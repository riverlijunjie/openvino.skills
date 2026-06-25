# GGUF 原生支持 — 实现总结（IMPLEMENT）

本文档记录按 [SPEC.md](SPEC.md) / [SUMMARY.md](SUMMARY.md) 落地 GGUF 原生支持时**实际**完成的
工作、修改/新增的全部文件，以及**如何测试**。配合三个本地 commit 阅读：

| PR | 仓库 | 分支 | commit | 规模 |
|----|------|------|--------|------|
| **PR-FE** | `openvino` | `river/gguf_support` | `7ee7804ca4` | OV Core 元类型 + 守卫 + FE（含 tokenizer `pre` rt-info key）|
| **PR-GPU** | `openvino` | `river/gguf_support` | `c7eb9315ce` | 16 files（含 transcode→OneDNN 路径）|
| **PR-GENAI** | `openvino.genai` | `river/gguf_support` | `3820c77f` | 入口切换 + tokenizer 走原生 FE rt-info |

> **硬约束 C6**：三个 commit 全部留在本地分支，**从未 push 到 remote**。
> commit hash 可能因 rebase 变化；用 `git log --oneline | grep '\[gguf\]'` 查最新值。

---

## 0. 端到端结果（验证状态）

在远程 Arc B580（`openvino-ci-74@10.239.140.155`，模型
`/home/openvino-ci-74/chenhu/openBMB/models/qwen3-4b-q4_0.gguf`，qwen3 架构 / Q4_0）上：

- ✅ `ov::Core::read_model("*.gguf")` 经原生 GGUF FE 产出合法 `ov::Model`（输出 `logits [?,?,151936]`）。
- ✅ GPU 编译 + 单次前向：`ocl::FCGGUFOpt` 内核被选中并执行，logits 全部有限。
- ✅ **decode（M=1）数值正确性**：对 `"The capital of France is"`（ids `[785,6722,315,9625,374]`）
  原生 FE→GPU GEMV 前向的 argmax = token **12095 = " Paris"**，top-5 语义合理。
- ✅ **prefill（M=64 > 阈值）transcode→OneDNN 路径正确性**：与强制走 GEMV 路径
  （`OV_GPU_GGUF_PREFILL_THRESHOLD=100000`）在同一输入上比对，argmax 基本一致
  （row0=804、row63=198 完全相同；个别 row 因对称重量化有 1 档差异），logit 量级吻合。
- ✅ 非 GPU 插件按 §3.5 **hard-fail**：CPU 上 `gguf_q4_0` 明确报错 “CPU plugin does not support gguf_q4_0”。
- ✅ openvino.genai 完整编译通过；`LLMPipeline` 默认走原生 FE（`OPENVINO_GENAI_USE_NATIVE_GGUF_FE=1`）。

**第二个模型：`/mnt/river/moe/models/Qwen3-8B-Q5_K_M.gguf`（qwen3 Instruct，mixed Q5_K(217)+Q6_K(37)+F32(145)）**：
- ✅ `read_model` + GPU 编译通过；单次前向对 `"The capital of France is"` argmax = **12095 " Paris"**（正确），
  logits 有限 —— 验证 **Q5_K / Q6_K** baseline 内核数值正确（含 §3.6 同模型多格式混用）。
- ✅ prefill transcode 路径（M=64）logits 有限、量级正常。
- ✅ **完整 LLMPipeline e2e（原生 FE + 原生 tokenizer + GPU GGUF 内核 + detokenize）生成连贯文本**：
  `"The capital of France is"` → `" Paris. The capital of Italy"`（贪心，6 token）。注意 in-tree reader
  **无法加载** Q5_K（只支持 Q4_0/Q4_1/Q4_K/Q8_0），原生 FE 路径才让该模型跑通。
- ✅ **性能优化（decode GEMV + prefill 显存）已落地**（详见 [OPTIMIZE_RESULT.md](OPTIMIZE_RESULT.md)）：
  decode GEMV 由 `local={1,1,1}`（~0.8% roofline）优化到 sub-group K-split + SWAR `dp4a`
  （单 kernel 最高 81% roofline）；prefill transcode 路径的**第二份 i8 权重副本**由
  *网络生命周期常驻* 改为 **per-engine 共享 scratch（每层用完即复用，不再常驻）**，使 8B/Q5_K_M 推理
  峰值显存 **~11.5 GiB → 7.42 GiB（省 ~4 GiB）**，TTFT 437ms 无回退、TPOT ~29ms/token、输出连贯（§2.4）。

---

## 1. PR-FE — OV Core 元类型 + Transformation 守卫 + 原生 GGUF FrontEnd

### 1.1 OV Core：23 个 GGUF block 元类型（SPEC §1）
- `src/core/include/openvino/core/type/element_type.hpp`：`Type_t` 追加 23 个 `gguf_*`
  枚举（顺序对齐 GGML `ggml_type`，**不可重排**）；新增 `block_byte_size()` /
  `block_elem_count()` / `is_gguf_block()` 访问器 + free function；
  `constexpr gguf_block_byte_size()/gguf_block_elem_count()` 表。
- `src/core/src/type/element_type.cpp`：23 行 `TypeInfo`（`is_real=false`、`is_quantized=true`、
  per-type `is_signed`、bitwidth=`ceil(block_bytes*8/elem)`）+ `EnumNames` 注册。
- `src/core/src/memory_util.cpp`：`get_memory_size` / `get_memory_size_safe` /
  `get_elements_capacity` 按 block 几何计算字节数（`ceil_div(n, elem)*bytes`）。
- `src/core/src/op/constant.cpp`：`calc_byte_strides` / `Constant(Tensor)` 对 GGUF 不按元素求 stride。
- `src/core/dev_api/openvino/core/type/element_iterator.hpp`：`is_byte_type` 排除 GGUF。
- `src/core/src/pass/visualize_tree.cpp` / `src/core/xml_util/src/xml_deserialize_util.cpp`：
  可视化 / IR 反序列化按 block 走（block-aware sizing，避免 bitwidth 取整误判）。
- `src/tests/.../common_test_utils/include/common_test_utils/type_ranges.hpp`：随机张量生成对 GGUF 走 default。

### 1.2 Transformation 守卫（SPEC §5.2，命中 GGUF Constant 即早退）
- `src/core/src/op/convert.cpp`：`Convert` 对 GGUF in/out 抛错。
- `src/core/src/pass/constant_folding.cpp`：扫到 GGUF Constant → 不折叠 + 钉 `disable_constant_folding`。
- `src/common/transformations/.../common_optimizations/nop_elimination.cpp`：reshape/squeeze 早退。
- `src/common/transformations/.../transpose_sinking/ts_base.cpp`：transpose-sinking 早退。
- `src/common/transformations/.../op_conversions/convert_fc_to_compressed.cpp`：通用 FC 压缩 pass 早退。
- `src/common/transformations/.../convert_precision.cpp`：`fuse_type_to_constant` 早退。
- `src/common/transformations/.../low_precision/mark_dequantization_subgraph.cpp`：DQ 标记早退。

### 1.3 原生 GGUF FrontEnd（SPEC §3）— `src/frontends/gguf/`
- `cmake/features.cmake`：`ENABLE_OV_GGUF_FRONTEND`（默认 ON）。
- `src/frontends/CMakeLists.txt` + `src/frontends/common/src/manager.cpp`：注册 `"gguf"` FE，
  `.gguf` 扩展名 + GGUF magic 映射，使 `FrontEndManager` 可发现。
- `src/frontends/gguf/CMakeLists.txt`、`src/CMakeLists.txt`：`ov_add_frontend(NAME gguf LINKABLE_FRONTEND ...)`。
- `src/frontends/gguf/include/openvino/frontend/gguf/{frontend,visibility}.hpp`：公开入口。
- `src/frontends/gguf/src/{frontend,gguf,gguf_reader,input_model,gguf_types,rt_info_keys}.{cpp,hpp}`：
  FrontEnd/InputModel 实现、mmap + header + tensor table 解析、GGUF type↔`element::Type` 映射、
  rt-info schema（顶层段 `"gguf"`，variadic 路径）。权重直接发射为 `gguf_*` Constant（**零拷贝、不 dequant**），
  FC 为 `FullyConnectedCompressed`、scale/zp 留空；非 qwen3 架构 `OPENVINO_THROW`。
- `src/frontends/gguf/src/builders/{builder.hpp,qwen3_builder.cpp,dequantize.{hpp,cpp}}`：
  qwen3 图重建（GQA attention、SwiGLU MLP、RMSNorm、RoPE、stateful KV-cache、lm_head）；
  `dequantize.*` 仅把 **token embedding** 解到 f16 给 `Gather`（非 FC，不违反 C1），覆盖
  F32/F16/BF16/Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K。
  > ⚠️ 注意：`builders/` 被仓库根 `.gitignore` 的 `[Bb]uild*/` 规则匹配而默认忽略，必须 `git add -f`。

### 1.4 PR-FE 修复的两个落地 bug
1. **FE 未被安装** → `read_model` 找不到 frontend：`src/frontends/gguf/src/CMakeLists.txt` 原带
   `DISABLE_CPP_INSTALL`（FE 编译但不进 `install_release/runtime/lib`）。**移除**。
2. **`rope.dimension_count` 缺失**：真实 qwen3 GGUF 不带该 key（带 `attention.key_length`）。
   `src/frontends/gguf/src/frontend.cpp` 改为从 `key_length`（或 `embedding_length/head_count`）派生，
   rt-info key 仍照常写入。

---

## 2. PR-GPU — `ocl::FCGGUFOpt` 单一 impl + baseline 内核

### 2.1 新增文件
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/fc_gguf_opt.cl`：**memory-bound GEMV 内核**。一个
  work-item 算一个 `(n, bm)` 输出，逐 block 在寄存器内解码权重再点乘激活；JIT `GGUF_IS_<TYPE>`
  选格式；baseline 解码体 Q4_0 / Q8_0 / Q4_K / Q5_K / Q6_K（镜像 `ggml-quants.c` 与 FE
  `dequantize.cpp`）。辅助函数用 `FUNC()/FUNC_CALL()` 装饰（batch 编译时避免重名冲突）；首参
  `OPTIONAL_SHAPE_INFO_ARG` 支持 shape-agnostic（行数 `bm` 取 `get_global_id(1)`）。
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/fc_gguf_transcode.cl`：**compute-bound transcode 内核**
  （§2.4）。一个 work-item 处理一个 `(n, K-group)`：复用同一套 block 解码体把 GGUF block 解到寄存器
  `half`，再**对称重量化**为 i4/i8（`TRANSCODE_TO_I4`/`QMAX`）+ per-group `f16` scale，写入两块
  scratchpad（权重 `[N,K]`、scale `[K/group, N]`）；**绝不**落 f16/f32 权重（C2）。
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/fc_gguf_opt.{hpp,cpp}`：
  `struct FCGGUFOpt : ImplementationManager`（`validate_impl` 只过 baseline 5 种 + f16/f32 激活；
  `support_shapes` 接受 dynamic 激活、静态时校验 `K % block_elem`）；GEMV JIT generator（从**权重**
  静态 shape `[N,K]` 取 K/N，手工发 `INPUT0`/`OUTPUT` type jit 以**避开**空 scale/zp 输入的
  shape-info `.at()`）。`FCGGUFOptImpl` **重写 `execute()` 按 M 分流**（§2.4）：小 M 走 GEMV stage；
  大 M 走 transcode stage + 直建 `dnnl::matmul`（LRU 缓存键 `(type,M,K,N)`）。含 transcode generator、
  `get_internal_buffer_descs`（scratchpad）、显式参数 `execute_stage` 重载。
- `src/plugins/intel_gpu/src/plugin/transformations/convert_gguf_fc_compressed.{hpp,cpp}`：
  把 FE 发的 `ov::op::internal::FullyConnectedCompressed`（gguf 权重）降为
  `ov::intel_gpu::op::FullyConnectedCompressed`；空 bias→`Placeholder`，scale→一个 dummy
  `f16 {1}=1.0` Constant（满足 cldnn FC 的 `!scale.empty()` 断言，内核忽略它），zp→`Placeholder`。

### 2.2 修改文件
- `src/plugins/intel_gpu/include/intel_gpu/runtime/layout.hpp`：`layout::bytes_count()` 与
  `data_type_traits::size_of()` 对 GGUF block 走 `ov::util::get_memory_size`（修正分配尺寸）。
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/utils/jitter.cpp`：`make_type_jit_constants`
  对 GGUF 映射为不透明 `uchar`。
- `src/plugins/intel_gpu/src/graph/registry/fully_connected_impls.cpp`：注册
  `ocl::FCGGUFOpt`（`shape_types::any`，作为动态 GGUF FC 节点的编译期 impl + 运行期按形状特化）；
  legacy dynamic FC 选择器对 `is_gguf_block` 早退。
- `src/plugins/intel_gpu/src/graph/impls/onednn/fully_connected_onednn.hpp`：OneDNN FC validator
  对 GGUF 权重早退（§3.5，OneDNN 会 repack 字节，绝不能选它）。
- `src/plugins/intel_gpu/src/plugin/transformations/convert_fc_to_compressed.cpp`：GPU 通用 FC 压缩
  pass 对 GGUF 早退（SPEC §5.2 #6）。
- `src/plugins/intel_gpu/src/plugin/transformations/fc_horizontal_fusion.cpp`：横向 FC 融合对 GGUF
  权重早退（融合会 `Concat` 不透明 block 常量 → ConstantFolding 崩溃）。
- `src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp`：注册
  `ConvertGGUFFullyConnectedCompressed`（在 `ConvertFullyConnectedToFullyConnectedCompressed` 之后）。
- `src/inference/include/openvino/runtime/properties.hpp` + `src/plugins/intel_gpu/src/plugin/plugin.cpp`：
  只读属性 `ov::supported_gguf_types` = `["Q4_0","Q4_K","Q5_K","Q6_K","Q8_0"]`（SPEC §4.4）。

### 2.3 PR-GPU 攻克的关键集成问题（按出现顺序）
1. **op-type 降级**：FE 发 `ie_internal_opset::FullyConnectedCompressed`，GPU 只有
   `gpu_opset` 的 factory → 加 `ConvertGGUFFullyConnectedCompressed` pass。
2. **ConstantFolding 段错误**：`FullyConnectedHorizontalFusion` 把多个 GGUF 权重常量
   `Concat` 后被 CF `evaluate` → 在 `fc_horizontal_fusion.cpp` 加 `is_gguf_block` 早退。
3. **动态 shape impl 绑定**：所有 253 个 FC 节点编译期都是 `[?,?,2560]` 动态。
   FCGGUFOpt 注册为 `shape_types::any` + generator dynamic-safe（从权重静态 shape 取 K/N）。
4. **OCL `CL_BUILD_PROGRAM_FAILURE`（redefinition）**：batch 编译多个 GGUF 内核时辅助函数重名
   → 全部用 `FUNC()/FUNC_CALL()` 装饰。
   > 改 `.cl` 后**必须**强制 codegen 重生：`touch *.cl && make run_ocl_codegen && touch utils/kernels_db.cpp`，
   > 否则增量 make 漏掉 `.cl→.inc→.o` 的过期依赖。
5. **`Different primitive id ... exists already`**：空 `Constant(dynamic,{0})` scale/zp 占位在 GPU
   topology 撞 id → 改用 `Placeholder` + dummy scale。

### 2.4 transcode → OneDNN WOQ 路径（compute-bound / 大 M prefill，SUMMARY §3.3.2/§3.3.3）

`FCGGUFOptImpl::execute()` 按激活行数 `M = derive_bm(activation)` 分流（阈值
`OV_GPU_GGUF_PREFILL_THRESHOLD`，默认 32）：

- **`M <= 阈值`（decode / 短 prompt）**：`execute_stage(gguf_stage)` 跑 memory-bound GEMV（原路径）。
- **`M > 阈值`（prefill / 长 prompt）**：`execute_transcode_plus_onednn_woq()`：
  1. **scratch（共享、不常驻）**：低比特权重 `layout([N,K], i4|i8)` + `f16` scale `layout([K/group, N])`。
     **不再**用 `get_internal_buffer_descs` 把这两块声明为网络生命周期常驻的 internal buffer（那会让每个
     GGUF FC 节点各留一份 i8 权重 → 与原生 GGUF 权重并存的"第二份权重"，8B 上常驻 ~5 GiB）。改为从
     **per-engine 共享 `TranscodeArena`** 取一块按 stream 分槽的 high-water scratch：弱引用静态注册表 +
     活实例持 `shared_ptr`（最后一个 GGUF FC impl 析构即释放），按需 grow-only 增长后 `reinterpret_buffer`
     视图成当前节点的精确 `[N,K]` 布局。**每层算完下一层直接覆盖复用**，全程只存在一份与最大 FC（lm_head
     i8 ≈ 622MB）等大的 scratch，原生 GGUF 权重始终是唯一一份权重；decode 路径仍直接读原生 block。
     > 复用安全性由 **oneDNN 强制 in-order 队列**保证（`ocl_stream.cpp` 拒绝 out-of-order）：相邻 FC 节点
     > FIFO 串行，上一节点 matmul 读完，下一节点 transcode 才覆盖；scratch 仅在增长时 `stream.finish()`。
     > `get_internal_buffer_descs` 现仅保留 decode 的 dp4a scratch（int8 激活 + f32 scale），不含权重。
  2. **transcode stage**：用自定义"显式参数 `execute_stage` 重载"派发 `fc_gguf_transcode.cl`
     （`global={N, K/group, 1}`），把原始 GGUF 权重重量化进 scratchpad。
  3. **直建 `dnnl::matmul`**（不经 `onednn::FullyConnectedImplementationManager`）：
     `src_md=[M,K]`、`dst_md=[M,N]`（dtype 取自 FC 输入/**输出** layout）、
     `wei_md=[K,N] format_tag::ba`（物理 `[N,K]`，与 transcode 写入顺序一致）、
     `attr.set_scales(DNNL_ARG_WEIGHTS, (1<<0)|(1<<1), {group,1}, f16)` + `fpmath_mode::f16`；
     pd 建好后用 `pd.{src,weights,dst}_desc()` 绑定（dnnl 可能选内部布局）；按 `(type,M,K,N)` LRU 缓存。
  4. 经 `mem->get_onednn_memory(md)` **零拷贝**把 cldnn scratchpad/激活/输出绑到 `DNNL_ARG_*`，
     `prim.execute(stream.get_onednn_stream(), args)`。OneDNN 与 OCL 共享同一 in-order 队列，
     transcode kernel 的写对随后的 matmul 可见。

**GGUF→OneDNN 目标档映射**（`transcode_target`，对称 per-group 重量化，group=32）：
`Q4_0`/`Q4_K` → `i4`（QMAX=7）；`Q5_K`/`Q6_K`/`Q8_0` → `i8`（QMAX=127）。这是**精度保持**的最小升档
（≤4-bit→i4，5/6/8-bit→i8，绝不退到 f16）。

**踩过的关键坑**（调试 transcode 路径时）：
1. **`make openvino_intel_gpu_plugin` 不会重编 `ocl_v2_obj`**：`fc_gguf_opt.cpp` 属于
   `openvino_intel_gpu_ocl_v2_obj` object library，必须显式 `make openvino_intel_gpu_ocl_v2_obj`
   （或 `touch` 源文件强制）后再 link plugin，否则跑的是旧 `.o`（表现为改了代码结果不变）。
2. **dst dtype 必须用输出 layout 的 dtype**：起初把 `dst_md` 硬编码成 `f16`，但该 FC 输出是 f16、
   激活也是 f16；真正的 bug 是早期把 f32 当 f16 读。最终 `src_dt`/`dst_dt` 分别取
   `convert_data_type(input0)`/`convert_data_type(output0)`，logits 从 ~7e6 乱码恢复到正常量级。
3. **internal buffer 常驻 → 显存双份**：经 `get_internal_buffer_descs` 声明的 buffer 按网络生命周期持有、
   execute 之间**不回收**到 pool（仅编译期 liveness 别名复用 ~26%），所以把 transcode 的 i8 权重放在那里
   会与原生 GGUF 权重**并存常驻**（8B 实测 +5.2 GiB）。修复 = 改用 per-engine 共享 scratch（上文 step 1）。
   实测峰值显存（`/proc/<PID>/fdinfo/*` 的 `drm-resident-vram0` 求和）从 ~11.5 GiB 降到 **7.42 GiB**，
   TTFT/TPOT 无回退。⚠️ 量显存别用 `grep -oE '[0-9]+'`（会把 `vram0` 里的 0 也匹配进去），用
   `awk '/drm-resident-vram0/{print $2;exit}'`。
4. **`run_cm_codegen` 复制到自身报错（与本改动无关的构建 flake）**：增量构建偶发
   `Error copying file (if different) from X to X` + `Circular X.cm <- X.cm`（陈旧 build.make 把
   `pa_*/xattn_*/xetla_*` 9 个 CM kernel 的 copy 规则源/目标都写成了 `codegen/cache/cm_kernels/X.cm`，
   而 cache 里只有 `cm_sdpa_vlen.cm`）。**最小修复**（不必 `cmake .` 重配，避免共享机风险）：把 `impls/cm/`
   下这 9 个源 `.cm` 拷进 `codegen/cache/cm_kernels/`，规则即视作已最新而跳过。

---

## 3. PR-GENAI — 入口切换 + tokenizer rt-info 迁移（`openvino.genai`）

- `src/cpp/src/utils.cpp` / `utils.hpp`（SPEC §6.1）：新增 `env_bool()` helper；`read_model()` 的
  `.gguf` 分支读 `OPENVINO_GENAI_USE_NATIVE_GGUF_FE`（默认 `true`）→ 走 `Core::read_model(path)`
  原生 FE；`false` 回退 in-tree `create_from_gguf`（保留一个 release 周期，§6.3）。
- `src/cpp/src/gguf_utils/gguf_tokenizer.{cpp,hpp}`（SPEC §6.2）：
  - 新增重载 `tokenizer_config_from_meta(const ov::Model&)`，从 FE rt-info
    （`"gguf","tokenizer","ggml",...`）取 tokenizer 配置：`model`、`pre`、`tokens`、`merges`、
    `token_type`/`scores`（物化为 `ov::Tensor`）、`bos/eos_token_id`（1 元素 u32 `ov::Tensor`，匹配
    `tokenizer_impl` 的读取方式）、`chat_template`。
  - **`create_tokenizer_from_config()` 改造**：当原生 FE 开启时（默认），用
    `Core::read_model(path)` + 上面的 rt-info 重载取配置，**不再**用 in-tree reader 重解析 `.gguf`。
    这是让 GenAI 能 tokenize **in-tree reader 加载不了的格式**（如 Q5_K/IQ\*）的关键 —— 没有它，
    `og.Tokenizer`/`LLMPipeline` 会在 `gguf.cpp:96 gguf_tensor_to_f16 failed` 处崩。
  - 对应 PR-FE 新增 `tokenizer.ggml.pre` rt-info key（SPEC §3.3 表外补充，pretokenizer 正则族，
    GenAI 用它选 BPE split regex）。
- in-tree reader / `gguf_quants` / `gguf_modeling` / `is_gguf_model` 全部**保留**为回退路径（§6.3，本期不删）。

---

## 4. 如何测试

### 4.1 远程环境与构建
```bash
ssh openvino-ci-74@10.239.140.155          # 密码: openvino
# OpenVINO（注意：本期为支持 genai 的 tokenizers，需要 ONNX FE 打开）
cd /mnt/river/moe/openvino
cd build-x86_64-release && cmake -DENABLE_OV_ONNX_FRONTEND=ON .. && make -j32 && make install
#   （若 onnx schema.h 编译 ICE/Bus error，多半是 -j 过高内存压力，改 make -j8 续编）
# openvino.genai
cd /mnt/river/moe/openvino.genai && source build.sh
```
本地→远程同步单个文件（不经过 git）：
```bash
sshpass -p openvino rsync -aR ./<相对路径> openvino-ci-74@10.239.140.155:/mnt/river/moe/openvino/
```
> 改 `.cl` 内核后需强制 codegen 重生（见 §2.3 #4）；改核心头（`layout.hpp` 等）会触发大范围重编。

### 4.2 测试模型
- 主用：`/home/openvino-ci-74/chenhu/openBMB/models/qwen3-4b-q4_0.gguf`（qwen3 / Q4_0，本地已有）。
- SPEC §7.1 推荐的 qwen3-0.6B 全量化变体（Q4_K_M/Q5_K_M/Q6_K/Q8_0/Q4_0）可作为更全面的 fixture。

### 4.3 GPU 内核功能 + 数值正确性（C++ 直测）
把下列程序编译并运行（在远程 `source install_release/setupvars.sh` 后）：
```cpp
// 载入 .gguf → GPU 编译 → 单次前向 → 检查 logits 有限 + argmax
#include <openvino/openvino.hpp>
int main(int, char** a){
  ov::Core core; auto m = core.read_model(a[1]);
  auto cm = core.compile_model(m, "GPU"); auto r = cm.create_infer_request();
  int64_t ids[]={785,6722,315,9625,374}; size_t S=5;            // "The capital of France is"
  std::vector<int64_t> mask(S,1),pos={0,1,2,3,4};
  r.set_tensor("input_ids",  ov::Tensor(ov::element::i64,{1,S},ids));
  r.set_tensor("attention_mask", ov::Tensor(ov::element::i64,{1,S},mask.data()));
  r.set_tensor("position_ids",   ov::Tensor(ov::element::i64,{1,S},pos.data()));
  ov::Tensor beam(ov::element::i32,{1}); beam.data<int32_t>()[0]=0; r.set_tensor("beam_idx",beam);
  r.infer(); auto o=r.get_tensor("logits"); const float* d=o.data<float>();
  size_t V=o.get_shape().back(), base=(S-1)*V, arg=0; float mx=-1e30f;
  for(size_t i=0;i<V;i++) if(d[base+i]>mx){mx=d[base+i];arg=i;}
  printf("argmax=%zu max=%.3f\n", arg, mx);             // 期望 argmax=12095 (" Paris")
}
```
```bash
I=install_release/runtime
g++ -O2 -std=c++17 t.cpp -o t -I$I/include -L$I/lib/intel64 -lopenvino
./t /home/openvino-ci-74/chenhu/openBMB/models/qwen3-4b-q4_0.gguf
# 期望: argmax=12095（用 og.Tokenizer 解码为 " Paris"）；logits 全部有限
```
> 多 prompt 复用同一 `infer_request` 时务必 `r.reset_state()`，否则 stateful KV-cache 串味。

### 4.4 §3.5 hard-fail 验证（非 GPU 插件）
```bash
benchmark_app -m qwen3-4b-q4_0.gguf -d CPU -niter 1 -hint none \
  -data_shape "input_ids[1,8],attention_mask[1,8],position_ids[1,8],beam_idx[1]"
# 期望: 明确报错 "CPU plugin does not support gguf_q4_0 ..."（不是崩溃/乱算）
```

### 4.4b transcode→OneDNN prefill 路径（大 M）验证
喂一个 `M=64`（> 默认阈值 32）的激活，使 FC 走 transcode+OneDNN 路径；再用
`OV_GPU_GGUF_PREFILL_THRESHOLD=100000` 强制同一输入走 GEMV，比对两条路径的逐行 argmax：
```cpp
// 关键: input_ids/attention_mask/position_ids 形状 [1, 64]，beam_idx [1]；infer 后逐行取 logits argmax
// （完整程序见远程 /tmp/gguf_prefill.cpp）
```
```bash
# transcode 路径（默认阈值 32，M=64 > 32）
./gguf_prefill qwen3-4b-q4_0.gguf
# GEMV 路径（强制阈值很大）
OV_GPU_GGUF_PREFILL_THRESHOLD=100000 ./gguf_prefill qwen3-4b-q4_0.gguf
# 期望: 两条路径 logits 全部有限、量级一致、逐行 argmax 基本相同
#       （对称重量化会让个别 row 差 1 档，属预期）
```
> 调试提示：改 `fc_gguf_opt.cpp` 后必须 `make openvino_intel_gpu_ocl_v2_obj` 再 `make
> openvino_intel_gpu_plugin`（见 §2.4 坑 1），否则跑的是旧 `.o`。

### 4.5 GenAI 端到端（LLMPipeline，原生 FE + 原生 tokenizer）
```bash
cd /mnt/river/moe/openvino.genai && source /mnt/river/moe/openvino/install_release/setupvars.sh
source venv/bin/activate
export OPENVINO_GENAI_USE_NATIVE_GGUF_FE=1     # 默认即 1；=0 走 in-tree 回退
python3 -c "
import openvino_genai as og
pipe = og.LLMPipeline('/mnt/river/moe/models/Qwen3-8B-Q5_K_M.gguf', 'GPU')
cfg = og.GenerationConfig(); cfg.max_new_tokens=6; cfg.do_sample=False; cfg.apply_chat_template=False
print(pipe.generate('The capital of France is', cfg))
"
# 实测输出: ' Paris. The capital of Italy'（连贯）。注意当前 GEMV 内核未优化，8B 上约 30s/token，
# 大 max_new_tokens 会很慢（正确性不受影响）。
```
- 重要：Q5_K/IQ\* 等格式 **in-tree reader 加载不了**（`gguf.cpp:96` 崩），必须走原生 FE
  （`OPENVINO_GENAI_USE_NATIVE_GGUF_FE=1`，默认）。`og.Tokenizer(path)`/`LLMPipeline(path)` 现在内部
  对原生路径用 `Core::read_model` 的 rt-info 构 tokenizer（§3 / §6.2）。
- 改 `gguf_tokenizer.cpp` 后同样要先 `make openvino_genai_obj` 再 `make openvino_genai`
  + `cmake --install`（genai 也是 object library，`make openvino_genai` 不重编 obj）。
- **正确性参照**：Q4_* 等 in-tree 支持的格式可用 in-tree + **CPU**
  （`OPENVINO_GENAI_USE_NATIVE_GGUF_FE=0`, device `CPU`）做 golden。

### 4.6 单元测试（可选，SPEC §7.3 oracle）
`build_release.sh` 默认 `ENABLE_TESTS=OFF`；若要跑 `ov_gpu_unit_tests --gtest_filter=*gguf*`，
需以 `-DENABLE_TESTS=ON` 重新配置并 `make ov_gpu_unit_tests`。本期未提供专门的 gguf gtest，
正确性以 §4.3 的 " Paris" 预测 + §4.5 in-tree/CPU 参照覆盖。

---

## 5. 仍未做 / 后续增量（不在本期范围）
- **已落地的性能优化**见 [OPTIMIZE_RESULT.md](OPTIMIZE_RESULT.md)：decode GEMV（K-split / Q6_K ILP /
  SWAR `dp4a`）+ prefill transcode 路径的共享 scratch（消除第二份常驻 i8 权重，省 ~4 GiB 显存）。
- transcode→OneDNN 的进一步优化：本期用**对称 per-group(32) 重量化**完成 transcode（数值正确、
  与 GEMV 逐行 argmax 基本一致），但 SUMMARY §3.3.2 设想的"把 K-quant 嵌套 super×sub scale 无损
  预乘进单一 per-group scale"未做（当前是有损但精度等级保持的重量化）；双缓冲流水、per-arch
  K_TILE/N_TILE 调参、Q4_1/Q5_0/Q5_1 等的 ZP 透传也留待增量。
- 18 种非 baseline element-type 的 GPU 内核（element-type 已全量定义，`validate_impl` 明确拒绝）。
- llama / qwen2 / mistral / phi 等其他架构 builder（本期仅 qwen3）。
- SPEC §7.1/§7.4 的字节不变量 / IR round-trip CI 脚本（`run_gguf_local.sh`）与 §7.2 llama.cpp token-equality
  oracle（远程无 `llama-cli`）。
- PR-GENAI §6.3 中 in-tree reader / `gguf_quants` 的删除（本期保留为回退）。
- GPU 多 token 生成退化的 root-cause（与 GGUF 无关，in-tree 路径同样复现）。
