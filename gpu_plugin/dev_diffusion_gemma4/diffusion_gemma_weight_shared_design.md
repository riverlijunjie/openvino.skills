# DiffusionGemma encoder/decoder 权重共享详细设计

> 目标：保留**两个** `ov::Model` / 两个 `CompiledModel`（不合并单图），但让 encoder 与 decoder
> 的**相同权重在 GPU 显存里只存一份**，并让**量化只做一次、主机也只存一份**。
>
> 范围：本文是可实现、可验证的工程设计。背景分析与日志/内存归因见同目录
> [`diffusion_gemma_flow_memory_cn.md`](./diffusion_gemma_flow_memory_cn.md)（§4 可行性、§6 概要）。
>
> ⚠️ 文中文件/行号为代码锚点。约束：GPU 首选精度 f16（不引入全局 f32）；仅本地修改，不 commit/push。

## ✅ 已实现并在远程 PTL（Arc B390 iGPU, Xe3）验证（2026-06-23）

**只实现了路径 2a（GenAI 打 `shared_weight_id`）+ 路径 1（GPU engine cache），路径 2b（SharedQuantWeightCache）暂缓** ——
仅设备 buffer 去重就足以验证收益。干净 A/B：同一 exe，只切换 GPU 插件 DLL（旧=无 cache / 新=有 cache），
其余完全一致（prompt "Why is the sky blue?"，`--output-tokens 128`）。

| 指标 | 基线（旧 DLL） | 共享（新 DLL） | 变化 |
|---|---|---|---|
| **吞吐** | 1.72 tok/s | **5.93 tok/s** | **3.45×** |
| decoder 编译 | 113.8 s | **17.9 s** | **6.3×**（命中 cache → 跳过上传） |
| encoder pass（生成期） | 17371 ms | **518 ms** | **33×**（不再换页） |
| decoder step #1 | 34085 ms | **936 ms** | **36×**（权重已驻留，无 PCIe thrash） |
| decoder 稳态/步 | ~385 ms | ~393 ms | 持平（compute 未变，符合预期） |
| 输出文本 | "Rayleigh scattering…" | **逐字节相同** | ✅ 正确性零回归 |
| nan/inf/exception | 0 | 0 | ✅ |
| peak working set | ~25.8 GB | ~25.6 GB | 持平（见下） |

**结论：达到甚至超出预期。** 真正的瓶颈是显存超额（两份权重 34.64GB > 31.62GB）导致的 **host-USM PCIe 换页**；
去掉 decoder 那份重复设备 buffer 后，权重全程驻留、首次访问不再 thrash —— encoder-pass 33×、decoder-step-1 36×
的提速正是换页消失的指纹（稳态步时不变佐证 compute 未动）。decoder 编译 6.3× 提速 = cache 命中跳过了 ~930 个
权重常量的 `allocate_memory + PCIe 上传`。

**关于 peak working set 持平**：当前只做了路径 1（设备去重），路径 2b（量化产物主机缓存）暂缓 —— decoder 仍
重新量化、其 int4 主机 buffer 在 build 期短暂存在，所以**主机** RSS 峰值由 build/量化主导、未降。但**设备侧**重复已
消除（这才是换页根因）。若进一步要降主机峰值，再补路径 2b（让 decoder 复用 encoder 已量化的 `ov::Tensor`）。

> 实现要点（与下文设计一致）：GenAI 侧 `detail::stamp_shared_weight_id` 给 3 个 MoE 专家权重
> （gate/up/down 的 `*_w_q_`）写 `rt_info["shared_weight_id"] = weight_name + "|{gate|up|down}_w"`；
> GPU 侧 `create_data`（constant.cpp）读该 id + layout 指纹，查 engine 级 `device_weight_cache`
> （`weak_ptr<memory>`），命中即复用设备 buffer、跳过 allocate+memcpy，未命中则分配并登记。

---

## ✅ 扩展到 FC 稠密权重并修复（2026-06-23，content-hash + stable 快路径）

继上节"仅 MoE 专家权重"之后，把设备 buffer 共享**扩展到 FC 稠密 / fused-QKV 权重**（`embed_tokens` + 各层 `q/k/v/o_proj`、`gate/up/down_proj`）。直接扩展先**翻车**（解码器输出 "3. 3. 8. 8…" 乱码），定位后修复并验证。

### 翻车根因（推翻了中途"scale/zp 是元凶"的判断）
verbose 抓了 249 条 cache HIT 的 key：**全是 `|fc` / `|gate_w|up_w|down_w` / fused-QKV，没有一条 scale/zp**。scale/zp 的 id 在 `ops.cpp` `normalize_aux` 的 `transpose({0,2,1})` 常量折叠处就丢了（多源 rt_info 合并，纯字符串被丢弃）——它们的"共享"一直是**静默空操作**。真正的元凶是 **FC 类权重在 content-blind key 下错配**：key = `shared_weight_id + "|" + layout.to_string()`，而 `layout::to_string()` 只编码 dtype/format/shape/padding，**不含字节内容**。FC 权重会被 per-graph **repack**（`ConvertFCToCompressed` reshape、`FullyConnectedHorizontalFusion` 按非确定的 `get_users()` 顺序 concat）→ 同 (id,layout) 但 encoder/decoder **最终字节不同** → HIT 把图1的字节 memcpy 给了图2 → 乱码。（Config C 当时误判 scale/zp，是因为 NEW exe 的 `|fc` 标记连旧 MoE-only DLL 都命中，如 `embed_tokens.weight|fc`。）

### 用户指令 → 修复思路
用户要求：**"按最终 transpose-fold / reshape / horizontal-fusion 之后的 GPU memory 来管理 cache，中间 repack 多余也没关系，只要最终 buffer 进 cache / 被命中即可。"** 关键事实：这三个变换都在 `transform_model` 阶段、**早于** program-build 的 `create_data`，所以 `op->get_data_ptr<char>()`（[`constant.cpp:92`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L92)）拿到的**就是最终 buffer**；`create_data` 之后唯一的权重变换是 FC 的 GPU reorder（`post_optimize_weights` 只处理 conv/fc/lstm/gru，**从不碰 MoE**），且**异地、只读**（[`reorder.cpp:230`](../../../../openvino.mx/src/plugins/intel_gpu/src/graph/impls/ocl/reorder.cpp#L230)）。⇒ "按最终 buffer 管理" = **用最终字节的 hash 当 key**。

### 修复（两半，缺一不可）
1. **内容寻址 key**（`create_data`）：key = `shared_weight_id + "|" + layout.to_string() + "|h<hash>|s<byte_size>"`。hash = `std::hash<std::string_view>(data_ptr, byte_size)`。**不用** `ov::runtime::compute_hash`（`core/dev_api/compute_hash.hpp` 声明了但**未从 core DLL 导出** → GPU 插件 LNK2019；`std::hash` 头文件即用、全 buffer，cache 进程内单例、非密码学足矣）。命中只发生在最终字节逐 bit 相同时；FC 跨图字节分叉 → MISS → 各自上传（按构造正确）。
2. **`shared_weight_stable` 快路径**（性能关键的一半）。hash 不是免费的：**读字节正是 HIT 想跳过的工作**。A/B 实测：content-hash DLL + MoE-only 标记 = 解码器编译 **119.9s**（Config F）vs 旧 MoE-only DLL **25.9s**（Config E）——约 **+94s** 全花在为算 hash 而读那 ~12 GB 冷的、刚量化的 MoE 权重（32GB 机器、内存吃紧）。修法：GenAI `stamp_shared_weight_id(t, id, stable=true)` 给 3 个 int4 MoE 专家权重（`*_w_q_`，确定性量化、被 `MOECompressed` 原样消费、无 repack ⇒ (id,layout) 已唯一定身）写 `rt_info["shared_weight_stable"]="1"`；插件对 stable 权重**跳过 hash**（恢复无字节快路径），只对会 repack 的 FC（~1.5 GB，且其 repack 本就碰了字节）算 hash。scale/zp 不标 stable、当前也不共享（id 折叠处丢失）——无害空操作。

### 验证（Config G：`NEW_stable` exe + `gpu_plugin_NEW_stable.dll`，iGPU）
| 指标 | 结果 |
|---|---|
| 输出 | "Rayleigh scattering…"，`RUN_EXIT=0` ✅ |
| decoder GPU 编译 | **31.1s**（vs hash-everything 114s；≈ MoE-only 基线） |
| cache 命中（累计） | **hits=248, bytes_saved=12.97 GB**, bytes_uploaded=14.89 GB（decoder 复用 93% 权重字节） |
| `memory swap` 警告 | **0** |
| 吞吐 | 6.0 tok/s |

> cache 计数器（新增）：`device_weight_cache` 加原子 hit/miss/bytes；`ProgramBuilder::build` 用 `GPU_DEBUG_COUT` 打**一行** `[weight-cache] cumulative: …`（不做 per-constant 日志——对 ~2k 常量的 decoder 太吵，且 verbose 满屏可能拖垮运行）。非 DiffusionGemma 模型计数恒 0、不打印。

### 端到端内存对比（`diffusion_gemma.log` 基线 → `dg_memory_opt_0623.log` 优化后）
同模型同 prompt（"Why is the sky blue?"，`--output-tokens 256`，int4_asym + group_size=64）下，从 `~MemoryTracker` 的 `max=` 峰值与逐笔 `Allocate` 统计提取。基线是**共享前的旧 build**（无 `[weight-cache]` 汇总行），优化后是 `NEW_stable` build：

| 指标（`usm_host`，除非注明） | 基线 `diffusion_gemma.log` | 优化 `dg_memory_opt_0623.log` | 变化 |
|---|---|---|---|
| **峰值工作集 (`max=`)** | **38.59 GiB** (41,435,075,638 B) | **14.89 GiB** (15,984,672,908 B) | **−23.70 GiB (−61.4%)** |
| 生命周期总申请 | 42.79 GiB | 16.09 GiB | −26.70 GiB |
| `usm_host` 申请次数 | 3090 | 3565 | +475（见注） |
| `usm_device` 峰值 | 0.44 GiB | 0.45 GiB | 持平（iGPU 设备侧未变） |
| 121 MiB MoE 专家块 | 180 笔 = 21.27 GiB | **90 笔 = 10.63 GiB** | **−90 笔 / −10.63 GiB**（精确减半） |
| decoder GPU 编译 | 180.6 s | **27.7 s** | **6.5×** |
| 吞吐 | 2.99 tok/s | **5.34 tok/s** | **1.79×** |
| 输出文本 | 正确 | "Rayleigh scattering…" 正确 | ✅ |
| `[weight-cache]` 汇总 | 无（旧 build） | `hits=248 misses=1139 bytes_saved=12.97 GB` | — |

**互相印证**：优化后 `bytes_saved=12.97 GB`(12.08 GiB) 恰等于在 0623 build 内 A/B（[`dg_0618_2.log`](#) 26.97 GiB → 14.89 GiB）的峰值差，且 `hits=248` = 申请次数差 248 = 121 MiB 块减半的 90 笔 + FC/embed 的 158 笔。

**为什么基线峰值更高（38.59 GiB）/ 申请次数反而更少（3090）**：基线是更早的 build，量化布局不同——其 `usm_host` 直方图含 **1408 MiB×4 (5.50 GiB)、22 MiB×200 (4.30 GiB)、44 MiB×40 (1.72 GiB)、11.34 MiB×366 (4.05 GiB)** 等优化 build 里**完全不存在**的大块（疑似旧的整层/未分组打包），用更少但更大的块堆出了更高的峰值。因此这是**端到端的版本前后对比**（共享 + 量化布局演进的合力），**不是**单一变量 A/B；纯 cache 单变量收益见上节 §"扩展到 FC" 的 Config G（同 exe 仅换 DLL 的 12.08 GiB / 248 hits）。

### 改动文件（均本地 + scp 到远程，未 commit）
- GenAI `modeling_diffusion_gemma_text.cpp`：`stamp_shared_weight_id` 加 `bool stable`；3 个 MoE `*_w_q_` 传 `stable=true`。
- GPU `ops/constant.cpp`：内容 hash key + stable 跳过 + cache 计数器。
- GPU `runtime/engine.hpp`：`device_weight_cache` 原子计数 + 注释更新。
- GPU `plugin/program_builder.cpp`：一行累计 cache 摘要。
- （前序：`convert_fc_to_compressed.cpp`/`fc_horizontal_fusion.cpp`/`shared_weight_id.hpp` 的 id 传播、`constant.cpp` 的 dGPU usm_device 提升——均仍有效；dGPU 路径在集显 B390 上 dormant。）

### 待办（可选 Phase 2）
让 scale/zp 的 id 活过 `transpose` 折叠（注册带 `merge()` 的 RuntimeAttribute，或折叠后再 stamp），使 ~851 MiB scale/zp 也共享——**现在有内容 hash 兜底，安全**。未做。

---

## ✅ 扩展到 IR 图缓存（Plan C）+ 修复缓存模型的共享回归（2026-06-25，可序列化 `SharedWeightId`）

> 这是上面"设备 buffer 共享"在 **IR 图缓存（`--cache-model`）路径**上的延续：缓存模型在加载提速后，
> decoder **推理** 反而回归到换页慢速。根因是共享身份（`shared_weight_id`）在 `ov::serialize` 时被丢弃。

### 背景：IR 图缓存（Plan C）省掉 build + 在线量化
为把 26B 模型的 ~722s 图构建 + RTN 在线量化在 warm run 跳过，sample 增加 `--cache-model`：首次 build 后
`ov::serialize` 出 IR（`.xml`/`.bin`，落模型目录旁），后续 `core.read_model` 直接读回 —— 只剩 mmap + compile。
唯一非标准 op 是 `ov::op::internal::MOECompressed`，用 `OpExtension` 注册即可反序列化。

### 回归：缓存模型 decoder 推理 14 → 6 tok/s
读 IR 回来的模型加载快了 ~16×，但 **decoder 推理从 14 tok/s 掉到 6 tok/s**。根因：
`shared_weight_id` / `shared_weight_stable` 是**纯字符串 rt_info**，而 `ov::serialize` **只写**
`is_deterministic()==true` 的 `RuntimeAttribute`，**纯字符串 rt_info 被丢弃**（实测 `decoder.xml` 里
0 个 `shared_weight_id`、0 个 `shared_weight_stable`、30 个 `MOECompressed`）。⇒ 缓存模型无 id →
`constant.cpp` 走普通路径 → encoder/decoder 各上传一份 ~12 GB MoE → ~27 GB > 16.5 GB 显存 → 换页 →
6 tok/s。**正是 §"扩展到 FC" 治好的同一个换页病，在 IR-cache 路径上复发。**

### 附带：built vs 反序列化模型的 blob-key 不一致（已先行修复）
GPU 编译 blob cache（`ov::cache_dir`）的 key 由 `compute_hash` 算，含 in-memory rt_info；**built 模型**
（带字符串 rt_info）与**反序列化模型**（不带）key 不同 → cold 导出的 blob 永远不被后续 warm run（读 IR）命中。
修复：cold 路径在 `ov::serialize` 后**重新 `read_model`** 一遍、编译那份反序列化模型 → cold 导出的 blob 与
每个 warm run 的 key 一致 → 首个 warm run 即命中 blob。

### 方案 2（采纳）：可序列化的 `SharedWeightId` RuntimeAttribute
让 id 真正活过 IR 往返，且 **GPU 侧消费逻辑（`constant.cpp` / 压缩 pass）完全不动**：

1. **core 新增** header-only `ov::SharedWeightId : public ov::RuntimeAttribute`
   （`OPENVINO_RTTI("SharedWeightId","0", ov::RuntimeAttribute)`，成员 `std::string id; bool stable;`，
   `visit_attributes` 读写 `id`/`stable`）。RTTI 名 `"SharedWeightId"` **故意区别于**纯字符串键
   `"shared_weight_id"`（RTTI 名即 rt_info map 键，撞键会冲突）。`visit_attributes` **内联在头文件**
   （仿 `Decompression`）⇒ **无新 .cpp** ⇒ 不触发 CMake GLOB 重配置。
2. **注册工厂**：`attributes.cpp` 的 `ov::pass::Attributes` 加 `register_factory<ov::SharedWeightId>()`
   —— 否则 IR 反序列化不会重建该属性。
3. **GenAI 加打**（与纯字符串**并存**，fresh 路径行为不变）：finalizer 两处 + model_text 的 stamp helper，
   在写 `rt_info["shared_weight_id"]=…` 的同时写
   `rt_info[ov::SharedWeightId::get_type_info_static()] = ov::SharedWeightId(id, stable)`。
4. **sample 读回后还原**：`load_or_build` 里**每次** `read_model`（warm 读 + cold 重读）后调
   `restore_shared_weight_ids(model)` —— 遍历 `get_ops()`，凡带 `SharedWeightId` 属性者
   `rt.emplace("shared_weight_id", swid.id)`（stable 则补 `"shared_weight_stable"="1"`）。
   GPU 插件仍只读这两个纯字符串键 ⇒ **插件零改动、无需重编 GPU plugin**。

> **blob-key 一致性**：还原写回的纯字符串**不被序列化**（compute_hash 时已丢），而 `SharedWeightId` 属性在
> cold-重读 与 warm-读 上**完全一致** → blob key 仍匹配 → warm 仍命中 blob。

### 验证（Arc B390 iGPU，output-tokens 256，`--cache-model`）
| 指标 | COLD（build→序列化→重读→还原→编译） | WARM（读 IR→还原→编译） |
|---|---|---|
| decoder-only 吞吐 | **14.22 tok/s** | **14.03 tok/s** |
| 图加载（build/量化 → 读 IR） | enc 289s + dec 373s ≈ **662s** | enc 0.18s + dec 0.21s ≈ **0.4s**（~1685×） |
| 到首 token | ~711s | ~43s（~16.6×） |
| `[weight-cache]` 累计 | hits=248 bytes_saved=12.97GB bytes_uploaded=14.89GB | 同左 |
| 输出文本 | "Rayleigh scattering…" 正确 | 同左，`RUN_EXIT=0` |

GPU 显存（总 17.72 GB / 16.5 GiB，`hello_query_device` 的 `GPU_DEVICE_TOTAL_MEM_SIZE`）：共享开启后
权重占 **14.89 GB（84%，无 `memory swap` 警告）**；关闭共享将是 **27.86 GB（157%）→ 溢出换页 → 6 tok/s**。
⇒ 方案 2 在 cold-重读 与 warm-读 两条路径上都把 decoder 推理拉回 ~14 tok/s。

### 改动文件（本地 + scp，未 commit）
- **core**：`transformations/include/transformations/rt_info/shared_weight_id_attribute.hpp`（新，header-only）、
  `transformations/include/transformations/rt_info/attributes.hpp`（include）、
  `transformations/src/transformations/rt_info/attributes.cpp`（注册工厂）。
- **genai**：`safetensors_weight_finalizer.cpp`（2 处与纯字符串并存打标）、
  `modeling_diffusion_gemma_text.cpp`（stamp helper 并存打标）、
  `modeling_diffusion_gemma.cpp`（`restore_shared_weight_ids` + 两次 `read_model` 后调用 + cold 重读 IR +
  `--cache-model` 时注册 `MOECompressed` OpExtension）。

> 另一处 core 修复（同属 IR 往返）：`ov_ops/moe_compressed.cpp` 的 `MOECompressed::visit_attributes` 在
> `MOE::visit_attributes(visitor)` 之后补 `static_cast<MOE::Config&>(m_config) = MOE::get_config();` ——
> 派生 `m_config` 遮蔽了基类的，反序列化后派生 config 字段停在默认值导致 GPU lowering 走错 MoE 路径
> （"No layout format … MOECompressed_*_moe_gather"）。修复后 IR-反序列化的 MOECompressed 才能正确编译。

### ⚠️ 关键：`--cache-dir`（GPU 编译 blob 缓存）与权重共享**互斥**（2026-06-25 修正）
手动测试发现：**第一次运行正常（~15GB / ~14 tok/s），第二次起 ~28GB / ~6 tok/s** —— 共享在第二次失效。
根因：`ov::cache_dir` 的 blob 在第二次走 `Plugin::import_model`（[`plugin.cpp:379`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/plugin.cpp#L379)）**直接从 blob 重建 CompiledModel 的权重**，
**完全不经过 `create_data`** → engine 级 `device_weight_cache` 跨图共享被**绕过** → encoder/decoder 各自
materialize ~14GB → ~28GB > 16.5GB 显存 → 换页 → 6 tok/s。复现：RUN1 14.32 tok/s（`hits=248`），
RUN2 6.53 tok/s（**完全没有 `[weight-cache]` 行** → `create_data` 一次没跑）。且本模型 blob import（53s）
比"共享 JIT 编译"（42s）还**慢** —— 对 fused-MoE 路径，`--cache-dir` 是**纯亏**。

> 为什么不能"只缓 encoder / 混用 import+JIT"：被 import 的 encoder 权重**不会登记进** engine cache，
> 故即便 decoder 走 JIT 也无可命中 → 必须两图都 JIT → 对 fused-MoE **整体禁用 blob 缓存**。

**修复（sample 单点）**：`use_fused_moe` 为真时**跳过** `core.set_property(ov::cache_dir(...))` 并打印
`[cache-dir] Ignored...`。`--cache-model`（IR 缓存）本就跳过了 ~662s 的 build+量化、且其 `create_data` 编译
**保留共享** → IR 缓存才是该路径唯一正确/最优的缓存。验证：FIX1 13.23 tok/s、FIX2 13.53 tok/s（两次都
`hits=248`、blob 目录恒空），第二次回归消失、跨次稳定。

> 因此上文"blob-key 一致性 → warm 仍命中 blob"对 fused-MoE 路径**已作废**：blob 命中本身有害，已禁用。
> cold 路径的"重读 IR 再编译"仍保留（让 cold/warm 都编译同一份反序列化模型、行为一致），只是其
> 原始动机（让 blob key 匹配）对 fused-MoE 不再适用。

---

## 实现前对抗式复审结论（2026-06-23，go/no-go = **GO，收窄范围**）

实现前用独立 agent 对抗式复审了"rt_info 能否活到 GPU `create_data`、且节点/数据指针不变"等 5 个关键假设，并实测核对了若干 API。结论：

| 权重角色 | rt_info 是否存活到 `create_data` | 数据指针是否不变 | 跨图是否一致 | 是否纳入共享 |
|---|---|---|---|---|
| **MoE 专家 gate/up/down 权重**（`*_w_q_`，121 MiB×90，**88% 显存**） | ✅ 存活（MOECompressed 直接吃原 Constant，不 repack；`KeepConstantsPrecision` 只**追加** rt_info） | ✅ clone 共享 `m_data`（[`constant.cpp:317`](../../../../openvino.mx/src/core/src/op/constant.cpp#L317)） | ✅ 确定性量化逐 bit 相同 | **✅ 是** |
| MoE scale/zp | ❌ 被 `reshape→transpose` **常量折叠**成新节点，rt_info 丢失（`transformations_pipeline.cpp:695-697`） | ❌ 折叠产生新 buffer | — | ❌ 否（且体积极小） |
| FC 语义权重（rank>2/转置） | ❌ `convert_fc_to_compressed.cpp:82` **repack** 成新 Constant，rt_info + friendly_name 双丢 | ✅（`m_data` 仍共享） | ✅ | ❌ 否（仅 1.55 GiB，风险大） |

**关键修正（相对初版设计）**：

1. **范围收窄为"仅 MoE 专家权重"**。scale/zp 与 FC 语义权重经实测会被后续 pass 折叠/重打包，rt_info 不可靠 —— 排除。MoE 专家权重是 88% 显存，且 `MOECompressed` 直接消费原始 Constant、不 repack，rt_info 与数据指针都稳定。
2. **身份载体 = `rt_info["shared_weight_id"]`（字符串、逻辑派生）成立**（复审证实其对 MoE 权重存活）。**不用** friendly_name（会被融合 pass 改名）、**不用**纯数据指针 key（主机地址释放后可能被复用 → 不同权重撞 key → 错配；字符串 id 对地址复用免疫）。
3. **路径1 正确性不依赖路径2 的缓存**：解码器即使重新量化到新 host buffer，`create_data` 凭 `shared_weight_id` 命中 cache → 复用 encoder 的设备 buffer（内容已证一致）。缓存（路径2 的 `SharedQuantWeightCache`）是**附加优化**（省 decoder 重量化 CPU + 主机一份），非正确性前提。
4. **设备共享安全性已确认**（复审 Q2）：`cldnn::data` 是只读、`can_share_buffer(false)`、不进内存池；`attach_or_copy_data` 对**同 engine** buffer 直接别名、不拷贝。**前提：两 CompiledModel 同 engine** —— 已确认（`weak_singleton_default_contexts` 进程级单例，[`plugin.cpp:207`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/plugin.cpp#L207)）。
5. **`hint_evict` 对 tensor-backed Constant 是 no-op**（[`shared_buffer.hpp:42`](../../../../openvino.mx/src/core/dev_api/openvino/runtime/shared_buffer.hpp#L42) 空 else 分支），不会释放量化产物主机 buffer —— 与 cache 的字符串 key 无关，但确认了主机侧行为。

> 下文 §3–§5 的"方案 B / rt_info" 即按此结论实现：**`shared_weight_id` 仅打在 MoE 专家 `*_w_q_` 上**；GPU cache 仅对带该 key 的 Constant 参与，按 `(shared_weight_id, layout)` 去重设备 buffer。

---

## 0. TL;DR

- **现状**：encoder 与 decoder 从同一 `WeightSource`（同一 mmap）加载，但 mmap 之后**各做一份** ——
  各自反量化、各自**重新量化**、各自烤 `ov::Constant`、各自在 GPU 上 `allocate_memory` 并上传。
  权重在设备上**烤了两份**，是 38.59GB 峰值 > 31.62GB 显存 → host-USM 换页 → 2.99 tok/s 的根因之一。
- **内容一致性已用源码证实**：encoder/decoder 对同一逻辑权重取**同一份 safetensor 字节**、施加**相同变换**、
  量化是**纯确定性映射**（逐 bit 相同）。⇒ 共享在数值上**安全**，不是近似。
- **方案**：两步，可分别上线、互相解耦。
  - **路径2（GenAI 侧）**：量化一次 + 主机一份 + 给共享权重打**稳定身份**。
  - **路径1（GPU 插件侧）**：engine 级 `DeviceWeightBank`，按稳定身份**去重设备 buffer**。
- **身份载体定为方案 B（friendly_name / 自定义 rt_info）**，方案 A（weightless）经 XML 实测否决。
  ⚠️ **不修改共享的 `ops::constant` 默认命名行为**（会污染其它模型）——命名只在 DiffusionGemma MoE loader 内显式注入。
- **预期收益**：设备权重 34.64GB → ~17–18GB；峰值 38.59GB → ~21–22GB；换页计数 1116 → ~0；吞吐显著回升。

---

## 1. 现状：实际共享边界（源码核对）

两个 builder 用**同一个 `DiffusionGemmaForBlockDiffusion` 结构**、同一套 packed_mapping、同一句
`load_model(model, source, finalizer)`，从**同一个 `source`** 加载（encoder
[`modeling_diffusion_gemma_text.cpp:1556-1558`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L1556)、
decoder [`:1653-1655`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L1653)）。
但"共享"仅到 mmap 这一层：

| 阶段 | encoder vs decoder | 证据 |
|---|---|---|
| (a) safetensor 磁盘字节 / OS page cache | **共享** | 单文件 |
| (b) 进程内 mmap 区域 | **共享** | 每文件一个 `shared_ptr<MmapHolder>`，所有 tensor 的 `mmap_info.holder` 指向它（[`safetensors_loader.cpp:211/243`](../../../../openvino.genai.mx/src/cpp/src/safetensors_utils/safetensors_loader.cpp#L211)）；`source` 全程同一对象 |
| (c) `get_tensor` 的 `ov::Tensor` view | **各一份对象（指向同一 mmap）** | encoder 编译后 `clear_tensor_cache()` 清掉 view（[`safetensors_weight_source.cpp:121`](../../../../openvino.genai.mx/src/cpp/src/safetensors_utils/safetensors_weight_source.cpp#L121)），decoder 再 `get_tensor` 重建（[`:96-101`](../../../../openvino.genai.mx/src/cpp/src/safetensors_utils/safetensors_weight_source.cpp#L96)） |
| (d) 反量化 f32 host buffer | **各一份（全新分配）** | MoE 把 mmap 值 `cast_vector<float>()` 拷成 `std::vector<float> tmp`，每图各拷一次 |
| (e) 量化后 int4 `ov::Constant` host buffer | **各一份（decoder 重新量化）** | loader 里 `quantize_moe_int4_asym_view` 无条件调、无任何按名缓存（[`modeling_diffusion_gemma_text.cpp:804/848`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L804)）；两图各自 `WeightFinalizer`，`cache_` 存绑定各自 `ov::Model` 的 `ov::Output<ov::Node>`，无法跨图 |
| (f) GPU 设备 buffer | **各一份** | 两次 `compile_model` 各自 `allocate_memory` + 上传 |

> sample 注释（[`modeling_diffusion_gemma.cpp:1664-1677`](../../../../openvino.genai.mx/src/cpp/src/modeling/samples/modeling_diffusion_gemma.cpp#L1664)）自陈：
> 共享 `source` 只为"不读两遍 safetensor"，并刻意 build→compile→**release** 串行，避免两图 `ov::Constant` 同时存在撑爆 host RAM。
> **当前共享的是"加载入口"，不是"加工产物"。**

要把 (e)(f) 也变成"只一份"，必须新增两段机制：路径2 收 (c)(d)(e)，路径1 收 (f)。

---

## 2. 前提：encoder 与 decoder 的权重内容【一致】（共享是安全的）

共享要安全，需三条充要条件全部成立。逐条用源码核对：

1. **取同一份源字节（weight-tying）**：HF DiffusionGemma 把 encoder.language_model 与 decoder.layers **权重绑定**，
   两图权重 key 全是 `model.decoder.*`，从同一 `source` 取到**同一 safetensor 偏移**。
   XML 实测：1009 个语义权重在两图**同名、shape/type 零失配**。
2. **施加相同变换**：两个 finalizer 收**同一个 `quant_config`**（[`modeling_diffusion_gemma.cpp:1628`](../../../../openvino.genai.mx/src/cpp/src/modeling/samples/modeling_diffusion_gemma.cpp#L1628) 只解析一次），
   `group_size_` 由同一 cfg 推出 → MoE 的 reshape / row_base / 分组完全一致。
3. **量化是纯确定性映射**：`quantize_moe_int4_asym_view`（[`:176-228`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L176)）
   只有逐组 min/max/`lround` 算术，**无 RNG / 无 seed / 无随机舍入**。相同输入 + 相同 group_size ⇒ **逐 bit 相同**。

**结论**：除 decoder 独有的 `self_conditioning.*`（~9 MiB，10 个权重，encoder 没有 → 不共享、各自保留）外，
其余 1009 个语义权重 + 全部 MoE 专家权重，encoder/decoder **内容一致**，(e)(f) 可**真正共享**，无错配风险
（仍按 §4.4 加 layout 断言防御）。

---

## 3. 稳定身份：为什么需要、用什么承载

### 3.1 为什么不能靠现有信息

GPU 现有的图内去重（[`constant.cpp:96` `blobMemCache`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L96)）
用**主机数据指针**当 key（[`:94`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L94)），上传后立刻
[`hint_evict`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L149) 释放主机 buffer：
- 指针易失：encoder 的指针在 decoder 编译时早已 evict/失效。
- 两图量化产物本就是两个不同 host buffer，指针天然不同。

XML 实测进一步确认现图**无任何可借用锚点**：

| 类别 | 个数(enc) | 字节 | 跨图能否按名匹配 |
|---|---|---|---|
| 语义命名 `model.decoder.…` | 1009 | 1.55 GiB | ✅ 同名、零失配 |
| 通用命名 `Constant_NNNN`（**MoE 专家权重，88% 内存**） | 1162 | 11.47 GiB | ❌ generic 名交集=0（enc `Constant_53` vs dec `Constant_103937`） |

- MoE 常量上**无 `rt_info`、无 `WeightlessCacheAttribute`、无 .bin 偏移**（grep 计数 0）。

所以必须**主动注入**一个与主机指针无关、跨模型稳定、代表"逻辑同一权重"的身份 ID。

### 3.2 命名机制（追溯：名字在哪里确定）

序列化层名直接取 `get_friendly_name()`：

```cpp
// openvino.mx/src/core/src/xml_util/xml_serialize_util.cpp:1002
layer.append_attribute("name").set_value(node->get_friendly_name().c_str());
// node.cpp:280 — 有显式名用显式名，否则退回 get_name()
// node.cpp:289 — get_name(): description()+"_"+m_instance_id  → "Constant_53"
// node.cpp:57  — m_instance_id 来自进程级原子自增计数器
```

| 权重 | 创建路径 | 是否命名 | 跨图 |
|---|---|---|---|
| 语义权重 | [`safetensors_weight_finalizer.cpp:274` `set_friendly_name(name)`](../../../../openvino.genai.mx/src/cpp/src/safetensors_utils/safetensors_weight_finalizer.cpp#L274) | ✅ safetensors key | **相同** |
| MoE 专家 gate/up/down | [`ops.cpp:80` `ops::constant`](../../../../openvino.genai.mx/src/cpp/src/modeling/ops/ops.cpp#L80) → `make_shared<Constant>(tensor)`，**从不命名** | ❌ 自增名 | **不同** |

### 3.3 身份载体决策

- **方案 A（已否决）**：复用 weightless `WeightlessCacheAttribute` + `wsh::Context`。
  ❌ XML 实测：在线量化常量根本不带该属性、无 .bin 偏移，机制不适用。
- **方案 B（采纳）**：给共享权重打一个**由逻辑位置派生的确定性 ID**。

> ⚠️ **关键约束（用户明确要求）**：**不修改 `ops::constant` 的默认命名行为**。
> `ops::constant` 是所有模型共用的 helper，给它加"自动命名"会污染其它模型的命名规则。
> 因此身份注入**只在 DiffusionGemma 的 MoE loader 内部局部进行**，且**不通过改 friendly_name**，
> 而是写一个 **DiffusionGemma 私有的 `rt_info["shared_weight_id"]`**（见 §4.1），对其它模型与通用 helper 零影响。

---

## 4. 路径2（GenAI 侧）：量化一次 + 主机一份 + 打稳定身份

**改动文件**：
[`modeling_diffusion_gemma_text.cpp`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp)（MoE loader）、
[`modeling_diffusion_gemma.cpp`](../../../../openvino.genai.mx/src/cpp/src/modeling/samples/modeling_diffusion_gemma.cpp)（build/compile 区段、缓存宿主）。

### 4.1 给每个共享权重打稳定身份（局部、私有、不动通用 helper）

在 MoE loader 内，量化产物 `ops::constant(...)` 返回后，对**该节点本身**写私有 rt_info：

```cpp
// 仅在 DiffusionGemma MoE loader 内，对刚建出的 Constant 节点写私有标记。
// 不改 ops::constant；用 shared_weight_id（字符串或其稳定哈希），由权重逻辑位置派生。
static void stamp_dg_id(const Tensor& t, const std::string& id) {
    if (auto node = t.node_ptr())                       // 取底层 ov::Node
        node->get_rt_info()["shared_weight_id"] = id;       // DiffusionGemma 私有键，pass 不会动 rt_info
}
// gate_up loader（:807-812 调用点之后）
stamp_dg_id(gate_w_q_, weight_name + "|gate|w");
stamp_dg_id(gate_s_q_, weight_name + "|gate|s");
stamp_dg_id(gate_z_q_, weight_name + "|gate|z");
stamp_dg_id(up_w_q_,   weight_name + "|up|w");
stamp_dg_id(up_s_q_,   weight_name + "|up|s");
stamp_dg_id(up_z_q_,   weight_name + "|up|z");
// down loader（:850-852 之后）：weight_name + "|down|w" / "|down|s" / "|down|z"
```

- `weight_name` 是 lambda 形参里的 safetensors key（gate_up loader [`:779`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L779)、
  down loader [`:823`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L823)），**两次 build 完全相同**；
  role 后缀固定 → encoder/decoder 同一逻辑权重得到**相同 `shared_weight_id`**。
- 语义权重（attn/embed/layernorm/router/scale/zp）本就同名，可直接用其 friendly_name 当 ID，不必再 stamp（GPU 侧两条都读，见 §5.2）。
- **为什么用 rt_info 而非 friendly_name**：① 不污染通用命名；② rt_info 不被图变换 pass 改写（friendly_name 可能被融合 pass 重命名）；③ DiffusionGemma 私有键，对其它模型透明。

### 4.2 量化产物缓存（跨两次 build 存活）

- 新建 `SharedQuantWeightCache`（放 sample/pipeline 层，**生命周期跨 encoder build + decoder build**），
  key = `shared_weight_id`，value = `{ ov::Tensor weights; ov::Tensor scales; ov::Tensor zps; }`。
- MoE loader 改为：先按 `shared_weight_id` 查缓存；
  - **命中** → 直接复用已量化的 `ov::Tensor`，**跳过 `quantize_moe_int4_asym_view`**；
  - **未命中** → 量化并存入缓存。
- 效果：① decoder 不再重做量化（省 decoder 构建里的量化耗时）；
  ② gate/up/down 的 `ov::Tensor` 在两图间是**同一个对象** —— `ops::constant(tensor,ctx)`
  （[`ops.cpp:80`](../../../../openvino.genai.mx/src/cpp/src/modeling/ops/ops.cpp#L80)）包的是
  `make_shared<Constant>(tensor)`，传同一 `ov::Tensor`（引用计数）⇒ **主机侧自动只一份**（收口 §1 的 (c)(d)(e)）。
- 因 §2 已证两图量化产物**逐 bit 相同**，缓存复用与"各自重量化"**数值完全等价**。

### 4.3 与现有 `clear_tensor_cache` 的关系（必须分清两个缓存）

- encoder 编译后的 `clear_tensor_cache()`（[`safetensors_weight_source.cpp:121`](../../../../openvino.genai.mx/src/cpp/src/safetensors_utils/safetensors_weight_source.cpp#L121)）
  清的是 **mmap view 缓存**（`m_tensor_cache`），为降 host RAM 峰值刻意保留 —— **不要动**。
- `SharedQuantWeightCache` 缓存的是**量化产物**（int4 `ov::Tensor`），是**独立缓存**，
  **绝不能塞进 `m_tensor_cache`**，否则会被这次 clear 误删、导致 decoder 退回重量化。

---

## 5. 路径1（GPU 插件侧）：按身份去重设备 buffer

### 5.1 前置事实（已核对）

同一 `ov::Core` + 同设备下多次 `compile_model` **复用同一个 `cldnn::engine`** ——
engine 由按设备单例 `RemoteContextImpl` 持有
（[`remote_context.hpp:93` `std::shared_ptr<cldnn::engine> m_engine`](../../../../openvino.mx/src/plugins/intel_gpu/include/intel_gpu/plugin/remote_context.hpp#L93)），
`ProgramBuilder` 的 engine 来自 `ctx->get_engine()`（[`plugin.cpp:357`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/plugin.cpp#L357)）。
**所以挂在 engine（或 RemoteContextImpl）上的 cache 天然跨两个 CompiledModel 存活。**

### 5.2 设备权重 cache

```cpp
// 概念结构，挂在 cldnn::engine 或 RemoteContextImpl 上
struct DeviceWeightBank {
    std::mutex mtx;
    // key = 稳定身份（shared_weight_id 或语义权重的 friendly_name）+ layout 指纹（dtype/shape/format）
    std::unordered_map<WeightKey, std::weak_ptr<cldnn::memory>> entries;
};
```
- 用 `weak_ptr` 持有，**不延长设备 buffer 生命周期**；实际 owner 仍是各 CompiledModel 的 `cldnn::data` primitive（见 §6）。

### 5.3 改 `create_data` 走 cache

锚点：[`constant.cpp:77 create_data`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L77)。
在现有 `blobMemCache`（保留，作图内去重）之外，新增 engine 级查表：

1. **取稳定 ID**：先读 `op->get_rt_info()` 里的 `shared_weight_id`；没有再看 `op->get_friendly_name()` 是否为
   "被显式命名的共享权重"（如以 `model.decoder.` 开头）。**两者都拿不到 → 走原逻辑**（申请新 buffer），
   保证非共享常量 / 其它模型零影响。
2. **查 cache**（key = `{id, layout}`）：
   - **命中且 `weak_ptr` 未失效** → 复用该 `cldnn::memory::ptr`，**跳过
     [`allocate_memory`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L105) + memcpy**
     （省一份显存 + 省一次 PCIe 上传），直接 `add_primitive(*op, cldnn::data(primID, shared_mem))`。
   - **未命中** → 走原逻辑（allocate + 上传），然后把 `weak_ptr` 存进 cache。
3. **`hint_evict` 时机不变**：cache 用稳定 ID，不依赖主机指针，evict 不影响命中。

### 5.4 layout 一致性校验

只有 `(id 相同) ∧ (layout 完全相同)` 才复用。§2 已证同名权重 layout 必一致，但加断言防御：
若某天 decoder 对某权重做了不同 reorder，宁可不共享也不能错配 → lock/layout 任一不符即降级为新建。

---

## 6. 生命周期与编译顺序（最易错的部分）

- **encoder 必须先编译，且其设备 buffer 在 decoder 编译时仍存活。** 当前 sample 是
  "build→compile→release **encoder 的 ov::Model**"（[`modeling_diffusion_gemma.cpp:1696-1721`](../../../../openvino.genai.mx/src/cpp/src/modeling/samples/modeling_diffusion_gemma.cpp#L1696)），
  但 `compiled_encoder` 这个 `ov::CompiledModel` **全程存活**（生成阶段 encoder 每 canvas 重跑），
  所以它持有的 `cldnn::data` 设备 buffer **天然在 decoder 编译时还在** → cache 的 `weak_ptr` 能 lock 成功。
  **结论：现有编译顺序已满足前置条件，无需把两图改成同时持有 `ov::Model`。** 释放的只是主机侧 `ov::Constant`（已 evict）。
- **owner 归属**：被复用 buffer 的强引用同时被 encoder 与 decoder 的 `cldnn::data` 持有（共享 `memory::ptr`），
  任一 CompiledModel 析构只减引用计数，最后一个析构才真正释放。cache 的 `weak_ptr` 不参与 owner。
- **线程安全**：`compile_model` 可能并发；cache 操作加锁。本 pipeline 串行编译，锁仅作防御。

---

## 7. 改动清单（按文件，标注是否可独立先行）

| 文件 | 改动 | 路径 | 可独立先行 |
|---|---|---|---|
| [`modeling_diffusion_gemma_text.cpp`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp) `:807-812`/`:850-852` | MoE Constant 后写私有 `rt_info["shared_weight_id"]`（§4.1），**不动 `ops::constant`** | 路径2 | ✅（仅打标记） |
| 同上 loader + [`modeling_diffusion_gemma.cpp`](../../../../openvino.genai.mx/src/cpp/src/modeling/samples/modeling_diffusion_gemma.cpp) | 新建跨两次 build 的 `SharedQuantWeightCache`，loader 命中复用量化 `ov::Tensor`、未命中才量化（§4.2） | 路径2 | ⬜（依赖上一行 ID） |
| [`constant.cpp`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp) `create_data` `:77` | 读 `shared_weight_id`/共享权重 friendly_name → 查 engine cache → 命中复用 / 未命中申请并登记（无 ID → 原逻辑） | 路径1 | ⬜（依赖 ID 落地） |
| `engine.hpp` / `engine.cpp` 或 `RemoteContextImpl` | 挂 `DeviceWeightBank`（含锁），提供 `find/insert(WeightKey)` | 路径1 | ⬜ |

> 注意：清单**不含**对 `ops.cpp` 通用 `constant()` 的任何改动 —— 按用户要求，避免污染其它模型的命名规则。

---

## 8. 验证步骤（待远程机器执行；本轮先不跑）

0. **离线先验（无需上设备）**：只落地 §4.1 的 ID 注入后，dump 两图 XML（`--cache-model`），
   写脚本确认同一逻辑权重在两图带**相同 `shared_weight_id`**（rt_info 会序列化进 XML），shape/type 零失配。
   这一步即可验证"前提满足"。
1. **正确性**：开 `OV_GPU_VERBOSE=4`，跑现有 prompt，确认输出与共享前一致（无 NaN、文本一致）。
2. **去重命中**：`create_data` 命中分支加 `GPU_DEBUG_LOG`；预期 decoder 编译期出现 ~930 条命中
   （decoder 侧本应新建、现复用的权重常量数：MoE 60 + dense/attn/embed/lm_head 等）。
3. **显存**：对比 `MemoryTracker current=` 高水位，预期 38.59GB → ~21–22GB；
   `performance might drop due to memory swap` 计数预期 1116 → ~0（生成期 351 → 0）。
4. **吞吐**：`Throughput: x tokens/s`，预期从 2.99 显著上升。
5. **回归**：未带 ID 的普通模型（非 DiffusionGemma）走原路径，确认零行为变化。

日志降噪：`grep -avE "GPU_Debug|MemoryTracker|check_allocatable|memory swap"`。

---

## 9. 风险与回退

- **风险1：layout/量化参数不一致导致错配** → §5.4 严格 key 校验 + 断言；不一致则不共享（退回各自申请）。
- **风险2：`shared_weight_id` rt_info 在某些图变换 pass 后丢失** → rt_info 通常被 pass 保留（不像 friendly_name 可能被融合 pass 改写）；
  仍在 GPU 侧 `create_data` 取值当下即用（早于多数图后处理），并加"找不到 ID 即降级新建"的兜底。
- **风险3：cache 的 `weak_ptr` 在 encoder buffer 被意外释放后失效** → §6 已论证 `compiled_encoder` 全程存活；
  命中后 lock 失败则降级为新建（加断言定位）。
- **风险4：`SharedQuantWeightCache` 误并入 `m_tensor_cache` 被 `clear_tensor_cache` 清掉** → §4.3，必须是独立缓存。
- **总回退**：路径1 的 cache 查询对"无 ID"常量完全透明（直接走原逻辑），所以即使路径2 没打上 ID，
  也只是退回现状（两份显存），不会出错 —— **两步解耦、可分别上线**。

---

## 10. 落地顺序建议（最小步先行）

1. **第一步（纯增量、可离线自检）**：只做 §4.1 的 ID 注入 + §8 步骤 0 的 XML 离线验证。
   不动 GPU 插件、不动 `clear_tensor_cache`、不动通用 helper。确认"同一权重跨图同 ID"成立。
2. **第二步（路径2 完整）**：加 `SharedQuantWeightCache`（§4.2/§4.3），让量化一次 + 主机一份。
3. **第三步（路径1）**：加 `DeviceWeightBank` + 改 `create_data`（§5），设备 buffer 去重。
4. 每步后跑 §8 对应验证项；任一步可独立回退。
