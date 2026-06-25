# diffusion_gemma pipeline 执行流程与内存使用分析

> 范围：基于一次完整的 verbose 端到端运行日志（`diffusion_gemma.log`，`OV_VERBOSE=4`，
> 91 MB / 583,636 行，UTF-16LE）分析 `modeling_diffusion_gemma.exe` 的执行流程、各阶段
> 耗时、以及主机/GPU 内存使用情况，并定位性能瓶颈。
> 运行命令：`modeling_diffusion_gemma.exe --model diffusiongemma-26B-A4B-it --device GPU
> --prompt "Why is the sky blue?" --output-tokens 256`
> 设备：Intel(R) Arc(TM) B390 GPU（可用显存 **31.62 GB** = 33,955,131,392 bytes），驱动 30.0.4。
> 关键编译属性：`INFERENCE_PRECISION_HINT=f16`、`KV_CACHE_PRECISION=f16`、
> `GPU_ENABLE_LARGE_ALLOCATIONS=YES`、`GPU_ENABLE_MEMORY_POOL=YES`、`EXECUTION_MODE_HINT=PERFORMANCE`。
> 该 log 为**性能修复之后**的版本（generation 阶段 0 个 `ocl:ref:any`、0 个 `dyn_quan is turned off`，
> 即 down_proj 已走优化 `jit:gemm` 路径；见 [diffusion_gemma_debug.md](diffusion_gemma_debug.md) §6.20）。
> 日期：2026-06-19

---

## 0. TL;DR

- **流程**：encoder/decoder 双图、共享 WeightSource。整体五个阶段：
  ①safetensors zero-copy mmap → ②encoder 图构建 → ③encoder GPU 编译 →
  ④decoder 图构建 → ⑤decoder GPU 编译 → ⑥生成（1 次 encoder pass + block-diffusion 去噪循环）。
- **耗时构成（总 ~14.5 分钟）由「图构建+编译」主导，不是推理**：
  encoder 构建 180 s + 编译 132 s；decoder 构建 192 s + 编译 181 s（合计 **~11.4 分钟**）；
  实际推理仅 encoder pass 45 s + 去噪 86 s（**~2.2 分钟**）。
- **内存是头号性能瓶颈**：模型峰值需要 **38.59 GB** GPU 内存，但设备只有 **31.62 GB**
  → 超出 ~7 GB → OpenVINO 把权重/缓冲回退到 **usm_host**（主机内存经 PCIe 暴露给 GPU）→
  **1116 次 "performance might drop due to memory swap" 警告**。
  累计 **usm_host 分配 42.79 GB**（3090 次），而真正落在显存的 **usm_device 仅 0.44 GB**（261 次）。
  推理阶段仍有 351 次 swap 警告 → **2.99 tokens/s 的吞吐主要被显存不足导致的 host-USM 换页拖垮**，
  而非算子本身。
- **主机内存（RSS）峰值 25.55 GB**，出现在图构建/权重加载期；编译完成后回落到 ~3.6–6.6 GB。

---

## 1. 端到端流程（按日志时间线）

日志里应用层用 `[xxx]` 前缀打点，GPU 插件用 `GPU_Debug:` 前缀。下表是各阶段的行号坐标与耗时：

| 阶段 | 日志行 | 耗时 | 说明 |
|---|---|---|---|
| 配置打印 | 25–26 | — | `mode=text device=GPU generation=on`；`vocab=262144 hidden=2816 layers=30 heads=16 kv=8 experts=128/topk=8 canvas=256` |
| ① safetensors zero-copy mmap | 27–50 | **~21.0 s** | 11 个分片、1047 个 tensor，**无内存拷贝**（mmap），每片 ~2 s |
| ② encoder 图构建 | 51 | **180.1 s** | `inputs=6 outputs=1 variables=60`（60 个 KV-cache 状态变量） |
| ③ encoder GPU 编译 | 51→212986 | **132.1 s** | 含 `clone_and_transform_model` + `CreateSingleLayerPrimitives` + 各编译 pass |
| ④ decoder 图构建 | 212996 | **191.6 s** | `inputs=4 outputs=1 variables=60` |
| ⑤ decoder GPU 编译 | 212996→445633 | **180.6 s** | 同 encoder，但图更大（含 self-conditioning、双向 mask） |
| ⑥a state-share | 445643–511744 | — | encoder→decoder 拷贝 **60 个 Variable**（KV cache 状态共享） |
| ⑥b encoder pass | →511263 | **45.2 s** | 对 19 个 prompt token 跑一次 encoder forward（30 次 SDPA） |
| ⑥c 去噪循环 | 511744→581211 | **85.7 s** | block-diffusion 去噪，提交 256-token canvas |
| 统计 | 581213–581217 | — | `Decoder passes: 1`、`Tokens/decoder-pass: 256`、`Throughput: 2.99 tokens/s` |

> **关键点：构建+编译（~11.4 分钟）远大于推理（~2.2 分钟）**。对 26B 模型这是一次性的图编译成本
> （可被 `--cache-model` 缓存规避），但本次运行未用 model cache，所以每次都重新编译。

### 1.1 模型规模（encoder 图算子统计，日志 56–91 行）

```
gpu_opset::FullyConnected 236     ie_internal_opset::RMS        301
gpu_opset::MoERouterFused  30     ie_internal_opset::RoPE        60
gpu_opset::SDPA            30     gpu_opset::KVCache             60
gpu_opset::ReadValue       60     opset1::Power(RMS variance)   60
opset7::Gelu               30     opset1::Add                  123
                                  Total ops: 3449
```

- 30 层，每层 1 个 SDPA、1 个 MoERouterFused（128 experts/topk8）、1 个 dense-MLP（GELU）。
- **RMS = 301 而非 ~210**：因为 NaN 修复让残差 highway 上的 norm **保持分解**（不被 RMSFusion 融合，
  以保留 f32 标记），所以它们以分解的 `Power/ReduceMean/Sqrt/...` 形式出现
  （见 [diffusion_gemma_debug.md](diffusion_gemma_debug.md) §6.17）。`Power 60 / ReduceMean 30 / Sqrt 30`
  正是这些分解 norm 的痕迹。
- decoder 图（日志 ~212998 起）多了 `Assign 60`（KV-cache 写回）、`CumSum 1`（位置）、更多 `Concat/Broadcast`
  （双向 canvas mask + self-conditioning）。

### 1.2 双图 + 共享权重的设计

- HF DiffusionGemma 把 encoder.language_model 的权重与 decoder.layers **绑定**（只有 self-conditioning 块
  是 decoder 独有）。pipeline 用**同一个 WeightSource** 构建两张图，避免重复加载 safetensors、内存里只保留一份
  （mmap 后端）。日志 `[gen] Reusing embed_tokens from shared WeightSource` 即此。
- **串行 构建→编译→释放**：encoder 图的常量在 decoder 图构建前释放，否则同时持有两张图会让峰值主机内存翻倍
  （≈2× 权重的 `ov::Constant`），在 32 GB 机器上 26B 模型会 OOM。
- encoder（causal，写 KV cache）→ decoder（双向 canvas attention，只读 encoder KV cache）。
  60 个 Variable 经 `query_state`/`set_state` 从 encoder 拷到 decoder（日志 `state-share: copied 60`）。

### 1.3 生成阶段是 block-diffusion，不是自回归

- `canvas_length=256`、`max_denoising_steps=48`、`t_min=0.40 t_max=0.80 entropy_bound=0.10`。
- **本次运行 `Decoder passes: 1`**：自适应停止（AdaptiveStopping）在本 prompt 上**第 1 步就收敛**
  （置信+稳定判据满足），整块 256-token canvas 一次提交。所以去噪循环里只观测到 **30 次 SDPA**
  （= 1 次 decoder infer × 30 层），耗时 85.7 s ——**单次 decoder 全图推理就花了 85.7 s**，
  这与显存换页直接相关（见 §2）。
- 对照 encoder pass：同样 30 次 SDPA、45.2 s（seq_len 仅 19，canvas 256 更大所以 decoder 单次更慢）。

---

## 2. 内存使用分析（核心）

日志有两套内存计量：
1. **主机 RSS**：GPU 插件每个编译 pass 后打 `print_mem_usage_info`（148 条），含 `current RSS` 与 `peak RSS`。
2. **GPU 设备内存（USM）**：`MemoryTracker` 的 `Allocate/Free ... usm_host|usm_device`，以及
   `check_allocatable` 的超额警告。

### 2.1 主机内存（RSS）轨迹

| 里程碑（日志行） | current RSS | peak RSS | 解读 |
|---|---|---|---|
| L93 `clone_and_transform_model` 后 | 16.89 GB | 25.31 GB | 图变换持有权重副本，主机内存高位 |
| L5336 `CreateSingleLayerPrimitives` 后 | 4.05 GB | **25.55 GB** | primitives 创建后释放了变换期副本（delta 为大负值） |
| L122877 build_implementations | 0.15 GB | 25.55 GB | encoder kernel JIT 编译，常量已下放设备 |
| L228854 decoder analyzed_graph | 19.79 GB | 25.55 GB | decoder 图变换期再次拉高 |
| L300428 decoder fuse/optimize | 19.95 GB | 25.55 GB | decoder 编译峰值 |
| L445632 编译结束 | 3.65 GB | 25.55 GB | 进入生成，主机内存回落 |

- **主机 RSS 峰值 = 25.55 GB**（26,786,600 KB），出现在权重加载 + 图变换期；encoder 与 decoder 两个阶段
  的峰值相同（25.55 GB），说明峰值由**单张图的权重展开**决定，串行构建成功避免了 2× 叠加。
- 编译完成后 RSS 回落到 3.6–6.6 GB，生成期稳定，**主机内存不是瓶颈**（机器内存足够）。

### 2.2 GPU 设备内存：超额 → host-USM 回退 → 换页（瓶颈所在）

最关键的一组数据：

| 指标 | 数值 | 来源 |
|---|---|---|
| GPU 可用显存 | **31.62 GB**（33,955,131,392 B） | `check_allocatable` 的 `available memory size` |
| 峰值 GPU 内存需求 | **38.59 GB**（41,435,075,638 B） | `MemoryTracker current=` 高水位 |
| 超出量 | **≈ 7 GB** | 38.59 − 31.62 |
| `usm_host` 累计分配 | **42.79 GB** / 3090 次 | 主机内存暴露给 GPU（PCIe 访问） |
| `usm_device` 累计分配 | **0.44 GB** / 261 次 | 真正落在显存 |
| "memory swap" 警告 | **1116 次** | `engine.cpp:336:check_allocatable` |

**机制**：26B 模型即使经 in-flight INT4 量化（experts）后，权重 + KV cache + 中间激活的总占用
（38.59 GB）仍超过 B390 的 31.62 GB 显存。`GPU_ENABLE_LARGE_ALLOCATIONS=YES` 允许超大单分配，
但当设备显存不够时，OpenVINO GPU 插件把绝大多数缓冲分配为 **`usm_host`**（host unified memory）。
`usm_host` 物理在主机内存、GPU 通过 PCIe 访问 —— 这就是日志反复出现的
`Please note that performance might drop due to memory swap.`。

- **`usm_host`(42.79 GB) ≫ `usm_device`(0.44 GB)**：几乎所有大缓冲（含权重）都退到了主机端，
  只有极少量小缓冲真正驻留显存。这是吞吐只有 2.99 tok/s 的根本原因 —— 计算受 PCIe 带宽而非算力限制。
- 最大的单类分配是 **180 × 121 MB 的 `usm_host`**（= 每层 MoE 专家的 INT4 打包权重块），
  以及 200 × 22 MB、366 × 11.3 MB 等。这些本应在显存的权重大量落到了主机。

### 2.3 换页警告的阶段分布

| 阶段 | swap 警告数 |
|---|---|
| encoder 编译（行 < 212986） | 0 |
| decoder 编译（212996–445633） | **765** |
| 生成（445643–581211） | **351** |

- encoder 编译期 0 次：此时显存尚未被两套权重/状态占满。
- decoder 编译期 765 次：decoder 图把 KV-cache 状态 + 双向 canvas 缓冲叠加进来，显存被压爆，
  开始大量 host-USM 回退。
- **生成期 351 次**：单次 decoder 全图推理（85.7 s）持续踩换页 → 直接拖慢推理。

### 2.4 USM buffer 分配模式

- `allocate_internal_buffer => allocate to usm_host` **480 次**、`=> usm_device` **480 次**：
  每个需要内部 scratch 的 primitive 各分配一对。
- `ocl_memory.cpp:lock: Copy usm_device buffer to host buffer` **60 次**：device→host 回拷
  （= 30 层 × 2，读出结果时的同步拷贝）。
- `set_kernel_arg (usm_host) ...`：kernel 入参直接绑定 usm_host 指针，印证算子在 host 内存上算。

### 2.5 GPU 内存按模块归因（权重/常量，权威）

每条权重/常量在 GPU 上创建时都有一行
`constant.cpp:create_data: [constant:<节点名>] layout: <dtype>:<shape>, mem_ptr(..., <字节> bytes)`，
把字节数与节点名/dtype/shape 绑在一起 —— 这是最可靠的归因来源。全部 **1860 个常量合计 34.64 GB**
（与 §2.2 的 38.59 GB 峰值差额 ≈ 4 GB，是运行时激活/scratch + logits 输出缓冲）。

> 注意：每个权重在 **encoder 图 + decoder 图各创建一份**（两图共享同一 WeightSource，但各自建 GPU 常量），
> 所以下表的 item 数基本是「30 层 × 2 图」=60 的倍数。

| 模块 | GPU 内存 | 占比 | dtype | 说明 |
|---|---|---|---|---|
| **MoE 专家 gate + up + down** | **21.27 GB** | **61%** | u4(INT4) | 120× `u4:128x704x2816`(**独立 gate + up 各一个**，14.18GB) + 60× `u4:128x2816x704`(down, 7.09GB)。HF 的 `gate_up_proj` 在加载时被 `chunk(2)` 拆成独立 gate/up，**不是融合张量**；120=「(gate+up) × 30 层 × 2 图」 |
| embed_tokens / lm_head | 5.50 GB | 16% | f16 | 4× `f16:262144x2816`(=1.408GB)：embed + 绑定的 lm_head，encoder/decoder 各一份 |
| dense MLP gate/up/down | 1.99 GB | 6% | INT4 | 每层 dense MLP（inter=2112），180 项 |
| MoE 专家量化 scale/zp | 1.66 GB | 5% | u4+f16 | `Transpose_*`：`128x704x44`(gate_up) + `128x2816x11`(down) 的量化元数据 |
| attn q_proj | 1.50 GB | 4% | **bf16** | `4096x2816`(sliding) / `8192x2816`(full-attn, head_dim 512)。**未量化** |
| attn o_proj | 1.50 GB | 4% | bf16 | 同上对称 |
| attn k_proj | 0.59 GB | 2% | bf16 | `2048x2816` / `1024x2816` |
| attn v_proj | 0.54 GB | 2% | bf16 | |
| MoE router | 0.04 GB | <1% | f16 | `router.proj.weight`(128×2816) + scale + per_expert_scale |
| self-conditioning MLP | 0.03 GB | <1% | INT4 | decoder 独有 |
| norms(RMS gamma) | 0.002 GB | — | f16 | 543 个 norm 权重（每个仅 2816×2B），可忽略 |
| 运行时缓冲（不在上表） | ~4 GB | — | f16 | `result:logits` 256MB（canvas256×vocab262144）+ KV cache + 中间激活 + 480 对 internal scratch |

**结论：GPU 内存压倒性地花在 MoE 专家权重上 —— 单独 21.27 GB（61%），加上量化元数据 1.66GB 共 ~23GB（66%）。**
其次是 embed/lm_head（5.5GB, 16%）。注意：

- **MoE 专家已是 INT4**（u4，半字节）。若不量化，128 experts × 30 层 × (gate_up 704×2816 + down 2816×704) × 2bytes
  会膨胀到 ~170GB，完全不可行 —— INT4 是让模型能在 GPU 上跑的前提。即便如此 21.27GB 仍是最大头。
- **attention 权重还是 bf16（未量化）**，q+k+v+o 合计 ~4.1GB；若也量化到 INT4 可省 ~3GB。
- **embed/lm_head 重复 4 份**（encoder/decoder × embed/lm_head 解压副本）是可优化点：5.5GB 里有冗余。
- 这些加总（34.64GB 权重）+ 运行时缓冲(~4GB) = 38.59GB 峰值 > 31.62GB 显存 → 触发 host-USM 换页（§2.2）。

> 减显存的最高杠杆：**所有这些权重都被 encoder/decoder 各烤了一份（item 数 = 60 的倍数 = 30 层 × 2 图）。
> 若两图共用一份设备权重，34.64GB → ~17–18GB，直接降到显存以下、消除换页 —— 比"进一步压缩 MoE 专家"杠杆更高、且不损精度。
> 详见 [§4 encoder/decoder 权重复用可行性分析](#4-encoderdecoder-权重复用可行性分析显存减半的最高杠杆)。**

---

## 3. 性能瓶颈结论与优化方向

### 3.1 两类成本

1. **一次性编译成本（~11.4 分钟，占总时 ~78%）**：图构建 + GPU 编译。
   - **直接可省**：用 `--cache-model` 缓存编译产物，二次运行跳过编译。
   - 26B 模型图大（3449 ops/图 × 2 图）、JIT kernel 多，首次编译慢属预期。

2. **推理成本（~2.2 分钟）受显存不足主导**：
   - **根因 = 38.59 GB 需求 > 31.62 GB 显存** → 42.79 GB 缓冲回退到 `usm_host` → PCIe 换页。
   - 这是 [diffusion_gemma_debug.md](diffusion_gemma_debug.md) 里多次提到的「26B-on-34GB-GPU swap」
     现象，与之前 0.78 tok/s、0.21→0.67 tok/s 等数字同源 —— 吞吐天花板由显存带宽而非算子效率决定。

### 3.2 可行的内存/性能优化方向（按收益排序）

1. **【最高杠杆】encoder/decoder 共用一份设备权重**（34.64GB→~17–18GB，直接降到显存以下、消除换页且不损精度）：
   见 [§4](#4-encoderdecoder-权重复用可行性分析显存减半的最高杠杆) 的可行路径与 weightless 可行性结论。
2. **降低显存占用到 < 31.62 GB**（次根本）：
   - 更激进的权重量化（如 lm_head/embeddings 也走 INT4 而非 INT8 backup；当前 `embed_tokens`
     是 INT8_ASYM per-channel，约 1.4 GB）。
   - KV-cache 量化（`KV_CACHE_PRECISION` 当前 f16，可评估 int8）。
   - 减小 `canvas_length` 或激活缓冲（生成期中间激活也吃显存）。
3. **用 model cache 跳过编译**（`--cache-model`）：直接砍掉 ~11 分钟首次成本。
4. **换更大显存的 GPU**：根因是 38.59 GB > 31.62 GB，>40 GB 显存的设备可让权重全驻留 `usm_device`，
   吞吐预计有数倍提升（消除 PCIe 换页）。
5. **算子层面**已基本到位：down_proj 等 compressed FC 已走 `jit:gemm`、MoE 走融合
   `moe_3gemm_swiglu`/`moe_router_fused`、SDPA 走 `ocl::sdpa::opt`、RoPE 走 `ocl::rope::opt`，
   无 `ocl:ref:any` 退化（见 §1 与 [diffusion_gemma_debug.md](diffusion_gemma_debug.md) §6.20）。
   再优化算子收益有限，瓶颈在内存。

### 3.3 生成阶段实际跑的 kernel（日志 445643–581211 统计）

| kernel | 次数 | 备注 |
|---|---|---|
| `execute rms` | 604 | 含分解 + 融合 RMSNorm |
| `execute jit:gemm:any` | 475 | 优化 FC/GEMM（含 down_proj，✅ 无 ref 退化） |
| `execute generic`(eltwise) | 255 | 残差 add 等 |
| `execute ocl::rope::opt_` | 120 | RoPE 优化 kernel |
| `execute ocl::sdpa::opt` | 60 | SDPA 优化（encoder+decoder 各 30） |
| `execute ocl::moe::moe_router_fused_opt_` | 60 | MoE router |
| `execute ocl::moe::moe_*`（3gemm swiglu） | 60 | MoE 专家融合 |
| `execute reduce/activation/gather/...` | — | 辅助算子 |

全部走优化实现，无 `ocl:ref:any`。**性能问题不在 kernel 选择，而在显存换页。**

---

## 4. encoder/decoder 权重复用可行性分析（显存减半的最高杠杆）

### 4.1 问题：同一份权重在设备上烤了两份

HF DiffusionGemma 把 `encoder.language_model` 与 `decoder.layers` 的权重**绑定**（只有
self-conditioning 块是 decoder 独有），所以 encoder/decoder 的 attn / dense-MLP / MoE 专家 /
embed / lm_head 在**数据上逐字节相同**。但当前实现把它们在 GPU 上各存了一份：

- encoder 与 decoder 是**两个完全独立的 `ov::Model`**：
  [`create_diffusion_gemma_encoder_model`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L1536)
  与 [`create_diffusion_gemma_decoder_model`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L1633)
  各自 `BuilderContext ctx` → 各自 `DiffusionGemmaForBlockDiffusion model(ctx, cfg.text)` → 各自 `load_model(...)`。
- MoE loader 每次 build 都调用 `ops::constant(...)`（INT4 量化后烤成 `ov::op::v0::Constant`），两次 build = 两套独立 Constant。
- sample 里 `core.compile_model(encoder_model,...)` 与 `core.compile_model(decoder_model,...)` 分别编译，
  生成两个 `CompiledModel`。

这正是 §2.5 表中「item 数 = 30 层 × 2 图 = 60 的倍数」、以及 log 里「120 个 gate/up 各有不同 mem_ptr」的根源。
若设备上只留一份，权重总量 **34.64 GB → ~17–18 GB**，加运行时 ~4GB ≈ **21–22 GB < 31.62 GB**，
**直接消除 §2.2 的 host-USM 换页（1116 次 swap 警告的根因），顶开 2.99 tok/s 天花板**——比进一步压缩 MoE 杠杆更高、且不损精度。

### 4.2 GPU 插件的去重边界（为什么"自然"不会共享）

| 层级 | 是否去重 | 代码证据 |
|---|---|---|
| 同一 `CompiledModel` 内、相同常量 | ✅ 会复用 | [`constant.cpp:94`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L94) 的 `blobMemCache`，key = `(const char* 主机数据指针, shape, dtype)` |
| **两个 `CompiledModel` 之间** | ❌ **不去重** | `blobMemCache` 是 [`ProgramBuilder` 成员](../../../../openvino.mx/src/plugins/intel_gpu/include/intel_gpu/plugin/program_builder.hpp#L100)；每次 `compile_model` 新建一个 ProgramBuilder，cache 互不可见 → 各自 `allocate_memory` 一块设备 buffer |

**关键**：GPU 设备常量上传是 **per-`CompiledModel`** 的。即使两图的 Constant 指向同一个主机指针，两次编译也各自
`std::memcpy` 上传到各自的 `usm_device`/`usm_host` buffer。所以"共享同一个 C++ Constant 对象"对设备显存**没用**。

### 4.3 weightless 机制能否作为基础？——结论：**只共享主机内存，不共享设备显存**

OpenVINO 确有一套 weight-sharing 机制（`ov::weight_sharing` / `ov::wsh`，
[`weight_sharing_util.hpp`](../../../../openvino.mx/src/core/dev_api/openvino/core/weight_sharing_util.hpp)），且**确实跨模型**：

- `SharedContextManager`（[`shared_context_manager.cpp`](../../../../openvino.mx/src/inference/src/shared_context_manager.cpp)）是
  **Core 级单例**（[`core_impl.cpp:298`](../../../../openvino.mx/src/inference/src/dev/core_impl.cpp#L298) `static SharedContextManager`），
  按 `cache_dir` 哈希分桶（[`core_impl.cpp:880`](../../../../openvino.mx/src/inference/src/dev/core_impl.cpp#L880)）。
  多个模型 import 到同一 cache 目录时，能复用同一个 `wsh::Context`（持有 `WeightSource` = 指向权重源 buffer 的 `weak_ptr`）。
- 它的作用是：weightless 缓存模式（`CacheMode::OPTIMIZE_SIZE` / `enable_weightless`）下，IR blob **不内嵌权重**，
  import 时把原始权重文件 **mmap** 成一块主机 buffer，多个 Constant（甚至多个模型）通过 `SharedBuffer` 指向**同一份主机内存**的不同 offset。

**但这套机制止步于主机内存**——它解决的是"两个模型的 `ov::Constant` 在 RAM 里只留一份"，不解决"GPU 显存只留一份"：

- weightless restore 路径里，每个 data primitive 仍调用
  [`data.hpp:417 allocate_memory(...)`](../../../../openvino.mx/src/plugins/intel_gpu/include/intel_gpu/primitives/data.hpp#L417)
  **各自申请一块全新设备 buffer**，再从共享的主机 mmap 拷进去。
- 搜遍 intel_gpu 树：**无任何跨 `CompiledModel` 的设备 buffer 去重 / weight bank / 共享 USM 常量池**。
  `RemoteContextImpl` 的 `m_memory_cache` 只服务 I/O tensor，不服务权重常量。

> 一句话：**weightless 能让用户的"safetensor 量化后只存一份主机权重、encoder+decoder 复用"在主机侧成立，
> 但到 GPU 这一层，两个 CompiledModel 仍会各申请一组显存** —— 它**不能直接满足"只申请一组 GPU memory 给两个模型共用"**。

### 4.4 为什么不能简单"把权重改成图输入 + 绑同一个 RemoteTensor"

- GPU 的 compressed-FC pass 明确只匹配 `ov::op::v0::Constant` 权重
  （[`convert_fc_to_compressed.cpp:64`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/transformations/convert_fc_to_compressed.cpp#L64)），
  `MOECompressed` 融合算子也要求常量权重输入。一旦权重变运行时输入，优化 `jit:gemm` / 融合 MoE 路径直接失效，
  退回 `ocl:ref:any`（正是 §6.20 刚修好的退化）。
- `If` 双分支合一图也无效：两个分支体各自是独立 Model，权重 Constant 落进两个 body → 同一 CompiledModel 内**仍是两份**。

### 4.5 要"只申请一组 GPU memory 给两个模型共用"，可行路径（按改动量排序）

> 路径 1 + 路径 2 的**可执行详细设计**见 [§6](#6-可执行设计路径2--路径1encoderdecoder-设备权重共享)。

1. **【推荐·符合用户"保留两个模型"意图】扩展 GPU 插件，做跨-CompiledModel 的设备常量去重。**
   思路：在共享同一 `RemoteContext`（同一 `ov::Core` + 同设备本就共享 context）的前提下，给 engine 增加一个
   **device-side weight bank**，key 用 weightless 的稳定身份（`wsh` 的 source_id + bin_offset/size，
   见 [`weight_sharing_util.hpp` `WeightOriginMetaData`](../../../../openvino.mx/src/core/dev_api/openvino/core/weight_sharing_util.hpp#L40)），
   而不是现在 `blobMemCache` 用的易失主机指针。第二个模型编译到同一常量时，
   `allocate_memory` 改为**复用已存在的设备 buffer**（`reinterpret_buffer` / 引用计数）。
   - 前提：encoder 先于 decoder 编译且其设备 buffer 在 decoder 编译时仍存活（当前 sample 是串行 build→release，
     需改为"encoder 编译产物常驻、其权重设备 buffer 进 bank"）。
   - 优点：保留两个 `ov::Model` / 两个 `CompiledModel`，对上层语义零侵入；只动 GPU 插件分配层。
   - 代价：要扩 engine/ProgramBuilder 的常量分配逻辑 + 给 Constant 打稳定身份标记（走 weightless rt_info），并在 GPU 上验证。

2. **【用户原始思路的落地形态】主机侧一份 + 显式共享设备 buffer。**
   加载 safetensor → 量化一次 → 用 `wsh::set_weight_source` 把量化后的权重 buffer 注册进一个跨两模型的 `Context`，
   两图的 Constant 都通过 `wsh::get_buffer` 指向同一主机 buffer（**主机侧已天然只一份**）。
   但**设备侧仍需配合路径 1 的 bank**，否则止步于"主机省一份、显存不省"。可把路径 2 当作路径 1 的主机侧前置。

3. **【架构最干净但最大改】合并成单图 stateful dual-mode 模型**（见下"备选"）：encoder+decoder 合一个 `ov::Model`、
   一个 `CompiledModel`，用输入区分 prefill/decode；权重在图里只出现一次，天然只上传一份。
   - 这是普通 LLM 的标准做法（prefill 与 decode 共用一张图，正是为了权重/KV-cache 算子共用）。
   - 差异都可参数化：mask（causal vs 双向 canvas）来自输入；KV-cache（decoder 已在做 `concat(encoder_k, canvas_k)`）
     统一成 prefill 写 Variable / decode 读 Variable；self-conditioning（decoder 独有，仅 0.03GB）走旁路。
   - 代价：实打实重构两个 builder + 注意力/mask 双模式 + KV-cache 双用；但与"保留两个模型"的用户意图相悖，列为备选。

### 4.6 收益估算

| | 现状（两图，两份设备权重） | 路径 1/2（设备去重） 或 路径 3（合并单图） |
|---|---|---|
| MoE 专家 | 21.27 GB | **~10.6 GB** |
| 量化元数据 | 1.66 GB | ~0.83 GB |
| attn + dense MLP + router | ~6 GB | ~3 GB |
| embed / lm_head | 5.50 GB（4 份） | ~2.75 GB（2 份） |
| 权重合计 | 34.64 GB | **~17–18 GB** |
| + 运行时 ~4GB → 峰值 | 38.59 GB **> 31.62 GB** → 换页 | **~21–22 GB < 31.62 GB** → 无换页 |

→ 预期吞吐从 2.99 tok/s 有数倍提升（消除 PCIe 换页）。

---

## 5. 复现与分析方法（备查）

- 日志为 UTF-16LE + ANSI 颜色码。先转码再分析：
  ```bash
  iconv -f UTF-16LE -t UTF-8 diffusion_gemma.log \
    | sed -E 's/\x1b\[[0-9;]*m//g; s/\r$//' > dg_clean.log
  ```
- 应用层流程：`grep -anE '^\[(diffusion_gemma|config|build|compile|gen|stats)\]'`。
- 主机内存：`grep 'print_mem_usage_info'`，解析 `current RSS` / `peak RSS`。
- GPU 内存：`grep 'already occupied'`（超额）、`MemoryTracker.*current=`（高水位）、
  `Allocate .* usm_(host|device)`（按类型累计）；用 awk 浮点比较避免 32-bit 整数溢出
  （`v=$1+0.0`，`printf "%.0f"`）。
- 换页警告：`grep -c 'performance might drop due to memory swap'`，按行号区间归到各阶段。
- 过滤 GPU 噪声看真实输出：`grep -avE 'GPU_Debug|MemoryTracker|check_allocatable|memory swap'`。

相关文档：[diffusion_gemma_debug.md](diffusion_gemma_debug.md)（NaN 根因 + 性能修复 §6.20）、
[diffusion_gemma_moe_support_cn.md](diffusion_gemma_moe_support_cn.md)（MoE 接入）。

---

## 6. 可执行设计：路径2 + 路径1（encoder/decoder 设备权重共享）

> 目标：保留两个 `ov::Model` / 两个 `CompiledModel`（不合并单图），但让 encoder 和 decoder
> 的**相同权重在 GPU 显存里只存一份**。把 §4 的结论落成可改、可验证的步骤。
> 分两步：**路径2** 在 GenAI 应用侧让量化产物只算一次、主机只一份并打上稳定身份；
> **路径1** 在 GPU 插件侧按该身份把设备 buffer 去重。两步合起来才等于"量化后只存一份、两模型共用一组 GPU memory"。
>
> ⚠️ **本设计尚未实现、未在远程机器验证。** 下方文件/行号为当前代码锚点，供实现时定位。

### 6.0 为什么必须两步、且必须先有"稳定身份"

GPU 现有的常量去重（[`constant.cpp:96`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L96)
的 `blobMemCache`）用 **主机数据指针** 当 key（[`constant.cpp:94`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L94)
`std::make_tuple(data, shape, type)`），而且上传后立刻
[`hint_evict(*op)`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L149) 释放主机 buffer。
两个问题：
1. **指针易失**：encoder 的 Constant 主机指针在 decoder 编译时早已 evict/失效，不能跨模型当 key。
2. **两图的量化产物本就是两个不同的主机 buffer**（loader 各跑一次 `quantize_moe_int4_asym_view` →
   各自 `ops::constant`，见 [`modeling_diffusion_gemma_text.cpp:807`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L807)），
   指针天然不同，纯指针 key 永远不会命中。

所以必须引入一个**与主机指针无关、跨模型稳定、且代表"逻辑同一权重"的身份 ID**。这正是路径2 的核心产出，
也是路径1 去重表的 key。

### 6.0.1 XML 实测验证（dump 两张图对比，2026-06-22）

把已落盘的 `diffusion_gemma_encoder.xml` / `diffusion_gemma_decoder.xml` 解析对比，实测坐实了上面的判断：
现有图里**不存在**可直接用作跨图身份的信息，必须由我们在 build 时主动注入。

**(1) 权重内存大头落在"通用名"常量上，恰恰是没有稳定名字的那批。**

| 常量类别 | 个数(encoder) | 字节 | 跨图能否按名匹配 |
|---|---|---|---|
| 语义命名 `model.decoder.…`（layernorm / router / attn proj / embed / scale / zp） | 1009 | **1.55 GiB** | ✅ 两图名字完全一致、shape/type **零失配** |
| 通用命名 `Constant_NNNN`（**MoE 专家 gate/up/down 权重**） | 1162 | **11.47 GiB（88%）** | ❌ 见 (2) |

每层 MoE 的 gate/up = `u4 [128,704,2816]`（121 MiB）、down = `u4 [128,2816,704]`（121 MiB），全部是
`Constant_NNNN`，占了常量总内存的 **88%**。也就是说，**最想共享的那部分正好落在没有稳定名字的常量上。**

**(2) 通用名在两图之间完全不重合（决定性）。**

```
generic 名字交集 = 0   （enc 1162 个 vs dec 1169 个，无一相同）
同一逻辑权重 layer0 MoE down [128,2816,704]：enc=Constant_53，dec=Constant_103937
```

名字由进程级自增计数器生成，decoder 在 encoder 之后构建，计数器已涨到 10 万+。**靠现有 name 做 ID 必失败。**

**(3) 常量上没有任何可借用的锚点。**

- Const 层**没有 `rt_info`**（XML 里 1234 处 rt_info 全是 `ReduceMean` 上的 `precise` 属性，与权重无关）。
- **没有 `WeightlessCacheAttribute` / `bin_offset` / weightless 属性**（grep 计数 0）—— 印证 §6.0：在线量化产物
  默认不带 weightless 身份，也没有 .bin 偏移可挂靠。

**(4) 但有一个可利用的规律：角色顺序稳定。** 两图都按 `layer0 → layer1 → …` 的顺序发射 MoE 常量，
所以"逻辑身份"完全可由 **层号 + 投影名 + 张量角色（weight/scale/zp）** 确定性重建 —— 而这正是 §6.1.3 揭示的、
modeling loader 在 build 时本就持有的信息。

> **实测结论对 §6.1 的影响：方案 A 出局，方案 B 成为正解。**
> XML 证明 MoE 常量既无 `WeightlessCacheAttribute` 也无 .bin 偏移 → **方案 A（复用 weightless 身份）不可用**；
> 必须走**方案 B：由权重逻辑位置派生一个确定性 ID 并主动写到 Constant 上**。下面 §6.1.3 给出"名字在哪里确定、
> 能否人为设置"的源码级答案。

### 6.0.2 加载链路实际共享边界 + 权重内容是否一致（源码核对）

> 详细独立设计见 [`diffusion_gemma_weight_shared_design.md`](./diffusion_gemma_weight_shared_design.md)。这里给结论与证据。

**(A) 当前 encoder/decoder 的加载链路，实际只共享到 mmap，那一层之后全部各做一份。**

| 阶段 | encoder vs decoder | 证据 |
|---|---|---|
| (a) safetensor 磁盘字节 / OS page cache | **共享** | 单文件 |
| (b) 进程内 mmap 区域 | **共享** | 每文件一个 `shared_ptr<MmapHolder>`，所有 tensor 的 `mmap_info.holder` 指向它（[`safetensors_loader.cpp:211/243`](../../../../openvino.genai.mx/src/cpp/src/safetensors_utils/safetensors_loader.cpp#L211)）；`source` 全程同一对象 |
| (c) `get_tensor` 的 `ov::Tensor` view | **各一份对象（指向同一 mmap）** | encoder 编译后 `clear_tensor_cache()` 清掉 view（[`safetensors_weight_source.cpp:121`](../../../../openvino.genai.mx/src/cpp/src/safetensors_utils/safetensors_weight_source.cpp#L121)），decoder 再 `get_tensor` 重建（[`:96-101`](../../../../openvino.genai.mx/src/cpp/src/safetensors_utils/safetensors_weight_source.cpp#L96)） |
| (d) 反量化 f32 host buffer | **各一份（全新分配）** | MoE 把 mmap 值 `cast_vector<float>()` 拷成 `std::vector<float> tmp`，每图各拷一次 |
| (e) 量化后 int4 `ov::Constant` host buffer | **各一份（decoder 重新量化）** | loader 里 `quantize_moe_int4_asym_view` 无条件调、无任何按名缓存（[`modeling_diffusion_gemma_text.cpp:804/848`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L804)）；两图各自 `WeightFinalizer`，`cache_` 存绑定各自 `ov::Model` 的 `ov::Output<ov::Node>`，无法跨图 |
| (f) GPU 设备 buffer | **各一份** | 两次 `compile_model` 各自 `allocate_memory` + 上传 |

> sample 注释自陈意图（[`modeling_diffusion_gemma.cpp:1664-1677`](../../../../openvino.genai.mx/src/cpp/src/modeling/samples/modeling_diffusion_gemma.cpp#L1664)）：共享 `source` 只为"不读两遍 safetensor"，并刻意 build→compile→**release** 串行避免两图 Constant 同时存在撑爆 host RAM。**当前共享的是"加载入口"，不是"加工产物"——§6 想要的"量化后只存一份、共用一组显存"现状完全没做到。**

**(B) encoder 与 decoder 的实际权重内容【一致】，因此共享在语义上是安全的（核对了三条充要条件）：**

1. **取的是同一份源字节（weight-tying）**：两个 builder 用**同一个 `DiffusionGemmaForBlockDiffusion` 结构**、同一套 packed_mapping、同一句 `load_model(model, source, finalizer)`（encoder [`:1556-1558`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L1556)、decoder [`:1653-1655`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L1653)）。HF DiffusionGemma 把 encoder.language_model 与 decoder.layers **权重绑定**，两图的权重 key 全是 `model.decoder.*`，从同一 `source` 取到**同一 safetensor 偏移**。§6.0.1 的 XML 实测佐证：1009 个语义权重两图**同名、shape/type 零失配**。
2. **施加完全相同的变换**：两个 finalizer 接收**同一个 `quant_config`**（[`modeling_diffusion_gemma.cpp:1628`](../../../../openvino.genai.mx/src/cpp/src/modeling/samples/modeling_diffusion_gemma.cpp#L1628) 只解析一次），`group_size_` 由同一 cfg 推出 → MoE 的 reshape/row_base/分组完全一致。
3. **量化是纯确定性映射**：`quantize_moe_int4_asym_view`（[`:176-228`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L176)）只有 min/max/`lround` 的逐组算术，**无 RNG / 无 seed / 无随机舍入**。相同输入字节 + 相同 group_size ⇒ **逐 bit 相同**的 int4 weights/scales/zps。

> **唯一例外**：decoder 独有的 `self_conditioning.*`（§6.0.1 的 10 个 decoder-only 权重，~9 MiB），encoder 没有 → 不参与共享、各自保留即可。其余 1009 个语义权重 + 全部 MoE 专家权重，encoder/decoder 内容一致、**可安全共享**。
>
> **结论**：内容一致性成立，(e) 主机量化产物与 (f) 设备 buffer 都**可以真正共享**，不存在错配风险（仍按 §6.2.3 加 layout 断言防御）。

### 6.1 路径2（GenAI 侧）：量化一次 + 主机一份 + 打稳定身份

**改动文件**：
[`modeling_diffusion_gemma_text.cpp`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp)
（MoE/权重 loader）、上层 sample
[`modeling_diffusion_gemma.cpp`](../../../../openvino.genai.mx/src/cpp/src/modeling/samples/modeling_diffusion_gemma.cpp)
（build/compile 区段 1693–1759）。

**6.1.1 量化产物缓存（跨两次 build 存活）**

- 新建一个 `SharedQuantWeightCache`（放在 sample 或 pipeline 层，**生命周期跨越 encoder build + decoder build**），
  key = 规范化权重名（如 `model.layers.{i}.mlp.experts.gate_proj` / `...up_proj` / `...down_proj` / 各 attn / embed / lm_head），
  value = `{ ov::Tensor weights; ov::Tensor scales; ov::Tensor zps; uint64_t weight_id; }`。
- MoE loader（[`:775` gate_up](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L775)、
  [`:820` down](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L820)）改为：
  先查缓存；命中则直接复用已量化的 `ov::Tensor`，**跳过 `quantize_moe_int4_asym_view`**；未命中才量化并存入缓存。
  - 这样 decoder build 不再重做量化（省 §1 表里 decoder 构建 191.6s 中的量化部分），且 gate/up/down 的
    `ov::Tensor` 在 encoder/decoder 两图间是**同一个对象**（同一主机 buffer）。
  - 注意：`ops::constant(tensor, ctx)`（[`ops.cpp:80`](../../../../openvino.genai.mx/src/cpp/src/modeling/ops/ops.cpp#L80)）
    包的是 `std::make_shared<ov::op::v0::Constant>(tensor)` —— 传入同一个 `ov::Tensor`，两图的 Constant 会**共享底层 buffer**
    （`ov::Tensor` 是引用计数的），主机侧自动只一份。
  - **与现有 `clear_tensor_cache` 的关系（重要）**：现状下 encoder 编译后会 `clear_tensor_cache()`（§6.0.2-(c)），
    清掉的是 **mmap view** 缓存，那是为降 host RAM 峰值刻意做的，**保留**。`SharedQuantWeightCache` 缓存的是**量化产物**
    （int4 `ov::Tensor`，~每权重几十 MiB），与 mmap view 是两个独立缓存，**不要混进 `m_tensor_cache`**，否则会被这次 clear 误删。
    因为 §6.0.2-(B) 已证 decoder 量化产物与 encoder **逐 bit 相同**，缓存复用与"各自重新量化"在数值上**完全等价**，只是省了 CPU 与一份内存。

GPU 插件要靠它跨模型识别"同一权重"。曾考虑两种实现：

- **方案 A（已否决）**：复用 weightless 身份 —— 给 Constant 写
  [`ov::WeightlessCacheAttribute`](../../../../openvino.mx/src/core/dev_api/openvino/core/rt_info/weightless_caching_attributes.hpp)
  并注册到跨模型 `wsh::Context`，GPU 侧用
  [`wsh::Extension::get_constant_origin`](../../../../openvino.mx/src/core/dev_api/openvino/core/weight_sharing_util.hpp#L95)
  读 `{source_id, bin_offset, size}`。
  - ❌ **§6.0.1 实测否决**：在线量化的 MoE 常量根本不带 `WeightlessCacheAttribute`，也没有 .bin 偏移可挂靠
    （这套机制是给"从 IR 读权重"用的，不是给"运行时计算出的权重"用的）。
- **方案 B（采纳）**：给 Constant 写一个稳定身份。最自然的载体就是 **friendly_name 本身** ——
  序列化时层名就取 `get_friendly_name()`（见 §6.1.3），且语义权重已经用它当稳定名。
  做法：在 6.1.1 的 loader 里，给每个 MoE 量化 Constant 调
  `set_friendly_name(weight_name + role_suffix)`（`role_suffix` ∈ `{_gate_compressed,_gate_scale,_gate_zp,
  _up_compressed,_up_scale,_up_zp,_down_compressed,_down_scale,_down_zp}`）。
  GPU 侧直接用 `op->get_friendly_name()`（或对其做稳定哈希）当身份 key。
  - 备选承载体：若不想把身份混进 friendly_name，可另写 `rt_info["dg_weight_id"]`；但 friendly_name 路线零额外
    字段、且天然进 XML 便于核对（§6.0.1 的对比脚本就能直接验证两图同名）。

> 关键不变量：**同一逻辑权重在 encoder 图和 decoder 图里的 Constant，必须携带相同的 ID。**
> 由 §6.1.3 保证：`weight_name` 是 safetensors key，两次 build 完全相同；`role_suffix` 由角色固定派生 → 同名。

### 6.1.3 layer 名在哪里确定、能否人为设置（源码级追溯）

序列化时，XML 里每个 `<layer name="...">` 直接取节点的 `get_friendly_name()`：

```cpp
// openvino.mx/src/core/src/xml_util/xml_serialize_util.cpp:1002
layer.append_attribute("name").set_value(node->get_friendly_name().c_str());
```

而 `get_friendly_name()` 的取值规则是"**有显式名就用显式名，否则退回自增计数名**"：

```cpp
// openvino.mx/src/core/src/node.cpp:280
const std::string& ov::Node::get_friendly_name() const {
    if (m_friendly_name.empty()) return get_name();   // 没 set 过 → 退回 get_name()
    return m_friendly_name;
}
// openvino.mx/src/core/src/node.cpp:289 — get_name() 的默认名
m_unique_name = description() + "_" + to_string(m_instance_id);  // 如 "Constant_53"
// openvino.mx/src/core/src/node.cpp:57 — m_instance_id 来自进程级原子自增计数器
atomic<size_t> ov::Node::m_next_instance_id(0);
```

由此两类常量的命名差异完全解释清楚：

| 权重 | 创建路径 | 是否 `set_friendly_name` | 结果名 | 跨图 |
|---|---|---|---|---|
| 语义权重（embed/attn/layernorm/router/scale/zp） | [`safetensors_weight_finalizer.cpp:274` `constant->set_friendly_name(name)`](../../../../openvino.genai.mx/src/cpp/src/safetensors_utils/safetensors_weight_finalizer.cpp#L274) | ✅ 用 safetensors key | `model.decoder.…`（稳定） | **相同** |
| MoE 专家 gate/up/down | [`ops.cpp:80` `ops::constant`](../../../../openvino.genai.mx/src/cpp/src/modeling/ops/ops.cpp#L80) → `make_shared<Constant>(tensor)`，**从不 set 名字** | ❌ 退回自增计数 | `Constant_NNNN`（每次 build 不同） | **不同** |

```cpp
// openvino.genai.mx/src/cpp/src/modeling/ops/ops.cpp:80 —— 问题根源：没有命名
Tensor constant(const ov::Tensor& tensor, OpContext* ctx) {
    auto node = std::make_shared<ov::op::v0::Constant>(tensor);  // 无 set_friendly_name
    return Tensor(node, ctx);
}
```

**能否人为设置：可以，而且代价极小。** 两个 MoE loader 的 lambda 形参里都已经有
`const std::string& weight_name`（即 safetensors key，两次 build 完全相同）：

```cpp
// openvino.genai.mx/.../modeling_diffusion_gemma_text.cpp:779（gate_up loader）
//                                          :823（down loader）
... set_weight_loader([this](WeightParameter& param, weights::WeightSource& source,
                             weights::WeightFinalizer& finalizer,
                             const std::string& weight_name,                 // ← 稳定 key 已在手
                             const std::optional<int>& shard_id) { ... });
```

当前 build 出的 Constant 未命名（[`:807-812` gate/up](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L807)、
[`:850-852` down](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp#L850)）。
最小改法：给 `ops::constant` 增一个可选名参数，或在调用后对返回节点 `set_friendly_name`：

```cpp
// 方案：ops::constant 增可选名（ops.cpp）
Tensor constant(const ov::Tensor& tensor, OpContext* ctx, const std::string& name = "") {
    auto node = std::make_shared<ov::op::v0::Constant>(tensor);
    if (!name.empty()) node->set_friendly_name(name);
    return Tensor(node, ctx);
}
// 调用处（gate_up loader :807-812）
gate_w_q_ = ops::constant(gate_q.weights, param.context(), weight_name + "_gate_compressed");
gate_s_q_ = ops::constant(gate_q.scales,  param.context(), weight_name + "_gate_scale");
gate_z_q_ = ops::constant(gate_q.zps,     param.context(), weight_name + "_gate_zp");
up_w_q_   = ops::constant(up_q.weights,    param.context(), weight_name + "_up_compressed");
up_s_q_   = ops::constant(up_q.scales,     param.context(), weight_name + "_up_scale");
up_z_q_   = ops::constant(up_q.zps,        param.context(), weight_name + "_up_zp");
// 调用处（down loader :850-852）：weight_name + "_down_compressed" / "_down_scale" / "_down_zp"
```

因为 `weight_name` 在 encoder build 和 decoder build 里是同一个 safetensors key，`role_suffix` 又固定，
所以两图的同一逻辑权重必然得到**完全相同的 friendly_name** → 满足"同一逻辑权重携带相同 ID"的前提。
改完后用 §6.0.1 的对比脚本（generic 名交集应从 0 变为覆盖全部 MoE 权重）即可离线确认，无需上设备。

> 注意：`set_friendly_name` 只改名字、不改数据/shape/类型，对 GPU 现有 kernel 选择与 `blobMemCache`
> 图内去重均无影响（图内仍按主机指针去重；跨图才用名字）。属纯增量、可独立先行落地的一步。

### 6.2 路径1（GPU 插件侧）：按身份 ID 去重设备 buffer

**前置事实（已核对）**：同一 `ov::Core` + 同设备下，多次 `compile_model` 复用**同一个 `cldnn::engine`**
—— engine 由按设备的单例 `RemoteContextImpl` 持有（[`remote_context.hpp:93` `std::shared_ptr<cldnn::engine> m_engine`](../../../../openvino.mx/src/plugins/intel_gpu/include/intel_gpu/plugin/remote_context.hpp#L93)），
`ProgramBuilder` 拿的 engine 来自 `ctx->get_engine()`（[`plugin.cpp:357`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/plugin.cpp#L357)）。
**所以 engine 级的 bank 天然跨两个 CompiledModel 存活。**

**6.2.1 在 engine（或 RemoteContextImpl）上加一个设备权重 bank**

```
// 概念结构，挂在 cldnn::engine 或 RemoteContextImpl 上
struct DeviceWeightBank {
    std::mutex mtx;
    // key = 稳定身份（方案 B：op->get_friendly_name() 的稳定哈希）+ layout 指纹
    std::unordered_map<WeightKey, std::weak_ptr<cldnn::memory>> entries;
};
```
- 用 `weak_ptr` 持有，**避免 bank 自己延长设备 buffer 生命周期**；实际 owner 仍是各 CompiledModel 的
  `cldnn::data` primitive（见 6.3 生命周期）。

**6.2.2 改 `create_data` 走 bank**

锚点：[`constant.cpp:77 create_data`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L77)。
在现有 `blobMemCache`（保留，作图内去重）之外，新增 engine 级查表：

1. 取稳定 ID = `op->get_friendly_name()`（方案 B，见 §6.1.3）；只对"被显式命名的共享权重"生效 ——
   **名字为空或仍是 `Constant_NNNN` 自增名 → 走原逻辑**（申请新 buffer），保证非共享常量零影响。
   可只对带特定前缀（如 `model.decoder.` 与 MoE 的 `..._gate_compressed` 等）的名字入表，进一步收窄作用域。
2. 用 `WeightKey{id, layout}` 查 engine bank：
   - **命中且 `weak_ptr` 未失效** → 复用该 `cldnn::memory::ptr`，**跳过 `allocate_memory` + `memcpy`**
     （省一份显存 + 省一次 PCIe 上传），直接 `add_primitive(*op, cldnn::data(primID, shared_mem))`。
   - **未命中** → 走原逻辑：`allocate_memory`（[`:105`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L105)）
     + 上传，然后把 `weak_ptr` 存进 bank。
3. **`hint_evict` 时机不变**（仍释放主机 buffer）——bank 用的是稳定 ID，不依赖主机指针，所以 evict 不影响命中。

**6.2.3 layout 一致性校验**

只有 `(id 相同) ∧ (layout 完全相同：dtype/shape/format)` 才复用。DiffusionGemma 里 encoder/decoder 的同名权重
layout 必然一致（同 cfg、同量化参数），但加断言防御 —— 若某天 decoder 对某权重做了不同 reorder，宁可不共享也不能错配。

### 6.3 生命周期与编译顺序（最易错的部分）

- **encoder 必须先编译、且其设备 buffer 在 decoder 编译时仍存活。** 当前 sample 是
  "build→compile→release **encoder 的 ov::Model**"（[`modeling_diffusion_gemma.cpp:1696-1721`](../../../../openvino.genai.mx/src/cpp/src/modeling/samples/modeling_diffusion_gemma.cpp#L1696)），
  但 `compiled_encoder` 这个 `ov::CompiledModel` 是**全程存活**的（生成阶段 encoder 每 canvas 重跑），
  所以它持有的 `cldnn::data` 设备 buffer **天然在 decoder 编译时还在** → bank 里的 `weak_ptr` 能 lock 成功。
  **结论：现有编译顺序已满足前置条件，无需把两图改成同时持有 `ov::Model`。** 释放的只是主机侧 `ov::Constant`（已 evict），
  不影响设备 buffer。
- **owner 归属**：被复用的 buffer 的强引用同时被 encoder 和 decoder 的 `cldnn::data` 持有（共享 `memory::ptr`），
  任一 CompiledModel 析构只减引用计数，最后一个析构才真正释放。bank 的 `weak_ptr` 不参与 owner。
- **线程安全**：`compile_model` 可能并发；bank 操作加锁（6.2.1 的 `mtx`）。本 pipeline 是串行编译，锁仅作防御。

### 6.4 改动清单（按文件）

| 文件 | 改动 | 路径 | 可独立先行 |
|---|---|---|---|
| [`ops.cpp`](../../../../openvino.genai.mx/src/cpp/src/modeling/ops/ops.cpp) `:80` `constant()` | 增可选 `const std::string& name` 形参，非空则 `set_friendly_name`（§6.1.3） | 路径2 | ✅ |
| [`modeling_diffusion_gemma_text.cpp`](../../../../openvino.genai.mx/src/cpp/src/modeling/models/diffusion_gemma/modeling_diffusion_gemma_text.cpp) `:807-812`/`:850-852` loader | 给每个 MoE 量化 Constant 传 `weight_name + role_suffix`（稳定 ID） | 路径2 | ✅（仅命名，不依赖缓存） |
| 同上 loader + [`modeling_diffusion_gemma.cpp`](../../../../openvino.genai.mx/src/cpp/src/modeling/samples/modeling_diffusion_gemma.cpp) + 两个 builder `create_*_model`（`:1536`/`:1633`） | 新建跨两次 build 的 `SharedQuantWeightCache` 并透传；loader 查缓存命中复用量化 `ov::Tensor`、未命中才量化并存（省 decoder 重复量化 + 主机一份） | 路径2 | ⬜（依赖上一行的命名） |
| [`constant.cpp`](../../../../openvino.mx/src/plugins/intel_gpu/src/plugin/ops/constant.cpp) `create_data` `:77` | 取 `op->get_friendly_name()` 当 ID → 查 engine bank → 命中复用 / 未命中申请并登记（名字为空或 `Constant_NNNN` 自增名 → 原逻辑） | 路径1 | ⬜（依赖命名落地） |
| `engine.hpp` / `engine.cpp` 或 `RemoteContextImpl` | 挂 `DeviceWeightBank`（含锁），提供 `find/insert(WeightKey)` | 路径1 | ⬜ |

### 6.5 验证步骤（待远程机器执行；本轮先不跑）

0. **离线先验（无需上设备）**：只落地"命名"那一步后，重新 dump 两图 XML，跑 §6.0.1 的对比脚本，
   确认 MoE 权重的 generic 名交集从 0 变为覆盖全部 `..._gate_compressed`/`..._up_*`/`..._down_*`，
   且两图同名权重 shape/type 零失配。这一步就能验证"前提满足"，再谈后续显存验证。
1. **正确性**：开 `OV_GPU_VERBOSE=4`，跑现有 prompt，确认输出与共享前一致（无 NaN、文本一致）。
2. **去重命中**：`create_data` 命中分支加一行 `GPU_DEBUG_LOG`；预期 decoder 编译期出现 ~930 条命中
   （= decoder 侧本应新建、现在复用的权重常量数；MoE 60 + dense/attn/embed/lm_head 等）。
3. **显存**：对比 `MemoryTracker current=` 高水位，预期 38.59GB → ~21–22GB（§4.6）；
   `performance might drop due to memory swap` 计数预期从 1116 → ~0（生成期 351 → 0）。
4. **吞吐**：`Throughput: x tokens/s`，预期从 2.99 显著上升。
5. **回归**：未带稳定 ID 的普通模型（非 DiffusionGemma）走原路径，确认零行为变化。

### 6.6 风险与回退

- **风险1：layout/量化参数不一致导致错配** → 6.2.3 的严格 key 校验 + 断言；不一致则不共享（退回各自申请）。
- **风险2：friendly_name 在某些图变换 pass 后被改写/丢失** → 方案 B 用 friendly_name 当 ID，需确认它在
  `clone_and_transform_model` 后仍保留（多数 pass 会沿用或合并 friendly_name，但融合类 pass 可能重命名）。
  缓解：在 GPU 侧 `create_data` 取名当下即生效（早于大多数图后处理），且可加"名字必须含约定前缀"的校验；
  若发现被改写，退化为另写 `rt_info["dg_weight_id"]`（pass 不会动 rt_info）作为更稳的承载体。
- **风险3：bank 的 `weak_ptr` 在 encoder buffer 被意外释放后失效** → 6.3 已论证 `compiled_encoder` 全程存活；
  加断言：命中后 lock 失败则降级为新建。
- **总回退**：路径1 的 bank 查询对"无 ID"常量完全透明（直接走原逻辑），所以即使路径2 没打上 ID，
  也只是退回现状（两份显存），不会出错 —— 两步解耦、可分别上线。
