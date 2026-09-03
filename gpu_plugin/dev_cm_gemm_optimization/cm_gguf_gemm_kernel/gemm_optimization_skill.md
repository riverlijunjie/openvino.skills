# Intel GPU GEMM / DPAS / XMX Kernel 优化方法论

> 本文汇总 `/mnt/river/skills` 全树中与 GEMM、GEMV、矩阵收缩、DPAS/XMX、ESIMD/SYCL、OpenCL、量化矩阵乘、oneDNN/SYCL*TLA、profiling 和框架集成有关的策略。它是一份**证据驱动的设计与验证手册**，不是脱离设备实测即可照抄的固定参数表。
>
> 结论分为三类：
> - **通用原则**：跨 Intel Xe 平台通常成立，但仍需测量。
> - **平台/工具链相关**：依赖设备、驱动、编译器或 API 合法性。
> - **本地实测结论**：来自本 workspace 的 BMG/B580/CM 实验，不能外推到所有 GPU。

## 1. 优化目标与基本模型

### 1.1 GEMM 合同

先固定并记录：

- 数学形式：`C = alpha * A * B + beta * C`，或 batched/grouped/convolution-lowered GEMM；
- `M/N/K`、batch、leading dimension、stride、转置和物理 layout；
- A/B/C/accumulator/epilogue 的数据类型；
- 对量化权重而言，block size、scale/min、zero-point、bit layout、VNNI layout；
- dispatch：GWS/LWS、subgroup size、每 work-item 的 tile；
- 正确性 reference、误差容限、尾部处理和是否需要 fallback。

不要把“输出数值相同”当作完整合同：layout、dtype、执行引擎和端到端成本也属于实现合同。

### 1.2 Roofline

对 dense GEMM，理想计算量为

$$F = 2MNK$$

实际 benchmark 应分别估算或测量：

$$t_{compute}=F/P_{compute},\qquad t_{memory}=B/P_{memory}$$

理想 roofline 延迟是

$$t_{roofline}=\max(t_{compute},t_{memory})$$

而不是二者相加。现实 kernel 由于 load、scoreboard、SLM round-trip、barrier 和依赖链可能无法完全重叠，因此还要检查 `load-to-use`、SBID/load stall、send、XVE/XMX occupancy 和实际 ISA。

推荐输出：arithmetic intensity、ridge point、compute/memory/XMX/XVE 各自时间、bound class、kernel-only latency、packing+kernel latency、端到端 latency 和误差统计。

## 2. 总体决策树：先选实现路径，再调参

1. **建立 baseline**：oneDNN/oneMKL、已有 SYCL/OpenCL kernel 或 scalar reference。
2. **分类**：标准 GEMM、GEMV、batched/grouped GEMM、attention matmul、MoE、量化 GEMM，还是并非 contraction 的其他操作。
3. **library-first**：标准 layout、常见 dtype、足够大的矩阵优先使用 oneDNN/oneMKL/SYCL*TLA；只有需要融合、特殊 layout/量化、库不支持，或实测 custom 更快时再手写 kernel。
4. **语义 gate**：必须确实存在沿 K 的重复乘加；elementwise、普通 reduction、scan、sort、copy、简单 transpose、pointer chasing 不应强行使用 XMX。
5. **合法性 gate**：查询设备 extension/aspect、subgroup、dtype、矩阵形状、alignment、compiler lowering；不能凭产品名假设。
6. **收益 gate**：估算并实测 packing、SLM staging、register/occupancy、tail 和 launch 开销。
7. 对每个候选只改一个主要变量：tile、layout、K-step、pipeline、prefetch、fusion 或 dispatch。
8. 编译、正确性测试、ISA/编译日志检查、重复 benchmark，然后保留最快的**正确**变体。

## 3. 瓶颈分类与对应杠杆

| 类别 | 典型信号 | 首选策略 |
|---|---|---|
| 极端带宽受限 | compute 远小于 memory，带宽接近上限 | 对齐 block load、减少往返、复用、SLM/L1、消除中间 buffer |
| 带宽主导 | memory 明显更大 | 先优化访问和 mandatory dequant/scale，不要增加额外计算 |
| 带宽/计算平衡 | 两侧都占显著时间 | 同时优化访问、指令数和 overlap；重点做 load/compute pipeline |
| XMX 主导 | XMX 忙、XVE 空闲 | 提高 DPAS tile/ILP，减少 accumulator 依赖和无效工作 |
| XVE 主导 | dequant、shuffle、address、转换占主导 | SIMD 化、branchless、预打包、减少 bit extraction，把 MAC 交给 XMX |
| SBID/latency 受限 | XMX/XVE 不满，load/SLM scoreboard stall 高 | prefetch、SLM/L1 reuse、load/compute 分离、减少 barrier/round-trip |
| occupancy/寄存器受限 | GRF 高、spill 或 resident threads 下降 | 缩 tile、减少 accumulator/临时数组、重新平衡 ILP 与 occupancy |
| launch/tail 受限 | 小 M/N/K、奇异 shape、padding 比计算多 | fallback、specialization、batch/grouped、避免为小形状付 packing |

“hard” 在策略表中表示难优化的近似平衡，并非强绑定。越接近 XMX/XVE/memory 的 1:1:1，越应优先做结构性 overlap 和 work reduction，而不是盲目扩大 tile。

## 4. DPAS/XMX 数学、gate 与 operand contract

### 4.1 基本操作

DPAS/XMX 计算：

$$C_{M\times N} \mathrel{+}= A_{M\times K}B_{K\times N}$$

常见 Xe 路径使用 8 行、8 列的基本 systolic tile；具体 `K` 和 N/subgroup 由 dtype、设备和 API 签名决定。OpenCL subgroup matrix builtin 中，`N` 通常等于 subgroup size，常见签名是 `M ∈ {1,2,4,8}`、`N ∈ {8,16}`，而 `K` 按类型固定，例如 FP16/BF16 的 k16、INT8 的 k32、INT4 的 k64；必须以 extension specification 和设备支持为准。

### 4.2 三重 gate

**Semantic gate**：有明确 M/N/K，有足够 K 和 arithmetic intensity，packing/setup/tail 能摊薄，并有 reference/tolerance。

**Legality gate**：设备支持 XMX/DPAS 或对应 joint-matrix/OpenCL extension；subgroup size 与 builtin 一致且可强制；dtype/accumulator 合法；subgroup collective 不经过 divergent control flow；layout、alignment、tail 合法。

**Profitability gate**：分别测 packing、kernel-only、端到端；评估 accumulator GRF、occupancy、SLM 和小形状的固定成本。任一 gate 不成立，都应保留 scalar/SIMD/library fallback。

### 4.3 Operand layout

- A 通常按 DPAS 约定以 row-major 或 thread-local K slice 提供；
- B 必须能被硬件按列的 K 连续片段消费，常需要 VNNI/K-major packing；
- accumulator 通常保持 row-major 语义；
- OpenCL builtin 参数顺序是 `(a, b, acc)`，acc 是最后一个参数；
- FP16/BF16 在某些 OpenCL 接口中通过 signed integer bit reinterpret 传递，不能误用数值转换；
- DPAS 的每个 work-item 只持有矩阵 slice，不能按普通标量矩阵直觉索引；
- subgroup 中所有 work-item 必须执行一致的 matrix builtin。

layout 不是“最后再修”的细节，而是 legality、load 数量、shuffle 成本和最终性能的一部分。任何新 layout 都应配套独立 layout/VNNI probe。

## 5. Tiling、ILP、K-loop 与 occupancy

### 5.1 Tile 选择

从硬件基本 tile 开始，分别 sweep：

- work-group 的 M/N tile；
- 每 subgroup 的 M/N tile 数；
- K-step、K unroll、stage 数；
- subgroup/work-item 数和 dispatch shape。

更大的 tile 提高复用和 XMX 利用率，但同时增加 accumulator、临时 decode buffer、SLM 和 tail 浪费。以 spill、resident threads、XMX active 和实际 latency 共同决定，而不是只看理论 FLOPs。

### 5.2 独立 accumulator

单一 DPAS accumulator chain 可能因 systolic latency 不能满发射。通常从多个独立 accumulator 开始，让后续 DPAS 在前一条链等待时发射；增加到出现寄存器压力或 occupancy 下降后退一步。经验上的“至少 4 条链”是 BMG 实测建议，不是所有 SKU 的硬阈值。

### 5.3 K-loop pipeline

理想循环是：

1. 预取/加载下一 K tile；
2. 对当前 tile reorder/dequant；
3. 发射一个或多个 DPAS；
4. 存储/交换 stage；
5. 只在必要位置同步。

K-loop unroll 可隐藏 DPAS latency，但每一级都可能增加 GRF。split barrier / named barrier 能减少无谓等待，但必须确认 compiler、SPIR-V extension 和 API 合法。

### 5.4 Occupancy 与 GRF

不要把“大 tile”自动等同于“高性能”。register spill、doubleGRF、过多 accumulator 或 SLM staging 可能减少 resident threads，使 load latency 更难隐藏。比较候选时至少记录：GRF count、spill、SLM bytes、barrier count、resident threads/XVE occupancy、XMX active 和 stall breakdown。

## 6. Memory hierarchy、LSC/block I/O 与 SLM

### 6.1 Global/L1/SLM

- 优先让 subgroup/work-items 访问 stride-1、对齐且可合并的连续地址；
- 规则 1D/2D tile 使用 block/LSC load，避免大量 scalar send；
- 对需要跨 work-item 复用、重排或广播的数据使用 SLM；
- SLM 只有在“复用收益 > copy + barrier + bank conflict 成本”时才值得；
- 二维 SLM tile 可通过 padding（常见为第二维 +1）降低规则 bank conflict，但要实测；
- SLM 大小、bank 数、cache 行和 alignment 必须 runtime/平台确认。

### 6.2 Weight-stationary 与激活复用

量化 GEMM 尤其适合让 weight tile 在 SLM 或 cache 中驻留，再由多个 token/row 复用；也可让 activation tile 被多个 N tile 复用。选择哪一个 stationary 方向，取决于 M/N/K、权重是否跨 token 复用、SLM 容量和 load stall。

### 6.3 双缓冲的边界

双缓冲应实现“当前计算与下一 tile load 重叠”，而不是简单复制两份 buffer。需要验证：stage 写入、fence、barrier、stage 切换和尾部。额外 SLM/GRF 可能越过 occupancy cliff；深 pipeline 并不必然快。

## 7. 量化 GEMM/GEMV：INT4/INT8/GGUF

### 7.1 解量化数据流

典型结构是 XVE 负责 unpack、scale/min、shuffle、address，XMX 负责 dense MAC。优先顺序：

1. 让权重物理布局适配 DPAS/VNNI，尽量一次转换、长期复用；
2. 用 SIMD/向量位操作替代热循环 scalar bit extraction；
3. 将 dequant 与 load/DPAS pipeline 重叠；
4. 将 scale/min 和 lookup table 放入合适的 SLM/寄存器/cache；
5. 避免 scalar sub-dword gather 和重复解码；
6. 尽量在 FP16/BF16 输入路径上以 FP32 accumulator 保持精度，除非明确接受 FP16 accumulation。

Q4/Q5/Q6 的 bit layout、block scale 和 VNNI packing 不可互换。每种 quant format 都应有独立 pack/dequant probe、非零随机输入测试和 tail 测试。

### 7.2 GEMM 与 GEMV 分流

- GEMM 的 token/M 足够大时，DPAS、weight-stationary、activation reuse 和大 tile 更容易摊薄固定成本；
- GEMV 或 decode 的 M 很小，DPAS packing/SLM/barrier 可能不划算，常需 SIMD/scalar、专用 subgroup 或预打包权重；
- MoE 还要考虑 token routing、expert 数、grouped/persistent dispatch、expert imbalance 和输出 scatter；
- 不要用一个 kernel 覆盖所有 token length，按 shape 设 dispatch threshold 并保留多个 variant。

### 7.3 GGUF 关注点

GGUF 权重解码的性能瓶颈经常是 layout/bit unpack 和 SLM round-trip，而不是 XMX 峰值。预打包可以降低 kernel 内开销，但必须把转换时间计入端到端；如果权重长期复用，离线或初始化阶段转换通常更合适。

## 8. Fused epilogue、attention、MoE 与结构性优化

- bias、ReLU/GELU/SiLU、scale、clamp 等 epilogue 可在 accumulator 仍在寄存器时融合，但必须检查 spill；
- attention 的 QK、softmax、AV 具有不同数据流，不应盲目按标准 GEMM 复制 tile；要分析 causal/sparse mask、GQA、paged KV 和 softmax stability；
- MoE 适合根据 token/expert 分布选择 dispatch、persistent 或 grouped GEMM；小 expert batch 往往受 launch 和 packing 支配；
- K-parallel 只有在并行收益大于 atomic/reduction 成本时采用，优先考虑 privatization 和 staged reduction；
- persistent workgroup、Hilbert/Z-order walk 可改善 cache/launch/reuse，但必须与 occupancy 和负载均衡共同评估；
- 真正接近 roofline 或 1:1:1 平衡时，fusion、减少中间写回、减少实际 FLOPs/bytes、改变 stationary 方向等结构性变化通常比微调更有价值。

## 9. API 与实现路径

### 9.1 SYCL joint_matrix / ESIMD

- joint_matrix 适合标准化的 subgroup matrix load/mad/store；
- ESIMD 适合显式 GRF、LSC 2D load、VNNI、DPAS、SLM 和 layout 控制；
- CuTe/SYCL*TLA 适合先使用经过封装的 collective builder，再在确有必要时下沉到 atom-level；
- SYCL*TLA 常用 `CollectiveBuilder`、Xe tensor-op、row-major D、`make_cute_packed_stride`，并要求匹配的 `icpx`、AOT/JIT target 和 split-barrier translator flag；
- 任何 API 的 subgroup size、tile shape、AOT target 和 extension 都应以编译日志与运行时 probe 验证。

### 9.2 OpenCL

- 查询 `CL_DEVICE_EXTENSIONS`；
- 为 DPAS kernel 指定 required subgroup size；
- block read/write 需要正确 alignment，不能置于 divergent control flow；
- packed B 的 `uint` 行布局必须与 block-read 语义一致；
- 扩展不存在或 shape 不支持时选择 fallback。

### 9.3 CM/C-for-Metal

CM 没有 OpenCL subgroup 语义；一个 CM thread 可以直接计算整个 vector，因此从 OpenCL subgroup(16) 移植时不能机械保留 local dimension。应重新核对：

- thread 与原 subgroup 的映射；
- `cm_dpas` 的 operand reinterpret/layout；
- half/ushort/char 的合法加载方式；
- `cm_fence(CM_LOCAL_BARRIER)` 必须先于 `cm_barrier()` 的 SLM producer/consumer 模式；
- `cm_ptr_load/store` 的类型限制、`vector_ref` 参数和 `uint` offset；
- SLM round-trip 是否真的减少全局流量，而不是制造额外 scoreboard stall。

## 10. oneDNN、oneMKL 与自定义 kernel 的协作

标准 GEMM/量化 GEMM 应首先以 oneDNN/oneMKL 作为正确性、性能和 layout 的参考。必须检查 `impl_info_str()`，确认没有落入 reference fallback。对于量化模型，常见稳妥路径是“解量化/转换 + oneDNN GEMM”分离 pipeline；只有在融合后的端到端收益能覆盖额外复杂度时，才实现 dequant+DPAS 融合。

自定义实现应明确优于库的原因：特殊 GGUF layout、MoE grouped shape、融合 epilogue、库不支持的 dtype，或实测端到端更快。不要仅凭 kernel-only 数字替换库实现。

## 11. 编译、profiling 与实验纪律

### 11.1 编译证据

每个候选记录：compiler/driver/device、编译选项、AOT/JIT、warning、GRF/spill、SLM、barrier、SIMD/subgroup，以及 binary/ISA 中是否真的出现 DPAS/XMX、block I/O 和预期的 load/store。

### 11.2 Benchmark 证据

- warm-up 后重复多次，报告 median、spread/min-max；
- 交错 A/B，避免时间漂移把候选差异伪装成收益；
- 需要 flush 或进程隔离时明确记录；
- 分离 kernel-only、packing、pre/post-process 和 end-to-end；
- 覆盖代表性 prefill/decode、常见 M/N/K、奇数和 tail shape；
- 只接受在目标设备真实测得的结论。

### 11.3 推荐工具

可使用 device probe、roofline calculator、compile/validate/benchmark/autotune 脚本、GTPin/VTune、ISA/compiler dump、带宽 probe、XMX probe 和框架级 benchmark。工具结果必须与实际 kernel contract 对齐，不能把微基准峰值当作应用性能。

## 12. 正确性与数值验证

最小测试矩阵应包含：

- 非零随机输入；
- `M/N/K` 不可整除 tile/K-step 的 tail；
- 小矩阵、标准矩阵、大矩阵；
- 不同 leading dimension、stride、transpose/layout；
- NaN/Inf guard；
- FP32 reference 与明确的 FP16/BF16 rounding policy；
- `rel RMS`、`max abs`、`argmax` 和端到端输出；
- layout/VNNI 独立 probe；
- SLM 写入、fence、barrier、读取的重复运行稳定性。

FP16/BF16 输入通常用 FP32 accumulator；若为性能采用 FP16 dequant 或 accumulator，必须同步更新 reference 和 tolerance，而不是把误差误判成 kernel bug。

## 13. 已有本地实验的负面结论

以下是 `/mnt/river/ovmx/cm_gguf_gemm_kernel` 中的本地经验，应视为平台相关，不应盲目复制：

- full GGUF Q4_K/Q5_K/Q6_K GEMM 在 Arc B580 上是 SBID/SLM latency bound，XMX 并未饱和；
- consumer-side SLM double buffer 在重要的大 prefill shape 上回归，额外 tile 越过 spill/occupancy cliff；
- 降低 token groups 没有稳定收益，DPAS 数量减少和 barrier 相对成本抵消了 occupancy 变化；
- fp16 decode 临时值没有解决瓶颈，且改变 rounding 后可能违反原有 tight tolerance；
- 更深 SLM pipeline 更慢且某些代码路径错误，不能因为“更多 stage”就默认更好；
- doubleGRF + 更大 C tile 因 resident threads 减少而回归；
- 当前结构若要继续提升，应研究减少 SLM round-trip、改变 weight-staging 或 N-direction reuse，而不是继续盲扫已耗尽的 knob。

这些结果说明：优化失败本身也是证据，必须记录并避免重复尝试。

## 14. 推荐的完整执行模板

### 阶段 A：合同与 baseline

1. 记录 M/N/K、dtype、layout、quant block、epilogue、dispatch。
2. 跑库/旧 kernel/reference，确认非零输入和正确性。
3. probe 设备 extension、subgroup、SLM、GRF、频率和实际带宽/XMX/XVE。

### 阶段 B：瓶颈诊断

1. 计算 roofline 和 arithmetic intensity。
2. 收集 latency、bandwidth、XMX/XVE active、occupancy、GRF/spill、SBID/send/barrier。
3. 判断是 memory、XVE、XMX、latency、occupancy、launch/tail 还是结构问题。

### 阶段 C：实现选择

1. 标准路径：oneDNN/oneMKL/SYCL*TLA。
2. 特殊 contraction：ESIMD/OpenCL/CM DPAS。
3. 小 M 或短 K：SIMD/GEMV/fallback。
4. 量化：预打包、VNNI、向量化解码、weight-stationary 或融合路径。

### 阶段 D：一次一个实验

按瓶颈选择 tile、B layout、SLM reuse、K unroll、独立 accumulator、prefetch、split barrier、fusion 或 dispatch。每次都执行 compile → correctness → ISA → benchmark → record。

### 阶段 E：验收

只有同时满足以下条件才接受：

- 所有目标 shape 正确，tail 和异常输入稳定；
- 编译产物包含预期的 DPAS/block I/O，且无未接受 spill；
- kernel-only 和端到端均有收益，或明确记录 trade-off；
- fallback、构建、运行时 capability gate 完整；
- 文档写清 accepted/rejected changes、风险和下一步。

## 15. 主要来源索引

### Intel GPU 通用优化

- `intel-gpu-kernel-skills/skills/intel-gpu-kernel-optimization/SKILL.md`
- `intel-gpu-kernel-skills/skills/intel-gpu-kernel-optimization/references/routing.md`
- `.../references/operations/matrix-contraction.md`
- `.../references/capabilities/xmx-dpas/{applicability,data-layouts,tuning,failure-modes}.md`
- `.../references/capabilities/{slm,block-io,prefetch,vectorization,autotuning}.md`
- `.../references/contracts/{correctness,benchmarking,numerical-accuracy,evidence-and-claims}.md`

### SYCL*TLA / ESIMD / OpenCL

- `intel-gpu-kernel-skills/skills/sycl-tla/sycl-tla-gemm/SKILL.md`
- `intel-gpu-kernel-skills/skills/sycl-tla/sycl-tla-cute-xe/SKILL.md`
- `applications.ai.intel-skills/skills/opencl/opencl-gemm/SKILL.md`
- `applications.ai.intel-skills/skills/opencl/opencl-gemm/dpas_reference.md`
- `applications.ai.intel-skills/skills/sycl/sycl-dpas-gemm/SKILL.md`

### GPU 架构与 memory

- `gpu.ai.gpuSkills/gpu-architecture-overview/SKILL.md`
- `gpu.ai.gpuSkills/gpu-memory-hierarchy/SKILL.md`
- `gpu.ai.gpuSkills/gpu-xmx-matrix-acceleration/SKILL.md`
- `gpu.ai.gpuSkills/gpu-threading-model/SKILL.md`

### Kernel optimizer 工作流与专项案例

- `kernel_optimizer/skills/optimization-strategy/SKILL.md`
- `kernel_optimizer/skills/roofline/SKILL.md`
- `kernel_optimizer/skills/kernel/kernel-developing/kernel_basic_skills/`
- `kernel_optimizer/skills/kernel/kernel-developing/kernel_specific_skills/esimd-fp16-gemm/`
- `.../quantized-gemm-gemv-patterns/SKILL.md`
- `.../moe-quant-gemm-kernels/`
- `.../onednn-fp16-gemm/SKILL.md`
- `.../sdp-kernels/`、`.../intel-esimd-{qk,qkv,kv}-gemm/`

### 本地工程记忆与 GGUF 实践

- `/memories/repo/cm_kernel_dev.md`
- `/memories/repo/openvino_gguf_notes.md`
- `/mnt/river/ovmx/cm_gguf_gemm_kernel/gemm_kernel_optimization_policy.md`

## 16. 一句话原则

**先证明这是一个值得用矩阵硬件解决的 contraction，再用正确的 layout 把数据送到 DPAS；随后用 roofline 和 stall/occupancy 证据选择唯一合适的杠杆，并以端到端正确性和实测收益决定是否保留。**
