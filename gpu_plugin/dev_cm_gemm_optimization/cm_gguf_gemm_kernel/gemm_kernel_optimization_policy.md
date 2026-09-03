# GEMM Kernel 优化策略与算法政策

> 本文总结 `/mnt/river/kernel_optimizer/skills` 中与 GEMM、量化 GEMM、DPAS/XMX、内存访问、调度、验证和性能分析相关的策略，并将其整理为适用于 `cm_gguf_gemm_kernel/` 的执行规范。
>
> 原始 skill 主要面向 Intel Xe2/Xe3 的 SYCL ESIMD；本文保留其算法原则，并在最后给出迁移到 CM kernel 时的对应关系。文中平台数值是特定平台的实测示例，不应直接当作所有 GPU 的固定阈值。

## 1. 总体原则

GEMM 优化必须遵循闭环：

```text
建立正确且可复现的 baseline
        ↓
profile / roofline 分类
        ↓
提出单一、可证伪的瓶颈假设
        ↓
只改一个优化变量
        ↓
先验正确性，再测性能
        ↓
记录收益或回归，继续下一轮
```

核心要求：

1. **先判断瓶颈，再选算法**：区分计算受限、显存带宽受限、SLM/缓存延迟受限、同步受限、寄存器受限和 launch/调度受限。
2. **一次只做一个结构改变**：同时修改 tile、布局、prefetch 和同步无法归因，也容易掩盖正确性问题。
3. **正确性是硬门槛**：任何 spill、NaN、越界、死锁、布局错位或超出误差门限的版本都不能进入性能比较。
4. **按 shape dispatch**：M/N/K、量化格式、batch/token 数和硬件平台决定最优实现；不要用一个 kernel 覆盖所有 regime。
5. **实测优先于直觉**：已有 GEMM 实验表明 prefetch、walk order、SLM 双缓冲和更大 K-step 都可能反而变慢。

## 2. Roofline 与瓶颈分类

对 `C = A × B`，基本模型为：

- FLOPs：$2MNK$
- 传输字节数：按实际读写的 A、B、C、scale、workspace 计算；不要把不会回写到内存的中间量计入。
- Arithmetic Intensity：$AI = FLOPs / Bytes$
- 计算下界：$t_{compute} = FLOPs / PeakCompute$
- 内存下界：$t_{memory} = Bytes / PeakBandwidth$
- 理想 roofline：$t_{roofline} = \max(t_{compute}, t_{memory})$

分类策略：

| 情况 | 首选方向 |
|---|---|
| 极端带宽受限 | 大块对齐加载、减少重复读、提高复用、消除中间 round-trip；不要优先优化 ALU |
| 带宽主导但仍有明显解量化开销 | 先压缩/融合必需的解量化与索引指令，再优化访问合并 |
| 内存与计算接近平衡 | 同时做 load/compute overlap、缓存/SLM staging 和指令精简 |
| XMX/XVE 计算受限 | 把 MAC 放到 DPAS/XMX；减少 XVE shuffle、转换、地址计算和寄存器搬运 |
| XMX 与 XVE 不平衡 | 将可移动的工作从忙碌管线迁移到空闲管线；同时保证数据供应不断流 |
| 近似 1:1:1 平衡 | 不要沉迷单点微调，优先考虑结构变化：融合、减少 FLOPs/bytes、改变数据流或并行化 |

重要的隐藏约束：roofline 假设内存与计算可完美重叠，但实际的 `load → wait → compute` 会串行化。即使模型标记为 compute-bound，也必须检查 load-to-use stall、scoreboard、send 和缓存命中率。

## 3. 实现路径选择：oneDNN、融合 ESIMD/CM 与 dispatch

### 3.1 量化 GEMM 的第一选择：分离解量化 + oneDNN

对于量化权重 GEMM，默认先评估：

1. 一个简单的带宽型 kernel，把量化权重一次解量化为 BF16/FP16；
2. 使用 oneDNN 的高质量 BF16/FP16 GEMM。

该方案的优势是解量化成本为 $O(NK)$，与 M 无关；当 M 较大或量化格式的 scale/min 解码较复杂时，通常优于每个 M 行都重复解量化的融合 kernel。skill 中 PTL 的 Q4K 示例为：oneDNN BF16 GEMM 19.55 ms，而最佳手写 ESIMD 为 26.93 ms；解量化约 2--3 ms，分离方案具有优势。

只有以下情况优先考虑手写融合 kernel：

- M 很小，额外的解量化 launch 成本占主导；
- oneDNN 不可用、实现回退到慢速 reference kernel，或不支持所需格式；
- 必须把后处理、SiLU、scale/min、scatter 等操作融合以消除大规模中间写回。

创建 oneDNN primitive 后必须检查 `impl_info_str()`；出现 `ref` 时不要把它当作高性能基线。primitive 应按 `(M,N,K,dtype,layout,quant format)` 缓存并复用。

### 3.2 标准 GEMM 的 ESIMD/CM 路径

在需要自定义融合或 oneDNN 性能不足时，采用：

- 大 tile + DPAS/XMX；
- double-GRF/LGRF 解决累加器和 staging 的寄存器压力；
- SLM weight-stationary 复用量化权重；
- 编译期 tile 参数 + host-side dispatch ladder；
- 针对小 M、中 M、大 M 分别选择 GEMV、batched GEMV 或 GEMM。

## 4. Tile、DPAS 与数据布局

### 4.1 DPAS 选择

只要有足够的可批处理 M（通常有效 M 至少约 4），应优先把 MAC 放到 DPAS/XMX；XVE 峰值通常约为 XMX 的四分之一。小 M 或 launch/内存占主导时，DPAS 的 setup 成本可能不划算，必须通过 shape sweep 决定。

Xe2/Xe3 skill 使用的 FP16/BF16 DPAS 约定是 `SD=8, RC=8`，一次调用覆盖约 `M=8, N=16, K=16` 的计算块。具体 CM 接口和 operand 形状以目标驱动/compiler 的可执行 probe 为准，不能只依赖理论形状。

### 4.2 推荐的 tile 层次

对大规模 dense GEMM，参考结构为：

- workgroup tile 约 `M=256, N=256`；
- 每线程负责多个 `8×16` DPAS 输出块；
- K-step 采用中等粒度（BMG 实测 `K_STEP=32`）；
- 两个 sub-step 做寄存器双缓冲；
- C accumulator 使用 FP32，输入可为 FP16/BF16。

大 C tile 可显著提高 DPAS/load 比，但会增加 GRF 压力。不要为了适配标准 GRF 而盲目缩小 tile：skill 的 PTL 实测显示，N-span=16 在标准 GRF 下发生 26,624 B spill 并降至 291 ms，而 doubleGRF 版本约 41 ms。

### 4.3 VNNI 和 no-shuffle 规则

DPAS 的 B operand 必须是 VNNI/K-major packing。优先采用硬件或 SOA 布局直接产生 DPAS 所需顺序：

- 连续的 `B_T[K,N]`：使用硬件 VNNI 2D load；
- `B[N,K]`：以 N 行作为 gather lanes，使 SOA gather 直接形成 VNNI；
- 需要转置的输出：优先使用能匹配 accumulator SOA 形式的 scatter/store；
- 避免在每个 K tile 前用标量循环手工 repack。

对 CM：应保持 `cm_dpas` 所需的 VNNI word order；SLM block read 本身不会自动提供 VNNI 转换，因此写入 SLM 前必须完成 packing，或在 CM 内显式构造满足 operand contract 的 tile。

### 4.4 访问接口的硬约束

采用 2D load/store 时必须验证：

- x offset 的单位与 width/pitch 的单位不同；
- width、height、pitch 是否要求减一；
- tile 宽度、tile 高度、总字节数和对齐要求满足驱动限制；
- FP16 transposed load 在目标驱动上可能不支持，应改用 `uint32` packing 或正确的 gather；
- FP16 2D store 的高度存在限制，必要时拆成多个 store；
- 对非连续访问，block load/store 优先于 gather/scatter；gather/scatter 只用于确实无法规整化的访问。

在 CM 中，首先用小型 layout probe 验证：每个 lane/row 的输入值、VNNI pair 顺序、输出 tile 行列位置，再接入完整 GEMM。

## 5. 量化 GEMM 的核心算法

### 5.1 量化格式解码

常见重建规则：

| 格式 | 解码 |
|---|---|
| Q4_0 | `(nibble - 8) * scale` |
| Q4_K | `scale * nibble - min` |
| Q5_K | `scale * (nibble | (high_bit << 4)) - min` |
| Q6_K | `scale * ((low | (high2 << 4)) - 32)` |
| Q8_0 | `scale * int8` |

量化位平面应尽量在 host 端或一次性预处理成 SIMD/CM 友好的 SOA 布局，避免热循环中的标量 bit extract、scalar sub-dword gather 和重复地址计算。

### 5.2 Weight-stationary SLM 解量化（大 M 首选）

朴素做法是每个 M 行线程都解量化自己的权重 tile，解量化成本变成 $O(MNK)$，M 大时会回归。

推荐算法：

1. workgroup 中少量 producer 线程按 N 行协作加载并解量化一个 K-block；
2. producer 把解量化后的权重 tile 写入 SLM；
3. producer/consumer 同步；
4. 多个 M-parallel consumer 线程从同一 SLM tile 做 DPAS；
5. 确认所有 consumer 已 drain 后，再覆盖 SLM 进入下一个 K-block。

总解量化成本降为 $O(NK)$，并在 M 方向复用。该结构是大 GEMM 的默认方向，但要先做 SLM 预算：SLM allocation、barrier 数、occupancy 和每线程 accumulator 共同决定是否可行。

M-block 在同一线程内进一步复用 SLM tile 是 shape/格式相关的 sweep 参数：某些大 N 格式有收益，另一些小 N 格式因寄存器压力增加而回归；一旦产生 spill，立即淘汰。

### 5.3 GEMV、batched GEMV 与 GEMM 的 dispatch

- M=1 或很小：通常带宽受限，优化重点是一次大块读取、预打包和减少 scalar gather；DPAS 的收益可能只有几个百分点。
- 中等多行 decode：当 M 增加导致解量化 ALU 成为瓶颈时，使用 DPAS，并让一个解量化 tile 服务多个 M 行。
- 大 M prefill：使用 weight-stationary SLM GEMM 或 oneDNN；不要把 GEMV 式的每行解量化直接扩展。

门限必须按格式、N、K、平台重新 sweep。已有 Q8_0 示例中 `M>=4, N>=1024, N<=2048` 是有效起点，而不是通用常量；深 K 或窄 N 可能使 DPAS setup 不值得。

## 6. 内存访问、复用与流水线

### 6.1 复用优先级

1. 大且重复的权重/激活：每个 consumer scope 尽量只读一次；
2. workgroup 内复用：放入 SLM 或依赖高命中率 L1；
3. workgroup 间复用：利用 L3，必要时使用 persistent tile walk；
4. 最后才考虑把数据写回 DRAM。

典型原则是“小 operand 常驻、大 operand 流式”：例如查询/激活 tile 留在寄存器或 SLM，权重 K-block 顺序流过。

### 6.2 Load/compute separation

不要把 `load → 立即等待 → compute` 固定串起来。将数据加载、DPAS、解量化和地址更新安排为可重叠的阶段：

- 当前 tile 计算时发起下一 tile 的 load；
- 将独立的地址计算和 scale 解码放在 load 与 DPAS 之间；
- 对确实需要的 load 使用软件流水，但 prefetch 深度必须 sweep。

注意：BMG 4096 GEMM 的实测结果显示所有显式 prefetch 距离都回归 8--24%，因为该场景已经 compute/XVE 受限；而 oneDNN 在 PTL 选中的 kernel 使用了 A/B 不同的 prefetch 距离 48/56。结论是 **prefetch 不是默认开启项**，必须在目标平台和至少两类 shape 上实测。

### 6.3 Payload CSE 与地址增量

- 2D payload 的静态 base、surface width/height/pitch 只构造一次；
- K-loop 内只更新 x/y offset；
- 用 induction variable `+=`，不要每次重新乘法和组合完整地址；
- page/expert/row 的间接查找移到较粗的循环边界；
- 变量 tile shape 用编译期实例化和 host dispatch，避免 kernel 内运行时分支。

Payload CSE 是已有 FP16 GEMM 中最大的一次低风险收益（约 +7.4%）；而显式 induction variable 在编译器已做 LICM 时可能只有约 +0.3%，仍可保留作为低成本规范。

## 7. SLM、同步与并行化

### 7.1 SLM 的四种角色

不要混淆：

1. **weight-stationary cache**：一次解量化，多行复用；
2. **LUT**：少量 centroid/scale 表发布一次，再按 index gather；
3. **reduction scratchpad**：K-split partial sum 写入 SLM 后归约；
4. **transpose staging**：通过 SLM 把非连续访问转换为连续访问。

先计算各角色字节预算，再决定 split factor、buffer 数量和 workgroup size。SLM allocation 越大，可能驻留的 workgroup 越少。

### 7.2 Barrier 规则

- 所有 work-item 必须以相同次数、相同顺序到达 barrier；
- SLM 写后、跨线程读前必须同步；
- SLM 被下一轮覆盖前，必须确保上一轮 consumer 已读完；
- 不允许跨 workgroup 用 barrier 等待；
- barrier 不能放在 data-dependent 分支或不一致循环中；
- CM 中对应的 `cm_fence(CM_LOCAL_BARRIER)` 应先于 `cm_barrier()`，尤其是 producer 写 SLM、随后 consumer 读取的场景。

### 7.3 Split/named barrier

如果 full barrier 的等待窗口中存在独立工作，可使用 arrive/wait：

```text
producer 写 SLM
arrive
    发起下一 tile 的全局 load
    做 scale/update/compensation
    发起 prefetch
wait
读取 SLM 并执行 DPAS
```

该策略只有在 arrive 与 wait 之间有足够独立工作时才有收益；已有案例中约 50--100 cycles 的窗口较有价值。双缓冲不是无条件收益：如果它额外占用 SLM、降低 Q tile 驻留或引入更多寄存器状态，可能没有收益甚至回归。

CM 没有等价的完整 nbarrier API 时，应优先使用简单且对称的 `cm_barrier`；只有在确认 CM/驱动支持对应 split synchronization 且能证明正确性时才引入。

### 7.4 K-split 与 persistent workgroup

K-split 能缩短单线程 K 方向 critical path，但会付出 partial reduction、SLM/barrier 或 atomic C 累加成本。split factor 不是越大越好，必须 sweep；已有经验中 K_SPLIT=2 常优于 1 和 4。

当 tile 数远大于硬件可驻留 workgroup 数，或小批量 grouped GEMM 的 launch 数量成为瓶颈，可采用 persistent workgroup：

- 启动约等于 XE core 数的 workgroup；
- 每个 workgroup 从全局 atomic counter 获取 tile；
- 按 grid-stride 或 Hilbert-like 顺序执行；
- 每 tile 只做一次 atomic，绝不能按 K-block 做 atomic。

该策略的收益来自 launch amortization 和 L2 权重复用；atomic contention、tail handling 和 SLM 预算必须单独验证。BMG 4096×4096 全部 WGs 同时运行时，Z-order/Hilbert walk 没有收益，甚至明显回归，因此 walk order 只在存在后续 wave、且 L2 warm-up 有意义时尝试。

## 8. 寄存器、occupancy 与 codegen hygiene

### 8.1 GRF 与 tile 取舍

- 大 accumulator tile、双缓冲 operand、深流水会快速消耗 GRF；
- spill 是编译失败级问题，不得拿 spilling binary 测性能；
- doubleGRF 是寄存器压力逃生通道，但会降低每 EU 的线程数，必须实测 occupancy；
- 先检查 spill 和 peak occupancy，再决定缩 tile 还是开 doubleGRF；
- 4 组以上 live payload/tile 可能直接导致 spill 或 kernel load failure。

### 8.2 Occupancy 调整

按 profile 信号选择：

| 信号 | 动作 |
|---|---|
| occupancy 低、idle 高 | 增加线程、减小单线程工作量或切分 tile |
| scoreboard/send stall 高且 occupancy 有余量 | 增加并发线程或 K-split；prefetch 是另一条独立路径 |
| spill/high mov | 缩小单线程 tile、减少 live state，或使用 doubleGRF |
| launch/dispatch 占主导 | 增大单线程工作量、合并小任务或 persistent kernel |
| tail imbalance | 调整二维映射、分组和 tile walk |

“更高 occupancy 总是更好”是错误的；当形状接近架构上限时，增加线程可能只会增加排队和同步开销。

### 8.3 热循环代码规范

- K-loop 内禁止 data-dependent `if`/`switch`/`?:`；用 loop split、compile-time dispatch 或 mask/merge；
- 对固定 trip count 的内层循环显式 unroll；
- 用 SIMD mask + merge 做逐元素选择；
- 稀有边界分支移到外层 coarse loop；
- 采用 zero-cost region/select，避免逐元素寄存器 transpose；
- 先转换为更窄类型，再进行 shuffle/transpose，以减少寄存器流量；
- 标量 sub-dword gather/scatter 不得出现在串行热循环；
- 检查生成 ISA：DPAS 应尽量连续，mov 不应承担真实 transpose，send 不应阻塞每条 DPAS。

## 9. 实验优先级与明确的负面结论

### 9.1 推荐优先级

1. 建立正确 baseline 与 roofline；
2. 消除 spill，确认 GRF 模式；
3. 确定 oneDNN split pipeline 是否胜出；
4. 选择 GEMV/DPAS-GEMV/WS-GEMM dispatch；
5. 确定 DPAS tile、VNNI layout 和 C tile；
6. 做 payload CSE、block/2D load、地址增量；
7. 建立 SLM weight-stationary staging；
8. 进行 load/compute overlap 与同步粒度 sweep；
9. 最后再尝试 persistent/walk、register-bank avoidance、显式 stall 和 K-parallel atomic。

### 9.2 已知容易回归的尝试

- **Prefetch**：compute-bound BMG GEMM 中可能降低 8--24%；
- **过大的 K-step**：BMG `K_STEP=64+` 约降 5--6%，原因是 EU scheduler window 压力；
- **Z-order/column swizzle**：当所有 WG 同时运行时无益，严重时降 25%；
- **L1 uncached hint**：破坏复用，曾降约 63%；
- **增加 payload 对象数量**：节省少量 set_x/set_y 不一定抵消 GRF、访问预测和调度损失；
- **SLM 双缓冲**：若占用关键 SLM/寄存器预算，可能 neutral 或负收益；
- **K-parallel atomic C**：只有 N 并行不足或 SLM/occupancy 真正受限时才值得，且会增加 atomic 写流量和浮点累加顺序误差。

这些是实验结论，不是绝对禁令；重新启用前必须在目标 hardware、目标 shape 上 A/B，并保留结果。

## 10. 正确性和性能验证政策

### 10.1 正确性

每个候选至少覆盖：

- `M=N=K=256` 的快速测试；
- 非零随机输入；
- K/N/M 非 tile 整数倍的 tail；
- 多个量化格式和 scale/min 边界；
- 大 K（累加误差）；
- NaN/Inf 检查；
- 输出布局、VNNI 顺序和 SLM capture 的独立 self-check。

量化 GEMM 应使用 FP32 CPU reference，并按照实际 kernel 的 FP16/BF16 rounding 规则比较：

- 普通 FP16/BF16 GEMM：报告 rel RMS、max abs 和 NaN 数；
- Q4/Q5/Q6：允许的误差必须与量化误差和解量化顺序匹配；
- 如果输出影响 argmax/speculative accept，必须增加 argmax preservation 或端到端 accept-rate 检查；仅看 RMS 不够。

### 10.2 Benchmark

- warmup 至少 20 次；
- compute-bound 至少 100 次，最好 1000 次；
- 计时前让 GPU 频率稳定；
- memory-bound 测试轮换独立 buffer，避免缓存让结果虚高；
- 使用非零随机数据，避免零值特殊路径；
- 记录 time、TFLOPS、GB/s、roofline%、occupancy、XMX/XVE busy、stall、L3/SLM hit、spill 和实际频率；
- 每次只比较一个变量，并在相同频率、驱动和进程条件下做交错 A/B。

### 10.3 CM 专用验证顺序

1. 先编译并运行单独的 `cm_dpas` operand probe；
2. 验证每种 Q 格式的 dequant block 与 host reference；
3. 验证一块 A/B/C 的 tile layout；
4. 验证 SLM write → fence → barrier → read；
5. 再扩大到非整除 tail、多 tile、多 workgroup；
6. 最后才进行完整模型 shape 的性能 profile。

## 11. 针对 `cm_gguf_gemm_kernel/` 的落地建议

当前 CM GGUF kernel 应优先遵循以下路线：

1. **保留统一的 host weight builder/layout**，所有 Q4_K/Q5_K/Q6_K kernel 使用同一份 SG-transposed 权重约定；
2. **prefill 与 decode 分开 dispatch**：小 token 使用 GEMV/小 tile，较大 token 使用 DPAS GEMM；
3. **full GEMM 默认使用真实 `cm_dpas`**，输入 half、累加 float，避免用 scalar dot product 代替原始 XMX 算法；
4. **统一 VNNI packing**：在 CM kernel 内从 SG-transposed layout 构造 DPAS B tile，确保 Q5_K 的 high-bit 位移和 Q6_K 的 2-bit 位域索引经过独立 probe 验证；
5. **优先降低 SLM round-trip 和 scoreboard stall**，但不要只增加 SLM buffer 数；每个改动都要检查 spill、SLM 占用和 barrier 数；
6. **保留现有 A/B 脚本**作为回归资产：`ab_cons_dbuf.py`、`ab_token_groups.py`、`ab_decode_half.py`、`ab_slm_pipe.py`；这些脚本的负面结果本身也是政策的一部分；
7. **针对 full GEMM 的下一轮结构实验**应优先考虑：减少 weight staging 的 SLM 往返、重新评估 OPG/DPAS tile、降低 accumulator/SLM ring 的资源峰值；不要继续盲扫已证伪的 consumer-side SLM double buffer、TOKEN_GROUPS 或 decode-half；
8. **所有优化必须保持 CM 的同步 fence 规则**：SLM producer 写入后先 fence，再 barrier；任何不对称 barrier 都视为 correctness failure。

## 12. 参考来源

主要依据：

- `skills/optimization-strategy/SKILL.md`
- `skills/roofline/SKILL.md`
- `skills/kernel/kernel-developing/kernel_basic_skills/intel-gpu-kernel-opt/SKILL.md`
- `skills/kernel/kernel-developing/kernel_basic_skills/esimd-codegen-hygiene/SKILL.md`
- `skills/kernel/kernel-developing/kernel_basic_skills/dpas-operand-patterns/SKILL.md`
- `skills/kernel/kernel-developing/kernel_basic_skills/esimd-lsc-2d-gather-scatter/SKILL.md`
- `skills/kernel/kernel-developing/kernel_basic_skills/esimd-lsc-slm/SKILL.md`
- `skills/kernel/kernel-developing/kernel_basic_skills/esimd-nbarrier-pipelining/SKILL.md`
- `skills/kernel/kernel-developing/kernel_basic_skills/cross-arch-hw-optimization-patterns/SKILL.md`
- `skills/kernel/kernel-developing/kernel_specific_skills/esimd-fp16-gemm/`
- `skills/kernel/kernel-developing/kernel_specific_skills/quantized-gemm-gemv-patterns/SKILL.md`
- `skills/kernel/kernel-developing/kernel_specific_skills/moe-quant-gemm-kernels/SKILL.md`
- `skills/kernel/kernel-developing/kernel_specific_skills/onednn-fp16-gemm/SKILL.md`
- `skills/kernel/kernel-developing/kernel_specific_skills/onednn-fp8-gemm/SKILL.md`

> 说明：oneDNN 与 ESIMD 的测量值来自原 skill 中的特定 Xe2/Xe3 平台；应用到 CM、Arc B580/BMG 或其他驱动时，必须重新测量 roofline、频率、occupancy、SLM 和 DPAS 利用率。
