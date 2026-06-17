# sdpa_micro__generate（PagedAttention MIXED）—— 保持布局不变的优化（PTL Xe3）

## 0. 核心结论

**硬约束：不得改变 kernel 输入/输出数据的 layout 与 format。** 只有在保持 PA 的
Q/K/V/输出张量布局与生产 kernel 逐字节一致的前提下做的优化才合格。因此优化空间仅限于
kernel 的启动/分块（tiling）配置及其内部代码——**不包括** I/O 布局。

采用进化式方法（MAP-Elites 参数信赖域搜索），在保持布局不变的前提下，针对 PA MIXED
解码场景（`tokens=16`、`history=512`、`D=128`、GQA 4:1）在 Panther Lake Xe3 上重新搜索
出最优配置。

**结果：配置 `16,16,16,16,8,1,8,1`。** xe3 真正的生产派发是
`xehpc_h128_pa = {16,16,16,16,8,2,8,2}`（`wg_n=2` → `wg_tile_n=32`），它会分块出 **32**
个 query 列，而实际只有 `q_new=16` 列有效——也就是说一半的 KQ/VS micro-GEMM 计算和一半
载入的 Q 都被掩码丢弃。把 `wg_n` 改为 1（`wg_tile_n=16`，恰好等于 `q_new`）即可消除这部分
浪费，**既不改布局，也不触发寄存器溢出崩溃**。

| 指标 | 生产配置 `8,2` | **最优配置 `8,1`** |
|---|---|---|
| warm（稳态）s1 / s8 | 0.0423 / 0.2827 ms | **0.0354 / 0.1660 ms** |
| cold s1 / s8 | 0.0568 / 0.3267 ms | **0.0503 / 0.2220 ms** |
| **warm 加速比** s1 / s8 | — | **1.19× / 1.70×** |
| 平均 DRAM(unique) s1–s10 | 23.4 % | **30.8 %（+31.6 %）** |
| 正确性（rel-L2） | 5.76e-4 PASS | 5.76e-4 PASS |
| s1–s10 全程无崩溃 | ✅ | ✅ |

已在远程 B390/Xe3 上验证（iters=50）。最优配置在 **每一个** batch（s1–s10）上都胜出，
且零 `OpenCL error`；kernel 的 `.cl` 文件保持 **原始未改动**（优化只是 `--cfg`/派发改动，
未改动 kernel 代码）。

## 1. 测量对象

对 OpenVINO GPU 的 `sdpa_micro__generate` kernel
（`SDPAMicroGenerator(prefill=false, gqa_single_token=false)`）做的独立抽取，针对
**PagedAttention MIXED 解码阶段** 配置，在远程 Panther Lake（Xe3）机器上基于
`D:\river\moe\openvino` 编译并运行。

- **为什么是 MIXED。** `get_paged_attention_stage()` 在 `num_tokens != num_seqs`
  **且** 存在 `past_len != 0` 时选用 micro kernel。当每序列有 `tokens` 个新 query
  token（`tokens > 1`）、共 `seqs` 个序列时，`num_tokens = tokens·seqs ≠ seqs`，且
  `history > 0`，因此本文所有配置都会派发带 `IS_PAGED_ATTENTION=1` 的 `micro_sdpa`
  PA generate kernel——这正是 `paged_attention_opt.cpp` 用于多序列解码的路径。（单
  token 解码 `tokens=1` 走的是 `pa_single_token` GENERATE 阶段，不会到这个 micro
  kernel，故有意排除。）

- **kernel 一致性。** 测试直接内联 `sdpa_micro.cl` 及其 batch 头文件、gemmstone
  micro-GEMM 垫片，来源是
  `D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2`
  （`--kernel-dir`）。在 `IS_PAGED_ATTENTION=1 && !IS_PREFILL` 下，主体运行全部
  **四个** microkernel——`ugemm_kq`（缓存 int8 K·Q）、`ugemm_kcq`（新 token f16
  Kc·Q）、`ugemm_vs`（缓存 int8 V·S）、`ugemm_vcs`（新 token f16 Vc·S）——因此测试
  发出四个垫片（`kq=0, vs=1, kcq=2, vcs=3`），所执行代码与 OpenVINO 的 PA generate
  kernel 逐字节一致。

- **真实 PA 输入。** 分页 int8 KV cache（block_size 16，每 token 的 f16 scale+zp 交错
  存放在每个 block 中，`ADJUSTED_*_HEAD_SIZE = head+4`），加上真实的索引缓冲
  （`subsequence_begins`、`past_lens`、`block_indices`、`block_indices_begins`、
  `blocked_indexes_start_and_gws_mapping`），以及 plugin 构建的 12 参数 kernel 顺序、
  派发网格和 `1/sqrt(D)` 的 f16 scale。新 token 以 f16 `Kc`/`Vc` 传入；历史 token 只
  存在于 int8 分页 cache 中。

- **参数。** `tokens = 16`、`seqs ∈ {1..10}`、`history = 512`（`past_len`，向上取整到
  KQ 的 `wg_tile_m = 128`）、`head-dim = 128`、`kv-heads = 8`、`heads = 32`（GQA
  4:1）、int8 BY_TOKEN 压缩 KV。`k_total = past_len + tokens = 528`。

- **冷 cache 平均。** 每个配置跑 N 次迭代（此处 30–50）取平均，同时报告冷 cache 设备
  时间和 warm（驻留 L3）设备时间。为防止较小（~1–11 MB）的 KV 工作集在迭代间一直驻留
  在 16 MB 的 L3 中（那样测到的是 L3 带宽而非真实冷 cache 解码时间），在 **每次** 计时
  的冷迭代前先运行一个 **`cache_flush` kernel**（128 MB 的读-改-写，≥ 4× LLC）清空
  L2/L3。`micro_sdpa` 时间取自它自己的 OpenCL profiling event（已排除 flush），并与
  cliloader 的逐 kernel 平均交叉校验（误差 < 1 %）。**`cold/warm` 比值** 是关键诊断：
  它把 DRAM 流量（仅冷）与计算（冷热都有）分离开。

## 2. 硬件

| | |
|---|---|
| 设备 | Intel(R) Arc(TM) B390 GPU（Panther Lake **Xe3**） |
| `gmdid` | `0x07800004` → xe3 (PTL) |
| Xe-core / EU | 12 core × 8 = 96 EU，systolic/XMX = 1 |
| GPU 频率 | 2400 MHz |
| 内存带宽（峰值） | 112 GB/s（LPDDR5x，规格书；读探测 ≈ 155–160 GB/s） |
| FP16 XMX（峰值） | ≈ 119.5 TFLOP/s @2.4 GHz |
| LLC (L3) | 16 MB（`CL_DEVICE_GLOBAL_MEM_CACHE_SIZE`） |

## 3. 生成的 kernel / microkernel（host JIT）

全部四个 micro-GEMM 由 gemmstone `selectGEMM` 生成并在运行时融进 SPIR-V，与
`init_microkernels()` 完全一致：

| micro-GEMM | 作用 | A ext | A 布局 | scale/zp | 二进制 | grfMin |
|---|---|---|---|---|---:|---:|
| `kq`  | 缓存 int8 K·Q | s8  | N | 有（aqGroupM=1, aqGroupK=D） | 95 376 B | 76 |
| `vs`  | 缓存 int8 V·S | s8  | N | 有（aqGroupM=D, aqGroupK=1） | 36 576 B | 72 |
| `kcq` | 新 token f16 Kc·Q | f16 | T | 无 | 5 984 B | 64 |
| `vcs` | 新 token f16 Vc·S | f16 | N | 无 | 5 744 B | 64 |

micro-GEMM 的分块通过 `--cfg`（8 个整数：`unroll_m/n_kq, unroll_m/n_vs, wg_m/n_kq,
wg_m/n_vs`）配置。生产默认（`xehpc_h128_pa`）为 `unroll={16,16}`、`wg={8,2}` →
`wg_tile=(128, 32)`；**保持布局的最优配置** 用 `wg={8,1}` → `wg_tile=(128, 16)`，
恰好匹配 `q_new=16`，无浪费列。派发 kernel：`micro_sdpa SIMD16 REG128 SLM=17536 B`。

### 配置结构约束（来自 kernel 的 tile/RSELECT 声明）

一个 `--cfg` 结构合法当且仅当同时满足：

1. `wg_tile_n_kq == wg_tile_n_vs`（即 `unroll_n_kq·wg_n_kq == unroll_n_vs·wg_n_vs`）。
2. `wg_tile_m_vs = unroll_m_vs·wg_m_vs == 128`（= `D`；VS 覆盖整个 head）。
3. `sg_per_wg_kq == sg_per_wg_vs`（`wg_m_kq·wg_n_kq == wg_m_vs·wg_n_vs`）。
4. `unroll_m_kq = 16`（其他值会出 NaN）；`unroll_n = 16`（取 32 → grf ≥ 108 →
   `CL_OUT_OF_RESOURCES`，见 §5d）。

当所有 unroll = 16 时，唯一可调旋钮是 `wg_n ∈ {1,2,4}`，即 `wg_tile_n ∈ {16,32,64}`。
由于 `q_new = 16`，`wg_n=1`（`wg_tile_n=16`）是精确匹配——更大的 tile 只会增加被掩码的
无效计算。

## 4. 结果

所有数据均为设备侧 `micro_sdpa` 时间（自身 profiling event），除非标注 *warm* 否则均为
冷 cache。FLOPs = `4·B·heads·tokens·k_total·D`（Q·K + P·V）。**`DRAM(unique)`** =
int8 KV **每个 KV head 只读一次**（GQA 4:1 下真实的冷 cache DRAM 流量）占 112 GB/s 峰值
的百分比。**`KV/head`** 则按 *每个 query head* 计一次读（含 GQA 4:1 复用，由 L3 提供，
故可能超过 DRAM 峰值——这正是 kernel 内 GQA L3 复用的直接指纹）。

### 4a. 保持布局的配置扫描（`wg_n` = `wg_tile_n`/16）

iters = 30，DRAM(unique) 占 112 GB/s roofline 的百分比。全部 40 次运行 PASS
（rel-L2 ≈ 5.8e-4），零崩溃。

| seqs | **`8,1`（wg_tile_n=16）** | `8,2` 生产（n=32） | `8,4`（n=64） | `m_vs=32` `4,2,4,2` |
|----:|---:|---:|---:|---:|
| 1  | **20.6 %** | 18.6 % | 9.1 %  | 14.8 % |
| 2  | **30.5 %** | 17.8 % | 11.5 % | 15.4 % |
| 3  | **26.4 %** | 23.2 % | 12.8 % | 20.3 % |
| 4  | **27.8 %** | 22.6 % | 12.7 % | 19.8 % |
| 5  | **30.0 %** | 20.8 % | 13.7 % | 23.2 % |
| 6  | **32.8 %** | 25.0 % | 12.8 % | 23.4 % |
| 7  | **33.1 %** | 25.5 % | 13.5 % | 25.7 % |
| 8  | **36.3 %** | 25.6 % | 14.4 % | 24.1 % |
| 9  | **33.9 %** | 27.1 % | 14.7 % | 27.1 % |
| 10 | **36.4 %** | 27.4 % | 14.5 % | 26.5 % |
| **平均** | **30.8 %** | 23.4 % | 13.0 % | 22.0 % |

`8,1` 在每个 batch 上都胜出；相对生产 `8,2` **平均 +31.6 %**。`8,4`（`wg_tile_n=64`）
浪费 4× 的列，崩塌；`m_vs=32` 居中。

### 4b. 占用率细化（固定 `wg_tile_n=16`，改变 `sg_per_wg`）

iters = 30。`A` sg8（最优）、`E` sg4（`unroll_m_vs=32`）、`G` sg2（`unroll_m_vs=64`）。
全部 PASS，零崩溃。（`unroll_m_vs=8` / sg16 编译失败，排除。）

| seqs | `A` sg8 = `8,1` | `E` sg4 = `32,16,4,1,4,1` | `G` sg2 = `64,16,2,1,2,1` |
|----:|---:|---:|---:|
| 1  | **20.4 %** | 15.1 % | 10.8 % |
| 4  | 27.9 % | 27.1 % | **33.8 %** |
| 6  | 32.0 % | 35.5 % | **41.3 %** |
| 8  | **37.2 %** | 34.1 % | 33.8 % |
| 10 | 36.3 % | 39.2 % | 39.2 % |
| **平均** | **~31.7 %** | ~31.7 % | ~30.8 % |

三者平均持平，但 `A`（sg8）最稳定，且在低 batch 明显最好（s1：20.4 % vs 10.8 %），
其他配置在低 batch 会让 96-EU 的 GPU 占用率饥饿。**`A` = `16,16,16,16,8,1,8,1` 即最优。**

### 4c. 与生产配置的正面对比（iters = 50，含 warm/驻留 L3 下限）

| seqs | 生产 `8,2` cold / warm | 最优 `8,1` cold / warm | warm 加速比 |
|----:|---:|---:|---:|
| 1 | 0.0568 / 0.0423 ms | **0.0503 / 0.0354 ms** | **1.19×** |
| 8 | 0.3267 / 0.2827 ms | **0.2220 / 0.1660 ms** | **1.70×** |

正确性完全一致（rel-L2 = 5.76e-4，均 PASS）。warm（稳态）增益是真实推理的指标：那正是
生产 `wg_n=2` 为其被掩码的列计算付出代价、而 `wg_n=1` 不付出代价的地方。

### 方法学交叉校验（cliloader）

cliloader 仅报告两个 *被 enqueue* 的 kernel——`micro_sdpa` 与 `cache_flush` 辅助
kernel（`ugemm_kq/kcq/vs/vcs` 是 *内联* 的，不是单独 enqueue）。其逐 kernel 平均与
profiling event 时间一致到 < 1 %。`cache_flush`（~1.05 ms，128 MB RMW）纯属清 cache
开销，已从上面所有 `micro_sdpa` 数字中 **排除**。

## 5. Roofline 分析

### 5a. 用哪条 roofline，以及为何用 `DRAM(unique)`

缓存的 int8 KV 每个 **KV head** 只存一份，但被其 GQA 组的全部 4 个 **query** head 读取。
诚实的冷 cache DRAM 指标按 **每个 KV head 一次** 计字节（`DRAM(unique)` =
`B·Hkv·past_len·2·(D+4)` ≈ s1 时 1.1 MB → s10 时 10.8 MB）；4× 的组内复用由 16 MB 的
L3 提供，而非 DRAM。这就是为什么 `KV/head`（计入全部 4 个 query head 读取）会超过
**112 GB/s**——它测的是 L3，不是 LPDDR5x。所有"占 roofline 百分比"都用
`DRAM(unique)` ÷ 112 GB/s（规格书；读探测约 155 GB/s，故 112 偏保守）。

### 5b. 此场景下 kernel 受计算/延迟约束——`cold/warm` 已证明

这种短上下文解码 **并非** 带宽受限。`cold/warm` 比值（高 batch 下 ≈ 1.3–1.4）把两类
成本分离：

- **warm（驻留 L3）** = matmul + softmax 的计算下限。KQ/VS tile（`128×16×128`）太小，
  无法喂饱 XMX；下限是 **指令 / softmax / barrier 延迟**，而非 flops。
- **cold − warm** = 唯一 KV 的 DRAM 读取，对它所搬运的字节而言 **已接近峰值带宽**。
  读取与计算 **基本串行**（未重叠），这才是真正的上限。

因此 `DRAM(unique) %` 受限于 计算 + 未重叠的首块延迟，而非裸带宽：
`利用率 = unique_bytes / (计算时间 + 未重叠读取)`。

### 5c. 为什么 `wg_n=1` 是关键收益——以及为何 kernel 已接近其天花板

生产 `8,2` 分块 `wg_tile_n=32`，但 `q_new=16`，因此 KQ/VS micro-GEMM 与 Q 载入做了
**2× 的列计算**，其中一半被掩码。降到 `wg_n=1`（`wg_tile_n=16`）即消除该浪费——同时降低
warm（计算）下限与 cold 时间，这就是 1.19–1.70× 的 warm 加速以及 23.4 % → 30.8 % 的平均
DRAM(unique) 提升。

关键在于：在最优配置下，kernel 搬运 KV 已达到 **可达到的单 head 带宽的约 99 %**
（s6+ 时 `KV/head` ≈ 100 % 峰值）。`DRAM(unique)` 仍停在 ~31 %，仅仅是因为 GQA 把每个
KV head 读了 4 次——而这些冗余读取 **大多命中 L3**（1–11 MB 的 KV 集放得进 16 MB 的 L3）。
因此纯配置调优 **几乎没有剩余空间**：唯一的结构性杠杆是消除这 4× 冗余，也就是 §7 中
（被否决）的工作。

### 5d. `CL_OUT_OF_RESOURCES` 约束与无崩溃选择

`unroll_n=32` 会把 VS micro-GEMM 推到 grf ≥ 108–190，并在某些 batch 上以
**`CL_OUT_OF_RESOURCES`（-5）** 放置失败（一种非单调的驱动驻留怪象，并非正确性故障）。
让 **每个 micro-GEMM 都保持 `unroll=16`（grf ≈ 76）** 可彻底消除该失败：所有扫过的
`unroll_n=16` 配置在每个 batch 上都成功放置，零 `OpenCL error`。最优配置
`16,16,16,16,8,1,8,1` 保持 `wg_tile_m_kq = wg_tile_m_vs = 128`、`wg_tile_n = 16 =
q_new`，且寄存器占用最低——是零崩溃约束下的 **可保证安全的最优解**。

## 6. 复现

```bat
:: 编译（MSVC，VS 开发命令行）—— 链接 D:\river\moe\openvino 的 oneDNN/gemmstone 库
call build_test.bat

:: 推荐的保持布局最优配置（wg_n=1），单个 batch：
sdpa_micro_generate_test.exe --tokens 16 --seqs 10 --history 512 ^
  --head-dim 128 --kv-heads 8 --heads 32 --cfg 16,16,16,16,8,1,8,1 --iters 50 ^
  --kernel-dir D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2

:: 与生产默认（wg_n=2）正面对比：
sdpa_micro_generate_test.exe --tokens 16 --seqs 8 --history 512 ^
  --head-dim 128 --kv-heads 8 --heads 32 --cfg 16,16,16,16,8,2,8,2 --iters 50 ^
  --kernel-dir D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2

:: 保持布局的配置扫描 + 占用率细化（s1-s10）：
call run_sweep_nolayout.bat & type sweep_nolayout.txt | findstr /C:"DRAM(unique)" /C:"OpenCL error"
call run_sweep_refine.bat   & type sweep_refine.txt   | findstr /C:"DRAM(unique)" /C:"OpenCL error"
```

每次计时迭代前都先运行 128 MB 的 `cache_flush` kernel，因此每个 `micro_sdpa` 测量都是
冷 cache；flush 已被 event 排除。所有运行都通过
`--kernel-dir D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2`
保证内联的 kernel 来自该代码树。

## 7. 备注与被否决的方案

- **最终交付物。** 保持布局的配置 `16,16,16,16,8,1,8,1`——相对生产默认 **warm 加速
  1.19×（s1）– 1.70×（s8）**、**平均 DRAM(unique) +31.6 %**、正确性完全一致
  （rel-L2 = 5.76e-4）、s1–s10 全程无崩溃。不改 I/O 布局；kernel 的 `.cl` 与上游逐字节
  一致。

- **将该收益产品化**（此处超出范围——按任务策略不提交/不推送）：在
  `sdpa_gen_micro.cpp` 中，让 xe3 PA 解码路径在 `q_new ≤ 16` 时选择 `wg_n=1`
  （`wg_tile_n = q_new`），而非固定的 `xehpc_h128_pa` `wg_n=2`。上线前需对其他 `q_new`
  取值做验证。

- **被否决——kernel 内 KV 共享（保持布局）。** 在一个 work-group 内处理某 KV 组的全部
  4 个 query head（从原生 `[token][head][D]` 布局聚合 Q，K/V 只读一次）是唯一剩下的
  结构性杠杆。但这是一次重大的 DPAS kernel 改写（外层按 block、内层按 head 重构，4×
  的 running max/sum/累加器状态），换来的仅 **~+11 % 且只在 cold** ——稳态 warm 下那 4×
  冗余 KV 读取本就命中 L3（§5c），对真实推理几乎无益。高风险、低回报 → 不实现。

- **被否决——通过 host 重排实现 GQA 共享 KV**（`16,16,16,16,8,4,8,4`）：可达约 34 % 平均
  DRAM(unique)，但需在 host 端把 Q **重排** 为 `[seq][kv_head][token][head_in_group][D]`
  ——这是 kernel **输入布局改动**，违反约束，故被取消资格。

- **被否决——激进配置 `16,32,32,32,4,2,4,2`**：s6 峰值约 48 %，但 `unroll_n=32`
  （grf=190）在 s4/s8 触发 **`CL_OUT_OF_RESOURCES`**。在零崩溃要求下被否决。

- **为何 ~37 % 已接近天花板。** 该解码受计算/延迟约束，存在 `cold/warm ≈ 1.3–1.4` 的
  重叠下限（§5b）；kernel 搬运 KV 已达到可达单 head 带宽的约 99 %（§5c）。要再提高
  DRAM(unique)，要么用被否决的 KV 共享（仅 cold），要么做会破坏与生产 kernel 一致性的
  算法改动（近似/降精度 softmax）。

- `cache_flush` kernel（128 MB RMW，≥ 4× LLC）对诚实测量至关重要——没有它，较小的 KV
  集会在迭代间驻留 L3，测出来的带宽是假的。它每次约 1 ms 的成本已被 event 排除。
