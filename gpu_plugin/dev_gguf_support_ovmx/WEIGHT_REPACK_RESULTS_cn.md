# GGUF 权重 Repack — 实现与实测结果

> 本文是 [WEIGHT_REPACK_DESIGN_cn.md](WEIGHT_REPACK_DESIGN_cn.md) 设计的**代码实现 + 真机实测**，
> 给出在 Intel Arc B580 上的**性能提升**与 **GPU 显存消耗**对比数据。
> 所有数字均为真机测得（非估算），correctness 全部 bit-exact。
>
> **三条路线的实测结论（先给结论）**：
> 1. **Tier-1 LISA 无损重排**：小 block 格式 Q4_0/Q8_0 **×2.7–3.5**；K-quant（Q4_K/Q5_K/Q6_K）中性。
> 2. **Tier-2 Q6_K i8 合并**：bit-exact 但 **×0.76–0.80（更慢）**——证伪了「Q6_K 是 decode-ALU-bound」
>    的设计假设，实测它是 **memory-parallelism-bound**，+30% 字节纯亏。**已放弃**（§6）。
> 3. **Q4_K int8-dp4a（无 repack）**：给 Qwen3 主力格式 Q4_K 补上 Q5_K 早有的 dp4a 路径，
>    kernel 级 **×1.63–1.80、bit-exact、零显存膨胀**——**已集成进 intel_gpu 插件并端到端验证**：
>    Qwen3-8B-Q4_K_M **decode 吞吐 +32.7%（28.83→38.26 tok/s）、TPOT −24.6%**，权重显存 +0（§7ter）。
> 4. **Q6_K LISA size-exact 重排（编译期图变换）**：把 Q6_K 权重在 `compile_model` 期原地字节转置成
>    coalesced 布局（0% 膨胀、bit-exact），decode kernel ×1.8–2.1 转化为端到端 **decode 吞吐
>    Q4_K_M +19.5%（14.64→17.50 tok/s）、Q5_K_M +18.2%（12.88→15.23 tok/s）**（PTL Arc B390，§7quater）。
>    关键是**编译期取代权重**而非首-decode lazy repack——后者的首 token 卡顿会把收益稀释到看不见。
>
> 路线 2→3 的转向源于一条关键实测：K-quant 的瓶颈不在 layout 而在**解码域**（float vs int8 dp4a），
> 与 [RESULTS_dp4a_evolve.md](../dev_kf_distill_master_opt/gguf_kernels/results/RESULTS_dp4a_evolve.md) 的 roofline 诊断一致。

**设备**：Intel Arc B580（Battlemage / Xe2），FP32 14.85 TFLOPS · DRAM 谱峰 456 GB/s · L2 18 MiB。
**方法**：复用 `dev_kf_distill_master_opt/gguf_kernels` 离线 harness——编译 **verbatim OV `.cl`**、
OV 一致的 JIT/GWS/LWS、迭代间 flush L2、CL profiling event 计时；120 iters / 20 warmup，
关键项 150 iters ×3 back-to-back 验证（消除时钟漂移）。这套 harness 正是产出
[RESULTS.md](../dev_kf_distill_master_opt/gguf_kernels/results/RESULTS.md) 的同一套，**此处优化的 kernel 可直接回灌 OV**。

---

## 1. 实现清单（已落地代码）

| 文件 | 作用 |
|---|---|
| [`harness/repack.py`](../dev_kf_distill_master_opt/gguf_kernels/harness/repack.py) | LISA Tier-1 重排（word 粒度 lane 内插 SoA）+ **逆置换可逆性证明**（全 10 格式 byte-equal） |
| [`kernels/fc_gguf_opt_packed.cl`](../dev_kf_distill_master_opt/gguf_kernels/kernels/fc_gguf_opt_packed.cl) | packed 取址的 GEMV kernel（`GGUF_PACKED` candidate 路径），含 Q4_0/Q8_0/Q4_K/Q5_K/Q6_K 解码器 |
| [`harness/ab_repack.py`](../dev_kf_distill_master_opt/gguf_kernels/harness/ab_repack.py) | 同一 binary 内 raw（baseline）vs packed（candidate）A/B：同激活、同 C_ref、同 L2-flush 计时 |
| [`harness/mem_model.py`](../dev_kf_distill_master_opt/gguf_kernels/harness/mem_model.py) | Qwen3-8B 模型级显存膨胀计算（按真实 FC 张量清单） |
| 产物 JSON | [`results/ab_repack.json`](../dev_kf_distill_master_opt/gguf_kernels/results/ab_repack.json) |

**LISA Tier-1 排布**（`repack.py`，与 kernel 取址一一对应）：`SG=16`，`bb=block_byte_size`，
`wpb=ceil(bb/4)`，`bpr=K/elem`，`G=ceil(bpr/16)`。block `kb` → group `g=kb//16`、lane `kb%16`；
其第 `w` 个 4-byte word 落在 uint 下标 `((n*G+g)*wpb+w)*16 + lane`。于是固定 `(g,w)` 时
**16 个 lane 读 16 个连续 uint = 64 B = 一条 cacheline（完全合并）**。kernel 里每个 lane 用
`wpb` 次合并宽读把自己 block 收进 private buffer，再跑**与现状完全相同**的解码 → **值 bit-exact**。

> 设计文档要求的「保留非-repack 路径做 A/B」「编译期一次性、inference 前完成」均已落地：
> baseline = raw kernel，candidate = packed kernel，二者数学一致、仅 layout 不同；repack 在
> harness 中是**计时之外**的一次性步骤（对应 OV `WeightsReorderParams` 在 `compile_model` 期执行）。

---

## 2. 性能对比（真机 B580，M=1 decode GEMV，NROW=1 干净 layout-only A/B）

| 格式 | shape | raw ms | packed ms | **加速** | raw %BW | **packed %BW** | correctness |
|---|---|--:|--:|--:|--:|--:|---|
| **Q4_0** | K4096 N4096 | 0.1386 | 0.0473 | **×2.93** | 15.0% | **48.7%** | relL2(AB)=0 |
| **Q4_0** | K12288 N4096 | 0.4193 | 0.1200 | **×3.49** | 14.8% | **57.5%** | relL2(AB)=0 |
| **Q8_0** | K4096 N4096 | 0.2056 | 0.0761 | **×2.70** | 19.0% | **54.4%** | relL2(AB)=0 |
| **Q8_0** | K12288 N4096 | 0.6112 | 0.1787 | **×3.42** | 19.2% | **69.5%** | relL2(AB)=0 |
| Q4_K | K4096 N4096 | 0.0798 | 0.0844 | ×0.95 | 26.0% | 24.6% | relL2(AB)=0 |
| Q4_K | K4096 N1024 | 0.0302 | 0.0306 | ×0.99 | 17.2% | 17.0% | relL2(AB)=0 |
| Q4_K | K4096 N12288 | 0.1847 | 0.1961 | ×0.94 | 33.7% | 31.7% | relL2(AB)=0 |
| Q4_K | K12288 N4096 | 0.2125 | 0.2098 | ×1.01 | 29.3% | 29.6% | relL2(AB)=0 |
| Q5_K | K4096 N4096 | 0.0904 | 0.1019 | ×0.89 | 28.0% | 24.9% | relL2(AB)=0 |
| Q5_K | K4096 N1024 | 0.0342 | 0.0373 | ×0.91 | 18.6% | 17.0% | relL2(AB)=0 |
| Q5_K | K4096 N12288 | 0.2152 | 0.2236 | ×0.96 | 35.3% | 34.0% | relL2(AB)=0 |
| Q5_K | K12288 N4096 | 0.2543 | 0.2476 | ×1.03 | 29.9% | 30.7% | relL2(AB)=0 |
| Q6_K | K4096 N4096 | 0.1166 | 0.1319 | ×0.88 | 25.9% | 23.1% | relL2(AB)=0 |
| Q6_K | K12288 N4096 | 0.3311 | 0.3445 | ×0.96 | 27.4% | 26.6% | relL2(AB)=0 |

**稳定性**（150 iters ×3 back-to-back）：Q4_0 K12288 = ×3.50/3.50/3.51（中位 **×3.50**）；
Q8_0 K12288 = ×3.45/3.44/3.44（中位 **×3.44**）。波动 < ±0.3%，结论可靠。

**14/14 全部 correctness PASS，且 packed 输出与 raw 输出逐元素 bit-exact（relL2(AB)=0.0e+00）**——
证明 LISA 是纯无损置换，零信息损失，零数值漂移（满足 SPEC §7.2/§7.3）。

### 2.1 关键结论（诚实的瓶颈读数）

- **小 block 格式（Q4_0 18 B / Q8_0 34 B）是 LISA 的主战场，×2.7–3.5 倍**。这正是设计文档预测的
  「瓶颈 A」：18/34 B 既非 64 B 约数又跨 cacheline，raw 路径只有 15–19% BW；LISA 把每条权重 load
  对齐成整 cacheline 合并后冲到 **48–70% BW**——逼近这些工作集尺寸的可达带宽上限。
- **K-quant 格式（Q4_K/Q5_K/Q6_K）LISA Tier-1 中性（×0.88–1.03）**。原因与
  [RESULTS_opt_evolve.md](../dev_kf_distill_master_opt/gguf_kernels/results/RESULTS_opt_evolve.md) 的诊断一致：
  这些 256-elem block（144/176/210 B）**本身连续**，raw 路径下 L2/coalescer 已吸收了「窗口内 strided」
  访问，瓶颈是**解包 ALU（decode-bound）而非访存对齐**。纯重排不改解码 ALU，自然推不动；个别
  shape（Q4_K/Q5_K K12288）的 +1~3% 是合并带来的边际改善，而 N4096/N12288 的 −5~12% 是 private-gather
  的轻微开销（见 §4）。
- **推论（指导后续）**：K-quant 要提速必须走**设计文档的 Tier-2（位平面合并）/ dp4a 整数路径**
  ——即「更便宜的解码」，而非 layout 重排。LISA Tier-1 对 K-quant 的价值是「为 Tier-2 提供已对齐的
  平面输入」，而非自身提速。

---

## 3. GPU 显存消耗对比

### 3.1 逐格式（每行字节，K=4096；`repack.py` 实测、逆置换证明无损）

| 格式 | raw B/行 | packed B/行 | **膨胀** | 膨胀来源 |
|---|--:|--:|--:|---|
| Q4_K | 2304 | 2304 | **+0.00%** | wpb 整除，零 pad |
| Q5_K | 2816 | 2816 | **+0.00%** | 同上 |
| Q6_K | 3360 | 3392 | **+0.95%** | 210 B → 53 word 的 2 B 尾 pad |
| Q8_0 | 4352 | 4608 | +5.88% | 34 B → 9 word 的 2 B pad（小 block） |
| Q4_0 | 2304 | 2560 | +11.11% | 18 B → 5 word 的 2 B pad（小 block） |
| Q3_K | 1760 | 1792 | +1.82% | 110 B → 28 word |
| IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S | — | — | +1.8~2.7% | 同 4-byte word 对齐 |

> 膨胀**只来自 4-byte word 对齐的尾部 pad**（≤2 B/block），与 block 大小成反比——所以小 block
> 的 Q4_0/Q8_0 相对膨胀大，但它们正是换来 ×3 加速的格式；Qwen3 主体的 Q4_K/Q5_K **零膨胀**。

### 3.2 模型级（Qwen3-8B，`mem_model.py`，按真实 FC 张量清单 L=36）

| 模型 | 主导格式 | raw 权重 | packed 权重 | **模型级膨胀** |
|---|---|--:|--:|--:|
| Qwen3-8B-Q4_K_M | Q4_K + 少量 Q6_K | 4.55–4.91 GiB | 4.55–4.93 GiB | **+0.17% ~ +0.38%** |
| Qwen3-8B-Q5_K_M | Q5_K + 少量 Q6_K | 5.38–5.56 GiB | 5.38–5.58 GiB | **+0.14% ~ +0.34%** |

（区间对应 Q6_K minority 占 12.5%/25%/50% 层的不同 K_M 配方假设；因 Q4_K/Q5_K 零膨胀，模型级膨胀
几乎只由 Q6_K 的 +0.95% × 其字节占比贡献，故无论配方如何都 **< +0.4%**。raw 体积与真实 GGUF 文件
大小吻合，验证张量清单正确。）

**编译期峰值**：repack 经 OV `propagate_constants` 子网执行（§4.5），期间单个权重 raw+packed 瞬时双份
（最大约 33–39 MiB）；逐 N-tile 流式 repack 可把峰值压到「+1 个 tile」。稳态只存一份 packed，
**模型级 < +0.4%**。

---

## 4. Tier-1 已知开销

- **private-gather 开销**：packed kernel 让每个 lane 把整 block 收进 `blk_u[wpb]` private buffer 再解码；
  对 128 B 量级 K-quant payload 增加寄存器压力，是 K-quant N4096 轻微回退的来源之一。但下文 §6 证明
  K-quant 的瓶颈根本不在 layout（无论怎么改 layout 都推不动），所以这条开销对 K-quant 已无意义——
  Tier-1 的价值锁定在小 block 格式。

---

## 6. Tier-2 Q6_K i8 合并：实现了、bit-exact，但**更慢**——证伪与放弃

设计 §6.1 假设 Q6_K 是 decode-ALU-bound、把 6-bit 码合并成 clean int8 喂 dp4a「消灭 SWAR 解包」即可提速。
**已完整实现并实测**（[`harness/tier2.py`](../dev_kf_distill_master_opt/gguf_kernels/harness/tier2.py)、
[`kernels/fc_gguf_dp4a_q6k_i8.cl`](../dev_kf_distill_master_opt/gguf_kernels/kernels/fc_gguf_dp4a_q6k_i8.cl)、
[`kernels/fc_gguf_prequant_sum.cl`](../dev_kf_distill_master_opt/gguf_kernels/kernels/fc_gguf_prequant_sum.cl)
预算 sum(a) 折叠 −32 偏移以省掉一半 dp4a）：

| shape | SWAR dp4a（现状） | Tier-2 i8 | **加速** | SWAR %BW | i8 %BW | 字节膨胀 | correctness |
|---|--:|--:|--:|--:|--:|--:|---|
| Q6_K K4096 N4096 | 0.1080 ms | 0.1347 ms | **×0.80** | 28.0% | 29.3% | +30.5% | bit-exact |
| Q6_K K12288 N4096 | 0.2697 ms | 0.3555 ms | **×0.76** | 33.6% | 33.2% | +30.5% | bit-exact |
| Q6_K K4096 N1024 | 0.0297 ms | 0.0368 ms | **×0.81** | 25.5% | 26.8% | +30.5% | bit-exact |

**诊断（关键）**：两个 kernel 跑出**同一个 ~152 GB/s（33% BW）墙**，与解码代价无关；且时间比
（1.31）≈ 字节比（51.4/39.4 = 1.30）。说明去掉 SWAR 解包**没有抬高吞吐**——**Q6_K 在这些 shape 上
不是 decode-ALU-bound，而是 memory-parallelism-bound**（每 subgroup 只 owns 16 个 block，in-flight
请求不够喂满总线）。这正是 [RESULTS_dp4a_evolve.md](../dev_kf_distill_master_opt/gguf_kernels/results/RESULTS_dp4a_evolve.md)
round-2 的结论。既然瓶颈是 MLP 而非 ALU，任何**增大字节**的重排（Tier-2 i8 +30%）都是纯负担。
**结论：Q6_K i8 Tier-2 放弃。** 设计文档 §6.1 的「Q6_K decode-bound」假设被实测证伪——这条记录本身
就是有价值的产出（它精确划定了「layout/字节优化无能为力」的边界）。

---

## 7. Q4_K int8-dp4a（无 repack）：对两个测试模型最有价值的优化 ✅

§6 的证伪把方向逼到正确处：K-quant 的杠杆是**解码域**而非 layout。RESULTS.md 同一 shape 上
**Q5_K dp4a 42% BW vs Q4_K float 19% BW**——差距不在重排，在于 **Q5_K 有 dp4a 整数路径而 Q4_K 没有**。
而 Q4_K 是 **Qwen3-8B-Q4_K_M 的主力格式（16/26 个 FC 张量）**。于是给 Q4_K 补一条与 Q5_K 同构的
int8-dp4a（[`kernels/fc_gguf_dp4a_q4k.cl`](../dev_kf_distill_master_opt/gguf_kernels/kernels/fc_gguf_dp4a_q4k.cl)，
= Q5_K dp4a 去掉 qh 高位面，**读 raw Q4_K block、零 repack、零显存膨胀**），A/B 对照
[`harness/ab_q4k.py`](../dev_kf_distill_master_opt/gguf_kernels/harness/ab_q4k.py)：

| Q4_K shape | float GEMV（现状） | **int8 dp4a** | **加速** | float %BW | **dp4a %BW** | best 旋钮 | correctness |
|---|--:|--:|--:|--:|--:|---|---|
| K4096 N4096 | 0.0797 ms | 0.0466 ms | **×1.71** | 26.0% | **44.5%** | NROW2 KSPLIT2 | dp4a relL2=2.1e-4 |
| K4096 N1024 | 0.0257 ms | 0.0151 ms | **×1.70** | 20.2% | **30.7%** | NROW1 | dp4a relL2=2.1e-4 |
| K4096 N12288 | 0.1849 ms | 0.1136 ms | **×1.63** | 33.6% | **54.7%** | NROW1 | dp4a relL2=2.1e-4 |
| K12288 N4096 | 0.2124 ms | 0.1180 ms | **×1.80** | 29.3% | **52.6%** | NROW2 KSPLIT2 | dp4a relL2=2.0e-4 |

**全部 bit-exact**（dp4a relL2=2.0–2.1e-4，与 float 参考同级；与 Q5_K dp4a 同精度语义）。
**稳定性** 3× back-to-back ±0.01。dp4a %BW 落在 30–55%，与 Q5_K dp4a 同档——证明「补 dp4a 路径」
就是把 Q4_K 从 float 的 ~19–34% 拉到 Q5_K 同级 roofline 的正确杠杆。**旋钮规律与 Q5_K 一致**：
N4096 大-K（q/o_proj、down_proj）用 NROW2 KSPLIT2，wide-N/tiny-N 用默认——可直接复用 OV 现成的
`gguf_q5k_dp4a_cfg` 表型。

**显存**：Q4_K dp4a 读 raw block，**零膨胀**（与现状 Q4_K float 完全相同的权重字节）。

---

## 7ter. 集成进 intel_gpu 插件 + 真实模型 E2E 实测 ✅

Q4_K dp4a 已**集成进 OpenVINO intel_gpu 插件**并跑通真实模型端到端 A/B。

**集成改动**（最小侵入，复用 Q5_K dp4a 全部基础设施）：

| 文件 | 改动 |
|---|---|
| `ocl_v2/fc_gguf_dp4a.cl` | 新增 `dp_block_dot_q4k` 解码器（= Q5_K 去 qh 高位面）＋ kernel dispatch `#elif GGUF_IS_Q4_K`；`get_scale_min_k4` 改 Q4_K/Q5_K 共享 |
| `ocl_v2/gguf/fc_gguf_opt.cpp` | 新增 `gguf_q4k_dp4a_cfg`（复用 Q5_K 的 NROW/KSPLIT 表）；generator split-K cache-key + cfg 选择纳入 Q4_K；新增 `m_use_q4k_dp4a`/`m_q4k_dp4a`、env `OV_GPU_GGUF_Q4K_DP4A`、stage 接线、clone、scratch、execute dispatch |

构建：`cmake --build ./build --target openvino_build`（增量重建 `libopenvino_intel_gpu_plugin.so`，
已确认 `.so` 内含 `dp_block_dot_q4k` 与 `OV_GPU_GGUF_Q4K_DP4A`）。

**E2E A/B（真机 B580，genai benchmark，Qwen3-8B-Q4_K_M，2048-token prompt → 128 token，n=1）**：

| 配置 | 2nd-token 延迟 (TPOT) | **decode 吞吐** | TTFT (1st token) |
|---|--:|--:|--:|
| `OV_GPU_GGUF_Q4K_DP4A=0`（baseline，Q4_K float GEMV） | 34.68 ms/tok | 28.83 tok/s | 695 ms |
| `OV_GPU_GGUF_Q4K_DP4A=1`（candidate，Q4_K int8 dp4a） | **26.14 ms/tok** | **38.26 tok/s** | 694 ms |
| **增益** | **−24.6%** | **+32.7%** | 不变 |

- **decode 吞吐 +32.7%**（28.83 → 38.26 tok/s）——Q4_K 是 Q4_K_M 解码期主力格式，kernel 级 ×1.6–1.8
  如预期转化为端到端 decode 提速。
- **TTFT（prefill）不变**（695→694 ms）：prefill 走 transcode + oneDNN 路径，未改动——符合预期。
- **Q5_K_M 无回归**：`OV_GPU_GGUF_Q4K_DP4A={0,1}` 下 27.68 vs 27.63 ms/tok（噪声内一致），因 Q5_K_M
  主体是 Q5_K（dp4a 早已启用），其 Q4_K 张量极少。

**GPU 显存（A/B 增量，由代码确定性给出）**：

| 项 | 增量 | 说明 |
|---|--:|---|
| 权重字节 | **+0** | Q4_K dp4a 绑定**同一个 raw Q4_K Constant**（无 repack、无 scratch 权重） |
| int8 激活 scratch | ≤ ~30 MiB（全模型）| 即 Q5_K/Q6_K dp4a 早已分配的 `[Mmax,K] i8 + [Mmax,K/32] f32`（Mmax=32），**复用同一基础设施** |
| 占模型权重比 | **+0.63%** | vs 4.7 GiB 模型权重；且非新增——本就为 dp4a 路径存在 |

> Q4_K dp4a **零权重膨胀**、仅复用既有极小激活 scratch——这正是它优于任何 repack 方案
> （Tier-2 要 +30% 权重）的根本：杠杆在解码域而非 layout/字节。

---

## 7quater. Q6_K LISA size-exact 重排：**编译期图变换**取代权重 + 真机 E2E ✅（PTL Arc B390）

> 本节是对设计 §4「编译期一次性、取代原始权重、prefill/decode 共享同一份」要求的**最终落地**。
> 此前实现把 repack 放在**第一次 decode**（lazy `ensure_packed_weight`）——首 token 的 repack 卡顿
> 把平均 decode 拉平，端到端看不到收益。本轮改为**真·编译期**：在 `compile_model` 期用一个 GPU
> plugin 图变换把 Q6_K 权重 Constant 原地字节置换，prefill 与每个 decode 都读同一份已打包权重。

**关键设计：size-exact 字节转置（0% 膨胀，保持 6-bit/210B 尺寸）**
- 组 = 16 个连续 block；组内 block-lane `gl=kb%16` 的第 i 字节存到 `(n*G+g)*16*bb + i*16 + gl`
  （`g=kb/16`）。解码时一个 subgroup 的 16 lane 对每个字节-index 命中 **16 个连续字节**（合并访存）。
- 仅当 `blocks_per_row % 16 == 0`（即 `K % 4096 == 0`）时是**精确同尺寸置换**；Qwen3-8B 全部 Q6_K
  张量（K∈{4096, 12288}）都满足，故**权重字节数完全不变**（对比早期 word-padded 版的 +0.95%）。

**实现改动**：

| 文件 | 改动 |
|---|---|
| `plugin/transformations/repack_gguf_weights.{hpp,cpp}` | **新增** `RepackGGUFWeightsQ6K` MatcherPass：host 端对 `gguf_q6_k` 权重 Constant 做 size-exact 字节转置（`parallel_for` 逐行），建同 type/同 shape/同字节数的新 Constant，`replace_source_output` 换掉 FC input(1) |
| `plugin/transformations_pipeline.cpp` | 在 `ConvertGGUFFullyConnectedCompressed` 之后注册该 pass |
| `ocl_v2/gguf/fc_gguf_opt.cpp` | 删除 lazy 设备端 repack（`FCGGUFRepackGenerator`/`repack_stage`/device 置换）；`ensure_packed_weight` 改为**直接别名** `weights_memory()`（权重已在编译期打包）；packed 路径门控与变换**同一纯条件**（pack env + Q6_K + `bpr%16==0`）以保持锁步；transcode A3 门控同步加 `bpr%16==0`；移除全部 `OV_GPU_GGUF_DIAG` 诊断 |

> 设计要点：变换与 impl 用**完全相同的纯条件**（在相同 `(dtype,N,K)` 上求值）决定是否打包，
> 无需把「已打包」标志透传到 primitive/impl。又因 horizontal-FC fusion **跳过 GGUF-block FC**
> （[`fc_horizontal_fusion.cpp:69`](../../../thirdparty/openvino/src/plugins/intel_gpu/src/plugin/transformations/fc_horizontal_fusion.cpp#L69)），
> 不存在 Q6_K 权重沿 N 拼接的情况；且置换是**逐行 N-无关**的，本身对任何 N 轴重排也安全。

**E2E A/B（真机 PTL Arc B390，12 Xe3 ~105 GB/s，genai benchmark，1024-token prompt → 64 token，n=3 取均）**：

| 模型 | 配置 | 2nd-token 延迟 (TPOT) | **decode 吞吐** | TTFT (1st token) |
|---|---|--:|--:|--:|
| **Q4_K_M** | `OV_GPU_GGUF_Q6K_PACK=0` | 68.31 ms/tok | 14.64 tok/s | 1317 ms |
| | `OV_GPU_GGUF_Q6K_PACK=1` | **57.13 ms/tok** | **17.50 tok/s** | 1369 ms |
| | **增益** | **−16.4%** | **+19.5%** | ~噪声 |
| **Q5_K_M** | `OV_GPU_GGUF_Q6K_PACK=0` | 77.63 ms/tok | 12.88 tok/s | 1406 ms |
| | `OV_GPU_GGUF_Q6K_PACK=1` | **65.66 ms/tok** | **15.23 tok/s** | 1445 ms |
| | **增益** | **−15.4%** | **+18.2%** | ~噪声 |

- **decode 显著提速**：Q4_K_M −16.4% TPOT、Q5_K_M −15.4%——与早期人工观测（65→56ms ≈14%）一致并更优。
  Q5_K_M 增益更大，因其 Q6_K 张量占比更高。
- **正确性**：相同 prompt 的 greedy 生成在 PACK=0 与 PACK=1 下**逐字符完全一致**（128 token），证明编译期
  字节置换 + packed 解码端到端 bit-exact（prefill A3 与 decode packed-dp4a 都正确读取置换后权重）。
- **权重显存 +0**：size-exact 置换不改字节数（与 lazy 版相比还省掉了第二份打包副本 + 首 token 的设备 repack）。
- **TTFT ~噪声内**：早期 lazy 版的「首 decode repack 卡顿」已被消除（repack 移到 compile_model），剩余 TTFT
  差异（+50ms 量级）落在跨 run 噪声内。

> **为什么 lazy 版看不到收益、编译期版立刻看到**：lazy 版的 repack 发生在第一个 decode token 内，把首 token
> 拉慢、并以「平均 decode」口径稀释；llm_bench 的 **2nd-token latency**（剔除首 token）才暴露稳态 decode 的
> 真实提速。把 repack 移到编译期后，2nd-token 口径下 Q6_K 的 ×1.8–2.1 kernel 收益干净地转化为端到端
> decode 的 −15~16%。这也回答了此前「kernel 像没被调用」的疑问：kernel 一直在跑，被掩盖的是**时机**（lazy）
> 与早期一个 NROW=4 配置 bug（packed 路径必须 NROW=1，否则 byte-transpose 合并访存被破坏，实测仅 19.6% BW）。

---

## 7bis. 重新审视 Tier-1 layout 的「更好选择」（回应建议）

建议提到「重新考虑 Tier-1 layout 重排是否有更好选择」。基于实测，结论是：**对 K-quant，
任何纯 layout 重排都不是正确的杠杆**——已被两组实测从两个方向夹证：

1. **§2 的 Tier-1 word-interleave（合并访存）对 K-quant 中性**：K-quant 256-elem block 本就连续，
   L2 已吸收窗口内 strided 访问，合并不再有红利。
2. **§6 的 Tier-2 字节合并（消解包）对 Q6_K 反而更慢**：Q6_K 在这些 shape 是 memory-parallelism-bound，
   去 SWAR 不抬吞吐、加字节纯亏。

两者夹出的唯一有效维度是**解码域（float → int8 dp4a，§7）**，它不改 layout、不加字节，却把 Q4_K
拉到 Q5_K 同级 roofline。因此「更好的 layout 选择」对 Qwen3 的 K-quant 而言**不存在于 layout 空间内**，
而在解码域——这正是 §7 采用的方向。

**Tier-1 layout 本身的「更好选择」只对小 block（Q4_0/Q8_0）有意义**（§2 已 ×2.7–3.5）；如需进一步压榨，
可用 `dev_kf_distill_master_opt` 的 **MAP-Elites 遗传搜索**（[`harness/evolve_opt.py`](../dev_kf_distill_master_opt/gguf_kernels/harness/evolve_opt.py)）
在 packed layout 上搜 word-tile 粒度 / lane 映射 / NROW 组合——但因 Qwen3-8B-Q4_K_M/Q5_K_M **不含**
Q4_0/Q8_0，对这两个目标模型无端到端价值，故未投入本轮（留作其他模型的后续项）。

> 一句话：layout 重排的收益边界已被实测划清——小 block 用 Tier-1（×3），K-quant 用解码域 dp4a（§7），
> Tier-2 字节合并对 memory-bound 的 Q6_K 是负优化。这是「更好选择」问题的数据驱动答案。

---

## 8. 一句话总结（修订）

> 在 B580 上实测三条路线后的诚实结论：(1) **Tier-1 LISA 无损重排** 对小 block 格式 Q4_0/Q8_0
> **×2.7–3.5**、bit-exact、模型级膨胀 <0.4%，但对 K-quant 中性；(2) **Tier-2 Q6_K i8 合并** bit-exact
> 却 **×0.76–0.80 更慢**，因 Q6_K 实为 memory-parallelism-bound（证伪设计假设，放弃）；(3) **Q4_K
> int8-dp4a** 给 Qwen3 主力格式补上 Q5_K 早有的整数路径，**×1.63–1.80、bit-exact、零显存膨胀**，
> 是对 Qwen3-8B-Q4_K_M **端到端最有价值**的优化——**已集成进 intel_gpu 插件、增量构建、真机
> E2E 验证**：Q4_K_M **decode 吞吐 28.83→38.26 tok/s（+32.7%）、TPOT 34.68→26.14 ms/tok（−24.6%）**，
> TTFT 不变、Q5_K_M 无回归、权重显存 +0（§7ter）。
>
> **方法论收获**：三条路线里两条（Tier-1 对 K-quant、Tier-2 Q6_K）实测为"无效/负优化"，正是这些
> 证伪把杠杆逼到正确的解码域，最终拿到端到端 +32.7%。这印证了「先在离线 harness 验证再构建插件」
> 的价值——若直接照设计文档把 Tier-1/Tier-2 全量接入插件，会耗费数小时构建却换来端到端零提升甚至回退。

### 后续可选项
> - **Q6_K decode 提速**：实测瓶颈是 memory-parallelism（非 ALU/layout），需重构 row→subgroup 映射
>   （多 subgroup 喂更宽请求流），非本轮 layout/dp4a 范畴。
> - **其他模型**：若目标模型含 Q4_0/Q8_0，Tier-1 LISA（已实现、×2.7–3.5）可接入插件的
>   `WeightsReorderParams` 编译期 repack；对 Qwen3-8B-Q4_K_M/Q5_K_M 无此格式故未接入。
> - **MAP-Elites**：可在 `dev_kf_distill_master_opt` 的 evolve harness 上进一步搜 Q4_K dp4a 的
>   per-shape (NROW,KSPLIT) 与 KB_UNROLL，预计在已得 ×1.6–1.8 之上再压榨个位数%。
