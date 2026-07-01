# GGUF Kernel 优化 — 性能提升数据总汇

> 本文汇总 `dev_gguf_support_ovmx` 全部 GGUF 权重 kernel 优化的**实测性能数据**（kernel 级 roofline
> + 端到端 E2E），按「已落地 / 实验证伪」分类。所有数字均真机测得，correctness 全部 bit-exact 或
> relL2 与参考同级。原始细节见 [OPTIMIZE_RESULT.md](OPTIMIZE_RESULT.md)、
> [WEIGHT_REPACK_RESULTS_cn.md](WEIGHT_REPACK_RESULTS_cn.md)、
> [RESULTS_dp4a_evolve.md](../dev_kf_distill_master_opt/gguf_kernels/results/RESULTS_dp4a_evolve.md)。

**测试硬件**
- **B580**（Battlemage / Xe2）：DRAM 456 GB/s spec（**实测 achievable 随工作集：3 MiB≈140、11 MiB≈189、33–39 MiB≈256–272、487 MiB≈428 GB/s**）· INT8 dp4a 59.39 TOPS · L2 18 MiB · 160 Xe-core · SIMD16。
- **PTL**（Arc B390）：12 Xe3 · DRAM **~105 GB/s**（random-data 实测；all-ones 因无损压缩假读 ~970 GB/s）· L3 ~1096 GB/s。

**测试模型**：Qwen3-8B-Q4_K_M.gguf、Qwen3-8B-Q5_K_M.gguf（主用 Q4_K/Q5_K/Q6_K）。

---

## 0. 一页速览（最重要的数字）

| # | 优化 | 目标格式 | 硬件 | kernel 级 | **端到端 decode 提升** | 状态 |
|--:|---|---|---|--:|--:|:--:|
| 1 | **Q5_K SWAR dp4a GEMV**（MAP-Elites 基石） | Q5_K | B580 | 0.8% → **54–81% roofline**（×60–100） | 见下 §1 E2E | ✅ 已落地 |
| 2 | **Q6_K 解码 ILP**（4 路独立累加器） | Q6_K | B580 | 24.9% → **31.7% roofline（+27%）** | 随 §1 一并计入 | ✅ 已落地 |
| 3 | **dp4a split-K / NROW 调优** | Q5_K/Q6_K | B580 | Q5_K ×1.05–1.15；Q6_K 头 ×1.09 | — | ✅ 已落地 |
| 4 | **Q4_K int8-dp4a**（补齐整数路径） | Q4_K | B580 | ×1.63–1.80（19–34%→30–55% BW） | **Q4_K_M +32.7%**（28.83→38.26 tok/s） | ✅ 已落地 |
| 5 | **Q6_K LISA size-exact 编译期 repack** | Q6_K | PTL | 52% → **89–93% BW（合并访存）** | **Q4_K_M +19.5% / Q5_K_M +18.2%** | ✅ 已落地 |
| 6 | **Tier-1 LISA 无损重排** | Q4_0/Q8_0 | B580 | **×2.70–3.49**（15–19%→48–70% BW） | 未接入（测试模型无此格式） | ⚙️ 备用 |
| 7 | **Tier-1 LISA** on K-quant | Q4_K/Q5_K/Q6_K | B580 | ×0.88–1.03（中性） | — | ❌ 证伪 |
| 8 | **Tier-2 Q6_K i8 位平面合并** | Q6_K | B580 | **×0.76–0.80（更慢）** | — | ❌ 证伪放弃 |

> **两条最有价值的落地**：#4（Q4_K dp4a，Q4_K_M **decode +32.7%**）和 #5（Q6_K 编译期 repack，
> 两模型 **decode +18~20%**）。两条证伪（#7/#8）精确划定了「纯 layout/字节优化对 K-quant 无能为力」
> 的边界，把杠杆逼到正确的**解码域**（float→int8 dp4a）与**访存合并**（coalesced packed）。

---

## 1. Q5_K SWAR dp4a GEMV — 从灾难基线到 roofline（B580，基石）

出货基线的 Q5_K decode GEMV **灾难性地慢**（roofline < 1%）；经 MAP-Elites（`dev_kf_distill`）搜出
streaming sub-group K-split GEMV + SWAR dp4a 后逼近 roofline。

**基线诊断（B580，roofline vs 421 GB/s 实测）**

| shape | 出货基线 roofline | 有效 BW | 诊断 |
|---|--:|--:|---|
| 4096×4096 | 0.81% | 3.4 GB/s | 纯流式读可达 **469 GB/s（111%）** → 非访存瓶颈 |
| 12288×4096 | 0.85% | 3.6 GB/s | float 解码上限 ~51% → **解码 ALU 瓶颈** |
| 1024×4096 | 0.86% | 3.6 GB/s | SWAR dp4a 把每 4 码解包压到 1 个 packed uint op |
| 4096×12288 | 0.79% | 3.3 GB/s | |

**优化后（SWAR dp4a，`dot_acc_sat_4x8packed_us_int`，4 MAC/op）**：1024×4096 冲到 **81.2% roofline**；
其余 4096²/大 MLP shape 落 54–73%。相较基线 **约 ×60–100** 的提升，是所有后续 dp4a 路径的基础。

---

## 2. Q6_K 解码 ILP — 4 路独立累加器（B580，+27%）

给 Q6_K 解码器 **4 条独立累加链**打破依赖、提高指令级并行（cell `(0,2,2,0)`，target-profile 搜出）：

| shape | ILP 前 | **ILP 后** | 提升 |
|---|--:|--:|--:|
| Q6_K 4096×12288 | 24.9% roofline | **31.7% roofline** | **+27%** |

> 同一 ILP 编辑对 Q5_K **回退**（寄存器压力），故 MAP-Elites 的 QD-transition 追踪把它 **format-local
> 到 Q6_K**——展示了「进化式搜索按格式局部保留 elite」的价值。

---

## 3. dp4a split-K / NROW 调优（B580，roofline vs size-matched achievable ceiling）

> 关键方法论修正：正确 roofline **不是** 456 GB/s spec 峰值（小工作集物理上达不到），而是
> **size-matched achievable ceiling**。据此，四个 ≤33 MiB 的 Q5_K shape **已在/超过其流式上限**（无带宽可抢）。

| 格式 | K | N | 权重 | 基线 | **best genome** | best BW | %spec | **% achievable** | 加速 |
|---|--:|--:|--:|---|---|--:|--:|--:|--:|
| Q5_K | 12288 | 4096 | 33 MiB | NROW1 228 GB/s | **NROW2 KSPLIT2** | 262 | 57.4% | **102%** | **×1.15** |
| Q5_K | 4096 | 4096 | 11 MiB | NROW1 192 | **NROW2 KSPLIT2** | 201 | 44.0% | **106%** | **×1.05** |
| Q5_K | 4096 | 1024 | 2.8 MiB | NROW1 154 | 默认 | 154 | 33.8% | **109%** | ×1.00（已满） |
| Q5_K | 4096 | 12288 | 33 MiB | NROW1 215 | 默认 | 215 | 47.2% | 84% | ×1.00 |
| Q6_K | 4096 | 151936（vocab 头）| 487 MiB | NROW4 164 | **NROW2** | 178 | 39.1% | 42% | **×1.09** |
| Q6_K | 12288 | 4096 | 39 MiB | NROW4 152 | 默认 | 152 | 33.3% | 56% | ×1.00（decode-ALU 墙）|
| Q6_K | 4096 | 1024 | 3.3 MiB | NROW4 115 | 默认 | 115 | 25.3% | 82% | ×1.00 |

**诊断**：Q6_K K4096/K12288 N4096 被钉在 ~152 GB/s、对所有 memory-parallelism 旋钮（NROW/KSPLIT/
KB_UNROLL/WGSG）**平坦 ±2%**——这是 **decode-ALU co-bound** 的签名（Q6_K 每 block 带 8-bit 有符号
per-16 scale + 2-bit 高位面 SWAR，约 2× Q5_K 解包 ALU）。故这些 shape 的 56% 就是该 dp4a 公式的
解码吞吐墙；vocab 头（487 MiB）**是**访存瓶颈，NROW 4→2 增加 live-subgroup 拿到 ×1.09。

---

## 4. Q4_K int8-dp4a — 对测试模型最有价值的落地 ✅（B580）

RESULTS.md 同 shape 上 **Q5_K dp4a 42% BW vs Q4_K float 19% BW**，差距不在 layout，在于
**Q4_K 缺整数路径**。而 Q4_K 是 **Q4_K_M 的主力（16/26 个 FC 张量）**。补一条与 Q5_K 同构的
int8-dp4a（= Q5_K 去 qh 高位面，**读 raw Q4_K block、零 repack、零显存膨胀**）：

**kernel 级（B580 harness A/B）**

| Q4_K shape | float GEMV | **int8 dp4a** | **加速** | float %BW | **dp4a %BW** | best 旋钮 |
|---|--:|--:|--:|--:|--:|---|
| K4096 N4096 | 0.0797 ms | 0.0466 ms | **×1.71** | 26.0% | **44.5%** | NROW2 KSPLIT2 |
| K4096 N1024 | 0.0257 ms | 0.0151 ms | **×1.70** | 20.2% | **30.7%** | NROW1 |
| K4096 N12288 | 0.1849 ms | 0.1136 ms | **×1.63** | 33.6% | **54.7%** | NROW1 |
| K12288 N4096 | 0.2124 ms | 0.1180 ms | **×1.80** | 29.3% | **52.6%** | NROW2 KSPLIT2 |

全部 bit-exact（dp4a relL2=2.0–2.1e-4，与 Q5_K dp4a 同精度语义）。

**端到端 E2E（B580，genai benchmark，Qwen3-8B-Q4_K_M，2048→128，n=1）**

| 配置 | TPOT (2nd-token) | **decode 吞吐** | TTFT |
|---|--:|--:|--:|
| `OV_GPU_GGUF_Q4K_DP4A=0`（baseline float GEMV） | 34.68 ms/tok | 28.83 tok/s | 695 ms |
| `OV_GPU_GGUF_Q4K_DP4A=1`（int8 dp4a） | **26.14 ms/tok** | **38.26 tok/s** | 694 ms |
| **增益** | **−24.6%** | **+32.7%** | 不变 |

- prefill（TTFT）不变（走 transcode+oneDNN，未改）；Q5_K_M 无回归（27.68 vs 27.63 ms/tok，噪声内）。
- **显存 +0**：绑定同一 raw Q4_K Constant，仅复用既有极小 int8 激活 scratch（≤~30 MiB / 全模型）。

---

## 5. Q6_K LISA size-exact 编译期 repack — 访存合并 ✅（PTL）

PTL 上 Q6_K 是 **L3-bandwidth-bound**：per-lane strided gather（16 lane 跨 210B 跳读）仅达 ~52%
DRAM 墙。把权重在 `compile_model` 期字节转置成 lane-interleaved（16 lane 读 16 连续字节 = 合并），
decode dp4a 达 89–93% BW。**关键：编译期取代权重**（非 lazy 首-decode repack）、**size-exact 0% 膨胀**
（`bpr%16==0` 时精确同尺寸，Qwen3-8B 全部 Q6_K 满足）、**NROW 必须=1**（NROW=4 破坏合并，实测仅 19.6% BW）。

**kernel 级（PTL harness）**：strided SWAR ~52% BW → **size-exact packed 89–93% BW（NROW=1）**（≈×1.8–2.1）。

**端到端 E2E（PTL，genai benchmark，1024→64，n=3 均值，2nd-token 口径）**

| 模型 | 配置 | TPOT (2nd-token) | **decode 吞吐** | TTFT |
|---|---|--:|--:|--:|
| **Q4_K_M** | `OV_GPU_GGUF_Q6K_PACK=0` | 68.31 ms/tok | 14.64 tok/s | 1317 ms |
| | `OV_GPU_GGUF_Q6K_PACK=1` | **57.13 ms/tok** | **17.50 tok/s** | 1369 ms |
| | **增益** | **−16.4%** | **+19.5%** | ~噪声 |
| **Q5_K_M** | `OV_GPU_GGUF_Q6K_PACK=0` | 77.63 ms/tok | 12.88 tok/s | 1406 ms |
| | `OV_GPU_GGUF_Q6K_PACK=1` | **65.66 ms/tok** | **15.23 tok/s** | 1445 ms |
| | **增益** | **−15.4%** | **+18.2%** | ~噪声 |

- 正确性：相同 prompt greedy 生成 PACK={0,1} **逐字符完全一致**（128 token，编译期置换 + packed 解码端到端 bit-exact）。
- 显存 **+0**（size-exact，且省掉 lazy 版的第二份副本 + 首 token 设备 repack）。
- **为何 lazy 版看不到、编译期版立刻看到**：lazy 把 repack 塞进首个 decode，被「平均 decode」稀释；
  移到编译期后 Q6_K 的 ×1.8–2.1 kernel 收益在 **2nd-token 口径**下干净转化为 decode −15~16%。

---

## 6. Tier-1 LISA 无损重排 — 小 block 格式 ×2.7–3.5 ⚙️（B580，备用）

小 block 格式（Q4_0 18B / Q8_0 34B）既非 64B 约数又跨 cacheline，raw 路径仅 15–19% BW；LISA 把每条
权重 load 对齐成整 cacheline 合并：

| 格式 | shape | raw ms | packed ms | **加速** | raw %BW | **packed %BW** |
|---|---|--:|--:|--:|--:|--:|
| **Q4_0** | K4096 N4096 | 0.1386 | 0.0473 | **×2.93** | 15.0% | **48.7%** |
| **Q4_0** | K12288 N4096 | 0.4193 | 0.1200 | **×3.49** | 14.8% | **57.5%** |
| **Q8_0** | K4096 N4096 | 0.2056 | 0.0761 | **×2.70** | 19.0% | **54.4%** |
| **Q8_0** | K12288 N4096 | 0.6112 | 0.1787 | **×3.42** | 19.2% | **69.5%** |

全部 bit-exact（relL2(AB)=0）。稳定性 150 iters×3：Q4_0 K12288 中位 ×3.50、Q8_0 中位 ×3.44（±<0.3%）。
**未接入插件**：两个测试模型（Q4_K_M/Q5_K_M）不含 Q4_0/Q8_0；若目标模型含此格式可经编译期 repack 接入。

**显存膨胀**（4-byte word 对齐尾 pad）：Q4_0 +11.1%、Q8_0 +5.88%（小 block，但换 ×3 加速）；
Q4_K/Q5_K **+0.00%**、Q6_K word-padded 版 +0.95%（size-exact 版 §5 为 0%）。

---

## 7. 两条证伪（同样有价值：划定优化边界）❌

### 7.1 Tier-1 LISA on K-quant — 中性（×0.88–1.03，B580）

| 格式 | shape | raw ms | packed ms | 加速 |
|---|---|--:|--:|--:|
| Q4_K | K4096 N4096 | 0.0798 | 0.0844 | ×0.95 |
| Q4_K | K12288 N4096 | 0.2125 | 0.2098 | ×1.01 |
| Q5_K | K4096 N4096 | 0.0904 | 0.1019 | ×0.89 |
| Q5_K | K12288 N4096 | 0.2543 | 0.2476 | ×1.03 |
| Q6_K | K4096 N4096 | 0.1166 | 0.1319 | ×0.88 |
| Q6_K | K12288 N4096 | 0.3311 | 0.3445 | ×0.96 |

256-elem block（144/176/210B）本身连续，L2/coalescer 已吸收窗口内 strided 访问，瓶颈是**解包 ALU**
而非访存对齐——纯重排推不动。**推论**：K-quant 要提速须走**解码域**（→ §4 Q4_K dp4a）或**合并访存**
（→ §5 PTL 上的 L3-bound Q6_K packed），而非通用 layout 重排。

### 7.2 Tier-2 Q6_K i8 位平面合并 — 更慢（×0.76–0.80，B580，放弃）

设计假设「Q6_K 是 decode-ALU-bound，合并成 clean int8 喂 dp4a 可消灭 SWAR 解包」。实现并实测：

| shape | SWAR dp4a | Tier-2 i8 | **加速** | 字节膨胀 |
|---|--:|--:|--:|--:|
| Q6_K K4096 N4096 | 0.1080 ms | 0.1347 ms | **×0.80** | +30.5% |
| Q6_K K12288 N4096 | 0.2697 ms | 0.3555 ms | **×0.76** | +30.5% |
| Q6_K K4096 N1024 | 0.0297 ms | 0.0368 ms | **×0.81** | +30.5% |

时间比（1.31）≈ 字节比（1.30）→ 去掉 SWAR 解包**没抬高吞吐**，两 kernel 撞同一 ~152 GB/s 墙。证明
Q6_K 在这些 shape 是 **memory-parallelism-bound**（每 subgroup 只 owns 16 block，in-flight 请求不足），
任何增大字节的重排都是纯负担。**放弃 Tier-2 i8。**

---

## 8. 一句话总结

> GGUF kernel 优化的杠杆有三处、边界清晰：
> 1. **解码域**（float→int8 dp4a）——给 Q4_K 补整数路径拿到 **Q4_K_M decode +32.7%**（§4），是对
>    测试模型最有价值的单点优化；Q5_K/Q6_K 的 SWAR dp4a + ILP 是其基石（§1/§2，×60–100 / +27%）。
> 2. **访存合并**（coalesced packed）——小 block（Q4_0/Q8_0）Tier-1 LISA **×2.7–3.5**（§6，备用）；
>    PTL 上 L3-bound 的 Q6_K 经 **编译期 size-exact repack** 拿到两模型 **decode +18~20%**（§5）。
> 3. **memory-level parallelism**（NROW/split-K）——Q5_K/Q6_K 微调 ×1.05–1.15（§3）。
>
> **纯 layout 重排对 decode-ALU/memory-parallelism-bound 的 K-quant 无能为力**（§7 两条证伪），这一
> 边界认知本身把工程投入导向了正确方向。
