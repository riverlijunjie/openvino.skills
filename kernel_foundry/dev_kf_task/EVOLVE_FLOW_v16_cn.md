# Kernel Foundry v16 冷启动实验报告

## 实验概述

| 项目 | 说明 |
|------|------|
| 实验版本 | v16 |
| 实验日期 | 2026-05-29 |
| 实验目标 | 验证冷启动（无seed kernel）条件下MAP-Elites进化能力 |
| 与v15区别 | **v15使用2D Block IO seed kernel（0.948ms），v16从朴素标量循环开始** |
| GPU | Intel B580 (Battlemage G21, 20 Xe2 cores) |
| 问题规模 | C[2048,2048] = A[2048,2560] × B[2560,2048], FP16, 21.5 GFLOP |
| 理论最优 | 0.224ms (96 TFLOPS XMX peak) |
| 参考基线 | 34.0ms（朴素标量实现，无任何优化） |
| 模型 | Claude Opus 4.6 (us.anthropic.claude-opus-4-6-v1) |
| 迭代次数 | 12 trials × 4 branches = 48个kernel变体 |
| 总耗时 | 1424.3s（23.7分钟） |

## 种子kernel（冷启动）

```c
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M, const int K, const int N)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += convert_float(A[row * K + k]) * convert_float(B[k * N + col]);
    C[row * N + col] = convert_half(acc);
}
```

性能: 34.0ms — 这是每个work-item计算一个输出元素的最朴素实现。

## 实验结果总览

| 指标 | 数值 |
|------|------|
| 总branches | 48 |
| 编译成功 | 33/48 (68.8%) |
| 正确性通过 | 27/48 (56.2%) |
| 编译失败 | 15/48 (31.2%) |
| 正确但无显著优化 | 4/48 (8.3%) — runtime ≈ 34ms |
| 有效优化 | 23/48 (47.9%) — runtime < 2ms |
| 最佳性能 | 1.30ms |
| 最佳加速比 | 26.2x（vs 34ms参考） |
| XMX利用率 | 17.2%（vs 0.224ms理论值） |
| MAP-Elites覆盖 | 4/256 cells |

## 逐轮迭代详情

| Trial | Branch | 正确性 | 运行时间(ms) | 加速比 | XMX利用率 | 优化策略 | 状态 |
|-------|--------|--------|-------------|--------|-----------|----------|------|
| **0** | v0 | ✗ | N/A | - | - | DPAS+SLM+2D Block Read (编译失败) | COMPILE_FAIL |
| **0** | v1 | ✓ | 34.0 | 1.0x | 0.66% | 基础DPAS tiling (无SLM，subgroup block read) | CORRECT (无加速) |
| **0** | v2 | ✓ | **1.46** | 23.2x | 15.3% | 64WI/4SG, SLM double-buffer, DPAS 32×64×32 tile | ✓ BEST (Trial 0) |
| **0** | v3 | ✓ | 34.0 | 1.0x | 0.66% | DPAS tiling (SLM bank conflict问题) | CORRECT (无加速) |
| **1** | v0 | ✗ | N/A | - | - | 增加vload+subgroup_shuffle (正确性失败) | INCORRECT |
| **1** | v1 | ✗ | N/A | - | - | K-loop 2x unroll (正确性失败) | INCORRECT |
| **1** | v2 | ✓ | 33.9 | 1.0x | 0.66% | 保守优化，未使用SLM double-buffer | CORRECT (无加速) |
| **1** | v3 | ✗ | N/A | - | - | 激进SLM策略 (正确性失败) | INCORRECT |
| **2** | v0 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **2** | v1 | ✓ | 1.46 | 23.3x | 15.3% | SLM double-buffer + interleaved load/compute | CORRECT |
| **2** | v2 | ✓ | **1.30** | 26.2x | 17.2% | K-loop 2x unroll + vload8 pairs + 指令重排 | ✓ **GLOBAL BEST** |
| **2** | v3 | ✓ | 1.38 | 24.6x | 16.2% | SLM double-buffer + K-loop unroll | CORRECT |
| **3** | v0 | ✓ | 1.55 | 21.9x | 14.5% | 基于v2最佳，if guard优化 | CORRECT |
| **3** | v1 | ✓ | 1.61 | 21.1x | 13.9% | 2x K-loop unroll + B prefetch | CORRECT |
| **3** | v2 | ✗ | N/A | - | - | 编译失败 (语法错误) | COMPILE_FAIL |
| **3** | v3 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **4** | v0 | ✓ | 1.64 | 20.7x | 13.7% | 32×64×32 tile, 4SG, K-loop unroll | CORRECT |
| **4** | v1 | ✓ | 1.57 | 21.7x | 14.3% | 同上，优化A load顺序 | CORRECT |
| **4** | v2 | ✓ | 1.62 | 21.0x | 13.8% | 同上，barrier优化 | CORRECT |
| **4** | v3 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **5** | v0 | ✓ | 1.57 | 21.7x | 14.3% | 保守实现，去除compound literals | CORRECT |
| **5** | v1 | ✗ | N/A | - | - | 正确性失败 | INCORRECT |
| **5** | v2 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **5** | v3 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **6** | v0 | ✓ | **1.30** | 26.2x | 17.2% | 2x K-loop unroll + if guard修复 | ✓ 追平BEST |
| **6** | v1 | ✓ | 1.39 | 24.5x | 16.1% | 类似v0，额外micro-opt | CORRECT |
| **6** | v2 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **6** | v3 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **7** | v0 | ✓ | 1.69 | 20.1x | 13.3% | 保守DPAS, 减少SLM bank conflict | CORRECT |
| **7** | v1 | ✗ | N/A | - | - | 正确性失败 | INCORRECT |
| **7** | v2 | ✓ | 1.61 | 21.1x | 13.9% | 标准32×64×32实现 | CORRECT |
| **7** | v3 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **8** | v0 | ✓ | 1.32 | 25.8x | 17.0% | 基于最佳实现，safe micro-opt | CORRECT |
| **8** | v1 | ✓ | 1.38 | 24.6x | 16.2% | precompute a_row_offset | CORRECT |
| **8** | v2 | ✓ | 1.61 | 21.1x | 13.9% | 标准实现 | CORRECT |
| **8** | v3 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **9** | v0 | ✓ | 1.37 | 24.8x | 16.4% | 返回proven-best结构 + safe micro-opt | CORRECT |
| **9** | v1 | ✓ | 1.64 | 20.7x | 13.7% | 标准double-buffer | CORRECT |
| **9** | v2 | ✓ | 1.31 | 26.0x | 17.1% | 最佳实现结构复现 | CORRECT |
| **9** | v3 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **10** | v0 | ✓ | 1.62 | 21.0x | 13.8% | 去除K-boundary check | CORRECT |
| **10** | v1 | ✓ | 1.61 | 21.1x | 13.9% | 标准实现 | CORRECT |
| **10** | v2 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **10** | v3 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **11** | v0 | ✓ | 1.64 | 20.7x | 13.7% | A prefetch overlap with compute | CORRECT |
| **11** | v1 | ✓ | 1.58 | 21.5x | 14.2% | 标准double-buffer | CORRECT |
| **11** | v2 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |
| **11** | v3 | ✗ | N/A | - | - | 编译失败 | COMPILE_FAIL |

## 性能进化轨迹

```
Trial 0:  34.0ms → 1.46ms  (首轮即从朴素循环跃迁到DPAS优化)
Trial 2:  1.46ms → 1.30ms  (K-loop 2x unroll + vload8优化, NEW BEST)
Trial 3~5: 1.30ms (无突破，变体在1.55-1.69ms范围)  
Trial 6:  1.30ms (追平最佳)
Trial 8:  1.32ms (接近最佳)
Trial 9:  1.31ms (接近最佳)
Trial 10~11: 1.58-1.64ms (轻微退化，未突破)
```

**MAP-Elites进化记录：**

| 时间点 | 程序ID | Score | 事件 |
|--------|--------|-------|------|
| Trial 0 | e04c04d0 | 8.00 | 首个正确kernel (34ms) |
| Trial 0 | fd95fb5e | 74.66 | 首个高性能kernel (1.46ms) |
| Trial 2 | 7971b985 | 74.86 | 新最佳 (1.30ms) |
| Trial 2 | 47d01e01 | 83.46 | 最终最佳 (score提升) |

## 推理时间与Token消耗估算

| Trial | 推理时间(s) | 评估时间(s) | 总时间(s) | 估算Token(input+output) |
|-------|------------|------------|-----------|------------------------|
| 0 | 140.4 | 143.1 | 283.5 | ~120K (首轮，含完整prompt) |
| 1 | 76.6 | 28.0 | 104.6 | ~80K |
| 2 | 82.7 | 42.4 | 125.1 | ~85K |
| 3 | 93.9 | 7.8 | 101.7 | ~90K |
| 4 | 90.5 | 11.1 | 101.6 | ~90K |
| 5 | 97.0 | 5.2 | 102.2 | ~95K |
| 6 | 92.9 | 7.8 | 100.7 | ~90K |
| 7 | 91.4 | 9.1 | 100.5 | ~90K |
| 8 | 91.9 | 11.5 | 103.4 | ~90K |
| 9 | 91.2 | 11.4 | 102.6 | ~90K |
| 10 | 91.6 | 7.8 | 99.4 | ~90K |
| 11 | 91.1 | 7.9 | 99.0 | ~90K |
| **合计** | **1131.2** | **293.1** | **1424.3** | **~1.1M tokens** |

> 注: Token消耗为估算值。基于Claude Opus 4.6平均输出速度~40 tok/s和每次推理90s计算，每branch约3600 output tokens，加上input prompt约15K tokens/branch。总计 48 branches × (~15K input + ~3.6K output) ≈ 894K tokens。考虑到系统开销，估计总消耗约 **1.0-1.2M tokens**。

## 核心发现

### 1. 冷启动进化能力验证

**结论：MAP-Elites从朴素标量循环到DPAS优化kernel的跃迁在首轮即完成。**

- Trial 0 Branch v2 直接从34ms跃升到1.46ms（23.2x加速）
- 这证明LLM（Claude Opus 4.6）能在单次推理中从零设计出完整的DPAS+SLM kernel
- 首轮花费140s推理时间（比后续90s长50%），表明首次设计需要更多"思考"

### 2. 性能瓶颈：1.3ms天花板

所有12轮迭代的最佳性能停留在1.30ms，未能突破。对比v15的0.274ms最佳结果：

| 对比项 | v15 (温启动) | v16 (冷启动) |
|--------|-------------|-------------|
| Seed kernel | 2D Block IO, 0.948ms | 朴素标量, 34.0ms |
| 最佳结果 | 0.274ms | 1.30ms |
| XMX利用率 | 81.6% | 17.2% |
| 达到最佳所需轮次 | ~8轮 | 2轮 |
| 使用2D Block IO | ✓ | ✗ |

**性能差距分析 (1.30ms vs 0.274ms = 4.7x):**

v16的最佳kernel使用以下架构：
- 64 work-items, 4 subgroups, SG_SIZE=16
- TILE: 32×64×32 (WG计算 32行 × 64列，K步长32)
- A通过SLM double-buffering
- B从global memory通过sub_group_block_read
- K-loop 2x unroll (步长64)
- DPAS: `intel_sub_group_f16_f16_matrix_mad_k16`

**缺少的关键优化（对比v15）：**
- ❌ 2D Block Load (intel_sub_group_2d_block_read) — v15使用此指令获得最高带宽
- ❌ B矩阵的VNNI格式转换 (2d_block_read_transform)
- ❌ 更大的WG tile覆盖 (v15用更大tile降低launch开销)
- ❌ 高级prefetch策略 (L1/L2 cooperative prefetch)

### 3. 编译失败率高 (31.2%)

15/48个branch编译失败，主要原因：
- 语法错误（括号不匹配、未声明变量）
- Intel扩展API使用错误（错误的参数类型）
- `reqd_work_group_size`与实际WG layout不匹配

### 4. 进化停滞现象

从Trial 2到Trial 11（10轮），最佳性能未改进。MAP-Elites算法在有限的4-cell覆盖下缺乏多样性压力，导致后续生成的kernel都在1.3-1.7ms范围内震荡。

### 5. 冷启动vs温启动对比结论

| 维度 | 冷启动(v16) | 温启动(v15) |
|------|------------|------------|
| 首轮突破能力 | 强 (34ms→1.46ms) | 中 (0.948ms→0.5ms) |
| 最终极限性能 | 弱 (1.30ms) | 强 (0.274ms) |
| 需要的知识 | LLM内化知识足够达到基础DPAS | 需seed提供2D Block IO模式 |
| 进化效率 | 低 (12轮无突破) | 高 (持续改进) |
| 适用场景 | 探索全新架构方向 | 精细优化已知最佳方案 |

## 结论与建议

1. **冷启动有效性有限**：虽然LLM能从零开始生成DPAS kernel，但无法通过进化达到2D Block IO级别的性能。2D Block Load是Xe2架构达到高XMX利用率的关键，但LLM难以在无示例的情况下正确使用。

2. **推荐混合策略**：使用冷启动发现新架构方向（如DPAS tiling方案），然后用温启动精细优化。

3. **USER_INSTRUCTIONS效果**：虽然instructions描述了"64 WI, SLM, TILE 32×64×32"架构，但在冷启动场景下LLM成功收敛到该架构，证明instructions的指导作用。

4. **投资回报**：48个branch/1.1M tokens，获得26.2x加速（vs朴素实现），但仅达17.2% XMX利用率。对比v15用类似token量达到81.6%利用率。

---

*实验环境: Intel B580 (BMG), oneAPI 2025.0.4, PyOpenCL, Intel NEO driver 25.48.36300.8*
*报告生成时间: 2026-05-29*
