# v14 实验分析报告：优化约束 + oneDNN Tips 指导进化

## 1. 实验概述

| 参数 | 值 |
|------|------|
| 任务名 | b580_matmul_dpas_v14 |
| GPU | Intel B580 (Battlemage G21, 20 Xe2 cores) |
| 问题 | C[2048,2560] = A[2048,2048] × B[2048,2560], FP16, 21.5 GFLOP |
| 理论峰值 | 96 TFLOPS FP16 XMX, 理论最小时间 0.224ms |
| 模型 | claude-4-6-opus |
| 温度 | 0.25 |
| 迭代 | 12 trials × 4 branches = 48 kernel variants |
| 种子kernel | v12最佳 (0.948ms, 23.6% XMX, 双缓冲A+B-global, 64 WIs) |
| 总耗时 | 1116秒 (~18.6分钟) |
| 进化模式 | evolve_mode + gradient_tracking + optimization_aware_prompting |

## 2. 核心策略

v14 采用 **强约束 + 丰富优化指导** 的策略，在v12严格约束基础上增加来自v13教训的新DO NOT规则，同时提供oneDNN gemmstone实践经验作为优化参考。

### 2.1 USER_INSTRUCTIONS（强化版）

```
- 保留v12的DO NOT约束:
    DO NOT change the fundamental architecture (proven best)
    DO NOT add B to SLM (causes regression)
    DO NOT increase WG beyond 64 WIs (causes regression)
- 新增v13验证的DO NOT:
    DO NOT use 32×256 tile (proven inferior to 32×64)
    DO NOT use K-step smaller than 32
- 允许的微优化方向:
    - Combine double-buffering with K-loop 2x unroll
    - More aggressive B prefetch strategies
    - Explore different SLM strides to avoid bank conflicts
    - Try async copy (intel_sub_group_block_read for A loads)
    - Use intel_sub_group_block_read_us for SLM A reads
    - Merge paired B scalar reads into vload2 or block reads
    - Remove K-remainder path (K=2048 divides evenly by 32)
    - Try TILE_M=48 or 64
    - Unroll K-loop 2x
    - Add prefetch for next B tile
- 全新: 37条oneDNN gemmstone优化Tips（涵盖计算指令、数据布局、
  内存访问、预取策略、SLM使用、K-loop流水线、WG调度、C写回等）
```

### 2.2 与前序实验配置对比
| 参数 | v12 | v13 | v14 |
|------|-----|-----|-----|
| 温度 | 0.2 | 0.25 | 0.25 |
| USER_INSTRUCTIONS | 3条DO NOT + 方向 | 无限制 | **5条DO NOT + 微优化列表 + 37条Tips** |
| 种子 | v10 best (1.07ms) | v12 best (0.948ms) | v12 best (0.948ms) |
| max_iters | 12 | 20 | 12 |
| 特殊内容 | — | — | oneDNN优化Tips |

## 3. 实验结果

### 3.1 逐Trial性能

| Trial | Branch 0 | Branch 1 | Branch 2 | Branch 3 | Best (ms) | XMX% |
|-------|----------|----------|----------|----------|-----------|------|
| 0 | 1.21 | FAIL | FAIL | FAIL | 1.21 | 18.5% |
| 1 | FAIL | FAIL | 3.80 | FAIL | 3.80 | 5.9% |
| 2 | 3.81 | 3.82 | FAIL | 2.77 | 2.77 | 8.1% |
| 3 | 3.82 | 1.33 | FAIL | FAIL | 1.33 | 16.8% |
| 4 | 3.13 | **1.14** | 1.41 | FAIL | **1.14** | 19.7% |
| 5 | **1.05** | 2.53 | 2.70 | 1.17 | **1.05** | 21.3% |
| 6 | **1.03** | 1.05 | 2.28 | FAIL | **1.03** | 21.7% |
| 7 | 3.75 | 2.60 | 2.86 | 2.99 | 2.60 | 8.6% |
| 8 | 3.48 | 3.78 | FAIL | 1.13 | 1.13 | 19.8% |
| 9 | **1.01** | 2.74 | 3.66 | FAIL | **1.01** | 22.2% |
| 10 | 2.23 | 1.02 | 2.80 | 1.02 | 1.02 | 22.0% |
| 11 | 2.73 | 2.44 | FAIL | 2.62 | 2.44 | 9.2% |

### 3.2 关键指标

| 指标 | v14值 | v12值 (对比) | v13值 (对比) |
|------|-------|-------------|-------------|
| **最佳性能** | **1.01ms** | **0.948ms** | 1.14ms |
| **XMX利用率** | 22.2% | 23.6% | 19.7% |
| 相对种子变化 | **+6.5% 回归** (0.948→1.01) | -11.4% 提升 | +20.3% 回归 |
| 正确率 | 36/48 = 75.0% | 46/48 = 95.8% | 70/80 = 87.5% |
| 编译率 | 48/48 = 100% | 48/48 = 100% | 80/80 = 100% |
| Sub-1.0ms | **0/48 = 0%** | 19/48 = 39.6% | 0/80 = 0% |
| Sub-1.1ms | 5/36 = 13.9% | — | 0/70 = 0% |
| Sub-1.2ms | 8/36 = 22.2% | — | 14/70 = 20.0% |
| Over 2.5ms | 14/36 = 38.9% | — | 48/70 = 68.6% |
| 中位数 | 2.26ms | ~0.99ms | 3.99ms |

### 3.3 性能分布

```
<1.0ms:   (0, 0%)          ← 种子的0.948ms未被复现
1.0-1.1:  ■■■■■ (5, 10.4%)   ← 最佳区间 (新！)
1.1-1.5:  ■■■■■■ (6, 12.5%)
1.5-2.5:  ■■ (2, 4.2%)
2.5-4.0:  ■■■■■■■■■■■■■■ (14, 29.2%)  ← 主要分布
>4.0:     ■ (1, 2.1%)       ← 极少数极差结果
FAIL:     ■■■■■■■■■■■■ (12, 25.0%)     ← 正确性失败
```

## 4. 进化动态分析

### 4.1 三阶段进化模式

**阶段1: 探索失败 (Trial 0-3)**
- Trial 0: 仅1/4通过正确性(1.21ms)，LLM尝试K-loop 2x unroll + double-buffering + SLM stride padding
- Trial 1-2: 大量失败，LLM在double-buffering中引入bug
- Trial 3: 开始恢复(1.33ms)
- 高失败率 (9/16 = 56%)
- 根因: LLM同时尝试太多优化(2x unroll + double-buffer + stride padding)导致bug

**阶段2: 核心发现 (Trial 4-6)**
- LLM学到: **单缓冲SLM + stride=34 padding + boustrophedon DPAS > 双缓冲**
- Trial 4: 1.14ms (修复了double-buffer中的bug)
- Trial 5: 1.05ms (切换到单缓冲，更简单更快)
- Trial 6: **1.03ms** (优化interleaving + boustrophedon)
- 关键洞察: 对于小A tile(2.2KB)，单缓冲+2 barriers比双缓冲+4 barriers更快

**阶段3: 稳定但回归震荡 (Trial 7-11)**
- Trial 7: 回归到2.60ms (LLM再次尝试double-buffer + 2x unroll)
- Trial 8: 1.13ms (修复回滚到单缓冲)
- Trial 9: **1.01ms** (最佳! 进一步优化interleaving)
- Trial 10: 1.02ms×2 (稳定维持)
- Trial 11: 全面回归 (2.44ms best, LLM尝试vload2导致正确性失败)

### 4.2 v14发现的关键微优化

| 优化技术 | 首次出现 | 效果 | 评价 |
|----------|---------|------|------|
| SLM stride=34 padding | Trial 0 | 减少bank conflict | ✅ 有效，被所有好结果采用 |
| 单缓冲 > 双缓冲 | Trial 5 | 1.05 vs 1.21 | ✅ 小A tile时单缓冲更优 |
| Boustrophedon DPAS | Trial 5 | ~2%改善 | ✅ 轻微改善，无代价 |
| B loads interleaving | Trial 6 | 1.03 vs 1.05 | ✅ 将B load穿插在A read之间 |
| K-loop 2x unroll | Trial 0 | 增加复杂度、bug频发 | ❌ 对此tile无益 |
| vload2 for B | Trial 10 | 正确性失败 | ❌ B行间距=N，不可vload2 |
| Double-buffer + 2x | Trial 7 | 2.60-3.75ms | ❌ 显著回归 |

### 4.3 v14最佳(1.01ms) vs v12种子(0.948ms)差异分析

两者的关键差异:

1. **SLM布局**: 
   - v12种子: stride=32 (无padding)，使用precomputed offsets
   - v14最佳: stride=34 (padded)，使用precomputed offsets
   - 分析: stride=34应该减少bank conflicts，但对齐变化可能影响block_read效率

2. **缓冲策略**:
   - v12种子: 双缓冲（load next A while computing current）
   - v14最佳: 单缓冲（load → barrier → compute → barrier → repeat）
   - 分析: v12的双缓冲在当前kernel中被验证为最优(0.948ms)，v14未能复现

3. **B访问模式**:
   - v12种子: B_us[b_off], B_us[b_off + N] with precomputed N2 stride
   - v14最佳: 相同模式但interleaved with A reads
   - 分析: interleaving有微小改善但无法弥补双缓冲的缺失

4. **DPAS顺序**:
   - v12种子: 顺序执行 acc0→acc1→acc2→acc3
   - v14最佳: 第二步boustrophedon (acc3→acc2→acc1→acc0)
   - 分析: 对性能影响微乎其微

**为什么v14达不到0.948ms**:
- v14的LLM在试图应用oneDNN tips时，放弃了双缓冲策略转向单缓冲
- 单缓冲在K-loop中有2个barriers per tile (load前barrier + load后barrier)
- 双缓冲只有1个barrier per tile (load完barrier, compute与下次load重叠)
- 对于64 K-tiles，双缓冲节省64个barriers的延迟
- 结论: **v12的双缓冲是达到sub-1ms的关键**

### 4.4 MAP-Elites状态

所有variant集中在cell `(2, 3, 3, 0)`:
- 最终精英score: 105.7 (vs v12的112.3, v13的94.2)
- 4次精英更替: 89.0 → 94.2 → 101.9 → 103.7 → 105.7
- 有效多样性: 极低（单一cell主导）

## 5. 关键发现

### 5.1 oneDNN Tips的效果评估

| Tips类别 | LLM采纳情况 | 实际效果 |
|----------|------------|---------|
| SLM bank conflict avoidance (stride padding) | ✅ 全部采纳 | ✅ 有效，所有好结果都用 |
| Boustrophedon DPAS ordering | ✅ 中期采纳 | ✅ 微小改善 |
| Load pipelining / interleaving | ✅ 反复尝试 | ⚠️ 有时改善，有时引入bug |
| Double/triple buffering | ✅ 多次尝试 | ❌ 在此tile下不如单缓冲 |
| K-loop unroll | ✅ 多次尝试 | ❌ 增加复杂度无收益 |
| vload2 / block load for B | ✅ Trial 10尝试 | ❌ 语义错误，B非连续 |
| Prefetch strategy | ⚠️ 未深入尝试 | — 未验证 |
| 2D Block Load | ❌ 未尝试 | — |
| Stream-K | ❌ 未尝试(被DO NOT限制) | — |
| Named barriers | ❌ 未尝试 | — |

**结论**: oneDNN Tips中只有SLM padding和boustrophedon被有效利用。大量Tips过于高级或与当前tile配置不兼容。

### 5.2 正确性问题分析

v14的正确性率(75%)显著低于v12(95.8%)和v13(87.5%):

| 失败原因 | 次数 | 占比 |
|----------|------|------|
| Double-buffer指针bug | 6 | 50% |
| vload2语义错误 | 3 | 25% |
| K-loop 2x unroll越界 | 2 | 17% |
| SLM stride不匹配 | 1 | 8% |

**根因**: 过多的优化Tips鼓励LLM进行复杂重构，增加了bug引入概率。

### 5.3 v14 vs v12 vs v13 三方对比

| 指标 | v12 (严格约束) | v13 (宽松约束) | v14 (约束+Tips) |
|------|---------------|---------------|-----------------|
| 最佳性能 | **0.948ms** | 1.14ms | 1.01ms |
| Sub-1ms率 | **39.6%** | 0% | 0% |
| 回归率(>2ms) | 33.3% | 72.9% | 38.9% |
| 正确率 | **95.8%** | 87.5% | 75.0% |
| 架构保持 | ✅ 高度一致 | ❌ 频繁改变 | ⚠️ 中等 |
| 双缓冲保持 | ✅ 始终保持 | ❌ 放弃 | ⚠️ 反复切换 |

## 6. 与全系列实验对比

| 实验 | 策略 | 温度 | 最佳 | XMX% | 对种子 |
|------|------|------|------|------|--------|
| v4 | 冷启动 | 0.0 | 11.4ms | 2.0% | — |
| v10 | 温启动(v4种子) | 0.25 | 1.07ms | 20.9% | -90.6% |
| v11 | 自由探索 | 0.3 | 1.31ms | 17.1% | +22.4% 回归 |
| **v12** | **严格约束** | 0.2 | **0.948ms** | **23.6%** | **-11.4% 提升** |
| v13 | 宽松约束 | 0.25 | 1.14ms | 19.7% | +20.3% 回归 |
| **v14** | **约束+Tips** | 0.25 | **1.01ms** | 22.2% | +6.5% 回归 |

**趋势**: v12 > v14 > v10 > v13 > v11 > v4

v14的定位: 比v13好得多(1.01 vs 1.14)，但仍不如v12(1.01 vs 0.948)。

## 7. 优化建议

### 7.1 v15方向

基于v14实验教训，v15应该：

1. **回归v12的exact seed kernel** (包括其双缓冲实现)
2. **新增DO NOT约束**:
   - DO NOT switch from double-buffer to single-buffer SLM
   - DO NOT try K-loop 2x unroll (adds complexity without benefit)
   - DO NOT use vload2 for B (B rows are N-stride apart)
   - DO NOT add SLM stride padding beyond 32 (the seed's stride=32 + double-buffer is proven best)
3. **精简优化Tips** (只保留对当前tile有效的):
   - Interleave B loads between A reads and DPAS (proven: 1.03→1.01)
   - Boustrophedon DPAS ordering (proven: minor improvement)
   - Precompute all address offsets outside loop
   - Try intel_sub_group_block_prefetch for B (未验证)
4. **温度降到0.15**: 减少创新冲动，聚焦微调
5. **保护性约束**: 明确声明 "The double-buffering approach is optimal and must be preserved"

### 7.2 为什么Tips过多反而有害

1. **信息过载**: 37条Tips中大部分不适用于当前tile，但LLM无法判断适用性
2. **鼓励重构**: Tips暗示"还有很多可优化的"，促使LLM进行不必要的架构改变
3. **正确性风险**: 复杂优化(double-buffer + stride + unroll)的组合容易引入subtle bugs
4. **分散注意力**: LLM花时间在inapplicable tips上而非focus on proven bottleneck

### 7.3 最佳实践总结

| 策略 | 效果 | 原因 |
|------|------|------|
| 严格DO NOT约束 | ✅ 最优 | 聚焦搜索空间 |
| 少量针对性Tips | ✅ 有效 | 引导正确方向 |
| 大量通用Tips | ⚠️ 降低正确率 | 鼓励不必要重构 |
| 无约束 | ❌ 最差 | 浪费探索预算 |

**最优配置 = 严格约束(DO NOT) + 2-3条针对性micro-optimization hints + 低温度**

## 8. 结论

v14实验是一个**混合结果**:

1. **正面**: 达到1.01ms（比v13的1.14好13%），证明约束+Tips优于无约束
2. **负面**: 仍未达到v12的0.948ms，且正确率(75%)是全系列最低
3. **关键发现**: 
   - oneDNN Tips中多数不适用于当前tile配置
   - 单缓冲+stride=34 < 双缓冲+stride=32 (v12的组合是最优的)
   - Tips过多导致LLM放弃已验证最优的双缓冲策略
   - Boustrophedon和interleaving是有效的micro-optimization
4. **核心教训**: **"Less is more" — 信息越精准越好，信息越多反而干扰决策**

v12仍是最佳实验，证明: 对于kernel micro-optimization，**minimal precise constraints > rich detailed guidance**。
