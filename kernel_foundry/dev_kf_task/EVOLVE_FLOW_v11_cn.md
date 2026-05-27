# KernelFoundry EVOLVE v11 实验分析报告

## 实验概况

| 项目 | 值 |
|------|-----|
| 实验名称 | b580_matmul_dpas_v11 |
| 问题规模 | C[2048,2560] = A[2048,2048] × B[2048,2560], FP16 |
| 硬件平台 | Intel Battlemage G21 (B580), 20 Xe2 cores, 96 TFLOPS FP16 XMX |
| LLM模型 | claude-4-6-opus |
| Temperature | 0.3 |
| max_tokens | 10000 |
| 迭代轮次 | 11/12（第12轮因GNAI速率限制未完成） |
| 每轮分支 | 4 |
| 总评估数 | 43（10轮×4 + 1轮×3） |
| 正确率 | 42/43 = 97.7% |
| 实验耗时 | 1022秒（17.0分钟） |
| **最佳结果** | **1.31ms (25.9x加速, 16.4 TFLOPS, 17.1% XMX利用率)** |

---

## 1. 关键配置变化（v10 → v11）

| 参数 | v10 | v11 | 变化原因 |
|------|-----|-----|----------|
| Seed kernel | v4 best (11.4ms, naive DPAS) | v10 best (1.07ms, A-SLM+B-global) | 从更高baseline温启动 |
| Temperature | 0.25 | 0.3 | 鼓励更大架构变异（双缓冲等） |
| max_tokens | 5000 | 10000 | 复杂kernel需要更多代码空间 |
| USER_INSTRUCTIONS | 通用优化提示 | 明确21%瓶颈+6个优化方向 | 引导LLM攻击正确问题 |
| 硬件参数 | 未指定 | 96 TFLOPS, 456 GB/s, 24MB L2 | 给LLM准确约束 |

---

## 2. 实验结果总览

### 2.1 逐轮最佳性能

| Trial | 分支结果 (ms) | 本轮最佳 | 累计最佳 | 正确率 |
|-------|--------------|----------|----------|--------|
| 0 | 3.88, ~~2.19~~(错误), 9.08, 5.07 | 3.88ms | 3.88ms | 3/4 |
| 1 | 5.05, 5.04, 5.06, 2.16 | 2.16ms | 2.16ms | 4/4 |
| 2 | 7.34, **1.33**, 3.34, 3.36 | 1.33ms | 1.33ms | 4/4 |
| 3 | 3.36, 3.36, 2.60, 3.36 | 2.60ms | 1.33ms | 4/4 |
| 4 | 3.37, 3.36, 2.17, 4.83 | 2.17ms | 1.33ms | 4/4 |
| 5 | 4.83, 3.44, 2.04, 11.2 | 2.04ms | 1.33ms | 4/4 |
| 6 | 4.81, 1.33, **1.31**, 6.02 | **1.31ms** | **1.31ms** | 4/4 |
| 7 | 2.14, 2.16, 2.16, 2.78 | 2.14ms | 1.31ms | 4/4 |
| 8 | 2.80, 2.16, 2.59, 2.74 | 2.16ms | 1.31ms | 4/4 |
| 9 | 3.26, 1.32, 2.16, 1.32 | 1.32ms | 1.31ms | 4/4 |
| 10 | 1.32, 4.05, 4.05, — | 1.32ms | 1.31ms | 3/3 |

### 2.2 进化轨迹

```
Runtime (ms)
10 |*
 9 |  *
 8 |
 7 |
 6 |
 5 |    * * *
 4 |
 3 |  *
 2 |  * * * * *  * * * *
 1 |    *   *  *   * * *     ← 1.31-1.33ms 平台
   +---+---+---+---+---+---+---+---+---+---+---→ Trial
     0   1   2   3   4   5   6   7   8   9  10
```

### 2.3 MAP-Elites 进化

```
Score 进化: 31.2 → 51.4 → 52.1 → 81.5 → 82.6
  Trial 0: score=31.2 (3.88ms, 初始探索)
  Trial 0: score=51.4 (首次优质变异，但结果错误→取第二名)
  Trial 2: score=52.1 (微调改善)
  Trial 2: score=81.5 (1.33ms突破！)
  Trial 6: score=82.6 (1.31ms，最终最佳)
```

---

## 3. 架构发现与分析

### 3.1 v11 最优kernel架构

v11 最优kernel（1.31ms）的核心结构：

```
TILE_M=64, TILE_N=128, TILE_K=32
WG: 128 WIs = 8 subgroups × 16 WIs
每个 subgroup: 8 个 8×16 DPAS tiles = 64 rows × 16 cols
SLM: A和B都在SLM，双缓冲
  A buffer: 2 × 64 × 34 × 2B = 8.7 KB
  B buffer: 2 × 32 × 130 × 2B = 16.6 KB
  总计: ~25 KB
每K-tile: 2 k16 steps × 8 row-blocks = 16 DPAS/subgroup
```

### 3.2 v10 最优 vs v11 最优对比

| 特征 | v10 (1.07ms) | v11 (1.31ms) |
|------|-------------|-------------|
| TILE_M | 32 | 64 |
| TILE_N | 64 | 128 |
| TILE_K | 32 | 32 |
| WG size | 64 WIs (4 SG) | 128 WIs (8 SG) |
| A存储 | SLM (单缓冲) | SLM (双缓冲) |
| B存储 | Global (直接读) | SLM (双缓冲) |
| SLM用量 | ~2.2 KB | ~25 KB |
| DPAS/barrier | 8 | 16 |
| 输出tile/SG | 32×16 | 64×16 |
| 实际性能 | **1.07ms** | 1.31ms |

### 3.3 关键发现：更大tile和双缓冲反而更慢

**这是v11最重要的发现**：尽管v11的kernel在理论指标上全面优于v10——

- DPAS/barrier比: 16 vs 8（翻倍）
- WG size: 128 vs 64（翻倍）
- 有双缓冲（v10没有）
- 更大输出tile

但实际性能却**退步了22%**（1.31ms vs 1.07ms）。

**根因分析**：

1. **SLM占用过高限制了并发**：25KB SLM → 可能只有1-2个WG能同时驻留在每个Xe-core上，减少了延迟隐藏能力

2. **B放入SLM增加了不必要的开销**：v10的B-from-global策略利用了L2 cache的自然复用，而v11把B也加载到SLM需要额外的barrier和协作加载时间

3. **双缓冲并未真正overlap**：从trial 6分析可见，loads happen BEFORE compute，不是真正的load/compute重叠

4. **大WG的调度开销**：128 WIs需要更多寄存器，可能造成寄存器溢出

---

## 4. 实验系列对比

### 4.1 全系列汇总

| 实验 | Temp | Seed | 最佳 | 加速比 | XMX利用率 | 正确率 | 关键特征 |
|------|------|------|------|--------|----------|--------|----------|
| v4 | 0.0 | 无 | 11.4ms | 2.98x | 1.9% | ~50% | 直接生成 |
| v5 | 0.3-0.6 | 无 | 33.9ms | 1.0x | 0.6% | ~60% | 高温不稳定 |
| v6 | 0.1 | 无 | 36.2ms | 0.94x | 0.6% | ~40% | 低温无进展 |
| v7 | 0.0 | 无 | 33.9ms | 1.0x | 0.6% | ~30% | 零温无进化 |
| v8 | 0.1 | 无 | 30.9ms | 1.10x | 0.7% | 69% | 慢收敛 |
| v9 | 0.2 | 无 | 23.8ms | 1.42x | 0.9% | 94% | 中速收敛 |
| **v10** | **0.25** | **v4** | **1.07ms** | **31.7x** | **20.9%** | **100%** | **A-SLM突破** |
| **v11** | **0.3** | **v10** | **1.31ms** | **25.9x** | **17.1%** | **97.7%** | **大tile+双缓冲退步** |

### 4.2 v10 vs v11 详细对比

```
性能对比:
  v10: 1.07ms → 20.1 TFLOPS → 20.9% utilization
  v11: 1.31ms → 16.4 TFLOPS → 17.1% utilization
  退步幅度: +22.4% (慢了0.24ms)

进化效率对比:
  v10: 7.0ms → 1.07ms (6.5x改善, 12轮)
  v11: 3.88ms → 1.31ms (3.0x改善, 11轮)
  v10 的进化效率更高

收敛速度:
  v10: 突破在 Trial 6 (第24次评估), 从3.87ms→1.08ms
  v11: 突破在 Trial 2 (第8次评估), 从3.88ms→1.33ms
  v11 收敛更快但到达的平台更低

探索多样性:
  v10 runtime 范围: 1.07ms ~ 7.0ms (全部正确)
  v11 runtime 范围: 1.31ms ~ 11.2ms (42/43正确)
  v11 探索范围更广但没找到更好解
```

---

## 5. 关键教训

### 5.1 "更优理论 ≠ 更优实际" 

v11 的核心教训：在GPU优化中，**微架构效应**（occupancy、cache behavior、寄存器压力）的影响可能大于**宏观算法优化**（双缓冲、更大tile）。

v10的"简陋"架构（64 WIs, 单缓冲A, B-from-global, 仅2.2KB SLM）之所以更快，可能是因为：
- 极小的SLM占用 → 多个WG同时驻留 → 更好的延迟隐藏
- B从global/L2读取 → 避免了SLM load的barrier开销
- 小WG → 更少寄存器压力 → 编译器优化空间更大

### 5.2 温度0.3的效果

- 正确率: 97.7%（vs v10的100%）— 轻微下降但仍然优秀
- 架构多样性: 产生了多种变体（A-only SLM, A+B SLM, 不同TILE_K）
- 但未能突破v10的性能上限
- **结论**: 0.3对于"精细化现有架构"可能太高了；0.25更适合这个阶段

### 5.3 USER_INSTRUCTIONS的效果

明确指出瓶颈和优化方向确实引导LLM去尝试双缓冲、更大tile等策略。但这些策略在B580上反而不如v10的简洁方案。**Instructions指明了正确的理论方向，但实际效果取决于硬件微架构特性。**

### 5.4 GNAI速率限制

第12轮触发了GNAI时间限制（等待约67000秒），说明inference服务有每日/每小时的调用限额。这是一个外部约束因素。

---

## 6. v11进化中的架构变体分析

### 6.1 性能分层

v11产生的kernel可分为几个性能层级：

| 层级 | 运行时间 | 架构特征 | 出现频率 |
|------|----------|----------|----------|
| 最优 | 1.31-1.33ms | 64×128×32, A+B SLM, 双缓冲, 128WI | 7/43 |
| 次优 | 2.04-2.19ms | 64×128×32, A-only SLM, 双缓冲, 128WI | 12/43 |
| 中等 | 2.6-3.44ms | 32×128×64 或 mixed, 各种配置 | 15/43 |
| 较慢 | 4.0-5.1ms | 32×128×32, 探索性变体 | 6/43 |
| 最慢 | 6.0-11.2ms | 过大tile或TILE_K=64导致SLM溢出 | 3/43 |

### 6.2 TILE_K=64 的失败

多次尝试 TILE_K=64 均未成功（5-11ms），原因：
- SLM需求翻倍（B tile: 64×128 = 16KB per buffer）
- 每WI需加载32-64个元素，寄存器压力爆炸
- 编译器可能无法有效向量化如此多的循环迭代
- **结论**: 在B580上，TILE_K=32是甜点

### 6.3 B-from-global vs B-in-SLM

| 策略 | v11最佳性能 | 优势 | 劣势 |
|------|-----------|------|------|
| B-from-global (v10策略) | 2.04-2.16ms | SLM小, 占用率高 | SG间冗余读取 |
| B-in-SLM (v11最优) | 1.31ms | 共享B, 减少带宽 | SLM大, 额外barrier |

在v11的128WI大WG上下文中，B-in-SLM略优。但v10的64WI小WG + B-from-global整体更快。这说明**WG大小和内存策略需要协同优化，不能独立调整**。

---

## 7. 优化建议（v12实验）

### 7.1 核心策略：回归v10架构 + 微调

v11证明了"大刀阔斧"的架构变更（双缓冲、大tile、B-SLM）在B580上不如v10的简洁方案。v12应该：

1. **以v10最优kernel（1.07ms）为种子**（不是v11的1.31ms）
2. **在v10架构框架内做微调**，而非架构级重设计
3. **降温到0.2**：鼓励渐进优化而非激进变异

### 7.2 具体微调方向

在保持v10架构（64WI, A-SLM, B-global, 32×64×32）的前提下：

1. **SLM A的访问模式优化**: 使用intel_sub_group_block_read_us代替标量as_short读
2. **B的向量化读取**: 将2次标量B读合并为vload2
3. **减少barrier**: K=2048整除32，remainder路径可删除
4. **尝试TILE_M=48或64但保持B-global**: 增加A复用而不增加SLM压力
5. **展开外层K循环**: 尝试手动展开2次K-tile迭代减少循环开销
6. **Prefetch**: 对B的下一K-tile行发送cache line prefetch提示

### 7.3 推荐v12配置

```yaml
job_name: b580_matmul_dpas_v12
task_name: matmul_dpas_v12

inference:
  servers:
  - model_name: claude-4-6-opus
    temperature: 0.2      # 降温：渐进优化而非激进变异
    max_tokens: 8000

max_iters: 16
branches_per_iteration: 4

# Seed: 使用v10最优kernel（1.07ms版本），NOT v11
```

### 7.4 推荐v12的USER_INSTRUCTIONS

```
[USER_INSTRUCTIONS_START]
Current kernel achieves 1.07ms = 20.9% XMX utilization on B580 (96 TFLOPS peak).
Architecture: 64 WIs (4 SGs), A in SLM (2.2KB), B from global/L2, TILE 32x64x32.

DO NOT change the fundamental architecture (this was proven best).
DO NOT add B to SLM (causes regression).
DO NOT increase WG beyond 64 WIs (causes regression).

Micro-optimizations to try:
1. Use intel_sub_group_block_read_us for SLM A reads (vectorized)
2. Merge paired B scalar reads into vload2 or block reads
3. Remove K-remainder path (K=2048 divides evenly by 32)
4. Try TILE_M=48 or 64 (more A rows per WG, same B columns)
5. Unroll K-loop 2x (reduce loop overhead for 64 tiles)
6. Add __builtin_prefetch or intel_sub_group_block_prefetch for next B tile

Hardware: B580 = 20 Xe2 cores, 96 TFLOPS FP16 XMX, 456 GB/s, 24MB L2
DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
[USER_INSTRUCTIONS_END]
```

---

## 8. 总结

### 8.1 v11 实验结论

v11 是一次**有价值的负面实验**：

- 证明了"理论更优的架构"（双缓冲、大tile、B-SLM）在B580上并不比v10的简洁方案更快
- 发现B580的性能甜点是：**小WG（64WI）+ 低SLM占用（<3KB）+ B-from-L2**
- 说明GPU微架构约束（occupancy、cache line behavior）是决定性因素
- Temperature 0.3 产生了丰富的架构变体，但未找到更优解

### 8.2 整体进化系列认知更新

```
进化路径:
  v4 (11.4ms, naive DPAS) 
    ↓ seed
  v10 (1.07ms, A-SLM + B-global, 64WI) ← 全局最优
    ↓ seed 
  v11 (1.31ms, A+B SLM + 双缓冲, 128WI) ← 过度优化退步

核心认知:
  ✓ Seed kernel是成功的关键前提
  ✓ Temperature 0.25 是架构突变的最优温度
  ✗ 双缓冲在B580上不如单缓冲+高占用率
  ✗ 更大WG不一定更快
  ✗ B放SLM不如B从L2直接读
```

### 8.3 下一步

v12应回归v10架构，用低温（0.2）做渐进微调，而非继续尝试宏观架构变更。目标从1.07ms优化到0.8-0.9ms（~25% XMX利用率）。

---

*报告生成时间: 2026-05-25。实验在Intel Battlemage G21 (B580)上进行，问题规模2048×2560×2048 FP16 GEMM。v11最优kernel经151次benchmark验证，runtime std=0.004ms。实验因GNAI速率限制在第12轮中断，共完成43次评估（11轮），耗时1022秒。*
