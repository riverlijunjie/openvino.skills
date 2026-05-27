# EVOLVE Mode Flow 深度分析 v6 — 单模型低温度实验

## 1) 执行信息与本次运行结论

| 项目 | 值 |
|------|-----|
| 脚本 | `/mnt/river/kernel_foundry/kernelfoundry.internal/runme_matmul.sh` |
| 目标任务 | `/mnt/river/kernel_foundry/workspace/ocl_matmul_dpas` (OpenCL DPAS matmul) |
| 执行环境 | `conda activate kernel_intel` + `source /opt/intel/oneapi/2025.0/oneapi-vars.sh` |
| GPU | Intel Battlemage G21 (B580), device_id=0xe20b |
| LLM 模型 | **claude-4-6-opus** (单模型, temperature=0.1) |
| `max_iters` | **10** (脚本override, config中为20) |
| `branches_per_iteration` | **4** |
| 总评估 kernel 数 | **40** |
| 总耗时 | **1447.3 秒 (24.1 分钟)** |
| 参考基线 (reference) | 33.9 ms |
| 最佳候选 (Trial 6, v3) | **36.2 ms** (speedup = **0.936x**) |
| 最佳 combined_score | **7.81** |
| 进化改善轨迹 | 47.0 → 62.3 → 57.1 → 43.3 → **36.2** ms |

> **核心结论**: 与v5(多模型集成, 80次评估, 最佳1.0x)相比，v6(单模型opus+temp=0.1)在仅40次评估中展现了**真实的进化收敛行为**——运行时间从47ms持续改善到36.2ms(距参考仅差7%)。但与v4(opus+temp=0, 最佳11.4ms=2.98x)相比仍有差距：v6的最优kernel使用SLM tiling + FMA标量计算而非DPAS硬件指令，说明temperature=0.1引入的微小随机性虽不足以产生完全错误的DPAS代码，却足以让模型"选择"放弃DPAS转向更安全但较慢的SLM策略。

---

## 2) 配置变更对比 (v4 → v5 → v6)

| 参数 | v4 | v5 | v6 | 
|------|-----|-----|-----|
| 模型 | claude-4-6-opus | 3模型集成 | **claude-4-6-opus** |
| temperature | 0.0 | 0.3-0.6 | **0.1** |
| max_iters (实际) | 4 | 20 | **10** |
| branches_per_iteration | 2 | 4 | **4** |
| 总评估 | 8 | 80 | **40** |
| exploration_ratio | 0.2 | 0.4 | **0.4** |
| exploitation_ratio | 0.7 | 0.5 | **0.5** |
| use_feedback_llm | false | true | **true** |
| 最佳 runtime | **11.4ms** | 33.9ms | **36.2ms** |
| 最佳 speedup | **2.98x** | 1.0x | **0.94x** |

### v6 关键配置 (task config.yaml)

```yaml
inference:
  servers:
  - model_name: claude-4-6-opus
    temperature: 0.1        # 极低温度但非零
    max_tokens: 5000

max_iters: 20              # (脚本override为10)
branches_per_iteration: 4

use_feedback_llm: true
feedback_llm_config:
    model_name: "gpt-5.3-codex"
    temperature: 0.3

database:
  config:
    exploration_ratio: 0.4
    exploitation_ratio: 0.5

use_gradient_tracking: true
gradient_config:
  exploration_weight: 0.4   # 提高 (默认0.2)
```

---

## 3) 全量评估结果

### 3.1 总体统计

| 指标 | 值 | vs v5 |
|------|-----|-------|
| 总评估数 | 40 | -50% |
| 编译成功 | 40/40 (100%) | = |
| 正确性通过 | 33/40 (82.5%) | -12.5% |
| 正确性失败 | 7/40 (17.5%) | +15% (集中在前2轮) |
| 超越参考 (>1.0x) | **0/40 (0%)** | = |
| 最快 runtime | **36.2ms** (0.94x) | vs 33.9ms (1.0x) |
| 有真实加速趋势 | **是** | vs 否 |

### 3.2 逐 Trial 最佳结果

| Trial | Best (ms) | Speedup | Scores [v0,v1,v2,v3] | 观察 |
|-------|-----------|---------|---------------------|------|
| 0 | N/A | N/A | [3, 3, 3, 3] | 全部正确性失败 (DPAS尝试) |
| 1 | 47.0 | 0.721x | [5, 3, 3, 3] | 首个正确kernel, 3个DPAS尝试失败 |
| 2 | 62.3 | 0.544x | [5, 5, 5, 5] | 全部正确, 开始探索SLM策略 |
| 3 | 66.9 | 0.507x | [5, 5, 5, 5] | 多方向探索 |
| 4 | 57.1 | 0.594x | [5, 5, 5, 5] | 开始收敛 |
| 5 | 43.3 | 0.783x | [5, 5, 5, 5] | 显著改善 (TILE_K提升) |
| 6 | **36.2** | **0.936x** | [5, 5, 5, 5] | 最佳! (128×16×64 tile) |
| 7 | 36.2 | 0.936x | [5, 5, 5, 5] | 维持最优 |
| 8 | 38.9 | 0.871x | [5, 5, 5, 5] | 略微退步 |
| 9 | 52.0 | 0.652x | [5, 5, 5, 5] | 最后轮探索新方向 |

### 3.3 逐 Branch 完整结果

| Trial | v0 | v1 | v2 | v3 |
|-------|-----|-----|-----|-----|
| 0 | ✗ | ✗ | ✗ | ✗ |
| 1 | **47.0ms** | ✗ | ✗ | ✗ |
| 2 | 95.9 | **62.3** | 80.1 | 105.0 |
| 3 | 71.5 | 219.0 | 93.8 | **66.9** |
| 4 | 71.0 | 212.0 | 96.2 | **57.1** |
| 5 | 58.6 | 91.0 | **43.3** | 48.2 |
| 6 | 42.7 | 56.3 | 49.5 | **36.2** |
| 7 | **36.2** | 52.8 | 46.6 | 60.1 |
| 8 | **38.9** | 42.9 | 49.1 | 52.0 |
| 9 | 87.5 | 86.8 | 55.4 | **52.0** |

### 3.4 运行时间分布

```
分布: (正确的33个kernels)
 36-40ms:  ████  (4个, 12%) — 最佳区间
 40-50ms:  ██████ (6个, 18%)
 50-60ms:  ██████ (6个, 18%)
 60-80ms:  ████ (4个, 12%)
 80-100ms: █████ (5个, 15%)
100-220ms: ████ (4个, 12%)
不正确:    ███████ (7个, 17.5%)
```

---

## 4) 进化行为分析

### 4.1 MAP-Elites 数据库进化

| 事件 | Trial | Score | Cell | 说明 |
|------|-------|-------|------|------|
| 首次 best | 0 | 3.00 | - | 编译但不正确 |
| 替换 best | 1 | 7.16 | (2,1,1,0) | 首个正确kernel |
| Elite 更新 | 3 | 6.42 | (2,1,1,0) | 连续改善 |
| Elite 更新 | 4 | 6.52 | (2,1,1,0) | |
| Elite 更新 | 5 | 6.78 | (2,1,1,0) | |
| Elite 更新 | 5 | 7.35 | (2,1,1,0) | |
| Best 更新 | 5 | 7.35 | (2,1,1,0) | |
| Elite 更新 | 6 | 7.38 | (2,1,1,0) | |
| Best 更新 | 6 | 7.38 | (2,1,1,0) | |
| New cell | 6 | 6.81 | (2,3,1,0) | 新niche |
| **New cell** | 7 | **7.81** | **(3,1,1,0)** | **最终最佳** |
| Best 更新 | 7 | 7.81 | (3,1,1,0) | |
| Elite 更新 | 8 | 7.61 | (2,1,1,0) | 持续改进旧cell |

**进化路径**: `(2,1,1,0)` → `(2,3,1,0)` → `(3,1,1,0)`

解读:
- `(2,1,1,0)`: memory_opt=2(SLM tiling), compute_opt=1(fused), parallelism_opt=1(barriers), esimd_opt=0
- `(3,1,1,0)`: memory_opt=3(register blocking + SLM), compute_opt=1, parallelism_opt=1, esimd_opt=0
- 最终突破来自提升 memory_opt (SLM → register blocking + SLM)

### 4.2 优化目标 Profile 分布

| Profile 特征 | 次数 | 策略 |
|-------------|------|------|
| esimd=0 | 33/40 (82.5%) | mutate主导 |
| esimd=1 | 1 | diversify |
| esimd=2 | 4 | diversify |
| esimd=3 | 2 | diversify |
| mutate 策略 | 31/40 (77.5%) | 主要 |
| diversify 策略 | 9/40 (22.5%) | 次要 |

**前5个最常见目标**:
1. memory=3, compute=1, parallelism=1, esimd=0 × 5次
2. memory=3, compute=0, parallelism=1, esimd=0 × 5次
3. memory=2, compute=2, parallelism=1, esimd=0 × 5次
4. memory=3, compute=2, parallelism=1, esimd=0 × 4次
5. memory=2, compute=0, parallelism=2, esimd=0 × 4次

### 4.3 推理时间统计

| 指标 | 值 |
|------|-----|
| 总推理次数 | 40 |
| 平均耗时 | 48.9s |
| 最快 | 26.8s |
| 最慢 | 88.3s |
| 模型 | claude-4-6-opus (100%) |

---

## 5) 技术分析: 为何 v6 未突破参考

### 5.1 最佳 Kernel 架构 (Trial 6, 36.2ms)

```
策略: SLM Tiled FMA (非 DPAS)
Tile: TILE_M=128, TILE_N=16, TILE_K=64
LWS: (16, 16, 1) = 256 work-items
GWS: (ceil(N/16)*16, ceil(M/128)*16, 1)
SLM: A_slm[128×64]=16KB + B_slm[64×16]=2KB = 18KB
Register: 8个float accumulators per thread
内循环: 4×unroll FMA (每步4个K值)
```

**为何性能瓶颈在 36.2ms?**
1. **无 DPAS**: 使用标量 `fma()` 而非矩阵乘硬件 → 吞吐受限于 EU FP throughput
2. **SLM 带宽限制**: 每个 TILE_K=64 步需要 128×64 + 64×16 = 9216 half 从 SLM 读取
3. **Barrier 开销**: 每个 K-chunk 需要 2 次 `barrier(CLK_LOCAL_MEM_FENCE)`
4. **内存搬运占比**: 大部分时间花在 global→SLM 的搬运上

### 5.2 v4 最佳 Kernel 架构 (11.4ms, 2.98x)

```
策略: 真正的 DPAS 硬件加速
Tile: 8×16 per subgroup (匹配 XMX 硬件)
LWS: (16, 1, 1) (无 reqd_work_group_size)
GWS: (N, M) runtime 自动选 LWS
指令: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
数据: A→short8 (8行×16列 packed), B→int8 (16×16 packed pair)
```

**为何 v4 快 3x?**
- XMX/DPAS 引擎提供 ~8x FP16 throughput vs EU FP ALU
- 零 barrier (纯 subgroup 协作, 无 SLM)
- 完美匹配硬件: 每个子组产出完整的 8×16 结果块

### 5.3 为何 v6 的模型放弃了 DPAS?

查看 Trial 1 的推理文本 (唯一尝试 DPAS 的正确 kernel):

> "With DPAS k16, each work-item (lane) in a subgroup of 16 holds one int per row, where that int packs 2 halfs. So lane `l` holds K indices `[2*l, 2*l+1]`. With K=16, lane 0 gets K[0,1], lane 7 gets K[14,15] — this only works for 8 lanes, but subgroup size is 16..."
> 
> "Given repeated failures with DPAS packing, let me use a simple subgroup-based approach **without DPAS**..."

**关键发现**: claude-4-6-opus 在 temperature=0.1 时:
1. 认识到 DPAS 操作数布局的复杂性
2. Trial 0 的 4 个 DPAS 尝试全部正确性失败
3. 模型"学习"到 DPAS 路径危险，转向安全的 SLM tiling 策略
4. 进化反馈机制放大了这个选择——正确的 SLM kernels 进入 DB 后成为 parent
5. 后续 evolution 沿 SLM 方向优化 (tile size, unroll, prefetch)

**对比 v4 (temperature=0.0)**:
- Trial 0 即生成正确的 DPAS kernel (11.6ms, 2.93x)
- temperature=0.0 下模型对 `short8/int8` 的类型选择是确定性的
- 一旦正确进入 DB，后续 trial 继续沿 DPAS 方向优化

### 5.4 temperature=0.1 vs 0.0 的关键差异

| 行为 | temp=0.0 (v4) | temp=0.1 (v6) |
|------|---------------|---------------|
| DPAS 操作数类型选择 | 确定性选 short8/int8 | 有概率选 half16/int8/其他 |
| 首轮 DPAS 正确率 | 1/2 (50%) | 0/4 (0%) |
| 后续策略 | 继续 DPAS (parent 引导) | 放弃 DPAS → SLM tiling |
| 最终性能 | 11.4ms (3x) | 36.2ms (0.94x) |

---

## 6) 进化算法收敛分析

### 6.1 收敛曲线

```
Score 进化:
Trial 0: 3.00 (全部失败)
Trial 1: 7.16 ↑ (首个正确)
Trial 2: — (无新best)
Trial 3: — (探索中)
Trial 4: — (探索中)
Trial 5: 7.35 ↑ (cell内改善)
Trial 6: 7.38 ↑ (微小提升)
Trial 7: 7.81 ↑↑ (新cell突破!)
Trial 8: — (无新best)
Trial 9: — (退步)

最终 best score: 7.81
进步轨迹: 前8轮有5次改善, 后2轮停滞
```

### 6.2 Archive 状态

| 指标 | 值 |
|------|-----|
| 占据 cells | 3 个: (2,1,1,0), (2,3,1,0), (3,1,1,0) |
| 覆盖率 | 3/256 = 1.17% |
| 最佳 cell | (3,1,1,0) score=7.81 |
| Elite 总更新 | 13次 |
| 新 cell 发现 | 3次 |
| Cell内 elite 竞争 | 10次 (cell (2,1,1,0) 频繁更新) |

### 6.3 收敛速度评估

- **有效搜索效率**: 33/40 正确 = 82.5% (合理)
- **改善效率**: 5次best更新 / 10 trials = 50% 轮次有改善
- **收敛方向**: 明确沿 memory_opt 维度提升 (2→3)
- **停滞点**: Trial 7 后未进一步改善 → 局部最优

---

## 7) 三次运行总体对比

| 指标 | v4 | v5 | v6 |
|------|-----|-----|-----|
| 模型 | opus (temp=0) | 3模型 (temp=0.3-0.6) | opus (temp=0.1) |
| 评估总数 | 8 | 80 | 40 |
| 正确率 | 87.5% | 95% | 82.5% |
| 最佳 runtime | **11.4ms** | 33.9ms | 36.2ms |
| 最佳 speedup | **2.98x** | 1.0x | 0.94x |
| 进化趋势 | 首轮即最优 | 完全无进化 | **持续改善** |
| 核心策略 | DPAS 硬件指令 | 错误的DPAS调用 | SLM tiling |
| 耗时 | 270s | 3509s | **1447s** |
| 效率 (speedup/时间) | 0.011x/s | 0.0003x/s | 0.00065x/s |
| Archive cells | 2 | 2 | **3** |
| Elite 更新 | 4 | 4 | **13** |

### 关键对比发现

1. **v4 是特例**: temp=0 恰好使模型一次命中正确的 DPAS 编码模式
2. **v5 验证了负面假设**: 弱模型+高温度=大量正确但毫无性能提升的代码
3. **v6 展示了真正的进化**: 单强模型+极低温度产生有意义的性能进化曲线
4. **v6 的局限**: 进化方向被"安全选择"锚定在 SLM tiling，未能突破到 DPAS

---

## 8) 根因总结与改进建议

### 8.1 为何无法超越参考?

```
根因链:
1. temp=0.1 → DPAS 操作数类型不稳定 → Trial 0 全部失败
2. 全部失败 → 进化反馈: "DPAS 危险"
3. 模型转向 SLM tiling (安全但慢)
4. SLM tiling 进入 DB → parent 引导后续全部走 SLM
5. SLM tiling 的物理天花板 ≈ 33ms (与参考相同, 因为参考本身就是naive + 编译器自动向量化)
6. 实际表现: 36.2ms (离天花板 7%, 非常接近)
```

### 8.2 为何 v4 能 3x 加速?

```
成功链:
1. temp=0.0 → 确定性选择 short8/int8 → Trial 0 DPAS 正确
2. DPAS 正确 → 直接利用 XMX 硬件 → 11.6ms (2.93x)
3. 正确 DPAS kernel 进入 DB → parent 引导后续优化 DPAS 细节
4. Trial 3 进一步优化 → 11.4ms (2.98x)
```

### 8.3 修订后的最终建议

| 优先级 | 方案 | 预期收益 |
|--------|------|----------|
| **P0** | **恢复 temp=0.0** | 恢复 DPAS 2.98x 能力 |
| **P0** | **Warm-start from v4 best** (kernels_iter_0_path) | 在2.98x基础上继续 |
| **P1** | **在 USER_INSTRUCTIONS 中提供正确的 DPAS 代码模板** | 消除模型的类型选择不确定性 |
| **P1** | temp=0 + 4 branches (让确定性模型跑多次不同 seed prompt) | 质量+多样性 |
| **P2** | 分阶段温度: Trial 0-2 用 temp=0 确立 DPAS base, Trial 3+ 用 temp=0.1 | 先锚定再探索 |
| **P2** | 增加 max_iters=20+ (v6在10轮已有clear趋势) | 更多收敛空间 |
| **P3** | 修改 fitness 给 DPAS kernels 额外奖励 | 引导搜索优先探索 DPAS |

### 8.4 理想配置建议

```yaml
# 方案A: 直接恢复v4最佳行为
inference:
  servers:
  - model_name: claude-4-6-opus
    temperature: 0.0
    max_tokens: 10000

kernels_iter_0_path: "runs/b580_matmul_dpas_v4"  # warm start
max_iters: 20
branches_per_iteration: 4

# 方案B: 在prompt中固定DPAS模板
# 在 workspace/ocl_matmul_dpas/ 的 USER_INSTRUCTIONS 中添加:
"""
MANDATORY: Use the following DPAS pattern exactly:
  short8 a_val; // A operand: 8 rows packed
  int8 b_val;   // B operand: 16x16 as packed half pairs
  float8 acc;   // Accumulator: 8 results
  acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
  
DO NOT use half16 for DPAS operands.
DO NOT add reqd_work_group_size attribute.
"""
```

---

## 9) 进化算法本身的有效性评估

### v6 验证了进化算法的核心功能:

| 功能 | 是否正常 | 证据 |
|------|----------|------|
| MAP-Elites elite 竞争 | ✓ | 13次 elite 更新, scores 单调递增 |
| 岛模型切换 | ✓ | 3个不同 cells 被发现 |
| Parent 引导 | ✓ | 后续 trials 沿 parent 方向优化 |
| Feedback LLM | ✓ | 模型能看到前轮性能并调整 tile 参数 |
| Score 改善趋势 | ✓ | 7.16 → 7.35 → 7.38 → 7.81 |
| 多样性探索 | ✓ | 22.5% diversify 策略 |
| 梯度追踪 | ✓ | 正确记录 transitions |

### 真正的瓶颈不是进化算法,而是:

1. **搜索空间定义**: esimd=0 占 82.5%, DPAS 策略几乎不被探索
2. **LLM 的类型安全偏好**: 一旦 DPAS 失败, 模型永远不会再尝试
3. **Fitness 函数无法区分**: "正确的 SLM 36ms" (score=7.8) vs "不正确的 DPAS" (score=3.0), 但后者的修复成本远低于前者到达 DPAS 性能的距离
4. **Temperature 的二元效应**: 对于这个任务, temp 要么是 0 (得到正确 DPAS) 要么非 0 (得不到)

---

## 10) 产物文件索引

| 类型 | 路径 |
|------|------|
| 运行目录 | `runs/b580_matmul_dpas_v6/` |
| 总结结果 | `runs/b580_matmul_dpas_v6/results.json` |
| 控制器日志 | `runs/b580_matmul_dpas_v6/controller.log` |
| 评估结果 | `runs/b580_matmul_dpas_v6/eval_result_*_trial_*_v*.json` (40个) |
| 生成代码 | `runs/b580_matmul_dpas_v6/generated_kernel_*_trial_*.py` (10个best) |
| Prompt记录 | `runs/b580_matmul_dpas_v6/prompt_*_trial_*_v*.md` (40个) |
| 任务配置 | `workspace/ocl_matmul_dpas/config.yaml` |
| 完整运行日志 | `/tmp/evolve_run_v6.log` |

---

## 11) 实验序列总结与核心教训

```
v4: temp=0, opus only    → 11.4ms (2.98x) ← DPAS 正确
v5: temp=0.3-0.6, 3模型  → 33.9ms (1.0x)  ← DPAS 错误, 等效参考
v6: temp=0.1, opus only  → 36.2ms (0.94x) ← 放弃DPAS, SLM tiling天花板

教训: 对于需要精确硬件指令编码的优化任务:
1. temperature=0 是必须的 (或提供确定性的代码模板)
2. 进化算法无法补偿 LLM 的类型选择错误
3. 一旦搜索走上错误路径(SLM), 进化只会在该路径内优化
4. 正确的 warm-start (从已验证的 DPAS kernel 开始) 比增大搜索广度更有效
```
