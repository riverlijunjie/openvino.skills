# EVOLVE Mode Flow 深度分析 v2

## 1) 执行信息与本次运行结论

| 项目 | 值 |
|------|-----|
| 脚本 | `/mnt/river/kernel_foundry/kernelfoundry.internal/runme_matmul.sh` |
| 目标任务 | `/mnt/river/kernel_foundry/workspace/ocl_matmul_dpas` (OpenCL DPAS matmul) |
| 执行环境 | `conda activate kernel_intel` + `source /opt/intel/oneapi/2025.0/oneapi-vars.sh` |
| GPU | Intel Battlemage G21 (B580), device_id=0xe20b |
| LLM 模型 | `claude-4-6-opus` (via Intel GNAI gateway) |
| `max_iters` | 4 |
| `branches_per_iteration` | 2 |
| 总评估 kernel 数 | 8 |
| 总耗时 | **269.64 秒** |
| 参考基线 (reference) | 34.0 ms |
| 最佳候选 (Trial 3, v1) | **11.4 ms** (speedup = **2.98x**) |
| 最佳首轮 (Trial 0, v0) | 11.6 ms (speedup = 2.93x) |

> 本次运行首轮即产出 ~3x 加速的正确 kernel，后续 Trial 1 出现一次正确性失败，Trial 2-3 持续产出 ~3x 水平的变体但未突破首轮最优。说明当前进化模式在少量迭代下尚未有效收敛到更优解。

---

## 2) 关键配置参数

从 `runme_matmul.sh` + `configs/run.yaml` + task config 合并后的实际运行参数：

```yaml
evolve_mode: true
max_iters: 4
branches_per_iteration: 2
use_queue: false           # 本地执行，不走 Celery
validate: true
stop_once_correct: false
has_reference_build_step: false
test_timeout: 180
build_timeout: 120
language: OCL
gpu_arch: bmg

# 进化数据库 (MAP-Elites)
database:
  _target_: kernelgen.evolve_database_optimization_aware.OptimizationAwareDatabase
  num_islands: 4
  programs_per_island: 10
  migration_interval: 10
  migration_rate: 0.1
  population_size: 1000
  exploration_ratio: 0.2
  exploitation_ratio: 0.7
  num_inspirations: 2

# 梯度追踪
use_gradient_tracking: true
gradient_sampling_weight: 0.3
gradient_config:
  max_history: 10000
  fitness_weight: 0.4
  improvement_rate_weight: 0.4
  exploration_weight: 0.2

# 优化感知提示
use_optimization_aware_prompting: true
exploration_strategy: mutate
guidance_exploration_rate: 0.0
```

---

## 3) 进化算法核心架构

### 3.1 整体架构图

```
┌────────────────────────────────────────────────────────────────────────┐
│                    CustomTaskController._run_single()                    │
│                                                                          │
│  for trial in range(max_iters):                                         │
│    ┌──────────────────────────────────────────────────────────┐         │
│    │  1. SAMPLE: program_database.sample()                     │         │
│    │     → parent + inspirations (from MAP-Elites archive)     │         │
│    │                                                           │         │
│    │  2. PROMPT: evolve_prompt_and_inference() × N branches    │         │
│    │     → Optimization-aware target profile selection          │         │
│    │     → PromptConstructor builds full prompt                 │         │
│    │     → LLM inference (claude-4-6-opus)                     │         │
│    │                                                           │         │
│    │  3. EVALUATE: evaluate_batch() (parallel ThreadPool)      │         │
│    │     → Build kernel → Test correctness → Benchmark perf    │         │
│    │                                                           │         │
│    │  4. UPDATE: program_database.add()                        │         │
│    │     → Classify optimization coordinates                    │         │
│    │     → MAP-Elites elite replacement                        │         │
│    │     → Island counter & migration check                    │         │
│    │                                                           │         │
│    │  5. REPORT: fitness → prompt evolution (if enabled)       │         │
│    └──────────────────────────────────────────────────────────┘         │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 MAP-Elites 程序数据库（核心）

**文件**: `kernelgen/evolve_database_optimization_aware.py`

MAP-Elites 是一种 Quality-Diversity (QD) 算法，核心思想是在**行为空间**中维护一个精英存档(archive)：

#### 行为空间定义（4D 优化特征网格）

| 维度 | 范围 | 含义 |
|------|------|------|
| `memory_opt` | 0-3 | 内存层次利用级别 (naive → coalesced → SLM tiling → register blocking + async) |
| `compute_opt` | 0-3 | 算法效率级别 (multi-pass → fused → single-pass → tiled/blocked) |
| `parallelism_opt` | 0-3 | 并行粒度级别 (thread-only → work-group barriers → sub-group → hierarchical) |
| `esimd_opt` | 0-3 | ESIMD 使用级别 (standard SYCL → basic simd → optimized LSC → expert DPAS/XMX) |

- **网格大小**: 4×4×4×4 = 256 个行为 niche（cell）
- **坐标赋值**: 通过**静态代码模式匹配** (`OptimizationFeatureClassifier`) 确定，确定性、与执行无关
- **精英选择**: 每个 cell 只保留 fitness 最高的程序（`combined_score`）

#### Fitness 计算

```python
combined_score = perf_score + 3 * int(correctness and runtime_improvement > 0) * runtime_improvement
```

- `perf_score`: 0-5 的性能评分（0=语法错误, 1=编译失败, 2=运行时错误, 3=正确性失败, 4=性能未提升, 5=完全通过）
- `runtime_improvement`: 相对参考的加速比 (reference_runtime / kernel_runtime)
- 加权：正确且有加速时，加速比的权重是 perf_score 的 3 倍

#### 岛模型 (Island Model)

- **4 个子种群**（islands）独立进化
- 每处理 `programs_per_island=10` 个程序后切换到下一个岛
- 每 `migration_interval=10` 代执行一次迁移：环形拓扑，最优个体复制到下一个岛
- 防止早熟收敛，维护种群多样性

### 3.3 采样策略（Parent Selection）

每次迭代从 MAP-Elites 存档中选择 parent：

```python
rand_val = random.random()
if rand_val < 0.2:   # exploration_ratio
    → 从当前岛的全部种群中随机选择（多样性探索）
elif rand_val < 0.9:  # exploration + exploitation
    → 从 MAP-Elites 精英存档中按 fitness 比例采样（softmax 概率）
else:                 # random (remaining 0.1)
    → 完全随机选择（新颖性注入）
```

**Inspiration 采样**（交叉参考）：
- 优先从**不同优化 niche** 的精英中采样
- 倾向选择**更高优化级别**的程序作为灵感
- 补充其他岛的最优程序

### 3.4 优化目标 Profile 选择

`_get_target_optimization_profile()` 决定本轮进化的目标方向：

1. **有 parent 且 parent 正确时**：
   - 获取 parent 的优化坐标
   - 执行变异策略（`mutate`）：在当前坐标基础上随机偏移
   - 考虑 ESIMD 升级策略（如果启用）
2. **无 parent 或 parent 不正确时**：
   - 从 underexplored regions（空 cell 或低质量 cell）中随机选取目标
3. **Guidance exploration rate**（本次为 0.0）：
   - 如果 > 0，即使有 parent，也有概率切换到 underexplored 目标

### 3.5 QD 梯度追踪

**文件**: `kernelgen/qd_gradient.py`

受 CMA-ME / PGA-MAP-Elites / DQD 启发，记录 parent→child 演化转换信息：

- **TransitionRecord**: 不可变记录，包含 parent/child 坐标、fitness delta、outcome
- **GradientEstimator**: 从历史转换中估计"伪梯度"
  - `fitness_weight=0.4`: fitness 值梯度
  - `improvement_rate_weight=0.4`: 改进成功率梯度
  - `exploration_weight=0.2`: 向未探索区域的探索梯度
- **用途**:
  - 指导 parent 选择（梯度加权采样概率）
  - 提供 mutation hints（告诉 LLM 应该向哪个方向优化）
  - 识别高潜力转换路径

### 3.6 Prompt 构造流程

`evolve_prompt_and_inference()` 的完整流程：

```
1. sample_evolve_programs()
   → parent (with feedback from eval log)
   → inspirations (diverse elites from other niches)
   → top_program (island best)

2. _get_target_optimization_profile()
   → target = {memory_opt: X, compute_opt: Y, parallelism_opt: Z, esimd_opt: W}

3. PromptConstructor()
   → reference code + parent code + feedback + inspirations
   → target optimization profile as guidance

4. _apply_optimization_aware_prompting()
   → build_exploration_prompt(): 将目标 profile 嵌入 prompt
   → (可选) ESIMD taxonomy、Tensor Core taxonomy

5. LLM Inference (claude-4-6-opus)
   → 生成新 kernel 代码
```

### 3.7 评估流程

`evaluate_batch()` → `evaluate_single()` → `CustomTaskEvaluator.run()`:

1. **代码注入**: `custom_task.with_blocks({"EVOLVE": program.code})` — 将生成的代码替换到 EVOLVE 标记位置
2. **编译**: `TaskRunner.build_custom_task()` — 调用 icpx/OpenCL 编译器
3. **正确性测试**: pytest 执行 `test_correctness`
4. **性能测试**: pytest 执行 `test_benchmark` (num_perf_trials=100, warmup_min_iters=10)
5. **结果汇总**: 生成 `EvalResult` 包含 runtime, perf_score, runtime_improvement 等

---

## 4) 本次运行详细分析

### Trial-by-Trial 结果

| Trial | Version | Compiled | Correct | Runtime (ms) | Speedup | combined_score |
|-------|---------|----------|---------|--------------|---------|----------------|
| 0 | v0 | ✓ | ✓ | 11.6 | 2.93x | 13.79 |
| 0 | v1 | ✓ | ✓ | 33.9 | 1.00x | 8.00 |
| 1 | v0 | ✓ | ✗ | - | - | ~3.0 |
| 1 | v1 | ✓ | ✓ | 11.4 | 2.98x | 13.95 |
| 2 | v0 | ✓ | ✓ | 35.7 | 0.95x | 5.0 |
| 2 | v1 | ✓ | ✓ | 11.4 | 2.98x | 13.95 |
| 3 | v0 | ✓ | ✓ | 35.7 | 0.95x | 5.0 |
| 3 | v1 | ✓ | ✓ | 11.4 | 2.98x | 13.95 |

### 优化目标 Profile 变化

| Trial | Branch | Profile | Strategy |
|-------|--------|---------|----------|
| 0 | v0 | memory=0, compute=3, parallelism=2, esimd=1 | diversify |
| 0 | v1 | memory=1, compute=3, parallelism=0, esimd=2 | diversify |
| 1 | v0 | memory=2, compute=3, parallelism=3, esimd=0 | mutate |
| 1 | v1 | memory=2, compute=3, parallelism=3, esimd=0 | mutate |
| 2 | v0 | memory=2, compute=3, parallelism=3, esimd=0 | mutate |
| 2 | v1 | memory=2, compute=3, parallelism=3, esimd=0 | mutate |
| 3 | v0 | memory=2, compute=3, parallelism=3, esimd=0 | mutate |
| 3 | v1 | memory=0, compute=1, parallelism=2, esimd=2 | diversify |

### 观察到的问题

1. **收敛停滞**: Trial 0 即达到 ~3x 加速，后续 3 轮未突破
2. **目标 Profile 重复**: Trial 1-3 大量重复 `(2,3,3,0)` 的 mutate 策略，缺乏有效探索
3. **Feedback 质量不足**: parent 的 eval log 信息（仅含 runtime 数字）缺乏具体优化方向指导
4. **多样性不足**: 只有 4 轮 × 2 分支 = 8 个样本，MAP-Elites 256 个 cell 覆盖率极低

---

## 5) 进化算法收敛速度提升方案

### 5.1 短期优化（配置调参，无代码改动）

#### A. 增大搜索广度

```yaml
# 当前值 → 建议值
max_iters: 4 → 20-50       # 更多迭代轮次
branches_per_iteration: 2 → 4-8  # 每轮更多分支并行
```

**原因**: 当前仅 8 个样本，MAP-Elites 的优势在大样本下才能体现。

#### B. 提高探索率和目标多样性

```yaml
exploration_ratio: 0.2 → 0.4      # 增加探索占比
exploitation_ratio: 0.7 → 0.5     # 减少过度利用
guidance_exploration_rate: 0.0 → 0.3  # 即使有 parent 也有 30% 概率探索 underexplored
```

**原因**: 当前 70% 利用率导致从稀疏的精英存档中反复采样相同个体。

#### C. 使用 Feedback LLM

```yaml
use_feedback_llm: true
feedback_llm_config:
  model_name: "gpt-4.1"
  temperature: 0.3  # 略有随机性
```

**原因**: 将原始 eval log 转化为结构化优化建议，帮助 LLM 理解性能瓶颈。

#### D. 调整 LLM 温度

```yaml
inference:
  temperature: 0.0 → 0.6-0.8  # 增加随机性
```

**原因**: temperature=0 导致同一 prompt 产出高度相似代码。

---

### 5.2 中期优化（需要代码改动）

#### E. Warm-Start 机制改进

**当前问题**: 进化数据库每次运行从空白开始。

**方案**: 实现 `resume_from_database=true` 或从上次运行的 checkpoint 恢复

```python
# 在 config 中添加
resume_from_archive: "runs/b580_matmul_dpas_v4/checkpoint.json"
# 或
resume_from_database: true
```

**效果**: 避免重复探索已知区域，从历史最优解继续进化。

#### F. 自适应目标 Profile 选择

**当前问题**: `_get_target_optimization_profile()` 中 mutate 策略重复产生相同目标 `(2,3,3,0)`。

**方案**: 引入 **Tabu List**（禁忌搜索）+ 衰减机制

```python
def _get_target_optimization_profile(self, parent):
    # 记录最近 N 个 target profiles
    if target in self._recent_targets:
        # 强制切换到不同方向
        target = self._sample_away_from_recent()
    self._recent_targets.append(target)
    
    # 或使用 UCB1 风格的探索奖励
    visit_count = self._target_visit_counts[target_key]
    ucb_bonus = sqrt(log(total_visits) / (visit_count + 1))
    exploration_score = base_score + C * ucb_bonus
```

#### G. 多阶段进化策略

**方案**: 根据进化进程动态调整策略

```python
Phase 1 (iter 0-10):  # 广泛探索
    exploration_ratio = 0.6
    temperature = 0.9
    num_inspirations = 4
    
Phase 2 (iter 10-30):  # 聚焦改进
    exploration_ratio = 0.3
    temperature = 0.5
    use_feedback_llm = True
    
Phase 3 (iter 30+):  # 精细调优
    exploitation_ratio = 0.8
    temperature = 0.2
    use_gradient_tracking = True (gradient_sampling_weight = 0.6)
```

#### H. 更细粒度的 Fitness Shaping

**当前问题**: `combined_score` 只区分"正确但慢"和"正确且快"，缺乏中间梯度。

**方案**: 引入多目标分层 fitness

```python
def compute_shaped_fitness(eval_result):
    # 层级奖励，确保编译 < 正确性 < 性能 之间有清晰梯度
    if not eval_result.compiled:
        return 0.1 * syntax_similarity_score  # 即使编译失败也有微弱信号
    if not eval_result.correctness:
        return 2.0 + 0.5 * partial_correctness  # 部分正确给部分分数
    # 正确后，指数奖励加速比
    speedup = eval_result.runtime_improvement
    return 5.0 + 10.0 * (1 - exp(-0.5 * speedup))  # saturating reward
```

#### I. 交叉(Crossover)操作

**当前问题**: 每个 child 完全由 LLM 独立生成，没有真正的代码级交叉。

**方案**: 在 prompt 中显式要求 LLM 合并两个 parent 的优化策略

```python
def crossover_prompt(parent_a, parent_b, target_profile):
    return f"""
    You are given two optimized kernels:
    
    Kernel A (speedup={parent_a.speedup}x, optimization: {parent_a.profile}):
    {parent_a.code}
    
    Kernel B (speedup={parent_b.speedup}x, optimization: {parent_b.profile}):
    {parent_b.code}
    
    Combine the best optimization strategies from both kernels into a single
    kernel targeting optimization profile: {target_profile}
    """
```

---

### 5.3 长期优化（架构级改进）

#### J. 分层进化 (Hierarchical Evolution)

将大 kernel 分解为独立优化模块：

```
Level 1: Memory access pattern (tiling, coalescing)
Level 2: Compute pattern (algorithm selection, fusion)
Level 3: Parallelism pattern (work distribution, synchronization)
Level 4: HW-specific optimizations (DPAS, LSC, prefetch)
```

每层独立进化然后组合，降低搜索空间维度。

#### K. Surrogate Model 加速评估

**当前瓶颈**: 每次评估需 ~15-75 秒（编译+测试）

**方案**: 训练一个轻量 surrogate model 预测 kernel 性能

```python
# 快速筛选（<1ms）
predicted_score = surrogate.predict(kernel_code_features)
if predicted_score < threshold:
    skip_evaluation()  # 跳过明显低质量的 kernel
else:
    actual_score = full_evaluation()  # 只评估有潜力的
    surrogate.update(kernel_code_features, actual_score)
```

**预期收益**: 每轮可评估 10-50 个 kernel 但只实际运行 2-5 个，等效将搜索效率提升 5-10x。

#### L. 多模型协同进化

使用不同 LLM 产生多样性：

```yaml
inference:
  _target_: kernelgen.inference_server.LLMEnsemble
  servers:
    - model_name: "claude-4-6-opus"        # 强推理能力
    - model_name: "gpt-5.4"                # 不同编码风格
    - model_name: "claude-4-5-sonnet"      # 快速但有创意
  weights: "round_robin"  # 轮流使用
```

#### M. Profiler-Guided Evolution

**方案**: 将 unitrace/VTune 的 profiler 数据转化为优化指导

```python
if eval_result.profiler_data:
    bottleneck = identify_bottleneck(profiler_data)
    # e.g., "Memory bound: 85% stalls on L3 cache misses"
    # → 指导 LLM 关注 memory tiling / prefetch
    optimization_hint = profiler_to_hint(bottleneck)
    prompt += f"\nProfiler analysis: {optimization_hint}"
```

当前已有 `profile_custom_model` 配置但默认关闭。

---

## 6) 推荐的优先级排序

| 优先级 | 方案 | 预期收益 | 实施难度 |
|--------|------|----------|----------|
| P0 | D. 增加 LLM temperature | 立即增加代码多样性 | 配置改动 |
| P0 | A. 增大 max_iters/branches | 更多样本覆盖更多 cells | 配置改动 |
| P1 | B. 调高 exploration_ratio + guidance_rate | 避免重复目标 | 配置改动 |
| P1 | E. Warm-start checkpoint | 跨 run 累积知识 | 小代码改动 |
| P2 | F. Tabu list for target profiles | 消除重复目标 | 中等改动 |
| P2 | C. 启用 Feedback LLM | 结构化优化建议 | 配置改动 |
| P2 | G. 多阶段策略 | 平衡探索/利用 | 中等改动 |
| P3 | H. Fitness shaping | 提供更细梯度 | 中等改动 |
| P3 | I. 显式 Crossover prompt | 组合多个好解 | 小改动 |
| P3 | K. Surrogate model | 10x 评估加速 | 大改动 |
| P4 | J. 分层进化 | 降低搜索空间 | 架构重构 |
| P4 | L. 多模型集成 | 代码风格多样性 | 中等改动 |
| P4 | M. Profiler-guided | 精确优化指导 | 大改动 |

---

## 7) 核心代码入口索引

| 组件 | 文件路径 | 关键类/函数 |
|------|----------|-------------|
| 入口脚本 | `scripts/run_custom_task.py` | `main()` |
| 控制器 | `kernelgen/custom_task_controller.py` | `CustomTaskController._run_single()` |
| 基类 | `kernelgen/controller.py` | `Controller.evolve_prompt_and_inference()` |
| MAP-Elites DB | `kernelgen/evolve_database_optimization_aware.py` | `OptimizationAwareDatabase` |
| 特征分类器 | 同上 | `OptimizationFeatureClassifier.classify_from_code()` |
| QD 梯度 | `kernelgen/qd_gradient.py` | `TransitionTracker`, `GradientEstimator` |
| Prompt 构造 | `kernelgen/prompts/prompt_constructor.py` | `PromptConstructor.__call__()` |
| Prompt 进化 | `kernelgen/prompts/prompt_evolution_integration.py` | `PromptEvolutionMixin` |
| Meta-Prompting | `kernelgen/prompts/meta_prompting.py` | `MetaPromptingManager` |
| 优化感知提示 | `kernelgen/prompts/optimization_aware.py` | `build_exploration_prompt()` |
| 评估器 | `kernelgen/custom_task_evaluator.py` | `CustomTaskEvaluator.run()` |
| 适应度计算 | `kernelgen/schemas.py:152` | `Program.add_eval_results()` |
| 最优选择 | `kernelgen/utils/score.py` | `select_best_solution()` |
| 模式匹配规则 | `kernelgen/utils/map_elites_patterns.py` | `*_OPT_PATTERNS` dicts |
| 数据库配置 | `configs/database/evolve_db_optimization_aware.yaml` | 全局 DB 参数 |

---

## 8) 算法流程伪代码总结

```python
def evolve_kernel(custom_task, config):
    db = OptimizationAwareDatabase(config)  # 4D MAP-Elites grid
    db.setup()  # 可选: 从 checkpoint 或数据库恢复
    
    program0 = create_initial_program(custom_task)  # 从 EVOLVE 标记提取初始代码
    
    for trial in range(max_iters):
        children = []
        
        # 并行生成 N 个分支
        for branch in range(branches_per_iteration):
            # 1. 采样
            if db.is_empty():
                parent = program0
                sampled = {"last_program": parent}
            else:
                parent, inspirations = db.sample()  # exploration/exploitation/random
                sampled = {"last_program": parent, "top_program": island_best, "inspirations": inspirations}
            
            # 2. 确定优化方向
            target_profile = select_optimization_target(parent, db)  # mutate/diversify
            
            # 3. 构造 prompt
            prompt = build_prompt(reference, sampled, target_profile, feedback)
            
            # 4. LLM 推理
            kernel_code = llm(prompt)
            child = parse_kernel(kernel_code, parent)
            children.append(child)
        
        # 5. 批量评估
        eval_results = evaluate_batch(children)  # compile + test + benchmark
        
        # 6. 更新数据库
        for child, result in zip(children, eval_results):
            child.add_eval_results(result)
            coords = classify_optimization_level(child.code)  # 静态模式匹配
            db.add(child)  # MAP-Elites: 只保留 cell 中 fitness 最高的
            
            # QD 梯度追踪
            db.record_transition(parent → child, coords, fitness_delta)
        
        # 7. 岛切换 & 迁移
        db.increase_island_counter_and_switch()
        if db.should_migrate():
            db.migrate_programs()  # 环形拓扑, top performers → next island
    
    return db.get_best_program()
```

---

## 9) 关键设计决策与权衡

| 决策 | 当前选择 | 替代方案 | 权衡 |
|------|----------|----------|------|
| 坐标赋值 | 静态代码模式匹配 | 运行时 profiler 数据 | 确定性 vs 准确性 |
| Fitness 函数 | `perf_score + 3×speedup` | 多目标 Pareto | 简单 vs 精细 |
| 变异操作 | 完全由 LLM 重写 | diff 格式增量修改 | 创造性 vs 稳定性 |
| 种群管理 | Island model (4 islands) | 单种群 MAP-Elites | 多样性 vs 收敛速度 |
| 采样策略 | 20% explore / 70% exploit / 10% random | 自适应比例 | 简单 vs 最优 |
| 梯度追踪 | 离散转换统计 | 连续梯度估计 | 稳健性 vs 精确性 |

---

## 10) 实验建议：快速验证最佳方案

### 实验 1：基础配置对比

```bash
# Baseline (当前)
python scripts/run_custom_task.py ... max_iters=20 branches_per_iteration=2

# 方案 A+B+D: 增大搜索 + 提高探索 + 高温度
python scripts/run_custom_task.py ... \
  max_iters=20 branches_per_iteration=4 \
  exploration_ratio=0.4 exploitation_ratio=0.5 \
  guidance_exploration_rate=0.3 \
  inference.servers.0.temperature=0.7
```

### 实验 2：Warm-Start 累积

```bash
# Run 1: 初始探索
python scripts/run_custom_task.py ... max_iters=20 job_name=matmul_explore

# Run 2: 从 Run 1 继续
python scripts/run_custom_task.py ... max_iters=20 \
  kernels_iter_0_path=best-matmul_explore \
  resume_from_database=true
```

### 度量指标

- **收敛速度**: 达到目标 speedup 所需的 iteration 数
- **Archive Coverage**: `occupied_cells / 256`
- **Best Fitness 曲线**: `max(combined_score)` vs iteration
- **多样性**: 占据的 unique optimization levels 数量
