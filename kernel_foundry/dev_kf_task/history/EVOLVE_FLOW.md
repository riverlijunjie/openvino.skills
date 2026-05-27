# EVOLVE Mode Flow Analysis (`evolve_mode=true`)

## 1) 执行信息与结论

- 脚本：`/mnt/river/kernel_foundry/kernelfoundry.internal/runme_matmul.sh`
- 目标任务：`/mnt/river/kernel_foundry/workspace/ocl_matmul_dpas`
- 本次实跑命令（成功）在 `kernel_intel` 环境下执行：
  - `cd /mnt/river/kernel_foundry/kernelfoundry.internal && conda activate kernel_intel && source runme_matmul.sh`
- 结果：完整跑完 `max_iters=4`，每轮 `branches_per_iteration=2`，共评估 8 个候选 kernel。
- 总耗时（日志）：`369.259s`
- 参考基线性能（任务 benchmark）：`ref_speed = 33.9 ms`
- 最好候选出现在 Trial 2（v1）：`53.6 ms`（speedup=0.632，相对参考仍慢于 1.0x）。

> 注：直接在 base 环境执行会报 `ModuleNotFoundError: autoroot`，需使用项目依赖环境（如 `kernel_intel`）。

---

## 2) 关键输入配置（本次运行）

来自 `runme_matmul.sh` + task config 合并后：

- `evolve_mode=true`（核心开关）
- `max_iters=4`
- `branches_per_iteration=2`
- `use_queue=false`（本地执行，不走 Celery 分布式）
- `validate=true`
- `test_reference=true`
- `use_gradient_tracking=true`
- `use_optimization_aware_prompting=true`
- `exploration_strategy=mutate`

运行产物目录：
- `/mnt/river/kernel_foundry/kernelfoundry.internal/runs/b580_matmul_dpas_v3`

核心产物包括：
- `controller.log`
- `results.json`
- `prompt_level_..._trial_*_v*.md`
- `generated_kernel_level_..._trial_*.py`
- `stdout_level_..._trial_*_v*.txt`
- `eval_result_level_..._trial_*_v*.json`

---

## 3) 代码级执行链路（从脚本到进化）

## 3.1 入口：`scripts/run_custom_task.py`

关键函数：`main(config)`

1. 读取 Hydra 配置与命令行 override，初始化数据库与日志。
2. `CustomTask.create(custom_task_path)` 解析任务目录，提取 block：`REFERENCE / EVOLVE / USER_INSTRUCTIONS`。
3. 合并配置（优先级：run.yaml < task config < metadata overrides < cmdline overrides）。
4. 若 `validate=true`：先调用 `validate_custom_task(...)` 做一次“当前 EVOLVE 代码可构建可测试”的预校验。
5. 若 `max_iters > 0`：创建 `CustomTaskController` 并执行 `controller.run_single(custom_task)`。

## 3.2 Controller 初始化：`kernelgen/custom_task_controller.py`

- `TaskRunner.init(use_queue=config.use_queue)`：
  - `use_queue=false` 时 `TaskRunner.app=None`，走本地 build/test 路径。
- `self.program_database` 在 `evolve_mode=true` 时由基类 `Controller` 初始化。

## 3.3 主循环：`CustomTaskController._run_single(...)`

每轮 Trial 的主要步骤：

1. **DB setup（仅 evolve_mode）**
   - `self.program_database.setup(...)`
   - 本次是 `OptimizationAwareDatabase`（MAP-Elites 4x4x4x4，总 256 cells）。

2. **生成候选（Inference）**
   - 因 `evolve_mode=true`，进入 `evolve_prompt_and_inference(...)` 分支；
   - 使用 `ThreadPoolExecutor` 并行发起 `branches_per_iteration` 次（本次=2）。

3. **评估候选（Build+Test）**
   - `evaluate_batch(...)` 并行评估每个候选：
     - `CustomTaskEvaluator.run(...)`
     - 调 `TaskRunner.build_custom_task(...)`
     - 调 `TaskRunner.test_custom_task(...)`
   - 本次 `test_reference=true`，因此早期轮次会测试 reference + custom（后续可复用参考结果）。

4. **写库与选择 best**
   - 每个 child 通过 `program_database.add(...)` 进入 MAP-Elites；
   - 按 `select_best_solution(eval_results)` 选当轮 best；
   - 保存 `stdout_*_best.txt`、`generated_kernel_*`、`eval_result_*`。

5. **终止条件**
   - 本次 `stop_once_correct=false`，所以即使 correctness=true 也会跑满 4 轮。

---

## 4) `evolve_mode=true` 与 `false` 的关键差异

## 4.1 Prompt 生成路径差异

- `evolve_mode=false`：走 `standard_prompt_and_inference(...)`，主要基于上轮输出/反馈。
- `evolve_mode=true`：走 `evolve_prompt_and_inference(...)`，可从 program DB 采样 parent/inspirations，形成“进化式提示词”。

## 4.2 数据库/种群管理差异

`evolve_mode=true` 时启用 `OptimizationAwareDatabase`：

- Feature 维度：`memory_opt`, `compute_opt`, `parallelism_opt`, `esimd_opt`（每维 0..3）
- Grid：`4×4×4×4=256` cells
- 每个 cell 只保留 elite（按 `combined_score`）
- 支持 island model、迁移、underexplored 区域探索
- 支持 QD gradient tracking（本次启用）

## 4.3 优化感知提示增强

在 `Controller._apply_optimization_aware_prompting(...)`：

- 先由 `_get_target_optimization_profile(...)` 选目标 profile（mutate/diversify 等策略）。
- 再 `build_exploration_prompt(...)` 注入优化目标（日志里可见：
  `Applied optimization-aware target profile: memory=..., compute=..., parallelism=..., esimd=...`）。

---

## 5) 评估流水细节（build/test/runtime）

## 5.1 Evaluator 执行顺序

`CustomTaskEvaluator.run(task)` 关键流程：

1. 校验代码抽取结果（extract code）。
2. 按配置决定是否 build reference/custom。
3. 执行 pytest correctness/performance（每个 GPU arch）。
4. 计算 runtime、runtime_improvement、perf_score，组装 `EvalResult`。

## 5.2 本次运行中的实际行为

- Build 基本都在 `0.36~0.64s`。
- Correctness 测试常见 `~11s`，性能测试 `~12~16s`。
- Trial 1 的一个分支出现更短评测时间（可能受缓存/测试路径影响）。

---

## 6) 本次 4 轮 Trial 结果汇总

来自 `results.json` + `controller.log`：

- **Trial 0**: 选中 best runtime = **65.8 ms**, speedup=0.515
- **Trial 1**: best runtime = **172.0 ms**, speedup=0.197
- **Trial 2**: best runtime = **53.6 ms**, speedup=0.632（全程最佳）
- **Trial 3**: best runtime = **102.0 ms**, speedup=0.332

额外日志事件：
- Trial 0：`New best program ... score=6.5456`
- Trial 2：
  - `New elite in cell (2,1,1,0) ... score=6.8974 replaced ...`
  - `New best program ... score=6.8974 replaced ...`

说明：进化 DB 的“best”与 runtime 直接相关但不是简单等价，内部依据 `combined_score` / 评估分数组合进行 elite 竞争。

---

## 7) 运行中可见的真实系统行为

1. **GNAI 模型调用**
   - 日志显示每轮并行 2 次 inference，耗时通常 25~40s。
   - 存在 `InsecureRequestWarning`（TLS verify 关闭）。

2. **use_queue=false 的副作用**
   - 本地线程并发评测，输出较密集；
   - `TaskRunner` 在 local 模式用 `_local_test_lock` 串行化 GPU 测试，降低设备冲突风险。

3. **DB 状态更新 warning**
   - 日志中有 `Could not update job status: 'NoneType' object is not callable`、`Could not add item to database...`。
   - 这不阻断主流程（本次 run 仍完成），但提示数据库写入链路有可用性问题。

---

## 8) `evolve_mode=true` 的“详细逻辑”总结（可作为排障清单）

可以把它理解为 9 个阶段：

1. **Task 装载与 block 提取**（REFERENCE/EVOLVE/USER_INSTRUCTIONS）
2. **配置合并与覆盖**（命令行最高优先级）
3. **预校验 validate**（build+test 当前 EVOLVE）
4. **初始化进化数据库**（MAP-Elites + islands + gradient tracker）
5. **每轮并行生成分支**（基于 parent/inspirations + target profile）
6. **每分支评估**（build + correctness + benchmark）
7. **入库竞争 elite**（按 cell 与 score 更新）
8. **记录产物**（prompt/kernel/stdout/eval_result/results.json）
9. **更新 best 与迭代进度**（直到 max_iters 或 stop 条件）

---

## 9) 本次 run 对你的 matmul task 的直接结论

- `evolve_mode=true` 链路是正常工作的：
  - 能生成候选
  - 能并行评估
  - 能更新 elite/best
  - 能输出完整 run artifacts
- 但性能上“进化候选仍慢于 reference（33.9ms）”，说明本轮搜索空间里还没挖到更优 kernel；后续可针对 DPAS 数据布局/子组映射/访存模式增加更强约束提示。

---

## 10) 关键源码索引（便于二次深挖）

- 入口脚本：
  - `scripts/run_custom_task.py`
- 控制器：
  - `kernelgen/custom_task_controller.py`
  - `kernelgen/controller.py`
- 评估：
  - `kernelgen/custom_task_evaluator.py`
- 任务模型：
  - `kernelgen/custom_task.py`
- 进化数据库：
  - `kernelgen/evolve_database_optimization_aware.py`
- 任务执行器：
  - `kernelgen/tasks/task_runner.py`
- 本次 run 目录：
  - `runs/b580_matmul_dpas_v3/`
