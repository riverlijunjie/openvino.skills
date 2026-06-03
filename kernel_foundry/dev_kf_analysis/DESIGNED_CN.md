# Kernel Optimization Agent 框架设计

> 日期：2026-06-01  
> 版本：v0.2  
> 范围：设计一个以 Kernel Foundry 为核心执行与数据底座的 kernel optimization agent；它整合成熟的 agent workflow 优点，并针对 Kernel Foundry 当前在 agent orchestration、动态任务生成、验证强化、E2E 集成与 evidence 管理上的局限进行补齐。  
> 状态：设计提案。

## 1. 愿景

Kernel Optimization Agent 将 Kernel Foundry 从一个基于配置的批量生成 harness，转变为面向用户、基于证据驱动、闭环优化的系统。

用户应该可以这样说：

> “优化 LNL/BMG 上的 OpenCL gated delta net kernel。优先优化 latency，数值误差保持在 1e-3 以内，并用 profiling 证据报告最佳 candidate。”

随后 agent 应该能够：

1. 理解自然语言需求，
2. 构建或定位 Kernel Foundry 任务，
3. profile baseline 或 workload，
4. 选择优化目标，
5. 启动一个 Kernel Foundry `CustomTask` 内层 evolution loop，生成并筛选多个 kernel variants，
6. 收集每次 KF 内部迭代的 correctness/perf/profile 结果，
7. 在内层循环结束后，将最佳 kernel 或少量 top-K / Pareto / diverse branch best kernels materialize 成外层 `Candidate`，
8. 仅在外层对少量 finalist candidates 做 runtime integration 和 E2E benchmark，
9. 如果 E2E 不达标，则基于外层反馈再次启动新一轮 KF `CustomTask`，
10. 生成包含全部证据的可复现报告。

核心设计原则是：

> 让 LLM 负责推理，但让确定性 tools 决定事实。

### 1.1 定位校准

本设计不是引入外部流程框架来替代 Kernel Foundry。主次关系应明确为：

- **Kernel Foundry 是核心**：负责 task abstraction、kernel generation/evolution、build/test/profile、RAG、database、template、benchmark 与未来 runtime integration。
- **Workflow discipline 是关键方法论**：提供 task contract、plan-first、candidate lineage、evidence records、promotion gates 和 human-readable artifacts 等 agent 工作流原则。
- **Kernel Optimization Agent 是 KF-first 的补强层**：围绕 Kernel Foundry 现有能力增加自然语言入口、agent orchestrator、动态测试/任务更新、anti-gaming validation、E2E promotion 与结构化 evidence，而不是另起一套独立优化框架。

因此，Kernel Foundry 在本设计中的角色是“执行环境、事实来源和持久化系统”。任何 correctness、performance、profile、promotion 的最终判断都必须来自 Kernel Foundry tool outputs，而不能来自 agent 的主观判断。

### 1.2 双层循环设计原则

必须避免把一个 KF `CustomTask` 误解为一次单独 kernel 迭代。真实设计应采用双层循环：

- **内层循环：KF CustomTask evolution loop**  
  由 Kernel Foundry 负责在一个 custom task 内部进行多轮 kernel 生成/变异、build、correctness tests、performance tests、可选 micro/profile 反馈和内部 best kernel 选择。
- **外层循环：Agent E2E optimization loop**  
  由 agent 在每个 KF custom task 内层循环结束后，把 KF 选出的最佳 kernel materialize 为外层 `Candidate`，再执行 integration、E2E benchmark、promotion decision 和下一轮 task planning。

关键约束：**E2E 测试不进入 KF custom task 内层循环**。E2E 代价太高，应只在外层对少量 finalist candidates 执行。内层只做 micro/correctness/perf tests，用于快速筛选 kernel；外层才验证真实 workload impact。

## 2. 设计目标

### 2.1 功能目标

- 由自然语言触发优化。
- 闭环 kernel 生成、测试、profiling、benchmarking 和 refinement。
- 支持多种 kernel 语言：SYCL、OpenCL、CUDA、Triton，并优先支持 Intel SYCL/OpenCL。
- 复用 Kernel Foundry task templates、evaluator、profiler feedback、RAG 和 database。
- 引入工程化 workflow discipline：task contract、planning、candidate tracking、evidence records 和 promotion discipline。
- 为 runtime/workload-level validation 提供 E2E evaluation interfaces。
- 支持 tools、skills 和 seed databases 资产库。
- 明确补齐 Kernel Foundry 当前作为 batch/search harness 时缺少的 agentic orchestration、candidate lifecycle、动态测试更新、E2E workload integration 和 report/evidence schema。

### 2.2 质量目标

- 正确性优先于性能。
- 性能声明必须有证据支持。
- 具备 robust anti-reward-hacking validation。
- 相比盲目批量搜索，降低 token 使用量。
- 通过 staged/tiered testing 降低评估成本。
- 后端、搜索策略和硬件知识可扩展。
- 报告可复现，并保留 candidate lineage。
- 新增 agent 能力必须以兼容当前 Kernel Foundry 为前提，优先通过 adapter、wrapper、feature flag 和可选 tool 扩展实现，避免破坏现有 batch experiments。
- 优化时间成本：减少无效 trial、减少 full profile 次数、复用缓存结果，并对搜索、验证、profiling 设置明确预算。

### 2.3 第一版非目标

- 初期不训练新的 kernel LLM。
- 第一里程碑不支持所有 runtime backend。
- 不替换 Kernel Foundry 现有 batch experiments；而是包装并复用它们。
- 不以外部流程框架重写 Kernel Foundry；KF 仍是核心执行与数据平面。
- 不将 MAP-Elites 作为强制主循环。

### 2.4 KF 局限到设计补齐的映射

本设计的出发点是补齐 Kernel Foundry 当前从“批量生成/搜索平台”走向“面向用户的 kernel optimization agent”时的缺口。对应关系如下：

| Kernel Foundry 当前局限 | 设计补齐方式 | 主要章节 |
|---|---|---|
| 配置和任务入口偏工程化，用户难以直接从自然语言发起优化 | `PARSE_REQUIREMENT` 将自然语言转成 `TaskContract`，再加载或生成 KF task | 5.1、5.2 |
| Batch generation/search 强，显式 agent planning 弱 | 引入 task contract、`docs/draft.md`、`docs/plan.md` 和 orchestrator state machine | 4.1、5.4、5.5 |
| Candidate 生命周期不够面向 agent 决策 | 区分 KF 内层 `InnerTrial` 与外层 `Candidate`；只有完成一个 custom task 内层循环后才生成外层 candidate，并支持 top-K / Pareto / diverse branch best kernels 受控进入外层 E2E | 4.2、4.2.1、5 |
| 测试/harness 通常随 task 静态定义，难以随 kernel 变体安全演化 | 增加 `UPDATE_TASK_TESTS` 与 `update_task_tests`，将 `TEST_SPEC`、`PERF_SPEC`、`HARNESS` 作为可版本化 blocks | 5.6.1、6.1.3 |
| 调用 LLM 时 prompt 过于臃肿，token 消耗量大且生成质量可能变差 | 优化 prompt 构造：按需求、candidate 状态和前期反馈动态生成更小更精准的 prompt；采用 delta prompts、结构化错误摘要、静态 skill 缓存、RAG/历史 kernel 精选和显式 token budget | 5.6、7.3、8.8 |
| Correctness gate 容易被 public fixed tests 过拟合 | 增加 randomized/hidden shapes、output poisoning、reference/custom isolation、profiler sanity 和 anti-gaming checks | 5.8、9.2 |
| Microbenchmark 与真实 workload impact 之间存在断层 | 增加 `integrate_runtime`、`benchmark_e2e` 与 E2E output equivalence | 5.11、5.12、6.8、6.9 |
| Profiling 主要发生在 candidate 级别，缺少 workload-level profiling 和 hotspot selection 作为一等阶段 | 在 `BASELINE_PROFILE` / `PLAN_OUTER_ROUND` 中加入 workload profiling、top hotspot selection、op contribution ranking 和 optimization opportunity estimation，使 agent 先决定“优化哪个 op/kernel”再进入 KF task 内层循环 | 5.3、5.4、13.3 |
| MCP/tool 接口粒度偏粗，当前更像单一 `build_and_test` 封装，不利于 agent 做分层验证和精细决策 | 将 KF 能力拆成稳定 tool APIs：`load_task`、`build_kernel`、`verify_correctness`、`benchmark_micro`、`profile_kernel`、`record_candidate` 等，并保持对原有接口的兼容 | 6、11、12 |
| Profiler feedback 与下一步策略选择之间缺少标准映射 | 增加 `profile_kernel` 的 structured diagnosis、bound-to-strategy mapping 和 profile-guided greedy refine | 5.10、8.1 |
| MAP-Elites 描述维度过于宏观，导致搜索集中在粗糙小区域且易偏离最优点 | 引入细粒度多维 descriptors、分阶段维度激活、代理模型排序与信任域局部精修的快速收敛策略 | 8.5、8.6 |
| RAG/DB 有基础，但缺少跨 run 的 agent memory/strategy statistics | 增加 `query_memory`、`record_candidate`、strategy statistics 与 warm-start | 4.5、6.6、6.7、8.7 |
| 实验 artifact 可用，但人类可审计报告不足 | 引入 final report、candidate timeline、evidence schema 和 reproducibility instructions | 10.2 |
| 新增 agent 能力可能破坏现有 KF 代码路径 | 采用兼容层、feature flag、可选 schema extension 和回滚机制，保证原有 CLI/batch/evaluator 可继续运行 | 3.4、11、12 |
| 优化 run 时间成本高，full profile 和 E2E benchmark 昂贵 | 引入分层验证、缓存、early stop、profile budget、异步队列和低成本 surrogate 排序 | 8.6、8.8 |

这张映射表也是判断设计是否跑偏的准则：凡是新增模块，都应直接服务于 KF 的现有能力增强或局限补齐；如果某个模块不连接 KF tool/data/execution，就不应进入核心架构。

## 3. 架构总览

框架分为五层，围绕 Kernel Foundry 重新组织为 agent 化分层：

1. 用户接口和 requirement parser。
2. KF-aware orchestrator agent。
3. Kernel Foundry deterministic tools/environment。
4. Skills/knowledge assets。
5. Memory/database/evidence store。

```text
用户自然语言需求
        |
        v
+------------------------------+
| Requirement Parser / Planner |
| - intent                     |
| - task spec                  |
| - metric/budget              |
+--------------+---------------+
               |
               v
+--------------------------------------------------+
| KF-aware Kernel Optimization Orchestrator          |
| - plan                                           |
| - choose target                                  |
| - choose strategy                                |
| - generate / repair / refine                     |
| - decide promote/reject                          |
+----------------------+---------------------------+
                       |
       +---------------+----------------+
       |                                |
       v                                v
+--------------+                 +---------------+
| KF Tools     |                 | Skills        |
| deterministic|                 | domain knowledge|
+--------------+                 +---------------+
       |                                |
       v                                v
+--------------------------------------------------+
| Kernel Foundry Infrastructure                     |
| CustomTask, TaskRunner, Evaluator, Profiler, DB,  |
| RAG, templates, kernel-eval tasks, MCP server     |
+--------------------------------------------------+
                       |
                       v
+--------------------------------------------------+
| Evidence / Memory                                 |
| candidates.jsonl, benchmark.csv, profile reports, |
| Kernel DB, RAG DB, strategy statistics            |
+--------------------------------------------------+
```

### 3.1 KF-core 分层原则

该架构中，Kernel Foundry 不只是底层 executor，而是 agent 的核心事实平面和数据平面：

- Task 必须落到 `CustomTask`、template task 或 kernel-eval task workspace。
- Build、correctness、benchmark、profile 必须通过 KF tools 或由 KF 包装的 runtime tools 执行。
- Candidate、benchmark、profile、RAG 和 report artifacts 应回写 KF run workspace/database。
- 可读 workflow artifacts 只作为 KF run 的 evidence layer，不能替代 KF 的结构化结果。
- Agent orchestration 只负责提出下一步动作；动作是否成功、candidate 是否正确、是否可以 promote，均由 KF deterministic tools 决定。

这种分层可以避免两类偏差：一是把 KF 降级成普通 shell command executor；二是让 workflow 层反客为主，绕开 KF 已有的 task/evaluator/profiler/database 能力。

### 3.2 架构细化（控制面 / 执行面 / 数据面）

为降低耦合并提升可维护性，建议将“总览五层”进一步细化为三个协作平面：

| 平面 | 核心职责 | 关键接口 | 失败处理 |
|---|---|---|---|
| 控制面（Control Plane） | 需求解析、计划、策略选择、预算控制、状态机推进 | `PARSE_REQUIREMENT`、`PLAN`、`SELECT_STRATEGY` | Budget stop、策略降级、人工澄清 |
| 执行面（Execution Plane） | 任务加载、代码变更、构建、验证、benchmark、profile、集成 | `load_task`、`build_kernel`、`verify_correctness`、`benchmark_micro`、`profile_kernel`、`integrate_runtime` | 自动回滚、candidate reject、进入 repair |
| 数据面（Data Plane） | candidate lineage、性能/正确性证据、memory 检索、报告与复现 | `record_candidate`、`query_memory`、`benchmark_e2e`、report writers | 数据校验失败即拒绝 promotion |

建议实现上保持“控制面无副作用、执行面幂等可重试、数据面可审计可追溯”。

### 3.3 总览优化建议

当前总览可继续优化为“主循环 + 快速路径 + 安全闸门”的可执行设计：

1. **外层主循环**：`PARSE_REQUIREMENT -> PLAN_OUTER_ROUND -> RUN_KF_CUSTOM_TASK_EVOLUTION -> SELECT_E2E_FINALISTS -> MATERIALIZE_CANDIDATE -> INTEGRATE -> BENCHMARK_E2E -> DECIDE_NEXT_ROUND`。
2. **内层快速循环**：`GENERATE_KERNEL -> UPDATE_TASK_TESTS -> BUILD -> VERIFY -> BENCHMARK_MICRO -> PROFILE_OPTIONAL -> SELECT_INNER_BEST`。
3. **快速路径**：对高置信模板任务允许 `NL -> implicit contract -> task`，但仍必须产出可审计 contract（见 4.1.1）。
4. **安全闸门**：内层任何 kernel 必须经过 correctness/perf tests；外层任何 candidate 必须经过 E2E evidence 才能 promote。
5. **收敛闸门**：内层按 micro budget 快速收敛，外层按 E2E budget 控制轮数，防止昂贵测试爆炸。

配套矢量图：

- Agent 执行流程图：`DESIGNED_FLOW.svg`
- Agent 框架架构图：`DESIGNED_ARCH.svg`

### 3.4 与当前 Kernel Foundry 的兼容性原则

本设计会引入新的执行逻辑、tools、skills、metadata schema 和 run artifacts，但必须保持与当前 KF 代码兼容。原则是“增量扩展，不破坏现有路径”：

| 扩展点 | 兼容性要求 | 推荐实现方式 |
|---|---|---|
| 执行逻辑 | 现有 batch generation、task evaluator、CLI scripts 不应被强制改写 | 新增 `AgentOrchestrator` 或 `agent/` module，通过配置启用 |
| Tools | 现有 `build_and_test` 能继续使用，新 tools 作为 superset | 在 MCP/tool server 中新增 `load_task`、`verify_correctness`、`profile_kernel` 等接口，并保留旧接口 |
| Task schema | 现有 `task.py/config.yaml` 仍可运行 | 通过 optional blocks/metadata 扩展 `TEST_SPEC`、`PERF_SPEC`、`HARNESS`，缺省时回退旧逻辑 |
| Database | 现有 `Kernel`、`Task`、`Job`、`JobLog` 不被破坏 | 新增可选表或 JSON metadata 字段，如 `AgentCandidate`、`E2EResult`、`test_version` |
| Skills | 不要求修改核心 evaluator 才能使用 skill | Skills 作为 planner/orchestrator 的只读知识输入，输出必须转成 KF tool call |
| Search | 保留现有 MAP-Elites/greedy/batch 策略 | 新增 strategy selector，在 feature flag 下选择多维收敛策略 |
| Reports | 不改变原始 logs/artifacts 结构 | 在 run workspace 下新增 `docs/`、`candidates.jsonl`、`final_report.md` |

建议引入如下 feature flags：

```yaml
agent_mode: false
agent_dynamic_tests: false
agent_e2e_integration: false
agent_multidim_search: false
agent_report_artifacts: true
```

默认情况下，现有 KF 行为保持不变；只有开启对应 flag 时，agent 才接管 state machine、动态测试更新、多维搜索或 E2E 集成。

兼容性验收标准：

- 现有 `run_test*.sh`、batch experiments 和 kernel-eval tasks 在不开启 agent flags 时行为不变。
- 新增 tools 能包装现有 evaluator/profiler，而不是复制一套独立执行系统。
- 新增 schema 字段必须 optional，并能被旧代码忽略。
- Agent run 失败时可以回退到原始 task workspace 和最后一个 correct candidate。

## 4. 核心概念

### 4.1 Task contract

每次优化 run 都从 task contract 开始。但该 contract 不是独立于 KF 的抽象文档；它必须能被解析为 Kernel Foundry 的 task、template、evaluator config 或 runtime integration target。

### 4.1.1 自然语言是否可以直接生成 KF task（跳过 TaskContract）

结论：**可以支持“直达模式”，但不能真正省略 contract 语义**。即使用户不显式看到 `TaskContract`，系统也必须在内部生成一个 `implicit contract` 并落盘，用于后续验证、预算控制和可复现。

原因：

- 没有 contract，`VERIFY` 无法确定容差、shape 覆盖和 hidden tests 约束；
- 没有 contract，`DECIDE` 无法判断 promotion 是否达标；
- 没有 contract，run 难以复现，也无法解释策略选择；
- 没有 contract，容易出现“生成了可编译 kernel，但目标不一致或验证降级”的风险。

建议采用“双模式”设计：

| 模式 | 入口 | 是否显式展示 contract | 适用场景 |
|---|---|---|---|
| 标准模式（默认） | `NL -> TaskContract -> KF task` | 是 | 新任务、复杂约束、E2E 优化 |
| 直达模式（Fast Path） | `NL -> KF task` | 对用户可隐藏，但系统必须自动生成并持久化 `implicit contract` | 模板任务、约束简单、低风险快速试验 |

优点与缺点：

| 方案 | 优点 | 缺点 |
|---|---|---|
| 显式 TaskContract | 目标清晰、可审计、易复现、利于预算与风险控制 | 首次交互步骤更多，前置成本略高 |
| 直达模式（内部隐式 contract） | 启动快、用户体验简洁、适合模板化任务 | 需求歧义风险更高；若隐式 contract 质量差，会放大后续修复成本 |

设计要求：直达模式只是在交互层“跳过显式展示”，而不是在系统层删除 contract。

字段：

| 字段 | 描述 |
|---|---|
| `task_name` | 人类可读任务名。 |
| `objective` | 要优化的目标。 |
| `source` | 现有任务路径、workload 路径、kernel 文件或模板。 |
| `target_hardware` | 示例：`b580`、`lnl`、`bmg`、`a770`、`sm100`。 |
| `runtime_backend` | 示例：Kernel Foundry task、OpenVINO、PyTorch-XPU、oneDNN。 |
| `kernel_language` | SYCL、OCL、CUDA、Triton。 |
| `correctness_policy` | 容差、shape 覆盖、hidden/random tests。 |
| `metric` | Latency、throughput、memory bandwidth、energy、E2E latency。 |
| `budget` | Token、时间、trials、profiler passes。 |
| `promotion_criteria` | 接受 final candidate 的条件。 |

示例：

```yaml
task_name: gated_delta_net_ocl_lnl
objective: reduce kernel latency while preserving numerical correctness
source: /mnt/river/kernel_foundry/kernelfoundry.kernel-eval/tasks/ov_ocl/gated_delta_net
target_hardware: lnl
runtime_backend: kernelfoundry-task
kernel_language: OCL
correctness_policy:
  rtol: 1e-3
  atol: 1e-3
  randomized_trials: 5
  include_hidden_shapes: true
metric:
  primary: mean_latency
  secondary: p95_latency
budget:
  max_trials: 12
  max_full_profiles: 3
  max_wall_time_minutes: 60
promotion_criteria:
  must_compile: true
  must_pass_correctness: true
  min_speedup: 1.05
  no_hidden_shape_regression: true
```

### 4.2 InnerTrial 与 Candidate

需要区分两个粒度：

- **`InnerTrial`**：KF `CustomTask` 内部 evolution loop 的一次 kernel variant 尝试。它有 build/correctness/perf/profile 结果，但不做 E2E，也不直接作为外层 promotion 单位。
- **`Candidate`**：一个 KF `CustomTask` 内层循环结束后，由该轮 task 选出的最佳 kernel 及其证据包。它是外层 agent loop 的评估单位，会进入 integration、E2E benchmark 和 promotion decision。

因此，外层 `Candidate` 不是“单次 KF 生成结果”，而是“一个 KF custom task 内部多轮 evolution 的汇总产物”。

字段：

| 字段 | 描述 |
|---|---|
| `candidate_id` | 唯一 ID。 |
| `parent_id` | Parent candidate 或 baseline。 |
| `source_task_run_id` | 生成该 candidate 的 KF custom task run。 |
| `inner_best_trial_id` | KF 内层循环中被选为最佳的 trial ID。 |
| `inner_trial_count` | 该 custom task 内部实际执行的 kernel variants 数量。 |
| `strategy` | `repair`、`greedy_refine`、`beam`、`map_elites`、`hyperparam_tune` 等。 |
| `kernel_code_ref` | 指向源代码的文件或 DB reference。 |
| `prompt_ref` | Prompt 或 prompt summary reference。 |
| `build_result` | 构建状态和错误。 |
| `verification_result` | 正确性结果。 |
| `benchmark_result` | Microbenchmark 统计。 |
| `profile_result` | Profiler 摘要。 |
| `e2e_result` | Runtime/workload 结果，如适用。 |
| `status` | `generated`、`build_failed`、`wrong`、`benchmarked`、`promoted`、`rejected`。 |
| `rejection_reason` | 若被拒绝，记录结构化原因。 |

KF 内层 trial results 应保存在 KF 原有 run logs/database 中；外层 candidates 会写入 `candidates.jsonl`，也可选写入 Kernel Foundry DB。

### 4.2.1 多分支 finalist 选择：top-K / Pareto / diverse branch best kernels

KF 内层 evolution loop 可能不是单一路径，而是包含多个策略分支或 family，例如不同 tiling、memory layout、vectorization、subgroup、SLM、unroll 或 launch-parameter 分支。此时不能简单假设 microbenchmark top-1 一定是 E2E 最优。设计上应支持将 **top-K / Pareto / diverse branch best kernels** 受控地 materialize 成外层 candidates，并进入外层 E2E 测试。

但该能力必须受预算和证据约束，不能无脑把很多 branch 都送去 E2E。E2E benchmark 成本很高，默认策略仍应是：

```text
default: top-1 best inner kernel
optional: top-K / Pareto / diverse branch best kernels under E2E budget
```

推荐增加 `SELECT_E2E_FINALISTS` 步骤，位于 `RUN_KF_INNER_EVOLUTION_LOOP` 与 `MATERIALIZE_OUTER_CANDIDATE` 之间：

```text
KF CustomTaskRun
  -> many InnerTrials
  -> branch-level best kernels
  -> SELECT_E2E_FINALISTS
  -> one or more outer Candidates
  -> integration / E2E / promotion decision
```

选择策略：

| 规则 | 说明 |
|---|---|
| Top-1 baseline | 若最佳 inner kernel 明显领先，默认只 materialize top-1。 |
| Near-best threshold | 允许选择与 best micro latency 差距在阈值内的候选，例如 3%～5%。 |
| Pareto frontier | 对 latency、variance、profile health、register pressure、memory traffic 等多指标取 Pareto 前沿。 |
| Branch diversity | 同一 branch/family 最多选择少量代表，避免 E2E 浪费在近似重复候选上。 |
| Correctness confidence | 必须通过 correctness、hidden/random checks 和 anti-gaming checks。 |
| Integration risk | 对 runtime 接入风险较低、接口稳定的候选优先。 |
| E2E budget cap | 由 `TaskContract.cost_budget.max_e2e_runs` 或 `max_e2e_candidates_per_round` 硬限制。 |

示例策略：

```yaml
e2e_finalist_policy:
  default_mode: top1
  enable_multi_finalists: true
  max_e2e_candidates_per_round: 3
  near_best_micro_threshold_pct: 5
  max_per_branch_family: 1
  require_correctness: true
  require_hidden_tests: true
  prefer_pareto_frontier: true
  diversity_keys:
    - strategy_family
    - memory_layout
    - tile_shape
    - launch_config
```

需要支持的中间概念：

| 概念 | 粒度 | 是否 E2E | 作用 |
|---|---|---|---|
| `InnerTrial` | 单次 KF 内层 kernel variant | 否 | build/correctness/micro/profile 记录。 |
| `BranchBest` | 某个 branch/family 内的最佳 inner trial | 否 | 用于多分支汇总和 diversity selection。 |
| `E2EFinalist` | 准备 materialize 的 finalist | 可选进入 | 受 top-K/Pareto/diversity/budget 过滤。 |
| `Candidate` | materialized 外层候选 | 是 | 进入 integration、E2E 和 promotion gate。 |

推荐决策逻辑：

- 如果 best inner kernel 显著优于其他分支（例如 micro latency 领先 > 10% 且 variance 稳定），只选择 top-1。
- 如果多个 branch 性能接近（例如差距 ≤ 3%～5%），选择 top-K 或 Pareto 前沿中的少量代表。
- 如果 micro/E2E 相关性历史上较弱，或 branch 的 integration 行为差异很大，适当增加 finalist 数量。
- 如果 E2E 非常昂贵或 budget 已接近耗尽，强制降级为 top-1 或 top-2。
- 如果候选只是在相同 family 内做微小参数变化，优先用 microbenchmark 和 profiler 决定，不重复消耗 E2E。

这一机制的目标是：既不因为只测 top-1 而错过真实 workload 最优，也不让 E2E 成本爆炸。外层 E2E 的输入应是“少量、高质量、结构多样且证据完整”的 finalists。

### 4.3 Environment

Environment 是围绕 Kernel Foundry 的确定性 executor。它暴露 tool APIs。LLM 不直接决定正确性或速度；它提交 candidates，并接收结构化反馈。

### 4.4 Skills

Skills 是静态、可复用的知识资产。它们应足够紧凑以支持 prompt caching，并足够模块化以按 backend/hardware 替换。

### 4.5 Memory

Memory 存储历史 kernels、task signatures、profiler findings、benchmark results 和 strategy success rates。它支持 warm-start 和跨 run 学习。

## 5. Agent 状态机

```text
INIT
  -> PARSE_REQUIREMENT
  -> PLAN_OUTER_ROUND
  -> LOAD_OR_CREATE_CUSTOM_TASK
  -> BASELINE_PROFILE
  -> RUN_KF_INNER_EVOLUTION_LOOP
    -> GENERATE_KERNEL_VARIANT
    -> UPDATE_TASK_TESTS optional
    -> BUILD
    -> VERIFY
    -> BENCHMARK_MICRO
    -> PROFILE_INNER_TRIAL optional
    -> RECORD_INNER_TRIAL
    -> SELECT_INNER_NEXT_STEP
    -> loop until inner budget/stop
  -> SELECT_INNER_BEST_KERNEL
  -> SELECT_E2E_FINALISTS optional
  -> MATERIALIZE_OUTER_CANDIDATE
  -> INTEGRATE optional
  -> BENCHMARK_E2E outer-only optional
  -> RECORD
  -> DECIDE
       -> done: REPORT
    -> continue: PLAN_OUTER_ROUND
       -> blocked: REPORT_BLOCKER
```

该状态机的核心变化是：`BUILD`、`VERIFY`、`BENCHMARK_MICRO` 和可选 inner profile 属于 KF custom task 内层循环；`SELECT_E2E_FINALISTS`、`INTEGRATE` 与 `BENCHMARK_E2E` 属于外层 agent loop。E2E 不应出现在内层 loop 中。

### 5.1 `PARSE_REQUIREMENT`

输入：自然语言。

输出：`TaskContract`。

职责：

- 推断 target hardware，
- 推断 kernel language，
- 推断 optimization metric，
- 定位 task/workload/kernel files，
- 判断是否需要模板，
- 仅在必要时询问最少的澄清问题。

### 5.2 `LOAD_OR_CREATE_CUSTOM_TASK`

输入：`TaskContract`。

输出：Kernel Foundry `CustomTask` 或生成的 task workspace。

职责：

- 加载现有 `task.py/config.yaml` 任务，
- 或从 `kernelfoundry.templates` 合成任务，
- 验证任务结构，
- 抽取 `REFERENCE` 和 `EVOLVE` blocks，
- 推导 shape 和 correctness policy。

### 5.2.1 `RUN_KF_INNER_EVOLUTION_LOOP`

输入：`CustomTask`、inner-loop budget、strategy hints、correctness/perf tests。

输出：KF 内部 trial logs、best inner kernel、microbenchmark summary、profile summary（可选）。

职责：

- 调用 KF 现有 `CustomTaskController` / evaluator / runner 进行多轮 kernel generation 或 mutation，
- 每个 inner trial 都执行 build、correctness tests 和 performance tests，
- 可选对 top inner trials 做 micro/profile 诊断，
- 保存每次 inner trial 的 correct/perf 结果，
- 根据 inner-loop stop criteria 结束循环并选择 best inner kernel。

内层循环禁止做：

- runtime integration，
- E2E benchmark，
- workload-level output equivalence，
- 外层 promotion decision。

这样可以保持 KF 内层 loop 低成本、高吞吐，并避免昂贵 E2E 被重复执行。

### 5.3 `BASELINE_PROFILE`

目的：避免盲目优化。

对于孤立任务：

- 运行 reference kernel/model，
- 收集 baseline latency，
- 可选收集 unitrace/ncu profile。

对于 workload/runtime：

- 运行 workload profiling，
- 识别 top kernels/operators，
- 估计 optimization opportunity。

### 5.4 `PLAN_OUTER_ROUND`

写入：

- `docs/draft.md`，
- `docs/plan.md`。

计划内容：

- baseline summary，
- risks，
- 按 expected value 和 risk 排序的 custom task directions，
- validation/evaluation commands 或 tool calls，
- budgets，
- promotion criteria。

### 5.5 `SELECT_STRATEGY`

Orchestrator 根据状态选择 strategy。

默认顺序：

1. 若 compile/runtime/correctness 失败，使用 `repair`。
2. 若已有 correct candidate 且 profile feedback 清晰，使用 `greedy_refine`。
3. 若代码结构良好但 tile/workgroup sizes 不确定，使用 `hyperparam_tune`。
4. 若存在多种合理 family，使用 `beam_search`。
5. 若搜索空间很宽或 greedy 停滞，使用 `map_elites`。
6. 若 memory 中存在更好的历史 candidate，使用 `fallback_to_seed`。

### 5.6 `GENERATE_KERNEL_VARIANT`（内层）

输入：

- task contract，
- 当前 inner trial 或 baseline，
- 简洁 feedback，
- 相关 skill snippets，
- RAG examples 或 seed kernels，
- 选中的 strategy。

输出：

- 修改后的 `EVOLVE` block，
- 可选的 test update proposal，
- inner trial metadata，
- 预期 optimization profile。

生成应优先使用 delta prompts 和 targeted changes，而不是巨大的 full-context prompts。

### 5.6.1 `UPDATE_TASK_TESTS`（内层）

目的：保证 Kernel Foundry 中的测试代码可以随着生成的 kernel candidate 动态更新，同时不破坏 correctness gate。

该步骤是可选但强烈建议的中间状态，发生在 `GENERATE_KERNEL_VARIANT` 之后、`BUILD` 和 `VERIFY` 之前。Agent 不能直接随意覆盖测试文件；它只能提交结构化的 test update proposal，由确定性 tool 在隔离 workspace 中应用、校验、版本化并记录 diff。

需要动态更新测试代码的典型场景包括：

- candidate 改变了 kernel entry、launch 参数、workspace buffer 或 JIT constants，
- candidate 引入新的 layout、padding、tiling metadata 或 epilogue behavior，
- task 从模板生成，需要根据 op signature 自动补齐 shape/dtype coverage，
- profiler 或失败样例暴露了新的 edge case，需要加入 regression test，
- E2E runtime integration 需要生成 runtime-level equivalence test。

更新机制：

1. `LOAD_OR_CREATE_TASK` 将 task 拆成稳定 blocks：`REFERENCE`、`EVOLVE`、`TEST_SPEC`、`PERF_SPEC`、`HARNESS`。
2. `GENERATE_KERNEL_VARIANT` 只产出 inner trial code 和可选 `test_patch`，其中 `test_patch` 必须说明修改原因、影响的 shapes/dtypes、以及是否改变 public API。
3. `UPDATE_TASK_TESTS` 使用 Kernel Foundry 的 code-block update mechanism 在临时 task workspace 中应用 patch。
4. Tool 对更新后的测试执行静态校验：schema 合法、required tests 未删除、reference path 仍独立、hidden/random policy 未降级。
5. Tool 先用 reference implementation 跑更新后的测试，确认测试本身有效，再用 candidate 跑 correctness。
6. 每次 test update 都写入 inner trial metadata：`test_version`、`test_patch_hash`、`test_diff_ref`、`test_update_reason`。
7. 若 test update 失败，inner trial 状态标记为 `invalid_test_update`，不能进入 build/performance 排名。

约束：

- Agent 可以新增测试、扩展 shape/dtype 覆盖、更新 harness 适配新的合法接口。
- Agent 不允许删除 baseline correctness tests，除非 `TaskContract` 显式批准并记录人工可审计原因。
- Agent 不允许降低 tolerance、减少 randomized/hidden checks、移除 output poisoning 或 reference/custom isolation。
- 对影响 correctness policy 的测试更新，必须同时跑 old tests 和 new tests。

因此，测试代码可以动态演化，但演化本身也被 correctness policy 约束，并被记录为 inner trial lineage 的一部分。外层 candidate 只引用最终选中的 `inner_best_trial_id` 和对应 test version。

### 5.7 `BUILD`

使用 Kernel Foundry build infrastructure：

- `TaskRunner.build_custom_task`，
- language-specific compilers，
- 可选 container/queue execution。

输出：

- 结构化 build status，
- 带 source locations 的 compiler errors summary，
- build artifacts。

### 5.8 `VERIFY`

正确性是硬 gate。

Verification policy：

- public task tests，
- randomized tests，
- hidden shape tests，
- output buffer poisoning，
- reference/custom isolation，
- dtype/tolerance checks，
- 相关场景下的 profiler sanity check。

未通过 verification 的 candidates 永不参与 performance 排名。

### 5.9 `BENCHMARK_MICRO`

运行代表性 performance tests。

输出：

- mean/p50/p95 latency，
- 相对 reference/baseline 的 speedup，
- variance，
- per-shape results。

### 5.10 `PROFILE_CANDIDATE`

仅对有希望的 inner trials、外层 finalist candidates 或诊断场景执行。内层 profile 应以 summary 为主；full/source-level profile 只对少数 inner best 或外层 finalist 执行。

Intel 路径：

- unitrace metrics，
- occupancy，
- memory bandwidth，
- SLM bandwidth/conflicts，
- XVE stalls，
- ALU/XMX utilization，
- roofline bound。

CUDA 路径：

- NCU full 和 source-level reports，
- occupancy，
- memory hierarchy，
- tensor core utilization，
- stalls，
- timeline/tail effects。

### 5.11 `INTEGRATE`

早期里程碑可选；E2E optimization 必需。该步骤只属于外层 agent loop，不属于 KF custom task 内层 evolution loop。

Backend-specific modes：

| Backend | 集成方式 |
|---|---|
| Kernel Foundry task | 仅替换 `EVOLVE` block。 |
| PyTorch-XPU | 注册 custom op / extension。 |
| OpenVINO | Custom op 或 extension registration。 |
| oneDNN | Primitive replacement 或 custom primitive path。 |
| OpenCL runtime | 替换 kernel source 并调优 GWS/LWS/JIT constants。 |

Integration 必须返回 rollback token。

### 5.12 `BENCHMARK_E2E`

测量真实 workload impact。该步骤只在外层对 materialized outer candidate 执行，不在 KF custom task 内层每次 trial 中执行。

输出：

- E2E latency/throughput，
- per-op deltas，
- final output equivalence，
- confidence/variance。

### 5.13 `RECORD`

写入：

- KF 内层 trial logs/database rows，
- 外层 `candidates.jsonl`，
- `benchmark.csv`，
- profiler summaries，
- Kernel Foundry DB rows，
- final report artifacts。

### 5.14 `DECIDE`

Promotion rules：

- compile success，
- correctness success，
- anti-gaming success，
- 内层 micro performance improvement 或 accepted neutral result，
- 若启用 E2E mode，则需要 E2E improvement，
- 未超出 budget，
- evidence 已持久化。

## 6. Tool Interface 设计

### 6.1 `load_task`

```yaml
input:
  path: string
output:
  ok: bool
  task_spec: object
  blocks:
    reference: object
    evolve: object
    user_instructions: object
  tests:
    correctness: list
    performance: list
  mutable_blocks:
    test_spec: object
    perf_spec: object
    harness: object
  config: object
```

### 6.1.1 `run_custom_task_evolution`

```yaml
input:
  task_ref: string
  inner_budget:
    max_inner_trials: int
    max_inner_wall_time_minutes: int
    max_inner_profiles: int
  strategy_hints:
    primary: greedy_refine | repair | beam_search | map_elites | hyperparam_tune
    descriptors: list
  cost_policy:
    allow_e2e_inside_inner_loop: false
    smoke_test_first: bool
    early_stop_no_improve_rounds: int
output:
  ok: bool
  task_run_id: string
  inner_trial_count: int
  best_inner_trial_id: string
  branch_best_trial_ids:
    - string
  pareto_trial_ids:
    - string
  best_kernel_ref: string
  best_micro_result: object
  inner_trials_ref: string
  stopped_reason: budget | converged | blocked | error
```

该 tool 是双层循环的核心边界：它把 KF 原生 `CustomTask` 内部 evolution loop 作为一个可调用的内层 kernel 生成循环暴露给外层 agent。它返回一个 inner best kernel，以及可选的 branch best / Pareto trial refs，但不直接返回外层 promoted candidate。

### 6.1.1.1 `select_e2e_finalists`

```yaml
input:
  task_run_id: string
  inner_trials_ref: string
  branch_best_trial_ids:
    - string
  pareto_trial_ids:
    - string
  policy:
    max_e2e_candidates_per_round: int
    near_best_micro_threshold_pct: float
    max_per_branch_family: int
    prefer_pareto_frontier: bool
    require_hidden_tests: bool
    diversity_keys:
      - string
output:
  ok: bool
  finalists:
    - finalist_id: string
      inner_trial_id: string
      branch_family: string
      kernel_ref: string
      selection_reason: string
      micro_result: object
      profile_summary: object
  rejected_near_best:
    - inner_trial_id: string
      reason: string
```

该 tool 负责把 KF 内层多分支结果压缩成少量外层 E2E finalists。它应优先选择 top-K / Pareto / diverse branch best kernels，但必须服从 E2E budget cap。未通过 correctness、hidden/random tests 或 anti-gaming checks 的 trial 不得进入 finalists。

### 6.1.2 `materialize_outer_candidate`

```yaml
input:
  task_run_id: string
  finalist_id: string optional
  best_inner_trial_id: string
  best_kernel_ref: string
  parent_candidate_id: string
output:
  ok: bool
  candidate_id: string
  candidate_ref: string
  source_task_run_id: string
  source_finalist_id: string optional
  inner_best_trial_id: string
  inner_trial_count: int
  candidate_summary: object
```

该 tool 将一个完成内层 evolution 的 KF task run 中被选中的 finalist 汇总为外层 `Candidate`。默认只 materialize `best_inner_trial_id`；启用多 finalist 策略时，可对 `select_e2e_finalists` 返回的多个 finalist 分别 materialize，但数量必须受 `max_e2e_candidates_per_round` 和总 E2E budget 限制。只有 materialized candidate 才进入 integration、E2E benchmark 和 promotion decision。

### 6.1.3 `update_task_tests`

```yaml
input:
  task_ref: string
  candidate_ref: string
  test_patch:
    reason: string
    affected_blocks:
      - TEST_SPEC | PERF_SPEC | HARNESS
    diff: string
    changes_public_api: bool
    shapes_added: list
    dtypes_added: list
  policy:
    allow_remove_required_tests: bool
    require_reference_pass: bool
    require_old_and_new_tests: bool
    forbid_tolerance_relaxation: bool
output:
  ok: bool
  updated_task_ref: string
  test_version: string
  test_patch_hash: string
  test_diff_ref: string
  static_checks:
    schema_valid: bool
    required_tests_preserved: bool
    hidden_policy_preserved: bool
    reference_isolation_preserved: bool
    tolerance_not_relaxed: bool
  reference_validation:
    passed: bool
    failures: list
  errors:
    - message: string
```

该 tool 是动态测试更新的关键边界：LLM 只提出 patch，Kernel Foundry tool 负责应用、校验、版本化和回滚。后续 inner-loop `build_kernel`、`verify_correctness`、`benchmark_micro` 都使用 `updated_task_ref`，从而确保 kernel code、test code、harness 和 benchmark spec 始终绑定到同一个 inner trial 版本。

### 6.2 `build_kernel`

```yaml
input:
  task_ref: string
  candidate_code: string
  language: string
  gpu_arch: string
  timeout_s: int
output:
  ok: bool
  compiled: bool
  artifacts_ref: string
  errors:
    - file: string
      line: int
      message: string
  raw_log_ref: string
```

### 6.3 `verify_correctness`

```yaml
input:
  task_ref: string
  candidate_ref: string
  policy:
    randomized_trials: int
    hidden_shapes: bool
    rtol: float
    atol: float
    poison_outputs: bool
output:
  ok: bool
  correct: bool
  failures:
    - shape: object
      dtype: string
      error_kind: string
      max_abs_error: float
      message: string
  anti_gaming:
    passed: bool
    checks: object
```

### 6.4 `benchmark_micro`

```yaml
input:
  task_ref: string
  candidate_ref: string
  shapes: list
  trials: int
  warmup: int
output:
  ok: bool
  stats:
    mean_us: float
    p50_us: float
    p95_us: float
    std_us: float
  per_shape: list
  speedup_vs_reference: float
```

### 6.5 `profile_kernel`

```yaml
input:
  task_ref: string
  candidate_ref: string
  profiler: unitrace | ncu | auto
  profile_level: summary | full | source
output:
  ok: bool
  bound: memory | compute | latency | balanced | unknown
  metrics: object
  diagnosis: list
  recommended_actions: list
  artifacts_ref: string
```

### 6.6 `query_memory`

```yaml
input:
  op_signature: string
  hardware: string
  language: string
  top_k: int
output:
  matches:
    - candidate_ref: string
      speedup: float
      correctness_policy: object
      strategy: string
      summary: string
```

### 6.7 `record_candidate`

```yaml
input:
  candidate: object
output:
  ok: bool
  candidate_id: string
  db_id: string
```

### 6.8 `integrate_runtime`

```yaml
input:
  runtime_backend: string
  workload_ref: string
  candidate_ref: string
  op_target: object
output:
  ok: bool
  integrated_ref: string
  rollback_token: string
  modified_files: list
  log_ref: string
```

### 6.9 `benchmark_e2e`

```yaml
input:
  integrated_ref: string
  workload_ref: string
  trials: int
  validation: bool
output:
  ok: bool
  output_equivalent: bool
  latency:
    mean_us: float
    p50_us: float
    p95_us: float
  throughput: float
  speedup_vs_baseline: float
  per_op_delta: list
```

约束：`benchmark_e2e` 只能接收 `materialize_outer_candidate` 或 `integrate_runtime` 之后的外层 candidate/integrated ref，不应被内层 `run_custom_task_evolution` 调用。

## 7. Skills 与 Assets

### 7.1 必需 skills

| Skill | 目的 | 初始来源 |
|---|---|---|
| `kf-task-authoring` | 创建 robust Kernel Foundry tasks 和 tests。 | `kernelfoundry.templates` + kernel-eval tasks。 |
| `intel-gpu-profile` | unitrace/Intel GPU profiling diagnosis。 | `UnitraceProfilerFeedback` + oneAPI docs。 |
| `sycl-ocl-optimization` | SYCL/OpenCL idioms 和 anti-patterns。 | `languages.py`、`optimization_aware.py`、existing kernels。 |
| `esimd-xmx-dpas` | Intel ESIMD、XMX、DPAS patterns。 | KF optimization prompts + curated examples。 |
| `runtime-integration` | OpenVINO/PyTorch-XPU/oneDNN integration。 | 新文档/工具。 |
| `search-strategies` | Greedy/beam/MAP-Elites/hyperparam tuning policies。 | `qd_gradient.py`、`map_elites_patterns.py`。 |
| `validation-hardening` | Anti-gaming 和 hidden-shape validation。 | 新文档/工具。 |

### 7.2 Seed databases

Seed data 应包括：

- 按 op signature 组织的 known-good kernels，
- task templates，
- profiler diagnosis examples，
- failed candidate patterns，
- compile error fixes，
- architecture-specific tuning constants，
- 来自真实 workloads 的 shape distributions。

### 7.3 Prompt caching 策略

静态内容应缓存：

- system instructions，
- hardware specs，
- language idioms，
- validation rules，
- runtime integration guides。

动态 prompts 应保持简洁：

- 当前 candidate diff，
- 精确 failure summary，
- top 3 profiler facts，
- selected strategy instruction。

### 7.4 Skills、Tools 与 KF 的协同方式

Skills 和 Tools 的边界必须清晰：**Skills 负责知识和决策建议，Tools 负责执行和事实验证，Kernel Foundry 负责承载状态与结果**。

其中，Tool Adapters 是 Agent 和 Kernel Foundry 之间的稳定契约层。它们不是新的 kernel generator，也不是替代 KF 的执行框架；它们的职责是把外层 agent 的计划、预算、候选选择和状态变更，转换为 Kernel Foundry 原生能力可以执行、记录和复现的 tool calls。

关系可以概括为：

```text
Agent reasoning layer
  -> Tool Adapters
  -> Kernel Foundry native modules
  -> structured evidence
  -> Agent outer-loop decision
```

Tool Adapters 与 KF 的核心边界如下：

| Tool Adapter 职责 | 对应 KF 能力 | 设计要求 |
|---|---|---|
| 参数结构化 | `CustomTask`、config、template、task workspace | 把 `TaskContract`、planner 输出和 skill hints 转为 KF 可消费输入。 |
| 内层执行封装 | `CustomTaskController`、`TaskRunner`、`CustomTaskEvaluator` | 只调用 KF 内层 evolution，不把 E2E 混入 inner loop。 |
| 动态测试更新 | code-block parser、task workspace、evaluator | LLM 只能提交 proposal；KF tool 负责应用、校验、版本化和回滚。 |
| 结果归一化 | build/test/benchmark/profile logs | 把 KF 输出转换成统一 schema，供 orchestrator 和 report 使用。 |
| 候选物化 | KF run artifacts、kernel refs、DB refs | 从 inner trial / finalist 生成外层 `Candidate`，保留 lineage。 |
| Evidence 写回 | KF DB、RAG、run workspace、artifacts | 所有事实证据必须可追踪、可复现、可审计。 |
| 成本控制 | queue、budget、profile policy、E2E policy | 强制执行 inner budget、profile budget 和 E2E budget。 |
| 错误隔离 | rollback token、workspace snapshot、structured errors | 编译失败、测试失败、timeout、环境错误必须结构化返回。 |

Tool Adapters 应该做：

- 将 agent plan 转成 KF task/config/test/profile/evolution 参数；
- 包装 KF 现有 evaluator、profiler、database 和 task runner，而不是复制一套执行系统；
- 强制执行双层边界：KF inner loop 只做 build/correctness/micro/profile，E2E 只在外层执行；
- 将 KF 返回的事实结果归一化为 `InnerTrial`、`BranchBest`、`E2EFinalist`、`Candidate` 和 report artifacts；
- 记录 prompt/config/seed/candidate hash/hardware/driver/commit 等复现信息；
- 在失败时提供可恢复状态，例如 rollback token、last correct candidate 和 rejected reason。

Tool Adapters 不应该做：

- 替代 `CustomTaskController` 或绕开 KF 自建一套不可复现的 inner loop；
- 让 skill hints 或 agent 主观判断直接覆盖 KF 实测结果；
- 隐式修改 KF workspace、DB 或 tests 但不记录 diff/version；
- 将未经 materialize 和 E2E 验证的 inner trial 直接 promote；
- 让 `benchmark_e2e` 被 `run_custom_task_evolution` 在内层循环中调用。

协作流程如下：

1. Orchestrator 根据 `TaskContract` 和当前 candidate 状态选择需要加载的 skill。
2. Skill 提供静态知识、优化模式、反模式、profiler 解释规则或 task authoring 规则。
3. Orchestrator 将 skill 建议转成结构化 tool call，例如 `build_kernel`、`verify_correctness`、`profile_kernel` 或 `update_task_tests`。
4. Tool 调用 KF 的 `CustomTask`、`TaskRunner`、`CustomTaskEvaluator`、profiler feedback 和 database。
5. KF 返回结构化事实：编译结果、正确性结果、benchmark 统计、profile 诊断和 artifact references。
6. Orchestrator 基于这些事实更新 inner trial state 或 outer candidate state，并决定继续内层搜索、materialize candidate、E2E、promote 或开启下一轮 custom task。

| 协作对象 | 输入 | 输出 | 不允许做的事 |
|---|---|---|---|
| Skill | task summary、profile summary、failure summary | 建议、规则、检查清单、策略 hint | 直接判断 correctness 或 speedup |
| Tool | 结构化参数、candidate/task refs | KF 执行结果、artifact refs、错误摘要 | 绕开 KF 自建不可复现执行路径 |
| Kernel Foundry | task workspace、candidate code、config | build/test/profile/database/report artifacts | 被 workflow 层替代或绕过 |
| Orchestrator | contract、tool results、skill hints、budget | 下一步 action、inner/outer decision、report | 基于主观判断 promote candidate |

这种协同方式保证：领域知识可以灵活扩展，但事实判断仍由 KF 的确定性执行链路完成。

## 8. Search Strategy 设计

### 8.1 默认：profile-guided greedy refine

适用场景：

- 存在 correct baseline，
- profiler 识别出清晰瓶颈，
- 可能只需要少量 transformations。

循环：

1. 做一个 targeted change，
2. build，
3. verify，
4. benchmark，
5. 若更好则保留，
6. 否则 revert 或 repair。

### 8.2 Repair mode

适用场景：

- compile failed，
- runtime crashed，
- correctness failed。

输入应是 compact error summaries，而不是 full logs。

### 8.3 Hyperparameter tuning

当代码结构良好且参数占主导时使用：

- tile size，
- block size，
- subgroup size，
- unroll factors，
- vector width，
- local memory padding，
- GWS/LWS/JIT constants。

这通常可以用确定性的 grid/random/Bayesian tuning 完成，而无需每次 trial 都调用 LLM 生成代码。

### 8.4 Beam search

当存在多种合理方案时使用：

- SLM tiling vs subgroup-only，
- vectorized global load vs local memory caching，
- one-pass vs multi-pass algorithm，
- OpenCL C vs SYCL/ESIMD rewrite。

### 8.5 MAP-Elites 局限与改造

当前 MAP-Elites 在 KF 中的典型问题是：描述维度过于宏观（例如 memory/compute/parallelism/ESIMD 四大类），导致 candidate 容易集中在很小且粗糙的区域，探索步长不匹配真实最优点邻域，进化轨迹可能长期偏离最优。

复用 KF 现有 MAP-Elites 的同时，应做两项改造：

1. 从“宏观标签”升级为“细粒度可度量 descriptors”；
2. 从“全维一次性展开”升级为“分阶段维度激活 + 快速收敛”。

细粒度 descriptors 可包含：

- 访存：global load/store 合并度、L2 命中、SLM 读写比、bank conflict 指标；
- 计算：FMA/TensorCore 利用率、指令混合、pipeline stall 分解；
- 并行：occupancy、subgroup 活跃率、warp/wavefront divergence；
- 编译：寄存器压力、spill 次数、关键循环 unroll 因子；
- 运行：shape bucket、dtype、backend-specific launch 参数。

只有当 greedy/beam/hyperparam 在局部停滞时，再进入该改造后的 MAP-Elites 探索分支。

### 8.6 多维搜索下的快速收敛方法

高维搜索会放大空间复杂度。为避免“维度越多越慢”，建议采用分层收敛 pipeline：

1. **阶段A：粗筛（低成本）**
  - 用静态代价模型 + 轻量 benchmark（小样本 trials）筛掉明显劣解；
  - 使用 Successive Halving / Hyperband 对候选做早停。

2. **阶段B：代理排序（中成本）**
  - 用历史 `candidates.jsonl` 训练在线 surrogate（如 GBDT/轻量回归）预测 speedup 与失败概率；
  - 仅保留 Top-K 进入真实 profile。

3. **阶段C：信任域局部精修（高收益）**
  - 围绕当前 Pareto 前沿做 trust-region 搜索（小步变异）；
  - 对连续参数用 Bayesian/CMA-ES 局部优化，对离散参数用 bandit/UCB 调度。

4. **阶段D：自适应维度激活**
  - 先激活影响最大的 2~4 个维度；
  - 仅当边际收益下降到阈值以下时再逐步增加维度；
  - 对低贡献维度进行冻结，减少搜索震荡。

5. **阶段E：收敛判定与回退**
  - 若连续 $n$ 轮提升 < $\epsilon$，触发 early stop；
  - 回退到最后一个“正确且稳定提升”的 candidate，避免过拟合噪声。

该方案的目标是把高维搜索复杂度从“盲目全空间扩展”改为“预算受控的分层收敛”，在不牺牲探索能力的前提下更快逼近最优邻域。

### 8.7 Fallback policy

如果多次 trial 失败：

- 查询 memory 获取 known-good seed，
- 降低 optimization ambition，
- 返回 last correct candidate，
- 仅在真正 blocked 时询问澄清 constraints。

### 8.8 时间成本优化策略

Kernel optimization 的主要时间成本来自 build、correctness tests、benchmark、full profile、E2E integration 和 LLM generation。设计中应显式管理这些成本。

建议采用分层成本模型：

| 阶段 | 成本等级 | 优化策略 |
|---|---|---|
| 生成 candidate | 中 | 使用 delta prompt、缓存静态 skill、限制一次只改一个主要方向 |
| Build | 中 | 编译缓存、增量构建、相同 candidate hash 跳过重复 build |
| Correctness | 中/高 | 先 smoke tests，再 public tests，再 hidden/random tests |
| Microbenchmark | 中 | 小样本 warmup + early reject；finalists 才做稳定统计 |
| Full profile | 高 | 只对 Top-K 或异常 candidate 执行；优先 summary profile |
| E2E benchmark | 很高 | 只对外层 materialized candidate / finalist 执行；禁止进入 KF 内层 trial 循环 |

执行预算应写入 `TaskContract`：

```yaml
cost_budget:
  max_wall_time_minutes: 60
  max_candidates: 12
  max_outer_rounds: 4
  max_inner_trials_per_round: 16
  max_full_profiles: 3
  max_e2e_runs: 2
  smoke_test_first: true
  early_stop_no_improve_rounds: 4
```

调度策略：

- **先便宜后昂贵**：compile/smoke/correctness 失败的 candidate 不进入 benchmark。
- **先 summary 后 full**：profile 先用 summary，只有 Top-K 才做 full/source-level profile。
- **内外层分离**：内层只做 correctness/perf/micro/profile；外层才做 integration 与 E2E。
- **先 micro 后 E2E**：只有一个 custom task 内层循环产出的 best kernel 被 materialize 为 candidate 后，才有资格进入 E2E。
- **缓存一切可缓存项**：candidate hash、build artifact、test result、profile summary、surrogate feature。
- **异步执行昂贵任务**：full profile 和 E2E benchmark 可进入 KF queue，不阻塞 agent 继续做低成本分析。
- **噪声控制**：对低于阈值的微小 speedup 不立即 promote，避免为噪声消耗 full profile budget。

## 9. Reward 与 Validation 设计

### 9.1 Reward components

Candidate score 只能在 correctness 通过后计算。

建议 score：

```text
score = correctness_gate * (
  w_micro * normalized_micro_speedup
  + w_e2e * normalized_e2e_speedup
  + w_robust * shape_robustness
  + w_profile * profiler_health
  - w_cost * normalized_cost
  - w_risk * validation_risk
)
```

其中 `correctness_gate` 在 correctness 失败时为 0。

### 9.2 Anti-reward-hacking checks

必需检查：

1. 每次运行使用随机输入。
2. Hidden shapes 和非 2 的幂维度。
3. Kernel 执行前对 output buffer poisoning。
4. Reference 和 custom output buffer isolation。
5. 如果任务支持，覆盖多个 dtypes。
6. 基于 tolerance 的数值比较，并记录 max error。
7. Profiler sanity：memory/compute activity 必须合理。
8. Runtime-integrated kernels 的 E2E output equivalence。
9. Candidate 不能读取 previous runs 的 expected outputs。

### 9.3 Promotion gates

Final candidate 必须满足：

- build success，
- correctness success，
- hidden/random validation success，
- 无 anti-gaming violation，
- 有统计意义的 performance improvement 或明确接受的 tradeoff，
- 所有 artifacts 已持久化，
- 提供 reproducibility instructions。

## 10. Evidence 与 Report 格式

每个 run workspace 应包含：

```text
run-workspace/
  docs/
    draft.md
    plan.md
  inner-runs/
    task-run-0001/
      inner_trials.jsonl
      best_kernel.*
  candidates.jsonl
  benchmark.csv
  profile/
    candidate-0003/
      REPORT.md
      artifacts/
  outputs/
    best_kernel.*
    final_report.md
  runs/
    kernelfoundry logs/artifacts
```

### 10.1 `candidates.jsonl`

每个 candidate 一个 JSON object：

```json
{
  "candidate_id": "c0007",
  "parent_id": "c0004",
  "source_task_run_id": "task-run-0003",
  "inner_best_trial_id": "trial-0012",
  "inner_trial_count": 16,
  "strategy": "greedy_refine",
  "status": "benchmarked",
  "compiled": true,
  "correct": true,
  "mean_us": 42.1,
  "speedup": 1.18,
  "profile_bound": "memory",
  "decision": "keep",
  "reason": "reduced global loads and passed hidden shapes"
}
```

### 10.2 Final report

Final report 应包括：

- requirement summary，
- task contract，
- baseline metrics，
- outer candidate timeline，
- per-round KF custom task inner-loop summary，
- best candidate code reference，
- correctness evidence，
- benchmark evidence，
- profiler evidence，
- E2E evidence（如适用），
- tradeoffs/risks，
- reproduction instructions，
- recommended next steps。

## 11. 现有 Kernel Foundry 组件如何复用

本节是架构的核心复用清单。设计默认优先复用和扩展 Kernel Foundry 现有组件；只有当 KF 缺少 agent-level workflow、evidence schema 或 runtime integration interface 时，才引入新的 wrapper/tool，而不是绕开原有框架。

| 组件 | 复用计划 |
|---|---|
| `CustomTask` | Task representation 和 code-block update mechanism。 |
| `CustomTaskController` | 保留为内层 kernel evolution loop 的核心执行器；外层 agent 只通过 tool 调用它并读取结果。 |
| `CustomTaskEvaluator` | Build/test/profile result conversion。 |
| Task code-block parser | 将 `REFERENCE`、`EVOLVE`、`TEST_SPEC`、`PERF_SPEC`、`HARNESS` 作为可版本化 blocks 管理，使 kernel 与测试可以同步演化。 |
| `TaskRunner` | Local/Celery/container execution backend。 |
| `PromptConstructor` | RAG/template helper，但由更小的 agent prompt layer 包装。 |
| `InferenceServer` | 非 Copilot workflow 的可选 model access layer。 |
| `UnitraceProfilerFeedback` | Intel profiling diagnosis engine。 |
| `NCUProfilerFeedback` | CUDA profiling diagnosis engine。 |
| `database/tables.py` | 持久 kernel/task/job/RAG memory。为 E2E metadata 扩展。 |
| `optimization_aware.py` | backend-specific skill content 和 strategy hints 来源。 |
| `map_elites_patterns.py` | 可选 MAP-Elites 的 behavior profile extraction。 |
| `qd_gradient.py` | Transition learning 和 gradient-inspired mutation hints。 |
| `mcp/server.py` | Tool-serving foundation；扩展到 `build_and_test` 之外。 |
| `kernelfoundry.templates` | Task synthesis templates。 |
| `kernelfoundry.kernel-eval` | Regression 和 benchmark suite。 |

## 12. 分阶段实施计划

### Phase 0：决策

选择：

- 第一个 runtime/backend，
- 第一个 hardware target，
- 第一个 workload/task，
- local vs queue execution mode。

推荐：

- 第一个 backend：Kernel Foundry task mode 或 OpenCL task mode。
- 第一个 hardware：现有 configs 中可用的 Intel GPU target（`lnl`/`b580`）。
- 第一个 task：`ov_ocl/gated_delta_net` 或较小的 matrix/reduction template。
- 第一次执行：如果硬件可用，使用 local deterministic mode；之后再使用 queue mode。

### Phase 1：Tool API extraction

交付物：

- `load_task`，
- `run_custom_task_evolution`，
- `materialize_outer_candidate`，
- `build_kernel`，
- `verify_correctness`，
- `benchmark_micro`，
- `profile_kernel`，
- `record_candidate`。

验收标准：

- 一个 KF custom task 可以通过 tool APIs 完成内层多轮 generation/build/test/benchmark/profile，并 materialize 成一个外层 candidate。

### Phase 2：KF run workspace + evidence artifacts

交付物：

- 自动 `docs/draft.md`，
- 自动 `docs/plan.md`，
- `candidates.jsonl`，
- `benchmark.csv`，
- `outputs/final_report.md`。

验收标准：

- 另一位工程师可以从 KF run workspace、database records 和 evidence artifacts 中重建优化过程。

### Phase 3：Minimal agent loop

交付物：

- NL requirement → task contract，
- outer loop → KF custom task inner evolution loop，
- compile/correctness/perf feedback repair，
- final promotion report。

验收标准：

- Agent 能运行至少一轮 KF custom task 内层 evolution，收集多个 inner trials，并将最佳 inner kernel materialize 为外层 candidate。

### Phase 4：Validation hardening

交付物：

- randomized/hidden shape policies，
- output poisoning，
- reference/custom isolation，
- anti-gaming summary。

验收标准：

- 对 public fixed shapes 过拟合的 candidates 会被拒绝。

### Phase 5：Profiling-guided optimization

交付物：

- structured unitrace/NCU diagnosis，
- bound-to-strategy mapping，
- ranked next actions。

验收标准：

- Agent 能基于 profiler evidence 解释为什么选择下一步优化。

### Phase 6：Memory 和 search strategies

交付物：

- 从 DB/RAG warm-start，
- strategy statistics，
- 可选 MAP-Elites integration，
- hyperparameter tuning mode。

验收标准：

- 重复类似任务相比 cold start 更快或使用更少 tokens/trials。

### Phase 7：E2E runtime integration

交付物：

- 一个选定 runtime integration backend，
- rollback support，
- E2E benchmark 和 output equivalence。

验收标准：

- 外层 Candidate 可以基于真实 workload improvement 被 promote；E2E 只在外层执行，不进入 KF inner trial loop。

## 13. 示例用例

### 13.1 OpenCL kernel optimization

输入：

> Optimize `gated_delta_net_ocl` for LNL. Keep fp16 error under 1e-3 and improve mean latency.

流程：

1. 加载现有 Kernel Foundry task。
2. 运行 baseline correctness/perf。
3. Profile reference/custom baseline。
4. 生成带 OpenCL subgroup/vector/local-memory changes 的 candidate。
5. Build/verify/benchmark。
6. 保留最佳 candidate 并报告 speedup。

### 13.2 SYCL matrix multiplication optimization

输入：

> Generate a SYCL matmul kernel for BMG with BF16 inputs and FP32 accumulation.

流程：

1. 从 PyTorch→SYCL template 创建 task。
2. 使用 Intel GPU hardware/SYCL skills。
3. 尝试 baseline tiled SLM implementation。
4. 如果支持，探索 sub-group/joint_matrix/ESIMD strategies。
5. 在 shape set 上 verify 并 benchmark。

### 13.3 Runtime-level OpenVINO optimization

输入：

> Find the slowest operation in this OpenVINO model on BMG and optimize it end-to-end.

流程：

1. Profile workload。
2. 选择 top hotspot。
3. 为该 op 生成 Kernel Foundry task。
4. 优化 candidate。
5. 通过 OpenVINO extension 集成。
6. Benchmark E2E 并验证 output equivalence。

## 14. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| Candidate 通过静态测试但泛化错误 | 高 | Hidden/random shapes、output poisoning、E2E checks。 |
| 评估过慢 | 高 | Tiered evaluation，仅对 finalists 做 profiler。 |
| Token 成本过高 | 中/高 | Prompt caching、delta prompts、structured summaries、warm-start。 |
| Runtime integration 是 backend-specific | 高 | 从一个 backend 开始，并定义 rollback interface。 |
| MAP-Elites 浪费 budget | 中 | 设为可选，并仅在 stagnation/broad search 时触发。 |
| Profiler metrics 有噪声 | 中 | 重复 trials，使用 p50/p95，要求统计上有意义的变化。 |
| Agent 做出不受支持的硬件假设 | 中 | Hardware skill + tool-reported worker info + compiler/profiler truth。 |

## 15. 开放决策

1. 第一个 production backend：OpenVINO、PyTorch-XPU、oneDNN，还是仅 Kernel Foundry task？
2. 第一个 target hardware：LNL、BMG/B580、Arc A770、PTL？
3. 第一个 workload/task：gated delta net、matmul、reduction、softmax，还是 model layer？
4. Evaluation mode：先 local only，还是一开始就 Celery/queue？
5. Database extension：修改现有 `Kernel` 表，还是新增独立 `AgentCandidate`/`E2EResult` 表？
6. Skill packaging：独立 repo skills，还是放在 `skills/` 下？

## 16. 推荐首个里程碑

在 E2E runtime integration 前先构建 task-only PoC。

目标：

- 现有任务：`kernelfoundry.kernel-eval/tasks/ov_ocl/gated_delta_net`。
- Backend：Kernel Foundry task evaluator。
- Language：OpenCL。
- Hardware：使用 task config target（`lnl`）或可用本地 Intel GPU。
- Strategy：greedy refine + repair。
- Evidence：`candidates.jsonl`、`benchmark.csv`、`final_report.md`。

成功标准：

- Agent 读取自然语言需求。
- Agent 加载 task 并写计划。
- Agent 至少生成一个 candidate。
- Agent 使用 Kernel Foundry build/verify/benchmark。
- Agent 记录每个 candidate。
- Agent 只 promote correct candidate。
- Final report 包含 metrics 和 reproducibility information。

## 17. 结论

拟议的 Kernel Optimization Agent 本质上是一个 **Kernel Foundry-centered agent architecture**。它的设计主线是让 Kernel Foundry 在保持自身 build/test/profile/RAG/database/template/search 能力的基础上，补齐成为可靠 optimization agent 所需的 orchestration、evidence、validation、高维搜索收敛与 E2E integration 能力。

系统角色边界如下：

- Kernel Foundry 是核心：提供 task abstraction、kernel generation/evolution、build/test/profile、RAG、database、templates、kernel-eval tasks、MCP/tool serving 和未来 runtime integration。
- Workflow discipline 是保障层：提供 task contract、planning、candidate lineage、evidence records、promotion gates 和 human-readable artifacts。
- Kernel Optimization Agent 是补强层：将上述流程保障嵌入 KF 的 deterministic tool loop，并针对 KF 的现有限制增加 requirement parsing、dynamic test update、anti-gaming validation、多维搜索快速收敛、E2E promotion、candidate memory 和 report schema。

最终结果应是一个 KF-first 的 closed-loop kernel optimization agent：相比盲目生成更高效，相比仅性能 reward loop 更安全，相比原始 batch harness 更面向用户和证据驱动，并且足够可扩展，既能支持 Intel-first kernel optimization，也能兼容未来 CUDA/Triton。
