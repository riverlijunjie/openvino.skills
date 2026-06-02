# Kernel Design Agents 框架分析

> 日期：2026-06-01  
> 范围：`/mnt/river/kernel_foundry/kda/kernel-design-agents`  
> 目标：理解 Kernel Design Agents workflow，识别可复用思想、缺口，以及为基于 Kernel Foundry 的 kernel optimization agent 所需的适配。

## 执行摘要

Kernel Design Agents（KDA）有意保持轻量。它不是一个完整的 kernel 生成系统，而是一个面向性能敏感实现任务的可复用 agent workflow 参考。它的核心贡献是流程设计：定义 task contract，让 agent 检查 workspace，起草计划，一次实现一个 candidate，验证、测量、记录证据，并且只在有证据时 promote。

这一理念与 Kernel Foundry 非常互补：

- Kernel Foundry 提供确定性的 kernel build/test/profile 基础设施。
- KDA 提供 evidence-driven agent loop 和文档纪律。
- KDA skills 提供领域知识资产，尤其是 CUDA/Blackwell profiling 与 kernel optimization 参考。

主要限制在于：KDA 当前是 task-agnostic，并且主要由文档/prompt/skill 组成。它缺少具体 environment API、reward system、training loop、candidate database 和 runtime integration layer。对于 Kernel Foundry kernel optimization agent，KDA 应作为 workflow shell 复用，而 Kernel Foundry 应提供可执行 tools 和 task environment。

## 仓库组件

| 路径 | 目的 |
|---|---|
| `README.md` | KDA 高层描述和 minimal workflow。 |
| `CLAUDE.md` | 面向仓库的 agent rules 和 expected workflow。 |
| `docs/agent-flow.md` | Task contract、minimal loop、evidence record、promotion rule。 |
| `prompts/basic-flow.md` | 面向 agent-driven implementation 的通用 starter prompt template。 |
| `prompts/README.md` | Prompt 使用指南。 |
| `skills/KernelWiki` | 结构化 Blackwell/Hopper kernel optimization knowledge base。 |
| `skills/ncu-report-skill` | 面向 B200/sm100 的 Nsight Compute profiling workflow 和 analysis skill。 |

该仓库有意避免存储 downstream task-specific code、datasets、benchmark logs 或私有 acceptance thresholds。

## 当前 Workflow 模型

KDA 的 minimal loop 如下：

1. 创建或进入一个独立的 task implementation workspace。
2. 定义 task contract：
   - objective，
   - inputs and outputs，
   - correctness requirements，
   - constraints，
   - validation command，
   - evaluation command，
   - promotion criteria。
3. 让 agent 检查 code/docs/tests。
4. 要求 agent 写入 `docs/draft.md`。
5. 将 draft 转换为 `docs/plan.md` 或可执行计划。
6. 一次实现一个 candidate。
7. 每次有意义变更后验证 correctness。
8. 在适用时测量 performance。
9. 记录 candidate relationships、benchmark evidence、profiling evidence 和 promotion decisions。
10. 重复，直到满足 promotion criteria 或明确剩余 blockers。

这对 coding agents 是一个强 workflow，因为它减少无纪律的 trial-and-error，并要求 promotion 前必须有证据。

## 架构特征

### 1. Workflow-first，而非 framework-heavy

KDA 避免硬编码 benchmark harness 或 hardware target。task workspace 拥有 code、tests、datasets、validators、profiling data 和 outputs。这让 KDA 具备可移植性，但也意味着 downstream systems 必须提供可执行 environment tools。

### 2. 可复用 workflow 与 task-specific artifacts 清晰分离

`CLAUDE.md` 明确说明：

- 将 task-specific prompts、datasets、validators、generated implementations、benchmark logs 和 candidates 保持在 generic repo 之外；
- 将 generated outputs 放入 `runs/`、`outputs/` 或 `profile/`；
- 保持仓库聚焦在 reusable flow mechanics。

这与 Kernel Foundry 很匹配：生成的 kernels 和 results 应存放在 `runs/`、database records 和 task workspaces 中，而不是污染 framework code。

### 3. Evidence records 是一等输出

KDA 推荐：

- `docs/draft.md`，
- `docs/plan.md`，
- `benchmark.csv`，
- `candidates.jsonl`，
- `profile/`，
- `runs/` 或 `outputs/`。

这些记录对 kernel optimization 很重要，因为性能声明往往很脆弱。未来工程师必须知道哪个 candidate 被测试、使用了什么 shape、在哪个硬件上测试，以及为什么接受或拒绝它。

### 4. Promotion 需要证据

Promotion rule 简单但重要：只有当 candidate 满足 task contract，并有证据证明它提升或保持 target metric 时，才能 promote。被拒绝的 candidates 应记录原因。

这一规则应成为新 optimization agent 中的硬 gate。

## Skills 分析

### `ncu-report-skill`

该 skill 是面向 B200/sm100 CUDA kernels 的详细 profiling playbook。它强制：

- 先 profile 再 diagnose，
- 每次 profiling run 创建一个独立 run directory，
- 必要时使用 standalone harness，
- 收集 full 和 source-level NCU reports，
- 用 Python APIs 解析 reports，而不是肉眼看 CLI 输出，
- 分析 occupancy、balance、stalls、tensor core usage、timeline 和 memory，
- 将观察结果匹配到 diagnosis playbook，
- 写 evidence-backed report。

对 Kernel Foundry 可复用的思想：

- “Profile → Diagnose → Plan” 应成为默认优化纪律。
- Profiler reports 应产生结构化证据和排序后的 action list。
- 每次 run 应隔离 harness、reports、analysis 和 final report。
- 建议必须引用具体 metrics，而不是模糊描述。

对 Kernel Foundry 的限制：

- 它是 CUDA/B200-specific。
- Kernel Foundry 的 Intel focus 需要等价的 unitrace/VTune/oneAPI profiling skill。
- 该 skill 不是 executable API；它是 workflow 和 knowledge asset。

### `KernelWiki`

这是一个聚焦 NVIDIA Blackwell/Hopper 的结构化 kernel optimization wiki。它包含：

- hardware feature pages，
- kernel technique pages，
- merged PR references，
- blog/doc summaries，
- query scripts，
- candidate ledgers，
- confidence 和 provenance rules。

可复用思想：

- 构建可搜索、带版本的 optimization knowledge base。
- 将 claims 关联到 confidence levels 和 sources。
- 按 symptom、technique、hardware feature、kernel type、language 和 repo 建索引。
- 使用具体 PR/code references 作为 few-shot 或 design evidence。

对 Kernel Foundry 的限制：

- 当前范围是 NVIDIA Blackwell/Hopper，而不是 Intel GPU/SYCL/OpenCL。
- 对 Kernel Foundry，需要并行建设 `IntelKernelWiki` 或 `IntelGpuOptimizationWiki`。
- 它的 query scripts 是本地脚本，尚未集成到 Kernel Foundry planner/tool API。

## 优势

### 1. 优秀的流程纪律

KDA 将 coding agents 经常跳过的环节形式化：

- task contract，
- plan draft，
- executable plan，
- candidate tracking，
- 每次变更后 validation，
- evidence-backed promotion。

对于 kernel 工作，这种纪律至关重要，因为“快但错”的 kernel 很常见。

### 2. Task-agnostic 且可复用

由于 KDA 避免绑定到某个 harness，它可以包装 Kernel Foundry、CUDA contests、compiler passes、runtime integrations 或 infrastructure work。

### 3. 基于 Skill 的领域知识

KDA 将领域知识视为可选 skills。这与 kernel optimization agent 设计一致：Intel GPU hardware、OpenCL/SYCL idioms、profiling diagnosis 和 runtime integration 都应作为模块化资产。

### 4. 人类可读 artifacts

推荐的 `draft.md`、`plan.md`、`benchmark.csv` 和 `candidates.jsonl` 让优化过程可检查。这对 debugging 和 review 很有价值。

### 5. Candidate lineage

KDA 要求 agents 记录 candidate parent relationships 和 outcomes。Kernel search 能从中受益，因为许多优化是 incremental mutations 或 fallback repairs。

## 面向 Kernel Optimization 的缺口

### 1. 没有具体 environment API

KDA 没有定义如下可执行接口：

- build candidate，
- run correctness tests，
- run performance tests，
- profile candidate，
- integrate candidate into runtime，
- rollback candidate。

Kernel Foundry 可以填补这一缺口。

### 2. 除 promotion criteria 外没有 reward system

KDA 使用人类可读的 promotion rules，但没有定义 reward shaping、score normalization、multi-objective tradeoffs 或 anti-gaming safeguards。

对 Kernel Foundry，reward 必须结合：

- compile success，
- correctness，
- micro performance，
- E2E performance，
- 跨 shapes 的 robustness，
- code maintainability/safety，
- budget cost。

### 3. 没有自动 training 或 learning loop

用户请求中提到了从 environment feedback 学习。KDA 本身不是 RL/training framework；它是一个 agentic implementation loop。Kernel Optimization Agent 初期可以通过 memory、strategy statistics 和 candidate databases 增加学习能力，而不是直接 fine-tuning model。

### 4. 没有内置 task synthesis

KDA 假设 task contract 已提供。Kernel Foundry 用户需要 natural-language requirement parsing 和 task/template generation。

### 5. 没有 E2E workload integration

KDA 可应用于 E2E 工作，但它不提供 runtime-specific integration 机制。对于真正 workload optimization，Kernel Foundry 必须增加 backend-specific integration tools。

### 6. Skills 偏 NVIDIA

当前 KDA domain assets 在 CUDA/B200 上最强。Kernel Foundry 需要 Intel-first skills：

- Xe architecture guide，
- BMG/LNL/PTL hardware facts，
- SYCL 2020 idioms，
- OpenCL subgroup patterns，
- ESIMD/DPAS/XMX usage，
- unitrace metric interpretation，
- OpenVINO/oneDNN/PyTorch-XPU integration。

### 7. Evidence recording 偏手动

KDA 推荐 evidence files，但没有强制 schema 或自动写入。Kernel Foundry 应从结构化评估输出自动生成 `candidates.jsonl`、benchmark tables 和 reports。

## 针对 Kernel Foundry 的推荐适配

### 1. 将 KDA 用作外层 agent process

采用以下 mandatory stages：

1. task contract generation，
2. workspace inspection，
3. plan draft，
4. executable plan，
5. candidate implementation，
6. correctness validation，
7. performance measurement，
8. evidence recording，
9. promotion/rejection decision。

### 2. 用 Kernel Foundry tools 替换手动命令

KDA 的 “validation command” 和 “evaluation command” 应映射到 Kernel Foundry tool calls：

- `build_kernel`，
- `verify_correctness`，
- `benchmark_micro`，
- `profile_kernel`，
- `benchmark_e2e`，
- `integrate_runtime`。

### 3. 标准化 evidence schema

使用 KDA 的文件，但让它们 machine-readable：

- `docs/draft.md`：初始 reasoning 和 risks。
- `docs/plan.md`：executable plan 和 budgets。
- `candidates.jsonl`：每个 candidate 一行，包含 parent、strategy、status、metrics、artifacts。
- `benchmark.csv`：标准化 performance stats。
- `profile/<candidate>/REPORT.md`：profiler evidence 和 diagnosis。
- `outputs/final_report.md`：最终 promoted kernel summary。

### 4. 建设 Intel-specific skills

创建与 KDA NVIDIA skills 对应的 Kernel Foundry skills：

- `intel-gpu-profile-skill`：unitrace/VTune workflow、metric names、common bottlenecks。
- `intel-kernel-wiki`：SYCL/OpenCL/ESIMD/DPAS/XMX examples 和 PR/code references。
- `kf-task-authoring-skill`：如何生成 robust Kernel Foundry tasks。
- `runtime-integration-skill`：OpenVINO/PyTorch-XPU/oneDNN integration recipes。

### 5. 强制 promotion gates

除非所有 gates 通过，否则 candidate 不应被 promote：

- compile success，
- public 和 hidden/randomized tests 上的 correctness，
- 无 anti-gaming violation，
- microbenchmark improvement 或 no regression，
- E2E improvement 或明确接受的 tradeoff，
- reproducible evidence 已写盘。

### 6. 保持 KDA 的 concerns separation

保持 generic agent workflow docs 与 downstream task workspaces 分离。生成 kernels、logs、datasets 和 profiles 应存放在 task-specific workspaces 或 Kernel Foundry run directories 中。

## 受 KDA 启发的 Agent 状态模型

| 状态 | 描述 | Kernel Foundry 映射 |
|---|---|---|
| `TASK_CONTRACT` | 将用户意图解析为 objective/constraints/metrics | NL→TaskSpec planner |
| `INSPECT` | 读取 task/runtime/harness context | workspace + template inspection |
| `DRAFT_PLAN` | 创建初始优化计划 | `docs/draft.md` |
| `EXEC_PLAN` | 转换为可执行循环 | `docs/plan.md` |
| `GENERATE` | 创建或变异 candidate | LLM + RAG + skills |
| `BUILD` | 编译 candidate | `TaskRunner.build_custom_task` |
| `VERIFY` | 正确性验证 | `CustomTaskEvaluator` / pytest |
| `BENCH` | Microbenchmark | `measure_runtime` / pytest performance |
| `PROFILE` | 收集瓶颈证据 | unitrace / ncu |
| `INTEGRATE` | 将 candidate 插入 runtime | 新 runtime-specific tool |
| `E2E_BENCH` | 测量 workload impact | 新 E2E evaluator |
| `RECORD` | 写 evidence 和 DB record | `candidates.jsonl` + DB |
| `PROMOTE_OR_REJECT` | 最终 candidate gate | KDA promotion rule |

## 结论

KDA 不是 Kernel Foundry 的替代品。它是围绕 Kernel Foundry 可执行基础设施所缺少的 agent discipline。最佳设计是复用 KDA 的 task-contract、planning、candidate、evidence 和 promotion workflow，同时让 Kernel Foundry 提供 generation、build、verification、profiling、benchmarking、memory 和未来 runtime integration 的确定性 tools。
