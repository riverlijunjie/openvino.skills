# Kernel Optimization Agent Framework Design

> Date: 2026-06-01  
> Version: v0.2  
> Scope: design a Kernel Foundry-first kernel optimization agent that reuses Kernel Foundry execution infrastructure and adds agentic workflow discipline around it.  
> Status: design proposal.

## 1. Vision

The Kernel Optimization Agent turns Kernel Foundry from a configured batch generation harness into a user-facing, evidence-driven, closed-loop optimization system.

A user should be able to say:

> “Optimize the OpenCL gated delta net kernel for LNL/BMG. Prioritize latency, keep numerical error under 1e-3, and report the best candidate with profiling evidence.”

The agent should then:

1. understand the natural-language requirement,
2. construct or locate the Kernel Foundry task,
3. profile the baseline or workload,
4. pick optimization targets,
5. generate or mutate kernel candidates,
6. build and verify them,
7. benchmark and profile promising candidates,
8. learn from feedback and historical memory,
9. integrate the best candidate when requested,
10. produce a reproducible report with all evidence.

The key design principle is:

> Let the LLM reason, but let deterministic tools decide truth.

### 1.1 Positioning alignment

This design does not introduce an external workflow framework to replace Kernel Foundry. The primary/secondary relationship must be explicit:

- **Kernel Foundry is the core**: it owns task abstraction, kernel generation/evolution, build/test/profile, RAG, database, templates, benchmark, and future runtime integration.
- **Workflow discipline is the key methodology**: it contributes agent workflow principles such as task contracts, plan-first execution, candidate lineage, evidence records, promotion gates, and human-readable artifacts.
- **The Kernel Optimization Agent is a KF-first enhancement layer**: it adds a natural-language entry point, agent orchestrator, dynamic test/task updates, anti-gaming validation, E2E promotion, and structured evidence around existing Kernel Foundry capabilities. It is not a separate optimization framework.

Therefore, Kernel Foundry’s role in this design is the **execution environment, source of truth, and persistence system**. Final judgments about correctness, performance, profiling, and promotion must come from Kernel Foundry tool outputs, not from subjective agent reasoning.

### 1.2 Two-level loop design principle

We must avoid misunderstanding one KF `CustomTask` as one single kernel iteration. The real design should use a two-level loop:

- **Inner loop: KF CustomTask evolution loop**  
  Kernel Foundry performs multiple rounds of kernel generation/mutation, build, correctness tests, performance tests, optional micro/profile feedback, and internal best-kernel selection inside one custom task.
- **Outer loop: Agent E2E optimization loop**  
  After each KF custom task inner loop finishes, the agent materializes the best kernel selected by KF into an outer `Candidate`, then performs integration, E2E benchmark, promotion decision, and planning for the next task round.

Key constraint: **E2E tests must not enter the KF custom task inner loop**. E2E is too expensive and should only run in the outer loop on a small number of finalist candidates. The inner loop only runs micro/correctness/performance tests to quickly filter kernels; the outer loop validates real workload impact.

## 2. Design Goals

### 2.1 Functional goals

- Natural-language triggered optimization.
- Closed-loop kernel generation, testing, profiling, benchmarking, and refinement.
- Support for multiple kernel languages: SYCL, OpenCL, CUDA, Triton, with Intel SYCL/OpenCL prioritized.
- Reuse Kernel Foundry task templates, evaluator, profiler feedback, RAG, and database.
- Reuse workflow discipline patterns such as task contracts, planning, candidate tracking, evidence records, and promotion gates.
- Provide E2E evaluation interfaces for runtime/workload-level validation.
- Support a library of tools, skills, and seed databases.
- Explicitly fill the current gaps in Kernel Foundry as a batch/search harness: agentic orchestration, candidate lifecycle, dynamic test updates, E2E workload integration, and report/evidence schema.

### 2.2 Quality goals

- Correctness before performance.
- Evidence-backed performance claims.
- Robust anti-reward-hacking validation.
- Lower token usage than blind batch search.
- Lower evaluation cost through staged/tiered testing.
- Extensible backends, search strategies, and hardware knowledge.
- Reproducible reports and candidate lineage.
- New agent capabilities must remain compatible with current Kernel Foundry, preferably through adapters, wrappers, feature flags, and optional tool extensions, without breaking existing batch experiments.
- Optimize time cost by reducing invalid trials, limiting full-profile passes, reusing cached results, and assigning explicit budgets to search, validation, and profiling.

### 2.3 Non-goals for the first version

- Do not train a new kernel LLM initially.
- Do not support every runtime backend in the first milestone.
- Do not replace Kernel Foundry’s existing batch experiments; wrap and reuse them.
- Do not rewrite Kernel Foundry with an external workflow framework; KF remains the core execution and data plane.
- Do not make MAP-Elites the mandatory main loop.

### 2.4 Mapping KF limitations to design additions

This design starts from the gaps Kernel Foundry has when evolving from a batch generation/search platform into a user-facing kernel optimization agent. The mapping is:

| Current Kernel Foundry limitation | Design addition | Main sections |
|---|---|---|
| Configuration and task entry are engineering-oriented; users cannot easily start optimization from natural language | `PARSE_REQUIREMENT` converts natural language into a `TaskContract`, then loads or generates a KF task | 5.1, 5.2 |
| Strong batch generation/search, but weak explicit agent planning | Add task contracts, `docs/draft.md`, `docs/plan.md`, and an orchestrator state machine | 4.1, 5.4, 5.5 |
| Candidate lifecycle is not sufficiently shaped for agent decisions | Separate inner KF `InnerTrial` from outer `Candidate`; only generate outer candidates after a custom task inner loop finishes, and support controlled top-K / Pareto / diverse branch best kernels entering outer E2E | 4.2, 4.2.1, 5 |
| Tests/harness are usually statically defined with the task and are hard to evolve safely with kernel variants | Add `UPDATE_TASK_TESTS` and `update_task_tests`, treating `TEST_SPEC`, `PERF_SPEC`, and `HARNESS` as versioned blocks | 5.6.1, 6.1.3 |
| LLM prompts can become overly bloated, causing high token cost and potentially worse generation quality | Optimize prompt construction: dynamically generate smaller and more precise prompts based on requirements, candidate state, and prior feedback; use delta prompts, structured error summaries, static skill caching, curated RAG/history kernels, and explicit token budgets | 5.6, 7.3, 8.8 |
| Correctness gates can overfit public fixed tests | Add randomized/hidden shapes, output poisoning, reference/custom isolation, profiler sanity checks, and anti-gaming checks | 5.8, 9.2 |
| Microbenchmark results can diverge from real workload impact | Add `integrate_runtime`, `benchmark_e2e`, and E2E output equivalence | 5.11, 5.12, 6.8, 6.9 |
| Profiling mainly happens at candidate level, while workload-level profiling and hotspot selection are not first-class stages | Add workload profiling, top-hotspot selection, op contribution ranking, and optimization opportunity estimation in `BASELINE_PROFILE` / `PLAN_OUTER_ROUND`, so the agent first decides which op/kernel to optimize before entering the KF task inner loop | 5.3, 5.4, 13.3 |
| MCP/tool interfaces are too coarse-grained and currently resemble a single `build_and_test` wrapper, limiting tiered validation and fine-grained agent decisions | Split KF capabilities into stable tool APIs such as `load_task`, `build_kernel`, `verify_correctness`, `benchmark_micro`, `profile_kernel`, and `record_candidate`, while preserving compatibility with existing interfaces | 6, 11, 12 |
| Profiler feedback lacks a standard mapping to the next strategy | Add structured diagnosis in `profile_kernel`, bound-to-strategy mapping, and profile-guided greedy refinement | 5.10, 8.1 |
| MAP-Elites descriptors are too coarse, concentrating search in rough small regions and drifting away from optima | Add fine-grained multi-dimensional descriptors, staged dimension activation, surrogate ranking, and trust-region local refinement for fast convergence | 8.5, 8.6 |
| RAG/DB exists, but cross-run agent memory and strategy statistics are missing | Add `query_memory`, `record_candidate`, strategy statistics, and warm-start | 4.5, 6.6, 6.7, 8.7 |
| Experiment artifacts exist, but human-auditable reports are insufficient | Add final reports, candidate timelines, evidence schema, and reproduction instructions | 10.2 |
| New agent capability may break existing KF paths | Use compatibility layers, feature flags, optional schema extensions, and rollback mechanisms to keep existing CLI/batch/evaluator paths working | 3.4, 11, 12 |
| Optimization runs can be expensive due to full profiles and E2E benchmarks | Add tiered validation, caching, early stopping, profile budgets, async queues, and low-cost surrogate ranking | 8.6, 8.8 |

This table is also the guardrail for the design: every new module should directly enhance an existing KF capability or fill a KF limitation. If a module does not connect to KF tools, data, or execution, it should not enter the core architecture.

## 3. Architecture Overview

The framework has five layers, reorganized around Kernel Foundry as an agentic stack:

1. User interface and requirement parser.
2. KF-aware orchestrator agent.
3. Kernel Foundry deterministic tools/environment.
4. Skills/knowledge assets.
5. Memory/database/evidence store.

```text
User NL Requirement
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

### 3.1 KF-core layering principle

In this architecture, Kernel Foundry is not merely a low-level executor; it is the agent’s core fact plane and data plane:

- Tasks must resolve to `CustomTask`, template tasks, or kernel-eval task workspaces.
- Build, correctness, benchmark, and profile must run through KF tools or runtime tools wrapped by KF.
- Candidate, benchmark, profile, RAG, and report artifacts should be written back to the KF run workspace/database.
- Human-readable workflow artifacts are only the evidence layer of a KF run; they must not replace KF structured results.
- Agent orchestration only proposes the next action; whether the action succeeds, whether a candidate is correct, and whether it may be promoted are all decided by KF deterministic tools.

This layering avoids two common mistakes: demoting KF to a generic shell-command executor, and letting the workflow layer bypass KF’s existing task/evaluator/profiler/database capabilities.

### 3.2 Architecture refinement: control plane / execution plane / data plane

To reduce coupling and improve maintainability, the five-layer overview should be refined into three cooperating planes:

| Plane | Core responsibility | Key interfaces | Failure handling |
|---|---|---|---|
| Control Plane | Requirement parsing, planning, strategy selection, budget control, state-machine progress | `PARSE_REQUIREMENT`, `PLAN`, `SELECT_STRATEGY` | Budget stop, strategy downgrade, human clarification |
| Execution Plane | Task loading, code changes, build, verification, benchmark, profile, integration | `load_task`, `build_kernel`, `verify_correctness`, `benchmark_micro`, `profile_kernel`, `integrate_runtime` | Automatic rollback, candidate rejection, enter repair |
| Data Plane | Candidate lineage, performance/correctness evidence, memory retrieval, reports and reproduction | `record_candidate`, `query_memory`, `benchmark_e2e`, report writers | Promotion is rejected if evidence validation fails |

Implementation should keep the control plane side-effect free, the execution plane idempotent and retryable, and the data plane auditable and traceable.

### 3.3 Architecture optimization suggestions

The overview can be further turned into an executable design with a “main loop + fast path + safety gates” structure:

1. **Outer main loop**: `PARSE_REQUIREMENT -> PLAN_OUTER_ROUND -> RUN_KF_CUSTOM_TASK_EVOLUTION -> SELECT_E2E_FINALISTS -> MATERIALIZE_CANDIDATE -> INTEGRATE -> BENCHMARK_E2E -> DECIDE_NEXT_ROUND`.
2. **Inner fast loop**: `GENERATE_KERNEL -> UPDATE_TASK_TESTS -> BUILD -> VERIFY -> BENCHMARK_MICRO -> PROFILE_OPTIONAL -> SELECT_INNER_BEST`.
3. **Fast path**: high-confidence template tasks may use `NL -> implicit contract -> task`, but still must produce an auditable contract (see 4.1.1).
4. **Safety gate**: every inner kernel must pass correctness/performance tests; every outer candidate must have E2E evidence before promotion.
5. **Convergence gate**: the inner loop converges under a micro budget; the outer loop controls rounds under an E2E budget to prevent expensive evaluation blow-up.

Related diagrams:

- Agent execution flow: `DESIGNED_FLOW.svg`
- Agent architecture diagram: `DESIGNED_ARCH.svg`

### 3.4 Compatibility principles with current Kernel Foundry

This design introduces new execution logic, tools, skills, metadata schema, and run artifacts, but it must remain compatible with current KF code. The principle is: **incremental extension, no breakage of existing paths**.

| Extension point | Compatibility requirement | Recommended implementation |
|---|---|---|
| Execution logic | Existing batch generation, task evaluator, and CLI scripts should not be forcibly rewritten | Add an `AgentOrchestrator` or `agent/` module enabled by config |
| Tools | Existing `build_and_test` remains usable; new tools are a superset | Add `load_task`, `verify_correctness`, `profile_kernel`, etc. in the MCP/tool server while preserving old interfaces |
| Task schema | Existing `task.py/config.yaml` tasks continue to run | Extend `TEST_SPEC`, `PERF_SPEC`, `HARNESS` through optional blocks/metadata, falling back to old logic when absent |
| Database | Existing `Kernel`, `Task`, `Job`, and `JobLog` are not broken | Add optional tables or JSON metadata fields such as `AgentCandidate`, `E2EResult`, and `test_version` |
| Skills | Skills must not require core evaluator changes | Skills are read-only planner/orchestrator inputs whose outputs must become KF tool calls |
| Search | Existing MAP-Elites/greedy/batch strategies remain available | Add a strategy selector and enable multidimensional convergence strategies behind feature flags |
| Reports | Raw logs/artifacts structure remains unchanged | Add `docs/`, `candidates.jsonl`, and `final_report.md` under the run workspace |

Recommended feature flags:

```yaml
agent_mode: false
agent_dynamic_tests: false
agent_e2e_integration: false
agent_multidim_search: false
agent_report_artifacts: true
```

By default, current KF behavior stays unchanged. The agent only takes over the state machine, dynamic test updates, multidimensional search, or E2E integration when the corresponding flag is enabled.

Compatibility acceptance criteria:

- Existing `run_test*.sh`, batch experiments, and kernel-eval tasks behave unchanged when agent flags are off.
- New tools wrap existing evaluator/profiler components instead of copying an independent execution system.
- New schema fields are optional and ignorable by old code.
- If an agent run fails, it can roll back to the original task workspace and the last correct candidate.

## 4. Core Concepts

### 4.1 Task contract

Every optimization run starts with a task contract, but that contract is not an external abstraction disconnected from Kernel Foundry; it must be reducible to a Kernel Foundry task, template, evaluator config, or runtime integration target.

### 4.1.1 Can natural language directly generate a KF task and skip TaskContract?

Conclusion: **a fast path can be supported, but contract semantics cannot truly be skipped**. Even if the user does not explicitly see a `TaskContract`, the system must generate and persist an internal `implicit contract` for later validation, budget control, and reproducibility.

Reasons:

- Without a contract, `VERIFY` cannot determine tolerance, shape coverage, or hidden-test constraints.
- Without a contract, `DECIDE` cannot determine whether promotion criteria are satisfied.
- Without a contract, a run is hard to reproduce and strategy choices are hard to explain.
- Without a contract, the system risks generating a compilable kernel whose target is inconsistent or whose validation has been weakened.

Recommended two-mode design:

| Mode | Entry | Contract shown explicitly? | Suitable scenarios |
|---|---|---|---|
| Standard mode (default) | `NL -> TaskContract -> KF task` | Yes | New tasks, complex constraints, E2E optimization |
| Fast Path | `NL -> KF task` | May be hidden from the user, but the system must automatically generate and persist an `implicit contract` | Template tasks, simple constraints, low-risk quick trials |

Tradeoffs:

| Approach | Advantages | Disadvantages |
|---|---|---|
| Explicit TaskContract | Clear target, auditable, reproducible, good for budget and risk control | More first-interaction steps and slightly higher upfront cost |
| Fast Path with internal implicit contract | Faster startup, simpler user experience, good for templated tasks | Higher ambiguity risk; a poor implicit contract amplifies later repair cost |

Design requirement: the fast path only skips explicitly showing the contract at the interaction layer; it must not remove contract semantics at the system layer.

Fields:

| Field | Description |
|---|---|
| `task_name` | Human-readable task name. |
| `objective` | What to optimize. |
| `source` | Existing task path, workload path, kernel file, or template. |
| `target_hardware` | Example: `b580`, `lnl`, `bmg`, `a770`, `sm100`. |
| `runtime_backend` | Example: Kernel Foundry task, OpenVINO, PyTorch-XPU, oneDNN. |
| `kernel_language` | SYCL, OCL, CUDA, Triton. |
| `correctness_policy` | Tolerance, shape coverage, hidden/random tests. |
| `metric` | Latency, throughput, memory bandwidth, energy, E2E latency. |
| `budget` | Token, time, trials, profiler passes. |
| `promotion_criteria` | Conditions for accepting final candidate. |

Example:

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

### 4.2 Candidate

A candidate is one kernel implementation plus its metadata.

Fields:

| Field | Description |
|---|---|
| `candidate_id` | Unique ID. |
| `parent_id` | Parent candidate or baseline. |
| `source_task_run_id` | The Kernel Foundry custom task run that generated this candidate. |
| `inner_best_trial_id` | The best trial selected from the inner loop. |
| `inner_trial_count` | Number of inner-loop kernel variants executed for the task. |
| `strategy` | `repair`, `greedy_refine`, `beam`, `map_elites`, `hyperparam_tune`, etc. |
| `kernel_code_ref` | File or DB reference to source. |
| `prompt_ref` | Prompt or summarized prompt reference. |
| `build_result` | Build status and errors. |
| `verification_result` | Correctness result. |
| `benchmark_result` | Microbenchmark stats. |
| `profile_result` | Profiler summary. |
| `e2e_result` | Runtime/workload result, if applicable. |
| `status` | `generated`, `build_failed`, `wrong`, `benchmarked`, `promoted`, `rejected`. |
| `rejection_reason` | Structured reason if rejected. |

Candidates are written to `candidates.jsonl` and optionally to Kernel Foundry’s DB.

### 4.2.1 Multi-finalist selection: top-K / Pareto / diverse branch best kernels

The inner Kernel Foundry evolution loop may contain multiple strategy branches or families, for example different tiling choices, memory layouts, vectorization strategies, subgroup strategies, SLM strategies, unroll factors, or launch-parameter families. In that case, we should not assume that the single best microbenchmark result is always the best E2E candidate. The design should support materializing **top-K / Pareto / diverse branch best kernels** into outer candidates and sending them to outer E2E evaluation.

However, this must be budget- and evidence-constrained; we cannot blindly send many branches to E2E. E2E benchmark cost is high, so the default policy should still be:

```text
default: top-1 best inner kernel
optional: top-K / Pareto / diverse branch best kernels under an E2E budget
```

Recommended `SELECT_E2E_FINALISTS` step placement:

```text
KF CustomTaskRun
  -> many InnerTrials
  -> branch-level best kernels
  -> SELECT_E2E_FINALISTS
  -> one or more outer Candidates
  -> integration / E2E / promotion decision
```

Selection rules:

| Rule | Meaning |
|---|---|
| Top-1 baseline | If the best inner kernel is clearly ahead, only materialize top-1. |
| Near-best threshold | Allow candidates within a small micro-latency gap, e.g. 3%–5%. |
| Pareto frontier | Use multiple metrics such as latency, variance, profiler health, register pressure, and memory traffic. |
| Branch diversity | Keep only a few representatives per branch/family to avoid near-duplicate E2E spend. |
| Correctness confidence | Must pass correctness, hidden/random checks, and anti-gaming checks. |
| Integration risk | Prefer candidates with low runtime integration risk and stable interfaces. |
| E2E budget cap | Hard-limited by `TaskContract.cost_budget.max_e2e_runs` or `max_e2e_candidates_per_round`. |

Example policy:

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

Useful intermediate concepts:

| Concept | Granularity | E2E? | Role |
|---|---|---|---|
| `InnerTrial` | one inner kernel variant | No | build/correctness/micro/profile record. |
| `BranchBest` | best inner trial inside one branch/family | No | supports branch aggregation and diversity selection. |
| `E2EFinalist` | finalist ready to materialize | Optional | filtered by top-K/Pareto/diversity/budget. |
| `Candidate` | materialized outer candidate | Yes | enters integration, E2E, and promotion gate. |

Decision logic:

- If the best inner kernel clearly beats the others, only select top-1.
- If multiple branches are close, choose a small top-K or Pareto subset.
- If micro and E2E correlation is weak or branch integration behavior differs materially, increase finalist count a bit.
- If E2E is very expensive or the budget is close to exhausted, downgrade to top-1 or top-2.
- If a candidate only makes a small parameter change inside the same family, let microbenchmark/profiler decide; do not repeatedly spend E2E budget.

The goal is to avoid both failure modes: missing the true workload-optimal kernel because only top-1 was tested, and exploding E2E cost by sending too many similar branches.

### 4.3 Environment

The environment is the deterministic executor around Kernel Foundry. It exposes tool APIs. The LLM does not directly decide correctness or speed; it submits candidates and receives structured feedback.

### 4.4 Skills

Skills are static, reusable knowledge assets. They should be compact enough for prompt caching and modular enough to swap by backend/hardware.

### 4.5 Memory

Memory stores historical kernels, task signatures, profiler findings, benchmark results, and strategy success rates. It supports warm-start and learning across runs.

## 5. Agent State Machine

```text
INIT
  -> PARSE_REQUIREMENT
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

### 5.1 `PARSE_REQUIREMENT`

Input: natural language.

Output: `TaskContract`.

Responsibilities:

- infer target hardware,
- infer kernel language,
- infer optimization metric,
- locate task/workload/kernel files,
- decide whether a template is needed,
- ask only minimal clarifying questions when required.

### 5.2 `LOAD_OR_CREATE_CUSTOM_TASK`

Input: `TaskContract`.

Output: Kernel Foundry `CustomTask` or generated task workspace.

Responsibilities:

- load an existing `task.py/config.yaml` task,
- or synthesize from `kernelfoundry.templates`,
- validate task structure,
- extract `REFERENCE` and `EVOLVE` blocks,
- derive shape and correctness policy.

### 5.2.1 `RUN_KF_INNER_EVOLUTION_LOOP`

Input: `CustomTask`, inner-loop budget, strategy hints, correctness/performance tests.

Output: KF inner-loop trial logs, best inner kernel, microbenchmark summary, and optional profile summary.

Responsibilities:

- call the existing Kernel Foundry `CustomTaskController` / evaluator / runner for multiple rounds of kernel generation or mutation,
- run build, correctness tests, and performance tests for every inner trial,
- optionally profile the top inner trials,
- save each inner-trial correctness/performance result,
- stop according to inner-loop criteria and choose the best inner kernel.

The inner loop must not do:

- runtime integration,
- E2E benchmarking,
- workload-level output equivalence,
- outer promotion decisions.

This keeps the Kernel Foundry inner loop low-cost and high-throughput while preventing expensive E2E work from being repeated unnecessarily.

### 5.3 `BASELINE_PROFILE`

Purpose: avoid optimizing blindly.

For isolated tasks:

- run reference kernel/model,
- collect baseline latency,
- optionally collect unitrace/ncu profile.

For workload/runtime:

- run workload profiling,
- identify top kernels/operators,
- estimate optimization opportunity.

### 5.4 `PLAN_OUTER_ROUND`

Writes:

- `docs/draft.md`,
- `docs/plan.md`.

Plan content:

- baseline summary,
- risks,
- candidate directions ranked by expected value and risk,
- validation/evaluation commands or tool calls,
- budgets,
- promotion criteria.

### 5.5 `SELECT_STRATEGY`

The orchestrator picks a strategy based on state.

Default order:

1. `repair` if compile/runtime/correctness failed.
2. `greedy_refine` if there is a correct candidate with clear profile feedback.
3. `hyperparam_tune` if code is structurally good but tile/workgroup sizes are uncertain.
4. `beam_search` if several plausible families exist.
5. `map_elites` if search is broad or greedy has stagnated.
6. `fallback_to_seed` if memory contains a better prior candidate.

### 5.6 `GENERATE_KERNEL_VARIANT`

Inputs:

- task contract,
- current candidate or baseline,
- concise feedback,
- relevant skill snippets,
- RAG examples or seed kernels,
- selected strategy.

Output:

- modified `EVOLVE` block,
- optional test update proposal,
- inner-trial metadata,
- expected optimization profile.

Generation should prefer delta prompts and targeted changes over huge full-context prompts.

### 5.6.1 `UPDATE_TASK_TESTS`

Purpose: ensure that Kernel Foundry test code can evolve dynamically with kernel candidates without breaking the correctness gate.

This step is optional but strongly recommended. It happens after `GENERATE_KERNEL_VARIANT` and before `BUILD` / `VERIFY`. The agent must not overwrite test files freely; it may only submit a structured test update proposal that deterministic tools apply, validate, version, and record in an isolated workspace.

Typical cases that require dynamic test updates:

- the candidate changes kernel entry points, launch parameters, workspace buffers, or JIT constants,
- the candidate introduces new layouts, padding, tiling metadata, or epilogue behavior,
- a task is generated from a template and shape/dtype coverage must be filled in from the op signature,
- profiler results or failures expose a new edge case that needs regression coverage,
- E2E runtime integration needs a runtime-level equivalence test.

Update mechanism:

1. `LOAD_OR_CREATE_TASK` splits the task into stable blocks: `REFERENCE`, `EVOLVE`, `TEST_SPEC`, `PERF_SPEC`, `HARNESS`.
2. `GENERATE_KERNEL_VARIANT` emits inner-trial code and an optional `test_patch`, and the patch must explain the reason, affected shapes/dtypes, and whether the public API changes.
3. `UPDATE_TASK_TESTS` applies the patch in a temporary task workspace through Kernel Foundry’s code-block update mechanism.
4. The tool performs static validation on the updated tests: schema validity, required tests preserved, reference path remains isolated, hidden/random policy not weakened.
5. The tool first runs the reference implementation against the updated tests to verify the tests themselves, then runs the candidate for correctness.
6. Every test update is written into inner-trial metadata: `test_version`, `test_patch_hash`, `test_diff_ref`, `test_update_reason`.
7. If test update fails, the inner-trial status becomes `invalid_test_update` and it cannot enter build/performance ranking.

Constraints:

- The agent may add tests, expand shape/dtype coverage, and update harness logic for new valid interfaces.
- The agent may not remove baseline correctness tests unless `TaskContract` explicitly approves it and records an auditable reason.
- The agent may not relax tolerances, reduce randomized/hidden checks, or remove output poisoning or reference/custom isolation.
- For test updates that affect correctness policy, the old tests and new tests must both be run.

This means tests may evolve dynamically, but that evolution is also constrained by correctness policy and recorded as part of the inner-trial lineage.

### 5.7 `BUILD`

Uses Kernel Foundry build infrastructure:

- `TaskRunner.build_custom_task`,
- language-specific compilers,
- optional container/queue execution.

Output:

- structured build status,
- compiler errors summarized with source locations,
- build artifacts.

### 5.8 `VERIFY`

Correctness is a hard gate.

Verification policy:

- public task tests,
- randomized tests,
- hidden shape tests,
- output buffer poisoning,
- reference/custom isolation,
- dtype/tolerance checks,
- profiler sanity check when relevant.

Candidates that fail verification are never ranked by performance.

### 5.9 `BENCHMARK_MICRO`

Runs representative performance tests.

Output:

- mean/p50/p95 latency,
- speedup vs reference/baseline,
- variance,
- per-shape results.

### 5.10 `PROFILE_CANDIDATE`

Only for promising candidates or diagnosis.

Intel path:

- unitrace metrics,
- occupancy,
- memory bandwidth,
- SLM bandwidth/conflicts,
- XVE stalls,
- ALU/XMX utilization,
- roofline bound.

CUDA path:

- NCU full and source-level reports,
- occupancy,
- memory hierarchy,
- tensor core utilization,
- stalls,
- timeline/tail effects.

### 5.11 `INTEGRATE`

Optional in early milestones; required for E2E optimization.

Backend-specific modes:

| Backend | Integration approach |
|---|---|
| Kernel Foundry task | Replace `EVOLVE` block only. |
| PyTorch-XPU | Register custom op / extension. |
| OpenVINO | Custom op or extension registration. |
| oneDNN | Primitive replacement or custom primitive path. |
| OpenCL runtime | Replace kernel source and tune GWS/LWS/JIT constants. |

Integration must return a rollback token.

### 5.12 `BENCHMARK_E2E`

Measures true workload impact.

Output:

- E2E latency/throughput,
- per-op deltas,
- final output equivalence,
- confidence/variance.

### 5.13 `RECORD`

Writes:

- `candidates.jsonl`,
- `benchmark.csv`,
- profiler summaries,
- Kernel Foundry DB rows,
- final report artifacts.

### 5.14 `DECIDE`

Promotion rules:

- compile success,
- correctness success,
- anti-gaming success,
- micro performance improvement or accepted neutral result,
- E2E improvement if E2E mode is enabled,
- budget not exceeded,
- evidence persisted.

## 6. Tool Interface Design

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

This tool is the core boundary of the two-level loop: it exposes the inner Kernel Foundry `CustomTask` evolution loop as a callable kernel-generation loop to the outer agent. It returns the best inner kernel, plus optional branch-best / Pareto trial refs, but not the outer promoted candidate.

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

This tool compresses multi-branch inner results into a small set of outer E2E finalists. It should prioritize top-K / Pareto / diverse branch best kernels, but it must obey the E2E budget cap. Any trial that fails correctness, hidden/random tests, or anti-gaming checks must not enter the finalists set.

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

This tool summarizes a completed inner evolution run into an outer `Candidate`. By default, it only materializes `best_inner_trial_id`. When multi-finalist mode is enabled, it may materialize multiple finalists returned by `select_e2e_finalists`, but the count must still respect `max_e2e_candidates_per_round` and the overall E2E budget.

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

This tool is the key boundary for dynamic test updates: the LLM only proposes a patch, while the Kernel Foundry tool applies, validates, versions, and rolls it back. Subsequent inner-loop `build_kernel`, `verify_correctness`, and `benchmark_micro` steps use `updated_task_ref`, ensuring that kernel code, test code, harness, and benchmark spec remain bound to the same inner-trial version.

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

## 7. Skills and Assets

### 7.1 Required skills

| Skill | Purpose | Initial source |
|---|---|---|
| `kf-task-authoring` | Create robust Kernel Foundry tasks and tests. | `kernelfoundry.templates` + kernel-eval tasks. |
| `intel-gpu-profile` | unitrace/Intel GPU profiling diagnosis. | `UnitraceProfilerFeedback` + oneAPI docs. |
| `sycl-ocl-optimization` | SYCL/OpenCL idioms and anti-patterns. | `languages.py`, `optimization_aware.py`, existing kernels. |
| `esimd-xmx-dpas` | Intel ESIMD, XMX, DPAS patterns. | KF optimization prompts + curated examples. |
| `runtime-integration` | OpenVINO/PyTorch-XPU/oneDNN integration. | New docs/tooling. |
| `search-strategies` | Greedy/beam/MAP-Elites/hyperparam tuning policies. | `qd_gradient.py`, `map_elites_patterns.py`. |
| `validation-hardening` | Anti-gaming and hidden-shape validation. | New docs/tooling. |

### 7.2 Seed databases

Seed data should include:

- known-good kernels by op signature,
- task templates,
- profiler diagnosis examples,
- failed candidate patterns,
- compile error fixes,
- architecture-specific tuning constants,
- shape distributions from real workloads.

### 7.3 Prompt caching strategy

Static content should be cached:

- system instructions,
- hardware specs,
- language idioms,
- validation rules,
- runtime integration guides.

Dynamic prompts should be concise:

- current candidate diff,
- exact failure summary,
- top 3 profiler facts,
- selected strategy instruction.

### 7.4 Skills, Tools, and KF Collaboration

Skills and Tools must have a clear boundary: **Skills provide knowledge and decision hints, Tools provide execution and factual validation, and Kernel Foundry carries state and results**.

Tool Adapters are the stable contract layer between the agent and Kernel Foundry. They are not a new kernel generator and not a replacement for KF execution; their job is to convert the outer agent’s plan, budget, candidate selection, and state transitions into tool calls that Kernel Foundry can execute, record, and reproduce.

The relationship can be summarized as:

```text
Agent reasoning layer
  -> Tool Adapters
  -> Kernel Foundry native modules
  -> structured evidence
  -> Agent outer-loop decision
```

Tool Adapter responsibilities:

| Responsibility | Corresponding KF capability | Design requirement |
|---|---|---|
| Parameter structuring | `CustomTask`, config, template, task workspace | Convert `TaskContract`, planner output, and skill hints into KF-consumable inputs. |
| Inner-loop execution wrapping | `CustomTaskController`, `TaskRunner`, `CustomTaskEvaluator` | Call only the KF inner evolution loop; do not mix E2E into the inner loop. |
| Dynamic test updates | code-block parser, task workspace, evaluator | The LLM only submits proposals; the KF tool applies, validates, versions, and rolls back patches. |
| Result normalization | build/test/benchmark/profile logs | Convert KF outputs into a unified schema for orchestration and reporting. |
| Candidate materialization | KF run artifacts, kernel refs, DB refs | Build outer `Candidate` records from inner trials/finalists while preserving lineage. |
| Evidence write-back | KF DB, RAG, run workspace, artifacts | All facts must be traceable, reproducible, and auditable. |
| Cost control | queue, budget, profile policy, E2E policy | Enforce inner budget, profile budget, and E2E budget. |
| Error isolation | rollback token, workspace snapshot, structured errors | Compile errors, test failures, timeouts, and environment errors must return structured status. |

Tool Adapters should:

- convert agent plans into KF task/config/test/profile/evolution parameters;
- wrap existing KF evaluator/profiler/database/task-runner capabilities rather than re-implementing them;
- enforce the two-level boundary: KF inner loop only does build/correctness/micro/profile, while E2E only happens outside;
- normalize KF facts into `InnerTrial`, `BranchBest`, `E2EFinalist`, `Candidate`, and report artifacts;
- record prompt/config/seed/candidate hash/hardware/driver/commit metadata for reproducibility;
- provide recoverable state such as rollback tokens, last correct candidate, and rejected reasons.

Tool Adapters should not:

- replace `CustomTaskController` or create a hidden, non-reproducible inner loop;
- let skill hints or subjective agent judgment override measured KF results;
- modify KF workspace/DB/tests implicitly without recording a diff/version;
- promote an inner trial that has not been materialized and E2E-validated;
- allow `benchmark_e2e` to be called inside `run_custom_task_evolution`.

Collaboration workflow:

1. The orchestrator picks a skill based on the `TaskContract` and current candidate state.
2. The skill supplies static knowledge, optimization patterns, anti-patterns, profiler interpretation rules, or task-authoring guidance.
3. The orchestrator converts skill advice into structured tool calls such as `build_kernel`, `verify_correctness`, `profile_kernel`, or `update_task_tests`.
4. The tool calls invoke Kernel Foundry `CustomTask`, `TaskRunner`, `CustomTaskEvaluator`, profiler feedback, and the database.
5. Kernel Foundry returns structured facts: compilation result, correctness result, benchmark statistics, profile diagnosis, and artifact references.
6. The orchestrator updates the inner-trial or outer-candidate state and decides whether to continue inner search, materialize a candidate, run E2E, promote, or start a new custom task round.

| Collaborator | Input | Output | Not allowed to do |
|---|---|---|---|
| Skill | task summary, profile summary, failure summary | suggestions, rules, checklists, strategy hints | directly judging correctness or speedup |
| Tool | structured parameters, candidate/task refs | KF execution results, artifact refs, error summaries | bypassing KF with a hidden execution path |
| Kernel Foundry | task workspace, candidate code, config | build/test/profile/database/report artifacts | being replaced or bypassed by workflow code |
| Orchestrator | contract, tool results, skill hints, budget | next action, inner/outer decision, report | promoting candidates based on subjective judgment |

This collaboration model ensures that domain knowledge can evolve flexibly while factual judgment still comes from the deterministic Kernel Foundry execution chain.

## 8. Search Strategy Design

### 8.1 Default: profile-guided greedy refine

Use when:

- a correct baseline exists,
- profiler identifies a clear bottleneck,
- a small number of transformations are likely.

Loop:

1. make one targeted change,
2. build,
3. verify,
4. benchmark,
5. keep if better,
6. otherwise revert or repair.

### 8.2 Repair mode

Use when:

- compile failed,
- runtime crashed,
- correctness failed.

Inputs should be compact error summaries, not full logs.

### 8.3 Hyperparameter tuning

Use when code structure is good and parameters dominate:

- tile size,
- block size,
- subgroup size,
- unroll factors,
- vector width,
- local memory padding,
- GWS/LWS/JIT constants.

This can be deterministic grid/random/Bayesian tuning without LLM code generation every time.

### 8.4 Beam search

Use when multiple approaches are plausible:

- SLM tiling vs subgroup-only,
- vectorized global load vs local memory caching,
- one-pass vs multi-pass algorithm,
- OpenCL C vs SYCL/ESIMD rewrite.

### 8.5 MAP-Elites limitations and modification

The typical problem with MAP-Elites in Kernel Foundry is that the descriptors are too coarse (for example, memory/compute/parallelism/ESIMD as large buckets). As a result, candidates often concentrate in a tiny and rough part of the space, the exploration step size does not match the true neighborhood around the optimum, and the evolution trajectory may drift away from the best region for a long time.

While reusing the existing MAP-Elites machinery, we should make two changes:

1. upgrade from coarse tags to fine-grained, measurable descriptors;
2. upgrade from one-shot full-dimensional expansion to staged dimension activation plus fast convergence.

Fine-grained descriptors may include:

- memory access: global load/store coalescing, L2 hit rate, SLM read/write ratio, bank conflict metrics;
- compute: FMA/TensorCore utilization, instruction mix, pipeline stall decomposition;
- parallelism: occupancy, subgroup activity, warp/wavefront divergence;
- compilation: register pressure, spill count, key loop unroll factor;
- runtime: shape bucket, dtype, backend-specific launch parameters.

Only when greedy/beam/hyperparameter search stagnates locally should this improved MAP-Elites branch be enabled.

### 8.6 Fast convergence under multi-dimensional search

High-dimensional search increases complexity quickly. To avoid “more dimensions means slower search”, use a staged convergence pipeline:

1. **Stage A: coarse filtering (low cost)**
  - Use a static cost model plus light benchmarks on a small sample to filter obvious losers.
  - Use Successive Halving / Hyperband to early-stop bad candidates.

2. **Stage B: surrogate ranking (medium cost)**
  - Train an online surrogate (e.g. GBDT or lightweight regression) from historical `candidates.jsonl` to predict speedup and failure probability.
  - Keep only Top-K candidates for real profiling.

3. **Stage C: trust-region local refinement (high yield)**
  - Search around the current Pareto frontier with small mutations.
  - Use Bayesian/CMA-ES for continuous parameters and bandit/UCB scheduling for discrete parameters.

4. **Stage D: adaptive dimension activation**
  - Activate only the top 2–4 most influential dimensions first;
  - add more dimensions only when marginal gains fall below threshold;
  - freeze low-contribution dimensions to reduce oscillation.

5. **Stage E: convergence test and rollback**
  - If the improvement for `n` consecutive rounds is below `epsilon`, trigger early stop;
  - roll back to the last candidate that was both correct and stable, to avoid noise overfitting.

The goal is to replace blind full-space expansion with budget-controlled staged convergence, so the search reaches the best neighborhood faster without losing exploration ability.

### 8.7 Fallback policy

If repeated trials fail:

- query memory for a known-good seed,
- lower the optimization ambition,
- return the last correct candidate,
- ask clarifying questions only when truly blocked.

### 8.8 Time cost optimization strategy

The main time costs in kernel optimization come from build, correctness tests, benchmark, full profiling, E2E integration, and LLM generation. The design should explicitly manage these costs.

Recommended cost model:

| Stage | Cost level | Optimization |
|---|---|---|
| Candidate generation | medium | Use delta prompts, cache static skills, change one main direction at a time. |
| Build | medium | Reuse compile cache, incremental builds, skip repeated builds for the same hash. |
| Correctness | medium/high | Smoke tests first, then public tests, then hidden/random tests. |
| Microbenchmark | medium | Warm up with small samples and early reject; only finalists get stable statistics. |
| Full profile | high | Run only on Top-K or suspicious candidates; prefer summary profiling. |
| E2E benchmark | very high | Run only for outer materialized candidates/finalists; never inside the KF inner trial loop. |

Budget should be recorded in `TaskContract`:

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

Scheduling strategy:

- **cheap first, expensive later**: compile/smoke/correctness failures do not go to benchmark;
- **summary before full**: profile with summary first, full/source-level only for top-K;
- **inner/outer separation**: inner loop only does correctness/perf/micro/profile; outer loop does integration and E2E;
- **micro before E2E**: only a best kernel from one inner loop can be materialized as an E2E candidate;
- **cache everything cacheable**: candidate hash, build artifact, test result, profile summary, surrogate feature;
- **async expensive tasks**: full profile and E2E can be queued to avoid blocking the agent;
- **noise control**: do not promote tiny speedups below threshold.

## 9. Reward and Validation Design

### 9.1 Reward components

A candidate score should be computed only after correctness passes.

Suggested score:

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

Where `correctness_gate` is 0 if correctness fails.

### 9.2 Anti-reward-hacking checks

Required checks:

1. Randomized inputs every run.
2. Hidden shapes and non-power-of-two dimensions.
3. Output buffer poisoning before kernel execution.
4. Reference and custom output buffer isolation.
5. Multiple dtypes if task supports them.
6. Tolerance-based numeric comparisons with max error recorded.
7. Profiler sanity: memory/compute activity must be plausible.
8. E2E output equivalence for runtime-integrated kernels.
9. No candidate can read expected outputs from previous runs.

### 9.3 Promotion gates

A final candidate must satisfy:

- build success,
- correctness success,
- hidden/random validation success,
- no anti-gaming violation,
- statistically meaningful performance improvement or explicit accepted tradeoff,
- all artifacts persisted,
- reproducibility instructions present.

## 10. Evidence and Report Format

Each run workspace should contain:

```text
run-workspace/
  docs/
    draft.md
    plan.md
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

One JSON object per candidate:

```json
{
  "candidate_id": "c0007",
  "parent_id": "c0004",
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

The final report should include:

- requirement summary,
- task contract,
- baseline metrics,
- candidate timeline,
- best candidate code reference,
- correctness evidence,
- benchmark evidence,
- profiler evidence,
- E2E evidence if applicable,
- tradeoffs/risks,
- reproduction instructions,
- recommended next steps.

## 11. How Existing Kernel Foundry Components Are Reused

| Component | Reuse plan |
|---|---|
| `CustomTask` | Task representation and code-block update mechanism. |
| `CustomTaskController` | Initial generation loop reference; later split into tools + orchestrator. |
| `CustomTaskEvaluator` | Build/test/profile result conversion. |
| `TaskRunner` | Local/Celery/container execution backend. |
| `PromptConstructor` | RAG/template helper, but wrapped by a smaller agent prompt layer. |
| `InferenceServer` | Optional model access layer for non-Copilot workflows. |
| `UnitraceProfilerFeedback` | Intel profiling diagnosis engine. |
| `NCUProfilerFeedback` | CUDA profiling diagnosis engine. |
| `database/tables.py` | Persistent kernel/task/job/RAG memory. Extend for E2E metadata. |
| `optimization_aware.py` | Source for backend-specific skill content and strategy hints. |
| `map_elites_patterns.py` | Behavior profile extraction for optional MAP-Elites. |
| `qd_gradient.py` | Transition learning and gradient-inspired mutation hints. |
| `mcp/server.py` | Tool-serving foundation; expand beyond `build_and_test`. |
| `kernelfoundry.templates` | Task synthesis templates. |
| `kernelfoundry.kernel-eval` | Regression and benchmark suite. |

## 12. Phased Implementation Plan

### Phase 0: Decisions

Choose:

- first runtime/backend,
- first hardware target,
- first workload/task,
- local vs queue execution mode.

Recommended:

- First backend: Kernel Foundry task mode or OpenCL task mode.
- First hardware: currently available Intel GPU target from existing configs (`lnl`/`b580`).
- First task: `ov_ocl/gated_delta_net` or a small matrix/reduction template.
- First execution: local deterministic mode if hardware is available; queue mode later.

### Phase 1: Tool API extraction

Deliverables:

- `load_task`,
- `build_kernel`,
- `verify_correctness`,
- `benchmark_micro`,
- `profile_kernel`,
- `record_candidate`.

Acceptance:

- A hand-written candidate can be built, tested, benchmarked, profiled, and recorded through tool APIs.

### Phase 2: KF run workspace

Deliverables:

- automatic `docs/draft.md`,
- automatic `docs/plan.md`,
- `candidates.jsonl`,
- `benchmark.csv`,
- `outputs/final_report.md`.

Acceptance:

- Another engineer can reconstruct the optimization process from artifacts.

### Phase 3: Minimal agent loop

Deliverables:

- NL requirement → task contract,
- greedy refine loop,
- compile/correctness/perf feedback repair,
- final promotion report.

Acceptance:

- On one Kernel Foundry task, the agent can run at least several candidates and select the best correct one.

### Phase 4: Validation hardening

Deliverables:

- randomized/hidden shape policies,
- output poisoning,
- reference/custom isolation,
- anti-gaming summary.

Acceptance:

- Candidates that overfit public fixed shapes are rejected.

### Phase 5: Profiling-guided optimization

Deliverables:

- structured unitrace/NCU diagnosis,
- bound-to-strategy mapping,
- ranked next actions.

Acceptance:

- The agent can explain why it chose the next optimization based on profiler evidence.

### Phase 6: Memory and search strategies

Deliverables:

- warm-start from DB/RAG,
- strategy statistics,
- optional MAP-Elites integration,
- hyperparameter tuning mode.

Acceptance:

- Repeated tasks improve faster or use fewer tokens/trials than cold start.

### Phase 7: E2E runtime integration

Deliverables:

- one selected runtime integration backend,
- rollback support,
- E2E benchmark and output equivalence.

Acceptance:

- A candidate can be promoted based on real workload improvement, not only microbenchmark speedup.

## 13. Example Use Cases

### 13.1 OpenCL kernel optimization

Input:

> Optimize `gated_delta_net_ocl` for LNL. Keep fp16 error under 1e-3 and improve mean latency.

Flow:

1. Load existing Kernel Foundry task.
2. Run baseline correctness/perf.
3. Profile reference/custom baseline.
4. Generate candidate with OpenCL subgroup/vector/local-memory changes.
5. Build/verify/benchmark.
6. Keep best candidate and report speedup.

### 13.2 SYCL matrix multiplication optimization

Input:

> Generate a SYCL matmul kernel for BMG with BF16 inputs and FP32 accumulation.

Flow:

1. Create task from PyTorch→SYCL template.
2. Use Intel GPU hardware/SYCL skills.
3. Try baseline tiled SLM implementation.
4. Explore sub-group/joint_matrix/ESIMD strategies if supported.
5. Verify across shape set and benchmark.

### 13.3 Runtime-level OpenVINO optimization

Input:

> Find the slowest operation in this OpenVINO model on BMG and optimize it end-to-end.

Flow:

1. Profile workload.
2. Select top hotspot.
3. Generate Kernel Foundry task for that op.
4. Optimize candidate.
5. Integrate via OpenVINO extension.
6. Benchmark E2E and validate output equivalence.

## 14. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Candidate passes static tests but is wrong generally | High | Hidden/random shapes, output poisoning, E2E checks. |
| Evaluation too slow | High | Tiered evaluation and profiler only for finalists. |
| Token cost too high | Medium/High | Prompt caching, delta prompts, structured summaries, warm-start. |
| Runtime integration is backend-specific | High | Start with one backend and define rollback interface. |
| MAP-Elites wastes budget | Medium | Make it optional and trigger only on stagnation/broad search. |
| Profiler metrics are noisy | Medium | Repeat trials, use p50/p95, require statistically meaningful changes. |
| Agent makes unsupported hardware assumptions | Medium | Hardware skill + tool-reported worker info + compiler/profiler truth. |

## 15. Open Decisions

1. First production backend: OpenVINO, PyTorch-XPU, oneDNN, or Kernel Foundry task-only?
2. First target hardware: LNL, BMG/B580, Arc A770, PTL?
3. First workload/task: gated delta net, matmul, reduction, softmax, or model layer?
4. Evaluation mode: local only first, or Celery/queue from day one?
5. Database extension: alter existing `Kernel` table or add separate `AgentCandidate`/`E2EResult` tables?
6. Skill packaging: separate repo skills or colocated under `skills/`?

## 16. Recommended First Milestone

Build a task-only PoC before E2E runtime integration.

Target:

- Existing task: `kernelfoundry.kernel-eval/tasks/ov_ocl/gated_delta_net`.
- Backend: Kernel Foundry task evaluator.
- Language: OpenCL.
- Hardware: use task config target (`lnl`) or available local Intel GPU.
- Strategy: greedy refine + repair.
- Evidence: `candidates.jsonl`, `benchmark.csv`, `final_report.md`.

Success criteria:

- Agent reads a natural-language requirement.
- Agent loads task and writes a plan.
- Agent generates at least one candidate.
- Agent uses Kernel Foundry build/verify/benchmark.
- Agent records every candidate.
- Agent promotes only a correct candidate.
- Final report includes metrics and reproducibility information.

## 17. Conclusion

The proposed Kernel Optimization Agent should combine the best parts of the two layers that matter most:

- Kernel Foundry contributes the executable substrate: task abstraction, build/test/profile, RAG, database, templates, and search infrastructure.
- The outer agent contributes the disciplined workflow around task contracts, planning, candidate lineage, evidence records, and promotion gates.

The result is a closed-loop kernel optimization agent that is more efficient than blind generation, safer than performance-only reward loops, and extensible enough to support Intel-first kernel optimization with future CUDA/Triton compatibility.
