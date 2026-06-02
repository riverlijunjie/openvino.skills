# Kernel Optimization Agent Framework Design

> Date: 2026-06-01  
> Version: v0.2  
> Scope: design a Kernel Foundry-based kernel optimization agent that reuses Kernel Design Agents workflow principles and Kernel Foundry execution infrastructure.  
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

## 2. Design Goals

### 2.1 Functional goals

- Natural-language triggered optimization.
- Closed-loop kernel generation, testing, profiling, benchmarking, and refinement.
- Support for multiple kernel languages: SYCL, OpenCL, CUDA, Triton, with Intel SYCL/OpenCL prioritized.
- Reuse Kernel Foundry task templates, evaluator, profiler feedback, RAG, and database.
- Reuse KDA’s task contract, planning, candidate tracking, evidence records, and promotion discipline.
- Provide E2E evaluation interfaces for runtime/workload-level validation.
- Support a library of tools, skills, and seed databases.

### 2.2 Quality goals

- Correctness before performance.
- Evidence-backed performance claims.
- Robust anti-reward-hacking validation.
- Lower token usage than blind batch search.
- Lower evaluation cost through staged/tiered testing.
- Extensible backends, search strategies, and hardware knowledge.
- Reproducible reports and candidate lineage.

### 2.3 Non-goals for the first version

- Do not train a new kernel LLM initially.
- Do not support every runtime backend in the first milestone.
- Do not replace Kernel Foundry’s existing batch experiments; wrap and reuse them.
- Do not make MAP-Elites the mandatory main loop.

## 3. Architecture Overview

The framework has five layers:

1. User interface and requirement parser.
2. Orchestrator agent.
3. Deterministic tools/environment.
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
| Kernel Optimization Orchestrator                  |
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
| Tools        |                 | Skills        |
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

## 4. Core Concepts

### 4.1 Task contract

Borrowed from KDA, every optimization run starts with a task contract.

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
  -> LOAD_OR_CREATE_TASK
  -> BASELINE_PROFILE
  -> PLAN
  -> SELECT_STRATEGY
  -> GENERATE_CANDIDATE
  -> BUILD
  -> VERIFY
  -> BENCHMARK_MICRO
  -> PROFILE_CANDIDATE optional
  -> INTEGRATE optional
  -> BENCHMARK_E2E optional
  -> RECORD
  -> DECIDE
       -> done: REPORT
       -> continue: SELECT_STRATEGY
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

### 5.2 `LOAD_OR_CREATE_TASK`

Input: `TaskContract`.

Output: Kernel Foundry `CustomTask` or generated task workspace.

Responsibilities:

- load an existing `task.py/config.yaml` task,
- or synthesize from `kernelfoundry.templates`,
- validate task structure,
- extract `REFERENCE` and `EVOLVE` blocks,
- derive shape and correctness policy.

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

### 5.4 `PLAN`

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

### 5.6 `GENERATE_CANDIDATE`

Inputs:

- task contract,
- current candidate or baseline,
- concise feedback,
- relevant skill snippets,
- RAG examples or seed kernels,
- selected strategy.

Output:

- modified `EVOLVE` block,
- candidate metadata,
- expected optimization profile.

Generation should prefer delta prompts and targeted changes over huge full-context prompts.

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

### 8.5 MAP-Elites as optional exploration

Reuse Kernel Foundry’s optimization profile dimensions:

- memory optimization,
- compute optimization,
- parallelism optimization,
- ESIMD optimization.

Use MAP-Elites only when:

- greedy search stagnates,
- broad diversity is useful,
- there is enough budget,
- behavior descriptors align with the task.

Enhancement:

- descriptors should be dynamically selected by the agent when needed, not always fixed.

### 8.6 Fallback policy

If repeated trials fail:

- query memory for known-good seed,
- reduce optimization ambition,
- return to last correct candidate,
- ask for clarifying constraints only if genuinely blocked.

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

### Phase 2: KDA-style run workspace

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

The proposed Kernel Optimization Agent should combine the best parts of both systems:

- KDA contributes the disciplined agent workflow: task contract, planning, candidate lineage, evidence records, and promotion gates.
- Kernel Foundry contributes the executable substrate: task abstraction, build/test/profile, RAG, database, templates, and search infrastructure.

The result is a closed-loop kernel optimization agent that is more efficient than blind generation, safer than performance-only reward loops, and extensible enough to support Intel-first kernel optimization with future CUDA/Triton compatibility.
