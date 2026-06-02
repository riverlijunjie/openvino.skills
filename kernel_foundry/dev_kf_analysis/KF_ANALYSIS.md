# Kernel Foundry Framework Analysis

> Date: 2026-06-01  
> Scope: `/mnt/river/kernel_foundry/kernelfoundry.internal`, `/mnt/river/kernel_foundry/kernelfoundry.kernel-eval`, `/mnt/river/kernel_foundry/kernelfoundry.templates`  
> Goal: understand the current Kernel Foundry architecture, identify bottlenecks/gaps, and propose improvements for a kernel optimization agent.

## Executive Summary

Kernel Foundry is already a rich kernel-generation and evaluation platform. It combines LLM-based code generation, task packaging, build/test/profiling execution, RAG examples, database logging, optional MAP-Elites-style evolution, and benchmark/task templates for SYCL/CUDA/Triton/OpenCL-oriented GPU kernels.

The strongest reusable assets are:

- `CustomTask` task abstraction: annotated `REFERENCE`, `EVOLVE`, and `USER_INSTRUCTIONS` blocks with build/test/profile metadata.
- `CustomTaskController`: iterative LLM generation loop with feedback, evaluation, database logging, and optional evolution.
- `CustomTaskEvaluator` + `TaskRunner`: deterministic build, correctness, performance, and profiler execution pipeline.
- Profiler feedback: unitrace-based Intel GPU analysis, plus NCU support for CUDA.
- RAG and program DB: persistent kernel/result storage and retrieval.
- Templates: practical task patterns for PyTorch→SYCL, SYCL→SYCL, OpenCL→OpenCL, model-layer tasks, and OpenCL PyTorch operations.

However, the current system is still primarily a batch experiment/search harness. It is not yet a user-facing end-to-end optimization agent. Key gaps include high task/config complexity, expensive trial loops, static benchmark orientation, limited runtime integration, weak anti-reward-hacking guarantees, and search policies that are more framework-driven than workload-driven.

## Repository Components

### 1. `kernelfoundry.internal`

This is the main generation and orchestration repo.

Important modules:

| Component | Path | Responsibility |
|---|---|---|
| Main controller | `kernelgen/controller.py` | Prompt construction, LLM calls, parent/inspiration sampling, optimization-aware prompting. |
| Custom task controller | `kernelgen/custom_task_controller.py` | Trial loop for custom tasks: generate, extract, evaluate, log, select best, early stop. |
| Evaluator | `kernelgen/custom_task_evaluator.py` | Converts build/test/profile outputs into `EvalResult` with correctness, runtime, speedup, profiler data. |
| Task abstraction | `kernelgen/custom_task.py` | In-memory task archive, block extraction/update, test names, task config, DB serialization. |
| Task dispatcher | `kernelgen/tasks/task_runner.py` | Local or Celery-based build/test/image execution. |
| Runtime test task | `kernelgen/tasks/test_custom_task.py` | Runs pytest correctness, pytest performance, and profiler passes. |
| LLM server | `kernelgen/inference_server.py` | OpenAI-compatible, AWS Bedrock, GNAI, IBM, local Llama Factory, ensembles. |
| Prompt construction | `kernelgen/prompts/prompt_constructor.py` | Template-based prompt assembly, RAG retrieval, code categorization. |
| Optimization-aware prompting | `kernelgen/prompts/optimization_aware.py` | Backend-aware memory/compute/parallelism/ESIMD guidance. |
| Pattern classifier | `kernelgen/utils/map_elites_patterns.py` | Regex-based behavioral profile detection for SYCL/CUDA/OCL/ESIMD. |
| QD gradient | `kernelgen/qd_gradient.py` | Transition tracking and gradient-like signals for MAP-Elites evolution. |
| Profiler feedback | `kernelgen/profiler_feedback.py` | unitrace and NCU metric collation and feedback text. |
| Database schema | `kernelgen/database/tables.py` | `Kernel`, `Task`, `Job`, `JobLog`, `Rag`, baseline timing tables. |
| MCP tool server | `kernelgen/mcp/server.py` | `build_and_test(folder_path)` wrapper for server-side validation. |
| Core compiler | `kernelfoundry/kernelfoundry/compiler.py` | `TorchCompiler` and `IcpxCompiler` build helpers. |

### 2. `kernelfoundry.kernel-eval`

This is the benchmark/evaluation package for Intel GPU kernel generation tasks.

It defines:

- Task groups: oneDNN, oneDAL, OpenVINO/OpenCL.
- Standard task structure: `task.py`, `conftest.py`, `config.yaml`, `*_kernel.*`, `*_reference.*`, optional build folders/scripts/CMake files.
- Python pytest-based validation and benchmark harnesses.
- Example: `tasks/ov_ocl/gated_delta_net/task.py` uses PyOpenCL, randomized inputs, OpenCL kernel compilation, correctness checks, and `measure_runtime`-based profiling.

The benchmark repository is valuable because it decouples task definition from generation logic. It can be reused as an evaluation backend for a new agent.

### 3. `kernelfoundry.templates`

This repo contains reusable task templates.

Supported use cases:

- PyTorch operation to SYCL kernel.
- SYCL kernel optimization.
- SYCL kernel for a model layer.
- PyTorch operation to OpenCL kernel.
- OpenCL kernel optimization.

The template model is important for onboarding new tasks because it gives developers concrete patterns for `REFERENCE`, `EVOLVE`, test design, and configuration.

## Current End-to-End Flow

The current custom task generation loop is roughly:

1. User supplies Hydra config and a custom task path.
2. `CustomTask.create()` loads task files, extracts blocks, test names, and config.
3. `CustomTaskController` initializes LLM, prompt constructor, feedback helper, optional program database.
4. For each trial:
   - Construct prompt from reference code, previous program/eval feedback, RAG examples, optimization guidance.
   - Query LLM server for one or more candidate implementations.
   - Extract generated code into the `EVOLVE` block.
   - Optionally perform a syntax precheck.
   - Evaluate each candidate with `CustomTaskEvaluator`.
   - Run build, correctness tests, performance tests, and profiler passes through `TaskRunner`.
   - Convert results into `EvalResult` with `compiled`, `correctness`, `perf_score`, `runtime`, `runtime_improvement`, and profiler metadata.
   - Store generated kernel/result/logs in run directory and optional database.
   - Select the best result and use it as parent or add it to the program database.
5. Stop on max iterations, correctness, early stagnation, or failure limits.

This is a robust experimental loop, but it still assumes the task, backend, metrics, and evaluation target are mostly preconfigured.

## Architecture Strengths

### 1. Clear task abstraction

`CustomTask` is a strong boundary. It stores task data in a memory file map, detects build/test/profile steps, extracts `REFERENCE` and `EVOLVE` regions, and can update code blocks without mutating unrelated files. This is exactly the kind of deterministic environment interface an optimization agent needs.

### 2. Deterministic evaluator

`CustomTaskEvaluator` separates:

- extraction errors,
- build failures,
- runtime/test failures,
- correctness failures,
- performance measurement,
- profiler feedback,
- speedup computation.

That structured result is suitable for agent feedback and reward design.

### 3. Multi-language kernel support

Current support includes:

- SYCL,
- CUDA,
- Triton,
- OpenCL.

Build support is narrower (`cuda`, `sycl`, `ocl`), but prompt and profiling logic already knows multiple backends.

### 4. Intel GPU profiling support

`UnitraceProfilerFeedback` extracts runtime, occupancy, memory bandwidth, SLM usage/conflicts, XVE stalls, ALU utilization, and roofline-style bound classification. This is a strong differentiator for Intel GPU-focused kernel optimization.

### 5. RAG and historical database

The `Kernel`, `Task`, `Job`, and `Rag` tables provide a foundation for:

- warm-starting from previous kernels,
- retrieving relevant examples,
- tracking model/config/profiler metadata,
- comparing performance over time.

### 6. Evolution infrastructure

The code already contains:

- behavior dimensions: memory, compute, parallelism, ESIMD,
- regex-based optimization profile detection,
- MAP-Elites-like program database,
- gradient-like transition tracking,
- optimization-aware prompt mutation.

This can be reused as one search strategy inside a broader agent.

### 7. MCP entrypoint

`kernelgen/mcp/server.py` exposes `build_and_test(folder_path)`, which is a useful tool boundary for a coding agent: the agent can generate/edit a task and ask Kernel Foundry to validate it.

## Main Gaps and Bottlenecks

### 1. Task/config complexity

The default workflow depends on many Hydra options:

- `job_name`, `task_name`, `custom_task`, `task_origin`, `language`, `gpu_arch`, `mode`, `evolve_mode`, queue settings, timeouts, profile settings, database settings, prompt settings, container settings.

This is powerful but difficult for a user who simply wants: “optimize this kernel/workload on this GPU.” The new agent should translate natural language into a compact task spec, then generate the Hydra/custom-task details automatically.

### 2. High per-trial latency

A full candidate evaluation can include:

- LLM inference,
- build,
- correctness pytest,
- performance pytest,
- multiple profiler passes,
- optional queue/container overhead.

`TaskRunner.test_custom_task()` multiplies test timeout by profiler/correctness/performance subprocesses. This is reasonable for reliable evaluation but expensive for inner-loop exploration.

The framework needs a tiered evaluator:

1. fast syntax/static checks,
2. minimal correctness smoke tests,
3. microbenchmark on a few representative shapes,
4. full correctness/performance/profile only for promising candidates,
5. E2E workload benchmark only for finalists.

### 3. Token cost and prompt bloat

Prompt construction may include:

- full reference source,
- vector-add examples,
- RAG examples,
- previous program,
- top programs,
- inspirations,
- profiler logs,
- feedback LLM output,
- optimization taxonomy/guidance.

This is useful but can be expensive. The design should promote:

- smaller delta prompts,
- cached static skills,
- structured error summaries instead of raw logs,
- warm-starting from database examples,
- explicit token budgets.

### 4. Batch search orientation

The current controller focuses on candidate generation over configured tasks. It does not yet own the full loop of:

- workload profiling,
- hotspot selection,
- kernel target extraction,
- runtime integration,
- E2E benchmarking,
- rollout/rollback decisions.

For real optimization, the agent must optimize the workload, not only the isolated task.

### 5. MAP-Elites should be a strategy, not the orchestrator

The current evolve mode can sample parents/inspirations from a program database and use behavior descriptors. This is valuable for diversity, but for many kernel tasks a focused profile-driven refine loop is more efficient.

Recommended change: keep MAP-Elites/gradient tracking as an optional `search_strategy` invoked only when:

- the search space is broad,
- greedy refinement stalls,
- multiple kernel families are plausible,
- diversity is valuable for escaping local optima.

### 6. Weak anti-reward-hacking guarantees

The current evaluator delegates correctness to task-defined pytest tests. Some tasks are robust, but the framework itself does not enforce:

- randomized hidden shapes,
- output buffer poisoning,
- separation from previous reference outputs,
- profiler-based “did real work happen?” checks,
- E2E output equivalence after runtime integration.

An optimization agent with reward feedback needs stronger default validation to avoid overfitting or gaming tests.

### 7. Runtime integration is missing as a first-class stage

Kernel Foundry can generate and validate task kernels, but it does not yet provide a general `integrate_runtime` abstraction for:

- OpenVINO custom op/extension,
- PyTorch-XPU custom op,
- oneDNN primitive replacement,
- OpenCL runtime kernel replacement,
- rollback and provenance.

This is the biggest gap for true end-to-end workload optimization.

### 8. Profiling is candidate-level, not workload-level

Profiler feedback is excellent after a candidate runs, but there is no high-level `profile_workload` stage to decide which op/kernel to target first based on E2E contribution.

### 9. Test cases can be static

Template tasks often hardcode shapes in `task.py`. This is useful for benchmarking, but generated kernels can overfit to those shapes. The new agent should support shape distribution specs and dynamic test generation.

## Improvement Recommendations

### P0: Make Kernel Foundry callable as deterministic tools

Define stable tool interfaces over existing components:

- `load_task(path) -> TaskSpec`
- `build_kernel(task, candidate) -> BuildResult`
- `verify_correctness(task, candidate, shape_policy) -> VerificationResult`
- `benchmark_micro(task, candidate, shape_policy) -> BenchmarkResult`
- `profile_kernel(task, candidate) -> ProfileResult`
- `query_kernel_memory(signature, hw) -> candidates`
- `record_candidate(...) -> id`

The MCP server already points in this direction; expand it beyond `build_and_test`.

### P1: Introduce tiered evaluation

Use several evaluation tiers:

| Tier | Purpose | Expected cost |
|---|---|---|
| T0 static/syntax | Catch extraction/compile obvious failures | seconds |
| T1 smoke correctness | One or two small randomized cases | seconds |
| T2 micro correctness/perf | Representative shape set | tens of seconds |
| T3 profiler | unitrace/ncu feedback for finalists | expensive |
| T4 E2E | Real workload comparison | most expensive |

This directly attacks long evaluation time per trial.

### P2: Add natural-language-to-task-spec conversion

A front-end planner should convert user intent into:

- target runtime,
- target hardware,
- source workload or task template,
- correctness tolerance,
- shape distribution,
- metric priority,
- budget,
- allowed languages/backends,
- integration mode.

Hydra config remains internal; users should not need to author it manually for common workflows.

### P3: Harden validation

Add default anti-gaming checks:

- multiple randomized seeds,
- hidden/held-out shapes,
- non-power-of-two dimensions,
- output buffer poisoning,
- reference/custom buffer isolation,
- profiler sanity check for memory/compute activity,
- E2E output equivalence after integration.

### P4: Promote profiler-driven search

Use `UnitraceProfilerFeedback` and `NCUProfilerFeedback` not only as text appended to prompts, but as structured strategy signals:

- memory-bound → coalescing, vector loads, SLM/register tiling, reduce transfers,
- compute-bound → DPAS/XMX/tensor core, ILP, unroll, FMA, algorithmic fusion,
- latency/stall-bound → reduce sync, improve occupancy, persistent/thread mapping changes,
- low occupancy → work-group/sub-group size tuning, register pressure reduction,
- SLM conflicts → padding/layout changes.

### P5: Make search strategies pluggable

Recommended strategy API:

- `greedy_refine`: default low-token serial loop.
- `beam_search`: maintain top-k candidates.
- `evolution`: use existing DB/parent/inspiration sampling.
- `map_elites`: use behavior descriptors for diversity.
- `fallback_repair`: compile/runtime error repair mode.
- `hyperparameter_tune`: tile/work-group/sub-group parameter sweeps.

### P6: Extend database schema for E2E optimization

Add or attach fields for:

- `workload_id`,
- `op_signature`,
- `shape_distribution`,
- `runtime_backend`,
- `integration_mode`,
- `e2e_latency_stats`,
- `micro_latency_stats`,
- `strategy_used`,
- `validation_policy`,
- `promotion_status`.

### P7: Separate knowledge as skills

Extract reusable prompt knowledge into compact skill docs/assets:

- Intel GPU architecture specs,
- SYCL/OpenCL optimization idioms,
- ESIMD/DPAS/XMX patterns,
- profiler diagnosis playbooks,
- runtime integration guides,
- task template authoring guide.

This reduces repeated prompt cost and makes agent behavior more maintainable.

## Reuse Map for a Kernel Optimization Agent

| Existing KF asset | Reuse as | Notes |
|---|---|---|
| `CustomTask` | Environment/task representation | Keep. Add shape policy and NL metadata. |
| `CustomTaskEvaluator` | Evaluation tool backend | Keep. Add tiered policies and anti-gaming. |
| `TaskRunner` | Local/queue executor | Keep. Add explicit scheduling and budget control. |
| `PromptConstructor` | Candidate generation helper | Reuse selectively; prefer smaller agent prompts. |
| `optimization_aware.py` | Strategy/skill source | Extract into skills and structured decision rules. |
| `map_elites_patterns.py` | Behavior descriptor extractor | Keep for optional QD strategy. |
| `qd_gradient.py` | Search telemetry | Keep for optional MAP-Elites/gradient search. |
| `UnitraceProfilerFeedback` | Profile diagnosis tool | Promote to structured tool output. |
| `database/tables.py` | Memory/experiment DB | Extend for E2E and strategy metadata. |
| `mcp/server.py` | Agent tool boundary | Expand to full build/verify/bench/profile tools. |
| `kernelfoundry.templates` | Task bootstrap library | Use for NL-to-task synthesis. |
| `kernelfoundry.kernel-eval/tasks` | Regression/eval suite | Use as benchmark and examples. |

## Prioritized Roadmap

### Phase 1: Tool stabilization

- Expand MCP/API from `build_and_test` to separate `build`, `verify`, `benchmark`, `profile`, `record` tools.
- Return structured JSON, not only eval logs.
- Add deterministic local mode for fast PoC.

### Phase 2: Tiered and hardened evaluation

- Implement smoke/full/hidden shape policies.
- Add output poisoning and reference/custom isolation helpers.
- Add profiler sanity checks.

### Phase 3: Agent loop

- Implement natural-language task parser.
- Add profile→select→generate→build→verify→benchmark→record loop.
- Default to greedy refine with compile/runtime feedback.

### Phase 4: Runtime integration

- Start with PyTorch-XPU or OpenCL task integration for fast PoC.
- Add OpenVINO extension path for production value.
- Add rollback and E2E validation.

### Phase 5: Search and memory

- Add warm-start retrieval from DB/RAG.
- Add MAP-Elites only as optional exploration strategy.
- Track strategy success by op type, bound type, hardware, and shape distribution.

## Conclusion

Kernel Foundry has most of the hard infrastructure needed for kernel optimization: build/test/profile, task templates, RAG, database, and evolution logic. The next step is architectural: wrap these pieces as deterministic tools around an agentic orchestrator that starts from natural-language requirements and real workload profiling, then performs evidence-based closed-loop optimization with strong validation and E2E promotion criteria.
