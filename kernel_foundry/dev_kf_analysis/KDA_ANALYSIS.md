# Kernel Design Agents Framework Analysis

> Date: 2026-06-01  
> Scope: `/mnt/river/kernel_foundry/kda/kernel-design-agents`  
> Goal: understand the Kernel Design Agents workflow, identify reusable ideas, gaps, and required adaptations for a Kernel Foundry-based kernel optimization agent.

## Executive Summary

Kernel Design Agents (KDA) is intentionally lightweight. It is not a full kernel generation system; it is a reusable agent workflow reference for performance-sensitive implementation tasks. Its core contribution is process design: define a task contract, let an agent inspect the workspace, draft a plan, implement one candidate at a time, validate, measure, record evidence, and promote only with proof.

This philosophy complements Kernel Foundry very well:

- Kernel Foundry provides deterministic kernel build/test/profile infrastructure.
- KDA provides an evidence-driven agent loop and documentation discipline.
- KDA skills provide domain-specific knowledge assets, especially CUDA/Blackwell profiling and kernel optimization references.

The main limitation is that KDA is currently task-agnostic and mostly documentation/prompt/skill oriented. It lacks a concrete environment API, reward system, training loop, candidate database, and runtime integration layer. For a Kernel Foundry kernel optimization agent, KDA should be reused as the workflow shell, while Kernel Foundry should supply the executable tools and task environment.

## Repository Components

| Path | Purpose |
|---|---|
| `README.md` | High-level KDA description and minimal workflow. |
| `CLAUDE.md` | Repository-facing agent rules and expected workflow. |
| `docs/agent-flow.md` | Task contract, minimal loop, evidence record, promotion rule. |
| `prompts/basic-flow.md` | Generic starter prompt template for agent-driven implementation. |
| `prompts/README.md` | Prompt usage guidance. |
| `skills/KernelWiki` | Structured Blackwell/Hopper kernel optimization knowledge base. |
| `skills/ncu-report-skill` | Nsight Compute profiling workflow and analysis skill for B200/sm100. |

The repository deliberately avoids storing downstream task-specific code, datasets, benchmark logs, or private acceptance thresholds.

## Current Workflow Model

KDA’s minimal loop is:

1. Create or enter a separate task implementation workspace.
2. Define the task contract:
   - objective,
   - inputs and outputs,
   - correctness requirements,
   - constraints,
   - validation command,
   - evaluation command,
   - promotion criteria.
3. Let the agent inspect code/docs/tests.
4. Ask the agent to write `docs/draft.md`.
5. Convert the draft into `docs/plan.md` or an executable plan.
6. Implement one candidate at a time.
7. Validate correctness after each meaningful change.
8. Measure performance when applicable.
9. Record candidate relationships, benchmark evidence, profiling evidence, and promotion decisions.
10. Repeat until promotion criteria are met or blockers are explicit.

This is a strong workflow for coding agents because it reduces undisciplined trial-and-error and requires evidence before promotion.

## Architecture Characteristics

### 1. Workflow-first, not framework-heavy

KDA avoids hardcoding a benchmark harness or hardware target. The task workspace owns code, tests, datasets, validators, profiling data, and outputs. This makes KDA portable, but it also means downstream systems must provide executable environment tools.

### 2. Clear separation of reusable workflow and task-specific artifacts

`CLAUDE.md` explicitly says:

- keep task-specific prompts, datasets, validators, generated implementations, benchmark logs, and candidates out of the generic repo;
- put generated outputs in `runs/`, `outputs/`, or `profile/`;
- keep the repository focused on reusable flow mechanics.

This maps well to Kernel Foundry, where generated kernels and results should live in `runs/`, database records, and task workspaces rather than polluting framework code.

### 3. Evidence records as first-class outputs

KDA recommends:

- `docs/draft.md`,
- `docs/plan.md`,
- `benchmark.csv`,
- `candidates.jsonl`,
- `profile/`,
- `runs/` or `outputs/`.

These records are important for kernel optimization because performance claims can be fragile. A future engineer must know which candidate was tested, with what shape, on what hardware, and why it was accepted or rejected.

### 4. Promotion requires proof

The promotion rule is simple but important: promote a candidate only when it satisfies the task contract and has evidence that it improves or preserves the target metric. Rejected candidates should be recorded with reasons.

This rule should become a hard gate in the new optimization agent.

## Skills Analysis

### `ncu-report-skill`

This skill is a detailed profiling playbook for CUDA kernels on B200/sm100. It enforces:

- profile before diagnosing,
- create one run directory per profiling run,
- use standalone harnesses when needed,
- collect full and source-level NCU reports,
- parse reports with Python APIs rather than eyeballing CLI output,
- analyze occupancy, balance, stalls, tensor core usage, timeline, and memory,
- match observations to a diagnosis playbook,
- write an evidence-backed report.

Reusable ideas for Kernel Foundry:

- “Profile → Diagnose → Plan” should be the default optimization discipline.
- Profiler reports should produce structured evidence and a ranked action list.
- Each run should isolate harness, reports, analysis, and final report.
- Recommendations must cite concrete metrics, not vague claims.

Limitations for Kernel Foundry:

- It is CUDA/B200-specific.
- Kernel Foundry’s Intel focus needs an equivalent unitrace/VTune/oneAPI profiling skill.
- The skill is not an executable API; it is a workflow and knowledge asset.

### `KernelWiki`

This is a structured kernel optimization wiki focused on NVIDIA Blackwell/Hopper. It contains:

- hardware feature pages,
- kernel technique pages,
- merged PR references,
- blog/doc summaries,
- query scripts,
- candidate ledgers,
- confidence and provenance rules.

Reusable ideas:

- Build a searchable, versioned optimization knowledge base.
- Attach claims to confidence levels and sources.
- Index knowledge by symptom, technique, hardware feature, kernel type, language, and repo.
- Use concrete PR/code references as few-shot or design evidence.

Limitations for Kernel Foundry:

- Current scope is NVIDIA Blackwell/Hopper, not Intel GPU/SYCL/OpenCL.
- For Kernel Foundry, a parallel `IntelKernelWiki` or `IntelGpuOptimizationWiki` is needed.
- Its query scripts are local, not integrated into Kernel Foundry’s planner/tool API.

## Strengths

### 1. Excellent process discipline

KDA formalizes the parts coding agents often skip:

- task contract,
- plan draft,
- executable plan,
- candidate tracking,
- validation after each change,
- evidence-backed promotion.

For kernel work, this discipline is crucial because “fast but wrong” kernels are common.

### 2. Task-agnostic and reusable

Because KDA avoids binding itself to one harness, it can wrap Kernel Foundry, CUDA contests, compiler passes, runtime integrations, or infrastructure work.

### 3. Skill-based domain knowledge

KDA treats domain knowledge as optional skills. This aligns with a kernel optimization agent design where Intel GPU hardware, OpenCL/SYCL idioms, profiling diagnosis, and runtime integration should be modular assets.

### 4. Human-readable artifacts

The recommended `draft.md`, `plan.md`, `benchmark.csv`, and `candidates.jsonl` make the optimization process inspectable. That is valuable for debugging and review.

### 5. Candidate lineage

KDA asks agents to record candidate parent relationships and outcomes. Kernel search benefits from this because many optimizations are incremental mutations or fallback repairs.

## Gaps for Kernel Optimization

### 1. No concrete environment API

KDA does not define executable interfaces like:

- build candidate,
- run correctness tests,
- run performance tests,
- profile candidate,
- integrate candidate into runtime,
- rollback candidate.

Kernel Foundry can fill this gap.

### 2. No reward system beyond promotion criteria

KDA uses human-readable promotion rules, but does not define reward shaping, score normalization, multi-objective tradeoffs, or anti-gaming safeguards.

For Kernel Foundry, reward must combine:

- compile success,
- correctness,
- micro performance,
- E2E performance,
- robustness across shapes,
- code maintainability/safety,
- budget cost.

### 3. No automated training or learning loop

The user request mentions learning from environment feedback. KDA itself is not an RL/training framework; it is an agentic implementation loop. A Kernel Optimization Agent can add learning through memory, strategy statistics, and candidate databases rather than model fine-tuning initially.

### 4. No built-in task synthesis

KDA assumes the task contract is provided. Kernel Foundry users need natural-language requirement parsing and task/template generation.

### 5. No E2E workload integration

KDA can be applied to E2E work, but it does not provide runtime-specific integration mechanisms. For true workload optimization, Kernel Foundry must add backend-specific integration tools.

### 6. NVIDIA-heavy skills

Current KDA domain assets are strongest for CUDA/B200. Kernel Foundry needs Intel-first skills:

- Xe architecture guide,
- BMG/LNL/PTL hardware facts,
- SYCL 2020 idioms,
- OpenCL subgroup patterns,
- ESIMD/DPAS/XMX usage,
- unitrace metric interpretation,
- OpenVINO/oneDNN/PyTorch-XPU integration.

### 7. Manual evidence recording

KDA recommends evidence files but does not enforce schema or automatic writes. Kernel Foundry should generate `candidates.jsonl`, benchmark tables, and reports automatically from structured evaluation outputs.

## Recommended Adaptation for Kernel Foundry

### 1. Use KDA as the outer agent process

Adopt these mandatory stages:

1. task contract generation,
2. workspace inspection,
3. plan draft,
4. executable plan,
5. candidate implementation,
6. correctness validation,
7. performance measurement,
8. evidence recording,
9. promotion/rejection decision.

### 2. Replace manual commands with Kernel Foundry tools

KDA’s “validation command” and “evaluation command” should map to Kernel Foundry tool calls:

- `build_kernel`,
- `verify_correctness`,
- `benchmark_micro`,
- `profile_kernel`,
- `benchmark_e2e`,
- `integrate_runtime`.

### 3. Standardize evidence schema

Use KDA’s files but make them machine-readable:

- `docs/draft.md`: initial reasoning and risks.
- `docs/plan.md`: executable plan and budgets.
- `candidates.jsonl`: one row per candidate with parent, strategy, status, metrics, artifacts.
- `benchmark.csv`: normalized performance stats.
- `profile/<candidate>/REPORT.md`: profiler evidence and diagnosis.
- `outputs/final_report.md`: final promoted kernel summary.

### 4. Build Intel-specific skills

Create Kernel Foundry skills mirroring KDA’s NVIDIA skills:

- `intel-gpu-profile-skill`: unitrace/VTune workflow, metric names, common bottlenecks.
- `intel-kernel-wiki`: SYCL/OpenCL/ESIMD/DPAS/XMX examples and PR/code references.
- `kf-task-authoring-skill`: how to generate robust Kernel Foundry tasks.
- `runtime-integration-skill`: OpenVINO/PyTorch-XPU/oneDNN integration recipes.

### 5. Enforce promotion gates

A candidate should not be promoted unless all gates pass:

- compile success,
- correctness across public and hidden/randomized tests,
- no anti-gaming violation,
- microbenchmark improvement or no regression,
- E2E improvement or explicit accepted tradeoff,
- reproducible evidence written to disk.

### 6. Preserve KDA’s separation of concerns

Keep generic agent workflow docs separate from downstream task workspaces. Generated kernels, logs, datasets, and profiles should live in task-specific workspaces or Kernel Foundry run directories.

## Proposed Agent State Model Inspired by KDA

| State | Description | Kernel Foundry mapping |
|---|---|---|
| `TASK_CONTRACT` | Parse user intent into objective/constraints/metrics | NL→TaskSpec planner |
| `INSPECT` | Read task/runtime/harness context | workspace + template inspection |
| `DRAFT_PLAN` | Create initial optimization plan | `docs/draft.md` |
| `EXEC_PLAN` | Convert to actionable loop | `docs/plan.md` |
| `GENERATE` | Create or mutate candidate | LLM + RAG + skills |
| `BUILD` | Compile candidate | `TaskRunner.build_custom_task` |
| `VERIFY` | Correctness validation | `CustomTaskEvaluator` / pytest |
| `BENCH` | Microbenchmark | `measure_runtime` / pytest performance |
| `PROFILE` | Gather bottleneck evidence | unitrace / ncu |
| `INTEGRATE` | Insert candidate into runtime | new runtime-specific tool |
| `E2E_BENCH` | Measure workload impact | new E2E evaluator |
| `RECORD` | Write evidence and DB record | `candidates.jsonl` + DB |
| `PROMOTE_OR_REJECT` | Gate final candidate | KDA promotion rule |

## Conclusion

KDA is not a replacement for Kernel Foundry. It is the missing agent discipline around Kernel Foundry’s executable infrastructure. The best design is to reuse KDA’s task-contract, planning, candidate, evidence, and promotion workflow while letting Kernel Foundry provide deterministic tools for generation, build, verification, profiling, benchmarking, memory, and future runtime integration.
