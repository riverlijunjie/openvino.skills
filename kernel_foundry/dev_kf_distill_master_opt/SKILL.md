---
name: dev_kf_distill_master
description: >-
  Distilled KernelFoundry kernel-optimization method: an AlphaEvolve/FunSearch-style evolutionary
  loop (MAP-Elites archive + island model + QD gradient) driven by optimization-aware LLM
  prompting, with correctness-gated speedup fitness. Use this when asked to optimize or generate a
  GPU kernel (SYCL/CUDA/Triton/OpenCL) by iterative LLM search, to set up or run a KernelFoundry
  task (config.yaml/task.py/EVOLVE block), to reimplement or reason about the
  evolve loop / MAP-Elites cells / optimization-aware prompts / profiler feedback, or to mimic this
  whole algorithm and workflow on a new problem. Also covers profiling a kernel to drive the search —
  with `unitrace` on Intel GPUs (ComputeBasic/MemoryProfile/VectorEngineProfile passes) or NCU on
  NVIDIA — and reading its roofline verdict onto the optimization ladder. Also covers convergence/diversity tuning of the
  search itself — fine-grained or measured MAP-Elites descriptors, staged coarse→fine activation,
  surrogate-assisted ranking, and trust-region local refinement.
---

# dev_kf_distill_master — Distilled KernelFoundry kernel-optimization skill

A faithful distillation of KernelFoundry (`kernelfoundry.internal`,
`kernelfoundry.kernel-eval`, `kernelfoundry.templates`): the algorithm, the workflow, and the
copyable artifacts needed to **mimic the entire method** — either by driving the real pipeline or
by reimplementing it anywhere.

## What this method is (one paragraph)

You evolve a **kernel source file** toward a faster, still-correct version. A population of
candidate kernels lives in a **MAP-Elites archive**: each kernel is mapped — *deterministically
from its code* — to a cell `(memory_opt, compute_opt, parallelism_opt[, esimd_opt])`, each axis an
integer **level 0–3** on an optimization ladder; every cell keeps only its best. Several **islands**
run semi-independent searches with periodic ring **migration** to avoid premature convergence. Each
generation: **sample** a parent (mostly fitness-proportional from elites, some island exploration,
some random) + diverse **inspirations** → pick a **target cell** to aim at (an under-explored region
or a mutation of the parent) → build an **optimization-aware prompt** that demands that cell's
concrete techniques and includes the parent + its profiler/eval feedback → call the LLM (often an
ensemble at temp 0.3) → **extract code → compile → correctness-test → benchmark → profile** →
score and insert. Fitness is a **correctness gate plus a speedup bonus** (`perf_score` 0–5, +
`gmean(ref/custom)`), so any correct kernel beats any incorrect one. An optional **QD gradient**
learns which axes are paying off and steers target-cell selection.

## When to use it

- "Optimize / speed up this GPU kernel", "generate a SYCL/CUDA/Triton/OpenCL kernel for this op",
  "port this PyTorch op to a kernel", "tune this kernel" — via iterative LLM search.
- Set up, configure, or run a KernelFoundry job; author a task; debug why a run isn't improving.
- Reimplement, extend, or explain any piece: the evolve loop, MAP-Elites cells, island/migration,
  optimization-aware prompts, meta-prompting, profiler→feedback, the scoring function.
- Apply the *same method* to a new code-optimization problem (the loop is backend-agnostic).

This is a method/algorithm skill, **not** a hardware tuning cheat-sheet. For Blackwell/Hopper
kernel *techniques* and PR references use the `KernelWiki` skill; for deep NVIDIA Nsight Compute
report analysis use `ncu-report-skill`. This skill is about the *search that orchestrates* them —
including the **profiler step** that closes the loop: on Intel GPUs it profiles with `unitrace`,
on NVIDIA with NCU, and feeds a roofline verdict back into the next prompt (see §"Profiling kernel
performance with `unitrace`" below and [references/evaluation_and_feedback.md §4](references/evaluation_and_feedback.md)).

## How to apply it

**First decide the regime** (it changes everything downstream):
- **Translation / correctness** (no working kernel yet) → single-lineage iterative fix-it:
  `evolve_mode:false`, `branches_per_iteration:1`, `max_iters≈10`, `stop_once_correct:true`,
  one strong low-temp model, feedback-LLM on. Template: [run_translation.yaml](templates/run_translation.yaml).
- **Optimization** (have a correct kernel, want it faster) → full evolutionary search:
  `evolve_mode:true`, `branches_per_iteration:4`, `max_iters≈20–40`, `start_from_best:true`,
  optimization-aware prompting + QD gradient on, model ensemble at temp 0.3.
  Template: [run_optimize_evolve.yaml](templates/run_optimize_evolve.yaml).

**Path A — drive the real KernelFoundry pipeline** (this repo, Intel/NVIDIA GPU + LLM keys):
0. **Ensure the profiler is installed.** On Intel GPUs the pipeline profiles via `unitrace`,
   which it expects already on `PATH` (or `$KERNELFOUNDRY_unitrace_cmd`) — nothing builds it on
   demand. Run `bash scripts/install_unitrace.sh` once: it's idempotent — it no-ops if `unitrace`
   is found, reuses an existing `pti-gpu` build, else clones `https://github.com/intel/pti-gpu.git`,
   builds it (cmake + make), and symlinks the binary into `~/.local/bin`. Ensure that dir is on
   `PATH`. (NVIDIA arches use NCU instead and don't need this.)
1. Author a task: copy [templates/task_template/](templates/task_template/) → fill `config.yaml`
   (task_name, job_name, gpu_arch, language), put the op in the `[REFERENCE]` block of `task.py`,
   write ≥1 correctness test + ≥1 `@pytest.mark.performance` test, leave the `[EVOLVE]` block empty
   (generate) or seeded (improve). Details + rules: [references/task_and_config.md](references/task_and_config.md).
2. Hand-check: `pytest --ref -s task.py` then `pytest -s task.py`.
3. Run: `python scripts/run_custom_task.py custom_task=/abs/my_task task_origin=local
   job_name=… task_name=… gpu_arch=… language=… <regime overrides>` (see the run-config templates).

**Path B — reimplement / mimic the method** (any environment, no GPU needed to study it):
- [scripts/evolve_loop.py](scripts/evolve_loop.py) is the whole algorithm in one dependency-light
  file. You supply two callbacks — `generate(prompt)->answer` (your LLM) and
  `evaluate(code)->EvalResult` (your compile+test+benchmark). `run_evolution(task, generate,
  evaluate, cfg, seed_code)` runs the MAP-Elites + island + gradient loop and returns the best
  program. **Run `python scripts/evolve_loop.py --demo`** to watch the search converge with a fake
  LLM/evaluator (no GPU, no keys) — it demonstrates the gate, elite replacement, and ladder-climb.
  For faster convergence on a hard search (finer/measured descriptors, staged activation, surrogate
  ranking, trust-region param refinement ④a, and edit-trust-region structure refinement ④b) see
  [references/advanced_convergence.md](references/advanced_convergence.md) and the demos
  `python scripts/evolve_loop.py --demo-advanced` (①②③④a) and `--demo-edit-tr` (④b).
- [scripts/optimization_classifier.py](scripts/optimization_classifier.py) is the deterministic
  code→cell mapping (the MAP-Elites behavior descriptor); [scripts/optimization_knowledge.json](scripts/optimization_knowledge.json)
  is the `[dimension][backend][level]` technique store the prompt draws from.

## The optimization ladder (the search space)

Each axis is climbed level by level; the prompt asks the model to reach the target cell's levels.

| Axis | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| **memory_opt** | naive global | vectorized/coalesced (float4) | shared/local tiling | register blocking + async/double-buffer |
| **compute_opt** | multi-pass | fusion / FMA | single-pass online (online-softmax, Welford) | blocked/tiled (Flash-Attention), Tensor Cores |
| **parallelism_opt** | thread-only | block/work-group tree reduction | warp/sub-group collectives (`__shfl`, `reduce_over_group`) | hierarchical block→warp→thread |
| **esimd_opt** (opt) | none | basic ESIMD / WMMA | optimized (LSC, cache hints) | expert DPAS / Tensor Cores |

Read profiler verdicts onto the ladder: *memory-bound* → raise memory_opt; *compute-bound* → raise
compute_opt + parallelism_opt; *low occupancy* → tune sizes / raise parallelism_opt; *bank
conflicts / high stall* → fix antipatterns before climbing. (Knowledge JSON `dimension_guidance`.)
Those verdicts come from the profiler step described next.

## Profiling kernel performance with `unitrace` (Intel) / NCU (NVIDIA)

Profiling is **how the loop earns its feedback**: a generic "make it faster" prompt wastes the
search; a roofline verdict from real counters tells the model *which axis to climb*. In the
pipeline this runs **automatically** after a kernel passes correctness (`perf_score == 5`) — it
benchmarks, then profiles, then renders a NL summary into the next prompt. You generally don't
invoke it yourself; you (a) make sure the profiler is installed, and (b) read its verdict and let
the ladder rules above pick the next move. Run it by hand only to diagnose a candidate out-of-loop.

**Which profiler runs** (`get_profilers` / `get_profiler_feedback_class`, keyed by `(language, arch)`):
- **Intel** (SYCL / OpenCL / Triton-on-XPU `lnl,ptl,bmg,dg2`) → **`unitrace`**, in **three passes**:
  `ComputeBasic`, `MemoryProfile`, `VectorEngineProfile`.
- **NVIDIA** (CUDA / Triton-on-CUDA) → **NCU** (`ncu --set detailed --csv`); ships with CUDA, no install.

**Step 0 — install `unitrace` once (Intel only).** It must already be on `PATH` (or
`$KERNELFOUNDRY_unitrace_cmd`); nothing builds it on demand and a missing binary fails the
profiler-wrap. Run `bash scripts/install_unitrace.sh` (idempotent: no-op if present, else reuse/
clone+cmake+make `intel/pti-gpu` `tools/unitrace`, symlink into `~/.local/bin`, and best-effort
enable non-root i915 profiling). Ensure that bin dir is on `PATH`.

**How the loop wraps a run** (`Unitrace.wrap_cmd`), per metric group:
```
unitrace --chrome-kernel-logging --chrome-itt-logging --start-paused --session <uuid> \
         --group <ComputeBasic|MemoryProfile|VectorEngineProfile> --metric-query \
         --output-dir-path <out> -o <out>/trace  <your run cmd>
```
`--chrome-*-logging` → the timeline; `--start-paused` → counters collect only inside the ITT
**`model run loop`** range (so warmup/teardown are excluded — the harness's perf tests mark this).
OpenCL uses `OCLUnitrace` + `retry_unitrace.sh` (adds `--opencl --metric-sampling`, retries ≤20× for
a ≥10-line metrics file). Output is read back as `timeline` (`*.json`) + `trace.metrics.<pid>` CSVs.

**Reading the verdict → next move** (what `analyze_kernel` extracts, and the ladder response):

| Signal (CSV column) | Verdict | Do |
|---|---|---|
| arithmetic intensity vs roofline (`GpuTime`, `GPU_MEMORY_BYTE_*_RATE[GBpS]`, `XVE_INST_EXECUTED_ALU*`) | **memory-bound** | raise **memory_opt** (vectorize → SLM tiling → register/double-buffer) |
| same, AI above roofline | **compute-bound** | raise **compute_opt** + **parallelism_opt** (fusion/online → Tensor/XMX, warp collectives) |
| `XVE_THREADS_OCCUPANCY_ALL[%]` low | **low occupancy** | tune launch sizes / raise **parallelism_opt** |
| `SLM_BANK_CONFLICT_COUNT` / `SLM_ACCESS_COUNT` high, `XVE_STALL[%]` high | **bank conflicts / stalls** | fix the access antipattern **before** climbing |
| `XVE_INST_EXECUTED_XMX_*` ≈ 0 on a matmul | **Tensor/XMX unused** | push **compute_opt** to L3 (XMX/Tensor Cores) |

**Profile a candidate by hand** (ad-hoc, outside the loop): run your kernel in a small benchmark
script (load `.so` → loop → `synchronize()`), once per group, then read `trace.metrics.<pid>`:
```bash
RUN="python bench.py"
for G in ComputeBasic MemoryProfile VectorEngineProfile; do
  unitrace --chrome-kernel-logging --chrome-itt-logging --group "$G" --metric-query \
           --output-dir-path "prof_$G" -o "prof_$G/trace"  $RUN
done
```
Timeline-only one-shot: add `-q`, drop `--group/--metric-query` (see
`scripts/experimental/profile_special_kernels.py`). NVIDIA: `ncu --set detailed --csv --log-file
ncu_report.csv <cmd>` (perf-counter perms; pipeline runs it under `sudo -E`) — for deep NCU report
reading use `ncu-report-skill`. Full column→verdict→template mapping and the ITT-segmentation
details are in [references/evaluation_and_feedback.md §4](references/evaluation_and_feedback.md).

## Reference docs (read the one you need)

- [references/algorithm.md](references/algorithm.md) — the evolutionary loop, MAP-Elites archive,
  island/migration, fitness, parent/inspiration sampling, QD gradient, target-cell selection.
  Pseudo-code for the whole search; the controller knob table.
- [references/optimization_aware_prompting.md](references/optimization_aware_prompting.md) — the
  14-section prompt, the three knowledge layers, the `[dimension][backend][level]` JSON, meta-prompting
  (evolving the prompt via SEARCH/REPLACE), feedback injection.
- [references/evaluation_and_feedback.md](references/evaluation_and_feedback.md) — Task/block
  markers, extract→splice→compile→test→benchmark→profile, `perf_score` 0–5 + speedup, profiler→NL
  feedback (roofline verdict), log cleaning.
- [references/task_and_config.md](references/task_and_config.md) — exact file formats for a task,
  Hydra run config, the three canonical run profiles, single-vs-ensemble inference, settings rules.
- [references/advanced_convergence.md](references/advanced_convergence.md) — **opt-in** fixes for
  "descriptors too macroscopic → search concentrates in coarse cells and drifts off the optimum":
  fine-grained/measured/CVT descriptors, staged coarse→fine activation, surrogate pre-ranking
  (SAIL/DSA-ME), and trust-region local refinement — both parameter (④a) and edit/structure (④b,
  reusing SEARCH/REPLACE diffs with a stay-in-cell constraint) — the convergence engine MAP-Elites
  lacks. All implemented in `evolve_loop.py`; demos: `--demo-advanced` (①②③④a), `--demo-edit-tr` (④b).

## Non-negotiables (get these right or the method fails)

- **Correctness gates performance.** Never reward speed before a kernel passes its tests; the
  `perf_score` 0–5 gate + speedup bonus encodes this. Use the formula in algorithm.md §5 verbatim.
- **The cell comes from the code, not the metrics.** Behavior descriptors are static; metrics only
  decide who wins a cell. This is what gives MAP-Elites its diversity.
- **The prompt is the mutation operator.** Its leverage is the *target-cell-specific* techniques +
  the parent's *profiler/eval feedback*. A generic "make it faster" prompt wastes the loop.
- **Diversity beats greedy — but pure illumination doesn't converge.** Islands + exploration
  sampling + diverse inspirations + an empty-cell pull stop the search collapsing onto one local
  optimum; don't strip them. Conversely, MAP-Elites *illuminates*, it does not *optimize* — to
  actually reach a local optimum add a local search (trust-region param refinement), and if the
  descriptors are too macroscopic make them measured/finer. Both are opt-in in
  [references/advanced_convergence.md](references/advanced_convergence.md); mind the diversity↔convergence trade-off.
- **Wider vs deeper budget:** `branches_per_iteration` widens each generation; `max_iters` deepens
  the search; more `num_islands` preserves diversity at high iteration counts.
