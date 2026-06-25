# Evaluation Pipeline & Profiler Feedback

Distilled from `kernelfoundry/eval_pipeline/{task.py, profiler_feedback.py, profiler_command.py}`,
`kernelfoundry/{compiler.py, testing.py, test_base.py}`, and
`kernelfoundry/algorithm/utils/{extract_code.py, code_editing.py, eval_helper.py, score.py}`.

This is the **fitness function plumbing**: it turns an LLM answer into a number plus a feedback
string. Correctness is always gated before performance.

---

## 1. The Task (unit of work)

A `Task` bundles everything needed to evaluate one kernel:
`task_data` (in-memory file map), `config` (from `config.yaml`), `blocks`
(`EVOLVE` / `REFERENCE` / `USER_INSTRUCTIONS`), `correctness_tests`, `profile_tests`,
build flags, and result slots (`build_result`, `test_result`, …).

- **`Task.create(path|tar|zip|bytes) → (Task, metadata)`** reads `config.yaml`, `task.py`,
  `conftest.py`; discovers block markers; runs `pytest --collect-only` to split tests into
  **correctness** (no marker) vs **performance** (`@pytest.mark.performance`).
- **`task.validate()`** asserts `task.py` & `conftest.py` exist, ≥1 correctness test, ≥1 perf test,
  an `EVOLVE` block, and `task_name`/`job_name` in config.
- **`task.with_blocks({"EVOLVE": code})`** splices the LLM's code into the kernel file between the
  markers, producing a new Task to evaluate.

### Block markers (the contract between author, model, and harness)
Plain comment lines anywhere in the task's files:
```
// [EVOLVE_START]   ... region the LLM rewrites ...        // [EVOLVE_END]
# [REFERENCE_START] ... ground-truth implementation ...    # [REFERENCE_END]
# [USER_INSTRUCTIONS_START] ... free-text guidance ...      # [USER_INSTRUCTIONS_END]
```
`EVOLVE` is what evolves (often an empty body in `*_kernel.EXT`); `REFERENCE` is correctness +
speedup baseline (usually a PyTorch function in `task.py`); `USER_INSTRUCTIONS` is optional.

---

## 2. End-to-end evaluation (`Evaluator.run` / `TaskRunner`)

```
extract code from LLM answer        → if it fails: perf_score=0, stop
splice into EVOLVE block (with_blocks)
build reference (if needed) + build custom kernel   → compile fail: perf_score=1, stop
for each gpu_arch: run correctness tests (pytest)   → runtime err 2 / shape 3 / value 4
if correct (perf_score=5):
    run performance tests → runtimes per benchmark
    speedup = gmean(ref_runtime / custom_runtime)   (timeout sentinels if it hangs)
    run profiler on custom (+ optionally reference) → metrics → NL feedback
assemble eval_log (compiler out + pytest out + profiler summary)
return EvalResult
```

### Code extraction (`extract_code.py`)
`extract_code_flexible(answer, tag)` tries, in order: triple-quoted blocks, `<lang>…</lang>` tags,
then ```` ```lang … ``` ```` fences; falls back to a C++-density heuristic
(`extract_cpp_code_heuristic`) when the model didn't fence its code. Also supports
`<<<<<<< SEARCH/======/>>>>>>> REPLACE` diff application for diff-mode evolution.

### Compilation (`compiler.py`)
Backend chosen by `(language, gpu_arch)`:
- **CUDA / PyTorch / Triton-on-NVIDIA** → `TorchCompiler` (`torch.utils.cpp_extension.load`,
  `TORCH_CUDA_ARCH_LIST` set, ~120 s timeout).
- **SYCL / OpenCL / Triton-on-Intel** → `IcpxCompiler`: `icpx -fsycl
  -fsycl-targets=spir64_gen,spir64 -c` → `-fsycl-link` → link `.so` against torch.
Output captured as `{stdout, stderr, returncode}`.

### Correctness (`testing.py`)
`assert_allclose(actual, expected, rtol=0.01, ratio_below_max_err=0.99)`: relative error per
element; pass iff ≥99 % of elements are within rtol. Author-written tests in `task.py` call torch's
`allclose` or this helper. Multi-GPU: tests run per arch and results combine (gmean).

---

## 3. Scoring (recap of algorithm §5)

`perf_score` 0–5 gate + speedup bonus. `runtime_improvement` = gmean over benchmarks of
`ref/custom`. `score.py` also maps gpu_arch → a baseline-times table when comparing against stored
torch-eager baselines (KernelBench-style), and computes geometric-mean speedup over correct cases.
Timeouts use sentinels (`RUNTIME_IF_TIMEOUT ≈ 1e11`, `SPEEDUP_IF_TIMEOUT ≈ 1e-10`) so a hang ranks
below any real kernel but doesn't crash the loop.

---

## 4. Profiler feedback (`profiler_feedback.py`, `profiler_command.py`)

Profiling runs **automatically inside the evaluation step**, but only after a kernel is correct
(`perf_score == 5`): the harness benchmarks it, then profiles the custom kernel (and optionally the
reference) and turns the raw output into a compact NL summary for the *next* prompt. Profiler is
chosen by `(language, arch)` in `get_profilers()` / `get_profiler_feedback_class()`:
**NCU (Nsight Compute)** on NVIDIA, **Unitrace** on Intel.

### 4.1 How the loop invokes `unitrace` (Intel SYCL / Triton-on-XPU)

`get_profilers("sycl", …)` returns **three** profiling passes — one per metric group:
`ComputeBasic`, `MemoryProfile`, `VectorEngineProfile`. Each pass wraps your run command
(`Unitrace.wrap_cmd`) as:

```
unitrace --chrome-kernel-logging --chrome-itt-logging \
         --start-paused --session <uuid-hex> \
         --group <ComputeBasic|MemoryProfile|VectorEngineProfile> --metric-query \
         --output-dir-path <out> -o <out>/trace  <your run command>
```

- `--chrome-kernel-logging --chrome-itt-logging` emit the **timeline** (`*.json` chrome trace);
  drop them (`timeline=False`) for metrics-only.
- `--start-paused` means collection is **paused until an ITT marker resumes it** — see segmentation
  below — so wrap the timed region of your benchmark in an ITT range named `model run loop`.
- `--metric-query` collects the named `--group`'s hardware counters into a `trace.metrics.<pid>` CSV.
- The binary is resolved from `$KERNELFOUNDRY_unitrace_cmd` (default literal `"unitrace"`).

`read_output()` then globs the out dir: every `*.json` → `timeline`, every `trace.*` →
keyed by filename (so the CSV lands under `trace.metrics.<pid>`).

### 4.2 OpenCL variant (`OCLUnitrace` + `retry_unitrace.sh`)

OpenCL metric collection is flaky, so the OCL path wraps the same idea in a **retry loop** (up to 20
attempts, each requiring ≥10 lines in the metrics file) and swaps two flags:
`--opencl` and `--metric-sampling` (instead of `--metric-query`), with `PTI_ENABLE_COLLECTION=0` in
the env. `prepare()` copies `retry_unitrace.sh` into the (container-mounted) output dir first.

### 4.3 What each pass surfaces, and where it lands on the ladder

`UnitraceProfilerFeedback.analyze_kernel` reads these columns and derives a verdict:

| Pass | Key CSV columns | Diagnoses | Ladder move |
|---|---|---|---|
| **ComputeBasic** | `GpuTime[ns]` (runtime), `GPU_MEMORY_BYTE_{READ,WRITE}_RATE[GBpS]` (gmem BW), `XVE_STALL[%]` (stall), `XVE_INST_EXECUTED_ALU{0,1,2}_ALL[events]` (ALU work) | runtime, bandwidth %-of-peak, the **memory / compute / balanced** roofline verdict (arithmetic intensity vs the hardware roofs) | memory-bound → raise `memory_opt`; compute-bound → raise `compute_opt` + `parallelism_opt` |
| **MemoryProfile** | `SLM_ACCESS_COUNT[events]`, `SLM_BANK_CONFLICT_COUNT[events]` | SLM (shared-local) bank-conflict % | conflicts high → fix access pattern *before* climbing |
| **VectorEngineProfile** | `XVE_THREADS_OCCUPANCY_ALL[%]` (occupancy), `XVE_INST_EXECUTED_{FP16,FP32,FP64,INT*,MATH,XMX_*}[events]` (instr mix) | low occupancy; whether **XMX / Tensor (ALU2)** is used at all | low occupancy → tune sizes / `parallelism_opt`; no XMX → `compute_opt` L3 (Tensor Cores) |

The bound is the arithmetic-intensity-vs-roofline test per ALU; metrics are matched per kernel by
median `GpuTime[ns]` across the three passes. The roofs come from `hardware_info.HardwareRoofs`
(`get_roofs(worker_info)`); peaks like `GPU_MEMORY_BYTE_READ`, `SLM_BYTE_READ/WRITE` set the %-of-peak.

### 4.4 Timeline segmentation (the ITT `model run loop` marker)

`collate_data` looks for `ittapi::model run loop[::<label>]` events in the timeline, takes the
`GlobalInstanceId`s of kernels inside each event's time range, and **restricts the metrics to that
window** — so the feedback reflects the *benchmark*, not warmup/teardown. With >3 segments it keeps
only the **median** and **slowest**. If no marker is found it warns and falls back to the whole run.
Mark your timed region accordingly (the harness's perf tests already do).

### 4.5 The NL summary the model sees

The per-kernel analysis is rendered through `intel_profiler_kernel_feedback.j2` into sections —
*Runtime & Occupancy*, *Roofline Analysis*, *Bottleneck Analysis* — prefixed with
"Your code has been analyzed with a profiler. Here is a summary of the results:". For example it
states the launch config, runtime (ms), occupancy/stall %, the memory/compute/balanced verdict,
the top ALU and its instruction mix, and concrete hints ("does not use SLM — consider SLM tiling",
"bank conflicts X% — fix access pattern", "ALU2/XMX not utilized"). On NVIDIA, `NCUProfilerFeedback`
emits SM/L1/L2/DRAM throughput + the roofline & bottleneck rule for the **longest-running** kernel.
This roofline-grounded summary is what lets the model pick the *right* next optimization
(e.g. "memory-bound → add SMEM tiling" maps to bumping `memory_opt`).

### 4.6 Running `unitrace` by hand (ad-hoc diagnosis / mimicking the feedback step)

To profile a kernel outside the loop — e.g. to diagnose a candidate before deciding which axis to
climb — drive a small benchmark script (load the `.so`, run the kernel N times, sync) under the same
three passes:

```bash
RUN="python bench.py"          # your script that runs the kernel in a loop and syncs
for G in ComputeBasic MemoryProfile VectorEngineProfile; do
  unitrace --chrome-kernel-logging --chrome-itt-logging \
           --group "$G" --metric-query \
           --output-dir-path "prof_$G" -o "prof_$G/trace"  $RUN
done
# Then read prof_*/trace.metrics.<pid>: GpuTime[ns], GPU_MEMORY_BYTE_*_RATE[GBpS],
# XVE_STALL[%], XVE_THREADS_OCCUPANCY_ALL[%], SLM_BANK_CONFLICT_COUNT, XVE_INST_EXECUTED_ALU*.
```

A minimal one-shot timeline-only capture (no per-group counters) is
`unitrace --chrome-kernel-logging --chrome-itt-logging -q --output-dir-path <out> -o <out>/trace <cmd>`
(this is exactly what `scripts/experimental/profile_special_kernels.py` does). On NVIDIA, the manual
equivalent is `ncu --set detailed --csv --log-file ncu_report.csv <cmd>` (needs perf-counter
permission — the pipeline runs it under `sudo -E`); for deeper NCU analysis use the `ncu-report-skill`.

### Installing the Intel profiler (`unitrace`)
The pipeline resolves the unitrace binary from `$KERNELFOUNDRY_unitrace_cmd` (default literal
`"unitrace"`) and **assumes it is already on `PATH`** — it is *not* auto-built on demand, and a
missing binary makes the profiler-wrap command fail. Build it once from Intel's
[`pti-gpu`](https://github.com/intel/pti-gpu) repo (`tools/unitrace`: cmake + make), then symlink
into a `PATH` dir. The skill ships [`scripts/install_unitrace.sh`](../scripts/install_unitrace.sh)
to do exactly this idempotently: no-op if already present, reuse an existing `pti-gpu` build,
else clone+build+symlink (and best-effort enable non-root i915 profiling). NVIDIA's NCU path
needs no equivalent (NCU ships with the CUDA toolkit).

---

## 5. Feedback cleaning (`eval_helper.py`)

Before logs reach the model they're shrunk and de-noised:
- **`postprocess_compiler_output`**: keep from the first compiler command to `ninja: build stopped`;
  shorten long command lines and include-stacks; drop warnings from non-source files; collapse
  duplicate lines; abbreviate toolchain paths; truncate to ~5000 chars while preserving errors.
- **`postprocess_pytest_output`**: start at the first `task.py::` line; drop dependency-skip noise;
  replace giant tensor dumps with `tensor(...)`.

Result: the eval_log the next prompt sees is dense, on-point, and within token budget.

---

## Reimplementation checklist
- Marker-based file with an EVOLVE region + a REFERENCE region.
- Extract → splice → compile → correctness-test → benchmark → profile, gating on each stage.
- `perf_score` 0–5 + speedup bonus as the single fitness number.
- A profiler→NL-summary step (roofline verdict is the high-value bit).
- Aggressive log cleaning so feedback stays short and actionable.
