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

Profiler chosen by `(language, arch)`: **NCU (Nsight Compute)** on NVIDIA
(`ncu --set detailed --csv`); **Unitrace** on Intel (3 passes: ComputeBasic, MemoryProfile,
VectorEngineProfile). Raw CSV/timeline is parsed, restricted to the kernel window, and turned into
a compact **natural-language summary** for the prompt, e.g.:

- **Intel/Unitrace**: runtime, launch config, thread occupancy %, global-mem bandwidth (abs + % of
  peak), SLM bandwidth %, vector-engine stall %, SLM bank-conflict %, ALU utilization & instr mix,
  and a **roofline verdict (memory / compute / balanced bound)**.
- **NVIDIA/NCU** (longest-running kernel): SM / L1 / L2 / DRAM throughput, roofline analysis, and a
  bottleneck hint, fed in as "Your kernel has been analyzed with a profiler: {summary}".

This roofline-grounded summary is what lets the model pick the *right* next optimization (e.g.
"memory-bound → add SMEM tiling" maps to bumping `memory_opt`).

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
