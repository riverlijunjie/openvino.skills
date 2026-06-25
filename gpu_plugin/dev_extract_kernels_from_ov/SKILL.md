---
name: dev_extract_kernels_from_ov
description: Extract any OpenVINO intel_gpu OCL kernel into a standalone, OV-faithful micro-benchmark for isolated optimization. Use when asked to pull a GPU kernel out of OV, build a separate kernel test harness, validate kernel correctness against a reference, profile a kernel with cliloader, or compute its roofline / bandwidth-utilisation on the target GPU. Reproduces OV's exact JIT constants, build options, and dispatch geometry so an optimised kernel drops straight back into OV. Works for GEMV/GEMM, decode, transcode, quant, attention, and similar ocl_v2 kernels; the bundled GGUF FC kernels are the worked reference. Triggers: "抽取kernel", "kernel单独测试", "kernel roofline", "cliloader profile", "extract OV kernel".
---

# Extracting & testing an OpenVINO GPU kernel standalone

This skill turns any OpenVINO `intel_gpu` OpenCL kernel into a **self-contained
micro-benchmark** that you can iterate on without a full OV build, then drop the optimised
`.cl` straight back into OV. The decisive property is **JIT faithfulness**: the kernel is
compiled with the *exact* preamble, JIT constants, build flags, and work-group geometry OV
uses at runtime, so numbers and behaviour transfer 1:1.

A complete worked instance lives in this directory (the GGUF FullyConnected kernels). Read
it as the reference while applying the steps below to a new kernel:

- `kernels/` — verbatim OV `.cl` + `gguf_iq_tables.hpp` + `common.cl` (deps), plus the only
  added files: `cache_flush.cl` and a `fetch_data.cl` stub.
- `harness/ov_jit.py` — reproduces OV's JIT preamble + per-kernel JIT constants.
- `harness/gguf_ref.py` — NumPy correctness oracle (mirrors each decoder).
- `harness/hw.py` — device capability + roofline model.
- `harness/bench.py` — build / dispatch / L2-flush / verify / roofline driver.
- `harness/make_results_md.py` — renders `results/report.json` → `RESULTS.md`.
- `configs/default.json` — shape / format / dtype sweep. `run.sh` — cliloader wrapper.
- `README.md` / `results/RESULTS.md` — the produced documentation & data.

Keep explanations conversational. Use the bundled GGUF files as copy-paste-and-adapt
templates rather than writing from scratch.

---

## When to use
- "把 X kernel 抽取出来单独测试 / 优化"
- "评估某 kernel 的 roofline / 带宽占比 / 是 memory-bound 还是 compute-bound"
- "用 cliloader profile 这个 kernel"
- Preparing a kernel for the `dev_kf_distill` / Blackwell-style optimisation loop where you
  need a fast, correctness-checked, repeatable micro-benchmark.

## Environment (this machine)
- GPU: **Intel Arc B580** (Battlemage / Xe2) — 160 Xe cores @ 2.9 GHz, 18 MiB L2,
  ~456 GB/s GDDR6, FP32 ~14.85 TFLOPS, INT8 dp4a ~59 TOPS, `cl_khr_integer_dot_product`.
- Tooling: `python3` + `pyopencl` + `numpy`, Intel OpenCL runtime (`ocloc`/`clinfo`),
  and `cliloader` built at `/tmp/opencl-intercept-layer/build/cliloader/cliloader`
  (source: https://github.com/intel/opencl-intercept-layer, build with
  `cmake -S . -B build -DENABLE_CLILOADER=ON && cmake --build build -j`).
- OV tree: `thirdparty/openvino/src/plugins/intel_gpu/src` — kernels under
  `graph/impls/ocl_v2/*.cl`, impls under `graph/impls/ocl_v2/**/*.cpp`.

---

## The procedure

### Step 0 — Identify the kernel and its impl
1. Find the `.cl` template and the C++ impl that builds it. Search the impl by kernel name:
   `grep -rn "KernelGenerator(\"<kernel_name>\")" graph/impls/ocl_v2`. The impl class(es)
   you want subclass `KernelGenerator` and define `get_entry_point`, `get_jit_constants`,
   `get_arguments_desc`, `get_dispatch_data_func`.
2. Note **every** kernel the primitive dispatches. One primitive often has several stages
   (the GGUF FC has 4: a GEMV, a transcode, a prequant, a dp4a GEMV). Extract them all if
   you want to profile the whole path.
3. Record the kernel's **inputs/outputs and their layouts** (from `get_arguments`/the
   explicit-args dispatch helpers) and any **internal scratch buffers**
   (`get_internal_buffer_descs`).

### Step 1 — Copy the kernel(s) verbatim + dependencies
- `cp` each `.cl` into `kernels/` unchanged. **Verify byte-identity** with `md5sum` against
  the OV source and re-check after any OV resync — the whole point is that the extracted
  kernel *is* the OV kernel.
- Copy every `#include`d dependency that isn't generated: `common.cl`
  (`kernel_selector/cl_kernels/include/batch_headers/common.cl`), any shared table header
  (e.g. `gguf/gguf_iq_tables.hpp`), etc. Preserve the include directory structure so the
  kernel's relative `#include` paths resolve under `-I kernels`.
- For `fetch_data.cl` (the layout-addressing header): if your kernel only uses the reduced
  `*_GET_INDEX` macros for contiguous tensors, a stub is enough (see below). If it uses
  blocked/padded formats, copy the real `fetch_data.cl` and emit the full layout constants.

### Step 2 — Reproduce OV's JIT codegen exactly (`ov_jit.py`)
OV does **not** compile the `.cl` directly; it prepends a generated preamble. Reproduce it:

1. **Decoration macros** — copy `build_preamble()` from `ov_jit.py` as-is. It reproduces
   `CodeBuilder::decoration_macro` (in `graph/common_utils/jitter.hpp`) for
   `FUNC`/`FUNC_CALL`/`CONST_ARRAY_DECL`/`CONST_ARRAY_REF` and the `KERNEL(name)`/`KERNEL_ID`
   value macros. This is **kernel-agnostic** — reuse it unchanged.
2. **Per-kernel JIT constants** — write one builder per OV `*Generator::get_jit_constants`,
   emitting the **same** `make_jit_constant` names/values. Cross-check against the impl
   line-by-line; a single wrong constant changes the compiled code. Include:
   - geometry/algorithm constants (`K_SIZE`, `N_SIZE`, tile sizes, `SG_SIZE`, unroll/`NROW`,
     group sizes, format flags, …) — copy the literals/derivations from the impl.
   - `make_type_jit_constants("INPUT0"/"OUTPUT"/…, dtype)` — the
     `*_TYPE`/`TO_*_TYPE`/`AS_*_TYPE`/`*_VAL_MAX`… family. The table in `ov_jit.py` already
     mirrors `jitter.cpp`; add types if your kernel needs them.
   - layout `*_GET_INDEX` / `*_FEATURE_NUM` macros (see Step 1 note).
   - the dynamic-shape pair `OPTIONAL_SHAPE_INFO_ARG`/`_TENSOR` (empty when static).
3. **Build options** — match `KernelGenerator::get_build_options`. For Intel + work-group
   collectives that is `["-cl-mad-enable", "-cl-std=CL3.0"]` (CL3.0 also exposes the
   `cl_khr_integer_dot_product` dp4a builtins). Keep them in `OV_BUILD_OPTIONS`.
4. **Entry point name** — mirror `get_entry_point` so cliloader's table shows OV-recognisable
   names and the build cache keys correctly.

Sanity gate: a probe-compile of every kernel × format/dtype on the real device **before**
writing the dispatch/verify code. If it builds clean, the preamble is faithful.

### Step 3 — Build the correctness oracle (`*_ref.py`)
- Reimplement the kernel's math in NumPy, mirroring the `.cl` **field offsets and op order**
  (not a from-scratch algorithm). For decoders, parse any lookup tables out of the **verbatim
  header** (one source of truth — see `load_iq_table()`), don't hand-copy them.
- Generate valid random inputs. Make the generator produce in-range data by construction
  (mask indices to table sizes, set scale fields to small controlled values), so any random
  pattern is decodable.
- Pick the right pass metric per kernel kind, and make it measure what the kernel actually
  computes:
  - GEMV/GEMM/decode → relative-L2 of `out` vs reference `< ~2e-2` (f16 accumulation slack)
    + cosine ≈ 1.
  - quant/transcode → **integer codes match** the reference quantizer (±1 LSB for rounding
    ties) **and** scales match the **f16-cast** of the ideal scale. Do NOT divide
    reconstruction error by the f16 quant step — near-zero groups blow that up to dozens of
    "steps" while the absolute error is ~1 LSB and OV behaves identically. Normalise by the
    group's data magnitude instead. (This exact trap bit the GGUF transcode; see
    `bench.py:run_transcode`.)

### Step 4 — Dispatch with OV's exact geometry (`bench.py`)
- Build via `ov_jit` preamble + `-I kernels`. Bind buffers in the **same order** as the OV
  argument descriptor / explicit-args helper.
- Set global/local work sizes **exactly** as `get_dispatch_data_func` (or the impl's explicit
  dispatch) computes them — e.g. GGUF GEMV `global=[N*SG, M, 1] local=[SG,1,1]`; dp4a
  `global=[ceil(N/NROW)*SG, M, 1]`; transcode `global=[ceil(N/SG)*SG, K/block_elem, 1]`. The
  geometry is part of the kernel's contract; getting it wrong invalidates the perf number.
- For multi-stage paths, chain the stages on one queue exactly as the impl's `execute()` does
  (e.g. prequant → dp4a), and time both the full path and the hot kernel.

### Step 5 — Accurate, repeatable timing
- **Flush the cache between timed iterations** with `cache_flush.cl`: an RMW sweep over a
  scratch buffer several × larger than L2 (default 128 MiB), twice. This both evicts a
  resident weight (forcing an honest DRAM re-read) and keeps the GPU at boost clock. Validate
  with an A/B: a sub-L2 weight should get *faster* with flushing (no downclock) and a >L2
  weight should be unchanged. Without this you measure L2 bandwidth or a downclocked artefact.
- Time with **CL profiling events** (`event.profile.end - .start`), median over N iters,
  with warmup. Report median/min/mean.

### Step 6 — Roofline (`hw.py`)
- Model the device: compute peaks (FP32 = CUs·SIMD·clock·2; INT8 dp4a = ×4), peak DRAM BW,
  L2 size. All overridable via config `"hw"` block + env (`OV_BENCH_PEAK_BW_GBPS`, …) so the
  same harness retargets another Xe GPU.
- Per case compute bytes moved (weight + activation + output + scratch) and FLOPs (`2·M·N·K`
  for GEMV/GEMM), then `min(peak_compute, AI·peak_bw)` attainable and the **% of peak BW** /
  **roofline %**. Memory-bound decode kernels: drive **% peak BW** toward 100%. Use the INT8
  ceiling for dp4a paths, FP32 ceiling for float-accumulate paths.

### Step 7 — Profile under cliloader & report
- Run through `run.sh` (wraps `cliloader -dv -q`) for an **independent** per-kernel device-time
  table (host overhead can't skew it) alongside the in-process numbers. cliloader also surfaces
  free optimisation signals: per-kernel `SIMD`, `REG`, and **`SPILL=…`** (register spill — a
  prime first target; the GGUF Q3_K decode spills 1600 bytes), plus the actual `GWS/LWS`.
- `make_results_md.py` renders `results/report.json` (+ cliloader table) into `RESULTS.md`:
  a per-case table (time, cliloader time, achieved BW, % peak, GFLOPS, roofline %, bound,
  correctness) and an analysis section ranking cases by BW utilisation with optimisation notes.

### Step 8 — Optimise & reintegrate
- Edit `kernels/*.cl` in place, re-run, compare roofline %. Iterate. When using the
  `dev_kf_distill` / optimisation loop, target ≥ ~80% roofline.
- To reintegrate: `cp` the improved `.cl` back to the OV tree. If you added a JIT constant,
  mirror it in **both** `ov_jit.py` and the OV `*Generator::get_jit_constants`, and add the
  matching dispatch change. Re-verify byte-diff of the unchanged parts.

---

## Usage (the bundled GGUF reference)
```bash
cd .github/skills/dev_extract_kernels_from_ov
./run.sh --config configs/default.json --out results/report.json   # full sweep under cliloader
python3 harness/bench.py --quick                                    # tiny smoke sweep
python3 harness/make_results_md.py                                  # report.json -> RESULTS.md
```
Each config `case`: `kernel` ∈ {opt, transcode, dp4a}; `format`, `K`, `N`, `M`; per-case
overrides `iters`/`warmup`/`flush_l2`/`in_dt`/`out_dt`/`seed`.

## Gotchas (learned the hard way)
- **Verbatim or it's worthless** — never edit the `.cl` while extracting; md5-check it.
- **Wrong JIT constant = silently different kernel** — diff every constant against the impl.
- **Dispatch geometry is part of the contract** — copy GWS/LWS from the impl, don't guess.
- **Cache flush is mandatory for honest BW** — and it doubles as keep-warm against downclock.
- **Pick a metric that measures the kernel, not f16 rounding** — see the transcode trap above.
- **CL3.0 build std** for any kernel using subgroup collectives or integer-dot builtins.
- **Tables: parse, don't copy** — re-derive lookup tables from the verbatim header.

## Key rules
- Don't push commits to remote repos.
- Re-sync verbatim kernels after upstream OV changes (it's just a `cp` + md5 re-check).
