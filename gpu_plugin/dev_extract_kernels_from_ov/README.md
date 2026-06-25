# dev_extract_kernels_from_ov — Standalone GGUF kernel test & roofline harness

Extracts the four OpenVINO **intel_gpu** GGUF FullyConnected OpenCL kernels into a
self-contained micro-benchmark so they can be tuned **outside** a full OV build and dropped
back in with zero source changes. Each kernel is compiled with the **exact same JIT
parameters OV uses at runtime**, so any optimisation measured here is directly transferable.

Target device for the bundled defaults: **Intel Arc B580** (Battlemage / Xe2) — 160 Xe
cores @ 2.9 GHz, 18 MiB L2, ~456 GB/s GDDR6, `cl_khr_integer_dot_product` (dp4a).

---

## What was extracted (verbatim)

From `thirdparty/openvino/.../graph/impls/ocl_v2/`:

| File (here)                         | OV source                          | Role |
|-------------------------------------|------------------------------------|------|
| `kernels/fc_gguf_opt.cl`            | `fc_gguf_opt.cl`                   | Memory-bound decode **GEMV** (any M; M=1 decode). One subgroup/output, K-blocks striped across lanes, dequant in registers. Handles all 10 baseline formats via a `GGUF_IS_*` JIT flag. |
| `kernels/fc_gguf_transcode.cl`      | `fc_gguf_transcode.cl`             | Compute-bound **prefill** repack: GGUF block → packed i4/i8 weight + f16 per-group scale (feeds a oneDNN WOQ matmul in OV). |
| `kernels/fc_gguf_prequant.cl`       | `fc_gguf_prequant.cl`             | Activation → int8 + per-32 f32 scale (feeds the dp4a path). |
| `kernels/fc_gguf_dp4a.cl`           | `fc_gguf_dp4a.cl`                 | Q5_K / Q6_K int8-activation **dp4a** decode GEMV (SWAR unpack + `dot_acc_sat_4x8packed_us_int`). |
| `kernels/gguf/gguf_iq_tables.hpp`   | `gguf/gguf_iq_tables.hpp`         | IQ codebook / sign tables (shared, verbatim). |
| `kernels/include/batch_headers/common.cl` | OV `cl_kernels/...common.cl`  | OV macro header the kernels `#include`. |

`kernels/cache_flush.cl` and `kernels/include/batch_headers/fetch_data.cl` are the only
**added** CL files (see below). The four kernels are byte-for-byte copies (verify with
`md5sum`), so re-syncing after an OV change is just a `cp`.

---

## JIT faithfulness (why results transfer to OV)

OV does **not** compile these `.cl` files directly — it prepends a generated preamble of
`#define`s. `harness/ov_jit.py` reproduces that preamble **exactly**:

1. **Decoration macros** — `FUNC`/`FUNC_CALL`/`CONST_ARRAY_DECL`/`CONST_ARRAY_REF`/`KERNEL`,
   formatted identically to `CodeBuilder::decoration_macro` in OV's
   `graph/common_utils/jitter.hpp`.
2. **Per-kernel JIT constants** — one builder per OV generator in
   `gguf/fc_gguf_opt.cpp` (`FCGGUFOptGenerator`, `FCGGUFTranscodeGenerator`,
   `FCGGUFDp4aGenerator`, `FCGGUFPrequantGenerator`): `K_SIZE`, `N_SIZE`,
   `GGUF_BLOCK_ELEM`, `GGUF_BLOCK_BYTES`, `SG_SIZE=16`, `NROW` (Q6_K=4), `REQUANT_GROUP=32`,
   `TRANSCODE_TO_I4`, `QMAX`, the `GGUF_IS_*` flag, and the `make_type_jit_constants`
   `INPUT0_*`/`OUTPUT_*`/`TO_*_TYPE` family.
3. **Build options** — `-cl-mad-enable -cl-std=CL3.0`, matching
   `KernelGenerator::get_build_options` for an Intel device with work-group collectives.
4. **Dispatch geometry** — global/local work sizes taken straight from each generator's
   `get_dispatch_data_func` / the impl's explicit-args dispatch in `fc_gguf_opt.cpp`
   (GEMV: `global=[N*SG, M, 1]`, `local=[SG,1,1]`; dp4a: `global=[ceil(N/NROW)*SG, M, 1]`;
   transcode: `global=[ceil(N/SG)*SG, K/block_elem, 1]`; prequant: `global=[K/32, M, 1]`).

Block geometry (`GGUF_BLOCK_ELEM`/`BYTES`) mirrors OV core `element_type.hpp`
(`gguf_block_elem_count` / `gguf_block_byte_size`).

**Addressing note:** the GGUF FC activation `[BM,K]` and output `[BM,N]` are contiguous
row-major. OV reaches them via `INPUT0_GET_INDEX`/`OUTPUT_GET_INDEX`; for a contiguous
bfyx tensor (`FEATURE_NUM=1`) that reduces to `bm*K+k` / `bm*N+n` — exactly what the dp4a
and prequant kernels hardcode. `ov_jit.py` emits the reduced index macros, byte-identical
to OV for this primitive. `fetch_data.cl` is therefore an empty stub (the kernel's
`#include` is satisfied; no helper from it is referenced).

---

## Correctness oracle

`harness/gguf_ref.py` is a NumPy reimplementation of **every** per-format decoder, matching
the `.cl` field offsets and unpack math line-for-line. The IQ codebook/sign tables are
**parsed out of the verbatim `gguf_iq_tables.hpp`** (one source of truth — no copy). Random
weight bytes are always valid: each grid index is masked to the table size, so any byte
pattern decodes in range; only the f16 `d`/`dmin` scale fields are set to small controlled
values to keep dequantised magnitudes sane.

Pass criteria per kernel:
- **opt / dp4a GEMV**: relative-L2 error vs `A @ deq(W)^T` `< 2e-2` (f16 accumulation slack)
  and cosine ≈ 1.
- **dp4a** additionally checks the prequant int8 / scale against its own NumPy reference.
- **transcode**: reconstruction `q*scale` within ≤ 1.5 quant steps of the decoded weight,
  and per-group scale relative-L2 `< 5e-2`.

---

## Cache-flush strategy (accurate, repeatable timing)

A small weight (e.g. 12.6 MiB Q6_K 4096²) fits in the 18 MiB L2, so naive back-to-back
timing would either measure **L2 bandwidth** (≈ TB/s, not the DRAM the decode GEMV is bound
by) or — worse — let the GPU **downclock** while the host sets up the next launch.

Between every timed iteration the harness runs `cache_flush.cl`: a read-modify-write sweep
over a scratch buffer several × larger than L2 (default 128 MiB), repeated twice. This both
**evicts** any resident weight (forcing a DRAM re-read) and **keeps the GPU at boost clock**.

Validated A/B on this machine:

| shape (Q6_K)                 | weight   | no-flush      | flush         |
|------------------------------|----------|---------------|---------------|
| 4096×4096 (fits in L2)       | 12.6 MiB | 27.6 GB/s     | **86.1 GB/s** |
| 12288×4096 (exceeds L2)      | 37.7 MiB | 89.6 GB/s     | 89.2 GB/s     |

When the weight exceeds L2 the two agree (cache can't help anyway); when it fits, flushing
restores the honest ~86–89 GB/s DRAM figure instead of a 27 GB/s downclocked artefact.

---

## Timing & roofline

- **In-process**: OpenCL command-queue profiling events (`CL_PROFILING_COMMAND_START/END`)
  — pure device time, median over N iters.
- **External cross-check**: run under **cliloader** (`run.sh`) for an independent per-kernel
  device-time table that host overhead cannot skew.
- **Roofline** (`harness/hw.py`): reports achieved BW vs 456 GB/s peak, achieved GFLOPS vs
  the attainable ceiling `min(peak_compute, AI·peak_bw)`, and whether the kernel is
  memory- or compute-bound. The float decode path uses the FP32 ceiling (kernels accumulate
  in f32); the dp4a path uses the INT8 dp4a ceiling. All HW numbers are config/env overridable.

---

## Usage

```bash
cd .github/skills/dev_extract_kernels_from_ov

# Full sweep (Qwen3-class shapes, all formats, all 4 kernels) under cliloader:
./run.sh --config configs/default.json --out results/report.json

# Quick built-in sweep, no config:
python3 harness/bench.py --quick

# Custom shapes: copy configs/default.json, edit `cases`, then:
./run.sh --config configs/my.json
```

Each `case`: `kernel` ∈ {`opt`,`transcode`,`dp4a`}; `format` = one of the 10 baseline
`gguf_*` types; `K` (multiple of block_elem), `N`, `M` (opt/dp4a). Per-case overrides:
`iters`, `warmup`, `flush_l2`, `in_dt`/`out_dt` (`f16`/`f32`), `seed`.

Tune a kernel by editing `kernels/*.cl` in place, re-run, compare roofline %, then `cp` the
improved `.cl` back to the OV tree (and mirror any new JIT constant in both `ov_jit.py` and
the OV `*Generator::get_jit_constants`).

### Requirements
`python3` + `pyopencl` + `numpy`, an Intel GPU OpenCL runtime, and (optional) `cliloader`
from [intel/opencl-intercept-layer](https://github.com/intel/opencl-intercept-layer).

---

## Layout
```
dev_extract_kernels_from_ov/
├── README.md
├── run.sh                       # cliloader wrapper (falls back to in-process timing)
├── kernels/                     # verbatim OV kernels + minimal added CL
│   ├── fc_gguf_opt.cl           #   (verbatim)
│   ├── fc_gguf_transcode.cl     #   (verbatim)
│   ├── fc_gguf_prequant.cl      #   (verbatim)
│   ├── fc_gguf_dp4a.cl          #   (verbatim)
│   ├── cache_flush.cl           #   (added) L2 flush / keep-warm
│   ├── gguf/gguf_iq_tables.hpp  #   (verbatim) IQ tables
│   └── include/batch_headers/{common.cl (verbatim), fetch_data.cl (stub)}
├── harness/
│   ├── ov_jit.py                # OV-faithful JIT preamble + per-kernel JIT constants
│   ├── gguf_ref.py              # block model + NumPy reference decoders (oracle)
│   ├── hw.py                    # B580 capability + roofline model
│   └── bench.py                 # build / dispatch / flush / verify / roofline driver
├── configs/default.json         # Qwen3-class shape & format sweep
└── results/                     # JSON reports + RESULTS.md
```
