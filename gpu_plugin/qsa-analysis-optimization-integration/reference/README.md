# QSA CM kernels — standalone test harness

## Latest: PTL 64K/128K Q3 + top-k optimization

See [root-cause analysis and validated optimizations](PTL_Q3_TOPK_OPTIMIZATION_CN.md)
and [same-session full-pipeline A/B results](PTL_Q3_TOPK_AB_RESULTS.md).
The retained opt-ins are **KV-head-major GQA scheduling + vector metadata** and
**`fast-wg16-cached` exact top-k**. Kernel math, stable selection semantics and
existing defaults are preserved; no OpenVINO plugin changes are included.

- Full PTL A/B: `benchmark_ptl_optimized.py --phase prefill --size 131072 --warmup 20 --samples 20`.
- Single optimized path: `benchmark_long_context.py --phase prefill --size 131072 --topk-finalizer fast-wg16-cached --q3-head-major`.
- GQA edge suite: `benchmark_ptl_q3_variants.py --edge-only`.
- Exact top-k regression: `validate_topk_fast.py` (B580/B390 allowlist).
- Offline results: `analyze_ptl_optimization.py`.

Use the configured remote PTL Python 3.12 environment; do not run GPU tests
locally. Historical reports below retain their original measurements/protocols.

Every QSA CM kernel from
`openvino.mx/.../graph/impls/cm/qsa/` extracted so each can be compiled, checked
and timed **in isolation**, with shapes taken from the model `config.json`. Useful
for iterating on a single kernel without rebuilding the whole GPU plugin.
`test_qsa_pipeline.py` also validates and benchmarks the **actual sequential
Q0 → Q1 → Q2 → Q3 standalone slice**, using GPU-produced summaries and selections.

## Layout

```
cm_kernel/
  clops/            # minimal OpenCL + CM python wrapper (builds its C++ ext on first import)
  kernels/          # standalone QSA .cm kernels; Q3 DPAS has local optimizations
  include/          # the shared CM headers they #include
  qsa_config.py     # parses ../config.json -> QSA JIT constants
  qsa_harness.py    # source flattener (inlines "..." includes) + clops build/bench + paged-attn metadata
  qsa_ref.py        # numpy/fp32 references (scoring, top-k, attention, prepare)
  test_q*.py        # one driver per kernel: build -> run -> check vs ref -> time
  test_qsa_pipeline.py # complete sequential slice, persistent state, strict refs and opt-in A/B
  qsa_score_chunked.py # pure query-row planner + bounded Q2 scorer/finalizer/scratch owner
  validate_q2_chunked.py # exact same-kernel chunk/unchunk regression; no benchmark
  run_all.sh        # original per-kernel sweep; integrated driver is invoked separately
```

## Kernels covered

| stage | kernel | test | reference |
|---|---|---|---|
| Q0 | `qsa_kv_cache_update`      | `test_q0_kv_cache_update.py`      | paged K/V scatter |
| Q1 | `qsa_prepare`             | `test_q1_prepare.py`              | proj + RMSNorm + partial RoPE + summary pooling |
| Q2 | `qsa_score_topk_fused`    | `test_q2_score_topk_fused.py`     | `sum_h relu(q·s)/√Di` + top-k |
| Q2 | `qsa_score_tile_dpas`     | `test_q2_score_tile_dpas.py`      | block scores (systolic) |
| Q2 | `qsa_score_partition`     | `test_q2_score_partition.py`      | block scores (decode split) |
| Q2 | `qsa_topk_finalization`   | `test_q2_topk_finalization.py`    | radix top-k of a score row |
| Q3 | `qsa_sparse_attention`    | `test_q3_sparse_attention.py`     | sparse softmax attention (scalar) |
| Q3 | `qsa_sparse_attention_dpas`| `test_q3_sparse_attention_dpas.py`| same, systolic prefill |

## Running (on the remote BMG GPU only)

The CM kernels compile/run only on the Intel GPU with the CM frontend; the local
box has no XMX/CM compiler. On the remote:

```bash
cd /mnt/river/qsa/cm_kernel
python3 -m venv .venv && ./.venv/bin/pip install pybind11 numpy   # once
export CM_FE_DIR=/mnt/river          # so the CM JIT finds libclangFEWrapper.so
./run_all.sh 50                      # iters (sweeps 1024/2048/4096 each)
# or a single kernel:
CM_FE_DIR=/mnt/river ./.venv/bin/python test_q3_sparse_attention_dpas.py --sizes 1024 2048 4096 --iters 50
```

Each test sweeps token sizes and prints, per size, a correctness line plus a
roofline row:

```
[      Q3-attn_dpas] n=1024  mean=  1.4163 min=  1.3561 median=  1.4155 ms | GFLOP=  12.897 MB= 2.10 | roofline=  0.1086 ms (compute) | eff=  7.7%
```

* **mean / min / median** — averaged over `--iters` samples (the request asked
  for averaging; min/median are shown for context).
* **cache flush** — before every measured iteration a 96 MB scratch kernel is run
  (in its own `finish`, so it is not timed) to evict the GPU last-level cache, so
  the kernel's inputs are cold each launch and cache-hit interference is removed.
* **roofline** — `max(FLOP/118.784 TFLOP·s⁻¹, bytes/456 GB·s⁻¹)` for the B580,
  using minimal *unique* DRAM traffic (each datum once); `bound` says which term
  dominates. **eff = roofline / measured** — how close the kernel runs to the
  hardware lower bound (a small % ⇒ far from peak, e.g. sparse-gather overhead).

Common flags: `--sizes N...` (token sweep), `--pa-block` (page size, default 16),
`--block-topk` (default `budget/r` = 512), `--iters`, `--no-flush` (disable the
cache flush), `--check` (run the slow numpy reference).

### Q3 DPAS optimization validation

`test_q3_sparse_attention_dpas.py --baseline` disables the three load/merge
optimizations for comparison. `--check` now fails with a nonzero exit code on
incorrect output, retaining its original 2e-2 tolerance.

`validate_q3_dpas.py` runs 18 paged/chunked correctness regressions by default.
Add `--benchmark` for same-input, cold-cache A/B timing with alternating variant
order, or `--benchmark --ablation` to measure each optimization step.
See [Q3 DPAS optimization results](Q3_DPAS_OPTIMIZATION_CN.md) for measured B580
results, supported coverage, compiler limitations, and reproduction parameters.
These changes are standalone only; the OpenVINO plugin copy is unchanged.

## Prefill vs decode

Every test takes `--mode {prefill,decode,both}`. The swept `--sizes` mean:

* **prefill** — `size` = query tokens of one sequence, no history (`past=0`); the
  row shows `q=<size>`.
* **decode** — one new token on top of `size` tokens of KV history; the row shows
  `past=<size>` (this is the "history token" count). `--decode-batch B` replicates
  B such sequences to fill the GPU for a throughput view (default 1 = per-request
  latency).

Defaults follow the production dispatch: the systolic kernels
(`Q2-score_dpas`, `Q3-attn_dpas`) are prefill-only, `Q2-score_part` is decode-only,
the scalar `Q3-attn_scalar` (QSA's decode kernel) and the shape-agnostic
`Q0`/`Q1`/`Q2-topk` run `both`.

## Config → JIT

`qsa_config.py` reads `text_config` from `config.json`: `Hq=24, Hkv=2, head_dim=256,
Hidx=4, Di=128, rotary_dim=32 (=Di·0.25), r=4, budget=2048 → block_topk=512,
scale=1/√256`. Override page size / block_topk from the CLI.

## Notes

* `qsa_harness.flatten_source` inlines each local `"..."` include once (mimicking
  the plugin codegen) and prepends the JIT `#define`s, then clops compiles with
  `-cmc -Qxcm_register_file_size=256`.
* Selection metadata for the **isolated** Q3 tests is synthesized directly
  (`make_selection`). The integrated driver below never uses this shortcut.
* Reference precision: f16 inputs, fp32 accumulation, tolerances 2e-2 (scalar) /
  6e-2 (dpas) matching the in-kernel f16 dpas rounding.

## Integrated pipeline: explicit opt-in, not plugin deployment

`test_qsa_pipeline.py` exposes `Fixture`, `State` and `Pipeline`. Construct buffers
and maps once; `Pipeline.enqueue()` executes the entire slice without device
allocation, tensor copies or intermediate `finish()`. A later `Pipeline` can
reuse the same `State` with advanced past lengths and the same fixed page pool.
The CLI performs validation by default; any mismatch raises and exits nonzero.

| Stage | Default baseline | `--optimized` path |
|---|---|---|
| Q0 | existing paged K/V update | same |
| Q1 | scalar prepare | DPAS projection + FP32 prepare |
| Q2 prefill | original DPAS scorer, scalar finalizer, dense flags **0** | coordinated dense bypass; WG8 finalizer only if **every** query is dense, otherwise scalar |
| Q2 decode | partition512 scorer + scalar finalizer, dense flags **0** | partition16 scorer + WG16 finalizer, coordinated dense bypass |
| Q3 prefill | **CURRENT optimized** DPAS W4/HT4 (all three load/merge flags 1) | one-token/all-GQA-heads DPAS W4 |
| Q3 decode | scalar | split-P16 only for K512, B1/B4, one query/sequence, all pasts ≥2048; otherwise scalar |

This is an explicitly selected **standalone integration policy**, not a plugin
default, an automatically deployed model optimization, or a measured universal
crossover. `--compare` runs both paths with independent mutable cache/output/
scratch allocations. Immutable weights and equal input values are shared.
The baseline is **not** the historical unoptimized Q3 kernel.

Scope is deliberately restricted to the default B580 model geometry:
D2560, Hq24/Hkv2, Dh=Dv256, Hidx4/Di128/rot32, r4/PA16, gated causal, K512.
K0/K1 are additional safety tests, not performance recommendations. CLI now
accepts prefill/past lengths 1..65536, B1/B4, for subsequent validation/measurement;
acceptance is **not** a claim that large prefill has been tested or timed.
Historic integrated results were measured at decode **past4096** only.

### Reproduce on the specified remote GPU (serial commands only)

From the local `cm_kernel/` directory, sync **named changed files only**, never
the whole workspace, `.env`, model data, other repos, or `--delete`:

```bash
rsync -av -e 'ssh -o BatchMode=yes' test_qsa_pipeline.py README.md PIPELINE_OPTIMIZATION_RESULTS_CN.md openvino-ci-74@10.239.140.245:/mnt/river/qsa/cm_kernel/

# Both paths, full edge/continuity validation (no benchmark).
timeout 300 ssh -o BatchMode=yes openvino-ci-74@10.239.140.245 'cd /mnt/river/qsa/cm_kernel && env CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 .venv/bin/python test_qsa_pipeline.py --compare --full-ref'

# Complete slice, 100 warmups and 20 alternating samples per path/shape.
timeout 300 ssh -o BatchMode=yes openvino-ci-74@10.239.140.245 'cd /mnt/river/qsa/cm_kernel && env CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 .venv/bin/python test_qsa_pipeline.py --compare --benchmark-only --warmup 100 --samples 20'

# Run AFTER the previous command completes: cold96MiB, all-query NumPy Q3 refs.
timeout 300 ssh -o BatchMode=yes openvino-ci-74@10.239.140.245 'cd /mnt/river/qsa/cm_kernel && env CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 300 .venv/bin/python test_qsa_pipeline.py --compare --benchmark-only --flush --full-ref --warmup 100 --samples 20'
```

Flags:
- No mode flag: baseline only. `--optimized`: optimized only. `--compare`: both;
  the latter two are mutually exclusive.
- `--benchmark`: edge suite followed by benchmark. `--benchmark-only`: skip
  edge cases, **never** skip validation of each timed shape.
- `--validate-shapes`: validate the requested shapes without the edge suite,
  warmups or timing. Useful for prefill4096 / long-history decode correctness.
- `--score-budget-mib 128`: allocation ceiling including 64-float guards;
  accepts 1..128. Both pipeline variants use bounded **query-row** chunks.
- `--score-row-cap 31`: optional smaller row limit for tests. All candidate
  columns remain present. `--score-unchunked` uses the old kernel ABI for a
  manageable comparison only, refuses full scores above budget, and cannot
  combine with a row cap. The existing standalone helper defaults are unchanged.
- `--mode prefill|decode|both` (default both); `--sizes 1024 2048 4096`;
  `--decode-past 4096`; `--decode-batches 1 4`.
- `--warmup 100`, `--samples 20`, `--flush` (default no flush).
- `--full-ref`: NumPy attention for every query. Otherwise all queries for
  batches ≤64, 32 evenly spaced queries for larger batches, boundary queries,
  and **every** baseline/optimized changed-selection query. Q0/Q1, valid scores,
  all selected IDs/counts/padding and guards are always checked in full.

### Correctness and timing contracts

- Prefill starts with past0/empty caches. Decode history is generated once by
  **actual scalar Q0/Q1**, checked against NumPy, then cloned independently.
  No random raw/summary cache masquerades as prior calls. History initialization
  and all reference work are outside timing. External mixed/Q/K/V and weights
  are synthetic inputs, not model traces; the *selections* are actual Q1/Q2 outputs.
- The regression includes mixed empty/nonempty sequences, shuffled pages,
  incomplete compression blocks, visible2051/2052, K0 dummy storage, K1 exact
  ties, and true persistent prefill2051→decode2052→decode2053 continuity.
- Q1 references pool old half raw cache plus fresh FP32 projections; fresh
  projections are not incorrectly reloaded as half before pooling. Untouched
  cache slots and deterministic replay are checked bit-exactly.
- Every finalizer must match the deterministic exact selector of **its own GPU
  scores**. Dense rows leave scratch untouched (initial poison or previous chunk
  data) and are never selected through score reads; every sparse valid score must match its own iq/summary
  reference. Q1 rounding can change a near-tie selection: JSON reports changed
  rows/IDs, kth margins, score perturbation bounds and output magnitudes.
- Q3 must match NumPy using **its own actual selected blocks**, with fixed
  `atol=rtol=2e-3` (not the loose legacy per-kernel tolerances above). No changed
  rows are excused from this check, and no attention tolerance is widened to
  hide selection changes. Equal-selection A/B outputs also meet that tolerance.
- All dispatches use clops' single **in-order** queue. Each timed sample ends
  with one finish; with C row chunks expected event counts are baseline3+2C /
  optimized-prefill4+2C / optimized-split-decode5+2C. JSON stage means sum all
  chunks of a stage, not just the last chunk. Event sum excludes inter-event gaps. Host elapsed
  includes enqueue/dispatch gaps/synchronization, but excludes setup/copies/refs.
- Cold96MiB flush happens **before the whole slice**, never between stages, and
  is not timed. Raw samples, mean/median/P95, stage means and host samples are
  printed as JSON. Post-timing outputs/caches must equal pre-timing validation.
- Q2 uses one reusable `[chunk_rows,max_blocks]` scratch, **at most 128MiB
  including guards**, independent of total query rows. Each chunk scores its
  full history, finalizes global selected/counts, then reuses that SAME scratch.
  DPAS tiles are rebuilt per sequence/chunk before timing. No column chunk or
  global top-k merge is introduced. Validation re-scores/consumes each chunk
  outside timing and checks every own-score selection; no full host/GPU score
  matrix is allocated at 64K. Projection/split scratch is preallocated.
  This is the QSA slice, excluding external Q/K/V projections, GR, o_proj,
  scheduling/page growth, and the rest of the model.

### Helper defaults remain unchanged

`PrepareDPAS` is explicit; original Q1 driver needs `--dpas`.
`TopKFinalizer` defaults to **scalar**, with dense bypass enabled;
`PartitionScorer` defaults to **512** and score bypass **off**.
`Q3Attention` defaults to **scalar** (`partitions=None`);
`Q3GQAAttention` defaults to **scalar** (`enabled=False`).
Original prefill Q3 driver still defaults to its current optimized DPAS.
None of these settings propagate into OpenVINO or other repositories.

Measured integrated results, exact selection differences, error bounds and log
paths are appended in [the integrated results section](PIPELINE_OPTIMIZATION_RESULTS_CN.md#integration真实-q0q1q2q3-standalone-slice).
That report is historical and unchanged by row chunking; no large benchmark was
rerun. See [query-row scratch implementation and exact tests](SCORE_SCRATCH_CHUNKING_CN.md)
for APIs, memory calculations, remote commands, and large-run limitations.

## Seven-size long-context results and offline roofline audit

The later [long-context performance report](LONG_CONTEXT_PERFORMANCE_CN.md)
contains all 28 actual optimized measurements through 65,536 tokens: seven sizes,
prefill/decode, cold96MiB/no-flush, 100 warmups and 20 raw event samples each.
These later measurements supersede the historical large-run limitations above;
they do not change the dispatch policy or benchmark the baseline.

See [the reproducible roofline analysis](LONG_CONTEXT_ROOFLINE_CN.md) for all-size
theoretical/measured times, gap ratios, and six-stage references. Its explicit
whole-graph external-I/O and independent-stage conventions **replace the legacy
isolated-kernel byte assumptions for these integrated results**. Neither gap nor
`eff` is a measured hardware-utilization or attainable-speedup claim. Dense scores
are excluded; sparse decode attends 2049 positions, not the entire past context;
scratch allocation and cached radix rereads are not equated with DRAM traffic.

Run `python3 analyze_long_context_roofline.py` from this directory to print the
report, or add `--format json` for all 28 full-precision results and per-stage
no-flush metrics. Add `--check-report LONG_CONTEXT_ROOFLINE_CN.md` to verify the
saved report. `python3 -m unittest test_analyze_long_context_roofline -v` runs the
offline regression tests. No GPU, NumPy, network, environment variables or package
installation is needed; input logs and recorded kernel sources remain unchanged.

## Long top-k: validated full-pipeline opt-in

See [the final full QSA A/B/C results](TOPK_LONG_OPTIMIZATION_CN.md#9-完整-qsa-slice-同输入-abc最终协议-v2).
Measured on B580 with identical Q0/Q1/Q2-score/Q3 programs, real validated decode
history, one shared buffer set and serial execution: **100 warmups + 20 alternating
samples per variant/cache**. All-row own-score selections/counts match exactly;
full output bits, finite checks, guards and sampled own-selection NumPy Q3 pass.

|Measured default geometry, B1|Recommended explicit flag|Cold full-GPU old → new|
|---|---|---:|
|prefill 8192|`--topk-finalizer fast`|32.466 → 26.247 ms|
|prefill 16384|`--topk-finalizer fast`|86.890 → 59.785 ms|
|prefill 32768|`--topk-finalizer fast`|248.547 → 139.130 ms|
|prefill 65536|`--topk-finalizer fast-wg16`|839.001 → 358.253 ms|
|decode past16384, B1|`--topk-finalizer fast-wg16`|0.228736 → 0.105866 ms|
|decode past32768, B1|`--topk-finalizer fast-wg16`|0.428919 → 0.131742 ms|
|decode past65536, B1|`--topk-finalizer fast-wg16`|0.618861 → 0.156919 ms|

**Defaults remain legacy**, including `--optimized`. There is no `auto-fast`
policy, no inferred crossover between measured lengths, and no recommendation
for B4, other geometry, >64K or other devices. Omitted/`--topk-finalizer legacy`
preserves the existing caller policy; short/unmeasured workloads should retain it.
`scalar` and `cooperative` remain explicit alternatives.

`ChunkedQ2(..., finalizer='fast')` selects exact SIMD threshold/CAP256.
`finalizer='fast-wg16'` pins WG16 in both phases; `fast-wg8` pins WG8.
Existing `fast-wg` still inherits `wg` (integration prefill defaults to **8**),
so use the numbered alias to reproduce the WG16 table. Finite sparse scores
use exact ordered-bit keys, stable ascending-ID ties and ascending output IDs;
candidate-cache overflow **never truncates candidates** and falls back to full
scans. Dense/K0 paths need no scores. Score scratch remains ≤128MiB including guards.

Run only via `ssh -o BatchMode=yes openvino-ci-74@10.239.140.245`, from
`/mnt/river/qsa/cm_kernel`, with the serial prefix
`CM_FE_DIR=/mnt/river OPENBLAS_NUM_THREADS=1 timeout 900 flock -n /mnt/river/qsa/cm_kernel/.qsa_gpu.lock .venv/bin/python`:

- Full three-way A/B: `benchmark_topk_pipeline.py --phase prefill --size 65536`.
  Defaults: legacy/fast/fast-wg16, cold96MiB, 100/20. Add
  `--cache cold96MiB no-flush`; long decode uses `--phase decode --size 65536`.
- Selected single path: `benchmark_long_context.py --phase prefill --size 32768 --topk-finalizer fast`.
  Use `--topk-finalizer fast-wg16` for 64K prefill or measured long B1 decode.
- Regression: `test_qsa_pipeline.py --optimized --topk-finalizer fast-wg16 --score-row-cap 17 --full-ref`.

Raw per-event/stage/host mean, median, p95 and evidence live in
`topk_long_logs/pipeline_ab_*.log`; `pipeline_timing_audit.log` contains all final log
hashes and tables. `python3 analyze_topk_pipeline.py --check-sources` audits all
33 measurements / 660 slices / 9360 events. Run CPU tests with
`python3 -m unittest test_topk_finalizer_options test_analyze_topk_pipeline -v`.
Initial readback-interrupted trials remain in `topk_long_logs/initial_ab_readback/`
and are excluded from final results; all seven shapes were rerun without
inter-sample readbacks/logging. No timing samples were selectively dropped.

Historical reports/logs/scripts are unchanged. The old source-strict roofline
command above now correctly detects the three evolved Python files. Use
`python3 audit_long_context_history.py` (or `--test` for its original 12 tests)
to reconstruct their **exact original SHA256-verified bytes** in a temporary
source-root and reproduce the unchanged report. This preserves historical source
context rather than disabling hash validation or relabelling old measurements.

## Latest fast-top-k seven-size rerun (2026-09-10 16:14 UTC)

[Fresh performance and full distributions](LONG_CONTEXT_FAST_TOPK_PERFORMANCE_CN.md)
and [fresh roofline / historical comparison](LONG_CONTEXT_FAST_TOPK_ROOFLINE_CN.md)
use only `long_context_fast_topk_logs_rerun_20260910T1614Z/`: 14 shapes,
28 cache-protocol measurements, 100 warmups + 20 samples each. The initial
`long_context_fast_topk_logs/` attempt is excluded. No old timings substitute
for fresh measurements. Cold 64K prefill is **358.769638 ms** (reference
33.786624 ms, gap **10.62×**); decode is **0.240304 ms** (reference 0.025684 ms,
gap **9.36×**). Ratios use the original full-precision external-I/O model;
`ref_as_percent` is not hardware utilization. Fast scan traffic is unknown,
not the old four-pass radix count or the one-pass optimistic stage reference.

Explicit rerun policy (not new defaults):

|Phase|Size|Actual `benchmark_long_context.py` flag|
|---|---|---|
|prefill|1024, 2048|`--topk-finalizer legacy` (WG8)|
|prefill|4096, 8192, 16384, 32768|`--topk-finalizer fast` (LWS1)|
|prefill|65536|`--topk-finalizer fast-wg16`|
|decode|1024, 2048|`--topk-finalizer legacy` (WG16)|
|decode|4096, 8192, 16384, 32768, 65536|`--topk-finalizer fast-wg16`|

Use the serial SSH/environment/lock prefix above, followed by, for example,
`benchmark_long_context.py --phase decode --size 65536 --topk-finalizer fast-wg16 --warmup 100 --samples 20 --cache cold96MiB no-flush`.
Substitute phase/size/finalizer according to the table for all 14 shapes, in
prefill-ascending then decode-ascending order. Each protocol has its own 100
**unflushed** warmups; cold samples flush96MiB before each full slice. This is
not the A/B/C v2 warmup protocol. Q3 remains GQA-W4 for prefill, scalar for
decode1024, and split-P16+reduce for decode≥2048.
The archived `run_measurements.sh` records exact commands and idle checks but
intentionally refuses to overwrite its existing run; any future GPU rerun
must use a new output directory, not replace these logs. No GPU rerun is
needed for the offline analysis below.

- New report: `python3 analyze_fast_topk_roofline.py --check-report LONG_CONTEXT_FAST_TOPK_ROOFLINE_CN.md`.
  Add `--format json` for full precision, all six stages in both protocols,
  measured-source hashes and separately identified analysis-source hashes.
- New tests: `python3 -m unittest test_analyze_fast_topk_roofline -v`.
- Raw/performance audit: `python3 long_context_fast_topk_logs_rerun_20260910T1614Z/audit_measurements.py --format tables --check-report LONG_CONTEXT_FAST_TOPK_PERFORMANCE_CN.md`.
- Raw audit tests: `python3 -m unittest discover -s long_context_fast_topk_logs_rerun_20260910T1614Z -p test_audit_measurements.py -v`.
- Unchanged historical context/report/tests: `python3 audit_long_context_history.py --test`.

Historical cold→new cold comparisons are **separate runs, not controlled A/B**:
policy differs and actual clock/cache state is not controlled. Fresh cold decode
16K/64K **0.179940/0.240304 ms** is retained despite exceeding the earlier A/B/C
fast-WG16 numbers above. No unsupported DVFS/cache/fallback explanation is given.
The source snapshot and its 67 measured-file hashes remain immutable; newly
added postprocessing files are not retroactively inserted into that manifest.

## PTL Arc B390 iGPU sweep (2026-09-11)

Same standalone optimized pipeline, run on a **different, separately
authorized remote machine**: the PTL Windows box in `remote_machine.txt`
(Intel Arc(TM) B390 iGPU, Xe3, 96 EUs @2400MHz). Hardware peaks are the
user-given **58e12 FP16 FLOP/s / 110e9 byte/s**, not the BMG numbers. This is
**not a controlled A/B against the B580** — different architecture, different
power class, different run.

7 sizes (1024, 2048, 4096, 8192, 16384, 65536, 131072 — no 32768 this round,
131072/128K is new and larger than any prior BMG measurement) × 2 phases ×
2 cache modes, 28 rows, 0 skipped/missing. [Full report](PTL_LONG_CONTEXT_ROOFLINE_CN.md)
gives every shape's compute-ms/memory-ms/ref-ms (roofline = max of the two,
shown explicitly, not just the max), measured mean/median/p95, gap× and eff%,
per-stage breakdowns, and the exact finalizer/warmup/samples used per shape
(1024/2048 legacy, 4096–16384 `fast`, 65536/131072 `fast-wg16` for prefill;
decode uses legacy WG16 throughout). 65536/131072 use reduced warmup=20/
samples=10 where noted in the report, not the standard 100/20 protocol.

Windows-specific build/portability fixes needed to run at all on this
machine (kept **behind opt-in env vars, zero effect on the existing Linux
path**): `clops/cl/CMakeLists.txt` guards `-march=core-avx2` behind
`if(MSVC)`/`else()` (adds `/arch:AVX2` for MSVC), pins the per-config
`RUNTIME_OUTPUT_DIRECTORY` for multi-config generators (Windows has no
Ninja/Makefiles toolchain available here, only the multi-config Visual
Studio generator) so the built `.pyd` lands next to `clops/cl/__init__.py`
like the Linux `.so` does, and adds an opt-in `OPENCL_SDK_ROOT` env var for
an external OpenCL-SDK include/lib path (unset on Linux, so it is a no-op
there). `clops/cl/__init__.py` now also passes `-DPYTHON_EXECUTABLE=`/
`-DPython_EXECUTABLE=` as `sys.executable`, so the build always targets the
interpreter actually running it instead of whatever `python` resolves to
first on `PATH` (this fixed a real ABI-tag mismatch on this machine and is
harmless/no-op on Linux). `test_qsa_pipeline.py` and
`benchmark_long_context.py`'s device allowlist now accepts `B580` **or**
`B390` (still rejects any other/local GPU by name), and the accepted
`--sizes`/`--size` upper bound is raised from 65536 to 131072 for the new
128K shape.

A first prefill-131072 run that omitted `--topk-finalizer` fell back to the
default legacy-scalar path and measured a **much worse** cold total (6592 ms,
archived at `ptl_long_context_logs/prefill_131072_legacyscalar_default.log`,
not used in any table); the canonical 131072 result explicitly passes
`--topk-finalizer fast-wg16` like 65536. See §9 of the PTL report for the
exact numbers — this is a real illustration of finalizer choice mattering at
long context, not a hidden discrepancy.

- Report: [PTL_LONG_CONTEXT_ROOFLINE_CN.md](PTL_LONG_CONTEXT_ROOFLINE_CN.md).
- Reproducible analyzer (stdlib-only, no GPU): `python3 analyze_ptl_roofline.py`.
  Reuses `analyze_long_context_roofline.py`'s unmodified FLOP/byte model with
  only `PEAK_FLOPS`/`PEAK_BW` monkeypatched to the PTL numbers.
- Raw logs: `ptl_long_context_logs/{prefill,decode}_{size}.log`, one JSONL
  file per shape with ENV/PLAN/WORK/PASS/BENCH/DONE records (both cache modes
  in one BENCH pair per file, matching the existing schema).
- Correctness: own-score selection exact on every row of every shape;
  sampled NumPy score/attention checks in the report's §4.
