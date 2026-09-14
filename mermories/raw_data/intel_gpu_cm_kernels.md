# intel_gpu CM kernel gotchas (verified on Arc B580, 2026-09)

## Remote BMG test box
- Creds/workspace in `/home/ov2022/workspace/qsa/remote_machine.txt` (Arc B580, 32 cores).
- Helper `/tmp/rbmg.py` (paramiko): `run|raw|put|get|sync`. Never print the password.
- **CRITICAL BUG (cost ~1 hour, 2026-09-11 second session)**: `rbmg.py`'s `load()` naively
  parses EVERY `key: value` line in `remote_machine.txt` into one flat dict, but that file
  lists TWO machines (BMG then PTL) with the SAME key names (`ip`/`usr`/`pwd`/`workspace`).
  The dict silently ends up with the LAST (PTL) section's values, so `rbmg.py run/put/sync`
  connects to the **PTL Windows box**, not BMG - even though the tool "succeeds" with no
  error (it just does something harmless-looking on the wrong machine: e.g. `sftp.put` to
  an absolute Linux-style path on the Windows sftp server silently created an orphan
  `/mnt/river/...` tree unrelated to the real `D:\river\qsa` project, and a rebuild
  afterwards reports everything already "Built target X" because BMG's real source tree
  was never touched). Symptom to watch for: after `rbmg.py put/sync` + rebuild, the build
  log has ZERO real compile lines (all targets already "Built") even though you just
  edited a source file - that means the edit never reached the box.
  FIX: never use `rbmg.py` for BMG until it's patched to key credentials by machine name;
  until then use plain `sshpass -p <pwd> ssh/scp openvino-ci-74@10.239.140.245 ...` for
  BMG (matches how PTL work already used direct `sshpass` in the same session). Verify any
  remote sync with an immediate `stat`/`grep` for a distinctive new string, from a SEPARATE
  ssh call, before trusting it and kicking off a multi-minute rebuild.
- For BMG-bound rebuild+test loops, prefer explicit `sshpass ssh ... "setsid nohup make ...
  > /tmp/x.log 2>&1 < /dev/null & disown; echo LAUNCHED"` (returns instantly) over piping
  through `tail` inside a `timeout`-wrapped foreground call - `cmd | tail -N` buffers ALL
  output until the pipeline's stdout closes, so if `timeout` kills the wrapper at the
  deadline the instant after `make` itself finished, you see nothing and exit code 124,
  even though the build (and even the link) already fully succeeded on disk. Poll the log
  file with a quick separate `tail -c N /tmp/x.log` call instead.
- MUST `export CM_FE_DIR=/mnt/river` (libclangFEWrapper.so lives there) or CM JIT fails.
- Build: `cd build-x86_64-release && make -j28 ov_gpu_unit_tests` (~2-4 min incremental).
- pyopencl venv at `/tmp/cmvenv` for standalone CM builds with `-cmc`.
- Local dev box (UHD 630, Gen9) cannot build CM at all and has no XMX -> no dpas testing.

## Getting CM compiler errors (the plugin hides them)
`CL_BUILD_PROGRAM_FAILURE` carries no log. Do:
`OV_GPU_DUMP_SOURCES_PATH=/tmp/cmdump/` then rebuild that dumped `.cm` with pyopencl
`-cmc` to get real diagnostics. `OV_GPU_VERBOSE` did NOT surface the build log.

## Multi-stage CM impls share ONE program (batched)
All stages of a primitive with the same build options are concatenated into a single
compilation unit. Consequences for shared headers:
- `#pragma once` does NOT work (codegen inlines per kernel, then concatenates).
- Use a CLASSIC include guard; `kernels_db_gen.py::detect_guard_patterns` recognizes
  `#ifndef X` + bodyless `#define X` and skips the auto `#undef`.
- Macros with a BODY (e.g. `#define ATTR [[type("svmptr_t")]]`) get an auto `#undef X`
  appended per kernel, so they MUST be defined OUTSIDE every include guard - put them in
  each `.cm`, otherwise kernels 2..N see them undefined ("expected ')'" in the signature).
- Existing repo headers have no guards because no two kernels of one primitive include
  the same function-defining header.

## CMake codegen dependency bug (fixed)
`impls/cm/CMakeLists.txt` globbed only `include/*.h`, but every CM header is `.hpp`,
so header edits never retriggered the codegen -> kernels silently used a stale inlined
copy. Now globs `*.h` and `*.hpp`. If in doubt: `rm -rf build/.../impls/cm/codegen`.

## cldnn multi-output primitives
- Pass `num_outputs` AND same-sized `output_data_types` / `output_paddings` to
  `primitive_base`, else `get_output_layout(1)` yields a default f32 layout.
- Tests need `config.set_property(ov::intel_gpu::allow_new_shape_infer(true))`, otherwise
  the legacy single-layout `calc_output_layout()` path is used.
- Read secondary outputs via `network.get_primitive(id)->output_memory_ptr(i)`
  (`outputs.at("id.1")` throws `map::at`).
- Implementation manager should override `query_formats()` for all in/outputs.

## PrimitiveImplOCL/CM runtime params
- `make_deep_copy()` sets `m_rt_params = nullptr` -> accessors must lazily recreate it.
- Override `update_rt_params(const primitive_inst&)` (base `execute()` calls it) and read
  i32 inputs from the INSTANCE; `kernel_impl_params::memory_deps` is empty for static shapes.
- `get_internal_buffer_descs()` must size from STATIC layouts, not runtime state: buffers
  are allocated per shape while page-table metadata changes every iteration.
- Scalar push order in `get_dispatch_data_func()` must match the kernel signature exactly;
  a mismatch silently zeroes outputs (e.g. num_tokens==0 -> every thread returns).

## cm_dpas facts (for QSA prefill work)
- `RepeatCount` (M) may be 1/2/4/8 - NOT necessarily 8. `pa_small_q.cm` uses
  `Q_ROWS = Q_head_chunk_size * TILE_Q <= 8`. M=1 still gives 256 MAC/instr vs ~3.2 for
  a `cm_sum` dot product, so small M is not a reason to avoid dpas.
- N is FIXED to the GRF width (`REG_N = CM_GRF_WIDTH/32`; 16 on Xe2, 8 on Xe1). N is the
  axis that must be filled with independent outputs -> in prefill use the QUERY tile.
- Operands: `cm_dpas(acc, src1=B[K][N] VNNI, src2=A[M][K] row-major)` -> `dst[M][N]`.
- fp16 B in VNNI == a TRANSPOSED d32 2D block load of row-major [N][K] memory:
  `lsc::block_2d_desc<uint,1,REG_N,REG_K/2>` + `cm_load<lsc::Transpose>`
  (see cm_pa_xe2.hpp:116). No register shuffle needed.
- `cm_load<lsc::VNNI>` does the re-layout in HW for the V operand.
- Register transposes: `Transpose_8x8` / `Transpose2DMatrix` in cm_attention_common.hpp
  (that header needs CMFLA_* macros, so copy the butterflies rather than including it).
- 2D block descriptor ctor: `(ptr, Height-1, Width_bytes-1, Pitch_bytes-1, blk_x, blk_y)`;
  `block_x` is in ELEMENTS. Surface height clamps out-of-range rows to zero (free masking).
  Surface width should be >= 64 bytes.
- Accumulator budget is the real limit: `rO = NQ * head_dim * 4 B` is 8 KB at Dv=128,
  i.e. the whole 128-GRF file -> a FlashAttention-style Q3 needs pa_multi_token's
  worker/SLM head-dim split. Do not attempt it without the hardware A/B + spill loop.

## Intra-work-group GLOBAL memory round-trip races (cost 1 debug cycle)
Symptom: a kernel passes in isolation and 12/12 on one run, then fails intermittently
(different tests each run) with deterministic input data.
Cause: threads of a WG cooperatively wrote a global buffer, then `cm_fence(); cm_barrier();`
and read back rows written by peers. A bare `cm_fence()` does NOT guarantee cross-thread
global visibility on this driver (the SLM rule `cm_fence(CM_LOCAL_BARRIER)` before every
barrier is the analogous, separate requirement).
Fix that worked: do NOT read back peer writes at all - split the kernel in two dispatches
(produce / consume) so the ordering is an event dependency. Bonus in QSA: the prefill
scorer now reuses the decode `qsa_topk_finalization` selector, so both paths select
identically by construction. 4x repeat runs green afterwards.

## Wide dpas stores overrun row strides (cost 3 debug cycles)
A tile-wide `cm_ptr_store<float, 8>` at the last tile of a row writes up to 7 entries past
the logical row length. If the row stride is not a multiple of the tile width the store
lands in the NEXT row and corrupts it (symptom: the first token of a work-group tile is
wrong, everything else fine). Fix: pad the runtime row stride to a multiple of the dpas
tile in BOTH `update_rt_params` and `get_internal_buffer_descs` - all kernels take the
stride as a scalar so they stay consistent, and padded entries are never read.

## Finite -inf sentinels break masked online softmax
With `NEG_INF = -3e38` and `running_max` initialised to the same value, a query whose
whole tile is masked gets `new_max == sentinel` and `exp(sentinel - sentinel) == 1`, i.e.
masked positions receive FULL weight. Always `e.merge(0.0f, St.row(i) <= NEG_INF)` after
the exp instead of trusting the arithmetic.

## Test design: QSA-style top-k tests are vacuous unless topk << n_complete
Two independent traps, both of which silently made the whole indexer untested:
- If `block_topk >= n_complete` every block is selected and the selection pipeline
  (projection, RMSNorm, RoPE, scoring, top-k) is a no-op no output comparison can see.
  A deliberately WRONG RoPE convention passed the entire suite because of this.
- `score = sum_h relu(q_h . k_b)` zeroes many blocks exactly, so the top-k boundary is
  often an exact tie; FP64 reference vs FP32 dpas then legitimately pick different blocks.
  Compare per token and skip tokens whose top-k margin is below ~2%, and assert that at
  least half the tokens survived the filter.

## Qwen3.8 RoPE is NEOX/rotate_half, not interleaved
`emb = cat((freqs, freqs))`, pairs are `(j, j + rd/2)`, NOT `(2j, 2j+1)`. Cannot be fixed
by permuting the cos/sin table alone - it changes which channels pair up. Keep the table
as rd/2 packed (cos_j, sin_j) pairs; only the channel pairing differs.

## Head-dim worker split unlocks dpas at head_size 256
`rO` is `q_tile * Dv * 4` = 16 KB at Dv=256, i.e. the entire 256-GRF file, so one thread
cannot own a head. Pattern (from cm_pa_xe2.hpp `enable_head_size_partition`): W threads
share one (query tile, head); worker w owns K dims [w*Dh/W, ...) and V dims [w*Dv/W, ...).
QK is a reduction over K, so partial St tiles are summed through SLM; mask + online
softmax are then recomputed identically by every worker (cheaper than a second barrier);
PV and the epilogue stay in each worker's V slice.
- Make the WG EXACTLY one (tile, head). Putting several tiles in one WG deadlocks: their
  loop trip counts differ, so the threads reach different numbers of barriers.
- Double-buffer the SLM exchange (`slot = step & 1`) so one barrier per step suffices - a
  thread can be at most one barrier ahead, hence never writes the buffer being read.
- Also note CM caps a single `matrix<>` at 16384 bytes (PA splits rO into lo/hi for that).

## dpas tiles may span several paged-attention pages
Do not require `page_rows % tile_rows == 0`. When a page is smaller than the tile (real
Qwen3.8: 16/4 = 4 summaries per page vs an 8-row A tile), assemble the tile from
`tile_rows / page_rows` sub-tiles, one 2D block load per page. Guard sub-tiles past the
last valid row: their page id is outside the sequence's page list, so indexing the page
table would read out of bounds.

## Selection-margin thresholds must be set by the STORED precision
Q1 writes indexer queries and block summaries back to the caches as f16, so GPU scores
carry ~1e-3 relative noise vs an FP64 reference - not the ~1e-5 of the FP32 dpas itself.
Threshold 2e-2. And require an ABSOLUTE number of cleanly-separated tokens (~8), never a
fraction: with a thousand candidate blocks the gap between the k-th and (k+1)-th score is
naturally a few percent, so most tokens sit near the boundary and a fraction-based rule
fails on correct code. For single-token decode tests, redraw the data instead.

## Measuring spill and per-kernel cost (verified recipe)
- Spill/asm: `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=/tmp/igc <binary>`. CM kernels land
  in `VC_asm*_<kernel>_cm_*.asm`. Read the `//.spill` header lines: `spill size N` means GRF
  spill; `spill flag store/load` is only predicate-register spill and is cheap. Also grep
  `^\s*(spill|fill)` for actual instructions. `-mCM_printregusage` produces nothing on this
  driver, and the max `rN` in the asm is useless (r255 is reserved).
- Per-stage timing: temporarily wrap `execute_stage` in a lambda that does
  `stream.finish(); t0; execute_stage(); stream.finish(); t1` under an env var.
  MUST also loop `network.execute()` ~10x and take the MIN - single-shot times were 2-3x
  inflated by first-run overhead (prepare 5.8 -> 2.2 ms, score_topk_fused 9.2 -> 4.6 ms).
- A/B: add `if (getenv("QSA_NO_DPAS_X")) return false;` to the capability gate so one binary
  can alternate paths; run A/B/A/B interleaved (the B580 throttles).
- unitrace exists but `--device-timing` gives no per-kernel breakdown for OpenCL (Level Zero
  only). It also needs `LD_LIBRARY_PATH=<oneapi>/advisor/*/lib64` for libsvml.so.
- Best bottleneck tool is ABLATION: delete one stage of the kernel, rebuild, re-time. An
  opcode histogram of the asm is misleading - `add/mov/shl` dominated the histogram in the
  QSA dpas attention kernel, but removing that work changed nothing.

## Optimizations that measured ZERO and cost spill (do not redo)
On `qsa_sparse_attention_dpas`, all three of these were neutral in time and together added
1536 bytes of GRF spill (extra live vectors), so they were reverted:
- caching the k-way merge heads in a live `vector<int32_t,16>` instead of re-reading;
- hoisting the page-table lookup / `kv_cache_offset` out of the head-dim loops;
- padding empty tile slots with block 0 to make the K/V loads branch-free.
Lesson: on a kernel bound by data movement, cutting scalar ALU/branch/load counts does
nothing, and the extra live registers can push a fitting kernel into spill.

## LSC 2D block load: a block row is capped at 64 BYTES
`lsc::block_2d_desc<half, 1, ROWS, COLS>` with COLS=64 (128 B/row) COMPILES and runs but
returns wrong data - silent corruption, no diagnostic. COLS=32 halfs is the limit. PA and
SDPA all use 16-half (32 B) rows but with 16 ROWS, so their message is 512 B.
Corollary for sparse attention: when the tile is only `r`=4 rows tall (scattered blocks),
grow the message with COLUMNS, and chunk at 32 halfs. One `r x 32` message per block per
chunk replaced one per dpas tile: 64 -> 16 messages per step, asm 16858 -> 8837 lines.

## QSA Q3 dpas: the real limiter is per-query selection, not the kernel
Measured on the production shape (Dh=Dv=256, 4096-token prefill), min-of-10:
  scalar 17.2 ms | dpas 22.3 ms (baseline) -> 20.3 ms (after wide loads)
Ablation of the dpas kernel: PV dpas ~0, QK dpas 0.16 ms, V loads+transpose 10.4 ms.
The dpas math is under 1% of the runtime; the kernel is pure data movement.
ROOT CAUSE: QSA selects blocks PER QUERY, so a 16-query dpas tile must walk the UNION of
16 independent top-k sets. With ~500 candidate blocks and topk=4 the union is ~60 blocks,
i.e. ~15x the useful work (measured: 4.56 GFLOP executed vs 0.30 GFLOP useful). No amount
of message/branch/address tuning recovers a 15x amplification - three such attempts each
measured exactly 0.
FIX DIRECTION: see "fixed by tiling the N axis over HEADS" below - implemented, 1.55x
faster than the scalar kernel.

## QSA Q3 dpas: fixed by tiling the N axis over HEADS (1.55x faster than scalar)
The dpas N axis must hold columns that share the A operand (the KV tile). Filling it with
16 query TOKENS forced the union of 16 independent per-token top-k sets = ~15x work, and
was 1.3x SLOWER than the scalar kernel no matter how the inner loop was tuned.
Fix: fill N with (head, token) pairs - HT heads x TT tokens, HT*TT = 16.
- All heads of a token share its selection, so the union shrinks to TT selections.
- Under GQA the n_rep = Hq/Hkv heads of a group share the same K/V, so K/V is read once
  for HT heads where the scalar kernel re-reads it per head.
- HT must divide 16 AND divide n_rep (else a head group straddles a KV head).
  n_rep=12 -> HT=4; n_rep=2 -> HT=2; n_rep=1 -> HT=1 (degenerates to the old tiling).
- Columns are HEAD-MAJOR (n = hh*TT + t) so one transposed d32 load of TT query rows fills
  TT contiguous columns; the Q tile needs HT such loads per k-tile instead of 1.
Measured (Dh=Dv=256, 4096-token prefill): 22.3 -> 20.4 (wide loads) -> 11.6 ms, vs scalar
17.2. With production head counts (Hq=24, Hkv=2, 1563 tokens): dpas 13.9 vs scalar 21.6.
Zero GRF spill throughout.

## cm_prefetch on QSA Q3 dpas is a 2.3x REGRESSION (do not redo)
Tried the cm_pa_xe2.hpp trick of prefetching the NEXT union step's K/V (added
`qsa_dpas::prefetch_wide` + a per-step lookahead that replays the TT-way merge on
a cursor COPY and issues `cm_prefetch<Cached,Cached>` for those blocks, behind a
`QSA_PREFETCH` compile toggle). Correctness: all 27 prefill tests still PASS.
Perf (roofline Q3_sparse_attention, prefill): 1024 tok 48.5->112 ms, 2048 tok
209->495 ms == 2.3-2.4x SLOWER. Fully reverted (git clean, re-measured 49/211 ==
baseline).
WHY: this kernel is bound by message-ISSUE-rate (see "LSC 2D block load" +
"real limiter is per-query selection" notes). cm_prefetch ADDS LSC messages that
fight for the same issue slots, and the lookahead re-runs the merge (extra scalar
+ selected_blocks gathers). Prefetch only helps latency-bound kernels WITH spare
message bandwidth; PA gets away with it because its dpas math is a real fraction
of runtime, QSA's is <1%. Right direction stays FEWER/WIDER messages, never more.
A/B recipe: `#define QSA_PREFETCH 0/1`, `make -j28 ov_gpu_unit_tests`, run
roofline configs 0(pf1024)/2(pf2048) with `export CM_FE_DIR=/mnt/river`.

## Standalone per-kernel CM harness (cm_kernel/, validated 2026-09-09)
Extracted all 8 QSA kernels into `qsa/cm_kernel/` for isolated compile+check+time,
shapes from `config.json` (Hq24/Hkv2/Dh256/Hidx4/Di128/rot32/r4/topk512).
- Uses a local `clops` pkg (cm_kernel/clops, cl C++ ext builds on first import via
  cmake+pybind11). API: `cl.tensor(np)`, `cl.kernels(src, "-cmc -Qxcm_register_file_size=256")`,
  `k.enqueue(name, GWS, LWS, *tensors, *int_scalars)`, `cl.profiling(True)`, `cl.finish()`->ns list.
- `qsa_harness.flatten_source` inlines local `"..."` includes ONCE + prepends JIT
  `#define`s (KERNEL_NAME=<name>); system `<...>` left alone. That's all the CM FE needs.
- Files: qsa_config.py, qsa_harness.py (bench, PagedBatch/make_batch, make_selection,
  make_score_tile_map, make_prepare_map, make_indexer_inputs), qsa_ref.py (numpy refs),
  test_q0..q3*.py, run_all.sh, README.md.
- RUN ONLY ON REMOTE BMG (local has no CM/XMX). Recipe:
  `cd /mnt/river/qsa/cm_kernel; export CM_FE_DIR=/mnt/river; ./run_all.sh 512 30`
  (venv at cm_kernel/.venv has pybind11+numpy; system python is PEP668-managed).
  config.json must be at /mnt/river/qsa/config.json (../config.json from cm_kernel).
- All 8 pass. Perf @512 tok prefill (min GPU ms): Q0 0.002, Q1 2.87, Q2fused 0.20,
  Q2dpas 0.003, Q2part 0.06, Q2topk 0.22, Q3scalar 1.43, Q3dpas 0.40.
- v2 (roofline sweep): bench() now averages N iters (mean/min/median) with a 96MB
  cache-flush kernel between measurements (own finish, not timed); qsa_work.py gives
  per-kernel FLOP/byte (minimal UNIQUE dram traffic) -> roofline = max(FLOP/118.784TF,
  bytes/456GB) vs BMG peak; report() prints eff=roofline/measured. Sweep 1024/2048/4096
  via --sizes. Correctness ref is O(tok*heads*pos) pure-python -> GATED behind --check
  (default OFF; intractable at >=1024 for Q3). run_all.sh <iters>.
- Roofline eff @4096 (mean ms | eff): Q0 0.039|95%(mem, near-peak BW), Q1 19.0|0.6%
  (scalar proj, no dpas), Q2score_fused 3.6|0.8%, Q2score_dpas 0.068|41%(dpas),
  Q2part 0.33|~0(decode, launch-overhead bound, roofline~0), Q2topk 2.1|1.7%,
  Q3scalar 65.4|2.0%, Q3dpas 14.1|9.3%. dpas Q3 ~4.6x faster than scalar. Q3 is
  COMPUTE-bound in roofline (KV unique tiny); sparse-gather overhead => low eff.
- GOTCHA: roofline bytes MUST be minimal-unique (each datum once) or measured beats
  roofline (eff>100%); first cut counted KV per-access -> 172% bug.

- v3 (prefill/decode split): every test takes --mode prefill|decode|both and
  --decode-batch B (default 1). prefill: (0,size) size=query tokens; decode:
  B*(size,1) size=KV history (1024/2048/4096). report() tags rows [name phase]
  with q=/past=. make_mode_batch() in harness. Defaults match dispatch: dpas score
  + dpas attn = prefill-only, score_partition = decode-only, scalar attn + Q0/Q1/
  topk = both. Decode Q3-scalar latency @past 1024/2048/4096: 0.28/0.56/0.57 ms
  (1 token, 24 heads -> small grid, launch/BW bound). Decode single-token is
  launch-bound for most kernels (roofline~0); use --decode-batch for throughput.

## KEY GOTCHA (cost 1 cycle): n_complete = (abs_pos+1)//r, NOT abs_pos//r (a block is
  complete when all r positions <= abs_pos). Applies to make_selection + refs + tail.
- Q3 dpas dispatch: workers=4, heads_per_tile=4 for this config (n_rep=12);
  GWS=[(Hq/ht)*workers, num_tiles*ht, 1], LWS=[workers,1,1].

## Known pre-existing failures (not caused by QSA/GR work)
`vlsdpa_gpu_test.mixed_input_runtime_pitches_*` (7 tests) fail with
"Reference output has zero magnitude" - reproduced with the changes reverted.

## Standalone Q1 Stage2 (2026-09-10)
- `qsa_project_dpas.cm`: M8 output channels/N16 tokens/K16, shared transposed-d32 VNNI loaders, token-major FP32 scratch; `QSA_PREPROJECTED=1` prepare consumes scratch, retains fresh raw K in FP32 SLM pooling. Default scalar ABI/driver unchanged; opt in with `test_q1_prepare.py --dpas`.
- `validate_q1_dpas.py` captures actual scalar FP32 projection for edge tests, handles old raw cache correctly (existing prepare_ref does not), and times both dispatches. B580 no-flush Q1 1K/2K/4K: scalar 5.033/9.491/18.880 ms vs DPAS 0.182/0.414/0.890; cold 5.216/9.668/19.013 vs 0.219/0.442/0.906. Full details appended to PIPELINE_OPTIMIZATION_RESULTS_CN.md. Stage1 full regression passed; Q3/shared DPAS header unchanged.
- clops `cl.tensor(empty_numpy)` fails on zero-byte copy: allocate a dummy element for unused empty inputs and do not dispatch empty batches. Avoid changing shared stage1 helpers just for empty-batch tests.
- Q1 DPAS requires D>=32, D%32=0, J%8=0, Di%16=0; tested D2560/Di128 and D96/Di48. Projection FP32 error ~1.7e-6 vs scalar on small tests, final half output differences up to 0.001953125 vs scalar; no top-k bit-exact/model-quality claim.


## Standalone Q2 Stage3 (2026-09-10)
- New `qsa_topk_cooperative.cm`: exact 4x8-bit private histograms + leader SLM reduction, WG8/16 (8336/16656 B SLM); contiguous-range count/prefix/emit preserves ascending IDs and exact threshold ties, no candidate cap. Dense/K0 zero reads; instrumentation verifies sparse 6*n reads. Two full strict runs and unchanged Stage1 regression passed.
- Explicit-only helpers in `qsa_topk_cooperative.py`; existing kernels/defaults unchanged. B580 B1 sparse decode WG16 + partition16 total Q2 no-flush past4K/16K/64K: 0.068437/0.140093/0.513926 ms vs scalar512 0.715332/2.134036/7.919400. Partition change is granularity tuning, NOT cooperative scoring. 64K P16/P32 essentially tied; no auto threshold.
- Dense prefill1K/2K WG8 faster; mixed4K prefill KEEP SCALAR (1.776ms vs WG8 1.849/WG16 2.442). Never auto-enable WG across all rows. Full 100 warmup/20 alternating no-flush+cold statistics appended to PIPELINE_OPTIMIZATION_RESULTS_CN.md.
- 100 short-iteration warmups did not prevent first-run 1K absolute-time anomaly; retain raw logs and repeat with long decode measured first, do not assume DVFS without telemetry.
- For append-only report verification, compare original byte length: old Stage1/2 report was 26075 bytes WITHOUT trailing newline; splitting at an appended Markdown delimiter adds a newline and yields a misleading hash mismatch.

## Standalone Q3 Stage4 (2026-09-10)
- `qsa_attention_split.py` Q3Attention defaults to scalar (`partitions=None`); explicit P1..16 dispatches new split producer + reducer, private exact scalar helper copy in `qsa_split_span.hpp`. Old scalar/DPAS kernels unchanged. Partition selected SLOT counts, last partition owns tail, reducer skips l=0 before exp/acc loads and gates once.
- B580 default config B1/B4 past2048/4096/16384: P16 fastest of P1/2/4/8/16 in no-flush/cold/repeat. First no-flush scalar→P16 B1: .450693→.038624, .451671→.038703, .457932→.039343 ms; B4 .486380→.119416, .518567→.121093, .627114→.142437. Short past1/3 ALL split paths slower; retain scalar. P1 adds overhead, not a cheap fallback. No auto route or Stage5.
- Strict 28 cases + P3/P15, exact zero-Q partition l/tail ownership, reducer NaN/-inf-empty states, and unchanged original 18 Q3 cases passed. Worst split output error vs FP32/scalar .000244140625. r16/DhDv256 compile warns 3136B spill for BOTH original scalar and split producers; correctness only, no speed recommendation.
- Full results appended to PIPELINE_OPTIMIZATION_RESULTS_CN.md, Stage1/2/3 original prefix 47719 bytes SHA d29a4a996f83bfcd4a8b5f5729eca977efd681fe21f9a725978c62fcb494c7ac preserved. Local/remote 39 QSA source/report hashes agree. P16 scratch B1 396288B/B4 1585152B. B4/16K absolute timings varied ~10–11% across runs; report full sweeps, not minima.


## Standalone Q3 Stage5 (2026-09-10)
- New `qsa_sparse_attention_gqa_dpas.cm`: one token/KVhead WG, N16 all nrep heads padded, W4 Dh/Dv slices32-half aligned, single sorted list (no union), 8192B double-buffer St SLM. Query transposed descriptor rows=nrep/pitch=headstride. `Q3GQAAttention` defaults scalar; explicit enabled=True, unsupported error or explicit scalar fallback. Old files unchanged.
- B580 default Hq24/Hkv2/D256/r4/PA16/K512, valid SERIAL 100/20 no-flush optimized-DPAS→GQA: past0 q1024 .878833→.752145, q2048 3.163447→2.740974, q4096 12.267104→9.542557, q8192 41.694765→23.621302ms independent; past16K q128 2.344760→.584796, q512 5.318109→2.060088. Cold/reverse confirm direction; keep opt-in, no unmeasured crossover. 29 new cases twice + unchanged original18 passed; worst GQA FP32 .000376284/scalar+optimized .00048828125.
- Extra OLD comparator limits: Dh128/Dv256/W4 optimized IGC Floating point exception reproducible (new GQA/scalar pass; validator explicitly omits old comparison there); old HT16/TT1 load_vnni_cols<1> expects uint16 vs uint8, use old HT8/TT2 comparator for nrep16, don't edit old source. Ungated padded headstride272 half passes with 32-byte helper alignment.
- Never parallelize terminal GPU benchmark calls, including through tool wrappers. Stage5 duplicate initial two sweeps were discarded, idle verified, all final measurements rerun serially. 1K absolute timing varied even after100 warmups/reverse; retain samples, no DVFS claim without telemetry.
- Stage5 report appended preserving68025-byte prefix SHA5fa001e05720e361cca6c06cd752e51475ca51e1363351e486cd8c27810cc60a; final report87831 bytes. Full36 valid benchmark rows and two old compile limitations documented. No real model traces, no Stage4 decode changes.


## Standalone final integration (2026-09-10)
- `test_qsa_pipeline.py`: actual Q0→Q1→Q2→Q3, explicit --optimized/--compare; baseline CURRENT optimized prefill DPAS Q3/scalar decode. Independent persistent states; real scalar Q0/Q1 history outside timing. Full reference edge9 + timed5 shapes: 9338 queries/path, exact metadata and all-query own-selection NumPy pass. No old kernels/helpers or other repos changed.
- Q1 half rounding changes 3/4096 prefill selections (tokens2735/2791/3421), kth gaps 1.64e-5..1.38e-4. A/B attention max .00223326683 is legitimate selection change, NOT a reason to widen Q3 tolerance. Every own-score selector exact and own-selection Q3 atol=rtol=.002; worst ref .00032120943. Empty boolean masks need explicit dtype=bool (np.array([]) defaults float).
- B580 serial100/20 complete pipeline no-flush GPU baseline→opt ms: prefill1024 6.454503→1.010185;2048 13.692805→3.255930;4096 33.572555→12.195325;decode past4096 B1 1.356638→.120107,B4 1.929326→.242184. Cold + host + exact margins in appended report/README. Full scores still all-token scratch, not chunked; no model/plugin quality/deployment claim.
- Integration report changed only title + appended section, original body87779B SHA cdaa0785a03f7593e4942cedba1d7479e3af64d6c4ea804f74159852b1874fc4 verified. Mem0 localhost8888 unavailable; repository memory used instead.


## Standalone Q2 row-bounded scratch (2026-09-10)
- qsa_score_chunked.py pure plan + ChunkedQ2 owner: full history columns, query-row chunk -> score -> finalize -> reuse SAME scratch, in-order, no enqueue allocation/copy/finish. 128MiB cap INCLUDES guards; no full score readback in integrated validator. Global iq/query/selected/count persistent. Never use old session Explore column-chunk proposal.
- Four Q2 kernels default0 QSA_SCORE_ROW_CHUNK appends row_begin/end (+DPAS tile_offset); maps are sequence/chunk-safe, valid query rows clamp to row_end, local scratch row/global metadata. Prior TopKFinalizer/PartitionScorer unchanged.
- B580 final148 exact same-kernel cases PASS (K0/1/512, scalar/WG16, bypass on/off, partition/DPAS, caps1/3/15/16/17/31, boundaries2051/2052, changed-input scratch reuse+guards); original dense/cooperative full suites PASS. Integrated edge9 cap17 full-ref PASS; prefill4096 cap31 full-ref PASS 127232B/133chunks; real65536-history decodeB1 cap1 PASS65792B. Automatic1MiB and old-ABI opt-out4096 also PASS. No large benchmarks yet; see SCORE_SCRATCH_CHUNKING_CN.md.
- Plan guard edge: clamp available row count to >=0 BEFORE NT0 dummy/guard budget validation; otherwise insufficient empty-batch budget can yield negative row dimensions.


## Full seven-size optimized pipeline measurements (2026-09-10)
- LONG_CONTEXT_PERFORMANCE_CN.md + long_context_logs/{prefill,decode}_*.log: remote B580 only, serial 14 shapes, cold96MiB AND no-flush each100 warmup/20 samples, all measured through65536. No baseline or kernels changed. benchmark_long_context.py reuses actual Pipeline; streamed ALL Q1 references, ALL own-score top-k rows, sampled NumPy Q2/Q3 incl chunk boundaries; real DPAS Q0/Q1 decode history outside timing.
- Cold prefill GPU ms1K/2K/4K/8K/16K/32K/64K:1.013883/3.254846/12.231065/32.477904/86.862772/248.551320/839.175710. Cold B1 decode same past sizes:.310701/.073211/.157023/.186742/.271122/.437800/.591835 (1K scalar Q3,>=2K split16).
- 64K prefill:134152448B score allocation including256B guard;2047rows x16384cols,33chunks,70events. Q2topk537.895496ms dominates,Q3 266.356692ms. Own-score exact all65536 rows;NumPy score98/Q3 104rows;full402653184 output elements finite;score err1.907e-6,Q3 err.000241339. Decode64K true optimized history GPU14.920728ms outside timing;fully validated65536 Q1 rows. No inferred roofline efficiencies.
- audit.txt recomputed264 stats triplets /560 slices /6960 events and84 report rows; hashes match. Driver stdout contains NUL: use grep -a for filters; raw JSON lines parse normally.


## Offline long-context roofline audit (2026-09-10)
- analyze_long_context_roofline.py is stdlib-only; 28 BENCH/560 slices/6960 events, 56 ENV source hashes, original 28 mean rows checked. Emits deterministic LONG_CONTEXT_ROOFLINE_CN.md or full JSON (168 stage comparisons); --check-report checks exact text. CPU unittest test_analyze_long_context_roofline:12 pass; python -S CLI passes.
- Global model: useful Q1+Q2 sparse+Q3 dot FLOPs/P plus explicitly conventional cold external-I/O max; NOT guaranteed DRAM lower bound. Stage references count logical inputs once + outputs, not allocation or radix rereads. eff is reference ratio, never hardware utilization/speedup. 64K cold prefill33.786624ms vs839.175710 (24.84x); decode.025684ms vs.591835 (23.04x).
- Actual decode unique2049, history2048; prefill union not saved, mandatory dense-prefix+own-tail subset U=3N/4+512 (N>2051), upper N. Either U endpoint leaves prefill Q3 compute dominant. Scalar topk reads4n+emit (K..n), WG reads6n; cannot use DRAM B for every read as rigorous bound.
- Historical long_context_logs/SHA256SUMS paths are relative to cm_kernel, not log dir; keep manifest unchanged after append. Original performance report28715B prefix SHA c5989fc00ce04401394c831b286746a0a3f2b4e700cc15487d1768a65d01d3a2 preserved;19 other manifest entries unchanged. apply_patch Add may omit final newline; exact report test caught it, fixed via explicit added blank line.


## Full top-k pipeline A/B final (2026-09-10)
- benchmark_topk_pipeline.py: one shared Fixture/State/scorer/scratch/outputs; only finalizer program+WG changes. Real DPAS Q0/Q1 decode history, all-row own-score oracle then SAME-score chunk replay all variants, full-bit payload SHA+finite/guards and sampled Q3 before/after.
- Final v2 protocol: serial B580 flock,100 cache-matched warmups/20 alternating samples; NO readback/logging between samples. Initial final-round per-variant readbacks caused legacy64K last sample909ms and are archived in topk_long_logs/initial_ab_readback (never quote as final).
- Cold GPU legacy→best ms prefill8K32.466102→26.246716,16K86.889569→59.785345,32K248.546678→139.129548 (fast);64K839.001299→358.252814 (fast-wg16,topk537.895158→57.097653). DecodeB1 past16K.228736→.105866,32K.428919→.131742,64K.618861→.156919 (fast-wg16). No inferred between-point crossover; default unchanged/no auto-fast.
- Explicit ChunkedQ2 finalizer legacy/None retain caller policy; fast-wg inherits wg, numbered fast-wg8/16 pin it. Both integration CLIs accept aliases. Final tests52 finalizer/288helper/148originalchunks/9full-refWG16/6CPU pass.
- analyze_topk_pipeline.py audits7raw logs,33BENCH,660slices,9360events + hashes; derived audit is pipeline_timing_audit.log (keep outside pipeline_ab_*.log raw namespace). Historical reports/logs/analyzer/tests untouched. audit_long_context_history.py restores3evolved Python files IN MEMORY via reverse-deltas; all56 exact historical hashes required, temp source-root reproduces unchanged report+12tests. Original direct analyzer appropriately rejects evolved current sources.


## GR (gated_residual) kernels extracted to standalone cm_kernel/ + PTL perf (2026-09-11)
- Copied `gr_read.cm`/`gr_write.cm` verbatim from
  `openvino.mx/.../graph/impls/cm/gated_residual/` into `cm_kernel/kernels/`.
  `cm_kernel/include/qsa_gr_common.hpp` ALREADY existed and is byte-identical to the
  plugin's copy (no need to (re)copy the header).
- New `cm_kernel/gr_ref.py`: FP64 numpy reference (`rbar_ref`/`gr_read_ref`/`gr_write_ref`)
  mirroring `gated_residual_gpu_test.cpp`'s `reference_rbar`/`reference_read` + the WRITE
  combine `R'[c,d]=R[c,d]+S[c]*Y[d]`.
- New `cm_kernel/test_gr_gated_residual.py`: standalone build+check+roofline-sweep for
  BOTH kernels (`--mode read|write|both`), no PagedBatch/qsa_config needed (GR has no
  paged-attention notion, just `rows`). Defaults = production shape from
  `gated_residual_gpu_test.cpp::run_point`: C=4, D=640 (hidden_size=2560), L=320, eps=1e-6,
  GR_WG_SIZE=16 (matches `gated_residual.cpp`'s `gr_wg_size`), produce_write_gate=True.
  Dispatch mirrors the plugin exactly: read GWS=[16,rows,1]/LWS=[16,1,1], write
  GWS=[D/16,C,rows]/LWS=[1,1,1]. Tolerances match the gtest: mixed/gate 2e-2 abs,
  write-combine 1e-2 abs (both measured well under, ~1e-3).
- Remote PTL (Windows, `10.239.132.213`/`Local_Admin`, see remote_machine.txt) has NO
  paramiko helper like BMG's `/tmp/rbmg.py`; it DOES run a plain OpenSSH server, so
  `sshpass -p <pwd> ssh/scp Local_Admin@10.239.132.213 "<cmd>"` / `'D:/river/...'` paths
  work directly (cmd.exe syntax, use `cd /d D:\...` to change drive+dir in one ssh call).
  Python is `D:\river\py312\Scripts\python.exe`; clops rebuilds its pyd via MSBuild
  automatically on first import after new files land (no explicit CM_FE_DIR needed on
  this Windows box - the `.env` file in cm_kernel is an unused Linux-only template).
- Correctness: ALL rows in {1,8,64,512,2048,4096,8192} PASS for both kernels (mixed_err/
  gate_err/write_err all ~3e-4 to 1e-3, write-gate correctly stays in (0,2)).
- Perf on PTL Arc B390 (96 EU @2400MHz, 58 TFLOP f16, 110 GB/s), D=640/C=4/L=320,
  cache-flushed and no-flush give IDENTICAL numbers (confirms gr_read is NOT memory-bound
  by DRAM-cache-hit interference - it's scalar-loop compute bound, no dpas):
  rows(GR-read ms): 1->1.44, 8->1.44, 64->3.38, 512->21.6, 2048->83.7, 4096->167, 8192->333
  (~40.7 us/row asymptotically; roofline eff only 0.1-0.5% - same class of inefficiency as
  the scalar Q1 projection kernel, expected since gr_read has no dpas/systolic path at all).
  rows(GR-write ms): 1->0.002, 8->0.004-0.006, 64->0.02-0.03, 512->0.14-0.20,
  2048->0.54, 4096->1.08, 8192->2.15 (memory-bound, eff ~9-10%, scales linearly/cleanly).
- No dpas/optimized variant exists yet for GR (this session only extracted+benchmarked
  the EXISTING plugin kernels as-is); if GR shows up as a real bottleneck in an end-to-end
  profile, the down/up low-rank projections (L x CD and CD x L GEMV per row) are the
  obvious dpas-conversion target, same pattern as `qsa_project_dpas.cm`.

## GR dpas optimization: gr_read (5-kernel dpas pipeline) + gr_write_wide (2026-09-11)
Extends the standalone GR extraction above. gr_read.cm/gr_write.cm themselves are
UNCHANGED (kept as the default scalar path); all new work is opt-in additional files,
same convention as every other dpas variant in this repo.

### gr_read -> GRReadDPAS (cm_kernel/gr_read_dpas.py), ~50x faster at scale
KEY INSIGHT: gr_read's down-proj (Rbar[tok,CD]@Wdown[L,CD]^T) and up-proj
(T[tok,L]@Wup[CD,L]^T) are EXACTLY qsa_project_dpas.cm's M8/N16/K16 GEMM shape
(token-major activations x channel-major weight) - copy that kernel's structure almost
verbatim, just swap D->CD/CD->L and add the SiLU/sigmoid activation before the store.
BLOCKER + fix: the transposed-load dpas trick (`qsa_dpas::load_vnni_tile`) needs a REAL
global/SVM pointer (LSC 2D block descriptors don't work on SLM), but gr_read.cm's Rbar
lives in SLM and is read across THREAD boundaries for the GEMM - exactly the documented
"intra-WG global round-trip race" gotcha (a single dispatch cannot safely produce Rbar
then dpas-consume it). Fix: split into 5 SEPARATE kernel dispatches on the same in-order
queue (mirrors qsa_prepare_dpas.py's PrepareDPAS class exactly):
  1. gr_rbar_dpas.cm: Rbar[tok,CD] to GLOBAL (literally gr_read.cm's Phase1a/1b unchanged
     except the final store target - low risk, subset of proven code).
  2. gr_down_proj_dpas.cm: T[tok,L]=SiLU(Rbar@Wdown^T/C) half, dpas GEMM (M=L/8 tiles,
     N=16 tokens, K=CD/16). Direct copy of qsa_project_dpas.cm's structure.
  3. gr_up_proj_dpas.cm: gate[tok,CD]=sigmoid(T@Wup^T) half, same structure, K=L now.
     Materializes the FULL read-gate tensor to global - the ORIGINAL fused kernel
     explicitly avoided this, but the extra traffic (~2*tokens*CD*2 bytes) is negligible
     next to the dpas speedup (verified: ~1-6% eff decrease at most, never the bottleneck).
  4. gr_writegate_dpas.cm (optional, only if produce_write_gate): tiny GEMM with
     RepeatCount=C(4) instead of 8 - cm_dpas RepeatCount may be 1/2/4/8, doesn't have to
     be 8. No ready 4x16->16x4 transpose helper (transpose_8x16 is 8x16-specific only) so
     it's a plain unrolled scalar copy loop (64 elements, negligible cost).
  5. gr_combine.cm: mixed[tok,D]=(1/C)*sum_c gate*Rbar - pure elementwise/bandwidth
     kernel, no dpas left to apply once the two GEMMs above ran.
Shape requirements (validated on production C=4/D=640/L=320): CD>=32 && CD%32==0,
L>=32 && L%32==0 (both 2D-surface constraints, same as Q1's D%32==0), L%8==0, CD%8==0
(dpas M-tile divisibility), C%2==0 (write-gate half-pair packing).
Tolerance: 6e-2 abs (repo's standard scalar-vs-dpas gap) though measured error was only
~5e-4 to 1.1e-3 in every test - far under budget.
MEASURED on PTL (Arc B390, cold+no-flush identical since scalar path is compute-bound,
not cache-bound): rows 1/8/64/512/2048/4096/8192, scalar->dpas ms:
  1.46->0.087 (16.8x), 1.43->0.087 (16.4x), 3.40->0.117 (29x), 21.7->0.389 (55.9x),
  83.8->1.63 (51.4x), 166.8->3.31 (50.4x), 332.9->6.55 (50.8x).
Speedup GROWS with rows (16x at rows=1 up to ~55x at rows>=512) because the 5 kernel
launches' fixed overhead (~0.08ms total) dominates at tiny rows but the O(CD*L) GEMM
FLOPs dominate at scale - dpas throughput advantage compounds there. Roofline eff of the
dpas pipeline is ~6-9% (bandwidth-bound on the 3 materialized scratch tensors, by design
trade-off) vs scalar's 0.1-0.5% (pure ALU-bound, no dpas at all) - both far from peak but
the ABSOLUTE dpas numbers are what matters here, not the eff%.
All correctness at every rows tested (1,7,8,16,17,64,512,2048,4096,8192) incl. explicit
partial-tile (rows%16!=0) and no-write-gate paths: PASS on first compile, no debug cycles
needed (unusual for this repo - the close mirroring of the already-validated
qsa_project_dpas.cm pattern paid off).

### gr_write -> gr_write_wide.cm, ~3-4x faster at scale (rows>=64)
Root cause identified from gr_write.cm's dispatch `[D/16, C, rows]` with one thread per
(d-tile, BRANCH, row): every branch's thread independently re-reads the SAME
`sublayer_y[row, d0:d0+16]` and re-derefs the same gate pair from DRAM - Y traffic was
inflated by a factor of GR_BRANCHES(4) for no reason, on top of D/16=40 threads/row/branch
each doing only a 16-half load/store (dispatch-overhead-bound at small D).
Fix (gr_write_wide.cm, JIT GR_WRITE_TILE default 64): dispatch `[D/TILE, rows]` (branch
loop moved INSIDE the kernel), so Y and the full C-branch gate vector are each read
exactly ONCE per thread (one `cm_ptr_load<int,C/2>` for all 4 gate halfs instead of C
separate overlapping pair-loads), and the D-tile widens from 16->64 elements via the
existing `qsa_gr::load_hv<TILE>`/`store_hv<TILE>` helpers - cuts thread count by
GR_BRANCHES*(TILE/16) = 16x.
MEASURED on PTL: rows 1/8/64/512/2048/4096/8192, scalar->wide ms:
  0.0020->0.0026 (small rows: launch-overhead bound, wide is NOT faster - expected, same
  pattern as every other "wide tile" opt in this repo), 0.0040->0.0024 (1.7x),
  0.0209->0.0055 (3.8x), 0.1432->0.0363 (3.9x), 0.5414->0.1407 (3.8x),
  1.0773->0.3467 (3.1x), 2.2458->0.7814 (2.9x). Correctness verified 1e-2 abs tol
  (same as scalar gr_write) at rows 1/7/64/512, error ~1e-3 throughout.

### Test CLI additions (test_gr_gated_residual.py)
`--read-variant {scalar,dpas,both}` and `--write-variant {scalar,wide,both}` (both
default "both", i.e. runs and compares all variants side by side). `--combine-tile`/
`--write-tile` expose the elements-per-thread JIT tile size (default 64, must divide D).
gr_ref.py unchanged (same FP64 reference serves both scalar and dpas correctness checks).

### Not done / follow-ups
No OpenVINO plugin integration yet (this session only touched cm_kernel/, mirroring the
scope of the prior GR-extraction turn). If/when integrated, mirror the QSA Q1
project_dpas pattern: capability-gate on the shape constraints above, keep scalar
gr_read.cm/gr_write.cm as the always-safe fallback, add the 5 new kernels' generators +
`can_use_dpas_gr()`-style predicate, and size the 3 new internal scratch buffers
(Rbar/T/gate, all `[tokens, *]`) similar to QSA's PROJECTED buffer.

## GR dpas pipeline integrated into the OpenVINO CM plugin (2026-09-11, third session)
Integrated the cm_kernel/ GR dpas work above into
openvino.mx/.../graph/impls/cm/gated_residual/{gated_residual.cpp,.hpp}. Branch
river/qsa_gr_impl. Old gr_read.cm/gr_write.cm's KERNEL NAMES are unchanged, but
gr_write.cm's BODY was rewritten in place (wide/deduped algorithm fully replaces the
old one - no capability gate needed, strictly faster or equal at every valid shape, so
unlike the dpas read path there is no reason to keep two write kernels side by side).
- New files: gr_rbar_dpas.cm, gr_down_proj_dpas.cm, gr_up_proj_dpas.cm,
  gr_writegate_dpas.cm, gr_combine.cm (copied verbatim from cm_kernel/kernels/,
  byte-identical), gr_stage_events.hpp (new, mirrors qsa_stage_events.hpp).
- `GatedResidualInternalBufIdx{RBAR,T_SCRATCH,GATE_SCRATCH}` added to gated_residual.hpp;
  always allocated (mirrors QSAInternalBufIdx's unconditional-buffer convention) sized
  from STATIC layouts in a new `get_internal_buffer_descs()` override.
- `can_use_dpas_gr_read()` (mirrors qsa.cpp's can_use_dpas_projection): arch>=xe2,
  C*D>=32 && %32==0, lowrank>=32 && %32==0, C%2==0 (write-gate half-pair packing).
  GatedResidualCMImpl ctor picks gr_write (WRITE mode) / the 5-stage dpas pipeline
  (READ|FINAL_MIX + capability holds) / scalar gr_read (fallback) - purely a STATIC,
  shape/arch-derived choice at construction time, no runtime re-decision needed (unlike
  QSA's use_split_attention which is genuinely per-request-dynamic).
- gr_write.cm rewrite: GR_WRITE_TILE JIT constant picked by `gr_best_tile(hidden_size)`
  = widest of {64,32,16} that evenly divides hidden_size (16 always valid per
  validate_impl's hidden_size%16==0). Same `gr_best_tile()` helper sizes gr_combine's
  GR_COMBINE_TILE. New WRITE test `write_combines_branches_narrow_tile` (D=48) exercises
  the tile=16 fallback branch (existing D=64 test only ever hit tile=64).
- New READ test cases added to the existing `INSTANTIATE_TEST_SUITE_P(smoke,
  gated_residual_read_test, ...)` list (same fixture/tolerance, zero new test
  infrastructure needed): `{32,64,32,true}` (exact 2x16 N-tile boundary),
  `{37,96,32,false}` (partial last N-tile, no write-gate), `{41,48,64,true}` (partial
  tile + GR_COMBINE_TILE=16 fallback). The PRE-EXISTING `{7,128,32,true}` case also
  became dpas-eligible after this change (D=128,L=32 satisfy the predicate) and now
  exercises the new pipeline "for free". 16/16 gated_residual + 51/51 combined
  qsa+gated_residual regression: PASS.
- **Debugging cost ~1.5h, two real bugs found**:
  1. `add_stage()` swallows exceptions from `get_kernel_data()` into a
     `GPU_DEBUG_TRACE_DETAIL`-only log (silently drops the stage from `_order` on
     failure) - so a broken stage manifests as a LATER, confusing
     "Kernel for {<primitive_id>} is not found in the kernel cache!" exception at
     EXECUTE time, not a clear error at the failure site. The message's `{name}` is the
     primitive's topology id (`params.desc->id`), NOT the CM kernel entry-point name -
     don't chase the literal string it prints.
  2. ROOT CAUSE of the above: **CMake's `file(GLOB ...)` for `.cm` kernel sources is
     cached and does NOT pick up newly-ADDED files just by re-running `make`** - only a
     full `cmake` reconfigure re-evaluates the glob (confirmed: the build log's CM
     codegen step listed "processing gr_read.cm"/"processing gr_write.cm" but NOT any of
     the 5 new files, until `cmake .` was re-run in the existing build dir). Symptom:
     build succeeds with 0 errors, `[100%] Built target ov_gpu_unit_tests`, but new
     kernels silently don't exist in the compiled binary. This is DIFFERENT from the
     already-documented "CMake codegen dependency bug" note above (that one was about
     `include/*.h` vs `*.hpp` header globs never including `*.hpp` at all; this one is
     about a glob that DOES include `.cm` correctly but is STALE after adding new files
     of an already-included pattern). FIX: after adding any brand-new `.cm`/`.hpp` file
     (not just editing an existing one), run `cmake .` in `build-x86_64-release/` before
     `make`, or just always use the full `build.sh` (which always reconfigures).
     Editing an EXISTING file needs no reconfigure (make's normal mtime tracking
     suffices) - this only bites when the FILE SET changes.
- Fixed the pre-existing roofline test bug this session's changes would have made worse:
  `gated_residual_gpu_test.cpp`'s `time_primitive_ms()` used
  `network::get_executed_primitives()`, which (like QSA before it added
  qsa_stage_events.hpp) only exposes the LAST dispatched kernel's event for a
  multi-kernel primitive. For the 5-kernel dpas pipeline this silently measured only
  gr_combine's tiny memory-bound tail (~5e-5 ms!) instead of the whole call. Added
  `gr_stage_events.hpp` + a `GatedResidualCMImpl::execute()` override (previously this
  impl relied entirely on the base class's default sequential-stage chaining) that
  explicitly dispatches stages in the fixed rbar->down->up->[writegate]->combine order
  AND records each one's event; `time_primitive_ms()` now sums `executing_ms()` over
  every recorded stage per iteration (exact mirror of qsa_gpu_test.cpp's
  `time_qsa_stages`/`executing_ms`). Verified fix: prefill 1024/2048/4096 production
  shape (D=640,C=4,L=320) on BMG Arc B580 now read 1.00/1.53/3.02 ms (sane, scales with
  tokens) vs the bogus ~5e-5 ms before the fix; decode (1 row) ~0.04ms. Scalar gr_read
  and gr_write paths are single-stage so were never affected by this bug.
- Sync gotcha (cost ~1h, separate from the two above): `/tmp/rbmg.py`'s credential
  loader is broken for this specific file (see "Remote BMG test box" section at top of
  this file for the full writeup) - used PLAIN `sshpass ssh/scp` for every BMG file
  transfer and build/test invocation in this session instead.

## Fast-topk seven-size remeasurement (2026-09-10 UTC)
- LONG_CONTEXT_FAST_TOPK_PERFORMANCE_CN.md + long_context_fast_topk_logs_rerun_20260910T1614Z/: all14shapes/28sets,100warmup20samples/560slices6960events; explicit shortlegacy, prefill4–32Kfast,64Kfast-wg16,decode4K+fast-wg16; originalQ3scalar1K/split16>=2K unchanged. No existing source edits; warmups noflush, cold measured samples flush96MiB (existing long-context protocol, not ABCv2).
- 64K cold prefillGPU358.769638/topk57.090920ms;decodeGPU.240304ms. Every all-row ownscore/metadata/guard/finite/replay passed;64K Q3NumPy104rows only. All event/host stats + explicit dispatch/source hashes independently checked by log-dir audit_measurements.py. 28raw protocol_json capsules, measurements.json, exact67source tar snapshot; no new roofline computed.
- Fresh regressions52finalizer/288helper/18integrated +21CPU tests pass. Prior performance/roofline/logs preserved. For new roofline maintain original unique-I/O references but never apply old fixed6nradix diagnostic to fast algorithm.
- Safety gotcha: fuser GPU owner opencl-language can be idle. Initial sweep safely stopped, archived separately; restarted ALL shapes after checking language service all drm-cycles-* ==0 before/after each shape under whole-sweep flock. No user processes killed. Use ps comm/PID rather than broad commandline searches (editors may put connection credentials in argv).
