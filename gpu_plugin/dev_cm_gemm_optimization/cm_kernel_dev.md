# CM (C-for-Metal) kernel development notes (ovmx workspace)

## Key facts
- Real CM kernel examples live in:
  /mnt/river/ovmx/openvino.pipeline.mx/thirdparty/openvino/src/plugins/intel_gpu/src/graph/impls/cm/*.cm
  Syntax: `#include <cm/cm.h>`, `extern "C" _GENX_MAIN_ void name(args [[type("svmptr_t")]])`,
  `vector<T,N>`, `matrix<T,R,C>`, `cm_ptr_load<T,N>(ptr, byte_offset)`, `cm_ptr_store<T,N>`,
  `cm_group_id(dim)`, `cm_local_id(dim)`, `cm_local_size(dim)`, `cm_group_count(dim)`,
  `cm_slm_init/alloc/read/write`, `cm_barrier()`.
- **CM kernels CAN be built/tested locally with pyopencl** on this machine's Intel NEO driver:
  `cl.Program(ctx, cm_source).build(options='-cmc')` works directly (confirmed on Arc B580).
  Kernel args declared `svmptr_t foo [[type("svmptr_t")]]` bind directly to plain pyopencl
  `cl.Buffer` objects via `krn.set_args(buf, ...)` -- exactly like OpenCL. No separate CM
  compiler/toolchain needed for standalone testing (a `cmc` binary does exist under
  /tmp/cm-workbench-sdk-*/pkg/usr/bin/cmc but is NOT required for this workflow).
- Real OpenVINO GPU plugin build flag for CM: options string is ` -cmc ` (see
  src/plugins/intel_gpu/src/graph/impls/cm/utils/kernel_generator.cpp), some kernels add
  `-Qxcm_register_file_size=256`.
- CM has NO OpenCL-style sub-group lanes: a single CM thread computes a full
  vector<T,16> (or wider) directly using explicit SIMD vector ops. This means porting an
  OpenCL `intel_sub_group_block_read`-based kernel to CM collapses the OPG(=16)-wide local
  dimension entirely: one CM thread replaces 16 OpenCL work-items in a sub-group.
- IMPORTANT layout insight: the existing "SG-transposed" weight packing used by the OpenCL
  q4k/q5k/q6k GEMV kernels (pack_qk_pqs_sg, pack_qk_psl_sg, pack_q5k_pqh_sg, pack_q6k_*)
  already stores, for a fixed (h, bid, j, c), 16 CONTIGUOUS lane values. This is exactly what
  CM wants for `cm_ptr_load<uint,16>(...)` (one contiguous vector read, no shuffle needed).
  So when porting OpenCL sub-group GEMV/MoE kernels to CM, KEEP the same Python packing
  functions unchanged; only rewrite the kernel (.cl -> .cm) and the dispatch/build code in
  the test harness (build with `-cmc`, drop the OPG local-dim, thread grid = groups count).

## CM correctness pitfall: SLM reduction race with plain cm_barrier()
When doing a producer/consumer SLM reduction (each thread in a group writes a
partial result to SLM, then thread 0 barriers and reads all partials), a bare
`cm_barrier()` is NOT always sufficient on this driver/hardware -- observed a
genuine non-deterministic race (correct on most runs, silently wrong on ~1/4
runs, only for the heavier Q6_K dense GEMV kernel, only at one specific
KSPLIT value) when doing:
    cm_slm_block_write(slm, ...); cm_barrier(); ... cm_slm_block_read(...);
Fix: always add `cm_fence(CM_LOCAL_BARRIER);` immediately before `cm_barrier();`
after the SLM write:
    cm_slm_block_write(slm, ...);
    cm_fence(CM_LOCAL_BARRIER);
    cm_barrier();
This matches the pattern used in real kernels (cm_pa_xe1.hpp, cm_sdpa_common.hpp
always pair cm_fence(CM_LOCAL_BARRIER) with barriers). Apply this to EVERY CM
kernel that does an SLM write -> barrier -> SLM read reduction pattern (dense
GEMV KSPLIT>1 path, and by extension the MoE GEMV KSPLIT>1 kernels too).

## Workspace kernel folder history (do not be confused by this)
- /mnt/river/ovmx/ocl_gguf_kernel/ = the FIRST (incorrect) attempt: these are actual OpenCL
  (.cl) kernels mistakenly created when asked to "convert SYCL kernels to CM kernels". They
  were renamed out of cm_gguf_kernel/ to ocl_gguf_kernel/ to make room for the real CM port.
  Useful only as a reference for weight layouts / dequant math / test structure.
- /mnt/river/ovmx/cm_gguf_kernel/ = the correct target: real .cm CM kernels + matching
  Python tests (built with -cmc), converted from /mnt/river/ovmx/sycl_gguf_kernel/.
- /mnt/river/ovmx/q4k_moe_gemv/test_moe_gemv_sg_kernels.py is the reference test template
  the user wants matched for style (numpy ref dequant, roofline harness, PASS/FAIL summary).

## STATUS (as of this session): CM port COMPLETE and hardware-validated
All 10 kernels ported from sycl_gguf_kernel/ -> real .cm files in
/mnt/river/ovmx/cm_gguf_kernel/, all passing on Intel Arc B580 (verified stable
across 5+ repeated runs each):
  - gemv_q4k_sg.cm / gemv_q5k_sg.cm / gemv_q6k_sg.cm (dense GEMV)
  - gemm_q4k_slim.cm / gemm_q5k_slim.cm / gemm_q6k_slim.cm (dense slim GEMM)
  - moe_gemv_q4k_sg.cm / moe_gemv_q5k_sg.cm / moe_gemv_q6k_sg.cm
    (group_up_gate_* + down_merge_* kernels each)
  - moe_gemv_q80_sg.cm (shared_gate_up_q8_0 + shared_down_merge_q8_0)
Tests: test_dense_gemv.py, test_dense_gemm_slim.py, test_moe_gemv.py (all in
cm_gguf_kernel/), built with `-cmc`, dispatch = (rows/OPG, KSPLIT, ...) threads
with local=(1,KSPLIT,1) -- no OPG lane dimension (CM has no sub-groups).

## Other CM syntax/API gotchas hit during this port (in addition to earlier notes)
- No `size_t` type in CM -- use `uint` (or explicit uint casts) for all
  pointer-offset arithmetic.
- `cm_ptr_load<T,N>`/`cm_ptr_store<T,N>` (transposed load/store) ONLY support
  T being 4 or 8 bytes wide (uint/float/uint64/double...). For half/ushort/char
  data, load as uint32 (N/2 or N/4 elements) then reinterpret with
  `.format<half>()` / `.format<ushort>()` / `.format<char>()`.
- Function parameters that need to be written by the callee must be
  `vector_ref<T,N>` (or `matrix_ref<T,R,C>`), NOT `vector<T,N>&` -- CM
  rejects plain C++ references to vector/matrix types. Plain `vector<T,N>`
  passed by value auto-converts to `vector_ref<T,N>` at the call site.
- `cm_exp<T>(x)` computes base-2 exp2(x), not natural exp -- multiply the
  argument by log2(e) (1.4426950408889634) first to get a real exp(). Also
  never explicitly specify the template argument as the element type (e.g.
  `cm_exp<float>(vec)`) for the vector overload -- its template parameter is
  the vector length (SZ), not the type; just call `cm_exp(vec)` and let it
  deduce from the argument.
- `cm_sum<T>(vector<T,N>)` gives a horizontal (lane-reduction) sum -- handy
  for GGUF's isum[j]/-min*isum correction terms.
- Reinterpreting a `vector<uint,16>` (16 lanes x 4 bytes) as
  `matrix<char,16,4>` via `.format<char,16,4>()` gives row i = lane i's 4
  bytes -- perfect for extracting "4 int8 weights packed per lane" (used in
  the Q8_0 kernel); `.column(k)` then extracts a `vector<char,16>` of the
  k-th weight across all 16 lanes.
- Arrays of `vector<T,N>` objects (e.g. `vector<float,32> in_j[8];`) compile
  fine in CM (plain C++ array-of-objects) -- useful for holding per-sub-block
  activation data reused across an unrolled `#pragma unroll` loop.

## Roofline model: use max(compute_latency, memory_latency), not AI-vs-ridge
`HW.roofline()` (duplicated in test_dense_gemv.py, test_moe_gemv.py,
test_dense_gemm_slim.py -- the latter is also imported by
test_dense_gemm_full.py/test_moe_gemm.py) was refactored per user request to
explicitly compute `t_compute_s = flops/(fp16_gflops*1e9)` and
`t_memory_s = bytes_moved/(peak_bw_gbps*1e9)` separately, then take
`roofline_s = max(t_compute_s, t_memory_s)` as the roofline baseline latency
and `bound_by = "compute" if t_compute_s >= t_memory_s else "memory"` (the
resource that takes LONGER is the bottleneck, since compute/memory can only
overlap, not add). This is mathematically identical to the old
AI-vs-ridge-point shortcut (`roof = min(peak_fp32, ai*peak_bw)`) -- verified
same output percentages on hardware -- just clearer/more literal about "take
the larger latency as the baseline". New dict keys returned by `roofline()`:
`compute_bound_ms`, `memory_bound_ms`, `roofline_ms` (all in addition to the
existing `roofline_gflops`, `roofline_pct`, `bound_by`, etc.). `_print_rl()`
in all 3 files now also prints a
`roofline latency: compute=X ms  memory=Y ms  ->  bound=Z ms` line. When
adding a new test file, prefer importing `HW`/`_print_rl` from
test_dense_gemm_slim (like test_dense_gemm_full.py/test_moe_gemm.py already
do) instead of copy-pasting this class again -- keeps future roofline-model
tweaks a single edit instead of 3+.

## Follow-up cleanup (later session)
- Deleted stray leftover `.cl` files that had been accidentally copied back into
  `/mnt/river/ovmx/cm_gguf_kernel/` (duplicates of files already preserved in
  `ocl_gguf_kernel/`, unreferenced by any test) -- verified byte-identical
  before removing.
- Converted the `_FLUSH_SRC` "cache_flush" helper kernel (used by all 3 test
  files' `Gpu.setup_flush`/`flush_l3` to defeat cache effects before timing)
  from plain OpenCL C to real CM, so the harnesses are fully OpenCL-C-free.
  Design: grid-stride loop over `vector<float,16>` chunks
  (`cm_ptr_load<float,16>`), reduced with `cm_sum<float>`, only threads with
  `gid < 4096` write to `sink` (matches original OpenCL semantics -- the sink
  write is just an anti-dead-code-elimination side effect, not a meaningful
  reduction, so exact sum values are irrelevant). Requires `_build_src` to
  accept a `build_opts` param so it can be built with `-cmc` like the other
  kernels.

## `token_len <= 16` in test_dense_gemm_slim.py is TEST-ONLY, not a kernel limit
The `gemm_q{4,5,6}k_slim.cm` kernels loop `for (uint v=0; v<token_len; v++)`
inside a single work-group dispatch; SLM sizing (`MAX_NBPR*OPG*sizeof(float)`)
does not depend on token_len at all. The `assert token_len <= 16` in the test
is purely a conservative small-batch test contract (mirrors the SYCL slim
kernel's intended small-batch use case), NOT a CM/HW/quant-format limit. Do
not just delete the assert to test large batches -- see `gemm_q*_full.cm`
below for the real large-batch path (better occupancy/parallelism).

## "full" (non-slim, large-token) dense GEMM port
User asked to also convert the non-slim SYCL GEMM kernels
(`sycl_gguf_kernel/gemm_q{4,5,6}k_L1.xmx-xe2.cpp`, functions
`gemm_q{4,5,6}kweights_xmx_xe2`). These use SYCL ESIMD XMX systolic `dpas`
matmul with an fp16 shuffle pre-pass and 256(rows)x256(tokens) 2D block-load
tiling -- extremely HW/ISA-specific (dpas intrinsics, load_2d, SLM double
buffering). Decided NOT to translate this instruction-for-instruction (too
high risk/effort for a CM port); instead created
`gemm_q{4,5,6}k_full.cm` + `test_dense_gemm_full.py` that:
  - Reuse the EXACT dequant + `vector<float,16>` dot-product math (and the
    same flat-shuffled weight layout / decode helpers) already validated in
    `gemm_q{4,5,6}k_slim.cm` -- verified the SYCL full kernels use the
    identical weight layout offsets (pqs/psl for q4k @ +128*num_blocks;
    pqs/pqh/psl for q5k @ +128/+160; pql/pqh/ps/pd for q6k @ +128/+192/+208),
    confirming this reuse is valid.
  - Add a 2nd dispatch-grid dimension: `cm_group_id(1)` = a tile of up to
    `TOKENS_PER_TILE=16` tokens (`#define TOKENS_PER_TILE 16` in each .cm,
    must match the Python test's `TOKENS_PER_TILE`). Loop bound inside the
    kernel becomes `tile_len = min(TOKENS_PER_TILE, token_len - tile_base)`
    instead of the whole `token_len` -- gives real GPU parallelism across
    tokens (work-group count scales with token_len) instead of slim's
    fixed work-group count with an ever-growing serial loop.
  - `test_dense_gemm_full.py` imports shared helpers from
    `test_dense_gemm_slim` (`build_q{4,5,6}k_flat`, `Gpu`, `HW`, `stats`,
    `check_close`, `_print_rl`, `cl_src`, `OPG`) instead of duplicating the
    ~200 lines of GGUF block-gen/shuffle/dequant code.
  - Dispatch: `gsize=(N/OPG*nbpr, ceil(token_len/16))`, `lsize=(nbpr,1)`.
  - Validated PASS on Arc B580 for token_len in {17 (tail-tile, not a
    multiple of 16), 1024, 4096} across q4k/q5k/q6k, K/N in {2048,4096}.
  - Note: raw benchmark GFLOPS is modest (~500-800 GFLOPS, roofline ~5-8%,
    compute-bound per the harness's AI calc) since it's a scalar/SIMD dot
    product, not a systolic dpas matmul -- expected, this port trades peak
    perf for a low-risk, numerically-verified CM implementation. If real XMX
    dpas-level performance is later required, that would need a dedicated
    CM `cm_dpas`-based rewrite (separate, much larger task).

## MoE GEMM (fused Up+Gate+SiLU, "mat" path) port
User asked to check whether sycl_gguf_kernel/ has MoE *GEMM* kernels (as
opposed to the already-ported MoE *GEMV* in moe_gemv_q{4,5,6}k_sg.cm, which
is only for token_len==1). Found: YES, but ONLY for Q4_K --
`gemm_q4k_L1_slim.xve.cpp::runUpAndGateSlimQ4KL1_xve` (1<token_len<=16) and
`gemm_q4k_L1.xmx-xe2.cpp::upandgate_q4kweights_xmx_xe2` /
`runUpAndGateMatQ4KL1_xmx_xe2` (token_len>16, XMX dpas-based in SYCL). Q5_K
and Q6_K have NO fused up+gate kernel in this tree at all (grepped
gemm_q5k/q6k_L1*.cpp for "gate" -> zero matches) -- their MoE "mat" path in
ffn_moe.mat.cpp reuses two plain GEMM calls (already-ported
gemm_q5k/q6k_slim/full.cm) + a separate elementwise SiLU kernel
(`gpu_silu` in ffn_moe.mat.cpp, plain SYCL, no ESIMD) which was NOT ported
(user only asked for GEMM kernels; flag this if elementwise SiLU port is
ever requested for q5k/q6k).
Ported (Q4_K only), following the same "reuse slim's validated dot product,
generalize full via a 2D token-tile dispatch grid" strategy as the dense
GEMM full/slim split:
  - `moe_gemm_q4k_upgate_slim.cm` (kernel `moe_gemm_q4k_upgate_slim`):
    same 1D dispatch as `gemm_q4k_slim.cm`, but takes TWO weight buffers
    (`ups`, `gates`), computes both dot products per (token, row-group),
    reduces via a two-thread SLM split (hh==0 reduces up, hh==1 reduces
    gate -- mirrors the SYCL original's slm layout/threading exactly, works
    correctly even when nbpr==1 since the raw per-thread partial already
    equals the full reduction in that case), then hh==0 combines with
    SiLU: `result = sigmoid(gate) * gate * up` using
    `1.0f/(1.0f+cm_exp(-gate*1.4426950408889634f))` (same idiom as
    moe_gemv_q4k_sg.cm's group_up_gate_q4k_sg).
  - `moe_gemm_q4k_upgate_full.cm` (kernel `moe_gemm_q4k_upgate_full`):
    identical math, 2D dispatch (row-group x token-tile, `TOKENS_PER_TILE=16`)
    like `gemm_q4k_full.cm`.
  - `test_moe_gemm.py`: builds two independent Q4_K flat weight sets (up,
    gate, different seeds) via `build_q4k_flat` (imported from
    `test_dense_gemm_slim`), numpy reference = `up*gate*sigmoid(gate)` with a
    numerically-stable sigmoid (plain `1/(1+exp(-x))` overflow-warns for
    very negative x -- split pos/neg branches). Validated PASS on Arc B580
    for slim (token_len 1,8,16) and full (token_len 17 tail-tile, 1024,
    4096) shapes.


## gemm_q{4,5,6}k_full.cm perf tuning (roofline %) -- MEASURED results
**FIRST: this machine's GPU clocks drift hugely between process runs.**
Comparing "run test_dense_gemm_full.py before change" vs "after change" is
WORTHLESS -- observed +-40% swings on byte-identical binaries, which once
led me to "confirm" a 33% speedup that was pure noise. ALWAYS A/B by
building both variants from the SAME source in ONE process (select with a
`-D` macro via the build options string) and timing them INTERLEAVED
round-robin with min-of-N. A throwaway harness doing exactly that is easy
to rewrite: import `Gpu`/`cl_src`/`OPG` from test_dense_gemm_slim and
`build_q{4,5,6}k_weight` from test_dense_gemv, then loop
`for _ in rounds: for v in variants: min(...)`.

WHAT WORKED (adopted, 1.11-1.31x end-to-end on Arc B580):
These kernels are limited by WEIGHT DEQUANT, not by the dpas they feed:
per (K-block, sub-block j) the old code issued ~32 SIMD16 dequant ops to
build operands for only 2 dpas ops. Fix = decode all 4 chunks of a
sub-block in ONE SIMD64 pass (`q{4,5,6}k_vnni_block_sg`): the SG-transposed
layout stores chunk c at +c*64 bytes, so 64 consecutive uints are exactly
[c=0..3][n=0..15], and the same 8 bit-field extractions then run at SIMD64
-> 8 ops instead of 32, bit-identical arithmetic. Scatter back into the
VNNI tile needs NO shuffle: source element (c,n) belongs at VNNI
[sd=2c+kpair][2n+kbit], so viewing the tile as `matrix<half,8,32>` via
`.format<half,8,32>()` the destination of a whole extraction is the single
strided 2D region `.select<4,2,16,2>(kpair, kbit)`.
Gotcha hit while doing this: q5k/q6k need a per-c shift for the 5th /
upper-2 bits, and materialising replicated H + a per-element c vector made
the kernel SPILL (net loss). Fix: `H >> (base + c) == (H >> c) >> base`,
so pre-shift once per c when building the replicated vector, then every
extraction is a shift by a scalar immediate and no extra vector stays live.

WHAT WAS MEASURED AND REJECTED (don't redo these):
- Raising TOKENS_PER_TILE to 16/32/64 to amortise dequant over more tokens
  per decode: 1.4-4.6x SLOWER. Extra live registers spill; the spill costs
  more than the redundant decode saves. (Also: CM hard-limits a single
  vector object to <8192 bytes -- "size of vector (8192 bytes) exceeds
  maximum supported size" -- so a flat vector<half,TPT*256> won't even
  compile at TPT>=16; an array of smaller vectors compiles but still spills.)
- fp16 instead of fp32 dequant arithmetic: ~1.25x SLOWER.
- `-Qxcm_register_file_size=256`: helps K=2048 (~1.2x) but HURTS K=4096
  (~1.15x slower) -- large-GRF mode cuts threads/EU. Not enabled.
- Parallel stride-halving tree SLM reduction instead of the serial hh==0
  sum: no measurable difference at any shape. The reduction is NOT the
  bottleneck.
ALWAYS watch build stderr for "Spill memory used = N bytes for kernel ..."
after any change that grows per-thread live vector state -- it is the
single best predictor of a perf regression in these kernels.

## cm_gguf_gemm_kernel/gemm_q{4,5,6}k_full.cm "v5" rewrite -- 2.0-3.2x MEASURED
NOTE: /mnt/river/ovmx/cm_gguf_gemm_kernel/ is a SECOND copy of the dense GEMM
kernels+tests (separate from cm_gguf_kernel/). The v5 work below was done there;
cm_gguf_kernel/ still has the older v4 versions.
Root cause of the old ~10% roofline: a decoded VNNI tile pair (K=32 x N=16)
was consumed by exactly ONE dpas pair (8 tokens), so ~90% of issued
instructions were dequant, not matrix math. Fix = amortize each decode over
TOKEN_GROUPS x 8 tokens. Three co-operating changes:
 1. DROP the K-split + SLM reduction. One thread owns all of K -> acc stays
    in registers, no barrier, no serial thread-0 sum, and an outer K-block
    loop exists to pipeline against. (SLM also can't scale: nbpr * TG * 8 *
    16 * 4 B > 64 KB once TG>1.)
 2. STREAM activations (64 B per token per sub-block j) instead of holding a
    [8 tokens x 256 k] 4 KB A-tile in registers. Still exactly one load per
    A element; this is what makes bigger TOKEN_GROUPS fit without spilling
    (earlier TOKENS_PER_TILE=16/32/64 attempts spilled *because* of the
    register A-tile).
 3. Register-free prefetch with `cm_ptr_prefetch<64, DataSize::U32,
    CacheHint::Cached, CacheHint::Cached>((const unsigned*)ptr, byte_off)`
    for weights and a `lsc::block_2d_desc<half,1,TOKENS_PER_THREAD,32>` +
    `cm_prefetch<CacheHint::Cached,CacheHint::Cached>(desc.set_block_x(x))`
    for activations. Headers: /mnt/CM/llvm-project/clang/lib/Headers/cm/
    include/cm/{cm_lsc.h,lsc/block2d.h}; real usage examples in
    openvino.mx/.../impls/cm/include/cm_pa_xe2.hpp. block_2d_desc ctor takes
    (ptr, Height-1, Width_bytes-1, Pitch_bytes-1, blockX, blockY);
    block_x/block_y are in ELEMENTS/ROWS.
## Long-K (K>>N) is an A-LAYOUT problem, not a tiling problem  [PTL, verified]
A is row-major in K, so the 8 tokens of one 2D block load are K*2 bytes apart;
one A tile load spans 8*K*2 = 65 KB at K=4096 but 196 KB at K=12288 -> TLB /
DRAM-page locality degrades linearly with K. Runtime is SUPER-linear in K
(q4k N=4096 t=4096, ms/K: .79e-3 @K4096 -> 1.22e-3 @K12288 -> 1.34e-3 @K16384).
Large N costs nothing (weights contiguous along K in a row group): 4096x12288
= 33.9 TFLOPS vs 12288x4096 = 25-30. NOT an L2-residency effect - both have the
same 28 MB weight matrix. Structural fix = repack A K-blocked in the caller.
In-kernel the ONLY lever that pays is a taller token tile: TOKEN_GROUPS=4
(121 regs on ptl, no spill) gives 1.08-1.21x for K>=10240, 0.79-0.96x below.
Implemented as `shape_tune()` in test_dense_gemm_full.py (threshold LONG_K=10240,
`--no-shape-tune` to disable). Rejected for long K: ROW_BLOCKS=4 0.89-0.96x,
ROW_BLOCKS=1 0.80-0.96x, TOKEN_GROUPS=8 0.15-0.18x (7040 B spill), WALK_HBLK
0.26-1.08x.
GOTCHA: two sequential test_dense_gemm_full.py runs are NOT comparable on PTL -
thermal drift moved an UNCHANGED shape 54.6% -> 48.7%. Always A/B configs with
the interleaved min-of-N harness (ab_rowgroups.py).

ADOPTED per quant (TOKEN_LOCAL=4, WALK_HBLK=0, PF_W=2, PF_A=1 for all):
  q4k TOKEN_GROUPS=4 -> 2.39/2.60/2.04x ; q5k TOKEN_GROUPS=3 -> 2.48/2.56/2.09x
  q6k TOKEN_GROUPS=4 -> 2.92/3.24/2.49x  (shapes 2048/1024, 2048/4096, 4096/1024)
  ~10-12 TFLOPS -> 25-31 TFLOPS on Arc B580.
Dispatch changed: gsize=(N/OPG, ceil(ntiles/TOKEN_LOCAL)*TOKEN_LOCAL),
lsize=(1,TOKEN_LOCAL), ntiles=ceil(token_len/(8*TOKEN_GROUPS)). TOKEN_LOCAL
threads share a row group h and differ only in token tile -> they read the
SAME weight bytes (L1 hits); worth ~10%.
MEASURED AND REJECTED (do not redo):
 - The "obvious" software pipeline (loading kb+1 / j+1 into REGISTERS early):
   2.3x SLOWER, spills 2048-3648 B. Cache prefetch is the right tool.
 - WALK_HBLK band swizzle of the group grid: +4% BEFORE prefetch existed,
   but 3-8% SLOWER once PF_W/PF_A are on (it breaks the sequential stream
   the HW prefetcher likes). Kept as a knob, defaults to 0.
 - TOKEN_GROUPS 5/6/8 (all quants) and 4 (q5k only): spill.
 - -Qxcm_register_file_size=256: much worse than plain v5.
 - Splitting the 64 B A load into two 32 B loads landing directly in the
   dpas operand rows: 0.9x.
 - Deeper prefetch (PF_W=4/8, PF_A=2/3/4/8): monotonically worse.
Test harness note: test_dense_gemm_full.py's TOKENS_PER_TILE / TOKEN_GROUPS /
TOKEN_LOCAL must stay in sync with the .cm #defines (they drive the dispatch).

## v6: LSC 2D BLOCK LOAD for the dpas A operands -- another 1.83-1.96x
Idea borrowed by reading the CM attention kernels
(openvino.mx/src/plugins/intel_gpu/src/graph/impls/cm/include/
 cm_attention_common.hpp, cm_sdpa_common.hpp, cm_pa_xe2.hpp).
Replace the 8 flat `cm_ptr_load<uint,16>` + 16 register moves + 8 bounds
clamps per (j, token-group) with ONE
  lsc::block_2d_desc<half, 2, 8, 16> b2dA(ptr, token_len-1,
       input_len*sizeof(half)-1, input_len*sizeof(half)-1, 0, 0);
  b2dA.set_block_y(row); cm_load<lsc::Normal>(A2, b2dA.set_block_x(col));
KEY TRICK: NBlocks=2 stacks the two column blocks (j*32+0 and j*32+16) along
X, so the destination vector<half, 2*8*16> comes back ALREADY as
[Alo | Ahi] in dpas operand order -- zero shuffle/moves. The descriptor's
surface height also clamps out-of-range rows in HW, removing the manual
`tok < token_len` selects. block_x is in ELEMENTS, block_y in ROWS, and the
ctor wants Height-1 / Width_bytes-1 / Pitch_bytes-1.
Effect: q4k 0.383->0.210 / 1.151->0.587 / 1.158->0.596 ms (1.83-1.96x).
Also FREED enough registers that q5k no longer spills at TOKEN_GROUPS=4
(1.2x over 3), so ALL THREE quants now use TOKEN_GROUPS=4.
CONSTRAINT: TOKENS_PER_THREAD (=8*TOKEN_GROUPS) must stay <= 32 -- max
BlockH of a 2D block descriptor. TOKEN_GROUPS>4 no longer compiles.
FINAL (all quants): TOKEN_GROUPS=4, TOKEN_LOCAL=4, WALK_HBLK=0, PF_W=2,
PF_A=1, A_LOAD_2D=1. Cumulative vs the original v4: q4k 4.3-5.0x,
q5k 4.2-4.8x, q6k 4.2-4.9x; 35-58 TFLOPS (30-50% of the 116 TFLOPS fp16
dpas roofline, was ~10%).

## v7: COOPERATIVE SLM WEIGHT-DECODE STAGING -- 1.20x (q4k) / 1.45x (q5k) / 1.70x (q6k)
DONE in cm_gguf_gemm_kernel/gemm_q{4,5,6}k_full.cm (`COOP_SLM`, default ON).
Ablation on v6 first (build variants in ONE process, -D selected) told us
where the time went at K=N=2048 tok=4096:
  base 0.578 ms | no dequant 0.352 (1.64x) | no A reload 0.487 (1.19x)
  | neither 0.277 (2.09x, 124 TFLOPS = dpas roofline)
=> dequant ~40% of runtime, and it was pure redundancy: the TOKEN_LOCAL
threads of a WG share row group h and each decoded all 8 sub-blocks.
v7: thread `lid` decodes ONLY sub-block lid, publishes the VNNI tile pair to
SLM, everyone reads all 8 back. Decode/thread drops 8x.
FINAL CONFIG (all 3 quants): TOKEN_GROUPS=4, TOKEN_LOCAL=8, WALK_HBLK=0,
A_LOAD_2D=1, COOP_SLM=1, SLM_JBLK=8, SLM_DEPTH=1, SLM_SLOTS=2,
SLM_FULLBAR=1, PF_W=0, PF_A=0  ->  16 KB SLM/WG = 2 KB/thread (no occupancy
loss), one cm_barrier per K-block. 60-73 TFLOPS = 52-63% of the 116 TFLOPS
fp16 dpas roofline (v4 was ~10%). Cumulative v4->v7: q4k ~5.6x, q5k ~6.5x,
q6k ~7.5x.
KEY LESSONS (do not redo):
 - SDPA's exact protocol (split barrier cm_sbarrier(1)/cm_sbarrier(0) +
   4-slot ring) is CORRECT but 0.97-1.49x, i.e. SLOWER than a 2-slot plain
   cm_barrier. Reason: the split barrier only proves everyone finished phase
   s-2 when phase s starts, so it needs SLOTS >= DEPTH+2 = 4 => 32 KB/WG,
   and the occupancy loss costs more than the barrier overlap saves. A plain
   cm_barrier proves everyone finished phase s-1 => SLOTS >= DEPTH+1 = 2.
   RULE OF THUMB: with SLM-heavy staging, minimize SLM footprint first.
 - SLM_DEPTH=1 with the SPLIT barrier is SILENTLY WRONG (relerr ~0.25): the
   signal releasing phase s is issued *before* phase s-1's producer stores.
   Split barrier requires DEPTH>=2.
 - Once COOP_SLM is on, PF_W/PF_A become a net LOSS (0.94-0.98x) -- staging
   already runs a phase ahead. WALK_HBLK still a loss. Large GRF still a loss.
 - TOKEN_LOCAL 4 (0.99-1.34x) and 16 (1.10-1.13x) both worse than 8.
   SLM_JBLK=4 (8 KB, 2 phases/K-block) slightly worse than 8.
 - Removing the SLM read-back entirely (diagnostic) only buys another
   1.16-1.36x, so the SLM read is NOT the next bottleneck.
 - TOKEN_LOCAL is now a CORRECTNESS requirement (WG size == producer count),
   not just a locality hint -- test_dense_gemm_full.py's TOKEN_LOCAL=8 must
   match the .cm #define or results are silently wrong.
 - `cm_slm_block_read(slm, GENX_NONE, off, vec)` / `cm_slm_block_write(slm,
   off, vec)` handle arbitrary N (they lower to plain __local vector
   load/store), 512 B and 1 KB both fine; 1 KB gave no speedup over 2x512 B.
 - PFA_ROWS/PFA_SETS added so TOKEN_GROUPS>4 compiles (2D block desc tops
   out at 32 rows).
NEXT IDEA IF MORE IS NEEDED: reduce the A-operand traffic (worth 1.19-1.31x
per the ablation) by giving each thread 2 row groups (32 output channels)
instead of 1 -- traffic/dpas 384 B -> 256 B -- but that needs 4 KB of
accumulators, i.e. half the default GRF.

## Other CM attention techniques NOT yet applied to the GGUF GEMMs
Worth trying if more perf is needed (from cm_sdpa_common.hpp):
 - `cm_load<lsc::VNNI>` -- HW does the VNNI re-layout during the load.
   Not directly usable for GGUF weights (they are decoded in registers from
   a quantized payload), but would apply if decoded weights were staged in
   SLM/global in plain [K][N].
 - `cm_load<lsc::Transpose>` + register-only `Transpose_16x16` /
   `Transpose_8x8` (strided `select<2,1,8,2>` butterflies) for changing
   operand orientation without SLM round-trips.
 - `REG_N = CM_GRF_WIDTH/32` -- architecture-adaptive tile width (8 on Xe1,
   16 on Xe2) instead of the hardcoded OPG=16.
 - `cm_load_2d_with_tail` -- zero-fill tail handling as an alternative to
   the clamp trick.


## v7 BOTTLENECK RE-MEASURED (ab_bottleneck.py, new) -- it is now the A OPERAND
Tool: cm_gguf_gemm_kernel/ab_bottleneck.py -- patches the .cm source TEXTUALLY
in memory (gpu._build_src) so every probe is built+timed interleaved in ONE
process. Variants: Abig / halfAtraf / noAtraf / noAload / noSLMrd / nodecode /
dpasonly. Speedup vs base on B580 (all 3 quants, K/N 2048&4096):
  nodecode 0.93-1.04x  noSLMrd 1.00-1.08x   <- dequant + SLM read are DONE
  halfAtraf 1.01-1.16x  noAtraf 1.11-1.35x  <- A cache traffic
  noAload 1.44-1.53x                        <- WHOLE A path = ~1/3 of runtime
  dpasonly 1.59-1.80x  (=~120 TFLOPS, the real dpas ceiling on this part)
Base today: 66-75 TFLOPS = ~60% of the dpas-only ceiling.
Static (ocloc -mCM_printregusage -Qxcm_print_asm_count, IGC_ShaderDumpEnable=1
+ IGC_DumpToCustomDir for .asm): 105 GRF, no spill, grf_count=128 (8 thr/XVE),
slm_size=16384 -> 8 WG x 16 KB = exactly the 128 KB local_mem_size, i.e. we sit
EXACTLY on the occupancy edge (any SLM growth costs a WG/core).
Main K-loop body = 458/496/531 instrs for q4k/q5k/q6k of which only 64 are
dpas.8x8, 32 are load_block2d (A), 16 load.slm (B), ~64 scalar mov/or just to
update the 2D descriptor, and 53 sync.nop + 16 sync.all* scoreboard waits.
Removing 294 of 1223 static instrs (nodecode) changes nothing => NOT ALU-issue
bound; it is LSC/latency bound on the A stream.
~45% of the dpas carry a {BC=1|2} GRF bank-conflict annotation (secondary).
REJECTED HERE (measured 0.78-0.99x): "Abig" = one wide
lsc::block_2d_desc<half,2,TOKENS_PER_THREAD,16> load per sub-block instead of
TOKEN_GROUPS separate 8-row loads (fewer messages, same bytes). Correct but
slower: 128 GRF (up from 105) and the whole 2 KB tile must land before the
first dpas, killing load/dpas overlap.
NEXT (unproven): N-blocking -- 2 row groups per thread with TOKEN_GROUPS=2 --
halves BOTH A bytes and A messages per dpas at constant accumulator GRF;
expected ~1.1-1.25x from the halfAtraf/noAtraf curve. Bigger win would need
staging A in SLM too (classic 2D WG tile), i.e. a real rewrite.

## v8: N-BLOCKING (ROW_GROUPS) -- ADOPTED, up to 1.39x on the big shapes
Implemented in cm_gguf_gemm_kernel/gemm_q{4,5,6}k_full.cm as a generic
`ROW_GROUPS` knob (default now 2; ROW_GROUPS=1 reproduces v7 exactly -- same
105 GRF / 1226 vs 1223 instrs). One thread owns ROW_GROUPS row groups
(16 out-channels each); the A tile of a (sub-block j, token group tg) is
identical for every row group, so the rg loop sits INSIDE the tg loop and the
ROW_GROUPS dpas pairs share ONE A register -> A bytes and A messages per dpas
both /ROW_GROUPS.
Changes needed: h -> h0 = gh*ROW_GROUPS; acc[ROW_GROUPS*TOKEN_GROUPS];
SLM producer work item w = lid (+u*TOKEN_LOCAL) decomposed as
rg = w/SLM_JBLK, jl = w%SLM_JBLK, SLM slot index = w; SLM_JBLK halved to 4 so
SLM_PHASE_B = SLM_NPROD*2*512 keeps the ring at 16 KB (8 WG/Xe-core);
per-row-group scale/min loads move INSIDE the producer loop; epilogue stores
(h0+rg)*OPG. Host dispatch dim0 becomes N/(OPG*ROW_GROUPS).
ADOPTED: ROW_GROUPS=2, TOKEN_GROUPS=4 (=> 8 acc tiles, 32 tokens/thread,
120-128 GRF, NO spill), SLM_JBLK=4, TOKEN_LOCAL=8.
MEASURED (ab_rowgroups.py, interleaved min-of-N, vs ROW_GROUPS=1):
  K=N=4096 tok=8192  q4k 1.39x  q5k 1.25x  q6k 1.14x
  K=N=4096 tok=4096  q4k 1.33x  q5k 1.21x  q6k 1.12x
  K=N=2048 tok=8192  1.04-1.11x ; K=N=2048 tok<=4096  0.97-1.03x (neutral)
  K=N=4096 tok=1024  0.89-0.94x (only regression; +0.12 ms there vs -0.63 ms
  saved at tok=8192, so net strongly positive)
REJECTED: ROW_GROUPS=2 + TOKEN_GROUPS=2 (acc held at 4 tiles, 114 GRF):
weaker, 1.02-1.18x. ROW_GROUPS=2 + SLM_JBLK=8 (32 KB SLM): 0.91-0.96x on big
shapes (occupancy). ROW_GROUPS=4 + TOKEN_GROUPS=1: 0.73-0.87x;
ROW_GROUPS=4 + TOKEN_GROUPS=2 spills 2560 B.
HOST-SIDE GOTCHA: every harness that dispatches these kernels must use
gsize dim0 = N/(OPG*ROW_GROUPS). test_dense_gemm_full.py was updated
(ROW_GROUPS=2 constant); all the older RG=1-era harnesses (ab_walker,
ab_slm_jblk, ab_slm_pipe, ab_slm_pad, ab_token_groups, ab_decode_half,
ab_cons_dbuf, ab_doublegrf, ab_bottleneck, explore_test_fp16vnni) were pinned
with `-DROW_GROUPS=1` in their build options instead.

## GRF=256 *combined with a bigger tile* -- MEASURED, A WASH. Do not redo.
The natural follow-up to v8 ("large GRF was only a loss because the tile did
not grow") was tested properly (ab_rowgroups.py, *_g256 configs):
  RG=2,TG=4 (same tile) + GRF256   0.87-0.96x   162 regs
  RG=2,TG=8 (2x tokens) + GRF256   0.95-0.99x   215 regs
  RG=4,TG=4 (4x tile)   + GRF256   0.94-1.08x   232 regs, mean ~1.00x
  RG=4,TG=2             + GRF256   0.75-0.91x   166 regs
(all three quants; without GRF256 every one of these spills 2.5-19 KB).
WHY it can never win here: -Qxcm_register_file_size=256 halves resident
threads/XVE (8 -> 4) and this kernel is A-LOAD LATENCY bound -- latency hiding
scales with THREADS, not registers. A tile N times bigger issues N times
fewer loads, so the two effects cancel exactly. Best case only TIES the
default while sitting at 232/256 registers (any later edit spills).
GENERAL RULE for these kernels: when the profile says "latency/SBID bound",
do not trade occupancy for registers; remove the latency exposure instead
(SLM staging, prefetch, more independent chains).

## v8 post-N-blocking ablation: BOTH operand fetches are now latency-exposed
Re-ran ab_bottleneck.py (retargeted at the ROW_GROUPS=2 default -- it no longer
pins -DROW_GROUPS=1) on all 3 quants:
  noAload 1.46-1.80x | noSLMrd 1.22-1.51x (was ~1.05x at RG=1!) |
  noAtraf 1.29-1.46x | halfAtraf 1.10-1.23x | nodecode 0.93-1.14x |
  dpasonly 1.58-1.93x
So halving the A traffic promoted the SLM READ to an equal-sized cost. Both
are LATENCY exposure, not bandwidth/message count.
TRIED AGAINST IT, ALL NEUTRAL OR WORSE (do not redo):
 - SLM_RD1K (new knob in gemm_q4k_full.cm): one 1 KB consumer SLM read per
   tile pair instead of 2x512 B, halving SLM messages: 0.99-1.00x.
 - Activation cache hints (new A_L1H/A_L2H knobs, q4k only):
   Cached/Cached and Streaming/Cached 0.99-1.01x (Default already acts as
   Cached); Uncached 0.91-0.95x; ConstCached does not compile
   ("unsupported cache hint").
 - SLM_DEPTH=2 + SLM_SLOTS=4: 0.72-0.80x (32 KB/WG halves WGs per Xe-core).
 - TOKEN_LOCAL=16 + SLM_JBLK=8 (32 KB but 16 threads, so occupancy is kept,
   1 barrier/K-block): 0.98-1.07x, inconsistent.
 - PF_W with or without PF_A: no better than PF_A alone.
 - **explore_gemm_fp16vnni.cm (weights PRE-DECODED to fp16 VNNI at load time,
   so the hot kernel has NO decode, NO SLM, NO barrier): 0.69-0.93x, i.e.
   SLOWER than the fused Q4_K kernel.** fp16 weights are 3.5x bigger than
   Q4_K in memory and that extra weight traffic costs more than the whole
   decode+SLM round-trip it removes. => the cooperative-decode/SLM structure
   is NOT the thing to attack; do not propose "pre-decode the weights" again.
ADOPTED: PF_A=1 (default when ROW_GROUPS>1) -- 1.01-1.08x on all 3 quants,
avg ~1.04x; it was a LOSS at ROW_GROUPS=1, which is why the default is
conditional on ROW_GROUPS. Kernels now report 120 GRF, no spill, ALL PASS.

## DATAFLOW IS PROVABLY AT ITS OPTIMUM: smaller acc / K-split CANNOT help
Asked to try "change the dpas dataflow (K-split + smaller acc)". Result: the
current RG=2/TG=4 tile is the analytic AND measured optimum. Model: with the
rg-inside-tg loop order the memory-hierarchy operand traffic per dpas is
    bytes/dpas = 256/ROW_GROUPS + 512/TOKEN_GROUPS
(A is 512 B per (j,tg) feeding 2*RG dpas; the B PAIR is 1 KB per (j,rg)
feeding 2*TG dpas -- B is twice as expensive per unit of reuse, so the
optimum wants TOKEN_GROUPS = 2*ROW_GROUPS). acc = RG*TG*512 B and must fit
under ~4 KB of the 8 KB (128 GRF) register file, i.e. RG*TG <= 8:
    (1,8) 320 B   (2,4) 256 B <-- min   (4,2) 320 B   (8,1) 544 B
MEASURED (q4k, vs the RG=2/TG=4 default) tracks the model exactly:
    RG=1/TG=8 0.63-0.66x | RG=4/TG=2 0.75-0.91x | RG=2/TG=2 0.86-0.96x
    RG=2/TG=1 0.56-0.63x
CONS_DBUF was generalized over ROW_GROUPS (the #error is gone) to spend the
registers freed by a smaller acc on an explicit double buffer of the SLM tile
reads. It does NOT help: at the SAME tile (RG=2/TG=2) DBUF=1 is 0.85-0.88x vs
DBUF=0's 0.86-0.93x. So operand REUSE dominates in-flight buffering; the
compiler was not the limitation.
K-SPLIT IS RULED OUT WITHOUT IMPLEMENTING IT: it adds parallelism and a
reduction while leaving operand traffic unchanged, and the machine is not
parallelism-starved -- at K=N=4096 the grid is 3.2 waves (tok=1024) to 12.8
waves (tok=4096) of the 1280 thread slots (160 XVE x 8). Every variant that
traded reuse for more thread-tiles (TG=1/2) lost 0.56-0.93x, which is the
same trade K-split makes. It would only pay at tiny token_len (<=32), where
the gemv/slim kernels are used anyway.

## ROOT CAUSE of the "clock drift": the B580 THROTTLES 2900 -> 1150 MHz
The +-40% "drift" the kernel headers warn about is thermal/power throttling,
and it is much worse than +-40%. Read the actual clock from
  /sys/class/drm/card*/device/tile0/gt0/freq0/{act_freq,max_freq,cur_freq}
Four back-to-back runs of the SAME shape with the SAME binary in ONE process:
  run0 1.72 ms @ 2900 MHz | run1 3.18 @ 1467 | run2 4.74 @ 1283 | run3 5.17 @ 1150
Latency tracks the clock ~1:1. Consequences:
 - The FIRST measurement in a process is always the fast one; later shapes of
   a long sweep look 3-4x worse for no code reason. This is why
   test_dense_gemm_full.py reported 8-19% roofline while the interleaved A/B
   harness measured 51-68% on the same binaries.
 - flush vs no-flush made NO difference once re-measured in alternating order
   (it looked like 1.8x at first -- pure ordering artifact).
test_dense_gemm_full.py now: `time_kernel_with_clock()` samples act_freq after
every timed iteration and reports the mean clock; roofline is back to the
post-warmup MEAN latency; when the GT is below 95% of peak it also prints
`roofline_pct_at_peak_clock = roofline_pct * max/act`. That normalised figure
comes out at a consistent 55-72% across all shapes (vs 20-51% raw), matching
the A/B numbers. Added `--no-flush`. `gt_clock=act/max MHz` is printed on the
timing line and `gt=...MHz (@peak clk ~X%)` in the summary.
ALWAYS check the clock before believing any absolute latency on this box.

## PTL / Xe3 iGPU (Arc B390) NEEDS THE OPPOSITE TILE FROM BMG
Remote box (details + credentials in cm_gguf_gemm_kernel/remote_machine.txt,
Windows, workdir D:\river\ovmx_rebase\cm_gguf_gemm_kernel, venv
D:\river\py312\Scripts\activate). No sshpass on this Linux box, but paramiko
5.0.0 IS installed -- drive it with a small helper (kept at /tmp/rptl.py:
`run`/`raw`/`py`/`put`/`sync` subcommands, creds parsed out of
remote_machine.txt, never printed).
DEVICE: Intel Arc B390 GPU, 96 XVE @ 2400 MHz, 10 threads/EU (vs 8 on Xe2),
local_mem 128 KB, L2 16 MB, shared LPDDR (~136 GB/s). Peak fp16 ~59 TFLOPS
(96*2.4e9*256); the `dpasonly` probe tops out ~52 TFLOPS there.
ABLATION ON PTL differs a lot from BMG: noAtraf 1.52-2.19x (A cache traffic
much worse), nodecode 1.14-1.45x (dequant ALU matters again), noSLMrd
1.36-1.59x.
MEASURED BEST ON PTL: ROW_GROUPS=2 TOKEN_GROUPS=2 TOKEN_LOCAL=16 SLM_JBLK=8
PF_A=0  ->  1.11-1.40x over the BMG-tuned config, on ALL 3 quants and every
shape. i.e. SMALLER acc (2 KB, to fit 10 threads/XVE) in a WIDER work-group
(decode shared 16 ways, one barrier per K-block).
REJECTED ON PTL: PF_A=1 (0.93-1.00x -- it is a WIN on BMG!), WALK_HBLK=4/8
(0.32-0.55x, catastrophic), ROW_GROUPS=4 (spills 2048 B), TOKEN_LOCAL=32
(0.92-1.11x), TOKEN_LOCAL=4, SLM_JBLK=4+TL16 (ties).
IMPLEMENTATION: the .cm source is IDENTICAL for both parts. Host-side
`TUNED_CONFIGS` / `pick_config()` / `build_opts()` in test_dense_gemm_full.py
select the macro set from the device name ("b390"/"panther" -> ptl, else bmg)
and match the dispatch; each entry also carries peak_bw/peak_fp16 so the
roofline % is meaningful per part (PTL then reads 34-55%, was 17-29% with
BMG's peaks). `--config bmg|ptl` forces it.
GOTCHA: `-mCM_printregusage` is SILENTLY IGNORED by the Windows driver on
that box (the spill warning does come through). fd-level dup2 and subprocess
pipes do not capture the compiler report there either -- screen configs by
measurement instead.

## PTL round 2: ROW_BLOCKS (A sharing inside the WG) + honest ceiling analysis
ADDED to all 3 kernels: `ROW_BLOCKS` (default 1 = old behaviour) splits a WG
into ROW_BLOCKS row-group blocks x TOKEN_SLOTS(=TOKEN_LOCAL/ROW_BLOCKS) token
slots. Threads sharing a token slot issue the SAME activation addresses, so
only the first misses L1 -> A traffic to L2 drops by ROW_BLOCKS, while the
cooperative decode is still shared across the whole WG (that is what makes it
different from raising ROW_GROUPS, which needs more accumulators and spills).
  rb = lid % ROW_BLOCKS; ts = lid / ROW_BLOCKS;
  h0 = (gh*ROW_BLOCKS + rb)*ROW_GROUPS;  t = gt*TOKEN_SLOTS + ts;
  producer w -> (rbp, rg, jl) with rgj = ROW_GROUPS*SLM_JBLK;
  consumer tile index = (rb*ROW_GROUPS + rg)*SLM_JBLK + jl;
  SLM_NPROD = SLM_JBLK*ROW_GROUPS*ROW_BLOCKS must equal TOKEN_LOCAL.
  Host: gsize=(N/(OPG*RG*RB), ceil(ntiles/TOKEN_SLOTS)*TOKEN_LOCAL).
Also added `MERGE_STORE` (default 1): one contiguous ROW_GROUPS*OPG store per
token instead of ROW_GROUPS 64 B ones. Bit-identical; measured NEUTRAL alone.
MEASURED ON PTL (q4k, iters=40): the winner is ROW_BLOCKS=2 with
TOKEN_LOCAL=32 / SLM_JBLK=8 -- that keeps SLM_NPROD==TOKEN_LOCAL so there is
still ONE barrier per K-block: 0.98/1.04/1.17/1.22x on
2048x4096 / 4096x1024 / 4096x4096 / 4096x8192.
  ROW_BLOCKS=2 with TOKEN_LOCAL=16 (SLM_JBLK=4, 2 barriers/K-block) only gets
  0.98-1.06x -- the extra barrier eats the traffic saving. ROW_BLOCKS=4 is
  0.83-0.87x, ROW_BLOCKS=8 (8 barriers) 0.59-0.64x.
ADOPTED ptl config: RG=2 TG=2 TL=32 RB=2 SLM_JBLK=8 PF_A=0.
FULL SWEEP (--warmup 100 --iters 1000, 13 shapes, all PASS):
  before 39.7% mean / 41.0% median / 44.6% min of the 59 TFLOPS roofline
  after  41.3% mean / 42.6% median / 46.8% min
ALSO REJECTED ON PTL: -Qxcm_register_file_size=96 (0.85-0.99x), 64 (0.17x,
spills), 128 (0.98x); ROW_GROUPS=4 with TG=1 (0.80-0.94x) or TG=2 (spill,
0.32x); ROW_GROUPS=8 (spill 6656 B, 0.15x); SLM_PAD=32 (0.74-0.79x);
TOKEN_GROUPS=1 (0.72-0.85x); A_LOAD_2D=0 does NOT COMPILE in the COOP_SLM
path (the flat A path only exists in the non-COOP branch).
WHY 60% IS NOT REACHABLE BY CONFIG TUNING HERE:
 - the dpas-only probe on PTL is 2.07-2.43x of base, so ~55 TFLOPS is the
   achievable ceiling and 60% of 59 = 35.4 TFLOPS would need a further 1.45x;
 - the remaining cost is dominated by A cache traffic (noAtraf 1.5-1.76x)
   which needs a structural change (WG covering ~16 row groups with A staged
   once), not a knob;
 - the average is also dragged by shapes that are not compute-limited at all:
   token_len=17 lands at 11.5% (1 wave, 14 of 16 token slots are padding) and
   the token_len=1024 shapes at 37-48%.
THROTTLING (measured): on PTL the min is nearly iteration-count-invariant
(3.79-3.83 ms for q4k 4096/4096/4096 at 50/300/1000 iters) while the mean
grows 3.99 -> 4.15 -> 5.10 ms, i.e. sustained runs are power-limited.
`--no-flush` makes it WORSE (7.4 vs 4.7 ms) because back-to-back kernels raise
the duty cycle. test_dense_gemm_full.py now prints mean/median/min roofline
per shape plus an "Average over N shapes" line, and has --shapes
quant:KxNxT,... for quick single-shape work.

## Q4_0 / Q4_1 / Q8_0 dense full GEMM added (cm_gguf_gemm_kernel/, BMG-validated)
NEW FILES: gemm_q40_full.cm / gemm_q41_full.cm / gemm_q80_full.cm (kernels
gemm_q{40,41,80}_full), builders+tests in test_dense_gemm_full.py, config A/B
harness ab_lowbit_config.py. All PASS on Arc B580, no spill (128/128/127 regs).
KEY INSIGHT that made this cheap: a GGUF Q4_0/Q4_1/Q8_0 block is 32 weights
with one scale, which is EXACTLY the shape of a Q4_K SUB-BLOCK, so the whole
v5-v8 dataflow (8 sub-blocks = one 256 K-block, COOP_SLM ring, VNNI tile pair,
ROW_GROUPS/TOKEN_GROUPS) is reused unchanged. All three map onto the SAME
`w = q*scale - minv` decoder:
   Q4_0  w=(q-8)*d      -> scale=d,  minv=8*d
   Q4_1  w=q*d+m        -> scale=d,  minv=-m
   Q8_0  w=q*d (int8)   -> scale=d,  no bit extraction: 8 strided byte regions
Q4_0/Q4_1 nibble packing already matches Q4_K's shuffled pqs convention
(byte p: low nibble = K-pos p, high = p+16), so pack_qk_pqs_sg is reused
verbatim. Q8_0 reuses moe_gemv_q80_sg.cm's pqs_T shape (nrg,nbpr,8j,8u,16n,4b),
u=0..3 -> Blo, u=4..7 -> Bhi. Scales: [rg][kb][j][lane] fp16 (pd_T,
num_blocks*OPG*16 B); Q4_1 stores (d,m) INTERLEAVED (pdm_T, *32 B) so the
producer gets both from ONE 64 B load + stride-2 region read.
Q8_0 decode trick: `vector<char,256> cl = q.format<char>();
w = cl.select<64,4>(b);` -- byte b of every uint, stride 4, lands exactly in
the (c,n) order the .select<4,2,16,2>(b>>1, b&1) VNNI destination wants.
REFERENCE must use the SAME expression order as the kernel (q*d - 8*d, not
(q-8)*d) -- both are exact in fp32 here, but do it anyway.

### A_OUTER=1 IS NOW THE BMG DEFAULT TOO (1.05-1.39x, ALL SIX QUANTS)
Measured with ab_lowbit_config.py (one process, per-variant correctness check,
round-robin interleaved, flush every launch, min-of-N):
  token_len=1024, (K,N) in (4096,4096)/(1024,4096)/(12288,4096)/(4096,12288):
    q40 1.13/1.13/1.11/1.25x  q41 1.11/1.14/1.13/1.34x  q80 1.23/1.15/1.12/1.39x
    q4k 1.11/1.14/1.07/1.06x  q5k 1.15/1.14/1.09/1.24x  q6k 1.20/1.13/1.05/1.34x
  token_len=4096/8192: 1.00-1.21x, never a loss beyond noise.
A_OUTER was originally a PTL register-pressure fix. THAT IS NOT WHY IT WINS ON
BMG: it COSTS registers there (120 -> 128 for the K-types) and still wins,
because the kernel is A-LOAD LATENCY bound. Holding the TOKEN_GROUPS A tiles
issues all 4 independent 2D block loads back-to-back before the first dpas
instead of injecting one into the middle of every dpas chain.
=> TUNED_CONFIGS["bmg"]["a_outer"] flipped 0 -> 1 in test_dense_gemm_full.py.
The .cm default stays A_OUTER=0 so the older ab_*.py baselines are unchanged.
Sweep effect on the 12 requested shapes: 41.1% mean / 45.8% min roofline ->
49.3% mean / 54.7% min (at-peak-clock ~58-69%).

### REJECTED for q40/q41/q80 on BMG (on top of A_OUTER=1, do not redo)
  TOKEN_GROUPS=2 0.81-1.00x | TOKEN_GROUPS=8 0.17-0.20x (spills)
  ROW_GROUPS=1 0.70-0.99x   | ROW_GROUPS=4 + TL=16 0.20-0.25x (8 KB acc spill)
  TOKEN_LOCAL=16 0.94-1.04x | TOKEN_LOCAL=32 + ROW_BLOCKS=2 0.91-1.09x
  PF_A=0 0.92-1.07x (inconsistent; kept at 1 to match the K-types)
  Q8_0's 2x payload does NOT move the optimum (TL=16/32 neutral) -- it is only
  ~16 B/dpas after COOP_SLM amortisation vs ~128 B/dpas of activation.
GOTCHA: ROW_GROUPS=3 does not compile -- MERGE_STORE does a
cm_ptr_store<float, ROW_GROUPS*OPG> and 48 is not a legal LSC vector size.
Use power-of-two ROW_GROUPS only.

### Q4_1 layout UNIFIED with the plugin: SEPARATE pd/pm planes (not interleaved)
The standalone tree briefly used one interleaved (d,m) fp16 plane (`pdm_T`,
num_blocks*OPG*32 B) so the producer got both from ONE 64 B load + stride-2
region read. The OpenVINO GPU plugin's compile-time repack
(`RepackGGUFWeightsShuffle`, src/plugins/intel_gpu/src/plugin/transformations/
repack_gguf_weights.cpp) instead emits TWO SoA planes:
  off_pqs = 0, off_pd = N*nbpr*128, off_pm = off_pd + N*nbpr*16
  field j of lane lid at j*kSG*2 + lid*2
The interleaved variant measured NEUTRAL (this kernel is activation-latency
bound, not producer-message bound), so the standalone was changed to match the
plugin -> ONE layout for the whole family. gemm_q41_full.cm now takes
`pqs_T || pd_T || pm_T` with pm_base = weights + num_blocks*144.
VERIFIED IDENTICAL between plugin and standalone (analytically + by reading
sg_scatter_chunks): Q4_0 and Q8_0 pqs/pd, and all K-type planes. The plugin
.cm files differ from the standalone ones ONLY by the f16/f32 output +
post-ops block (`CM_OUTPUT_F32`, `CM_POST_OPS_DECLS`, `CM_APPLY_POST_OP_*`).

### A_OUTER=1 CHANGES THE COST MODEL: the SLM read became FREE
`ab_lowbit_config.py --ablate` (new mode; patches one component out of the
A_OUTER=1 source in memory, interleaved min-of-N) on q40 with the adopted BMG
config, 4 shapes at token_len=1024:
  noAtraf 0.98-1.39x | noAload 1.28-1.56x | noSLMrd 1.02-1.08x
  dpasonly 1.31-1.59x
noAload lands within noise of dpasonly => the ACTIVATION PATH IS THE ONLY COST
LEFT. noSLMrd ~free => the old model bytes/dpas = 256/RG + 512/TG (which is
what pinned RG=2/TG=4) NO LONGER APPLIES; with the B term free it collapses to
~256/ROW_GROUPS.
(WARNING: the `nodecode` probe reads 0.36-0.39x, i.e. slower than base. It is
a BROKEN probe for these kernels -- writing the raw payload into the VNNI tile
changes the register/format dependency. Do not read it as a result.)

### ADOPTED: bmg long-K shape rule RG=4/TG=2 for K >= 6144 (1.05-1.13x)
RG=4/TG=2 holds RG*TG=8, i.e. the SAME 4 KB accumulator and the same 128
registers with NO spill (A_OUTER=1 is what makes RG=4 compile), while halving
activation traffic per dpas. Implemented in `shape_tune(cfg, K, N, label)` in
test_dense_gemm_full.py (BMG_LONG_K=6144); it must also set
slm_jblk = TOKEN_LOCAL/4 so SLM_NPROD == TOKEN_LOCAL, and it is gated on
N % 64 == 0.
MEASURED (q40, N=4096, t=1024, vs RG=2/TG=4): K2048 0.98x, K4096 1.02x,
K6144 1.13x, K8192 1.05x, K10240 1.05x, K12288 1.12x, K16384 1.11x.
Generalizes at K=12288: q41 1.11x, q80 1.11x, q4k 1.10x, q6k 1.06x; and over
token length: t=2048 1.10x, t=4096 1.09x.

### REJECTED on BMG against the A_OUTER=1 baseline (do not redo)
 - A_BIG=1 (new knob in gemm_q40_full.cm): ONE
   block_2d_desc<half,2,TOKENS_PER_THREAD,16> load for all token groups
   instead of TOKEN_GROUPS 8-row loads. 122 regs (vs 128) and 1/4 the LSC
   messages, yet 0.95-1.00x. The A cost is L2 traffic + latency, NOT message
   count. (This is NOT the old "Abig" rejection, which applied at A_OUTER=0.)
 - A_DESC_TG=1 (new knob): one 2D descriptor per token group so only
   set_block_x stays on the critical path. 0.67-0.71x -- the extra descriptor
   payloads spill 512 B.
 - PF_A=2: 0.93-0.97x.  DECODE_HALF=1: 0.97-0.99x (and double-rounds).
 - ROW_BLOCKS on BMG (the traffic lever that WORKS on PTL): RB2/TL8/JBLK2
   0.91-1.06x, RB2/TL16/JBLK4 0.94-1.05x, RB4/TL16/JBLK2 0.81-0.97x. Only the
   long-K shape gains; elsewhere the extra barriers (SLM_JPS = 8/SLM_JBLK)
   cost more than the traffic saved. ROW_GROUPS=4 gets the same traffic cut
   with NO extra barrier -- that is why the shape rule uses RG, not RB.
 - RG=8/TG=1: 0.62-0.78x.  RG=4/TG=4 with TL=16: 0.20-0.24x (8 KB acc, spill).
GOTCHA: ROW_GROUPS=8 makes MERGE_STORE emit cm_ptr_store<float,128>, which
does not compile (LSC flat store tops out at 64 elements). gemm_q40_full.cm
now auto-falls back to MERGE_STORE=0 when ROW_GROUPS*OPG > 64.

### PTL Q4_1 investigation -> NEW ptl config RG=4/RB=1/TL=16 (1.01-1.63x, all quants)
Remote PTL box driven with /tmp/rptl.py (paramiko; `run`/`raw`/`py`/`put`/`sync`,
creds parsed out of remote_machine.txt, never printed). Recreate it if /tmp is
wiped -- it wraps every command as
  cmd /c "call <act>.bat && cd /d <workdir> && <cmd>"
`ab_lowbit_config.py --ablate --base-variant ptl` on q41, four (K,N) shapes at
token_len=1024, OLD ptl config (RG2 TG2 TL32 RB2 JBLK8 PF_A0 A_OUTER1):
  noAtraf 1.20-1.70x | noAload 1.36-2.05x | noSLMrd 1.04-1.12x
  nodecode 1.07-1.22x | dpasonly 1.63-2.11x
=> PTL is much more activation-CACHE-traffic bound than BMG, the SLM read is
free there too, and 1.6-2.1x of headroom was still available.
ADOPTED (TUNED_CONFIGS["ptl"]): RG=4 TG=2 TL=16 RB=1 SLM_JBLK=4 PF_A=1
A_OUTER=1. 128 registers, NO spill on any of the six kernels.
  q41 vs old ptl: K4096/N4096 1.16x, K1024/N4096 1.03x, K12288/N4096 1.56x,
                  K4096/N12288 1.32x
  q40 1.08-1.63x, q80 1.07-1.55x, q4k 1.01-1.56x, q5k 1.03-1.42x, q6k 1.05-1.36x
  token_len 64/256/2048/4096/8192: 1.08-1.25x; K=2048 1.04x, K=16384 1.54x
  => no regression anywhere.
Q4_1 PTL roofline (4 shapes, --iters 100): mean 50.8% -> 54.5%, min 53.4% -> 62.8%.
KEY MECHANISM (counter-intuitive, remember it): RG=4/RB=1 has the SAME
activation traffic per dpas as RG=2/RB=2 -- 256/(ROW_GROUPS*ROW_BLOCKS) either
way. The win is (a) the reuse moving from L1 (ROW_BLOCKS threads issuing the
same addresses and hoping for a hit) into REGISTERS (one A tile feeding
ROW_GROUPS dpas), and (b) TOKEN_LOCAL halving, which shrinks the SLM ring
64 -> 32 KB so 4 WGs fit per Xe-core instead of 2.
=> GENERAL RULE: prefer ROW_GROUPS over ROW_BLOCKS at equal RG*RB. ROW_BLOCKS
is only worth it when ROW_GROUPS is register-blocked, and A_OUTER=1 is what
unblocks ROW_GROUPS=4.
PF_A flips to 1 on PTL (1.03-1.15x) -- it was rejected at 0.93-1.00x under the
old config. At RG=4 each A load feeds 4x as many dpas, so there are 4x fewer
loads and each one's latency is that much more exposed.
REJECTED ON PTL against the new entry: RG4/RB2/TL32 0.84-1.09x (64 KB ring),
RG4/TL8 0.74-1.07x, RG4/TL32 0.81-0.92x, RG4/RB2/TL16 0.78-1.10x,
RG4/TG1 0.62-0.85x, RG4/TG4 0.19-0.27x (8 KB acc, spills), plain TG4
0.94-1.13x, plain RB1/TL16 1.00-1.16x, TL16/RB2 0.89-1.09x.
INTERLOCK ADDED to shape_tune(): the old PTL "K>=10240 -> TOKEN_GROUPS=4"
promotion is now gated on row_groups <= 2, because firing it on top of
ROW_GROUPS=4 gives 8 KB of accumulators -> spill -> 0.19-0.27x. It is dead
code for the adopted entry anyway (RG=4 buys 1.54-1.63x at long K vs the
1.11-1.21x that promotion used to).
NOTE PTL sequential sweeps still disagree with the interleaved A/B on
individual shapes (mean is dragged by sustained power limits and shape
ordering); the min column tracks the A/B. Trust ab_lowbit_config.py.

## cm_dpasw + GRF=256 (asked, both REJECTED)
- cm_dpasw NOT SUPPORTED on Battlemage/bmg: compile probe hard-errors
  "CM_HAS_DPASW is NOT defined for this target". dpasw (fused-EU shared
  Src2/activation, halves A read bw) was an older-Xe (DG2/PVC) feature, removed
  on the Xe2 XVE. Cannot be used at all on B580.
- GRF=256 (-Qxcm_doubleGRF) already measured fatal (0.64-0.91x): halves resident
  threads/XVE 8->4 on a latency-bound kernel; ASM shows 0 spill so more GRF is
  useless anyway. Even if dpasw existed, exploiting it needs a paired-threads-
  share-activation / different-N layout that conflicts with COOP_SLM decode
  sharing (same wall as the N-direction rewrite).
