# Latest follow-up COMPLETE: fresh measurements AND roofline
- New analyze_fast_topk_roofline.py reuses unchanged A.model_work/roofline; drops old radix read-range diagnostics (fast scans unknown). LONG_CONTEXT_FAST_TOPK_ROOFLINE_CN.md has all28 global mean/median/p95/ref/gap, cold6stage diagnostics, separate-run old cold comparison.
- Independent integer F/B tests and runtime exact comparison verify original14 global references unchanged. 64K prefill reference33.786624 happens to equal six-decimal display exactly; do not require every reference differ from rounded display.
- Cold64K prefill358.769638/ref33.786624/gap10.62/ref9.42%; decode.240304/ref.025684/gap9.36/ref10.69%. Separate-run old/new2.339x/2.463x; topk9.42x/5.83x. Keep new decode16K.179940 and64K.240304, no substitution of earlier A/B.
- Audited new560samples/6960events, all28 capsules, exact measurements.json, 67source bytes and immutable tar SHA. New script/test are postprocessing, NOT added to measured ENV manifest.
- Local33CPU tests PASS (12new+6existing+3measurement+12historical); both new --check-report checks PASS and original historical report exact. Remote12new tests and report check PASS.
- Named5 files synced remote and SHA-equal: new analyzer/test/roofline, README updated explicit policies/commands, performance report append-only roofline link. No GPU rerun; no raw logs or old analyzer/report changes.
- New report cm_kernel/LONG_CONTEXT_FAST_TOPK_PERFORMANCE_CN.md; final logs long_context_fast_topk_logs_rerun_20260910T1614Z/{prefill,decode}_SIZE.log; each has cold/no-flush BENCH. 28 protocol_json/*.json capsules and full measurements.json. 100 warmups/20 samples EACH; 560 slices/6960 events, all actually rerun remote B580 UTC2026-09-10 16:14:30–16:18:38.
- Explicit prefill1K2K legacyWG8,4K8K16K32K fastLWS1,64K fast-wg16; decode1K2K legacyWG16,4K+fast-wg16. Q3 unchanged scalar decode1K/P16>=2K. No benchmark/kernel/default source edits. Existing driver cold samples flush96MiB, warmups no flush (NOT ABC v2 cache-matched warmup).
- Cold GPU ms prefill 1.012878/3.254696/10.623992/26.242925/59.841272/139.211444/358.769638; decode .310456/.080430/.111117/.106492/.179940/.128925/.240304. noflush prefill1.010524/3.252670/10.611029/26.234565/59.797944/139.158861/358.693435;decode.246045/.059295/.076795/.086273/.088321/.111622/.135106. Keep variability; no cherry-picking.
- Full own-score allrows, guards/finite/replay before/post pass;64K Q3 NumPy104rows (NOT full), score98rows,full402653184 outputs finite. Real DPAS Q0/Q1 history fully validated. New GPU regressions52finalizer/288helper/9fast+9WG16 passed,21CPU tests incl12historical and3new(10tamper cases).
- New log-dir audit_measurements.py checks all statistics/dispatch/plans/WORK/67source hashes; no roofline. source_snapshot.tar.gz holds exact67source bytes. Existing historical report/analyzers/logs remain untouched. New report audit --format tables --check-report ../../? Run from cm_kernel with full relative script and report paths.
- Initial long_context_fast_topk_logs/ stopped after1K because new OpenCL language server heldGPU; archived, excluded. New runner whole-sweep flock plus28before/after checks service all drm-cycles-* zero, no otherGPU use, no processes killed. Idle render handle is NOT proof of active GPU compute.
- Next roofline: preserve prior unique-I/O formulas, independently adapt fast event labels; old category/audit_plan intentionally rejects fast. Old model_work fixed6n/4n logical radix read diagnostic DOES NOT apply fast; mark unknown, do not change unique-score memory reference. New roofline NOT computed. All logs/reports/new artifacts copied by named rsync both ways.


# COMPLETED VERIFIED UPDATE (2026-09-10)
Prior exploration below includes obsolete/incorrect proposals; do not implement from it. Actual final algorithms are exact SIMD threshold CAP256 with uncapped overflow fallback and WG8/16. Full integrated long A/B completed in TOPK_LONG_OPTIMIZATION_CN.md section9. Best explicit fast for prefill8/16/32K, fast-wg16 for64K/B1 decode16/32/64K; defaults legacy. Seven v2 logs in topk_long_logs/pipeline_ab_*.log, audit33BENCH/660slices/9360events. Source/API regressions all passed. Initial readback-interrupted samples archived; only v2 runs valid for final tables. See repository memory final section for measured numbers and historical source overlay.


# Q2 Topk Optimization Investigation (2026-09-10 follow-up)

Investigating expensive loop structure in qsa_topk_cooperative.cm and optimization opportunities preserving exact deterministic ordering and ordered-key handling of ties.

## Current Expensive Structure

### qsa_topk_cooperative.cm Lines 70–105: Four-Pass Radix Loop
- **Passes**: 4 × 8-bit histogram, shift={24,16,8,0}
- **Per-thread work**: `for(uint i=begin; i<end; i++)` scans contiguous range [tid*chunk, min(tid*chunk+chunk, n))
  - **Load**: `scores[i]` single float
  - **Transform**: `uint key=qsa::float_to_ordered_key(scores[i])`
  - **Key check**: if `(key&mask)==prefix` accumulate `hist[(key>>shift)&255u]++`
  - **Histogram storage**: `vector<uint,256>` local register (per thread)
- **SLM write**: `cm_slm_block_write(slm, tid*1024, hist)` - 256×4 bytes per thread
- **Leader aggregation** (lines 108–118): serial loop over WG threads reading SLM

### Measured Bottleneck (64K Prefill)
- Q2topk: **537.895496ms** (dominant)
- Q3dpas: 266.356692ms
- **Root cause**: 6n score reads per token (source radix_select_topk), but cooperative has fixed 6n reads per pass regardless of K or n_complete

### Dense Bypass Effect
- k==n_complete → arange return (no score loads)
- Mixed/sparse rows → full 4n+emit loop still executes
- 64K prefill 2047 rows → only ~0.15% dense

## Exact Constraints (INVIOLABLE)

1. **Ordered float→key monotonicity**: `float_to_ordered_key` must preserve IEEE-754 finite ordering
2. **Deterministic tie-break**: (score DESC, block_id ASC) must be EXACT, not approximate
3. **Ascending output order**: Selected indices written [0,k) in ascending block ID
4. **No candidate truncation**: Every key > threshold taken; threshold level contributes EXACTLY k_remaining via ascending ID scan
5. **Radix correctness**: Each of 4 passes must resolve bit prefix [31:24], [23:16], [15:8], [7:0] exactly once

## Optimization Menu (Implementable in CM)

### 1. SIMD Block Load for Histogram Accumulation (IMMEDIATE, LOW-RISK)

**Current bottleneck**: Per-thread histogram is `vector<uint,256>` in registers; leader reads one thread at a time via SLM.

**Proposal**:
- Instead of `cm_slm_block_write(slm, tid*1024, hist)`, split hist into 8×32 chunks
- Use 8 separate `cm_slm_block_write(slm, tid*256+chunk_off*128, chunk32)` with 32-element vectors
- Leader thread then does leader_stride=2 or 4 to exploit wider SLM loads, reading `vector<uint,32>` or `vector<uint,64>` blocks
- **Benefit**: Reduces leader loop iteration count; message-pipelining advantage if SLM is latency-bottleneck
- **Risk**: NONE if message width <= 64 bytes (proven safe in cm_pa_xe2.hpp)
- **Preservation**: Exact same histogram bins, same prefix resolution, same tie-break determinism

**Estimate**: ~5–10% if leader aggregation is >10% of total (unlikely given 4 passes dominate)

### 2. Adaptive Radix with Early-Stop Threshold (MEDIUM, MODERATE-RISK)

**Insight**: After pass k, if bin-count ordering becomes "sparse" (few non-zero bins around prefix), can stop early.

**Proposal**:
- After leader aggregates pass k histogram, compute **entropy** or **Gini index** of bin counts
- If entropy < threshold (e.g., top 3 bins contain >95% of histogram), mark `early_stop=1` in SLM state
- All threads skip passes k+1..3 (but still consume SLM state to consume barrier)
- If not early-stopped, continue normally

**Correctness**: 
- Early stop **only when the prefix is unambiguous** – i.e., the next pass cannot change which bins cross the k_remaining threshold
- Proof sketch: if top 3 bins account for >95% and remaining < 5%, AND k_remaining > 0, the threshold must fall within one of those 3 bins (or the high-entropy bins from before)
- Worst case (false trigger): re-run 1 more pass to verify; cost is one SLM read per thread

**Risk**: Entropy threshold tuning is data-dependent; threshold=95% may vary 90–99% across shapes; requires profiling per shape or conservative 99% (rare benefit)

### 3. Narrow Threshold Cohort + Safe Full-Scan Fallback (MEDIUM, HIGH-CONFIDENCE)

**Insight**: Once prefix is narrowed to single 8-bit range, can use **scalar loop to scan for exact threshold**.

**Proposal** (after pass 3, once prefix is narrowed):
- Cooperative kernel has identified threshold byte (prefix bits [7:0])
- **Threshold cohort**: All scores with key matching prefix are candidates for tie-break
- Count scores > prefix (taken unconditionally) + scores == prefix (taken in block-ID ascending order)
- **Current approach**: Cooperative reads full range 4 times, then count/emit once
- **Optimized approach**:
  - Thread range [begin,end) filters scores to only those keys >= (prefix|0x00) and < (prefix|0x100)
  - Gather all equal-key block IDs into **SLM cohort buffer** (max K=512 entries per token)
  - Emit blocks > prefix first, then blocks == prefix in sorted order (radix of block ID if needed)
- **Cost**: One more SLM write (cohort IDs); one reduction to sort cohort IDs (if needed)
- **Benefit**: Eliminates 1–2 wasteful passes for highly-biased distributions; preserves exact tie-break

**Risk**: SLM cohort buffer sizing; if K=512 and all scores tie, need 512×4 bytes per thread (feasible: 16 threads×512×4 = 32KB << 64KB SLM)

### 4. SLM Row Cache for Large n (<=16384, ADVANCED, IMPLEMENTATION-DEPENDENT)

**Constraint**: 64K SLM budget; current histogram+state = ~16.4K for WG16.

**Proposal** (only for decode n >> K):
- On first pass, prefetch first ~4K scores into SLM (16KB row-cache, e.g., 4096×4B)
- Subsequent passes read from SLM row-cache instead of global scores
- After SLM fills, fallback to global loads
- **Trade-off**: 
  - Save: 3 global loads per score (passes 2–4) for cached rows
  - Cost: 1 global load + SLM write (pass 1), + SLM BW (reads × 4 passes)
  - Crossover: n > 4000 and cache-miss rate on global K-cache > 50%

**Limitation**:
- Only applicable if n_complete >> K (e.g., decode 65536 >> 512)
- Not beneficial for prefill or sparse decode
- Requires profiling to confirm L1/L3 miss rate

**Risk**: SLM oversubscription if multiple tokens share cache; need careful WG partitioning

### 5. CM Helper Conventions (NO CROSS-THREAD ATOMICS, PRIMARY SOURCES)

**Current architecture** (qsa_topk_cooperative.cm):
- Leader thread (tid==0) produces prefix after each pass via SLM state write
- All threads synchronize via `cm_barrier()`
- No atomic operations; explicit barrier-based producer-consumer

**Robustness Principle**:
- Do NOT introduce `cm_atomic_*` for histogram aggregation; leader serial loop is safe and deterministic
- If cohort buffer needs merging, use SLM reduction (leader serial or bitonic network over SLM) NOT atomics
- Block-wide prefix distribution: leader writes once, all read via SLM block-read (not via broadcast or shared GRF)
- **Tie-break ordering**: Guarantee via **contiguous range emission** (lines 145–155 current code) – each thread emits its cohort in ascending ID order, no reordering race

**Verified safe pattern** (from code):
```
cm_slm_block_write(slm, COUNT_BASE + tid*16, counts);  // each thread writes its own counts
cm_fence(CM_LOCAL_BARRIER);  // SLM fence
cm_barrier();                 // work-group synchronize
// leader updates offset array (serial, no atomics)
cm_fence(CM_LOCAL_BARRIER);  // SLM fence
cm_barrier();                 // all threads read updated offsets
```

## Recommendation Summary (Ranked by Confidence & Implementability)

| Priority | Option | Risk | Gain Estimate | Implementation Effort |
|---|---|---|---|---|
| 1 | SIMD histogram block load | LOW | 5–10% if leader is bottleneck | 2–3 hours |
| 2 | Narrow threshold cohort + sorted emit | MEDIUM | 10–15% for biased tie distributions | 4–6 hours (requires SLM size audit) |
| 3 | Adaptive early-stop with entropy gate | HIGH | 5–8% only on high-entropy configs | 6–8 hours (profiling per shape) |
| 4 | SLM row cache for large n | MEDIUM | 10–20% only for n>4000, decode-heavy | 8–12 hours + cache-miss profiling |

**Do NOT implement** without:
- Profiling per stage (passes 1–4, leader aggregation, count reduction, emission) separately
- Measured roofline (FLOPs nominal, actual bytes from instrumentation)
- Strict checker re-running equal-key block-ID tie-break validation per row

--- Archived incorrect exploration follows; not implementation guidance ---


# QSA Q2 Score/Finalizer Investigation (2026-09-10)

## Current Architecture

### Score Storage & Layout
- **Block scores buffer**: `[num_tokens, max_blocks]` float32, allocated once per pipeline
- **Selection output**: `[num_tokens, block_topk]` int32, pre-allocated with GUARD sentinels
- **Score chunk sizing**: Currently unlimited per token row; streaming loads per-block in kernel
- **Padding**: `max_blocks = ceil8(n_complete.max())` for work alignment

### Q2 Pipeline Stages (test_qsa_pipeline.py)

#### Prefill Path:
1. **Q1 (Prepare)**: Projects tokens to indexer space; optional DPAS-accelerated (qsa_prepare_dpas)
2. **Q2a (Score)**: DPAS-based tile scoring (qsa_score_tile_dpas.cm)
   - Work-group owns 16 query tokens (NQ=16 fixed to GRF width)
   - Queries staged in SLM (64KB budget check at compile time)
   - 8-block DPAS tiles (M=8)
   - Broadcasts summaries across dpas K steps
3. **Q2b (Finalize)**: Scalar or cooperative radix selection (qsa_topk_finalization.cm or qsa_topk_cooperative.cm)
   - Scalar: 1 thread per token (simple loop)
   - Cooperative: QSA_TOPK_WG=8/16 threads per token sharing SLM histograms

#### Decode Path:
1. **Q2a (Score)**: Partition-based scoring (qsa_score_partition.cm)
   - 1 token per GWS[0]; QSA_PARTITION_BLOCKS (512 default) split along block range
   - Query hoisted to registers (reused across partitions)
   - Each partition scans its block range, writes scores[token, block]
2. **Q2b (Finalize)**: Scalar or cooperative selection (same as prefill)

### Buffer Allocation in Pipeline.__init__
- `self.scores = Buffer(self.poison)` shape `[nt, mb]` where `mb = max(8, ceil8(nc.max()))`
- `self.poison`: alternating NaN/sentinel for skip-even-rows test
- Guard padding: +64 elements on each named buffer (qsa_harness.py::Buffer class)

### Data Flow & Kernels

#### Dense Score Bypass (QSA_DENSE_SCORE_BYPASS=1 only in partition scorer)
- **Prefill DPAS scorer**: Always computes all scores (no bypass)
- **Decode partition scorer**: Skips dense rows (n_complete <= block_topk)
- **Both finalizers**: Bypass reads if QSA_DENSE_BYPASS=1 (default) and k==n_complete

#### Key Selection (qsa_common.hpp::radix_select_topk)
- 4 passes × 8-bit radix histogram → deterministic tie-break on ID
- Input: `scores[0..n_valid)` float32 (NOT padded by caller)
- Output: ascending index order (required for Q3 k-way merge in tile scatter)
- Early return if `k == n_valid` and bypass enabled (no score reads)

## Benchmark CLI Validation (validate_q2_cooperative.py)

### CLI Options
```
--benchmark / --benchmark-only
--warmup N (default 100)
--samples N (default 20)
--flush (96 MiB cold-cache between iterations)
--modes {prefill|decode|both}
--prefill-sizes [1024, 2048, 4096] (validated hardcoded)
--decode-sizes [4096, 16384, 65536] (validated hardcoded)
```

### Expensive Validations & Pitfalls
1. **Hardcoded size validation**: CLI rejects sizes not in {1024,2048,4096} for prefill
   - 65536 decode size requires partition_blocks tuning (16/32/64 variants tested)
   - 65536 → ~128 partitions with default 512, expensive histogram aggregation

2. **Memory pitfall: 64K block counts**
   - Cooperative finalizer SLM: `HIST_BYTES = WG * 256 * sizeof(uint) = 16*1024` (WG=16)
   - `COUNT_BASE = 16K + 16*16 = 16.25K`; total `16.4K < 64K` ✓
   - Partition decoder with part=16 on 65536: 4 passes × (65536 edges + 512 partition boundaries) → 262K score reads
   - **Risk**: L1 thrashing if score row doesn't fit L1 TLB (expect 1-2 cache misses per token)

3. **Instrumentation overhead** (validate_dense_bypass.py):
   - Modified kernel inlines `qsa_counted_score()` to track score accesses
   - Replacement replaces two `scores[i]` sites + function signature + call site
   - Verify source drift before instrumentation: `src.count("scores[i]") == 2`

4. **Reference loops** (test_qsa_pipeline.py):
   - `attention_reference()` iterates selected blocks per token for Q3 check
   - Selected list is sparse (k << n_complete typical)
   - Only ~500 reference rows computed (not all nt tokens) except when `full_ref=True`

## Implementation Requirements: Bounded Score Query-Row Chunks (128 MiB Default)

### Goal
Split score matrix row computation across multiple kernel launches to stay within memory budget while maintaining persistent selection/output state.

### Proposed ABI Changes

#### 1. **Score Chunking Definition**
```c
#define QSA_SCORE_CHUNK_BYTES (128 << 20)  // 128 MiB default
// Compute per-token: max_blocks_in_chunk = CHUNK_BYTES / (num_tokens * sizeof(float))
// At token batch N_T, full_mb blocks: chunk_blocks = min(max_blocks_in_chunk, full_mb)
```

#### 2. **Host-side Partition Loop** (Python caller)
```python
# Before: enqueue_scorer(scores[nt, mb])
# After:
chunk_blocks = (128 * 1024 * 1024) // (nt * 4)
for chunk_start in range(0, mb, chunk_blocks):
    chunk_end = min(chunk_start + chunk_blocks, mb)
    scorer.enqueue(..., scores, chunk_start, chunk_end)  # new args
```

#### 3. **Modified Scorer Signature**
```c
// qsa_score_partition / qsa_score_tile_dpas (NEW):
extern "C" _GENX_MAIN_ void KERNEL_NAME(
    const half* indexer_q, const half* summary_cache,
    const int32_t* past_lens, const int32_t* subsequence_begins,
    const int32_t* block_indices, const int32_t* block_indices_begins,
    float* block_scores,  // NOW [num_tokens, mb] PERSISTENT across chunks
    uint32_t num_tokens, uint32_t num_seqs,
    uint32_t max_blocks,
    uint32_t chunk_block_start,  // NEW: first block to score in this launch
    uint32_t chunk_block_end      // NEW: exclusive end block for this chunk
)
```

#### 4. **Score Computation Loop Modification** (qsa_score_partition.cm example)
```c
// OLD:
for (uint b = begin; b < end; b++) {
    block_scores[row + b] = qsa_block_score(q, summary_cache, block_indices, page_begin, b);
}

// NEW:
uint actual_begin = cm_max<uint>(begin, chunk_block_start);
uint actual_end = cm_min<uint>(end, chunk_block_end);
for (uint b = actual_begin; b < actual_end; b++) {
    block_scores[row + b] = qsa_block_score(q, summary_cache, block_indices, page_begin, b);
}
// Unscored blocks in this chunk stay unwritten; Q3 must skip them
```

#### 5. **Persistent Batch Selection/Output Arrays** (Architecture)
```python
# Key insight: selection already persistent across Q1→Q2→Q3
# scores tensor changes (chunked writes), sel/count DO NOT change allocation

class Pipeline:
    def __init__(self, ...):
        # Pre-allocate ONCE, reuse across score chunks:
        self.sel = Buffer(np.full((nt, block_topk), -1, np.int32))
        self.count = Buffer(np.full(nt, 0, np.int32))
        self.out = Buffer(np.zeros((nt, heads, v_head_size), np.float16))
    
    def enqueue_score_chunked(self):
        chunk_blocks = (128 << 20) // (self.nt * 4)
        for start in range(0, self.mb, chunk_blocks):
            end = min(start + chunk_blocks, self.mb)
            self.scorer.enqueue(...scores, start, end)  # SAME scores buffer
        # After all chunks, finalizer runs ONCE on the complete scores matrix
        self.final.enqueue(self.scores, ...)
```

#### 6. **Global Token ID / Seq Maps (Local Scratch)**
```c
// Q2 score kernel:
// Per-token absolute position (already computed from past_lens + token offset)
// Seq assignment resolved via binary search or linear scan
// These are CONTROL-FLOW only, not buffered to global memory

// Recommendation: NO change needed - already pure local computation
// q absolute_pos = past_lens[seq] + (token_idx - subsequence_begins[seq])
// per-block: id = block_id (simple offset, no encoding)

// Q3 (attention) consumes:
// selected_blocks[token, :count[token]] → physical page indices via block_indices
// No dual mapping needed; page table indirection handles the abstraction
```

### Validation & CLI Constraints

#### Benchmark CLI Changes
```python
# New option:
ap.add_argument("--score-chunk-mib", type=int, default=128,
                help="max memory per score chunk (default 128 MiB)")

# Validation:
if args.score_chunk_mib not in (64, 128, 256, 512):
    ap.error("validated chunk sizes: 64, 128, 256, 512 MiB")

# Usage in benchmark():
for shape in shapes:
    chunk_blocks = (args.score_chunk_mib * 1024 * 1024) // (nt * 4)
    # enqueue_chunked(chunk_blocks)
```

#### Default Preservation
- Default: `--score-chunk-mib 128` → no chunking if nt*max_blocks*4 < 128 MiB (common)
- Chunk arithmetic is idempotent: `chunk_end = start + min(chunk_blocks, remaining)` handles N=1 chunk

## Reference Loop Integration

### Current q3_validation (test_qsa_pipeline.py::attention_reference)
- Iterates over sparse selected blocks per token
- ~500 rows with `rows = set(range(nt)) if nt <= 64 else linspace(32)` + divergent tokens
- Skips expensive full-row computation unless `--full-ref` (CLI flag added)

### Chunked Score Implication
- Selection still produces ascending block IDs per token
- Reference loops unchanged (just reads final sel/count arrays)
- No per-chunk reference computation needed (incremental scores are transparent)

## Summary Modifications

| Component | File | Change | Rationale |
|-----------|------|--------|-----------|
| **Score partition ABI** | qsa_score_partition.cm | Add `chunk_block_start/end` params | Bounded per-launch memory |
| **Score DPAS ABI** | qsa_score_tile_dpas.cm | Add `chunk_block_start/end` params | Consistency, prefill support |
| **Scoring loop** | Both scorers | Clamp `[begin,end)` to chunk | Partial-row writes per launch |
| **Python harness** | test_qsa_pipeline.py | `enqueue_score_chunked()` loop | Multi-launch orchestration |
| **Cooperative finalizer** | qsa_topk_cooperative.cm | **NO change** | Processes complete scores after chunking |
| **Dense bypass logic** | qsa_common.hpp | **NO change** | Works on full row regardless of chunks |
| **CLI validation** | validate_q2_cooperative.py | Add `--score-chunk-mib` option | Benchmark chunk size control |

## ABI Macro Suggestion

```c
// In qsa_config.py .jit() output, add:
"QSA_SCORE_CHUNK_BLOCKS": computed_chunk_blocks_from_host,  // or leave to runtime

// Alternative: encode in kernel as preprocessor const
// qsa_config.py line ~60:
if kernel_name in ("qsa_score_partition", "qsa_score_tile_dpas"):
    d["QSA_SCORE_CHUNK_MSTART"] = runtime_chunk_start  # or 0 for compile-time constant
    d["QSA_SCORE_CHUNK_MEND"] = runtime_chunk_end      # or -1 for "no limit"
```

## Memory Pitfalls & Existing Reference Loops

1. **64K histogram SLM** (cooperative finalizer): Validated < 64K; scales linearly with WG
2. **Score read locality**: Decoder with 65536 blocks = 4 passes × 256 histogram bins per thread
3. **Reference computation**: Sparse (only changed-selection + boundary tokens)
4. **Score persistence**: Key insight - scores can be partial and incremental
