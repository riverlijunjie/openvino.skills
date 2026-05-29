# Kernel Foundry v16 Cold-Start Experiment Report

## Experiment Overview

| Item | Details |
|------|---------|
| Version | v16 |
| Date | 2026-05-29 |
| Objective | Validate MAP-Elites evolution capability from cold start (no seed kernel) |
| Key Difference from v15 | **v15 uses 2D Block IO seed kernel (0.948ms); v16 starts from naive scalar loop** |
| GPU | Intel B580 (Battlemage G21, 20 Xe2 cores) |
| Problem Size | C[2048,2048] = A[2048,2560] × B[2560,2048], FP16, 21.5 GFLOP |
| Theoretical Optimum | 0.224ms (96 TFLOPS XMX peak) |
| Reference Baseline | 34.0ms (naive scalar implementation) |
| Model | Claude Opus 4.6 (us.anthropic.claude-opus-4-6-v1) |
| Iterations | 12 trials × 4 branches = 48 kernel variants |
| Total Duration | 1424.3s (23.7 minutes) |

## Seed Kernel (Cold Start)

```c
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M, const int K, const int N)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += convert_float(A[row * K + k]) * convert_float(B[k * N + col]);
    C[row * N + col] = convert_half(acc);
}
```

Performance: 34.0ms — one work-item per output element, no optimizations.

## Results Summary

| Metric | Value |
|--------|-------|
| Total branches | 48 |
| Compiled successfully | 33/48 (68.8%) |
| Passed correctness | 27/48 (56.2%) |
| Compile failures | 15/48 (31.2%) |
| Correct but no speedup | 4/48 (8.3%) — runtime ≈ 34ms |
| Effectively optimized | 23/48 (47.9%) — runtime < 2ms |
| Best performance | 1.30ms |
| Best speedup | 26.2x (vs 34ms reference) |
| XMX utilization | 17.2% (vs 0.224ms theoretical) |
| MAP-Elites coverage | 4/256 cells |

## Per-Iteration Details

| Trial | Branch | Correct | Runtime(ms) | Speedup | XMX% | Optimization Strategy | Status |
|-------|--------|---------|-------------|---------|------|----------------------|--------|
| **0** | v0 | ✗ | N/A | - | - | DPAS+SLM+2D Block Read (compile fail) | COMPILE_FAIL |
| **0** | v1 | ✓ | 34.0 | 1.0x | 0.66% | Basic DPAS tiling (no SLM, subgroup block read) | CORRECT (no speedup) |
| **0** | v2 | ✓ | **1.46** | 23.2x | 15.3% | 64WI/4SG, SLM double-buffer, DPAS 32×64×32 tile | ✓ BEST (Trial 0) |
| **0** | v3 | ✓ | 34.0 | 1.0x | 0.66% | DPAS tiling (SLM bank conflict issue) | CORRECT (no speedup) |
| **1** | v0 | ✗ | N/A | - | - | Added vload+subgroup_shuffle (incorrect) | INCORRECT |
| **1** | v1 | ✗ | N/A | - | - | K-loop 2x unroll (incorrect) | INCORRECT |
| **1** | v2 | ✓ | 33.9 | 1.0x | 0.66% | Conservative, no SLM double-buffer | CORRECT (no speedup) |
| **1** | v3 | ✗ | N/A | - | - | Aggressive SLM strategy (incorrect) | INCORRECT |
| **2** | v0 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **2** | v1 | ✓ | 1.46 | 23.3x | 15.3% | SLM double-buffer + interleaved load/compute | CORRECT |
| **2** | v2 | ✓ | **1.30** | 26.2x | 17.2% | K-loop 2x unroll + vload8 pairs + instruction reorder | ✓ **GLOBAL BEST** |
| **2** | v3 | ✓ | 1.38 | 24.6x | 16.2% | SLM double-buffer + K-loop unroll | CORRECT |
| **3** | v0 | ✓ | 1.55 | 21.9x | 14.5% | Based on best, if-guard optimization | CORRECT |
| **3** | v1 | ✓ | 1.61 | 21.1x | 13.9% | 2x K-loop unroll + B prefetch | CORRECT |
| **3** | v2 | ✗ | N/A | - | - | Compile failure (syntax error) | COMPILE_FAIL |
| **3** | v3 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **4** | v0 | ✓ | 1.64 | 20.7x | 13.7% | 32×64×32 tile, 4SG, K-loop unroll | CORRECT |
| **4** | v1 | ✓ | 1.57 | 21.7x | 14.3% | Same + optimized A load order | CORRECT |
| **4** | v2 | ✓ | 1.62 | 21.0x | 13.8% | Same + barrier optimization | CORRECT |
| **4** | v3 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **5** | v0 | ✓ | 1.57 | 21.7x | 14.3% | Conservative, removed compound literals | CORRECT |
| **5** | v1 | ✗ | N/A | - | - | Correctness failure | INCORRECT |
| **5** | v2 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **5** | v3 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **6** | v0 | ✓ | **1.30** | 26.2x | 17.2% | 2x K-loop unroll + if-guard fix | ✓ TIED BEST |
| **6** | v1 | ✓ | 1.39 | 24.5x | 16.1% | Similar + extra micro-opt | CORRECT |
| **6** | v2 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **6** | v3 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **7** | v0 | ✓ | 1.69 | 20.1x | 13.3% | Conservative DPAS, reduced SLM bank conflict | CORRECT |
| **7** | v1 | ✗ | N/A | - | - | Correctness failure | INCORRECT |
| **7** | v2 | ✓ | 1.61 | 21.1x | 13.9% | Standard 32×64×32 implementation | CORRECT |
| **7** | v3 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **8** | v0 | ✓ | 1.32 | 25.8x | 17.0% | Based on best, safe micro-optimizations | CORRECT |
| **8** | v1 | ✓ | 1.38 | 24.6x | 16.2% | Precompute a_row_offset | CORRECT |
| **8** | v2 | ✓ | 1.61 | 21.1x | 13.9% | Standard implementation | CORRECT |
| **8** | v3 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **9** | v0 | ✓ | 1.37 | 24.8x | 16.4% | Return to proven-best + safe micro-opt | CORRECT |
| **9** | v1 | ✓ | 1.64 | 20.7x | 13.7% | Standard double-buffer | CORRECT |
| **9** | v2 | ✓ | 1.31 | 26.0x | 17.1% | Best implementation structure replicated | CORRECT |
| **9** | v3 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **10** | v0 | ✓ | 1.62 | 21.0x | 13.8% | Removed K-boundary check | CORRECT |
| **10** | v1 | ✓ | 1.61 | 21.1x | 13.9% | Standard implementation | CORRECT |
| **10** | v2 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **10** | v3 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **11** | v0 | ✓ | 1.64 | 20.7x | 13.7% | A prefetch overlap with compute | CORRECT |
| **11** | v1 | ✓ | 1.58 | 21.5x | 14.2% | Standard double-buffer | CORRECT |
| **11** | v2 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |
| **11** | v3 | ✗ | N/A | - | - | Compile failure | COMPILE_FAIL |

## Performance Evolution Trajectory

```
Trial 0:  34.0ms → 1.46ms  (first-round leap from naive to DPAS-optimized)
Trial 2:  1.46ms → 1.30ms  (K-loop 2x unroll + vload8 optimization, NEW BEST)
Trial 3~5: 1.30ms plateau (variants range 1.55-1.69ms)  
Trial 6:  1.30ms (tied best)
Trial 8:  1.32ms (near best)
Trial 9:  1.31ms (near best)
Trial 10~11: 1.58-1.64ms (slight regression, no breakthrough)
```

**MAP-Elites Evolution Log:**

| Timepoint | Program ID | Score | Event |
|-----------|-----------|-------|-------|
| Trial 0 | e04c04d0 | 8.00 | First correct kernel (34ms) |
| Trial 0 | fd95fb5e | 74.66 | First high-performance kernel (1.46ms) |
| Trial 2 | 7971b985 | 74.86 | New best (1.30ms) |
| Trial 2 | 47d01e01 | 83.46 | Final best (score improvement) |

## Inference Time and Token Consumption Estimate

| Trial | Inference(s) | Evaluation(s) | Total(s) | Est. Tokens (in+out) |
|-------|-------------|--------------|-----------|---------------------|
| 0 | 140.4 | 143.1 | 283.5 | ~120K (first round, full prompt) |
| 1 | 76.6 | 28.0 | 104.6 | ~80K |
| 2 | 82.7 | 42.4 | 125.1 | ~85K |
| 3 | 93.9 | 7.8 | 101.7 | ~90K |
| 4 | 90.5 | 11.1 | 101.6 | ~90K |
| 5 | 97.0 | 5.2 | 102.2 | ~95K |
| 6 | 92.9 | 7.8 | 100.7 | ~90K |
| 7 | 91.4 | 9.1 | 100.5 | ~90K |
| 8 | 91.9 | 11.5 | 103.4 | ~90K |
| 9 | 91.2 | 11.4 | 102.6 | ~90K |
| 10 | 91.6 | 7.8 | 99.4 | ~90K |
| 11 | 91.1 | 7.9 | 99.0 | ~90K |
| **Total** | **1131.2** | **293.1** | **1424.3** | **~1.1M tokens** |

> Note: Token consumption is estimated. Based on Claude Opus 4.6 average output rate of ~40 tok/s and ~90s per inference, each branch produces ~3,600 output tokens plus ~15K input prompt tokens. Total: 48 branches × (~15K input + ~3.6K output) ≈ 894K tokens. With system overhead, estimated total is approximately **1.0-1.2M tokens**.

## Key Findings

### 1. Cold-Start Evolution Capability Validated

**Conclusion: MAP-Elites achieves the leap from naive scalar loop to DPAS-optimized kernel in the very first trial.**

- Trial 0 Branch v2 jumps directly from 34ms to 1.46ms (23.2x speedup)
- This proves Claude Opus 4.6 can design a complete DPAS+SLM kernel from scratch in a single inference
- First round takes 140s inference time (50% longer than subsequent ~90s), indicating more "reasoning" needed for initial design

### 2. Performance Ceiling: 1.3ms Barrier

All 12 iterations plateau at 1.30ms best. Compared to v15's 0.274ms:

| Comparison | v15 (Warm Start) | v16 (Cold Start) |
|-----------|-----------------|-----------------|
| Seed kernel | 2D Block IO, 0.948ms | Naive scalar, 34.0ms |
| Best result | 0.274ms | 1.30ms |
| XMX utilization | 81.6% | 17.2% |
| Trials to reach best | ~8 trials | 2 trials |
| Uses 2D Block IO | ✓ | ✗ |

**Performance gap analysis (1.30ms vs 0.274ms = 4.7x):**

The v16 best kernel architecture:
- 64 work-items, 4 subgroups, SG_SIZE=16
- TILE: 32×64×32 (WG computes 32 rows × 64 cols, K-step 32)
- A via SLM double-buffering
- B from global memory via sub_group_block_read
- K-loop 2x unroll (stride 64)
- DPAS: `intel_sub_group_f16_f16_matrix_mad_k16`

**Missing critical optimizations (vs v15):**
- ❌ 2D Block Load (intel_sub_group_2d_block_read) — v15 uses this for peak bandwidth
- ❌ B matrix VNNI format transform (2d_block_read_transform)
- ❌ Larger WG tile coverage (v15 uses larger tiles to reduce launch overhead)
- ❌ Advanced prefetch strategies (L1/L2 cooperative prefetch)

### 3. High Compile Failure Rate (31.2%)

15/48 branches failed to compile. Primary causes:
- Syntax errors (unmatched parentheses, undeclared variables)
- Incorrect Intel extension API usage (wrong parameter types)
- `reqd_work_group_size` mismatch with actual WG layout

### 4. Evolution Stagnation

From Trial 2 to Trial 11 (10 rounds), best performance did not improve. The MAP-Elites algorithm with only 4/256 cells covered lacks diversity pressure, causing subsequent kernels to oscillate in the 1.3-1.7ms range without breakthrough.

### 5. Cold Start vs Warm Start Comparison

| Dimension | Cold Start (v16) | Warm Start (v15) |
|-----------|-----------------|-----------------|
| First-round breakthrough | Strong (34ms→1.46ms) | Moderate (0.948ms→0.5ms) |
| Ultimate performance | Weak (1.30ms) | Strong (0.274ms) |
| Required knowledge | LLM internalized knowledge sufficient for basic DPAS | Seed provides 2D Block IO pattern |
| Evolution efficiency | Low (no improvement in 12 rounds) | High (continuous improvement) |
| Best use case | Exploring new architecture directions | Fine-tuning known best approach |

## Conclusions and Recommendations

1. **Cold start has limited effectiveness**: While the LLM can generate DPAS kernels from scratch, it cannot evolve to 2D Block IO performance levels. 2D Block Load is critical for high XMX utilization on Xe2, but the LLM struggles to use it correctly without examples.

2. **Recommended hybrid strategy**: Use cold start to discover new architectural directions (e.g., DPAS tiling schemes), then switch to warm start for fine-tuning optimization.

3. **USER_INSTRUCTIONS effectiveness**: Although instructions describe "64 WI, SLM, TILE 32×64×32" architecture, the LLM successfully converges to this architecture in cold start, proving the guidance value of instructions.

4. **ROI analysis**: 48 branches / ~1.1M tokens yields 26.2x speedup (vs naive) but only 17.2% XMX utilization. Compare to v15 achieving 81.6% utilization with similar token investment.

---

*Environment: Intel B580 (BMG), oneAPI 2025.0.4, PyOpenCL, Intel NEO driver 25.48.36300.8*
*Report generated: 2026-05-29*
