# v13 Experiment Analysis: Relaxed-Constraint Exploratory Evolution

## 1. Experiment Overview

| Parameter | Value |
|-----------|-------|
| Task Name | b580_matmul_dpas_v13 |
| GPU | Intel B580 (Battlemage G21, 20 Xe2 cores) |
| Problem | C[2048,2560] = A[2048,2048] × B[2048,2560], FP16, 21.5 GFLOP |
| Theoretical Peak | 96 TFLOPS FP16 XMX, theoretical min time 0.224ms |
| Model | claude-4-6-opus |
| Temperature | 0.25 |
| Iterations | 20 trials × 4 branches = 80 kernel variants |
| Seed Kernel | v12 best (0.948ms, 23.6% XMX, double-buffered A + B-global, 64 WIs) |
| Total Time | 1758 seconds (~29.3 minutes) |
| Evolution Mode | evolve_mode + gradient_tracking + optimization_aware_prompting |

## 2. Core Strategy

v13 adopts a **relaxed-constraint exploratory** strategy, forming a controlled contrast experiment with v12's strict constraints:

### 2.1 USER_INSTRUCTIONS (Relaxed Version)
```
- Current kernel achieves 0.948ms = 23% XMX utilization on B580 (96 TFLOPS peak).
    Hardware: B580 = 20 Xe2 cores, 96 TFLOPS FP16 XMX, 456 GB/s, 32MB L2, 128 KB SLM per core.
    DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
- Focus on improving memory access efficiency and compute utilization.
- Adopts better thread walker and blocking strategy to maximize DPAS usage and hide memory latency.
- Provide explicit launch metadata (GWS/LWS/subgroup hints) in kernel comments.
```

**Key change: All "DO NOT" constraints removed. LLM allowed to freely explore architecture changes.**

### 2.2 Configuration Comparison with v12
| Parameter | v12 | v13 |
|-----------|-----|-----|
| Temperature | 0.2 | 0.25 (more exploratory) |
| USER_INSTRUCTIONS | 3× "DO NOT" constraints | No restrictions, directional guidance only |
| Seed | v10 best (1.07ms) | v12 best (0.948ms) |
| max_iters | 12 | 20 |
| gradient_sampling_weight | 0.3 | 0.2 |
| L2 cache note | 24MB | 32MB (corrected) |
| SLM note | Not mentioned | 128KB per core |

## 3. Experimental Results

### 3.1 Per-Trial Performance

| Trial | Branch 0 | Branch 1 | Branch 2 | Branch 3 | Best (ms) | XMX% |
|-------|----------|----------|----------|----------|-----------|------|
| 0 | 5.13 | 4.73 | FAIL | FAIL | 4.73 | 4.7% |
| 1 | 11.7 | 11.8 | FAIL | 6.43 | 6.43 | 3.5% |
| 2 | 4.35 | 11.8 | 5.14 | 8.92 | 4.35 | 5.2% |
| 3 | 4.66 | FAIL | 1.72 | 4.55 | 1.72 | 13.0% |
| 4 | 2.23 | 2.57 | 2.57 | **1.14** | **1.14** | 19.7% |
| 5 | 1.19 | 1.18 | 1.18 | 1.20 | 1.18 | 19.0% |
| 6 | 1.16 | 3.07 | 1.18 | 5.12 | 1.16 | 19.3% |
| 7 | 5.15 | 5.15 | 3.09 | FAIL | 3.09 | 7.3% |
| 8 | 1.16 | 1.16 | 3.13 | 3.37 | 1.16 | 19.3% |
| 9 | 3.51 | 5.15 | 4.62 | 3.08 | 3.08 | 7.3% |
| 10 | 1.16 | 3.99 | FAIL | 1.16 | 1.16 | 19.3% |
| 11 | 3.07 | 3.99 | 4.60 | 3.71 | 3.07 | 7.3% |
| 12 | 5.15 | FAIL | FAIL | 4.69 | 4.69 | 4.8% |
| 13 | 4.69 | 4.39 | 3.90 | 3.74 | 3.74 | 6.0% |
| 14 | 4.71 | 4.35 | 1.16 | 4.72 | 1.16 | 19.3% |
| 15 | 3.84 | 3.70 | 4.70 | 4.05 | 3.70 | 6.1% |
| 16 | 1.16 | 1.16 | **1.15** | 7.69 | **1.15** | 19.5% |
| 17 | 1.48 | 5.98 | 1.83 | 5.98 | 1.48 | 15.1% |
| 18 | 3.13 | 6.02 | 5.74 | 1.48 | 1.48 | 15.1% |
| 19 | 6.06 | FAIL | 6.05 | FAIL | 6.05 | 3.7% |

### 3.2 Key Metrics

| Metric | v13 Value | v12 Value (comparison) |
|--------|-----------|----------------------|
| **Best Performance** | **1.14ms** | **0.948ms** |
| **XMX Utilization** | 19.7% | 23.6% |
| Relative to Seed | **+20.3% regression** (0.948→1.14) | -11.4% improvement |
| Correctness Rate | 70/80 = 87.5% | 46/48 = 95.8% |
| Compilation Rate | 80/80 = 100% | 48/48 = 100% |
| Sub-1.0ms | **0/80 = 0%** | 19/48 = 39.6% |
| Sub-1.2ms | 14/70 = 20.0% | — |
| Sub-2.0ms | 19/70 = 27.1% | 38/46 = 82.6% |
| Over 3.0ms | **48/70 = 68.6%** | 13/46 = 28.3% |
| Over 5.0ms | 19/70 = 27.1% | 0/46 = 0% |
| Median | 3.99ms | ~0.99ms |

### 3.3 Performance Distribution

```
<1.0ms:   (0, 0%)         ← seed's 0.948ms never reproduced
1.0-1.2:  ■■■■■■■ (14, 17.5%)  ← best region
1.2-2.0:  ■■■ (5, 6.3%)
2.0-3.5:  ■■■■■■■ (13, 16.3%)
3.5-5.5:  ■■■■■■■■■■■■ (24, 30.0%)  ← primary distribution
>5.5:     ■■■■■■■ (14, 17.5%)
FAIL:     ■■■■■ (10, 12.5%)
```

## 4. Evolution Dynamics Analysis

### 4.1 Four-Phase Evolution Pattern

**Phase 1: Architecture Collapse (Trial 0-3)**
- Seed's double-buffered 64-WI architecture completely rewritten by LLM
- Performance dropped to 4.35-11.8ms
- LLM attempted: 32×128(8SG), 32×256(16SG), 64×128(16SG), B-in-SLM
- High failure rate (4/16 = 25%)
- Root cause: relaxed constraints allowed LLM to change the proven-optimal architecture

**Phase 2: Partial Recovery (Trial 4-6)**
- LLM learned from failures, found 32×256/16SG as viable architecture
- Best 1.14ms, still 20% slower than seed
- Trial 5 showed stability (all 4 branches at 1.18-1.20ms)
- Uses `intel_sub_group_block_read_us` for B loads (since subgroups are aligned to B columns)

**Phase 3: Persistent Oscillation (Trial 7-15)**
- Constantly alternating between 1.16ms and 3.07-5.15ms
- LLM repeatedly tries "better" architectures (64×128 with A+B in SLM), repeatedly fails
- Successful trials maintain 32×256/16SG architecture
- Failed trials use oversized WG (>128 WIs) or B-in-SLM strategy

**Phase 4: Marginal Convergence (Trial 16-19)**
- Trial 16 reached 1.15ms (final best)
- Still unable to break the 1.14ms barrier
- Trial 19 completely collapsed (6.05ms + 2 FAILs)

### 4.2 Architecture Variants Explored by LLM

| Architecture | WG Size | Tile | SLM Strategy | Best Performance | Assessment |
|-------------|---------|------|-------------|-----------------|-----------|
| 32×64, 4SG (seed) | 64 | 32×64×32 | A double-buffered | 0.948ms | Optimal (v12 proved) |
| **32×256, 16SG** | 256 | 32×256×16 | A double-buffered | **1.14ms** | v13 best |
| 32×128, 8SG | 128 | 32×128×32 | A+B SLM | 3.09ms | B loading bottleneck |
| 64×128, 16SG | 256 | 64×128×32 | A+B SLM | 4.35ms | Register spilling |
| 32×256, 16SG (K=64) | 256 | 32×256×64 | A SLM 40KB | 11.8ms | SLM too large |

### 4.3 Why 32×256/16SG is Worse Than the Seed

The seed (32×64/4SG) outperforms v13's best (32×256/16SG) because:

1. **Occupancy**: 64 WIs per WG allows more concurrent WGs on 20 Xe2 cores; 256 WIs limits concurrency
2. **SLM efficiency**: Seed's 4KB SLM leaves room for more WGs to share a core; 256-WI version needs more SLM
3. **B access pattern**: Seed reads 16 B columns per SG (contiguous); 256-version also reads 16 per SG but the WG needs 256 columns total bandwidth
4. **K tile size**: Seed uses K=32 (2×k16); 256-version uses K=16 with only 1×k16, relatively higher barrier overhead
5. **Double-buffering effectiveness**: Seed fully hides A loading behind compute; 256-version loads less A per fill but has more frequent barriers

### 4.4 MAP-Elites State

Same as v12, all variants concentrated in cell `(2, 3, 3, 0)`:
- Final elite score: 94.2 (vs v12's 112.3)
- Island 0: 10 programs (dominant)
- Other Islands: 1 program each
- Effective diversity: extremely low

## 5. Key Findings

### 5.1 Constraint Importance Strongly Validated

| Metric | v12 (Strict Constraints) | v13 (Relaxed Constraints) |
|--------|--------------------------|--------------------------|
| Best Performance | 0.948ms | 1.14ms (+20%) |
| Sub-1ms Rate | 39.6% | 0% |
| Regression Rate (>2ms) | 33.3% | 72.9% |
| Failure Rate | 4.2% | 12.5% |
| Exploration Efficiency | High | Very Low |
| Exceeded Seed | ✅ Yes | ❌ No |

**Conclusion: Relaxed constraints not only failed to find a better architecture, they couldn't even reproduce the seed's performance.**

### 5.2 B580-Specific Constraints Confirmed

v13's numerous failures further confirm these B580 hardware constraints:
- **B should NOT go into SLM**: Even with 32×128/8SG, B-SLM loading overhead (64+ elements/WI) far exceeds benefits
- **WG should NOT exceed 64 WIs**: 256 WIs causes occupancy degradation
- **K=32 is optimal**: K=16 has too many barriers, K=64 requires too much SLM

### 5.3 LLM Behavior Under Relaxed Constraints

1. **Over-innovation**: Given "freedom", the LLM tends to design "theoretically better" but actually worse architectures
2. **Cannot roll back**: Even after seeing regression results, the LLM continues trying variants instead of returning to the seed architecture
3. **Analysis correct but decisions wrong**: LLM correctly identifies the bottleneck (B access latency) but chooses the wrong solution (B-SLM)
4. **Slow new-architecture learning**: Takes 4-5 rounds of failure before learning that 32×256/16SG is the viable point under relaxed constraints

## 6. Comparison with Full Experiment Series

| Experiment | Strategy | Temp | Best | XMX% | vs Seed |
|------------|----------|------|------|------|---------|
| v4 | Cold start | 0.0 | 11.4ms | 2.0% | — |
| v10 | Warm start (v4 seed) | 0.25 | 1.07ms | 20.9% | -90.6% |
| v11 | Free exploration | 0.3 | 1.31ms | 17.1% | +22.4% regression |
| **v12** | **Strict constraints** | 0.2 | **0.948ms** | **23.6%** | **-11.4% improvement** |
| **v13** | **Relaxed constraints** | 0.25 | **1.14ms** | 19.7% | **+20.3% regression** |

**Trend**: Stricter constraints → better results. v12 > v10 > v13 > v11 > v4

## 7. Optimization Recommendations

### 7.1 v14 Direction (Based on v12+v13 Lessons)

1. **Return to v12's strict constraint strategy**: Use 0.948ms kernel as seed
2. **Add new "DO NOT" constraints learned from v13**:
   - DO NOT increase WG beyond 64 WIs
   - DO NOT use 32×256 tile (proven inferior to 32×64)
   - DO NOT put B in SLM
   - DO NOT use K-step smaller than 32
3. **New permitted exploration directions**:
   - Combine double-buffering with K-loop 2x unroll
   - More aggressive B prefetch strategies
   - Explore different SLM strides to avoid bank conflicts
   - Try async copy (intel_sub_group_block_read for A loads)
4. **Temperature**: Keep 0.2 or lower to 0.15

### 7.2 Framework-Level Improvements

1. **Seed protection mechanism**: Guarantee at least 1 branch per trial uses the seed's exact structure
2. **Performance gate**: Automatically discard variants >2x slower than seed, don't let them enter elite archive
3. **Architecture lock**: Allow configuration to "lock" certain code structure parameters (WG size, tile dims)

### 7.3 Experiment Design Lessons

- **Constraints are not limitations, they are accelerators**: Good constraints focus the search space on promising regions
- **"Free innovation" is ineffective for LLMs**: On highly specialized hardware (B580 XMX), LLM "theoretical intuition" is unreliable
- **Empirical > theoretical**: Compiler behavior, hardware scheduling, and occupancy cannot be reasoned from code surface

## 8. Conclusion

The v13 experiment is a **valuable negative result**:

1. Clearly demonstrates that **relaxed constraints are inferior to strict constraints**
2. The LLM discovered a 32×256/16SG architecture (1.14ms) through free exploration, but this is still 20% worse than v12's seed
3. 0 out of 80 variants reached the seed's 0.948ms level, while v12 had 19 that exceeded the seed
4. Validates that all "DO NOT" constraints in the v12 report were correct
5. Additionally confirms: K=16 tile inferior to K=32, 256 WIs inferior to 64 WIs

**Core conclusion**: For kernel micro-optimization, "constraint-driven evolution" >> "free exploratory evolution". The next step should be to add finer-grained micro-optimization guidance on top of v12's strict constraints.

## Appendix A: Full Results Table

| Trial | V0 (ms) | V1 (ms) | V2 (ms) | V3 (ms) | Best | Architecture Explored |
|-------|---------|---------|---------|---------|------|----------------------|
| 0 | 5.13 | 4.73 | FAIL | FAIL | 4.73 | 32×128 A+B SLM |
| 1 | 11.7 | 11.8 | FAIL | 6.43 | 6.43 | 32×128 K=64 |
| 2 | 4.35 | 11.8 | 5.14 | 8.92 | 4.35 | 64×128 / 32×128 B-SLM |
| 3 | 4.66 | FAIL | 1.72 | 4.55 | 1.72 | 32×256 first attempt |
| 4 | 2.23 | 2.57 | 2.57 | 1.14 | 1.14 | 32×256/16SG K-pair |
| 5 | 1.19 | 1.18 | 1.18 | 1.20 | 1.18 | 32×256 refined |
| 6 | 1.16 | 3.07 | 1.18 | 5.12 | 1.16 | Mixed 32×256 + others |
| 7 | 5.15 | 5.15 | 3.09 | FAIL | 3.09 | 64×128 regression |
| 8 | 1.16 | 1.16 | 3.13 | 3.37 | 1.16 | 32×256 + exploration |
| 9 | 3.51 | 5.15 | 4.62 | 3.08 | 3.08 | B-SLM attempts |
| 10 | 1.16 | 3.99 | FAIL | 1.16 | 1.16 | Mixed |
| 11 | 3.07 | 3.99 | 4.60 | 3.71 | 3.07 | Architecture exploration |
| 12 | 5.15 | FAIL | FAIL | 4.69 | 4.69 | Failed architectures |
| 13 | 4.69 | 4.39 | 3.90 | 3.74 | 3.74 | B-SLM variants |
| 14 | 4.71 | 4.35 | 1.16 | 4.72 | 1.16 | One good branch |
| 15 | 3.84 | 3.70 | 4.70 | 4.05 | 3.70 | All exploration |
| 16 | 1.16 | 1.16 | 1.15 | 7.69 | 1.15 | 32×256 K=32 |
| 17 | 1.48 | 5.98 | 1.83 | 5.98 | 1.48 | Mixed results |
| 18 | 3.13 | 6.02 | 5.74 | 1.48 | 1.48 | Late exploration |
| 19 | 6.06 | FAIL | 6.05 | FAIL | 6.05 | Final collapse |

## Appendix B: v12 vs v13 Head-to-Head

```
v12 (strict):  1.23 → 1.01 → 0.97 → 0.97 → 0.97 → 0.96 → [trap] → 0.95
               ─────────────────────── steady improvement ───────────────────

v13 (relaxed): 4.73 → 6.43 → 4.35 → 1.72 → 1.14 → 1.18 → [oscillation] → 1.15
               ─── architecture collapse ──┤── recovery ──┤── never beat seed ──
```

The v12 experiment started producing sub-1ms kernels by Trial 2. The v13 experiment never produced a sub-1ms kernel across all 80 variants.
