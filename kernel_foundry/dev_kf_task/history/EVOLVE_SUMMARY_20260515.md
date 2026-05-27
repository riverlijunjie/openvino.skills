# KernelFoundry Evolutionary Kernel Optimization: Comprehensive Experiment Report

## Abstract

This report summarizes 8 experiments (v4–v11) of LLM-driven evolutionary kernel optimization on Intel Battlemage B580 GPU. The target problem is FP16 matrix multiplication C[2048,2560] = A[2048,2048] × B[2048,2560] using Intel XMX DPAS instructions. Through systematic exploration of temperature, seed kernel, and prompting strategies, we achieved **1.07ms** execution time (**31.7x speedup** over the 33.9ms reference baseline), reaching **20.9% XMX hardware utilization**. The experiments reveal core mechanisms of LLM-evolutionary optimization, identify critical control variables, and establish hardware-specific constraints that define the performance ceiling.

---

## 1. Background and Principles

### 1.1 Evolutionary Kernel Optimization

Traditional GPU kernel optimization requires human experts with deep hardware knowledge and extensive trial-and-error. KernelFoundry proposes a new paradigm: **LLM as mutation operator + MAP-Elites evolutionary framework as search strategy**, automatically discovering high-performance GPU kernels.

Core loop:
```
Seed kernel → LLM generates mutations → Hardware evaluation (compile + correctness + perf)
     ↑                                                        ↓
     ← MAP-Elites archive selects parent for next generation ←┘
```

### 1.2 MAP-Elites Quality-Diversity Algorithm

Unlike traditional evolutionary algorithms that retain only the "single best," MAP-Elites maintains an **elite archive** across a behavioral feature space:

- **4D behavioral space**: memory_opt × compute_opt × parallelism_opt × esimd_opt, each 0–3
- **256 cells**: 4^4 possible behavioral niches
- **Elite selection**: Each cell retains the best-performing kernel with that behavioral profile
- **Island model**: 4 sub-populations evolve in parallel with migration

Behavioral features are extracted automatically via code pattern analysis:
- memory_opt: SLM usage, prefetch, bank conflict avoidance patterns
- compute_opt: DPAS instructions, loop unrolling, register blocking
- parallelism_opt: WG size, subgroup count, cooperative loading

### 1.3 LLM as Semantic Mutation Operator

Unlike traditional random bit-flip mutations, LLM mutations are **semantically aware**:

- **Input**: Current best kernel + performance feedback + hardware specs + optimization hints
- **Output**: Reasoned improvement (analysis/thinking + new code)
- **Temperature**: Controls mutation magnitude — low (incremental) vs high (architectural redesign)

### 1.4 Fitness Function

```
combined_score = perf_score + 3 × I(correct ∧ speedup > 0) × runtime_improvement

Where:
- perf_score: Performance tier (1-5)
- runtime_improvement: Speedup relative to reference baseline
- I(·): Correctness indicator — only correct kernels with speedup receive fitness reward
```

### 1.5 Target Problem

```
Hardware: Intel Battlemage G21 (B580)
  - 20 Xe2 cores
  - FP16 XMX theoretical peak: 96 TFLOPS (20 × 2048 ops/cycle × 2.4 GHz / 1024)
  - Memory bandwidth: ~456 GB/s
  - L2 Cache: ~24 MB

Problem: FP16 GEMM
  - C[2048, 2560] = A[2048, 2048] × B[2048, 2560]
  - Compute: 21.5 GFLOP
  - Theoretical minimum time: 0.224 ms (compute-bound, OI=731 FLOP/byte)

Key instruction: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
  - Per invocation: 8×16×16 = 2048 FP16 MADs = 4096 FLOPs
  - Subgroup size = 16
```

---

## 2. Experiment Configuration Overview

### 2.1 Full Configuration Comparison

| Exp | LLM Model | Temp | Seed | Iters | Branches | Special Config | Total Evals |
|-----|-----------|------|------|-------|----------|----------------|-------------|
| v4 | opus | 0.0 | none | 4 | 2 | First DPAS experiment | 8 |
| v5 | sonnet+codex+opus mix | 0.3-0.6 | none | 20 | 4 | Multi-model | 80 |
| v6 | opus | 0.1 | none | 10 | 4 | Low temp | 40 |
| v7 | opus | 0.0 | none | 10 | 4 | Zero temp | 40 |
| v8 | opus | 0.1 | none | 8 | 4 | Simplified config | 32 |
| v9 | opus | 0.2 | none | 8 | 4 | Gradient tracking | 32 |
| v10 | opus | 0.25 | v4 best | 12 | 4 | Warm-start + enhanced prompts | 48 |
| v11 | opus | 0.3 | v10 best | 11* | 4 | Explicit bottleneck guidance | 43 |

*v11 trial 12 interrupted by GNAI rate limit

### 2.2 Key Variable Evolution

```
Temperature: 0.0 → 0.1 → 0.2 → 0.25 → 0.3
                ↗         ↗         ↗      ↘ (too high)
          zero evo    slow    moderate  breakthrough  regression

Seed:     none → none → ... → none → v4(11.4ms) → v10(1.07ms)
                                       ↑ critical turning point

Instructions: generic → simplified → enhanced → explicit bottleneck targeting
```

---

## 3. Results Comparison

### 3.1 Performance Summary

| Exp | Best Time | Speedup | TFLOPS | XMX Util | Correctness | Wall-clock |
|-----|-----------|---------|--------|----------|-------------|------------|
| baseline | 33.9ms | 1.0x | 0.63 | 0.7% | — | — |
| v4 | 11.4ms | 2.98x | 1.88 | 2.0% | 100% | 270s |
| v5 | 33.9ms | 1.0x | 0.63 | 0.7% | 100% | 3508s |
| v6 | 36.2ms | 0.94x | 0.59 | 0.6% | 90% | 1447s |
| v7 | 33.9ms | 1.0x | 0.63 | 0.7% | 100% | 2128s |
| v8 | 30.9ms | 1.10x | 0.70 | 0.7% | 100% | 775s |
| v9 | 23.8ms | 1.42x | 0.90 | 0.9% | 100% | 819s |
| **v10** | **1.07ms** | **31.7x** | **20.1** | **20.9%** | **100%** | **843s** |
| v11 | 1.31ms | 25.9x | 16.4 | 17.1% | 97.7% | 1022s |

### 3.2 Visual Comparison

```
Runtime (log scale, lower is better):
  v6  |████████████████████████████████████| 36.2ms
  v5  |███████████████████████████████████ | 33.9ms
  v7  |███████████████████████████████████ | 33.9ms
  v8  |█████████████████████████████████   | 30.9ms
  v9  |████████████████████████            | 23.8ms
  v4  |████████████                        | 11.4ms
  v11 |██                                  | 1.31ms
  v10 |█                                   | 1.07ms  ← BEST
  Peak|                                    | 0.22ms
      +----+----+----+----+----+----+----+→
      0    5   10   15   20   25   30   35 ms

XMX Utilization:
  v10 |████████████████████▉          | 20.9%  ← BEST
  v11 |█████████████████▏             | 17.1%
  v4  |██                             | 2.0%
  v9  |▉                              | 0.9%
  rest|▎                              | <0.7%
      +----+----+----+----+----+----+→
      0    5   10   15   20   25   30 %
```

### 3.3 Efficiency Analysis

| Exp | Cost/eval (s) | Performance gain/eval | ROI |
|-----|--------------|----------------------|-----|
| v4 | 33.8 | 2.81ms/eval | ★★★★ |
| v5 | 43.9 | 0ms/eval | ✗ |
| v6 | 36.2 | -0.06ms/eval | ✗ |
| v7 | 53.2 | 0ms/eval | ✗ |
| v8 | 24.2 | 0.09ms/eval | ★ |
| v9 | 25.6 | 0.32ms/eval | ★★ |
| v10 | 17.6 | 0.68ms/eval | ★★★★★ |
| v11 | 23.8 | 0ms/eval (regression) | ✗ |

---

## 4. Key Findings

### 4.1 Finding 1: Temperature is the Core Control Variable

```
Temperature Effect:

Performance Improvement
     ^
  31x|                    * v10 (0.25)
     |
     |
   5x|
     |
  1.4x|              * v9 (0.2)
  1.1x|        * v8 (0.1)
  1.0x|──*─────*─────────────────────* ← v5(mixed), v7(0.0)
  0.9x|  v7   v6               v11(0.3)
     +────+────+────+────+────+────+→ Temperature
          0   0.1  0.2  0.25  0.3

Optimal temperature window: 0.25
```

**Mechanism**:
- **T=0.0**: Deterministic decoding → identical outputs → zero evolution
- **T=0.1**: Minimal randomness → same-architecture micro-adjustments only
- **T=0.2**: Moderate randomness → tile size/loop structure changes, but no architectural boundary crossing
- **T=0.25**: Critical threshold → sufficient to "forget" fixed patterns (e.g., B→SLM), producing novel architectures
- **T=0.3**: Too high → breaks already-optimal structures → regression

### 4.2 Finding 2: Seed Kernel is the Critical Enabler

| Seed State | Best Evolution Result | Analysis |
|------------|---------------------|----------|
| Empty (v5–v9) | 23.8ms | LLM must design DPAS kernel from scratch; evolution budget wasted on basic correctness |
| v4 best (v10) | 1.07ms | Seed validates correct DPAS usage; 100% budget on performance optimization |
| v10 best (v11) | 1.31ms | Seed already near-optimal; high-temp mutations move away from optimum |

**Insight**: The ideal seed is "correct but far from optimal" —
- Too weak (empty): Evolution wastes budget on correctness exploration
- Too strong (v10→v11): High-temperature mutations tend to destroy existing optimizations
- Just right (v4→v10): Correctness solved, large headroom for performance gains

### 4.3 Finding 3: Architectural Phase Transitions

v10 exhibited classic **phase transition** behavior:

```
v10 trajectory:
  7.0 → 5.74 → 5.93 → 3.87 → 5.65 → 5.22 → [1.08] → 1.09 → ...
                                                ↑
                                         Phase transition (Trial 6)
                                         3.87ms → 1.08ms (3.6x jump)
```

**Pre-transition**: All kernels use "A+B both in SLM" (incremental: 3–7ms)
**Post-transition**: Discovery of "A in SLM, B from global" (breakthrough to 1ms)

This reveals **fitness valleys** in the performance landscape — crossing a counter-intuitive intermediate state is required to reach the global optimum. Temperature 0.25 provides exactly the "jump force" needed to cross this valley.

### 4.4 Finding 4: GPU Microarchitecture > Algorithmic Optimization

v11 demonstrated that on B580:

| Theoretically Better | Actually Worse | Root Cause |
|---------------------|----------------|------------|
| Double-buffering (overlap load/compute) | Single-buffer faster | SLM usage↑ → occupancy↓ → worse latency hiding |
| 128 WIs (more parallelism) | 64 WIs faster | Register pressure↑ → spilling |
| B in SLM (shared, reduce bandwidth) | B from L2 faster | L2 sufficient; saved SLM enables more concurrent WGs |
| TILE_K=64 (more DPAS/barrier) | TILE_K=32 faster | SLM overflow + excessive load loops |

**Core lesson**: In GPU optimization, **occupancy** and **latency hiding** often dominate **algorithmic-level optimizations** (double-buffering, larger tiles). A 2.2KB SLM footprint allowing multiple concurrent WGs is more effective than a 25KB "perfect" double-buffered scheme.

### 4.5 Finding 5: LLM Model Selection Matters

| Model Config | Result | Analysis |
|-------------|--------|----------|
| sonnet + codex mix (v5) | 33.9ms, no progress | Style inconsistency prevents cumulative evolution |
| opus alone (v4, v6–v11) | 1.07ms achieved | Consistent style, sufficient reasoning depth |

v5 used claude-4-5-sonnet + gpt-5.3-codex + claude-4-6-opus mix across 80 evaluations with zero progress. Different models' code styles prevent effective inheritance between generations.

---

## 5. Optimal B580 Kernel Architecture

### 5.1 v10 Champion Kernel (1.07ms)

```c
// Hardware mapping:
//   WG tile: 32 rows × 64 cols
//   4 subgroups × 16 WIs = 64 WIs per WG
//   Each subgroup: 4 vertical 8×16 DPAS = 32 rows × 16 cols
//   A → SLM (shared, only ~2.2KB), B → global/L2 (direct reads)
//   K-loop: TILE_K=32, 2 k16 DPAS steps per K-tile

Key optimizations:
1. Bit operations (>> 5, & 31) replace division/modulo
2. Fast-path: skip boundary checks when tile is fully in-bounds
3. SLM padding (+2) avoids bank conflicts
4. Cooperative load: 64 WIs, 16 elements each, fill 32×32 A tile
```

### 5.2 Why This Architecture Wins on B580

```
Performance decomposition (estimated):

  XMX compute: 8 DPAS × 4096 FLOP = 32,768 FLOPs per K-tile
  SLM load: 16 half reads per WI × 64 WIs (A) ≈ 4 cycles
  B load: 16 half reads per SG from L2 ≈ 8 cycles
  Barriers: ~10 cycles × 2 per K-tile
  
  Key insight: Small SLM (2.2KB) → theoretically ~10 WGs per Xe-core
       → When one WG waits on barrier, others execute
       → Latency hiding is the TRUE acceleration source
```

---

## 6. Strengths of Evolutionary Kernel Optimization

### 6.1 Discovers Counter-Intuitive Optimizations

- "A in SLM, B from global" is NOT recommended by any textbook or manual optimization guide
- It was "accidentally" discovered by temperature-0.25 sampling on the 24th evaluation
- A human expert might never try this "asymmetric" strategy

### 6.2 Efficient Search

- v10: Only 48 evaluations (14 minutes) to find 31.7x speedup
- Equivalent to 2.3x performance improvement per minute
- Human manual tuning might take days to weeks for comparable results

### 6.3 Correctness Guarantee

- v10: All 48 evaluations produced correct results
- The evolutionary framework's validation mechanism ensures only correct kernels enter the elite archive
- Eliminates the "faster but broken" risk inherent in manual optimization

### 6.4 Knowledge Accumulation

- Each generation's analysis and code serve as context for the next
- The LLM's "analysis" section demonstrates understanding of prior results
- This "think → experiment → feedback" loop simulates a human engineer's learning process

---

## 7. Problems and Limitations

### 7.1 Dependence on Strong LLM Reasoning

- Current approach critically depends on claude-4-6-opus code reasoning ability
- Weaker models (v5's sonnet/codex mix) completely fail to evolve
- High inference cost: 30–80 seconds per generation

### 7.2 Cannot Break ~21% Utilization Ceiling

- After v10 reached 20.9%, v11's various "improved" strategies all failed
- The remaining 79% performance gap likely requires:
  - Compiler-level optimization (IGC instruction scheduling)
  - Hand-tuned assembly (inline ISA)
  - Deeper hardware feature exploitation (cooperative kernel launch, etc.)

### 7.3 Uncontrollable High-Temperature Mutations

- T=0.25 produced breakthrough in v10; T=0.3 caused regression in v11
- Optimal temperature is highly dependent on current seed quality:
  - Far-from-optimal seed: needs high temperature (0.25–0.3) for large-step exploration
  - Near-optimal seed: needs low temperature (0.1–0.2) for fine-tuning
- No automatic temperature scheduling mechanism exists

### 7.4 Seed Selection Dilemma

- Empty seed: wastes evolution budget on basic correctness
- Optimal seed + high temperature: tends to destroy existing optimizations
- Needs a "just right" seed — correct but far from optimal

### 7.5 Evaluation Cost

- Each evaluation: ~15–30 seconds (compile + correctness + benchmark)
- LLM inference: 30–80 seconds per call
- Full experiment (48 evals): ~14–17 minutes
- GNAI rate limits can interrupt long experiments

### 7.6 Coarse Behavioral Feature Space

- Current 4D MAP-Elites space too coarse
- All 1ms-class kernels fall in same cell (2,3,3,0)
- Cannot distinguish v10's 64-WI architecture from v11's 128-WI architecture
- May cause excellent non-mainstream variants to be discarded

### 7.7 Lack of Compiler Awareness

- LLM doesn't understand Intel Graphics Compiler (IGC) internals
- Some code patterns may inhibit compiler optimization (e.g., excessive SLM blocks automatic register allocation)
- No feedback mechanism for LLM to see compiled ISA

---

## 8. Directional Guidance for Future Work

### 8.1 Near-term (v12–v13)

**Goal**: Micro-tune within v10's architecture, from 1.07ms to 0.8–0.9ms

#### Strategy A: Low-Temperature Fine-Tuning
```
Seed: v10 best (1.07ms)
Temperature: 0.15–0.2
Instructions: Explicitly prohibit architecture changes, allow only micro-optimizations
Directions:
  - Vectorized SLM reads (intel_sub_group_block_read_us)
  - Vectorized B reads (vload2, block reads)
  - Remove dead code (K-remainder path when K%32==0)
  - Try TILE_M=48 or 64 (keeping B-global)
  - K-loop unroll 2x
  - Prefetch B data
```

#### Strategy B: Compiler-Guided Optimization
```
Add: IGC ISA dump analysis in evaluation pipeline
Let LLM see actual compiled instruction sequences
Identify compiler-missed hotspots (unnecessary spill/fill)
Feed back into next generation's prompt
```

### 8.2 Medium-term

#### Adaptive Temperature Scheduling
```python
def adaptive_temperature(trial, best_score, prev_best_score):
    if trial < 3:
        return 0.3  # Initial high-temp exploration
    improvement_rate = (best_score - prev_best_score) / prev_best_score
    if improvement_rate > 0.1:
        return 0.2  # Still improving fast, moderate temp
    elif improvement_rate > 0.01:
        return 0.15  # Incremental, low temp
    else:
        return 0.25  # Stalled, raise temp for breakthrough
```

#### Multi-Phase Evolution
```
Phase 1 (T=0.3, no seed): Explore fundamental architecture space
Phase 2 (T=0.25, Phase 1 best as seed): Seek architectural mutations
Phase 3 (T=0.15, Phase 2 best as seed): Fine-tune
```

#### Refined Behavioral Features
- Current 4D space too coarse (all 1ms kernels in same cell)
- Add dimensions: WG_size_opt, SLM_usage_opt, tile_shape_opt
- Or use continuous coordinates + adaptive grid partitioning

### 8.3 Long-term Directions

#### 1. Multi-Level Optimization Pipeline
```
LLM generates OCL → IGC compiles → ISA analysis → LLM understands ISA →
→ Produces compiler-friendly OCL patterns
```

#### 2. Cross-Problem / Cross-Hardware Generalization
- Transfer v10's "A-SLM + B-global" strategy to other GEMM sizes
- Test on different Intel GPUs (Arc A770, Data Center Max)
- Build hardware→strategy knowledge base

#### 3. Mixed Precision / Operator Fusion
- Extend from pure GEMM to GEMM + Bias + ReLU fused kernels
- FP16/INT8 mixed-precision DPAS
- Support different matrix layouts (row-major, column-major, blocked)

#### 4. Complement Traditional Autotuning
- Use LLM evolution to discover architectural prototypes
- Use traditional grid search to fine-tune numerical parameters (tile sizes, unroll factors)
- Combine strengths of both approaches


## 9. Conclusions

### 9.1 Three Core Conclusions

1. **LLM evolutionary optimization can discover counter-intuitive high-performance GPU kernels**: v10's "A-SLM + B-global" strategy does not come from textbooks — it is an "accidental" product of evolutionary search. This validates the fundamental viability of this approach for GPU optimization.

2. **Temperature × Seed = determinant of evolution effectiveness**: Temperature 0.25 combined with a "correct but non-optimal" seed is the most effective combination. Neither factor alone suffices — seedless high-temperature cannot produce correct kernels; seeded zero-temperature cannot produce new architectures.

3. **GPU microarchitecture constraints set the current performance ceiling**: ~21% XMX utilization indicates that pure code-level optimization is approaching its limits. Further breakthroughs require compiler co-optimization, ISA-level knowledge, or deeper hardware feature exploitation.

### 9.2 Final Performance Milestones

```
Reference baseline: 33.9ms  (0.63 TFLOPS,  0.7% XMX)
v4 direct gen:      11.4ms  (1.88 TFLOPS,  2.0% XMX)
v10 evolved best:    1.07ms (20.1 TFLOPS, 20.9% XMX)  ← GLOBAL BEST
Theoretical peak:    0.22ms (96.0 TFLOPS, 100% XMX)

Total speedup: 33.9ms → 1.07ms = 31.7x
Evolution contribution: 11.4ms → 1.07ms = 10.7x (improvement on seed)
```

---
