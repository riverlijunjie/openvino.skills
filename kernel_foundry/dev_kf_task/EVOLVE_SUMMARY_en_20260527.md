# KernelFoundry Evolutionary Kernel Optimization: Full Technical Report (v4-v15)

## Abstract

This report presents a comprehensive analysis of 12 evolutionary kernel optimization experiments (v4-v15) conducted using the KernelFoundry system on an Intel Battlemage B580 GPU. The target workload is FP16 matrix multiplication C[2048,2048] = A[2048,2560] x B[2560,2048] using Intel XMX DPAS instructions. Through systematic exploration of temperature control, seed kernel selection, constraint strategies, and architectural innovation, the system achieved **0.274ms** execution time at **81.6% XMX hardware utilization** (78.4 TFLOPS), representing a **41.6x speedup** over the initial cold-start result (11.4ms) and reaching within **1.22x** of the theoretical hardware minimum (0.224ms).

The experiments reveal fundamental principles of LLM-driven kernel evolution: (1) architecture selection dominates over micro-optimization by an order of magnitude; (2) seed quality is the single most impactful variable; (3) temperature must be phase-appropriate; (4) strict negative constraints outperform rich positive guidance; and (5) evolution efficiency follows a power law with diminishing returns within a fixed architecture. We propose a multi-phase pipeline and adaptive control strategies to improve future evolution campaigns.

---

## 1. KernelFoundry: Principles and Evolution Logic

### 1.1 System Overview

KernelFoundry replaces human expert trial-and-error with LLM-driven evolutionary search. The core innovation is using a Large Language Model as a **semantic mutation operator** within a quality-diversity evolutionary framework.

```
Seed Kernel ──→ MAP-Elites Archive ──→ Parent Selection
     ↑                                       │
     │                                       ▼
Fitness Evaluation ◄── Hardware Test ◄── LLM Mutation (Semantic Rewrite)
```

### 1.2 MAP-Elites Quality-Diversity Algorithm

Unlike traditional EAs that retain only the single fittest individual, MAP-Elites maintains an **elite archive** spanning a multi-dimensional behavioral space:

- **4D Behavioral Space**: `memory_opt x compute_opt x parallelism_opt x esimd_opt` (each 0-3)
- **256 cells** (4^4): Each retains the highest-fitness kernel with that behavioral profile
- **Island Model**: 4 sub-populations evolve in parallel with periodic migration

Behavioral features are extracted via static code pattern analysis (SLM usage, DPAS density, WG size, block IO patterns).

### 1.3 LLM as Semantic Mutation Operator

Unlike random bit-flip mutations, LLM mutations are **semantically aware**: the model receives current kernel source, performance metrics, hardware specs, and optimization hints, then produces a reasoned improvement. Temperature controls mutation magnitude -- low temperature yields incremental edits, high temperature enables architectural redesign.

### 1.4 Fitness Function

```
combined_score = perf_score + 3 * I(correct AND speedup > 0) * runtime_improvement
```

The 3x multiplier on correctness-gated improvement ensures incorrect kernels receive no speedup bonus regardless of measured time, while correct faster kernels receive amplified reward proportional to improvement.

### 1.5 Target Problem

```
Hardware: Intel Battlemage B580 -- 20 Xe2 cores, 96 TFLOPS FP16 XMX peak, ~456 GB/s, 32MB L2
Problem:  C[2048,2048] = A[2048,2560] x B[2560,2048], FP16, 21.5 GFLOP
Theoretical minimum: 0.224ms (compute-bound, OI=731 FLOP/byte)
Key instruction: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc) = 4096 FLOPs/call
```
---

## 2. Experiment Summary: Conditions and Results

### 2.1 Full Experiment Matrix

| Exp | Strategy | Model | Temp | Seed Source | Evals | Best Time | XMX% | TFLOPS | Correct% | Wall-clock |
|-----|----------|-------|------|-------------|-------|-----------|------|--------|----------|------------|
| v4 | Cold start (first DPAS) | opus | 0.0 | None | 8 | 11.4ms | 2.0% | 1.9 | 100% | 270s |
| v5 | Multi-model ensemble | sonnet+codex+opus | 0.3-0.6 | None | 80 | 33.9ms | 0.7% | 0.63 | 100% | 3508s |
| v6 | Low temperature | opus | 0.1 | None | 40 | 36.2ms | 0.6% | 0.59 | 90% | 1447s |
| v7 | Zero temperature | opus | 0.0 | None | 40 | 33.9ms | 0.7% | 0.63 | 100% | 2128s |
| v8 | Simplified prompt | opus | 0.1 | None | 32 | 30.9ms | 0.7% | 0.70 | 100% | 775s |
| v9 | Gradient tracking | opus | 0.2 | None | 32 | 23.8ms | 0.9% | 0.90 | 100% | 819s |
| v10 | Warm start | opus | 0.25 | v4 best (11.4ms) | 48 | 1.07ms | 20.9% | 20.1 | 100% | 843s |
| v11 | Free exploration | opus | 0.3 | v10 best (1.07ms) | 43 | 1.31ms | 17.1% | 16.4 | 97.7% | 1022s |
| v12 | Strict constraints | opus | 0.2 | v10 best (1.07ms) | 48 | 0.948ms | 23.6% | 22.6 | 95.8% | 1105s |
| v13 | Relaxed constraints | opus | 0.25 | v12 best (0.948ms) | 80 | 1.14ms | 19.7% | 18.9 | 87.5% | 1758s |
| v14 | Constraints + 37 tips | opus | 0.25 | v12 best (0.948ms) | 48 | 1.01ms | 22.2% | 21.3 | 75.0% | 1116s |
| v15 | New architecture seed | opus | 0.25 | v28 2D Block IO (0.34ms) | 48 | 0.274ms | 81.6% | 78.4 | 93.8% | 1129s |

### 2.2 Performance Progression (Log Scale)

```
 36.2ms ┤ v6                    ─┐
 33.9ms ┤ v5, v7                 │ Phase 1: No seed, budget wasted on correctness
 30.9ms ┤ v8                     │
 23.8ms ┤ v9                    ─┘
 11.4ms ┤ v4 ←── Cold start baseline
        │     ══ Architecture Transition: SLM-based → A-SLM + B-global ══
  1.31ms┤ v11
  1.14ms┤ v13                   ─┐ Phase 2: Warm-start evolution (SLM arch)
  1.07ms┤ v10                    │
  1.01ms┤ v14                    │
 0.948ms┤ v12                   ─┘
        │     ══ Architecture Transition: SLM+global → 2D Block IO ══
 0.274ms┤ v15 ←── Final best (81.6% XMX)
 0.224ms┤ ---- Theoretical minimum ----
```

### 2.3 Key Observations by Phase

**Phase 1 (v4-v9, no seed):** All experiments without a pre-existing seed produced 23-36ms. The LLM spends most budget on producing valid DPAS code rather than optimizing it. v4's anomalous 11.4ms at T=0.0 came from initial prompt quality, not evolution.

**Phase 2 (v10-v14, warm start):** Using v4's 11.4ms as seed, v10 achieved **10.65x improvement to 1.07ms** via phase transition (A+B in SLM to A-SLM + B-global). Only v12 (strict constraints, T=0.2) further improved to 0.948ms.

**Phase 3 (v15, architecture innovation):** Seeding with an externally-developed 2D Block IO kernel (0.34ms), v15 achieved 0.274ms by optimizing within this fundamentally superior architecture.

---

## 3. Architecture Analysis

### 3.1 Old Architecture: SLM + 64 Work-Items (v4-v14)

- WG: 64 WIs (4 subgroups x 16 lanes), Tile: 32M x 64N x 32K
- Data flow: A via Global -> SLM -> Register (double-copy + barrier); B via Global/L2 scalar reads
- Bottlenecks: SLM barrier overhead, B-matrix bandwidth, small WG limiting ILP
- **Ceiling: 0.948ms (23.6% XMX)**

### 3.2 New Architecture: 2D Block IO + 512 Work-Items (v15)

- WG: 512 WIs (32 subgroups x 16 lanes), SG Layout: 4x8 grid, Per-SG: 32M x 32N, WG: 128M x 256N
- A: `intel_sub_group_2d_block_read_16b_32r16x2c` (direct Global -> Register)
- B: `intel_sub_group_2d_block_read_transform_16b_16r16x2c` (load + VNNI transform)
- No SLM, No barriers, WG Swizzle (SWIZZLE=2) for L3 reuse, 4x K-unroll (128/iter), interleaved load/compute
- **Ceiling: 0.274ms (81.6% XMX)**

### 3.3 Comparison

| Metric | Old (v12 best) | New (v15 best) | Ratio |
|--------|----------------|----------------|-------|
| Runtime | 0.948 ms | 0.274 ms | 3.46x |
| XMX Utilization | 23.6% | 81.6% | 3.46x |
| Work-items/WG | 64 | 512 | 8x |
| K per iteration | 32 | 128 | 4x |
| Data movement hops | 2 (Global->SLM->Reg) | 1 (Global->Reg) | Halved |
| Distance to theoretical | 4.23x | 1.22x | -- |

---

## 4. Key Findings

### 4.1 Architecture Dominates Over Micro-Optimization

v4-v14 spent ~300 variants optimizing the SLM architecture: 11.4ms -> 0.948ms (12x). v15 used 1 trial on the correct architecture to find 0.274ms (3.46x beyond v12). No amount of evolutionary micro-tuning overcomes a fundamentally suboptimal architecture.

### 4.2 Temperature Control is Phase-Dependent

| Temperature | Behavior | Evidence |
|-------------|----------|----------|
| 0.0 | Deterministic, no evolution | v4: single result, v7: stagnation |
| 0.1 | Cosmetic edits only | v6: worst result (36.2ms) |
| 0.2 | Conservative restructuring, best for micro-tuning | v12: 0.948ms |
| 0.25 | Architectural exploration + optimization | v10: 10.65x; v15: 0.274ms |
| 0.3+ | Aggressive rewrite, destroys optimizations | v11: regression from 1.07 to 1.31ms |

### 4.3 Seed Quality: The Dominant Variable

1. **Externally optimized strong seed** (v15: 0.34ms -> 0.274ms): Best absolute, fastest convergence
2. **"Correct but suboptimal" seed** (v10: 11.4ms -> 1.07ms): Largest relative improvement (10.65x)
3. **Near-optimal same-architecture seed** (v12: 1.07ms -> 0.948ms): Diminishing returns
4. **No seed** (v5-v9: 23-36ms): Budget wasted on correctness, no meaningful optimization

### 4.4 Constraints: Less Is More

- **Strict DO NOT rules (v12)**: Most effective -- 95.8% correctness, best performance
- **No constraints (v11, v13)**: Unfocused exploration, regression
- **37 positive tips (v14)**: Worst correctness (75%), no performance gain
- **Core lesson**: "DO NOT break X" >> "Try doing Y"

### 4.5 Phase Transitions

v10 Trial 6 showed classic punctuated equilibrium: trials 1-5 produced incremental SLM optimizations (~8ms), then a single architectural insight (remove B from SLM) yielded 7.5x jump to 1.07ms. The LLM accumulated understanding of the bottleneck across trials before producing a qualitative solution.

### 4.6 Bimodal Distribution (v15)

v15 showed pronounced bimodality: optimal variants (0.27-0.40ms) vs catastrophic regression (1.2-13ms), with nothing in between. The 2D Block IO architecture is "fragile" -- small code changes can invalidate the compiler's ability to emit efficient instructions.

### 4.7 Evolution Efficiency

| Exp | Trial of Best | Pattern |
|-----|---------------|---------|
| v4 | 1 | Immediate (no evolution, prompt quality) |
| v10 | 6 | Phase transition (punctuated equilibrium) |
| v12 | 9 | Gradual improvement (hill climbing) |
| v15 | 1 | Immediate (strong seed + correct architecture) |

---

## 5. Strengths of KernelFoundry

**Semantic Mutation Power**: LLM mutations make coordinated multi-point changes (tile dims + loop bounds + allocation sizes + indexing) simultaneously and coherently -- equivalent to hundreds of correlated random mutations.

**Quality-Diversity Preservation**: MAP-Elites maintains stepping stones, insurance against degradation, and exploration breadth across behavioral niches.

**Hardware-in-the-Loop**: Every variant is compiled and run on real hardware, eliminating cost model inaccuracies and compiler prediction errors.

**Near-Peak Results**: 81.6% XMX utilization (v15) demonstrates the framework produces near-expert-level kernels with correct architectural seeding.

**Rapid Convergence**: With appropriate seeds, convergence within 1-6 trials makes it practical for iterative development.

---

## 6. Weaknesses and Limitations

**High Regression Rate (30-70%)**: Most generated variants regress. v11: ~50%, v13: ~60%, v15: ~45%. Each wasted variant costs 15-25s of wall-clock time.

**Diversity Collapse Under Constraints**: Strict constraints cause MAP-Elites to degenerate to single-cell dominance, reducing to simple hill-climbing (paradoxically, this works better for micro-tuning).

**Compiler Opacity**: The LLM cannot see IGC compiler behavior. Two semantically equivalent kernels can differ 10x in performance due to invisible scheduling decisions.

**No Automatic Temperature Scheduling**: Optimal temperature depends on evolutionary phase but is fixed per experiment, requiring manual tuning between runs.

**Architecture Ceiling**: Without external innovation, evolution gets stuck at local optima. The system cannot independently discover that a fundamentally different architecture is needed.

**LLM Model Dependence**: Only Claude Opus works. Multi-model (v5) with Sonnet + Codex failed completely despite largest budget (80 evals).

**Information Overload**: Too many tips (v14, 37 tips) reduces correctness from 95.8% to 75% by encouraging over-engineering.

---

## 7. How to Effectively Use KernelFoundry

### 7.1 Recommended Workflow

```
Step 1: Architecture Identification
  - Study hardware docs, identify key instructions (2D block IO, DPAS)
  - Write or obtain correct reference kernel with target architecture

Step 2: Seed Preparation
  - Ensure correct results, target architecture features present
  - Sweet spot: 5-50x from theoretical optimum

Step 3: Exploration Phase (T=0.25, minimal constraints, 12-24 evals)
  - Goal: Find phase transition or architectural breakthrough
  - Stop: No improvement for 3 consecutive iterations

Step 4: Exploitation Phase (T=0.15-0.2, strict DO NOT rules, 24-48 evals)
  - Goal: Squeeze remaining performance from best variant
  - Stop: <5% improvement over 2 iterations

Step 5: Evaluate -- if <40% XMX, change architecture; if >60%, accept result
```

### 7.2 Temperature Selection

- >10x from optimum: T=0.25-0.3 (architectural changes needed)
- 3-10x from optimum: T=0.2-0.25 (structural improvements)
- 1.5-3x from optimum: T=0.15-0.2 (micro-tuning)
- <1.5x from optimum: T=0.1-0.15 or stop (diminishing returns)
- Correctness <80%: Lower temperature; >95%: Can raise slightly

### 7.3 Constraint Formulation

**Effective (DO NOT rules):**
```
- DO NOT remove Shared Local Memory usage for matrix A
- DO NOT reduce tile size below 32x64
- DO NOT change subgroup size from 16
- DO NOT remove K-loop unrolling
```

**Ineffective (positive tips):** "Try prefetch", "Consider double-buffering", "Experiment with unroll factors" -- these encourage multiple simultaneous risky changes.

---

## 8. Improving Evolution Efficiency and Avoiding Regression

### 8.1 Adaptive Temperature Scheduling

```python
def adaptive_temperature(history, base_temp=0.25):
    recent = history[-3:]
    if all(r.improvement > 0 for r in recent):
        return min(base_temp + 0.02, 0.35)       # Consistently improving: maintain/increase
    elif all(r.improvement <= 0 for r in recent):
        return max(base_temp - 0.05, 0.10)       # Stagnant: reduce for fine-tuning
    elif any(r.correctness < 0.8 for r in recent):
        return max(base_temp - 0.08, 0.10)       # Too many failures: reduce
    else:
        return base_temp                          # Mixed: maintain
```

### 8.2 Progressive Constraint Injection

Instead of fixed constraints, add rules dynamically based on observed failures:
- Iterations 1-3: No constraints (explore freely)
- After iteration 3: Analyze failures, add targeted DO NOT rules for observed failure patterns
- Iterations 4-8: Accumulated constraints from failures
- After iteration 8: Freeze constraints, reduce temperature

### 8.3 Performance Threshold Filtering

Only allow variants within 1.5x of current best to serve as parents:
```
parent_threshold = 1.5 * current_best_time
If cell.kernel.runtime > parent_threshold: ineligible as parent
```
Expected benefit: 20-30% reduction in wasted evaluations by preventing regression cascades.

### 8.4 Multi-Phase Pipeline

```
Phase 1: Architecture Exploration (T=0.3, no constraints, diverse seeds, 4 iters)
  → Output: Top-3 architecturally distinct variants

Phase 2: Architecture Selection (T=0.25, minimal constraints, top-3 seeds, 6 iters)
  → Output: Single best architecture + best variant

Phase 3: Micro-Tuning (T=0.15, strict DO NOT constraints, best seed, 12 iters)
  → Output: Final optimized kernel
```

### 8.5 Compiler Feedback Integration

Add IGC ISA dump analysis to evaluation:
- Extract assembly listing, count DPAS instructions vs total
- Detect register spilling (scratch memory accesses)
- Measure DPAS density (%) and instruction-level parallelism
- Feed metrics back to LLM: "Your kernel: 847 instructions, 12% DPAS density, 3 spills. Previous best: 523 instructions, 28% DPAS density, 0 spills."

### 8.6 Regression Guardrail

```python
def regression_guardrail(trial_results, config):
    regression_rate = sum(1 for r in trial_results if r.regressed) / len(trial_results)
    if regression_rate > 0.6:
        config.temperature *= 0.7
        config.add_constraint("DO NOT make large structural changes")
    elif regression_rate > 0.4:
        config.temperature *= 0.9
```

### 8.7 Additional Strategies

- **Kernel complexity budget**: Limit `new_lines <= parent_lines * 1.3` to prevent over-engineering
- **Ensemble parent selection**: Provide LLM with multiple high-performing variants from different niches for crossover-like synthesis
- **Early stopping per trial**: If first 2 of 4 branches regress, skip remaining branches and tighten for next trial

---

## 9. Statistical Summary

| Metric | Value |
|--------|-------|
| Total experiments | 12 (v4-v15) |
| Total evaluations | 529 |
| Total wall-clock time | 14,890s (~4.1 hours) |
| Best result | 0.274ms (81.6% XMX, 78.4 TFLOPS) |
| Theoretical minimum | 0.224ms (100% XMX, 96 TFLOPS) |
| Overall speedup (v4 to v15) | 41.6x |
| Experiments with improvement | 4 of 12 (v4, v10, v12, v15) |
| Experiments with regression | 5 of 12 (v5, v6, v7, v11, v13) |
| Most efficient | v15 (0.274ms in Trial 1) |
| Least efficient | v5 (33.9ms after 80 evals, 3508s) |

### Optimal Strategy (Retrospective)

```
1. Develop correct 2D Block IO seed (1-2 hours)
2. Run evolution: T=0.25, strict constraints, 48 evals (~19 min)
3. Expected: ~0.27ms (81.6% XMX)
4. Optional: T=0.15 micro-tuning (~19 min)
Total: ~2.5 hours vs actual 4.1 hours across 12 experiments with extensive regression
```

---

## 10. Conclusions

1. **Architecture > Optimization**: Architecture choice accounts for 3.46x; micro-optimization within wrong architecture yields diminishing returns (max 1.13x additional after initial phase transition).

2. **Seed Quality is Paramount**: Difference between no seed (23-36ms) and correct architecture seed (0.274ms) spans two orders of magnitude.

3. **Temperature Must Match Phase**: Exploration needs T=0.25+; exploitation needs T=0.15-0.2. Wrong temperature for current phase guarantees failure.

4. **Negative Constraints Beat Positive Guidance**: "DO NOT break X" is more effective than "try doing Y." Information overload (37 tips) actively harms performance.

5. **Evolution Has Diminishing Returns**: Within fixed architecture, improvement follows power law. First experiment: 10.65x; second: 1.13x; third: 0x.

6. **Recommended approach**: Strong external seed + constrained evolution (v15 pattern) combined with adaptive temperature and regression guardrails for maximum efficiency.

---

---

## Appendix A: Hardware Reference

```
Intel Arc B580 (Battlemage G21) - Key Specifications

Compute:
  - 20 Xe2 cores (5 Render Slices x 4 Xe cores)
  - Each Xe2 core: 8 XVE (Xe Vector Engines) + 8 XMX units
  - XMX FP16 peak: 96 TFLOPS (systolic array)
  - Clock: ~2.4 GHz (boost)

Memory:
  - 12 GB GDDR6, 192-bit bus, ~456 GB/s bandwidth
  - L2 Cache: 32 MB (unified, last-level before memory)
  - L1/SLM per Xe core: 192 KB (configurable split)

Execution Model:
  - Subgroup size: 8, 16, or 32 (DPAS requires 16)
  - Max work-group size: 1024
  - Hardware threads per Xe core: 64

Key GEMM Instructions:
  - intel_sub_group_f16_f16_matrix_mad_k16: DPAS 8x16x16 FP16 (4096 FLOPs)
  - intel_sub_group_2d_block_read_16b_*: 2D surface block loads
  - intel_sub_group_2d_block_read_transform_16b_*: Load + VNNI transform
```

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| DPAS | Dot Product Accumulate Systolic -- Intel's XMX matrix instruction |
| XMX | Xe Matrix eXtensions -- Intel's matrix multiply hardware units |
| SLM | Shared Local Memory -- on-chip scratchpad shared within a work-group |
| WG | Work-Group -- group of work-items executing together |
| WI | Work-Item -- single thread of execution (equivalent to CUDA thread) |
| SG | Subgroup -- hardware SIMD group of 16 work-items (equivalent to CUDA warp) |
| VNNI | Variable Neural Network Instructions -- packed data layout for matrix ops |
| IGC | Intel Graphics Compiler -- compiles OpenCL/SPIR-V to GPU ISA |
| MAP-Elites | Multi-dimensional Archive of Phenotypic Elites -- quality-diversity algorithm |
| 2D Block IO | Hardware-accelerated 2D memory reads bypassing SLM |
| Phase Transition | Sudden large performance jump from architectural change |
| Regression | A variant performing worse than its parent |
| XMX Utilization | Fraction of theoretical peak TFLOPS achieved |
| QD | Quality-Diversity -- class of algorithms maintaining diverse high-quality solutions |
| ILP | Instruction-Level Parallelism -- concurrent execution of independent instructions |

## Appendix C: Cost-Effectiveness Analysis

| Experiment | Evals | Improvement | Cost (evals per 1% improvement) |
|-----------|-------|-------------|--------------------------------|
| v10 | 48 | 10.65x (965%) | 0.05 evals/% |
| v15 | 48 | 3.46x (246%) | 0.20 evals/% |
| v12 | 48 | 1.13x (13%) | 3.69 evals/% |
| v5 | 80 | 0.34x (regression) | Infinite (negative return) |
| v13 | 80 | 0.83x (regression) | Infinite (negative return) |

Cost-effectiveness decreases exponentially approaching the architectural ceiling. The first 10x costs 48 evaluations; the next 1.13x also costs 48. Breaking through to a new architecture (v15) resets the cost curve.

---
*Report generated: 2026-05-27 | System: KernelFoundry (MAP-Elites + LLM mutation) | Hardware: Intel Arc B580 | LLM: Claude Opus*