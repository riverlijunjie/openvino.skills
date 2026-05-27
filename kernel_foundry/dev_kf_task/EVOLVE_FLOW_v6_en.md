# Evolutionary Kernel Optimization for Intel Battlemage GPU

## A Comprehensive Analysis of MAP-Elites-Based LLM-Driven GPU Kernel Search on OpenCL DPAS Matmul

- **Target Hardware**: Intel Battlemage G21 (B580)
- **Task**: FP16 Matrix Multiplication (2048x2560x2048)
- **Framework**: KernelFoundry EVOLVE Mode
- **Date**: May 2026

---

## 1. Introduction

This report analyzes an evolutionary kernel search system that combines:

- MAP-Elites (Quality-Diversity search)
- Island-model evolution
- QD gradient tracking
- LLM-driven code mutation

The optimization target is FP16 matmul on Intel B580, with a focus on leveraging **DPAS/XMX** instructions for high throughput.

Three experiments (v4/v5/v6) were run to evaluate how model choice and temperature impact convergence and final performance.

---

## 2. Evolutionary Algorithm Architecture

### 2.1 Core Loop

For each trial:

1. **Sample** parent/inspirations from MAP-Elites archive
2. **Target** an optimization profile (mutate/diversify)
3. **Prompt** LLM with parent, feedback, and target profile
4. **Generate** multiple candidate kernels (branches)
5. **Evaluate** compile -> correctness -> benchmark
6. **Update** archive/islands with fitness scores

### 2.2 MAP-Elites Behavior Space (4D)

| Dimension | Range | Meaning |
|---|---|---|
| `memory_opt` | 0-3 | Memory hierarchy usage |
| `compute_opt` | 0-3 | Compute algorithm efficiency |
| `parallelism_opt` | 0-3 | Parallel granularity |
| `esimd_opt` | 0-3 | ESIMD/DPAS usage level |

Total cells: $4\times4\times4\times4=256$.

### 2.3 Fitness

$$
\text{combined\_score} = \text{perf\_score} + 3 \cdot I(\text{correct} \land \text{speedup}>0) \cdot \text{runtime\_improvement}
$$

Where `runtime_improvement = reference_runtime / kernel_runtime`.

---

## 3. Experimental Results

### 3.1 Overview

| Exp | Model | Temp | Iters x Branches | Total Evals | Best Runtime | Best Speedup |
|---|---|---:|---:|---:|---:|---:|
| v4 | claude-4-6-opus | 0.0 | 4 x 2 | 8 | **11.4 ms** | **2.98x** |
| v5 | 3-model ensemble | 0.3-0.6 | 20 x 4 | 80 | 33.9 ms | 1.0x |
| v6 | claude-4-6-opus | 0.1 | 10 x 4 | 40 | 36.2 ms | 0.94x |

### 3.2 v4 (single model, temp=0.0)

- Hit high-performance DPAS pattern early
- Correct DPAS operand packing (`short8`/`int8`)
- Reached **2.98x** speedup with only 8 evaluations

### 3.3 v5 (ensemble, higher temperature)

- High correctness, but no performance gain over reference
- Typical failure mode: DPAS intrinsic compiled but used ineffective operand encoding
- Archive coverage stayed low (2/256 cells)

### 3.4 v6 (single model, temp=0.1)

- Showed real convergence trend: ~47 ms -> ~43 ms -> **36.2 ms**
- Abandoned DPAS after early failures, converged to SLM-tiling path
- Better than random search behavior, but trapped in a local optimum

---

## 4. Key Findings

1. **Temperature is critical for DPAS correctness**
   - `temp=0.0` reliably produced correct DPAS coding patterns.
   - `temp>0` introduced enough randomness to derail DPAS path.

2. **Model diversity did not help this precision-sensitive task**
   - Ensemble increased diversity but reduced hardware-specific correctness.

3. **Path dependency is strong**
   - Early failed DPAS generations led search toward “safe” SLM kernels.

4. **Current fitness has blind spots**
   - It cannot represent long-term potential of “almost-fixable” DPAS attempts.

---

## 5. Recommendations

### 5.1 Immediate (config-only)

- Use `claude-4-6-opus` with **temperature=0.0** for DPAS tasks
- Warm-start from known-good v4 kernel archive
- Keep single strong model; diversity should come from prompt targets/branches

### 5.2 Short-term (minor code changes)

- Add mandatory DPAS operand template in `USER_INSTRUCTIONS`
- Add tabu/visit-penalty to reduce repeated target profiles
- Improve launch compatibility for subgroup-oriented kernels

### 5.3 Mid-term (architecture)

- Phased temperature strategy
- Potential-aware fitness (reward DPAS-capable code path)
- Separate DPAS and non-DPAS tracks with migration

### 5.4 Long-term

- Surrogate model pre-screening
- Profiler-guided feedback (e.g., XMX utilization)
- Hierarchical generation and AST-level post-validation/fixups

---

## 6. Conclusion

The evolutionary framework itself is functional and can converge, but performance ceiling is dominated by whether the LLM enters and stays on the correct **DPAS/XMX** path.

For this task class, the most effective setup is:

- strong single model
- deterministic generation (`temperature=0.0`)
- warm-start from proven DPAS kernels

This combination is the most promising route to surpass the current best speedup baseline.
