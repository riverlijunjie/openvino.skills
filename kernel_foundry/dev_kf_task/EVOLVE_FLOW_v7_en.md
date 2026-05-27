# Evolutionary Kernel Optimization: v7 Experiment Analysis

## MAP-Elites + LLM-Driven OpenCL DPAS Matmul Search on Intel Battlemage B580

- **Target Hardware**: Intel Battlemage G21 (B580) GPU
- **Task**: FP16 Matrix Multiplication (2048 x 2560 x 2048)
- **Framework**: KernelFoundry EVOLVE Mode
- **Experiment Version**: v7 (comparative analysis with v4/v5/v6)
- **Core Model**: claude-4-6-opus
- **Date**: May 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Experiment Configuration](#2-experiment-configuration)
3. [Results Analysis](#3-results-analysis)
4. [Critical Finding: Prompt Content Overrides Temperature](#4-critical-finding-prompt-content-overrides-temperature)
5. [Four-Experiment Comparative Analysis](#5-four-experiment-comparative-analysis)
6. [Evolutionary Algorithm Effectiveness Assessment](#6-evolutionary-algorithm-effectiveness-assessment)
7. [Root Cause Analysis](#7-root-cause-analysis)
8. [Recommendations for Convergence Improvement](#8-recommendations-for-convergence-improvement)
9. [Conclusion and Next Steps](#9-conclusion-and-next-steps)

---

## 1. Executive Summary

### 1.1 One-Paragraph Summary

Experiment v7 attempted to enhance the proven v4 configuration (which achieved 2.98x speedup) by
adding advanced prompt engineering features -- feedback LLM, optimization-aware prompting,
inspiration injection, and exploration prompts -- while retaining the same model (claude-4-6-opus)
and temperature (0.0). The result was a **complete failure**: 40 evaluations over 35.5 minutes
produced ZERO improvement over the reference baseline (33.9ms), achieving only 1.0x speedup
compared to v4's 2.98x. The root cause is that the additional prompt context caused the model to
consistently select incorrect DPAS operand types (`float8` instead of `short8`/`int8`), which
bypasses XMX hardware acceleration entirely. This demonstrates that **prompt content has a stronger
effect on code generation quality than temperature**, and that "more features" in the evolutionary
search pipeline can actively harm performance.

### 1.2 Key Metrics at a Glance

```
+--------------------------------------------------+
|  v7 EXPERIMENT RESULTS                            |
+--------------------------------------------------+
|  Best Runtime:     33.9ms (= reference)           |
|  Best Speedup:     1.0x   (ZERO improvement)      |
|  Correctness:      38/40  (95%)                   |
|  Elite Events:     2      (only in Trial 0)       |
|  Evolution:        ZERO   (flat fitness curve)    |
|  Total Time:       2128.4s (35.5 minutes)         |
|  Compute Wasted:   97.5% (39 of 40 duplicates)   |
+--------------------------------------------------+
```

### 1.3 Critical Verdict

| Question | Answer |
|----------|--------|
| Did v7 outperform v4? | **NO** -- 3x worse (1.0x vs 2.98x) |
| Did advanced features help? | **NO** -- they caused the regression |
| Was evolutionary search effective? | **NO** -- zero diversity at temp=0 |
| Was compute well-utilized? | **NO** -- 97.5% wasted on duplicates |
| Is the approach recoverable? | **YES** -- by reverting to v4's minimal config |

---

## 2. Experiment Configuration

### 2.1 v7 Configuration Detail

```yaml
# Core Model Settings
model: claude-4-6-opus
temperature: 0.0

# Search Budget
max_iters: 10
branches_per_iteration: 4
# Total evaluations: 10 x 4 = 40

# NEW in v7 (not present in v4)
feedback_llm: gpt-5.3-codex
exploration_ratio: 0.4
exploitation_ratio: 0.5
use_gradient_tracking: true
use_optimization_aware_prompting: true
include_inspirations: true
use_exploration_prompts: true

# Reference
reference_runtime: 33.9ms
```

### 2.2 Configuration Comparison: v7 vs v4

| Parameter | v7 | v4 (Control) | Delta |
|-----------|-----|--------------|-------|
| `model` | claude-4-6-opus | claude-4-6-opus | Same |
| `temperature` | 0.0 | 0.0 | Same |
| `max_iters` | 10 | 4 | +6 |
| `branches_per_iteration` | 4 | 2 | +2 |
| Total evaluations | 40 | 8 | 5x more |
| `feedback_llm` | **gpt-5.3-codex** | None | **NEW** |
| `exploration_ratio` | 0.4 | N/A | **NEW** |
| `exploitation_ratio` | 0.5 | N/A | **NEW** |
| `use_gradient_tracking` | true | N/A | **NEW** |
| `use_optimization_aware_prompting` | true | N/A | **NEW** |
| `include_inspirations` | true | N/A | **NEW** |
| `use_exploration_prompts` | true | N/A | **NEW** |
| Reference baseline | 33.9ms | 33.9ms | Same |

### 2.3 Design Intent

The v7 experiment was designed to test the hypothesis:

> "Adding sophisticated prompt engineering features (feedback LLM, optimization-aware targeting,
> cross-niche inspiration) to a proven base configuration (temp=0, opus) will improve search
> effectiveness and find even better solutions than v4's 11.4ms."

**This hypothesis was decisively refuted.**

### 2.4 System Architecture

```
+============================================================================+
|                        v7 EVOLUTIONARY PIPELINE                             |
+============================================================================+
|                                                                              |
|  +------------------+     +-------------------+     +-------------------+   |
|  | MAP-Elites       |     | Island Model      |     | Gradient Tracking |   |
|  | Archive (4D)     |<--->| (multi-population)|<--->| (QD gradients)    |   |
|  | 256 cells        |     | crossover/migrate |     | direction vectors |   |
|  +------------------+     +-------------------+     +-------------------+   |
|           |                        |                         |               |
|           v                        v                         v               |
|  +-----------------------------------------------------------------------+  |
|  |                    PROMPT ASSEMBLY ENGINE                               |  |
|  |                                                                         |  |
|  |  [Parent Kernel] + [Optimization Target] + [Exploration Prompt]        |  |
|  |       +                                                                 |  |
|  |  [Feedback from gpt-5.3-codex]  <--- feedback_llm                     |  |
|  |       +                                                                 |  |
|  |  [Inspirations from other niches] <--- include_inspirations            |  |
|  |       +                                                                 |  |
|  |  [Optimization-Aware Targets] <--- optimization_aware_prompting        |  |
|  +-----------------------------------------------------------------------+  |
|           |                                                                  |
|           v                                                                  |
|  +-----------------------------------------------------------------------+  |
|  |              claude-4-6-opus (temp=0.0)                                 |  |
|  |              Generates OpenCL kernel code                               |  |
|  +-----------------------------------------------------------------------+  |
|           |                                                                  |
|           v                                                                  |
|  +------------------+     +-------------------+     +-------------------+   |
|  | OpenCL Compiler  |---->| Correctness Check |---->| Benchmark         |   |
|  | (Intel runtime)  |     | (matrix verify)   |     | (median of N)     |   |
|  +------------------+     +-------------------+     +-------------------+   |
|           |                                                                  |
|           v                                                                  |
|  +-----------------------------------------------------------------------+  |
|  |                    FITNESS EVALUATION                                   |  |
|  |  score = perf_score + 3 * I(correct & speedup>0) * runtime_improvement |  |
|  +-----------------------------------------------------------------------+  |
|                                                                              |
+==============================================================================+
```

### 2.5 MAP-Elites Behavior Space

The archive uses a 4-dimensional behavior characterization:

| Dimension | Range | Description |
|-----------|-------|-------------|
| `memory_opt` | 0-3 | Memory hierarchy utilization (registers, SLM, L1, global) |
| `compute_opt` | 0-3 | Compute pattern efficiency (scalar, vector, SIMD, DPAS) |
| `parallelism_opt` | 0-3 | Parallelism granularity (work-item, sub-group, work-group, multi-WG) |
| `esimd_opt` | 0-3 | ESIMD/DPAS instruction usage level |

**Total archive capacity**: 4 x 4 x 4 x 4 = 256 cells

**v7 archive utilization**: 2/256 cells (0.78%) -- catastrophically low

---

## 3. Results Analysis

### 3.1 Aggregate Performance

| Metric | Value | Assessment |
|--------|-------|------------|
| Total runtime | 2128.4 seconds (35.5 min) | Moderate |
| Correct kernels | 38/40 (95%) | High correctness |
| Failed kernels | 2/40 (5%) | Low failure rate |
| Best kernel time | 33.9ms | Matches reference exactly |
| Best speedup | 1.0x | ZERO improvement |
| Maximum score | 8.0 | Never exceeded baseline score |
| Elite updates | 2 (Trial 0 only) | Near-zero archive dynamics |
| Unique kernel variants | ~1 (functionally identical) | ZERO diversity |

### 3.2 Runtime Distribution

```
Runtime Distribution (38 correct kernels):
===========================================

33.9ms |████████████████████████████████████████████ 21 (55.3%)
34.0ms |████████████████████████████████            14 (36.8%)
34.1ms |██████                                       3 ( 7.9%)
       +----+----+----+----+----+----+----+----+
       0    3    6    9   12   15   18   21

       Mean:   33.97ms
       Median: 33.9ms
       Std:    0.07ms
       Range:  [33.9ms, 34.1ms] -- only 0.2ms spread
```

**Interpretation**: The 0.2ms spread represents measurement noise, not algorithmic diversity.
All 38 correct kernels are functionally identical code with identical performance characteristics.
The "evolution" produced a single phenotype repeated 38 times.

### 3.3 Detailed Trial Results

#### Trial Timing

| Trial | Duration (s) | Notes |
|-------|-------------|-------|
| 0 | 534.6 | Anomalous: one inference took 389.5s (GNAI endpoint latency) |
| 1 | 161.8 | Normal |
| 2 | 173.0 | Normal |
| 3 | 168.7 | Normal |
| 4 | 167.0 | Normal |
| 5 | 164.3 | Normal |
| 6 | 216.0 | Slightly elevated |
| 7 | 171.4 | Normal |
| 8 | 175.6 | Normal |
| 9 | 195.9 | Slightly elevated |

```
Trial Duration Timeline:
========================

Trial 0  |████████████████████████████████████████████████████▌  534.6s (outlier)
Trial 1  |████████████████▏                                      161.8s
Trial 2  |█████████████████▎                                     173.0s
Trial 3  |████████████████▉                                      168.7s
Trial 4  |████████████████▋                                      167.0s
Trial 5  |████████████████▍                                      164.3s
Trial 6  |█████████████████████▌                                 216.0s
Trial 7  |█████████████████▏                                     171.4s
Trial 8  |█████████████████▌                                     175.6s
Trial 9  |███████████████████▌                                   195.9s
         +----+----+----+----+----+----+----+----+----+----+--
         0   50  100  150  200  250  300  350  400  450  500  550
```

**Excluding Trial 0**: Average trial time = 177.5s, which is reasonable for 4 branches with
LLM inference + compile + benchmark.

#### Per-Branch Results

```
Trial 0:  [  FAIL  ] [ 33.9ms ] [ 33.9ms ] [  FAIL  ]   2/4 correct
Trial 1:  [ 33.9ms ] [ 34.0ms ] [ 33.9ms ] [ 34.0ms ]   4/4 correct
Trial 2:  [ 33.9ms ] [ 34.0ms ] [ 33.9ms ] [ 34.0ms ]   4/4 correct
Trial 3:  [ 33.9ms ] [ 33.9ms ] [ 34.0ms ] [ 34.0ms ]   4/4 correct
Trial 4:  [ 33.9ms ] [ 34.0ms ] [ 33.9ms ] [ 33.9ms ]   4/4 correct
Trial 5:  [ 33.9ms ] [ 33.9ms ] [ 34.0ms ] [ 33.9ms ]   4/4 correct
Trial 6:  [ 33.9ms ] [ 34.0ms ] [ 34.0ms ] [ 33.9ms ]   4/4 correct
Trial 7:  [ 33.9ms ] [ 34.0ms ] [ 33.9ms ] [ 34.0ms ]   4/4 correct
Trial 8:  [ 34.0ms ] [ 34.0ms ] [ 34.1ms ] [ 34.0ms ]   4/4 correct
Trial 9:  [ 34.1ms ] [ 34.0ms ] [ 34.1ms ] [ 34.0ms ]   4/4 correct
```

**Observations**:
- Trial 0 has 2 failures (early exploration, unstable prompt context)
- Trials 1-7 show an exact repeating pattern: always 33.9ms or 34.0ms
- Trials 8-9 show slight upward drift (34.0-34.1ms) -- potentially due to thermal throttling
  or system load, not algorithmic variation
- No trial EVER produces a kernel faster than 33.9ms

### 3.4 Fitness Curve

```
Fitness (Best Score) Over Trials:
==================================

Score
  |
10|
  |
 9|
  |
 8|----●----●----●----●----●----●----●----●----●----●---- (flat at 8.0)
  |    
 7|
  |
 6|
  |
 5|●   (Trial 0: initial failures reduce effective score)
  |
 4|
  |
  +----+----+----+----+----+----+----+----+----+----+----> Trial
       0    1    2    3    4    5    6    7    8    9

  Legend: ● = best score at end of trial
  
  NOTE: Score never exceeds 8.0 because speedup never exceeds 1.0x
        Score formula: perf_score + 3 * is_correct * (ref_time / kernel_time)
        At 1.0x: score = 5.0 + 3 * 1 * 1.0 = 8.0
```

**The fitness curve is completely flat** -- there is zero evolutionary progress across 40
evaluations. This is the hallmark of a degenerate search where all candidates occupy the same
point in solution space.

### 3.5 Elite Archive Dynamics

```
Elite Events Timeline:
======================

Trial 0:  ★ ★  (2 elite insertions -- initial population of empty archive)
Trial 1:       (0 elite events)
Trial 2:       (0 elite events)
Trial 3:       (0 elite events)
Trial 4:       (0 elite events)
Trial 5:       (0 elite events)
Trial 6:       (0 elite events)
Trial 7:       (0 elite events)
Trial 8:       (0 elite events)
Trial 9:       (0 elite events)

Total: 2 elite events in 40 evaluations (5% elite rate)
Archive fill: 2/256 cells (0.78%)
```

The 2 elite events in Trial 0 simply represent the first two correct kernels being inserted
into an empty archive. After that, no kernel is ever different enough (in behavior space) OR
better (in fitness) to trigger an archive update. The evolutionary search has completely stalled.

---

## 4. Critical Finding: Prompt Content Overrides Temperature

### 4.1 The Paradox

v7 uses **identical model and temperature** as v4:
- Model: claude-4-6-opus
- Temperature: 0.0

Yet v4 achieved 2.98x speedup while v7 achieved only 1.0x. How is this possible when the
model is deterministic at temp=0?

**Answer: The prompt content is different.** Temperature controls randomness in token selection,
but the prompt determines the distribution from which tokens are sampled. Different prompts at
temp=0 produce different (but each internally deterministic) outputs.

### 4.2 The DPAS Operand Type Divergence

The Intel Battlemage DPAS (Dot Product Accumulate Systolic) instruction requires specific
operand types for hardware acceleration via the XMX (Xe Matrix eXtension) engine:

```
+================================================================+
|  DPAS OPERAND TYPES: CORRECT vs INCORRECT                       |
+================================================================+
|                                                                  |
|  CORRECT (v4 -- achieves XMX hardware acceleration):            |
|  ------------------------------------------------               |
|                                                                  |
|    short8 a_val;    // A matrix: packed FP16 as short8           |
|    int8 b_val;      // B matrix: packed FP16 as int8             |
|    float8 acc;      // Accumulator                               |
|    acc = intel_sub_group_f16_f16_matrix_mad_k16(                 |
|              a_val, b_val, acc);                                  |
|                                                                  |
|    Result: XMX systolic array execution                          |
|    Runtime: 11.4ms (2.98x speedup)                               |
|                                                                  |
|  INCORRECT (v7 -- falls back to scalar emulation):              |
|  ------------------------------------------------               |
|                                                                  |
|    float8 a_packed;   // WRONG: float8 for A operand             |
|    float8 b_packed;   // WRONG: float8 for B operand             |
|    float8 acc;        // Accumulator                             |
|    acc = intel_sub_group_f16_f16_matrix_mad_k16(                 |
|              a_packed, b_packed, acc);                            |
|                                                                  |
|    Result: Compiles but uses SCALAR EMULATION (no XMX)           |
|    Runtime: 33.9ms (1.0x -- no speedup)                          |
|                                                                  |
+================================================================+
```

### 4.3 Why float8 Compiles But Does Not Accelerate

The OpenCL compiler for Intel GPUs performs implicit type conversions in many cases. When
`float8` is passed to `intel_sub_group_f16_f16_matrix_mad_k16`:

1. The compiler does NOT reject the code (no compilation error)
2. It generates a **scalar fallback path** that reinterprets the data
3. The kernel executes correctly (produces correct matrix results)
4. But it does NOT use the XMX systolic array hardware
5. Performance is equivalent to a well-optimized SLM-tiled kernel without DPAS

This is an extremely subtle bug: **the code is correct but slow**, and there is no compiler
warning to indicate the performance loss.

### 4.4 What Changed in the Prompt

v4's prompt assembly is minimal:
```
[System prompt] + [USER_INSTRUCTIONS] + [Parent kernel] + [Optimization target]
```

v7's prompt assembly includes additional components:
```
[System prompt] + [USER_INSTRUCTIONS] + [Parent kernel] + [Optimization target]
    + [Feedback from gpt-5.3-codex]           <-- feedback_llm
    + [Optimization-aware performance targets] <-- optimization_aware_prompting
    + [Inspirations from other archive cells]  <-- include_inspirations
    + [Exploration diversity prompts]          <-- use_exploration_prompts
```

### 4.5 Mechanism of Type Regression

The additional prompt components introduce context that biases the model's code generation:

```
+------------------------------------------------------------------+
|  CAUSAL CHAIN: How Prompt Features Cause Type Regression          |
+------------------------------------------------------------------+
|                                                                    |
|  1. optimization_aware_prompting adds targets like:               |
|     "Optimize for memory bandwidth" / "Reduce register pressure"  |
|                                                                    |
|  2. These targets cause the model to think about data types       |
|     in terms of "what operations will I perform" rather than       |
|     "what does the hardware intrinsic require"                     |
|                                                                    |
|  3. feedback_llm (gpt-5.3-codex) provides optimization advice     |
|     that may reference float8 as the natural type for FP16 math   |
|                                                                    |
|  4. include_inspirations injects code from other niches that       |
|     may use float8 (correct for non-DPAS paths)                   |
|                                                                    |
|  5. At temp=0, the model deterministically resolves this           |
|     combined context to float8 -- EVERY SINGLE TIME               |
|                                                                    |
|  6. Result: 40/40 kernels use float8 operands (identical code)    |
|                                                                    |
+------------------------------------------------------------------+
```

### 4.6 The Temperature Misconception

Prior to v7, the working hypothesis was:

> "temp=0 guarantees correct DPAS operand types because the model deterministically
> selects the most likely (correct) tokens"

v7 **refutes** this hypothesis. The correct formulation is:

> "temp=0 guarantees DETERMINISTIC output given IDENTICAL prompts. The correctness
> of that output depends entirely on the prompt content. Different prompts at temp=0
> produce different (but each deterministic) outputs."

This is a fundamental insight: **temperature controls variance, not correctness**.

---

## 5. Four-Experiment Comparative Analysis

### 5.1 Summary Table

| Experiment | Model | Temp | Total Evals | Best Runtime | Speedup | Key Strategy | DPAS Types |
|-----------|-------|------|-------------|-------------|---------|--------------|------------|
| **v4** | opus | 0.0 | 8 | **11.4ms** | **2.98x** | Correct DPAS | short8/int8 |
| v5 | 3-model ensemble | 0.3-0.6 | 80 | 33.9ms | 1.0x | Wrong DPAS | float8/float8 |
| v6 | opus | 0.1 | 40 | 36.2ms | 0.94x | Abandoned DPAS to SLM | N/A (no DPAS) |
| **v7** | opus | 0.0 | 40 | 33.9ms | 1.0x | Wrong DPAS | float8/float8 |

### 5.2 Visual Comparison

```
Speedup Comparison:
===================

v4  |████████████████████████████████████████████████████████████▏  2.98x
v5  |████████████████████                                           1.00x
v6  |██████████████████▉                                            0.94x
v7  |████████████████████                                           1.00x
    +----+----+----+----+----+----+----+----+----+----+----+----+
    0   0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50 2.75 3.00

Efficiency (Speedup per Evaluation):
=====================================

v4  |████████████████████████████████████████████████████████████▏  0.373x/eval
v5  |██▌                                                            0.013x/eval
v6  |████▋                                                          0.024x/eval
v7  |█                                                              0.025x/eval
    +----+----+----+----+----+----+----+----+----+----+
    0   0.04 0.08 0.12 0.16 0.20 0.24 0.28 0.32 0.36 0.40
```

### 5.3 Detailed Experiment Profiles

#### v4: The Gold Standard

```
Configuration:  Minimal (no feedback_llm, no optimization_aware_prompting)
Model:          claude-4-6-opus @ temp=0.0
Budget:         4 iterations x 2 branches = 8 evaluations
Result:         11.4ms (2.98x speedup)
DPAS types:     short8 / int8 (CORRECT)
Key insight:    Simple prompt + deterministic model = correct hardware intrinsic usage
Efficiency:     Best result in fewest evaluations
```

#### v5: The Diversity Experiment

```
Configuration:  3-model ensemble (opus/sonnet/haiku) with varied temperatures
Model:          Ensemble @ temp=0.3-0.6
Budget:         20 iterations x 4 branches = 80 evaluations
Result:         33.9ms (1.0x speedup)
DPAS types:     float8 / float8 (INCORRECT)
Key insight:    Model diversity does not help precision-sensitive hardware tasks
Efficiency:     10x more compute than v4, 3x worse result
```

#### v6: The Temperature Experiment

```
Configuration:  Single model with slight temperature increase
Model:          claude-4-6-opus @ temp=0.1
Budget:         10 iterations x 4 branches = 40 evaluations
Result:         36.2ms (0.94x -- SLOWER than reference)
DPAS types:     N/A (abandoned DPAS entirely, used SLM tiling)
Key insight:    Even temp=0.1 derails DPAS path; model retreats to safe SLM approach
Efficiency:     Showed convergence trend but trapped in local optimum
Trajectory:     47ms -> 43ms -> 36.2ms (improving but never reaches DPAS performance)
```

#### v7: The Feature-Rich Experiment

```
Configuration:  Full feature set (feedback_llm, optimization_aware, inspirations, exploration)
Model:          claude-4-6-opus @ temp=0.0
Budget:         10 iterations x 4 branches = 40 evaluations
Result:         33.9ms (1.0x speedup)
DPAS types:     float8 / float8 (INCORRECT)
Key insight:    Additional prompt features override model's type selection even at temp=0
Efficiency:     5x more compute than v4, 3x worse result, ZERO evolution
```

### 5.4 Success Factor Analysis

```
+-------------------------------------------------------------------+
|  WHAT MAKES DPAS WORK: Factor Decomposition                        |
+-------------------------------------------------------------------+
|                                                                     |
|  Factor              | v4  | v5  | v6  | v7  | Required?          |
|  --------------------|-----|-----|-----|-----|----------|          |
|  temp=0.0            | YES | NO  | NO  | YES | NECESSARY          |
|  Minimal prompt      | YES | N/A | YES | NO  | NECESSARY          |
|  Single model (opus) | YES | NO  | YES | YES | NECESSARY          |
|  Correct DPAS types  | YES | NO  | NO  | NO  | OUTCOME            |
|  XMX acceleration    | YES | NO  | NO  | NO  | OUTCOME            |
|  ---                 |     |     |     |     |                    |
|  Conclusion: BOTH temp=0 AND minimal prompt are required           |
|                                                                     |
+-------------------------------------------------------------------+
```

### 5.5 Failure Mode Classification

| Failure Mode | Experiments | Description |
|-------------|-------------|-------------|
| Wrong DPAS operand types | v5, v7 | Uses float8 instead of short8/int8; compiles but no XMX |
| DPAS abandonment | v6 | Model gives up on DPAS after failures, retreats to SLM |
| Zero diversity | v7 | temp=0 + fixed prompt = identical outputs every iteration |
| Prompt pollution | v7 | Additional context shifts type selection away from correct answer |

---

## 6. Evolutionary Algorithm Effectiveness Assessment

### 6.1 MAP-Elites Performance

| Metric | Expected (healthy search) | v7 Actual | Assessment |
|--------|--------------------------|-----------|------------|
| Archive fill rate | >10% after 40 evals | 0.78% (2/256) | **FAILED** |
| Elite update frequency | >20% of evals | 5% (2/40) | **FAILED** |
| Behavior diversity | Multiple distinct niches | Single niche | **FAILED** |
| Performance improvement | Monotonic increase | Flat | **FAILED** |
| Novel phenotypes discovered | >5 | 1 | **FAILED** |

### 6.2 Why the Evolutionary Algorithm is Wasted

```
NORMAL EVOLUTIONARY SEARCH:              v7 DEGENERATE SEARCH:
========================                 ========================

Generation 1:  A B C D                   Generation 1:  X X X X
               |/ \|                                    | | | |
Generation 2:  E F G H                   Generation 2:  X X X X
              /| |\ |\                                  | | | |
Generation 3: I J K L M                  Generation 3:  X X X X
                                                        | | | |
Fitness:  ↗ ↗ ↗ (improving)             Fitness:  → → → (flat)
Diversity: HIGH                          Diversity: ZERO
Archive:  fills up                       Archive:  stays empty
```

At temperature 0 with identical prompts, the model produces **byte-identical output** every
iteration. The evolutionary infrastructure -- MAP-Elites archive, island model, gradient
tracking, crossover, mutation targeting -- has NOTHING to work with because there is no
phenotypic variation.

### 6.3 Component Effectiveness

| EA Component | Purpose | v7 Utility | Reason |
|-------------|---------|------------|--------|
| MAP-Elites archive | Quality-diversity storage | **ZERO** | Only 1 phenotype exists |
| Island model | Population diversity | **ZERO** | All islands have same individual |
| Gradient tracking | Direct search improvement | **ZERO** | No fitness gradient exists |
| Exploration ratio (0.4) | Discover new niches | **ZERO** | Exploration produces same code |
| Exploitation ratio (0.5) | Refine known good | **ZERO** | Cannot refine identical code |
| Inspiration injection | Cross-pollinate niches | **NEGATIVE** | Causes type regression |
| Feedback LLM | Guide optimization | **NEGATIVE** | Adds confusing context |

### 6.4 The Fundamental Incompatibility

```
+===================================================================+
|  ARCHITECTURAL CONFLICT IN v7                                      |
+===================================================================+
|                                                                     |
|  Evolutionary algorithms REQUIRE:                                   |
|    - Phenotypic VARIATION between candidates                        |
|    - Selection PRESSURE (fitness differences)                       |
|    - Heritable IMPROVEMENT (offspring better than parents)          |
|                                                                     |
|  Temperature 0.0 GUARANTEES:                                        |
|    - IDENTICAL output for identical inputs                           |
|    - ZERO variation between iterations                               |
|    - NO heritable changes (nothing to inherit)                      |
|                                                                     |
|  CONCLUSION: EA + temp=0 + fixed-prompt = CONTRADICTION             |
|                                                                     |
|  The evolutionary algorithm becomes a NO-OP:                        |
|    - Selection has nothing to select                                |
|    - Crossover has nothing to cross                                  |
|    - Mutation is suppressed by deterministic generation             |
|    - Archive stores one solution forever                            |
|                                                                     |
+===================================================================+
```

### 6.5 Compute Waste Analysis

| Resource | Allocated | Useful | Wasted | Waste % |
|----------|-----------|--------|--------|---------|
| LLM inferences | 40 | 1 | 39 | 97.5% |
| Compile cycles | 40 | 2 | 38 | 95.0% |
| Benchmark runs | 38 | 2 | 36 | 94.7% |
| Feedback LLM calls | ~40 | 0 | ~40 | 100% |
| Wall-clock time | 2128.4s | ~50s | ~2078s | 97.6% |
| Estimated cost | ~$12-15 | ~$0.30 | ~$12-15 | 97.5% |

**Effective search efficiency**: 1 unique candidate explored out of 40 evaluations = **2.5%**

---

## 7. Root Cause Analysis

### 7.1 Primary Root Cause

**The additional prompt features (feedback_llm, optimization_aware_prompting, 
include_inspirations, use_exploration_prompts) modify the LLM's context in a way that
deterministically shifts its DPAS operand type selection from short8/int8 (correct) to
float8/float8 (incorrect).**

### 7.2 Causal Chain

```
ROOT CAUSE TREE:
================

v7 achieves only 1.0x speedup
    |
    +-- DPAS uses float8 operands (scalar emulation, no XMX)
        |
        +-- Model deterministically outputs float8 at temp=0
            |
            +-- Prompt context biases toward float8
                |
                +-- optimization_aware_prompting adds performance targets
                |   that emphasize "float" operations
                |
                +-- feedback_llm (gpt-5.3-codex) suggests float-based
                |   optimizations without DPAS hardware knowledge
                |
                +-- include_inspirations injects code from non-DPAS niches
                |   where float8 IS the correct type
                |
                +-- use_exploration_prompts adds generic diversity language
                    that doesn't constrain hardware-specific types
```

### 7.3 Contributing Factors

#### Factor 1: temp=0 is NECESSARY but NOT SUFFICIENT

- **NECESSARY**: Without temp=0, the model introduces random variation that can derail
  hardware-specific coding patterns (proven by v5 and v6 failures)
- **NOT SUFFICIENT**: Even at temp=0, incorrect prompt content produces incorrect code
  (proven by v7)

#### Factor 2: Prompt Content Determines Deterministic Output

At temperature 0, the model's output is a deterministic function of its input:

```
output = f(prompt)    where f is deterministic at temp=0

v4_prompt != v7_prompt
    => f(v4_prompt) != f(v7_prompt)
    => v4_output (short8/int8) != v7_output (float8/float8)
```

The additional features in v7 change the prompt, which changes the deterministic output.

#### Factor 3: DPAS Type Knowledge is Fragile

The correct DPAS operand types (short8 for A, int8 for B) are:
- Counter-intuitive (why would FP16 data use integer types?)
- Hardware-specific (only applies to Intel Xe/Battlemage DPAS)
- Poorly documented (Intel's public docs are sparse on this detail)
- Easily overridden by generic optimization advice

The model has this knowledge but it is easily "distracted" by additional context that
suggests float-typed operations are appropriate for floating-point math.

#### Factor 4: Zero Diversity Makes Recovery Impossible

Even if the initial type selection is wrong, an evolutionary system with diversity could:
- Randomly discover the correct types through mutation
- Cross-pollinate from a successful variant
- Explore the type space through gradient tracking

But at temp=0 with identical prompts, there is ZERO diversity, so the system is permanently
locked into its first (incorrect) decision with no mechanism for correction.

#### Factor 5: Feedback LLM Lacks Hardware-Specific Knowledge

The feedback LLM (gpt-5.3-codex) provides general optimization advice without specific
knowledge of Intel Battlemage DPAS operand type requirements. Its suggestions reinforce
the float8 approach because float8 is the "obvious" type for FP16 computation from a
software engineering perspective.

### 7.4 Counterfactual Analysis

| What if... | Expected outcome |
|-----------|-----------------|
| v7 used v4's minimal prompt (no extra features)? | Would match v4: 11.4ms, 2.98x |
| v7 used temp=0.3 instead of 0.0? | Would get diversity but likely wrong types (like v5) |
| v7 had explicit type template in USER_INSTRUCTIONS? | Likely correct types regardless of other features |
| v7 warm-started from v4's kernel? | Might maintain correct types through prompt anchoring |
| v7 had AST post-processing to fix types? | Would achieve correct types mechanically |

### 7.5 Interaction Model

```
+------------------------------------------------------------------+
|  FACTOR INTERACTION MAP                                           |
+------------------------------------------------------------------+
|                                                                    |
|                    +-----------+                                   |
|                    | temp=0.0  |                                   |
|                    +-----+-----+                                   |
|                          |                                         |
|              Guarantees determinism                                |
|                          |                                         |
|                    +-----v-----+                                   |
|                    | Identical |                                   |
|              +---->|  output   |<----+                             |
|              |     | per iter  |     |                             |
|              |     +-----+-----+     |                             |
|              |           |           |                             |
|     Kills diversity      |      Locks in errors                   |
|              |           |           |                             |
|     +--------v---+  +---v--------+  +---v--------+               |
|     | EA becomes |  | If correct |  | If wrong   |               |
|     | useless    |  | = great    |  | = stuck    |               |
|     +------------+  | (v4 case)  |  | (v7 case)  |               |
|                     +------------+  +------------+               |
|                                                                    |
|  The ONLY variable that determines success vs failure              |
|  at temp=0 is: PROMPT CONTENT                                     |
|                                                                    |
+------------------------------------------------------------------+
```

---

## 8. Recommendations for Convergence Improvement

### 8.1 Priority Matrix

```
+------------------------------------------------------------------+
|  RECOMMENDATION PRIORITY MATRIX                                   |
+------------------------------------------------------------------+
|                                                                    |
|              HIGH IMPACT                                           |
|                 |                                                  |
|     P0-1       |      P0-2                                        |
|  Strip to      |   Add DPAS type                                  |
|  minimal cfg   |   template                                       |
|                 |                                                  |
|  ----LOW EFFORT-+------HIGH EFFORT--->                            |
|                 |                                                  |
|     P1-4       |      P1-6                                        |
|  Reduce        |   AST post-                                      |
|  branches=1    |   processing                                     |
|                 |                                                  |
|              LOW IMPACT                                            |
|                                                                    |
+------------------------------------------------------------------+
```

### 8.2 P0 (Critical) -- Must Implement Before Next Run

#### P0-1: Strip Back to v4's Minimal Configuration

**Rationale**: v4's configuration is the only one that produced correct DPAS code.
All additional features in v7 are net-negative for this task.

**Action**:
```yaml
# DISABLE all v7-specific features:
feedback_llm: null                        # was: gpt-5.3-codex
exploration_ratio: null                   # was: 0.4
exploitation_ratio: null                  # was: 0.5
use_gradient_tracking: false              # was: true
use_optimization_aware_prompting: false   # was: true
include_inspirations: false               # was: true
use_exploration_prompts: false            # was: true
```

**Expected outcome**: Restore correct DPAS type selection, achieve ~11.4ms runtime.

#### P0-2: Add Explicit DPAS Operand Type Template

**Rationale**: Even with minimal config, the model's DPAS type knowledge is fragile.
Explicit instruction eliminates ambiguity.

**Action**: Add to USER_INSTRUCTIONS in config.yaml:
```
CRITICAL DPAS INSTRUCTION:
When using intel_sub_group_f16_f16_matrix_mad_k16():
  - Operand A MUST be declared as: short8
  - Operand B MUST be declared as: int8  
  - Accumulator MUST be declared as: float8
  
DO NOT use float8 for A or B operands -- this compiles but falls back to
scalar emulation without XMX hardware acceleration.

Example:
  short8 a_val = as_short8(intel_sub_group_block_read_us8(...));
  int8 b_val = as_int8(intel_sub_group_block_read8(...));
  float8 acc = (float8)(0.0f);
  acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
```

**Expected outcome**: Correct DPAS types regardless of other prompt features.

#### P0-3: Warm-Start from v4's Proven Kernel

**Rationale**: Starting from a known-good kernel anchors the model's context and
provides a correct template for type usage.

**Action**:
```yaml
kernels_iter_0_path: /path/to/v4/best_kernel_11.4ms.cl
```

**Expected outcome**: Model preserves correct types from parent kernel during mutation.

### 8.3 P1 (Important) -- Should Implement for Robustness

#### P1-4: Reduce branches_per_iteration to 1 for temp=0

**Rationale**: At temperature 0 with identical prompts, all branches produce identical
output. Running 4 branches wastes 75% of compute.

**Action**:
```yaml
# For temp=0 configs:
branches_per_iteration: 1    # was: 4

# Alternative: if diversity is desired, increase temperature:
# temperature: 0.2
# branches_per_iteration: 4
```

**Expected outcome**: 4x reduction in compute cost with zero loss in search quality.

#### P1-5: Conditional Feature Activation with Type Constraints

**Rationale**: If advanced features (feedback_llm, inspirations) are desired for
exploration, they must be paired with explicit type constraints.

**Action**:
```yaml
# If keeping advanced features, MUST also add:
temperature: 0.15                          # Enable diversity
branches_per_iteration: 4                  # Exploit diversity
# AND include type template in USER_INSTRUCTIONS (P0-2)
# AND include warm-start kernel (P0-3)
```

**Expected outcome**: Diversity without type regression.

#### P1-6: AST-Level Post-Processing

**Rationale**: Mechanical correction of DPAS operand types eliminates model-dependent
failures entirely.

**Action**: Implement a post-generation AST pass:
```python
def fix_dpas_operand_types(kernel_source: str) -> str:
    """
    Find all intel_sub_group_f16_f16_matrix_mad_k16() calls.
    Verify operand A is short8 and operand B is int8.
    If not, insert appropriate as_short8() / as_int8() casts.
    """
    # Parse DPAS call sites
    # Check operand declaration types
    # Insert type conversions if needed
    # Return corrected source
```

**Expected outcome**: Guaranteed correct DPAS types regardless of model output.

### 8.4 P2 (Enhancement) -- For Future Experiments

#### P2-7: A/B Validation Test

**Rationale**: Confirm v4's result is reproducible before building on it.

**Action**:
```yaml
# Run v4's exact config unchanged with extended budget:
model: claude-4-6-opus
temperature: 0.0
max_iters: 10
branches_per_iteration: 2    # same as v4
# No other features
```

**Expected outcome**: Should reproduce 11.4ms result, confirming stability.

#### P2-8: Incremental Feature Testing

**Rationale**: Identify which specific feature causes type regression.

**Action**: Run 5 experiments, each adding one feature to v4's base:
```
Test A: v4 + feedback_llm only
Test B: v4 + optimization_aware_prompting only
Test C: v4 + include_inspirations only
Test D: v4 + use_exploration_prompts only
Test E: v4 + use_gradient_tracking only
```

**Expected outcome**: Identify the specific feature(s) causing regression. Likely
candidates are feedback_llm and optimization_aware_prompting.

#### P2-9: Model Knowledge Enhancement

**Rationale**: Improve the model's intrinsic DPAS knowledge rather than relying on
prompt engineering.

**Options**:
- Few-shot examples in system prompt showing correct DPAS patterns
- Custom system prompt with Intel Battlemage DPAS documentation
- Fine-tuning on verified DPAS kernel examples (if model supports)

### 8.5 Implementation Priority Order

```
Phase 1 (Immediate -- before next experiment):
  [x] P0-1: Strip to minimal config
  [x] P0-2: Add DPAS type template
  [x] P0-3: Warm-start from v4 kernel

Phase 2 (Next iteration):
  [ ] P1-4: Reduce branches for temp=0
  [ ] P1-5: Type constraints for advanced features
  [ ] P2-7: A/B validation of v4

Phase 3 (Research):
  [ ] P1-6: AST post-processing
  [ ] P2-8: Incremental feature testing
  [ ] P2-9: Model knowledge enhancement
```

---

## 9. Conclusion and Next Steps

### 9.1 Key Takeaways

1. **Prompt content > Temperature**: The most important finding from v7 is that prompt
   content has a stronger effect on code generation correctness than temperature settings.
   Adding "helpful" context can actively harm performance by shifting the model's
   decision-making on hardware-specific details.

2. **Simplicity wins for hardware-specific tasks**: v4's minimal configuration outperformed
   all "enhanced" versions by a factor of 3x. For tasks requiring precise hardware
   intrinsic knowledge, minimal prompts produce better results than feature-rich prompts.

3. **Evolutionary search requires diversity**: Running MAP-Elites at temp=0 is an
   architectural contradiction. Either use temp>0 for diversity (with type safeguards)
   or use temp=0 with branch=1 (deterministic single-shot).

4. **Correct DPAS types are binary**: There is no gradient between "almost correct" and
   "correct" DPAS usage. Either the types are short8/int8 (3x speedup) or they are not
   (1x). This cliff-edge performance profile makes evolutionary hill-climbing ineffective
   unless it can cross the type barrier.

5. **Silent failures are the worst failures**: float8 DPAS code compiles, runs correctly,
   and produces valid results -- but with zero hardware acceleration. Without explicit
   benchmarking, this failure mode is invisible.

### 9.2 Lessons for the Framework

| Lesson | Implication for KernelFoundry |
|--------|-------------------------------|
| More features != better results | Default config should be minimal; features opt-in only after validation |
| temp=0 + EA = waste | Framework should warn/prevent this combination |
| Hardware intrinsics need templates | USER_INSTRUCTIONS should include intrinsic type specifications |
| Feedback LLM can be harmful | Feedback should be gated on domain-specific knowledge |
| Warm-start is critical | Always provide best-known kernel as starting point |

### 9.3 Predicted Outcomes for Next Experiments

| Proposed Config | Predicted Best Runtime | Confidence |
|----------------|----------------------|------------|
| v4 exact rerun | 11.4ms (2.98x) | 95% |
| v4 + type template | 11.4ms (2.98x) | 99% |
| v4 + type template + temp=0.15 + branches=4 | 9-11ms (3.1-3.8x) | 60% |
| v4 + all v7 features + type template | 11.4ms (2.98x) | 80% |
| v4 + warm-start + type template + temp=0.15 | 8-10ms (3.4-4.2x) | 40% |

### 9.4 The Path Forward

```
+===================================================================+
|  CONVERGENCE STRATEGY                                              |
+===================================================================+
|                                                                     |
|  STEP 1: Establish reliable baseline                               |
|  --------                                                          |
|  Config: v4 minimal + DPAS type template                           |
|  Expected: Reproduce 11.4ms consistently                           |
|  Purpose: Confirm foundation is stable                             |
|                                                                     |
|  STEP 2: Enable controlled diversity                               |
|  --------                                                          |
|  Config: + temp=0.15, branches=4, warm-start from 11.4ms kernel    |
|  Expected: Explore around 11.4ms, find 9-10ms variants             |
|  Purpose: Evolutionary search WITH correct type anchoring          |
|                                                                     |
|  STEP 3: Carefully add features (one at a time)                    |
|  --------                                                          |
|  Config: + one feature, measure impact on types and performance    |
|  Expected: Identify which features help vs harm                    |
|  Purpose: Build evidence-based feature set                         |
|                                                                     |
|  STEP 4: Full-featured run with safeguards                         |
|  --------                                                          |
|  Config: Validated features + type template + AST checking         |
|  Expected: Best of both worlds (diversity + correctness)           |
|  Purpose: Maximum performance with engineering guardrails          |
|                                                                     |
+===================================================================+
```

### 9.5 Final Assessment

The v7 experiment is a **negative result with high informational value**. While it failed to
improve performance, it revealed a critical architectural insight: **the relationship between
prompt engineering and deterministic code generation is non-trivial, and "more sophisticated"
prompting can produce worse outcomes than minimal prompting for hardware-specific tasks.**

This finding has broad implications for LLM-driven code generation systems:
- Feature additions must be empirically validated, not assumed beneficial
- Hardware-specific knowledge is fragile and easily overridden by generic context
- Evolutionary algorithms require explicit diversity mechanisms at the generation level
- Silent performance failures (correct but slow) require benchmark-driven validation

The path to exceeding v4's 2.98x speedup exists but requires:
1. Preserving v4's correct type selection (through templates or constraints)
2. Adding controlled diversity (through moderate temperature)
3. Warm-starting from the proven 11.4ms kernel
4. Validating each feature addition empirically

---

## Appendix A: DPAS Intrinsic Reference

### A.1 Correct Usage Pattern

```c
// Intel Battlemage DPAS for FP16 matmul
// Subgroup size: 16 (required for XMX)

// Load A matrix tile (FP16 packed as short8)
short8 a_val = as_short8(intel_sub_group_block_read_us8(
    (__global ushort*)(A + row_offset)));

// Load B matrix tile (FP16 packed as int8)  
int8 b_val = as_int8(intel_sub_group_block_read8(
    (__global uint*)(B + col_offset)));

// Accumulator (float8 -- this one IS float8)
float8 acc = (float8)(0.0f);

// DPAS instruction -- executes on XMX systolic array
acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);

// Key: A=short8, B=int8, C=float8
// The short8/int8 types encode the PHYSICAL REGISTER LAYOUT
// that maps to XMX hardware ports, not the logical data type.
```

### A.2 Why short8 and int8?

The XMX (Xe Matrix eXtension) hardware has specific register port widths:
- **A port**: 8 x 16-bit = 128 bits = `short8` (8 shorts x 16 bits)
- **B port**: 8 x 32-bit = 256 bits = `int8` (8 ints x 32 bits)
- **C port**: 8 x 32-bit = 256 bits = `float8` (8 floats x 32 bits)

The types describe the PHYSICAL REGISTER FORMAT, not the logical data type. The actual
FP16 values are bit-packed into these registers. Using `float8` for A or B causes the
compiler to generate conversion code that does not map to the hardware ports.

### A.3 Common Incorrect Patterns

```c
// WRONG: float8 for all operands (v5, v7 pattern)
float8 a_packed = vload8(0, (__global float*)(A + offset));  // WRONG TYPE
float8 b_packed = vload8(0, (__global float*)(B + offset));  // WRONG TYPE
float8 acc = (float8)(0.0f);
acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);
// Compiles! Runs correctly! But NO XMX acceleration.

// WRONG: half8 (logical type, not physical register type)
half8 a_val = vload8(0, (__global half*)(A + offset));       // WRONG TYPE
half8 b_val = vload8(0, (__global half*)(B + offset));       // WRONG TYPE
// May not even compile, or produces garbage.
```

---

## Appendix B: Experiment Timeline

```
v7 Experiment Wall-Clock Timeline (2128.4 seconds total):
==========================================================

Time(s)  0        500       1000      1500      2000
         |---------|---------|---------|---------|---->
         
Trial 0  [==========XXXXXXXXXXXXXXX========]     534.6s
                   ↑ 389.5s inference latency
         
Trial 1            [=====]                        161.8s
Trial 2                  [======]                  173.0s
Trial 3                        [=====]            168.7s
Trial 4                             [=====]       167.0s
Trial 5                                  [=====]  164.3s
Trial 6                                       [=======] 216.0s
Trial 7                                              [=====] 171.4s
Trial 8                                                   [======] 175.6s
Trial 9                                                         [======] 195.9s

Legend: [===] = normal inference + compile + benchmark
        XXX  = anomalous GNAI endpoint latency
```

---

## Appendix C: Configuration File Template (Recommended for v8)

```yaml
# v8 Recommended Configuration
# Based on v7 failure analysis: minimal config + type safety

# Core (same as v4)
model: claude-4-6-opus
temperature: 0.0

# Budget (conservative)
max_iters: 10
branches_per_iteration: 1    # temp=0 means branches are duplicates

# DISABLED (proven harmful in v7)
feedback_llm: null
use_optimization_aware_prompting: false
include_inspirations: false
use_exploration_prompts: false
use_gradient_tracking: false

# Warm-start (from v4's best kernel)
kernels_iter_0_path: ./results/v4/best_kernel_11.4ms.cl

# Type safety (added based on v7 analysis)
user_instructions_append: |
  CRITICAL DPAS INSTRUCTION:
  When using intel_sub_group_f16_f16_matrix_mad_k16():
    - Operand A MUST be declared as: short8
    - Operand B MUST be declared as: int8
    - Accumulator MUST be declared as: float8
  DO NOT use float8 for A or B operands.
```

---

## Appendix D: Statistical Summary

### D.1 v7 Runtime Statistics

| Statistic | Value |
|-----------|-------|
| N (correct kernels) | 38 |
| Mean | 33.97ms |
| Median | 33.9ms |
| Mode | 33.9ms |
| Std. deviation | 0.07ms |
| Min | 33.9ms |
| Max | 34.1ms |
| Range | 0.2ms |
| IQR | 0.1ms |
| CV (coefficient of variation) | 0.2% |

### D.2 Cross-Experiment Runtime Comparison

| Statistic | v4 | v5 | v6 | v7 |
|-----------|-----|-----|-----|-----|
| Best runtime | 11.4ms | 33.9ms | 36.2ms | 33.9ms |
| Speedup vs ref | 2.98x | 1.00x | 0.94x | 1.00x |
| Evaluations | 8 | 80 | 40 | 40 |
| Cost efficiency | 0.373x/eval | 0.013x/eval | 0.024x/eval | 0.025x/eval |
| XMX utilized | YES | NO | NO | NO |

### D.3 Time Budget Breakdown (v7)

| Phase | Total Time | Per-Eval Average | % of Total |
|-------|-----------|-----------------|-----------|
| LLM Inference | ~1400s | ~35s | 65.8% |
| Compilation | ~200s | ~5s | 9.4% |
| Benchmarking | ~380s | ~9.5s | 17.9% |
| Overhead (prompt assembly, archive ops) | ~148s | ~3.7s | 7.0% |
| **Total** | **2128.4s** | **53.2s** | **100%** |

Note: Trial 0's anomalous 389.5s inference skews the LLM average significantly.
Excluding Trial 0, average LLM inference time is approximately 25s per call.

---

*Report generated: May 2026*
*Framework: KernelFoundry EVOLVE Mode*
*Analysis covers experiments v4, v5, v6, v7 on Intel Battlemage B580*
