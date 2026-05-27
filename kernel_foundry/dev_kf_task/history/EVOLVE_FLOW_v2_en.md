# EVOLVE Mode Flow — Concise Work Summary (v2)

## Executive Summary
This run validated that the EVOLVE pipeline can quickly produce a high-quality OpenCL DPAS matmul kernel on Intel B580.

- Best runtime: **11.4 ms** vs reference **34.0 ms**
- Best speedup: **2.98x**
- Total candidates evaluated: **8** (4 iterations × 2 branches)
- Total runtime: **269.64 s**

Key takeaway: We reached ~3x speedup in the first iteration, but later iterations did not improve further, indicating early stagnation and limited exploration.

---

## Run Snapshot

| Item | Value |
|---|---|
| Script | `kernelfoundry.internal/runme_matmul.sh` |
| Task | `workspace/ocl_matmul_dpas` |
| GPU | Intel Battlemage G21 (B580, `0xe20b`) |
| Model | `claude-4-6-opus` (Intel GNAI gateway) |
| `max_iters` | 4 |
| `branches_per_iteration` | 2 |
| Validation | Enabled |
| Queue mode | Disabled (local execution) |

---

## How the Evolution Loop Works (Short)
1. Sample parent/inspirations from optimization-aware MAP-Elites database.
2. Build optimization-aware prompt (target profile + feedback + examples).
3. Generate branch candidates via LLM.
4. Evaluate each candidate: compile → correctness → benchmark.
5. Update MAP-Elites archive and island populations.

Archive behavior space is 4D:
- memory optimization
- compute optimization
- parallelism optimization
- ESIMD optimization

Each niche keeps only the best-scoring program.

---

## What Happened in This Run

### Best Results
- Trial 1 v1: **11.4 ms**, **2.98x**
- Trial 0 v0: 11.6 ms, 2.93x

### Failure/plateau signals
- One correctness failure (Trial 1 v0).
- Repeated target profile selection in Trials 1–3 (especially `(2,3,3,0)`).
- Most later “correct” variants were either near-tie best or clearly slower (~35.7 ms).

Interpretation: exploitation dominated too early, while search space coverage remained very low (8 samples for a 256-cell behavior grid).

---

## Main Bottlenecks
- **Low sample budget**: too few iterations/branches for MAP-Elites to show strength.
- **Insufficient diversity pressure**: repeated target profiles.
- **Weak feedback signal**: runtime-only logs provide limited optimization guidance.
- **Low archive coverage**: tiny explored fraction of behavior niches.

---

## Recommended Actions (Priority)

### P0 (immediate, config-only)
1. Increase search budget:
   - `max_iters`: 20–50
   - `branches_per_iteration`: 4–8
2. Increase generation diversity:
   - LLM temperature: `0.6–0.8` (instead of `0.0`)

### P1 (config-only)
3. Improve exploration/exploitation balance:
   - `exploration_ratio`: ~0.4
   - `exploitation_ratio`: ~0.5
   - `guidance_exploration_rate`: ~0.3

### P2 (light code + config)
4. Add warm-start/resume from previous archive/checkpoint.
5. Add anti-repetition target selection (e.g., tabu list / visit-penalty).
6. Enable feedback LLM to convert eval logs into actionable guidance.

---

## Suggested Next Validation Plan
Run a controlled A/B test:

- **Baseline**: current settings with longer horizon.
- **Improved config**: P0 + P1 together.

Track:
- Best speedup vs iteration
- Iterations to reach target speedup
- Archive coverage (`occupied_cells / 256`)
- Diversity (unique optimization profiles)

Success criterion: improved profile reaches higher final speedup and better archive coverage at similar compute cost.

---

## Conclusion
The current EVOLVE setup is capable of finding strong kernels quickly, but it plateaus early under a small-budget, low-diversity regime. The highest-ROI next step is to increase search breadth and randomness, then add warm-start and anti-repetition mechanisms for steadier convergence gains.