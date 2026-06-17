# The Evolutionary Optimization Algorithm

Distilled from `kernelfoundry/algorithm/{controller.py, evolve_database_optimization_aware.py,
qd_gradient.py, utils/map_elites_patterns.py, evaluator.py}`.

This is an **AlphaEvolve / FunSearch-style evolutionary loop** with a **MAP-Elites archive**, an
**island model**, and an optional **quality-diversity (QD) gradient** that steers exploration.
The unit being evolved is a *kernel source file*; fitness is *correctness-gated speedup*.

---

## 1. The top-level loop (`Controller.run_single` → `_run_single`)

```
build/validate task  →  for trial in range(max_iters):
    1. PROPOSE   spawn `branches_per_iteration` parallel branches (ThreadPoolExecutor).
                 Each branch:
                   a. sample (parent, inspirations) from the program database
                   b. choose a TARGET optimization profile (a MAP-Elites cell to aim at)
                   c. build an optimization-aware prompt (see optimization_aware_prompting.md)
                   d. call the LLM (single model or ensemble) → raw answer
                   e. extract code from the answer (answer_processor)
    2. EVALUATE  evaluate_batch(children): compile → correctness → benchmark → score
                 (ThreadPoolExecutor, max_workers≈8). See evaluation_and_feedback.md.
    3. STORE     for each evaluated child:
                   - program_database.add(child, iteration=trial)   # MAP-Elites insert
                   - increase_island_counter_and_switch()
                   - if should_migrate(): migrate_programs()         # ring migration
                   - (optionally) persist to SQL database
    4. SELECT    best = select_best_solution(eval_results)          # by combined score
                 save best code + stdout; report prompt fitness (meta-prompting)
    5. STOP?     break if stop_once_correct and best is correct & compiles
return all_programs, id(best_program_overall)
```

Two execution modes:
- **`evolve_mode: false`** — single branch, iterative refinement of one lineage (no archive/islands).
  Used for *translation* / *fix-it* runs (`max_iters` ≈ 10, `branches_per_iteration` 1).
- **`evolve_mode: true`** — the full island/MAP-Elites search above. Used for *optimization* runs
  (`max_iters` ≈ 20–40, `branches_per_iteration` ≈ 4).

### Key controller knobs (Hydra config, see `configs/run.yaml`)
| Knob | Typical | Meaning |
|---|---|---|
| `max_iters` | 3 / 20 / 36 | number of generations |
| `branches_per_iteration` | 1 / 4 | candidates proposed per generation (parallelism) |
| `evolve_mode` | true | enable island + MAP-Elites |
| `stop_once_correct` | false | stop at first correct kernel (translation jobs) |
| `start_from_best` | false | seed iteration 0 from the best kernel found so far (DB) |
| `kernels_iter_0_path` | null | seed iteration 0 from a file |
| `use_optimization_aware_prompting` | true | inject dimension/level technique guidance |
| `use_feedback_llm` | false | a 2nd LLM digests the eval log into feedback |
| `use_gradient_tracking` | true | enable the QD gradient |
| `gradient_sampling_weight` | 0.3 | blend 30 % gradient-informed sampling with 70 % standard |
| `exploration_strategy` | mutate | `mutate` \| `diversify` \| `intensify` \| `esimd_upgrade` |
| `test_reference` | true | benchmark the reference too (enables real speedup) |

---

## 2. The program (genome)

A `Program` (see `algorithm/schemas.py`) carries:
`id, code, language, parent_id, generation, iteration_found, metrics{}, metadata{},
kernel_exec_result (EvalResult), feedback`.

`metadata["feature_coords"]` = the program's MAP-Elites cell (its behavior descriptor).
`metrics["combined_score"]` = its fitness (see §5).

---

## 3. MAP-Elites archive (`OptimizationAwareDatabase`)

A discrete behavioral grid where each cell keeps only its single best (elite) program.

- **Grid**: dimensions = `[memory_opt, compute_opt, parallelism_opt]` (+ optional `esimd_opt`),
  each binned into **4 levels (0–3)** → **64 cells** (256 with ESIMD).
- **`feature_map: dict[coords → elite program id]`** — the archive.
- **`programs: dict[id → Program]`** — every program ever added.
- **Cell assignment is deterministic from the *code*** (static pattern analysis, §4) — runtime
  metrics never move a program between cells; they only decide *who wins a cell*.

**Insert (`add`)**: classify code → coords → if cell empty, occupy; if occupied, replace iff
`combined_score(new) > combined_score(current_elite)`. Returns whether the program became an elite.

**Population cap**: `population_size` (1000). When exceeded, a min-heap drops the worst
*non-elite* programs (elites + global best are protected).

### Island model (on top of the grid)
- `num_islands` (4) sub-populations; programs are tagged with an island id.
- A counter advances every `programs_per_island` (10) inserts, switching the "active" island.
- Every `migration_interval` (10) generations, the top performers of each island are copied to
  the next island (ring topology), at rate `migration_rate` (0.1). Copy-mode preserves source
  diversity. Islands keep several independent search frontiers alive (anti-premature-convergence).

---

## 4. Behavior descriptor: classifying code into a cell

`OptimizationFeatureClassifier.classify_from_code(code) → (memory, compute, parallelism[, esimd])`.

For each dimension, level patterns (regexes) are scored top-down (level 3 → 1); the first level
whose weighted match score clears a threshold `≈ 0.15 + (level-1)*0.03` wins; else level 0.
The four dimensions and what each level *means* (the optimization ladder the search climbs):

- **memory_opt** — 0 naive global → 1 vectorized/coalesced (float4, `block_load`) →
  2 shared/local-memory tiling (`__shared__`, `local_accessor`, barriers) →
  3 register blocking + async/double-buffer (`async_work_group_copy`, prefetch).
- **compute_opt** — 0 multi-pass → 1 fusion/FMA → 2 single-pass online (online-softmax, Welford) →
  3 blocked/tiled algorithms (Flash-Attention style, `joint_matrix`).
- **parallelism_opt** — 0 thread-only → 1 work-group/block reductions (tree, barriers) →
  2 sub-group/warp collectives (`reduce_over_group`, `__shfl_*`) → 3 hierarchical (block→warp→thread).
- **esimd_opt** (optional) — 0 none → 1 basic ESIMD/WMMA → 2 optimized (LSC, cache hints) →
  3 expert (DPAS / Tensor Cores, software pipelining).

See `scripts/optimization_classifier.py` for a runnable, faithful subset.

---

## 5. Fitness (`EvalResult.compute_performance_score`)

A **correctness gate plus a speedup bonus**, so any correct kernel beats any incorrect one:

```python
score = perf_score                      # 0..5 discrete gate (see evaluation_and_feedback.md)
if runtime_improvement > 0:             # test_reference=True → real speedup ref/custom
    score += runtime_improvement
elif runtime_improvement == -1 and correct and runtime > 0:
    score += 1.0 / runtime              # no reference → prefer faster kernels
```

`perf_score`: 0 extraction/syntax error · 1 not compiled · 2 runtime error · 3 shape mismatch ·
4 value mismatch · 5 correct. `runtime_improvement` = geometric mean of per-benchmark
`ref_runtime / custom_runtime`. This single function is the source of truth for elite selection,
best-kernel selection, and prompt-evolution fitness.

---

## 6. Parent & inspiration sampling (the selection operator)

`OptimizationAwareDatabase.sample() → (parent, inspirations)`.

**Parent** — three strategies by configured ratio:
- **exploitation** (`exploitation_ratio` 0.7): fitness-proportional softmax over the archive elites.
- **exploration** (`exploration_ratio` 0.2): sample from the full current island (not just elites).
- **random** (remaining ~0.1): uniform over all programs.

**Inspirations** (`num_inspirations` 2) — drawn from *different* cells than the parent to force
cross-pollination, with an **optimization-level bonus** `max(0, sum(child_levels) −
sum(parent_levels)) * 0.1` so the model learns from *more sophisticated* kernels; then global
top programs (`num_top_programs` 1), then other islands, then random correct programs.

---

## 7. QD gradient (optional steering) — `qd_gradient.py`

Tracks every parent→child **transition** as a `TransitionRecord`
(`parent_coords, child_coords, fitness_delta, outcome ∈ {IMPROVEMENT, NEUTRAL, REGRESSION,
CELL_DISCOVERY, ELITE_REPLACEMENT}`). From the history it estimates a per-dimension gradient:

```
gradient[d] = α·fitness_grad[d] + β·improvement_rate_grad[d] + γ·exploration_grad[d]
              (α=0.4 fitness, β=0.4 success-rate, γ=0.2 toward empty/weak cells)
```

`fitness_grad` = time-decayed mean Δfitness in direction ±d; `improvement_rate_grad` =
P(improve | +d) − P(improve | −d); `exploration_grad` points toward empty/low-quality cells.
The controller blends this with standard sampling at weight `gradient_sampling_weight` (0.3) to
pick the **target cell** the next prompt should aim for. Buffers: `max_history` 10000,
`max_cell_cache` 256, `checkpoint_interval` 100.

---

## 8. Choosing the target optimization profile (`_get_target_optimization_profile`)

Before each generation, decide which cell to aim at:
- If the parent is correct and `random() > guidance_exploration_rate`: pick an
  **under-explored region** (`get_underexplored_regions`) — empty/weak cells the gradient favors.
- Otherwise **mutate the parent's profile** per `exploration_strategy`:
  `mutate` (±1 on one random dimension), `diversify` (jump several dims),
  `intensify` (push the weakest dimension up), `esimd_upgrade` (raise the ESIMD axis).
The chosen profile becomes the "Required Optimizations" the prompt demands (see prompting doc).

---

## 9. Minimal pseudo-code for the whole search

```python
db = MapElitesArchive(dims=4, levels=4, num_islands=4)
db.add(seed_program)                         # iteration 0 (reference / scaffold / best-so-far)
for it in range(max_iters):
    children = []
    for _ in range(branches_per_iteration):  # parallel
        parent, insps = db.sample()
        target = pick_target_cell(parent, db, gradient)     # §7–8
        prompt = build_prompt(task, parent, insps, target)  # optimization-aware, §prompting
        code   = extract_code(llm(prompt))
        child  = Program(code=code, parent_id=parent.id)
        children.append(child)
    for child in evaluate_batch(children):    # compile→test→benchmark→score, §evaluation
        db.add(child, iteration=it)           # MAP-Elites insert + island switch + migration
    if stop_once_correct and any(c.correct for c in children): break
return db.best()
```

`scripts/evolve_loop.py` is a runnable, backend-agnostic implementation of exactly this.
