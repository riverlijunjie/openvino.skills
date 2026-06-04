# 5. Quality-Diversity Gradient Tracking (Optional)

> Back to [SKILL.md](SKILL.md). Reads [BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md) and [MAP_ELITES_LOOP.md](MAP_ELITES_LOOP.md).

This component is **optional**. The loop works without it; with it,
parent-selection and prompt-mutation become history-aware.

The mechanism is a stripped-down version of ideas from the
quality-diversity literature:

- CMA-ME (Fontaine et al. 2020) — Covariance Matrix Adaptation MAP-Elites
- PGA-MAP-Elites (Nilsson & Cully 2021) — Policy Gradient Assisted
- DQD (Fontaine & Nikolaidis 2022) — Differentiable Quality-Diversity

The version described here is much simpler: no neural net, no policy
gradient, just empirical accounting of "which delta in descriptor
space, applied from which cell, historically produced an improvement."

## 5.1 What it tracks

Every parent → child transition produces a record:

```python
TransitionRecord(
    parent_id, child_id,
    parent_coords=(m,c,p,e), child_coords=(m,c,p,e),
    parent_fitness, child_fitness,
    fitness_delta = child_fitness - parent_fitness,
    outcome ∈ {IMPROVEMENT, NEUTRAL, REGRESSION,
               CELL_DISCOVERY, ELITE_REPLACEMENT},
    timestamp, iteration, mutation_hint,
)
```

Records are stored in a circular buffer of capped size (`max_history`,
default 10000). Per-cell statistics (counts, mean delta, success rate
per outgoing direction) are kept in an LRU cache (`max_cell_cache`,
default 256).

## 5.2 What it produces

Three outputs feed back into the loop:

### 5.2.1 Gradient vector at a cell

```python
get_gradient_at_coords(coords) → (d_m, d_c, d_p, d_e), metadata
```

Each component is a real number; positive means "transitions from this
cell that *increased* this dimension yielded improvements." The
gradient is a weighted sum of:

- a **fitness component** — average fitness delta per outgoing direction,
- an **improvement-rate component** — fraction of transitions that
  improved fitness per direction,
- an **exploration component** — bias toward empty / underexplored cells.

Default weights are `0.4 / 0.4 / 0.2` (`fitness / improvement_rate /
exploration`).

### 5.2.2 Mutation hints

```python
get_mutation_hints_for_parent(parent, max_hints=3)
→ ["increase parallelism_opt — historical success rate 67% over 12 trials",
   "decrease memory_opt — historical success rate 43% over 7 trials",
   ...]
```

These are appended to the prompt as a "Based on history…" block (see
[PROMPT_PIPELINE.md §3.6](PROMPT_PIPELINE.md#36-gradient-mutation-hints)
or just §3.6 of that file).

### 5.2.3 Sampling weights

```python
get_gradient_weighted_sampling_probabilities(candidate_ids, strategy)
→ {pid: probability, ...}
```

Reweights the parent-sampling softmax so parents in productive
neighborhoods are picked more often. Blended with the standard
fitness-proportional softmax via `gradient_sampling_weight` (default 0.3
— so 30% gradient, 70% fitness).

## 5.3 Why this matters

The unmodified MAP-Elites archive treats every elite equally — a parent
that has produced 5 improvements in past trials gets sampled with the
same probability as a parent that has produced 5 regressions. The
gradient tracker breaks that symmetry:

- parents with positive local gradient are preferred,
- prompts include "increase X" hints for parents whose neighborhood
  shows X is the productive direction,
- the loop avoids re-trying directions that have consistently regressed.

Empirically, this reduces wasted trials by roughly 20–30% on tasks with
>10 trials of history. On the first few trials (history empty) it has no
effect, gracefully degrading to plain MAP-Elites.

## 5.4 What it does NOT do

- It does **not** train a model. There's no neural net, no learned
  embedding. It's all empirical-frequency accounting.
- It does **not** affect *cell coordinates*. Coordinates remain a pure
  function of code text (see [BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md)).
- It does **not** replace the descriptor classifier or the optimization-
  aware prompts. It augments them.

## 5.5 When to skip it

Skip the gradient tracker if any of these is true:

- you're targeting <20 total trials (no history to learn from),
- you only have one parent strategy (`mutate` only) — there's no
  direction to choose,
- you're porting to a backend with very different optimization patterns
  and you haven't tuned the descriptor yet — gradient based on a noisy
  descriptor is worse than no gradient.

## 5.6 Cross-references

- This skill: [BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md) (defines the directions the gradient is over).
- This skill: [MAP_ELITES_LOOP.md](MAP_ELITES_LOOP.md) (sampling that this layer reweights).
- This skill: [PROMPT_PIPELINE.md](PROMPT_PIPELINE.md) (consumes the mutation hints).
- This skill: [PSEUDOCODE.py](PSEUDOCODE.py) (`TransitionTracker` class —
  minimum viable implementation; expand as needed).
