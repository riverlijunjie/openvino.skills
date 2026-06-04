# 2. The MAP-Elites Loop

> Back to [SKILL.md](SKILL.md). Reads [BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md).

The archive is a MAP-Elites grid keyed by the 4D descriptor (see
[BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md)), with optional island
sharding and migration. The "mutation operator" is the LLM call — there
is no GA-style crossover or random bit-flip.

This document covers four things:

1. The data structure.
2. How `add()` decides whether a candidate becomes an elite.
3. How `sample()` picks a parent for the next trial.
4. The island model and migration.

## 2.1 The data structure

```
feature_map: {(m, c, p, e): program_id}     # one elite per cell
programs:    {program_id: Program}          # all programs ever seen
islands:     [set[program_id], ...]         # N disjoint subsets of programs
```

The grid shape is fixed at `(4, 4, 4, 4) = 256` cells. Each cell holds at
most one program — the **elite** — chosen by `combined_score` (see
[EVALUATOR.md](EVALUATOR.md) for the score's definition).

Programs that are evicted from their cell still live in `programs` (so the
gradient tracker can refer to them) but no longer participate in sampling.
A separate population-size limit prunes the lowest-scoring evicted
programs back to a configurable cap (default 500).

## 2.2 `add(program)` — elite competition

```python
def add(program):
    coords = classify(program.code)         # static, deterministic
    program.metadata["feature_coords"] = coords
    programs[program.id] = program

    current_elite_id = feature_map.get(coords)
    if current_elite_id is None:
        feature_map[coords] = program.id    # empty cell → fill it
        return True
    if score(program) > score(programs[current_elite_id]):
        feature_map[coords] = program.id    # better → replace
        return True
    return False                            # worse → kept in programs but not elite
```

Three things to notice:

1. **Coordinates come from the code, not from the metrics.** A candidate
   with a bad runtime can still occupy a cell as long as no better
   candidate has landed in that cell yet.

2. **Elite replacement uses `combined_score`** (see [EVALUATOR.md](EVALUATOR.md)),
   not just runtime, so that correct-but-slow candidates beat
   fast-but-incorrect ones (`perf_score` ladder dominates the score below
   the speedup-multiplier threshold).

3. **The cell is decided BEFORE the score is checked.** A candidate that
   reaches a new (m,c,p,e) tuple is automatically a "discovery" even if
   its score is mediocre — `cell_discovery` outcome is recorded in the
   gradient tracker. This is what gives MAP-Elites its diversity bias.

## 2.3 `sample()` — parent selection

Sample a parent from the current island using a three-way mix:

```python
def sample():
    r = random()
    if r < exploitation_ratio:           # default 0.7
        # fitness-proportional over the top-K elites in this island
        parent = softmax_sample(island_top_k(current_island, k=num_top_programs))
    elif r < exploitation_ratio + exploration_ratio:   # default +0.2 → 0.9
        # uniform over underexplored cells (empty + low-quality)
        parent = uniform_from(underexplored_cells())
    else:                                # default 0.1
        # uniform over all elites in this island
        parent = uniform_from(island_elites(current_island))

    inspirations = pick_inspirations(parent, k=num_inspirations)
    return parent, inspirations
```

`underexplored_cells()` returns a mix of empty cells (priority) and
occupied cells whose elite score is below the population's median. This
is how the loop avoids getting stuck in one corner of the grid.

The QD gradient (see [QD_GRADIENT.md](QD_GRADIENT.md)) optionally reweights
the softmax by historical improvement rates.

## 2.4 Inspirations

When constructing a prompt, the loop includes:

- the **parent** (whose code the LLM should improve),
- the **top program** in the parent's island (best performer overall),
- a few **inspirations** — additional elites from the same island, chosen
  for diversity (different cells from the parent).

Inspirations are not parents; they appear in the prompt as "previous
versions" the model can borrow ideas from. Empirically, 1 parent +
1 top + 2 inspirations is the sweet spot — adding more bloats the prompt
without measurable gain.

## 2.5 Islands and migration

The archive is split into N "islands" — disjoint subsets of programs. The
controller cycles through islands round-robin: every `programs_per_island`
adds, the current_island advances. Each island therefore evolves
semi-independently.

Every `migration_interval` adds (default 10), the top fraction
(`migration_rate`, default 0.1) of each island is cloned into the next
island. This is "ring migration" — keeps islands from diverging
permanently while preserving local search momentum.

For small budgets (~30 trials) one island is fine. For longer runs
(100+ trials) 2–3 islands measurably improve diversity coverage.

## 2.6 Population cap

If `programs.size > population_size` (default 500), the lowest-scoring
non-elite programs are deleted until we're back under cap. Elites are
*never* deleted by this path — they are only displaced by a better elite
in the same cell.

This matters because the gradient tracker's history can refer to
deleted programs by ID; the tracker handles that gracefully by treating
them as "parent unknown."

## 2.7 What's load-bearing vs scaffolding

Load-bearing for search quality:

- the descriptor + cell competition (item 2.2),
- the explore/exploit mix in sampling (item 2.3),
- the inspirations injection (item 2.4).

Scaffolding (helps marginally, can be skipped in re-implementation):

- island model (only matters at >100 trials),
- population cap (memory hygiene),
- migration (only matters with >1 island).

If you want a 1-day port of the algorithm, drop islands and population
caps. Keep the descriptor, the cell-elite rule, and the explore/exploit
mix.

## 2.8 Cross-references

- This skill: [BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md) (defines the cells).
- This skill: [PROMPT_PIPELINE.md](PROMPT_PIPELINE.md) (consumes parent + inspirations).
- This skill: [QD_GRADIENT.md](QD_GRADIENT.md) (optionally reweights sampling).
- This skill: [PSEUDOCODE.py](PSEUDOCODE.py) (`Archive` class — concrete reference implementation).
