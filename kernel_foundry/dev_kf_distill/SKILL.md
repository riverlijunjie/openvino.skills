---
name: dev_kf_distill
description: Distilled, self-contained algorithm for an LLM-driven, MAP-Elites-based GPU kernel optimizer — closed-loop generation guided by a static behavior descriptor, optimization-aware prompting, optional QD-gradient tracking, and a tiered evaluator with a continuous combined score. Use this skill when you want to (a) understand what makes such a loop work, (b) re-implement it in a smaller harness or new project, or (c) port it to a new GPU backend.
---

# Rules

- This skill is **research / documentation**. Do not commit, push, or open PRs from it.

# What this skill is

This skill documents one closed-loop algorithm for generating high-performance
GPU kernels from a reference implementation, using an LLM as the only
"mutation operator." It is a **fully self-contained** description: every
file in this folder reads on its own, with no required reading outside the
skill. There are no required cross-references to other skills, source
trees, or experiment logs.

The algorithm and its parts:

```
┌─────────────────────────────────────────────────────────────────────────┐
│   sample_parent → build_prompt → LLM → extract_code → evaluate          │
│       ↑                                                       │         │
│       └────────────── update_archive ←────── score ───────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

Six things make the loop work as well as it does. Each is documented in
its own file in this folder; this `SKILL.md` is the executive summary.

## 1. The behavior descriptor (4D, static, deterministic)

Every candidate is mapped to a 4-tuple of integer levels in `[0..3]`:

```
(memory_opt, compute_opt, parallelism_opt, esimd_opt)
```

The mapping is done by **regex pattern matching on the source code** —
no execution, no LLM, no normalization. This is the single most important
design choice: behavioral coordinates are a property of the *text*, not
of the runtime, so they are reproducible across runs and cheap to
compute.

The full grid is `4 × 4 × 4 × 4 = 256` cells. Each cell stores at most
one elite — the highest-scoring candidate that ever landed there.

See: [BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md)

## 2. MAP-Elites with island model

The archive is a MAP-Elites grid over the descriptor above, optionally
sharded into N islands that evolve in parallel and migrate elites every
K generations. Sampling from the archive is a blend of:

- **exploitation** (sample fitness-proportionally from current island top-K),
- **exploration** (sample from underexplored cells, including empty ones),
- **uniform** (random elite — break stagnation).

Mix ratios (default `0.7 / 0.2 / 0.1`) and per-island migration are the
only "evolutionary" hyperparameters — there is no crossover, no mutation
operator in the GA sense. The mutation **is the LLM call**.

See: [MAP_ELITES_LOOP.md](MAP_ELITES_LOOP.md)

## 3. Optimization-aware prompting

Before every LLM call, a **target profile** is selected — a 4D coordinate
the loop wants the next candidate to land on. The selection is one of:

- `mutate`     — small delta from parent's profile (most common),
- `diversify`  — sample from underexplored cells,
- `esimd_upgrade` — push toward explicit-SIMD / Tensor-Core if the parent
                   already has memory+compute+parallelism but no ESIMD.

The selected profile is rendered as **concrete, actionable instructions**
(e.g. "use `intel_sub_group_block_read_us` for B loads, add `restrict`")
and appended to the prompt. The instructions are keyed by
`(dimension, backend, level)`.

This is the "WHAT TO DO, not WHY" half of the prompt — it tells the
model which optimization to apply, while the descriptor classifier later
confirms whether the model actually did it.

See: [PROMPT_PIPELINE.md](PROMPT_PIPELINE.md)

## 4. QD-gradient transition tracking (optional)

For every parent → child transition the loop records:

```
(parent_coords, child_coords, parent_fitness, child_fitness, outcome)
```

`outcome ∈ {improvement, neutral, regression, cell_discovery, elite_replacement}`.

From the transition history a coarse "gradient" is estimated: which delta
in descriptor space, applied from a given cell, *historically* yielded
improvements. This gradient feeds back into:

- parent sampling (prefer parents whose neighborhood has positive gradient),
- prompt mutation hints ("based on history, try increasing parallelism_opt"),
- exploration (avoid directions with consistent regression).

Optional. The loop works without it; with it, it reduces wasted trials
once the archive has any history.

See: [QD_GRADIENT.md](QD_GRADIENT.md)

## 5. Tiered evaluator + combined score

A candidate passes through up to four stages:

| Tier | What                                                    | Cost     |
|------|---------------------------------------------------------|----------|
| T0   | extract code, syntax pre-check (compile in process)     | seconds  |
| T1   | full build (out-of-process)                             | seconds  |
| T2   | correctness pytest (reference vs custom outputs)        | ~10s     |
| T3   | performance pytest (timed runs, min over N)             | ~10–20s  |
| T4*  | profiler (vendor profiler) — only on promising         | expensive|

The fitness score combines all of them:

```
combined_score = perf_score                                    # 0..5 ladder
               + 3 * 1[correct AND speedup > 0] * speedup
```

`perf_score` is a coarse 0..5 ladder (0 = syntax error, 5 = correct), so
the score is monotone in "how far did this candidate get." The
`+3 * speedup` term makes correct-and-faster candidates dominate
correct-but-not-faster ones.

See: [EVALUATOR.md](EVALUATOR.md)

## 6. Compile-error + log feedback loop

Whatever the evaluator produces — compile error text, pytest traceback,
profiler summary — is summarized and **injected into the next trial's
prompt** as:

- `[COMPILE ERROR FEEDBACK FROM PREVIOUS TRIAL]` block (top-3 most recent),
- the parent's stdout/eval log, optionally rewritten by a "feedback LLM",
- structured profiler hints from a vendor profiler.

This is the closed-loop part. Without it, the LLM rediscovers the same
build errors every iteration; with it, the loop converges roughly 2–3×
faster on real tasks.

See: [PROMPT_PIPELINE.md](PROMPT_PIPELINE.md)

# When to use this skill

Pick this skill when you want to:

1. **Understand the loop** — read SKILL.md (here) plus the six files it
   links. Together they describe the algorithm completely.

2. **Re-implement the loop somewhere else** — a smaller agent, a
   different orchestration framework, a custom Python harness. Read the
   six files and follow [PSEUDOCODE.py](PSEUDOCODE.py) as a template.
   The pseudocode includes representative starter pattern dictionaries
   and instruction text so it is runnable end-to-end once you wire in
   an LLM client and a build/test harness.

3. **Port to a new backend** — adding TPU / ROCm / Metal / etc.: only
   two parts of the design are backend-specific —
   [BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md) (the regex
   patterns) and [PROMPT_PIPELINE.md](PROMPT_PIPELINE.md) (the
   per-`(dim, level)` instruction text). The rest is backend-agnostic.

4. **Reason about a failure mode** — if the loop is generating slow
   kernels, the bug is almost always in one of: descriptor
   mis-classification (item 1), bad target-profile selection (item 3),
   or score not rewarding what you want (item 5).

# When NOT to use this skill

- You want a single, one-shot kernel from an LLM (no archive, no loop) —
  this skill is overkill. Just call the LLM with the reference + a good
  prompt.
- You only have one sample per generation budget (e.g. one LLM call
  total) — there is no archive to update; the algorithm degenerates.
- The reference kernel cannot be evaluated reliably (no build, no test) —
  the score channel collapses and elite selection becomes random.

# Pseudocode summary

The whole algorithm in one page:

```python
archive = MapElitesArchive(grid_shape=(4,4,4,4), num_islands=N)
archive.add(initial_program(reference_code))
gradient_tracker = TransitionTracker()           # optional

for trial in range(max_iters):
    branches = []
    for _ in range(branches_per_iteration):
        # 1. SAMPLE PARENT
        if archive.is_empty():
            parent = initial_program(reference_code)
        else:
            parent = archive.sample(
                exploit=0.7, explore=0.2, random=0.1,
                gradient_weight=0.3,             # reweight by gradient_tracker
            )

        # 2. SELECT TARGET PROFILE (in optimization space)
        target = pick_target_profile(
            parent_profile=classify(parent.code),
            strategy=config.exploration_strategy,   # mutate / diversify / esimd_upgrade
            underexplored=archive.underexplored_cells(),
        )

        # 3. BUILD PROMPT
        prompt = render_main_template(
            reference=reference_code,
            parent=parent,
            inspirations=archive.top_k_in_island(parent.island, k=3),
            rag_examples=rag.lookup(reference_code, target),
            hardware_specs=hardware,
        )
        prompt += optimization_instructions_for(target, backend)
        prompt += previous_compile_errors(top_n=3)
        prompt += gradient_tracker.mutation_hints(parent.profile)

        # 4. LLM
        candidate_code = extract_code(llm.complete(prompt))

        # 5. EVALUATE (tiered)
        result = evaluate(candidate_code)         # syntax → build → correct → perf [→ profile]

        # 6. SCORE + UPDATE
        score = result.perf_score + 3 * (result.correct and result.speedup > 0) * result.speedup
        candidate = Program(code=candidate_code, profile=classify(candidate_code), score=score)
        became_elite = archive.add(candidate)
        gradient_tracker.record(parent, candidate, became_elite)
        branches.append(candidate)

    # 7. EARLY-STOP / MIGRATE
    if early_stopping_satisfied(archive, patience=2): break
    if trial % migration_interval == 0: archive.migrate_islands()

return archive.best()
```

That's the entire algorithm. Each step is documented in one of the six
files in this folder.

# How to read this skill

Recommended order:

1. **First**: this `SKILL.md` (you're here).
2. [BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md) — what the cells are.
3. [MAP_ELITES_LOOP.md](MAP_ELITES_LOOP.md) — how the archive moves.
4. [PROMPT_PIPELINE.md](PROMPT_PIPELINE.md) — what a prompt looks like.
5. [EVALUATOR.md](EVALUATOR.md) — how scoring works.
6. [QD_GRADIENT.md](QD_GRADIENT.md) — the optional gradient layer.
7. [PSEUDOCODE.py](PSEUDOCODE.py) — a single-file runnable sketch with
   representative starter pattern dictionaries and instruction snippets.
   Wire in an LLM client, a build, and a test harness; the rest is
   complete.

Each supporting doc cross-links back here. None of them refer outside
this folder.

# Empirical anchor (Intel B580 / matmul DPAS, in-house run)

A representative deep run of this algorithm produced these numbers,
which serve as a sanity check that the loop, configured as documented,
actually converges:

| Setting                          | Value                                          |
|----------------------------------|------------------------------------------------|
| Hardware                         | Intel B580 (Battlemage G21, 20 Xe2 cores)      |
| Problem                          | C[2048, 2048] = A[2048, 2560] × B[2560, 2048], FP16 |
| Reference (naive scalar)         | 34.0 ms                                        |
| LLM                              | Claude Opus 4.6 (one-shot per call, no tools)  |
| Iterations                       | 7 trials × 4 branches = 28 candidates          |
| Total wall time                  | 1115.7 s (18.6 min)                            |
| Compile success rate             | 96.4 % (27 / 28)                               |
| Correctness pass rate            | 89.3 % (25 / 28)                               |
| Best runtime                     | **1.22 ms**                                    |
| Best speedup vs reference        | **27.9 ×**                                     |
| XMX peak utilization             | 18.3 % (of 96 TFLOPS theoretical)              |
| Estimated total tokens           | ~348K input+output (~12.4K avg per branch)     |
| Critical breakthrough            | trial 4 — model discovered `intel_sub_group_block_read_us` for B loads + `restrict`, after `mutate` from the prior best |

Not a benchmark — an existence proof. Read it as: "the algorithm
described in the rest of this skill, configured at default settings, can
take a 34 ms naive matmul to 1.22 ms in 7 LLM-driven trials." The loop
will not always achieve this; randomness in the LLM and the search
matters. But the breakthrough at trial 4 is a textbook example of the
optimization-aware prompt landing the model on the right cell after two
mutate steps from the prior best, exactly as item 3 above describes.

# Re-implementation effort

Roughly:

| Scope                                                                      | LoC   |
|----------------------------------------------------------------------------|-------|
| Minimum viable loop (single island, no QD gradient, T0–T3 only)            | ~500  |
| Production-quality, all six layers, multi-backend                          | 1500–2500 |

Most of the size in a production port goes into (a) backend-specific
regex pattern dictionaries (each backend needs its own), (b) backend-
specific instruction text per `(dim, level)`, and (c) the per-task build
and test harness. The control loop itself is small.
