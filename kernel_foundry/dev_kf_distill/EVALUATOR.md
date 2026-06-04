# 4. The Evaluator and the Score

> Back to [SKILL.md](SKILL.md).

The evaluator is what turns a candidate string into a number that the
archive can compare. The evaluator described here is structured as a
**5-tier ladder** plus a **combined score** that maps every possible
outcome to a monotonically increasing real value. This is more
important than it sounds: a non-monotone or noisy score destroys
MAP-Elites' elite selection.

## 4.1 The five tiers

A candidate progresses through tiers until it fails. Each tier has a
characteristic time cost and produces a `perf_score` value if it's the
last tier reached.

| Tier | Stage              | What                                            | perf_score on fail | perf_score on pass |
|------|--------------------|-------------------------------------------------|--------------------|--------------------|
| T0   | Code extraction    | LLM output → cleaned code block                 | 0 (extraction failed) | continue          |
| T0.5 | Syntax pre-check   | In-process compile (e.g. PyOpenCL Program build) | 1 (syntax/compile error) | continue       |
| T1   | Full build         | Out-of-process build (icpx / nvcc / OCL build)  | 1                  | continue          |
| T2   | Correctness pytest | Reference vs custom outputs, randomized inputs  | 2 (runtime error) / 3 (shape) / 4 (value) | continue |
| T3   | Performance pytest | Timed runs, min/median over N iterations        | 5 (correct), record `runtime` | continue |
| T4   | Profiler (opt)     | unitrace / NCU metrics                          | (no score change, attaches structured data) | — |

`perf_score` is a 0..5 ladder. At every tier failure the candidate gets
the highest score it earned. The ladder is *deliberately coarse* — it
ranks "how close to working did this candidate get" with five buckets:

- 0 — no code extracted
- 1 — code extracted but didn't compile
- 2 — compiled but crashed at runtime
- 3 — ran but produced wrong shape
- 4 — produced right shape but wrong values
- 5 — fully correct

This coarse ladder is what lets the archive accept incorrect candidates
into cells (so descriptor coverage is not blocked on getting working
kernels) while still preferring correct candidates within each cell.

## 4.2 The combined score

The score that the archive compares is computed as:

```python
combined_score = perf_score \
               + 3 * int(correct AND speedup > 0) * speedup
```

where `speedup = baseline_time / candidate_time` (a number > 1 means the
candidate is faster than the reference).

Walking through the cases:

- `compile error`              → `combined_score = 1`
- `runtime error`              → `combined_score = 2`
- `correct, but slower`        → `combined_score = 5` (since `int(speedup>0) * speedup` is multiplied into the term but `speedup<=1` produces `5 + 3*1 = 8` — wait, see below)
- `correct, 2× faster`         → `combined_score = 5 + 3*2 = 11`
- `correct, 27.9× faster (anchor-run best)` → `combined_score ≈ 5 + 3*27.9 = 88.7`

The reference implementation is:

```python
combined_score = (
    eval_result.perf_score
    + 3 * int(eval_result.correctness and eval_result.runtime_improvement > 0)
        * eval_result.runtime_improvement
)
```

`runtime_improvement = speedup` is a real number. If `speedup ≤ 0` the
boolean test gates it out — the candidate gets only `perf_score`. The
multiplier `3` is the convention used in [PSEUDOCODE.py](PSEUDOCODE.py)
and is the value that produced the empirical-anchor numbers in
[SKILL.md](SKILL.md); raise it if you want speed to dominate
correctness more aggressively, lower it if the search overshoots into
unstable territory.

This formula has three properties that matter:

1. **Monotone in correctness:** any incorrect candidate (perf_score ≤ 4)
   loses to any correct one (perf_score = 5 + ...).
2. **Monotone in speed within correct candidates:** faster always wins.
3. **Continuous:** scores form a real-number axis, not buckets, so MAP-
   Elites' elite-comparison works without edge cases.

## 4.3 Selecting the trial best

A trial generates `branches_per_iteration` candidates in parallel. After
all are evaluated, a `select_best_solution` helper picks one:

```python
def select_best_solution(eval_results):
    runtimes = [r.runtime if r.runtime != -1 else NaN for r in eval_results]
    if all_nan(runtimes):
        return argmax(perf_score)        # nobody compiled+correct → take "closest to working"
    return argnanmin(runtimes)           # at least one timed → take fastest
```

This is *trial-best*, not *archive-best*. Archive-best is determined by
the scoring above. Trial-best matters for logging and for deciding
parent in non-evolve mode.

## 4.4 Speedup measurement

The evaluator measures runtime as a min (or median) over N timed runs
of a single shape. Reasonable defaults:

- Warm-up: 5 runs discarded.
- Measured: 30 runs, take both min and median.
- Speedup = `reference_runtime / candidate_runtime`, where the reference
  is the kernel the user provided as the unoptimized baseline (in the
  task convention used here, the baseline lives between
  `[REFERENCE_START]` and `[REFERENCE_END]` tags).
- Multiple shapes: geometric mean of per-shape speedups.

When porting, pick one unit (ms or µs) for runtime and stick to it
across all stored fields. Mixed units in a single field are a real bug
class.

## 4.5 Anti-reward-hacking (largest open gap)

The evaluator's main weakness is that it inherits whatever pytest test
the task author wrote. The minimum viable design described here has
**no** built-in cross-task safeguards: all shapes that show up in the
test come from the task itself (they aren't randomized across trials);
there's no output-buffer poisoning; there's no "did real work happen?"
profiler check.

This is the largest open gap in the algorithm. For a re-implementation,
add at minimum:

1. **Randomized seeds per trial** — same shape, different RNG, every
   trial.
2. **Hidden held-out shapes** — the LLM never sees them in the prompt
   but the evaluator runs them. Catches the model that overfits to a
   single hard-coded `M=N=K`.
3. **Profiler sanity check** — confirm the kernel actually executed
   instructions for the expected duration, not a 1-cycle shortcut. A
   kernel that "completes in 5 ns and produces zeros" should fail
   correctness; if it passes, the test is broken.

These three additions cost very little but change the loop's behavior
fundamentally — without them, the score channel is gameable, and
MAP-Elites will eventually find the gameable corners.

## 4.6 Building the eval log

Every tier's stdout/stderr is captured into `EvalResult.eval_log`, which
becomes part of the next trial's prompt as `parent.feedback` (or is
rewritten by the optional feedback-LLM into a 5-line summary).

The eval_log is the second feedback channel (the first is the structured
fields like `compiled` / `correctness` / `runtime`). It's verbose but
valuable: real compile errors with line numbers go through this channel.

## 4.7 Caching the reference

In each trial, the evaluator builds and times *both* the reference (for
baseline) and the candidate. Doing this every trial is wasteful since
the reference doesn't change. Cache the first successful reference
result and reuse it for all subsequent candidates in the same run:

```python
if reference_test_result is None:
    reference_test_result = first_non_None(et.test_result_reference for et in eval_tasks)
for child in child_list:
    child.custom_task.test_result_reference = reference_test_result
```

This is a small thing but matters: without it, every trial pays the
reference-build + reference-time cost (~5–15s) for nothing.

## 4.8 Cross-references

- This skill: [MAP_ELITES_LOOP.md](MAP_ELITES_LOOP.md) (consumes the score for elite selection).
- This skill: [PROMPT_PIPELINE.md](PROMPT_PIPELINE.md) (consumes `eval_log` for feedback).
- This skill: [PSEUDOCODE.py](PSEUDOCODE.py) (`evaluate`, `combined_score` —
  reference implementation).
