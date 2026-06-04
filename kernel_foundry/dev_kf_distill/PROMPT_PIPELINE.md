# 3. The Prompt Pipeline

> Back to [SKILL.md](SKILL.md). Reads [BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md) and [MAP_ELITES_LOOP.md](MAP_ELITES_LOOP.md).

A prompt in this design is not a single template — it's a layered
build. Six layers are stacked in order, each emitting a fragment that's
concatenated into the final string. Understanding the layers is more
useful than seeing the final 8K-token render.

```
┌─ BASE TEMPLATE  (main_prompt.j2)
│   • role, language, reference code, hardware specs, requirements
├─ RAG / EXAMPLES
│   • vector-add example (first iter only)
│   • database-retrieved kernels matching reference + target profile
├─ EVOLUTION CONTEXT
│   • parent code + parent.feedback (or stdout)
│   • top program in island
│   • 1–3 inspirations
├─ TARGET-PROFILE INSTRUCTIONS  ← the "what to do" layer
│   • per-(dimension, level, backend) actionable text
│   • optional ESIMD / Tensor-Core taxonomy
├─ GRADIENT MUTATION HINTS  (optional)
│   • "historically, increasing parallelism_opt from this cell helps"
└─ ERROR FEEDBACK  ← the closed-loop layer
    • last 3 compile errors
    • profiler summary (memory-bound? compute-bound? stall reasons?)
```

## 3.1 Base template

The base template is a Jinja2-style file with three modes:

- `status="translate"` — first iteration, no parent, no errors. Pure
  reference→language translation prompt.
- `status="error"` — parent failed (compile error or incorrect output).
  The prompt asks the model to *fix* and emphasizes the error log.
- `status="correct"` — parent compiles and is correct. The prompt asks
  the model to *optimize* and emphasizes performance strategies.

Status is selected automatically based on the parent's eval result. The
LLM behaves very differently in error-fix mode vs optimization mode, so
this branching is more important than it looks.

The base template also bakes in:

- the GPU's hardware spec sheet (cores, SLM size, EU count, peak FLOPS),
- the language-specific requirements section (e.g. SYCL: "Use SYCL 2020
  API, name all kernels explicitly, do NOT use `.shuffle_down()` member
  functions" — these are pitfalls discovered the hard way),
- a strict output format ("Provide the complete code in a code block").

## 3.2 RAG / examples

The RAG layer is responsible for two things:

1. **First-iteration vector-add example.** When there's no parent yet
   (the LLM has never been called for this task), the prompt includes a
   vector-add example pair (PyTorch + target language). This dramatically
   improves first-iter compile rate by anchoring the output format.

2. **Database-retrieved kernels.** A second RAG database is keyed by
   (reference_keywords, target_optimization_profile). For a task with
   keywords `["gemm", "matrix"]` and target `(m=2, c=1, p=2, e=1)`, the
   RAG returns 1–3 prior successful kernels with similar properties.
   These are inserted as additional examples.

The RAG is optional. Without it, the loop still works but the first 1–2
iterations are wasted on "discovering the syntax" — particularly painful
for less-common backends like ESIMD.

## 3.3 Evolution context

This is the layer that turns a "single-shot LLM call" into "iterative
search":

- `parent.code` — the LLM sees what it (or a sibling) produced last.
- `parent.feedback` — either raw stdout/eval log, or a feedback-LLM-
  rewritten summary. The feedback-LLM is a small model whose only job is
  to compress 200-line logs into 5-line summaries. Optional but cheap.
- `top_program.code` — the best kernel in the parent's island. Acts as
  an "aspirational" version.
- `inspirations[]` — 1–3 additional elites from the same island, chosen
  for diversity. These let the model see *several* working approaches.

The template formats them as "Previous version (score=…), Best implementation
so far (score=…), Inspirations…". The model is explicitly told to
"identify why one is better than another."

## 3.4 Target-profile instructions

This is the half of the prompt that comes from a per-`(dim, backend, level)`
instruction table (a JSON dict, in practice — see
[PSEUDOCODE.py](PSEUDOCODE.py) `INSTRUCTIONS` for a representative
starter set). The selected `target_profile` (a 4-tuple, e.g.
`(memory=2, compute=1, parallelism=2, esimd=0)`) is expanded into
actionable text by walking each non-zero dimension and concatenating the
matching entry:

```
## Required Optimizations

Apply the following optimization techniques in your implementation:

**SLM Tiling**: Allocate `__local` memory for the inner-loop tiles of A and B.
Cooperatively load using `barrier(CLK_LOCAL_MEM_FENCE)`. Choose tile size
TM × TN such that TM*TN*sizeof(half) <= 64KB.

**Fused Operations**: Combine the matmul and the bias-add into one kernel.
Don't write the unbiased product to global memory.

**Sub-group Intrinsics**: Use `intel_sub_group_block_read_us` for cooperative
B loads. Use `sub_group_broadcast` for sharing K-strip across the sub-group.
```

Three things to notice about this layer:

1. **It's WHAT, not WHY.** The text doesn't explain *why* SLM tiling is
   faster (the model knows that). It says *what* to do, in concrete
   syntactic terms. This is the result of iterative tuning — earlier
   versions with rationale-heavy prompts wasted tokens.

2. **It's per-backend.** SYCL, CUDA, OCL, Triton each have their own
   instruction set keyed by `(dimension, level)`. Adding a new backend
   means adding a new key to the JSON.

3. **It's coupled to the descriptor.** The instructions for "memory=2,
   SYCL" produce text whose *implementation* will pattern-match as
   memory_opt level 2 in the classifier (see
   [BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md)). This is
   intentional: the classifier and the instruction text are two views
   of the same taxonomy. Drift between them is one of the main bug
   classes — if you change a pattern dictionary, audit the
   corresponding instruction text, and vice versa.

## 3.5 Strategy selection (which target to pick)

Before the instructions are rendered, a target profile must be selected.
The strategies are:

| Strategy        | When                                          | What it does                       |
|-----------------|-----------------------------------------------|------------------------------------|
| `mutate`        | Default. Parent exists and is correct.        | Pick a 1-step neighbor of the parent's profile. |
| `diversify`     | Parent doesn't exist, or stagnation detected. | Pick from underexplored cells.     |
| `esimd_upgrade` | Parent has high non-ESIMD opt but esimd=0.    | Force an esimd-target profile.     |

A reasonable strategy-selection rule is to scale the chance of
`esimd_upgrade` by how well-optimized the parent already is along the
non-ESIMD dimensions — a heavily-optimized parent (sum of
memory+compute+parallelism ≥ 6 of 9) gets ~60% chance to escalate to
ESIMD; a lightly-optimized one (sum ≥ 1) gets ~20%. See the
`pick_target_profile` reference implementation in
[PSEUDOCODE.py](PSEUDOCODE.py).

In practice the loop spends ~70% of its time on `mutate`, 20% on
`diversify` (especially early), 10% on `esimd_upgrade` (when applicable).

## 3.6 Gradient mutation hints

If QD-gradient tracking is enabled (see [QD_GRADIENT.md](QD_GRADIENT.md)),
hints like these are appended:

```
Based on history, transitions from your current cell that yielded
improvements:
- increase parallelism_opt (success rate: 67%, samples: 12)
- increase memory_opt (success rate: 50%, samples: 8)
```

These hints don't replace the target-profile instructions (3.4); they're
additional bias toward directions that historically worked.

## 3.7 Compile-error feedback

The simplest, most effective feedback layer. After every trial, if any
branch failed to compile, its error string (truncated to 500 chars) is
saved. On the next trial's prompt:

```
[COMPILE ERROR FEEDBACK FROM PREVIOUS TRIAL]
The following compile errors occurred in the previous iteration. Avoid
these mistakes in your implementation:
Branch 0: error: identifier "intel_sub_group_2d_block_read_8u" is undefined
Branch 1: error: 'sycl::shuffle_down' is not a member of 'sycl::sub_group'
[END COMPILE ERROR FEEDBACK]
```

Two design choices:

- Top-3 only. Beyond 3, returns diminish and tokens compound.
- Truncated to 500 chars per error — keeps the most actionable part
  (the line with `error:` and the offending identifier).

This layer alone accounts for a large fraction of why the loop converges
at all on unfamiliar backends.

## 3.8 Profiler feedback (optional)

For candidates that pass T3 (correct + timed), an optional T4 profiling
run produces structured signals:

```
The kernel is memory-bound:
- DRAM bandwidth utilization: 87% (of 512 GB/s peak)
- L3 hit rate: 41% — consider larger SLM tiles
- XVE thread occupancy: 23% — consider smaller work-group size
```

This text is added to the *next* iteration's `parent.feedback`. The
model is told it ran on hardware X and saw bottleneck Y, and asked to
change strategy accordingly.

A reasonable port keeps two backend-specific implementations — one per
vendor profiler (e.g. unitrace for Intel, NCU for NVIDIA). Both should
produce the same text shape; only the metric extraction differs.

## 3.9 What a final prompt looks like

For a typical mutate-mode iteration on a matmul-class task, the final
prompt is roughly:

- 200 lines of base template (role + requirements + hardware),
- 100 lines of RAG examples,
- 200 lines of parent + top + inspirations (code blocks),
- 30 lines of target-profile instructions,
- 5 lines of gradient hints,
- 10 lines of compile-error feedback.

Total: **~10K input tokens + ~2.4K output tokens** per inference call.
For 4 branches × 7 trials = 28 inferences, the total is ~350K tokens —
in the empirical-anchor run referenced in [SKILL.md](SKILL.md), this
budget reached a 27.9× speedup over the naive reference.

## 3.10 Cross-references

- This skill: [BEHAVIOR_DESCRIPTOR.md](BEHAVIOR_DESCRIPTOR.md) (defines the dimensions referenced here).
- This skill: [QD_GRADIENT.md](QD_GRADIENT.md) (sources the mutation hints).
- This skill: [PSEUDOCODE.py](PSEUDOCODE.py) (`build_prompt`, `INSTRUCTIONS` table — concrete reference).
