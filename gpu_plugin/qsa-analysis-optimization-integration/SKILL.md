---
name: qsa-analysis-optimization-integration
description: "Use when analyzing, designing, implementing, optimizing, validating, or integrating sparse-attention / gated-residual style GPU kernels (e.g. QSA, Qwen Sparse Attention, Gated Residual / Hyper-Connection, Intel GPU CM/DPAS kernels, long-context top-k, paged KV attention) into a GPU compiler stack such as an OpenVINO-style GPU plugin. Covers the full workflow: problem analysis, design-document authoring, implementation, kernel optimization, correctness validation, performance measurement, and plugin integration."
---

# QSA / Sparse-Attention Kernel Analysis, Optimization & Integration

## Purpose and Scope

This skill packages a complete, self-contained workflow for working on a
sparse/indexed attention system (indexer-selects-blocks + sparse-core-attention,
as in Qwen-style QSA) and an associated multi-branch residual mechanism
(Gated Residual / Hyper-Connection), implemented as custom GPU kernels
(e.g. Intel "C-for-Metal"/CM, DPAS/systolic-array kernels) and integrated into
a GPU inference-engine plugin (OpenVINO-style: internal ops → primitives →
graph nodes → kernel-generator implementations).

It is written to be usable **without any external files**: all domain
invariants, templates, checklists and protocols needed are embedded below.
Apply it whenever the task is one of:

- Root-causing a performance or correctness problem in a sparse-attention /
  gated-residual kernel pipeline.
- Writing a design document before changing kernel or plugin behavior.
- Implementing a change against an approved design document.
- Optimizing a GPU kernel (memory access pattern, systolic/DPAS matmul,
  occupancy, register pressure).
- Validating correctness of a new kernel variant against a reference.
- Measuring and reporting kernel/pipeline performance.
- Integrating a validated standalone kernel into a production GPU plugin.

Do **not** use this workflow to silently turn exact top-k / exact attention
into an approximate algorithm, to change model math semantics without
explicit sign-off, to reuse an unrelated existing attention implementation
as a drop-in dependency, to change production default dispatch without
validation evidence, or to report only theoretical FLOPs/roofline numbers as
if they were measured speedups.

## Core Domain Invariants (must not be broken silently)

These are the algorithmic invariants of the QSA + Gated-Residual family of
mechanisms. Any analysis, design, or optimization must explicitly state
whether it preserves each one; changing one requires an explicit, called-out
decision in the design document, not an incidental side effect.

### Indexer / block-selection (the "which tokens matter" path)

1. The indexer uses its **own** Q/K projection, separate from the main
   attention Q/K/V projection. Do not conflate the two.
2. Raw indexer keys are **pooled first** (mean over a fixed compression ratio
   `r` of raw, not-yet-rotated keys, accumulated in higher precision than the
   storage dtype), **then** normalized, **then** rotary-embedded using the
   position of the block's **start token** (not the block index, not the
   block's end token). Never normalize/rotate per-token keys and then average
   the rotated results — that mixes incompatible rotation phases.
3. The indexer query is normalized, then rotary-embedded using **its own**
   absolute token position.
4. Scoring across indexer heads is: **per-head dot product → per-head
   nonlinearity (e.g. ReLU) → sum across heads → scale**. Do not reorder to
   "sum across heads then apply the nonlinearity" — the two are not
   equivalent for a nonlinearity like ReLU.
5. A compressed block only becomes visible to a query once all `r` of its
   raw positions are causally visible: `complete_blocks = floor((position+1)/r)`.
   An incomplete trailing block is **never** scored or top-k'd; instead its
   already-visible raw tokens are appended to the attention set directly
   (the "causal tail").
6. When `complete_blocks <= block_budget`, every complete block is selected
   by construction — this is an exact "dense" fast path: skip scoring and
   top-k entirely and just emit the ascending block-id sequence. Even when
   skipping selection for the *current* query, the indexer key/summary state
   for that token must still be updated because *future* queries depend on it.
7. Selected block ids (after expansion to raw token ids) must be emitted in
   **ascending order**, with a stable, deterministic tie-break rule, and
   unused output slots padded with an explicit sentinel (e.g. -1) plus an
   explicit valid-count. Any optimized top-k must reproduce this exactly.

### Sparse core attention (the "read the real values" path)

8. Sparse attention reads the **original, unpooled, un-normalized** main K/V
   — never the compressed summaries, which exist only to drive selection.
9. Within one query token, all attention heads share the **same** selected
   set; different query tokens generally do **not** share a selected set.
10. Standard causal masking, softmax numerics (a safe online-softmax with a
    correct masked/empty-tile fallback), GQA head-sharing, and any output
    gating must be unaffected by memory-layout or scheduling optimizations.

### Gated Residual / Hyper-Connection (multi-branch residual)

11. The residual stream has `C` parallel branches. "READ" (pre-sublayer
    branch mixing) and "WRITE" (post-sublayer per-branch update) are two
    **different** gates computing two different things — do not merge their
    logic or reuse one gate's activation for the other.
12. Normalization before mixing is **branch-wise**: reshape the last
    dimension into `[C, D]` and normalize each branch's `D` channels
    independently, never the flattened `C*D` vector as one normalization
    group.
13. Any `1/C` scaling terms in the read-gate low-rank projection, the
    branch-mean reduction, or the write-gate logits are part of the model
    math and must not be silently dropped; if folded into weights during an
    optimization, that folding must be explicit and tested.
14. The write gate is a **per-branch scalar** (not per-channel); the read
    gate is per-channel-per-branch; a third, unrelated gate may exist on the
    attention output itself — keep all three distinct.

### General correctness discipline

15. Any exact algorithm (exact top-k, exact attention) must stay exact after
    optimization: no candidate truncation, no dropped ties, no candidate-cap
    silently discarding valid entries — a bounded working set must fall back
    to a full, correct path when it does not fit, not silently degrade
    quality.
16. A kernel invariant test suite is under-constrained (and can hide a wrong
    implementation, e.g. a wrong rotary convention) whenever the selection
    budget is not meaningfully smaller than the number of eligible
    candidates. Always include at least one test configuration where
    `budget << eligible_candidates`.

## Step 1 — Problem Analysis

1. **State the target precisely**: which stage (indexer projection,
   compression/pooling, scoring, top-k, sparse core attention, gated-residual
   read/write), which regime (prefill vs. decode, sequence-length range,
   batch size), which device, and which of {correctness bug, performance
   regression, new capability, scalability limit} is being addressed.
2. **Read before touching code**: read the algorithm/model semantics
   description end-to-end (not just a summary), the current kernel source,
   the current host-side dispatch/integration code, and any existing tests
   for that stage. Prefer reading a large, coherent chunk of a file over many
   tiny fragments so control flow and invariants are understood in context.
3. **Classify every claim you are about to make** into one of:
   - *source-code fact* (directly observed in the code),
   - *mathematical derivation* (follows from the algorithm's equations),
   - *test-suite fact* (an existing/added automated test demonstrates it),
   - *measured evidence* (an actual same-input A/B run on target hardware),
   - *unverified hypothesis* (plausible but not yet checked).
   Never present an unverified hypothesis as if it were measured evidence.
4. **Establish a baseline before proposing a fix.** For performance work,
   this means isolating the suspected stage and measuring it standalone
   (a per-kernel or per-stage micro-benchmark), not just the end-to-end
   pipeline number. For correctness work, this means reproducing the bug
   with the smallest input that still triggers it.
5. **Find the true bottleneck via ablation, not instruction counting.**
   Removing a suspected expensive block of work and re-measuring is far more
   reliable than reading an instruction histogram: a kernel bound by memory
   message issue rate can have its ALU/branch instruction count dominate a
   static histogram while contributing near-zero to actual runtime.
6. **Check whether the problem is amplified by a structural mismatch**, e.g.:
   - a per-query-independent selection being processed in a batched tile
     that forces computing the *union* of many independent selections
     (amplifies work by the union size vs. any single selection);
   - a "partition"/"tile" granularity that leaves very little parallelism
     when the actual working set (e.g. decode with short history) is small;
   - an unbounded intermediate buffer whose size scales with sequence length
     squared or with total context length, which is fine at small scale and
     becomes a scalability wall at long context.
7. **Write down what is explicitly out of scope** for this analysis (e.g.
   "not evaluating end-to-end model quality", "not covering other hardware
   families", "not covering batch sizes above N") so later readers do not
   over-generalize the conclusions.

## Step 2 — Writing the Design Document

Before changing default behavior, write a design document with the following
sections. Keep every claim tagged per the classification in Step 1.

1. **Problem statement and scope** — the target stage/regime, why it
   matters, and explicit non-goals.
2. **Current behavior** — the existing algorithm, data layout, and dispatch
   as they are today, including any existing fallback paths and their
   trigger conditions.
3. **Invariants that must be preserved** — reference the relevant items from
   the domain-invariants list (or the project-specific equivalent) explicitly;
   do not just say "keep it correct".
4. **Proposed change** — the new algorithm/data layout/dispatch, described
   precisely enough to implement without further judgment calls: new buffer
   shapes and lifetimes, new kernel parameters/flags and their default
   values, new host-side control flow, and exactly which existing code paths
   remain as a fallback and under what condition they are chosen.
5. **Alternatives considered and rejected**, with the concrete reason each
   was rejected (a failed experiment is valuable evidence — record it rather
   than deleting it).
6. **Correctness validation plan** — the specific edge cases that must be
   covered (see Step 5's checklist as a starting point), the reference
   implementation/oracle to compare against, and the tolerance policy with a
   justification tied to the numeric precision actually used on-device.
7. **Performance validation plan** — the exact measurement protocol (see
   Step 6), the shapes/regimes to be measured, and what would count as
   success vs. a wash vs. a regression.
8. **Risk and rollback** — what default behavior changes (if any), how to
   disable the new path (an explicit opt-in flag is strongly preferred over
   silently replacing the old default), and what monitoring/tests catch a
   regression.
9. **Open questions / follow-ups** — anything intentionally deferred, called
   out so it is not silently forgotten (e.g. "this closes the standalone gap
   but the production integration of the same bound is a follow-up").

Treat the design document as a contract: implementation should not
introduce behavior not described in it without updating it first.

## Step 3 — Implementing Against the Design Document

1. **Trace the impact order** before editing: kernel source and its
   compile-time parameters → the host-side generator/dispatch code that
   builds kernel arguments and work-group sizes → the buffer/lifetime
   management code → any capability/fallback selection logic → tests.
   Read each of these fully for the stage being touched before editing any
   of them.
2. **Prefer additive, opt-in changes.** New kernel variants should be
   selected by an explicit flag/parameter with the old behavior remaining
   the default until validated; new kernel-parameter additions should be
   appended after existing parameters so the old calling convention keeps
   working when the new feature is compiled out.
3. **Prototype in the smallest possible harness first.** If a standalone,
   isolated per-kernel test harness exists (or can be created cheaply),
   validate the new kernel there — with a numeric reference and edge-case
   coverage — before touching the full plugin/integration code. This keeps
   iteration cheap and keeps the eventual integration diff small and
   reviewable.
4. **Never allocate, copy, rebuild maps, clear buffers, or synchronize
   inside a region that will be timed or that is on a performance-critical
   dispatch path** unless the design document explicitly calls for it; those
   operations belong in one-time setup.
5. **Keep old code paths intact and reachable** until the new path has full
   correctness and performance evidence; do not delete a fallback as part of
   the same change that introduces its replacement.
6. **Update the design document if implementation reveals it was wrong** —
   do not silently diverge from a written design without recording why.

## Step 4 — Kernel Optimization Workflow

Apply in this order; do not skip straight to micro-optimizations before the
structural bottleneck is identified (see Step 1.5-1.6).

1. **Identify what actually bounds the kernel**: memory message issue rate,
   DRAM bandwidth, register-file/spill pressure, occupancy/launch overhead,
   or genuine compute (matmul) throughput. Use ablation (temporarily remove
   a block of work and re-measure) as the primary diagnostic; use an
   assembly/spill dump as a secondary check, not instruction-histogram
   guessing.
2. **If bound by per-query independent selection processed in a batched
   tile**: restructure the tiling axis to group work that shares state
   (e.g. put multiple attention heads that share the same selection and the
   same K/V source into one tile, instead of grouping independent query
   tokens whose selections must be unioned). Sharing collapses "N-way union
   of independent selections" down to "the selection actually needed",
   which is normally the dominant amplification factor, far larger than any
   register/message tuning can recover.
2b. **If bound by decode-time granularity that is too coarse**: split the
    scan of history/candidates into smaller, more numerous units so more
    hardware threads/lanes participate, especially when the total candidate
    count at decode time is small (a single work item scanning the whole
    history is not "using a partition", it is serializing all of it).
3. **Widen memory messages, but respect hardware message-size limits**: pack
   more useful bytes per load/store instruction (2D block loads over
   multiple rows/columns, wide store types) — but any given hardware/driver
   combination usually caps a single message row width; verify empirically,
   because exceeding it can silently return/write wrong data with no
   compile or runtime error.
4. **Reuse loaded data across independent consumers when legally possible**
   (e.g. cache a merged cursor/head across a loop instead of re-reading
   unchanged data), but verify the gain: on a data-movement-bound kernel,
   reducing scalar ALU/branch work or hoisting address computation can be a
   *net loss* if it adds live registers and pushes the kernel into spill —
   always re-measure after such changes, do not assume they help.
5. **For a systolic/DPAS-style matmul path**: choose the tiling axis that is
   naturally shared (e.g. tokens/heads that read the same weight tile) as
   the "N" (broadcast/output-column) axis, and keep the accumulator small
   enough to fit the register file at the target head/tile size — split
   large per-thread accumulators across multiple worker threads/lanes
   (cooperative head-dimension or K-dimension partition with a barrier and
   an intermediate reduction) rather than trying to force one thread to own
   an accumulator too large for the register file.
6. **Prefetching and speculative optimizations are not free wins.** On a
   kernel already bound by memory message issue rate, adding a prefetch
   instruction stream competes for the same issue slots and can regress
   performance materially; always A/B it, do not assume prefetch helps.
7. **Every optimization must ship with an A/B or ablation result** (see
   Step 6) — a change that "compiles and looks reasonable" is not evidence
   it is faster.
8. **Watch for numerically-fragile shortcuts**: a finite negative "sentinel"
   used to represent "masked out" in an online-softmax-style reduction can
   silently produce full (wrong) weight for an entirely masked tile if the
   running maximum degenerates to exactly the sentinel — always explicitly
   zero out masked contributions after the exponential, do not rely on the
   arithmetic cancelling correctly.

## Step 5 — Correctness Validation Checklist

At minimum, cover:

- Empty and very short sequences; zero-length batches.
- Selection-budget edge cases: zero budget, budget of one, budget covering
  every eligible candidate (the "dense" fast path), and — critically — a
  configuration where budget is much smaller than eligible candidates (see
  invariant 16) so the selection logic is actually exercised.
- Exact ties in the scoring function, all-zero scores, and both signs of
  zero if the implementation uses a bit-level ordering key.
- Non-tile-aligned row/column counts, sequences that cross a physical
  page/block boundary, and (if paging is involved) a shuffled/non-contiguous
  physical page order.
- Incomplete trailing groups of 0..r-1 elements at every valid remainder.
- Mixed batches combining different regimes (e.g. one sequence still in
  "prefill" together with one already in "decode" in the same dispatch).
- Continuity across repeated calls that extend the same sequence
  incrementally, verifying persisted state (caches, history) is correctly
  reused, not just correctness of a single isolated call.
- Any bounded/pre-allocated working buffer: verify behavior exactly at the
  boundary of what fits, one unit below it, and one unit above it (forcing
  the multi-chunk/fallback path), and verify output correctness is
  identical whether the fallback path was taken or not.
- Sentinel/guard/padding regions: verify they are never read as if they
  were valid data, and that writes never spill past their declared bounds.

Comparison policy:

- Reduced-precision on-device paths need not be bit-exact against a
  higher-precision reference; compare final decisions (e.g. which items
  were selected) and the downstream numeric result computed from those
  *actual* on-device decisions, with a tolerance justified by the on-device
  storage precision, not the reference precision.
- When two implementations legitimately produce a different selection near
  a tie (due to rounding), do not widen the downstream numeric tolerance to
  hide it — instead report the tie margin and confirm it is within the
  expected rounding-induced perturbation, and still check the downstream
  numeric result against a reference computed from each implementation's
  *own* actual selection.
- An exact algorithm (exact top-k, exact attention over a given selection)
  must match a brute-force reference computed from its own actual inputs,
  independent of whether any nearby optimization changed selection.

## Step 6 — Performance Measurement Protocol

1. Only report numbers measured on the actual target hardware; never
   fabricate or extrapolate device numbers from a machine that cannot run
   the kernel (e.g. lacks the required GPU/compiler).
2. Run GPU benchmarks **serially** (not concurrently with other GPU work on
   the same device) to avoid cross-job interference.
3. Compare variants with the **same input data**, alternating execution
   order across repetitions (A/B/B/A, or reversed on a second pass) to
   average out any drift (frequency/power/cache-state drift).
4. Fix and disclose the measurement protocol per number: warmup count,
   sample count, whether the cache was flushed before each sample or left
   warm, and what is included in the timed region (device kernel events vs.
   host-observed elapsed time including dispatch/sync overhead).
5. Never allocate memory, copy data, rebuild metadata, or flush caches
   *inside* the timed loop; a cache flush (if used) is a full, untimed
   operation applied once before the whole timed unit, not between internal
   sub-stages.
6. For a multi-kernel-dispatch stage, sum every kernel's device event time,
   not just the last dispatched kernel's event — a common bug is to only
   capture whichever event a generic "last executed primitive" accessor
   happens to expose.
7. Report mean, median, and a high percentile (not only the minimum), plus
   the raw sample count and the exact shapes/config measured.
8. Every optimization claim needs either an A/B comparison or a clearly
   labeled ablation; state the evidence's scope explicitly (e.g. "measured
   at these shapes on this device; not extrapolated to other shapes/devices").
9. Do not conflate "compute-bound roofline efficiency is low" with "there is
   necessarily more speedup available" — a kernel dominated by irregular
   gather/scatter overhead may be nowhere near a naive FLOP roofline while
   still being close to that access pattern's realistic limit; state
   roofline numbers as a reference bound, not a promised target.

## Step 7 — Integrating a Validated Kernel into a GPU Inference Plugin

Only start integration once the standalone kernel has passed the Step 5
correctness checklist and has Step 6-style A/B evidence. Typical structure
for a plugin built as internal-op → primitive → graph-node → kernel-impl
(the layering used by OpenVINO-style GPU plugins; adapt names to the actual
target framework):

1. **Op / primitive / graph-node layer**: define (or extend) an internal
   operation type carrying the fixed compile-time-ish configuration
   (head counts, dimensions, compression ratio, budget, page size, causal
   flag, dtype/layout constraints) and validate the configuration's internal
   consistency at construction time (e.g. a budget must be an exact multiple
   of the compression ratio; a derived "block budget" must equal
   `budget / compression_ratio`).
2. **Kernel-generator / implementation layer**: add the new kernel source
   file(s) and a generator class that builds the compile-time JIT constants,
   the work-group/global dispatch sizes, and the kernel-argument order.
   Preserve the existing kernel's entry point and old default constants so
   old call sites keep working unchanged when the new feature is compiled
   with its flag left at the old default.
3. **Capability gating**: add an explicit predicate function that checks the
   static shape/dtype/hardware-generation constraints the new kernel
   requires (e.g. minimum divisibility constraints for a systolic/DPAS
   layout, required architecture generation). Select the new kernel only
   when the predicate holds; otherwise fall back to the existing
   scalar/reference implementation. Never let an unsupported configuration
   silently produce wrong output — either gate it out statically or fail
   loudly.
4. **Internal scratch/buffer sizing**: size any new internal buffer from the
   *static* layout information available at graph-compile time, not from
   runtime-only state (runtime dependency maps may be empty at that point in
   the framework's lifecycle). If a buffer's natural size scales unboundedly
   with total sequence length (e.g. an all-pairs score matrix), size it as a
   **bounded, reusable** scratch region instead (see the row/one-axis
   chunking pattern below) rather than allocating proportional to the worst
   case total context length.
5. **Bounded chunked processing pattern** (for any stage whose natural
   working set is quadratic or otherwise unbounded in sequence length):
   process one bounded slice of the "query" axis at a time — score that
   slice against the *full* history, finalize/select for that slice, then
   reuse the *same* fixed-size scratch buffer for the next slice. Do **not**
   chunk the "candidate/history" axis (that would require a separate
   partial-selection merge step, which is a materially different and more
   complex algorithm). Persist only the small, per-query outputs (selected
   ids, counts, metadata) across slices; never persist the large per-slice
   scratch. Choose a scratch budget as a fixed upper bound (independent of
   total sequence length), round down to a whole number of query rows, and
   fail loudly if even one row does not fit rather than silently shrinking
   candidate width. Keep the chunk-capable kernel variant behind a
   compile-time flag defaulting to the old, unchunked calling convention;
   when the flag is enabled, append the new chunk-boundary parameters after
   the existing parameter list so the ABI stays backward compatible.
6. **Dynamic/runtime metadata construction**: build any per-call metadata
   (position/page/sequence maps, chunk plans) in the runtime-parameter-update
   step that has access to actual instance data, mirroring the exact
   planning logic validated in the standalone harness, not re-deriving a
   subtly different version of it.
7. **Multi-stage / multi-dispatch primitives**: if a single logical
   operation spans multiple kernel dispatches, make sure the execution path
   records **every** stage's event/timing, not just the last one, and — if
   any stage's dispatch function is invoked more than once per top-level
   execute call (e.g. once per chunk in the bounded-chunking pattern above)
   — explicitly force that stage's "dispatch data" and "kernel arguments"
   to be recomputed/rebound before each invocation. Frameworks commonly
   assume a stage is dispatched at most once per execute call and silently
   reuse the first call's scalars/work sizes otherwise, producing
   plausible-looking but wrong output with no error.
8. **Build-system integration**: after adding brand-new kernel/header source
   files (not just editing existing ones), make sure the build's source
   discovery step is re-run (a cached file-glob step may not pick up new
   files just from an incremental rebuild) — otherwise the build reports
   success while the new kernel is silently absent from the compiled
   binary.
9. **Shared-compilation-unit hazards**: when multiple kernel stages of one
   primitive (or the whole primitive family) are compiled together into a
   single build/program, watch for helper-function name collisions between
   a new private copy of a helper and an existing one already compiled
   into the same unit — a standalone single-kernel compile can succeed while
   the combined production build fails with a redefinition error; rename the
   new private copy rather than editing the shared original.
10. **Cross-thread visibility inside one work-group**: if several
    threads/lanes of one dispatch cooperatively write a buffer and then
    other threads in the *same* dispatch need to read what peers wrote, a
    single generic memory fence is not always sufficient for full
    cross-thread visibility on every driver; the reliable pattern is to
    split "producer" and "consumer" into two separate dispatches ordered by
    the command queue/event dependency, not by an in-dispatch fence+barrier
    alone.
11. **Preserve default production behavior** unless the design document
    explicitly changed it: new capability-gated fast paths should be
    strictly at-least-as-correct as the fallback, and any change to a
    production default must be backed by the same correctness + performance
    evidence required for a new opt-in path.
12. **Test coverage for integration-specific concerns**: add tests that
    exercise multiple independent sequences in one batched call (not only
    the single-sequence case most unit tests default to), and add at least
    one test that forces the bounded/chunked path to actually chunk (e.g. by
    temporarily shrinking its scratch budget) rather than relying on it
    happening to chunk at realistic production sizes.

## Common Failure Patterns and How to Diagnose Them

- **A kernel compile failure with no useful diagnostic from the plugin
  layer**: dump the fully preprocessed/expanded kernel source the plugin
  actually built and recompile it directly with the underlying
  device-compiler CLI to get a real error message and line number.
- **A newly added kernel file behaves as if it were never compiled in**:
  the build's source-file discovery is stale; force a full reconfigure, not
  just an incremental rebuild.
- **A multi-stage primitive's measured time looks implausibly small**: the
  timing code is likely only capturing the last dispatched kernel's event;
  sum every stage's event explicitly.
- **Intermittent wrong results that differ between runs with the same
  input**: suspect a cross-thread global-memory write-then-read within one
  dispatch relying on a fence that does not guarantee visibility; split into
  producer/consumer dispatches.
- **A chunked/looped dispatch produces results that look structured but
  wrong** (e.g. correct for the first chunk, wrong afterward): check whether
  the stage's dispatch arguments/work sizes are being refreshed before every
  chunk's dispatch, or silently reused from the first call.
- **A wide/tiled store corrupts the next row**: the store's tile width does
  not evenly divide the row's actual stride; pad the runtime stride to a
  multiple of the store tile width everywhere that stride is computed.
- **A masked/empty attention tile yields NaN or full (wrong) weight**:
  the online-softmax path likely relies on arithmetic cancellation against
  a finite "negative infinity" sentinel; explicitly zero the contribution
  after the exponential for masked entries instead.
- **A selection/top-k test suite passes even though the algorithm is wrong**:
  the test's selection budget is not meaningfully smaller than the number of
  eligible candidates, so every candidate is trivially selected regardless
  of correctness; add a configuration where budget is much smaller than the
  candidate count.
- **A rotary-position-embedding implementation looks reasonable but is
  wrong**: confirm which pairing convention the target model actually uses
  (adjacent-interleaved pairs vs. "rotate-half"/first-half-second-half
  pairs) directly from the reference model definition — the two conventions
  cannot be reconciled by only permuting a cos/sin lookup table; the
  channel-pairing itself differs.
- **A performance change that "should" help measures as neutral or worse**:
  re-check whether it added live registers/spill, added memory messages that
  compete with the real bottleneck, or only mattered at a scale much larger
  than the shapes actually being optimized for; always re-measure rather
  than trusting the direction of the change.
- **Synchronizing to the wrong remote/target machine or environment**:
  after any deployment/sync step, independently verify (e.g. checking for a
  distinctive marker) from a separate connection before starting a
  potentially long rebuild or benchmark run on top of it.

## Deliverable / Report Format

When concluding a unit of work under this workflow, report:

1. Which files/stages were changed and the purpose of each change.
2. Which old ABI/behavior/default/fallback paths were explicitly preserved.
3. Correctness evidence: what was tested, on what hardware/config, and the
   result — including edge cases from the Step 5 checklist.
4. Performance evidence: the exact measurement protocol used, the
   baseline/variant compared, and the A/B or ablation result, with its
   scope (shapes/devices) explicitly stated.
5. What remains unverified or out of scope (other hardware, other shapes,
   production integration status, end-to-end model-quality impact).
6. Reproduction steps for anyone who needs to re-run the validation or
   benchmark.

Every conclusion must be labeled as verified-fact, derivation, or
unverified-hypothesis; never present an unverified optimization idea as an
already-realized performance gain.
