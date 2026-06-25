---
name: dev_gpu_acc_issue
description: Diagnose and fix GPU inference accuracy issues — NaN/Inf logits, empty or garbage output, CPU-vs-GPU mismatch — that stem from f16 dynamic-range overflow under the GPU plugin's default inference_precision=f16. Use when a model runs fine on CPU (f32) but produces NaN, all-zeros, empty, or wrong output on GPU.
---

When a model is **correct on CPU but NaN / empty / garbage on GPU**, the cause is almost always
**f16 dynamic-range overflow** (GPU default `inference_precision=f16`, max representable ≈ 65504;
CPU runs f32 so it never overflows and masks the bug). This skill distills a battle-tested
methodology for finding the overflow site and fixing it **without abandoning f16 as the global
precision** (only a thin sub-stream goes f32).

Worked end-to-end case study (read it for a concrete instance of every step below):
`../dev_moe_update/diffusion_gemma_debug.md` — DiffusionGemma 26B produced empty GPU output;
root cause = f16 residual-backbone overflow; fix = f32 highway + skip RMSFusion for the marked norms.

Remote build/test setup (Windows GPU box): `../dev_remote_debug_windows/`.

## Top-priority policies
- Don't create commits or push. Modify locally, remote-copy to the build machine, build & test there.
- Keep f16 as the **global** precision. Only mark the minimal overflowing sub-stream f32. A global
  `inference_precision=f32` / `ACCURACY` usually won't even compile (fused ops are often f16-only)
  and isn't the goal.
- Match existing code style.

## 0. Triage the symptom → is it f16 overflow?
Read the sample's per-step diagnostics first (no instrumentation needed). The signature of f16
overflow through a normalization is very specific:
- **`nan = <all elements>`, `inf = 0`, `max_logit = -inf`, argmax all 0** → this is the RMSNorm/
  LayerNorm `0×inf` pattern: an activation crosses 65504 → `inf` → `rsqrt(mean(inf²)) = 0` →
  `0 × inf = NaN`, then a matmul sprays NaN everywhere. **Zero inf + all NaN is the tell** (the inf
  was consumed inside the norm; only NaN escapes).
- CPU f32 correct + GPU f16 wrong, and dummy/small-magnitude weights are fine on GPU but real
  weights aren't → confirms **weight-magnitude-driven f16 overflow**, not a logic/kernel bug.
- Models trained in **bf16** (exponent range = f32) are especially prone: they carry values f16
  can't hold (e.g. big-gamma RMSNorm outputs, residual streams that grow with depth).

If the signature matches, proceed. If you see `inf != 0` surviving to logits, or values are sane but
wrong, it's a different class (logic/layout/quant) — this skill won't apply cleanly.

## 1. ⚠️ THE CARDINAL RULE: never measure with f32-promoting probes
This is the trap that cost the case study days. Any diagnostic that inserts `.to(f32)` +
`disable_fp16_compression` into the graph (per-layer max-abs taps, "keep this output f32 to print
it", etc.) **changes the computation you are trying to measure**: the GPU's `AlignMixedFP32FP16Types`
/ `ConvertPrecision` then carries the tapped edges in f32, **raising the overflow ceiling** and
*partially masking* the bug. You will see a tame, self-contradictory picture (e.g. "inputs ≤ 43 yet
output = inf at only 7/256 positions") and chase phantom "structural miscompile / memory-aliasing"
hypotheses. **All of those are tap artifacts.**

Use **tapless instruments only**:
- **Plugin-side buffer-scan probe** (reads device buffers, adds *no* graph ops). Env-gate it in the
  relevant primitive's `get_arguments` (e.g. `eltwise.cpp`): for outputs of interest (filter by a
  known last-dim), `mem_lock` each INPUT buffer and host-scan for `maxabs / nan / inf`, print
  `ptr / layout.to_short_string() / dtype`. Inputs are valid there (producers already ran); output
  isn't written yet. This shows the **real f16 values and the real dtype** each op receives.
- **The sample's own logit/denoise NaN-count** (tap-independent: derived from real output logits).
- A **pass-side print** (e.g. `DG_PASS_DEBUG`) inside a transformation callback to dump which nodes
  it visits, their out-type, consumers, and rt_info marks.

Keep these env-gated and **remove them before shipping**.

## 2. Localize the first overflow site (tapless)
With the buffer-scan probe on and **no f32 taps**, trace the magnitude of the suspect stream layer by
layer. You're looking for the first buffer whose `maxabs` crosses ~65504 → `inf` appears → next op's
output is all-NaN. Typical findings:
- The overflow is usually on a **residual backbone / accumulating stream**, not inside the norm —
  the norm's *input* is already too big. (embed × √hidden start, + attn/MLP/MoE each layer → grows
  with depth, crosses 65504 at some mid layer.)
- **Encoder often fine, decoder overflows** (or vice-versa): different input magnitude / statefulness.
- Beware **truncated runs** (`--num-layers N`): they under-accumulate the residual stream and
  under-represent the bug. Reproduce at full depth before trusting a localization.

Rule out red herrings with **rebuild-free** toggles before touching precision: memory-reuse / pool
(`OV_GPU_DISABLE_MEMORY_REUSE`, `OV_GPU_ENABLE_MEMORY_POOL=0`), fusion, fake-alignment, async. If the
NaN is byte-identical with these flipped, they're not the cause. Quantization is rarely the cause —
prove it with a same-depth same-input A/B (e.g. force the eager/f32 expert path) showing identical
overflow.

## 3. Fix: carry the overflowing sub-stream in f32 (keep everything else f16)
Mark only the minimal stream precision-sensitive via `ov::disable_fp16_compression` (genai:
`ops::precision::disable_fp16_compression(tensor)`). For a residual-overflow bug that means the
residual adds + the norm outputs that feed them. The compute-heavy islands (MoE experts, SDPA, big
matmuls) **stay f16** — don't widen the marking into them or you'll hit f16-only-fused-op layout
errors and lose the perf.

### Two traps that make "I marked it f32" silently not work
The case study burned a full cycle on each — check both with the tapless probe:

1. **Fusion drops the mark.** `DisableFP16Compression::is_copyable()` returns **false**, so when a
   later pass fuses your marked subgraph (e.g. `RMSFusion` folding a hand-rolled RMSNorm into
   `ov::op::internal::RMS` via `copy_runtime_info`), the mark is **discarded** and the output reverts
   to f16. Verify with a pass-side print that your node still carries the mark *after* fusion.

2. **Even a surviving mark can be ignored by the fused op's kernel.** Marking the *fused* op f32
   does **not** guarantee an f32 output buffer — some kernels (e.g. `rms_gpu_bfyx_opt`) downcast
   their output to f16 regardless of the op's declared output type. Re-marking the fused op is the
   wrong lever.

**The reliable lever:** keep the marked subgraph **unfused** so it stays a chain of decomposed f32
primitives (the same way un-fused eltwise Adds honor their f32 mark). Do this by extending the
relevant fusion pass's skip-callback in `transformations_pipeline.cpp`:
```cpp
pass_config->set_callback<ov::pass::RMSFusion>([...](const_node_ptr& root) -> bool {
    if (ov::fp16_compression_is_disabled(root))   // genai marked this norm's output f32
        return true;                              // → skip fusion, keep it a decomposed f32 primitive
    ... existing guard ...
});
```
(`root` is the fusion match-root = the marked node; it still carries the mark at match time, before
fusion would drop it. Needs `#include "transformations/rt_info/disable_fp16_compression.hpp"`.
No-op for models that don't carry the mark, so default behavior is unchanged.)

## 4. Verify the fix (tapless, then real e2e)
- Re-run with the **buffer-scan probe**: every target op must now show its f32 operand as `dtype=f32`
  and `inf=0 / nan=0`. Confirm the f16 islands you intentionally left (MoE/MLP) still read `out=f16`.
- Run **real end-to-end** (full depth, full token budget, **no diagnostic env vars**, default build):
  output must be non-empty/coherent and `nan=0 inf=0` across all steps. A single short or truncated
  run is not proof.

## 5. Minimize and de-risk the change set before shipping
- **Re-test necessity of each piece.** Marks/edits that were needed under an earlier, wider strategy
  may be **dead code** under the final minimal fix. Revert each suspected-unnecessary edit, rebuild,
  and confirm e2e is **byte-identical** (compile + per-step stats + output). The case study removed 6
  MoE-plugin edits this way after proving the MoE ops actually receive f16, not f32.
- **Remove all diagnostics** (probes, pass-prints, clamp/scale experiment knobs). Restore touched
  files that only carried instrumentation back to upstream baseline (`git checkout <file>`), so the
  final diff is exactly the fix.
- Sanity-check perf: the f32 sub-stream adds bounded memory-bandwidth/reorder overhead on small
  norm/residual tensors; the FLOP-heavy f16 islands are untouched. If perf matters, consider marking
  only the deep layers where overflow actually begins.

## 6. Anti-patterns / proven dead ends (don't repeat)
- Global `inference_precision=f32` / `execution_mode=ACCURACY` / `bf16`: usually won't compile
  (f16-only fused ops) and fights the deliberate f16 islands.
- `activations_scale_factor` (ASF): the `×S` restore is retained at residual-add boundaries, so it
  can't shrink scale-invariant gamma-driven activations, and it may push the *encoder* into overflow.
- Trusting truncated (`--num-layers`) runs, `clamp`/value-cap workarounds, or any conclusion drawn
  while f32 taps were active. Clamp "fixing" it is often just the inserted op blocking f16-compression
  on tap-promoted inputs — another artifact.

Keep explanations conversational. Always confirm a fix with the tapless probe **and** a real
no-env-var e2e run before declaring victory.
