# Advanced Convergence: fine-grained descriptors, staged activation, surrogate ranking, trust-region refinement

An **opt-in** extension to the baseline loop (`references/algorithm.md`) for the failure mode
observed in practice: *MAP-Elites descriptors too macroscopic → the search concentrates in a few
coarse cells and drifts away from the true optimum.* All of this is implemented in
[../scripts/evolve_loop.py](../scripts/evolve_loop.py) and demoed with
`python evolve_loop.py --demo-advanced`. **Everything here defaults OFF** — turn pieces on via `cfg`.

> First read the diagnosis: the two symptoms have *different* root causes, so they need *different*
> fixes. Don't reach for "more descriptor dimensions" as a cure-all — it can make convergence worse.

---

## 0. Diagnosis (why the two symptoms are not the same bug)

- **Concentration in coarse cells** = *illumination collapse*. With only 3 axes × 4 levels = 64
  cells and a static code-pattern classifier, many distinct kernels hash into the same cell; each
  cell keeps one elite, so real diversity is discarded. → fix with **finer, performance-correlated
  descriptors** (①), kept affordable by **bounded archives** (CVT) and **staged activation** (②).
- **Drift from the optimum** = *descriptor↔fitness mismatch* **and** *MAP-Elites is an illuminator,
  not an optimizer*. Static "levels" correlate only weakly with runtime; and even a perfect archive
  is not designed to converge to a local optimum. → fix with **measured descriptors** (①) **and,
  decisively, a local optimizer**: **trust-region refinement** (④). **Adding descriptor dimensions
  alone will not fix drift** — it can worsen it via sampling starvation.

Key dependency: ① (finer descriptors) is only affordable once ③ (surrogate) and ④ (local search)
make evaluations cheaper / better-spent. Stage the rollout accordingly (§6).

---

## ① Fine-grained, performance-correlated descriptors (`cfg['descriptor']`)

Two-descriptor split — the crux that makes this fit the existing loop:
- **Targeting** happens *before* generation → must be **a-priori** → keep the **static** code-pattern
  ladder (`Program.coords`). The gradient and inspiration bonus also use `coords`.
- **Placement** happens *after* evaluation → can be **a-posteriori** → use a richer descriptor
  (`Program.cell`). For the static descriptor `cell == coords`, so the baseline is unchanged.

`cfg['descriptor']`:
- `"static"` (default) — original ladder; `cell == coords`.
- `"measured"` — bins profiler features that *directly* track runtime
  (`occupancy × arithmetic_intensity × dram_bw_pct`, default 6 bins each). Populate
  `EvalResult.features` from your profiler feedback (the real pipeline already computes these — see
  `evaluation_and_feedback.md` §4). Falls back to static when a kernel didn't run. **This is the
  recommended fix for both "too macroscopic" and "drift".**
- `"cvt"` — CVT-MAP-Elites (Vassiliades et al., 2018): map a continuous featurizer onto `k` fixed
  Voronoi centroids (`cfg['cvt_cells']`, default 512). Archive size is bounded **independent of
  descriptor dimensionality**, so you can go high-dimensional without cell explosion.

Pick measured feature keys via `cfg['measured_feature_keys']`. Good candidates beyond the default:
register pressure, shared-mem bytes/CTA, achieved DRAM BW, warp-stall reason mix, tile aspect ratio.

---

## ② Staged dimension activation — coarse→fine curriculum (`cfg['stages']`)

Start coarse (broad coverage, locate promising regions), then activate finer descriptors **only
after** the search has something to refine. Configure an ascending list; each entry applies until
its `until_iter`, then the loop switches to the next entry's descriptor and **re-projects** the
archive:

```python
cfg["stages"] = [
    {"until_iter": 8, "descriptor": "static"},                 # phase 1: coarse ladder
    {"descriptor": "measured", "measured_bins": 6},            # phase 2: fine, measured
]
```

**Re-projection** (`Archive.reproject`) re-classifies every stored program under the new descriptor
and rebuilds the elite map. It's cheap: code + `EvalResult` are already stored, so **no
re-evaluation** — just re-binning. The best program per new cell is kept. This is what lets you
add dimensions mid-run without throwing away history.

---

## ③ Surrogate pre-ranking — spend the eval budget on the best candidates (`cfg['surrogate']`)

The bottleneck is `evaluate` (compile + test + benchmark + profile). So **over-generate** `k·B`
cheap candidates, score them with a surrogate, and **evaluate only the top `B`** (SAIL, Gaier et
al. 2018; DSA-ME, Bhatt et al. 2022). `cfg['surrogate']`:
- `True` → built-in `HeuristicSurrogate`: transparent, ML-free (compile-likelihood × climbed-levels
  − antipattern hits). Good default to start.
- any object with `.score(code: str) -> float` → plug in a learned ranker or an LLM-judge (reuse the
  existing feedback-LLM infra; small model, low temp).
- `cfg['surrogate_overgen']` = `k` (default 3 → generate 3·B, keep B).

The existing `qd_gradient` is itself a primitive surrogate (predicts which *direction* improves);
upgrade it to predict Δfitness / measures when you train a real model.

**Honesty / pitfalls (encoded in the code, respect them in production):**
- **Cold start** — a freshly-initialized learned surrogate is blind; bootstrap with full evaluation
  for the first N generations before trusting it.
- **Surrogate bias** — keep `acquisition = top-k`, never top-1 (the code breaks ties randomly for
  diversity), and periodically evaluate some surrogate-*rejected* candidates to de-bias the model.
- **The correctness gate is hard to surrogate** — prefer predicting `P(compile)·P(pass)·speedup`
  with a safety margin over regressing the raw continuous score.

---

## ④ Trust-region local refinement — the actual convergence engine

MAP-Elites illuminates **structure**; this converges to the **optimum within a structure**. Two
layers, both runnable in `evolve_loop.py` (④a `--demo-advanced`, ④b `--demo-edit-tr`):

### ④a Parameter trust region (recommended, highest ROI) — `cfg['refine_every'] / refine_at_end`
KernelFoundry already supports **templated kernels with tunable params** (`param_optim_prompt.j2`,
`is_templated`, `template_results`). So: once MAP-Elites surfaces a strong elite, **fix its
structure** and optimize its continuous params (`TILE_M/N`, `BLOCK`, `num_stages`, `num_warps`,
`unroll`) inside a **shrinking trust region** (TuRBO-style pattern search):

```python
cfg.update(refine_every=5, refine_at_end=True,
           param_space={"TILE_M": (16, 256), "TILE_N": (16, 256), "num_stages": (1, 6)},
           tr_max_evals=18, tr_init_radius=0.4, tr_shrink=0.5, tr_grow=1.6)
# and pass an evaluator that compiles the center's STRUCTURE with given params:
run_evolution(task, generate, evaluate, cfg, seed_code, eval_params=eval_params)
```

`trust_region_refine` polls ± along each param axis; on improvement the radius **grows**, otherwise
it **shrinks**, until budget or `tr_min_radius`. Structure/params are **decoupled**, so this is
low-risk and directly attacks "drift from the optimum". In `--demo-advanced` it drives the toy score
from 14.3 → ~22 by homing `tile`/`stages` onto a hidden optimum while structure stays fixed.

### ④b Edit trust region (LLM-operator layer, complementary) — `cfg['edit_refine_every'] / edit_refine_at_end`
A derivative-free trust region whose **variable is the LLM's diff edit** and whose **step size is
the edit magnitude**. It reuses the diff-mode machinery (`extract_diffs` / `apply_diff`, faithful to
`algorithm.utils.extract_code`): ask the model for a patch of `≤ radius` changed lines, apply it,
and — the defining constraint — **reject any candidate whose static descriptor leaves the center's
cell, *before* paying for evaluation**. On improvement the radius **grows**; on failure / rejection /
non-applying / over-budget patch it **shrinks** toward micro-edits.

```python
cfg.update(edit_refine_every=5, edit_refine_at_end=True,
           etr_max_evals=12, etr_init_radius=8, etr_min_radius=2, etr_grow=1.6, etr_shrink=0.5)
# supply an edit_fn(prompt)->answer that returns SEARCH/REPLACE blocks (your LLM in diff mode):
run_evolution(task, generate, evaluate, cfg, seed_code, edit_fn=edit_fn)
```

`build_edit_prompt` instructs the model to keep the optimization *profile* constant (same tiling /
sync / vectorization) and tune only local detail (indexing, unroll factor, fusion of adjacent ops,
register reuse, bounds); the hard `classify(patched) == center.coords` check then enforces it. This
refines **structure** locally where ④a refines **continuous params** — run ④b then ④a to first
settle the structure, then tune its knobs. Run it: `python evolve_loop.py --demo-edit-tr` — the demo
shows out-of-cell edits (which inject a higher-level technique) rejected pre-eval, non-applying
patches shrinking the radius, and the local `UNROLL` knob converging onto its hidden optimum while
the search provably stays in one cell (`archive_cells=1`).

**Why "edit magnitude = step size" works:** a small trust region forces the model toward surgical,
low-risk edits that are unlikely to break correctness or change the structure; only after such an
edit pays off do we widen the budget. This is the无导数 (derivative-free) analogue of shrinking a
TR radius after a rejected step — adapted to a mutation operator you can't differentiate.

**Honesty / limits:** an LLM may ignore the line budget (handled: over-budget patches are distrusted
and shrink the radius) or keep proposing the same out-of-cell edit (handled: each rejection shrinks
the region, so the round terminates); but a model that *only* knows how to change structure will make
no in-cell progress — ④b assumes the model can make meaningful local edits. The stay-in-cell check
is only as good as the descriptor: under a coarse static descriptor a "local" edit can still shift
real performance a lot, which is one more reason to pair ④b with measured descriptors (①).

### Principled whole-archive version — CMA-ME / CMA-MAE
For a single coherent algorithm, **CMA-ME** (Fontaine et al. 2020) / **CMA-MAE** (2023) run
self-adapting covariance emitters that search locally inside the archive. **CMA-MAE's archive
learning-rate `α` is exactly the "illumination ↔ convergence" knob** you want. Reference
implementation: **pyribs** (archive / emitter / scheduler abstractions, with CVT, CMA-ME, CMA-MAE
built in) — a good target if you outgrow the in-file pattern search.

---

## 5. Trade-offs you must not ignore
- **Diversity ↔ convergence is a real trade-off.** Bolting convergence onto MAP-Elites (CMA-MAE,
  trust region) sacrifices some coverage. Control it explicitly (CMA-MAE `α`, or how often/aggressively
  you refine); don't let it collapse into pure greedy.
- **The four pieces are interdependent.** Fine descriptors (①) without cheaper/better-spent
  evaluations (③/④) → sampling starvation and *slower* convergence. Roll out in the order below.
- **Reproducibility.** Surrogates and self-adapting covariance weaken the `random_seed=42`
  determinism; checkpoint surrogate/emitter state if you need reproducible runs.

---

## 6. Recommended rollout (by ROI / risk)
1. **Phase 0 (cheap, do first):** `descriptor:"measured"` (reuse existing profiler features) +
   bounded archive (`"cvt"` if dims grow) + parameter trust region on templated elites (④a).
   Fixes *too-macroscopic*, *mismatch*, and *drift* with essentially no new ML.
2. **Phase 1:** surrogate pre-ranking (③), starting with compile-filter + LLM-judge, to amplify the
   evaluation budget.
3. **Phase 2:** staged coarse→fine activation (②); add edit-trust-region structure refinement (④b)
   once you run the LLM in diff mode; graduate to CMA-MAE emitters for a principled, single-algorithm
   explore/exploit balance when you outgrow the in-file pattern/edit search.

---

## 7. Config quick-reference (all default OFF)
| `cfg` key | Default | Strategy | Meaning |
|---|---|---|---|
| `descriptor` | `"static"` | ① | `static` \| `measured` \| `cvt` placement descriptor |
| `measured_feature_keys` | occupancy, AI, dram_bw% | ① | which `EvalResult.features` to bin |
| `measured_bins` / `cvt_cells` | 6 / 512 | ① | bins per measured axis / Voronoi centroid count |
| `stages` | `[]` | ② | `[{until_iter, descriptor, ...}, ...]` coarse→fine schedule |
| `surrogate` | `False` | ③ | `True` (heuristic) or object with `.score(code)` |
| `surrogate_overgen` | 3 | ③ | over-generation factor `k` (generate `k·B`, keep `B`) |
| `refine_every` / `refine_at_end` | `None` / `False` | ④a | trust-region cadence (needs `eval_params`) |
| `param_space` | `None` | ④a | `{name: (lo, hi)}` continuous params to tune |
| `tr_max_evals` / `tr_init_radius` / `tr_shrink` / `tr_grow` / `tr_min_radius` | 16 / 0.4 / 0.5 / 1.6 / 0.02 | ④a | trust-region budget & dynamics |
| `edit_refine_every` / `edit_refine_at_end` | `None` / `False` | ④b | edit-trust-region cadence (needs `edit_fn`) |
| `etr_max_evals` / `etr_init_radius` / `etr_min_radius` / `etr_max_radius` | 12 / 8 / 2 / 40 | ④b | eval budget & radius bounds (radius = max changed lines) |
| `etr_grow` / `etr_shrink` | 1.6 / 0.5 | ④b | radius grow on improvement / shrink on failure-reject |

Demos (no LLM/GPU): `--demo` baseline · `--demo-advanced` (①②③④a) · `--demo-edit-tr` (④b).

## References
- Mouret & Clune 2015, *Illuminating search spaces* (MAP-Elites). Vassiliades et al. 2018 (CVT-MAP-Elites).
- Gaier et al. 2018 (SAIL surrogate-assisted QD). Bhatt et al. 2022 (Deep Surrogate Assisted MAP-Elites).
- Fontaine et al. 2020 (CMA-ME), 2023 (CMA-MAE). Eriksson et al. 2019 (TuRBO trust-region BO). Library: pyribs.
