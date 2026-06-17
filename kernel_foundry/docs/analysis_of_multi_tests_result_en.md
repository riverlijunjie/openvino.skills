# KernelFoundry — Analysis of Repeated Identical-Command Experiments

**Hardware**: Intel Arc B580 (Battlemage G21, device_id `0xe20b`, XMX/DPAS)
**Task**: OCL matmul fp16, shape (M,K,N) = (2048, 2560, 2048), ≈21.5 GFLOP
**Reference baseline**: reference kernel ≈ **33.9 ms** (consistent and stable across all six runs)
**Date**: 2026-06-15
**Code version**: git commit `98f26eb`, oneAPI 2025.0, icpx 2025.0.4, NEO 25.48.36300.8, torch 2.9.0+xpu

---

## 1. Experiment Design

The **exact same command** (only the `job_name` / `task_name` suffix differs, to isolate the DB) was run 6 times in a row:

```
python scripts/run_custom_task.py custom_task=.../ocl_matmul_dpas \
  task_origin=local_gemm gpu_arch=bmg language=OCL \
  use_queue=false use_container=false store_generated_kernels_in_db=false \
  validate=true max_iters=6 branches_per_iteration=4 \
  stop_once_correct=false has_reference_build_step=false \
  eval_config.profile_custom_model=false eval_config.profile_original_model=false \
  test_timeout=180 build_timeout=120
```

> Note: the first run `b580_matmul_dpas` used `max_iters=12`; the later runs used `max_iters=6`. But in every run the divergence already happens at trial 0/1, so the number of iterations is not the main driver of the differences.

---

## 2. Six-Run Comparison

| Run | Best runtime | **Speedup** | Compute | Correct | First high-perf hit | Peak landed | Best candidate |
|---|---|---|---|---|---|---|---|
| `b580_matmul_dpas` | 33.9 ms | **1.00×** | ~0.6 TFLOPS | 45/48 | **never** | low | trial_5_v2 |
| `b580_matmul_dpas_v1` | 13.5 ms | **2.52×** | ~1.6 TFLOPS | 18/24 | trial 1 | mid | trial_5_v2 |
| `b580_matmul_dpas_v2` | 1.05 ms | **32.3×** | ~20.5 TFLOPS | 17/24 | trial 1 | high | trial_4_v2 |
| `b580_matmul_dpas_v3` | 1.01 ms | **33.6×** | ~21.3 TFLOPS | 16/23 | trial 0 | high | trial_3_v2 |
| `b580_matmul_dpas_v4` | 33.9 ms | **1.00×** | ~0.6 TFLOPS | 7/24 | **never** | low | trial_4_v1 |
| `b580_matmul_dpas_v5` | **0.623 ms** | **54.4×** | **~34.5 TFLOPS** | 21/24 | **trial 0** | **high (best)** | trial_1_v1 |

**The same command produced six different outcomes: 1.00× / 2.52× / 32.3× / 33.6× / 1.00× / 54.4×.**

**The fifth run (v4) is a key control**: it broke the appearance of "all later runs are high" by **falling back to the low-performance peak (1.0×)**. This decisively falsifies the "monotonic cross-run accumulation" leakage hypothesis — if accumulation were real, v4 could not have dropped back to 1.0×.

**The sixth run (v5) is another key control**, and it **falsifies the accumulation hypothesis twice over**:
1. v5 came right after v4 (which had fallen back to the low peak), yet **jumped straight to the global best of 54.4×** — the sequence goes from 1.0× (v4) to 54.4× (v5), again proving that each run is independent and there is no "relay improvement."
2. v5 **set a new high-peak record for the first time**: 0.623 ms ≈ 34.5 TFLOPS, **markedly faster than v2/v3's ~1.0 ms / ~21 TFLOPS**. This overturns the earlier inference that "~21 TFLOPS is the physical ceiling" — the high peak is itself a **range with width (~21–35 TFLOPS)**, not a single convergence point; the true ceiling is higher and is simply rarely reached by sampling.
3. v5 had **all 6 trials land in the high peak** (21 of 24 candidates correct, none regressing to any low peak other than 1.0×) — the only "clean sweep" run of the six, the most extreme contrast with v4's "all-failures opening."

The six-run sequence **{1.0×, 2.5×, 32×, 34×, 1.0×, 54×}** swings wildly and disorderly, exactly as expected from "each run independently samples at random and lands in one of the two peaks."

### v4's distinctive trajectory (further evidence of randomness)

- **All 12 candidates in trials 0/1/2 failed correctness** (all -1.0): three consecutive rounds of LLM sampling produced no usable kernel — the only such "cold opening" among the runs;
- the bad kernels in trial 2 triggered **each correctness test running for the full ~100 seconds** (normally ~11 s), indicating pathological loops / faulty launches inside the kernels;
- a single **Bedrock `InternalServerException`** also occurred during the run (auto-retry succeeded) — occasional backend errors are another uncontrollable factor;
- not until trials 3–5 did correct candidates appear, but they all stalled at 33.9–34.0 ms (low peak) and never escaped.

### v5's distinctive trajectory (high peak throughout, still improving via evolution)

- **broke into the peak right at trial 0**: v0/v1 failed correctness, but v3 immediately hit 1.11 ms / 30.5×, locking in a high-peak elite on the very first trial;
- **steady refinement trial by trial** (classic MAP-Elites mutation around the high-peak elite): the best runtime within each trial went roughly 1.11 → 0.623 → 0.749 → 0.905 → 0.736 → 0.636 ms, all within the 0.6–1.1 ms high-peak band;
- **global best trial_1_v1 = 0.623 ms / 54.4×**, the fastest single candidate across all six runs;
- **21 of 24 candidates correct**, with only 3 correctness failures (trial_0_v0/v1, trial_1_v2) — the highest hit rate of the six;
- no Bedrock errors, no pathological timeouts, and the shortest wall-clock time (about 7 minutes) — a polar opposite to v4.

### Authenticity check of the best candidates (all verified as genuine measurements, not measurement artifacts)

| Run | runtime | std | # measurements | correctness | fallback? |
|---|---|---|---|---|---|
| b580 | 33.9 ms | 0.023 ms | 29 | True (but = baseline) | no |
| v1 | 13.5 ms | 0.035 ms | 74 | True | no |
| v2 | 1.05 ms | 0.023 ms | 235 | True | no |
| v3 | 1.01 ms | 0.0025 ms | 140 | True | no |
| v4 | 33.9 ms | 0.127 ms | 29 | True (but = baseline) | no |
| v5 | 0.623 ms | 0.00496 ms | 700 | True | no |

- All "fast" results pass full pytest correctness (wrt_pytorch + wrt_reference);
- runtime_stats are statistics over hundreds of repeated measurements, std < 2% — not a -1.0 sentinel or a 0.0 placeholder;
- the `matmul_reference` / `fallback launch` matches in the logs are all benign (launch warnings from failed candidates + the test fixture loading baseline source), and **no candidate regressed to the reference implementation**.

---

## 3. Core Investigation: "Did later runs reference earlier results?"

Because the first three runs increased monotonically (1.0× → 2.5× → 32×), we suspected cross-run state leakage (later runs reading the good kernels found by earlier ones). **After systematic investigation plus the fifth and sixth controls, this is fully ruled out:**

| Potential leakage channel | Finding | Conclusion |
|---|---|---|
| `/tmp` build/test temp dirs | tempfile mechanism auto-deletes after the run; no residue | ✗ no leakage |
| `/tmp` persistent db/checkpoint/archive | only files from another task on June 11 (flash-attention), unrelated to matmul | ✗ unrelated |
| `runs/kernels.sqlite3` database | contains only baseline rows, **all 33.9–34.0 ms / score=5**, no record of any fast 13.5/1.05 ms kernel | ✗ no fast kernel to reference |
| MAP-Elites archive inheritance | all six startup logs show `Total programs: 0 / Available for evolution: 0`; every run cold-starts from an empty archive | ✗ no inheritance |
| DB inspiration-retrieval isolation | code filters by `task_name` (`evolve_database_optimization_aware.py:1521`, `:1590`); the six task_names all differ, so they are naturally isolated | ✗ mutually invisible |

**Decisive evidence**: even if cross-run reads existed, the DB only ever stored the 33.9 ms baseline — **v1's 13.5 ms or v2/v3/v5's sub-millisecond results simply do not exist anywhere for a later run to reference.**

**Three independent falsifications**:
1. **v3 (4th run) refutes "monotonic accumulation"**: v3's speedup of 33.6× ≈ v2's 32.3× — it did not keep climbing significantly.
2. **v4 (5th run) refutes "getting better each time"**: v4 **fell back to 1.00×** (low-performance peak). If any form of cross-run accumulation / referencing existed, v4 could not possibly be 30× worse than v2/v3.
3. **v5 (6th run) refutes again + falsifies in reverse**: v5 came right after v4 yet **exploded from 1.0× to 54.4×** (global best), and **overtook** v2/v3. If results "relayed" in run order or were influenced by the previous run, neither the v4→v5 jump nor v5 surpassing the earlier v2/v3 could be explained. The six-run sequence **{1.0×, 2.5×, 32×, 34×, 1.0×, 54×} swings wildly and disorderly** — hard proof of "each run independently samples at random and lands in one of the two peaks."

---

## 4. Root Cause of the Huge Differences

Differences = **two-layer randomness × evolutionary max-extraction × bimodal solution space**.

### 4.1 Two-layer randomness (two independent, non-reproducible sources)

| Layer | Source of randomness | Controlled by `random_seed=42`? | Consequence |
|---|---|---|---|
| **① Prompt construction** | `np.random.choice(KERNEL_OPTIMIZATION_TIPS, n_tips, replace=False)` (`template_manager.py:183`) randomly picks which optimization tips to inject | Nominally controlled, but **drifts** with call count / task_name; not stably reproducible across runs | Different runs feed the LLM different "tips" |
| **② LLM sampling** | Bedrock `invoke_model`, temperature=0.1, **no `seed` field in the request body** (`gnai_inference.py:411-425`) | **Completely uncontrolled** | Even with an identical prompt, it produces a different kernel |

**Empirical evidence**:
- The first three runs had identical trial_0 prompt md5 (`a19bd733`, 16672 B), yet the generated kernel md5s were **all different** → proving ② LLM sampling is non-deterministic (same input → different output).
- The trial_0 prompt md5 took **three distinct values** across the six runs: `a19bd733` (b580/v1/v2), `f20d0784` (v3/v4), and `823e7ffa` (v5, 16695 B). v5's injected tips were yet another set (Aggressive B prefetch + SLM bank-conflict striding + cooperative/three-level prefetch + double/triple buffering), different from both prior sets → further confirming that ① prompt construction itself carries randomness.
- **The six trial_0 generated-kernel md5s are all different**:
  - b580: `f89ce718` (11903 B)
  - v1: `4a162a5c` (21639 B)
  - v2: `ce71a0be` (15586 B)
  - v3: `893fb11f` (15178 B)
  - v4: `5b8bfa69` (12974 B)
  - v5: `b9ba2e6e` (15477 B)

> `random_seed=42` (`evolve_database_optimization_aware.py:741-744`) only seeds the framework-side Python/numpy RNG (parent selection, dimension sampling). It **neither reaches the LLM nor stably fixes the prompt-tip sampling across runs / across task_names**.

### 4.2 The evolutionary algorithm amplifies "sampling luck" into "final performance"

MAP-Elites takes the **maximum (max) over the result distribution, not the average**. Each run = drawing 6×4 = 24 samples from the solution space, keeping the best and refining around it. Once some trial hits an efficient solution, the elitism mechanism immediately locks that cell and keeps mutating/refining it; conversely, if the initial elite is a "correct but slow" local optimum, the whole population is locked into its genes (this is exactly what happened to b580 / v4, stuck at 33.9 ms the whole time).

**v4 and v5 are the two extremes of this amplification mechanism**:
- **v4 (amplifying a bad opening)**: trials 0–2 did not even draw a single correct candidate, wasting half the trials; the population never escaped the low peak → 1.0×.
- **v5 (amplifying a good opening)**: trial 0 drew a high-peak elite (v3 = 1.11 ms), and subsequent trials kept mutating/refining around it, gradually pushing down to 0.623 ms, with 21 of 24 candidates correct → 54.4×.

**The same algorithm: the quality of the opening sample is amplified manyfold by the evolutionary process into wildly different final performance** — this is the core amplifier behind the "huge differences."

### 4.3 The matmul solution space is "bimodal"

- **Low-performance peak**: "formally correct DPAS, but with degenerate memory-access / data-reuse patterns" → bandwidth-bound, 1–2.5× (b580 and v4 at 1.0×; v1 slightly toward the middle at 2.5×).
- **High-performance peak**: "correct AND genuinely saturating XMX (correct SLM layout / sub-group dataflow / K-tile pipelining / multi-level prefetch)" → **~21–35×, and itself a range with width** (v2/v3 land on the ~21 TFLOPS side, v5 surges to ~34.5 TFLOPS).

Writing "formally correct DPAS" is easy, but which peak you land in depends on sampling luck. This explains why the differences are not a continuous 1.5×/2× gradient but a **Bernoulli-style "hit / miss" bimodal split**: of the six runs, two landed low (b580/v4), one in the middle (v1, 2.5×), and three high (v2/v3/v5). A rough estimate of the per-run probability of hitting the high-performance peak is about **3/6 ≈ 50%** (small sample, order-of-magnitude only). **Note there is also significant variance within the high peak**: even when both hit the high peak, v5 (0.623 ms) is still ~40% faster than v2/v3 (~1.0 ms), showing a second layer of randomness on top of "hitting the high peak" — namely "how well you can refine within the high peak."

### 4.4 The high peak is a range; a hardware ceiling exists but is far from reached

Initially, based on v2/v3 (both ~1.0 ms / ~21 TFLOPS), we inferred "~21 TFLOPS is the physical ceiling." **v5 overturns this inference**: it achieved 0.623 ms ≈ **34.5 TFLOPS**, ~40% faster than v2/v3. So:

- The high peak is not a single convergence point but a **range of ~21–35 TFLOPS**; the earlier "ceiling" was just an illusion from two samples happening to land on the lower edge of the range;
- For the B580 (fp16 XMX theoretical peak ~230 TFLOPS), even v5's 34.5 TFLOPS is only ~15% of the theoretical peak, so the **true hardware ceiling is far from reached** — meaning there is still substantial optimization headroom within the high peak that sampling does not reliably hit;
- But the conclusion "a ceiling exists, growth is not unbounded" still holds: after hitting an efficient solution, performance approaches (rather than exceeds) the hardware's capability, and none of the six runs produced numbers outside the physically reasonable range — mechanistically ruling out "unbounded-increase leakage." v4 falling back to 1.0× also confirms from the opposite direction: performance does not accumulate monotonically across runs.

---

## 5. Conclusions and Methodological Recommendations

1. **This pipeline is inherently non-deterministic and non-reproducible.** "Same command, different result" is by design (two-layer randomness + no LLM seed) — **not a bug, and not cross-run leakage.**

2. **It is definitely not "later runs referencing earlier ones."** All six cold-start from an empty archive, the DB stores only the baseline, and there is no leakage channel; v3 ≈ v2 stops rising, v4 falls back to 1.0×, and v5 right after v4 explodes to 54× while overtaking the earlier v2/v3 — three independent pieces of evidence that decisively refute the accumulation hypothesis.

3. **A single run's result is not trustworthy.** With the same config, the outcome can land anywhere from 1.0× to 54×. The six-run distribution is **{1.0×, 2.5×, 32×, 34×, 1.0×, 54×}**, median ≈ 18× (between 2.5× and 32×), but clearly bimodal (low peak ~1× twice, middle 2.5× once, high peak ~21–54× three times). **A single run reporting 54× or 1× does not represent the method's true capability**; and even when both hit the high peak, final performance can still differ by 40% (v5 vs v2/v3), so even "the number within the high peak" needs multiple runs to characterize as a distribution.

4. **Practical recommendations** (for trustworthy / reproducible conclusions):
   - **Run multiple times and report the distribution** (at least 3–5 runs; report min / median / max and the bimodal split). These 6 runs already show extreme variance (1.0×–54×); production evaluation should run even more;
   - **Fix the LLM `seed` + lock temperature** (requires modifying `gnai_inference.py` to pass a seed in the invoke_model body);
   - **Lock the prompt tips** (fix the tip sampling in `template_manager.py`, or explicitly pass the full tip set);
   - **Seed the DB with verified efficient kernels** to avoid re-rolling the lottery on every cold start — start refining directly from the high-performance peak;
   - **Increase `branches_per_iteration` / `max_iters`** to raise the per-run probability of hitting the high-performance peak (more sampling = more likely to draw the long-tail efficient solution), at the cost of more Bedrock calls and time.

---

## Appendix: Key Code Locations

- LLM call has no seed: `kernelgen/gnai_inference.py:411-425` (`BedrockInferenceServer.__call__`; body contains only messages/max_tokens/anthropic_version/temperature/top_p)
- Framework RNG seed: `kernelfoundry/kernelfoundry/algorithm/evolve_database_optimization_aware.py:741-744`
- Random prompt-tip sampling: `kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py:183`
- DB retrieval isolated by task_name: `kernelfoundry/kernelfoundry/algorithm/evolve_database_optimization_aware.py:1521, 1590`
- kernels DB default path: `kernelfoundry/configs/paths/default.yaml` (`kernels_db_path: sqlite:///runs/kernels.sqlite3`; readonly/insertonly default to null)
