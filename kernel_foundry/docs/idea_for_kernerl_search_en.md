# KernelFoundry Kernel-Search Optimization Plan

**Target hardware**: Intel Arc B580 (Battlemage G21, XMX/DPAS)
**Benchmark task**: OCL matmul fp16 (2048×2560×2048)
**Basis**: six identical-command experiments (see `analysis_of_multi_tests_result.md`) + earlier tuning experience
**Code version**: git commit `98f26eb`

---

## 0. Starting Point: Three Core Pain Points Exposed by the Six Experiments

| Observation (measured) | Root cause | Corresponding improvement |
|---|---|---|
| Six runs of the same command gave {1.0×, 2.5×, 32×, 34×, 1.0×, 54×}; high-peak hit rate only ~50% | Two-layer randomness + evolutionary max-extraction + bimodal solution space; **no early-termination/early-stop mechanism at all** — it only stops after running the full `max_iters` | §1 early stop + §4 multi-run distribution |
| In v4, three trials (12 candidates) all failed correctness, with each bad kernel dragging the correctness test out to ~100 s; yet high-peak candidates run the full 100 perf trials | Fixed evaluation cost: `num_perf_trials=100` and full profiling treat **good and bad candidates identically**, so bad candidates waste a lot of budget | §2 coarse→fine tiered evaluation |
| trial_0 prompt showed 3 different md5s; the prompt only randomly injects **2** generic tips, with **no XMX/DPAS-specific guidance**, and `use_feedback_llm=False` by default | Prompt construction is random and low in information; the feedback loop is weak; hardware priors are missing | §3 more effective prompts |

**Overall goal**: under the **same or lower Bedrock-call budget**, raise the probability of "hitting the high-performance peak" from ~50% to as high as possible, and make a single run converge faster and more reproducibly.

---

## 1. Search Early-Stopping

### 1.1 Current state

The main loop `for trial in range(config.max_iters)` at [controller.py:615](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L615) has **only one early exit**, at [controller.py:798-799](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L798):

```python
if config.stop_once_correct and kernel_exec_result.compiled and kernel_exec_result.correctness:
    break
```

Problem: `stop_once_correct=true` is too aggressive (it stops at the first **correct** candidate, which usually lands in the low peak at 1.0×), while `false` runs all iterations. **Between these two extremes there is no middle strategy for "stop once performance has converged" or "stop once it's clearly hopeless."**

### 1.2 Plan: three classes of early-stop conditions (inserted alongside L798)

At the same location in `controller.py:798`, add a `should_stop_early(trial, database, config)` call:

```python
# existing
if config.stop_once_correct and kernel_exec_result.compiled and kernel_exec_result.correctness:
    break
# new
stop, reason = should_stop_early(trial, self.program_database, best_history, config)
if stop:
    logging.info(f"[early-stop] trial {trial}: {reason}")
    break
```

Three conditions (union; stop if any is satisfied):

**(A) Performance-plateau early stop (patience)** — addresses "still burning money after convergence"
- Maintain `best_history` (the global-best `runtime_improvement` at the end of each trial).
- If for `early_stop_patience` consecutive trials (suggested **3**) the relative improvement of the best value is < `early_stop_min_delta` (suggested **2%**), stop.
- Against the measurements: v5 reached 0.623 ms already at trial 1, and the following 4 trials only jittered in the 0.6–0.9 ms range → it could have stopped around trial 3, saving ~40% of the budget.

**(B) Target-reached early stop (target speedup)** — addresses "pointless refinement after hitting the peak"
- If the global best `runtime_improvement >= early_stop_target_speedup` (suggested **30×** for B580, i.e., solidly in the high peak), run `target_patience` more trials (suggested **1**) to confirm no regression, then stop.
- Difference from `stop_once_correct`: it requires **high performance** rather than mere **correctness**, avoiding lock-in at the low peak.

**(C) No-hope early stop** — addresses the v4-style "all-failures opening yet still running to the full count"
- If the first `nohope_warmup` trials (suggested **3**) **never produced any correct candidate**, or the best is still `< 1.5×`, judge that this cold start got unlucky.
- One of two actions (configured via `nohope_action`):
  - `stop`: abandon this run immediately and let the upper-level "multi-run" strategy re-roll (recommended, combined with §4);
  - `reseed`: force the next trial to use a "verified high-peak kernel" as seed (combined with §3.4), rescuing a bad run.

### 1.3 New config (`run.yaml`)

```yaml
early_stop:
  enable: true
  patience: 3              # (A) number of consecutive trials with no significant improvement
  min_delta: 0.02          # (A) significant-improvement threshold (relative 2%)
  target_speedup: 30.0     # (B) prepare to stop once reached (B580 high-peak lower edge ~21×, with margin set to 30)
  target_patience: 1       # (B) trials to confirm after target is reached
  nohope_warmup: 3         # (C) observation window for judging "no hope"
  nohope_min_speedup: 1.5  # (C) best below this within the window counts as no-hope
  nohope_action: stop      # (C) stop | reseed
```

### 1.4 Expected benefits
- High-peak runs (v2/v3/v5 type) save **30–40%** of trials on average;
- No-hope runs (v4 type) cut losses early or are rescued via reseed;
- Combined with §4, more independent runs fit into a fixed total budget → higher overall probability of hitting the high peak.

---

## 2. Search Coarsely First, Then Precisely (Coarse-to-Fine)

### 2.1 Current state: evaluation cost treats all candidates equally

Every candidate is evaluated at full settings per [run.yaml:96-102](kernelfoundry.internal/configs/run.yaml#L96):
- `num_perf_trials: 100`, `warmup_min_iters: 10`, `warmup_min_time: 0.1`, `inner_loop_min_time: 0.01`;
- `profile_custom_model: true` (full profiling).

Measured cost: a single v5 high-peak candidate runs **600–1200** perf measurements; v4's bad candidates also drag the correctness test out to ~100 s. **Bad candidates cost the same as good ones** — extremely wasteful.

The warmup/trials of the performance measurement are **adaptive and config-driven** ([performance.py:251-276](kernelfoundry.internal/kernelfoundry/kernelfoundry/eval_pipeline/utils/performance.py#L251)), so **lowering these values yields a cheap coarse measurement** — this is the key lever for tiered evaluation.

### 2.2 Plan: two-stage evaluation (coarse screening → fine measurement)

Change "one candidate = one full evaluation" into a **two-stage funnel**:

```
branches_per_iteration candidates
        │
   ┌────▼─────────────────────────┐
   │ Coarse:                       │  cheap, only for ranking
   │  · correctness must pass (short timeout)│
   │  · a few perf trials to estimate the order of magnitude│
   │  · no profiling               │
   └────┬─────────────────────────┘
        │ take top-K (by coarse runtime)
   ┌────▼─────────────────────────┐
   │ Fine:                         │  expensive, only for winners
   │  · full num_perf_trials=100   │
   │  · full profiling             │
   └──────────────────────────────┘
```

**Coarse config** (temporarily override eval_config; saves ~90–95% of measurement time):
```yaml
coarse_eval:
  num_perf_trials: 3
  warmup_min_iters: 2
  warmup_min_time: 0.02
  inner_loop_min_time: 0.002
  profile_custom_model: false
  test_timeout: 40          # kill bad kernels fast, avoiding the v4-style ~100 s drag
```

**Fine config**: keep the existing `run.yaml` defaults (`num_perf_trials=100` + profiling), run **only for the coarse top-K**.

### 2.3 Implementation notes

- **Insertion point**: [controller.py:651-691](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L651). Currently the `branches_per_iteration` candidates run in parallel and then go straight into `evaluate_batch` at full settings; change it to first `evaluate_batch(coarse_cfg)` → sort → select `coarse_top_k` → then `evaluate_batch(fine_cfg)`.
- **Evolution uses only fine results**: write to the MAP-Elites database / update `best_program` using the **fine** `runtime_improvement` (only fine is accurate); coarse is used only for "who advances to the finals."
- **Correctness short-circuit**: candidates that fail correctness in the coarse stage are eliminated immediately and never reach fine, naturally avoiding the v4 case of "a bad candidate maxing out the timeout while also consuming perf budget."
- **Coarse-noise tolerance**: the std of a 3-trial coarse measurement is large, so use it only for **rough ranking** (distinguishing 1× / 5× / 30× orders of magnitude is enough), not for the precise comparison of 0.62 vs 0.64 ms.

### 2.4 Going further: lift "coarse→fine" to the search-phase level (optional)

Beyond tiering within a single trial, the whole run can also be split into two phases (echoing "coarse first, fine later"):
- **Exploration phase (first ~60% of trials)**: high `branches_per_iteration`, high temperature, coarse evaluation → cast a wide net so that some trial hits a high-peak cell as soon as possible;
- **Refinement phase (last ~40% of trials)**: low temperature, sample parents only from the **high-peak elites** (parents taken only from `get_top_programs` top-K), full fine measurement → squeeze within the high peak (pushing v2/v3's ~1.0 ms toward v5's 0.62 ms).

### 2.5 New config
```yaml
coarse_to_fine:
  enable: true
  coarse_top_k: 2          # number of candidates advancing to fine per iter (filters out half when branches=4)
  coarse_eval: { num_perf_trials: 3, warmup_min_iters: 2, warmup_min_time: 0.02,
                 inner_loop_min_time: 0.002, profile_custom_model: false, test_timeout: 40 }
  explore_fraction: 0.6    # first 60% of trials are the exploration phase
  refine_parent_top_k: 4   # in the refinement phase, sample parents only from the top-K elites
```

### 2.6 Expected benefits
- Per-iter evaluation time roughly **-50% to -70%** (only the top-K go into full settings + profiling);
- The saved budget is reinvested into a **higher `branches_per_iteration`** (more sampling = more likely to hit the high peak, see §4);
- Bad candidates (especially v4-style) are eliminated cheaply and quickly, no longer hogging the perf budget.

---

## 3. More Effective Prompts

### 3.1 Three weaknesses of the current state

1. **Too few tips, and random**: [template_manager.py:183](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L183) uses `np.random.choice(..., n_tips, replace=False)` with default `n_tips=2`, picking 2 from OCL's ~18 **generic** tips **without a seed**. → The guidance fed to the LLM each time is both scant and drifting (3 different prompt md5s were measured).
2. **No XMX/DPAS-specific guidance**: the OCL/SYCL tips only cover vague vectorize / sub-group / bank-conflict advice, with **not a single hardware prior targeting Battlemage XMX/DPAS, SLM double-buffering, K-tile pipelining, or `intel_sub_group_block_read`**. The reason v5 set a record is precisely that it (by chance) drew the prefetch + SLM striding + double-buffer set of tips.
3. **Weak feedback loop**: `use_feedback_llm=False` (default, [run.yaml:54](kernelfoundry.internal/configs/run.yaml#L54)), and the prompt does not feed back **profiling data** (only source + runtime + console). The LLM gets no quantitative signal about "where the bottleneck is" and can only guess blindly.

### 3.2 Option A: fix + expand high-quality tips (low cost, do first)

- **Increase `n_tips`**: from 2 to **5–6** (OCL has ~18, and the assertion at [template_manager.py:62](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L62) `n_tips < min(len)` still holds). More information.
- **Optionally disable randomness**: add a `tips_deterministic` switch — when true, take a **fixed high-priority subset of tips** (or rotate by trial), eliminating the "prompt's own randomness" layer (combined with §5 for reproducibility).
- **Add a B580/XMX-specific tip section** (append into the OCL list in [languages.py](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/languages.py), or create a separate `BMG_DPAS_TIPS`), fixing the empirically effective patterns as priors:
  - use `intel_sub_group_block_read_us/_uc` for vectorized SLM/global reads of A/B;
  - double-buffer the A-tile in SLM + one barrier per K-iter for pipelining;
  - choose the right sub-group size and use `intel_reqd_sub_group_size` to fully feed the DPAS 8×16 tiles;
  - three-level prefetch (L1/L2 distance = prefetch×unrollK, L3 long-distance cooperative prefetch);
  - tune the SLM stride to avoid bank conflicts; accumulate C with register blocking.

### 3.3 Option B: feed profiling data back to the LLM (medium cost, high benefit)

- The current prompt feedback ([template_manager.py:199-221](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L199)) contains only `code + result + console_output`, and **no profiler data** (EvalResult actually carries `profiler_data`).
- Improvement: for the **high-peak candidates** in the fine stage (§2), summarize the collected occupancy / bandwidth / DPAS-utilization metrics and inject them into a "bottleneck analysis of the previous kernel" section. This turns the LLM from "blind editing" into "editing while looking at the roofline."
- Enable `use_feedback_llm=true` and customize [feedback_llm_prompt.j2](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/templates/feedback_llm_prompt.j2) so the feedback LLM specifically outputs "which optimization dimension to act on next."

### 3.4 Option C: use the high-peak kernel as a few-shot seed (high benefit, addresses the root)

- Save the verified **v5 best kernel (0.623 ms / 54.4×)** as a seed, either into the DB (`store_generated_kernels_in_db` / `kernels_db_readonly_path`) or as the prompt's "Best Kernel So Far" / RAG example ([main_prompt.j2](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/templates/main_prompt.j2) already has that section).
- Effect: each run no longer "re-rolls the lottery" from an empty archive but **starts refining near the high peak** — directly erasing the ~50% low-peak probability of the bimodal distribution. This is the most direct fix for "a single run's result is untrustworthy."
- Interlocks with §1(C)'s `reseed` action: a no-hope run injects this seed to rescue itself.

### 3.5 LLM sampling parameters (combined with §5)

The `invoke_model` body at [gnai_inference.py:411-425](kernelfoundry.internal/kernelgen/gnai_inference.py#L411) has **no `seed` field**, and temperature only takes effect when explicitly passed. Suggestions:
- add `seed` to the body (where Bedrock/Anthropic supports it) → reproducibility;
- use a higher temperature in the exploration phase (diversity) and lower it to 0.0–0.1 in the refinement phase (stable convergence) — set per phase per §2.4.

---

## 4. Multiple Runs + Budget Reallocation (Characterize the Distribution Clearly)

The experimental iron law: **a single run's result is untrustworthy** (anything from 1× to 54× is possible). At the methodology level we must:
- **Run N times and report the distribution** (≥3–5), reporting min / median / max + high-peak hit rate, rather than a single number;
- **Budget reallocation**: the time saved by §1 early stop + §2 coarse screening goes to
  - raising `branches_per_iteration` (each run is more likely to hit the high peak), and/or
  - increasing the number of independent runs N (`1-(1-p)^N`; with p≈0.5, N=4 already gives a ~94% hit rate);
- **Automatic best-of-N**: an outer script runs N times, automatically takes the global-best kernel into the DB, and exposes only one stable high-peak result to the user.

---

## 5. Reproducibility Switches (optional, for debugging / controlled experiments)

To support "control-variable" controlled experiments, provide one-click determinization:
- **LLM**: pass `seed` + fixed `temperature` in the `gnai_inference.py` body;
- **Prompt tips**: the `tips_deterministic=true` from §3.2;
- **Framework RNG**: `random_seed=42` already seeds numpy/python ([evolve_database_optimization_aware.py:741](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/evolve_database_optimization_aware.py#L741)), but ensure the tip sampling also uses this seed (currently `np.random.choice` is affected by the global numpy state and drifts across task_names).
> Note: production search should stay random (diversity is the source of hitting the high peak); determinization is only for the controlled experiment of "verifying whether a change actually helps."

---

## 7. Other Strategies to Reduce Per-Iteration Time / Accelerate Convergence

§1–§2 address "**run less, run cheaper**" (fewer trials, cheaper evaluation). This section targets an orthogonal axis: **the wall-clock time of each trial itself**, and **approaching the high peak in fewer trials**. The bottlenecks below have all been verified by reading the code.

### 7.1 Cut LLM inference wall-clock (often the longest segment of a single trial)

Measured: a single v4 Bedrock inference reached **163 s**, one of the main causes of slowing the whole run. Three directly cuttable points:

**(a) Enable prompt caching (biggest benefit, smallest change)**
- Current state: the `invoke_model` body at [gnai_inference.py:411-425](kernelfoundry.internal/kernelgen/gnai_inference.py#L411) has **no `cache_control` at all**. Yet the **large static blocks** in the prompt (system instructions, hardware specs, reference source, format template — see [main_prompt.j2](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/templates/main_prompt.j2)) are **resent in full and re-prefilled on every call**.
- Improvement: apply `cache_control: ephemeral` to the segments that are **invariant across trials / across branches** — system + reference + format instructions. Once Anthropic prompt caching hits, the **input-token billing for this part drops to ~1/10, and the TTFT (time-to-first-token) drops significantly**.
- Excellent fit: this search is exactly "the same large context + only the trailing last_kernel/feedback changes each time" — an ideal scenario for prompt caching.

**(b) Lower the `max_tokens` cap**
- Current state: [run.yaml:59](kernelfoundry.internal/configs/run.yaml#L59) `max_tokens: 2000` (earlier experiments once used 10000). Kernel source is usually 200–400 lines; the more output tokens, the longer the decode.
- Improvement: tighten `max_tokens` to just enough for one kernel + a brief analysis (e.g., cap at 4000 but with a stop sequence), preventing the model from dragging out the decode with long-winded analysis.

**(c) Parallelize multiple completions (when using `num_completions>1`)**
- Current state: [gnai_inference.py:426](kernelfoundry.internal/kernelgen/gnai_inference.py#L426) `outputs = [self._invoke_with_retry(body) for _ in range(n)]` is a **serial for loop**.
- Improvement: the file already imports `ThreadPoolExecutor`; just submit these `n` completions concurrently. The current default `num_completions=1` ([run.yaml:62](kernelfoundry.internal/configs/run.yaml#L62)) is unaffected, but it becomes linearly slower once increased — fix it preemptively.

### 7.2 Break the "generate → evaluate → next trial" pipeline bubble (large benefit)

- Current state: [controller.py:652-691](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L652) has **two serial barriers**: ① a trial's `branches_per_iteration` candidates are **all generated** (L652-654) → ② only then does `evaluate_batch` **evaluate them all** (L691) → ③ only then does the next trial's generation begin.
- Problem: during generation the GPU is idle (waiting on Bedrock), and during evaluation the LLM is idle (running on the GPU). The two resources **idle alternately**, so the per-trial wall-clock ≈ generation time + evaluation time (summed, not overlapped).
- Improvement: **pipeline generation and evaluation** — as soon as candidate A finishes generating, send it to evaluation while candidate B is still generating; one can even overlap trial N's evaluation with trial N+1's "generation based on the current best." Ideally the per-trial wall-clock ≈ max(generation, evaluation) rather than their sum, **saving roughly the shorter of the two segments**.
- Implementation: connect the "generation thread pool" at L652 and the "evaluation thread pool" at L868 with a producer-consumer queue (enqueue evaluation as soon as a generation future completes), instead of two `as_completed` barriers.

### 7.3 Candidate de-duplication / evaluation memoization (saves redundant GPU time)

- Current state: there is **no kernel-source hash / dedup / evaluation cache** anywhere in [controller.py](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py) (grep for `md5|hash|seen|cache|dedup` in controller.py finds nothing).
- Problem: at low temperature (refinement phase temp→0), the LLM very easily **generates a kernel nearly identical, or even completely identical, to the previous version**, yet it still goes through a full build + correctness + 100 perf measurements.
- Improvement: compute a hash of the normalized (comments/whitespace stripped) kernel source and maintain a `hash → EvalResult` table; on a hit, **reuse the result directly and skip the entire evaluation**. Especially effective combined with the low-temp refinement phase of §7.2.

### 7.4 Cache the reference build and measurement (saves the fixed per-startup overhead)

- Current state: the reference's measurement result is already cached **within one run** ([test_custom_task.py:87](kernelfoundry.internal/kernelfoundry/kernelfoundry/eval_pipeline/tasks/test_custom_task.py#L87) only re-measures when `task.test_result_reference is None`), but **every new run startup rebuilds + correctness-tests + benchmarks the reference again** (measured: that ~33.9 ms baseline measurement + two rounds of correctness at every startup).
- Improvement: persist the reference's build artifact and the `ref_speed=33.9 ms` pair (fixed hardware + fixed shape) in a **cross-run cache** (keyed by `gpu_arch + shape + ref_src_hash`). Since §4's best-of-N runs N times, this fixed overhead can be saved N-1 times.

### 7.5 Speed up compilation/build (saves the build segment of each candidate)

- `build_timeout: 200` ([run.yaml:78](kernelfoundry.internal/configs/run.yaml#L78)) is a cap; OCL online compilation itself is not slow (measured: build 'custom' ~0.27 s), but bad kernels may trigger long compiles. Options:
  - enable a **binary cache** for OCL kernels (NEO's `cl_cache` / offline `ocloc` build-artifact cache) so identical sources are not recompiled;
  - shorten `build_timeout` to fail pathological compiles fast (combined with §2.2's coarse-screening short timeout).

### 7.6 Approach the high peak in fewer trials (accelerate "convergence" itself)

Besides "faster per trial," we can also "reach the peak in fewer trials":

- **Island / mini-batch early elimination**: with `branches_per_iteration` candidates in parallel within a trial, use §2's coarse measurement to **cut clearly bad ones immediately** and concentrate the fine budget on the promising ones — equivalent to exploring more directions in the same wall-clock.
- **Temperature annealing**: the exploration phase uses high temp to cast a wide net and quickly hit a high-peak cell, while the refinement phase uses temp→0 for fast convergence (already mentioned in §2.4 / §3.5). Annealing converges faster than a fixed temperature throughout.
- **High-peak seed + warm start** (§3.4): start directly from a verified high-peak kernel, skipping the many trials of "climbing out of the low peak" — the most direct way to compress the "convergence trial count" from ~6 to 1–2.

> Summary: §7.1 (prompt caching) + §7.2 (pipelining) are the two biggest levers on **per-trial wall-clock**, and both leave the search semantics unchanged at zero risk; §7.3/7.4/7.5 cut redundancy; §7.6 reduces the number of trials needed.

---

## 6. Implementation Priorities and Expectations

| Priority | Change | Cost | Expected benefit |
|---|---|---|---|
| **P0** | §1 early stop (A plateau + C no-hope) | Low (~30 lines in controller.py + config) | Save 30–40% of trials; cut losses on v4-style runs |
| **P0** | §3.2 increase tip count + B580/XMX-specific tips | Low (edit languages.py / config) | Directly raise the per-run high-peak hit rate |
| **P0** | §7.1(a) enable prompt caching | Low (add cache_control in gnai_inference.py) | LLM input tokens ~1/10, TTFT↓, **directly cuts the longest segment of a single trial** |
| **P1** | §2 coarse→fine tiered evaluation | Medium (rework controller.py evaluation flow) | Per-iter -50~70% evaluation time |
| **P1** | §7.2 generation/evaluation pipelining | Medium (queue linking the two thread pools) | Per-trial wall-clock ≈ max(generation, evaluation) rather than summed |
| **P1** | §3.4 high-peak kernel as seed (= §7.6 warm start) | Medium (DB insert + prompt injection) | Erase the ~50% low-peak probability + convergence trial count 6→1~2 |
| **P2** | §7.3 candidate dedup / evaluation memoization | Low-medium (hash table) | Save redundant evaluation in the low-temp phase |
| **P2** | §7.4 cross-run reference cache | Low-medium | best-of-N saves N-1 baseline overheads |
| **P2** | §3.3 profiling feedback + feedback_llm | Medium-high | From blind editing to targeted optimization |
| **P2** | §4 best-of-N outer layer + §5 reproducibility switches | Low | Stable delivery + ability to run controlled experiments |

**Overall goal**: use §1+§2 to **halve the budget** of a single run, then use §7 to **drive down each trial's wall-clock** (prompt caching + pipelining + warm start), and reinvest the saved budget per §4 into **more sampling / more runs**; use §3 to **raise the baseline per-run probability of hitting the high peak**. Stacking these, the goal is to raise "hitting the high peak with the same command" from the measured ~50% to a stable ≥90%, compress the "trials needed to converge" from ~6 to 1–2, and steadily push the in-peak performance from v2/v3's ~21 TFLOPS toward v5's ~34.5 TFLOPS and beyond.

---

## Appendix: Key Code Anchors (where the changes land)

- Main search loop / early-stop insertion point: [controller.py:615](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L615) (loop), [:798](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L798) (the only existing break)
- Candidate generation + evaluation (coarse→fine insertion point): [controller.py:651-691](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/controller.py#L651)
- Evaluation-cost config: [run.yaml:96-102](kernelfoundry.internal/configs/run.yaml#L96) (`num_perf_trials/warmup_*/inner_loop_min_time/profile_*`), `test_timeout`/`build_timeout` ([:79](kernelfoundry.internal/configs/run.yaml#L79)/[:78](kernelfoundry.internal/configs/run.yaml#L78))
- Adaptive warmup/trials calculation: [performance.py:251-276](kernelfoundry.internal/kernelfoundry/kernelfoundry/eval_pipeline/utils/performance.py#L251)
- Prompt assembly: [template_manager.py:75-176](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L75), template [main_prompt.j2](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/templates/main_prompt.j2)
- Tip definitions / random sampling: [languages.py:65](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/languages.py#L65) (OCL ~18 tips), [template_manager.py:183](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L183) (`n_tips` default 2)
- Feedback loop: [template_manager.py:199-221](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/prompts/template_manager.py#L199), `use_feedback_llm` [run.yaml:54](kernelfoundry.internal/configs/run.yaml#L54)
- LLM call (no seed): [gnai_inference.py:411-425](kernelfoundry.internal/kernelgen/gnai_inference.py#L411)
- Framework RNG seed: [evolve_database_optimization_aware.py:741](kernelfoundry.internal/kernelfoundry/kernelfoundry/algorithm/evolve_database_optimization_aware.py#L741)
