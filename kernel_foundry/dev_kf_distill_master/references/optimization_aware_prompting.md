# Optimization-Aware Prompting & Meta-Prompting

Distilled from `kernelfoundry/algorithm/prompts/` (`prompt_constructor.py`, `optimization_aware.py`,
`meta_prompting.py`, `prompt_evolution_integration.py`, `feedback_llm.py`, `languages.py`,
`optimization_aware_prompts.json`, and the `templates/*.j2`).

The prompt is the *mutation operator* of the evolutionary loop. Its quality is what actually
produces better kernels, so the system layers three sources of optimization knowledge into it and
even evolves the prompt itself.

---

## 1. How the main prompt is assembled

`PromptConstructor.__call__` builds the user prompt in five stages; `templates/main_prompt.j2`
renders the final text. Sections, in order:

1. **Role / objective** — "You are a {LANGUAGE} GPU kernel optimization expert. Given a reference
   {ref_language} implementation, create a performant {LANGUAGE} kernel with identical functionality."
2. **(Evolvable) Optimization philosophy** — high-level mindset; replaceable by meta-prompting.
3. **Execution context** — exactly how the code will be loaded/called (torch `cpp_extension.load`,
   Triton function replacement, existing-project signature, …) so the LLM emits a usable artifact.
4. **User instructions** — optional, from the task's `[USER_INSTRUCTIONS]` block.
5. **RAG examples** — retrieved before/after examples (iteration 0 gets a simple vector-add).
6. **Reference code** — the PyTorch/C++/SYCL reference (the `[REFERENCE]` block).
7. **Inspirations** — previous versions *with their scores* (`num_inspirations`).
8. **Best kernel so far** — the top program's code + result line.
9. **Last kernel + feedback** — the parent's code and either its raw console output or the
   feedback-LLM digest. This closes the loop: errors/profiler findings drive the next attempt.
10. **Hardware spec** — target GPU details.
11. **Task block** — status-dependent objective: `translate` (write from scratch) /
    `error` (analyze log → fix) / `correct` (analyze → optimize further).
12. **(Evolvable) Optimization strategies** — concrete techniques (static tips, or the
    profile-targeted "Required Optimizations", or meta-evolved text).
13. **(Evolvable) Common pitfalls** + **Critical requirements** — language-specific hard constraints
    (e.g. SYCL `range<N>` only N∈{1,2,3}; sub-group free-functions not `.shuffle()`; name kernels;
    don't capture `constexpr` in device lambdas; Triton `program_id` axis ∈{0,1,2}).
14. **(Evolvable) Analysis guidance** + response format ("1. Analysis … 2. Code in a ```LANG block").

Sections 2, 12, 13, 14 are the four **evolvable regions** (meta-prompting, §4).

---

## 2. The three layers of optimization knowledge

1. **Static tips** (`languages.py: KERNEL_OPTIMIZATION_TIPS`) — ~13–18 one-liners per language
   (SYCL/CUDA/Triton/OCL). `num_optimization_tips` (2) are sampled per prompt. Quick, always-on.
2. **Parametric, dimension×level knowledge** (`optimization_aware_prompts.json`) — the heart of
   "optimization-aware". Indexed by `[dimension][backend][level]` (§3). When a target cell is
   chosen (algorithm §8), the techniques for that cell's levels are formatted into a
   **"## Required Optimizations"** section the kernel must apply (`build_exploration_prompt`).
3. **Evolved knowledge** (meta-prompting, §4) — LLM-rewritten versions of the evolvable regions,
   learned from which prompts produced the fastest kernels.

`scripts/optimization_knowledge.json` ships a compact but faithful copy of layer 2.

---

## 3. The optimization-knowledge JSON (`optimization_aware_prompts.json`)

Top-level keys and what they hold:

| Key | Sub-keys | Content |
|---|---|---|
| `memory` | sycl, cuda, triton, opencl | levels 1–3 of the memory ladder (algorithm §4) |
| `compute` | sycl, cuda, triton, opencl | levels 1–3 of the compute ladder |
| `parallelism` | sycl, cuda, triton, opencl | levels 1–3 of the parallelism ladder |
| `explicit_simd` | sycl_esimd, cuda_tensor_cores, opencl | DPAS / WMMA / Tensor-Core guidance |
| `antipatterns` | sycl, cuda, opencl, triton | what NOT to do (bank conflicts, over-sync, divergence) |
| `performance_hints` | sycl, cuda, triton, opencl | occupancy, work-group sizes, fast math |
| `upgrades` | sycl_esimd, cuda_tensor_cores, opencl, triton | how to move up a level |
| `dimension_guidance` | sycl, cuda, opencl, triton | how to reason about each dimension |

Each `[dimension][backend][level]` is a markdown block: **Objective → Required Transformations
→ BEFORE/AFTER code**. Representative content (CUDA):

- **memory L1** vectorized & coalesced access (`float4`, alignment); **L2** SMEM tiling (16×16/32×32,
  `__syncthreads()`, +1 padding vs bank conflicts); **L3** register blocking THREAD_M×THREAD_N +
  `#pragma unroll` + double buffering.
- **compute L1** kernel fusion + `fmaf`; **L2** single-pass online algorithms (online softmax:
  `scale=exp(m_old-m_new); l = l*scale + exp(x-m_new)`; Welford variance; Kahan sum); **L3**
  Flash-Attention-style blocked processing with running `(m_i,l_i,acc)`, Tensor Cores.
- **parallelism L1** tree reductions (replace atomics); **L2** warp shuffle (`__shfl_xor_sync`);
  **L3** hierarchical block→warp→thread decomposition.

SYCL mirrors this with `local_accessor`, `group_barrier`, `reduce_over_group`, ESIMD `simd<>` /
`block_load` / DPAS. Triton adds `tl.make_block_ptr` + `tl.advance`, `num_stages=3`,
`@triton.autotune`. The point: **the model is told the next rung of the ladder, concretely.**

---

## 4. Meta-prompting (evolving the prompt itself) — `meta_prompting.py`

A second evolutionary loop over *prompts*, "Science-CodeEvolve"-style:

- A **`PromptProgram`** = `{template_overrides{region→text}, base_template, fitness, generation,
  parent_id, best_child_code, best_child_metrics}`. Its **fitness = best score of any kernel it
  generated**.
- A holistic-prompt database stores prompts; `sample()` selects by
  `softmax(fitness − usage_penalty + exploration_bonus)` (under-used prompts get a bonus).
- Every `evolution_interval` (10) generations, the meta-prompter (an LLM, `meta_prompting_*.j2`)
  reads a prompt's evolvable sections + its best kernel + that kernel's metrics, then emits
  **SEARCH/REPLACE diffs** to improve one of the four regions:

  ```
  <<<<<<< SEARCH [optimization_strategies]
  - Coalesced memory access: ...
  =======
  - Coalesced memory access: ensure adjacent threads hit adjacent addresses; cast to float4 for
    16-byte transactions; ...
  >>>>>>> REPLACE
  ```

- Strategies: `IMPROVE`, `SPECIALIZE`, `GENERALIZE`, `SIMPLIFY`, `ELABORATE`.

`prompt_evolution_integration.py` wires this in: each kernel generation samples a prompt
(`session_id`), and after evaluation the kernel's score is reported back to update that prompt's
fitness — so prompts that yield fast kernels are selected and refined over time.

---

## 5. Feedback injection — `feedback_llm.py`

After evaluation the eval log (compiler errors, pytest failures, profiler summary) is fed into the
*next* prompt as the parent's feedback. Two modes:
- **Raw**: cleaned console output (see evaluation doc's postprocessing) goes in directly.
- **`use_feedback_llm: true`**: a cheaper LLM (`feedback_llm_prompt.j2`) digests the log into
  `(I) general feedback · (II) console summary · (III) numbered errors → fix suggestions`.
- Optional SYCL docs lookup: `<kw>...</kw>` keywords in feedback are replaced with spec excerpts.

This is what turns the loop from "random restarts" into directed iterative refinement.

---

## Reimplementation checklist
- A template with the 14 sections; mark 4 as evolvable.
- A `[dimension][backend][level]` knowledge store (ship the JSON).
- A function that, given a target cell, emits a "Required Optimizations" section from that store.
- Feedback ingestion (raw or LLM-digested) into the next prompt.
- (Optional) a prompt database + meta-prompter applying SEARCH/REPLACE diffs, fitness = best child.
