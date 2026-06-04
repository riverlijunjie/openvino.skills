# 1. The Behavior Descriptor

> Back to [SKILL.md](SKILL.md).

Every kernel candidate is mapped to a single, deterministic 4-tuple:

```
(memory_opt, compute_opt, parallelism_opt, esimd_opt) ∈ [0..3]^4
```

These are **NOT** runtime measurements. They are **static properties of the
source code**, derived by regex pattern matching. Same code → same tuple,
forever, across machines, runs, processes.

This is the choice that makes the rest of the loop work, so it's worth
being precise about.

## Why static, not dynamic

You could imagine using runtime metrics — DRAM traffic, occupancy,
register pressure — as the descriptor. We deliberately don't:

| Static (this design)                  | Dynamic (the alternative)             |
|---------------------------------------|---------------------------------------|
| Cheap (regex on text)                 | Requires successful build + run       |
| Defined for failed candidates         | Failed candidates have no descriptor  |
| Reproducible                          | Hardware-dependent                    |
| LLM can be told what cell to aim for  | LLM can't directly target a metric    |

The cost is that the descriptor is a *proxy* — it measures whether the
LLM applied a known optimization technique, not whether that technique
actually improved performance. Combining static descriptor (for cell
assignment) with dynamic fitness (for elite selection within a cell)
gives the system the best of both: fast, complete coverage of behavior
space plus runtime-grounded ranking inside each cell.

## The four dimensions

Each dimension has 4 levels (0..3). Levels are intended as a ladder, but
the classifier does **not** require lower levels to be present at higher
levels — patterns are independent.

### memory_opt — memory hierarchy exploitation

| Level | Pattern                                      | OpenCL hint                                       |
|-------|----------------------------------------------|---------------------------------------------------|
| 0     | naive global loads/stores                    | `c[i] = a[i] + b[i]`                              |
| 1     | coalesced + vectorized                       | `vload4`, `vstore4`, `intel_sub_group_block_read` |
| 2     | SLM / shared-memory tiling                   | `__local`, `barrier(CLK_LOCAL_MEM_FENCE)`         |
| 3     | register blocking + async / multi-stage      | `async_work_group_copy`, double-buffer            |

### compute_opt — algorithmic efficiency

| Level | Pattern                                      | Hint                                              |
|-------|----------------------------------------------|---------------------------------------------------|
| 0     | multi-pass naive                             | separate kernels per stage                        |
| 1     | fused operations                             | one kernel does several ops                       |
| 2     | streaming / online algorithm                 | online softmax, single-pass reduction             |
| 3     | tiled / blocked algorithm                    | block-wise GEMM, hierarchical tiling              |

### parallelism_opt — parallelism granularity

| Level | Pattern                                      | Hint                                              |
|-------|----------------------------------------------|---------------------------------------------------|
| 0     | thread-only                                  | per-element work, no cooperation                  |
| 1     | work-group barriers                          | `barrier`, work-group reductions                  |
| 2     | sub-group / warp intrinsics                  | `sub_group_reduce_*`, `__shfl_*`                  |
| 3     | hierarchical (thread + warp + block)         | combined sub-group + work-group + grid            |

### esimd_opt — explicit SIMD (SYCL only) / Tensor Cores (CUDA)

| Level | SYCL                                         | CUDA                                              |
|-------|----------------------------------------------|---------------------------------------------------|
| 0     | no ESIMD                                     | no Tensor Core                                    |
| 1     | basic ESIMD (`sycl::ext::intel::esimd`)      | basic `wmma`                                      |
| 2     | optimized ESIMD (cooperative groups)         | `mma.sync` / CUTLASS-style                        |
| 3     | expert ESIMD (full subgroup mapping)         | grouped MMA, async copy, TMA                      |

For OpenCL the `esimd_opt` dimension is repurposed as "OCL sub-group /
DPAS" intrinsics (`intel_sub_group_dpas`, `intel_sub_group_2d_block_read`).

## How classification works

`classify_from_code(code, language) → (m, c, p, e)`:

1. Pick the language-specific pattern dict (SYCL / CUDA / OpenCL).
2. For each dimension, for each level 3 → 2 → 1, compute a weighted score:
   - `score = (sum of weights of categories with at least one matching regex)
              / (sum of all category weights)`
   - bonus: if any single category with weight ≥ 0.9 matches, score is
     floored at 0.25 — so a single definitive intrinsic (e.g.
     `async_work_group_copy`) is enough.
3. Take the highest level whose score crosses a threshold:
   - level 1 ≥ 0.15, level 2 ≥ 0.18, level 3 ≥ 0.21
4. If no level crosses, return 0.

The classifier is intentionally permissive: false positives at level 1
are cheap (just a wider exploration), false negatives at level 3 are
expensive (the model thinks it hasn't applied the technique and drifts).

## How patterns are organized

The pattern dictionaries are shaped like this:

```python
MEMORY_OPT_PATTERNS = {
    1: {  # level
        "vectorized_load": {"weight": 0.8, "patterns": [r"vload\d", r"float\d\b", ...]},
        "coalesced_index": {"weight": 0.5, "patterns": [r"get_global_id\s*\(0\)", ...]},
        ...
    },
    2: {
        "slm_decl":   {"weight": 0.9, "patterns": [r"__local\s+\w+", r"local_accessor", ...]},
        "barrier":    {"weight": 0.7, "patterns": [r"barrier\s*\(\s*CLK_LOCAL", ...]},
        ...
    },
    3: {
        "async_copy": {"weight": 1.0, "patterns": [r"async_work_group_copy", ...]},
        "double_buf": {"weight": 0.9, "patterns": [r"buf\s*\[\s*ping", ...]},
        ...
    },
}
```

There are typically three sets — `MEMORY/COMPUTE/PARALLELISM_OPT_PATTERNS`
(generic across all backends) — plus three backend-specific overlays
(`CUDA_*_PATTERNS`, `OPENCL_*_PATTERNS`) and one `ESIMD_OPT_PATTERNS`.
The merge function ORs the overlay onto the base before classification.

A representative starter dictionary covering all four dimensions is
included in [PSEUDOCODE.py](PSEUDOCODE.py); copy and extend.

## Mistakes the descriptor catches

Concrete failures the system has caught in practice:

- **"I'm using SLM" but no `barrier`** → classified as level 0 memory.
  Forces the LLM to actually emit synchronization on the next mutate.
- **`async_work_group_copy` without double-buffer** → still level 3
  because the async-copy weight is 1.0 alone. Encourages level-3
  exploration even before the model figures out the rest of the
  pattern.
- **"Tensor Core" via custom matmul** without `wmma`/`mma.sync` →
  level 0 esimd_opt. Pushes the model to use the actual intrinsic on
  the next esimd_upgrade.

## Re-implementation notes

If you port this to a new backend:

1. Copy the structure of `MEMORY_OPT_PATTERNS` etc.
2. The patterns must be **regex strings**, not callables — the
   classifier compiles them once at startup and assumes they're cheap.
3. The four-dimension shape is not load-bearing in principle, but is
   in practice: the prompts and gradient code assume 4D. Going 3D is
   easy (drop esimd / tensor-core); going 5D requires changing the
   feature-map keys and the gradient vectors.
4. Keep cells small (4–6 levels max). A 4×4×4×4 grid has 256 cells; a
   6-level 4D grid is 1296 cells, which is too many to cover in any
   reasonable budget.

## Cross-references

- This skill: [MAP_ELITES_LOOP.md](MAP_ELITES_LOOP.md) (consumes the descriptor).
- This skill: [PROMPT_PIPELINE.md](PROMPT_PIPELINE.md) (uses descriptor as `target_profile`).
- This skill: [PSEUDOCODE.py](PSEUDOCODE.py) (`PATTERNS` dict at the top —
  extend per backend).
