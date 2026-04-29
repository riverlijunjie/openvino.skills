# Grouped GEMM Integration Details

This document covers the oneDNN grouped GEMM integration for the MoE prefill path, including the `DNNL_ARG_HINT_MAX_GROUP_SIZE` mechanism, performance analysis, and optimization opportunities specific to the prefill execution path.

---

## 1. DNNL_ARG_HINT_MAX_GROUP_SIZE

### API Design

The grouped GEMM dispatches workgroups as `gws ~ N * total_M * E`. Without a hint, each expert dispatches `div_up(total_M, tile)` workgroups — massive oversubscription since most experts process far fewer tokens.

`DNNL_ARG_HINT_MAX_GROUP_SIZE` (value=384) is an optional runtime execute argument:
- **Type**: Host scalar `s32` memory object
- **Semantics**: Upper bound on per-group M dimension for grid dispatch
- **Safety**: Values outside `(0, m_all]` fall back to `m_all`; kernel `if (wg_j0 >= m) return;` handles over-dispatch

### oneDNN Implementation

In `grouped_micro_gemm.cpp execute()`:
```cpp
const int32_t *max_var_dim = CTX_IN_MEM(const int32_t *, DNNL_ARG_HINT_MAX_GROUP_SIZE);
dim_t m_dispatch = m_all;
if (max_var_dim && *max_var_dim > 0 && *max_var_dim <= m_all)
    m_dispatch = *max_var_dim;
```

### OpenVINO Integration

In `moe_3gemm_swiglu_opt.cpp`:
- `get_grouped_kernel()`: Creates grouped descriptors without dispatch hint. Cache key = `total_tokens` only.
- Runtime hint: Before each of three grouped GEMMs, creates host_scalar memory with `max_tokens_per_expert`:
  ```cpp
  auto hint_md = dnnl::memory::desc::host_scalar(dnnl::memory::data_type::s32);
  dnnl::memory hint_mem(hint_md, static_cast<int32_t>(max_tokens_per_expert));
  {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}
  ```

### Dispatch Improvement Example

Balanced routing, 1024 tokens, 128 experts:
- Without hint: each expert dispatches `div_up(1024, tile)` ≈ 32 workgroups
- With hint: `max_per_expert ≈ 15` → ~1 workgroup per expert (32× reduction)

---

## 2. CPU vs GPU Mask Generation

The `moe_mask_gen.cl` kernel has an inherently serial inner loop — each work-item (one per expert) iterates `num_tokens × NUM_EXPERTS_PER_TOKEN` entries linearly to find its tokens. With 128 experts, only 128 threads run.

**Verdict**: CPU mask gen is preferred. CPU benefits from:
- Better branch prediction and speculative execution
- Cache locality on sequential reads
- The CPU→GPU sync cost is smaller than GPU under-utilization penalty

---

## 3. Prefill Optimization Opportunities

### 3.1 Fuse Gate+Up into Single Grouped Matmul

**Current**: Gate and Up are two separate `grouped_matmul.execute()` calls sharing the same source tensor.

**Optimization**: Fuse into single `[E, hidden, 2*inter]` matmul. Output `[total, 2*inter]` split for SiLU.
- Halves GPU kernel launches (3→2 grouped GEMMs)
- Improves GPU occupancy by doubling work per dispatch
- Requires changing weight layout in `convert_moe_to_compressed.cpp`

### 3.2 Fuse SwiGLU into Gate GEMM Post-Op

**Current**: `prefill_swiglu` is a separate OCL kernel after both gate and up GEMMs.

**Optimization**: Use oneDNN post-ops (binary multiply + eltwise SiLU) on gate matmul to eliminate kernel + round-trip read/write of `[total_tokens, inter_size]`.

### 3.3 Primitive Caching Over-Compiles

One cache entry per distinct `total_tokens`. Since `total_tokens = token_num * top_k` varies with every prompt, this causes JIT compilation for each new prompt length.

**Optimization**: `DNNL_RUNTIME_DIM_VAL` for M-dimension (pending oneDNN support), or bucketing to powers-of-2.

### 3.4 Scatter-Reduce Phase 2 Overhead

At 8K+ tokens, the linear token search in Phase 2 of `moe_scatter_reduction_opt.cl` becomes ~24% overhead.

**Optimization**: Build inverse index `token_id → (expert_slot_offset, expert_weight_index)` during CPU mask gen (trivially cheap). Phase 2 becomes a single direct lookup.

Expected: ~1.9% improvement at 8K tokens (~79ms over 48 layers).

---

## 4. Performance Data (Qwen3-30B-A3B)

### Prefill + Decode Comparison

| Input Tokens | Prefill (per-expert loop) | Prefill (grouped) | Prefill Δ | Decode (loop) | Decode (grouped) | Decode Δ |
|-------------|--------------------------|-------------------|-----------|--------------|-------------------|----------|
| 32 | 149 ms | 154 ms | +3.3% | 22.0 ms | 23.7 ms | +7.9% |
| 1024 | 500 ms | 490 ms | **-2.0%** | 24.1 ms | 25.0 ms | +3.7% |
| 2048 | 823 ms | 841 ms | +2.2% | 25.6 ms | 25.9 ms | +1.4% |
| 4096 | 1682 ms | 1725 ms | +2.6% | 25.6 ms | 26.0 ms | +1.4% |
| 8192 | 4735 ms | 4183 ms | **-11.7%** | 31.8 ms | 29.4 ms | **-7.6%** |
| 32768 | 65611 ms | 66926 ms | +2.0% | 1128 ms | 1057 ms | **-6.3%** |

**Key observations**:
- Grouped GEMM shows clear prefill benefit **only** at 8K tokens (-11.7%)
- Decode benefits at long contexts (8K/32K) due to reduced KV cache overhead
- At 32K tokens, decode explodes (1056+ ms/tok) — dominated by attention/KV, not MOE

### Recommended Path Selection Heuristic

| token_num | Recommended Path |
|-----------|-----------------|
| < 512 | oneDNN per-expert loop |
| 512 – 4096 | micro_gemm |
| ≥ 4096 | Grouped GEMM |
