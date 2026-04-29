# MOE Full Work Summary

This document consolidates the detailed architecture, optimization history, and performance analysis from the MoE decode kernel development and grouped GEMM prefill integration work.

---

## 1. Decode Path: Custom GEMV Kernels

### Kernel Architecture (moe_3gemm_swiglu_mlp.cl)

**Preprocessing — Activation Loading to SLM**:
Both `mlp_gate_up` and `mlp_down` start by loading activation from global memory to SLM:
- **4-bit path** (`WEIGHT_COMPRESSEION_DT==0`): Interleaves even/odd elements for optimal sub-group block read alignment
- **8-bit/f16 path**: Direct copy to SLM
- **xg_sum**: Per-group activation sum, computed only when `HAS_ZP=1` (asymmetric ZP compensation)

**GEMV Functions**:

| Function | Weight Type | Status |
|----------|-------------|--------|
| `gate_up_gemv_n2x_u4` | u4/i4 | **Active** — two-call gate+up GEMV |
| `gate_up_gemv_n2x_u8` | u8/i8 | **Active** — two-call gate+up GEMV |
| `gate_up_gemv_n2x_f16` | f16 | **Active** — two-call gate+up GEMV |
| `gate_up_gemv_fused_u4/u8/f16` | various | Experimental fused (slower, commented out) |
| `down_gemv_n2x_u4/u8/f16` | various | Down projection GEMV × routing_weight |

**Entry Point Kernels**:
- `mlp_gate_up`: Loads activation → (optional) scalar gate for shared expert → gate+up GEMV
- `mlp_down`: Loads activation → down GEMV × routing_weight
- `mlp_reduce`: Cross-expert sum reduction

### Optimization A: Fused Gate+Up GEMV — Attempted, Reverted

**Hypothesis**: Fusing gate and up saves 1 SLM read per group, eliminates `y[]` intermediate write, exploits ILP.

**Result**: Fused version **slower** than two separate calls.

**Root Cause** (by impact):
1. **Register pressure → occupancy drop** (primary): Non-fused ~10-12 GRF vs fused ~20-24 GRF → 2× allocation → fewer concurrent HW threads → exposed memory latency (~300+ cycles)
2. **Cache thrashing**: Fused reads from 4 disjoint addresses per iteration vs non-fused 1 contiguous region → 2× working set
3. **I-cache pressure**: Unrolled fused ~384 instructions vs non-fused ~192

**Key lesson**: For memory-bound GEMV, **low register pressure + high occupancy + sequential memory access** always beats reducing SLM/ALU operations.

### Optimization B: Parallelized Shared Expert Scalar Gate — Applied

Single thread computing `dot(x, gate_weight)` over HIDDEN_SIZE replaced with workgroup-wide parallel reduction (256 threads, ~14 iterations each), two-level reduce (sub_group + cross-subgroup SLM).

### Optimization C: Symmetric xg_sum Skip — Applied

All `xg_sum` computation guarded with `#if HAS_ZP`. Symmetric path (`HAS_ZP=0`) skips per-element accumulation, `sub_group_reduce_add()`, and SLM write in both gate_up and down kernels.

### Optimization D: Shared Expert Transformation Fix — Applied

- **Problem**: Standard MOE matcher greedily consumed inner MOE node before shared expert matcher could match `Add(MOE, SharedExpert)` pattern
- **Solution**: Rejection predicate `!ov::is_type<ov::op::v1::Add>` on standard matcher; context-aware `replace_node` that swaps correct root; test infrastructure updated for 22-parameter shared expert layout

### Decode Performance (Qwen3.5 MoE, single-batch, 40 layers)

| Kernel | Before | After | Delta |
|--------|--------|-------|-------|
| **moe_gate_up** | 6.271 ms | 5.214 ms | **−16.9%** |
| moe_down | 2.737 ms | 2.619 ms | −4.3% |
| **MoE pipeline total** | **9.324 ms** | **8.186 ms** | **−12.2%** |

Hardware metrics (moe_gate_up): ALU0 −58.8%, ALU1 −56.3%, SBID stall −7.5pp.

---

## 2. Prefill Path: Grouped GEMM Integration

### DNNL_ARG_HINT_MAX_GROUP_SIZE

Runtime execute argument that provides a tighter upper bound on per-expert token count, reducing dispatch grid size:

- **Without hint**: Each expert dispatches `div_up(total_tokens, tile)` workgroups — massive oversubscription
- **With hint**: Each expert dispatches `div_up(max_tokens_per_expert, tile)` workgroups — up to `num_experts×` fewer

**Key design**: Hint is a pure runtime argument — no effect on primitive compilation. Single compiled primitive serves all requests regardless of routing distribution.

### Grouped GEMM Performance (Qwen3-30B-A3B)

| Input Tokens | Prefill (loop) | Prefill (grouped) | Δ | Decode (loop) | Decode (grouped) | Δ |
|-------------|---------------|-------------------|------|--------------|-------------------|------|
| 32 | 149 ms | 154 ms | +3.3% | 22.0 ms | 23.7 ms | +7.9% |
| 1024 | 500 ms | 490 ms | **-2.0%** | 24.1 ms | 25.0 ms | +3.7% |
| 8192 | 4735 ms | 4183 ms | **-11.7%** | 31.8 ms | 29.4 ms | **-7.6%** |
| 32768 | 65611 ms | 66926 ms | +2.0% | 1128 ms | 1057 ms | **-6.3%** |

**Key finding**: Grouped GEMM benefits prefill only at 8K+ tokens; decode benefits at long contexts.

### Scatter-Reduce Kernel Analysis

Phase 2 (token search) becomes significant overhead at large token counts:

| Token Count | Phase 2 fraction |
|-------------|-----------------|
| 1024 | ~4% |
| 4096 | ~14% |
| 8192 | ~24% |

Possible fix: Inverse index `token_id → (expert_slot_offset, expert_weight_index)` built during CPU mask gen.

---

## 3. Prioritized Future Optimizations

| # | Optimization | Path | Est. Impact | Complexity |
|---|-------------|------|-------------|------------|
| 1 | Fuse gate+up into single grouped matmul | Prefill | High | Medium |
| 2 | Fuse SwiGLU post-op into gate GEMM | Prefill | Medium | Low-Medium |
| 3 | Profile-guided path selection by token count | Both | Medium | Low |
| 4 | Fuse reduce into down kernel | Decode | Low-Medium | Low |
| 5 | Runtime-dim grouped GEMM | Prefill | Low-Medium | Low |
| 6 | Inverse mapping for scatter-reduce | Prefill | Low | Low |

### Rejected

| Optimization | Reason |
|-------------|--------|
| GPU mask gen | GPU serial scan slower than CPU; sync cost justified |
| Decode gate+up weight interleaving | Prefill shares weights; no separate layout possible |
| K-split across subgroups | Memory-bound; doesn't reduce traffic; adds barrier |

---

## 4. Known Constraints

1. **No inline functions in .cl**: Use macros only (file compiled 3× as sub-kernels into single program)
2. **Symmetric quant needs dummy ZP tensor**: All-zero tensor matching weight type to satisfy kernel signature
3. **Shared expert must match sparse expert INTERMEDIATE_SIZE**: Current assumption in code
4. **No gate+up fusion for prefill**: Prefill uses oneDNN matmul, not custom GEMV
5. **Shared expert prefill**: Separate sequential oneDNN calls, not integrated into grouped dispatch
6. **Grouped GEMM primitive caching**: New cache entry per distinct `total_tokens` — potential JIT overhead
