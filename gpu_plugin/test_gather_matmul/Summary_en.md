# GatherMatmul Deep Analysis & Comparison with Previous MoE Implementation

> This document is based on a file-by-file analysis of the OpenVINO GPU Plugin source code. It covers the complete execution logic of the `GatherMatmul` operator and provides a precise, code-level comparison with the legacy MoE implementation.

---

## 1. Background: Two MoE Paradigms

The OpenVINO GPU Plugin contains two MoE implementation paradigms:

### Legacy Approach (`moe_gemm` family) — Two Separate Codepaths

**Single Token (Decode)** and **Multi-Token (Prefill)** are handled by completely independent code:

| Phase | Kernel Family | Core Idea |
|:---|:---|:---|
| **Decode (single token)** | `moe_3gemm_swiglu_fuse.cl` | Single monolithic fused kernel; Gate/Up/SwiGLU/Down all execute in registers — zero intermediate Global Memory I/O |
| **Prefill (multi-token)** | `moe_mask_gen` → `moe_gather_ref` → `moe_gemm` → `moe_scatter_reduction` | A chain of loosely-coupled kernels; GPU-parallel prefix scan for token grouping, followed by Grouped GEMM with shared weights |

### New Approach (`GatherMatmul` family) — Unified "LEGO Blocks"

One general-purpose `GatherMatmul` operator expresses "index-based lookup + batched matrix multiply". The backend automatically selects the execution path based on `n_tokens`:

```
n_tokens == 1       → regular_micro_single_token   (gather_matmul.cl)
1 < n_tokens ≤ 16  → regular_micro_multi_tokens    (gather_matmul.cl, per-token dispatch)
n_tokens > 16       → 3-stage batched pipeline      (Sort → Gather → gather_matmul_batched.cl)
```

---

## 2. GatherMatmul Complete Execution Flow

### 2.1 C++ Dispatch Logic (`gather_matmul.cpp`)

```
GatherMatmulOCLImpl::execute()
     │
     ├─ is_prefill_stage()? → n_tokens > 1?
     │       │
     │       ├─ Yes (Prefill)
     │       │       └─ use_batched_prefill()? → n_tokens > BATCHED_PREFILL_THRESHOLD(16)?
     │       │               ├─ Yes → [Sort] → [Gather] → [batched_gemm] 3-stage pipeline
     │       │               └─ No  → regular_micro_multi_tokens (per-token dispatch)
     │       │
     │       └─ No (Decode, n_tokens == 1)
     │               └─ regular_micro_single_token
```

### 2.2 Seven Internal Buffers (Intermediate Storage)

In the batched path, kernels communicate via these pre-allocated internal buffers:

| Index | Name | Type | Size | Semantics |
|:---:|:---|:---:|:---|:---|
| 0 | `GATHERED_A` | f16 | `n_tokens × top_k × K` | Contiguous activations reordered by expert group |
| 1 | `GROUP_EXPERT_IDS` | i32 | `max_groups` | Expert ID for each group |
| 2 | `GROUP_SLOT_IDS` | i32 | `max_groups` | Which top_k slot each group belongs to |
| 3 | `GROUP_OFFSETS` | i32 | `max_groups` | Starting row offset of each group in `GATHERED_A` |
| 4 | `GROUP_SIZES` | i32 | `max_groups` | Number of tokens in each group |
| 5 | `TOKEN_MAP` | i32 | `max(n_tokens×top_k, max_groups)` | Sorted position → original token index mapping |
| 6 | `NUM_GROUPS` | i32 | 1 | Number of active groups |

> `max_groups = n_all_experts × top_k` — one potential group per (Expert, Slot) combination.

---

## 3. Per-Kernel Detailed Analysis

### Kernel A: `gather_matmul.cl` → `batch_gather_matmul` (Per-token Path)

**Scope**: `n_tokens ≤ 16` or Decode phase (`n_tokens == 1`)

#### Inputs
| Parameter | Shape | Description |
|:---|:---|:---|
| `input_ptr` (A) | `[n_act, n_tokens, K]` | Token activations. `n_act=1` (broadcast) for the first GatherMatmul, `n_act=top_k` for subsequent ones |
| `weight_ptr` (B) | `[n_all_experts, N, K]` | All expert weights (K dimension transposed) |
| `indices` | `[n_tokens, top_k]` | Per-token expert routing indices |
| `m`, `k` | scalar | Output feature dimension N and reduction dimension K |

#### Dispatch Strategy
```
global[0] = ceil(M / wg_tile_m)    ← output feature tiles
global[1] = 1                       ← 1 token per workgroup
global[2] = n_tokens × top_k       ← all (token, slot) pairs flattened
```

Decoded from `flat_idx = get_group_id(2)`:
```c
token_idx   = flat_idx / top_k;
expert_slot = flat_idx % top_k;
expert_id   = indices[token_idx * top_k + expert_slot];  // routing table lookup
```

#### Core Computation
```c
input_ptr  += a_slot * n_tokens * K + token_idx * K;      // current token row
weight_ptr += expert_id * EXPERT_STRIDE;                    // current expert weights
out_ptr    += expert_slot * n_tokens * M + token_idx * M;  // output position

// cur_n_tokens=1 → essentially a GEMV
c_tile = ugemm_gm(weight_ptr, input_ptr, M, 1, K, ...);
tile_store(c_tile, out_ptr);
```

#### Output
| Shape | Description |
|:---|:---|
| `[top_k, n_tokens, N]` | MLP projection result for every `(slot, token)` pair, written sequentially |

#### Key Concern
If multiple tokens are routed to the same expert, each workgroup **independently reloads** that expert's weights — causing severe memory bandwidth thrashing.

---

### Kernel B1: `gathermatmul_sort.cl` → `bgm_sort` (Sort Stage)

**Scope**: Batched path when `n_tokens > 16`

#### Inputs
| Parameter | Shape | Description |
|:---|:---|:---|
| `indices` (INPUT0) | `[n_tokens, top_k]` | Per-token expert routing indices |

#### Dispatch
```
global = {1, 1, 1},  local = {1, 1, 1}   ← single thread, fully sequential
```

#### Execution Logic (3 passes)

**Pass 1 — Histogram**: Uses `token_map` as scratch to count how many tokens fall into each `(slot, expert)` bin.

**Pass 2 — Compaction + Prefix Sum**: Skips empty bins; assigns compact group IDs to non-empty bins; computes `group_offsets` via prefix sum.

**Pass 3 — Token Scatter**: For each group, scans all tokens to find those belonging to that `(slot, expert)`, and writes them into `token_map`.

#### Outputs (written to internal buffers)
`GROUP_EXPERT_IDS`, `GROUP_SLOT_IDS`, `GROUP_OFFSETS`, `GROUP_SIZES`, `TOKEN_MAP`, `NUM_GROUPS`

#### Key Concern
The entire kernel is **fully sequential** — it cannot exploit the GPU's massive parallelism at all. Compare with the legacy `moe_mask_gen` which uses GPU hardware parallel prefix scan.

---

### Kernel B2: `gathermatmul_gather.cl` → `bgm_gather` (Gather Stage)

#### Inputs
| Parameter | Shape | Description |
|:---|:---|:---|
| `input_ptr` (A) | `[n_act, n_tokens, K]` | Original token activations |
| `token_map` | `[n_tokens × top_k]` | Position-to-token mapping from Sort output |
| `group_slot_ids` | `[max_groups]` | Slot index per group |
| `group_offsets` | `[max_groups]` | Starting offset in `GATHERED_A` per group |
| `group_sizes` | `[max_groups]` | Token count per group |
| `num_groups` | `[1]` | Number of active groups |

#### Dispatch (3D, highly parallel)
```
global[2] = max_groups                    ← one z-slice per group
global[1] = max_tokens_per_group          ← y-dim per token within group
global[0] = ceil(K / COPY_BLOCK=8)       ← K dimension parallel copy
```

#### Core Computation (128-bit vectorized copy)
```c
int orig_token = token_map[group_offsets[group_id] + token_in_group];
int a_slot     = min(slot, n_act - 1);   // clamp for broadcast case

// Source: A[a_slot, orig_token, :] — random (scattered) access
src = input_ptr + (a_slot * n_tokens + orig_token) * K;
// Dest: GATHERED_A[offset + token_in_group, :] — contiguous write
dst = gathered_ptr + (group_offsets[group_id] + token_in_group) * K;

half8 val = vload8(0, src + k_offset);  // 128-bit vectorized read
vstore8(val, 0, dst + k_offset);        // 128-bit vectorized write
```

#### Output
| Buffer | Shape | Content |
|:---|:---|:---|
| `GATHERED_A` | `[n_tokens × top_k, K]` | Contiguous activations laid out by group: all tokens of Group 0 first, then Group 1, etc. |

---

### Kernel B3: `gather_matmul_batched.cl` → `gather_matmul_batched` (Batched GEMM Stage)

#### Inputs
| Parameter | Shape | Description |
|:---|:---|:---|
| `gathered_input_ptr` | `[n_tokens×top_k, K]` | Contiguous activations from Gather |
| `weight_ptr` (B) | `[n_all_experts, N, K]` | All expert weights (transposed) |
| `group_expert_ids` | `[max_groups]` | Which expert's weights to use per group |
| `group_slot_ids` | `[max_groups]` | Which slot each group belongs to |
| `group_offsets` | `[max_groups]` | Starting row in the contiguous buffer per group |
| `group_sizes` | `[max_groups]` | Token count per group |
| `token_map` | `[n_tokens×top_k]` | Used during scattered store to restore original token order |
| `num_groups` | `[1]` | Number of active groups |

#### Dispatch
```
global[0] = ceil(M / wg_tile_m)          ← output feature tiles
global[1] = ceil(n_tokens / wg_tile_n)   ← token tiles (early-exit per group_size)
global[2] = max_groups                    ← one z-slice per group
```

#### Core Computation (two steps)

**Step 1 — Batched GEMM (shared weight load across all tokens in a group)**:
```c
int expert_id    = group_expert_ids[group_id];
int cur_n_tokens = group_sizes[group_id];          // tokens in this group

// ALL tokens in this group share a single weight load — this is the key optimization!
input_ptr  = gathered_input_ptr + group_offsets[group_id] * k;
weight_ptr += expert_id * EXPERT_STRIDE;

// Standard GEMM with n = cur_n_tokens (can be tens or hundreds)
c_tile = ugemm_gm(weight_ptr, input_ptr, M, cur_n_tokens, K, ...);
```

**Step 2 — Scattered Store (reverse token_map to write back to original positions)**:
```c
// Cannot write sequentially! Must scatter based on each row's original token position.
for (int j = 0; j < cur_n_tokens; j++) {
    int orig_token = token_map[offset + sg_j0 + j];     // reverse lookup
    row_ptr = out_ptr + slot * n_tokens * M + orig_token * M;
    row_ptr[sg_i0 + i] = c_tile_half.x[...];             // scattered write
}
```

#### Output
| Shape | Description |
|:---|:---|
| `[top_k, N, n_tokens]` | Results written back in original token order |

---

## 4. End-to-End Data Flow Example

Using `n_tokens=32, top_k=2, n_experts=8, K=4096, N=2048`:

```
Original Inputs
  A: [1, 32, 4096]   indices: [32, 2]   B: [8, 2048, 4096]

── Kernel B1: bgm_sort ────────────────────────────────────────────────
  indices: [32,2] → count tokens for 16 (slot,expert) bins
  Output: GROUP_EXPERT_IDS, GROUP_SIZES, GROUP_OFFSETS, TOKEN_MAP, NUM_GROUPS
  Example: Group0=(slot=0, expert=3, size=5, offset=0)
           Group1=(slot=0, expert=7, size=4, offset=5) ...
           NUM_GROUPS = 12

── Kernel B2: bgm_gather ──────────────────────────────────────────────
  TOKEN_MAP[0..4] = [2, 8, 11, 19, 27]  (original token indices for Group0)
  Copy A[0,2,:], A[0,8,:], A[0,11,:], A[0,19,:], A[0,27,:] into
       GATHERED_A[0,:], [1,:],  [2,:],  [3,:],  [4,:]
  Repeat for all groups → GATHERED_A: [64, 4096]

── Kernel B3: gather_matmul_batched ───────────────────────────────────
  Group0: weight_ptr = B[3,:,:]   (Expert 3 weights — loaded ONCE!)
          GEMM: [2048,4096] × [4096,5] → C[2048,5]   (5 tokens share Expert 3)
          Scatter: C[:,0] → out[0,2,:], C[:,1] → out[0,8,:] ...

  Group1: weight_ptr = B[7,:,:]   (Expert 7 weights — loaded ONCE!)
          GEMM: [2048,4096] × [4096,4] → C[2048,4]
          Scatter: C[:,i] → out[0, orig_token, :] ...

Final Output: out[2, 32, 2048]  (top_k=2 slots, 32 tokens, 2048 output features)
```

---

## 5. Code-Level Comparison: Key Differences

### 5.1 Sort/Mask Kernel: Legacy has far superior parallel efficiency

**Legacy `moe_mask_gen.cl`** (GPU wavefront-level parallel prefix scan):
```c
const size_t expert_id = get_local_id(0);  // n_experts threads run in parallel!
// ... each thread counts tokens for its own expert ...
int tokens_per_expert_iter = work_group_scan_exclusive_add(num_tokens_per_curr_expert);
int experts_id_iter        = work_group_scan_exclusive_add(is_used);
// ↑ Hardware instruction: completes full group prefix sum in ~O(1) time
```

**New `gathermatmul_sort.cl`** (purely sequential):
```c
// Dispatch: single workgroup (1,1,1) global, (1,1,1) local
// All work done sequentially — n_tokens * top_k is small enough.
wgs.global = {1, 1, 1};   // completely sequential!
for (int s = 0; s < top_k; s++)
    for (int e = 0; e < n_all_experts; e++)
        // O(n_tokens × top_k × n_experts) sequential loop
```

### 5.2 SwiGLU Activation: Legacy stays in registers, new approach goes through Global Memory

**Legacy `moe_gemm.cl`** (fused inside the GEMM kernel — zero intermediate memory traffic):
```c
#ifdef POST_PROC_SILU_MUL
    // c_tile_half is a register variable — entire SwiGLU runs in registers
    float gate_val = post_op_row[i];                              // read Gate branch output
    float up_val   = c_tile_half.x[reg_idx_i][reg_idx_j];        // Up branch is in registers
    float res      = gate_val * (up_val / (1.0f + native_exp(-up_val)));  // SiLU(Up) × Gate
    c_tile_half.x[reg_idx_i][reg_idx_j] = res;                    // stays in registers!
#endif
```

**New approach** (SwiGLU is a separate graph OP):
```c
// No POST_PROC_SILU_MUL — activation functions are separate ops in the GatherMatmul graph.
tile_store(c_tile_half, out_ptr, m, cur_n_tokens, sg_i0, sg_j0);
// ↑ Gate result → write to Global Memory → SwiGLU OP reads from Global Memory
//   → compute → write to Global Memory → next GatherMatmul reads from Global Memory
```

### 5.3 Scatter + Reduce: Legacy performs weighted accumulation in one kernel

**Legacy `moe_scatter_reduction_ref.cl`** (multiply router scores + multi-expert accumulate in one kernel):
```c
for (int e_iter = 0; e_iter < ACTIVE_EXPERTS; ++e_iter) {
    INPUT2_TYPE weight = expert_weights[token_id * ACTIVE_EXPERTS + e_iter];
    // ...
    if (e_iter == 0)
        output[out_pos + h] = input[in_pos + h] * weight;   // multiply router score
    else
        output[out_pos + h] += input[in_pos + h] * weight;  // accumulate experts
}
```

**New `gather_matmul_batched.cl`** (positional scatter only — no weighting):
```c
int orig_token = token_map[offset + sg_j0 + j];
row_ptr = out_ptr + slot * n_tokens * m + orig_token * m;
row_ptr[sg_i0 + i] = c_tile_half.x[...];  // raw scatter — no weight multiplication
```

---

## 6. Comprehensive Pros and Cons

| Dimension | Legacy (`moe_gemm` family) | New GatherMatmul | Winner |
|:---|:---|:---|:---:|
| **Sort parallelism** | GPU parallel prefix scan (`work_group_scan_exclusive_add`), near O(1) | Single-thread sequential loop, O(n × top_k × n_exp) | **Legacy** |
| **Gather vectorization** | Generic `VLOAD/VSTORE`, dynamic vector width | Hard-coded `vload8/vstore8` (128-bit), K+Token dual-dim parallel | **New** |
| **SwiGLU fusion** | `POST_PROC_SILU_MUL` runs in registers — zero intermediate Global Memory I/O | SwiGLU is a separate OP — 2 full Global Memory round-trips | **Legacy** |
| **GEMM core computation** | Same `ugemm_micro` microkernel (gemmstone) | Same `ugemm_micro` microkernel (gemmstone) | **Tie** |
| **Scatter + weighted reduce** | One kernel: multiply router score + multi-expert accumulation | Scatter only; weighting requires additional downstream OPs | **Legacy** |
| **Code maintainability** | Deeply coupled; adding new data types requires extensive MoE-specific changes | Standard GEMM interface; inherits all GEMM optimizations (int4, fp8) automatically | **New** |
| **Single/multi-token unification** | Two completely separate codepaths; graph-level branching required | One operator; backend auto-selects by n_tokens; unified graph representation | **New** |
| **Operator generality** | Only handles fixed 3-GEMM SwiGLU MoE structure | Composable with any activation; usable for non-MoE grouped projection | **New** |

---

## 7. Performance Analysis by Inference Phase

### 7.1 Prefill (First Token Latency)

| Approach | Performance | Reason |
|:---|:---:|:---|
| Legacy (multi-token `micro_gemm` path) | ✅ Good | GPU-parallel prefix scan for grouping; Grouped GEMM amortizes weight loading |
| New (GatherMatmul batched) | ✅ Good (comparable) | Inherits the same Grouped GEMM concept; sequential Sort is a latent bottleneck at very large n_tokens |

**Conclusion: Roughly equivalent at typical Prefill sizes. The sequential Sort becomes a measurable bottleneck at very large n_tokens.**

### 7.2 Decode (Token Rate / Second Token Latency)

| Approach | Performance | Reason |
|:---|:---:|:---|
| Legacy (`moe_3gemm_swiglu_fuse` Megakernel) | ✅✅ Excellent | Single token → single monolithic fused kernel; Gate/Up/SwiGLU/Down all in registers; zero intermediate Global Memory I/O; single kernel launch |
| New (GatherMatmul per-token) | ⚠️ Fair | SwiGLU is forcibly separated — intermediate Global Memory round-trips; multiple kernel launches overhead |

**Conclusion: New approach has a measurable latency regression for single-token Decode.**

### 7.3 Selection Recommendation

The new GatherMatmul approach is better suited for **modern serving scenarios** (Continuous Batching, high concurrency, large Prefill), as well as for **long-term maintainability and future hardware extension**. The legacy approach's single-token Decode advantage is only relevant in strict BS=1 scenarios, which are rarely dominant in production serving deployments.

---

## 8. Concrete Optimization Recommendations

### Optimization 1 (High Priority): Parallelize `bgm_sort` with GPU prefix scan

**Current code** (`gathermatmul_sort_gen.cpp`):
```cpp
// Single workgroup — all work done sequentially
wgs.global = {1, 1, 1};
wgs.local  = {1, 1, 1};
```

**Proposed change**: Mirror the approach of legacy `moe_mask_gen` — change dispatch to `{n_all_experts × top_k, 1, 1}`, one thread per `(slot, expert)` bin, using `work_group_scan_exclusive_add` for hardware-accelerated prefix sum.

**Expected gain**: Sort latency reduced by 10x+ (from O(n_tokens × top_k × n_experts) serial to O(n_tokens) parallel).

---

### Optimization 2 (High Priority): Add optional SwiGLU post-fusion to `gather_matmul_batched`

**Root cause** (comment in `gather_matmul.cl`):
```c
// No POST_PROC_SILU_MUL — activation functions are separate ops in the GatherMatmul graph.
```

**Proposed change**: Add a `POST_PROC_SILU_MUL` compile-time switch to `gather_matmul_batched.cl`, mirroring `moe_gemm.cl`. Apply the SwiGLU computation in registers before the Scattered Store. Extend `moe_op_fusion` graph transforms to detect `GatherMatmul + SwiGLU` patterns and set the fusion flag.

**Expected gain**: Eliminates 2 full Global Memory round-trips of size `[top_k × n_tokens × intermediate_size]`. For large models (e.g., Deepseek-R1: hidden=5120), this translates to hundreds of MB saved per MoE layer.

---

### Optimization 3 (Medium Priority): Fused Gather-GEMM to eliminate `GATHERED_A` buffer

**Problem**: `GATHERED_A` buffer size = `n_tokens × top_k × K`. At `n_tokens=2048, top_k=4, K=5120` this is ~84 MB — fully written then fully read every inference step.

**Proposed change**: Merge `bgm_gather` and `gather_matmul_batched` into a single kernel. In the GEMM inner loop, look up the original token address directly via `token_map` instead of reading from `GATHERED_A`.

**Trade-off**: Activation access becomes non-coalesced (random), which hurts L3 cache hit rate. For large K, the contiguous layout of `GATHERED_A` is more cache-friendly. Actual benefit needs profiling to confirm.

---

### Optimization 4 (Medium Priority): Dynamic `COPY_BLOCK` vectorization width

**Current** (`gathermatmul_gather.cl`):
```c
#define COPY_BLOCK 8  // hardcoded 128-bit vector for fp16
half8 val = vload8(0, src + k_offset);
```

**Proposed change**: In `gathermatmul_gather_gen.cpp`, dynamically select vector width based on K size and hardware capabilities (analogous to legacy `moe_gather_ref.cpp`'s `GetBlockSize`). Also explore `intel_sub_group_block_read/write` for better cache utilization on coalesced accesses.

---

### Optimization 5 (Medium Priority): Adaptive `BATCHED_PREFILL_THRESHOLD`

**Current** (`gather_matmul.cpp`):
```cpp
static constexpr int32_t BATCHED_PREFILL_THRESHOLD = 16;  // hardcoded
```

**Proposed change**: Compute dynamically based on average tokens per expert reaching the GEMM vs GEMV break-even point (typically ≥ 4-8 tokens/group):
```cpp
const int avg_tokens_per_expert = (rtp->n_tokens * rtp->top_k) / n_all_experts;
return avg_tokens_per_expert >= 4;  // replaces fixed threshold of 16
```

---

### Optimization Summary

| Optimization | Phase | Expected Gain | Difficulty |
|:---|:---:|:---:|:---:|
| **Sort parallelization** (P1) | Prefill (large n_tokens) | 10x+ Sort latency reduction | Medium |
| **SwiGLU post-fusion** (P2) | Decode + Prefill | Eliminate 2 Global Memory round-trips; remove 1 kernel launch | High (requires graph-level changes) |
| **Fused Gather-GEMM** (P3) | Prefill | Eliminate `GATHERED_A` buffer memory I/O | High (requires GEMM kernel restructure) |
| **Dynamic vector width** (P4) | Prefill Gather stage | ~1.5-2x Gather bandwidth improvement | Low |
| **Adaptive threshold** (P5) | All scenarios | Avoid wasted Sort/Gather overhead for small n_tokens | Low |

**Overall recommendation**: Prioritize P1 (Sort parallelization) to eliminate the serial bottleneck, then P2 (SwiGLU fusion) to eliminate unnecessary intermediate memory traffic. Together, these two optimizations yield the most significant throughput improvement for large Prefill workloads.

---

## 9. Key File Index

| File | Purpose |
|:---|:---|
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul/gather_matmul.cpp` | C++ dispatch logic and kernel orchestration |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gathermatmul_sort.cl` | Sort kernel (sequential — determines group structure) |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gathermatmul_gather.cl` | Gather kernel (copies activations into contiguous buffer) |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul.cl` | Per-token GEMV kernel (Decode / small Batch) |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul_batched.cl` | Batched GEMM + Scattered Store kernel (large Batch Prefill) |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul/gather_matmul_gen_micro.cpp` | Micro GEMM generator (gemmstone-based) |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_mask_gen.cl` | Legacy Sort kernel (GPU parallel prefix scan — reference for P1 optimization) |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_gemm.cl` | Legacy GEMM kernel (with `POST_PROC_SILU_MUL` fused activation — reference for P2 optimization) |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_fuse.cl` | Legacy single-token monolithic fused kernel (performance reference for Decode) |
| `src/common/transformations/include/transformations/common_optimizations/moe_op_fusion.hpp` | Graph transform: GatherMatmul sequence → MoE OP fusion (extension point for SwiGLU post-fusion) |
