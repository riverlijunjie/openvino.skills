# DNNL_ARG_HINT_MAX_GROUP_SIZE Integration for Grouped GEMM

## oneDNN Changes Analysis

The oneDNN branch `rls-v3.12` adds an optional runtime execute argument `DNNL_ARG_HINT_MAX_GROUP_SIZE` for grouped matmul dispatch optimization. Key commits:

### API Layer (`4f31488c29`)
- **Constant**: `DNNL_ARG_HINT_MAX_GROUP_SIZE = 384` defined in `dnnl_types.h` (gated by `DNNL_EXPERIMENTAL_GROUPED_MEMORY`)
- **`grouped()` signature**: no `max_variable_dim` parameter. Signature: `grouped(dims, data_type, variable_dim_idx, group_count, offsets_dt)`
- The hint is a pure runtime execute argument, not part of the memory descriptor

### Common Layer (`a0460febab`)
- `DNNL_ARG_HINT_MAX_GROUP_SIZE` added to the set of accepted arguments in `matmul_pd_t::arg_usage()` — accepted as input when src is a grouped descriptor, unused otherwise
- `primitive_exec_types.cpp` updated to allow the hint arg as a host-scalar input

### GPU Micro GEMM Kernel (`98e97b7f1d`)
- In `grouped_micro_gemm.cpp` `execute()`, the kernel reads the hint at runtime:
  ```cpp
  const int32_t *max_var_dim = CTX_IN_MEM(const int32_t *, DNNL_ARG_HINT_MAX_GROUP_SIZE);
  dim_t m_dispatch = m_all;
  if (max_var_dim && *max_var_dim > 0 && *max_var_dim <= m_all)
      m_dispatch = *max_var_dim;
  ```
- **Without** hint (or hint=0): each expert dispatches `div_up(total_tokens, tile)` workgroups — massive oversubscription
- **With** hint: each expert dispatches `div_up(m_dispatch, tile)` workgroups — up to `num_experts`× fewer workgroups

### Host Scalar Memory Pattern
- The hint is passed as a host_scalar memory:
  ```cpp
  auto hint_md = dnnl::memory::desc::host_scalar(dnnl::memory::data_type::s32);
  dnnl::memory hint_mem(hint_md, static_cast<int32_t>(max_tokens_per_expert));
  // In execute args:
  {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}
  ```

## OpenVINO Integration Changes

### What was changed

#### `moe_3gemm_swiglu_opt.cpp`

- **Primitive creation (`get_grouped_kernel`)**: creates grouped descriptors without any dispatch hint. Cache key is `total_tokens` only (single `int`). One compiled primitive per `total_tokens` value.

- **Runtime hint passing**: Before executing the three grouped GEMMs (gate, up, down), a host_scalar memory is created with the actual `max_tokens_per_expert` value and passed as `DNNL_ARG_HINT_MAX_GROUP_SIZE` in each execute call:
  ```cpp
  auto hint_md = dnnl::memory::desc::host_scalar(dnnl::memory::data_type::s32);
  dnnl::memory hint_mem(hint_md, static_cast<int32_t>(max_tokens_per_expert));
  // Added to all three GEMM execute args:
  {DNNL_ARG_HINT_MAX_GROUP_SIZE, hint_mem}
  ```

- **Removed**: `bucket_max_variable_dim()`, adaptive bucketing, `PairHash`, `max_variable_dim` in grouped() calls, `pair<int,int>` cache key.

### Design rationale: runtime execute arg vs. compile-time parameter

`DNNL_ARG_HINT_MAX_GROUP_SIZE` controls how many GPU workgroups are launched per expert. The key advantage over the old `max_variable_dim` approach:

| Approach | Recompilations | Dispatch quality |
|----------|---------------|-----------------|
| Old: `max_variable_dim` baked into pd | Every change triggers recompile | Optimal but costly |
| **New: `DNNL_ARG_HINT_MAX_GROUP_SIZE` at execute** | **Zero** | **Optimal** |

The hint is a pure runtime argument — it has no effect on primitive compilation. The same compiled primitive serves all requests regardless of routing distribution.

### Safety

- The kernel guards: `*max_var_dim > 0 && *max_var_dim <= m_all` — values outside this range fall back to `m_all` (full dispatch)
- `max_tokens_per_expert <= total_gathered_tokens` always holds by definition
- The kernel's `if (wg_j0 >= m) return;` guard handles any over-dispatched workgroups

### Performance Impact

**Compilation**: Exactly one primitive set (gate/up/down) per distinct `total_tokens`. Zero additional compilations from routing variation.

**Dispatch efficiency**: Exact `max_tokens_per_expert` per request — no over-dispatch.
- Balanced routing, 1024 tokens, 128 experts: `max_per_expert ≈ 15` → ~1 workgroup per expert vs. ~32 without
- Imbalanced routing: still exact — each request gets its actual value via the runtime hint

---

# MOE Optimization Analysis (Prefill + Decode)

## Benchmark Environment

- **Model**: Qwen3-30B-A3B (128 experts, top_k=2, hidden=2048, inter=2048, u4 weights)
- **GPU**: Intel discrete GPU (Xe2 architecture)
- **Build**: OpenVINO 2026.2.0 (`river/grouped_gemm_integration` branch)

## Current Performance Numbers (Qwen3-30B-A3B)

### Log 1: `MOE_USE_GROUPED_GEMM_PREFILL=0` (per-expert oneDNN loop)

| Input Tokens | 1st Token (Prefill, ms) | 2nd Token (Decode, ms/tok) |
|-------------|------------------------|---------------------------|
| 32          | 156.59                 | 22.59                     |
| 1024        | 490.16                 | 24.18                     |
| 2048        | 822.69                 | 25.56                     |

### Log 2: `MOE_USE_GROUPED_GEMM_PREFILL=0` vs `=1` side-by-side

| Input Tokens | Prefill (loop) ms | Prefill (grouped) ms | Prefill Δ | Decode (loop) ms | Decode (grouped) ms | Decode Δ |
|-------------|------------------|---------------------|-----------|-----------------|---------------------|----------|
| 32          | 149.37           | 154.34              | +3.3%     | 21.96           | 23.70               | +7.9%    |
| 1024        | 500.03           | 490.16              | **-2.0%** | 24.07           | 24.96               | +3.7%    |
| 2048        | 822.69           | 840.89              | +2.2%     | 25.56           | 25.92               | +1.4%    |
| 4096        | 1682.36          | 1725.48             | +2.6%     | 25.64           | 26.01               | +1.4%    |
| 8192        | 4734.63          | 4182.80             | **-11.7%**| 31.79           | 29.38               | **-7.6%**|
| 32768       | 65610.79         | 66926.02            | +2.0%     | 1128.25         | 1056.69             | **-6.3%**|

**Key observations**:
- Grouped GEMM shows clear prefill benefit **only** at 8K tokens (-11.7%); at other lengths it's flat or slightly worse.
- Decode latency is generally similar; grouped GEMM helps slightly at long contexts (8K/32K).
- At 32K tokens, decode latency explodes (1056-1128 ms/tok) — this is dominated by attention/KV, not MOE.

---

## Part 1: Prefill Path — Optimization Opportunities

### 1.1 CPU Mask Generation — Trade-off Analysis

**Current behavior**: Both the oneDNN-loop path and grouped GEMM path call `get_expert_mask_from_gpu()` which does `topk_event->wait()` followed by `mem->copy_to(stream, ...)` — a full GPU→CPU sync to read topk_ids, then builds the expert mask on CPU, then uploads the result back to GPU.

**Why GPU mask gen is NOT better**: The `moe_mask_gen.cl` kernel has an inherently serial inner loop — each work-item (one per expert) iterates `num_tokens × NUM_EXPERTS_PER_TOKEN` entries linearly to find its tokens. With 128 experts, only 128 threads run, and each scans up to `tokens × topk` entries. This is essentially a hash/lookup operation that cannot be parallelized effectively across the K-dimension / SIMD lanes within each work-item. The GPU's advantage (massive parallelism) doesn't help here; a single CPU core can iterate through this data much faster due to better branch prediction, speculative execution, and cache locality on sequential reads. The CPU sync overhead is smaller than the penalty of GPU under-utilization.

**Status**: ~~Optimization candidate~~ **Not viable** — CPU mask gen is the better choice. The CPU→GPU sync cost is justified.

### 1.2 Fuse Gate + Up GEMM into a Single Grouped Matmul

**Current behavior** (grouped GEMM path): Gate and Up are two separate `grouped_matmul.execute()` calls. They share the same source tensor (`scratch.x`) and the same weight layout/quantization.

**Optimization**: Fuse gate+up weights into a single `[E, hidden, 2*inter]` matmul. The output would be `[total, 2*inter]` which is then split for SiLU activation. This halves the number of GPU kernel launches (3→2 grouped GEMMs) and improves GPU occupancy by doubling the work per dispatch.

**Complexity**: Medium — requires changing weight layout in `convert_moe_to_compressed.cpp` and adjusting the grouped memory descriptor construction.

### 1.3 Fuse SwiGLU into Gate GEMM Post-Op

**Current behavior**: `prefill_swiglu` is a separate OCL kernel `swiglu_ref` dispatched after both gate and up GEMMs complete. It reads `scratch.up` and `scratch.gate`, computes `gate = up * silu(gate)`, writes back.

**Optimization**: The micro_gemm path already does this with `POST_PROC_SILU_MUL` — the gate micro-kernel fuses `silu(gate) * up` in registers, reading the up output from memory. For grouped GEMM, using OneDNN post-ops (binary multiply + eltwise SiLU) on the gate matmul could eliminate this kernel entirely.

**Expected benefit**: Saves one kernel launch + one round-trip read/write of `[total_tokens, inter_size]` buffer.

### 1.4 Grouped Matmul Primitive Caching Over-Compiles

**Current behavior**: `get_grouped_kernel()` caches by `total_tokens`. Since `total_tokens = token_num * top_k`, and `token_num` varies with every prompt, there's a new cache entry (and potentially a new JIT compilation) for every distinct prompt length.

**Optimization**: Use `DNNL_RUNTIME_DIM_VAL` for the M-dimension in the grouped descriptor, similar to how the per-expert oneDNN loop does it. This would create a single compiled primitive that works for any token count, with only the offsets buffer and hint changing at runtime.

**Note**: This depends on OneDNN grouped GEMM support for runtime M — may not be available yet. If not, bucketing to powers-of-2 would reduce cache pollution significantly.

### 1.5 Scatter-Reduce Kernel: Detailed Performance Impact Analysis

**Current behavior** (`moe_scatter_reduction_opt.cl`): For each output token, the kernel needs to locate where its data resides in the expert-sorted flat buffer. The algorithm has three phases:

**Phase 1 — Expert-ID lookup** (lines 43-55): Each of `ACTIVE_EXPERTS` (=topk=2) threads performs a linear scan over `actual_used_expert_num` (up to 128) activated experts to find the mapping `expert_id → position_in_compact_list`. At topk=2, this is 2 threads × up to 128 iterations = **~256 comparisons**. Very fast.

**Phase 2 — Token search** (lines 62-84): For each matched expert, search its token list (`tokens_per_expert[offset..offset+token_len]`) to find the current `token_group_id`. Two sub-paths:
- `token_len < 256`: Single-thread linear scan (threads_index==0 only; rest idle)
- `token_len >= 256`: Parallel scan across all workgroup threads

For Qwen3-30B with 128 experts, balanced routing:
- 1024 tokens × topk=2: `total_gathered = 2048`, average `token_len = 2048/128 = 16` per expert
- 8192 tokens × topk=2: `total_gathered = 16384`, average `token_len = 128` per expert

Both hit the `token_len < 256` single-thread branch. **Impact**: at 8K tokens, 2 experts × ~128 sequential comparisons per token = ~256 comparisons/token, single-threaded. The remaining workgroup threads are idle.

**Phase 3 — Accumulation** (lines 97-111): Vectorized load + FMA over `HIDDEN_SIZE`. This is the bulk of work, and it's well-parallelized: ~512 threads (for hidden=2048, VEC_BLK_SIZE=4) each do `BATCHES_PER_THREAD` loads * `ACTIVE_EXPERTS` iterations. Memory traffic: `topk × hidden_size × 2 bytes (f16)` per token = 2×2048×2 = 8KB read + 2048×2 = 4KB write.

**Quantitative estimate for Phase 2 overhead**:

| Token Count | token_len/expert | Phase 2 comparisons | Phase 2 time (est.) | Phase 3 time (est.) | Phase 2 fraction |
|-------------|-----------------|---------------------|--------------------|--------------------|-----------------|
| 1024        | 16              | 2×16=32             | ~0.02 µs           | ~0.5 µs            | ~4%             |
| 4096        | 64              | 2×64=128            | ~0.08 µs           | ~0.5 µs            | ~14%            |
| 8192        | 128             | 2×128=256           | ~0.16 µs           | ~0.5 µs            | ~24%            |
| 32768       | 512             | 2×512=1024 (parallel)| ~0.3 µs           | ~0.5 µs            | ~37%            |

**Assessment**: At small-to-medium token counts (≤4K), Phase 2 is a minor overhead (<14%). At large token counts (8K+), it becomes significant (~24-37%). An inverse mapping would eliminate Phase 2 entirely, but the absolute time savings per-token are <0.2µs — across 8K tokens and 48 layers, that's `0.2µs × 8192 × 48 ≈ 79ms`, which would be notable for 8K prefill latency (~4200ms), roughly **~1.9% improvement**.

**Possible optimization**: Build an inverse index `token_id → (expert_slot_offset, expert_weight_index)` during CPU mask gen. This is trivially cheap on CPU (one extra pass). Store it as an additional internal buffer of shape `[token_num, topk, 2]`. Then Phase 2 becomes a single direct lookup: `input_offset = inverse_map[token_group_id * topk + i]`.

---

## Part 2: Decode Path (Single-Token) — Optimization Opportunities

### 2.0 Roofline Analysis: Decode is Memory-Bandwidth Bound

Before evaluating compute optimizations, we need to understand the bottleneck.

**Assumed hardware**: Intel Arc B580 (Xe2), ~456 GB/s memory bandwidth, ~109 TFLOPS FP16 DPAS.

**Decode = GEMV (M=1)**: For a single-token MoE layer, the MOE part consists of:
- **gate_up**: 2 experts × (gate + up) = 4 GEMVs of `[1, 2048] × [2048, 2048]`
- **down**: 2 experts × 1 GEMV of `[1, 2048] × [2048, 2048]`
- Total: 6 GEMVs per MoE layer (not counting shared expert)

Per GEMV with u4 weights:
- **Weight read**: `K × N / 2` bytes = `2048 × 2048 / 2` = **2 MB**
- **Scale/ZP read**: `(K/group_size) × N × 2` bytes (f16 scale) + `(K/group_size) × N / 2` bytes (u4 ZP) ≈ **~32-64 KB** (for group_size=128)
- **Input (x)**: 2048 × 2 bytes = **4 KB** (in SLM, read once)
- **Output**: N × 2 bytes = **4 KB**
- **Total traffic per GEMV**: ~2 MB (weight-dominated)

Per MoE layer (6 GEMVs, not counting shared expert):
- **Total traffic**: 6 × 2 MB = **12 MB**
- **Total FLOPs**: 6 × 2 × 2048 × 2048 = **~50 MFLOP**

**Arithmetic intensity**: 50 MFLOP / 12 MB ≈ **4.2 FLOP/byte**

**Roofline crossover point**: `109 TFLOPS / 456 GB/s ≈ 239 FLOP/byte`

Since 4.2 << 239, decode is **deeply memory-bandwidth bound** — the GPU's compute units are **~57× underutilized**. The bottleneck is purely how fast we can stream weights from memory.

**Implication**: Any optimization that reduces **compute** but doesn't reduce **memory reads** will have **zero effect** on decode latency. Only optimizations that either:
1. Reduce total bytes read from memory, or
2. Improve memory access patterns for better effective bandwidth
will help.

### ~~2.1 Gate+Up Double Weight Memory Traversal~~ (Revised: NOT applicable)

**Original idea**: Interleave gate+up weights to halve weight memory traffic in a single streaming pass.

**Why this doesn't work**: Prefill and decode **share the same weight tensors**. The weight layout is set once at model load time in `convert_moe_to_compressed.cpp`. Interleaving for decode would break the layout assumptions of all prefill paths (oneDNN loop, micro_gemm, grouped GEMM), which expect standard `[E, N, K]` / `[E, K, N]` weight layouts. There is no practical way to maintain separate weight copies without doubling the model memory footprint (~15GB for Qwen3-30B).

**Revised assessment**: In the current single-kernel `mlp_gate_up`, both gate and up weight reads traverse **separate memory regions** but they go through the **same SLM `x2` buffer**, so the input vector is loaded only once. The "double traversal" actually means two sequential streaming passes over different weight blocks, which is fine for a memory-bound workload — the total bytes read are `2 × K × N / 2` regardless of ordering. Interleaving wouldn't reduce total bytes, only change the access pattern. Since modern GPU memory controllers handle sequential streaming well, the benefit would be negligible.

**Status**: ~~Optimization candidate~~ **Not viable** — shared weight layout constraint and no byte reduction.

### 2.2 K-Split Across Subgroups: Deep Analysis

**Original idea**: Split K-dimension across `SUBGROUP_NUM=8` subgroups so each handles `K/8` elements, then combine via SLM.

**Current kernel behavior** (detailed walkthrough for u4, Xe2 SUBGROUP_SIZE=32):

```
Dispatch: global = [topk, 32, INTER_SIZE/N_BLOCK], local = [1, 32, 8]
          → get_global_id(0) = expert_no
          → get_global_id(1) = lane within subgroup (unused explicitly, handled by sub_group intrinsics)
          → get_global_id(2) = N-block index
```

Each workgroup: 8 subgroups × 32 lanes = 256 threads.
Each workgroup processes `N_BLOCK = 4` output elements (2 at a time in a `n+=2` loop).
All 8 subgroups cooperate on loading `x` into SLM, then ALL 8 subgroups independently compute the SAME K-reduction for the same N outputs.

**Why all 8 subgroups do the same work**: The code has:
```c
int n_start = get_global_id(2) * N_BLOCK;  // same for all subgroups in same workgroup
...
for (int gk = 0; gk < K / FAKE_GROUP_SIZE; gk++) {
    // All subgroups iterate ALL K tiles
    half4 a = intel_sub_group_block_read_us4(x2 + gk * FAKE_GROUP_SIZE);  // from SLM, same data
    uchar2 b = intel_sub_group_block_read_uc2(B + gk * FAKE_GROUP_SIZE/2); // from global, same address
}
```
The 8 subgroups read the same weight data and get the same result. Only subgroup id_local==0 writes.

**Wait — re-reading more carefully**: Actually looking at `get_global_id(2)`: with `global[2] = INTER_SIZE / N_BLOCK` and `local[2] = SUBGROUP_NUM = 8`, `get_global_id(2)` gives each subgroup in a workgroup a **different** N-block index. The N dimension is split across subgroups within the workgroup!

Let me re-derive:
- `local = [1, SUBGROUP_SIZE, SUBGROUP_NUM]` → workgroup has SUBGROUP_NUM subgroups
- `get_global_id(2) = get_group_id(2) * SUBGROUP_NUM + get_local_id(2)`
- `get_local_id(2)` = 0..7 (the subgroup index)
- So subgroup 0 handles `n_start = (group_id*8 + 0) * 4`, subgroup 1 handles `(group_id*8 + 1) * 4`, etc.

**Corrected understanding**: The 8 subgroups in one workgroup process **different N-block ranges** (32 consecutive N outputs total per workgroup). They share the SLM `x2[]` buffer (loaded cooperatively). Each subgroup independently does the full K-reduction for its own 4 N-outputs. This is correct N-parallel design, NOT redundant K.

**Now — would K-splitting help?**

Since decode is memory-bound, the question is whether K-splitting changes the number of bytes read:

**Bytes read per workgroup (current, u4)**:
- `x` loaded into SLM: `HIDDEN_SIZE × 2 bytes` = 4 KB (shared, loaded once by all 8 subgroups)
- **Weight reads per subgroup**: Each subgroup processes `N_BLOCK=4` output elements. For each of the 4 outputs, reads `K/2` bytes (u4) = `4 × 2048/2 = 4 KB` per subgroup.
- Total weight reads per workgroup: `8 × 4 KB = 32 KB`
- Scale/ZP: ~a few hundred bytes per subgroup (small)

**With K-splitting (hypothetical)**:
If we split K across, say, 2 of the 8 subgroups and use the other 6 for more N-blocks:
- This doesn't reduce total weight bytes — same weights must be read.
- It reduces latency per-output-element (two subgroups complete faster than one) but only matters if we're compute-bound in the K-reduction loop, which we're not.

**Measured decode time**: ~22 ms/token for Qwen3-30B at 32 input tokens.

**Per-layer MOE time estimate**: The model has 48 MoE layers. If MOE takes ~50% of decode time:
`22ms × 50% / 48 layers ≈ 0.23 ms per MoE layer`

**Roofline prediction**: 12 MB / 456 GB/s × 1000 = **0.026 ms** — the theoretical minimum for reading 12 MB of weights.

**Gap**: Measured ~0.23 ms vs. theoretical ~0.026 ms ≈ **~9× gap**. This gap comes from:
1. Kernel launch overhead (3 kernels: gate_up, down, reduce)
2. Shared expert (3 additional oneDNN GEMV calls)
3. Softmax/topk kernel
4. SLM loading latency
5. `sub_group_reduce_add` + conditional write overhead
6. Effective bandwidth << peak (not perfectly sequential access)

**K-splitting verdict**: Since the current kernel is reaching only ~11% of peak bandwidth (0.026/0.23), the bottleneck is likely kernel launch overhead and the many sequential kernel launches (6+ kernels per MoE layer), not the inner K-reduction loop length. K-splitting would make each subgroup's inner loop shorter, but wouldn't reduce total memory reads. For a memory-bound kernel, the subgroups are mostly waiting for memory anyway — making them wait for different parts of K and then combining via SLM would add overhead (barrier + SLM reduction) without reducing memory traffic.

**Verdict**: **Unlikely to help significantly.** The 9× gap to roofline is dominated by launch overhead and serial kernel sequencing, not by K-loop efficiency. K-splitting adds barrier + SLM reduction cost that would likely make things slightly worse.

### 2.3 Fuse Reduce into Down Kernel

**Current behavior** (`mlp_reduce`): A separate kernel sums `topk` expert outputs element-wise. For topk=2, this is just `output[i] = y[0*H+i] + y[1*H+i]`.

**Optimization**: Fuse the reduce into `mlp_down`. The down kernel already writes `y[expert_no * hidden_size + n] = sum * routing_weight`. Instead of writing per-expert results, have both expert workgroups atomically accumulate into the same output row.

**Concern**: The two experts run as separate `get_global_id(0)` workgroups. Since they process the same output addresses, atomic f16 add would be needed. Intel Xe2 does not have native atomic f16 — this would require f32 atomic or a two-pass approach. Alternatively, since topk=2 is small, a simple read-after-write with SLM coordination between the two expert workgroups could work, but the workgroups are independent and may not be co-scheduled.

**Practical alternative**: More realistically, keep separate output buffers but fuse the reduce into the host-side dispatch — just change the down kernel to write directly to `output[] += down_result * weight` with an initialization pass to zero the output first. But this is essentially what the per-expert onednn loop path's `index_add` scatter does.

**Expected benefit**: Eliminates one kernel launch (~5-10 µs per launch × 48 layers = ~0.24-0.48 ms total) and saves reading `topk × hidden_size × 2` bytes = `2 × 2048 × 2 = 8KB` (negligible).

### 2.4 Shared Expert Sequential Execution

**Current behavior**: When `_has_shared_expert == true` (Qwen3-30B-A3B has shared experts), the decode path already fuses the shared expert into the `mlp_gate_up` and `mlp_down` kernels via `SHARED_EXPERT_ENABLE`. The shared expert is processed as expert_no = `MAX_TOPK` (i.e., expert index 2 for topk=2) in the same kernel launch. This means the shared expert GEMV runs in parallel with the routed experts.

However, for **prefill** with oneDNN primitives, `execute_shared_expert()` runs 3 separate oneDNN calls after the main MoE computation. The comment in code says:
```
// execute_shared_expert will be serialized with the following kernels by onednn stream
```

**For decode**: Already fused. No action needed.
**For prefill**: The sequential oneDNN shared expert adds latency. The grouped_gemm/micro_gemm paths should incorporate the shared expert into the same grouped dispatch if possible.

### 2.5 Kernel Launch Overhead: The Real Decode Bottleneck

From the roofline analysis (Section 2.2), the 9× gap between theoretical bandwidth minimum and measured decode time suggests **kernel launch overhead** is the primary decode bottleneck, not compute or memory access patterns.

For single-token decode, one MoE layer dispatches:
1. `softmax_topk` or `sigmoid_bias_topk` — routing
2. `mlp_gate_up` — fused gate+up GEMV for all topk+1 experts
3. `mlp_down` — down GEMV for all topk+1 experts
4. `mlp_reduce` — element-wise sum

= **4 kernel launches** per MoE layer.

At ~5-10 µs per OCL kernel launch, that's 20-40 µs overhead per MoE layer × 48 layers = **~1-2 ms** total just in launch overhead — roughly **5-9%** of the 22 ms decode latency.

**Optimization**: Reduce the number of kernel launches. The `mlp_reduce` kernel is the easiest candidate to fold into `mlp_down`. Beyond that, further fusion (e.g., gate_up + swiglu + down in a single mega-kernel) would be complex but could save 1-2 more launches per layer.

---

## Part 3: Cross-Cutting Optimization Opportunities

### 3.1 Fuse gate+up weights for decode AND prefill simultaneously

**Problem** (from 2.1): Gate and up weights cannot be interleaved for decode only, because prefill shares them.

**Revised approach**: Fuse gate+up into a single `[E, hidden, 2*inter]` weight tensor used by BOTH paths:
- **Decode**: The `mlp_gate_up` OCL kernel already processes gate and up sequentially. With a fused layout, it could still read them as two halves of the larger N dimension. No weight traffic reduction, but 1 GEMV becomes `[1, K] × [K, 2N]` — fewer kernel launches if we restructure.
- **Prefill (grouped GEMM)**: A single `[total, hidden] × [E, hidden, 2*inter] → [total, 2*inter]` grouped matmul replaces two separate calls.

This is essentially optimization 1.2 but ensuring it works for both paths.

### 3.2 Profile-Guided Path Selection

**Current behavior**: `use_grouped_gemm_prefill` is set once at init (default=true). But the benchmark data shows grouped GEMM is only faster at large prompt lengths (8K+).

**Optimization**: Runtime heuristic that selects the execution path based on `token_num`:
- `token_num < 512`: Use oneDNN per-expert loop (lower overhead, pre-compiled)
- `512 ≤ token_num < 4096`: Use micro_gemm (best intermediate behavior) 
- `token_num ≥ 4096`: Use grouped GEMM (best GPU utilization at scale)

---

## Summary: Priority-Ranked Optimization Opportunities (Revised)

| # | Optimization | Path | Est. Impact | Complexity | Notes |
|---|-------------|------|-------------|------------|-------|
| 1 | Fuse gate+up into single grouped matmul | Prefill | Medium-High (halve GEMM launches) | Medium | Also benefits decode if weight layout shared |
| 2 | Fuse SwiGLU post-op into gate GEMM | Prefill | Medium (save kernel + buffer R/W) | Low-Medium | micro_gemm already does this |
| 3 | Profile-guided path selection | Both | Medium (always pick best path) | Low | Data shows grouped only helps at 8K+ |
| 4 | Fuse reduce into down kernel (decode) | Decode | Low-Medium (save ~1ms over 48 layers) | Low | Mainly kernel launch overhead savings |
| 5 | Runtime-dim grouped GEMM (avoid recompiles) | Prefill | Low-Medium (reduce JIT overhead) | Low | Pending oneDNN support |
| 6 | Inverse mapping for scatter-reduce | Prefill | Low (~1.9% at 8K tokens) | Low | Only helps at long prompt lengths |

### Rejected / Not Viable

| Optimization | Reason |
|-------------|--------|
| GPU mask gen for grouped GEMM | GPU single-thread execution slower than CPU; CPU sync cost is worth it |
| Interleave gate+up weights (decode only) | Prefill+decode share weights; no separate layout possible without 2× memory |
| K-split across subgroups (decode) | Decode is memory-bound, K-split doesn't reduce memory traffic; adds barrier overhead |
