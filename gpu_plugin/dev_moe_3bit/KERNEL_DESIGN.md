# 3-Bit GPU Kernel Design for OpenVINO

This document describes the architecture, requirements, and optimization strategy for 3-bit quantized GPU kernels in OpenVINO's Intel GPU plugin. The effective 3-bit average precision is achieved by NNCF assigning each weight layer as either **U2 (2-bit)** or **I4/U4 (4-bit)** during model conversion. At inference time some layers carry U2 weights and others carry I4/U4 weights — both types can coexist across layers of the same model, but a single weight tensor is always one type.

---

## 1. Background and Constraints

### 1.1 Weight Format

3-bit quantization achieves an effective average bit-width by having NNCF assign each weight layer to one of two precisions during model conversion. The assignment is **per layer, not per element**: every element within a given weight tensor shares the same data type.

| Tensor | Data Type | Notes |
|--------|-----------|-------|
| Weight (2-bit layers) | U2 | 4 values packed per byte; assigned by NNCF |
| Weight (4-bit layers) | I4 or U4 | 2 values packed per byte; assigned by NNCF |
| Zero-point | U8 | Per quantization group |
| Scale | FP16 | Per quantization group |

> **Key constraint for the inference runtime:** a layer's weight dtype is fixed at compile time (determined by NNCF conversion). The kernel scheduler must read `weight_dtype` per primitive and route each layer independently — U2 layers go to the custom 3-bit kernel; I4/U4 layers go to the existing oneDNN 4-bit path.

### 1.2 oneDNN Limitation

**oneDNN GPU GEMM does not support U2/I2 weight precision.** The minimum supported integer weight type in oneDNN is S4/U4. Therefore, 3-bit kernels must be implemented as custom GPU primitives and scheduled separately from the existing 4-bit/8-bit oneDNN paths.

---

## 2. Runtime Kernel Scheduler

Because oneDNN handles 4-bit and 8-bit quantization natively, 3-bit requires a **custom scheduler at the primitive level** that selects the appropriate kernel implementation based on the actual weight data type:

- If `weight_dtype ∈ {U4, I4, U8, S8}` → dispatch to existing oneDNN GEMM/GEMV primitive
- If `weight_dtype == U2` → dispatch to custom 3-bit GPU kernel

This scheduler must be integrated into the GPU primitive's `execute()` path and must cache compiled kernel objects to avoid JIT recompilation overhead across inference calls.

---

## 3. Kernel Requirements

### 3.1 Prefill Stage (GEMM, Multiple Tokens)

The prefill stage processes batches of tokens through matrix–matrix operations. It is **compute-bound** at medium-to-large batch sizes.

#### 3.1.1 Dense GEMM (Non-MoE Layers)

| Property | U2-weight layers | I4/U4-weight layers |
|----------|-----------------|---------------------|
| Activation type | INT8 (DQ) or FP16 | INT8 (DQ) or FP16 |
| Weight type | **U2** (assigned by NNCF) | **I4 or U4** (assigned by NNCF) |
| Output type | FP16 | FP16 |
| Core computation | `dpas(int8 × int8 → int32 → fp16)` (U2 expanded to INT8) | `dpas(int8 × int4 → int32 → fp16)` |
| Kernel path | Custom 3-bit kernel | Existing oneDNN path |

**Kernel implementation options for U2-weight layers** (in order of preference):

1. **Gemmstone microkernel API** *(preferred)* — Highest performance and flexibility; fuses U2→INT8 unpack directly into the microkernel prologue before DPAS.
2. **C for Metal (CM)** — Lower-level control; suitable for experimental tiling strategies.
3. **OpenCL + DPAS extension** — Portable baseline; easier to prototype but harder to tune.

#### 3.1.2 MoE GEMM

| Property | U2-weight layers | I4/U4-weight layers |
|----------|-----------------|---------------------|
| Activation type | INT8 (DQ) or FP16 | INT8 (DQ) or FP16 |
| Weight type | **U2** (assigned by NNCF) | **I4 or U4** (assigned by NNCF) |
| Output type | FP16 | FP16 |
| Core computation | `dpas(int8 × int8 → int32 → fp16)` (U2 pre-expanded to INT8) | `dpas(int8 × int8 → int32 → fp16)` via oneDNN |

**Kernel implementation options for U2-weight MoE layers** (in order of preference):

1. **Micro-GEMM enhancement via Gemmstone API** *(preferred)* — Extends the existing `grouped_micro_gemm` path; fuses U2→INT8 expansion into the tile load phase before DPAS.
2. **C for Metal (CM)** — Full control over register allocation; required if Gemmstone does not expose sufficient U2 hooks.

---

### 3.2 Decode Stage (GEMV, Single Token)

The decode stage processes one token at a time through matrix–vector operations. It is **memory-bound** regardless of expert count or model size.

#### 3.2.1 Dense GEMV (Non-MoE Layers)

| Property | U2-weight layers | I4/U4-weight layers |
|----------|-----------------|---------------------|
| Activation type | FP16 | FP16 |
| Weight type | **U2** (assigned by NNCF) | **I4 or U4** (assigned by NNCF) |
| Output type | FP16 | FP16 |
| Core computation | `fma(fp16 × fp16 → fp32 → fp16)` after U2 dequant | `fma(fp16 × fp16 → fp32 → fp16)` after I4/U4 dequant |
| Kernel path | Custom 3-bit kernel | Existing oneDNN / custom GEMV path |

**Kernel implementation options for U2-weight layers** (in order of preference):

1. **Gemmstone microkernel API** *(preferred)* — Reuses tile infrastructure; clean integration with existing GEMV tuning.
2. **OpenCL** — Portable and easy to profile; suitable baseline implementation.
3. **C for Metal (CM)** — Maximum register control; use when OpenCL register pressure is too high.

#### 3.2.2 MoE GEMV

| Property | U2-weight layers | I4/U4-weight layers |
|----------|-----------------|---------------------|
| Activation type | FP16 | FP16 |
| Weight type | **U2** (assigned by NNCF) | **I4 or U4** (assigned by NNCF) |
| Output type | FP16 | FP16 |
| Core computation | `fma(fp16 × fp16 → fp32 → fp16)` after U2 dequant | Existing MoE OpenCL GEMV path (U4/U8) |
| Kernel path | Extended MoE OpenCL kernel with U2 dequant path | Existing `moe_3gemm_swiglu_mlp.cl` |

**Kernel implementation options for U2-weight MoE layers** (in order of preference):

1. **Enhanced MoE OpenCL kernel** *(preferred)* — Extends the existing `moe_3gemm_swiglu_mlp.cl` decode path with a new U2 dequant branch alongside the existing U4/U8 paths. The branch is selected at JIT time via `WEIGHT_COMPRESSION_DT` constant.
2. **C for Metal (CM)** — Fallback for architectures where OpenCL register pressure is unmanageable.

---

## 4. Kernel Optimization

### 4.1 Dense GEMM Optimization (Prefill, Compute-Bound)

**Performance target:** ≥80% of hardware compute roofline.

**Hardware context (Intel LNL / Xe2):**

| Resource | Value |
|----------|-------|
| Xe Cores per GPU | 8 |
| EUs per Xe Core | 8 |
| Threads per EU | 8 |
| Subgroup size | 16 or 32 |
| GRF registers per EU | 256 × 256 bytes |
| SLM per Xe Core | 32 KB |

#### Compute Strategy

- **Do not chain multiple separate kernels** for a single GEMM layer. Fuse as many phases as possible (weight unpack → DPAS → epilogue) into a single kernel dispatch.
- Use **rectangular subgroup tiles** to maximize B-matrix reuse and amortize the per-group scale/ZP load cost across multiple output elements.
- Compute per-group B scales either as a **separate upfront reduction kernel** or fused into the GEMM microkernel prologue — choose based on register pressure.
- Fuse **bias addition and post-ops** (e.g., SiLU, GELU) into the GEMM epilogue to avoid extra global memory round-trips.

#### DPAS/XMX Instruction Selection

The instruction choice depends on the weight dtype of the layer being executed, which is fixed at compile time:

```
U2-weight layers (dense):   dpas(int8 × int8 → int32 → fp16)  [U2 expanded to INT8 in tile prologue]
U2-weight layers (MoE):     dpas(int8 × int8 → int32 → fp16)  [U2 expanded to INT8 in tile prologue]
I4/U4-weight layers:        dpas(int8 × int4 → int32 → fp16)  [standard oneDNN WOQ path]
```

#### U2 Weight Unpacking (U2-weight layers only)

- 4 U2 values are packed per byte. Unpack using shift and mask:
  ```c
  val_i = (packed_byte >> (2 * i)) & 0x3;   // i in {0,1,2,3}
  ```
- Load weight bytes via vectorized `uchar16`/`uchar8` block reads, then bitfield-extract in GRF. No SLM round-trip needed for dequant.
- Dequant formula: `w_f16 = (cast_f16(u2_val) - zp) * scale`
- I4/U4 layers use the standard oneDNN dequant path and do not enter this code path.

#### Tiling Strategy

| Parameter | Recommended Value |
|-----------|------------------|
| M tile | 16–32, aligned to `subgroup_size` |
| N tile | 16–32, aligned to `subgroup_size` |
| K tile | Equal to quantization group size (128) |
| SLM usage | B-matrix tiles to reduce HBM round-trips |
| Buffering | Double-buffer A and B: load tile `i+1` while computing tile `i` |

#### Register Management

- 3-bit dequant adds approximately 4–6 GRF over the equivalent INT4 path. Monitor GRF usage carefully.
- Avoid global memory spilling. If register pressure exceeds capacity, spill intermediate values to SLM, not to global memory.
- Keep scale/ZP loads **outside** the inner K-loop; load once per group and broadcast across subgroup lanes.
- Unroll the K-loop by 2× or 4× to increase instruction-level parallelism and hide memory latency.

---

### 4.2 Dense GEMV Optimization (Decode, Memory-Bound)

**Performance target:** ≥85% of memory bandwidth roofline.

**Memory roofline reference:**

| GPU | HBM Bandwidth | Target Throughput |
|-----|--------------|-------------------|
| Intel LNL (Xe2) | ~100 GB/s | ≥85 GB/s |
| Intel BMG (B580) | ~456 GB/s | ≥388 GB/s |

> GEMV at 3-bit precision has an operational intensity of approximately 4 FLOPs/byte, which is 60× below the XE2 compute crossover point (~239 FLOPs/byte). **ALU optimizations have negligible impact; only data movement optimizations matter.**

#### Work-Group Layout

```
subgroup_size:  32 (XE2+) or 16 (older)
N_BLOCK:        4   (output elements per workgroup)
SUBGROUP_NUM:   8   (subgroups per workgroup)
threads/WG:     256 (= SUBGROUP_SIZE × SUBGROUP_NUM)

gws = {1, SUBGROUP_SIZE, N / N_BLOCK}
lws = {1, SUBGROUP_SIZE, SUBGROUP_NUM}
```

Each subgroup handles a distinct N-block range. Subgroups within a workgroup are **N-parallel, not K-parallel** — there is no K-reduction across subgroups (no barrier needed).

#### U2 Weight Loading and Dequant

```c
// Load: 16-byte (64 U2 values) aligned block read per lane
uchar16 chunk = intel_sub_group_block_read_uc16(weight_ptr + offset);

// Unpack 4 U2 values per byte
for (int i = 0; i < 4; i++)
    u2_val[i] = (chunk[byte_idx] >> (2 * i)) & 0x3;

// Dequant: w_f16 = (u2_val - zp) * scale
float w_f16 = ((float)u2_val - (float)zp) * scale;
```

- Load quantization scale (FP16) and zero-point (U8) once per group; broadcast to all lanes in the subgroup.
- This dequant path applies only to **U2-weight layers**. Layers with I4/U4 weights use the standard oneDNN dequant path and are not dispatched to this kernel.

#### Critical Anti-Pattern: Do NOT Fuse gate + up into a Single GEMV Kernel

Empirical testing on the existing INT4 MoE path shows fused gate+up GEMV is **slower**, not faster:

| Metric | Separate kernels | Fused kernel |
|--------|-----------------|--------------|
| GRF usage | ~10–12 | ~20–24 |
| Occupancy | High | Drops significantly |
| Cache behavior | Sequential weight access | Thrashing across two disjoint regions |
| Net result | Baseline | **Slower** |

Keep gate and up GEMV as **separate kernel dispatches**.

#### FMA Accumulation

- Accumulate partial sums in **FP32** to avoid precision loss; convert to FP16 only at the final store.
- Use `sub_group_reduce_add` for the K-dimension reduction across lanes.
- Each subgroup lane handles `K / SUBGROUP_SIZE` elements independently, then reduces.
- Prefer `sub_group_reduce_add` over hand-written shuffle trees for maintainability and correctness.

#### Latency Hiding

- **Double-buffer weight tiles**: issue the `block_read` for tile `i+1` before beginning the FMA loop for tile `i`. This overlaps memory transfers with computation and can hide ~50% of HBM latency.

---

### 4.3 MoE Prefill Optimization (Compute-Bound at Medium-to-Large Batch)

The MoE prefill path becomes compute-bound above approximately 16 tokens per active expert.

#### Execution Path Selection

| Condition | Path | Notes |
|-----------|------|-------|
| `token_num == 1` | Decode GEMV | Never use prefill path for single token |
| `1 < token_num < 4096` | Per-expert micro_gemm loop | oneDNN micro_gemm, one call per active expert |
| `token_num ≥ 8192` | Grouped GEMM single dispatch | One grouped matmul covers all experts |
| Crossover threshold | Profile-guided | Measure per model + GPU pair; crossover varies |

#### U2 Weight Expansion Strategy for DPAS (U2-weight layers only)

For layers with U2 weights, DPAS requires INT8 operands; U2 values must be widened. Three strategies:

| Option | Description | Trade-off |
|--------|-------------|-----------|
| A — Offline expansion | Store INT8 copy of U2 weights alongside the U2 buffer | 4× weight memory for U2 layers |
| B — JIT expansion kernel | Run U2→INT8 kernel before each GEMM dispatch | Extra kernel launch per layer |
| **C — Fused in microkernel** *(preferred)* | Unpack U2→INT8 inside gemmstone tile load prologue | No extra memory, no extra launch |

I4/U4 layers feed INT4 directly to DPAS via the standard oneDNN WOQ path and do not need this expansion.

#### Top-Priority Fusion: Gate + Up into a Single Grouped Matmul

- Gate and up GEMMs share the same input activation matrix A. Dispatching them together halves the number of grouped GEMM launches.
- Technique: use a **double-width N tile** `[gate_out | up_out]` in a single dispatch, then apply SwiGLU as a fused epilogue: `silu(gate_out) × up_out`.
- SwiGLU fusion eliminates the intermediate activation buffer write/read, saving approximately `INTERMEDIATE_SIZE × token_num × 2 × 2` bytes of memory traffic.

#### Rectangular Tile Shape for B Reuse

- Use K-tiles aligned to the quantization **group size** (typically 128). This ensures scale and zero-point are loaded exactly once per tile, with no fractional group overhead.
- Wider K tiles spread the per-group dequant cost across more output elements, reducing effective quantization overhead.

#### Grouped GEMM Dispatch Efficiency

- Pass `DNNL_ARG_HINT_MAX_GROUP_SIZE` as an **execute-time runtime argument** (not baked into the primitive descriptor). This avoids recompilation when the routing distribution changes between requests.
- Cache grouped GEMM primitives keyed by `total_tokens`. Routing variation within the same total token count causes **zero additional JIT compilations**.

#### Mask Generation

- **Prefer CPU-side mask generation**: read `topk_ids` from GPU to CPU, build the expert token mask on CPU, then transfer the mask back. The GPU serial scan alternative (one thread scanning all `tokens × topk` entries) is slower due to lack of branch prediction.
- CPU→GPU synchronization cost is justified by the superior cache and branch-prediction behavior of sequential CPU reads.

#### Gather / Scatter Around GEMM

- **Gather** activated tokens before the GEMM to eliminate all-zero rows from inactive experts, reducing effective matrix size.
- **Scatter + reduce** after the down-projection GEMM using an inverse token mapping.
- Fuse the routing weight multiplication into the scatter-reduce kernel to avoid a separate pass over the output buffer.

---

### 4.4 MoE Decode Optimization (Memory-Bound, Single Token)

**Roofline context:**

| Metric | Value |
|--------|-------|
| Operational intensity (3-bit GEMV) | ~4 FLOPs/byte |
| XE2 compute crossover | ~239 FLOPs/byte |
| Regime | Memory-bound (60× below crossover) |
| **Conclusion** | Only data-movement reductions are effective; ALU savings are irrelevant |

#### Kernel Pipeline

```
softmax_topk  →  mlp_gate_up  →  mlp_down  →  mlp_reduce
(4 separate kernel dispatches)
```

#### Dispatch Geometry (gate_up and down Kernels)

```
gws = { num_experts,  SUBGROUP_SIZE,  INTERMEDIATE_SIZE / N_BLOCK }
lws = { 1,            SUBGROUP_SIZE,  SUBGROUP_NUM }

N_BLOCK     = 4    (output elements per workgroup)
SUBGROUP_NUM = 8   (subgroups per workgroup)
SUBGROUP_SIZE = 32 (XE2+) or 16 (older GPU)

Threads per workgroup: 256 = SUBGROUP_SIZE × SUBGROUP_NUM
Each workgroup: handles N_BLOCK=4 output elements
Subgroups within workgroup: N-parallel (different N ranges), NOT K-parallel
```

#### U2 Weight Dequant in Decode Kernel

```c
// Applies to U2-weight layers only.
// I4/U4-weight layers are routed to the existing oneDNN dequant path.

// Each subgroup lane loads 64 U2 values (16 bytes, 32-byte aligned)
uchar16 chunk = intel_sub_group_block_read_uc16(weight_ptr + offset);

// Unpack: 4 U2 values per byte
val = (chunk[byte] >> (2 * lane_idx)) & 0x3;

// Dequant: w_f16 = (u2_val - zp) * scale
float w_f16 = ((float)val - (float)zp) * scale;
```

#### Zero-Point Optimization (`xg_sum`)

For **symmetric** 3-bit quantization (no zero-point), the ZP compensation accumulation (`xg_sum`) can be entirely eliminated at JIT compile time:

```c
// JIT constant controls compilation
// HAS_ZP=0: symmetric — all ZP code is compiled out
// HAS_ZP=1: asymmetric — ZP compensation code is included
#if HAS_ZP
    xg_sum += dequant_val;  // accumulate for ZP offset correction
#endif
```

This avoids dead code execution in the common symmetric case.

#### Fusion Opportunities

| Fusion | Benefit | Complexity |
|--------|---------|------------|
| `mlp_reduce` into `mlp_down` kernel | Eliminates ~1ms latency per 48-layer model by avoiding a separate global memory write of down-projection outputs | Low |
| Shared expert as extra workgroup | Avoids a separate kernel launch for shared expert; expert uses index `MAX_TOPK` in dim-0 | Low (already supported via `SHARED_EXPERT_ENABLE` JIT constant) |

> **Do NOT fuse gate + up GEMV into a single kernel** (see Section 4.2 anti-pattern analysis above — the same reasoning applies to MoE decode).

#### Shared Expert Fusion (Qwen3.5-Style Models)

- The shared expert is treated as workgroup index `MAX_TOPK` in the first GWS dimension, dispatched alongside sparse experts in the same kernel launch.
- The shared expert's scalar routing gate uses a **workgroup-wide parallel reduction** (256 threads → `reduce_add` → sigmoid), replacing the naive single-thread sequential computation.
- Enable via JIT constant `SHARED_EXPERT_ENABLE=1`.

#### SLM Usage

| Use Case | Details |
|----------|---------|
| Activation preprocessing | Interleave even/odd elements for 4-bit sub-byte compatibility; direct copy for 8-bit or FP16 |
| Scale/ZP broadcast | One subgroup loads scale and ZP from global memory; all others read via SLM broadcast to eliminate redundant global loads |

#### Double-Buffering Weight Tiles

- Issue the `block_read` for weight tile `i+1` before beginning the FMA loop for tile `i`.
- Hides approximately 50% of HBM round-trip latency.
- Cost: 2× the working-set register count for weight tiles. Monitor GRF pressure carefully.

---

## 5. Summary: Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| U2 weight expansion strategy | Fuse into microkernel prologue | No extra memory or launch overhead |
| gate + up kernel structure | Separate dispatches | Fused variant causes occupancy collapse |
| Prefill grouped vs. per-expert | Threshold at ~8K tokens | Verified via roofline and end-to-end measurement |
| Grouped GEMM dispatch hint | `DNNL_ARG_HINT_MAX_GROUP_SIZE` at execute-time | Zero recompilations under routing variation |
| Mask generation | CPU-side | Faster than GPU serial scan for typical sizes |
| ZP handling | `#if HAS_ZP` JIT guard | Compiles out dead code for symmetric case |
| SLM for scale/ZP | Broadcast via SLM | Avoids `num_subgroups × group_count` global reads |
