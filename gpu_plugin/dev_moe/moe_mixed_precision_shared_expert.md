# Mixed Precision MoE: Shared Expert F16 + Sparse Expert i4/i8

## 1. Problem Statement

Some MoE models (e.g., Qwen3.5 MoE, DeepSeek-V3) keep the shared expert in full FP16 precision while compressing sparse expert weights to INT4/INT8. The current `moe_3gemm_fused_compressed` primitive assumes **uniform weight precision** across all experts (including shared). When the shared expert weights are F16 and sparse expert weights are i4/i8, the decode path's GEMV kernel will misinterpret the shared expert's F16 data as compressed integers, producing garbage results.

**Goal**: Support mixed precision within a single fused MoE kernel dispatch, using the recommended **single-kernel workgroup-level branch** approach (zero additional kernel submits).

---

## 2. Current Architecture Analysis

### 2.1 Execution Paths & Their Readiness

| Path | Shared Expert Handling | F16 Ready? |
|------|----------------------|------------|
| **Decode (GEMV kernel)** | Same workgroup set, same `MOE_WEI_DT` type for all | ❌ No — hardcoded type |
| **Prefill (oneDNN loop)** | `init_shared_primitives()` reads dtype dynamically | ⚠️ Almost — `ic_group_size` bug |
| **Prefill (micro_gemm)** | Separate `execute_shared_expert()` at end | ⚠️ Same `ic_group_size` bug |
| **Prefill (grouped_gemm)** | Separate `execute_shared_expert()` at end | ⚠️ Same `ic_group_size` bug |

### 2.2 Decode Path — The Core Problem

In `moe_3gemm_swiglu_mlp.cl`, the shared expert kernel parameters are declared with the same `MOE_WEI_DT` type as sparse experts:

```c
// Current: shared expert uses same type as sparse
const __global MOE_WEI_DT* shared_gate_weight,    // ← If sparse=u4, this is uchar
const __global MOE_SCALE_DT* shared_gate_scale,
const __global MOE_ZP_DT* shared_gate_zp,
```

And the GEMV dispatch is a single compile-time branch:

```c
#if WEIGHT_COMPRESSEION_DT == 0
    gate_up_gemv_n2x_u4(...)   // ALL experts use u4 path
#elif WEIGHT_COMPRESSEION_DT == 1
    gate_up_gemv_n2x_u8(...)   // ALL experts use u8 path
#elif WEIGHT_COMPRESSEION_DT == 2
    gate_up_gemv_n2x_f16(...)  // ALL experts use f16 path
#endif
```

When shared expert is F16 but `WEIGHT_COMPRESSEION_DT == 0` (sparse=i4), the shared expert workgroup executes the u4 dequant path on F16 data → **incorrect results**.

### 2.3 Prefill Path — Nearly Ready

`init_shared_primitives()` (line 1152) already reads weight dtype dynamically:

```cpp
auto gate_w_dt = convert_data_type(addr.shared_weight[0]->get_layout().data_type);
// Creates oneDNN matmul with the actual weight type
```

**Bug**: When shared weights are F16 (no scale/zp), the code still passes `_gate_up_group_size` as `ic_group_size`, causing oneDNN to expect non-existent scale tensors.

### 2.4 Transformation Pipeline

```
FuseMOE3GemmCompressed (line 566)     ← matches pattern, creates fused op
  ↓
ConvertPrecision (line 658)           ← may convert f32 → f16
  ↓
KeepMOE3GemmConstPrecision (line 669) ← protects compressed constants from conversion
```

`KeepMOE3GemmConstPrecision` only matches `u4` typed constants for shared weights. F16 shared weights don't need protection (F16 is already the target inference precision), so this is fine as-is.

---

## 3. Implementation Plan (Recommended Approach: Single Kernel, Workgroup Branch)

### Design Principle

- **Zero additional kernel submits** — shared expert remains in the same dispatch as sparse experts
- **Zero SIMD divergence** — branch is at workgroup level (all threads in a workgroup take the same path)
- Shared expert workgroup always uses `gate_up_gemv_n2x_f16` regardless of sparse expert compression type
- Minimal code change — reuse existing `_f16` GEMV functions

### 3.1 Layer 1: Config Extension

**File**: `src/common/transformations/include/ov_ops/moe_compressed.hpp`

Add shared expert precision metadata to `MOECompressed::Config`:

```cpp
struct Config : public MOE::Config {
    // ... existing fields ...

    // Shared expert weight precision (f16 = uncompressed, u4/i4/u8/i8 = compressed)
    ov::element::Type shared_weight_type = ov::element::dynamic;  // NEW
    size_t shared_group_size = 0;   // NEW: 0 = no quantization (raw f16/f32)
    bool shared_has_zp = false;     // NEW
};
```

**Impact**: Carries shared expert precision info through compilation, JIT constant generation, and kernel argument binding.

### 3.2 Layer 2: Transformation — Populate Config

**File**: `src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.cpp`

In the matcher callback, detect shared expert weight precision and set config:

```cpp
if (has_shared_expert) {
    auto shared_gate_const = ov::as_type_ptr<ov::op::v0::Constant>(
        pattern_map.at(shared_gate_wei_m).get_node_shared_ptr());
    config.shared_weight_type = shared_gate_const->get_element_type();

    if (config.shared_weight_type == ov::element::f16 ||
        config.shared_weight_type == ov::element::f32) {
        config.shared_group_size = 0;   // no quantization
        config.shared_has_zp = false;
    } else {
        // Same compression as sparse experts
        config.shared_group_size = config.group_size;
        config.shared_has_zp = config.has_zp;
    }
}
```

### 3.3 Layer 3: JIT Constants — Add `SHARED_WEIGHT_COMPRESSEION_DT`

**File**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp`

In `add_common_consts()`, add JIT constants for shared expert:

```cpp
// After existing WEIGHT_COMPRESSEION_DT logic (line ~737)
if (desc->_config.num_shared_expert > 0) {
    ov::element::Type shared_wt = desc->_config.shared_weight_type;
    if (shared_wt == ov::element::f16 || shared_wt == ov::element::f32) {
        jit.make("SHARED_WEIGHT_COMPRESSEION_DT", 2);  // f16 path
        jit.make("SHARED_HAS_ZP", 0);
        jit.make("SHARED_WEIGHT_IS_SIGNED", 0);
    } else if (shared_wt == ov::element::u4 || shared_wt == ov::element::i4) {
        jit.make("SHARED_WEIGHT_COMPRESSEION_DT", 0);  // 4-bit path
        jit.make("SHARED_HAS_ZP", desc->_config.shared_has_zp ? 1 : 0);
        jit.make("SHARED_WEIGHT_IS_SIGNED",
                 (shared_wt == ov::element::i4) ? 1 : 0);
    } else {
        jit.make("SHARED_WEIGHT_COMPRESSEION_DT", 1);  // 8-bit path
        jit.make("SHARED_HAS_ZP", desc->_config.shared_has_zp ? 1 : 0);
        jit.make("SHARED_WEIGHT_IS_SIGNED",
                 (shared_wt == ov::element::i8) ? 1 : 0);
    }
}
```

### 3.4 Layer 4: OpenCL Kernel — Workgroup-Level Branch

**File**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_mlp.cl`

#### 3.4.1 Change shared expert pointer types to `half*`

When shared expert is F16, the weight pointers must be `half*` not `uchar*`:

```c
#if SHARED_EXPERT_ENABLE
    // Shared expert weight pointers — always f16 in mixed-precision mode
    #if SHARED_WEIGHT_COMPRESSEION_DT == 2
    const __global half* shared_gate_weight,
    const __global half* shared_gate_scale,    // unused but passed for signature compat
    const __global half* shared_gate_zp,       // unused
    const __global half* shared_up_weight,
    const __global half* shared_up_scale,      // unused
    const __global half* shared_up_zp,         // unused
    #else
    const __global MOE_WEI_DT* shared_gate_weight,
    const __global MOE_SCALE_DT* shared_gate_scale,
    const __global MOE_ZP_DT* shared_gate_zp,
    const __global MOE_WEI_DT* shared_up_weight,
    const __global MOE_SCALE_DT* shared_up_scale,
    const __global MOE_ZP_DT* shared_up_zp,
    #endif
    const __global half* shared_gate_gate_weight,
    __global MOE_DTYPE* routing_weights,
#endif
```

#### 3.4.2 Workgroup-level GEMV dispatch branch

Replace the current monolithic GEMV dispatch with a workgroup-level branch for shared expert:

```c
    // After SLM load + barrier

#if SHARED_EXPERT_ENABLE
    if (is_shared) {
        // Shared expert: ALWAYS uses f16 GEMV (no dequant)
        #if SHARED_WEIGHT_COMPRESSEION_DT == 2
        gate_up_gemv_n2x_f16(
            (__global half*)shared_up_weight,
            (__global half*)shared_up_zp,
            y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, false);
        gate_up_gemv_n2x_f16(
            (__global half*)shared_gate_weight,
            (__global half*)shared_gate_zp,
            y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, true);
        #elif SHARED_WEIGHT_COMPRESSEION_DT == 0
        gate_up_gemv_n2x_u4(shared_up_weight, shared_up_scale, shared_up_zp,
                            y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
        gate_up_gemv_n2x_u4(shared_gate_weight, shared_gate_scale, shared_gate_zp,
                            y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
        #elif SHARED_WEIGHT_COMPRESSEION_DT == 1
        gate_up_gemv_n2x_u8(shared_up_weight, shared_up_scale, shared_up_zp,
                            y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
        gate_up_gemv_n2x_u8(shared_gate_weight, shared_gate_scale, shared_gate_zp,
                            y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
        #endif
    } else
#endif
    {
        // Sparse experts: use the model's compression type
        #if WEIGHT_COMPRESSEION_DT == 0
        gate_up_gemv_n2x_u4(up_weight, up_scale, up_zp,
                            y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
        gate_up_gemv_n2x_u4(gate_weight, gate_scale, gate_zp,
                            y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
        #elif WEIGHT_COMPRESSEION_DT == 1
        gate_up_gemv_n2x_u8(up_weight, up_scale, up_zp,
                            y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, false);
        gate_up_gemv_n2x_u8(gate_weight, gate_scale, gate_zp,
                            y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, xg_sum, true);
        #elif WEIGHT_COMPRESSEION_DT == 2
        gate_up_gemv_n2x_f16(up_weight, up_zp,
                             y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, false);
        gate_up_gemv_n2x_f16(gate_weight, gate_zp,
                             y, INTERMEDIATE_SIZE, HIDDEN_SIZE, x2, true);
        #endif
    }
```

**Same pattern applies to `mlp_down` kernel**.

#### 3.4.3 SLM xg_sum handling for mixed precision

When shared expert is F16 (`SHARED_WEIGHT_COMPRESSEION_DT == 2`), the shared expert workgroup does NOT need `xg_sum` (no ZP compensation). The current code already computes `xg_sum` for all workgroups uniformly. Two options:

- **Option A (simple)**: Let the shared expert workgroup compute `xg_sum` anyway — it will simply be unused. Minimal code change, negligible perf cost (memory-bound, a few extra ALU ops don't matter).
- **Option B (optimal)**: Guard `xg_sum` computation with `if (!is_shared || SHARED_HAS_ZP)`. Saves a few SLM writes.

**Recommendation**: Option A for initial implementation (KISS). Optimize later if profiling shows benefit.

### 3.5 Layer 5: Prefill Path — Fix `ic_group_size`

**File**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp`

In `init_shared_primitives()`, fix the group size logic:

```cpp
// Before creating shared up/gate/down projections:
int shared_gate_up_gs = (addr.shared_scale[0]) ? _gate_up_group_size : -1;
int shared_down_gs    = (addr.shared_scale[2]) ? _down_group_size    : -1;
bool shared_has_zp    = (addr.shared_zp[0] != nullptr);

// Then use shared_gate_up_gs / shared_down_gs / shared_has_zp
// instead of _gate_up_group_size / _down_group_size / config.has_zp
_shared_up_proj = std::make_shared<onednn_linear>(onednn_linear::create(eng,
    dnnl::memory::data_type::f16,
    up_w_dt,
    batch,
    _hidden_size,
    _shared_intermediate_size,
    shared_gate_up_gs,        // ← -1 for f16 (no quantization)
    t::none,
    up_w,
    up_s,                     // ← dnnl::memory() if no scale
    up_z));                   // ← dnnl::memory() if no zp
```

### 3.6 Layer 6: `KeepMOE3GemmConstPrecision` — No Change Required

When shared weights are F16:
- They won't match `type_matches(ov::element::u4)` → pattern doesn't fire for them
- `ConvertPrecision` won't touch them (F16 is already the inference precision on GPU)
- No intervention needed

---

## 4. Execution Flow (After Changes)

### 4.1 Decode Path (Single Token)

```
Submit 1: mlp_gate_up  gws={MAX_TOPK + 1, SUBGROUP_SIZE, INTER_SIZE/N_BLOCK}
  ├── Workgroup 0..MAX_TOPK-1: sparse experts
  │     → gate_up_gemv_n2x_u4() (i4 dequant + scale/zp)
  └── Workgroup MAX_TOPK: shared expert
        → gate_up_gemv_n2x_f16() (raw f16, no dequant)

Submit 2: mlp_down  gws={MAX_TOPK + 1, SUBGROUP_SIZE, HIDDEN_SIZE/N_BLOCK}
  ├── Workgroup 0..MAX_TOPK-1: sparse experts
  │     → down_gemv_n2x_u4() * routing_weight
  └── Workgroup MAX_TOPK: shared expert
        → down_gemv_n2x_f16() * sigmoid_gate

Submit 3: mlp_reduce  gws={1, HIDDEN_SIZE}
  → sum across MAX_TOPK + 1 experts
```

**Total submits: 3 (unchanged)**

### 4.2 Prefill Path

```
grouped_gemm / micro_gemm / onednn_loop:
  → Only sparse experts (E groups)
  → Uses compressed weights (i4/i8 + scale + zp)

execute_shared_expert():
  → onednn_linear with f16 weights, no scale/zp
  → ic_group_size = -1
```

---

## 5. Performance Impact Analysis

### 5.1 Decode (Critical Path)

| Factor | Impact | Severity |
|--------|--------|----------|
| Extra submit | 0 (workgroup branch) | None |
| SIMD divergence | 0 (all threads in a WG take same path) | None |
| Kernel binary size | +1 GEMV function body included | Negligible (I-cache: only executed pages are fetched) |
| Register allocation | Compiler sees both paths, allocates max | Low (f16 GEMV uses FEWER registers than i4) |
| Branch overhead | 1 workgroup-level `if` per kernel | ~0 cycles (branch predictor, uniform across WG) |

**Net impact: Zero measurable performance difference for decode.**

### 5.2 Prefill

- Shared expert was already a separate `onednn_linear` call
- F16 weights with no quantization overhead → potentially **faster** than compressed
- No architectural change

---

## 6. File Change Summary

| File | Change Type | Description |
|------|------------|-------------|
| `src/common/transformations/include/ov_ops/moe_compressed.hpp` | **Extend** | Add `shared_weight_type`, `shared_group_size`, `shared_has_zp` to Config |
| `src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.cpp` | **Modify** | Detect shared expert precision, populate config |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp` | **Modify** | (1) Add `SHARED_WEIGHT_COMPRESSEION_DT` JIT constants; (2) Fix `init_shared_primitives` group_size |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_mlp.cl` | **Modify** | (1) Conditional pointer types for shared; (2) Workgroup-level GEMV dispatch branch |
| `src/plugins/intel_gpu/tests/unit/test_cases/moe_3gemm_gpu_test.cpp` | **Add** | Mixed-precision test cases (shared=f16, sparse=i4/i8) |

---

## 7. Test Plan

### New Test Cases

```cpp
// In moe_3gemm_gpu_test.cpp, add:
// Shared expert F16 + Sparse expert i4 (symmetric)
TEST(moe_3gemm_compressed_gpu_mixed_precision, shared_f16_sparse_i4_seq1)
TEST(moe_3gemm_compressed_gpu_mixed_precision, shared_f16_sparse_i4_seq16)

// Shared expert F16 + Sparse expert u4 (asymmetric)
TEST(moe_3gemm_compressed_gpu_mixed_precision, shared_f16_sparse_u4_seq1)
TEST(moe_3gemm_compressed_gpu_mixed_precision, shared_f16_sparse_u4_seq16)

// Shared expert F16 + Sparse expert i8 (symmetric)
TEST(moe_3gemm_compressed_gpu_mixed_precision, shared_f16_sparse_i8_seq1)
TEST(moe_3gemm_compressed_gpu_mixed_precision, shared_f16_sparse_i8_seq16)
```

### Build & Run

```bash
cd build-x86_64-release
make ov_gpu_unit_tests -j$(nproc)
./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter="*moe*mixed_precision*"
# Also run full MoE suite for regression:
./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter="*moe*"
```

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Workgroup branch (not dual submit) | Zero additional submit overhead; zero SIMD divergence |
| Compile-time `#if SHARED_WEIGHT_COMPRESSEION_DT` (not runtime) | GPU compile-time branch eliminates dead code → smaller register footprint, no branch cost |
| Shared expert pointer type changes via `#if` | OpenCL requires typed pointers; can't reinterpret `uchar*` as `half*` safely at runtime |
| Keep `xg_sum` computation for shared WG | Memory-bound kernel — extra ALU is free; simplifies SLM array sizing |
| Config extension in `MOECompressed::Config` | Clean propagation from transformation → JIT → kernel; no runtime detection needed |
| `ic_group_size = -1` when no scale | Standard oneDNN convention for unquantized matmul |

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Kernel binary bloat (includes both i4 and f16 GEMV) | Both already exist in current .cl file; no new code, just new dispatch path |
| Config serialization break (new fields) | Add proper `save()`/`load()` for new Config fields in primitive |
| `shared_gate_gate_weight` is already F16 | No change needed — it was always treated as f16 (`ConvertPrecision` converts it) |
| Models where shared expert IS compressed | `shared_weight_type` falls through to existing compressed path — backward compatible |

---

## 10. Backward Compatibility

- Models with **uniform precision** (shared expert also compressed): `shared_weight_type == sparse weight type` → `SHARED_WEIGHT_COMPRESSEION_DT == WEIGHT_COMPRESSEION_DT` → same branch, functionally identical
- Models with **no shared expert**: `SHARED_EXPERT_ENABLE == 0` → all shared-related code compiled out, zero impact
- Default `shared_weight_type = ov::element::dynamic`: transformation detects actual type from constant; if not found, inherits sparse type
