# Primitive Scheduler Runtime Implementation Switching — Technical Summary

> **Branch**: `river/enhance_kernel_scheduler` (13 files, +2143/−2 lines vs upstream)  
> **Target Hardware**: Intel Arc B580 (BMG, Xe2) — 20 XC, 8 EU/XC, SG_SIZE=16, 456 GB/s GDDR6  
> **Benchmark Model**: Qwen3-8B (hidden=4096, intermediate=12288, 36 layers, 4-bit WOQ)

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Original Problems](#2-original-problems)
3. [Solution 1: Runtime Implementation Switching](#3-solution-1-runtime-implementation-switching)
4. [Solution 2: Sub-byte Weight Compatibility Mechanism](#4-solution-2-sub-byte-weight-compatibility-mechanism)
5. [Solution 3: M=1 Decode Synthetic Retry](#5-solution-3-m1-decode-synthetic-retry)
6. [Solution 4: W4A8 Dynamic Quantization Handling](#6-solution-4-w4a8-dynamic-quantization-handling)
7. [OCL GEMV Kernel Optimization Details](#7-ocl-gemv-kernel-optimization-details)
8. [E2E Verification Results and Analysis](#8-e2e-verification-results-and-analysis)
9. [Unresolved Issues](#9-unresolved-issues)
10. [File Inventory and Code Location Index](#10-file-inventory-and-code-location-index)
11. [Detailed Flowcharts](#11-detailed-flowcharts)

---

## 1. System Architecture Overview

```
┌─────────────────────────── GPU Plugin Runtime ────────────────────────────┐
│                                                                           │
│  primitive_inst::execute()                                                │
│      │                                                                    │
│      ├─ _switching_policy != NONE?                                        │
│      │   YES ──► select_best_impl_for_inputs()                            │
│      │              │                                                     │
│      │              ├─ extract_selection_criteria()                        │
│      │              │     batch, seq_len, M, K, workload                  │
│      │              │                                                     │
│      │              └─ evaluate_best_impl_type()                          │
│      │                    ├─ PROFILING: EMA latency comparison             │
│      │                    └─ HEURISTIC: M × K vs threshold                │
│      │                                                                    │
│      │   best != current? ──► switch_impl_to(best)                        │
│      │                           └─ _impl = _impl_pool[best]             │
│      │                                                                    │
│      └─ _impl->execute()                                                  │
│                                                                           │
│  ┌─────────────────── ImplPool (built at first prefill) ────────────────┐ │
│  │  impl_types::onednn  → PrimitiveImplOneDNN (prefill-optimized FC)   │ │
│  │  impl_types::ocl     → PrimitiveImplOCL (W4A16 GEMV for decode)   │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  enable_multi_impl_mode() — Pool builder (called at first update_impl)    │
│      ├─ Determine primary impl type                                       │
│      ├─ Compile candidate impls for each GPU-native type                  │
│      ├─ Weight IO Contract check (Rules 1-3)                              │
│      └─ add_impl_to_pool()                                                │
└───────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Core Data Structures

```cpp
// primitive_inst.h

/// Implementation pool — maps impl_type to compiled implementation
using ImplPool = std::unordered_map<impl_types, std::shared_ptr<PrimitiveImpl>>;

/// EMA statistics tracker — per-impl-type exponential moving average latency
struct ImplStats {
    double ema_latency_us;         // Exponential moving average latency (μs)
    int64_t sample_count;          // Number of collected samples
    static constexpr double alpha = 0.3;  // EMA smoothing factor
};

/// Workload predictor — performance characterization of the current call
struct WorkloadPredictor {
    double compute_threshold;      // HEURISTIC mode switching boundary
    // auto-detect: gpu_fp16_gflops × 10
};

/// Extracted feature vector for the current call
struct ImplSelectionCriteria {
    int64_t batch_size, seq_length, M, K;
    double compute_workload;       // M × K
    bool is_prefill;               // seq_length > 1
};
```

### 1.2 Environment Variables

| Variable | Values | Default | Purpose |
|----------|--------|---------|---------|
| `OV_GPU_MULTI_IMPL_POLICY` | NONE / AUTO_HEURISTIC / PROFILING / MANUAL | NONE | Switching policy |
| `OV_GPU_MULTI_IMPL_COMPUTE_THRESHOLD` | Floating-point value | auto-detect | HEURISTIC threshold |

---

## 2. Original Problems

### P1: Registry Static Priority

The `fully_connected_impls.cpp` registry records all candidate impls per op type.  
At compile time the registry is traversed top-to-bottom, and the **first valid** impl is selected permanently:

```
registry order: OneDNN FC → ... → OCL FC (default)
```

Once selected, there is no mechanism to switch at runtime. A single call site cannot use different impls for different workloads.

### P2: Binary OneDNN Gate

The original logic was:
```
if (OneDNN impl exists && validation passes)
    → use OneDNN
else
    → fallback to OCL
```

No middle ground — either all-OneDNN or all-OCL. No threshold-based or per-shape selection possible.

### P3: Compile-time Static Decision

Shape information at graph compile time (typically prefill phase, M >> 1) does not match runtime shapes in the decode phase (M=1).  
A compile-time decision optimized for prefill necessarily underperforms in decode, yet there is no way to revisit the choice.

---

## 3. Solution 1: Runtime Implementation Switching

### 3.1 Design Philosophy

- Build impl pool at first `update_impl` call (prefill phase)
- Evaluate optimal impl per call based on extracted shape features
- Switch in-place with zero memory allocation (pre-compiled impls)

### 3.2 Core Call Chain

```
update_impl()
    └─ enable_multi_impl_mode(policy)      // One-time initialization
         ├─ determine primary_impl_type
         ├─ compile candidates
         └─ build _impl_pool

execute()
    └─ select_best_impl_for_inputs(params)
         ├─ extract_selection_criteria()    // shape → feature extraction
         └─ evaluate_best_impl_type()      // feature → decision
    └─ switch_impl_to(best)                // swap _impl pointer + update weights
    └─ _impl->execute()
```

### 3.3 Threshold Calculation Logic

```cpp
// evaluate_best_impl_type()

// AUTO_HEURISTIC mode:
if (is_prefill && workload > threshold)     → OneDNN   // prefill: large GEMM
if (!is_prefill && workload <= threshold)    → OCL      // decode: small GEMV
if (workload > 2 × threshold)               → OneDNN   // very large → force OneDNN
fallback                                     → OneDNN

// auto-detect threshold:
threshold = gpu_fp16_gflops × 10
// B580: ~152 TFLOPS fp16 → threshold ≈ 1.52e6
// M=1, K=4096: workload = 4096 ≪ 1.52e6 → OCL ✓
// M=128, K=4096: workload = 524,288 < threshold → borderline

// PROFILING mode:
if (both impls have sufficient samples)     → pick lower EMA latency
else                                        → fallback to HEURISTIC
```

### 3.4 Shape-Cache Fast Path

```cpp
// select_best_impl_for_inputs() — O(1) decode path
if (cached_impl != any && input_shape == cached_shape)
    return cached_impl;    // All tokens in decode phase have same shape → direct hit
```

This ensures the decode phase (all tokens share the same M=1 shape) does not repeatedly execute the full `extract_selection_criteria` and `evaluate_best_impl_type` logic.

---

## 4. Solution 2: Sub-byte Weight Compatibility Mechanism

### 4.1 Problem

Both OneDNN and OCL directly read raw u4/i4 packed weight bytes (no weight reorder).  
The **Weight IO Contract**'s Rule 3 originally unconditionally rejected sub-byte cross-backend sharing:

```
Rule 3 (original): !reorder + sub-byte dtype + different backend → reject
```

However, in practice OneDNN WOQ FC and our OCL GEMV kernel both use the **low-nibble-first** packing convention, allowing safe sharing of the same weight buffer.

### 4.2 Solution

Introduce a virtual method `raw_sub_byte_weight_compatible()` to let impl managers declare their sub-byte reading convention:

```cpp
// implementation_manager.hpp — base class defaults to false
virtual bool raw_sub_byte_weight_compatible() const noexcept { return false; }

// fc_compressed_generate_opt.hpp — OCL GEMV declares compatible
bool raw_sub_byte_weight_compatible() const noexcept override { return true; }

// fully_connected_onednn.hpp — OneDNN FC declares compatible
bool raw_sub_byte_weight_compatible() const noexcept override { return true; }
```

Rule 3 is modified to conditional rejection:

```cpp
// primitive_inst.cpp — within enable_multi_impl_mode()
} else if (weight_is_sub_byte && t != primary_type) {
    const bool primary_raw_ok = primary_mgr && primary_mgr->raw_sub_byte_weight_compatible();
    const bool alt_raw_ok = alt_impl->m_manager && alt_impl->m_manager->raw_sub_byte_weight_compatible();
    if (!(primary_raw_ok && alt_raw_ok))
        reject_reason = "rule3: sub-byte weight cross-backend packing incompatible";
    // Both declare compatible → Rule 3 passes ✓
}
```

### 4.3 Edge Case: m_manager is null

When the primary impl is created through the legacy path (graph optimizer selects directly without going through `ImplementationManager::update_impl`), `_impl->m_manager` may be `nullptr`.

Solution: fall back to looking up the first manager of the same type in `m_available_impls`:

```cpp
auto find_first_manager_of_type = [&](impl_types type) -> const ImplementationManager* {
    for (const auto& mgr : _impls_factory->m_available_impls)
        if (mgr->get_impl_type() == type)
            return mgr.get();
    return nullptr;
};
const ImplementationManager* primary_mgr =
    _impl->m_manager ? _impl->m_manager : find_first_manager_of_type(primary_type);
```

---

## 5. Solution 3: M=1 Decode Synthetic Retry

### 5.1 Problem

`enable_multi_impl_mode()` is triggered for the first time during the **prefill phase** (M >> 1). At that point:

1. `create_impl_for_type(ocl, params_with_M=2048)` is called
2. Internally `get_fake_aligned_params_if_possible()` may further inflate M
3. `FCCompressedGenerateOpt::support_shapes(M=2048)` → **false** (M≠1)
4. Falls through to the default OCL FC impl
5. Default OCL FC's `raw_sub_byte_weight_compatible() = false`
6. Rule 3 rejects

→ Even with Rule 3 fixed, the wrong impl causes the check to fail.

### 5.2 Solution

In the Step 3 loop of `enable_multi_impl_mode()`, for candidates with sub-byte weights and non-primary type, add an **M=1 synthetic retry**:

```
enable_multi_impl_mode() Step 3 loop:
  for each candidate_type:
    ├─ create_impl_for_type(*_impl_params, type) → alt_impl
    ├─ Check: weight_is_sub_byte && type != primary?
    │   ├─ alt_impl exists and has raw_sub_byte_weight_compatible()?
    │   │   ├─ Yes → use directly (found correct OCL GEMV)
    │   │   └─ No → need M=1 retry ↓
    │   │
    │   └─ M=1 synthetic retry:
    │       ├─ Copy _impl_params → m1_params
    │       ├─ Set all non-last dims of activation shape to 1
    │       │   [40, 1, 4096] → [1, 1, 4096]
    │       ├─ Sync output shape
    │       ├─ Clear dynamic padding mask
    │       │
    │       ├─ (W4A8 special handling)
    │       │   If dynamic_quantized_activation && activation is i8:
    │       │   Change activation dtype to f16 (dyn_quant skipped during decode)
    │       │
    │       ├─ Directly iterate m_available_impls (bypass fake alignment):
    │       │   for each manager of type:
    │       │     if validate(node) && support_shapes(m1_params):
    │       │       create → compile → check raw_sub_byte_compatible
    │       │       → if OK: alt_impl = m1_impl; break
    │       │
    │       └─ Found → alt_impl replaced with M=1 compiled GEMV kernel
    │
    ├─ Weight IO Contract (Rules 1-3) check
    └─ add_impl_to_pool(type, alt_impl)
```

### 5.3 Why Bypass `create_impl_for_type`

`create_impl_for_type()` internally calls `get_fake_aligned_params_if_possible()`, which inflates M=1 to values like M=16 (for OneDNN alignment optimization). This causes `FCCompressedGenerateOpt::support_shapes(M=16)` to always fail.

The synthetic retry directly iterates `m_available_impls` using exact M=1 parameters, completely bypassing fake alignment.

---

## 6. Solution 4: W4A8 Dynamic Quantization Handling

### 6.1 Problem

When OneDNN uses W4A8 mode:
- Prefill phase: the upstream `dynamic_quantize` node converts f16 activations to i8
- Decode phase (M=1, batch=1): the `dynamic_quantize` node **skips execution** and passes through the original f16

Pool building occurs during the prefill phase, when `_impl_params` has i8 activations.  
If the OCL kernel is compiled with i8 (IS_ACT_INT8=1), it will produce incorrect results when it receives f16 data during the decode phase.

### 6.2 Solution

During the M=1 synthetic retry, detect the `dynamic_quantized_activation` condition and proactively change the activation dtype from i8 to f16:

```cpp
if (get_node().is_type<fully_connected>()) {
    const auto& fc_desc = *m1_params.typed_desc<fully_connected>();
    if (fc_desc.dynamic_quantized_activation &&
        m1_params.input_layouts[0].data_type == data_types::i8) {
        m1_params.input_layouts[0].data_type = data_types::f16;
    }
}
```

This ensures the OCL GEMV kernel in the pool is compiled in W4A16 mode, correctly receiving the f16 activations passed through when `dynamic_quantize` is skipped during the decode phase.

### 6.3 Trigger Condition Separation

```cpp
// Two independent retry trigger conditions
const bool need_retry =          // Found impl is incorrect
    !alt_impl ||
    !(alt_impl->m_manager && alt_impl->m_manager->raw_sub_byte_weight_compatible());

bool need_dyn_quant_retry =      // Found correct impl, but wrong dtype
    !need_retry &&
    desc.dynamic_quantized_activation &&
    input_layouts[0].data_type == data_types::i8;
```

---

## 7. OCL GEMV Kernel Optimization Details

### 7.1 Kernel Architecture

```
gemm_generate_opt.cl — Dual-branch design
├─ Branch A: f16/f16 GEMM (#if !IS_WEIGHT_INT4)
│   Small GEMV, each work-item computes one output
│
└─ Branch B: INT4 WOQ GEMV (#if IS_WEIGHT_INT4)
    ├─ B1: W4A16 (#if !IS_ACT_INT8)
    │   ├─ SLM activation caching: entire K vector → __local
    │   ├─ UNPACK8_UINT: uint read + bitwise unpack
    │   ├─ Dual float8 accumulators (ILP)
    │   ├─ opencl_unroll_hint(4)
    │   └─ TILE_N=1 (one output per WI) / TILE_N=2 (two outputs per WI)
    │
    └─ B2: W4A8 (#if IS_ACT_INT8)
        ├─ Activation is i8, appends per-token activation scale
        └─ Structure similar to B1, multiplied by act_scale at the end
```

### 7.2 Key Optimization Techniques

| Technique | Effect | Status |
|-----------|--------|--------|
| **UNPACK8_UINT**: single uint read + bit-shift unpack of 8 INT4 values | +15% BW vs uchar4 per-byte read | ✅ Deployed |
| **Dual float8 accumulators (ILP)**: `acc0`, `acc1` alternating | +8% BW, keeps EU pipeline saturated | ✅ Deployed |
| **SLM activation caching**: shared K vector within WG | Reduces global memory reads, benefits when multiple subgroups share activation | ✅ Deployed |
| **opencl_unroll_hint(4)**: inner loop | +5% vs no hint | ✅ Deployed |
| **Two-pass K-split**: split into K_SPLIT WGs + reduce | +38–74% for small-N shapes | ⚠ Prototype |
| **uchar4 vs uint**: 128-bit read comparison | uint is better (fewer transactions) | ✅ Chose uint |
| **TILE_N=2**: two outputs per WI | −30% (register pressure) | ❌ Abandoned |
| **half8/half16 accumulators**: halved precision | −20% (precision-related stalls) | ❌ Abandoned |
| **ulong (8-byte) reads**: aligned access | No significant benefit | ❌ Abandoned |
| **Full unroll(8)**: complete unrolling | −10% (register spilling) | ❌ Abandoned |

### 7.3 JIT Constant Generation (fc_compressed_generate_opt.cpp)

```
Fixed constants:                  Dynamic constants (derived from runtime params):
  SG_SIZE = 16                     K_SIZE, N_SIZE, B_SIZE
  VEC_SIZE = 8                     IS_WEIGHT_INT4 = 1
  WG_SIZE = 256                    WEIGHT_IS_SIGNED (i4=1, u4=0)
                                   HAS_ZP, ZP_IS_U8
                                   GROUP_SIZE, NUM_GROUPS = K/GROUP_SIZE
                                   IS_ACT_INT8 (W4A8=1, W4A16=0)

Dispatch: global = {ceil(N/WG_SIZE) × WG_SIZE, B, 1}
          local  = {WG_SIZE, 1, 1}
```

### 7.4 Standalone Kernel Benchmark Results

**Hardware**: Intel Arc B580 (BMG, Xe2) — 20 XC, 456 GB/s GDDR6, SG_SIZE=16

#### Exhaustive Variant Comparison (N=6144, K=4096)

| Variant | Avg GB/s | Peak GB/s | % of 456 GB/s |
|---------|----------|-----------|----------------|
| Raw BW ceiling (minimal compute) | 421 | 435 | **95.4%** |
| float4×4 accumulators | 368 | 373 | 81.8% |
| **K16 dual float8 accumulators (deployed)** | **366** | **371** | **81.4%** |
| uint + unroll(4) baseline | 341 | 347 | 74.8% |
| ulong reads | 341 | 346 | 75.9% |
| half8 accumulators | 286 | 290 | 63.7% |
| TILE_N=2 | 235 | 251 | 55% |

#### Breakdown by Qwen3-8B FC Layer Shape

| FC Layer | N | K | WG Count | BW GB/s | Roofline % | Status |
|----------|---|---|----------|---------|------------|--------|
| gate_proj / up_proj | 12288 | 4096 | 48 | 391 | **88.6%** | ✅ >85% target met |
| qkv_proj | 6144 | 4096 | 24 | 366 | 81.4% | ⚠ Borderline |
| o_proj | 4096 | 4096 | 16 | 248 | 64.8% | ❌ GPU underutilized |
| down_proj | 4096 | 12288 | 16 | 206 | 46.0% | ❌ Severely underutilized |

**Root cause**: N=4096, WG_SIZE=256 → only 16 WGs, but B580 has 20 XC cores → 20% compute idle.

#### K-Split Prototype Results (two-pass)

| FC Layer | Config | Total WGs | BW GB/s | Roofline % | Improvement |
|----------|--------|-----------|---------|------------|-------------|
| **down_proj** | K_SPLIT=4, WG128 | 64 | **347** | **76.2%** | **+71%** |
| **o_proj** | K_SPLIT=4, WG128 | 64 | **377** | **82.7%** | **+74%** |
| **qkv_proj** | K_SPLIT=2, WG256 | 48 | **361** | **79.2%** | **+38%** |
| gate_proj | K_SPLIT=1, WG256 | 48 | 387 | 84.8% | baseline |

---

## 8. E2E Verification Results and Analysis

### 8.1 Qwen3-8B Decode Performance

**Configuration**: 256 output tokens, 6-token input, 3 iterations (excluding warmup)

| Configuration | 2nd Token (ms/tok) | Throughput (tok/s) | vs Baseline |
|---------------|--------------------|--------------------|-------------|
| **Baseline (NONE/OneDNN)** | **13.36** | **74.87** | — |
| AUTO_HEURISTIC (OCL for generate) | 14.26–14.35 | 69.70–70.14 | **−6.9%** |
| Force OneDNN (threshold=−1) | 13.30 | 75.19 | +0.4% |
| PROFILING mode | 14.32 | 69.81 | −6.7% |
| N threshold ≥6144 (reverted) | 14.26 | 70.14 | −6.5% |

### 8.2 Correctness Verification

- ✅ All configurations produce valid text (MD5 checksum verified)
- ✅ No ",,," / "!!!" / "???" artifacts
- ✅ "impl switch onednn → ocl" log messages confirm switching
- ✅ "Enqueue stage gemm_generate_opt" confirms kernel is selected
- ✅ Layer 0 q_proj input/output comparison within tolerance

### 8.3 E2E Regression Root Cause Analysis

| Factor | Contribution | Explanation |
|--------|-------------|-------------|
| **DPAS/XMX instruction advantage** | Primary | OneDNN uses DPAS systolic array processing ~128 INT4/instruction vs OCL vector EU ~16/iteration |
| **Per-dispatch overhead difference** | Secondary | OCL path incurs ~8μs overhead per FC dispatch × 180 times = ~1.4ms/token |
| **Small-N GPU underutilization** | Secondary | N=4096 FC layers (o_proj, down_proj) have only 16 WGs / 20 XC |

**Key finding**: Framework switching overhead is **not** the bottleneck. The threshold=−1 test (force OneDNN but via the AUTO_HEURISTIC path) yields 13.30ms ≈ NONE baseline, proving the ImplPool mechanism itself incurs < 0.5% overhead.

---

## 9. Unresolved Issues

### 9.1 OCL Has Not Yet Surpassed OneDNN (SKILL.md Requirement Not Met)

**Requirement**: "OCL implementation has better performance than OneDNN implementation in generate stage"  
**Status**: OCL is ~7% slower than OneDNN

**Follow-up Approaches** (by priority):

| Approach | Expected Impact | Complexity |
|----------|-----------------|------------|
| **K-Split framework integration**: Support two-stage dispatch (GEMV + reduce) in PrimitiveImplOCL | +38–74% BW for small-N, may close the E2E gap but uncertain whether it can surpass OneDNN | Medium |
| **DPAS Kernel**: Rewrite using `cl_intel_subgroup_matrix_multiply_accumulate` instructions | Theoretical 2–4× instruction count reduction → may surpass OneDNN | High |
| **Dispatch overhead reduction**: kernel fusion / persistent kernel | Reduce ~1.4ms/token overhead | High |

### 9.2 Memory Optimization (SKILL.md Phase 2/3 Not Implemented)

- Hot-Swap Memory Management (design only, not coded)
- Resource Pooling (proof-of-concept only)
- Lazy Compilation (not implemented)
- LRU Weights Cache (not implemented)

### 9.3 Dynamic TILE_N Selection

Currently JIT does not set TILE_N (kernel defaults to 1). For very large N, TILE_N=2 might be beneficial, but current tests show register pressure causes a 30% performance drop.

---

## 10. File Inventory and Code Location Index

### 10.1 Production Code (13 files, +2143 / −2 lines)

| File | Line Changes | Core Function |
|------|-------------|---------------|
| `primitive_inst.cpp` | +917 | Scheduling engine: `enable_multi_impl_mode`, `evaluate_best_impl_type`, `switch_impl_to`, M=1 retry, Weight IO rules |
| `primitive_inst.h` | +90 | `ImplPool`, `WorkloadPredictor`, `ImplSelectionCriteria`, API declarations |
| `gemm_generate_opt.cl` | +384 | GEMV kernel (f16/f16 + W4A16 + W4A8), UNPACK8_UINT, dual accumulators, SLM |
| `fc_compressed_generate_opt.cpp` | +281 | JIT generator: WG_SIZE=256, SG_SIZE=16, dynamic derivation of K/N/B/GROUP_SIZE |
| `fc_compressed_generate_opt.hpp` | +177 | `FCCompressedGenerateOpt` IM: validate, support_shapes, raw_sub_byte |
| `gemm_generate_opt.cpp` | +125 | Non-compressed GEMM JIT path |
| `gemm_generate_opt.hpp` | +139 | Non-compressed GEMM validation |
| `fully_connected_onednn.hpp` | +5 | `raw_sub_byte_weight_compatible() = true` |
| `fully_connected_impls.cpp` | +6/−2 | FCCompressedGenerateOpt registration (after OneDNN, before default OCL) |
| `gemm_impls.cpp` | +6 | GEMM OCL impl registration |
| `implementation_manager.hpp` | +9 | Virtual method `raw_sub_byte_weight_compatible()` |
| `internal_properties.hpp` | +3 | Property declarations |
| `options.inl` | +3 | Option registration |

### 10.2 Test Code (gemv_test/w4a16/ — 24 files)

| File | Purpose |
|------|---------|
| `w4a16_test.cpp` | Main test framework: 500 warmup + 500 timed, wall-clock + CL profiling |
| `w4a16_gemv.cl` | Best standalone kernel (matches production) |
| `bw_bench.cpp` + `bw_test.cl` | Raw bandwidth ceiling test |
| `ksplit_bench.cpp` + `w4a16_ksplit.cl` | Two-pass K-split benchmark |
| `ksplit_intra_bench.cpp` + `w4a16_ksplit_intra.cl` | Intra-WG K-split (SLM reduce) |
| Multiple `w4a16_gemv_*.cl` | Kernel variants (block, f4acc, fullunroll, h16acc, kpar, nounroll, ulong, wg512) |

### 10.3 Git Commit History

```
380023493b Add FCCompressedGenerateOpt OCL W4A8 kernel implementation
cf386457bb Debug ocl_gemm kernel
c6265d84fe Add i8 support for gemm_ocl
fa43e17dac Fix gemm ocl impl cannot be chosen issue
610b56489a Add ocl type gemm_impl
6c6db79d2a Add_impl_pool policy for different impls
fbfcc26721 Runtime Primitive Implementation Switching Policy
```

---

## 11. Detailed Flowcharts

> Detailed flowcharts are saved as separate files in the same directory in Mermaid and SVG formats:
> - `flowchart_enable_multi_impl.md` — enable_multi_impl_mode() complete flow
> - `flowchart_runtime_switching.md` — Runtime impl switching decision flow
> - `flowchart_overall_architecture.md` — Overall architecture and data flow

### 11.1 enable_multi_impl_mode() Flow (Simplified)

```
enable_multi_impl_mode(policy)
│
├─ _switching_policy != NONE?  ─── Yes → return (already initialized)
│
├─ Step 1: Determine primary impl type
│   └─ _impl->m_manager ? manager->get_impl_type()
│                         : (is_onednn ? onednn : ocl)
│
├─ Step 2: Collect candidate types
│   for each mgr in m_available_impls:
│     └─ GPU-native? → not seen? → has_impl_for(node, type, static)?
│                                   → candidate_types.push_back(type)
│
├─ MANUAL policy? → parse_manual_impl → no rule? → return
│
├─ OneDNN fused post-ops? → non-OneDNN cannot handle → return
│
├─ add_impl_to_pool(primary_type, _impl)
│
└─ Step 3: for each candidate_type:
    ├─ create_impl_for_type(params, type)
    │
    ├─ Sub-byte weight + type ≠ primary?
    │   ├─ need_retry? (alt doesn't exist / not raw-sub-byte-compatible)
    │   │   └─ YES → M=1 synthetic retry (bypass fake alignment)
    │   ├─ need_dyn_quant_retry? (W4A8 dtype correction)
    │   │   └─ YES → M=1 synthetic retry + f16 dtype
    │   └─ Found correct FCCompressedGenerateOpt
    │
    ├─ Weight IO Contract check:
    │   ├─ Rule 1: reorder flag consistency
    │   ├─ Rule 2: reorder source layout consistency
    │   └─ Rule 3: sub-byte compatibility (raw_sub_byte_weight_compatible)
    │
    └─ add_impl_to_pool(type, alt_impl)
```

### 11.2 Runtime Switching Decision Flow (Simplified)

```
execute()
│
├─ _switching_policy != NONE && _impl_pool != null?
│   │
│   YES → select_best_impl_for_inputs(params)
│   │   ├─ shape-cache hit? → return cached_impl (O(1))
│   │   │
│   │   ├─ extract_selection_criteria():
│   │   │   batch_size, seq_length, M, K, compute_workload, is_prefill
│   │   │
│   │   └─ evaluate_best_impl_type():
│   │       ├─ PROFILING + sufficient samples? → pick lower EMA latency
│   │       ├─ is_prefill && workload > threshold? → OneDNN
│   │       ├─ !is_prefill && workload ≤ threshold? → OCL
│   │       ├─ workload > 2×threshold? → OneDNN
│   │       └─ fallback → OneDNN
│   │
│   ├─ best ≠ current?
│   │   └─ switch_impl_to(best):
│   │       ├─ _impl = pool[best]
│   │       ├─ update_weights()
│   │       ├─ OOO queue? → discard null event
│   │       ├─ set_arguments()
│   │       └─ log: "impl switch prev → best"
│   │
│   └─ OOO queue + actually switched? → stream.finish()
│
├─ PROFILING? → record start time
│
├─ _impl->execute()
│
└─ PROFILING? → record end time → update EMA stats
```
