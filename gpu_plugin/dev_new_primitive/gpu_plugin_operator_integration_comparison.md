# Three Approaches for Integrating Custom Operators in OpenVINO GPU Plugin

**Document Version:** 1.0  
**Date:** March 11, 2026  
**Use Case:** MatMul→RMSNorm→MatMul Fusion Integration  

---

## Executive Summary

This document compares three architectural approaches for integrating custom operators into the OpenVINO Intel GPU plugin, specifically analyzing the implementation of a fused `MatMul→RMSNorm→MatMul` operator. The analysis covers file modifications, development effort, architectural complexity, and maintenance considerations.

**Key Findings:**
- **Approach 2 (Fused Primitives)**: 82% less work, recommended for production
- **Approach 3 (OCL_V2)**: 31% faster than traditional approach, best for new operators
- **Approach 1 (Traditional OCL)**: Legacy architecture, avoid for new development

---

## Table of Contents

1. [Overview of Three Approaches](#overview-of-three-approaches)
2. [Complete File Modification Comparison](#complete-file-modification-comparison)
3. [Overall Statistics](#overall-statistics)
4. [Development Effort Breakdown](#development-effort-breakdown)
5. [Architectural Analysis](#architectural-analysis)
6. [Complexity Assessment](#complexity-assessment)
7. [Recommendation Matrix](#recommendation-matrix)

---

## Overview of Three Approaches

### Approach 1: Traditional OCL with Full Custom Operator Stack

A complete end-to-end implementation creating a new `MatRmsMul` operator that traverses all five architectural layers: from OpenVINO IR to OpenCL kernel execution. This follows the legacy kernel_selector architecture.

**Architecture Flow:**
```
OV IR → Plugin Frontend → Primitive Definition → Kernel Selector → OCL Implementation
  ↓           ↓                    ↓                    ↓                  ↓
New Op → Translation Logic → Primitive Struct → JIT Gen + Selector → typed_primitive_impl_ocl
```

**Characteristics:**
- Complete vertical stack traversal
- Heavyweight kernel_selector layer with GetDispatchData(), GetJitConstants(), Validate()
- Manual registration via REGISTER_OCL_IMPL macro
- Complex parameter passing through multiple abstraction layers
- **Effort:** 26 person-days
- **Complexity:** ⭐⭐⭐⭐⭐

---

### Approach 2: Fused Primitives Mechanism (Recommended for Production)

Leverages OpenVINO's built-in post-ops fusion framework. Instead of creating a new operator, attaches the first MatMul and RMSNorm as fused operations to the second MatMul node, enabling automatic kernel code generation through JIT macros.

**Architecture Flow:**
```
OV IR (existing) → Fusion Pass → Existing Primitive + Fused Desc → Existing Kernel + JIT Extension
                      ↓
                Only modifies graph optimizer layer
```

**Characteristics:**
- Zero new primitive types
- Leverages existing high-performance GEMM kernels (including OneDNN/XMX)
- Fusion logic injected via JIT macro expansion at kernel compile time
- Minimal code footprint
- **Effort:** 4.5 person-days
- **Complexity:** ⭐⭐

---

### Approach 3: Modern OCL_V2 Architecture

Uses the modernized OCL_V2 implementation framework which eliminates the kernel_selector layer, provides a cleaner KernelGenerator abstraction, and uses the Registry-based implementation management system.

**Architecture Flow:**
```
OV IR → Plugin Frontend → Primitive Definition → OCL_V2 Implementation → Registry
  ↓           ↓                    ↓                    ↓                  ↓
New Op → Translation Logic → Primitive Struct → PrimitiveImplOCL + .cl → ImplementationManager
                                        ↓
                            (kernel_selector layer eliminated!)
```

**Characteristics:**
- Eliminates kernel_selector directory structure
- Uses KernelGenerator abstraction with cleaner interfaces
- Registry<T> pattern for implementation management
- Native multi-stage support (e.g., mean→variance→normalize)
- Files co-located in ocl_v2/ directory
- **Effort:** 18 person-days
- **Complexity:** ⭐⭐⭐⭐

---

## Complete File Modification Comparison

### Layer 1: Graph Optimization & Frontend Translation

| File Path | Approach 1 | Approach 2 | Approach 3 | Description |
|-----------|-----------|-----------|-----------|-------------|
| `src/plugins/intel_gpu/include/intel_gpu/op/matrmsmul.hpp` | 🆕 New | ❌ Not needed | 🆕 New | Internal op definition header with validate_and_infer_types |
| `src/plugins/intel_gpu/src/plugin/op/matrmsmul.cpp` | 🆕 New | ❌ Not needed | 🆕 New | Internal op implementation with shape inference logic |
| `src/plugins/intel_gpu/src/plugin/transformations/matrmsmul_fusion.hpp` | 🆕 New | ❌ Not needed | 🆕 New | Fusion pass header declaring MatcherPass |
| `src/plugins/intel_gpu/src/plugin/transformations/matrmsmul_fusion.cpp` | 🆕 New | ❌ Not needed | 🆕 New | Pattern matching: wrap_type<MatMul>(wrap_type<RMSNorm>(wrap_type<MatMul>())) |
| `src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp` | 🔧 Modify | ❌ Not needed | 🔧 Modify | Register pass: manager.register_pass<MatRmsMulFusion>() |
| `src/plugins/intel_gpu/src/plugin/ops/matrmsmul.cpp` | 🆕 New | ❌ Not needed | 🆕 New | CreateMatRmsMulOp() - OV Node → cldnn::topology translation |
| `src/plugins/intel_gpu/src/plugin/ops/CMakeLists.txt` | 🔧 Modify | ❌ Not needed | 🔧 Modify | Add matrmsmul.cpp to compilation target list |
| `src/plugins/intel_gpu/src/graph/graph_optimizer/prepare_primitive_fusing.cpp` | ❌ Not needed | 🔧 Modify (~50-100 lines) | ❌ Not needed | Add fusion pattern detection and fused_primitive_desc attachment |
| **Subtotal** | **5 new + 2 modified** | **1 modified** | **5 new + 2 modified** | |
| **Estimated Effort** | **3 person-days** | **0.5 person-days** | **3 person-days** | |

---

### Layer 2: GPU Primitive & Graph Node Definition

| File Path | Approach 1 | Approach 2 | Approach 3 | Description |
|-----------|-----------|-----------|-----------|-------------|
| `src/plugins/intel_gpu/include/intel_gpu/primitives/matrmsmul.hpp` | 🆕 New | ❌ Not needed | 🆕 New | struct matrmsmul : public primitive_base<matrmsmul> with all runtime parameters |
| `src/plugins/intel_gpu/src/graph/matrmsmul_inst.hpp` | 🆕 New | ❌ Not needed | 🆕 New | typed_program_node<matrmsmul> and typed_primitive_inst<matrmsmul> declarations |
| `src/plugins/intel_gpu/src/graph/matrmsmul_inst.cpp` | 🆕 New | ❌ Not needed | 🆕 New | calc_output_layouts(), get_shape_infer_dependencies(), BIND_BINARY_BUFFER_WITH_TYPE |
| `src/plugins/intel_gpu/include/intel_gpu/graph/program.hpp` | 🔧 Modify | ❌ Not needed | 🔧 Modify | Forward declaration: struct matrmsmul; |
| `src/plugins/intel_gpu/src/graph/CMakeLists.txt` | 🔧 Modify | ❌ Not needed | 🔧 Modify | Add matrmsmul_inst.cpp to sources |
| **Subtotal** | **3 new + 2 modified** | **None** | **3 new + 2 modified** | |
| **Estimated Effort** | **2 person-days** | **0 person-days** | **2 person-days** | |

---

### Layer 3: Kernel Implementation & Selector (Critical Architectural Difference)

#### Traditional Kernel Selector Path (Approach 1)

| File Path | Approach 1 | Description |
|-----------|-----------|-------------|
| `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/matrmsmul_ref.cl` | 🆕 New | Full OpenCL kernel: KERNEL(matrmsmul_ref)(...) with MatMul+RMS+MatMul logic using __local memory |
| `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/CMakeLists.txt` | 🔧 Modify | Add matrmsmul_ref.cl to KERNEL_CL_FILES list |
| `src/plugins/intel_gpu/src/kernel_selector/core/actual_kernels/matrmsmul/matrmsmul_kernel_base.h` | 🆕 New | class MatRmsMulKernelBase : public KernelBaseOpenCL with params/optional_params |
| `src/plugins/intel_gpu/src/kernel_selector/core/actual_kernels/matrmsmul/matrmsmul_kernel_base.cpp` | 🆕 New | Implement GetJitConstants() and GetCommonDispatchData() |
| `src/plugins/intel_gpu/src/kernel_selector/core/actual_kernels/matrmsmul/matrmsmul_kernel_ref.h` | 🆕 New | class MatRmsMulKernelRef : public MatRmsMulKernelBase |
| `src/plugins/intel_gpu/src/kernel_selector/core/actual_kernels/matrmsmul/matrmsmul_kernel_ref.cpp` | 🆕 New | Implement GetDispatchData(), GetKernelsPriority(), Validate() with NDRange calculations |
| `src/plugins/intel_gpu/src/kernel_selector/core/actual_kernels/matrmsmul/matrmsmul_kernel_selector.h` | 🆕 New | class matrmsmul_kernel_selector : public kernel_selector_base |
| `src/plugins/intel_gpu/src/kernel_selector/core/actual_kernels/matrmsmul/matrmsmul_kernel_selector.cpp` | 🆕 New | Constructor with Attach<MatRmsMulKernelRef>() and GetBestKernels() |
| `src/plugins/intel_gpu/src/kernel_selector/kernel_selector.h` | 🔧 Modify | Add DECLARE_KERNEL_SELECTOR(matrmsmul) |
| `src/plugins/intel_gpu/src/kernel_selector/kernel_selector.cpp` | 🔧 Modify | Add DEFINE_KERNEL_SELECTOR(matrmsmul) |
| `src/plugins/intel_gpu/src/kernel_selector/CMakeLists.txt` | 🔧 Modify | Add all matrmsmul kernel .cpp files to sources |
| **Subtotal** | **8 new + 4 modified** | |
| **Estimated Effort** | **8 person-days** | ~600 LOC of boilerplate, complex NDRange tuning |

#### OCL_V2 Modern Path (Approach 3)

| File Path | Approach 3 | Description |
|-----------|-----------|-------------|
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/matrmsmul_ref.cl` | 🆕 New | OpenCL kernel with JIT macro placeholders (INPUT0_GET_INDEX, OUTPUT0_TYPE, etc.) |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/matrmsmul_ref.hpp` | 🆕 New | class MatRmsMulRef : public PrimitiveImplOCL with Stage definitions |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/matrmsmul_ref.cpp` | 🆕 New | Implement using KernelGenerator: get_arguments_desc(), get_jit_constants(), get_dispatch_data_func() |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/matrmsmul_base.hpp` | 🆕 New (optional) | Shared base class if multiple implementations (ref/opt) need common logic |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/matrmsmul_opt.cl` | 🆕 New (optional) | Optimized kernel with tiling/blocking strategies |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/matrmsmul_opt.hpp` | 🆕 New (optional) | class MatRmsMulOpt : public PrimitiveImplOCL |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/matrmsmul_opt.cpp` | 🆕 New (optional) | Optimized implementation with shape-specific dispatch strategies |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/CMakeLists.txt` | 🔧 Modify | Add matrmsmul_ref.cpp (and _opt.cpp if exists) to sources |
| **Subtotal** | **4-7 new + 1 modified** | |
| **Estimated Effort** | **4 person-days** | ~300 LOC, cleaner API |

#### GEMM Fusion Extension (Approach 2)

| File Path | Approach 2 | Description |
|-----------|-----------|-------------|
| `src/plugins/intel_gpu/src/kernel_selector/core/actual_kernels/gemm/gemm_kernel_base.cpp` | 🔧 Modify (~30 lines) | Extend GetJitConstants() to handle RMSNorm in fused_ops list |
| `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/gemm_ref.cl` | 🔧 Modify (~5 lines) | Add #if HAS_FUSED_OPS ... FUSED_OPS_RESULT ... #endif block |
| `src/plugins/intel_gpu/src/kernel_selector/core/actual_kernels/gemm/fused_ops/matmul_rms_matmul.hpp` | 🆕 New (optional) | Encapsulate complex fusion logic if needed |
| **Subtotal** | **2 modified + 0-1 new** | |
| **Estimated Effort** | **1 person-day** | Minimal changes, maximum reuse |

---

### Layer 4: Implementation Binding & Registration

#### Traditional OCL Binding (Approach 1)

| File Path | Approach 1 | Description |
|-----------|-----------|-------------|
| `src/plugins/intel_gpu/src/graph/impls/ocl/matrmsmul.cpp` | 🆕 New | struct matrmsmul_impl : public typed_primitive_impl_ocl<matrmsmul> with kernel selector binding |
| `src/plugins/intel_gpu/src/graph/impls/ocl/register.hpp` | 🔧 Modify | Add DECLARE_OCL_IMPL_FACTORY(matrmsmul) |
| `src/plugins/intel_gpu/src/graph/impls/ocl/CMakeLists.txt` | 🔧 Modify | Add matrmsmul.cpp to sources |
| **Subtotal** | **1 new + 2 modified** | |
| **Estimated Effort** | **2 person-days** | |

#### OCL_V2 Registry Mechanism (Approach 3)

| File Path | Approach 3 | Description |
|-----------|-----------|-------------|
| `src/plugins/intel_gpu/src/graph/registry/matrmsmul_impls.cpp` | 🆕 New | Registry<matrmsmul>::get_implementations() returning vector<ImplementationManager> |
| `src/plugins/intel_gpu/src/graph/registry/CMakeLists.txt` | 🔧 Modify | Add matrmsmul_impls.cpp to sources |
| **Subtotal** | **1 new + 1 modified** | |
| **Estimated Effort** | **1 person-day** | Cleaner registration pattern |

---

### Layer 5: Testing

| File Path | Approach 1 | Approach 2 | Approach 3 | Description |
|-----------|-----------|-----------|-----------|-------------|
| `src/plugins/intel_gpu/tests/unit/test_matrmsmul.cpp` | 🆕 New | ❌ Not needed | 🆕 New | Unit tests: basic functionality, shape combinations, transpose flags, numerical accuracy |
| `src/plugins/intel_gpu/tests/functional/shared_tests_instances/single_layer_tests/matrmsmul.cpp` | 🆕 New | ❌ Not needed | 🆕 New | Parameterized tests: FP32/FP16/FP64, dynamic shapes, CPU plugin comparison |
| `src/plugins/intel_gpu/tests/unit/transformations/matrmsmul_fusion_test.cpp` | 🆕 New | ❌ Not needed | 🆕 New | Test fusion pass: pattern matching triggers correctly, no fusion when intermediate has users |
| `src/plugins/intel_gpu/tests/unit/CMakeLists.txt` | 🔧 Modify | ❌ Not needed | 🔧 Modify | Add test_matrmsmul.cpp to test target |
| `src/plugins/intel_gpu/tests/functional/shared_tests_instances/single_layer_tests/gemm_fused.cpp` | ❌ Not needed | 🆕 New (optional) | ❌ Not needed | Integration tests for fused GEMM patterns |
| **Subtotal** | **3 new + 1 modified** | **0-1 new** | **3 new + 1 modified** | |
| **Estimated Effort** | **3 person-days** | **1 person-day** | **3 person-days** | |

---

## Overall Statistics

| Metric | Approach 1 (Traditional OCL) | Approach 2 (Fused Primitives) | Approach 3 (OCL_V2) | Delta (3 vs 1) |
|--------|----------------------------|------------------------------|-------------------|----------------|
| **New Files** | 🆕 20 files | 🆕 0-2 files | 🆕 12-15 files | **−25% to −40%** |
| **Modified Files** | 🔧 11 files | 🔧 3 files | 🔧 7 files | **−36%** |
| **Total File Operations** | 31 files | 3-5 files | 19-22 files | **−29% to −39%** |
| **Core Code Volume** | ~3000-4000 LOC | ~150-300 LOC | ~1500-2500 LOC | **−37% to −50%** |
| **Architectural Layers** | 5 complete layers | 1-2 layers | 4 layers (no kernel_selector) | **−1 layer** |
| **CMakeLists Changes** | 5 locations | 0 locations | 3 locations | **−40%** |
| **Learning Curve** | ⭐⭐⭐⭐⭐ Very steep | ⭐⭐ Low | ⭐⭐⭐⭐ Medium-high | Moderate |
| **Maintainability** | ❌ Full-stack maintenance | ✅ Follows mainline | ✅ Modern architecture | High |
| **Future-Proof** | ⚠️ Legacy architecture | ✅ Stable API | ✅ Official direction | Yes |

---

## Development Effort Breakdown

| Development Phase | Approach 1 | Approach 2 | Approach 3 | Notes |
|-------------------|-----------|-----------|-----------|-------|
| **Graph optimization & translation** | 3 days | 0.5 days | 3 days | Approach 2 skips custom op creation |
| **Primitive definition** | 2 days | 0 days | 2 days | Approach 2 reuses existing primitives |
| **Kernel implementation** | 8 days | 1 day | 4 days | OCL_V2 simplifies by 50% |
| **Registration & binding** | 2 days | 0 days | 1 day | Registry cleaner than macros |
| **Test development** | 3 days | 1 day | 3 days | Equal test coverage required |
| **Debugging & optimization** | 5 days | 1 day | 3 days | Fewer layers = faster debug |
| **Documentation** | 1 day | 0.5 days | 1 day | API documentation |
| **Code review iterations** | 2 days | 0.5 days | 1 day | More files = more review cycles |
| **Total Effort** | **26 days** | **4.5 days** | **18 days** | |
| **Efficiency vs Approach 1** | Baseline | **82% faster** | **31% faster** | |

**Key Insight:** Approach 2 delivers the same functionality with 82% less effort by eliminating the need to create new primitives, kernels, and implementations.

---

## Architectural Analysis

### Approach 1: Five-Layer Full Stack Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: OpenVINO IR                                            │
│ - ov::op::internal::MatRmsMul (new internal op)                │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Plugin Frontend                                        │
│ - CreateMatRmsMulOp() translation                              │
│ - Fusion pass: pattern matching MatMul→RMS→MatMul              │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Primitive Definition                                   │
│ - cldnn::matrmsmul primitive struct                            │
│ - matrmsmul_node, matrmsmul_inst                               │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: Kernel Selector (Heavy Abstraction)                   │
│ - MatRmsMulKernelBase                                          │
│ - MatRmsMulKernelRef with GetDispatchData(), Validate()        │
│ - matrmsmul_kernel_selector with scoring                       │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 5: OCL Implementation                                     │
│ - typed_primitive_impl_ocl<matrmsmul>                          │
│ - Binds to kernel selector                                     │
└─────────────────────────────────────────────────────────────────┘
```

**Pain Points:**
- 600+ LOC of boilerplate in kernel_selector layer
- Complex parameter marshalling across 6+ intermediate structures
- JIT macro string concatenation with no compile-time checking
- Easy to introduce bugs in NDRange calculations

---

### Approach 2: Fusion Framework Reuse

```
┌─────────────────────────────────────────────────────────────────┐
│ OpenVINO IR (unchanged)                                         │
│ - Standard MatMul, RMSNorm operators                            │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Graph Optimizer (minimal change ~100 LOC)                      │
│ - Pattern detection in prepare_primitive_fusing.cpp            │
│ - Attach RMS+MatMul1 as fused_primitive_desc to MatMul2        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Existing GEMM Primitive (no new type)                          │
│ - cldnn::gemm with fused_desc vector populated                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Existing GEMM Kernel Selector (minor JIT extension ~30 LOC)    │
│ - GetJitConstants() reads fused_ops list                       │
│ - Generates FUSED_OPS macro definitions                        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Existing GEMM Kernels (OneDNN/XMX/Tiled variants)              │
│ - gemm_ref.cl with #if HAS_FUSED_OPS ... FUSED_OPS ... #endif │
│ - Automatically inherits all GEMM optimizations                 │
└─────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Zero new primitives or implementations
- Automatic optimization inheritance (if GEMM gets faster, fusion gets faster)
- Minimal maintenance burden
- Production-proven fusion framework

---

### Approach 3: OCL_V2 Modern Simplification

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: OpenVINO IR                                            │
│ - ov::op::internal::MatRmsMul (same as Approach 1)             │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Plugin Frontend                                        │
│ - CreateMatRmsMulOp() translation (same as Approach 1)         │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Primitive Definition                                   │
│ - cldnn::matrmsmul (same as Approach 1)                        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: OCL_V2 Implementation (kernel_selector ELIMINATED)    │
│                                                                 │
│ class MatRmsMulRef : public PrimitiveImplOCL {                 │
│     Stage::Ptr stage = make_stage<MatRmsMulGenerator>();       │
│                                                                 │
│     class MatRmsMulGenerator : public KernelGenerator {        │
│         Arguments get_arguments_desc() { ... }                 │
│         JitConstants get_jit_constants() { ... }               │
│         DispatchDataFunc get_dispatch_data_func() { ... }      │
│     }                                                           │
│ }                                                               │
│                                                                 │
│ Files co-located in ocl_v2/:                                   │
│ - matrmsmul_ref.cl, matrmsmul_ref.hpp, matrmsmul_ref.cpp      │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 5: Registry-Based Selection                              │
│ - Registry<matrmsmul>::get_implementations()                   │
│ - Returns vector<ImplementationManager> for ref/opt variants   │
└─────────────────────────────────────────────────────────────────┘
```

**Improvements over Approach 1:**
- Eliminates 600 LOC kernel_selector boilerplate
- Cleaner KernelGenerator API (declarative vs imperative)
- Files organized in single directory
- Multi-stage support built-in (e.g., mean→variance→normalize)
- Registry pattern enables shape-type-specific selection

---

## Complexity Assessment

### Approach 1: Traditional OCL Pain Points

#### 1. Kernel Selector Boilerplate Hell (~600 LOC)

Must implement 8+ virtual methods:
```cpp
class MatRmsMulKernelRef : public MatRmsMulKernelBase {
    ParamsKey GetSupportedKey() const override;  // 20 LOC feature flags
    KernelsData GetKernelsData(const Params& params) const override;  // 80 LOC
    JitConstants GetJitConstants(const params&) const override;  // 100 LOC
    DispatchData GetDispatchData(const params&) const override;  // 150 LOC with NDRange tuning
    bool Validate(const Params& params) const override;  // 50 LOC validation logic
    KernelsPriority GetKernelsPriority() const override;  // 10 LOC
    // + 6 more methods for multi-output, optional params, etc.
};
```

**Issue:** Massive boilerplate with repetitive patterns across all custom kernels.

#### 2. Parameter Passing Nightmare

Data flows through 6+ intermediate structures:
```cpp
ov::Node attributes
    ↓ [CreateMatRmsMulOp()]
cldnn::matrmsmul primitive
    ↓ [matrmsmul_impl::create()]
kernel_selector::matrmsmul_params
    ↓ [MatRmsMulKernelSelector::GetBestKernels()]
kernel_selector::KernelData
    ↓ [typed_primitive_impl_ocl constructor]
cldnn::kernel
    ↓ [execute()]
OpenCL clKernel
```

**Issue:** Easy to lose attributes during translation; each layer has different naming conventions.

#### 3. JIT Macro String Concatenation

```cpp
JitConstants MatRmsMulKernelRef::GetJitConstants(...) const {
    JitConstants jit;
    jit.AddConstant(MakeJitConstant("EPSILON", params.epsilon));
    jit.AddConstant(MakeJitConstant("TRANSPOSE_A", params.transpose_a ? 1 : 0));
    jit.AddConstant(MakeJitConstant("INPUT0_TYPE", toCLType(params.inputs[0].GetDType())));
    // ... 50 more lines of string generation
    return jit;
}
```

**Issue:** No compile-time checking; typos in macro names cause runtime OpenCL compilation failures with cryptic error messages.

#### 4. NDRange Calculation Complexity

```cpp
DispatchData GetDispatchData(const params& p) const override {
    DispatchData dispatchData;
    
    // Must calculate optimal work-group sizes considering:
    // - Sub-group size (8/16/32 depending on architecture)
    // - L3 cache size and blocking factor
    // - Register pressure vs occupancy tradeoff
    // - Local memory usage
    
    const auto& output = p.outputs[0];
    size_t M = output.Y().v;
    size_t N = output.X().v;
    
    // Complex heuristics for choosing global/local work sizes
    dispatchData.gws = {RoundUp(N, 16), RoundUp(M, 8), output.Batch().v};
    dispatchData.lws = {16, 8, 1};
    
    // ... 100 more lines of tuning logic
    return dispatchData;
}
```

**Issue:** Requires deep understanding of GPU architecture; easy to create suboptimal kernels.

---

### Approach 2: Fusion Framework Advantages

#### 1. Automatic Optimization Inheritance

```cpp
// In prepare_primitive_fusing.cpp (~100 LOC addition)
if (matmul2_node.has_pattern(matmul1 → rms → matmul2)) {
    matmul2_node.add_fused_primitive(matmul1);
    matmul2_node.add_fused_primitive(rms);
    p.mark_optimized_out(matmul1, rms);
}

// That's it! MatMul2 now automatically:
// - Uses OneDNN XMX kernels if available (INT8/BF16 acceleration)
// - Uses tiled GEMM for large matrices
// - Uses blocked layouts for cache efficiency
// - Benefits from future GEMM optimizations
```

**Benefit:** Zero maintenance cost for kernel performance.

#### 2. Minimal JIT Extension

```cpp
// In gemm_kernel_base.cpp (~30 LOC)
JitConstants GemmKernelBase::GetJitConstants(const gemm_params& p) const {
    // ... existing code ...
    
    if (!p.fused_ops.empty()) {
        // Fused ops framework handles JIT generation automatically
        FusedOpsConfiguration conf = {"", {"b", "f", "y", "x"}, "gemm_result", output_dt};
        jit.Merge(MakeFusedOpsJitConstants(p, {conf}));
    }
    
    return jit;
}
```

```opencl
// In gemm_ref.cl (~5 LOC modification)
KERNEL(gemm_ref)(...) {
    // ... MatMul computation ...
    OUTPUT_TYPE result = accumulator;
    
    #if HAS_FUSED_OPS
        FUSED_OPS;  // Macro expanded by fusion framework
        OUTPUT[idx] = FUSED_OPS_RESULT;
    #else
        OUTPUT[idx] = result;
    #endif
}
```

**Benefit:** ~35 LOC total to enable fusion across all GEMM variants.

---

### Approach 3: OCL_V2 Simplifications

#### 1. Declarative Kernel API

```cpp
// OLD (Approach 1): Imperative with boilerplate
class MatRmsMulKernelRef : public KernelBaseOpenCL {
    KernelsData GetKernelsData(const Params& p) const override {
        KernelsData kd = {};
        KernelData kernel_data = KernelData::Default<matrmsmul_params>(p);
        
        // 50 lines of argument setup
        kernel_data.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel_data.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
        // ... repeat for all arguments
        
        // 100 lines of JIT constants
        auto jit = GetJitConstants(p);
        
        // 30 lines of dispatch data
        auto dispatch = GetDispatchData(p);
        kernel_data.kernels.push_back({GetKernelString(), jit, dispatch});
        
        kd.push_back(kernel_data);
        return kd;
    }
};
```

```cpp
// NEW (Approach 3): Declarative with clean separation
class MatRmsMulGenerator : public KernelGenerator {
    // Just describe WHAT, not HOW
    Arguments get_arguments_desc(const RuntimeParams& p) const override {
        return {{ArgumentDescriptor::Types::INPUT, 0},
                {ArgumentDescriptor::Types::INPUT, 1},
                {ArgumentDescriptor::Types::INPUT, 2},
                {ArgumentDescriptor::Types::OUTPUT, 0}};
    }
    
    JitConstants get_jit_constants(const RuntimeParams& p) const override {
        auto jit = KernelGenerator::get_jit_constants(p);
        jit.make("EPSILON", p.typed_desc<matrmsmul>()->epsilon);
        return jit;
    }
    
    DispatchDataFunc get_dispatch_data_func() const override {
        return [](const RuntimeParams& p, KernelData& kd, ImplRuntimeParams*) {
            kd.params.workGroups.global = {p.output_layouts[0].count(), 1, 1};
            kd.params.workGroups.local = {256, 1, 1};
        };
    }
};
```

**Benefit:** 70% less code, cleaner interfaces, easier to test.

#### 2. Multi-Stage Built-In Support

```cpp
// Example: GroupNormalization requires 3 kernels
class GroupNormalizationRef : public PrimitiveImplOCL {
    Stage::Ptr calc_mean_stage = make_stage<CalcMeanGenerator>();
    Stage::Ptr calc_var_stage = make_stage<CalcVarianceGenerator>();
    Stage::Ptr normalize_stage = make_stage<NormalizeGenerator>();
    
    void create_impl(const RuntimeParams& p) override {
        add_stage(calc_mean_stage, p);   // Stage 1: mean reduction
        add_stage(calc_var_stage, p);    // Stage 2: variance with mean as input
        add_stage(normalize_stage, p);   // Stage 3: normalize using mean+var
        // Execution order automatically managed
    }
};
```

**Benefit:** Natural expression of multi-kernel algorithms; explicit stage dependencies.

#### 3. Registry-Based Selection

```cpp
// In matrmsmul_impls.cpp
const std::vector<std::shared_ptr<ImplementationManager>>& 
Registry<matrmsmul>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        // Optimized implementation for static shapes
        OV_GPU_CREATE_INSTANCE_OCL(ocl::MatRmsMulOpt, shape_types::static_shape)
        
        // Reference implementation for dynamic shapes
        OV_GPU_CREATE_INSTANCE_OCL(ocl::MatRmsMulRef, shape_types::dynamic_shape)
        
        // Can add FP16-specific optimizations
        // OV_GPU_CREATE_INSTANCE_OCL(ocl::MatRmsMulFP16Opt, shape_types::static_shape)
    };
    return impls;
}
```

**Benefit:** Explicit implementation prioritization; supports multiple variants for different scenarios.

---

## Recommendation Matrix

### Use Case Based Recommendations

| Scenario | Recommended Approach | Rationale | Risk Level |
|----------|---------------------|-----------|------------|
| **Fast production deployment (< 1 week)** | ✅ **Approach 2** | 82% less work, leverages battle-tested GEMM kernels, minimal surface area for bugs | 🟢 Very Low |
| **Prototyping new operator concept** | ✅ **Approach 3** | 31% faster than Approach 1, cleaner APIs enable rapid iteration | 🟡 Medium |
| **Educational/learning complete stack** | ⚠️ Approach 1 | Touches all architectural layers, but extremely time-consuming | 🔴 High |
| **Multiple kernel optimization variants** | ✅ **Approach 3** | OCL_V2 natively supports _ref/_opt/_fp16 split with Registry selection | 🟡 Medium |
| **Leveraging hardware accelerators (XMX)** | ✅ **Approach 2** | Direct access to OneDNN's XMX/Systolic Array implementations | 🟢 Very Low |
| **Long-term maintainable solution** | ✅ **Approach 3** | Modern architecture aligned with OpenVINO's future direction | 🟡 Medium |
| **Complex multi-stage algorithm** | ✅ **Approach 3** | Stage mechanism purpose-built for this (e.g., mean→var→normalize) | 🟡 Medium |
| **Minimal code review cycles** | ✅ **Approach 2** | Only ~150 LOC to review vs 3000+ LOC | 🟢 Very Low |
| **Need custom algorithmic fusion** | ⚠️ Approach 3 | If fusion logic too complex for Approach 2's macro system | 🟡 Medium |
| **Legacy codebase maintenance** | ⚠️ Approach 1 | Only if existing code already uses this pattern | 🔴 High |

---

### Technology Alignment

| Consideration | Approach 1 | Approach 2 | Approach 3 | Winner |
|--------------|-----------|-----------|-----------|---------|
| **Intel XMX (Xe Matrix Extension)** | ❌ No access | ✅ Via OneDNN | ⚠️ Manual impl | Approach 2 |
| **OneDNN Integration** | ❌ Would need separate OneDNN impl | ✅ Automatic | ⚠️ Would need separate OneDNN impl | Approach 2 |
| **Dynamic Shapes** | ⚠️ Requires extra handling | ✅ Already supported | ✅ Well supported | Tie (2 & 3) |
| **Multi-Device Scope** | ❌ GPU only | ✅ GPU only (but composes with other devices) | ❌ GPU only | Approach 2 |
| **OpenVINO Model Optimizer** | ⚠️ May require custom MO pass | ✅ Pattern recognized automatically | ⚠️ May require custom MO pass | Approach 2 |
| **Inference At Scale** | ⚠️ Untested perf | ✅ Production-proven | ⚠️ Requires tuning | Approach 2 |

---

### Resource Constraints

| Constraint | Approach 1 | Approach 2 | Approach 3 | Recommendation |
|-----------|-----------|-----------|-----------|----------------|
| **Timeline: 3-5 days** | ❌ Not possible | ✅ Achievable | ❌ Too tight | Approach 2 |
| **Timeline: 2 weeks** | ⚠️ Tight | ✅ Comfortable | ✅ Achievable | Approach 2 or 3 |
| **Timeline: 1 month** | ✅ Possible | ✅ Overkill | ✅ Polish + optimize | Approach 3 |
| **Team: 1 developer** | ⚠️ Risky | ✅ Manageable | ✅ Feasible | Approach 2 or 3 |
| **Team: Multiple developers** | ✅ Can parallelize | ⚠️ Little parallelism | ✅ Can split implementations | Approach 3 |
| **GPU expertise: Low** | ❌ Requires deep knowledge | ✅ Minimal GPU knowledge | ⚠️ Moderate knowledge | Approach 2 |
| **GPU expertise: High** | ⚠️ Still tedious | ✅ Quick win | ✅ Flexible optimization | Approach 2 or 3 |

---

## Final Recommendation for MatMul→RMSNorm→MatMul

### Priority 1: Approach 2 (Fused Primitives) - **STRONGLY RECOMMENDED**

**Choose when:**
- Production deployment timeline < 2 weeks
- Need guaranteed high performance
- Team has limited GPU kernel expertise
- Want to leverage hardware acceleration (XMX/OneDNN)

**Implementation Strategy:**
1. **Week 1 Days 1-2:** Implement pattern detection in `prepare_primitive_fusing.cpp`
   - Detect MatMul→RMSNorm→MatMul chain
   - Verify no intermediate users
   - Attach as fused_primitive_desc to second MatMul

2. **Week 1 Days 3-4:** Extend GEMM JIT generation
   - Add RMSNorm support to `gemm_kernel_base.cpp`
   - Modify `gemm_ref.cl` with fusion macro

3. **Week 1 Day 5 - Week 2:** Testing & validation
   - Numerical accuracy tests vs unfused
   - Performance benchmarking
   - Dynamic shape coverage

**Expected Outcome:** 1.5-2x speedup over unfused baseline with 4.5 person-days effort.

---

### Priority 2: Approach 3 (OCL_V2) - **RECOMMENDED FOR NEW OPERATORS**

**Choose when:**
- Custom fusion logic required (Approach 2's JIT macros insufficient)
- Need multiple optimization variants (_ref, _opt, _fp16)
- Long-term maintenance preferred over quick turnaround
- Building new operator family (e.g., MoEOps, CustomNormalization)

**Implementation Strategy:**
1. **Week 1:** Frontend & Primitive Definition (same as Approach 1)
   - Internal op definition
   - Fusion pass
   - Primitive struct

2. **Week 2-3:** OCL_V2 Implementation
   - `matrmsmul_ref.cl` with basic implementation
   - `MatRmsMulRef` class with KernelGenerator
   - Registry registration

3. **Week 3-4:** Optimization & Testing
   - `matrmsmul_opt.cl` with tiling/blocking
   - Performance tuning
   - Comprehensive test suite

**Expected Outcome:** Maintainable codebase with 50% less complexity than Approach 1.

---

### Avoid: Approach 1 (Traditional OCL) - **NOT RECOMMENDED FOR NEW DEVELOPMENT**

**Only choose when:**
- Maintaining existing legacy kernels in this style
- Educational purposes (learning complete architecture)
- Time and resources are unlimited

**Disadvantages:**
- 5.8x more effort than Approach 2
- 1.4x more effort than Approach 3
- 31 files to modify vs 3-22
- Heavyweight kernel_selector boilerplate
- High risk of NDRange misconfiguration
- No access to hardware accelerators

---

## Implementation Checklist

### Approach 2 Quick Start Checklist

- [ ] **Day 1 Morning:** Study `prepare_primitive_fusing.cpp` - understand existing fusion patterns
- [ ] **Day 1 Afternoon:** Implement MatMul→RMS→MatMul pattern detection (~80 LOC)
- [ ] **Day 2 Morning:** Add fused_primitive_desc attachment logic (~50 LOC)
- [ ] **Day 2 Afternoon:** Extend `gemm_kernel_base.cpp` GetJitConstants() (~30 LOC)
- [ ] **Day 3 Morning:** Modify `gemm_ref.cl` with FUSED_OPS macro (~10 LOC)
- [ ] **Day 3 Afternoon:** Unit test pattern matching
- [ ] **Day 4:** Functional testing & numerical validation
- [ ] **Day 5:** Performance benchmarking & optimization tuning

**Total LOC:** ~170 lines across 2-3 files

---

### Approach 3 Implementation Checklist

**Week 1: Foundation**
- [ ] Define `ov::op::internal::MatRmsMul` (matrmsmul.hpp/cpp)
- [ ] Implement fusion pass with MatcherPass (matrmsmul_fusion.hpp/cpp)
- [ ] Register pass in transformations pipeline
- [ ] Define `cldnn::matrmsmul` primitive (primitives/matrmsmul.hpp)
- [ ] Implement graph node (matrmsmul_inst.hpp/cpp)

**Week 2: OCL_V2 Core**
- [ ] Write `matrmsmul_ref.cl` OpenCL kernel
- [ ] Create `MatRmsMulGenerator` with KernelGenerator interface
- [ ] Implement `get_arguments_desc()`
- [ ] Implement `get_jit_constants()`
- [ ] Implement `get_dispatch_data_func()`
- [ ] Create `MatRmsMulRef` implementation class

**Week 3: Registration & Testing**
- [ ] Create `matrmsmul_impls.cpp` with Registry<matrmsmul>
- [ ] Unit tests (test_matrmsmul.cpp)
- [ ] Functional tests (single_layer_tests/matrmsmul.cpp)
- [ ] Transformation tests (transformations/matrmsmul_fusion_test.cpp)

**Week 4: Optimization (Optional)**
- [ ] Create `matrmsmul_opt.cl` with tiling
- [ ] Implement `MatRmsMulOpt` class
- [ ] Performance tuning with different block sizes
- [ ] Benchmark vs Approach 2

**Total LOC:** ~1500-2500 lines across 15-20 files

---

## Conclusion

For the specific use case of fusing **MatMul→RMSNorm→MatMul** in OpenVINO's Intel GPU plugin:

1. **Production environments:** Use **Approach 2 (Fused Primitives)** - it delivers the best performance with 82% less effort by leveraging existing highly-optimized GEMM implementations.

2. **New operator development:** Use **Approach 3 (OCL_V2)** - it provides a modern, maintainable architecture that reduces complexity by 31-39% compared to the traditional approach.

3. **Traditional full-stack approach:** **Avoid Approach 1** - it's a legacy architecture that requires 5.8x more effort than Approach 2 with no significant benefits for this use case.

The choice between Approach 2 and 3 depends primarily on whether existing kernels (GEMM) can be extended via fusion (Approach 2) or if truly custom kernel logic is required (Approach 3). For most production scenarios, Approach 2 is the clear winner.

---

## Appendix: Quick Reference

### File Count Summary

| Metric | Approach 1 | Approach 2 | Approach 3 |
|--------|-----------|-----------|-----------|
| **New files** | 20 | 0-2 | 12-15 |
| **Modified files** | 11 | 3 | 7 |
| **Total operations** | 31 | 3-5 | 19-22 |

### Effort Summary

| Phase | Approach 1 | Approach 2 | Approach 3 |
|-------|-----------|-----------|-----------|
| **Total effort** | 26 days | 4.5 days | 18 days |
| **LOC written** | ~3000-4000 | ~150-300 | ~1500-2500 |

### Risk Assessment

| Factor | Approach 1 | Approach 2 | Approach 3 |
|--------|-----------|-----------|-----------|
| **Development risk** | 🔴 High | 🟢 Very Low | 🟡 Medium |
| **Maintenance risk** | 🔴 High | 🟢 Very Low | 🟡 Medium |
| **Performance risk** | 🟡 Medium | 🟢 Very Low | 🟡 Medium |

---

**Document End**
