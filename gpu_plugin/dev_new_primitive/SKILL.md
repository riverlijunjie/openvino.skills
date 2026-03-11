---
name: dev_new_primitive
description: Develop new primitives for better performance. Use when working on new operator development or improving existing operator efficiency.
---

When integrating new primitive feature, always include:
1. **Read primitive ocl_v2 framework code**: Read primitive related code to understand its structure and optimizations. This includes:
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2
    - src/plugins/intel_gpu/src/plugin/transformations

2. **Identify optimization opportunities**: Look for areas in the code where performance can be improved, such as inefficient algorithms, redundant computations, or memory bottlenecks.
3. **Implement optimizations**: Apply code changes to optimize the identified areas. This may include algorithmic improvements, better memory management, or leveraging hardware capabilities.
4. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance.
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.

---

## Three Approaches for Custom Operator Integration

When adding a new operator or fusing existing operators in the Intel GPU plugin, choose from three architectural approaches based on your requirements. **For detailed analysis, see [gpu_plugin_operator_integration_comparison.md](../../../docs/dev/gpu_plugin_operator_integration_comparison.md)**.

### Quick Decision Guide

| Scenario | Recommended Approach | Effort | Risk |
|----------|---------------------|--------|------|
| **Production fusion (< 1 week)** | Approach 2: Fused Primitives | 4.5 days | 🟢 Very Low |
| **New operator prototype** | Approach 3: OCL_V2 | 18 days | 🟡 Medium |
| **Learning/Education** | Approach 1: Traditional OCL | 26 days | 🔴 High |

---

### Approach 1: Traditional OCL (Legacy - Avoid for New Development)

**Full five-layer stack traversal:** OV IR → Plugin Frontend → Primitive → Kernel Selector → OCL Implementation

**File Operations:** 20 new + 11 modified = **31 files**  
**Effort:** **26 person-days** | **Complexity:** ⭐⭐⭐⭐⭐

**Key Files:**
- `src/plugins/intel_gpu/src/kernel_selector/core/actual_kernels/<op>/` - Kernel base, ref, selector (~600 LOC boilerplate)
- `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/<op>_ref.cl` - OpenCL kernel
- `src/plugins/intel_gpu/src/graph/impls/ocl/<op>.cpp` - typed_primitive_impl_ocl binding
- Plus: primitive definition, graph node, frontend translation, tests

**Pain Points:**
- Heavyweight kernel_selector boilerplate (GetDispatchData, GetJitConstants, Validate)
- Complex parameter passing through 6+ intermediate structures
- JIT macro string concatenation with no compile-time checking
- Requires deep GPU architecture knowledge for NDRange tuning

**When to Use:** Legacy maintenance only. **Not recommended for new development.**

---

### Approach 2: Fused Primitives (Recommended for Production)

**Leverage existing operators:** Attach fusion logic to existing high-performance kernels via post-ops framework

**File Operations:** 0-2 new + 3 modified = **3-5 files**  
**Effort:** **4.5 person-days** (82% less than Approach 1) | **Complexity:** ⭐⭐

**Key Files:**
- `src/plugins/intel_gpu/src/graph/graph_optimizer/prepare_primitive_fusing.cpp` - Pattern detection (~50-100 LOC)
- `src/plugins/intel_gpu/src/kernel_selector/core/actual_kernels/gemm/gemm_kernel_base.cpp` - JIT extension (~30 LOC)
- `src/plugins/intel_gpu/src/kernel_selector/cl_kernels/gemm_ref.cl` - Fusion macro placeholder (~5 LOC)

**Example Pattern (MatMul→RMSNorm→MatMul):**
```cpp
// In prepare_primitive_fusing.cpp
if (matmul2_node.has_pattern(matmul1 → rms → matmul2)) {
    matmul2_node.add_fused_primitive(matmul1);
    matmul2_node.add_fused_primitive(rms);
    p.mark_optimized_out(matmul1, rms);
}
```

**Advantages:**
- ✅ Zero new primitives - reuses existing implementations
- ✅ Automatic optimization inheritance (OneDNN/XMX for free)
- ✅ Minimal code footprint (~150 LOC total)
- ✅ Production-proven fusion framework
- ✅ No kernel_selector boilerplate

**When to Use:**
- Fast production deployment (< 1 week timeline)
- Standard operator combinations that can be expressed as post-ops
- Want to leverage hardware accelerators (XMX/Systolic Array)
- Minimal maintenance burden desired

**Limitations:**
- Fusion logic limited to JIT macro capabilities
- Must fit into existing operator's execution model
- Complex multi-stage algorithms may not fit

---

### Approach 3: OCL_V2 (Recommended for New Operators)

**Modern simplified architecture:** Eliminates kernel_selector layer, uses KernelGenerator + Registry pattern

**File Operations:** 12-15 new + 7 modified = **19-22 files**  
**Effort:** **18 person-days** (31% less than Approach 1) | **Complexity:** ⭐⭐⭐⭐

**Key Files:**
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/<op>_ref.cl` - OpenCL kernel with JIT macros
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/<op>_ref.hpp/cpp` - PrimitiveImplOCL implementation
- `src/plugins/intel_gpu/src/graph/registry/<op>_impls.cpp` - Registry<op> registration
- Plus: primitive definition, graph node, frontend translation, tests

**Architecture Simplification:**
```cpp
// OLD (Approach 1): Imperative with 600 LOC boilerplate
class OpKernelRef : public KernelBaseOpenCL {
    KernelsData GetKernelsData(...) { /* 50 lines arg setup, 100 lines JIT, 30 lines dispatch */ }
};

// NEW (Approach 3): Declarative with clean API
class OpGenerator : public KernelGenerator {
    Arguments get_arguments_desc(...) { return {{INPUT, 0}, {OUTPUT, 0}}; }
    JitConstants get_jit_constants(...) { jit.make("EPSILON", desc->epsilon); return jit; }
    DispatchDataFunc get_dispatch_data_func() { return [](p, kd, rt) { kd.workGroups.global = {...}; }; }
};
```

**Advantages:**
- ✅ 50%+ less code than Approach 1
- ✅ Cleaner KernelGenerator API (declarative vs imperative)
- ✅ No kernel_selector boilerplate
- ✅ Native multi-stage support (mean→variance→normalize)
- ✅ Files co-located in ocl_v2/ directory
- ✅ Registry pattern enables shape-type-specific selection
- ✅ OpenVINO's modernization direction

**Multi-Stage Example:**
```cpp
class GroupNormalizationRef : public PrimitiveImplOCL {
    Stage::Ptr calc_mean = make_stage<CalcMeanGenerator>();
    Stage::Ptr calc_var = make_stage<CalcVarianceGenerator>();
    Stage::Ptr normalize = make_stage<NormalizeGenerator>();
    
    void create_impl(const RuntimeParams& p) {
        add_stage(calc_mean, p);   // Stage 1
        add_stage(calc_var, p);     // Stage 2
        add_stage(normalize, p);    // Stage 3
    }
};
```

**When to Use:**
- New operator requiring custom kernel logic
- Need multiple optimization variants (_ref, _opt, _fp16)
- Complex multi-stage algorithms
- Long-term maintainable solution desired
- 2+ week development timeline available

**Key Directories:**
- Implementation: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/`
- Registration: `src/plugins/intel_gpu/src/graph/registry/`
- Utils: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/utils/` (jitter, fused_ops_jitter, kernel_generator)

---

## Comparison Summary

| Metric | Approach 1 (Traditional) | Approach 2 (Fused Primitives) | Approach 3 (OCL_V2) |
|--------|------------------------|------------------------------|-------------------|
| **Total Files** | 31 | 3-5 | 19-22 |
| **Code Volume** | ~3000-4000 LOC | ~150-300 LOC | ~1500-2500 LOC |
| **Effort** | 26 days | 4.5 days | 18 days |
| **Complexity** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **XMX/OneDNN Access** | ❌ | ✅ | ⚠️ Manual |
| **Future-Proof** | ❌ Legacy | ✅ Stable | ✅ Modern |
| **Risk Level** | 🔴 High | 🟢 Very Low | 🟡 Medium |

---

## Implementation Workflow

### For Approach 2 (Fused Primitives):

1. **Pattern Detection** - Add fusion rule in `prepare_primitive_fusing.cpp`:
   ```cpp
   if (pattern_matches) {
       primary_node.add_fused_primitive(secondary_node);
       mark_optimized_out(secondary_node);
   }
   ```

2. **JIT Extension** - Extend primary kernel's JIT generation:
   ```cpp
   if (!fused_ops.empty()) {
       jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
   }
   ```

3. **Kernel Modification** - Add fusion macro in .cl file:
   ```opencl
   #if HAS_FUSED_OPS
       FUSED_OPS;
       OUTPUT[idx] = FUSED_OPS_RESULT;
   #endif
   ```

### For Approach 3 (OCL_V2):

1. **Frontend** - Define internal op + fusion pass:
   - `include/intel_gpu/op/<op>.hpp`
   - `src/plugin/transformations/<op>_fusion.cpp`

2. **Primitive** - Define cldnn primitive:
   - `include/intel_gpu/primitives/<op>.hpp`
   - `src/graph/<op>_inst.hpp/cpp`

3. **OCL_V2 Implementation** - Create kernel generator:
   ```cpp
   // ocl_v2/<op>_ref.hpp
   class OpRef : public PrimitiveImplOCL {
       Stage::Ptr stage = make_stage<OpGenerator>();
   };
   ```

4. **Registry** - Register implementations:
   ```cpp
   // registry/<op>_impls.cpp
   Registry<op>::get_implementations() {
       return {OV_GPU_CREATE_INSTANCE_OCL(ocl::OpRef, shape_types::static_shape)};
   }
   ```

---

## Best Practices

### When to Use Each Approach:

**Use Approach 2 if:**
- Timeline < 2 weeks
- Fusion fits into existing operator execution model
- Want hardware acceleration (XMX)
- Minimal risk tolerance

**Use Approach 3 if:**
- Building new operator family
- Need custom fusion logic
- Want multiple variants (_ref, _opt)
- Timeline 2-4 weeks

**Avoid Approach 1 unless:**
- Maintaining legacy code
- Educational purposes only

### Performance Optimization Tips:

1. **For Approach 2:**
   - Ensure fusion reduces global memory traffic
   - Profile to verify OneDNN path is taken
   - Test dynamic shapes thoroughly

2. **For Approach 3:**
   - Start with _ref implementation, optimize later
   - Use `__local` memory for fusion intermediate results
   - Profile with `ovc` cache warming
   - Consider sub-group operations for reduction

3. **Common to All:**
   - Benchmark vs unfused baseline
   - Test multiple batch sizes and shapes
   - Verify numerical accuracy (especially for FP16)
   - Document roofline analysis

---

For complete details including file-by-file breakdown, architectural diagrams, and effort estimation, refer to:
**[gpu_plugin_operator_integration_comparison.md](../../../docs/dev/gpu_plugin_operator_integration_comparison.md)**

