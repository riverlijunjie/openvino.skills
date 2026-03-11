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

---

## OCL_V2 Architecture Optimization Proposals

The OCL_V2 architecture can be further accelerated through automation and abstraction improvements. Here are **five optimization proposals** to reduce development time from 18 days to **8-12 days**.

### Proposal 1: Primitive Code Generator Tool ⚡ (High Impact)

**Problem:** Creating new primitives requires ~12-15 boilerplate files with repetitive patterns.

**Solution:** Extend existing `kernels_db_gen.py` to create a comprehensive primitive generator.

**Implementation:**
```bash
# New tool: src/plugins/intel_gpu/scripts/gen_primitive.py
python3 gen_primitive.py \
    --name matrmsmul \
    --type fusion \
    --inputs "input:fp16,weights1:fp16,rms_scale:fp16,weights2:fp16" \
    --outputs "output:fp16" \
    --params "epsilon:float,transpose_a:bool,transpose_b:bool"
```

**Auto-Generated Files:**
1. ✅ `include/intel_gpu/primitives/matrmsmul.hpp` - Primitive definition with all parameters
2. ✅ `src/graph/matrmsmul_inst.hpp/cpp` - Graph node with shape inference template
3. ✅ `src/graph/impls/ocl_v2/matrmsmul_ref.hpp/cpp` - Reference implementation skeleton
4. ✅ `src/graph/impls/ocl_v2/matrmsmul_ref.cl` - OpenCL kernel template with JIT placeholders
5. ✅ `src/graph/registry/matrmsmul_impls.cpp` - Registry registration
6. ✅ `tests/unit/test_matrmsmul.cpp` - Basic test template

**Effort Reduction:** 2-3 days → **0.5 days** (83% faster)

**Example Generated Code:**
```cpp
// Auto-generated: include/intel_gpu/primitives/matrmsmul.hpp
#pragma once
#include "primitive.hpp"

namespace cldnn {
struct matrmsmul : public primitive_base<matrmsmul> {
    CLDNN_DECLARE_PRIMITIVE(matrmsmul)
    
    matrmsmul(const primitive_id& id,
              const input_info& input,
              const input_info& weights1,
              const input_info& rms_scale,
              const input_info& weights2,
              float epsilon = 1e-5f,
              bool transpose_a = false,
              bool transpose_b = false)
        : primitive_base(id, {input, weights1, rms_scale, weights2}),
          epsilon(epsilon),
          transpose_a(transpose_a),
          transpose_b(transpose_b) {}
    
    float epsilon;
    bool transpose_a;
    bool transpose_b;
    
    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, epsilon);
        seed = hash_combine(seed, transpose_a);
        seed = hash_combine(seed, transpose_b);
        return seed;
    }
};
}
```

---

### Proposal 2: Unified Base Template for Simple Operators 🎯 (Medium Impact)

**Problem:** Even simple element-wise or reduction operators require full OCL_V2 implementation.

**Solution:** Create `SimpleOpBase` template that handles common patterns.

**Implementation:**
```cpp
// New: src/graph/impls/ocl_v2/utils/simple_op_base.hpp
template<typename PrimitiveType>
class SimpleUnaryOpBase : public PrimitiveImplOCL {
protected:
    Stage::Ptr stage = make_stage<SimpleUnaryGenerator<PrimitiveType>>();
    
public:
    SimpleUnaryOpBase() {
        add_stage(stage, get_runtime_params());
    }
    
    // Automatically handles:
    // - Standard argument mapping (1 input, 1 output)
    // - Basic JIT constants
    // - 1D/2D/3D/4D dispatch data calculation
    // - Dynamic shape support
};
```

**Usage Example:**
```cpp
// matrmsmul_ref.hpp - Now just 15 lines instead of 80!
#include "ocl_v2/utils/simple_op_base.hpp"

class MatRmsMulRef : public SimpleTripleInputOpBase<matrmsmul> {
protected:
    JitConstants get_custom_jit_constants(const RuntimeParams& p) const override {
        auto jit = SimpleTripleInputOpBase::get_custom_jit_constants(p);
        auto desc = p.typed_desc<matrmsmul>();
        jit.make("EPSILON", desc->epsilon);
        jit.make("TRANSPOSE_A", desc->transpose_a);
        jit.make("TRANSPOSE_B", desc->transpose_b);
        return jit;
    }
};
```

**Effort Reduction:** 4 days kernel implementation → **2 days** (50% faster)

**Supported Patterns:**
- `SimpleUnaryOpBase` - ReLU, Exp, Log, Sqrt
- `SimpleBinaryOpBase` - Add, Mul, Sub, Div
- `SimpleTripleInputOpBase` - Custom fusions
- `SimpleReductionOpBase` - ReduceMax, ReduceSum, ReduceMean

---

### Proposal 3: YAML-Based Primitive Definition 📋 (High Impact

**Problem:** C++ primitive definitions are verbose and require recompilation for changes.

**Solution:** Define primitives in YAML, generate C++ at build time.

**Implementation:**
```yaml
# New: src/plugins/intel_gpu/primitives/matrmsmul.yaml
name: matrmsmul
type: fusion
category: neural_network

inputs:
  - name: input
    type: [fp16, fp32]
    rank: [2, 3, 4]
  - name: weights1
    type: [fp16, fp32]
    rank: 2
  - name: rms_scale
    type: [fp16, fp32]
    rank: 1
  - name: weights2
    type: [fp16, fp32]
    rank: 2

outputs:
  - name: output
    type: auto  # Inherits from input
    rank: auto  # Computed by shape inference

parameters:
  - name: epsilon
    type: float
    default: 1e-5
    range: [1e-10, 1e-3]
  
  - name: transpose_a
    type: bool
    default: false
  
  - name: transpose_b
    type: bool
    default: false

shape_inference:
  # Python-like expression
  output_shape: "matmul_shape(input, weights1, transpose_a) -> rms -> matmul_shape(rms_out, weights2, transpose_b)"

ocl_v2_implementations:
  - name: ref
    kernel: matrmsmul_ref.cl
    priority: 10
    shapes: [static, dynamic]
  
  - name: opt
    kernel: matrmsmul_opt.cl
    priority: 100
    shapes: [static]
    constraints:
      - "input.dims[0] >= 16"  # Batch size >= 16
      - "weights1.dims[1] % 64 == 0"  # Hidden size multiple of 64
```

**Build Integration:**
```cmake
# In CMakeLists.txt
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated/matrmsmul.hpp
    COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/gen_from_yaml.py
            --input ${CMAKE_CURRENT_SOURCE_DIR}/primitives/matrmsmul.yaml
            --output-dir ${CMAKE_CURRENT_BINARY_DIR}/generated
    DEPENDS primitives/matrmsmul.yaml
)
```

**Effort Reduction:** 3 days boilerplate → **0.5 days** (83% faster)

**Benefits:**
- ✅ Single source of truth for primitive definition
- ✅ Automatic validation and documentation generation
- ✅ Easy to review (YAML diffs are cleaner than C++)
- ✅ Type safety enforced by schema
- ✅ Can generate Python bindings simultaneously

---

### Proposal 4: Macro-Based Single-File Registration 🔧 (Low Impact, High Convenience)

**Problem:** Separate registry file requires CMakeLists.txt modification and boilerplate.

**Solution:** Self-registering primitives using static initializers.

**Implementation:**
```cpp
// In matrmsmul_ref.cpp - add at the end:
REGISTER_OCL_V2_IMPL_AUTO(
    matrmsmul,            // Primitive type
    MatRmsMulRef,         // Implementation class
    shape_types::static_shape | shape_types::dynamic_shape,
    data_types::f16 | data_types::f32
);

// Expands to (auto-generated):
namespace {
    struct MatRmsMulRefRegistrar {
        MatRmsMulRefRegistrar() {
            Registry<matrmsmul>::register_impl<MatRmsMulRef>(
                shape_types::static_shape | shape_types::dynamic_shape,
                data_types::f16 | data_types::f32
            );
        }
    };
    static MatRmsMulRefRegistrar _matrmsmul_ref_registrar;
}
```

**Effort Reduction:** 1 day registry setup → **0.1 days** (90% faster)

**Benefits:**
- ❌ No separate `*_impls.cpp` file needed
- ❌ No CMakeLists.txt modification for registration
- ✅ Implementation and registration co-located
- ✅ Impossible to forget registration

---

### Proposal 5: Test Template Generator with Golden Data 🧪 (Medium Impact)

**Problem:** Writing comprehensive tests is time-consuming (3 days per operator).

**Solution:** Auto-generate parameterized tests from primitive YAML + golden data.

**Implementation:**
```yaml
# In matrmsmul.yaml - add test section:
tests:
  unit:
    - name: basic_fp16
      input: [1, 128, 512]  # [batch, seq, hidden]
      weights1: [512, 1024]
      rms_scale: [1024]
      weights2: [1024, 512]
      epsilon: 1e-5
      dtype: fp16
      golden_output: tests/golden/matrmsmul_basic_fp16.bin
      tolerance: 1e-3
    
    - name: transposed_fp32
      input: [4, 256, 1024]
      weights1: [2048, 1024]  # Transposed
      transpose_a: true
      golden_output: tests/golden/matrmsmul_transposed_fp32.bin
      tolerance: 1e-6
  
  functional:
    parameterized:
      batch_sizes: [1, 4, 16, 64]
      seq_lengths: [128, 256, 512, 1024]
      hidden_sizes: [512, 1024, 2048, 4096]
      dtypes: [fp16, fp32]
      compare_against: cpu_plugin
```

**Generated Test Code:**
```cpp
// Auto-generated: tests/unit/test_matrmsmul.cpp
#include <gtest/gtest.h>
#include "test_utils.h"

class MatRmsMulTest : public PrimitiveTestBase<matrmsmul> {};

TEST_F(MatRmsMulTest, basic_fp16) {
    this->load_golden_data("tests/golden/matrmsmul_basic_fp16.bin");
    this->run_test({1, 128, 512}, /*weights1*/{512, 1024}, ...);
    this->compare_with_golden(1e-3f);
}

TEST_F(MatRmsMulTest, transposed_fp32) {
    // ... auto-generated test code
}

// Parameterized tests
INSTANTIATE_TEST_SUITE_P(
    AllCombinations,
    MatRmsMulTest,
    ::testing::Combine(
        ::testing::Values(1, 4, 16, 64),      // batch_sizes
        ::testing::Values(128, 256, 512, 1024), // seq_lengths
        ...
    )
);
```

**Golden Data Generation:**
```bash
# One-time golden data generation against reference implementation
python3 scripts/generate_golden_data.py \
    --primitive matrmsmul \
    --tests matrmsmul.yaml \
    --reference numpy  # or torch, onnxruntime
```

**Effort Reduction:** 3 days testing → **1 day** (67% faster)

---

## Combined Optimization Impact

### Original OCL_V2 Timeline (18 days):
- Day 1-3: Frontend & Primitive Definition
- Day 4-8: OCL_V2 Implementation
- Day 9-10: Registration
- Day 11-13: Testing
- Day 14-18: Debugging & Optimization

### Optimized Timeline (8-12 days):

| Phase | Original | With Tools | Savings | Tools Used |
|-------|----------|------------|---------|------------|
| **Boilerplate Generation** | 3 days | 0.5 days | -83% | Proposal 1 (Generator) |
| **Kernel Implementation** | 4 days | 2 days | -50% | Proposal 2 (Base Templates) |
| **Registration** | 1 day | 0.1 days | -90% | Proposal 4 (Auto-register) |
| **Testing** | 3 days | 1 day | -67% | Proposal 5 (Test Generator) |
| **Debugging & Tuning** | 5 days | 4 days | -20% | Less boilerplate = fewer bugs |
| **Documentation** | 2 days | 0.4 days | -80% | Auto-generated from YAML |
| **Total** | **18 days** | **8-10 days** | **-44% to -56%** | |

**Additional 2-day buffer for custom optimization brings total to 10-12 days.**

---

## Implementation Roadmap for Optimization Tools

### Phase 1: Quick Wins (1-2 weeks)
1. **Proposal 4** - Macro-based registration (easiest, immediate value)
2. **Proposal 2** - Create 3-4 base templates for common patterns
   - SimpleUnaryOpBase
   - SimpleBinaryOpBase
   - SimpleReductionOpBase

### Phase 2: Automation (2-3 weeks)
3. **Proposal 1** - Primitive code generator tool
   - Start with basic template generation
   - Iterate based on developer feedback

4. **Proposal 5** - Test template generator
   - Integrate with existing test utils
   - Support CPU plugin comparison

### Phase 3: Advanced (4-6 weeks)
5. **Proposal 3** - YAML-based definition system
   - Design schema carefully
   - Implement build-time code generation
   - Add IDE support (YAML completion)

---

## Example: matrmsmul Development with All Optimizations

### Step 1: Define Primitive (15 minutes)
```bash
# Create YAML definition
vim primitives/matrmsmul.yaml  # Write 40 lines of YAML

# Generate all boilerplate
python3 scripts/gen_from_yaml.py primitives/matrmsmul.yaml
```

**Generated:** 12 files (~2000 LOC), ready to compile

### Step 2: Implement OpenCL Kernel (1-2 days)
```opencl
// Only file you write manually: matrmsmul_ref.cl
KERNEL(matrmsmul_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* weights1,
    const __global INPUT2_TYPE* rms_scale,
    const __global INPUT3_TYPE* weights2,
    __global OUTPUT_TYPE* output)
{
    // Your custom fusion logic here
    // Use pre-generated JIT macros: INPUT0_GET_INDEX, EPSILON, etc.
}
```

### Step 3: Customize Generated Code (0.5 days)
```cpp
// matrmsmul_ref.cpp - override only what's needed
JitConstants MatRmsMulRefGenerator::get_jit_constants(const RuntimeParams& p) const {
    auto jit = SimpleTripleInputOpBase::get_jit_constants(p);
    // Add custom constants
    jit.make("USE_FAST_MATH", true);
    return jit;
}
```

### Step 4: Generate Tests & Golden Data (0.5 days)
```bash
# Generate test suite
python3 scripts/gen_tests.py primitives/matrmsmul.yaml

# Generate golden data from PyTorch reference
python3 scripts/gen_golden_data.py --primitive matrmsmul --backend torch
```

### Step 5: Build & Test (0.5 days)
```bash
# Everything auto-registered, just build
cmake --build . --target test_matrmsmul

# Run tests
./test_matrmsmul --gtest_filter=MatRmsMulTest.*
```

### Step 6: Optimize (2-3 days)
- Write `matrmsmul_opt.cl` with tiling/blocking
- Profile and tune work-group sizes
- Add to YAML `implementations:` section

**Total Time with All Optimizations: 5-7 days (vs 18 days original)**

---

## Comparison: Manual vs Automated Development

| Aspect | Manual OCL_V2 | With Optimization Tools | Improvement |
|--------|---------------|------------------------|-------------|
| **Lines of Code Written** | ~2500 | ~500 | **80% less** |
| **Files Created Manually** | 15-20 | 2-3 | **85% less** |
| **CMakeLists Edits** | 4-5 places | 0-1 places | **80-100% less** |
| **Boilerplate Errors** | High (manual typing) | Very Low (generated) | **90% less bugs** |
| **Documentation** | Manual | Auto-generated | **100% sync** |
| **Time to First Build** | 2-3 days | 1-2 hours | **95% faster** |
| **Test Coverage** | Depends on diligence | Comprehensive | **2x better** |
| **Onboarding New Devs** | 2-3 weeks | 3-5 days | **70% faster** |

---

## Next Steps to Adopt These Optimizations

### For Individual Developers:
1. **Use Proposal 2 (Base Templates)** immediately - no infrastructure needed
2. Request Proposal 4 (Auto-registration) from team - low-hanging fruit
3. Prototype Proposal 1 (Code Generator) for your own use

### For Team Leads:
1. **Week 1-2:** Implement Proposal 4 (macro-based registration)
2. **Week 3-4:** Create 3-4 base templates (Proposal 2)
3. **Month 2:** Build primitive generator tool (Proposal 1)
4. **Month 3:** Design YAML schema and test generators (Proposals 3 & 5)
5. **Month 4+:** Gradual migration of existing primitives to new system

### Metrics to Track Success:
- Time from idea to working primitive (target: < 1 week)
- Number of boilerplate bugs found in code review (target: 50% reduction)
- Lines of code per primitive (target: < 600 LOC including tests)
- Developer onboarding time (target: < 1 week to first contribution)

---

**With these optimizations, OCL_V2 development becomes 2-3x faster, making it competitive with even Approach 2 (Fused Primitives) for new operator development, while maintaining full flexibility and ownership of the implementation.**

