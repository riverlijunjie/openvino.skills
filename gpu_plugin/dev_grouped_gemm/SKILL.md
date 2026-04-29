---
name: dev_grouped_gemm
description: Develop Grouped GEMM operations for better performance. Use when working on grouped GEMM models or improving grouped GEMM operation efficiency.
---

When integrating grouped GEMM feature, always include:
1. **Read all grouped GEMM code**: Read grouped GEMM related code to understand its structure and optimizations. This includes:
    oneDNN grouped GEMM code:
        - src/plugins/intel_gpu/thirdparty/onednn_gpu
    OpenVINO integration code:
        - src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp
        - src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_mask_gen.cpp
        - src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_mask_gen.cl
        - src/plugins/intel_gpu/src/plugin/transformations/convert_moe_to_compressed.cpp
        - src/plugins/intel_gpu/tests/unit/test_cases/moe_mask_gen_gpu_test.cpp
        - src/plugins/intel_gpu/tests/unit/test_cases/moe_gemm_gpu_test.cpp
2. **Identify optimization opportunities**: Look for areas in the code where performance can be improved, such as inefficient algorithms, redundant computations, or memory bottlenecks.
3. **Implement optimizations**: Apply code changes to optimize the identified areas. This may include algorithmic improvements, better memory management, or leveraging hardware capabilities.
    - Follow the same code style and conventions used in the existing codebase to maintain consistency.
    - Don't create new commit
    - Don't push code to remote repository until.
    - Don't modify oneDNN code.
4. **Document optimization**:
    - Summarize the optimizations applied and insert their impact on performance to "SUMMARY.md"
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.

