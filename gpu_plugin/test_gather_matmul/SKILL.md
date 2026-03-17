---
name: test_gather_matmul
description: Develop gather_matmul operations to support MoE.
---

When develop gather_matmul to support MoE, always include:

1. **Read gather_matmul source code**: read the existing code of gather_matmul operations to understand its structure and optimizations.
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul.cl
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2/gather_matmul_batched.cl
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2/gathermatmul_gather.cl
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2/gathermatmul_sort.cl
    - src/common/transformations/include/ov_ops/gather_matmul_compressed.hpp
    - src/plugins/intel_gpu/include/intel_gpu/primitives/gather_matmul.hpp
    - src/common/transformations/include/ov_ops/moe_compressed.hpp
    - src/common/transformations/include/transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp
    - src/common/transformations/include/transformations/common_optimizations/moe_op_fusion.hpp
    - src/common/transformations/include/transformations/op_conversions/convert_gather_matmul_to_compressed.hpp
    - src/common/transformations/src/ov_ops/gather_matmul_compressed.cpp
    - src/common/transformations/src/transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.cpp
    - src/common/transformations/src/transformations/op_conversions/convert_gather_matmul_to_compressed.cpp


2. **Read previous MoE implementation on GPU plugin**: read the existing code of previous MoE operations to understand its structure and optimizations
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_fuse.cl
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_mlp.cl
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_*.cl
    - src/plugins/intel_gpu/src/plugin/transformations/op/moe_3gemm_fused_compressed.cpp
    - src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.cpp
    - src/plugins/intel_gpu/src/plugin/transformations/keep_moe_3gemm_const_precision.hpp

3. **Understand previous MoE and gather_matmul relationship**: understand how gather_matmul operations are used in MoE implementations, and identify the differences between them, including advantages and disadvantages.
     - Previous MoE implementation on GPU plugin: mutli-token use micro_gemm(gather every experts' tokens together to do batch gemm) and corresponding gather/reduce/gen_mask kernels; sigle-token use moe_3gemm_swiglu kernels.
     - gather_matmul operation is designed to support both single-token and multi-token MoE implementations.

Keep explanations conversational. For complex concepts, use multiple analogies.