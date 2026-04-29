# MOE Source File Reference

## Op Definition
- `src/core/dev_api/openvino/op/moe.hpp` — Base MOE op
- `src/core/dev_api/openvino/op/moe_compressed.hpp` — Compressed MOE op
- `src/core/dev_api/openvino/op/moe_3gemm_fused_compressed.hpp` — 3GEMM fused op (optional inputs 11-20 for shared expert)

## GPU Primitive
- `src/plugins/intel_gpu/include/intel_gpu/primitives/moe_3gemm_fused_compressed.hpp` — Primitive definition with `Config` struct
- `src/plugins/intel_gpu/src/graph/moe_3gemm_fused.cpp` — Primitive instance
- `src/plugins/intel_gpu/src/graph/include/moe_3gemm_fused_inst.h` — Instance header
- `src/plugins/intel_gpu/src/graph/registry/moe_3gemm_swiglu_impls.cpp` — Implementation registry

## Host-side Implementation
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.hpp` — Class declaration
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp` — JIT constants, dispatch, argument binding, decode/prefill execution
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_mask_gen.cpp` — Mask generation for grouped GEMM
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_mask_gen.cl` — GPU mask generation kernel

## OpenCL Kernels
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_mlp.cl` — GEMV kernels (gate_up, down, reduce)
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_fuse.cl` — Softmax/TopK, gather, scatter, mask-gen kernels

## Graph Transformations
- `src/plugins/intel_gpu/src/plugin/transformations/convert_moe_to_compressed.cpp` — Pattern match MOE subgraph → MOECompressed
- `src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.cpp` — Fuse MOECompressed → 3GEMM primitive

## oneDNN Grouped GEMM
- `src/plugins/intel_gpu/thirdparty/onednn_gpu` — oneDNN grouped GEMM implementation (do NOT modify)

## Tests
- `src/plugins/intel_gpu/tests/unit/test_cases/moe_3gemm_gpu_test.cpp` — GPU accuracy tests (75 tests)
- `src/plugins/intel_gpu/tests/unit/test_cases/transform_moe_to_compressed_test.cpp` — Transformation tests
- `src/plugins/intel_gpu/tests/unit/test_cases/fuse_moe_3gemm_compressed_test.cpp` — Fusion tests
- `src/plugins/intel_gpu/tests/unit/test_cases/moe_mask_gen_gpu_test.cpp` — Mask generation tests
- `src/plugins/intel_gpu/tests/unit/test_cases/moe_gemm_gpu_test.cpp` — micro_gemm/grouped GEMM tests
- `tests/unit/transformations/convert_moe_to_compressed_test.cpp` — Core pattern matching tests

## GenAI Integration
- `openvino.genai/src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_moe.cpp` — Qwen3.5 MoE model implementation
- `openvino.genai/src/cpp/src/modeling/ops/ops.cpp` — Op construction helpers
