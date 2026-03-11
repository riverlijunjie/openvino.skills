moe op:
    openvino/src/core/dev_api/openvino/op/moe.hpp
    openvino/src/core/dev_api/openvino/op/moe_compressed.hpp
    openvino/src/core/dev_api/openvino/op/moe_3gemm_fused_compressed.hpp
    openvino/src/plugins/intel_gpu/include/intel_gpu/primitives/moe_3gemm_fused_compressed.hpp
    openvino/src/plugins/intel_gpu/src/graph/moe_3gemm_fused.cpp
    openvino/src/plugins/intel_gpu/src/graph/registry/moe_3gemm_swiglu_impls.cpp
    openvino/src/plugins/intel_gpu/src/graph/include/moe_3gemm_fused_inst.h

moe transformation implementation:
    openvino/src/plugins/intel_gpu/src/plugin/transformations/convert_moe_to_compressed.cpp
    openvino/src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.cpp

moe transformation unit test:
    openvino/src/plugins/intel_gpu/tests/unit/test_cases/transform_moe_to_compressed_test.cpp
    openvino/src/plugins/intel_gpu/tests/unit/test_cases/fuse_moe_3gemm_compressed_test.cpp
    
moe primitive implementation:
    openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.hpp
    openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp

moe kernel implementation:
    openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_mlp.cl
    openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_fuse.cl

moe unit test:
    openvino/src/plugins/intel_gpu/tests/unit/test_cases/moe_3gemm_gpu_test.cpp

moe genai:
    openvino.genai/src/cpp/src/modeling/ops/ops.cpp
    openvino.genai/src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_moe.cpp
    openvino.genai/src/cpp/src/modeling/models/qwen3_moe/modeling_qwen3_moe.cpp
    openvino.genai/src/cpp/src/modeling/models/qwen3_next/modeling_qwen3_next.cpp
    openvino.genai/src/cpp/src/modeling/tests/ops_test.cpp