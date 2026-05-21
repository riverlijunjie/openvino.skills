moe op:
    src/core/dev_api/openvino/op/moe.hpp
    src/core/dev_api/openvino/op/moe_compressed.hpp
    src/core/dev_api/openvino/op/moe_3gemm_fused_compressed.hpp
    src/plugins/intel_gpu/include/intel_gpu/primitives/moe_3gemm_fused_compressed.hpp
    src/plugins/intel_gpu/src/graph/moe_3gemm_fused.cpp
    src/plugins/intel_gpu/src/graph/registry/moe_3gemm_swiglu_impls.cpp
    src/plugins/intel_gpu/src/graph/include/moe_3gemm_fused_inst.h

moe transformation implementation:
    src/plugins/intel_gpu/src/plugin/transformations/convert_moe_to_compressed.cpp
    src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.cpp

moe transformation unit test:
    src/plugins/intel_gpu/tests/unit/test_cases/transform_moe_to_compressed_test.cpp
    src/plugins/intel_gpu/tests/unit/test_cases/fuse_moe_3gemm_compressed_test.cpp
    
moe primitive implementation:
    src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.hpp
    src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp

moe kernel implementation:
    src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_mlp.cl
    src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_fuse.cl

moe unit test:
    src/plugins/intel_gpu/tests/unit/test_cases/moe_3gemm_gpu_test.cpp
