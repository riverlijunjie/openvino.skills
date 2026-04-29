cmake_minimum_required(VERSION 3.16)
project(roofline_bench LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# OV_SRC_DIR should point to the OpenVINO source root (for internal/dev_api headers)
set(OV_SRC_DIR "" CACHE PATH "Path to OpenVINO source tree (for dev_api headers)")

# FC bench: standard dequantization pattern (u4 → Convert → Sub → Mul → Reshape → MatMul)
# The GPU plugin's KeepConstPrecision pass marks the u4 constant, preventing u4→u8 conversion.
add_executable(fc_bench fc_bench/main.cpp)
target_link_libraries(fc_bench PRIVATE openvino::runtime)

# PA bench: needs dev_api header for PagedAttentionExtension
add_executable(pa_bench pa_bench/main.cpp)
target_link_libraries(pa_bench PRIVATE openvino::runtime)
if(OV_SRC_DIR)
    target_include_directories(pa_bench PRIVATE
        "${OV_SRC_DIR}/src/core/dev_api"
        "${OV_SRC_DIR}/src/inference/dev_api"
    )
endif()

# SDPA bench: ov::op::v13::ScaledDotProductAttention (uncompressed attention).
# Use this when the model's attention is NOT converted to PagedAttention, or
# when comparing SDPA vs PA roofline on the same shapes. KV-cache compression
# is NOT modelled here (use pa_bench for INT8 KV).
add_executable(sdpa_bench sdpa_bench/main.cpp)
target_link_libraries(sdpa_bench PRIVATE openvino::runtime)

# Small ops bench: RMSNorm, SwiGLU, Add, RoPE (decomposed public ops → GPU fusion)
add_executable(small_ops_bench small_ops_bench/main.cpp)
target_link_libraries(small_ops_bench PRIVATE openvino::runtime)

# MoE bench: canonical softmax-routing 3-GEMM MoE pattern → fused by MoE3GemmFusion
# into the cldnn::moe_3gemm_fused_compressed primitive. References
# src/plugins/intel_gpu/tests/unit/test_cases/moe_3gemm_gpu_test.cpp and the
# canonical pattern in src/tests/test_utils/common_test_utils/src/node_builders/moe_builders.cpp.
add_executable(moe_bench moe_bench/main.cpp)
target_link_libraries(moe_bench PRIVATE openvino::runtime)

# GDN bench: linear-attention via ov::op::internal::GatedDeltaNet → GPU plugin's
# gated_delta_net_ref OCL kernel. Used for Qwen3.5-MoE / Qwen3-Next style hybrid
# attention layers. Needs OV_SRC_DIR for the dev_api header.
add_executable(gdn_bench gdn_bench/main.cpp)
target_link_libraries(gdn_bench PRIVATE openvino::runtime)
if(OV_SRC_DIR)
    target_include_directories(gdn_bench PRIVATE
        "${OV_SRC_DIR}/src/core/dev_api"
    )
endif()
