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

# Small ops bench: RMSNorm, SwiGLU, Add, RoPE (decomposed public ops → GPU fusion)
add_executable(small_ops_bench small_ops_bench/main.cpp)
target_link_libraries(small_ops_bench PRIVATE openvino::runtime)

# MoE bench: builds the canonical softmax-routing 3-GEMM SwiGLU MoE subgraph
# (and optionally an always-on shared expert) so the GPU plugin's transformation
# chain MatMulExpertsFusion -> ConvertMOEToMOECompressed -> FuseMOE3GemmCompressed
# (with FuseMOESharedExpert in front when shared_I>0) collapses everything into
# the single fused primitive moe_3gemm_fused_compressed. Verifies fusion at
# runtime; fails the run if the fused node is missing or any shared-expert
# MatMul survived.
add_executable(moe_bench moe_bench/main.cpp)
target_link_libraries(moe_bench PRIVATE openvino::runtime)
