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

# VLSDPA bench: variable-length SDPA via ov::op::internal::VLSDPA → GPU plugin's
# CM-only cm_sdpa_vlen kernel. Used for ViT-style block-diagonal attention
# (Qwen2-VL / Qwen2.5-VL). Requires Xe2/Xe3 GPU + CM-JIT support at runtime;
# OV_SRC_DIR is needed for the internal ov_ops header.
add_executable(vlsdpa_bench vlsdpa_bench/main.cpp)
target_link_libraries(vlsdpa_bench PRIVATE openvino::runtime)
if(OV_SRC_DIR)
    target_include_directories(vlsdpa_bench PRIVATE
        "${OV_SRC_DIR}/src/common/transformations/include"
    )
endif()

# Conv bench: 3D Convolution for video VAE encoder/decoder (SeedVR2). Covers
# both ov::op::v1::Convolution ("plain", 1x1x1/stride-2 taps) and the
# SeedVR2-project-specific ov::op::internal::CausalConv3DCarry ("causal",
# kernel_t==3 resnet convs with cross-segment temporal carry state). The
# latter is a dev_api-only op, so OV_SRC_DIR is needed.
add_executable(conv_bench conv_bench/main.cpp)
target_link_libraries(conv_bench PRIVATE openvino::runtime)
if(OV_SRC_DIR)
    target_include_directories(conv_bench PRIVATE
        "${OV_SRC_DIR}/src/core/dev_api"
    )
endif()

# GroupNorm bench: ov::op::internal::GroupNormalizationZSplit → GPU plugin's
# cldnn::group_normalization_zsplit primitive. This is the actual op SeedVR2's
# exported VAE encoder/decoder IR uses for every ResnetBlock/attention-prenorm
# GroupNorm (not the plain opset12 GroupNormalization). Header-only, part of
# the installed core include tree -- no OV_SRC_DIR needed.
add_executable(groupnorm_bench groupnorm_bench/main.cpp)
target_link_libraries(groupnorm_bench PRIVATE openvino::runtime)
