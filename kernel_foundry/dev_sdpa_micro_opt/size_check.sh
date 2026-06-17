#!/usr/bin/env bash
# Compare GEMMProblem struct layout under the lib's c++11 flags vs the probe's c++17 flags.
set -u
cd /home/ov2022/workspace/remote_debug/openvino || exit 1
O=src/plugins/intel_gpu/thirdparty/onednn_gpu
B=build-x86_64-release/src/plugins/intel_gpu/thirdparty/onednn_gpu_build
INC="-I$O/src/gpu/intel/gemm/jit -I$O/src/gpu/intel/gemm/jit/dnnl_gpu_intel_gemm_jit -I$B/include -I$O/include -Ithirdparty/ocl/cl_headers -I$O/third_party -I$O/src -I$O/src/gpu/intel/jit/config -I$O/third_party/ngen -I$O/src/gpu/intel/gemm/jit/include"
LIBDEF="-DCL_TARGET_OPENCL_VERSION=120 -DDNNL_ENABLE_CONCURRENT_EXEC -DDNNL_ENABLE_CPU_ISA_HINTS -DDNNL_ENABLE_MAX_CPU_ISA -DDNNL_X64=1 -DGEMMSTONE_BUILD_12HP -DGEMMSTONE_BUILD_12LP -DGEMMSTONE_BUILD_12P7 -DGEMMSTONE_BUILD_12P8 -DGEMMSTONE_BUILD_XE2 -DGEMMSTONE_BUILD_XE3 -DGEMMSTONE_BUILD_XE3P -DGEMMSTONE_CONFIG -DNGEN_CONFIG -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS"
SP=.github/skills/dev_sdpa_micro_opt/size_probe.cpp

echo "===== LIB flags (c++11) ====="
c++ -O3 -DNDEBUG -std=c++11 -fopenmp -fsigned-char $LIBDEF $INC "$SP" -o /tmp/size_lib 2>&1 | head -40
/tmp/size_lib

echo "===== PROBE flags (c++17) ====="
c++ -O2 -DNDEBUG -std=c++17 -fopenmp -fsigned-char $LIBDEF $INC "$SP" -o /tmp/size_probe17 2>&1 | head -40
/tmp/size_probe17
