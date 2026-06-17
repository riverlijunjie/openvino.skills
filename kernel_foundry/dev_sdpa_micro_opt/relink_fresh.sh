#!/usr/bin/env bash
# Link the probe against freshly-rebuilt gemm_jit objects (current-header layout)
# placed ahead of the stale aggregate .a, then run it.
set -u
cd /home/ov2022/workspace/remote_debug/openvino || exit 1
R=$PWD
O=$R/src/plugins/intel_gpu/thirdparty/onednn_gpu
B=$R/build-x86_64-release/src/plugins/intel_gpu/thirdparty/onednn_gpu_build
GEMMJIT_DIR=$B/src/gpu/intel/gemm/jit/CMakeFiles/dnnl_gpu_intel_gemm_jit.dir
STALE_A=$R/build-x86_64-release/src/plugins/intel_gpu/thirdparty/onednn_gpu_install/lib/libopenvino_onednn_gpu.a

echo "fresh .o total: $(find "$GEMMJIT_DIR" -name '*.o' | wc -l)"
echo "fresh .o newer than stale .a: $(find "$GEMMJIT_DIR" -name '*.o' -newer "$STALE_A" | wc -l)"

OBJS=$(find "$GEMMJIT_DIR" -name '*.o' | sort | tr '\n' ' ')
INC="-I$O/src/gpu/intel/gemm/jit -I$B/include -I$O/include -I$R/thirdparty/ocl/cl_headers -I$O/third_party -I$O/src -I$O/src/gpu/intel/jit/config -I$O/third_party/ngen -I$O/src/gpu/intel/gemm/jit/include"
DEF="-DCL_TARGET_OPENCL_VERSION=120 -DDNNL_ENABLE_CONCURRENT_EXEC -DDNNL_ENABLE_CPU_ISA_HINTS -DDNNL_ENABLE_MAX_CPU_ISA -DDNNL_X64=1 -DGEMMSTONE_BUILD_12HP -DGEMMSTONE_BUILD_12LP -DGEMMSTONE_BUILD_12P7 -DGEMMSTONE_BUILD_12P8 -DGEMMSTONE_BUILD_XE2 -DGEMMSTONE_BUILD_XE3 -DGEMMSTONE_BUILD_XE3P -DGEMMSTONE_CONFIG -DNGEN_CONFIG -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -DNDEBUG"

echo "=== linking probe: fresh gemm_jit .o ahead of stale .a (first definition wins) ==="
# shellcheck disable=SC2086
c++ -O2 -std=c++17 -fopenmp -fsigned-char $DEF $INC \
  "$R/.github/skills/dev_sdpa_micro_opt/probe_micro.cpp" \
  $OBJS "$STALE_A" \
  -Wl,--allow-multiple-definition \
  -fopenmp -lpthread -ldl -o /tmp/probe_fresh 2>&1 | tail -25

echo "=== RUN ==="
/tmp/probe_fresh
