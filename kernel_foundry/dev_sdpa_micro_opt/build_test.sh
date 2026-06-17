#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Build the standalone sdpa_micro generate test.
#
# This links the host-side gemmstone microkernel JIT (selectGEMM / generateShim /
# fuse) from the OpenVINO oneDNN-GPU build, plus OpenCL for the runtime path.
#
# Just like build_probe.sh, the prebuilt libopenvino_onednn_gpu.a on a dev box
# may be STALE relative to the current gemmstone headers (the GEMMProblem struct
# layout drifts), which crashes at runtime inside selectGEMM/transpose. To stay
# robust we pack the freshly-compiled gemm-jit + per-arch generator objects into
# a small archive and list it FIRST inside the link group (resolved on demand,
# so only the symbols we use are pulled from it; identical to the install lib on
# a consistent build). Set USE_FRESH_GEMM_JIT=0 to link the install lib only.
#
# If you change gemmstone sources, rebuild the objects first:
#   make -C "$B" -j"$(nproc)" dnnl_gpu_intel_gemm_jit \
#       generatorXE2 generatorXE3 generatorXE3P
#
# Usage:
#   ./build_test.sh            # -O2 release  -> sdpa_micro_generate_test
#   DEBUG=1 ./build_test.sh    # -O0 -g       -> sdpa_micro_generate_test_dbg
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
O="$ROOT/src/plugins/intel_gpu/thirdparty/onednn_gpu"
B="$ROOT/build-x86_64-release/src/plugins/intel_gpu/thirdparty/onednn_gpu_build"

ONEDNN_LIB="$ROOT/build-x86_64-release/src/plugins/intel_gpu/thirdparty/onednn_gpu_install/lib/libopenvino_onednn_gpu.a"

INCLUDES=(
  -I"$O/src/gpu/intel/gemm/jit"
  -I"$B/include"
  -I"$O/include"
  -I"$ROOT/thirdparty/ocl/cl_headers"
  -I"$O/third_party"
  -I"$O/src"
  -I"$O/src/gpu/intel/jit/config"
  -I"$O/third_party/ngen"
  -I"$O/src/gpu/intel/gemm/jit/include"
)

DEFINES=(
  -DCL_TARGET_OPENCL_VERSION=300
  -DDNNL_ENABLE_CONCURRENT_EXEC
  -DDNNL_ENABLE_CPU_ISA_HINTS
  -DDNNL_ENABLE_MAX_CPU_ISA
  -DDNNL_GPU_ISA_XE2
  -DDNNL_X64=1
  -DGEMMSTONE_BUILD_12HP
  -DGEMMSTONE_BUILD_12LP
  -DGEMMSTONE_BUILD_12P7
  -DGEMMSTONE_BUILD_12P8
  -DGEMMSTONE_BUILD_XE2
  -DGEMMSTONE_BUILD_XE3
  -DGEMMSTONE_BUILD_XE3P
  -DGEMMSTONE_CONFIG
  -DNGEN_CONFIG
  -D__STDC_CONSTANT_MACROS
  -D__STDC_LIMIT_MACROS
  -DNDEBUG
)

# Locate an OpenCL ICD loader to link against.
OCL_LIB=""
for cand in \
  /usr/lib/x86_64-linux-gnu/libOpenCL.so \
  /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 \
  /usr/lib/libOpenCL.so \
  /usr/local/lib/libOpenCL.so; do
  if [[ -e "$cand" ]]; then OCL_LIB="$cand"; break; fi
done
if [[ -z "$OCL_LIB" ]]; then
  # Fall back to -lOpenCL and let the linker find it.
  OCL_LIB="-lOpenCL"
fi

if [[ "${DEBUG:-0}" == "1" ]]; then
  OPT=(-O0 -g)
  OUT="$HERE/sdpa_micro_generate_test_dbg"
else
  OPT=(-O2)
  OUT="$HERE/sdpa_micro_generate_test"
fi

JIT_BASE="$B/src/gpu/intel/gemm/jit/CMakeFiles"
FRESH_DIRS=(
  "$JIT_BASE/dnnl_gpu_intel_gemm_jit.dir"   # selectGEMM / generateShim / fuse / transpose
  "$JIT_BASE/generatorXE2.dir"              # BMG  (Core::Xe2)
  "$JIT_BASE/generatorXE3.dir"              # PTL  (Core::Xe3)
  "$JIT_BASE/generatorXE3P.dir"             # Xe3p (Core::Xe3p)
)
FRESH_LIB=""
if [[ "${USE_FRESH_GEMM_JIT:-1}" == "1" ]]; then
  FRESH_OBJS=()
  for d in "${FRESH_DIRS[@]}"; do
    [[ -d "$d" ]] || continue
    while IFS= read -r f; do FRESH_OBJS+=("$f"); done < <(find "$d" -name '*.o')
  done
  if [[ ${#FRESH_OBJS[@]} -gt 0 ]]; then
    FRESH_LIB="$HERE/libgemmjit_fresh.a"
    rm -f "$FRESH_LIB"
    ar qc "$FRESH_LIB" "${FRESH_OBJS[@]}"
    ranlib "$FRESH_LIB"
    echo "Packed ${#FRESH_OBJS[@]} fresh gemm-jit objects -> $(basename "$FRESH_LIB")"
  fi
fi

set -x
c++ "${OPT[@]}" -std=c++17 -fopenmp -fsigned-char \
  "${DEFINES[@]}" "${INCLUDES[@]}" \
  "$HERE/sdpa_micro_generate_test.cpp" \
  -Wl,--allow-multiple-definition \
  -Wl,--start-group \
  ${FRESH_LIB:+"$FRESH_LIB"} \
  "$ONEDNN_LIB" \
  -Wl,--end-group \
  "$OCL_LIB" \
  -fopenmp -lpthread -ldl \
  -o "$OUT"
set +x
echo "Build OK -> $OUT"
