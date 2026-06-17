#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Build the gemmstone compile/link probe.
set -euo pipefail

# Workspace root (3 levels up from .github/skills/dev_sdpa_micro_opt)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
  -DCL_TARGET_OPENCL_VERSION=120
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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# DEBUG=1 builds an unoptimized, instrumented binary for gdb.
if [[ "${DEBUG:-0}" == "1" ]]; then
  OPT=(-O0 -g)
  OUT="$HERE/probe_micro_dbg"
else
  OPT=(-O2)
  OUT="$HERE/probe_micro"
fi

# The prebuilt libopenvino_onednn_gpu.a may be STALE relative to the current
# gemmstone headers (e.g. the GEMMProblem struct gained members after the lib
# was archived). Linking against a stale archive produces a struct-layout/ABI
# mismatch that crashes at runtime inside selectGEMM()/GEMMProblem::transpose().
#
# Fix: package the freshly compiled gemm-jit objects (the dnnl_gpu_intel_gemm_jit
# target which contains selectGEMM/generateShim/fuse/transpose, PLUS the per-arch
# generator targets generatorXE2/XE3/XE3P which contain
# Generator<Core::XeN>::gemmMicrokernelPackage) into a fresh archive and list it
# BEFORE the stale archive. The linker resolves on demand, so only the members
# the probe actually needs are pulled from the fresh archive (with the current
# struct layout), and everything else still comes from the stale archive. On a
# consistent build the two are identical, so this is always safe. Rebuild the
# objects first with:
#   make -C "$B" -j"$(nproc)" dnnl_gpu_intel_gemm_jit \
#       generatorXE2 generatorXE3 generatorXE3P
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
  "$HERE/probe_micro.cpp" \
  -Wl,--allow-multiple-definition \
  -Wl,--start-group \
  ${FRESH_LIB:+"$FRESH_LIB"} \
  "$ONEDNN_LIB" \
  -Wl,--end-group \
  -fopenmp -lpthread -ldl \
  -o "$OUT"
set +x
echo "Build OK -> $OUT"
