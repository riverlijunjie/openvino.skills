#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Runs the GGUF kernel benchmark under cliloader (Intel OpenCL Intercept Layer) so each
# kernel's INDEPENDENT device-time table is captured alongside the harness's in-process
# CL-event timings, then prints both. cliloader gives a neutral, tool-measured ground truth
# for the per-kernel GPU time (it cannot be skewed by host overhead in the Python loop).
#
# Usage:
#   ./run.sh [--quick | --config configs/myshapes.json] [--out results/report.json]
#   CLILOADER=/path/to/cliloader ./run.sh ...     # override cliloader location
#
# Without cliloader on PATH it falls back to a plain run (in-process CL-event timing only),
# which is already accurate; cliloader just adds an external cross-check + per-enqueue trace.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/harness"

# Locate cliloader: env override, PATH, or the source build we create under /tmp.
CLILOADER="${CLILOADER:-}"
if [[ -z "$CLILOADER" ]]; then
  if command -v cliloader >/dev/null 2>&1; then
    CLILOADER="$(command -v cliloader)"
  elif [[ -x /tmp/opencl-intercept-layer/build/cliloader/cliloader ]]; then
    CLILOADER="/tmp/opencl-intercept-layer/build/cliloader/cliloader"
  fi
fi

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  ARGS=(--config "../configs/default.json")
fi

export PYOPENCL_COMPILER_OUTPUT=0
PYBIN="$(command -v python3)"

if [[ -n "$CLILOADER" && -x "$CLILOADER" ]]; then
  echo "### Running under cliloader: $CLILOADER"
  # -dv  : device timing, verbose summary at exit (per-kernel GPU ns + call counts)
  # -q   : quiet host-call logging (keep only the device-timing table)
  # KernelNameHashTracking=0 keeps our JIT entry-point names readable in the report.
  CLI_OPTS=(-dv -q)
  "$CLILOADER" "${CLI_OPTS[@]}" "$PYBIN" bench.py "${ARGS[@]}"
else
  echo "### cliloader not found -- running with in-process CL-event timing only."
  echo "### (build it: cd /tmp/opencl-intercept-layer && cmake -S . -B build -DENABLE_CLILOADER=ON && cmake --build build -j)"
  "$PYBIN" bench.py "${ARGS[@]}"
fi
