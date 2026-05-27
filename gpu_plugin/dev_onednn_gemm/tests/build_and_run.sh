#!/bin/bash
# Build and run the GEMM f16 performance test
# Usage: ./build_and_run.sh
#
# Remote machine (B580):
#   workspace: /mnt/river/kernel_foundry/workspace/copilot
#   Copy this entire 'tests' folder there, then run this script.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Building GEMM f16 test ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

echo ""
echo "=== Running test ==="
cd "${BUILD_DIR}"
./test_gemm_f16
