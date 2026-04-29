#!/usr/bin/env bash
set -euo pipefail

# Template runner for Linux hosts.
# Fill in the placeholders below for your environment and model.

SKILL_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_NAME="replace_me"
REMOTE_WORKDIR="/path/to/roofline_workdir"
TOKEN_POINTS=(1024 2048 4096 8192)

# Example environment contract:
#   OPENVINO_ROOT  - OpenVINO source or install root
#   CLILOADER_BIN  - path to cliloader binary
#   BENCH_BUILD    - build directory for utils/CMakeLists.txt
#   OUTPUT_DIR     - where raw logs should be written

: "${OPENVINO_ROOT:?Set OPENVINO_ROOT}"
: "${CLILOADER_BIN:?Set CLILOADER_BIN}"
: "${BENCH_BUILD:?Set BENCH_BUILD}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"

for S in "${TOKEN_POINTS[@]}"; do
  echo "[template] Run prefill/decode measurements for ${MODEL_NAME} at S=${S}"
  echo "[template] Invoke the appropriate bench binaries here with cliloader."
  echo "[template] Save each log to: ${OUTPUT_DIR}/${MODEL_NAME}_<mode>_${S}.log"
done

cat <<EOF
Template complete.
Next steps:
  1. Replace echo statements with actual bench invocations.
  2. Copy logs into outputs/${MODEL_NAME}/logs*.
  3. Parse with utils/parse_logs.py.
  4. Rebuild db/metrics.db with utils/build_db.py.
EOF
