#!/usr/bin/env bash
# Run every standalone QSA kernel perf sweep (1024/2048/4096) on the remote BMG GPU.
# Each test averages over N iterations with a GPU-cache flush between measurements,
# and prints the measured mean vs the analytic roofline.
#
#   ./run_all.sh [ITERS]
set -u
cd "$(dirname "$0")"

ITERS="${1:-50}"
PY="./.venv/bin/python"
export CM_FE_DIR="${CM_FE_DIR:-/mnt/river}"
export cl_cache_dir="${cl_cache_dir:-/tmp/cl_cache_qsa}"

run() { echo "===== $1 ====="; timeout 1200 "$PY" "$@" --iters "$ITERS" 2>&1 | grep -aE 'eff=|correctness:|FAIL|Error|Traceback'; }

run test_q0_kv_cache_update.py
run test_q1_prepare.py
run test_q2_score_topk_fused.py
run test_q2_score_tile_dpas.py
run test_q2_score_partition.py
run test_q2_topk_finalization.py
run test_q3_sparse_attention.py
run test_q3_sparse_attention_dpas.py
