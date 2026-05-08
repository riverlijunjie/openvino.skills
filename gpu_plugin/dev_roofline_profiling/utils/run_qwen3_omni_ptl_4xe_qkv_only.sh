#!/usr/bin/env bash
# Re-run only fused QKV cases (decode + prefill S=1024/2048/4096/8192)
source ~/river/openvino/install_release/setupvars.sh
export LD_LIBRARY_PATH=$HOME/river/openvino/temp/Linux_x86_64/tbb/lib:$LD_LIBRARY_PATH
CLI=$HOME/river/clintercept-3.0.6-Linux/bin/cliloader
BUILD=$HOME/river/roofline_test_utils/build
LOGS=$HOME/river/roofline_results/qwen3_omni/ptl_4xe
mkdir -p "$LOGS"

run() { local tag=$1; shift; echo "=== $tag : $*" >> "$LOGS/_index.txt"
  "$CLI" -d "$@" > "$LOGS/$tag.log" 2>&1; }

# Remove obsolete separate q/k/v logs
rm -f "$LOGS"/fc_decode_q.log "$LOGS"/fc_decode_k.log "$LOGS"/fc_decode_v.log
for S in 1024 2048 4096 8192; do
  rm -f "$LOGS/fc_prefill_q_S$S.log" "$LOGS/fc_prefill_k_S$S.log" "$LOGS/fc_prefill_v_S$S.log"
done

# Fused QKV: 2560 -> 6144 (= 4096+1024+1024 for GQA NH=32, NKV=8, HD=128)
run fc_decode_qkv  "$BUILD/fc_bench" 1 2560 6144 128 3000 150 8
for S in 1024 2048 4096 8192; do
  case $S in
    1024) IT=200 ;; 2048) IT=120 ;; 4096) IT=60 ;; 8192) IT=30 ;;
  esac
  run fc_prefill_qkv_S$S "$BUILD/fc_bench" $S 2560 6144 128 $IT 10 4
done
echo "qkv re-run done."
