#!/bin/bash
# Extension sweep: only S=16K/32K/64K/128K configs (appended to existing logs/bmg/qwen3_moe).
set -e
ROOT=/mnt/river/model_loading/roofline_test_utils
BUILD=$ROOT/build
OV_BIN=/mnt/river/model_loading/openvino/build-x86_64-release/bin/intel64/Release
CLI=/mnt/river/model_loading/clintercept-3.0.6-Linux/bin/cliloader
LOG_DIR=$ROOT/logs/bmg/qwen3_moe
mkdir -p "$LOG_DIR"
export LD_LIBRARY_PATH=$OV_BIN:$LD_LIBRARY_PATH

run() {
  local tag=$1; shift
  echo "== $tag == $@" | tee -a "$LOG_DIR/_index.txt"
  $CLI -d "$@" > "$LOG_DIR/$tag.log" 2>&1 || { echo "FAIL $tag"; tail -10 "$LOG_DIR/$tag.log"; }
}

cd "$BUILD"

# MoE prefill
run moe_prefill_S16384  ./moe_bench 1 16384  2048 768 128 8 128 6 2 1 64
run moe_prefill_S32768  ./moe_bench 1 32768  2048 768 128 8 128 4 1 1 64
run moe_prefill_S65536  ./moe_bench 1 65536  2048 768 128 8 128 2 1 1 64
run moe_prefill_S131072 ./moe_bench 1 131072 2048 768 128 8 128 2 1 1 64

# FC prefill
for S in 16384 32768 65536 131072; do
  run fc_qkv_prefill_S${S} ./fc_bench $S 2048 5120 128 20 5 2 u4 64
  run fc_o_prefill_S${S}   ./fc_bench $S 4096 2048 128 20 5 2 u4 64
done

# PA decode (kv-cache size)
for KV in 16384 32768 65536 131072; do
  run pa_decode_kv${KV} ./pa_bench decode 1 $KV 8000 200 4 i8
done

# PA prefill
run pa_prefill_S16384  ./pa_bench prefill 16384  0 15 3 2 i8
run pa_prefill_S32768  ./pa_bench prefill 32768  0 8  2 2 i8
run pa_prefill_S65536  ./pa_bench prefill 65536  0 4  1 2 i8
run pa_prefill_S131072 ./pa_bench prefill 131072 0 2  1 1 i8

echo "Ext done. Logs in $LOG_DIR"
