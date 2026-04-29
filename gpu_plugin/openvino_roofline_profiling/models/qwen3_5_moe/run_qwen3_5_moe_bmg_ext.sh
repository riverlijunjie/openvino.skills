#!/bin/bash
# Qwen3.5-MoE-35B-A3B extension sweep on BMG: S/kv = 16K, 32K, 64K, 128K.
# Iterations are reduced relative to S<=8K because:
#   - MoE has 256 experts + shared expert -> ~2x heavier than qwen3_moe
#   - PA prefill is O(S^2)
#   - PA decode at HD=256 is 2x heavier than HD=128
set -e
ROOT=/mnt/river/model_loading/roofline_test_utils
BUILD=$ROOT/build
OV_BIN=/mnt/river/model_loading/openvino/build-x86_64-release/bin/intel64/Release
CLI=/mnt/river/model_loading/clintercept-3.0.6-Linux/bin/cliloader
LOG_DIR=$ROOT/logs/bmg/qwen3_5_moe
mkdir -p "$LOG_DIR"
export LD_LIBRARY_PATH=$OV_BIN:$LD_LIBRARY_PATH

run() {
  local tag=$1; shift
  echo "== $tag == $@" | tee -a "$LOG_DIR/_index.txt"
  $CLI -d "$@" > "$LOG_DIR/$tag.log" 2>&1 || { echo "FAIL $tag"; tail -10 "$LOG_DIR/$tag.log"; }
}
run_env() {
  local tag=$1; shift
  local env_str=$1; shift
  echo "== $tag == [$env_str] $@" | tee -a "$LOG_DIR/_index.txt"
  env $env_str $CLI -d "$@" > "$LOG_DIR/$tag.log" 2>&1 || { echo "FAIL $tag"; tail -10 "$LOG_DIR/$tag.log"; }
}

cd "$BUILD"

# ---------- MoE prefill (NE=256, TK=8, shared_I=512) ----------
run moe_prefill_S16384  ./moe_bench 1 16384  2048 512 256 8 128 5 1 1 64 512
run moe_prefill_S32768  ./moe_bench 1 32768  2048 512 256 8 128 3 1 1 64 512
run moe_prefill_S65536  ./moe_bench 1 65536  2048 512 256 8 128 2 1 1 64 512
run moe_prefill_S131072 ./moe_bench 1 131072 2048 512 256 8 128 2 1 1 64 512

# ---------- FC prefill (full-attn QKV, O) ----------
for S in 16384 32768 65536 131072; do
  run fc_qkv_prefill_S${S}          ./fc_bench $S 2048   5120 128 15 3 2 u4 64
  run fc_o_prefill_S${S}            ./fc_bench $S 4096   2048 128 15 3 2 u4 64
  run fc_linattn_proj_prefill_S${S} ./fc_bench $S 2048  12288 128 10 2 2 u4 64
done

# ---------- PA decode (NH=16 NKV=2 HD=256) ----------
for KV in 16384 32768 65536 131072; do
  run_env pa_decode_kv${KV}         "PA_NH=16 PA_NKV=2 PA_HD=256" ./pa_bench decode 1 $KV 4000 100 4 i8
done

# ---------- PA prefill (NH=16 NKV=2 HD=256) ----------
run_env pa_prefill_S16384  "PA_NH=16 PA_NKV=2 PA_HD=256" ./pa_bench prefill 16384  0 8 2 2 i8
run_env pa_prefill_S32768  "PA_NH=16 PA_NKV=2 PA_HD=256" ./pa_bench prefill 32768  0 4 1 2 i8
run_env pa_prefill_S65536  "PA_NH=16 PA_NKV=2 PA_HD=256" ./pa_bench prefill 65536  0 2 1 1 i8
run_env pa_prefill_S131072 "PA_NH=16 PA_NKV=2 PA_HD=256" ./pa_bench prefill 131072 0 1 1 1 i8

# ---------- GDN prefill (HK=32, K=V=128) ----------
for S in 16384 32768 65536 131072; do
  run gdn_prefill_S${S} ./gdn_bench 1 $S 32 32 128 10 2 2
done

echo "Ext done. Logs in $LOG_DIR"
