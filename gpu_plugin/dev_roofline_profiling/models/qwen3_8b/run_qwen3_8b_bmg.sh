#!/bin/bash
# Qwen3-8B (dense) full roofline sweep on BMG (Arc B580, 2850 MHz, 456 GB/s, 116.74 FP16 TFLOPS).
# Architecture: NL=36, H=4096, I=12288, NH=32, NKV=8, HD=128, vocab=151936
# QKV concat = (NH+2*NKV)*HD = 6144;  O = NH*HD x H = 4096x4096
# MLP gate/up = H x I (4096x12288); down = I x H (12288x4096)
# pa_bench/small_ops_bench defaults already match Qwen3-8B (NH=32 NKV=8 HD=128, H=4096, I=12288).
set -e
ROOT=/mnt/river/model_loading/roofline_test_utils
BUILD=$ROOT/build
OV_LIB=/mnt/river/model_loading/openvino/install_release/runtime/lib/intel64
CLI=/mnt/river/model_loading/clintercept-3.0.6-Linux/bin/cliloader
LOG_DIR=$ROOT/logs/bmg/qwen3_8b
mkdir -p "$LOG_DIR"
export LD_LIBRARY_PATH=$OV_LIB:$LD_LIBRARY_PATH

run() {
  local tag=$1; shift
  echo "== $tag == $@" | tee -a "$LOG_DIR/_index.txt"
  $CLI -d "$@" > "$LOG_DIR/$tag.log" 2>&1 || { echo "FAIL $tag"; tail -30 "$LOG_DIR/$tag.log"; }
}

cd "$BUILD"

# ---------- fc_bench: QKV / O / MLP / LM_Head ----------
# fc_bench  <M> <K> <N> [group_size] [iters] [warmup] [num_bufs] [precision] [flush_mb]
# decode (M=1) — INT4 g=128 weights, FP16 act
run fc_qkv_decode_M1                ./fc_bench 1     4096   6144  128 8000 500 8 u4 64
run fc_o_decode_M1                  ./fc_bench 1     4096   4096  128 8000 500 8 u4 64
run fc_gate_decode_M1               ./fc_bench 1     4096  12288  128 4000 300 8 u4 64
run fc_up_decode_M1                 ./fc_bench 1     4096  12288  128 4000 300 8 u4 64
run fc_down_decode_M1               ./fc_bench 1    12288   4096  128 4000 300 8 u4 64
run lm_head_decode_M1               ./fc_bench 1     4096 151936  128 1000 100 8 u8 64

# prefill (M=S) — INT4 weights still, but FC primitives use INT8 XMX path
for S in 1024 2048 4096 8192 16384 32768; do
  run fc_qkv_prefill_S${S}          ./fc_bench $S    4096   6144  128 30 5 4 u4 64
  run fc_o_prefill_S${S}            ./fc_bench $S    4096   4096  128 30 5 4 u4 64
  run fc_gate_prefill_S${S}         ./fc_bench $S    4096  12288  128 30 5 4 u4 64
  run fc_up_prefill_S${S}           ./fc_bench $S    4096  12288  128 30 5 4 u4 64
  run fc_down_prefill_S${S}         ./fc_bench $S   12288   4096  128 30 5 4 u4 64
done

# ---------- pa_bench: paged attention (NH=32, NKV=8, HD=128, INT8 KV) ----------
for KV in 1024 2048 4096 8192 16384 32768 65536 131072; do
  run pa_decode_kv${KV}             ./pa_bench decode  1 $KV 10000 300 4 i8
done
for S in 1024 2048 4096 8192 16384 32768; do
  run pa_prefill_S${S}              ./pa_bench prefill $S 0 30 5 2 i8
done

# ---------- small ops (Qwen3-8B specific shapes) ----------
# decode shapes
run so_rmsnorm_h4096_decode         ./small_ops_bench rmsnorm   1 4096            --iters 50000 --warmup 500 --bufs 8
run so_rmsnorm3d_qnorm_decode       ./small_ops_bench rmsnorm3d 1 32 128          --iters 50000 --warmup 500 --bufs 8
run so_rmsnorm3d_knorm_decode       ./small_ops_bench rmsnorm3d 1  8 128          --iters 50000 --warmup 500 --bufs 8
run so_rope_q_decode                ./small_ops_bench rope      1 32 128          --iters 50000 --warmup 500 --bufs 8
run so_rope_k_decode                ./small_ops_bench rope      1  8 128          --iters 50000 --warmup 500 --bufs 8
run so_add_decode                   ./small_ops_bench add       1 4096            --iters 50000 --warmup 500 --bufs 8
run so_swish_decode                 ./small_ops_bench swish     1 12288           --iters 50000 --warmup 500 --bufs 8
run so_multiply_decode              ./small_ops_bench multiply  1 12288           --iters 50000 --warmup 500 --bufs 8

# prefill shapes
for S in 1024 2048 4096 8192; do
  run so_rmsnorm_h4096_prefill_S${S}  ./small_ops_bench rmsnorm  $S 4096           --iters 500 --warmup 50 --bufs 4
  run so_rope_q_prefill_S${S}         ./small_ops_bench rope     $S 32 128         --iters 500 --warmup 50 --bufs 4
done

echo ===
echo "All logs in $LOG_DIR"
ls "$LOG_DIR"
