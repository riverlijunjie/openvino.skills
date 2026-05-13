#!/usr/bin/env bash
# qwen3_omni Thinker text decoder roofline sweep on PTL 4Xe Linux
# Target: intel@10.239.152.140 — Intel PTL 4Xe iGPU, 32 EUs, 2450 MHz, 110 GB/s
#
# Architecture (config.json thinker_config.text_config):
#   hidden=2560, layers=36, GQA NH=32 NKV=8, head_dim=128,
#   intermediate=9728, vocab=151936, tie_word_embeddings=true,
#   hidden_act=silu (SwiGLU), rope_theta=1e6
#
# GQA shapes:
#   Q  proj : 2560 -> 32*128=4096   (M x 2560 x 4096)
#   K  proj : 2560 -> 8*128=1024    (M x 2560 x 1024)
#   V  proj : 2560 -> 8*128=1024    (M x 2560 x 1024)
#   O  proj : 32*128=4096 -> 2560   (M x 4096 x 2560)
#   gate/up : 2560 -> 9728          (M x 2560 x 9728)
#   down    : 9728 -> 2560          (M x 9728 x 2560)
#   lm_head : 2560 -> 151936 (INT8, single-token)

source ~/river/openvino/install_release/setupvars.sh
export LD_LIBRARY_PATH=$HOME/river/openvino/temp/Linux_x86_64/tbb/lib:$LD_LIBRARY_PATH

CLI=$HOME/river/clintercept-3.0.6-Linux/bin/cliloader
BUILD=$HOME/river/roofline_test_utils/build
LOGS=$HOME/river/roofline_results/qwen3_omni/ptl_4xe
mkdir -p "$LOGS"
echo "=== START $(date)" > "$LOGS/_index.txt"

export PA_NH=32
export PA_NKV=8
export PA_HD=128

run() {
  local tag=$1; shift
  echo "=== $tag : $*" >> "$LOGS/_index.txt"
  "$CLI" -d "$@" > "$LOGS/$tag.log" 2>&1
  if [ $? -ne 0 ]; then echo "FAIL $tag" >> "$LOGS/_index.txt"; fi
}

# ============ PA decode (S_q=1, S_kv = ctx) ============
run pa_decode_kv1024  "$BUILD/pa_bench" decode 1   1024 300 30 4 i8 ocl
run pa_decode_kv2048  "$BUILD/pa_bench" decode 1   2048 300 30 4 i8 ocl
run pa_decode_kv4096  "$BUILD/pa_bench" decode 1   4096 200 20 4 i8 ocl
run pa_decode_kv8192  "$BUILD/pa_bench" decode 1   8192 150 15 4 i8 ocl

# ============ PA prefill (S_q=S, S_kv=0) ============
run pa_prefill_S1024  "$BUILD/pa_bench" prefill 1024 0 100 10 4 i8 ocl
run pa_prefill_S2048  "$BUILD/pa_bench" prefill 2048 0  60 10 4 i8 ocl
run pa_prefill_S4096  "$BUILD/pa_bench" prefill 4096 0  40  8 4 i8 ocl
run pa_prefill_S8192  "$BUILD/pa_bench" prefill 8192 0  25  5 2 i8 ocl

# ============ FC decode (M=1) - INT4 ============
# Fused QKV: 2560 -> Q_dim+K_dim+V_dim = 4096+1024+1024 = 6144
run fc_decode_qkv      "$BUILD/fc_bench" 1 2560   6144  128 3000 150 8
run fc_decode_o        "$BUILD/fc_bench" 1 4096   2560  128 5000 200 8
run fc_decode_gate     "$BUILD/fc_bench" 1 2560   9728  128 2000 100 8
run fc_decode_up       "$BUILD/fc_bench" 1 2560   9728  128 2000 100 8
run fc_decode_down     "$BUILD/fc_bench" 1 9728   2560  128 2000 100 8
run fc_decode_lm_head  "$BUILD/fc_bench" 1 2560 151936  128  150  10 4 u8

# ============ FC prefill (M=S) - INT4 ============
for S in 1024 2048 4096 8192; do
  case $S in
    1024) IT=200 ;;
    2048) IT=120 ;;
    4096) IT=60 ;;
    8192) IT=30 ;;
  esac
  run fc_prefill_qkv_S$S    "$BUILD/fc_bench" $S 2560   6144 128 $IT 10 4
  run fc_prefill_o_S$S      "$BUILD/fc_bench" $S 4096   2560 128 $IT 10 4
  run fc_prefill_gate_S$S   "$BUILD/fc_bench" $S 2560   9728 128 $IT 10 4
  run fc_prefill_up_S$S     "$BUILD/fc_bench" $S 2560   9728 128 $IT 10 4
  run fc_prefill_down_S$S   "$BUILD/fc_bench" $S 9728   2560 128 $IT 10 4
done
run fc_prefill_lm_head      "$BUILD/fc_bench" 1 2560 151936 128 150 10 4 u8

# ============ Small ops decode (M=1) ============
run small_decode_rmsnorm        "$BUILD/small_ops_bench" rmsnorm   1 2560   --iters 5000 --warmup 200
run small_decode_rmsnorm3d_q    "$BUILD/small_ops_bench" rmsnorm3d 1 32 128 --iters 5000 --warmup 200
run small_decode_rmsnorm3d_k    "$BUILD/small_ops_bench" rmsnorm3d 1 8  128 --iters 5000 --warmup 200
run small_decode_rope_q         "$BUILD/small_ops_bench" rope      1 32 128 --iters 5000 --warmup 200
run small_decode_rope_k         "$BUILD/small_ops_bench" rope      1 8  128 --iters 5000 --warmup 200
run small_decode_swish          "$BUILD/small_ops_bench" swish     1 9728   --iters 5000 --warmup 200
run small_decode_multiply       "$BUILD/small_ops_bench" multiply  1 9728   --iters 5000 --warmup 200
run small_decode_add            "$BUILD/small_ops_bench" add       1 2560   --iters 5000 --warmup 200

# ============ Small ops prefill ============
for S in 1024 2048 4096 8192; do
  case $S in
    1024) ITS=1500 ;;
    2048) ITS=800 ;;
    4096) ITS=400 ;;
    8192) ITS=200 ;;
  esac
  run small_prefill_rmsnorm_S$S      "$BUILD/small_ops_bench" rmsnorm   $S 2560    --iters $ITS --warmup 30
  run small_prefill_rmsnorm3d_q_S$S  "$BUILD/small_ops_bench" rmsnorm3d $S 32 128  --iters $ITS --warmup 30
  run small_prefill_rmsnorm3d_k_S$S  "$BUILD/small_ops_bench" rmsnorm3d $S 8  128  --iters $ITS --warmup 30
  run small_prefill_rope_q_S$S       "$BUILD/small_ops_bench" rope      $S 32 128  --iters $ITS --warmup 30
  run small_prefill_rope_k_S$S       "$BUILD/small_ops_bench" rope      $S 8  128  --iters $ITS --warmup 30
  run small_prefill_swish_S$S        "$BUILD/small_ops_bench" swish     $S 9728    --iters $ITS --warmup 30
  run small_prefill_multiply_S$S     "$BUILD/small_ops_bench" multiply  $S 9728    --iters $ITS --warmup 30
  run small_prefill_add_S$S          "$BUILD/small_ops_bench" add       $S 2560    --iters $ITS --warmup 30
done

echo "=== END $(date)" >> "$LOGS/_index.txt"
echo "Done. Logs in $LOGS"
