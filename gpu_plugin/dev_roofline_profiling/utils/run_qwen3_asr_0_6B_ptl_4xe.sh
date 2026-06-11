#!/usr/bin/env bash
# ============================================================================
# Qwen3-ASR-0.6B roofline sweep on PTL 4Xe Linux  (FP16 weights / INT8 KV)
# Target: intel@10.239.152.140 — Intel PTL 4Xe iGPU, 32 EUs (4 Xe×8), 2450 MHz, 110 GB/s
#
# Text decoder (Qwen3, dense):
#   hidden=1024, layers=28, GQA NH=16 NKV=8 HD=128, intermediate=3072,
#   vocab=151936, tie_word_embeddings=true, hidden_act=silu (SwiGLU)
#
# Audio encoder (Whisper-style, MHA):
#   d_model=896, layers=18, NH=14 HD=64, FFN=3584, S=1500 (fixed)
#   Runs ONCE per inference; profiled below at M=1500.
#
# All FC weights: precision=f16 (NO compression, per user request). LM_Head also f16.
# Text-decoder PA: INT8 KV cache (i8), opencl + micro_kernel.
# Audio encoder SDPA: FP16 (bidirectional, no causal mask, no KV cache).
#
# Input token sweep (text decoder prefill / decode KV): 512, 1024, 4096, 8192.
#
# NOTE: 4Xe has ~1/3 the XMX compute of 12Xe but the SAME 110 GB/s shared-memory
# bandwidth. Compute-bound ops (FC/SDPA prefill, encoder GEMMs) run ~3x slower;
# memory-bound ops (FC/PA decode, small ops, LM_Head) run at ~12Xe speed.
# ============================================================================

source ~/river/openvino/install_release/setupvars.sh
export LD_LIBRARY_PATH=$HOME/river/openvino/temp/Linux_x86_64/tbb/lib:$LD_LIBRARY_PATH

CLI=$HOME/river/clintercept-3.0.6-Linux/bin/cliloader
BUILD=$HOME/river/roofline_test_utils/build
LOGS=$HOME/river/roofline_results/qwen3_asr_0_6B/ptl_4xe
mkdir -p "$LOGS"
echo "=== START $(date)" > "$LOGS/_index.txt"

run() {
  local tag=$1; shift
  echo "=== $tag : $*" >> "$LOGS/_index.txt"
  "$CLI" -d "$@" > "$LOGS/$tag.log" 2>&1
  if [ $? -ne 0 ]; then echo "FAIL $tag" >> "$LOGS/_index.txt"; fi
}

# ===================== Text decoder PA (GQA 16:8, HD=128), INT8 KV =====================
export PA_NH=16
export PA_NKV=8
export PA_HD=128

# PA decode (S_q=1, S_kv = ctx). i8 KV, ocl impl.
run pa_decode_kv512   "$BUILD/pa_bench" decode 1  512 500 50 4 i8 ocl
run pa_decode_kv1024  "$BUILD/pa_bench" decode 1 1024 400 40 4 i8 ocl
run pa_decode_kv4096  "$BUILD/pa_bench" decode 1 4096 200 20 4 i8 ocl
run pa_decode_kv8192  "$BUILD/pa_bench" decode 1 8192 150 15 4 i8 ocl

# PA prefill (S_q=S, S_kv=0). Causal mask -> Sq*(Sq+1)/2 effective pairs.
run pa_prefill_S512   "$BUILD/pa_bench" prefill  512 0 150 20 4 i8 ocl
run pa_prefill_S1024  "$BUILD/pa_bench" prefill 1024 0  80 10 4 i8 ocl
run pa_prefill_S4096  "$BUILD/pa_bench" prefill 4096 0  25  5 4 i8 ocl
run pa_prefill_S8192  "$BUILD/pa_bench" prefill 8192 0  12  3 2 i8 ocl

# ===================== Text decoder FC (decode, M=1, FP16) ====================
# fc_bench: M K N gs iters warmup bufs precision flush_mb
run fc_decode_qkv      "$BUILD/fc_bench" 1 1024   4096 128 6000 200 8 f16
run fc_decode_o        "$BUILD/fc_bench" 1 2048   1024 128 8000 200 8 f16
run fc_decode_gate     "$BUILD/fc_bench" 1 1024   3072 128 6000 200 8 f16
run fc_decode_up       "$BUILD/fc_bench" 1 1024   3072 128 6000 200 8 f16
run fc_decode_down     "$BUILD/fc_bench" 1 3072   1024 128 6000 200 8 f16
run fc_decode_lm_head  "$BUILD/fc_bench" 1 1024 151936 128  300  20 4 f16

# ===================== Text decoder FC (prefill, M=S, FP16) ===================
for S in 512 1024 4096 8192; do
  case $S in
    512)  IT=200 ;;
    1024) IT=150 ;;
    4096) IT=40 ;;
    8192) IT=20 ;;
  esac
  run fc_prefill_qkv_S$S    "$BUILD/fc_bench" $S 1024   4096 128 $IT 10 4 f16
  run fc_prefill_o_S$S      "$BUILD/fc_bench" $S 2048   1024 128 $IT 10 4 f16
  run fc_prefill_gate_S$S   "$BUILD/fc_bench" $S 1024   3072 128 $IT 10 4 f16
  run fc_prefill_up_S$S     "$BUILD/fc_bench" $S 1024   3072 128 $IT 10 4 f16
  run fc_prefill_down_S$S   "$BUILD/fc_bench" $S 3072   1024 128 $IT 10 4 f16
done
run fc_prefill_lm_head      "$BUILD/fc_bench" 1 1024 151936 128 300 20 4 f16

# ===================== Text decoder small ops (decode, M=1) ===================
run small_decode_rmsnorm      "$BUILD/small_ops_bench" rmsnorm   1 1024   --iters 8000 --warmup 200
run small_decode_rmsnorm3d_q  "$BUILD/small_ops_bench" rmsnorm3d 1 16 128 --iters 8000 --warmup 200
run small_decode_rmsnorm3d_k  "$BUILD/small_ops_bench" rmsnorm3d 1  8 128 --iters 8000 --warmup 200
run small_decode_rope_q       "$BUILD/small_ops_bench" rope      1 16 128 --iters 8000 --warmup 200
run small_decode_rope_k       "$BUILD/small_ops_bench" rope      1  8 128 --iters 8000 --warmup 200
run small_decode_add          "$BUILD/small_ops_bench" add       1 1024   --iters 8000 --warmup 200

# ===================== Text decoder small ops (prefill, M=S) ==================
for S in 512 1024 4096 8192; do
  case $S in
    512)  ITS=2000 ;;
    1024) ITS=1500 ;;
    4096) ITS=400 ;;
    8192) ITS=200 ;;
  esac
  run small_prefill_rmsnorm_S$S      "$BUILD/small_ops_bench" rmsnorm   $S 1024   --iters $ITS --warmup 30
  run small_prefill_rmsnorm3d_q_S$S  "$BUILD/small_ops_bench" rmsnorm3d $S 16 128 --iters $ITS --warmup 30
  run small_prefill_rmsnorm3d_k_S$S  "$BUILD/small_ops_bench" rmsnorm3d $S  8 128 --iters $ITS --warmup 30
  run small_prefill_rope_q_S$S       "$BUILD/small_ops_bench" rope      $S 16 128 --iters $ITS --warmup 30
  run small_prefill_rope_k_S$S       "$BUILD/small_ops_bench" rope      $S  8 128 --iters $ITS --warmup 30
  run small_prefill_add_S$S          "$BUILD/small_ops_bench" add       $S 1024   --iters $ITS --warmup 30
done

# ===================== Audio encoder FC (M=1500 fixed, FP16) ==================
# Encoder runs once per inference. K=896 is divisible by 128 (=7 groups); precision=f16
# bypasses the group_size constraint anyway (plain FP16 MatMul).
run fc_enc_qkv_S1500     "$BUILD/fc_bench" 1500  896 2688 128 150 10 4 f16
run fc_enc_o_S1500       "$BUILD/fc_bench" 1500  896  896 128 150 10 4 f16
run fc_enc_fc1_S1500     "$BUILD/fc_bench" 1500  896 3584 128 150 10 4 f16
run fc_enc_fc2_S1500     "$BUILD/fc_bench" 1500 3584  896 128 150 10 4 f16
run fc_enc_outproj_S1500 "$BUILD/fc_bench" 1500  896 1024 128 150 10 4 f16

# ===================== Audio encoder SDPA (bidirectional MHA, FP16) ===========
# Bench the PA-prefill (causal) path as a lower bound; analysis scales it x2 to
# approximate full-square bidirectional attention.
export PA_NH=14
export PA_NKV=14
export PA_HD=64
run pa_prefill_encS1500  "$BUILD/pa_bench" prefill 1500 0 40 10 4 f16 ocl

echo "=== END $(date)" >> "$LOGS/_index.txt"
echo "Done. Logs in $LOGS"
