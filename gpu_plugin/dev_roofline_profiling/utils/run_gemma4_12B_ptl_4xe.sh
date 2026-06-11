#!/usr/bin/env bash
# ========================================================================
#  Gemma4-12B-it (dense) roofline sweep — PTL 4Xe Linux
#  Target: intel@10.239.152.140 — Intel PTL 4Xe iGPU, 32 EUs, 2450 MHz, 110 GB/s
# ========================================================================
#  Architecture (Gemma4-12B-it dense text decoder):
#    vocab_size=262144, hidden_size=3840, num_hidden_layers=48
#    (40 sliding + 8 full, pattern: 5 sliding → 1 full)
#    Sliding attn: NH=16, NKV=8, HD=256 → Q_dim=4096, KV_dim=2048
#    Full attn: NH=16, NKV=1, HD=512 → Q_dim=8192, KV_dim=512
#    sliding_window=1024, attention_k_eq_v=true (full: V reuses K proj)
#    Dense MLP: GEGLU, intermediate=15360
#    tie_word_embeddings=true
#
#  FC shapes (fused):
#    FC_QKV_sliding: 3840 → 8192 (Q:4096+K:2048+V:2048) ×40 layers
#    FC_O_sliding:   4096 → 3840                         ×40 layers
#    FC_QK_full:     3840 → 8704 (Q:8192+K:512, V=K)    ×8  layers
#    FC_O_full:      8192 → 3840                         ×8  layers
#    MLP_gate:       3840 → 15360                        ×48 layers
#    MLP_up:         3840 → 15360                        ×48 layers
#    MLP_down:      15360 → 3840                         ×48 layers
#    LM_head:        3840 → 262144 (INT8 g=128)          ×1
#
#  Quant: INT4 g=128 body, INT8 g=128 LM_head, INT8 KV cache, FP16 act
#  SDPA: PagedAttention OCL + micro_kernel
#  Token sweep: S ∈ {256, 1024, 2048, 4096, 8192}
# ========================================================================

source ~/river/openvino/install_release/setupvars.sh
export LD_LIBRARY_PATH=$HOME/river/openvino/temp/Linux_x86_64/tbb/lib:$LD_LIBRARY_PATH

CLI=$HOME/river/clintercept-3.0.6-Linux/bin/cliloader
BUILD=$HOME/river/roofline_test_utils/build
LOGS=$HOME/river/roofline_results/gemma4_12B/ptl_4xe
mkdir -p "$LOGS"
echo "=== START $(date)" > "$LOGS/_index.txt"

run() {
  local tag=$1; shift
  echo "=== $tag : $*" >> "$LOGS/_index.txt"
  "$CLI" -d "$@" > "$LOGS/$tag.log" 2>&1
  if [ $? -ne 0 ]; then echo "FAIL $tag" >> "$LOGS/_index.txt"; fi
  sleep 2
}

# =====================================================================
#  FC decode (M=1) — Memory-bound (weights dominate at M=1)
#  PTL 4Xe: 110 GB/s → FC decode ~same as 12Xe
# =====================================================================

# FC_QKV_sliding: K=3840, N=8192, ~16 MB weights → ~0.15 ms → 7000 iters
run fc_qkv_sliding_decode_M1    "$BUILD/fc_bench" 1 3840  8192 128 5000 200 8 u4 64
# FC_O_sliding: K=4096, N=3840, ~8 MB weights
run fc_o_sliding_decode_M1      "$BUILD/fc_bench" 1 4096  3840 128 8000 300 8 u4 64
# FC_QK_full: K=3840, N=8704, ~17 MB weights
run fc_qk_full_decode_M1        "$BUILD/fc_bench" 1 3840  8704 128 5000 200 8 u4 64
# FC_O_full: K=8192, N=3840, ~16 MB weights
run fc_o_full_decode_M1         "$BUILD/fc_bench" 1 8192  3840 128 5000 200 8 u4 64
# MLP_gate: K=3840, N=15360, ~30 MB weights
run fc_gate_dense_decode_M1     "$BUILD/fc_bench" 1 3840 15360 128 2500 100 8 u4 64
# MLP_up: K=3840, N=15360
run fc_up_dense_decode_M1       "$BUILD/fc_bench" 1 3840 15360 128 2500 100 8 u4 64
# MLP_down: K=15360, N=3840
run fc_down_dense_decode_M1     "$BUILD/fc_bench" 1 15360 3840 128 2500 100 8 u4 64
# LM_head: K=3840, N=262144, INT8 g=128, ~999 MB → ~9 ms → 120 iters
run lm_head_decode_M1           "$BUILD/fc_bench" 1 3840 262144 128  120  10 4 u8 64

# =====================================================================
#  FC prefill — Compute-bound at large S (INT8 XMX path)
#  PTL 4Xe: INT8 XMX = 40.14 TOPS, FP16 XMX = 20.07 TFLOPS
# =====================================================================
for S in 256 1024 2048 4096 8192; do
  case $S in
    256)  IT=400; WU=20 ;;
    1024) IT=100; WU=10 ;;
    2048) IT=50;  WU=5 ;;
    4096) IT=25;  WU=3 ;;
    8192) IT=12;  WU=2 ;;
  esac
  run fc_qkv_sliding_prefill_S$S  "$BUILD/fc_bench" $S 3840  8192 128 $IT $WU 4 u4 64
  run fc_o_sliding_prefill_S$S    "$BUILD/fc_bench" $S 4096  3840 128 $IT $WU 4 u4 64
  run fc_qk_full_prefill_S$S      "$BUILD/fc_bench" $S 3840  8704 128 $IT $WU 4 u4 64
  run fc_o_full_prefill_S$S       "$BUILD/fc_bench" $S 8192  3840 128 $IT $WU 4 u4 64
  run fc_gate_dense_prefill_S$S   "$BUILD/fc_bench" $S 3840 15360 128 $IT $WU 4 u4 64
  run fc_up_dense_prefill_S$S     "$BUILD/fc_bench" $S 3840 15360 128 $IT $WU 4 u4 64
  run fc_down_dense_prefill_S$S   "$BUILD/fc_bench" $S 15360 3840 128 $IT $WU 4 u4 64
done
# LM_head is always single-token
run lm_head_prefill             "$BUILD/fc_bench" 1 3840 262144 128 120 10 4 u8 64

# =====================================================================
#  PA decode — Sliding attention (NH=16, NKV=8, HD=256)
#  Sliding window = 1024 → effective KV = min(ctx, 1024)
# =====================================================================
export PA_NH=16
export PA_NKV=8
export PA_HD=256

# For kv <= 1024, effective is kv; for kv > 1024, effective is 1024
run pa_sliding_decode_kv256      "$BUILD/pa_bench" decode 1  256  6000 200 4 i8 ocl
run pa_sliding_decode_kv1024     "$BUILD/pa_bench" decode 1 1024  3000 150 4 i8 ocl

# =====================================================================
#  PA decode — Full attention (NH=16, NKV=1, HD=512)
# =====================================================================
export PA_NH=16
export PA_NKV=1
export PA_HD=512

for K in 256 1024 2048 4096 8192; do
  case $K in
    256)  IT=6000; WU=200 ;;
    1024) IT=3000; WU=150 ;;
    2048) IT=2000; WU=100 ;;
    4096) IT=1000; WU=50 ;;
    8192) IT=500;  WU=30 ;;
  esac
  run pa_full_decode_kv$K        "$BUILD/pa_bench" decode 1 $K $IT $WU 4 i8 ocl
done

# =====================================================================
#  PA prefill — Sliding attention (NH=16, NKV=8, HD=256)
#  S ≤ 1024 only (sliding window = 1024)
# =====================================================================
export PA_NH=16
export PA_NKV=8
export PA_HD=256

run pa_sliding_prefill_S256      "$BUILD/pa_bench" prefill 256  0 400 30 4 i8 ocl
run pa_sliding_prefill_S1024     "$BUILD/pa_bench" prefill 1024 0 150 15 4 i8 ocl

# =====================================================================
#  PA prefill — Full attention (NH=16, NKV=1, HD=512)
# =====================================================================
export PA_NH=16
export PA_NKV=1
export PA_HD=512

for S in 256 1024 2048 4096 8192; do
  case $S in
    256)  IT=400; WU=30 ;;
    1024) IT=120; WU=15 ;;
    2048) IT=50;  WU=8 ;;
    4096) IT=20;  WU=3 ;;
    8192) IT=8;   WU=2 ;;
  esac
  run pa_full_prefill_S$S        "$BUILD/pa_bench" prefill $S 0 $IT $WU 4 i8 ocl
done

# =====================================================================
#  Small ops — Decode (M=1)
# =====================================================================
# RMSNorm on hidden: 3840
run small_decode_rmsnorm         "$BUILD/small_ops_bench" rmsnorm   1 3840   --iters 10000 --warmup 300 --bufs 8
# RMSNorm3D on Q (sliding): [1, 16, 256]
run small_decode_rmsnorm3d_q_sl  "$BUILD/small_ops_bench" rmsnorm3d 1 16 256 --iters 10000 --warmup 300 --bufs 8
# RMSNorm3D on K (sliding): [1, 8, 256]
run small_decode_rmsnorm3d_k_sl  "$BUILD/small_ops_bench" rmsnorm3d 1 8  256 --iters 10000 --warmup 300 --bufs 8
# RMSNorm3D on Q (full): [1, 16, 512]
run small_decode_rmsnorm3d_q_f   "$BUILD/small_ops_bench" rmsnorm3d 1 16 512 --iters 10000 --warmup 300 --bufs 8
# RoPE on Q (sliding): [1, 16, 256]
run small_decode_rope_q_sl       "$BUILD/small_ops_bench" rope      1 16 256 --iters 10000 --warmup 300 --bufs 8
# RoPE on K (sliding): [1, 8, 256]
run small_decode_rope_k_sl       "$BUILD/small_ops_bench" rope      1 8  256 --iters 10000 --warmup 300 --bufs 8
# RoPE on Q (full): [1, 16, 512]
run small_decode_rope_q_f        "$BUILD/small_ops_bench" rope      1 16 512 --iters 10000 --warmup 300 --bufs 8
# Add (residual): [1, 3840]
run small_decode_add             "$BUILD/small_ops_bench" add       1 3840   --iters 10000 --warmup 300 --bufs 8

# =====================================================================
#  Small ops — Prefill
# =====================================================================
for S in 256 1024 2048 4096 8192; do
  case $S in
    256)  ITS=3000 ;;
    1024) ITS=1000 ;;
    2048) ITS=500 ;;
    4096) ITS=250 ;;
    8192) ITS=120 ;;
  esac
  run small_prefill_rmsnorm_S$S        "$BUILD/small_ops_bench" rmsnorm   $S 3840    --iters $ITS --warmup 30 --bufs 4
  run small_prefill_rmsnorm3d_q_sl_S$S "$BUILD/small_ops_bench" rmsnorm3d $S 16 256  --iters $ITS --warmup 30 --bufs 4
  run small_prefill_rmsnorm3d_k_sl_S$S "$BUILD/small_ops_bench" rmsnorm3d $S 8  256  --iters $ITS --warmup 30 --bufs 4
  run small_prefill_rmsnorm3d_q_f_S$S  "$BUILD/small_ops_bench" rmsnorm3d $S 16 512  --iters $ITS --warmup 30 --bufs 4
  run small_prefill_rope_q_sl_S$S      "$BUILD/small_ops_bench" rope      $S 16 256  --iters $ITS --warmup 30 --bufs 4
  run small_prefill_rope_k_sl_S$S      "$BUILD/small_ops_bench" rope      $S 8  256  --iters $ITS --warmup 30 --bufs 4
  run small_prefill_rope_q_f_S$S       "$BUILD/small_ops_bench" rope      $S 16 512  --iters $ITS --warmup 30 --bufs 4
  run small_prefill_add_S$S            "$BUILD/small_ops_bench" add       $S 3840    --iters $ITS --warmup 30 --bufs 4
done

echo "=== END $(date)" >> "$LOGS/_index.txt"
echo "Done. Logs in $LOGS"
