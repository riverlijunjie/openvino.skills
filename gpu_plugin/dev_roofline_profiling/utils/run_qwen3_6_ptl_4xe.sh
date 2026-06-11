#!/usr/bin/env bash
# ============================================================================
# Qwen3.6-35B-A3B roofline sweep on PTL 4Xe Linux
# Target: intel@10.239.152.140 — Intel PTL 4Xe iGPU, 32 EUs (4 Xe×8), 2450 MHz, 110 GB/s
#
# FP16 XMX peak = 4 × 8 × 256 × 2.45 GHz = 20.07 TFLOPS
# INT8 XMX peak = 4 × 8 × 512 × 2.45 GHz = 40.14 TOPS
#
# Architecture = qwen3_5_moe + attention output gate (attn_output_gate=true):
#   - full-attn QKV+gate fused FC width = 2*16*256 + 2*2*256 = 9216 (was 5120)
#   - extra attn output gate elementwise (attn_output * sigmoid(gate)) H=4096
#   - MoE: H=2048 I=512 NE=256 TK=8 + shared_I=512 (always-on shared expert)
#   - LM_Head: vocab=248320 (INT8 g=128); body FC INT4 g=128; KV cache INT8
#   - PA: NH=16 NKV=2 HD=256 (GQA=8)
#   - GatedDeltaNet linear-attention bench (PagedGatedDeltaNet opt; qk_heads=16, v_heads=32, K=V=128)
#
# DECODE-512 WINDOW: we measure the decoding cost of generating 512 tokens for
# each prompt P in {1024,2048,4096,8192}. Only PA grows with KV length over the
# window; FC/MoE/GDN/small ops are M=1 constant. We therefore sweep PA decode at
# KV = P (window start), P+256 (window average), P+512 (window end).
#
# NOTE: 4Xe has 1/3 the XMX compute of 12Xe but same 110 GB/s bandwidth.
# Compute-bound ops (FC/MoE/PA prefill) run ~3x slower; memory-bound ops
# (FC/PA decode, small ops) run at ~12Xe speed.
# ============================================================================

source ~/river/openvino/install_release/setupvars.sh
export LD_LIBRARY_PATH=$HOME/river/openvino/temp/Linux_x86_64/tbb/lib:$LD_LIBRARY_PATH

CLI=$HOME/river/clintercept-3.0.6-Linux/bin/cliloader
BUILD=$HOME/river/roofline_test_utils/build
LOGS=$HOME/river/roofline_results/qwen3_6/ptl_4xe
mkdir -p "$LOGS"
echo "=== START $(date)" > "$LOGS/_index.txt"

run() {
  local tag=$1; shift
  echo "=== $tag : $*" >> "$LOGS/_index.txt"
  "$CLI" -d "$@" > "$LOGS/$tag.log" 2>&1
  if [ $? -ne 0 ]; then echo "FAIL $tag" >> "$LOGS/_index.txt"; fi
}

# ============ MoE fused (with shared expert) ============
# moe_bench: B M H I NE TK gs iters warmup bufs flush_mb shared_I
run moe_decode_M1         "$BUILD/moe_bench" 1 1    2048 512 256 8 128 100 10 4 64 512
run moe_prefill_S1024     "$BUILD/moe_bench" 1 1024 2048 512 256 8 128  8  2 2 64 512
run moe_prefill_S2048     "$BUILD/moe_bench" 1 2048 2048 512 256 8 128  5  2 2 64 512
run moe_prefill_S4096     "$BUILD/moe_bench" 1 4096 2048 512 256 8 128  4  2 2 64 512
run moe_prefill_S8192     "$BUILD/moe_bench" 1 8192 2048 512 256 8 128  3  2 2 64 512

# ============ FC decode (M=1) ============
# fc_bench: M K N gs iters warmup bufs quant flush_mb
run fc_qkv_decode_M1      "$BUILD/fc_bench" 1    2048   9216 128 5000 200 8 u4 64
run fc_o_decode_M1        "$BUILD/fc_bench" 1    4096   2048 128 5000 200 8 u4 64
run lm_head_decode_M1     "$BUILD/fc_bench" 1    2048 248320 128  300  30 4 u8 64

# ============ FC prefill (M=S) ============
for S in 1024 2048 4096 8192; do
  case $S in
    1024) IT=15 ;;
    2048) IT=10 ;;
    4096) IT=6 ;;
    8192) IT=4 ;;
  esac
  run fc_qkv_prefill_S$S  "$BUILD/fc_bench" $S 2048 9216 128 $IT 3 4 u4 64
  run fc_o_prefill_S$S    "$BUILD/fc_bench" $S 4096 2048 128 $IT 3 4 u4 64
done

# ============ Linear-attention input projection FC (2048 -> 12288) ============
run fc_linattn_proj_decode_M1 "$BUILD/fc_bench" 1 2048 12288 128 1500 100 4 u4 64
for S in 1024 2048 4096 8192; do
  case $S in
    1024) IT=10 ;;
    2048) IT=6 ;;
    4096) IT=4 ;;
    8192) IT=3 ;;
  esac
  run fc_linattn_proj_prefill_S$S "$BUILD/fc_bench" $S 2048 12288 128 $IT 2 4 u4 64
done

# ============ PA decode (NH=16 NKV=2 HD=256, i8 KV) ============
export PA_NH=16
export PA_NKV=2
export PA_HD=256
# Decode KV sweep: window start (P), average (P+256), end (P+512) for each prompt P.
for K in 1024 1280 1536 2048 2304 2560 4096 4352 4608 8192 8448 8704; do
  run pa_decode_kv$K      "$BUILD/pa_bench" decode 1 $K 8000 200 4 i8 ocl
done

# ============ PA prefill (causal, S_kv=0) ============
run pa_prefill_S1024      "$BUILD/pa_bench" prefill 1024 0 30 5 4 i8 ocl
run pa_prefill_S2048      "$BUILD/pa_bench" prefill 2048 0 15 3 4 i8 ocl
run pa_prefill_S4096      "$BUILD/pa_bench" prefill 4096 0 10 3 2 i8 ocl
run pa_prefill_S8192      "$BUILD/pa_bench" prefill 8192 0  5 2 2 i8 ocl

# ============ GDN (GatedDeltaNet, linear attention) ============
# gdn_bench: B T qk_heads v_heads head_dim iters warmup bufs cache_interval
run gdn_decode_T1         "$BUILD/gdn_bench" 1 1    16 32 128 4000 150 4 0
run gdn_prefill_S1024     "$BUILD/gdn_bench" 1 1024 16 32 128  8   2 2 0
run gdn_prefill_S2048     "$BUILD/gdn_bench" 1 2048 16 32 128  5   2 2 0
run gdn_prefill_S4096     "$BUILD/gdn_bench" 1 4096 16 32 128  3   2 2 0
run gdn_prefill_S8192     "$BUILD/gdn_bench" 1 8192 16 32 128  2   1 2 0

# ============ Small ops decode (M=1) ============
# hidden=2048; full-attn 16/2 heads HD=256; gate H=4096
run so_rmsnorm_h2048_decode   "$BUILD/small_ops_bench" rmsnorm   1 2048 --iters 30000 --warmup 300 --bufs 8
run so_rmsnorm3d_qnorm_decode "$BUILD/small_ops_bench" rmsnorm3d 1 16 256 --iters 30000 --warmup 300 --bufs 8
run so_rmsnorm3d_knorm_decode "$BUILD/small_ops_bench" rmsnorm3d 1  2 256 --iters 30000 --warmup 300 --bufs 8
run so_rope_q_decode          "$BUILD/small_ops_bench" rope      1 16 256 --iters 30000 --warmup 300 --bufs 8
run so_rope_k_decode          "$BUILD/small_ops_bench" rope      1  2 256 --iters 30000 --warmup 300 --bufs 8
run so_add_decode             "$BUILD/small_ops_bench" add       1 2048 --iters 30000 --warmup 300 --bufs 8
run so_gate_decode            "$BUILD/small_ops_bench" gate      1 4096 --iters 30000 --warmup 300 --bufs 8

# ============ Small ops prefill ============
for S in 1024 2048 4096 8192; do
  case $S in
    1024) ITS=300 ;;
    2048) ITS=200 ;;
    4096) ITS=100 ;;
    8192) ITS=60 ;;
  esac
  run so_rmsnorm_h2048_prefill_S$S "$BUILD/small_ops_bench" rmsnorm $S 2048   --iters $ITS --warmup 30 --bufs 4
  run so_rope_q_prefill_S$S       "$BUILD/small_ops_bench" rope    $S 16 256 --iters $ITS --warmup 30 --bufs 4
  run so_gate_prefill_S$S         "$BUILD/small_ops_bench" gate    $S 4096   --iters $ITS --warmup 30 --bufs 4
done

echo "=== END $(date)" >> "$LOGS/_index.txt"
echo "Done. Logs in $LOGS"
