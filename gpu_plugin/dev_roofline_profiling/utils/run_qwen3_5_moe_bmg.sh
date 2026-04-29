#!/bin/bash
# Qwen3.5-MoE-35B-A3B full roofline sweep on BMG (Arc B580, 2850 MHz, 456 GB/s).
# Differences from Qwen3-Coder sweep:
#   - MoE: H=2048 I=512 NE=256 TK=8 + shared_I=512 (always-on shared expert)
#   - LM_Head: vocab=248320 (1.6× Qwen3-Coder's 151936)
#   - PA: NH=16 NKV=2 HD=256 (head_dim 2× larger than Qwen3-Coder's 128)
#   - Linear-attn: GatedDeltaNet kernel (HK=H=32, K=V=128) — new bench
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
  $CLI -d "$@" > "$LOG_DIR/$tag.log" 2>&1 || { echo "FAIL $tag"; tail -30 "$LOG_DIR/$tag.log"; }
}
run_env() {
  local tag=$1; shift
  local env_str=$1; shift
  echo "== $tag == [$env_str] $@" | tee -a "$LOG_DIR/_index.txt"
  env $env_str $CLI -d "$@" > "$LOG_DIR/$tag.log" 2>&1 || { echo "FAIL $tag"; tail -30 "$LOG_DIR/$tag.log"; }
}

cd "$BUILD"

# ---------- MoE fused primitive (with shared expert) ----------
# moe_bench <B> <S> <H> <I> <NE> <TK> <g> <iters> <warm> <bufs> <flush_mb> <shared_I>
run moe_decode_M1                   ./moe_bench 1 1    2048 512 256 8 128  200  20  4  64 512
run moe_prefill_S1024               ./moe_bench 1 1024 2048 512 256 8 128   30   5  2  64 512
run moe_prefill_S2048               ./moe_bench 1 2048 2048 512 256 8 128   20   3  2  64 512
run moe_prefill_S4096               ./moe_bench 1 4096 2048 512 256 8 128   15   3  2  64 512
run moe_prefill_S8192               ./moe_bench 1 8192 2048 512 256 8 128   10   2  2  64 512

# ---------- fc_bench: full-attn QKV, O, LM_Head ----------
run fc_qkv_decode_M1                ./fc_bench 1    2048   5120 128 8000 500 8 u4 64
run fc_o_decode_M1                  ./fc_bench 1    4096   2048 128 8000 500 8 u4 64
run lm_head_decode_M1               ./fc_bench 1    2048 248320 128  600  60 4 u8 64

for S in 1024 2048 4096 8192; do
  run fc_qkv_prefill_S${S}          ./fc_bench $S   2048   5120 128 50 10 8 u4 64
  run fc_o_prefill_S${S}            ./fc_bench $S   4096   2048 128 50 10 8 u4 64
done

# Linear-attention input projections (Q+K=16*128 ×2 + V=32*128 + gate=32*128 ≈ 4096+4096+4096=12288;
# we measure as N=12288, K=2048, INT4). This is a lower-bound projection roofline; conv1d is small.
run fc_linattn_proj_decode_M1       ./fc_bench 1    2048  12288 128 2000 200 4 u4 64
for S in 1024 2048 4096 8192; do
  run fc_linattn_proj_prefill_S${S} ./fc_bench $S   2048  12288 128 30 5 4 u4 64
done

# ---------- pa_bench: NH=16 NKV=2 HD=256 ----------
for KV in 1024 2048 4096 8192; do
  run_env pa_decode_kv${KV}         "PA_NH=16 PA_NKV=2 PA_HD=256" ./pa_bench decode  1 $KV 20000 500 4 i8
done
for S in 1024 2048 4096 8192; do
  run_env pa_prefill_S${S}          "PA_NH=16 PA_NKV=2 PA_HD=256" ./pa_bench prefill $S 0 50 10 4 i8
done

# ---------- gdn_bench: linear attention (GatedDeltaNet) ----------
# HK=H=32, K=V=128 (Qwen3.5 linear_num_value_heads=32; q must be expanded to 32 heads pre-op)
run gdn_decode_T1                   ./gdn_bench 1 1    32 32 128 5000 200 4
for S in 1024 2048 4096 8192; do
  run gdn_prefill_S${S}             ./gdn_bench 1 $S   32 32 128 30   5   2
done

# ---------- small ops (Qwen3.5 hidden=2048, full-attn Q heads=16/HD=256, K heads=2/HD=256) ----------
run so_rmsnorm_h2048_decode         ./small_ops_bench rmsnorm   1 2048 --iters 50000 --warmup 500 --bufs 8
run so_rmsnorm3d_qnorm_decode       ./small_ops_bench rmsnorm3d 1 16 256 --iters 50000 --warmup 500 --bufs 8
run so_rmsnorm3d_knorm_decode       ./small_ops_bench rmsnorm3d 1  2 256 --iters 50000 --warmup 500 --bufs 8
run so_rope_q_decode                ./small_ops_bench rope      1 16 256 --iters 50000 --warmup 500 --bufs 8
run so_rope_k_decode                ./small_ops_bench rope      1  2 256 --iters 50000 --warmup 500 --bufs 8
run so_add_decode                   ./small_ops_bench add       1 2048 --iters 50000 --warmup 500 --bufs 8

for S in 1024 2048 4096 8192; do
  run so_rmsnorm_h2048_prefill_S${S}  ./small_ops_bench rmsnorm   $S 2048 --iters 500 --warmup 50 --bufs 4
  run so_rope_q_prefill_S${S}         ./small_ops_bench rope      $S 16 256 --iters 500 --warmup 50 --bufs 4
done

echo ===
echo "All logs in $LOG_DIR"
ls "$LOG_DIR"
