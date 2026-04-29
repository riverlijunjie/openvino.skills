#!/bin/bash
# Qwen3-MoE full roofline sweep on BMG (Arc B580, 2850 MHz, 456 GB/s, 116.74 FP16 TFLOPS).
# Produces one .log per (op, shape) under $LOG_DIR so parse_logs_v2.py can pick them up.
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
  $CLI -d "$@" > "$LOG_DIR/$tag.log" 2>&1 || { echo "FAIL $tag"; tail -30 "$LOG_DIR/$tag.log"; }
}

cd "$BUILD"

# ---------- MoE fused primitive ----------
# Args: <B> <S> <H> <I> <NE> <TK> <g> <iters> <warm> <bufs> <flush_mb>
# Iter budgets sized for ~1s GPU time per config.
run moe_decode_M1                   ./moe_bench 1 1    2048 768 128 8 128  200  20  4  64
run moe_prefill_S1024               ./moe_bench 1 1024 2048 768 128 8 128   30   5  2  64
run moe_prefill_S2048               ./moe_bench 1 2048 2048 768 128 8 128   20   3  2  64
run moe_prefill_S4096               ./moe_bench 1 4096 2048 768 128 8 128   15   3  2  64
run moe_prefill_S8192               ./moe_bench 1 8192 2048 768 128 8 128   10   2  2  64
run moe_prefill_S16384              ./moe_bench 1 16384 2048 768 128 8 128   6   2  1  64
run moe_prefill_S32768              ./moe_bench 1 32768 2048 768 128 8 128   4   1  1  64
run moe_prefill_S65536              ./moe_bench 1 65536 2048 768 128 8 128   2   1  1  64
run moe_prefill_S131072             ./moe_bench 1 131072 2048 768 128 8 128  2   1  1  64

# ---------- fc_bench: QKV, O, LM_Head ----------
# <M> <K> <N> [group_size] [iters] [warmup] [num_bufs] [precision] [flush_mb]
run fc_qkv_decode_M1                ./fc_bench 1    2048   5120 128 8000 500 8 u4 64
run fc_o_decode_M1                  ./fc_bench 1    4096   2048 128 8000 500 8 u4 64
run lm_head_decode_M1               ./fc_bench 1    2048 151936 128 1000 100 8 u8 64

for S in 1024 2048 4096 8192 16384 32768 65536 131072; do
  run fc_qkv_prefill_S${S}          ./fc_bench $S   2048   5120 128 30 5 4 u4 64
  run fc_o_prefill_S${S}            ./fc_bench $S   4096   2048 128 30 5 4 u4 64
done

# ---------- pa_bench: paged attention ----------
for KV in 1024 2048 4096 8192 16384 32768 65536 131072; do
  run pa_decode_kv${KV}             ./pa_bench decode  1 $KV 10000 300 4 i8
done
for S in 1024 2048 4096 8192 16384 32768 65536 131072; do
  run pa_prefill_S${S}              ./pa_bench prefill $S 0 30 5 2 i8
done

# ---------- small ops (Qwen3 specific shapes) ----------
# decode shapes
run so_rmsnorm_h2048_decode         ./small_ops_bench rmsnorm   1 2048 --iters 50000 --warmup 500 --bufs 8
run so_rmsnorm3d_qnorm_decode       ./small_ops_bench rmsnorm3d 1 32 128 --iters 50000 --warmup 500 --bufs 8
run so_rmsnorm3d_knorm_decode       ./small_ops_bench rmsnorm3d 1  4 128 --iters 50000 --warmup 500 --bufs 8
run so_rope_q_decode                ./small_ops_bench rope      1 32 128 --iters 50000 --warmup 500 --bufs 8
run so_rope_k_decode                ./small_ops_bench rope      1  4 128 --iters 50000 --warmup 500 --bufs 8
run so_add_decode                   ./small_ops_bench add       1 2048 --iters 50000 --warmup 500 --bufs 8

# prefill shapes (S=2048 representative)
for S in 1024 2048 4096 8192; do
  run so_rmsnorm_h2048_prefill_S${S}  ./small_ops_bench rmsnorm   $S 2048 --iters 500 --warmup 50 --bufs 4
  run so_rope_q_prefill_S${S}         ./small_ops_bench rope      $S 32 128 --iters 500 --warmup 50 --bufs 4
done

echo ===
echo "All logs in $LOG_DIR"
ls "$LOG_DIR"
