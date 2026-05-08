#!/bin/bash
# Qwen3-Next-80B-A3B-Instruct roofline sweep on BMG (Arc B580, 2850 MHz, 456 GB/s).
# Only the parts that DIFFER from qwen3_5_moe-bmg are re-measured here:
#   - MoE: NE=512, TK=10 (vs qwen3_5_moe NE=256, TK=8). Same H=2048, I=512, SI=512.
#   - LM_Head: vocab=151936 (vs qwen3_5_moe vocab=248320).
#   - small_ops at S>=16K (qwen3_5_moe-bmg only ran S<=8K).
# All other ops (fc_qkv, fc_o, fc_linattn, PA NH=16/NKV=2/HD=256, GDN HK=32/K=V=128,
# small_ops S<=8K + decode) have identical shapes and are reused from qwen3_5_moe BMG logs.
set -e
ROOT=/mnt/river/model_loading/roofline_test_utils
BUILD=$ROOT/build
OV_BIN=/mnt/river/model_loading/openvino/build-x86_64-release/bin/intel64/Release
CLI=/mnt/river/model_loading/clintercept-3.0.6-Linux/bin/cliloader
LOG_DIR=$ROOT/logs/bmg/qwen3_next
mkdir -p "$LOG_DIR"
export LD_LIBRARY_PATH=$OV_BIN:$LD_LIBRARY_PATH

run() {
  local tag=$1; shift
  echo "== $tag == $@" | tee -a "$LOG_DIR/_index.txt"
  $CLI -d "$@" > "$LOG_DIR/$tag.log" 2>&1 || { echo "FAIL $tag"; tail -10 "$LOG_DIR/$tag.log"; }
}

cd "$BUILD"
echo "=== START $(date) ===" > "$LOG_DIR/_index.txt"

# ---------- MoE fused (with shared expert): H=2048 I=512 NE=512 TK=10 ----------
# moe_bench <B> <S> <H> <I> <NE> <TK> <g> <iters> <warm> <bufs> <flush_mb> <shared_I>
run moe_decode_M1                ./moe_bench 1 1     2048 512 512 10 128 200 20 4 64 512
run moe_prefill_S1024            ./moe_bench 1 1024  2048 512 512 10 128  30  5 2 64 512
run moe_prefill_S2048            ./moe_bench 1 2048  2048 512 512 10 128  20  3 2 64 512
run moe_prefill_S4096            ./moe_bench 1 4096  2048 512 512 10 128  15  3 2 64 512
run moe_prefill_S8192            ./moe_bench 1 8192  2048 512 512 10 128  10  2 2 64 512
run moe_prefill_S16384           ./moe_bench 1 16384 2048 512 512 10 128   5  1 1 64 512
run moe_prefill_S32768           ./moe_bench 1 32768 2048 512 512 10 128   3  1 1 64 512
run moe_prefill_S65536           ./moe_bench 1 65536 2048 512 512 10 128   2  1 1 64 512
run moe_prefill_S131072          ./moe_bench 1 131072 2048 512 512 10 128  1  1 1 64 512

# ---------- LM head (INT8 g=128, vocab=151936) ----------
run lm_head_decode_M1            ./fc_bench 1 2048 151936 128 600 60 4 u8 64

# ---------- small_ops at S>=16K (rest reused from qwen3_5_moe-bmg) ----------
for S in 16384 32768 65536 131072; do
  run so_rmsnorm_h2048_prefill_S${S}  ./small_ops_bench rmsnorm $S 2048   --iters 200 --warmup 20 --bufs 4
  run so_rope_q_prefill_S${S}         ./small_ops_bench rope    $S 16 256 --iters 200 --warmup 20 --bufs 4
done

echo "=== END $(date) ===" >> "$LOG_DIR/_index.txt"
echo "Done. Logs in $LOG_DIR"
