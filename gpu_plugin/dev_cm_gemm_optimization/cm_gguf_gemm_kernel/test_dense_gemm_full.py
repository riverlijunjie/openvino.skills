"""
test_dense_gemm_full.py

Correctness + benchmark + roofline harness for dense "full" GEMM CM
(C-for-Metal) kernels:
    gemm_q4k_full.cm : gemm_q4k_full
    gemm_q5k_full.cm : gemm_q5k_full
    gemm_q6k_full.cm : gemm_q6k_full
    gemm_q40_full.cm : gemm_q40_full
    gemm_q41_full.cm : gemm_q41_full
    gemm_q80_full.cm : gemm_q80_full

Converted from sycl_gguf_kernel/gemm_q{4,5,6}k_L1.xmx-xe2.cpp -- the SYCL
"full"/prefill dense GEMM path built on the XMX systolic (dpas) matrix engine
with an fp16 shuffle pre-pass and 256-row x 128-token 2D tiling. Per the "1:1
restore data types/layout/execution logic" requirement, all three CM kernels
use the REAL systolic matrix engine via `cm_dpas` (CM_PRECISION_HF,
SystolicDepth=8, RepeatCount=8=TOKENS_PER_TILE) with VNNI-packed
fp16-dequantized weight operands against fp16 activations -- matching the
original's data types and core execution logic, NOT a scalar dot product.
The one simplification vs. the SYCL original (documented in each .cm file) is
that the exact `shuffle_input` 16-token memory-reordering pre-pass (a pure
perf/coalescing optimization) is skipped -- fp16 activations are read in plain
row-major order instead. Weight layout, dequant formulas and the dpas
matrix-engine compute itself are unchanged/faithful.

Input activations: fp16 [token_len * input_len] (converted from fp32 on host;
                    matches the SYCL original's fp16 shuffle_input output dtype)
Output:            fp32 [token_len * output_len]

Weight layout (identical to the slim kernels):
  Q4K: pqs || psl
  Q5K: pqs || pqh || psl
  Q6K: pql || pqh || ps || pd

Dispatch ("v8" kernels): each thread owns a full K range, ROW_GROUPS weight row
groups and a TOKENS_PER_THREAD-token tile, so there is no K-split and no SLM
reduction of the accumulators. The TOKEN_LOCAL threads of a work-group share
those row groups and COOPERATIVELY dequantize them into an SLM ring buffer
(COOP_SLM in the .cm files), so TOKEN_LOCAL must match the kernel's macro
exactly -- it is a correctness requirement, not just a cache-locality hint.
ROW_GROUPS > 1 is the N-blocking that lets one activation load feed several
weight row groups (the activation operand is the measured bottleneck).
  global (threads) = (output_len/(OPG*ROW_GROUPS),
                      ceil(ntiles/TOKEN_LOCAL)*TOKEN_LOCAL)
  local  (threads) = (1, TOKEN_LOCAL)
  OPG = 16, TOKENS_PER_TILE = 8 (dpas RepeatCount),
  TOKENS_PER_THREAD = TOKENS_PER_TILE * TOKEN_GROUPS,
  ntiles = ceil(token_len / TOKENS_PER_THREAD)
This file DRIVES those values: it passes them to the kernel as -D macros and
builds the matching dispatch, choosing the set from the device (see
TUNED_CONFIGS / pick_config -- BMG dGPU and the PTL Xe3 iGPU want different
tiles). Override with --config bmg|ptl.

Usage:
    python test_dense_gemm_full.py [--device 0] [--iters 50] [--warmup 5]
        [--flush_mib 256] [--no-flush] [--peak_bw 110] [--peak_fp16 116000]
        [--verbose] [--no-bench]

NOTE ON THE NUMBERS THIS FILE PRINTS: it is a CORRECTNESS harness first. The
roofline is computed from the MEAN of the timed iterations (the --warmup
launches run before timing starts and are excluded from it).

The absolute figures are still only indicative because THIS BOARD THROTTLES
HARD under sustained load. Measured on B580, four back-to-back runs of the
SAME shape with the SAME binary:
    run0 1.72 ms @ 2900 MHz   run1 3.18 ms @ 1467 MHz
    run2 4.74 ms @ 1283 MHz   run3 5.17 ms @ 1150 MHz
The latency tracks the clock 1:1, so a shape's number depends mostly on where
in the sweep it lands. Each result therefore prints the GT clock it was
measured at, and warns when the GT is below 90% of its peak clock. (This is
also why --no-flush looked slower at first: re-measuring in alternating order
showed flush/no-flush are identical, it was purely the ordering.)

Do NOT use this file to compare kernel versions: use the interleaved A/B
harnesses (ab_rowgroups.py, ab_bottleneck.py, ...), which alternate the
variants launch by launch so every variant sees the same clock state.
"""

import argparse
import glob

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    raise SystemExit("pyopencl is required: pip install pyopencl")

from test_dense_gemm_slim import (
    OPG, cl_src, HW, Gpu, stats, check_close, _print_rl,
)
from test_dense_gemv import build_q4k_weight as build_q4k_flat
from test_dense_gemv import build_q5k_weight as build_q5k_flat
from test_dense_gemv import build_q6k_weight as build_q6k_flat


# ===========================================================================
# Q4_0 / Q4_1 / Q8_0 weight builders  (gemm_q40_full.cm / q41 / q80)
# ===========================================================================
# These three GGUF types have a 32-weight block with ONE scale (plus an
# offset for Q4_1), which is exactly the shape of a Q4_K SUB-BLOCK, so they
# reuse the K-type kernels' whole dataflow unchanged: 8 sub-blocks make one
# 256-wide K-block, one decoded VNNI tile pair covers K=32 x N=16, and the
# SG-transposed packing is the same "[row group][K-block][sub-block][chunk]
# [lane][byte]" order.
#
#   Q4_0   w = (q - 8) * d            q 4-bit    -> scale = d,  minv = 8*d
#   Q4_1   w =  q * d + m             q 4-bit    -> scale = d,  minv = -m
#   Q8_0   w =  q * d                 q int8     -> scale = d
#
# so all three go through the SAME q*scale - minv decoder the Q4_K kernel
# uses (see the .cm headers).  As with the K-type builders in
# test_dense_gemv.py, the QUANTIZED payload is generated directly (random
# nibbles/bytes + random scales) rather than by quantizing a float matrix:
# the kernel under test is the dequant+GEMM path, and the reference is the
# exact same dequant formula the kernel evaluates, in the same order and the
# same precision (fp32 decode, fp16 dpas operands), so a mismatch is a real
# bug and not quantization noise.


def _ascon(a):
    return np.ascontiguousarray(a)


def _pack_nibble_pqs_sg(qs, K, N, opg=OPG):
    """[N, K/32, 16] 4-bit payload -> SG-transposed pqs_T.

    Identical to test_dense_gemv.pack_qk_pqs_sg: a lane's uint32 for chunk c
    holds bytes 4c..4c+3 of the sub-block, i.e. low nibbles are K-pos 4c+b
    and high nibbles K-pos 4c+b+16 -- which is what makes the Q4_K SIMD64
    decoder reusable verbatim.
    """
    nbpr = K // 256
    nrg = N // opg
    src = qs.reshape(nrg, opg, nbpr, 8, 4, 4)
    return _ascon(src.transpose(0, 2, 3, 4, 1, 5)).reshape(-1).astype(np.uint8)


def _pack_int8_pqs_sg(qs, K, N, opg=OPG):
    """[N, K] int8 payload -> SG-transposed pqs_T (same shape as
    moe_gemv_q80_sg.cm's): a lane's uint32 u holds K-pos 4u..4u+3, so u=0..3
    fill the Blo tile and u=4..7 the Bhi tile."""
    nbpr = K // 256
    nrg = N // opg
    src = qs.reshape(nrg, opg, nbpr, 8, 8, 4)
    return _ascon(src.transpose(0, 2, 3, 4, 1, 5)).reshape(-1).astype(np.uint8)


def _pack_subblock_meta_sg(meta, K, N, nbyte, opg=OPG):
    """[N, K/32, nbyte] per-32-block metadata -> [rg][K-block][j][lane][bytes].
    nbyte = 2 for a single fp16 scale (Q4_0 / Q8_0 pd_T), 4 for the
    interleaved (d, m) fp16 pair of Q4_1 (pdm_T)."""
    nbpr = K // 256
    nrg = N // opg
    src = meta.reshape(nrg, opg, nbpr, 8, nbyte)
    return _ascon(src.transpose(0, 2, 3, 1, 4)).reshape(-1).astype(np.uint8)


def build_q40_weight(K, N, seed=1234):
    """Q4_0: 32 weights, one fp16 scale, fixed zero point 8. 18 B/block."""
    assert K % 256 == 0 and N % OPG == 0
    nblk = N * K // 32
    rng = np.random.default_rng(seed)
    d16 = np.float16(rng.uniform(0.001, 0.10, nblk))
    qs = rng.integers(0, 256, (nblk, 16), dtype=np.uint8)

    d32 = d16.astype(np.float32)[:, None]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    W = np.empty((nblk, 32), dtype=np.float32)
    # same expression the kernel evaluates: q*scale - minv, minv = 8*d
    W[:, 0:16] = lo * d32 - 8.0 * d32
    W[:, 16:32] = hi * d32 - 8.0 * d32
    ref_W = W.reshape(N, K)

    pqs_T = _pack_nibble_pqs_sg(qs.reshape(N, K // 32, 16), K, N)
    pd_T = _pack_subblock_meta_sg(
        d16.reshape(N, K // 32).view(np.uint8).reshape(N, K // 32, 2), K, N, 2)
    return np.concatenate([pqs_T, pd_T]), ref_W, nblk * 18


def build_q41_weight(K, N, seed=1234):
    """Q4_1: 32 weights, fp16 scale AND fp16 offset. 20 B/block."""
    assert K % 256 == 0 and N % OPG == 0
    nblk = N * K // 32
    rng = np.random.default_rng(seed)
    d16 = np.float16(rng.uniform(0.001, 0.10, nblk))
    m16 = np.float16(rng.uniform(-0.05, 0.05, nblk))
    qs = rng.integers(0, 256, (nblk, 16), dtype=np.uint8)

    d32 = d16.astype(np.float32)[:, None]
    m32 = m16.astype(np.float32)[:, None]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    W = np.empty((nblk, 32), dtype=np.float32)
    W[:, 0:16] = lo * d32 + m32
    W[:, 16:32] = hi * d32 + m32
    ref_W = W.reshape(N, K)

    # d and m go into two SEPARATE SoA planes, matching the OpenVINO GPU
    # plugin's compile-time repack (RepackGGUFWeightsShuffle:
    # off_pd = N*nbpr*128, off_pm = off_pd + N*nbpr*16). An interleaved
    # (d, m) plane -- one 64 B producer load instead of two 32 B ones -- was
    # measured NEUTRAL, so the family keeps ONE layout.
    pqs_T = _pack_nibble_pqs_sg(qs.reshape(N, K // 32, 16), K, N)
    pd_T = _pack_subblock_meta_sg(
        d16.reshape(N, K // 32).view(np.uint8).reshape(N, K // 32, 2), K, N, 2)
    pm_T = _pack_subblock_meta_sg(
        m16.reshape(N, K // 32).view(np.uint8).reshape(N, K // 32, 2), K, N, 2)
    return np.concatenate([pqs_T, pd_T, pm_T]), ref_W, nblk * 20


def build_q80_weight(K, N, seed=1234):
    """Q8_0: 32 int8 weights, one fp16 scale. 34 B/block."""
    assert K % 256 == 0 and N % OPG == 0
    nblk = N * K // 32
    rng = np.random.default_rng(seed)
    d16 = np.float16(rng.uniform(0.001, 0.10, nblk))
    qs = rng.integers(-127, 128, (nblk, 32), dtype=np.int8)

    ref_W = (qs.astype(np.float32) *
             d16.astype(np.float32)[:, None]).reshape(N, K)

    pqs_T = _pack_int8_pqs_sg(qs.view(np.uint8).reshape(N, K), K, N)
    pd_T = _pack_subblock_meta_sg(
        d16.reshape(N, K // 32).view(np.uint8).reshape(N, K // 32, 2), K, N, 2)
    return np.concatenate([pqs_T, pd_T]), ref_W, nblk * 34


# quant -> (weight builder, .cm file, kernel entry point)
QUANTS = {
    "q4k": (build_q4k_flat, "gemm_q4k_full.cm", "gemm_q4k_full"),
    "q5k": (build_q5k_flat, "gemm_q5k_full.cm", "gemm_q5k_full"),
    "q6k": (build_q6k_flat, "gemm_q6k_full.cm", "gemm_q6k_full"),
    "q40": (build_q40_weight, "gemm_q40_full.cm", "gemm_q40_full"),
    "q41": (build_q41_weight, "gemm_q41_full.cm", "gemm_q41_full"),
    "q80": (build_q80_weight, "gemm_q80_full.cm", "gemm_q80_full"),
}


# ---------------------------------------------------------------------------
# GT clock telemetry -- this board throttles HARD under sustained load
# (measured 2900 -> 1467 -> 1283 -> 1150 MHz over four back-to-back runs of
# the same shape, with the latency tracking it 1:1), so a latency printed
# without the clock it was measured at is uninterpretable.
# ---------------------------------------------------------------------------
_FREQ_GLOB = "/sys/class/drm/card*/device/tile0/gt0/freq0/"


def gt_clock_mhz():
    """(act_freq, max_freq) of the discrete GT in MHz, or (None, None)."""
    for d in sorted(glob.glob(_FREQ_GLOB)):
        try:
            with open(d + "act_freq") as f:
                act = int(f.read().strip())
            with open(d + "max_freq") as f:
                mx = int(f.read().strip())
            if mx > 0:
                return act, mx
        except (OSError, ValueError):
            continue
    return None, None


def time_kernel_with_clock(gpu, enq, iters, warmup, do_flush):
    """Like Gpu.time_kernel, but also samples the GT clock after every timed
    iteration so the reported latency can be interpreted (and normalised)
    against the clock it was actually measured at.
    Returns (times_seconds, mean_act_mhz_or_None, max_mhz_or_None)."""
    for _ in range(warmup):
        if do_flush:
            gpu.flush_l3()
        enq(gpu.queue)
        gpu.queue.finish()
    ts, mhz, mx = [], [], None
    for _ in range(iters):
        if do_flush:
            gpu.flush_l3()
        ev = enq(gpu.queue)
        ev.wait()
        ts.append((ev.profile.end - ev.profile.start) * 1e-9)
        a, m = gt_clock_mhz()
        if a:
            mhz.append(a)
            mx = m
    return ts, (sum(mhz) / len(mhz) if mhz else None), mx


# ---------------------------------------------------------------------------
# Per-architecture tuned launch configuration
# ---------------------------------------------------------------------------
# The kernels take ROW_GROUPS / TOKEN_GROUPS / TOKEN_LOCAL / SLM_JBLK / PF_A as
# -D macros and the host dispatch must match them exactly (TOKEN_LOCAL is a
# CORRECTNESS requirement -- it is the COOP_SLM producer count, and
# SLM_JBLK*ROW_GROUPS has to equal it).
#
# The two parts measured so far want OPPOSITE tiles:
#
#  bmg  Arc B580, Xe2, 160 XVE @ 2.9 GHz, 8 threads/XVE, 128 KB SLM/Xe-core,
#       456 GB/s GDDR6. Compute-rich, so the big 4 KB accumulator tile that
#       minimises operand traffic (256/RG + 512/TG bytes per dpas) wins:
#       ROW_GROUPS=2 TOKEN_GROUPS=4 TOKEN_LOCAL=8 (=> SLM_JBLK=4) PF_A=1.
#       A_OUTER=1 since the Q4_0/Q4_1/Q8_0 work (ab_lowbit_config.py,
#       interleaved min-of-N, ALL SIX quants): 1.00-1.34x, never a loss.
#       A_OUTER was originally adopted on PTL to cut REGISTER PRESSURE, and
#       that is NOT why it wins here -- on BMG it costs registers (120 -> 128
#       for the K-types) and still wins, because this kernel is A-LOAD
#       LATENCY bound (ab_bottleneck.py: noAload 1.46-1.80x). Holding the
#       TOKEN_GROUPS activation tiles issues all 4 independent 2D block loads
#       back-to-back before the first dpas instead of dropping one into the
#       middle of every dpas chain, so the load latency overlaps compute.
#       Biggest at token_len=1024 (1.06-1.34x) where there are fewest waves
#       to hide it with; ~1.00-1.09x at token_len=8192.
#
#  ptl  Arc B390 iGPU, Xe3, 96 XVE @ 2.4 GHz, 10 threads/XVE, 16 MB L2 and
#       SHARED LPDDR. Memory-poor, so the winning tile is the one that gets
#       the most reuse out of each activation load:
#       ROW_GROUPS=4 TOKEN_GROUPS=2 TOKEN_LOCAL=16 ROW_BLOCKS=1 SLM_JBLK=4
#       PF_A=1 A_OUTER=1 (128 registers, no spill on any of the six kernels).
#       MEASURED 1.01-1.63x over the previous ptl entry (RG=2 TG=2 TL=32
#       RB=2 SLM_JBLK=8 PF_A=0) on ALL SIX quants and all four (N,K) shapes,
#       and 1.04-1.58x across token_len 64..8192 and K 1024..16384 --
#       ab_lowbit_config.py, interleaved min-of-N.
#       WHY, and it is NOT what the old ROW_BLOCKS story predicted: RG=4/RB=1
#       has the SAME activation traffic per dpas as RG=2/RB=2
#       (256/(ROW_GROUPS*ROW_BLOCKS) either way). Two other things change.
#       (a) The reuse moves from L1 into REGISTERS -- with ROW_BLOCKS the
#       threads sharing a token slot merely issue the same addresses and hope
#       for an L1 hit, whereas one A tile in a register feeding ROW_GROUPS
#       dpas needs no cache at all. (b) TOKEN_LOCAL halves, so the SLM ring
#       goes 64 -> 32 KB and 4 work-groups fit per Xe-core instead of 2.
#       PF_A also flips to 1 here (1.03-1.15x): at RG=4 each activation load
#       feeds 4x as many dpas, so there are 4x fewer of them and each one's
#       latency is that much more exposed.
#       ALSO MEASURED ON PTL, all worse than the adopted entry (do not redo):
#       RG=4/RB=2/TL=32 0.84-1.09x (64 KB ring), RG=4/TL=8 0.74-1.07x,
#       RG=4/TL=32 0.81-0.92x, RG=4/RB=2/TL=16 0.78-1.10x,
#       RG=4/TOKEN_GROUPS=1 0.62-0.85x, RG=4/TOKEN_GROUPS=4 0.19-0.27x
#       (8 KB of accumulators, spills).
TOKENS_PER_TILE = 8    # dpas RepeatCount -- fixed by the ISA

TUNED_CONFIGS = {
    "bmg": dict(row_groups=2, token_groups=4, token_local=8, row_blocks=1,
                slm_jblk=None, pf_a=None, a_outer=1,
                peak_bw=456.0, peak_fp16=116000.0),
    "ptl": dict(row_groups=4, token_groups=2, token_local=16, row_blocks=1,
                slm_jblk=4, pf_a=1, a_outer=1,
                # 96 XVE x 2.4 GHz x 256 fp16 FLOP/cycle/XVE = 59 TFLOPS; the
                # `dpasonly` ablation probe tops out around 55 TFLOPS on this
                # part, so 59000 is the right order. BW is LPDDR5x, shared
                # with the CPU.
                peak_bw=136.0, peak_fp16=59000.0),
}


def pick_config(device_name, force=None):
    """(label, cfg) for a device; `force` overrides the auto-detection."""
    if force:
        return force, TUNED_CONFIGS[force]
    n = (device_name or "").lower()
    if "b390" in n or "panther" in n:
        return "ptl", TUNED_CONFIGS["ptl"]
    return "bmg", TUNED_CONFIGS["bmg"]


# Long-K shapes (K >> N) are the weak spot of this kernel and the reason is
# the A LAYOUT, not the tiling: A is row-major in K, so the 8 tokens of one
# TOKENS_PER_TILE 2D block load sit K*2 bytes apart.  One A tile load therefore
# scatters over 8*K*2 bytes -- 65 KB at K=4096 but 196 KB at K=12288 -- and the
# TLB / DRAM-page locality of every A fetch degrades linearly with K.  That is
# why runtime is SUPER-linear in K (PTL, q4k, N=4096, token_len=4096):
#     K       2048   4096   6144   8192  10240  12288  16384
#     ms      1.93   3.23   5.26   8.32  11.04  14.95  21.98
#     ms/K   .94e-3 .79e-3 .86e-3 1.02e-3 1.08e-3 1.22e-3 1.34e-3
# and why a large N costs nothing (weights are contiguous along K inside a row
# group): 4096x12288 runs at 33.9 TFLOPS while 12288x4096 runs at 25-30.
# The structural fix is to repack A into a K-blocked layout so the stride stops
# depending on K; that has to happen in the caller.  In-kernel, the only lever
# that pays is a taller token tile.
LONG_K = 10240        # ptl: TOKEN_GROUPS 2 -> 4
BMG_LONG_K = 6144     # bmg: RG=2/TG=4 -> RG=4/TG=2 (see shape_tune)


def shape_tune(cfg, K, N, label="bmg"):
    """Per-SHAPE overrides on top of the per-device config.

    ---- bmg: long K wants ROW_GROUPS=4 / TOKEN_GROUPS=2 --------------------
    The RG=2/TG=4 tile was derived from the operand-traffic model
        bytes/dpas = 256/ROW_GROUPS + 512/TOKEN_GROUPS
    which weights the B (SLM) operand twice as heavily as the activation.
    `ab_lowbit_config.py --ablate` on the A_OUTER=1 dataflow invalidates that
    weighting: noSLMrd is only 1.02-1.08x (the SLM read is ~free, A_OUTER
    hides it behind the batched activation loads) while noAload is
    1.28-1.56x and lands within noise of the dpasonly ceiling. So the model
    collapses to ~256/ROW_GROUPS and the optimum moves to the largest
    ROW_GROUPS the accumulator budget allows. RG=4/TG=2 holds RG*TG=8, i.e.
    the SAME 4 KB of accumulators (and the same 128 registers, no spill --
    A_OUTER is what makes RG=4 compile at all), while halving the activation
    traffic per dpas.
    It pays only where the activation path actually dominates, which is long
    K -- A is row-major in K, so one 8-token 2D block load spans 8*K*2 bytes
    (65 KB at K=4096, 196 KB at K=12288) and its locality degrades with K.
    MEASURED (ab_lowbit_config.py, interleaved min-of-N, q40 N=4096 t=1024,
    RG=4/TG=2 vs RG=2/TG=4):
        K= 2048  0.98x     K= 8192  1.05x
        K= 4096  1.02x     K=10240  1.05x
        K= 6144  1.13x     K=12288  1.12x
        K=16384  1.11x
    and at K=12288 it generalizes across quants (q40 1.12x, q41 1.11x,
    q80 1.11x, q4k 1.10x, q6k 1.06x) and token lengths (t=2048 1.10x,
    t=4096 1.09x). Below the threshold it is a small loss (K=4096/N=12288
    0.96x, K=1024 0.93x), hence the K rule.
    SLM_JBLK must be re-derived: SLM_NPROD = SLM_JBLK*RG*RB has to equal
    TOKEN_LOCAL, so RG=4 with TOKEN_LOCAL=8 needs SLM_JBLK=2.
    REJECTED at the same accumulator budget: RG=8/TG=1 0.62-0.78x (8
    tokens/thread under-amortizes everything else), RG=4/TG=4 with
    TOKEN_LOCAL=16 0.20-0.24x (8 KB of accumulators, spills).

    ---- ptl: long K wants TOKEN_GROUPS=4 ----------------------------------
    TOKEN_GROUPS=4 makes each thread issue 4 vertically adjacent token tiles
    out of one A descriptor region, so the poor long-K A locality is paid once
    per 4 tiles instead of once per 2.  It costs registers (121 vs 102 on ptl,
    still no spill), which is why it is a loss on short K.

    MEASURED on PTL, interleaved min-of-N, TOKEN_GROUPS 4 vs 2:
        K=16384          1.08-1.14x
        K=12288          1.11-1.21x   (q4k / q5k / q6k all agree)
        K=10240          1.20x
        K= 8192          0.95x
        K= 6144          0.95x
        K= 4096          0.79-0.96x   <- short K, smaller tile wins
        K= 2048          0.90x
    Keys on K only: K=4096/N=12288 (1.01x) has the same 28 MB weight matrix as
    K=12288/N=4096 but does not benefit, so it is not an L2-residency effect.
    REJECTED for long K (interleaved, vs the TOKEN_GROUPS=2 baseline):
        ROW_BLOCKS=4 with TOKEN_GROUPS=4   0.89-0.96x
        ROW_BLOCKS=1 with TOKEN_GROUPS=4   0.80-0.96x
        TOKEN_GROUPS=8                     0.15-0.18x (spills 7040 B)
        WALK_HBLK band swizzle             0.26-1.08x
    """
    cfg = dict(cfg)
    if K is None:
        return cfg
    if label == "bmg":
        if (K >= BMG_LONG_K and cfg.get("row_groups") == 2
                and cfg.get("token_groups") == 4
                and cfg.get("row_blocks", 1) == 1
                and N % (OPG * 4) == 0):
            cfg["row_groups"] = 4
            cfg["token_groups"] = 2
            cfg["slm_jblk"] = cfg["token_local"] // 4   # SLM_NPROD == TOKEN_LOCAL
    elif (K >= LONG_K and cfg.get("token_groups") == 2
            and cfg.get("row_groups", 2) <= 2):
        # NOTE: dead for the ADOPTED ptl entry, which is already ROW_GROUPS=4.
        # The row_groups guard is a SAFETY interlock, not a tuning choice:
        # promoting TOKEN_GROUPS 2 -> 4 on top of ROW_GROUPS=4 gives 4*4
        # accumulator tiles = 8 KB, which spills and measures 0.19-0.27x.
        # ROW_GROUPS=4 covers the long-K case far better anyway (1.54-1.63x
        # at K=12288/16384 vs the 1.11-1.21x this promotion used to buy).
        cfg["token_groups"] = 4
    return cfg


def build_opts(cfg):
    o = ["-cmc",
         f"-DROW_GROUPS={cfg['row_groups']}",
         f"-DTOKEN_GROUPS={cfg['token_groups']}",
         f"-DTOKEN_LOCAL={cfg['token_local']}",
         f"-DROW_BLOCKS={cfg.get('row_blocks', 1)}",
         f"-DA_OUTER={cfg.get('a_outer', 0)}"]
    if cfg["slm_jblk"] is not None:
        o.append(f"-DSLM_JBLK={cfg['slm_jblk']}")
    if cfg["pf_a"] is not None:
        o.append(f"-DPF_A={cfg['pf_a']}")
    return " ".join(o)


# ===========================================================================
# Test runner
# ===========================================================================
def test_gemm_full(gpu, hw, quant,
                   K=2048, N=2048, token_len=1024,
                   seed=42, verbose=False,
                   iters=50, warmup=5, do_bench=True, do_flush=True,
                   cfg=None, cfg_label="bmg", shape_tuning=True):
    """
    Test gemm_q{4,5,6}k_full CM kernel.
    y = W * X  where W: [N, K] quantized (SG-transposed), X: [token_len, K] fp16
    Output: [token_len, N] fp32
    All three quants: fp16 inputs, SG-transposed weights, cm_dpas,
    TOKENS_PER_TILE=8, TOKENS_PER_THREAD tokens per thread, full K per thread.
    """
    label = f"gemm_{quant}_full  N={N} K={K} token_len={token_len}"
    print(f"\n  [{label}]")
    cfg = cfg or TUNED_CONFIGS["bmg"]
    if shape_tuning:
        cfg = shape_tune(cfg, K, N, cfg_label)
    row_groups = cfg["row_groups"]
    token_groups = cfg["token_groups"]
    token_local = cfg["token_local"]
    row_blocks = cfg.get("row_blocks", 1)
    token_slots = token_local // row_blocks
    tokens_per_thread = TOKENS_PER_TILE * token_groups
    assert N % (OPG * row_groups * row_blocks) == 0, \
        "N must be a multiple of OPG*ROW_GROUPS*ROW_BLOCKS"
    assert K % 256 == 0, "K must be a multiple of 256"

    nbpr = K // 256

    if quant not in QUANTS:
        raise ValueError(f"Unknown quant: {quant} (have {sorted(QUANTS)})")
    builder, cm_file, kernel_name = QUANTS[quant]
    flat, ref_W, wbytes = builder(K, N, seed)
    rng = np.random.default_rng(seed + 1)
    x_f32 = rng.standard_normal((token_len, K)).astype(np.float32)
    x_in = x_f32.astype(np.float16)

    # Reference: numpy matmul (fp64 for precision), rounding BOTH the
    # activation and the dequantized weight to fp16 first -- the real
    # cm_dpas systolic engine only accepts fp16 operands (same precision the
    # original SYCL XMX kernel computes at), so rounding the reference weight
    # to fp16 too gives a fair, bug-catching comparison instead of conflating
    # expected fp16 quantization noise with real bugs.
    x_ref = x_in.astype(np.float64)
    w_ref = ref_W.astype(np.float16).astype(np.float64)
    ref = (w_ref @ x_ref.T).astype(np.float32).T

    # Build & run GPU kernel (CM front-end: '-cmc' plus the tuned -D macros)
    prog = gpu.build(cl_src(cm_file), build_opts(cfg))
    krn = cl.Kernel(prog, kernel_name)

    x_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=np.ascontiguousarray(x_in))
    w_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=flat)
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=token_len * N * 4)

    krn.set_args(x_b, w_b, out_b,
                 np.uint32(token_len), np.uint32(K), np.uint32(N))

    # Dispatch: 2D, global (threads) = (N/(OPG*ROW_GROUPS),
    #                                   ceil(ntiles/TOKEN_LOCAL)*TOKEN_LOCAL),
    #           local = (1, TOKEN_LOCAL). One thread owns the whole K range,
    #           ROW_GROUPS row groups and TOKENS_PER_THREAD tokens.
    ngroups = N // (OPG * row_groups * row_blocks)
    ntiles = (token_len + tokens_per_thread - 1) // tokens_per_thread
    gsize = (ngroups, ((ntiles + token_slots - 1) // token_slots) * token_local)
    lsize = (1, token_local)
    print(f"    cfg={cfg_label} RG={row_groups} TG={token_groups} "
          f"TL={token_local} RB={row_blocks}  gsize={gsize}  lsize={lsize}  "
          f"ngroups={ngroups}  nbpr={nbpr}  ntiles={ntiles}")

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    ev = enq(gpu.queue); ev.wait()
    got = np.empty(token_len * N, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, out_b)
    gpu.queue.finish()
    got = got.reshape(token_len, N)

    # fp16-rounded reference already accounts for the dpas path's inherent
    # precision, so the same tight tolerance applies to both paths.
    ok = check_close(kernel_name, ref, got, verbose=verbose)

    timing = rl = moved = None
    if do_bench:
        ts, act, mx = time_kernel_with_clock(gpu, enq, iters, warmup, do_flush)
        timing = stats(ts)
        flops = token_len * 2 * N * K
        moved = wbytes + token_len * K * 4 + token_len * N * 4
        # Roofline is reported off the MEAN of the timed iterations. The
        # `warmup` launches run before timing starts and are NOT part of `ts`,
        # so they never enter this average.
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        # Same roofline against the median and the min. On a power-limited
        # part (the PTL iGPU) a long sustained run throttles: the min stays
        # put while the tail grows, so mean << median << min tells you how
        # much of the gap is the kernel and how much is the power budget.
        rl["roofline_pct_median"] = rl["roofline_ms"] / timing["median_ms"] * 100.0
        rl["roofline_pct_min"] = rl["roofline_ms"] / timing["min_ms"] * 100.0
        timing["gt_mhz"] = act
        timing["gt_max_mhz"] = mx
        clk = f"  gt_clock={act:.0f}/{mx} MHz" if act else ""
        print(f"    mean={timing['mean_ms']:.3f} ms  median={timing['median_ms']:.3f} ms  "
              f"min={timing['min_ms']:.3f} ms  max={timing['max_ms']:.3f} ms{clk}")
        _print_rl(rl, moved)
        if act and act < 0.95 * mx:
            # Latency scales ~1:1 with the GT clock here, so scaling the
            # roofline % by max/act estimates what it would have been had the
            # board held its peak clock. That is the only figure in this file
            # that is even roughly comparable across a long sweep.
            rl["roofline_pct_at_peak_clock"] = rl["roofline_pct"] * mx / act
            print(f"    GT throttled to {act*100/mx:.0f}% of peak clock "
                  f"({act:.0f}/{mx} MHz) -> roofline at peak clock would be "
                  f"~{rl['roofline_pct_at_peak_clock']:.1f}%")
    return ok, timing, rl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--flush_mib", type=int, default=128)
    ap.add_argument("--no-flush", action="store_true",
                    help="do not run the L3 cache-flush kernel before each "
                         "timed launch, i.e. measure the cache-warm steady "
                         "state instead of the cold-cache one. (Measured to "
                         "make no difference on this board once the ordering "
                         "effect of GT throttling is controlled for.)")
    ap.add_argument("--peak_bw", type=float, default=None,
                    help="peak DRAM bandwidth in GB/s (default: per-device)")
    ap.add_argument("--peak_fp16", type=float, default=None,
                    help="FP16 peak compute in GFLOPS (default: per-device; "
                         "BMG 116000, PTL 59000)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--config", choices=sorted(TUNED_CONFIGS),
                    help="force a tuned launch config instead of picking one "
                         "from the device name (bmg = Arc B580 / Xe2 dGPU, "
                         "ptl = Arc B390 / Xe3 iGPU)")
    ap.add_argument("--no-shape-tune", action="store_true",
                    help="disable the per-shape overrides (see shape_tune) "
                         "and use the plain per-device config everywhere")
    ap.add_argument("--shapes", type=str, default=None,
                    help="comma list of quant:KxNxTOKENS (e.g. "
                         "q4k:4096x4096x4096,q6k:2048x2048x1024) instead of "
                         "the built-in sweep")
    args = ap.parse_args()

    gpu = Gpu(args.device)
    cfg_label, cfg = pick_config(gpu.device.name, args.config)
    hw = HW(peak_bw_gbps=args.peak_bw or cfg["peak_bw"],
            fp16_gflops=args.peak_fp16 or cfg["peak_fp16"])
    print(f"Tuned config: {cfg_label}  RG={cfg['row_groups']} "
          f"TG={cfg['token_groups']} TL={cfg['token_local']} "
          f"RB={cfg.get('row_blocks', 1)} SLM_JBLK={cfg['slm_jblk']} "
          f"PF_A={cfg['pf_a']} A_OUTER={cfg.get('a_outer', 0)}  "
          f"peak={hw.fp16_gflops/1000:.0f} TFLOPS / {hw.peak_bw_gbps:.0f} GB/s")
    do_flush = not args.no_flush
    if not args.no_bench and do_flush:
        gpu.setup_flush(args.flush_mib)

    kw = dict(verbose=args.verbose, iters=args.iters, warmup=args.warmup,
              do_bench=not args.no_bench, do_flush=do_flush,
              cfg=cfg, cfg_label=cfg_label, shape_tuning=not args.no_shape_tune)

    SEP = "=" * 80
    print(SEP)
    print("  Dense full GEMM CM kernels: gemm_q{4k,5k,6k,40,41,80}_full")
    print(SEP)

    results = []
    shapes = [
        # (quant, K, N, token_len)
        ("q4k", 4096, 1024, 1024),
        ("q4k", 4096, 1024, 2048),
        ("q4k", 4096, 1024, 4096),
        ("q4k", 4096, 4096, 1024),
        ("q4k", 4096, 4096, 2048),
        ("q4k", 4096, 4096, 4096),
        ("q4k", 4096, 4096, 8192),
        ("q6k", 4096, 1024, 1024),
        ("q6k", 4096, 1024, 2048),
        ("q6k", 4096, 1024, 4096),
        ("q6k", 4096, 1024, 8192),
        ("q6k", 12288, 4096, 1024),
        ("q6k", 12288, 4096, 2048),
        ("q6k", 12288, 4096, 4096),
        ("q6k", 12288, 4096, 8192),
        ("q4k", 4096, 12288, 1024),
        ("q4k", 4096, 12288, 2048),
        ("q4k", 4096, 12288, 4096),
        ("q4k", 4096, 12288, 8192),
        ("q5k", 4096, 1024, 1024),
        ("q5k", 4096, 1024, 2048),
        ("q5k", 4096, 1024, 4096),
        ("q5k", 4096, 1024, 8192),
        ("q5k", 12288, 4096, 1024),
        ("q5k", 12288, 4096, 2048),
        ("q5k", 12288, 4096, 4096),
        ("q5k", 12288, 4096, 8192),

        # ---- Q4_0 / Q4_1 / Q8_0 (gemm_q40_full.cm / q41 / q80) ------------
        # The four (N, K) shapes of a typical 4096-hidden transformer block
        # -- square projections, the 1024-wide KV projection, and the two
        # FFN directions -- at the prefill token length. K=12288/N=4096 vs
        # K=4096/N=12288 is deliberately BOTH ways round: they have the same
        # weight-matrix size but the long-K one is the harder case (A is
        # row-major in K, so an 8-token 2D block load spans 8*K*2 bytes; see
        # the shape_tune() note above).
        ("q40", 4096, 4096, 1024),
        ("q40", 1024, 4096, 1024),
        ("q40", 12288, 4096, 1024),
        ("q40", 4096, 12288, 1024),
        ("q41", 4096, 4096, 1024),
        ("q41", 1024, 4096, 1024),
        ("q41", 12288, 4096, 1024),
        ("q41", 4096, 12288, 1024),
        ("q80", 4096, 4096, 1024),
        ("q80", 1024, 4096, 1024),
        ("q80", 12288, 4096, 1024),
        ("q80", 4096, 12288, 1024),
    ]
    if args.shapes:
        shapes = []
        for s in args.shapes.split(","):
            q, dims = s.split(":")
            K, N, tl = (int(v) for v in dims.split("x"))
            shapes.append((q, K, N, tl))
    for quant, K, N, token_len in shapes:
        ok, timing, rl = test_gemm_full(gpu, hw, quant, K=K, N=N,
                                        token_len=token_len, **kw)
        results.append((quant, K, N, token_len, ok, rl, timing))

    print(f"\n{SEP}")
    print("  Summary:")
    for quant, K, N, token_len, ok, rl, timing in results:
        status = "PASS" if ok else "FAIL"
        if rl is None:
            roofline_info = "roofline=n/a  bound=n/a"
        else:
            roofline_ratio = rl["roofline_pct"] / 100.0
            roofline_info = (f"roofline_ratio={roofline_ratio:.3f} "
                             f"({rl['roofline_pct']:.1f}%)  "
                             f"med={rl['roofline_pct_median']:.1f}% "
                             f"min={rl['roofline_pct_min']:.1f}%  "
                             f"bound={rl['bound_by']}")
        clk = ""
        if timing and timing.get("gt_mhz"):
            clk = f"  gt={timing['gt_mhz']:.0f}MHz"
            if rl and "roofline_pct_at_peak_clock" in rl:
                clk += f"  (@peak clk ~{rl['roofline_pct_at_peak_clock']:.1f}%)"
        print(f"    [{status}] gemm_{quant}_full  N={N} K={K} token_len={token_len}  "
              f"{roofline_info}{clk}")
    all_ok = all(r[4] for r in results)
    rls = [r[5] for r in results if r[5] is not None]
    if rls:
        n = len(rls)
        print(f"\n  Average over {n} shapes:  mean-based "
              f"{sum(r['roofline_pct'] for r in rls)/n:.1f}%   median-based "
              f"{sum(r['roofline_pct_median'] for r in rls)/n:.1f}%   min-based "
              f"{sum(r['roofline_pct_min'] for r in rls)/n:.1f}%")
    print(f"\n  Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
