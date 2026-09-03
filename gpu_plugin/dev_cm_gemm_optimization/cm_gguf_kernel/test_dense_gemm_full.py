"""
test_dense_gemm_full.py

Correctness + benchmark + roofline harness for dense "full" GEMM CM
(C-for-Metal) kernels:
    gemm_q4k_full.cm : gemm_q4k_full
    gemm_q5k_full.cm : gemm_q5k_full
    gemm_q6k_full.cm : gemm_q6k_full

Converted from sycl_gguf_kernel/gemm_q{4,5,6}k_L1.xmx-xe2.cpp -- the SYCL
"full"/prefill dense GEMM path built on the XMX systolic (dpas) matrix engine
with an fp16 shuffle pre-pass and 256-row x 128-token 2D tiling. Per the "1:1
restore data types/layout/execution logic" requirement, all three CM kernels
now use the REAL systolic matrix engine via `cm_dpas` (CM_PRECISION_HF,
SystolicDepth=8, RepeatCount=8=TOKENS_PER_TILE) with VNNI-packed
fp16-dequantized weight operands against fp16 activations -- matching the
original's data types and core execution logic, NOT a scalar dot product.
Simplifications vs. the SYCL original (documented in each .cm file): the
exact `shuffle_input` 16-token memory-reordering pre-pass (a pure
perf/coalescing optimization) is skipped -- fp16 activations are read in
plain row-major order instead -- and the 256-row x 128-token 2D block-load /
tik-tok SLM double-buffering schedule is replaced by a simpler per-K-block
SLM reduction. Weight layout, dequant formulas and the dpas matrix-engine
compute itself are unchanged/faithful.

Input activations: fp16 [token_len * input_len] (converted from fp32 on host;
                    matches the SYCL original's fp16 shuffle_input output dtype)
Output:            fp32 [token_len * output_len]

Weight layout (identical to the slim kernels):
  Q4K: pqs || psl
  Q5K: pqs || pqh || psl
  Q6K: pql || pqh || ps || pd

Dispatch:
  global (threads) = (output_len/OPG * nbpr, num_token_tiles)   [2D]
  local  (threads) = (nbpr, 1)
  OPG = 16, nbpr = input_len // 256, num_token_tiles = ceil(token_len / 8)
  (TOKENS_PER_TILE=8 = dpas RepeatCount). gemm_q4k_full.cm additionally uses
  a parallel stride-halving SLM reduction tree (instead of a serial
  hh==0-only sum) to cut roofline%-limiting reduction overhead.

Usage:
    python test_dense_gemm_full.py [--device 0] [--iters 50] [--warmup 5]
        [--flush_mib 256] [--peak_bw 456] [--peak_fp16 116000] [--verbose]
        [--no-bench]
"""

import argparse

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

TOKENS_PER_TILE = 8  # must match TOKENS_PER_TILE (dpas RepeatCount) in gemm_q{4,5,6}k_full.cm



# ===========================================================================
# Test runner
# ===========================================================================
def test_gemm_full(gpu, hw, quant,
                   K=2048, N=2048, token_len=1024,
                   seed=42, verbose=False,
                   iters=50, warmup=5, do_bench=True):
    """
    Test gemm_q{4,5,6}k_full CM kernel.
    y = W * X  where W: [N, K] quantized (SG-transposed), X: [token_len, K] fp16
    Output: [token_len, N] fp32
    All three quants: fp16 inputs, SG-transposed weights, cm_dpas, TOKENS_PER_TILE=8.
    """
    label = f"gemm_{quant}_full  N={N} K={K} token_len={token_len}"
    print(f"\n  [{label}]")
    assert N % OPG == 0, "N must be a multiple of OPG=16"
    assert K % 256 == 0, "K must be a multiple of 256"

    nbpr = K // 256
    tokens_per_tile = TOKENS_PER_TILE

    if quant == "q4k":
        flat, ref_W, wbytes = build_q4k_flat(K, N, seed)
        cm_file = "gemm_q4k_full.cm"
        kernel_name = "gemm_q4k_full"
    elif quant == "q5k":
        flat, ref_W, wbytes = build_q5k_flat(K, N, seed)
        cm_file = "gemm_q5k_full.cm"
        kernel_name = "gemm_q5k_full"
    elif quant == "q6k":
        flat, ref_W, wbytes = build_q6k_flat(K, N, seed)
        cm_file = "gemm_q6k_full.cm"
        kernel_name = "gemm_q6k_full"
    else:
        raise ValueError(f"Unknown quant: {quant}")

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

    # Build & run GPU kernel (CM front-end: '-cmc')
    prog = gpu.build(cl_src(cm_file), "-cmc")
    krn = cl.Kernel(prog, kernel_name)

    x_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=np.ascontiguousarray(x_in))
    w_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=flat)
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.WRITE_ONLY, size=token_len * N * 4)

    krn.set_args(x_b, w_b, out_b,
                 np.uint32(token_len), np.uint32(K), np.uint32(N))

    # Dispatch: 2D, global (threads) = (N/OPG * nbpr, num_token_tiles),
    #           local = (nbpr, 1)
    ngroups = N // OPG
    ntiles = (token_len + tokens_per_tile - 1) // tokens_per_tile
    gsize = (ngroups * nbpr, ntiles)
    lsize = (nbpr, 1)
    print(f"    gsize={gsize}  lsize={lsize}  ngroups={ngroups}  nbpr={nbpr}  ntiles={ntiles}")

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
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
        timing = stats(ts)
        flops = token_len * 2 * N * K
        moved = wbytes + token_len * K * 4 + token_len * N * 4
        rl = hw.roofline(flops, moved, timing["mean_ms"] * 1e-3)
        print(f"    mean={timing['mean_ms']:.3f} ms  median={timing['median_ms']:.3f} ms  "
              f"min={timing['min_ms']:.3f} ms  max={timing['max_ms']:.3f} ms")
        _print_rl(rl, moved)
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
    ap.add_argument("--peak_bw", type=float, default=456.0)
    ap.add_argument("--peak_fp16", type=float, default=116000.0,
                    help="FP16 peak compute throughput in GFLOPS (BMG: 116000)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    hw = HW(peak_bw_gbps=args.peak_bw, fp16_gflops=args.peak_fp16)
    gpu = Gpu(args.device)
    if not args.no_bench:
        gpu.setup_flush(args.flush_mib)

    kw = dict(verbose=args.verbose, iters=args.iters, warmup=args.warmup,
              do_bench=not args.no_bench)

    SEP = "=" * 80
    print(SEP)
    print("  Dense full GEMM CM kernels: gemm_q4k_full / gemm_q5k_full / gemm_q6k_full")
    print(SEP)

    results = []
    shapes = [
        # (quant, K, N, token_len)
        ("q4k", 2048, 2048,   17),   # non-multiple-of-16 tail tile
        ("q4k", 2048, 2048, 1024),
        ("q4k", 2048, 2048, 4096),
        ("q4k", 4096, 4096, 1024),
        ("q4k", 4096, 4096, 4096),
        ("q5k", 2048, 2048, 1024),
        ("q5k", 2048, 2048, 4096),
        ("q5k", 4096, 4096, 1024),
        ("q6k", 2048, 2048, 1024),
        ("q6k", 2048, 2048, 4096),
        ("q6k", 4096, 4096, 1024),
    ]
    for quant, K, N, token_len in shapes:
        ok, timing, rl = test_gemm_full(gpu, hw, quant, K=K, N=N,
                                        token_len=token_len, **kw)
        results.append((quant, K, N, token_len, ok))

    print(f"\n{SEP}")
    print("  Summary:")
    for quant, K, N, token_len, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] gemm_{quant}_full  N={N} K={K} token_len={token_len}")
    all_ok = all(r[-1] for r in results)
    print(f"\n  Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
