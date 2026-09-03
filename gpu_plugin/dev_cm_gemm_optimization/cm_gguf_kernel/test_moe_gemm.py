"""
test_moe_gemm.py

Correctness + benchmark + roofline harness for the fused Up+Gate+SiLU MoE
GEMM CM (C-for-Metal) kernels:
    moe_gemm_q4k_upgate_slim.cm : moe_gemm_q4k_upgate_slim  (1 < token_len <= 16)
    moe_gemm_q4k_upgate_full.cm : moe_gemm_q4k_upgate_full  (token_len > 16)

Converted from:
  sycl_gguf_kernel/gemm_q4k_L1_slim.xve.cpp   (runUpAndGateSlimQ4KL1_xve)
  sycl_gguf_kernel/gemm_q4k_L1.xmx-xe2.cpp    (upandgate_q4kweights_xmx_xe2 /
                                                runUpAndGateMatQ4KL1_xmx_xe2)

These are the "mat" MoE dense-GEMM path selected by
ffn_moe.mat.cpp's `up_gate_silu_fusion()` once tokens routed to one expert
have been scattered/grouped into a contiguous [token_len, input_len] matrix
(token_len > 1); it fuses the up-projection, gate-projection and SiLU
activation into a single kernel:
    out = up(x) * gate(x) * sigmoid(gate(x))

NOTE: only Q4_K has this fused kernel in sycl_gguf_kernel/ -- for Q5_K/Q6_K
the "mat" MoE path in ffn_moe.mat.cpp falls back to two plain dense GEMM calls
(gemm_q5k_slim/full.cm, gemm_q6k_slim/full.cm -- already ported) plus a
separate elementwise SiLU pass (gpu_silu in ffn_moe.mat.cpp), so no
Q5_K/Q6_K fused up+gate CM kernel is generated here.

Weight layout (identical to gemm_q4k_slim.cm / gemm_q4k_full.cm), one copy
each for the up and gate expert:
  Q4K: pqs || psl

Dispatch:
  slim: global (threads) = (output_len/OPG * nbpr,)               [1D]
        local  (threads) = (nbpr,)
  full: global (threads) = (output_len/OPG * nbpr, num_token_tiles) [2D]
        local  (threads) = (nbpr, 1)
        num_token_tiles = ceil(token_len / 16)

Usage:
    python test_moe_gemm.py [--device 0] [--iters 50] [--warmup 5]
        [--flush_mib 128] [--peak_bw 456] [--peak_fp16 116000] [--verbose]
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

TOKENS_PER_TILE = 8  # must match TOKENS_PER_TILE (dpas RepeatCount) in moe_gemm_q4k_upgate_full.cm


def _sigmoid(x):
    # numerically stable logistic sigmoid (avoids exp() overflow warnings for
    # large negative -x, i.e. large positive x)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


# ===========================================================================
# Test runner
# ===========================================================================
def test_moe_gemm_upgate(gpu, hw, variant,
                         K=2048, N=2048, token_len=8,
                         seed=42, verbose=False,
                         iters=50, warmup=5, do_bench=True):
    """
    Test moe_gemm_q4k_upgate_{slim,full} CM kernel.
    up = Wup * X, gate = Wgate * X  (W: [N, K] quantized, X: [token_len, K] fp16)
    out = up * gate * sigmoid(gate)   (SiLU-gated)
    Output: [token_len, N] fp32
    """
    assert variant in ("slim", "full")
    label = f"moe_gemm_q4k_upgate_{variant}  N={N} K={K} token_len={token_len}"
    print(f"\n  [{label}]")
    assert N % OPG == 0, "N must be a multiple of OPG=16"
    assert K % 256 == 0, "K must be a multiple of 256"
    if variant == "slim":
        assert token_len <= 16, "slim kernel requires token_len <= 16"

    nbpr = K // 256
    use_dpas = (variant == "full")  # moe_gemm_q4k_upgate_full.cm uses real cm_dpas + fp16

    up_flat, up_ref_W, up_wbytes = build_q4k_flat(K, N, seed)
    gate_flat, gate_ref_W, gate_wbytes = build_q4k_flat(K, N, seed + 7)
    cm_file = f"moe_gemm_q4k_upgate_{variant}.cm"
    kernel_name = f"moe_gemm_q4k_upgate_{variant}"

    rng = np.random.default_rng(seed + 1)
    x_f32 = rng.standard_normal((token_len, K)).astype(np.float32)
    x_in = x_f32.astype(np.float16)  # all variants (slim & full) take fp16 inputs

    # Reference: fp16-rounded activations. For full (dpas), also round weights.
    x_ref = x_in.astype(np.float64)
    up_w_ref = (up_ref_W.astype(np.float16).astype(np.float64) if use_dpas
                else up_ref_W.astype(np.float64))
    gate_w_ref = (gate_ref_W.astype(np.float16).astype(np.float64) if use_dpas
                  else gate_ref_W.astype(np.float64))
    up_ref = (up_w_ref @ x_ref.T).T
    gate_ref = (gate_w_ref @ x_ref.T).T
    ref = (up_ref * gate_ref * _sigmoid(gate_ref)).astype(np.float32)

    # Build & run GPU kernel (CM front-end: '-cmc')
    prog = gpu.build(cl_src(cm_file), "-cmc")
    krn = cl.Kernel(prog, kernel_name)

    x_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=np.ascontiguousarray(x_in))
    up_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                     hostbuf=np.ascontiguousarray(up_flat))
    gate_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                       hostbuf=np.ascontiguousarray(gate_flat))
    out_b = cl.Buffer(gpu.ctx, cl.mem_flags.READ_WRITE, size=token_len * N * 4)

    krn.set_args(x_b, up_b, gate_b, out_b,
                 np.uint32(token_len), np.uint32(K), np.uint32(N))

    ngroups = N // OPG
    if variant == "slim":
        # No SLM kernel: 1 thread per row-group, iterates all K-blocks itself
        gsize = (ngroups, token_len)
        lsize = (1, 1)
    else:
        ntiles = (token_len + TOKENS_PER_TILE - 1) // TOKENS_PER_TILE
        gsize = (ngroups * nbpr, ntiles)
        lsize = (nbpr, 1)
    print(f"    gsize={gsize}  lsize={lsize}  ngroups={ngroups}  nbpr={nbpr}")

    def enq(q):
        return cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

    ev = enq(gpu.queue); ev.wait()
    got = np.empty(token_len * N, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, out_b)
    gpu.queue.finish()
    got = got.reshape(token_len, N)

    ok = check_close(kernel_name, ref, got, verbose=verbose)

    timing = rl = moved = None
    if do_bench:
        ts = gpu.time_kernel(enq, iters=iters, warmup=warmup, do_flush=True)
        timing = stats(ts)
        flops = token_len * 2 * 2 * N * K  # up + gate matmuls
        moved = up_wbytes + gate_wbytes + token_len * K * 4 + token_len * N * 4
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
    print("  Fused MoE Up+Gate+SiLU GEMM CM kernels: moe_gemm_q4k_upgate_slim / _full")
    print(SEP)

    results = []
    shapes = [
        # (variant, K, N, token_len)
        ("slim", 2048, 2048,    1),
        ("slim", 2048, 2048,    8),
        ("slim", 2048, 2048,   16),
        ("slim", 4096, 4096,   16),
        ("full", 2048, 2048,   17),
        ("full", 2048, 2048, 1024),
        ("full", 2048, 2048, 4096),
        ("full", 4096, 4096, 1024),
    ]
    for variant, K, N, token_len in shapes:
        ok, timing, rl = test_moe_gemm_upgate(gpu, hw, variant, K=K, N=N,
                                              token_len=token_len, **kw)
        results.append((variant, K, N, token_len, ok))

    print(f"\n{SEP}")
    print("  Summary:")
    for variant, K, N, token_len, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"    [{status}] moe_gemm_q4k_upgate_{variant}  N={N} K={K} token_len={token_len}")
    all_ok = all(r[-1] for r in results)
    print(f"\n  Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
