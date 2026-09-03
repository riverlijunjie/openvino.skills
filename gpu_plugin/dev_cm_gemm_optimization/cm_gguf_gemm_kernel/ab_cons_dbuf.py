"""
A/B: consumer-side SLM read double-buffer (CONS_DBUF) for the dense full GEMM
CM kernels.  Builds each .cm twice in ONE process (-DCONS_DBUF=0 baseline vs
-DCONS_DBUF=1 new), benchmarks them INTERLEAVED with a cache flush between
every timed launch, and reports min-of-N -- the methodology the kernel headers
require because this machine's clocks drift +-40% between separate process
runs.  Also correctness-checks both builds against the fp16-rounded reference.
"""
import argparse
import numpy as np
import pyopencl as cl

from test_dense_gemm_slim import OPG, cl_src, HW, Gpu, stats, check_close
from test_dense_gemv import (
    build_q4k_weight as build_q4k_flat,
    build_q5k_weight as build_q5k_flat,
    build_q6k_weight as build_q6k_flat,
)

TOKENS_PER_TILE = 8
TOKEN_LOCAL = 8
TOKEN_GROUPS = 4
TOKENS_PER_THREAD = TOKENS_PER_TILE * TOKEN_GROUPS

BUILDERS = {"q4k": build_q4k_flat, "q5k": build_q5k_flat, "q6k": build_q6k_flat}
CMFILE = {"q4k": "gemm_q4k_full.cm", "q5k": "gemm_q5k_full.cm", "q6k": "gemm_q6k_full.cm"}


def prep(gpu, quant, K, N, token_len, seed=42):
    flat, ref_W, wbytes = BUILDERS[quant](K, N, seed)
    rng = np.random.default_rng(seed + 1)
    x_in = rng.standard_normal((token_len, K)).astype(np.float32).astype(np.float16)
    ref = (ref_W.astype(np.float16).astype(np.float64) @
           x_in.astype(np.float64).T).astype(np.float32).T
    mf = cl.mem_flags
    x_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                    hostbuf=np.ascontiguousarray(x_in))
    w_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat)
    out_b = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=token_len * N * 4)
    ngroups = N // OPG
    ntiles = (token_len + TOKENS_PER_THREAD - 1) // TOKENS_PER_THREAD
    gsize = (ngroups, ((ntiles + TOKEN_LOCAL - 1) // TOKEN_LOCAL) * TOKEN_LOCAL)
    lsize = (1, TOKEN_LOCAL)
    return flat, ref, wbytes, x_b, w_b, out_b, gsize, lsize


def make_kernel(gpu, quant, dbuf, x_b, w_b, out_b, token_len, K, N):
    prog = gpu.build(cl_src(CMFILE[quant]), f"-cmc -DROW_GROUPS=1 -DCONS_DBUF={dbuf}")
    krn = cl.Kernel(prog, CMFILE[quant][:-3])
    krn.set_args(x_b, w_b, out_b, np.uint32(token_len), np.uint32(K), np.uint32(N))
    return krn


def check(gpu, krn, out_b, ref, token_len, N, gsize, lsize, tag):
    cl.enqueue_nd_range_kernel(gpu.queue, krn, gsize, lsize).wait()
    got = np.empty(token_len * N, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, out_b); gpu.queue.finish()
    return check_close(tag, ref, got.reshape(token_len, N), verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--flush_mib", type=int, default=128)
    ap.add_argument("--peak_fp16", type=float, default=116000.0)
    args = ap.parse_args()

    gpu = Gpu(args.device)
    gpu.setup_flush(args.flush_mib)

    shapes = [
        ("q4k", 2048, 2048, 1024), ("q4k", 2048, 2048, 4096), ("q4k", 4096, 4096, 1024),
        ("q5k", 2048, 2048, 1024), ("q5k", 2048, 2048, 4096), ("q5k", 4096, 4096, 1024),
        ("q6k", 2048, 2048, 1024), ("q6k", 2048, 2048, 4096), ("q6k", 4096, 4096, 1024),
    ]

    print(f"{'shape':<34}{'base min':>10}{'dbuf min':>10}{'speedup':>9}"
          f"{'base TF':>9}{'dbuf TF':>9}  ok")
    print("-" * 92)
    for quant, K, N, tl in shapes:
        flat, ref, wbytes, x_b, w_b, out_b, gsize, lsize = prep(gpu, quant, K, N, tl)
        k0 = make_kernel(gpu, quant, 0, x_b, w_b, out_b, tl, K, N)
        k1 = make_kernel(gpu, quant, 1, x_b, w_b, out_b, tl, K, N)
        ok0 = check(gpu, k0, out_b, ref, tl, N, gsize, lsize, f"{quant}/base")
        ok1 = check(gpu, k1, out_b, ref, tl, N, gsize, lsize, f"{quant}/dbuf")

        def enq(krn):
            return lambda q: cl.enqueue_nd_range_kernel(q, krn, gsize, lsize)

        # interleave A/B: same flush cadence, min-of-N neutralizes clock drift
        for _ in range(args.warmup):
            gpu.flush_l3(); enq(k0)(gpu.queue); gpu.queue.finish()
            gpu.flush_l3(); enq(k1)(gpu.queue); gpu.queue.finish()
        t0, t1 = [], []
        for _ in range(args.iters):
            gpu.flush_l3()
            e = cl.enqueue_nd_range_kernel(gpu.queue, k0, gsize, lsize); e.wait()
            t0.append((e.profile.end - e.profile.start) * 1e-6)
            gpu.flush_l3()
            e = cl.enqueue_nd_range_kernel(gpu.queue, k1, gsize, lsize); e.wait()
            t1.append((e.profile.end - e.profile.start) * 1e-6)
        m0, m1 = min(t0), min(t1)
        flops = tl * 2 * N * K
        tf0 = flops / (m0 * 1e-3) / 1e12
        tf1 = flops / (m1 * 1e-3) / 1e12
        label = f"{quant} K={K} N={N} tok={tl}"
        print(f"{label:<34}{m0:>9.3f} {m1:>9.3f} {m0/m1:>8.2f}x"
              f"{tf0:>9.1f}{tf1:>9.1f}  {'OK' if ok0 and ok1 else 'FAIL'}")


if __name__ == "__main__":
    main()
