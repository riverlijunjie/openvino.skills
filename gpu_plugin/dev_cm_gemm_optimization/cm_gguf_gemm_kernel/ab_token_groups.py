"""
A/B: TOKEN_GROUPS sweep for the dense full GEMM CM kernels, on the CURRENT
COOP_SLM=1 code.  Hypothesis (opt "②"): since COOP_SLM already shares the
dequant across the work-group, the "amortize dequant over more token groups"
argument that motivated TOKEN_GROUPS=4 is weaker, so a smaller TOKEN_GROUPS
(fewer fp32 accumulators -> lower register pressure -> higher occupancy ceiling
than the current 0.8) might hide the SBID load-latency stalls better.

Builds each .cm once per TOKEN_GROUPS value in ONE process (-DTOKEN_GROUPS=n),
matching the host dispatch (ntiles = ceil(token_len/(8*n))), benchmarks them
INTERLEAVED with a cache flush between every timed launch, min-of-N -- the
drift-neutral methodology the kernel headers require.
"""
import argparse
import numpy as np
import pyopencl as cl

from test_dense_gemm_slim import OPG, cl_src, HW, Gpu, check_close
from test_dense_gemv import (
    build_q4k_weight as build_q4k_flat,
    build_q5k_weight as build_q5k_flat,
    build_q6k_weight as build_q6k_flat,
)

TOKENS_PER_TILE = 8
TOKEN_LOCAL = 8   # COOP_SLM correctness requirement; unchanged across variants

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
    return ref, wbytes, x_b, w_b, out_b


def build_variant(gpu, quant, tg, x_b, w_b, out_b, token_len, K, N):
    prog = gpu.build(cl_src(CMFILE[quant]), f"-cmc -DROW_GROUPS=1 -DTOKEN_GROUPS={tg}")
    krn = cl.Kernel(prog, CMFILE[quant][:-3])
    krn.set_args(x_b, w_b, out_b, np.uint32(token_len), np.uint32(K), np.uint32(N))
    tpt = TOKENS_PER_TILE * tg
    ntiles = (token_len + tpt - 1) // tpt
    gsize = (N // OPG, ((ntiles + TOKEN_LOCAL - 1) // TOKEN_LOCAL) * TOKEN_LOCAL)
    lsize = (1, TOKEN_LOCAL)
    return krn, gsize, lsize


def check(gpu, krn, out_b, ref, token_len, N, gsize, lsize, tag):
    cl.enqueue_nd_range_kernel(gpu.queue, krn, gsize, lsize).wait()
    got = np.empty(token_len * N, dtype=np.float32)
    cl.enqueue_copy(gpu.queue, got, out_b); gpu.queue.finish()
    return check_close(tag, ref, got.reshape(token_len, N), verbose=False)


def bench_min(gpu, krn, gsize, lsize, iters):
    ts = []
    for _ in range(iters):
        gpu.flush_l3()
        e = cl.enqueue_nd_range_kernel(gpu.queue, krn, gsize, lsize); e.wait()
        ts.append((e.profile.end - e.profile.start) * 1e-6)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--flush_mib", type=int, default=128)
    ap.add_argument("--tgs", type=str, default="4,3,2",
                    help="comma list of TOKEN_GROUPS to compare (first = baseline)")
    args = ap.parse_args()

    tgs = [int(x) for x in args.tgs.split(",")]
    base = tgs[0]
    gpu = Gpu(args.device)
    gpu.setup_flush(args.flush_mib)

    shapes = [
        ("q4k", 2048, 2048, 1024), ("q4k", 2048, 2048, 4096), ("q4k", 4096, 4096, 1024),
        ("q5k", 2048, 2048, 1024), ("q5k", 2048, 2048, 4096), ("q5k", 4096, 4096, 1024),
        ("q6k", 2048, 2048, 1024), ("q6k", 2048, 2048, 4096), ("q6k", 4096, 4096, 1024),
    ]

    hdr = f"{'shape':<30}"
    for tg in tgs:
        hdr += f"{'TG='+str(tg)+' ms':>12}"
    for tg in tgs[1:]:
        hdr += f"{'vs TG'+str(base):>10}"
    hdr += "  ok"
    print(hdr)
    print("-" * len(hdr))

    for quant, K, N, tl in shapes:
        ref, wbytes, x_b, w_b, out_b = prep(gpu, quant, K, N, tl)
        variants = {}
        allok = True
        for tg in tgs:
            krn, gs, ls = build_variant(gpu, quant, tg, x_b, w_b, out_b, tl, K, N)
            allok &= check(gpu, krn, out_b, ref, tl, N, gs, ls, f"{quant}/TG{tg}")
            variants[tg] = (krn, gs, ls)
        # warmup all
        for _ in range(args.warmup):
            for tg in tgs:
                krn, gs, ls = variants[tg]
                gpu.flush_l3(); cl.enqueue_nd_range_kernel(gpu.queue, krn, gs, ls).wait()
        # interleaved timing
        mins = {tg: [] for tg in tgs}
        for _ in range(args.iters):
            for tg in tgs:
                krn, gs, ls = variants[tg]
                gpu.flush_l3()
                e = cl.enqueue_nd_range_kernel(gpu.queue, krn, gs, ls); e.wait()
                mins[tg].append((e.profile.end - e.profile.start) * 1e-6)
        mn = {tg: min(mins[tg]) for tg in tgs}
        row = f"{quant+' K='+str(K)+' N='+str(N)+' tok='+str(tl):<30}"
        for tg in tgs:
            row += f"{mn[tg]:>12.3f}"
        for tg in tgs[1:]:
            row += f"{mn[base]/mn[tg]:>9.2f}x"
        row += f"  {'OK' if allok else 'FAIL'}"
        print(row)


if __name__ == "__main__":
    main()
