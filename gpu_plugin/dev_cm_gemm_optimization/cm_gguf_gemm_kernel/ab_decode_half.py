"""
A/B: fp32 vs fp16 weight-decode (DECODE_HALF) for the dense full GEMM kernels.
Builds each .cm twice (-DDECODE_HALF=0 fp32 decode / =1 fp16 decode) in ONE
process, interleaved min-of-N with a flush between every launch (drift-neutral).
Reports perf only -- correctness of the fp16 path is a separate (expected fp16
rounding) question handled in the test harness.  Also prints the build's spill
bytes if the driver emits them.
"""
import argparse
import numpy as np
import pyopencl as cl

from test_dense_gemm_slim import OPG, cl_src, Gpu
from test_dense_gemv import (
    build_q4k_weight as b4, build_q5k_weight as b5, build_q6k_weight as b6,
)

TOKENS_PER_TILE = 8
TOKEN_LOCAL = 8
TOKEN_GROUPS = 4
TPT = TOKENS_PER_TILE * TOKEN_GROUPS
BUILD = {"q4k": b4, "q5k": b5, "q6k": b6}
CMF = {"q4k": "gemm_q4k_full.cm", "q5k": "gemm_q5k_full.cm", "q6k": "gemm_q6k_full.cm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--flush_mib", type=int, default=128)
    args = ap.parse_args()
    gpu = Gpu(0); gpu.setup_flush(args.flush_mib)

    shapes = [
        ("q4k", 2048, 2048, 1024), ("q4k", 2048, 2048, 4096), ("q4k", 4096, 4096, 1024),
        ("q5k", 2048, 2048, 1024), ("q5k", 2048, 2048, 4096), ("q5k", 4096, 4096, 1024),
        ("q6k", 2048, 2048, 1024), ("q6k", 2048, 2048, 4096), ("q6k", 4096, 4096, 1024),
    ]
    print(f"{'shape':<30}{'fp32 ms':>10}{'fp16 ms':>10}{'speedup':>9}{'fp32 TF':>9}{'fp16 TF':>9}")
    print("-" * 77)
    for quant, K, N, tl in shapes:
        flat, ref_W, wbytes = BUILD[quant](K, N, 42)
        rng = np.random.default_rng(43)
        x_in = rng.standard_normal((tl, K)).astype(np.float32).astype(np.float16)
        mf = cl.mem_flags
        x_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(x_in))
        w_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat)
        out_b = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=tl * N * 4)
        gs = (N // OPG, ((( (tl + TPT - 1)//TPT) + TOKEN_LOCAL - 1)//TOKEN_LOCAL)*TOKEN_LOCAL)
        ls = (1, TOKEN_LOCAL)

        krns = {}
        for dh in (0, 1):
            prog = gpu.build(cl_src(CMF[quant]), f"-cmc -DROW_GROUPS=1 -DDECODE_HALF={dh}")
            k = cl.Kernel(prog, CMF[quant][:-3])
            k.set_args(x_b, w_b, out_b, np.uint32(tl), np.uint32(K), np.uint32(N))
            krns[dh] = k

        for _ in range(args.warmup):
            for dh in (0, 1):
                gpu.flush_l3(); cl.enqueue_nd_range_kernel(gpu.queue, krns[dh], gs, ls).wait()
        t = {0: [], 1: []}
        for _ in range(args.iters):
            for dh in (0, 1):
                gpu.flush_l3()
                e = cl.enqueue_nd_range_kernel(gpu.queue, krns[dh], gs, ls); e.wait()
                t[dh].append((e.profile.end - e.profile.start) * 1e-6)
        m0, m1 = min(t[0]), min(t[1])
        flops = tl * 2 * N * K
        print(f"{quant+' K='+str(K)+' N='+str(N)+' tok='+str(tl):<30}"
              f"{m0:>10.3f}{m1:>10.3f}{m0/m1:>8.2f}x"
              f"{flops/(m0*1e-3)/1e12:>9.1f}{flops/(m1*1e-3)/1e12:>9.1f}")


if __name__ == "__main__":
    main()
