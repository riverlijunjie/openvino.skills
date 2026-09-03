"""
A/B: doubleGRF + larger per-thread C tile (bigger TOKEN_GROUPS) for the dense
full GEMM kernels.  At default GRF=128 the kernels do NOT spill at TG=4 but the
header records TG>4 spilling (0.44-0.95x); -Qxcm_doubleGRF (256 GRF) lifts that
ceiling, so a bigger M-tile may now win by giving more independent dpas chains
to hide the SBID/SLM-read stalls.  Interleaved min-of-N, flush between launches
(drift-neutral).  Correctness checked against the fp16-rounded reference.

Configs: (label, token_groups, extra_opts).  First = baseline.
"""
import argparse
import numpy as np
import pyopencl as cl

from test_dense_gemm_slim import OPG, cl_src, Gpu, check_close
from test_dense_gemv import (
    build_q4k_weight as b4, build_q5k_weight as b5, build_q6k_weight as b6,
)

TOKENS_PER_TILE = 8
TOKEN_LOCAL = 8
BUILD = {"q4k": b4, "q5k": b5, "q6k": b6}
CMF = {"q4k": "gemm_q4k_full.cm", "q5k": "gemm_q5k_full.cm", "q6k": "gemm_q6k_full.cm"}

CONFIGS = [
    ("base tg4 128grf", 4, ""),
    ("dgrf tg4",        4, "-Qxcm_doubleGRF"),
    ("dgrf tg6",        6, "-Qxcm_doubleGRF"),
    ("dgrf tg8",        8, "-Qxcm_doubleGRF"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--flush_mib", type=int, default=128)
    args = ap.parse_args()
    gpu = Gpu(0); gpu.setup_flush(args.flush_mib)

    shapes = [
        ("q4k", 2048, 2048, 1024), ("q4k", 2048, 2048, 4096), ("q4k", 4096, 4096, 1024),
        ("q5k", 2048, 2048, 1024), ("q5k", 2048, 2048, 4096), ("q5k", 4096, 4096, 1024),
        ("q6k", 2048, 2048, 1024), ("q6k", 2048, 2048, 4096), ("q6k", 4096, 4096, 1024),
    ]
    labels = [c[0] for c in CONFIGS]
    hdr = f"{'shape':<26}" + "".join(f"{l+' ms':>18}" for l in labels)
    hdr += "".join(f"{'x'+labels[i].split()[1]:>9}" for i in range(1, len(labels))) + "  ok"
    print(hdr); print("-" * len(hdr))

    for quant, K, N, tl in shapes:
        flat, ref_W, wbytes = BUILD[quant](K, N, 42)
        rng = np.random.default_rng(43)
        x_in = rng.standard_normal((tl, K)).astype(np.float32).astype(np.float16)
        ref = (ref_W.astype(np.float16).astype(np.float64) @
               x_in.astype(np.float64).T).astype(np.float32).T
        mf = cl.mem_flags
        x_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(x_in))
        w_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat)
        out_b = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=tl * N * 4)

        krns = []; dispatch = []; allok = True
        for label, tg, opts in CONFIGS:
            prog = gpu.build(cl_src(CMF[quant]), f"-cmc -DROW_GROUPS=1 -DTOKEN_GROUPS={tg} {opts}".strip())
            k = cl.Kernel(prog, CMF[quant][:-3])
            k.set_args(x_b, w_b, out_b, np.uint32(tl), np.uint32(K), np.uint32(N))
            tpt = TOKENS_PER_TILE * tg
            ntiles = (tl + tpt - 1) // tpt
            gs = (N // OPG, ((ntiles + TOKEN_LOCAL - 1)//TOKEN_LOCAL)*TOKEN_LOCAL)
            ls = (1, TOKEN_LOCAL)
            cl.enqueue_nd_range_kernel(gpu.queue, k, gs, ls).wait()
            got = np.empty(tl * N, dtype=np.float32)
            cl.enqueue_copy(gpu.queue, got, out_b); gpu.queue.finish()
            allok &= check_close(f"{quant}/{label}", ref, got.reshape(tl, N))
            krns.append(k); dispatch.append((gs, ls))

        for _ in range(args.warmup):
            for k, (gs, ls) in zip(krns, dispatch):
                gpu.flush_l3(); cl.enqueue_nd_range_kernel(gpu.queue, k, gs, ls).wait()
        mins = [[] for _ in krns]
        for _ in range(args.iters):
            for i, (k, (gs, ls)) in enumerate(zip(krns, dispatch)):
                gpu.flush_l3()
                e = cl.enqueue_nd_range_kernel(gpu.queue, k, gs, ls); e.wait()
                mins[i].append((e.profile.end - e.profile.start) * 1e-6)
        mn = [min(m) for m in mins]
        row = f"{quant+' K'+str(K)+' N'+str(N)+' t'+str(tl):<26}" + "".join(f"{v:>18.3f}" for v in mn)
        row += "".join(f"{mn[0]/mn[i]:>8.2f}x" for i in range(1, len(mn)))
        row += f"  {'OK' if allok else 'FAIL'}"
        print(row)


if __name__ == "__main__":
    main()
