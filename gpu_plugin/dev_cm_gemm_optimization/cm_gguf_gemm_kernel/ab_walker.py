"""
A/B: thread-walker band-remap (WALK_HBLK) for the dense full GEMM kernels
(user request #3).  WALK_HBLK re-linearizes the (row-group, token-tile) grid
into WALK_HBLK-wide bands of row groups to keep a weight band L2-resident while
token tiles sweep past.  Header recorded 2/4/8/16/32 as net loss; re-verify in
the current state.  Interleaved min-of-N, flush between launches.
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
TOKEN_GROUPS = 4
TPT = TOKENS_PER_TILE * TOKEN_GROUPS
BUILD = {"q4k": b4, "q5k": b5, "q6k": b6}
CMF = {"q4k": "gemm_q4k_full.cm", "q5k": "gemm_q5k_full.cm", "q6k": "gemm_q6k_full.cm"}

CONFIGS = [
    ("hb0",  ""),
    ("hb2",  "-DWALK_HBLK=2"),
    ("hb4",  "-DWALK_HBLK=4"),
    ("hb8",  "-DWALK_HBLK=8"),
    ("hb16", "-DWALK_HBLK=16"),
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
        ("q5k", 2048, 2048, 4096), ("q6k", 2048, 2048, 4096), ("q6k", 4096, 4096, 1024),
    ]
    labels = [c[0] for c in CONFIGS]
    hdr = f"{'shape':<26}" + "".join(f"{l+' ms':>10}" for l in labels)
    hdr += "".join(f"{'x'+labels[i]:>8}" for i in range(1, len(labels))) + "  ok"
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
        gs = (N // OPG, (((tl + TPT - 1)//TPT + TOKEN_LOCAL - 1)//TOKEN_LOCAL)*TOKEN_LOCAL)
        ls = (1, TOKEN_LOCAL)

        krns = []; allok = True
        for label, opts in CONFIGS:
            prog = gpu.build(cl_src(CMF[quant]), ("-cmc -DROW_GROUPS=1 " + opts).strip())
            k = cl.Kernel(prog, CMF[quant][:-3])
            k.set_args(x_b, w_b, out_b, np.uint32(tl), np.uint32(K), np.uint32(N))
            cl.enqueue_nd_range_kernel(gpu.queue, k, gs, ls).wait()
            got = np.empty(tl * N, dtype=np.float32)
            cl.enqueue_copy(gpu.queue, got, out_b); gpu.queue.finish()
            allok &= check_close(f"{quant}/{label}", ref, got.reshape(tl, N))
            krns.append(k)

        for _ in range(args.warmup):
            for k in krns:
                gpu.flush_l3(); cl.enqueue_nd_range_kernel(gpu.queue, k, gs, ls).wait()
        mins = [[] for _ in krns]
        for _ in range(args.iters):
            for i, k in enumerate(krns):
                gpu.flush_l3()
                e = cl.enqueue_nd_range_kernel(gpu.queue, k, gs, ls); e.wait()
                mins[i].append((e.profile.end - e.profile.start) * 1e-6)
        mn = [min(m) for m in mins]
        row = f"{quant+' K'+str(K)+' N'+str(N)+' t'+str(tl):<26}" + "".join(f"{v:>10.3f}" for v in mn)
        row += "".join(f"{mn[0]/mn[i]:>7.2f}x" for i in range(1, len(mn)))
        row += f"  {'OK' if allok else 'FAIL'}"
        print(row)


if __name__ == "__main__":
    main()
