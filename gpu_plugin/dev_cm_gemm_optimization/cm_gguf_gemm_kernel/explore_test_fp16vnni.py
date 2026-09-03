"""
explore_test_fp16vnni.py -- correctness + A/B for the EXPLORATORY no-SLM
pre-decoded fp16 VNNI dpas GEMM (explore_gemm_fp16vnni.cm).

Structural rewrite thesis: hoist the Q4/Q5/Q6 weight decode OUT of the hot path
(do it once at model-load), so the per-call kernel is a pure fp16 dpas GEMM with
NO decode and NO SLM round-trip.  This measures the per-call (pass-2) latency --
the decode pass is one-time and amortized across all forward passes, so it is
NOT counted here (that is the whole point).  We validate numerics and A/B the
pass-2 latency against the current fused COOP_SLM kernel (gemm_q4k_full).

Weight buffer built here on host = what the load-time pre-decode pass would emit.
"""
import argparse
import numpy as np
import pyopencl as cl

from test_dense_gemm_slim import OPG, cl_src, HW, Gpu, stats, check_close, _print_rl
from test_dense_gemv import build_q4k_weight

TOKENS_PER_TILE = 8
TOKEN_GROUPS = 4
TOKEN_LOCAL = 8
TPT = TOKENS_PER_TILE * TOKEN_GROUPS


def pack_fp16_vnni(W_fp16, N, K):
    """Dequantized fp16 weight [N,K] -> contiguous VNNI tile buffer matching
    explore_gemm_fp16vnni.cm:  tile(h,kb,j,lohi) index = (((h*nbpr+kb)*8+j)*2+lohi),
    Blo[d][2n+kbit] = W[h*16+n][kb*256+j*32 + 2d+kbit], hi adds 16 to k_local."""
    Nrg, nbpr = N // OPG, K // 256
    Wr = W_fp16.reshape(Nrg, OPG, nbpr, 8, 32)              # [h,n,kb,j,klocal]
    lo = Wr[..., 0:16].reshape(Nrg, OPG, nbpr, 8, 8, 2)     # [h,n,kb,j,d,kbit]
    hi = Wr[..., 16:32].reshape(Nrg, OPG, nbpr, 8, 8, 2)
    Blo = lo.transpose(0, 2, 3, 4, 1, 5)                    # [h,kb,j,d,n,kbit]
    Bhi = hi.transpose(0, 2, 3, 4, 1, 5)
    buf = np.stack([Blo, Bhi], axis=3)                      # [h,kb,j,lohi,d,n,kbit]
    return np.ascontiguousarray(buf.reshape(-1).astype(np.float16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--flush_mib", type=int, default=128)
    ap.add_argument("--peak_fp16", type=float, default=116000.0)
    args = ap.parse_args()
    hw = HW(peak_bw_gbps=456.0, fp16_gflops=args.peak_fp16)
    gpu = Gpu(0); gpu.setup_flush(args.flush_mib)

    shapes = [
        (2048, 2048, 17), (2048, 2048, 1024), (2048, 2048, 4096),
        (4096, 4096, 1024), (4096, 4096, 4096),
    ]
    print(f"{'shape':<26}{'fused ms':>10}{'fp16vnni ms':>13}{'speedup':>9}"
          f"{'fused TF':>10}{'vnni TF':>9}  ok")
    print("-" * 90)

    for K, N, tl in shapes:
        flat, ref_W, wbytes = build_q4k_weight(K, N, 42)
        W_fp16 = ref_W.astype(np.float16)
        wvnni = pack_fp16_vnni(W_fp16, N, K)
        rng = np.random.default_rng(43)
        x_in = rng.standard_normal((tl, K)).astype(np.float32).astype(np.float16)
        ref = (W_fp16.astype(np.float64) @ x_in.astype(np.float64).T).astype(np.float32).T

        mf = cl.mem_flags
        x_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(x_in))
        wq_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat)
        wv_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=wvnni)
        out_b = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=tl * N * 4)
        nt = (tl + TPT - 1) // TPT
        gs = (N // OPG, ((nt + TOKEN_LOCAL - 1) // TOKEN_LOCAL) * TOKEN_LOCAL)
        ls = (1, TOKEN_LOCAL)
        # the fused kernel now defaults to ROW_GROUPS=2 -> half the dim0 groups
        ROW_GROUPS = 2
        gsf = (N // (OPG * ROW_GROUPS), gs[1])

        # exploratory pass-2 kernel
        kv = cl.Kernel(gpu.build(cl_src("explore_gemm_fp16vnni.cm"), "-cmc"), "gemm_fp16vnni")
        kv.set_args(x_b, wv_b, out_b, np.uint32(tl), np.uint32(K), np.uint32(N))
        cl.enqueue_nd_range_kernel(gpu.queue, kv, gs, ls).wait()
        got = np.empty(tl * N, dtype=np.float32); cl.enqueue_copy(gpu.queue, got, out_b); gpu.queue.finish()
        okv = check_close(f"fp16vnni K{K} N{N} t{tl}", ref, got.reshape(tl, N))

        # fused baseline (current default: ROW_GROUPS=2)
        kf = cl.Kernel(gpu.build(cl_src("gemm_q4k_full.cm"), "-cmc"), "gemm_q4k_full")
        kf.set_args(x_b, wq_b, out_b, np.uint32(tl), np.uint32(K), np.uint32(N))

        def bench(k):
            for _ in range(args.warmup):
                gpu.flush_l3(); cl.enqueue_nd_range_kernel(gpu.queue, k, gs, ls).wait()
            ts = []
            for _ in range(args.iters):
                gpu.flush_l3()
                e = cl.enqueue_nd_range_kernel(gpu.queue, k, gs, ls); e.wait()
                ts.append((e.profile.end - e.profile.start) * 1e-6)
            return ts
        # interleave
        tf, tv = [], []
        for _ in range(args.warmup):
            gpu.flush_l3(); cl.enqueue_nd_range_kernel(gpu.queue, kf, gsf, ls).wait()
            gpu.flush_l3(); cl.enqueue_nd_range_kernel(gpu.queue, kv, gs, ls).wait()
        for _ in range(args.iters):
            gpu.flush_l3(); e = cl.enqueue_nd_range_kernel(gpu.queue, kf, gsf, ls); e.wait()
            tf.append((e.profile.end - e.profile.start) * 1e-6)
            gpu.flush_l3(); e = cl.enqueue_nd_range_kernel(gpu.queue, kv, gs, ls); e.wait()
            tv.append((e.profile.end - e.profile.start) * 1e-6)
        mf_, mv_ = min(tf), min(tv)
        fl = tl * 2 * N * K
        print(f"q4k K{K} N{N} t{tl:<5}{mf_:>10.3f}{mv_:>13.3f}{mf_/mv_:>8.2f}x"
              f"{fl/(mf_*1e-3)/1e12:>10.1f}{fl/(mv_*1e-3)/1e12:>9.1f}  {'OK' if okv else 'FAIL'}")


if __name__ == "__main__":
    main()
