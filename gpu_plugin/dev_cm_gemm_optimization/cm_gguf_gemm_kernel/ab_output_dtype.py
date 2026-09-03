"""
ab_output_dtype.py -- interleaved A/B of the GEMM output element type
(fp16 default vs the original fp32) for all six dense "full" GEMM CM kernels
(gemm_q{4k,5k,6k,40,41,80}_full.cm).

gemm_q*_full.cm now default to a `half* outputs` store (`cm_output_t`,
selected by `-DCM_OUTPUT_F32=1` to restore the original `float* outputs`).
This script measures whether that halves the store bandwidth/message count
enough to matter, using the SAME interleaved-min-of-N methodology as
ab_lowbit_config.py (this board's GT clock drifts +-40% between process
runs, with latency tracking it 1:1, so a sequential before/after comparison
is worthless -- both variants must be built from the SAME source in ONE
process and alternated launch-by-launch).

Usage:
    python ab_output_dtype.py [--quants q4k,q5k,q6k,q40,q41,q80] [--iters 40]
        [--shapes q4k:4096x4096x1024,...] [--config bmg|ptl]
"""
import argparse

import numpy as np
import pyopencl as cl

from test_dense_gemm_slim import Gpu, check_close, cl_src
from test_dense_gemm_full import (
    QUANTS, TUNED_CONFIGS, build_opts, shape_tune, TOKENS_PER_TILE, OPG,
)

# dispatch() does not exist in test_dense_gemm_full (it is inlined in
# test_gemm_full); re-derive it here from the same formulas so this script
# has no other dependency than QUANTS/TUNED_CONFIGS/build_opts/shape_tune.


def dispatch(cfg, N, token_len):
    tpt = TOKENS_PER_TILE * cfg["token_groups"]
    row_blocks = cfg.get("row_blocks", 1)
    token_slots = cfg["token_local"] // row_blocks
    ntiles = (token_len + tpt - 1) // tpt
    gsize = (N // (OPG * cfg["row_groups"] * row_blocks),
             ((ntiles + token_slots - 1) // token_slots) * cfg["token_local"])
    return gsize, (1, cfg["token_local"])


def prep(gpu, quant, K, N, token_len, seed=42):
    builder, _, _ = QUANTS[quant]
    flat, ref_W, _ = builder(K, N, seed)
    rng = np.random.default_rng(seed + 1)
    x_in = rng.standard_normal((token_len, K)).astype(np.float32).astype(np.float16)
    ref = (ref_W.astype(np.float16).astype(np.float64) @
           x_in.astype(np.float64).T).astype(np.float32).T
    mf = cl.mem_flags
    x_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                    hostbuf=np.ascontiguousarray(x_in))
    w_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat)
    # sized for the larger (fp32) variant; the fp16 variant only touches the
    # first half of each row's bytes, which is exactly what gets read back.
    out_b = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=token_len * N * 4)
    return ref, x_b, w_b, out_b


VARIANTS = ["f32", "f16"]  # f32 = baseline (original behaviour), f16 = new default


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--flush_mib", type=int, default=128)
    ap.add_argument("--quants", type=str, default="q4k,q5k,q6k,q40,q41,q80")
    ap.add_argument("--config", type=str, default=None, choices=[None, "bmg", "ptl"])
    ap.add_argument("--shapes", type=str, default=None,
                    help="comma list of quant:KxNxTOKENS; default is the four "
                         "(K,N) transformer shapes at token_len=1024")
    args = ap.parse_args()

    gpu = Gpu(args.device)
    gpu.setup_flush(args.flush_mib)
    label = args.config or ("ptl" if "b390" in gpu.device.name.lower()
                             or "panther" in gpu.device.name.lower() else "bmg")
    base_cfg = TUNED_CONFIGS[label]

    if args.shapes:
        shapes = []
        for s in args.shapes.split(","):
            q, dims = s.split(":")
            K, N, tl = (int(v) for v in dims.split("x"))
            shapes.append((q, K, N, tl))
    else:
        shapes = [(q, K, N, 1024)
                  for q in args.quants.split(",")
                  for (K, N) in ((4096, 4096), (1024, 4096),
                                 (12288, 4096), (4096, 12288))]

    hdr = f"{'shape':<28}{'f32 ms':>10}{'f16 ms':>10}{'speedup':>10}"
    print(hdr)
    print("-" * len(hdr))

    for quant, K, N, tl in shapes:
        cfg = shape_tune(base_cfg, K, N, label)
        ref, x_b, w_b, out_b = prep(gpu, quant, K, N, tl)
        _, cm_file, kernel_name = QUANTS[quant]
        gs, ls = dispatch(cfg, N, tl)

        krns, bad = {}, []
        for v in VARIANTS:
            opts = build_opts(cfg) + (" -DCM_OUTPUT_F32=1" if v == "f32" else "")
            prog = gpu.build(cl_src(cm_file), opts)
            krn = cl.Kernel(prog, kernel_name)
            krn.set_args(x_b, w_b, out_b, np.uint32(tl), np.uint32(K), np.uint32(N))
            cl.enqueue_nd_range_kernel(gpu.queue, krn, gs, ls).wait()
            if v == "f32":
                got = np.empty(tl * N, dtype=np.float32)
            else:
                got = np.empty(tl * N, dtype=np.float16)
            cl.enqueue_copy(gpu.queue, got, out_b)
            gpu.queue.finish()
            ok = check_close(f"{quant}/{v}", ref, got.reshape(tl, N))
            if not ok:
                bad.append(v)
            krns[v] = krn

        for _ in range(args.warmup):
            for v in VARIANTS:
                gpu.flush_l3()
                cl.enqueue_nd_range_kernel(gpu.queue, krns[v], gs, ls).wait()

        ts = {v: [] for v in VARIANTS}
        for _ in range(args.iters):
            for v in VARIANTS:                     # round-robin: same clock state
                gpu.flush_l3()
                e = cl.enqueue_nd_range_kernel(gpu.queue, krns[v], gs, ls)
                e.wait()
                ts[v].append((e.profile.end - e.profile.start) * 1e-6)
        mn = {v: min(ts[v]) for v in VARIANTS}

        row = (f"{quant+' K'+str(K)+' N'+str(N)+' t'+str(tl):<28}"
               f"{mn['f32']:>10.3f}{mn['f16']:>10.3f}{mn['f32']/mn['f16']:>9.2f}x")
        if bad:
            row += "  WRONG:" + ",".join(bad)
        print(row)


if __name__ == "__main__":
    main()
