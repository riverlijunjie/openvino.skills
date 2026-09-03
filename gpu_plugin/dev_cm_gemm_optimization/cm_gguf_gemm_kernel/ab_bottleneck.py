"""
ab_bottleneck.py -- component ablation of gemm_q{4,5,6}k_full.cm (v7 COOP_SLM).

Answers "what is the current bottleneck?" by building, IN ONE PROCESS, several
variants of the SAME source with one component surgically removed, and timing
them interleaved with min-of-N (this machine's clocks drift +-40% between
processes, so cross-process A/B is worthless -- see the kernel headers).

Variants (source is patched textually in memory; only `base` is numerically
correct, the others are timing probes):
  base       unmodified kernel
  noAtraf    A operand still loaded, but always from block (0,0): same
             instruction count / same LSC messages, ~zero cache misses
             -> isolates the A-operand MEMORY TRAFFIC cost
  noAload    A operand replaced by a constant (load removed entirely)
             -> isolates the A-operand TOTAL cost (issue + latency + traffic)
  noSLMrd    consumer's cm_slm_block_read removed (producer still decodes and
             still writes SLM) -> isolates the SLM READ cost
  nodecode   producer's dequant ALU replaced by a raw copy of the payload
             (payload load kept) -> isolates the DEQUANT ALU cost
  dpasonly   noAload + noSLMrd + nodecode -> the dpas issue ceiling of this
             exact loop structure (still has the barrier + payload load)

Usage: python ab_bottleneck.py [--iters 20] [--shapes ...]
"""
import argparse
import re

import numpy as np
import pyopencl as cl

from test_dense_gemm_slim import OPG, cl_src, Gpu
from test_dense_gemm_full import pick_config, build_opts, TOKENS_PER_TILE
from test_dense_gemv import (
    build_q4k_weight as build_q4k_flat,
    build_q5k_weight as build_q5k_flat,
    build_q6k_weight as build_q6k_flat,
)

TOKEN_GROUPS = 4
TOKEN_LOCAL = 8
ROW_GROUPS = 2          # overwritten in main() from the device's tuned config
TPT = TOKENS_PER_TILE * TOKEN_GROUPS
BUILD_OPTS = "-cmc"

BUILDERS = {"q4k": build_q4k_flat, "q5k": build_q5k_flat, "q6k": build_q6k_flat}
CMFILE = {q: f"gemm_{q}_full.cm" for q in BUILDERS}

# ---- textual patches (the consumer loop text is identical in all 3 kernels) --
A_LOAD = """                vector<half, 2 * TOKENS_PER_TILE * 16> A2;
                b2dA.set_block_y((int)(tile_base + tg * TOKENS_PER_TILE));
                cm_load<lsc::Normal>(A2, b2dA.set_block_x((int)(kb * 256u) + j * 32));"""
A_LOAD_CONST = """                vector<half, 2 * TOKENS_PER_TILE * 16> A2 = (half)1.0f;"""

SLM_RD = """            vector<half, 8*32> Blo[ROW_GROUPS], Bhi[ROW_GROUPS];
            #pragma unroll
            for (int rg = 0; rg < ROW_GROUPS; rg++) {
                cm_slm_block_read(slm, GENX_NONE,
                    (int)(sbase + (uint)((rg * SLM_JBLK + jl) * 2 + 0) * SLM_TILE_B), Blo[rg]);
                cm_slm_block_read(slm, GENX_NONE,
                    (int)(sbase + (uint)((rg * SLM_JBLK + jl) * 2 + 1) * SLM_TILE_B), Bhi[rg]);
            }"""
SLM_RD_CONST = """            vector<half, 8*32> Blo[ROW_GROUPS], Bhi[ROW_GROUPS];
            #pragma unroll
            for (int rg = 0; rg < ROW_GROUPS; rg++) { Blo[rg] = (half)1.0f; Bhi[rg] = (half)1.0f; }"""

DECODE_RE = re.compile(r"^[ \t]*q[456]k_vnni_decode\([^;\n]*\);", re.MULTILINE)
DECODE_RAW = ("Blo.format<uint>().select<4*OPG,1>(0) = q; "
              "Bhi.format<uint>().select<4*OPG,1>(0) = q;")


def patch(src, variant):
    n = 0
    if variant == "halfAtraf":
        src2 = src.replace("b2dA.set_block_y((int)(tile_base + tg * TOKENS_PER_TILE));",
                           "b2dA.set_block_y((int)(tile_base + (tg & ~1) * TOKENS_PER_TILE));")
        assert src2 != src, "halfAtraf patch failed"
        src = src2
    if variant in ("noAtraf",):
        src2 = src.replace("b2dA.set_block_y((int)(tile_base + tg * TOKENS_PER_TILE));",
                           "b2dA.set_block_y(0);")
        src2 = src2.replace("b2dA.set_block_x((int)(kb * 256u) + j * 32)",
                            "b2dA.set_block_x(0)")
        assert src2 != src, "noAtraf patch failed"
        src = src2
    if variant in ("noAload", "dpasonly"):
        assert A_LOAD in src, "noAload patch failed"
        src = src.replace(A_LOAD, A_LOAD_CONST)
    if variant in ("noSLMrd", "dpasonly"):
        assert SLM_RD in src, "noSLMrd patch failed"
        src = src.replace(SLM_RD, SLM_RD_CONST)
    if variant in ("nodecode", "dpasonly"):
        src, n = DECODE_RE.subn(DECODE_RAW, src)
        assert n >= 1, "nodecode patch failed"
    return src


VARIANTS = ["base", "halfAtraf", "noAtraf", "noAload", "noSLMrd",
            "nodecode", "dpasonly"]
EXACT = ("base",)          # variants that must still produce correct results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--flush_mib", type=int, default=128)
    args = ap.parse_args()

    gpu = Gpu(args.device)
    gpu.setup_flush(args.flush_mib)

    # profile whatever tuned config this device actually runs
    global TOKEN_GROUPS, TOKEN_LOCAL, ROW_GROUPS, TPT, BUILD_OPTS
    label, cfg = pick_config(gpu.device.name)
    ROW_GROUPS = cfg["row_groups"]
    TOKEN_GROUPS = cfg["token_groups"]
    TOKEN_LOCAL = cfg["token_local"]
    TPT = TOKENS_PER_TILE * TOKEN_GROUPS
    BUILD_OPTS = build_opts(cfg)
    print(f"Tuned config: {label}  RG={ROW_GROUPS} TG={TOKEN_GROUPS} "
          f"TL={TOKEN_LOCAL}  opts='{BUILD_OPTS}'")

    shapes = [
        ("q4k", 2048, 2048, 4096), ("q4k", 4096, 4096, 1024),
        ("q5k", 2048, 2048, 4096), ("q5k", 4096, 4096, 1024),
        ("q6k", 2048, 2048, 4096), ("q6k", 4096, 4096, 1024),
    ]

    hdr = f"{'shape':<26}" + "".join(f"{v:>11}" for v in VARIANTS) + f"{'TFLOPS':>9}"
    print(hdr)
    print("-" * len(hdr))

    for quant, K, N, tl in shapes:
        flat, ref_W, _ = BUILDERS[quant](K, N, 42)
        rng = np.random.default_rng(43)
        x_in = rng.standard_normal((tl, K)).astype(np.float32).astype(np.float16)
        mf = cl.mem_flags
        x_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=np.ascontiguousarray(x_in))
        w_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat)
        out_b = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=tl * N * 4)

        with open(cl_src(CMFILE[quant])) as f:
            src0 = f.read()
        nt = (tl + TPT - 1) // TPT
        gs = (N // (OPG * ROW_GROUPS),
              ((nt + TOKEN_LOCAL - 1) // TOKEN_LOCAL) * TOKEN_LOCAL)
        ls = (1, TOKEN_LOCAL)

        krns = {}
        for v in VARIANTS:
            prog = gpu._build_src(patch(src0, v), (quant, v), BUILD_OPTS)
            k = cl.Kernel(prog, f"gemm_{quant}_full")
            k.set_args(x_b, w_b, out_b, np.uint32(tl), np.uint32(K), np.uint32(N))
            krns[v] = k

        ref = (ref_W.astype(np.float16).astype(np.float64) @
               x_in.astype(np.float64).T).astype(np.float32).T
        bad = []
        for v in EXACT:
            cl.enqueue_nd_range_kernel(gpu.queue, krns[v], gs, ls).wait()
            got = np.empty(tl * N, dtype=np.float32)
            cl.enqueue_copy(gpu.queue, got, out_b)
            gpu.queue.finish()
            got = got.reshape(tl, N)
            rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-9)
            if rel > 1e-2:
                bad.append(f"{v}:{rel:.2e}")

        for _ in range(args.warmup):
            for v in VARIANTS:
                gpu.flush_l3()
                cl.enqueue_nd_range_kernel(gpu.queue, krns[v], gs, ls).wait()

        acc = {v: [] for v in VARIANTS}
        for _ in range(args.iters):
            for v in VARIANTS:
                gpu.flush_l3()
                e = cl.enqueue_nd_range_kernel(gpu.queue, krns[v], gs, ls)
                e.wait()
                acc[v].append((e.profile.end - e.profile.start) * 1e-6)
        mn = {v: min(acc[v]) for v in VARIANTS}
        tf = tl * 2 * N * K / (mn["base"] * 1e-3) / 1e12
        row = f"{quant+' K'+str(K)+' N'+str(N)+' t'+str(tl):<26}"
        row += f"{mn['base']:>11.3f}"
        for v in VARIANTS[1:]:
            row += f"{mn['base']/mn[v]:>10.2f}x"
        row += f"{tf:>9.1f}"
        if bad:
            row += "  WRONG:" + ",".join(bad)
        print(row)


if __name__ == "__main__":
    main()
