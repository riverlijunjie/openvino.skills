"""
ab_lowbit_config.py -- interleaved A/B of launch configurations for the
Q4_0 / Q4_1 / Q8_0 dense full GEMM CM kernels (gemm_q40_full.cm,
gemm_q41_full.cm, gemm_q80_full.cm).

The three new quant types inherit gemm_q4k_full.cm's dataflow verbatim, but
their operand balance is NOT the same: Q4_0/Q4_1 have a cheaper scale
front-end than Q4_K, and Q8_0 carries a 2x larger quantized payload. The
adopted BMG tile (ROW_GROUPS=2 / TOKEN_GROUPS=4 / TOKEN_LOCAL=8) was tuned
for the K-types, so it has to be re-measured here rather than assumed.

Methodology is the one the kernel headers require, because this board's GT
clock drifts +-40% between process runs (2900 -> 1150 MHz under sustained
load, with latency tracking it 1:1):
  - every variant is built from the SAME source in ONE process, selected
    only by -D macros;
  - all variants share the same input buffers and the same reference;
  - correctness is checked once per variant BEFORE timing (a config with a
    wrong dim-0 divisor or a TOKEN_LOCAL that does not match SLM_NPROD
    produces wrong output that would otherwise look like a fast result);
  - every variant is warmed up, then the variants are alternated
    round-robin for every timed round with a constant flush policy;
  - the reported number is min-of-N, which is the drift-neutral signal.

Usage:
    python ab_lowbit_config.py [--quants q40,q41,q80] [--iters 40]
        [--variants base,aouter,tl16,...] [--shapes q80:4096x4096x1024,...]
"""
import argparse
import re

import numpy as np
import pyopencl as cl

from test_dense_gemm_slim import OPG, cl_src, Gpu, check_close
from test_dense_gemm_full import QUANTS, TOKENS_PER_TILE

# Candidate configs. Keys mirror the .cm macros exactly; the host dispatch is
# derived from them (see dispatch()) so a variant can never silently disagree
# with the kernel it built.
#   SLM_NPROD = SLM_JBLK * ROW_GROUPS * ROW_BLOCKS must equal TOKEN_LOCAL.
VARIANTS = {
    #                      RG TG  TL RB JBLK PFA AOUT
    "base":     dict(row_groups=2, token_groups=4, token_local=8,  row_blocks=1,
                     slm_jblk=4, pf_a=1, a_outer=0),   # adopted BMG K-type tile
    "aouter":   dict(row_groups=2, token_groups=4, token_local=8,  row_blocks=1,
                     slm_jblk=4, pf_a=1, a_outer=1),
    "nopfa":    dict(row_groups=2, token_groups=4, token_local=8,  row_blocks=1,
                     slm_jblk=4, pf_a=0, a_outer=0),
    "rg1":      dict(row_groups=1, token_groups=4, token_local=8,  row_blocks=1,
                     slm_jblk=8, pf_a=1, a_outer=0),
    "tg2":      dict(row_groups=2, token_groups=2, token_local=8,  row_blocks=1,
                     slm_jblk=4, pf_a=1, a_outer=0),
    "tl16":     dict(row_groups=2, token_groups=4, token_local=16, row_blocks=1,
                     slm_jblk=8, pf_a=1, a_outer=0),
    "tl16rb2":  dict(row_groups=2, token_groups=4, token_local=16, row_blocks=2,
                     slm_jblk=4, pf_a=1, a_outer=1),
    "rg4":      dict(row_groups=4, token_groups=4, token_local=8,  row_blocks=1,
                     slm_jblk=2, pf_a=1, a_outer=1),
    # ---- round 2: built on top of the A_OUTER=1 winner -------------------
    "ao_nopfa": dict(row_groups=2, token_groups=4, token_local=8,  row_blocks=1,
                     slm_jblk=4, pf_a=0, a_outer=1),
    "ao_tg2":   dict(row_groups=2, token_groups=2, token_local=8,  row_blocks=1,
                     slm_jblk=4, pf_a=1, a_outer=1),
    "ao_tg8":   dict(row_groups=2, token_groups=8, token_local=8,  row_blocks=1,
                     slm_jblk=4, pf_a=1, a_outer=1),
    "ao_rg1":   dict(row_groups=1, token_groups=4, token_local=8,  row_blocks=1,
                     slm_jblk=8, pf_a=1, a_outer=1),
    "ao_rg3":   dict(row_groups=3, token_groups=4, token_local=12, row_blocks=1,
                     slm_jblk=4, pf_a=1, a_outer=1),
    "ao_tl16":  dict(row_groups=2, token_groups=4, token_local=16, row_blocks=1,
                     slm_jblk=8, pf_a=1, a_outer=1),
    "ao_tl32":  dict(row_groups=2, token_groups=4, token_local=32, row_blocks=2,
                     slm_jblk=8, pf_a=1, a_outer=1),
    "ao_rg4tl16": dict(row_groups=4, token_groups=4, token_local=16, row_blocks=1,
                       slm_jblk=4, pf_a=1, a_outer=1),
    # ---- round 3: attack the A-load path itself (A_OUTER=1 baseline) -----
    # A_BIG    = one wide block_2d_desc<half,2,TOKENS_PER_THREAD,16> load for
    #            all token groups instead of TOKEN_GROUPS 8-row loads
    # A_DESC_TG= one descriptor per token group (block_y baked in), so only
    #            set_block_x stays on the critical path
    "ao_abig":   dict(row_groups=2, token_groups=4, token_local=8, row_blocks=1,
                      slm_jblk=4, pf_a=1, a_outer=1, extra="-DA_BIG=1"),
    "ao_desctg": dict(row_groups=2, token_groups=4, token_local=8, row_blocks=1,
                      slm_jblk=4, pf_a=1, a_outer=1, extra="-DA_DESC_TG=1"),
    "ao_pfa2":   dict(row_groups=2, token_groups=4, token_local=8, row_blocks=1,
                      slm_jblk=4, pf_a=2, a_outer=1),
    "ao_dech":   dict(row_groups=2, token_groups=4, token_local=8, row_blocks=1,
                      slm_jblk=4, pf_a=1, a_outer=1, extra="-DDECODE_HALF=1"),
    "abig_nopfa": dict(row_groups=2, token_groups=4, token_local=8, row_blocks=1,
                       slm_jblk=4, pf_a=0, a_outer=1, extra="-DA_BIG=1"),
    "abig_tl16": dict(row_groups=2, token_groups=4, token_local=16, row_blocks=1,
                      slm_jblk=8, pf_a=1, a_outer=1, extra="-DA_BIG=1"),
    "abig_tg2":  dict(row_groups=2, token_groups=2, token_local=8, row_blocks=1,
                      slm_jblk=4, pf_a=1, a_outer=1, extra="-DA_BIG=1"),
    "abig_rg4":  dict(row_groups=4, token_groups=2, token_local=8, row_blocks=1,
                      slm_jblk=2, pf_a=1, a_outer=1, extra="-DA_BIG=1"),
    # ---- round 4: cut A CACHE TRAFFIC via ROW_BLOCKS ---------------------
    # The --ablate run says noAload ~= dpasonly (the A path is the ONLY cost
    # left) and noAtraf carries 1.14-1.39x of it, i.e. most of it is L2
    # traffic rather than message count. Traffic per dpas is
    # 256/(ROW_BLOCKS*ROW_GROUPS) bytes; ROW_GROUPS is capped by the
    # accumulator register wall, so ROW_BLOCKS is the only lever left.
    # ROW_BLOCKS threads sharing a token slot issue the SAME A addresses, so
    # only the first misses L1. Cost: SLM_JBLK = TOKEN_LOCAL/(RG*RB) shrinks,
    # i.e. more barriers per K-block, and the SLM ring grows with TOKEN_LOCAL.
    # SLM per WG = 2 * SLM_JBLK*RG*RB * 2 * 512 = 2*TOKEN_LOCAL*1024 B, and a
    # BMG Xe-core has 128 KB SLM / 64 thread slots -- so TOKEN_LOCAL<=16 keeps
    # full occupancy, TOKEN_LOCAL=32 halves it.
    "rb2_tl8":   dict(row_groups=2, token_groups=4, token_local=8, row_blocks=2,
                      slm_jblk=2, pf_a=1, a_outer=1),   # 16 KB, 4 barriers/Kblk
    "rb2_tl16":  dict(row_groups=2, token_groups=4, token_local=16, row_blocks=2,
                      slm_jblk=4, pf_a=1, a_outer=1),   # 32 KB, 2 barriers/Kblk
    "rb4_tl16":  dict(row_groups=2, token_groups=4, token_local=16, row_blocks=4,
                      slm_jblk=2, pf_a=1, a_outer=1),   # 32 KB, 4 barriers/Kblk
    "rb2_tl8_abig": dict(row_groups=2, token_groups=4, token_local=8, row_blocks=2,
                         slm_jblk=2, pf_a=1, a_outer=1, extra="-DA_BIG=1"),
    "rb2_tl16_nopfa": dict(row_groups=2, token_groups=4, token_local=16, row_blocks=2,
                           slm_jblk=4, pf_a=0, a_outer=1),
    # ---- round 5: re-balance RG vs TG now that the SLM read is FREE ------
    # The RG=2/TG=4 optimum was derived from bytes/dpas = 256/RG + 512/TG,
    # which weights the B (SLM) operand twice as heavily as A. The --ablate
    # run invalidates that weighting for the A_OUTER=1 dataflow: noSLMrd is
    # only 1.02-1.08x, i.e. the B term is ~free, so the cost model collapses
    # to ~256/ROW_GROUPS and the optimum should move to the largest
    # ROW_GROUPS that keeps the accumulator (RG*TG*512 B) within budget.
    # RG=4/TG=2 and RG=8/TG=1 hold RG*TG=8, i.e. the SAME 4 KB accumulator as
    # the adopted RG=2/TG=4, while cutting A traffic per dpas 2x and 4x.
    "rg4_tg2":   dict(row_groups=4, token_groups=2, token_local=8, row_blocks=1,
                      slm_jblk=2, pf_a=1, a_outer=1),
    "rg8_tg1":   dict(row_groups=8, token_groups=1, token_local=8, row_blocks=1,
                      slm_jblk=1, pf_a=1, a_outer=1),
    "rg4_tg2_nopfa": dict(row_groups=4, token_groups=2, token_local=8, row_blocks=1,
                          slm_jblk=2, pf_a=0, a_outer=1),
    "rg4_tg4_tl16": dict(row_groups=4, token_groups=4, token_local=16, row_blocks=1,
                         slm_jblk=4, pf_a=1, a_outer=1),
    # ---- PTL / Xe3 iGPU (Arc B390) --------------------------------------
    # `ptl` is the adopted TUNED_CONFIGS["ptl"] entry; the rest probe around
    # it. PTL has 10 threads/EU (80 thread slots per Xe-core) and 128 KB SLM,
    # so SLM/WG = 2*TOKEN_LOCAL*1024 B caps the resident work-groups:
    # TL=32 -> 64 KB -> 2 WG = 64 threads, TL=16 -> 32 KB -> 4 WG = 64.
    "ptl":       dict(row_groups=2, token_groups=2, token_local=32, row_blocks=2,
                      slm_jblk=8, pf_a=0, a_outer=1),
    "ptl_tg4":   dict(row_groups=2, token_groups=4, token_local=32, row_blocks=2,
                      slm_jblk=8, pf_a=0, a_outer=1),
    "ptl_pfa1":  dict(row_groups=2, token_groups=2, token_local=32, row_blocks=2,
                      slm_jblk=8, pf_a=1, a_outer=1),
    "ptl_rg4":   dict(row_groups=4, token_groups=2, token_local=32, row_blocks=2,
                      slm_jblk=4, pf_a=0, a_outer=1),
    "ptl_rg4rb1": dict(row_groups=4, token_groups=2, token_local=16, row_blocks=1,
                       slm_jblk=4, pf_a=0, a_outer=1),
    "ptl_rb1":   dict(row_groups=2, token_groups=2, token_local=16, row_blocks=1,
                      slm_jblk=8, pf_a=0, a_outer=1),
    "ptl_tl16":  dict(row_groups=2, token_groups=2, token_local=16, row_blocks=2,
                      slm_jblk=4, pf_a=0, a_outer=1),
    "ptl_rg4tg4": dict(row_groups=4, token_groups=4, token_local=32, row_blocks=2,
                       slm_jblk=4, pf_a=0, a_outer=1),
    # ---- PTL round 2: on top of the RG=4/RB=1/TL=16 winner ---------------
    # RG=4/RB=1 has the SAME A traffic per dpas as the old RG=2/RB=2
    # (256/(RG*RB) either way), so the 1.10-1.39x it measures is NOT a
    # traffic effect: the reuse moves from L1 (ROW_BLOCKS threads issuing the
    # same addresses) into REGISTERS (one A tile feeding ROW_GROUPS dpas),
    # and TOKEN_LOCAL halving shrinks the SLM ring 64 -> 32 KB, i.e. 4
    # resident work-groups per Xe-core instead of 2.
    "rg4rb1_pfa1": dict(row_groups=4, token_groups=2, token_local=16, row_blocks=1,
                        slm_jblk=4, pf_a=1, a_outer=1),
    "rg4rb1_tl8":  dict(row_groups=4, token_groups=2, token_local=8, row_blocks=1,
                        slm_jblk=2, pf_a=0, a_outer=1),
    "rg4rb1_tl32": dict(row_groups=4, token_groups=2, token_local=32, row_blocks=1,
                        slm_jblk=8, pf_a=0, a_outer=1),
    "rg4rb2_tl16": dict(row_groups=4, token_groups=2, token_local=16, row_blocks=2,
                        slm_jblk=2, pf_a=0, a_outer=1),
    "rg4rb1_tg1":  dict(row_groups=4, token_groups=1, token_local=16, row_blocks=1,
                        slm_jblk=4, pf_a=0, a_outer=1),
}


def opts(cfg):
    return (f"-cmc -DROW_GROUPS={cfg['row_groups']} "
            f"-DTOKEN_GROUPS={cfg['token_groups']} "
            f"-DTOKEN_LOCAL={cfg['token_local']} "
            f"-DROW_BLOCKS={cfg['row_blocks']} "
            f"-DSLM_JBLK={cfg['slm_jblk']} "
            f"-DPF_A={cfg['pf_a']} -DA_OUTER={cfg['a_outer']} "
            f"{cfg.get('extra', '')}")


# ---------------------------------------------------------------------------
# Component ablation (--ablate), the A_OUTER=1 counterpart of ab_bottleneck.py
# ---------------------------------------------------------------------------
# Each probe removes ONE component from the SAME source by an in-memory
# textual patch, so the answer to "what is the bottleneck now?" comes from
# interleaved timings of one process rather than from a guess. Only `base` is
# numerically correct; the probes are timing instruments.
_A_LOAD = """                b2dA.set_block_y((int)(tile_base + tg * TOKENS_PER_TILE));
                cm_load<lsc::Normal, CacheHint::A_L1H, CacheHint::A_L2H>(
                    A2[tg], b2dA.set_block_x((int)(kb * 256u) + j * 32));"""
_A_LOAD_CONST = """                A2[tg] = (half)1.0f;"""
_A_LOAD_NOTRAF = """                b2dA.set_block_y(0);
                cm_load<lsc::Normal, CacheHint::A_L1H, CacheHint::A_L2H>(
                    A2[tg], b2dA.set_block_x(0));"""

_SLM_RD = """                vector<half, 8*32> Blo, Bhi;
                cm_slm_block_read(slm, GENX_NONE,
                    (int)(sbase + (ti * 2 + 0) * SLM_TILE_B), Blo);
                cm_slm_block_read(slm, GENX_NONE,
                    (int)(sbase + (ti * 2 + 1) * SLM_TILE_B), Bhi);"""
_SLM_RD_CONST = """                vector<half, 8*32> Blo = (half)1.0f, Bhi = (half)1.0f;"""

_DECODE_RE = re.compile(r"^[ \t]*q\w+_vnni_decode\((?:[^;]|\n)*?\);", re.MULTILINE)
_DECODE_RAW = ("Blo.format<uint>().select<4*OPG,1>(0) = q; "
               "Bhi.format<uint>().select<4*OPG,1>(0) = q;")

ABLATIONS = ["base", "noAtraf", "noAload", "noSLMrd", "nodecode", "dpasonly"]


def ablate(src, variant):
    """Patch one component out of the A_OUTER=1 consumer/producer path."""
    if variant == "noAtraf":
        assert _A_LOAD in src, "noAtraf patch failed"
        src = src.replace(_A_LOAD, _A_LOAD_NOTRAF)
    if variant in ("noAload", "dpasonly"):
        assert _A_LOAD in src, "noAload patch failed"
        src = src.replace(_A_LOAD, _A_LOAD_CONST)
    if variant in ("noSLMrd", "dpasonly"):
        assert _SLM_RD in src, "noSLMrd patch failed"
        src = src.replace(_SLM_RD, _SLM_RD_CONST)
    if variant in ("nodecode", "dpasonly"):
        src, n = _DECODE_RE.subn(_DECODE_RAW, src)
        assert n >= 1, "nodecode patch failed"
    return src


def dispatch(cfg, K, N, token_len):
    tpt = TOKENS_PER_TILE * cfg["token_groups"]
    token_slots = cfg["token_local"] // cfg["row_blocks"]
    ntiles = (token_len + tpt - 1) // tpt
    gsize = (N // (OPG * cfg["row_groups"] * cfg["row_blocks"]),
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
    out_b = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=token_len * N * 4)
    return ref, x_b, w_b, out_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--flush_mib", type=int, default=128)
    ap.add_argument("--variants", type=str,
                    default="base,aouter,nopfa,rg1,tg2,tl16,tl16rb2,rg4",
                    help="comma list of VARIANTS keys; the first is the baseline")
    ap.add_argument("--quants", type=str, default="q40,q41,q80")
    ap.add_argument("--shapes", type=str, default=None,
                    help="comma list of quant:KxNxTOKENS; default is the four "
                         "(N,K) transformer shapes at token_len=1024")
    ap.add_argument("--ablate", action="store_true",
                    help="component ablation instead of a config sweep: build "
                         "the ADOPTED config with one component patched out "
                         "per variant (see ABLATIONS) and report the speedup, "
                         "i.e. what each component currently costs")
    ap.add_argument("--base-variant", type=str, default="aouter",
                    help="which VARIANTS entry --ablate builds its probes "
                         "from (use 'ptl' on the Xe3 iGPU)")
    args = ap.parse_args()

    names = ABLATIONS if args.ablate else args.variants.split(",")
    base = names[0]
    gpu = Gpu(args.device)
    gpu.setup_flush(args.flush_mib)

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

    hdr = f"{'shape':<28}"
    for n in names:
        hdr += f"{n+' ms':>13}"
    for n in names[1:]:
        hdr += f"{'vs '+base:>10}"
    print(hdr)
    print("-" * len(hdr))

    for quant, K, N, tl in shapes:
        ref, x_b, w_b, out_b = prep(gpu, quant, K, N, tl)
        _, cm_file, kernel_name = QUANTS[quant]
        variants, bad = {}, []
        src0 = open(cl_src(cm_file)).read()
        for n in names:
            cfg = VARIANTS[args.base_variant] if args.ablate else VARIANTS[n]
            assert cfg["slm_jblk"] * cfg["row_groups"] * cfg["row_blocks"] == \
                cfg["token_local"], f"{n}: SLM_NPROD != TOKEN_LOCAL"
            if args.ablate:
                prog = gpu._build_src(ablate(src0, n), (quant, n), opts(cfg))
            else:
                prog = gpu.build(cl_src(cm_file), opts(cfg))
            krn = cl.Kernel(prog, kernel_name)
            krn.set_args(x_b, w_b, out_b,
                         np.uint32(tl), np.uint32(K), np.uint32(N))
            gs, ls = dispatch(cfg, K, N, tl)
            cl.enqueue_nd_range_kernel(gpu.queue, krn, gs, ls).wait()
            got = np.empty(tl * N, dtype=np.float32)
            cl.enqueue_copy(gpu.queue, got, out_b)
            gpu.queue.finish()
            ok = check_close(f"{quant}/{n}", ref, got.reshape(tl, N))
            # ablation probes are deliberately wrong -- only `base` must pass
            if not ok and (not args.ablate or n == "base"):
                bad.append(n)
            variants[n] = (krn, gs, ls)

        for _ in range(args.warmup):
            for n in names:
                krn, gs, ls = variants[n]
                gpu.flush_l3()
                cl.enqueue_nd_range_kernel(gpu.queue, krn, gs, ls).wait()

        ts = {n: [] for n in names}
        for _ in range(args.iters):
            for n in names:                       # round-robin: same clock state
                krn, gs, ls = variants[n]
                gpu.flush_l3()
                e = cl.enqueue_nd_range_kernel(gpu.queue, krn, gs, ls)
                e.wait()
                ts[n].append((e.profile.end - e.profile.start) * 1e-6)
        mn = {n: min(ts[n]) for n in names}

        row = f"{quant+' K'+str(K)+' N'+str(N)+' t'+str(tl):<28}"
        for n in names:
            row += f"{mn[n]:>13.3f}"
        for n in names[1:]:
            row += f"{mn[base]/mn[n]:>9.2f}x"
        if bad:
            row += "  WRONG:" + ",".join(bad)
        print(row)


if __name__ == "__main__":
    main()
