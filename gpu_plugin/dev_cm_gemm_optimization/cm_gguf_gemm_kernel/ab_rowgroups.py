"""
ab_rowgroups.py -- A/B for N-blocking (ROW_GROUPS) in gemm_q{4,5,6}k_full.cm.

Motivation (ab_bottleneck.py): dequant and the SLM read are no longer measurable
costs; the ACTIVATION operand is, at ~1/3 of the runtime. The A tile of a given
(sub-block, token group) does not depend on the weight row group, so letting one
thread own ROW_GROUPS row groups makes each A load feed ROW_GROUPS times as many
dpas -- A bytes and A messages per dpas both drop by ROW_GROUPS.
TOKEN_GROUPS is halved with it so the accumulator count (register pressure) and
the 16 KB SLM ring are unchanged; the kernel picks those defaults itself.

Builds RG=1 and RG=2 from the SAME source in ONE process, checks both against a
numpy reference, and times them INTERLEAVED with a cache flush before every
timed launch, min-of-N (this machine's clocks drift +-40% between processes).

Usage: python ab_rowgroups.py [--rgs 1,2] [--iters 20]
"""
import argparse
import re

import numpy as np
import pyopencl as cl

from test_dense_gemm_slim import OPG, cl_src, Gpu
from test_dense_gemv import (
    build_q4k_weight as build_q4k_flat,
    build_q5k_weight as build_q5k_flat,
    build_q6k_weight as build_q6k_flat,
)

TOKENS_PER_TILE = 8
BUILDERS = {"q4k": build_q4k_flat, "q5k": build_q5k_flat, "q6k": build_q6k_flat}

# label -> (ROW_GROUPS, TOKEN_GROUPS, TOKEN_LOCAL, extra build options).
# All four are listed explicitly because the host dispatch depends on them.
# The *_g256 entries are the large-GRF experiment: -Qxcm_register_file_size=256
# halves the resident threads per XVE (8 -> 4), so it can only pay off if the
# per-thread tile grows enough to keep the same work in flight (it does not --
# measured a wash, see gemm_q4k_full.cm).
G256 = "-Qxcm_register_file_size=256"
CONFIGS = {
    "tg4rb4":  (2, 4, 32, "-DROW_BLOCKS=4 -DA_OUTER=1 -DSLM_JBLK=8 -DPF_A=0"),
    "tg4rb1":  (2, 4, 32, "-DROW_BLOCKS=1 -DA_OUTER=1 -DSLM_JBLK=8 -DPF_A=0"),
    "tg8":     (2, 8, 32, "-DROW_BLOCKS=2 -DA_OUTER=1 -DSLM_JBLK=8 -DPF_A=0"),
    "rg1":        (1, 4, 8, ""),                    # v7
    "rg2":        (2, 2, 8, ""),
    "rg2j8":      (2, 2, 8, "-DSLM_JBLK=8"),        # 32 KB SLM (4 WG/Xe-core)
    "rg2tg4":     (2, 4, 8, ""),                    # adopted default
    "rg4tg1":     (4, 1, 8, "-DSLM_JBLK=2"),
    "rg2tg4g256": (2, 4, 8, G256),
    "rg2tg8g256": (2, 8, 8, G256),                  # 64 tokens/thread
    "rg4tg2g256": (4, 2, 8, f"-DSLM_JBLK=2 {G256}"),
    "rg4tg4g256": (4, 4, 8, f"-DSLM_JBLK=2 {G256}"),  # 64 channels x 32 tokens
    # ---- operand-fetch knobs on top of the adopted default ----------------
    "pfa1":       (2, 4, 8, "-DPF_A=1"),            # prefetch next K-block's A
    "pfa2":       (2, 4, 8, "-DPF_A=2"),
    "pfw2":       (2, 4, 8, "-DPF_W=2"),            # prefetch next phase's payload
    "pfaw":       (2, 4, 8, "-DPF_A=1 -DPF_W=2"),
    "tl16":       (2, 4, 16, "-DSLM_JBLK=8"),       # 16 thr/WG, 32 KB, 1 bar/K-blk
    "depth2":     (2, 4, 8, "-DSLM_DEPTH=2 -DSLM_SLOTS=4"),  # stage 2 phases ahead
    "rd1k":       (2, 4, 8, "-DSLM_RD1K=1"),        # one 1 KB SLM read per pair
    "rd1kpfa":    (2, 4, 8, "-DSLM_RD1K=1 -DPF_A=1"),
    "rd1ktl16":   (2, 4, 16, "-DSLM_RD1K=1 -DSLM_JBLK=8"),
    "acc_":       (2, 4, 8, "-DA_L1H=Cached -DA_L2H=Cached"),
    "astream":    (2, 4, 8, "-DA_L1H=Streaming -DA_L2H=Cached"),
    "aunc":       (2, 4, 8, "-DA_L1H=Uncached -DA_L2H=Cached"),
    "accpfa":     (2, 4, 8, "-DA_L1H=Cached -DA_L2H=Cached -DPF_A=1"),
    # ---- "smaller acc + more operands in flight" dataflow ------------------
    # acc owns ROW_GROUPS*TOKEN_GROUPS*512 B; at 2x4 that is half the register
    # file, which is what stops the compiler from running the operand loads
    # further ahead. These shrink acc and spend the freed registers on an
    # explicit double buffer of the SLM tile reads (CONS_DBUF, now generic
    # over ROW_GROUPS). RG=2/TG=4 + DBUF spills 3072 B, hence the smaller ones.
    "rg2tg2db":   (2, 2, 8, "-DCONS_DBUF=1"),       # acc 2 KB, 128 regs
    "rg1tg4db":   (1, 4, 8, "-DCONS_DBUF=1"),       # acc 2 KB, 128 regs
    "rg2tg1db":   (2, 1, 8, "-DCONS_DBUF=1"),       # acc 1 KB, 107 regs
    "rg1tg8db":   (1, 8, 8, "-DCONS_DBUF=1"),       # acc 4 KB, 125 regs
    # ---- knobs re-opened for the PTL / Xe3 iGPU ----------------------------
    # PTL (Arc B390: 96 XVE @ 2.4 GHz, 10 threads/EU, 16 MB L2, shared LPDDR)
    # has a very different balance from BMG: the ablation there shows the A
    # cache traffic (noAtraf 1.52-2.19x) and the dequant ALU (nodecode
    # 1.14-1.45x) both matter much more, so cache-locality and prefetch knobs
    # that lost on BMG are worth re-measuring.
    "pfa0":       (2, 4, 8, "-DPF_A=0"),            # A prefetch off
    "pfa2b":      (2, 4, 8, "-DPF_A=2"),
    "pfw2b":      (2, 4, 8, "-DPF_W=2"),
    "pfa1w2":     (2, 4, 8, "-DPF_A=1 -DPF_W=2"),
    "walk2":      (2, 4, 8, "-DWALK_HBLK=2"),       # band-swizzle the walker
    "walk4":      (2, 4, 8, "-DWALK_HBLK=4"),
    "walk8":      (2, 4, 8, "-DWALK_HBLK=8"),
    "walk16":     (2, 4, 8, "-DWALK_HBLK=16"),
    "tl4":        (2, 4, 4, "-DSLM_JBLK=2"),        # 4 thr/WG, 8 KB SLM
    "tl16b":      (2, 4, 16, "-DSLM_JBLK=8"),       # 16 thr/WG, 32 KB SLM
    "grf160":     (2, 4, 8, "-Qxcm_register_file_size=160"),
    "grf192":     (2, 4, 8, "-Qxcm_register_file_size=192"),
    # wider work-groups: on PTL (10 threads/EU) a 16-thread WG shares the
    # cooperative decode 16 ways instead of 8 and needs one barrier per
    # K-block instead of two. SLM_NPROD = SLM_JBLK*ROW_GROUPS must equal
    # TOKEN_LOCAL for every producer to have exactly one tile pair.
    "tl16pfa0":   (2, 4, 16, "-DSLM_JBLK=8 -DPF_A=0"),
    "tl16tg2":    (2, 2, 16, "-DSLM_JBLK=8 -DPF_A=0"),
    "tl16rg4":    (4, 2, 16, "-DSLM_JBLK=4 -DPF_A=0"),
    "tl32rg4":    (4, 2, 32, "-DSLM_JBLK=8 -DPF_A=0"),
    "tl32":       (2, 4, 32, "-DSLM_JBLK=8 -DPF_A=0"),
    "tl16tg1":    (2, 1, 16, "-DSLM_JBLK=8 -DPF_A=0"),
    "tl16tg3":    (2, 3, 16, "-DSLM_JBLK=8 -DPF_A=0"),
    "tl16tg2p":   (2, 2, 16, "-DSLM_JBLK=8 -DPF_A=1"),
    "tl16tg2j4":  (2, 2, 16, "-DSLM_JBLK=4 -DPF_A=0"),
    "tl16tg2s4":  (2, 2, 16, "-DSLM_JBLK=8 -DPF_A=0 -DSLM_SLOTS=4 -DSLM_DEPTH=2"),
    "tl8tg2j8":   (2, 2, 8, "-DSLM_JBLK=8 -DPF_A=0"),
    "ptl":        (2, 2, 16, "-DSLM_JBLK=8 -DPF_A=0"),
    "ptl_g96":    (2, 2, 16, "-DSLM_JBLK=8 -DPF_A=0 -Qxcm_register_file_size=96"),
    "ptl_g64":    (2, 2, 16, "-DSLM_JBLK=8 -DPF_A=0 -Qxcm_register_file_size=64"),
    "ptl_g128":   (2, 2, 16, "-DSLM_JBLK=8 -DPF_A=0 -Qxcm_register_file_size=128"),
    "ptl_tg1":    (2, 1, 16, "-DSLM_JBLK=8 -DPF_A=0"),
    "ptl_tg1g96": (2, 1, 16, "-DSLM_JBLK=8 -DPF_A=0 -Qxcm_register_file_size=96"),
    "ptl_flatA":  (2, 2, 16, "-DSLM_JBLK=8 -DPF_A=0 -DA_LOAD_2D=0"),
    "ptl_pad32":  (2, 2, 16, "-DSLM_JBLK=8 -DPF_A=0 -DSLM_PAD=32"),
    # trade cheap SLM-side B traffic for expensive L2-side A traffic: on PTL
    # noAtraf (1.5-1.8x) >> noSLMrd (1.0-1.2x), so a bigger ROW_GROUPS pays
    # even though it raises total operand bytes/dpas.
    "rg4tg1":     (4, 1, 16, "-DSLM_JBLK=4 -DPF_A=0"),
    "rg4tg1p":    (4, 1, 16, "-DSLM_JBLK=4 -DPF_A=1"),
    "rg4tg2j4":   (4, 2, 16, "-DSLM_JBLK=4 -DPF_A=0"),
    "rg8tg1":     (8, 1, 16, "-DSLM_JBLK=2 -DPF_A=0"),
    "rg4tg1tl8":  (4, 1, 8, "-DSLM_JBLK=2 -DPF_A=0"),
    # ROW_BLOCKS: several row-group blocks per WG so the threads sharing a
    # token slot also share the activation tile in L1 (A L2 traffic / ROW_BLOCKS)
    "rb2":        (2, 2, 16, "-DSLM_JBLK=4 -DPF_A=0 -DROW_BLOCKS=2"),
    "rb4":        (2, 2, 16, "-DSLM_JBLK=2 -DPF_A=0 -DROW_BLOCKS=4"),
    "rb8":        (2, 2, 16, "-DSLM_JBLK=1 -DPF_A=0 -DROW_BLOCKS=8"),
    "rb2tg4":     (2, 4, 16, "-DSLM_JBLK=4 -DPF_A=0 -DROW_BLOCKS=2"),
    "rb4tg4":     (2, 4, 16, "-DSLM_JBLK=2 -DPF_A=0 -DROW_BLOCKS=4"),
    "nomerge":    (2, 2, 16, "-DSLM_JBLK=8 -DPF_A=0 -DMERGE_STORE=0"),
    "merge":      (2, 2, 16, "-DSLM_JBLK=8 -DPF_A=0 -DMERGE_STORE=1"),
    "merge_rb2":  (2, 2, 16, "-DSLM_JBLK=4 -DPF_A=0 -DROW_BLOCKS=2 -DMERGE_STORE=1"),
    # A-sharing (ROW_BLOCKS) WITHOUT paying an extra barrier: a 32-thread WG
    # keeps SLM_NPROD == TOKEN_LOCAL at SLM_JBLK=8, i.e. one barrier/K-block.
    "rb2tl32":    (2, 2, 32, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=2"),
    "rb4tl32":    (2, 2, 32, "-DSLM_JBLK=4 -DPF_A=0 -DROW_BLOCKS=4"),
    "rb2tl32t4":  (2, 4, 32, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=2"),
    # A_OUTER: hold the activation tiles, stream B one row group at a time.
    # Live B stops scaling with ROW_GROUPS (102 regs instead of 119), which is
    # what finally makes ROW_GROUPS=4 compile without spilling.
    "ao1":        (2, 2, 32, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1"),
    "rg4ao":      (4, 2, 32, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=1 -DA_OUTER=1"),
    "rg4rb2ao":   (4, 2, 32, "-DSLM_JBLK=4 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1"),
    "rg4ao16":    (4, 2, 16, "-DSLM_JBLK=4 -DPF_A=0 -DROW_BLOCKS=1 -DA_OUTER=1"),
    "ao_rb4":     (2, 2, 32, "-DSLM_JBLK=4 -DPF_A=0 -DROW_BLOCKS=4 -DA_OUTER=1"),
    "ao_rg1rb4":  (1, 2, 32, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=4 -DA_OUTER=1"),
    "ao_rb8":     (2, 2, 32, "-DSLM_JBLK=2 -DPF_A=0 -DROW_BLOCKS=8 -DA_OUTER=1"),
    "ao_tl64":    (2, 2, 64, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=4 -DA_OUTER=1"),
    # fp16 dequant arithmetic (what the SYCL original does). PTL's ablation
    # says the dequant ALU is 1.08-1.40x there, so halving its width may pay.
    # SLM footprint vs occupancy: SLM = SLM_SLOTS*SLM_JBLK*RG*RB KB per WG.
    # At JBLK=8 that is 64 KB, i.e. only 2 WGs (64 threads) per Xe-core against
    # the 80 thread slots Xe3 has -- SLM-limited occupancy.
    "slm32":      (2, 2, 32, "-DSLM_JBLK=4 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1"),
    "slm16":      (2, 2, 32, "-DSLM_JBLK=2 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1"),
    "slm32tl16":  (2, 2, 16, "-DSLM_JBLK=4 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1"),
    "slm16tl16":  (2, 2, 16, "-DSLM_JBLK=2 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1"),
    # weight-resident walker: when the weight matrix no longer fits in L2
    # (K or N much larger than the other), the default dim0-fastest walker
    # re-streams ALL the weights once per token block. WALK_HBLK keeps a band
    # of row groups resident and sweeps the token blocks past it instead.
    "w2":         (2, 2, 32, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1 -DWALK_HBLK=2"),
    "w4":         (2, 2, 32, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1 -DWALK_HBLK=4"),
    "w8":         (2, 2, 32, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1 -DWALK_HBLK=8"),
    "w16":        (2, 2, 32, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1 -DWALK_HBLK=16"),
    "tg4":        (2, 4, 32, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1"),
    # tokens per WG = (TOKEN_LOCAL/ROW_BLOCKS)*8*TOKEN_GROUPS. Bigger = fewer
    # token blocks = fewer full re-streams of a weight matrix that does not
    # fit in L2 (the K>>N / N>>K shapes).
    "tl64":       (2, 2, 64, "-DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1 -DSLM_JBLK=8"),
    "tg4tl64":    (2, 4, 64, "-DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1 -DSLM_JBLK=8"),
    "tl128":      (2, 2, 128, "-DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1 -DSLM_JBLK=8"),
    "dh1":        (2, 2, 32, "-DSLM_JBLK=8 -DPF_A=0 -DROW_BLOCKS=2 -DA_OUTER=1 -DDECODE_HALF=1"),
}


def dispatch(rg, tg, tlocal, N, token_len, rb=1):
    """rb = ROW_BLOCKS: row-group blocks per work-group. The WG then covers
    rb*rg row groups and tlocal/rb distinct token tiles."""
    tpt = TOKENS_PER_TILE * tg
    ntiles = (token_len + tpt - 1) // tpt
    slots = tlocal // rb
    gsize = (N // (OPG * rg * rb),
             ((ntiles + slots - 1) // slots) * tlocal)
    return gsize, (1, tlocal)


def row_blocks_of(extra):
    m = re.search(r"-DROW_BLOCKS=(\d+)", extra)
    return int(m.group(1)) if m else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--flush_mib", type=int, default=128)
    ap.add_argument("--variants", type=str, default="rg1,rg2",
                    help=f"comma list from {sorted(CONFIGS)} (first = baseline)")
    ap.add_argument("--quants", type=str, default="q4k")
    ap.add_argument("--shapes", type=str,
                    default="2048x2048x1024,2048x2048x4096,2048x2048x8192,"
                            "4096x4096x1024,4096x4096x4096,4096x4096x8192",
                    help="comma list of KxNxtoken_len")
    args = ap.parse_args()

    labels = args.variants.split(",")
    base = labels[0]
    gpu = Gpu(args.device)
    gpu.setup_flush(args.flush_mib)

    shapes = []
    for q in args.quants.split(","):
        for s in args.shapes.split(","):
            K, N, tl = (int(v) for v in s.split("x"))
            shapes.append((q, K, N, tl))

    hdr = f"{'shape':<26}" + "".join(f"{l+' ms':>12}" for l in labels)
    hdr += "".join(f"{'vs '+base:>10}" for l in labels[1:])
    hdr += f"{'TFLOPS':>9}  ok"
    print(hdr)
    print("-" * len(hdr))

    for quant, K, N, tl in shapes:
        flat, ref_W, _ = BUILDERS[quant](K, N, 42)
        rng = np.random.default_rng(43)
        x_in = rng.standard_normal((tl, K)).astype(np.float32).astype(np.float16)
        ref = (ref_W.astype(np.float16).astype(np.float64) @
               x_in.astype(np.float64).T).astype(np.float32).T
        mf = cl.mem_flags
        x_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=np.ascontiguousarray(x_in))
        w_b = cl.Buffer(gpu.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat)
        out_b = cl.Buffer(gpu.ctx, mf.WRITE_ONLY, size=tl * N * 4)

        variants, errs = {}, {}
        for label in labels:
            rg, tg, tlocal, extra = CONFIGS[label]
            prog = gpu.build(cl_src(f"gemm_{quant}_full.cm"),
                             f"-cmc -DROW_GROUPS={rg} -DTOKEN_GROUPS={tg} "
                             f"-DTOKEN_LOCAL={tlocal} {extra}")
            k = cl.Kernel(prog, f"gemm_{quant}_full")
            k.set_args(x_b, w_b, out_b, np.uint32(tl), np.uint32(K), np.uint32(N))
            gs, ls = dispatch(rg, tg, tlocal, N, tl, row_blocks_of(extra))
            variants[label] = (k, gs, ls)

            cl.enqueue_nd_range_kernel(gpu.queue, k, gs, ls).wait()
            got = np.empty(tl * N, dtype=np.float32)
            cl.enqueue_copy(gpu.queue, got, out_b)
            gpu.queue.finish()
            errs[label] = (np.abs(got.reshape(tl, N) - ref).max() /
                           max(np.abs(ref).max(), 1e-9))

        for _ in range(args.warmup):
            for label in labels:
                k, gs, ls = variants[label]
                gpu.flush_l3()
                cl.enqueue_nd_range_kernel(gpu.queue, k, gs, ls).wait()

        acc = {l: [] for l in labels}
        for _ in range(args.iters):
            for label in labels:
                k, gs, ls = variants[label]
                gpu.flush_l3()
                e = cl.enqueue_nd_range_kernel(gpu.queue, k, gs, ls)
                e.wait()
                acc[label].append((e.profile.end - e.profile.start) * 1e-6)
        mn = {l: min(acc[l]) for l in labels}

        row = f"{quant+' K'+str(K)+' N'+str(N)+' t'+str(tl):<26}"
        for label in labels:
            row += f"{mn[label]:>12.3f}"
        for label in labels[1:]:
            row += f"{mn[base]/mn[label]:>9.2f}x"
        best = min(mn.values())
        row += f"{tl * 2 * N * K / (best * 1e-3) / 1e12:>9.1f}"
        bad = [f"{l}:{errs[l]:.1e}" for l in labels if errs[l] > 1e-2]
        row += "  " + ("OK" if not bad else "FAIL " + ",".join(bad))
        print(row)


if __name__ == "__main__":
    main()
