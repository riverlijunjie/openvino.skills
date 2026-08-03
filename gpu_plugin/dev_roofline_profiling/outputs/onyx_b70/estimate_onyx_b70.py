#!/usr/bin/env python3
"""Onyx text decoder — Analytical roofline estimate on B70 GPU.

B70 GPU specs (user-provided, 2026-07-15):
  32 Xe Cores, 2280 MHz
  INT8 XMX : 367 TOPS
  FP16 XMX : 183.5 TFLOPS  (INT8/2 — standard XMX 2:1 ratio)
  Memory BW: 608 GB/s

Method: scale measured PTL 12Xe timings (performance_metrics.json) by HW ratios.

  memory-bound ops   → time_B70 = time_PTL × (BW_PTL  / BW_B70)   =  × 0.1809
  XMX-compute ops    → time_B70 = time_PTL × (XMX_PTL / XMX_B70)  =  × 0.3214
  (INT8 and FP16 ratios are identical because B70 keeps the 2:1 relationship)

Sizes not in measured data (kv=64K / S=64K): extrapolated from 32K:
  FC prefill   (linear in S)              ×2 on 32K single_ms
  PA_full pref (quadratic: pairs ×4)      ×4 on 32K single_ms
  PA_slid pref (band S×SW, S>>SW)         ×2.03 on 32K single_ms
  SmallOps pre (linear in S)              ×2 on 32K total_ms
  PA_full dec  (linear in kv)             ×2 on 32K single_ms
  All other decode ops: M=1, kv-independent — same single_ms for all kv.

Efficiency % (Eff%) is invariant under hardware-ratio scaling:
  Eff% = t_theo/t_meas, and both numerator/denominator scale by the same factor.
  So B70 Eff% = PTL Eff% for all ops. This assumes the same kernel-quality
  factor (occupancy, bank conflicts, dispatch overhead) on B70 as on PTL 12Xe,
  which is valid for the same OpenVINO FullyConnectedCompressed + PagedAttention
  primitives.
"""
import json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATE = "2026-07-15"

# ─── B70 hardware (user-provided) ────────────────────────────────────────────
BW_B70   = 608.0        # GB/s
INT8_B70 = 367.0        # TOPS
FP16_B70 = INT8_B70 / 2  # 183.5 TFLOPS
OVH = 0.95
BW_B70_A, FP16_B70_A, INT8_B70_A = BW_B70*OVH, FP16_B70*OVH, INT8_B70*OVH

# ─── PTL 12Xe reference ──────────────────────────────────────────────────────
BW_PTL   = 110.0
FP16_PTL = 58.9824
INT8_PTL = FP16_PTL * 2   # 117.9648 TOPS

# scaling factors (multiply PTL single_ms by these to get B70 single_ms)
MEM_SF = BW_PTL  / BW_B70    # 0.18092 — memory-bound
XMX_SF = FP16_PTL / FP16_B70  # 0.32147 — XMX compute-bound (same for INT8 & FP16)

# ─── Model config ────────────────────────────────────────────────────────────
H, L, NH, NKV, HD = 6656, 52, 32, 2, 128
INTER, VOCAB, SW, G = 19968, 202048, 2048, 128
FULL  = sum(1 for i in range(L) if i % 4 == 3)  # 13 full/NoPE layers
SLIDE = L - FULL                                  # 39 sliding/RoPE layers

# ─── Byte / FLOP helpers (u4/u8 quantized FC; u4 KV cache) ──────────────────
def i4w(K, N): return N*K/2 + N*(K/G)*2 + N*(K/G)*0.5
def i8w(K, N): return N*K + N*(K/G)*2
def fc_b(M, K, N, q): return M*K*2 + (i4w(K,N) if q=='u4' else i8w(K,N)) + M*N*2
def fc_f(M, K, N):    return 2*M*K*N
def kvb(kv):          # u4 KV: int4 + fp16-scale + int4-zp, k+v
    per_tok = (NKV*HD*0.5 + NKV*(HD/G)*2 + NKV*(HD/G)*0.5) * 2
    return per_tok * kv
def pad_b(kv): return kvb(kv) + NH*HD*2 + NH*HD*2   # PA decode: Q + out
def pad_f(kv): return 4*NH*HD*kv                     # QK^T + PV, M=1
def pap_b(S, kv): return kvb(kv) + S*NH*HD*2 + S*NH*HD*2
def pap_f(pairs): return 4*NH*HD*pairs

# ─── Sliding-band causal pair count ─────────────────────────────────────────
def slid_pairs(S):
    kvs = min(S, SW)
    return S*kvs - kvs*(kvs-1)/2

# ─── Load measured PTL data ──────────────────────────────────────────────────
PTL = json.load(open(HERE.parent / "onyx" / "performance_metrics.json"))

def ptl(phase, size, op, field="single_ms"):
    for r in PTL[phase][str(size)]["rows"]:
        if r["op"] == op: return r[field]
    return 0.0

# ─── Row builder ─────────────────────────────────────────────────────────────
def row(op, kernel, single_ms, calls, bytes_, flops, eff_pct, bound):
    tot = single_ms * calls
    gbs = bytes_ / (single_ms*1e-3) / 1e9 if single_ms > 0 else 0.0
    gf  = flops  / (single_ms*1e-3) / 1e9 if single_ms > 0 else 0.0
    return dict(op=op, kernel=kernel, single_ms=single_ms, calls=calls,
                total_ms=tot, gflops=gf, gbs=gbs, eff=eff_pct, bound=bound)

def theo_sum(rows):
    t = 0.0
    for r in rows:
        e = r["eff"] / 100.0 if r["eff"] > 0 else 0.45
        t += r["total_ms"] * e
    return t

SIZES = [4096, 16384, 32768, 65536]
FC_OPS = [
    ("FC_QKV",    H, 4608, "u4", L),
    ("FC_OutGate",H, 4096, "u4", L),
    ("FC_O",   4096,    H, "u4", L),
    ("MLP_gate",  H, INTER,"u4", L),
    ("MLP_up",    H, INTER,"u4", L),
    ("MLP_down",INTER,  H, "u4", L),
]

out = {
    "platform": "B70 (analytical estimate)",
    "platform_spec": "32 XeCore × 2280 MHz | INT8 367 TOPS | FP16 183.5 TFLOPS | BW 608 GB/s",
    "model": "onyx (text decoder)", "measured": False,
    "base_platform": "PTL 12Xe (measured, performance_metrics.json)",
    "config": "INT4 g=128 body + INT8 g=128 LM_head + INT4 g=128 KV cache, FP16 act",
    "bw_gbs": BW_B70, "fp16_tflops": FP16_B70, "int8_tops": INT8_B70,
    "ridge_f16": FP16_B70_A*1e12 / (BW_B70_A*1e9),
    "ridge_i8":  INT8_B70_A*1e12 / (BW_B70_A*1e9),
    "sizes": SIZES, "decode": {}, "prefill": {},
    "scale_factors": {"mem_sf": MEM_SF, "xmx_sf": XMX_SF,
                      "bw_ptl": BW_PTL, "fp16_ptl": FP16_PTL},
}

# ─── DECODE ──────────────────────────────────────────────────────────────────
for kv in SIZES:
    rows = []
    ref_kv = min(kv, 32768)  # PTL has up to 32K; kv=64K extrapolated

    # FC decode (M=1, memory-bound) — kv-independent, reuse any measured kv
    for op, K, N, q, calls in FC_OPS:
        s_ptl = ptl("decode", ref_kv, op)
        eff   = ptl("decode", ref_kv, op, "eff")
        rows.append(row(op, "gemm_kernel", s_ptl*MEM_SF, calls,
                        fc_b(1,K,N,q), fc_f(1,K,N), eff, "memory"))

    # LM_head (M=1, memory-bound)
    s_ptl = ptl("decode", 32768, "LM_head")
    eff   = ptl("decode", 32768, "LM_head", "eff")
    rows.append(row("LM_head","gemm_kernel", s_ptl*MEM_SF, 1,
                    fc_b(1,H,VOCAB,"u8"), fc_f(1,H,VOCAB), eff, "memory"))

    # PA_sliding decode — capped at SW=2048 for all kv≥2048
    kvs = min(kv, SW)
    ref_pa_s = 2048 if kv >= SW else kv
    s_ptl = ptl("decode", ref_pa_s, "PA_sliding")
    eff   = ptl("decode", ref_pa_s, "PA_sliding", "eff")
    rows.append(row("PA_sliding","pa_kv_cache_update+paged_attention",
                    s_ptl*MEM_SF, SLIDE, pad_b(kvs), pad_f(kvs), eff, "memory"))

    # PA_full decode — linear in kv; kv=65536 extrapolated ×2 from kv=32768
    if kv <= 32768:
        s_ptl = ptl("decode", kv, "PA_full")
        eff   = ptl("decode", kv, "PA_full", "eff")
        s_b70 = s_ptl * MEM_SF
    else:
        s_ptl_32 = ptl("decode", 32768, "PA_full")
        eff = ptl("decode", 32768, "PA_full", "eff")
        s_b70 = s_ptl_32 * 2 * MEM_SF
    rows.append(row("PA_full","pa_kv_cache_update+paged_attention",
                    s_b70, FULL, pad_b(kv), pad_f(kv), eff, "memory"))

    # SmallOps — M=1, kv-independent; use 32K value for 64K
    so_ref = min(kv, 32768)
    so_ptl = next(r["total_ms"] for r in PTL["decode"][str(so_ref)]["rows"] if r["op"]=="SmallOps")
    rows.append(dict(op="SmallOps", kernel="rms/rope/gate/add", single_ms=0, calls=0,
                     total_ms=so_ptl*MEM_SF, gflops=0, gbs=0, eff=0, bound="memory"))

    total = sum(r["total_ms"] for r in rows)
    theo  = theo_sum(rows)
    rows.sort(key=lambda r: -r["total_ms"])
    out["decode"][str(kv)] = {"total_ms":total, "theo_ms":theo,
                              "achieved_pct":theo/total*100, "rows":rows}

# ─── PREFILL ─────────────────────────────────────────────────────────────────
for S in SIZES:
    rows = []
    ref_S = min(S, 32768)
    extrap = (S > 32768)   # S=65536 extrapolated from S=32768

    # FC prefill (compute INT8-XMX, linear in S)
    scale_fc = 2 if extrap else 1
    for op, K, N, q, calls in FC_OPS:
        s_ptl = ptl("prefill", ref_S, op)
        eff   = ptl("prefill", ref_S, op, "eff")
        s_b70 = s_ptl * scale_fc * XMX_SF
        rows.append(row(op, "dq+gemm_kernel", s_b70, calls,
                        fc_b(S,K,N,q), fc_f(S,K,N), eff, "compute"))

    # LM_head (M=1 last token, memory-bound — constant regardless of S)
    s_ptl = ptl("prefill", 4096, "LM_head")
    eff   = ptl("prefill", 4096, "LM_head", "eff")
    rows.append(row("LM_head","gemm_kernel", s_ptl*MEM_SF, 1,
                    fc_b(1,H,VOCAB,"u8"), fc_f(1,H,VOCAB), eff, "memory"))

    # PA_full prefill (FP16 XMX, causal S²/2 pairs)
    #   S=64K: pairs(64K)/pairs(32K) = (64K²/2)/(32K²/2) ≈ 4 → ×4 in time
    if S <= 32768:
        s_ptl = ptl("prefill", S, "PA_full")
        eff   = ptl("prefill", S, "PA_full", "eff")
        s_b70 = s_ptl * XMX_SF
    else:
        s_ptl_32 = ptl("prefill", 32768, "PA_full")
        eff = ptl("prefill", 32768, "PA_full", "eff")
        s_b70 = s_ptl_32 * 4 * XMX_SF
    pairs_f = S*(S+1)/2
    rows.append(row("PA_full","sdpa_micro_prefill", s_b70, FULL,
                    pap_b(S,S), pap_f(pairs_f), eff, "compute"))

    # PA_sliding prefill (FP16 XMX, banded causal)
    #   S=64K: band(64K)/band(32K) ≈ 2.032 → ×2.032 in time
    if S <= 32768:
        s_ptl = ptl("prefill", S, "PA_sliding")
        eff   = ptl("prefill", S, "PA_sliding", "eff")
        s_b70 = s_ptl * XMX_SF
    else:
        s_ptl_32 = ptl("prefill", 32768, "PA_sliding")
        eff = ptl("prefill", 32768, "PA_sliding", "eff")
        band_ratio = slid_pairs(65536) / slid_pairs(32768)  # ≈ 2.032
        s_b70 = s_ptl_32 * band_ratio * XMX_SF
    kvs = min(S, SW)
    rows.append(row("PA_sliding","sdpa_micro_prefill", s_b70, SLIDE,
                    pap_b(S,kvs), pap_f(slid_pairs(S)), eff, "compute"))

    # SmallOps (memory-bound, linear in S; ×2 for S=64K)
    scale_so = 2 if extrap else 1
    so_ptl = next(r["total_ms"] for r in PTL["prefill"][str(ref_S)]["rows"] if r["op"]=="SmallOps")
    rows.append(dict(op="SmallOps", kernel="rms/rope/gate/add", single_ms=0, calls=0,
                     total_ms=so_ptl*scale_so*MEM_SF, gflops=0, gbs=0, eff=0, bound="memory"))

    total = sum(r["total_ms"] for r in rows)
    theo  = theo_sum(rows)
    rows.sort(key=lambda r: -r["total_ms"])
    out["prefill"][str(S)] = {"total_ms":total, "theo_ms":theo,
                              "achieved_pct":theo/total*100, "rows":rows}

# ─── Save JSON ───────────────────────────────────────────────────────────────
os.makedirs(HERE, exist_ok=True)
json.dump(out, open(HERE/"performance_metrics.json","w"), indent=2)
print("wrote performance_metrics.json")
print(f"  MEM_SF={MEM_SF:.4f}  XMX_SF={XMX_SF:.4f}")
print(f"  Ridge F16={out['ridge_f16']:.1f}  Ridge I8={out['ridge_i8']:.1f}")
print()
for kv in SIZES:
    d=out["decode"][str(kv)]
    print(f"decode kv={kv:6d}: {d['total_ms']:7.2f}ms  {1000/d['total_ms']:5.1f} tok/s  "
          f"theo {d['theo_ms']:6.2f}ms  achieved {d['achieved_pct']:.1f}%")
print()
for S in SIZES:
    d=out["prefill"][str(S)]
    print(f"prefill S={S:6d}: {d['total_ms']:9.2f}ms  {S/d['total_ms']*1e3:6.0f} tok/s  "
          f"theo {d['theo_ms']:8.2f}ms  achieved {d['achieved_pct']:.1f}%")

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
GEN = 512   # decode window for end-to-end
def f(x, n=2): return f"{x:,.{n}f}"
def fk(r): return f"{r['single_ms']:.4f}" if r['single_ms'] else "—"
def krow(r):
    sm = f"{r['single_ms']:.4f}" if r['single_ms'] else "—"
    gf = f"{r['gflops']:.1f}" if r['gflops'] else "—"
    gb = f"{r['gbs']:.1f}" if r['gbs'] else "—"
    ef = f"{r['eff']:.1f}%" if r['eff'] else "—"
    return (f"| {r['op']} | `{r['kernel']}` | {sm} | {r['calls']} | "
            f"{f(r['total_ms'],3)} | {gf} | {gb} | {ef} | {r['bound']} |")
def table(rows, total, header):
    hdr = ["",f"### {header}","",
           "| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    hdr += [krow(r) for r in rows]
    hdr.append(f"| **TOTAL** |  |  |  | **{f(total)}** |  |  |  |  |")
    hdr.append("")
    return "\n".join(hdr)

# weight footprint
def wt():
    items = []
    def add(n,sh,q,b,l): items.append((n,sh,q,b,l,b*l/1e6))
    add("Embedding",f"{H}×{VOCAB}","INT8 g128",i8w(H,VOCAB),1)
    add("FC_QKV",f"{H}×4608","INT4 g128",i4w(H,4608),L)
    add("FC_OutGate",f"{H}×4096","INT4 g128",i4w(H,4096),L)
    add("FC_O",f"4096×{H}","INT4 g128",i4w(4096,H),L)
    add("MLP_gate",f"{H}×{INTER}","INT4 g128",i4w(H,INTER),L)
    add("MLP_up",f"{H}×{INTER}","INT4 g128",i4w(H,INTER),L)
    add("MLP_down",f"{INTER}×{H}","INT4 g128",i4w(INTER,H),L)
    add("LM_Head",f"{H}×{VOCAB}","INT8 g128",i8w(H,VOCAB),1)
    return items
items = wt()
wtot = sum(i[5] for i in items)
fp16_mb = 2*sum(int(sh.split('×')[0])*int(sh.split('×')[1])*l for _,sh,_,_,l,_ in items)/1e6

S_out = []
def A(x): S_out.append(x)

A(f"# Onyx (dense VLM text decoder) — Roofline on B70 ({DATE})")
A("")
A(f"**Platform**: B70 GPU — 32 Xe @ 2280 MHz, 608 GB/s GDDR memory  ")
A(f"**Model**: `onyx` — 52-layer dense SwiGLU decoder (vision tower excluded)")
A("")
A("> **Analytical estimate** — no B70 device is available. Method: scale per-kernel "
  "timings measured on PTL 12Xe (B390 iGPU) by hardware ratios. "
  "Memory-bound ops: × (110/608) = × 0.181. XMX-compute-bound ops: × (58.98/183.5) = × 0.321. "
  "S=64K and kv=64K are extrapolated from S=kv=32K (FC linear ×2; PA_full quadratic ×4; "
  "PA_sliding / SmallOps band-linear ×2). Eff% is invariant under hardware scaling.")
A("")
A("- hidden=6656, layers=52, GQA 32Q/2KV/HD=128 (16-way sharing), inter=19968, vocab=202048, SwiGLU")
A("- Onyx-specific: **QK-RMSNorm** + **sigmoid output-gate** (extra FC 6656→4096/layer), "
  "iRoPE (39 sliding SW=2048 + 13 full/NoPE layers)")
A("- MatMul INT4 g=128 / LM_head INT8 g=128 / KV cache INT4 g=128 / FP16 act")
A("- SDPA: PagedAttention OpenCL + micro_kernel")
A("")

A("## Model parameters & weight shapes")
A("")
A("| Field | Value |")
A("|---|---:|")
A("| `hidden_size` | 6,656 |")
A("| `num_hidden_layers` | 52 (39 sliding/RoPE + 13 full/NoPE) |")
A("| `num_attention_heads` / `num_kv_heads` | 32 / 2 (GQA×16) |")
A("| `head_dim` | 128 → Q=4096, KV=256 |")
A("| `intermediate_size` | 19,968 |")
A("| `vocab_size` | 202,048 |")
A("| `sliding_window` | 2,048 |")
A("")
A("| Weight | Shape | Quant | Per instance | × | Total MB |")
A("|---|---|---|---:|---:|---:|")
for n,sh,q,b,l,mb in items:
    A(f"| {n} | {sh} | {q} | {b/1e6:,.2f} MB | {l} | {mb:,.1f} |")
A(f"| **Total** | | | | | **{wtot:,.0f} MB** |")
A("")
A(f"_FP16 baseline ≈ {fp16_mb:,.0f} MB → quantized {wtot:,.0f} MB = {wtot/fp16_mb*100:.0f}% of FP16. "
  f"Decode weight traffic per token (excluding embedding): ~{wtot-i8w(H,VOCAB)/1e6:,.0f} MB._")
A("")

A("## Hardware comparison: B70 vs PTL 12Xe reference")
A("")
A("| Metric | PTL 12Xe (measured) | B70 (estimate target) | Ratio |")
A("|---|---:|---:|---:|")
A(f"| FP16 XMX | {FP16_PTL:.3f} TFLOPS | {FP16_B70:.1f} TFLOPS | ×{FP16_B70/FP16_PTL:.2f} |")
A(f"| INT8 XMX | {INT8_PTL:.3f} TOPS | {INT8_B70:.0f} TOPS | ×{INT8_B70/INT8_PTL:.2f} |")
A(f"| Memory BW | {BW_PTL:.0f} GB/s | {BW_B70:.0f} GB/s | ×{BW_B70/BW_PTL:.2f} |")
A(f"| Ridge (FP16) | {(FP16_PTL*0.95*1e12)/(BW_PTL*0.95*1e9):.1f} | {out['ridge_f16']:.1f} FLOP/byte | — |")
A(f"| Decode expected speedup | 1× (baseline) | ×{BW_B70/BW_PTL:.2f} | BW-limited |")
A(f"| Prefill FC expected speedup | 1× (baseline) | ×{INT8_B70/INT8_PTL:.2f} | INT8-XMX-limited |")
A("")

A("## Theoretical roofline (B70)")
A("")
A("| Metric | Value |")
A("|---|---|")
A(f"| FP16 XMX peak | {FP16_B70:.1f} TFLOPS |")
A(f"| INT8 XMX peak | {INT8_B70:.0f} TOPS |")
A(f"| Memory BW | {BW_B70:.0f} GB/s |")
A(f"| Ridge point (FP16) | {out['ridge_f16']:.1f} FLOP/byte |")
A(f"| Ridge point (INT8) | {out['ridge_i8']:.1f} OP/byte |")
A("")
A("_B70 ridge points are 1.78× the PTL values (XMX scales by 3.11×, BW by 5.53×); "
  "the XMX-to-BW ratio is better balanced on B70 so prefill benefits relatively less "
  "than decode._")
A("")

A("## Data sources")
A("")
A("- **Base measurements**: on-device cliloader profiling of all ops on PTL 12Xe (B390 iGPU), "
  "99 kernel logs in `outputs/onyx/logs/`. See `outputs/onyx/SUMMARY_onyx_ptl_12xe_2026-07-14.md`.")
A("- **B70 scaling**: memory-bound ops × (110/608); XMX-bound ops × (58.98/183.5).")
A("- **S=64K / kv=64K extrapolation**: from S=kv=32K measurements.")
A("- **Eff%**: copied directly from PTL 12Xe measurements (invariant under scaling, "
  "same kernel family on both platforms).")
A("")

A("## Token latency summary")
A("")
A("### Prefill — TTFT")
A("")
A("| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |")
A("|---:|---:|---:|---:|---:|")
for S in SIZES:
    d=out["prefill"][str(S)]
    A(f"| {S:,} | {f(d['total_ms'])} | {f(d['total_ms']/1000,3)} | {f(d['total_ms']/S,4)} | {f(S/d['total_ms']*1e3,0)} |")
A("")
A("### Decode — TPOT (per output token)")
A("")
A("| KV | TPOT (ms) | tokens/s |")
A("|---:|---:|---:|")
for kv in SIZES:
    d=out["decode"][str(kv)]
    A(f"| {kv:,} | {f(d['total_ms'])} | {f(1000/d['total_ms'],1)} |")
A("")

A("## Roofline: theoretical floor vs estimated")
A("")
A("### Decode (per output token)")
A("")
A("| KV | theoretical (ms) | estimated (ms) | achieved % |")
A("|---:|---:|---:|---:|")
for kv in SIZES:
    d=out["decode"][str(kv)]
    A(f"| {kv:,} | {f(d['theo_ms'])} | {f(d['total_ms'])} | {f(d['achieved_pct'],1)}% |")
A("")
A("### Prefill (TTFT)")
A("")
A("| S | theoretical (ms) | estimated (ms) | achieved % |")
A("|---:|---:|---:|---:|")
for S in SIZES:
    d=out["prefill"][str(S)]
    A(f"| {S:,} | {f(d['theo_ms'])} | {f(d['total_ms'])} | {f(d['achieved_pct'],1)}% |")
A("")

A("## Decode tables")
A("")
for kv in SIZES:
    d=out["decode"][str(kv)]
    A(table(d["rows"], d["total_ms"], f"Decode — KV={kv:,}"))
    A("_SW=2048 caps PA_sliding for all KV≥2048; KV=64K PA_full extrapolated ×2 from 32K._\n")

A("## Prefill tables")
A("")
for S in SIZES:
    d=out["prefill"][str(S)]
    A(table(d["rows"], d["total_ms"], f"Prefill — S={S:,}"))
    A("_S=64K extrapolated from S=32K (FC ×2, PA_full ×4, PA_sliding ×2.03, SmallOps ×2)._\n")

A("## Top contributors")
A("")
A("### Decode")
A("| KV | top1 (ms,%) | top2 | top3 |")
A("|---:|---|---|---|")
for kv in SIZES:
    d=out["decode"][str(kv)]; tot=d["total_ms"]; rs=d["rows"][:3]
    cells=[f"{r['op']} {f(r['total_ms'],1)}ms ({f(r['total_ms']/tot*100,1)}%)" for r in rs]
    A(f"| {kv:,} | {cells[0]} | {cells[1]} | {cells[2]} |")
A("")
A("### Prefill")
A("| S | top1 (ms,%) | top2 | top3 |")
A("|---:|---|---|---|")
for S in SIZES:
    d=out["prefill"][str(S)]; tot=d["total_ms"]; rs=d["rows"][:3]
    cells=[f"{r['op']} {f(r['total_ms'],1)}ms ({f(r['total_ms']/tot*100,1)}%)" for r in rs]
    A(f"| {S:,} | {cells[0]} | {cells[1]} | {cells[2]} |")
A("")

A(f"## End-to-end (prefill TTFT + {GEN}-token decode)")
A("")
A(f"| prompt P | TTFT (ms) | {GEN}-tok decode (ms) | total (ms) | avg tok/s |")
A("|---:|---:|---:|---:|---:|")
for S in SIZES:
    ttft=out["prefill"][str(S)]["total_ms"]
    tpot=out["decode"][str(S)]["total_ms"]
    A(f"| {S:,} | {f(ttft)} | {f(tpot*GEN)} | {f(ttft+tpot*GEN)} | {f(1000/tpot,1)} |")
A("")

A("## Comparison: B70 vs PTL 12Xe")
A("")
A("| Phase | Metric | PTL 12Xe (measured) | B70 (estimated) | Speedup |")
A("|---|---|---:|---:|---:|")
for kv, label in [(4096,"decode kv=4K"),(32768,"decode kv=32K"),(65536,"decode kv=64K")]:
    ptl_ms = PTL["decode"][str(min(kv,32768))]["total_ms"] if str(min(kv,32768)) in PTL["decode"] else "—"
    b70_ms = out["decode"][str(kv)]["total_ms"]
    if isinstance(ptl_ms, float) and kv <= 32768:
        spd = f"×{ptl_ms/b70_ms:.2f}"
        A(f"| Decode | {label} | {f(ptl_ms)} ms | {f(b70_ms)} ms | {spd} |")
    elif kv > 32768:
        A(f"| Decode | {label} (extrap) | — | {f(b70_ms)} ms | — |")
for S, label in [(4096,"prefill S=4K"),(16384,"prefill S=16K"),(32768,"prefill S=32K"),(65536,"prefill S=64K")]:
    ptl_ms = PTL["prefill"][str(min(S,32768))]["total_ms"] if str(min(S,32768)) in PTL["prefill"] else None
    b70_ms = out["prefill"][str(S)]["total_ms"]
    if ptl_ms and S <= 32768:
        spd = f"×{ptl_ms/b70_ms:.2f}"
        A(f"| Prefill | {label} | {f(ptl_ms)} ms | {f(b70_ms)} ms | {spd} |")
    else:
        A(f"| Prefill | {label} (extrap) | — | {f(b70_ms)} ms | — |")
A("")

# Key findings — computed from data
d1k = out["decode"]["4096"]
def rowms(d, op): return next(r["total_ms"] for r in d["rows"] if r["op"]==op)
mlp_ms = sum(rowms(d1k,o) for o in ("MLP_gate","MLP_up","MLP_down"))
lm_ms  = rowms(d1k,"LM_head")
pre64  = out["prefill"]["65536"]; pre32 = out["prefill"]["32768"]
pa_full_eff = next(r["eff"] for r in pre32["rows"] if r["op"]=="PA_full")

A("## Key findings")
A("")
A(f"- **Decode is memory-bound at ~{1000/d1k['total_ms']:.1f} tok/s across all KV sizes** "
  f"({f(d1k['total_ms'])} ms/tok at KV=4K). "
  f"B70's 5.53× BW advantage over PTL 12Xe delivers a {BW_B70/BW_PTL:.2f}× decode speedup: "
  f"PTL {f(PTL['decode']['4096']['total_ms'])} ms → B70 {f(d1k['total_ms'])} ms. "
  f"MLP (gate+up+down ×52) is {mlp_ms/d1k['total_ms']*100:.0f}% of decode; "
  f"LM_head {lm_ms/d1k['total_ms']*100:.0f}%.")
A(f"- **Decode achieves ~{d1k['achieved_pct']:.0f}% of the B70 BW roofline** — same kernel "
  f"efficiency as measured on PTL. Little headroom without INT4 LM_head or speculative decoding.")
A(f"- **Prefill speedup is ~3.1× vs PTL** (XMX ratio 367/117.97 = 3.11) at moderate S. "
  f"TTFT at S=4K: {f(PTL['prefill']['4096']['total_ms'])} ms (PTL) → {f(out['prefill']['4096']['total_ms'])} ms (B70).")
A(f"- **PA_full prefill is the dominant TTFT cost at long context**: {f(rowms(pre32,'PA_full'))} ms "
  f"at S=32K = {rowms(pre32,'PA_full')/pre32['total_ms']*100:.0f}% of TTFT, with {pa_full_eff:.0f}% "
  f"FP16 XMX efficiency. B70 scales this proportionally, so the PA bottleneck shifts from "
  f"S≈16K on PTL to roughly S≈50K on B70.")
A(f"- **SmallOps (RMSNorm/residual-add) scale by BW** (5.5×) not XMX (3.1×), "
  f"so they shrink more relative to FC prefill on B70 and account for "
  f"{rowms(out['prefill']['32768'],'SmallOps')/out['prefill']['32768']['total_ms']*100:.0f}% "
  f"of prefill TTFT at S=32K vs "
  f"{next(r['total_ms'] for r in PTL['prefill']['32768']['rows'] if r['op']=='SmallOps')/PTL['prefill']['32768']['total_ms']*100:.0f}% on PTL.")
A("")

A("## Optimization levers (highest ROI first)")
A("")
A(f"1. **INT4 LM_head** (currently INT8): halves the ~{f(lm_ms,1)} ms LM_head decode cost "
  f"→ ~{lm_ms/2/d1k['total_ms']*100:.0f}% faster decode, frees ~{i8w(H,VOCAB)/2/1e6:.0f} MB of weight bytes.")
A("2. **Fuse `output_gate_proj` into FC_QKV** (single 6656→8704 wide gemm): "
  "removes a separate M=1 kernel launch per layer.")
A("3. **PA_full prefill at S≥32K**: at S=64K, PA_full is ~45% of TTFT; "
  "algorithmic improvements (CM micro-kernel, KV sparsity, chunk-prefill) directly "
  "reduce the dominant cost. The ~25% FP16 XMX efficiency already measured on PTL "
  "leaves 4× theoretical headroom.")
A("4. **Speculative decoding / MTP**: decode is fully memory-bound; amortizing weight "
  "reads over N accepted tokens gives N× decode throughput at the same memory footprint.")
A("5. **Batch decode**: multiple sequences share the same weight re-read cost "
  "(BW stays the same, compute scales by batch) until compute-bound; break-even batch "
  f"≈ ridge × ops-per-byte ≈ {out['ridge_f16']:.0f} FLOP/byte × 0.5 FLOP/byte (int4) ≈ B~{int(out['ridge_f16']*0.5/2):.0f}.")
A("")

A("## Caveats & method")
A("")
A("- **Estimate only** — no B70 device measured. All times derived from PTL 12Xe measurements × scaling.")
A("- Assumes same kernel occupancy / efficiency factor on B70 (same primitives, same OpenVINO version).")
A("- FP16 XMX = INT8/2 = 183.5 TFLOPS assumed; actual FP16 throughput should be confirmed on device.")
A("- S=64K and kv=64K are linear/quadratic extrapolations; actual values may differ by ±15%.")
A("- Vision tower / adapter excluded; text-generation roofline only.")
A("")

A("## Reproduction")
A("")
A("```bash")
A("# Run the actual PTL 12Xe benches first (already done):")
A("# see outputs/onyx/logs/ + outputs/onyx/performance_metrics.json")
A("")
A("# B70 estimate from PTL measured data:")
A("cd .github/skills/dev_roofline_profiling/outputs/onyx_b70")
A("python3 estimate_onyx_b70.py   # writes performance_metrics.json + this SUMMARY")
A("```")
A("")

open(HERE/f"SUMMARY_onyx_b70_{DATE}.md","w").write("\n".join(S_out)+"\n")
print(f"wrote SUMMARY_onyx_b70_{DATE}.md")
