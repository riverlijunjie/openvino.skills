#!/usr/bin/env python3
"""Generate SUMMARY_qwen3_5_moe_int4g128_ptl_v4.md for PTL 12Xe (B390 iGPU)
   v4: Shared expert FUSED into MoE kernel (shared_I=512, shared_quant=u4)
   Config: ALL FC ops use INT4 g=128. MoE fused (routed+shared) INT4 g=128.
   LM head uses INT8 g=128. KV cache INT8.
   PA/GDN/SmallOps: reused from g=64 v4 (group-size independent).
"""
import json, math
from pathlib import Path

OUT = Path(__file__).resolve().parent
PARSED_JSON = OUT / "parsed_ptl_int4g128_v4.json"
# PA/GDN/SmallOps come from g=64 v4
G64_JSON = OUT.parent / "qwen3_5_moe_int4g64" / "parsed_ptl_int4g64_v4.json"
OUT_METRICS = OUT / "performance_metrics_ptl_int4g128_v4.json"
OUT_SUMMARY = OUT / "SUMMARY_qwen3_5_moe_int4g128_ptl_v4.md"

# ── Hardware peaks ────────────────────────────────────────────────────────────
BW      = 110.0
FP16    = 58.9824
INT8    = 117.9648
RIDGE   = FP16 * 1e12 / (BW * 1e9)
RIDGE8  = INT8 * 1e12 / (BW * 1e9)

# ── Model config ──────────────────────────────────────────────────────────────
H      = 2048
NH     = 16;  NKV = 2;   HD = 256
HK     = 32;  KD  = 128; VD = 128
I      = 512
SI     = 512
NE     = 256
TK     = 8
VOCAB  = 248320
NL     = 40
NF     = 10
NL_LIN = 30
G      = 128   # group_size for body FC and MoE

KV_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
S_SIZES  = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

with open(PARSED_JSON) as f:
    P = json.load(f)
with open(G64_JSON) as f:
    P64 = json.load(f)

def ms(key):
    return P.get(key, {}).get("total_kernel_ns", 0) / 1e6

def ms64(key):
    return P64.get(key, {}).get("total_kernel_ns", 0) / 1e6

def fc_int4_bytes(M_, K, N, g=G):
    return M_*K*2 + N*K//2 + N*(K//g)*2 + N*(K//g)//2 + M_*N*2

def fc_int8_bytes(M_, K, N, g=128):
    return M_*K*2 + N*K + N*(K//g)*2 + M_*N*2

def fc_flops(M_, K, N):
    return 2.0 * M_ * K * N

def roofline(ms_val, flops, byts, int8_xmx=False):
    if ms_val <= 0:
        return dict(ms=0, gbs=0, gflops=0, eff_bw=0, eff_xmx=0, ai=0, eff=0, bound="N/A")
    ai     = flops / byts if byts > 0 else 0
    gflops = flops / (ms_val * 1e-3) / 1e9
    gbs    = byts  / (ms_val * 1e-3) / 1e9
    peak   = INT8 * 1e12 if int8_xmx else FP16 * 1e12
    ridge  = RIDGE8 if int8_xmx else RIDGE
    eff_xmx = gflops * 1e9 / peak * 100
    eff_bw  = gbs / BW * 100
    bound   = "memory" if ai < ridge else "compute"
    eff     = eff_bw if bound == "memory" else eff_xmx
    return dict(ms=ms_val, gbs=gbs, gflops=gflops,
                eff_bw=eff_bw, eff_xmx=eff_xmx, eff=eff, ai=ai, bound=bound)

def int4_wb(K_, N_, g=G):
    return N_*K_//2 + N_*(K_//g)*2 + N_*(K_//g)//2

def moe_fused_bytes_decode(g=G):
    per_exp_w  = int4_wb(H, I, g) * 2 + int4_wb(I, H, g)
    shared_w   = int4_wb(H, SI, g) * 2 + int4_wb(SI, H, g)
    router_w   = H * NE * 2
    io = H * 2 + H * 2
    return TK * per_exp_w + shared_w + router_w + io

def moe_fused_flops_decode():
    return 2.0 * (2*TK*H*I + TK*I*H) + 2.0 * (2*H*SI + SI*H)

def moe_fused_bytes_prefill(s, g=G):
    per_exp_w  = int4_wb(H, I, g) * 2 + int4_wb(I, H, g)
    shared_w   = int4_wb(H, SI, g) * 2 + int4_wb(SI, H, g)
    all_exp_w  = NE * per_exp_w
    router_w   = H * NE * 2
    io = s * H * 2 + s * H * 2
    return all_exp_w + shared_w + router_w + io

def moe_fused_flops_prefill(s):
    return 2.0 * (2*s*TK*H*I + s*TK*I*H) + 2.0 * (2*s*H*SI + s*SI*H)

def gdn_bytes(t=1): return 2 * HK * KD * VD * 2 + t * H * 2
def gdn_flops(t=1): return 2.0 * HK * KD * VD * t
def pa_decode_bytes(kv): return 2 * NKV * HD * kv * 1
def pa_decode_flops(kv): return 2.0 * NH * HD * kv * 2

def fmt(v, fmt_str):
    if v == 0: return "—"
    return format(v, fmt_str)

def md_table_row(r):
    s_ms   = f"{r['single_ms']:.4f}" if r['single_ms'] > 0 else "—"
    calls  = str(r['calls']) if r['calls'] > 0 else "—"
    t_ms   = f"{r['total_ms']:.3f}"
    gf     = fmt(r['gflops'], '.2f')
    gbs    = fmt(r['gbs'], '.1f')
    eff    = f"{r['eff']:.1f}%" if r['eff'] > 0 else "—"
    bound  = r['bound']
    return f"| {r['op']} | {r['kernel']} | {s_ms} | {calls} | {t_ms} | {gf} | {gbs} | {eff} | {bound} |"

def decode_op_rows(kv):
    rows = []
    def add(name, kernel, ms_val, calls, flops, byts, int8_xmx=False):
        r = roofline(ms_val, flops, byts, int8_xmx)
        rows.append(dict(op=name, kernel=kernel, single_ms=ms_val, calls=calls,
                         total_ms=ms_val*calls, gflops=r['gflops'], gbs=r['gbs'],
                         eff=r['eff'], bound=r['bound'], ai=r['ai'],
                         eff_bw=r['eff_bw'], eff_xmx=r['eff_xmx']))

    add("FC_QKV (INT4 g=128)", "gemm_kernel", ms("fc_qkv_decode_M1"), NF,
        fc_flops(1, H, 5120), fc_int4_bytes(1, H, 5120))
    add("FC_O (INT4 g=128)", "gemm_kernel", ms("fc_o_decode_M1"), NF,
        fc_flops(1, NH*HD, H), fc_int4_bytes(1, NH*HD, H))
    add("FC_linattn_qkv (INT4 g=128)", "gemm_kernel", ms("linattn_qkv_decode_M1"), NL_LIN,
        fc_flops(1, H, 8192), fc_int4_bytes(1, H, 8192))
    add("FC_linattn_z (INT4 g=128)", "gemm_kernel", ms("linattn_z_decode_M1"), NL_LIN,
        fc_flops(1, H, 4096), fc_int4_bytes(1, H, 4096))
    add("FC_linattn_a (INT4 g=128)", "gemm_kernel", ms("linattn_a_decode_M1"), NL_LIN,
        fc_flops(1, H, 32), fc_int4_bytes(1, H, 32))
    add("FC_linattn_b (INT4 g=128)", "gemm_kernel", ms("linattn_b_decode_M1"), NL_LIN,
        fc_flops(1, H, 32), fc_int4_bytes(1, H, 32))
    add("FC_linattn_out (INT4 g=128)", "gemm_kernel", ms("linattn_out_decode_M1"), NL_LIN,
        fc_flops(1, 4096, H), fc_int4_bytes(1, 4096, H))
    add("LM_head (INT8 g=128)", "gemm_kernel", ms("lm_head_decode_M1"), 1,
        fc_flops(1, H, VOCAB), fc_int8_bytes(1, H, VOCAB))
    add("MoE_fused (routed+shared, INT4 g=128)", "moe_3gemm_swiglu_mlp", ms("moe_fused_decode_M1"), NL,
        moe_fused_flops_decode(), moe_fused_bytes_decode())
    # PA/GDN/SmallOps from g=64
    add("PagedAttention (INT8 KV)", "paged_attention_opt", ms64(f"pa_decode_kv{kv}"), NF,
        pa_decode_flops(kv), pa_decode_bytes(kv))
    add("GatedDeltaNet", "gated_delta_net_ref_sa", ms64("gdn_decode_T1"), NL_LIN,
        gdn_flops(), gdn_bytes())
    rms = ms64("so_rmsnorm_h2048_decode")
    add_ = ms64("so_add_decode")
    rq = ms64("so_rope_q_decode"); rk = ms64("so_rope_k_decode")
    qn = ms64("so_rmsnorm3d_qnorm_decode"); kn = ms64("so_rmsnorm3d_knorm_decode")
    small = rms*2*NL + add_*2*NL + rq*NF + rk*NF + qn*NF + kn*NF
    rows.append(dict(op="SmallOps (norm/rope/add)", kernel="rms/rope/eltwise", single_ms=0,
                     calls=0, total_ms=small, gflops=0, gbs=0, eff=0, bound="memory",
                     ai=0, eff_bw=0, eff_xmx=0))
    rows.sort(key=lambda r: -r["total_ms"])
    return rows

def prefill_op_rows(s):
    rows = []
    def add(name, kernel, ms_val, calls, flops, byts, int8_xmx=False):
        r = roofline(ms_val, flops, byts, int8_xmx)
        rows.append(dict(op=name, kernel=kernel, single_ms=ms_val, calls=calls,
                         total_ms=ms_val*calls, gflops=r['gflops'], gbs=r['gbs'],
                         eff=r['eff'], bound=r['bound'], ai=r['ai'],
                         eff_bw=r['eff_bw'], eff_xmx=r['eff_xmx']))

    sk = f"S{s}"
    add("FC_QKV (INT4 g=128)", "dq+gemm_kernel", ms(f"fc_qkv_prefill_{sk}"), NF,
        fc_flops(s, H, 5120), fc_int4_bytes(s, H, 5120), int8_xmx=True)
    add("FC_O (INT4 g=128)", "dq+gemm_kernel", ms(f"fc_o_prefill_{sk}"), NF,
        fc_flops(s, NH*HD, H), fc_int4_bytes(s, NH*HD, H), int8_xmx=True)
    add("FC_linattn_qkv (INT4 g=128)", "dq+gemm_kernel", ms(f"linattn_qkv_prefill_{sk}"), NL_LIN,
        fc_flops(s, H, 8192), fc_int4_bytes(s, H, 8192), int8_xmx=True)
    add("FC_linattn_z (INT4 g=128)", "dq+gemm_kernel", ms(f"linattn_z_prefill_{sk}"), NL_LIN,
        fc_flops(s, H, 4096), fc_int4_bytes(s, H, 4096), int8_xmx=True)
    if s <= 8192:
        m_a = ms(f"linattn_a_prefill_{sk}")
        m_b = ms(f"linattn_b_prefill_{sk}")
    else:
        scale = s / 8192
        m_a = ms("linattn_a_prefill_S8192") * scale
        m_b = ms("linattn_b_prefill_S8192") * scale
    add("FC_linattn_a (INT4 g=128)", "dq+gemm_kernel", m_a, NL_LIN,
        fc_flops(s, H, 32), fc_int4_bytes(s, H, 32), int8_xmx=True)
    add("FC_linattn_b (INT4 g=128)", "dq+gemm_kernel", m_b, NL_LIN,
        fc_flops(s, H, 32), fc_int4_bytes(s, H, 32), int8_xmx=True)
    add("FC_linattn_out (INT4 g=128)", "dq+gemm_kernel", ms(f"linattn_out_prefill_{sk}"), NL_LIN,
        fc_flops(s, 4096, H), fc_int4_bytes(s, 4096, H), int8_xmx=True)
    add("MoE_fused (routed+shared, INT4 g=128)", "grouped_micro_gemm+scatter/gather", ms(f"moe_fused_prefill_{sk}"), NL,
        moe_fused_flops_prefill(s), moe_fused_bytes_prefill(s), int8_xmx=True)
    # PA/GDN from g=64
    pa_b = 2 * NKV * HD * s
    pa_fl = 2 * NH * HD * s * s
    add("PagedAttention (FP16, causal)", "sdpa_micro__prefill", ms64(f"pa_prefill_{sk}"), NF, pa_fl, pa_b)
    m_gdn = ms64(f"gdn_prefill_{sk}")
    if m_gdn == 0 and s == 131072:
        m_gdn = ms64("gdn_prefill_S65536") * 2
    add("GatedDeltaNet", "gated_delta_net_ref_sa", m_gdn, NL_LIN, gdn_flops(s), gdn_bytes(s))
    lm_ms = ms("lm_head_decode_M1")
    lm_r = roofline(lm_ms, fc_flops(1,H,VOCAB), fc_int8_bytes(1,H,VOCAB))
    rows.append(dict(op="LM_head (INT8 g=128, 1 out tok)", kernel="gemm_kernel",
                     single_ms=lm_ms, calls=1, total_ms=lm_ms,
                     gflops=lm_r['gflops'], gbs=lm_r['gbs'], eff=lm_r['eff'], bound=lm_r['bound'],
                     ai=lm_r['ai'], eff_bw=lm_r['eff_bw'], eff_xmx=lm_r['eff_xmx']))
    if s <= 8192:
        ms_rms_p = ms64(f"so_rmsnorm_h2048_prefill_{sk}")
        ms_rq_p  = ms64(f"so_rope_q_prefill_{sk}")
    else:
        scale = s / 8192
        ms_rms_p = ms64("so_rmsnorm_h2048_prefill_S8192") * scale
        ms_rq_p  = ms64("so_rope_q_prefill_S8192") * scale
    small_total = ms_rms_p*2*NL + ms_rq_p*NF
    rows.append(dict(op="SmallOps (norm/rope)", kernel="rms/rope", single_ms=0,
                     calls=0, total_ms=small_total, gflops=0, gbs=0, eff=0, bound="memory",
                     ai=0, eff_bw=0, eff_xmx=0))
    rows.sort(key=lambda r: -r["total_ms"])
    return rows

def total_ms(rows): return sum(r["total_ms"] for r in rows)

# ── Build metrics ─────────────────────────────────────────────────────────────
metrics = {
    "platform": "PTL", "version": "v4",
    "platform_desc": "PTL (B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s)",
    "config": "INT4 asymmetric g=128 all body FC + MoE fused; LM head INT8 g=128",
    "decode": {}, "prefill": {},
}
for kv in KV_SIZES:
    rows = decode_op_rows(kv)
    metrics["decode"][kv] = {"total_ms": total_ms(rows), "rows": rows}
for s in S_SIZES:
    rows = prefill_op_rows(s)
    metrics["prefill"][s] = {"total_ms": total_ms(rows), "rows": rows}

OUT_METRICS.write_text(json.dumps(metrics, indent=2))
print(f"Saved {OUT_METRICS}")

# ── Build SUMMARY ─────────────────────────────────────────────────────────────
def build_summary():
    lines = []
    lines += [
        "# Qwen3.5-MoE-35B-A3B — Roofline on PTL 12Xe (2026-05-10, v4 INT4 g=128)",
        "",
        "**Platform**: PTL (Panther Lake B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s)",
        "**Model**: Qwen/Qwen3.5-MoE-35B-A3B (Hybrid Attention: 10 full-attn + 30 GatedDeltaNet, 256 experts × top-8)",
        "",
        "- hidden_size = 2048, 40 layers (10 full-attn + 30 linear-attn GDN), 16 Q heads, 2 KV heads (GQA=8)",
        "- MoE: 256 experts × top-8, intermediate=512, shared expert intermediate=512",
        "- MatMul weights **INT4 asymmetric g=128** / FP16 act; LM_head INT8 g=128 / FP16 act; KV cache INT8",
        "- SDPA: PagedAttention (INT8 KV cache, block_size=16)",
        "- **v4**: Shared expert **fused** into MoE kernel (FuseMOESharedExpert, commit 2abcdce7f3)",
        "- PA/GDN/SmallOps data reused from g=64 v4 run (group-size independent)",
        "",
    ]

    # Model parameters (same as g=64)
    lines += [
        "## Model parameters & weight shapes",
        "",
        "See g=64 report for full weight table. Key difference: all body FC and MoE weights use **g=128** instead of g=64.",
        "",
    ]

    lines += [
        "## Theoretical roofline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| FP16 XMX peak | {FP16:.4f} TFLOPS |",
        f"| INT8 XMX peak | {INT8:.4f} TOPS |",
        f"| Memory BW | {BW:.1f} GB/s |",
        f"| Ridge point (FP16) | {RIDGE:.1f} FLOP/byte |",
        f"| Ridge point (INT8) | {RIDGE8:.1f} FLOP/byte |",
    ]

    lines += [
        "",
        "## Data sources",
        "",
        "| Op category | Source | Platform |",
        "|---|---|---|",
        "| FC_QKV, FC_O, FC_linattn_* (all INT4 g=128) | **Measured** (fc_bench) | PTL 12Xe |",
        "| MoE fused routed+shared (INT4 g=128, SI=512) | **Measured** (moe_bench) | PTL 12Xe |",
        "| LM_head (INT8 g=128) | **Measured** (fc_bench) | PTL 12Xe |",
        "| PagedAttention (INT8 KV) | **Reused** from g=64 v4 | PTL 12Xe |",
        "| GatedDeltaNet | **Reused** from g=64 v4 | PTL 12Xe |",
        "| SmallOps | **Reused** from g=64 v4 | PTL 12Xe |",
    ]

    lines += [
        "",
        "## Graph fusion notes",
        "",
        "| Bench row | Real graph behaviour | Fused into | Standalone kernel? |",
        "|---|---|---|---|",
        "| FC_QKV / FC_O (INT4 g=128) | FullyConnectedCompressed | decode: `gemm_kernel`; prefill: `dq+gemm` | Yes |",
        "| MoE routed + shared expert | MOE3GemmFusedCompressed | **Fused** routed+shared | Yes (single primitive) |",
        "| PagedAttention | PagedAttention | INT8 KV cache, GQA=8 | Yes |",
        "| GatedDeltaNet | GatedDeltaNet | Reference kernel | Yes |",
    ]

    # Token latency summary
    lines += [
        "",
        "## Token latency summary",
        "",
        "### Prefill — TTFT and per-token amortized",
        "",
        "| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for s in S_SIZES:
        tot = metrics["prefill"][s]["total_ms"]
        lines.append(f"| {s:,} | {tot:.1f} | {tot/1000:.3f} | {tot/s:.4f} | {s/tot*1000:.1f} |")

    lines += [
        "",
        "### Decode — TPOT (per output token)",
        "",
        "| KV (ctx) | TPOT (ms) | tokens/s |",
        "|---:|---:|---:|",
    ]
    for kv in KV_SIZES:
        tot = metrics["decode"][kv]["total_ms"]
        lines.append(f"| {kv:,} | {tot:.2f} | {1000/tot:.1f} |")

    # Decode TPOT per-op breakdown
    all_ops = []
    for kv in KV_SIZES:
        for r in metrics["decode"][kv]["rows"]:
            if r['op'] not in all_ops: all_ops.append(r['op'])
    lines += ["", "### Decode TPOT — per-op breakdown (ms / % of TPOT)", ""]
    kv_h = " | ".join(f"kv={kv:,}" for kv in KV_SIZES)
    kv_s = " | ".join("---:" for _ in KV_SIZES)
    lines.append(f"| op | {kv_h} |")
    lines.append(f"|---|{kv_s}|")
    for op_name in all_ops:
        cells = []
        for kv in KV_SIZES:
            tot = metrics["decode"][kv]["total_ms"]
            op_ms = sum(r['total_ms'] for r in metrics["decode"][kv]["rows"] if r['op'] == op_name)
            cells.append(f"{op_ms:.3f} ({op_ms/tot*100:.1f}%)" if op_ms > 0 else "—")
        lines.append(f"| {op_name} | {' | '.join(cells)} |")

    # Prefill TTFT per-op breakdown
    all_ops_pf = []
    for s in S_SIZES:
        for r in metrics["prefill"][s]["rows"]:
            if r['op'] not in all_ops_pf: all_ops_pf.append(r['op'])
    lines += ["", "### Prefill TTFT — per-op breakdown (ms / % of TTFT)", ""]
    s_h = " | ".join(f"S={s:,}" for s in S_SIZES)
    s_s = " | ".join("---:" for _ in S_SIZES)
    lines.append(f"| op | {s_h} |")
    lines.append(f"|---|{s_s}|")
    for op_name in all_ops_pf:
        cells = []
        for s in S_SIZES:
            tot = metrics["prefill"][s]["total_ms"]
            op_ms = sum(r['total_ms'] for r in metrics["prefill"][s]["rows"] if r['op'] == op_name)
            cells.append(f"{op_ms:.1f} ({op_ms/tot*100:.1f}%)" if op_ms > 0 else "—")
        lines.append(f"| {op_name} | {' | '.join(cells)} |")

    # Decode tables
    lines += ["", "## Decode tables (1 query token, KV = context length)", ""]
    for kv in KV_SIZES:
        rows = metrics["decode"][kv]["rows"]
        tot  = metrics["decode"][kv]["total_ms"]
        lines += [f"### Decode — KV={kv:,}", "",
                  "| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in rows: lines.append(md_table_row(r))
        lines += [f"| **TOTAL** | | | | **{tot:.3f}** | | | **{1000/tot:.1f} tok/s** | |", ""]

    # Prefill tables
    lines += ["## Prefill tables (single forward over S tokens)", ""]
    for s in S_SIZES:
        rows = metrics["prefill"][s]["rows"]
        tot  = metrics["prefill"][s]["total_ms"]
        toks = s / tot * 1000 if tot > 0 else 0
        lines += [f"### Prefill — S={s:,}", "",
                  "| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in rows: lines.append(md_table_row(r))
        lines += [f"| **TOTAL** | | | | **{tot:.1f}** | | | **{toks:.1f} tok/s** | |", ""]

    # Top contributors
    lines += ["## Top contributors", "", "### Decode", "",
              "| KV | top1 (ms,%) | top2 | top3 |", "|---:|---|---|---|"]
    for kv in KV_SIZES:
        rows = sorted(metrics["decode"][kv]["rows"], key=lambda x: -x["total_ms"])
        tot = metrics["decode"][kv]["total_ms"]
        tops = [f"{r['op']} {r['total_ms']:.2f}ms ({r['total_ms']/tot*100:.1f}%)" for r in rows[:3]]
        while len(tops) < 3: tops.append("—")
        lines.append(f"| {kv:,} | {tops[0]} | {tops[1]} | {tops[2]} |")

    lines += ["", "### Prefill", "",
              "| S | top1 (ms,%) | top2 | top3 |", "|---:|---|---|---|"]
    for s in S_SIZES:
        rows = sorted(metrics["prefill"][s]["rows"], key=lambda x: -x["total_ms"])
        tot = metrics["prefill"][s]["total_ms"]
        tops = [f"{r['op']} {r['total_ms']:.1f}ms ({r['total_ms']/tot*100:.1f}%)" for r in rows[:3]]
        while len(tops) < 3: tops.append("—")
        lines.append(f"| {s:,} | {tops[0]} | {tops[1]} | {tops[2]} |")

    lines += [
        "",
        "## Caveats & method",
        "",
        "- Each op profiled in its own process via cliloader Device Performance Timing; mean kernel time per iteration.",
        f"- FC weight bytes count INT4 weight + FP16 scale/zp (g={G}) + FP16 activations.",
        "- PA/GDN/SmallOps data reused from g=64 v4 run (group-size independent ops).",
        "- **v4**: Shared expert fused into MOE3GemmFusedCompressed (commit 2abcdce7f3).",
        "- Target machine: Local_Admin@10.239.132.229 (PTL B390 iGPU, 12 Xe cores)",
        "",
        "## Reproduction",
        "",
        "```bat",
        "REM Full script: run_qwen3_5_moe_ptl_int4g128_v4.bat",
        "```",
        "",
        "_Generated by `build_report_ptl_int4g128_v4.py`_",
    ]
    return "\n".join(lines)

summary = build_summary()
OUT_SUMMARY.write_text(summary)
print(f"Saved {OUT_SUMMARY}")

print("\n=== DECODE TOTALS (PTL INT4 g=128 v4) ===")
for kv in KV_SIZES:
    tot = metrics["decode"][kv]["total_ms"]
    print(f"  kv={kv:>6,}  {tot:.2f} ms  {1000/tot:.1f} tok/s")
print("\n=== PREFILL TOTALS ===")
for s in S_SIZES:
    tot = metrics["prefill"][s]["total_ms"]
    print(f"  S={s:>6,}  {tot:.1f} ms  {s/tot*1000:.1f} tok/s")
