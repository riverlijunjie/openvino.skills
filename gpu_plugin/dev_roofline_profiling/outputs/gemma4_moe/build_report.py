#!/usr/bin/env python3
"""Generate performance_metrics.json and SUMMARY_gemma4_moe_ptl_12xe.md
   for Gemma4-26B-A4B-it on PTL 12Xe (B390 iGPU).

   Architecture specifics:
   - 30 layers total: 25 sliding attention + 5 full attention (5:1 pattern)
   - Each layer has dense MLP (GEGLU) + MoE (128 experts, top-8)
   - Sliding attn: NH=16, NKV=8, HD=256, sliding_window=1024
   - Full attn: NH=16, NKV=2, HD=512, attention_k_eq_v=true (V reuses K)
   - MoE bench uses Swish proxy (GPU plugin only supports Swish fusion)
"""
import json, math, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
PARSED_JSON = OUT / "parsed.json"
OUT_METRICS = OUT / "performance_metrics.json"
OUT_SUMMARY = OUT / "SUMMARY_gemma4_moe_ptl_12xe.md"

# ── Hardware peaks ────────────────────────────────────────────────────────────
BW      = 110.0       # GB/s
FP16    = 58.9824     # TFLOPS  (12 Xe × 8 EU × 256 FP/cyc × 2.4 GHz)
INT8    = 117.9648    # TOPS    (2× FP16)
RIDGE   = FP16 * 1e12 / (BW * 1e9)   # ≈ 536.2 FLOP/byte
RIDGE8  = INT8 * 1e12 / (BW * 1e9)   # ≈ 1072.4 FLOP/byte

# ── Model config ──────────────────────────────────────────────────────────────
H      = 2816    # hidden size
NL     = 30      # total layers
NL_S   = 25      # sliding attention layers
NL_F   = 5       # full attention layers

# Sliding attention
NH_S   = 16;  NKV_S = 8;   HD_S = 256   # Q=4096, KV=2048
QKV_S_N = 8192   # Q(4096)+K(2048)+V(2048)
O_S_K  = NH_S * HD_S  # 4096

# Full attention
NH_F   = 16;  NKV_F = 2;   HD_F = 512   # Q=8192, K=1024, V=K (no V proj)
QK_F_N = 9216    # Q(8192)+K(1024), no V proj
O_F_K  = NH_F * HD_F  # 8192

# Dense MLP (GEGLU)
I_DENSE = 2112   # intermediate_size for dense MLP

# MoE
I_MOE  = 704     # moe_intermediate_size
NE     = 128     # num_experts
TK     = 8       # top_k_experts

VOCAB  = 262144
SW     = 1024    # sliding_window

KV_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
S_SIZES  = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# ── Load parsed logs ──────────────────────────────────────────────────────────
if not PARSED_JSON.exists():
    print(f"ERROR: {PARSED_JSON} not found. Run parse_logs.py first.")
    sys.exit(1)

with open(PARSED_JSON) as f:
    P = json.load(f)

def ms(key):
    """Get total kernel time in ms for a benchmark log."""
    return P.get(key, {}).get("total_kernel_ns", 0) / 1e6

# ── Roofline helpers ──────────────────────────────────────────────────────────
def fc_int4_bytes(M_, K, N, g=128):
    """INT4 g=128 FC bytes: input(f16) + weight(u4) + scale(f16) + zp(u4) + output(f16)."""
    return M_*K*2 + N*K//2 + N*(K//g)*2 + N*(K//g)//2 + M_*N*2

def fc_int8_bytes(M_, K, N, g=128):
    """INT8 g=128 FC bytes: input(f16) + weight(u8) + scale(f16) + output(f16)."""
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

# ── MoE bytes/flops ──────────────────────────────────────────────────────────
def moe_decode_bytes(tk=TK, ne=NE, h=H, i=I_MOE, g=64):
    """Weight bytes for TK selected experts (gate+up+down) INT4 g=64 + router + io."""
    # Per expert: gate(h→i) + up(h→i) + down(i→h) weights
    per_exp_w = (i*h//2 + (h//g)*i*2 + (h//g)*i//2) * 2 + \
                (h*i//2 + (i//g)*h*2 + (i//g)*h//2)
    router_w = ne*h//2 + (h//g)*ne*2 + (h//g)*ne//2
    io = h*2 + h*2  # input + output
    return tk * per_exp_w + router_w + io

def moe_decode_flops(tk=TK, h=H, i=I_MOE):
    return 2.0 * (2*tk*h*i + tk*i*h)  # gate+up: 2×(tk×h×i), down: tk×i×h

def moe_prefill_bytes(s, tk=TK, ne=NE, h=H, i=I_MOE, g=64):
    """Prefill: all NE expert weights read (not just TK) + io."""
    by_gate = s*h*2 + ne*h*i//2 + ne*(h//g)*i*2 + ne*(h//g)*i//2
    by_up   = s*h*2 + ne*h*i//2 + ne*(h//g)*i*2 + ne*(h//g)*i//2
    by_down = s*i*2 + ne*i*h//2 + ne*(i//g)*h*2 + ne*(i//g)*h//2
    by_router = s*h*2 + ne*h//2 + (h//g)*ne*2 + s*ne*2
    by_out  = s*h*2
    return by_gate + by_up + by_down + by_router + by_out

def moe_prefill_flops(s, tk=TK, h=H, i=I_MOE):
    return 2.0 * (2*s*tk*h*i + s*tk*i*h)

# ── PA bytes/flops ────────────────────────────────────────────────────────────
def pa_decode_sliding_bytes():
    """Sliding PA decode: KV capped at SW=1024, INT8 KV cache."""
    return 2 * NKV_S * HD_S * SW * 1  # INT8 KV read

def pa_decode_sliding_flops():
    return 2.0 * NH_S * HD_S * SW * 2  # QK + QV

def pa_decode_full_bytes(kv):
    return 2 * NKV_F * HD_F * kv * 1  # INT8 KV read

def pa_decode_full_flops(kv):
    return 2.0 * NH_F * HD_F * kv * 2

def pa_prefill_sliding_flops(s):
    """Sliding prefill: causal + sliding window. For S<=SW: O(S²), for S>SW: ~S*SW."""
    if s <= SW:
        return 2.0 * NH_S * HD_S * s * s
    else:
        return 2.0 * NH_S * HD_S * s * SW

def pa_prefill_full_flops(s):
    """Full prefill: O(S²) causal attention."""
    return 2.0 * NH_F * HD_F * s * s

# ── Dense MLP bytes/flops ────────────────────────────────────────────────────
def dense_decode_bytes():
    """Gate + up + down for dense MLP, INT4 g=128/g=64, M=1."""
    return (fc_int4_bytes(1, H, I_DENSE) +       # gate: K=2816 g=128 OK
            fc_int4_bytes(1, H, I_DENSE) +       # up:   K=2816 g=128 OK
            fc_int4_bytes(1, I_DENSE, H, g=64))  # down: K=2112 g=64 (2112%128!=0)

def dense_decode_flops():
    return (fc_flops(1, H, I_DENSE) * 2 +  # gate + up
            fc_flops(1, I_DENSE, H))         # down

# ── Decode per-op table ──────────────────────────────────────────────────────
def decode_op_rows(kv):
    rows = []
    def add(name, kernel, ms_val, calls, flops, byts, int8_xmx=False):
        r = roofline(ms_val, flops, byts, int8_xmx)
        rows.append(dict(op=name, kernel=kernel,
                         single_ms=ms_val, calls=calls,
                         total_ms=ms_val*calls,
                         gflops=r['gflops'], gbs=r['gbs'],
                         eff=r['eff'], bound=r['bound']))

    # FC — sliding attention (×25)
    m = ms("fc_qkv_sliding_decode_M1")
    add("FC_QKV_sliding (INT4)", "gemm_kernel", m, NL_S,
        fc_flops(1, H, QKV_S_N), fc_int4_bytes(1, H, QKV_S_N))
    m = ms("fc_o_sliding_decode_M1")
    add("FC_O_sliding (INT4)", "gemm_kernel", m, NL_S,
        fc_flops(1, O_S_K, H), fc_int4_bytes(1, O_S_K, H))

    # FC — full attention (×5)
    m = ms("fc_qk_full_decode_M1")
    add("FC_QK_full (INT4)", "gemm_kernel", m, NL_F,
        fc_flops(1, H, QK_F_N), fc_int4_bytes(1, H, QK_F_N))
    m = ms("fc_o_full_decode_M1")
    add("FC_O_full (INT4)", "gemm_kernel", m, NL_F,
        fc_flops(1, O_F_K, H), fc_int4_bytes(1, O_F_K, H))

    # Dense MLP (×30) — gate/up/down combined
    mg = ms("fc_gate_dense_decode_M1")
    mu = ms("fc_up_dense_decode_M1")
    md = ms("fc_down_dense_decode_M1")
    dense_ms = mg + mu + md
    add("DenseMLP_gate+up+down (INT4)", "gemm_kernel×3", dense_ms, NL,
        dense_decode_flops(), dense_decode_bytes())

    # MoE fused (×30)
    m = ms("moe_decode_M1")
    add("MoE_fused (INT4, TK=8/128)", "moe_3gemm+scatter+gather", m, NL,
        moe_decode_flops(), moe_decode_bytes())

    # LM head (×1)
    m = ms("lm_head_decode_M1")
    add("LM_head (INT8)", "gemm_kernel", m, 1,
        fc_flops(1, H, VOCAB), fc_int8_bytes(1, H, VOCAB))

    # PA sliding (×25) — KV capped at SW=1024
    m = ms("pa_sliding_decode_kv1024")
    add("PA_sliding (INT8 KV, sw=1024)", "paged_attn_single_token", m, NL_S,
        pa_decode_sliding_flops(), pa_decode_sliding_bytes())

    # PA full (×5) — full KV context
    m = ms(f"pa_full_decode_kv{kv}")
    add("PA_full (INT8 KV)", "paged_attn_single_token", m, NL_F,
        pa_decode_full_flops(kv), pa_decode_full_bytes(kv))

    # Small ops (aggregate)
    ms_rms = ms("so_rmsnorm_h2816_decode")
    ms_add = ms("so_add_h2816_decode")
    ms_rq_s = ms("so_rope_q_sliding_decode")
    ms_rk_s = ms("so_rope_k_sliding_decode")
    ms_qn_s = ms("so_rmsnorm3d_q_sliding_decode")
    ms_kn_s = ms("so_rmsnorm3d_k_sliding_decode")
    ms_rq_f = ms("so_rope_q_full_decode")
    ms_rk_f = ms("so_rope_k_full_decode")
    ms_qn_f = ms("so_rmsnorm3d_q_full_decode")
    ms_kn_f = ms("so_rmsnorm3d_k_full_decode")
    # 4 RMSNorm per layer × 30 + 1 final = 121 total
    # rope_q + rope_k per sliding layer (25) + full layer (5)
    # rmsnorm3d_q + rmsnorm3d_k per sliding (25) + full (5)
    # add: 2 per layer × 30
    small_total = (ms_rms * (4*NL + 1) +
                   ms_add * 2*NL +
                   (ms_rq_s + ms_rk_s) * NL_S +
                   (ms_qn_s + ms_kn_s) * NL_S +
                   (ms_rq_f + ms_rk_f) * NL_F +
                   (ms_qn_f + ms_kn_f) * NL_F)
    rows.append(dict(op="SmallOps (norm/rope/add)", kernel="rms/rope/eltwise",
                     single_ms=0, calls=0, total_ms=small_total,
                     gflops=0, gbs=0, eff=0, bound="memory"))

    rows.sort(key=lambda r: -r["total_ms"])
    return rows

# ── Prefill per-op table ─────────────────────────────────────────────────────
def prefill_op_rows(s):
    rows = []
    def add(name, kernel, ms_val, calls, flops, byts, int8_xmx=False):
        r = roofline(ms_val, flops, byts, int8_xmx)
        rows.append(dict(op=name, kernel=kernel,
                         single_ms=ms_val, calls=calls,
                         total_ms=ms_val*calls,
                         gflops=r['gflops'], gbs=r['gbs'],
                         eff=r['eff'], bound=r['bound']))

    sk = f"S{s}"

    # FC — sliding attention (×25), prefill uses INT8 XMX for INT4 weights
    m = ms(f"fc_qkv_sliding_prefill_{sk}")
    add("FC_QKV_sliding (INT4→INT8 XMX)", "dq+gemm_kernel", m, NL_S,
        fc_flops(s, H, QKV_S_N), fc_int4_bytes(s, H, QKV_S_N), int8_xmx=True)
    m = ms(f"fc_o_sliding_prefill_{sk}")
    add("FC_O_sliding (INT4→INT8 XMX)", "dq+gemm_kernel", m, NL_S,
        fc_flops(s, O_S_K, H), fc_int4_bytes(s, O_S_K, H), int8_xmx=True)

    # FC — full attention (×5)
    m = ms(f"fc_qk_full_prefill_{sk}")
    add("FC_QK_full (INT4→INT8 XMX)", "dq+gemm_kernel", m, NL_F,
        fc_flops(s, H, QK_F_N), fc_int4_bytes(s, H, QK_F_N), int8_xmx=True)
    m = ms(f"fc_o_full_prefill_{sk}")
    add("FC_O_full (INT4→INT8 XMX)", "dq+gemm_kernel", m, NL_F,
        fc_flops(s, O_F_K, H), fc_int4_bytes(s, O_F_K, H), int8_xmx=True)

    # Dense MLP (×30) — gate/up/down combined
    mg = ms(f"fc_gate_dense_prefill_{sk}")
    mu = ms(f"fc_up_dense_prefill_{sk}")
    md_val = ms(f"fc_down_dense_prefill_{sk}")
    dense_ms = mg + mu + md_val
    add("DenseMLP_gate+up+down (INT4→INT8 XMX)", "dq+gemm_kernel×3", dense_ms, NL,
        (fc_flops(s, H, I_DENSE)*2 + fc_flops(s, I_DENSE, H)),
        (fc_int4_bytes(s, H, I_DENSE)*2 + fc_int4_bytes(s, I_DENSE, H, g=64)),
        int8_xmx=True)

    # MoE fused (×30)
    m = ms(f"moe_prefill_{sk}")
    add("MoE_fused (INT4, TK=8/128)", "grouped_micro_gemm×3+scatter+gather", m, NL,
        moe_prefill_flops(s), moe_prefill_bytes(s), int8_xmx=True)

    # PA sliding prefill (×25)
    # We only benchmarked pa_sliding_prefill_S1024; for S>1024, scale linearly
    m_pa_s = ms("pa_sliding_prefill_S1024")
    if s <= SW:
        pa_s_ms = m_pa_s * (s / 1024.0) ** 2 if m_pa_s > 0 else 0
    else:
        pa_s_ms = m_pa_s * (s / 1024.0) if m_pa_s > 0 else 0
    add("PA_sliding (FP16 prefill, sw=1024)", "sdpa_micro_prefill", pa_s_ms, NL_S,
        pa_prefill_sliding_flops(s), 2 * NKV_S * HD_S * min(s, SW) * 2)  # approx bytes

    # PA full prefill (×5) — O(S²) attention
    m = ms(f"pa_full_prefill_{sk}")
    pa_f_bytes = 2 * NKV_F * HD_F * s * 2  # simplified: KV cache write+read
    add("PA_full (FP16 prefill, causal)", "sdpa_micro_prefill", m, NL_F,
        pa_prefill_full_flops(s), pa_f_bytes)

    # LM head (1 output token, same as decode)
    lm_ms_val = ms("lm_head_decode_M1")
    r_lm = roofline(lm_ms_val, fc_flops(1, H, VOCAB), fc_int8_bytes(1, H, VOCAB))
    rows.append(dict(op="LM_head (INT8, 1 out tok)", kernel="gemm_kernel",
                     single_ms=lm_ms_val, calls=1, total_ms=lm_ms_val,
                     gflops=r_lm['gflops'], gbs=r_lm['gbs'],
                     eff=r_lm['eff'], bound="memory"))

    # Small ops — prefill: scale from measured sizes
    if s <= 8192:
        ms_rms_p = ms(f"so_rmsnorm_h2816_prefill_{sk}")
        ms_rq_p  = ms(f"so_rope_q_sliding_prefill_{sk}")
        ms_add_p = ms(f"so_add_h2816_prefill_{sk}")
    else:
        scale = s / 8192
        ms_rms_p = ms("so_rmsnorm_h2816_prefill_S8192") * scale
        ms_rq_p  = ms("so_rope_q_sliding_prefill_S8192") * scale
        ms_add_p = ms("so_add_h2816_prefill_S8192") * scale
    small_total = ms_rms_p * (4*NL + 1) + ms_add_p * 2*NL + ms_rq_p * NL
    rows.append(dict(op="SmallOps (norm/rope/add)", kernel="rms/rope/eltwise",
                     single_ms=0, calls=0, total_ms=small_total,
                     gflops=0, gbs=0, eff=0, bound="memory"))

    rows.sort(key=lambda r: -r["total_ms"])
    return rows


def total_ms(rows):
    return sum(r["total_ms"] for r in rows)


# ── Build performance_metrics JSON ────────────────────────────────────────────
metrics = {
    "platform": "PTL_12Xe",
    "platform_desc": "PTL (B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s, FP16 XMX 58.982 TFLOPS)",
    "bw": BW, "fp16_tflops": FP16, "int8_tops": INT8, "ridge_f16": RIDGE,
    "model": "google/gemma-4-26B-A4B-it",
    "config_summary": "INT4 g=128 body (MoE g=64) + INT8 g=128 LM_head + INT8 KV cache",
    "decode": {},
    "prefill": {},
}
for kv in KV_SIZES:
    rows = decode_op_rows(kv)
    metrics["decode"][kv] = {"total_ms": total_ms(rows), "rows": rows}
for s in S_SIZES:
    rows = prefill_op_rows(s)
    metrics["prefill"][s] = {"total_ms": total_ms(rows), "rows": rows}

OUT_METRICS.write_text(json.dumps(metrics, indent=2))
print(f"Saved {OUT_METRICS}")


# ── Markdown helpers ──────────────────────────────────────────────────────────
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


# ── Build SUMMARY markdown ────────────────────────────────────────────────────
def build_summary():
    lines = []

    lines += [
        "# Gemma4-26B-A4B-it — Roofline on PTL 12Xe (B390 iGPU)",
        "",
        "**Date:** 2026-05-12",
        "**Target:** PTL (Panther Lake B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s)",
        "**Config:** INT4 g=128 body (MoE g=64, I=704 not divisible by 128) + INT8 g=128 LM_head + INT8 KV cache",
        "**Profiler:** cliloader 3.0.6 `--device-performance-timing`, mean kernel time",
        "**Token sweep:** S/kv ∈ {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}",
        "**Bench:** `.github/skills/dev_roofline_profiling/utils/{fc,moe,pa,small_ops}_bench`",
        "",
        "---",
        "",
        "## 1. Hardware Peaks",
        "",
        "| Platform | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | Ridge point (F16) |",
        "|---|---:|---:|---:|---:|",
        f"| PTL (B390 iGPU, 12 Xe @ 2400 MHz) | {BW:.1f} | {FP16:.3f} | {INT8:.3f} | {RIDGE:.1f} FLOP/byte |",
        "",
        "---",
        "",
        "## 2. Model Configuration",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| `vocab_size` | {VOCAB:,} |",
        f"| `hidden_size` (H) | {H:,} |",
        f"| `num_hidden_layers` | **{NL}** (25 sliding + 5 full, 5:1 pattern) |",
        f"| Sliding attn: NH / NKV / HD | {NH_S} / {NKV_S} / {HD_S} (Q={NH_S*HD_S}, KV={NKV_S*HD_S}) |",
        f"| Full attn: NH / NKV / HD | {NH_F} / {NKV_F} / {HD_F} (Q={NH_F*HD_F}, K={NKV_F*HD_F}, V=K) |",
        f"| `sliding_window` | {SW} |",
        f"| `attention_k_eq_v` | true (full attn: V reuses K projection) |",
        f"| Dense MLP `intermediate_size` | {I_DENSE} (GEGLU: gate+up {H}→{I_DENSE}, down {I_DENSE}→{H}) |",
        f"| MoE: `moe_intermediate_size` | {I_MOE} per expert |",
        f"| MoE: `num_experts` / `top_k` | {NE} / {TK} |",
        f"| `hidden_activation` | gelu_pytorch_tanh (GEGLU) |",
        f"| `tie_word_embeddings` | true |",
        f"| `final_logit_softcapping` | 30.0 |",
        "| Body weight quant | INT4 g=128 (asymmetric) |",
        "| LM head weight quant | INT8 g=128 |",
        "| KV cache | INT8 |",
        "| Activation dtype | FP16 |",
        "",
        "> **Note**: Dense MLP and MoE run in parallel per layer. Both are profiled.",
        "> GPU plugin MoE fusion only supports Swish/SiLU — bench uses Swish as proxy for GEGLU.",
        "",
        "---",
        "",
        "## 3. Graph Fusion Notes",
        "",
        "| Op variant | GPU primitive | Fused? | Notes |",
        "|---|---|---|---|",
        "| FC_QKV/O sliding/full (INT4) | FullyConnectedCompressed | Partial | decode: gemm_kernel; prefill: dq+gemm_kernel |",
        "| Dense MLP gate/up/down (INT4) | FullyConnectedCompressed×3 | ❌ NOT fused | 3 separate FC ops (gate+up need GEGLU activation between) |",
        "| MoE routed experts (INT4) | MOE3GemmFusedCompressed | ✅ Fused | gate_up+down+scatter/gather; Swish proxy |",
        "| PagedAttention sliding | PagedAttention | ✅ Fused | INT8 KV, GQA group=2, sw=1024 |",
        "| PagedAttention full | PagedAttention | ✅ Fused | INT8 KV, GQA group=8 |",
        "| GEGLU multiply | silu(gate)·up in DenseMLP | SwiGLU primitive | Fused — bench-only |",
        "| add (residual) | eltwise | Not fused | 2× per layer |",
        "| rmsnorm | RMSNorm primitive | Not fused | 4× per layer + 1 final |",
        "",
        "---",
    ]

    # ── Decode totals ─────────────────────────────────────────────────────
    lines += [
        "",
        "## 4. Decode Performance — Totals",
        "",
        "| kv tokens | FC_attn/L (ms) | DenseMLP/L (ms) | MoE/L (ms) | PA_sliding/L (ms) | PA_full/L (ms) | LM head (ms) | SmallOps (ms) | **total ms** | **tok/s** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kv in KV_SIZES:
        d = metrics["decode"][kv]
        rows = d["rows"]
        fc_attn = sum(r["total_ms"] for r in rows if r["op"].startswith("FC_Q") or r["op"].startswith("FC_O"))
        dense = sum(r["total_ms"] for r in rows if "DenseMLP" in r["op"])
        moe = sum(r["total_ms"] for r in rows if "MoE" in r["op"])
        pa_s = sum(r["total_ms"] for r in rows if "PA_sliding" in r["op"])
        pa_f = sum(r["total_ms"] for r in rows if "PA_full" in r["op"])
        lm = sum(r["total_ms"] for r in rows if "LM_head" in r["op"])
        sm = sum(r["total_ms"] for r in rows if "SmallOps" in r["op"])
        tot = d["total_ms"]
        tps = 1000.0 / tot if tot > 0 else 0
        lines.append(
            f"| {kv:>7,} | {fc_attn/NL:.4f} | {dense/NL:.4f} | {moe/NL:.4f} | "
            f"{pa_s/NL_S:.4f} | {pa_f/NL_F:.4f} | {lm:.3f} | {sm:.3f} | "
            f"**{tot:.2f}** | **{tps:.1f}** |"
        )

    lines += ["", "---"]

    # ── Prefill totals ────────────────────────────────────────────────────
    lines += [
        "",
        "## 5. Prefill Performance — Totals (TTFT)",
        "",
        "| S tokens | FC_attn/L (ms) | DenseMLP/L (ms) | MoE/L (ms) | PA_sliding/L (ms) | PA_full/L (ms) | **TTFT (ms)** | **tok/s** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in S_SIZES:
        p = metrics["prefill"][s]
        rows = p["rows"]
        fc_attn = sum(r["total_ms"] for r in rows if r["op"].startswith("FC_Q") or r["op"].startswith("FC_O"))
        dense = sum(r["total_ms"] for r in rows if "DenseMLP" in r["op"])
        moe = sum(r["total_ms"] for r in rows if "MoE" in r["op"])
        pa_s = sum(r["total_ms"] for r in rows if "PA_sliding" in r["op"])
        pa_f = sum(r["total_ms"] for r in rows if "PA_full" in r["op"])
        tot = p["total_ms"]
        tps = s / (tot / 1000.0) if tot > 0 else 0
        lines.append(
            f"| {s:>7,} | {fc_attn/NL:.3f} | {dense/NL:.3f} | {moe/NL:.3f} | "
            f"{pa_s/NL_S:.3f} | {pa_f/NL_F:.3f} | "
            f"**{tot:.1f}** | **{tps:.0f}** |"
        )

    lines += ["", "---"]

    # ── Decode time-share at kv=4096 ──────────────────────────────────────
    ref_kv = 4096
    d_ref = metrics["decode"][ref_kv]
    tot_ref = d_ref["total_ms"]
    lines += [
        "",
        f"## 6. Decode Time-Share (kv={ref_kv:,})",
        "",
        f"**Total decode @ kv={ref_kv}: {tot_ref:.2f} ms → {1000/tot_ref:.1f} tok/s**",
        "",
        "| Group | ms | Share |",
        "|---|---:|---:|",
    ]
    for r in d_ref["rows"]:
        share = r["total_ms"] / tot_ref * 100 if tot_ref > 0 else 0
        lines.append(f"| {r['op']} (×{r['calls']}) | {r['total_ms']:.3f} | {share:.1f}% |")

    lines += ["", "---"]

    # ── Decode per-KV tables ──────────────────────────────────────────────
    lines += ["", "## 7. Decode Tables (1 query token, KV = context length)", ""]
    lines.append("_Columns: op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound_")
    lines.append("_Eff% = GB/s / 110 GB/s × 100 for memory-bound; GFLOPS / XMX_peak × 100 for compute-bound._")

    for kv in KV_SIZES:
        d = metrics["decode"][kv]
        tot = d["total_ms"]
        tps = 1000.0 / tot if tot > 0 else 0
        lines += [
            "",
            f"### Decode — KV={kv:,}",
            "",
            "| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in d["rows"]:
            lines.append(md_table_row(r))
        lines.append(f"| **TOTAL** | | | | **{tot:.3f}** | | | | |")
        lines.append(f"| | | | | | | | **{tps:.1f} tok/s** | |")

    lines += ["", "---"]

    # ── Prefill per-S tables ──────────────────────────────────────────────
    lines += ["", "## 8. Prefill Tables (single forward over S tokens)", ""]

    for s in S_SIZES:
        p = metrics["prefill"][s]
        tot = p["total_ms"]
        tps = s / (tot / 1000.0) if tot > 0 else 0
        lines += [
            "",
            f"### Prefill — S={s:,}",
            "",
            "| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in p["rows"]:
            lines.append(md_table_row(r))
        lines.append(f"| **TOTAL** | | | | **{tot:.1f}** | | | | |")
        lines.append(f"| | | | | | | | **TTFT {tot/1000:.2f}s, {tps:.0f} tok/s** | |")

    lines += [
        "",
        "---",
        "",
        "## Caveats & Method",
        "",
        "- Each op profiled in its own process via cliloader Device Performance Timing; mean kernel time per iteration.",
        "- FC weight bytes: INT4 weight + FP16 scale/zp(g=128) + FP16 act + FP16 out.",
        "- PA bytes: INT8 KV cache + FP16 Q/out.",
        "- Decode FC is **memory-bound** (weight-read dominates at M=1).",
        "- Prefill FC is **compute-bound** (INT8 XMX path via dynamic quantize).",
        "- Sliding PA decode always uses kv=1024 (sliding_window cap). Full PA grows with context.",
        "- Dense MLP and MoE run in parallel per layer — model total includes both.",
        "- MoE bench uses Swish activation as proxy for GEGLU (GPU plugin limitation).",
        "- LM head: 1 token only, both decode and prefill.",
        "- Small ops prefill for S>8192: linearly extrapolated from S=8192 measurement.",
        f"- Target: Local_Admin@10.239.132.229 (PTL 12Xe, B390 iGPU)",
    ]

    return "\n".join(lines)


summary = build_summary()
OUT_SUMMARY.write_text(summary)
print(f"Saved {OUT_SUMMARY}")
print(f"\nDecode @ kv=4096: {metrics['decode'][4096]['total_ms']:.2f} ms "
      f"→ {1000/metrics['decode'][4096]['total_ms']:.1f} tok/s")
print(f"Prefill @ S=4096: {metrics['prefill'][4096]['total_ms']:.1f} ms "
      f"→ {4096/(metrics['prefill'][4096]['total_ms']/1000):.0f} tok/s")
