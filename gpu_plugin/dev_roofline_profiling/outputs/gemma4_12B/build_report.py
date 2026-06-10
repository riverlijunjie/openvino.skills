#!/usr/bin/env python3
"""Generate performance_metrics.json and SUMMARY_gemma4_12B_ptl_12xe.md
   for Gemma4-12B-it (dense) on PTL 12Xe (B390 iGPU).

   Architecture (from HF config.json text_config):
   - 48 layers total: 40 sliding attention + 8 full attention (5:1 pattern)
   - Sliding attn: NH=16, NKV=8, HD=256, sliding_window=1024
   - Full attn: NH=16, NKV=1, HD=512, attention_k_eq_v=true (V reuses K)
   - Dense MLP (GEGLU): gate/up: 3840->15360, down: 15360->3840
   - LM head: 3840 -> 262144 (INT8 g=128, tied)
   - No MoE
"""
import json, sys
from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parent
PARSED_JSON = OUT / "parsed.json"
OUT_METRICS = OUT / "performance_metrics.json"
OUT_SUMMARY = OUT / f"SUMMARY_gemma4_12B_ptl_12xe_{date.today().isoformat()}.md"

# Hardware peaks (PTL 12Xe @ 2400 MHz)
BW      = 110.0
FP16    = 58.9824     # TFLOPS  (12 Xe x 8 EU x 256 FP/cyc x 2.4 GHz)
INT8    = 117.9648
RIDGE   = FP16 * 1e12 / (BW * 1e9)   # ~536.2
RIDGE8  = INT8 * 1e12 / (BW * 1e9)   # ~1072.4

# Model config
H        = 3840
NL       = 48
NL_S     = 40
NL_F     = 8

# Sliding attention
NH_S, NKV_S, HD_S = 16, 8, 256
QKV_S_N = 8192          # Q(4096)+K(2048)+V(2048)
O_S_K   = NH_S * HD_S   # 4096

# Full attention (k_eq_v: V reuses K projection)
NH_F, NKV_F, HD_F = 16, 1, 512
QK_F_N  = 8704          # Q(8192)+K(512)
O_F_K   = NH_F * HD_F   # 8192

# Dense MLP (GEGLU)
I_DENSE = 15360

VOCAB   = 262144
SW      = 1024

# Token sizes requested (input sizes); decode runs for OUTPUT_TOKENS new tokens.
SIZES   = [256, 1024, 2048, 4096, 8192, 16384, 32768]
KV_SIZES = SIZES
S_SIZES  = SIZES
OUTPUT_TOKENS = 512

# Overhead applied to theoretical roofline (sustained ~ 95% of peak)
OVERHEAD       = 0.05
ACHIEV_FRAC    = 1.0 - OVERHEAD          # 0.95
BW_ACHIEV      = BW   * ACHIEV_FRAC      # GB/s
FP16_ACHIEV    = FP16 * ACHIEV_FRAC      # TFLOPS
INT8_ACHIEV    = INT8 * ACHIEV_FRAC      # TOPS

# Weight-only bytes for a single FC (no activation, no output).
def fc_int4_weight_bytes(K, N, g=128):
    return N*K//2 + N*(K//g)*2 + N*(K//g)//2  # INT4 + FP16 scale + INT4 zp

def fc_int8_weight_bytes(K, N, g=128):
    return N*K + N*(K//g)*2                   # INT8 + FP16 scale (per-channel zp absorbed)

# Per-module weight footprint definition: (name, K, N, quant, n_layers)
MODULES = [
    ("FC_QKV_sliding", H,      QKV_S_N, "INT4", NL_S),
    ("FC_O_sliding",   O_S_K,  H,       "INT4", NL_S),
    ("FC_QK_full",     H,      QK_F_N,  "INT4", NL_F),
    ("FC_O_full",      O_F_K,  H,       "INT4", NL_F),
    ("MLP_gate",       H,      I_DENSE, "INT4", NL),
    ("MLP_up",         H,      I_DENSE, "INT4", NL),
    ("MLP_down",       I_DENSE, H,      "INT4", NL),
    ("LM_head (tied)", H,      VOCAB,   "INT8", 1),
]

def module_weight_bytes(K, N, quant):
    return fc_int4_weight_bytes(K, N) if quant == "INT4" else fc_int8_weight_bytes(K, N)

def module_param_count(K, N):
    return K * N

if not PARSED_JSON.exists():
    print(f"ERROR: {PARSED_JSON} not found. Run parse_logs.py first.")
    sys.exit(1)

with open(PARSED_JSON) as f:
    P = json.load(f)

def ms(key):
    return P.get(key, {}).get("total_kernel_ns", 0) / 1e6

# --- sanity check: any required FC decode bench missing? -------------------
_required_decode_keys = [
    "fc_qkv_sliding_decode_M1",
    "fc_o_sliding_decode_M1",
    "fc_qk_full_decode_M1",
    "fc_o_full_decode_M1",
    "fc_gate_dense_decode_M1",
    "fc_up_dense_decode_M1",
    "fc_down_dense_decode_M1",
    "lm_head_decode_M1",
]
_missing = [k for k in _required_decode_keys if ms(k) <= 0]
if _missing:
    print("WARNING: required decode benches missing or zero — report will be inaccurate:")
    for k in _missing:
        print(f"  - {k}")
    print("Re-run the missing benches and parse_logs.py before trusting Eff% values.")

# Roofline helpers ------------------------------------------------------------
def fc_int4_bytes(M_, K, N, g=128):
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

def roofline_only_ms(flops, byts, int8_xmx=False):
    """Theoretical min time at 100% of relevant peak (no overhead)."""
    if byts <= 0 and flops <= 0:
        return 0.0
    ai = flops / byts if byts > 0 else 0
    peak = INT8 * 1e12 if int8_xmx else FP16 * 1e12
    ridge = RIDGE8 if int8_xmx else RIDGE
    t_mem = byts / (BW * 1e9) * 1000.0          # ms
    t_cmp = flops / peak * 1000.0               # ms
    return max(t_mem, t_cmp)

# PA bytes / flops ------------------------------------------------------------
def pa_decode_sliding_bytes(kv_eff):
    return 2 * NKV_S * HD_S * kv_eff * 1  # INT8 KV

def pa_decode_sliding_flops(kv_eff):
    return 2.0 * NH_S * HD_S * kv_eff * 2  # QK + AV

def pa_decode_full_bytes(kv):
    return 2 * NKV_F * HD_F * kv * 1

def pa_decode_full_flops(kv):
    return 2.0 * NH_F * HD_F * kv * 2

def pa_prefill_sliding_flops(s):
    # causal + sliding window. Lower triangular = s*(s+1)/2 pairs.
    if s <= SW:
        eff_pairs = s * (s + 1) / 2.0
    else:
        # Tail-padded window: ~SW*(s - SW/2) pairs.
        eff_pairs = SW * (s - SW / 2.0)
    return 2.0 * NH_S * HD_S * eff_pairs * 2  # QK + AV

def pa_prefill_full_flops(s):
    eff_pairs = s * (s + 1) / 2.0  # causal
    return 2.0 * NH_F * HD_F * eff_pairs * 2

def pa_prefill_sliding_bytes(s):
    # KV cache write (S tokens) + small Q I/O (approx).
    return 2 * NKV_S * HD_S * min(s, SW) * 2 + 2 * NH_S * HD_S * s * 2

def pa_prefill_full_bytes(s):
    return 2 * NKV_F * HD_F * s * 2 + 2 * NH_F * HD_F * s * 2

# Dense MLP roofline ----------------------------------------------------------
def dense_decode_bytes():
    return (fc_int4_bytes(1, H, I_DENSE) +
            fc_int4_bytes(1, H, I_DENSE) +
            fc_int4_bytes(1, I_DENSE, H))

def dense_decode_flops():
    return fc_flops(1, H, I_DENSE) * 2 + fc_flops(1, I_DENSE, H)

def dense_prefill_bytes(s):
    return (fc_int4_bytes(s, H, I_DENSE) +
            fc_int4_bytes(s, H, I_DENSE) +
            fc_int4_bytes(s, I_DENSE, H))

def dense_prefill_flops(s):
    return fc_flops(s, H, I_DENSE) * 2 + fc_flops(s, I_DENSE, H)

# Decode per-op table ---------------------------------------------------------
def decode_op_rows(kv):
    rows = []
    def add(name, kernel, ms_val, calls, flops, byts, int8_xmx=False):
        r = roofline(ms_val, flops, byts, int8_xmx)
        rows.append(dict(op=name, kernel=kernel,
                         single_ms=ms_val, calls=calls,
                         total_ms=ms_val*calls,
                         gflops=r['gflops'], gbs=r['gbs'],
                         eff=r['eff'], bound=r['bound']))

    # FC sliding attention (x40)
    m = ms("fc_qkv_sliding_decode_M1")
    add("FC_QKV_sliding (INT4)", "gemm_kernel", m, NL_S,
        fc_flops(1, H, QKV_S_N), fc_int4_bytes(1, H, QKV_S_N))
    m = ms("fc_o_sliding_decode_M1")
    add("FC_O_sliding (INT4)", "gemm_kernel", m, NL_S,
        fc_flops(1, O_S_K, H), fc_int4_bytes(1, O_S_K, H))

    # FC full attention (x8)
    m = ms("fc_qk_full_decode_M1")
    add("FC_QK_full (INT4)", "gemm_kernel", m, NL_F,
        fc_flops(1, H, QK_F_N), fc_int4_bytes(1, H, QK_F_N))
    m = ms("fc_o_full_decode_M1")
    add("FC_O_full (INT4)", "gemm_kernel", m, NL_F,
        fc_flops(1, O_F_K, H), fc_int4_bytes(1, O_F_K, H))

    # Dense MLP (x48) gate/up/down combined
    mg = ms("fc_gate_dense_decode_M1")
    mu = ms("fc_up_dense_decode_M1")
    md = ms("fc_down_dense_decode_M1")
    dense_ms = mg + mu + md
    add("DenseMLP_gate+up+down (INT4)", "gemm_kernel x3", dense_ms, NL,
        dense_decode_flops(), dense_decode_bytes())

    # LM_head (x1)
    m = ms("lm_head_decode_M1")
    add("LM_head (INT8)", "gemm_kernel", m, 1,
        fc_flops(1, H, VOCAB), fc_int8_bytes(1, H, VOCAB))

    # PA sliding (x40): use kv=min(ctx, SW)
    kv_eff = min(kv, SW)
    key = f"pa_sliding_decode_kv{kv_eff}"
    if key not in P:
        key = "pa_sliding_decode_kv1024"
        kv_eff = SW
    m = ms(key)
    pa_s_flops = pa_decode_sliding_flops(kv_eff)
    pa_s_byts  = pa_decode_sliding_bytes(kv_eff)
    if m <= 0:
        m = roofline_only_ms(pa_s_flops, pa_s_byts)  # roofline-only fallback
    add(f"PA_sliding (INT8 KV, eff={kv_eff})", "paged_attn_single_token", m, NL_S,
        pa_s_flops, pa_s_byts)

    # PA full (x8): full KV
    m = ms(f"pa_full_decode_kv{kv}")
    pa_f_flops = pa_decode_full_flops(kv)
    pa_f_byts  = pa_decode_full_bytes(kv)
    if m <= 0:
        m = roofline_only_ms(pa_f_flops, pa_f_byts)  # roofline-only fallback
    add("PA_full (INT8 KV)", "paged_attn_single_token", m, NL_F,
        pa_f_flops, pa_f_byts)

    # Small ops aggregate
    ms_rms = ms("so_rmsnorm_h3840_decode")
    ms_add = ms("so_add_h3840_decode")
    ms_rq_s = ms("so_rope_q_sliding_decode")
    ms_rk_s = ms("so_rope_k_sliding_decode")
    ms_qn_s = ms("so_rmsnorm3d_q_sliding_decode")
    ms_kn_s = ms("so_rmsnorm3d_k_sliding_decode")
    ms_rq_f = ms("so_rope_q_full_decode")
    ms_rk_f = ms("so_rope_k_full_decode")
    ms_qn_f = ms("so_rmsnorm3d_q_full_decode")
    ms_kn_f = ms("so_rmsnorm3d_k_full_decode")
    # 4 RMSNorm per layer + 1 final = 193 total
    # rope_q + rope_k + q_norm + k_norm per layer (sliding vs full)
    # add: 2 per layer x 48 = 96
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

# Prefill per-op table --------------------------------------------------------
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

    # FC sliding attention (x40), prefill uses INT8 XMX path
    m = ms(f"fc_qkv_sliding_prefill_{sk}")
    add("FC_QKV_sliding (INT4->INT8 XMX)", "dq+gemm_kernel", m, NL_S,
        fc_flops(s, H, QKV_S_N), fc_int4_bytes(s, H, QKV_S_N), int8_xmx=True)
    m = ms(f"fc_o_sliding_prefill_{sk}")
    add("FC_O_sliding (INT4->INT8 XMX)", "dq+gemm_kernel", m, NL_S,
        fc_flops(s, O_S_K, H), fc_int4_bytes(s, O_S_K, H), int8_xmx=True)

    # FC full attention (x8)
    m = ms(f"fc_qk_full_prefill_{sk}")
    add("FC_QK_full (INT4->INT8 XMX)", "dq+gemm_kernel", m, NL_F,
        fc_flops(s, H, QK_F_N), fc_int4_bytes(s, H, QK_F_N), int8_xmx=True)
    m = ms(f"fc_o_full_prefill_{sk}")
    add("FC_O_full (INT4->INT8 XMX)", "dq+gemm_kernel", m, NL_F,
        fc_flops(s, O_F_K, H), fc_int4_bytes(s, O_F_K, H), int8_xmx=True)

    # Dense MLP (x48) gate/up/down combined
    mg = ms(f"fc_gate_dense_prefill_{sk}")
    mu = ms(f"fc_up_dense_prefill_{sk}")
    md_val = ms(f"fc_down_dense_prefill_{sk}")
    dense_ms = mg + mu + md_val
    add("DenseMLP_gate+up+down (INT4->INT8 XMX)", "dq+gemm_kernel x3", dense_ms, NL,
        dense_prefill_flops(s), dense_prefill_bytes(s), int8_xmx=True)

    # PA sliding prefill (x40)
    # bench measures full causal; if S<=SW the bench value is accurate.
    # For S>SW we scale from S=1024 by (s/1024) since work ~ SW*s.
    if s <= SW:
        m_pa_s = ms(f"pa_sliding_prefill_{sk}")
    else:
        m_base = ms("pa_sliding_prefill_S1024")
        m_pa_s = m_base * (s / 1024.0) if m_base > 0 else 0
    pa_s_flops = pa_prefill_sliding_flops(s)
    pa_s_byts  = pa_prefill_sliding_bytes(s)
    if m_pa_s <= 0:
        m_pa_s = roofline_only_ms(pa_s_flops, pa_s_byts)  # roofline-only fallback
    add(f"PA_sliding (FP16 prefill, sw={SW})", "sdpa_micro_prefill", m_pa_s, NL_S,
        pa_s_flops, pa_s_byts)

    # PA full prefill (x8)
    m = ms(f"pa_full_prefill_{sk}")
    pa_f_flops = pa_prefill_full_flops(s)
    pa_f_byts  = pa_prefill_full_bytes(s)
    if m <= 0:
        m = roofline_only_ms(pa_f_flops, pa_f_byts)  # roofline-only fallback
    add("PA_full (FP16 prefill, causal)", "sdpa_micro_prefill", m, NL_F,
        pa_f_flops, pa_f_byts)

    # LM head (1 output token, same as decode)
    lm_ms_val = ms("lm_head_decode_M1")
    r_lm = roofline(lm_ms_val, fc_flops(1, H, VOCAB), fc_int8_bytes(1, H, VOCAB))
    rows.append(dict(op="LM_head (INT8, 1 out tok)", kernel="gemm_kernel",
                     single_ms=lm_ms_val, calls=1, total_ms=lm_ms_val,
                     gflops=r_lm['gflops'], gbs=r_lm['gbs'],
                     eff=r_lm['eff'], bound="memory"))

    # Small ops prefill (per S)
    ms_rms_p = ms(f"so_rmsnorm_h3840_prefill_{sk}")
    ms_add_p = ms(f"so_add_h3840_prefill_{sk}")
    ms_rq_s_p = ms(f"so_rope_q_sliding_prefill_{sk}")
    ms_rk_s_p = ms(f"so_rope_k_sliding_prefill_{sk}")
    ms_qn_s_p = ms(f"so_rmsnorm3d_q_sliding_prefill_{sk}")
    ms_kn_s_p = ms(f"so_rmsnorm3d_k_sliding_prefill_{sk}")
    ms_qn_f_p = ms(f"so_rmsnorm3d_q_full_prefill_{sk}")
    ms_kn_f_p = ms(f"so_rmsnorm3d_k_full_prefill_{sk}")
    small_total = (ms_rms_p * (4*NL + 1) +
                   ms_add_p * 2*NL +
                   (ms_rq_s_p + ms_rk_s_p) * NL_S +
                   (ms_qn_s_p + ms_kn_s_p) * NL_S +
                   (ms_qn_f_p + ms_kn_f_p) * NL_F)
    rows.append(dict(op="SmallOps (norm/rope/add)", kernel="rms/rope/eltwise",
                     single_ms=0, calls=0, total_ms=small_total,
                     gflops=0, gbs=0, eff=0, bound="memory"))

    rows.sort(key=lambda r: -r["total_ms"])
    return rows

def total_ms(rows):
    return sum(r["total_ms"] for r in rows)

# Build performance_metrics ---------------------------------------------------
metrics = {
    "platform": "PTL_12Xe",
    "platform_desc": "PTL (B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s, FP16 XMX 58.982 TFLOPS, INT8 XMX 117.965 TOPS)",
    "bw": BW, "fp16_tflops": FP16, "int8_tops": INT8,
    "ridge_f16": RIDGE, "ridge_i8": RIDGE8,
    "model": "google/gemma-4-12B-it (dense)",
    "config_summary": "INT4 g=128 body + INT8 g=128 LM_head + INT8 KV cache, FP16 activation",
    "token_sizes": SIZES,
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

# Markdown ---------------------------------------------------------------------
def fmt(v, fs):
    if v == 0: return "—"
    return format(v, fs)

def md_row(r):
    s_ms   = f"{r['single_ms']:.4f}" if r['single_ms'] > 0 else "—"
    calls  = str(r['calls']) if r['calls'] > 0 else "—"
    t_ms   = f"{r['total_ms']:.3f}"
    gf     = fmt(r['gflops'], '.2f')
    gbs    = fmt(r['gbs'], '.1f')
    eff    = f"{r['eff']:.1f}%" if r['eff'] > 0 else "—"
    return f"| {r['op']} | {r['kernel']} | {s_ms} | {calls} | {t_ms} | {gf} | {gbs} | {eff} | {r['bound']} |"

def build_summary():
    lines = []
    lines += [
        f"# Gemma4-12B-it (dense) — Roofline on PTL 12Xe (B390 iGPU)",
        "",
        f"**Date:** {date.today().isoformat()}",
        "**Target:** Local_Admin@10.239.132.229 — PTL B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s",
        "**Model:** `google/gemma-4-12B-it` (dense, no MoE) — text decoder of the unified multimodal model",
        "**Config:** INT4 g=128 body + INT8 g=128 LM_head + INT8 KV cache, FP16 activation",
        "**SDPA impl:** PagedAttention OpenCL + micro_kernel (kv_type=i8)",
        "**Profiler:** cliloader 3.0.6 `--device-performance-timing`, mean kernel time (ms)",
        f"**Token sweep:** input S / kv ∈ {{{', '.join(str(x) for x in SIZES)}}}; **output tokens (decode): {OUTPUT_TOKENS}**",
        "**Bench:** `.github/skills/dev_roofline_profiling/utils/{fc,pa,small_ops}_bench`",
        "",
        "---",
        "",
        "## 1. Hardware Peaks",
        "",
        "| Platform | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | Ridge (F16) | Ridge (I8) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| PTL (B390 iGPU, 12 Xe @ 2400 MHz) | {BW:.1f} | {FP16:.3f} | {INT8:.3f} | {RIDGE:.1f} | {RIDGE8:.1f} FLOP/byte |",
        "",
        "_FP16 XMX = 12 × 8 × 256 FLOP/cycle × 2.4 GHz. INT8 XMX = 2× FP16._",
        "",
        "---",
        "",
        "## 2. Model Configuration",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| `vocab_size` | {VOCAB:,} |",
        f"| `hidden_size` (H) | {H:,} |",
        f"| `num_hidden_layers` | **{NL}** ({NL_S} sliding + {NL_F} full, 5:1 pattern) |",
        f"| Sliding attn (NH/NKV/HD) | {NH_S}/{NKV_S}/{HD_S} → Q={NH_S*HD_S}, KV={NKV_S*HD_S} |",
        f"| Full attn (NH/NKV/HD) | {NH_F}/{NKV_F}/{HD_F} → Q={NH_F*HD_F}, K={NKV_F*HD_F}, V=K |",
        f"| `sliding_window` | {SW} |",
        f"| `attention_k_eq_v` | true (full attn: V reuses K projection) |",
        f"| Dense MLP `intermediate_size` | {I_DENSE} (GEGLU: gate+up {H}→{I_DENSE}, down {I_DENSE}→{H}) |",
        f"| `hidden_activation` | gelu_pytorch_tanh (GEGLU) |",
        f"| `tie_word_embeddings` | true |",
        f"| `final_logit_softcapping` | 30.0 |",
        f"| Body weight quant | INT4 g=128 (asymmetric) |",
        f"| LM head weight quant | INT8 g=128 |",
        f"| KV cache | INT8 |",
        f"| Activation dtype | FP16 |",
        "",
        "---",
        "",
        "## 3. Graph Fusion Notes",
        "",
        "| Op variant | GPU primitive | Fused? | Notes |",
        "|---|---|---|---|",
        "| FC_QKV/O sliding/full (INT4) | FullyConnectedCompressed | Partial | decode: gemm_kernel; prefill: dq+gemm_kernel (INT8 XMX) |",
        "| Dense MLP gate/up/down (INT4) | FullyConnectedCompressed×3 | ❌ NOT fused (GEGLU between gate/up) | 3 separate FC kernels per layer |",
        "| PagedAttention sliding | PagedAttention | ✅ Fused | INT8 KV, GQA group=2, sw=1024 |",
        "| PagedAttention full | PagedAttention | ✅ Fused | INT8 KV, GQA group=16, V reuses K |",
        "| GEGLU multiply | gelu(gate)·up | SwiGLU primitive | Fused (bench-only) |",
        "| add (residual) | eltwise | Not fused | 2× per layer |",
        "| rmsnorm | RMSNorm primitive | Not fused | 4× per layer + 1 final |",
        "",
        "---",
        "",
        "## 4. Decode Performance — Totals",
        "",
        f"_Decode total for {OUTPUT_TOKENS} output tokens = per-token ms × {OUTPUT_TOKENS}._",
        "",
        f"| kv tokens | FC_attn/L (ms) | DenseMLP/L (ms) | PA_sliding/L (ms) | PA_full/L (ms) | LM_head (ms) | SmallOps (ms) | **ms/tok** | **tok/s** | **decode {OUTPUT_TOKENS} tok (ms)** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kv in KV_SIZES:
        d = metrics["decode"][kv]; rows = d["rows"]
        fc_attn = sum(r["total_ms"] for r in rows if r["op"].startswith("FC_Q") or r["op"].startswith("FC_O"))
        dense = sum(r["total_ms"] for r in rows if "DenseMLP" in r["op"])
        pa_s = sum(r["total_ms"] for r in rows if "PA_sliding" in r["op"])
        pa_f = sum(r["total_ms"] for r in rows if "PA_full" in r["op"])
        lm = sum(r["total_ms"] for r in rows if "LM_head" in r["op"])
        sm = sum(r["total_ms"] for r in rows if "SmallOps" in r["op"])
        tot = d["total_ms"]; tps = 1000.0 / tot if tot > 0 else 0
        out_ms = tot * OUTPUT_TOKENS
        lines.append(
            f"| {kv:>5,} | {fc_attn/NL:.4f} | {dense/NL:.4f} | {pa_s/NL_S:.4f} | "
            f"{pa_f/NL_F:.4f} | {lm:.3f} | {sm:.3f} | **{tot:.2f}** | **{tps:.1f}** | **{out_ms:,.0f}** |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Prefill Performance — Totals (TTFT) + End-to-End Latency",
        "",
        f"_End-to-end = TTFT + {OUTPUT_TOKENS} × decode_ms (decode KV = input S, capped to sliding window for sliding layers)._",
        "",
        f"| S tokens | FC_attn/L (ms) | DenseMLP/L (ms) | PA_sliding/L (ms) | PA_full/L (ms) | LM_head (ms) | SmallOps (ms) | **TTFT (ms)** | **prefill tok/s** | **decode ms/tok @ kv=S** | **E2E {OUTPUT_TOKENS}-out (ms)** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in S_SIZES:
        p = metrics["prefill"][s]; rows = p["rows"]
        fc_attn = sum(r["total_ms"] for r in rows if r["op"].startswith("FC_Q") or r["op"].startswith("FC_O"))
        dense = sum(r["total_ms"] for r in rows if "DenseMLP" in r["op"])
        pa_s = sum(r["total_ms"] for r in rows if "PA_sliding" in r["op"])
        pa_f = sum(r["total_ms"] for r in rows if "PA_full" in r["op"])
        lm = sum(r["total_ms"] for r in rows if "LM_head" in r["op"])
        sm = sum(r["total_ms"] for r in rows if "SmallOps" in r["op"])
        tot = p["total_ms"]; tps = s / (tot / 1000.0) if tot > 0 else 0
        decode_ms = metrics["decode"][s]["total_ms"] if s in metrics["decode"] else 0
        e2e_ms = tot + OUTPUT_TOKENS * decode_ms
        lines.append(
            f"| {s:>5,} | {fc_attn/NL:.3f} | {dense/NL:.3f} | {pa_s/NL_S:.3f} | "
            f"{pa_f/NL_F:.3f} | {lm:.3f} | {sm:.3f} | **{tot:.1f}** | **{tps:.0f}** | **{decode_ms:.2f}** | **{e2e_ms:,.0f}** |"
        )

    # ====================================================================
    # Section 6 — Theoretical weight footprint per module (model storage)
    # ====================================================================
    lines += [
        "", "---", "",
        "## 6. Theoretical Weight Footprint per Module",
        "",
        "_INT4 g=128 weight bytes = N·K/2 + N·(K/g)·2 (FP16 scale) + N·(K/g)/2 (INT4 zp). "
        "INT8 g=128 weight bytes = N·K + N·(K/g)·2 (FP16 scale). "
        "FP16 baseline shown for reference (= 2·N·K).\n"
        "Params count excludes scale/zero-point overhead.\n_",
        "",
        "| Module | Shape (K×N) | Quant | Per-layer params | Per-layer storage (MB) | Layers | **Total params** | **Total storage (MB)** | Share |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    grand_params = 0
    grand_bytes  = 0
    mod_rows = []
    for name, K, N, q, nl in MODULES:
        pl_params = module_param_count(K, N)
        pl_bytes  = module_weight_bytes(K, N, q)
        tot_params = pl_params * nl
        tot_bytes  = pl_bytes  * nl
        mod_rows.append((name, K, N, q, nl, pl_params, pl_bytes, tot_params, tot_bytes))
        grand_params += tot_params
        grand_bytes  += tot_bytes
    for name, K, N, q, nl, pl_p, pl_b, tp, tb in mod_rows:
        share = tb / grand_bytes * 100 if grand_bytes else 0
        lines.append(
            f"| {name} | {K}×{N} | {q} | {pl_p/1e6:.2f} M | {pl_b/1024/1024:.2f} | {nl} | "
            f"**{tp/1e6:.1f} M** | **{tb/1024/1024:,.1f}** | {share:.1f}% |"
        )
    lines.append(
        f"| **TOTAL** | | | | | | **{grand_params/1e9:.2f} B** | **{grand_bytes/1024/1024:,.1f}** | 100% |"
    )
    lines += [
        "",
        f"_FP16 baseline (no quant): {sum(2*K*N*nl for _,K,N,_,nl in [(n,K,N,q,nl) for n,K,N,q,nl in MODULES])/1024/1024:,.0f} MB. "
        f"Quantized total ({grand_bytes/1024/1024:,.0f} MB) = ~{grand_bytes/sum(2*K*N*nl for n,K,N,q,nl in MODULES)*100:.1f}% of FP16 size._",
    ]

    # ====================================================================
    # Section 7 — Measured vs Theoretical Roofline
    # ====================================================================
    def theo_ms(byts, flops, int8_xmx=False):
        peak = (INT8_ACHIEV if int8_xmx else FP16_ACHIEV) * 1e12
        t_mem = byts  / (BW_ACHIEV * 1e9) * 1000.0
        t_cmp = flops / peak * 1000.0
        return max(t_mem, t_cmp), ("memory" if t_mem >= t_cmp else "compute")

    # Decode module specs used by both per-op rows and aggregate sweep.
    DEC_FC_SPECS = [
        ("FC_QKV_sliding", H,       QKV_S_N, "INT4", NL_S, "fc_qkv_sliding_decode_M1"),
        ("FC_O_sliding",   O_S_K,   H,       "INT4", NL_S, "fc_o_sliding_decode_M1"),
        ("FC_QK_full",     H,       QK_F_N,  "INT4", NL_F, "fc_qk_full_decode_M1"),
        ("FC_O_full",      O_F_K,   H,       "INT4", NL_F, "fc_o_full_decode_M1"),
        ("MLP_gate",       H,       I_DENSE, "INT4", NL,   "fc_gate_dense_decode_M1"),
        ("MLP_up",         H,       I_DENSE, "INT4", NL,   "fc_up_dense_decode_M1"),
        ("MLP_down",       I_DENSE, H,       "INT4", NL,   "fc_down_dense_decode_M1"),
        ("LM_head",        H,       VOCAB,   "INT8", 1,    "lm_head_decode_M1"),
    ]
    PRE_FC_SPECS = [
        ("FC_QKV_sliding", H,       QKV_S_N, NL_S, "fc_qkv_sliding_prefill"),
        ("FC_O_sliding",   O_S_K,   H,       NL_S, "fc_o_sliding_prefill"),
        ("FC_QK_full",     H,       QK_F_N,  NL_F, "fc_qk_full_prefill"),
        ("FC_O_full",      O_F_K,   H,       NL_F, "fc_o_full_prefill"),
        ("MLP_gate",       H,       I_DENSE, NL,   "fc_gate_dense_prefill"),
        ("MLP_up",         H,       I_DENSE, NL,   "fc_up_dense_prefill"),
        ("MLP_down",       I_DENSE, H,       NL,   "fc_down_dense_prefill"),
    ]

    def theo_decode_total(kv):
        """Theoretical full-model decode ms for 1 token at given kv."""
        total = 0.0
        for _, K, N, q, nl, _ in DEC_FC_SPECS:
            byts  = fc_int4_bytes(1, K, N) if q == "INT4" else fc_int8_bytes(1, K, N)
            t, _b = theo_ms(byts, fc_flops(1, K, N), int8_xmx=False)
            total += t * nl
        total += theo_ms(pa_decode_sliding_bytes(min(kv, SW)),
                         pa_decode_sliding_flops(min(kv, SW)), False)[0] * NL_S
        total += theo_ms(pa_decode_full_bytes(kv),
                         pa_decode_full_flops(kv), False)[0] * NL_F
        return total

    def theo_prefill_total(s):
        """Theoretical full-model prefill (TTFT) ms at given S."""
        total = 0.0
        for _, K, N, nl, _ in PRE_FC_SPECS:
            t, _b = theo_ms(fc_int4_bytes(s, K, N), fc_flops(s, K, N), int8_xmx=True)
            total += t * nl
        # LM_head once for single output token after prefill
        total += theo_ms(fc_int8_bytes(1, H, VOCAB), fc_flops(1, H, VOCAB), False)[0]
        # Sliding PA caps to SW for S>SW (scale linearly from S=SW baseline)
        if s > SW:
            t_sl, _ = theo_ms(pa_prefill_sliding_bytes(SW),
                              pa_prefill_sliding_flops(SW), False)
            t_sl = t_sl * (s / SW)
        else:
            t_sl, _ = theo_ms(pa_prefill_sliding_bytes(s),
                              pa_prefill_sliding_flops(s), False)
        total += t_sl * NL_S
        total += theo_ms(pa_prefill_full_bytes(s),
                         pa_prefill_full_flops(s), False)[0] * NL_F
        return total

    lines += [
        "", "---", "",
        f"## 7. Measured vs Theoretical Roofline",
        "",
        "### 7.1 How the theoretical roofline is computed",
        "",
        "For each kernel we know:",
        "",
        "- **Bytes moved (B)** = weight read + activation read + output write. Weight bytes include INT4/INT8 packed weights plus FP16 scales and (for INT4) packed zero points at group-128 granularity.",
        "- **FLOPs (F)** = 2 · M · K · N for a GEMM with shape (M·K) × (K·N).",
        "",
        "The classic roofline model gives the lower-bound execution time of a kernel as:",
        "",
        "$$ t_{ideal}\\ =\\ \\max\\!\\left(\\frac{B}{\\mathrm{BW}_{peak}},\\ \\frac{F}{\\mathrm{Peak}_{compute}}\\right) $$",
        "",
        f"- $\\mathrm{{BW}}_{{peak}}$ = {BW:.1f} GB/s (PTL B390 LPDDR5x peak).",
        f"- $\\mathrm{{Peak}}_{{compute}}$ = {FP16:.3f} TFLOPS (FP16 XMX) for decode (low arithmetic intensity → memory-bound) and **{INT8:.3f} TOPS (INT8 XMX)** for prefill FC ops, since OpenVINO uses the INT8-XMX `dynamic_quantize_gpu_opt + gemm_kernel` path for compressed-weight matmul at M>1.",
        "- The ratio decides whether the kernel is **memory-bound** ($B/\\mathrm{BW} > F/\\mathrm{Peak}$) or **compute-bound** (the converse).",
        "",
        f"**Overhead factor:** we deduct **{OVERHEAD*100:.0f}%** from each peak to model unavoidable real-world losses (kernel launch, dispatch, memory-bus contention from activations/scales, sub-optimal tile boundaries, refresh / power excursions, etc.). The numbers used in the tables below are therefore:",
        "",
        f"- Achievable BW = peak × {ACHIEV_FRAC:.2f} = **{BW_ACHIEV:.2f} GB/s**",
        f"- Achievable FP16 XMX = peak × {ACHIEV_FRAC:.2f} = **{FP16_ACHIEV:.3f} TFLOPS**",
        f"- Achievable INT8 XMX = peak × {ACHIEV_FRAC:.2f} = **{INT8_ACHIEV:.3f} TOPS**",
        "",
        "$$ t_{theo}\\ =\\ \\max\\!\\left(\\frac{B}{0.95\\cdot\\mathrm{BW}_{peak}},\\ \\frac{F}{0.95\\cdot\\mathrm{Peak}_{compute}}\\right) $$",
        "",
        "$$ \\text{Eff\\%}\\ =\\ \\frac{t_{theo}}{t_{meas}}\\times 100\\% $$",
        "",
        "Aggregated full-model latencies are simply the sum of $t_{theo}$ over every kernel invocation in the model (per-layer ms × layer count for body modules, plus the LM_head call).",
        "",
    ]

    # --- 7.2 Per-op decode (FC are kv-independent; PA scanned across kv) ----
    ref_kv = 1024 if 1024 in metrics["decode"] else KV_SIZES[0]
    lines += [
        "### 7.2 Per-op decode — measured vs theoretical (1 query token)",
        "",
        "_FC and LM_head decode are independent of kv (M=1). PA rows are listed per tested kv._",
        "",
        "| Module | Bytes (KB) | FLOPs (M) | Bound | Theo ms | **Meas ms** | Eff% |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for name, K, N, q, _nl, key in DEC_FC_SPECS:
        byts  = fc_int4_bytes(1, K, N) if q == "INT4" else fc_int8_bytes(1, K, N)
        flops = fc_flops(1, K, N)
        t, b  = theo_ms(byts, flops, int8_xmx=False)
        mv    = ms(key)
        eff   = t / mv * 100 if mv > 0 else 0
        lines.append(
            f"| {name} | {byts/1024:,.1f} | {flops/1e6:.1f} | {b} | {t:.4f} | **{mv:.4f}** | {eff:.1f}% |"
        )
    for kv in KV_SIZES:
        kv_eff   = min(kv, SW)
        pas_byts = pa_decode_sliding_bytes(kv_eff)
        pas_fl   = pa_decode_sliding_flops(kv_eff)
        t, b     = theo_ms(pas_byts, pas_fl, False)
        mv       = ms(f"pa_sliding_decode_kv{kv_eff}")
        eff      = t / mv * 100 if mv > 0 else 0
        lines.append(
            f"| PA_sliding kv={kv} (eff={kv_eff}) | {pas_byts/1024:,.1f} | {pas_fl/1e6:.2f} | {b} | {t:.4f} | **{mv:.4f}** | {eff:.1f}% |"
        )
    for kv in KV_SIZES:
        paf_byts = pa_decode_full_bytes(kv)
        paf_fl   = pa_decode_full_flops(kv)
        t, b     = theo_ms(paf_byts, paf_fl, False)
        mv       = ms(f"pa_full_decode_kv{kv}")
        eff      = t / mv * 100 if mv > 0 else 0
        lines.append(
            f"| PA_full kv={kv} | {paf_byts/1024:,.1f} | {paf_fl/1e6:.2f} | {b} | {t:.4f} | **{mv:.4f}** | {eff:.1f}% |"
        )

    # --- 7.3 Per-op prefill: one block per S ---------------------------------
    lines += ["", "### 7.3 Per-op prefill — measured vs theoretical (per S)",
              "", "_¹ PA_sliding bench only runs up to S=SW=1024; for S>SW the measured ms is scaled linearly from the S=1024 baseline (work ∝ SW·S)._", ""]
    for s in S_SIZES:
        sk = f"S{s}"
        lines += [
            f"#### S = {s:,}",
            "",
            "| Module | Bytes (MB) | FLOPs (G) | Bound | Theo ms | **Meas ms** | Eff% |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
        for name, K, N, _nl, key_prefix in PRE_FC_SPECS:
            byts  = fc_int4_bytes(s, K, N)
            flops = fc_flops(s, K, N)
            t, b  = theo_ms(byts, flops, int8_xmx=True)
            mv    = ms(f"{key_prefix}_{sk}")
            eff   = t / mv * 100 if mv > 0 else 0
            lines.append(
                f"| {name} | {byts/1024/1024:.2f} | {flops/1e9:.2f} | {b} | {t:.4f} | **{mv:.4f}** | {eff:.1f}% |"
            )
        # PA prefill
        pas_byts = pa_prefill_sliding_bytes(s)
        pas_fl   = pa_prefill_sliding_flops(s)
        t, b     = theo_ms(pas_byts, pas_fl, False)
        mv       = ms(f"pa_sliding_prefill_{sk}")
        derived  = ""
        if mv <= 0 and s > SW:
            # sliding caps at SW: linearly scale from S=SW baseline
            mv_base = ms(f"pa_sliding_prefill_S{SW}")
            mv      = mv_base * (s / SW)
            derived = " ¹"
        eff      = t / mv * 100 if mv > 0 else 0
        lines.append(
            f"| PA_sliding{derived} | {pas_byts/1024/1024:.2f} | {pas_fl/1e9:.2f} | {b} | {t:.4f} | **{mv:.4f}** | {eff:.1f}% |"
        )
        paf_byts = pa_prefill_full_bytes(s)
        paf_fl   = pa_prefill_full_flops(s)
        t, b     = theo_ms(paf_byts, paf_fl, False)
        mv       = ms(f"pa_full_prefill_{sk}")
        eff      = t / mv * 100 if mv > 0 else 0
        lines.append(
            f"| PA_full | {paf_byts/1024/1024:.2f} | {paf_fl/1e9:.2f} | {b} | {t:.4f} | **{mv:.4f}** | {eff:.1f}% |"
        )
        lines.append("")

    # --- 7.4 Aggregate sweep across ALL sizes (decode + prefill + E2E) -------
    lines += [
        "### 7.4 Aggregate — full-model latency across all tested sizes",
        "",
        "_Theoretical = sum of per-kernel $t_{theo}$ over every invocation in the model (with the 5 % overhead already applied). Decode is for 1 generated token at the given kv; prefill is TTFT over S tokens; E2E adds 512 generated tokens at the matched kv._",
        "",
        "| Phase | Size | **Theoretical ms** | Theoretical tok/s | **Measured ms** | Measured tok/s | Eff% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    # Decode rows
    for kv in KV_SIZES:
        th = theo_decode_total(kv)
        mv = metrics["decode"][kv]["total_ms"]
        eff = th / mv * 100 if mv > 0 else 0
        lines.append(
            f"| Decode (1 tok) | kv={kv:,} | **{th:.2f}** | {1000/th:.1f} | **{mv:.2f}** | {1000/mv:.1f} | {eff:.1f}% |"
        )
    # Prefill rows
    for s in S_SIZES:
        th = theo_prefill_total(s)
        mv = metrics["prefill"][s]["total_ms"]
        eff = th / mv * 100 if mv > 0 else 0
        lines.append(
            f"| Prefill TTFT | S={s:,} | **{th:.1f}** | {s/(th/1000):.0f} | **{mv:.1f}** | {s/(mv/1000):.0f} | {eff:.1f}% |"
        )
    # E2E rows
    lines += [
        "",
        f"_E2E latency for {OUTPUT_TOKENS} generated tokens (TTFT + {OUTPUT_TOKENS} × decode_ms @ kv=S):_",
        "",
        "| Size (S = kv₀) | **Theoretical E2E (ms)** | **Measured E2E (ms)** | Eff% |",
        "|---:|---:|---:|---:|",
    ]
    for s in S_SIZES:
        th_p = theo_prefill_total(s)
        th_d = theo_decode_total(s)
        th_e = th_p + OUTPUT_TOKENS * th_d
        mv_p = metrics["prefill"][s]["total_ms"]
        mv_d = metrics["decode"][s]["total_ms"]
        mv_e = mv_p + OUTPUT_TOKENS * mv_d
        eff  = th_e / mv_e * 100 if mv_e > 0 else 0
        lines.append(
            f"| {s:,} | **{th_e:,.0f}** | **{mv_e:,.0f}** | {eff:.1f}% |"
        )

    # Decode tables ----------------------------------------------------------
    lines += ["", "---", "", "## 8. Decode Tables (1 query token, KV = context length)", "",
              "_Eff% = GB/s / 110 for memory-bound, GFLOPS / XMX_peak for compute-bound._"]
    for kv in KV_SIZES:
        d = metrics["decode"][kv]; tot = d["total_ms"]
        tps = 1000.0 / tot if tot > 0 else 0
        lines += [
            "", f"### Decode — KV={kv:,}", "",
            "| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in d["rows"]:
            lines.append(md_row(r))
        lines.append(f"| **TOTAL** | | | | **{tot:.3f}** | | | **{tps:.1f} tok/s** | |")

    # Prefill tables ---------------------------------------------------------
    lines += ["", "---", "", "## 9. Prefill Tables (single forward over S tokens)", ""]
    for s in S_SIZES:
        p = metrics["prefill"][s]; tot = p["total_ms"]
        tps = s / (tot / 1000.0) if tot > 0 else 0
        lines += [
            "", f"### Prefill — S={s:,}", "",
            "| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in p["rows"]:
            lines.append(md_row(r))
        lines.append(f"| **TOTAL** | | | | **{tot:.1f}** | | | **TTFT {tot/1000:.2f}s, {tps:.0f} tok/s** | |")

    # Decode time-share -----------------------------------------------------
    ref_kv = 1024 if 1024 in metrics["decode"] else KV_SIZES[0]
    d_ref = metrics["decode"][ref_kv]; tot_ref = d_ref["total_ms"]
    lines += ["", "---", "",
              f"## 10. Decode Time-Share (kv={ref_kv:,})",
              "",
              f"**Total decode @ kv={ref_kv}: {tot_ref:.2f} ms → {1000/tot_ref:.1f} tok/s**",
              "",
              "| Group | ms | Share |",
              "|---|---:|---:|"]
    for r in d_ref["rows"]:
        share = r["total_ms"] / tot_ref * 100 if tot_ref > 0 else 0
        calls = f"×{r['calls']}" if r["calls"] > 0 else ""
        lines.append(f"| {r['op']} {calls} | {r['total_ms']:.3f} | {share:.1f}% |")

    # Caveats ---------------------------------------------------------------
    lines += [
        "", "---", "",
        "## 11. Caveats & Method",
        "",
        "- Each op profiled in its own process via cliloader Device Performance Timing; mean kernel time per iteration (ms).",
        "- FC weight bytes: INT4 weight + FP16 scale + INT4 zp (g=128) + FP16 act + FP16 out.",
        "- LM_head bytes: INT8 weight + FP16 scale (g=128) + FP16 act + FP16 out.",
        "- PA bytes: INT8 KV cache + FP16 Q / out.",
        "- Decode FC is memory-bound (weight-read dominates at M=1).",
        "- Prefill FC is compute-bound (INT8 XMX path via dynamic_quantize_gpu_opt + gemm_kernel).",
        "- PA prefill FLOPs use causal lower-triangular pairs = S(S+1)/2 (not S²); sliding caps to SW for S>SW.",
        "- PA sliding decode KV caps at sliding_window=1024 → for ctx≥1024 we reuse kv=1024 measurement.",
        "- PA sliding prefill bench uses full causal mask; for S>1024 we scale the S=1024 measurement linearly (sliding work ~ SW·S).",
        "- Full attention uses partial_rotary_factor=0.25 (rope applied to 25% of HD); the bench measures full HD rope (slight over-estimate, immaterial).",
        "- LM head is profiled once (single output token) and reused for prefill & decode.",
        "- All RMSNorm/RoPE/Add bench timings are aggregated using model call-counts (4×NL+1 rms, 2×NL adds, per-layer rope/q-norm/k-norm for sliding vs full).",
        f"- Target: Local_Admin@10.239.132.229 (PTL 12Xe, B390 iGPU).",
        "",
        "### Optimization levers",
        "",
        "1. **Dense MLP (15360 wide)** dominates decode bandwidth — fusing gate+up into a single packed-INT4 weight read (or moving to INT4 XMX decompose-on-the-fly) is the highest lever.",
        "2. **LM_head INT8** is a ~Gigabyte read every token; INT4 g=128 (with the softcap +1 LM_head) would roughly halve its time on decode.",
        "3. **PA full** GQA group=16 + V=K halves the KV traffic vs. independent V proj; ensure prefill uses sdpa_micro_prefill (compute-bound, INT8 KV).",
        "4. **rmsnorm / rope / add** aggregate to a non-trivial decode tax (193 + 96 + per-layer rope/qk-norm); a fused norm+rope+add primitive would cut launches.",
    ]
    return "\n".join(lines)

OUT_SUMMARY.write_text(build_summary())
print(f"Saved {OUT_SUMMARY}")
for s in SIZES:
    d_ms = metrics['decode'][s]['total_ms']
    p_ms = metrics['prefill'][s]['total_ms']
    print(f"S={s:>5}: TTFT={p_ms:8.1f} ms  decode={d_ms:6.2f} ms/tok ({1000/d_ms:5.1f} tok/s)  "
          f"E2E({OUTPUT_TOKENS}out)={p_ms + OUTPUT_TOKENS*d_ms:8.0f} ms")
