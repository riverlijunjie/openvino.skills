#!/usr/bin/env python3
"""Generate performance_metrics_ptl_f16qkvo.json and
   SUMMARY_qwen3_5_moe_0509_ptl_f16qkvo.md for PTL 12Xe (B390 iGPU)
   with FP16 QKV/O and FP16 shared expert (3 separate FC ops).

   Key differences vs 0429 report:
   - FC_QKV (2048→5120): FP16 uncompressed  (was INT4 g=128)
   - FC_O   (4096→2048): FP16 uncompressed  (was INT4 g=128)
   - Shared expert gate/up/down: FP16, NOT fused into MoE primitive
     → runs as 3 separate FullyConnected kernels; benchmarked independently
   - MoE routed experts: INT4 g=128 (unchanged), benchmarked with SI=0
"""
import json, math
from pathlib import Path

OUT = Path(__file__).resolve().parent
PARSED_JSON = OUT / "parsed_ptl_f16qkvo.json"
PARSED_LINATTN_JSON = OUT / "parsed_ptl_linattn.json"
OUT_METRICS  = OUT / "performance_metrics_ptl_f16qkvo.json"
OUT_SUMMARY  = OUT / "SUMMARY_qwen3_5_moe_0509_ptl_f16qkvo.md"

# ── Hardware peaks ────────────────────────────────────────────────────────────
BW      = 110.0       # GB/s
FP16    = 58.9824     # TFLOPS  (12 Xe × 8 EU × 256 FP/cyc × 2.4 GHz)
INT8    = 117.9648    # TOPS    (2× FP16)
RIDGE   = FP16 * 1e12 / (BW * 1e9)   # FP16 ridge point: FLOP/byte ≈ 536.2
RIDGE8  = INT8 * 1e12 / (BW * 1e9)   # INT8 ridge point: FLOP/byte ≈ 1072.4

# ── Model config ──────────────────────────────────────────────────────────────
H      = 2048   # hidden size
NH     = 16;  NKV = 2;   HD = 256   # full-attn heads
HK     = 32;  KD  = 128; VD = 128   # linear-attn (GDN)
I      = 512   # MoE intermediate
SI     = 512   # shared expert intermediate
NE     = 256   # num experts
TK     = 8     # top-k per token
VOCAB  = 248320
NL     = 40    # total layers
NF     = 10    # full-attn layers
NL_LIN = 30    # linear-attn layers

KV_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
S_SIZES  = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

with open(PARSED_JSON) as f:
    P = json.load(f)

# Merge in linear-attention broken-down benchmarks
if PARSED_LINATTN_JSON.exists():
    with open(PARSED_LINATTN_JSON) as f:
        P_LA = json.load(f)
    P.update(P_LA)

def ms(key):
    return P.get(key, {}).get("total_kernel_ns", 0) / 1e6

# ── Roofline helpers ──────────────────────────────────────────────────────────
def fc_f16_bytes(M_, K, N):
    """FP16 uncompressed FC: input(f16) + weight(f16) + output(f16)."""
    return M_*K*2 + N*K*2 + M_*N*2

def fc_int4_bytes(M_, K, N, g=128):
    """INT4 g=128 FC bytes (weight u4 + scale f16 + zp u4 + io f16)."""
    return M_*K*2 + N*K//2 + N*(K//g)*2 + N*(K//g)//2 + M_*N*2

def fc_int8_bytes(M_, K, N, g=128):
    """INT8 g=128 FC bytes (weight u8 + scale f16 + io f16)."""
    return M_*K*2 + N*K + N*(K//g)*2 + M_*N*2

def fc_flops(M_, K, N):
    return 2.0 * M_ * K * N

def roofline(ms_val, flops, byts, int8_xmx=False):
    """Return dict with all roofline metrics."""
    if ms_val <= 0:
        return dict(ms=0, gbs=0, gflops=0, eff_bw=0, eff_xmx=0, ai=0, eff=0, bound="N/A")
    ai     = flops / byts
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

# ── MoE routed bytes (TK selected from NE, INT4 g=128) ───────────────────────
def moe_routed_bytes(tk=TK, ne=NE, h=H, i=I, g=128):
    """Weight bytes for TK selected experts (gate+up+down) + router + io."""
    # Each expert: gate(h,i)+up(h,i)+down(i,h), INT4 g=128 with scale+zp
    per_exp = fc_int4_bytes(1, h, i, g) + fc_int4_bytes(1, h, i, g) + fc_int4_bytes(1, i, h, g)
    # Subtract the M_=1 io from each (io is separate), add back once
    per_exp_w = (i*h//2 + (h//g)*i*2 + (h//g)*i//2) * 2 + (i*h//2 + (i//g)*h*2 + (i//g)*h//2)
    # Router: FC(h, ne) INT4
    router_w  = ne*h//2 + (h//g)*ne*2 + (h//g)*ne//2
    io        = h*2 + h*2
    return tk * per_exp_w + router_w + io

def moe_routed_flops(tk=TK, h=H, i=I):
    """FLOPs for TK experts gate+up+down."""
    return 2.0 * (2*tk*h*i + tk*i*h)   # gate+up: 2×(tk×h×i), down: tk×i×h

# ── Shared expert bytes: 3 separate F16 FC ops ───────────────────────────────
def shared_f16_bytes(h=H, si=SI):
    """Bytes for shared expert gate+up+down (all F16)."""
    return (fc_f16_bytes(1, h, si) + fc_f16_bytes(1, h, si) + fc_f16_bytes(1, si, h))

def shared_f16_flops(h=H, si=SI):
    return fc_flops(1, h, si)*2 + fc_flops(1, si, h)   # gate+up+down

def shared_f16_prefill_bytes(s, h=H, si=SI):
    return (fc_f16_bytes(s, h, si) + fc_f16_bytes(s, h, si) + fc_f16_bytes(s, si, h))

def shared_f16_prefill_flops(s, h=H, si=SI):
    return fc_flops(s, h, si)*2 + fc_flops(s, si, h)

# ── GDN bytes/flops ───────────────────────────────────────────────────────────
def gdn_bytes(t=1, hk=HK, kd=KD, vd=VD):
    """Per-token GDN: read+write state + small io."""
    state = hk * kd * vd * 2      # FP16 state matrix
    return 2 * state + t * H * 2  # read+write state

def gdn_flops(t=1, hk=HK, kd=KD, vd=VD):
    return 2.0 * hk * kd * vd * t

# ── PA decode bytes/flops ─────────────────────────────────────────────────────
def pa_decode_bytes(kv, nh=NH, nkv=NKV, hd=HD):
    """KV cache bytes read for one PA decode step (INT8 KV)."""
    return 2 * nkv * hd * kv * 1   # INT8 KV cache

def pa_decode_flops(kv, nh=NH, hd=HD):
    return 2.0 * nh * hd * kv * 2  # QK + QV

# ── Decode per-op tables ──────────────────────────────────────────────────────
def decode_op_rows(kv):
    rows = []

    def add(name, kernel, ms_val, calls, flops, byts, int8_xmx=False):
        r = roofline(ms_val, flops, byts, int8_xmx)
        rows.append(dict(op=name, kernel=kernel,
                         single_ms=ms_val, calls=calls,
                         total_ms=ms_val*calls,
                         gflops=r['gflops'], gbs=r['gbs'],
                         eff=r['eff'], bound=r['bound'],
                         ai=r['ai'], eff_bw=r['eff_bw'], eff_xmx=r['eff_xmx']))

    # FC_QKV  f16  M=1
    m = ms("fc_qkv_decode_M1")
    add("FC_QKV (FP16)", "gemm_kernel", m, NF,
        fc_flops(1, H, 5120), fc_f16_bytes(1, H, 5120))

    # FC_O  f16  M=1
    m = ms("fc_o_decode_M1")
    add("FC_O (FP16)", "gemm_kernel", m, NF,
        fc_flops(1, NH*HD, H), fc_f16_bytes(1, NH*HD, H))

    # FC linattn broken-down: qkv, z, a, b, out_proj  (all INT4)
    m_qkv = ms("linattn_qkv_decode_M1")
    add("FC_linattn_qkv (INT4)", "gemm_kernel", m_qkv, NL_LIN,
        fc_flops(1, H, 8192), fc_int4_bytes(1, H, 8192))
    m_z = ms("linattn_z_decode_M1")
    add("FC_linattn_z (INT4)", "gemm_kernel", m_z, NL_LIN,
        fc_flops(1, H, 4096), fc_int4_bytes(1, H, 4096))
    m_a = ms("linattn_a_decode_M1")
    add("FC_linattn_a (INT4)", "gemm_kernel", m_a, NL_LIN,
        fc_flops(1, H, 32), fc_int4_bytes(1, H, 32))
    m_b = ms("linattn_b_decode_M1")
    add("FC_linattn_b (INT4)", "gemm_kernel", m_b, NL_LIN,
        fc_flops(1, H, 32), fc_int4_bytes(1, H, 32))
    m_out = ms("linattn_out_decode_M1")
    add("FC_linattn_out (INT4)", "gemm_kernel", m_out, NL_LIN,
        fc_flops(1, 4096, H), fc_int4_bytes(1, 4096, H))

    # LM head  INT8
    m = ms("lm_head_decode_M1")
    add("LM_head (INT8)", "gemm_kernel", m, 1,
        fc_flops(1, H, VOCAB), fc_int8_bytes(1, H, VOCAB))

    # MoE routed (fused, SI=0, INT4)
    m = ms("moe_routed_decode_M1")
    add("MoE3GEMM_fused (routed, INT4)", "moe_3gemm_swiglu_mlp_gate_up+down", m, NL,
        moe_routed_flops(), moe_routed_bytes())

    # Shared expert: 3 separate F16 FC  (combined)
    mg = ms("shared_gate_decode_M1")
    mu = ms("shared_up_decode_M1")
    md = ms("shared_down_decode_M1")
    ms_sh = mg + mu + md
    add("SharedExpert_gate+up+down (FP16 FC×3)", "gemm_kernel×3", ms_sh, NL,
        shared_f16_flops(), shared_f16_bytes())

    # PA decode  (INT8 KV)
    m = ms(f"pa_decode_kv{kv}")
    add("PagedAttention (INT8 KV)", "paged_attention_opt__single_token", m, NF,
        pa_decode_flops(kv), pa_decode_bytes(kv))

    # GDN decode
    m = ms("gdn_decode_T1")
    add("GatedDeltaNet", "gated_delta_net_ref_sa", m, NL_LIN,
        gdn_flops(), gdn_bytes())

    # Small ops (aggregate)
    ms_rms = ms("so_rmsnorm_h2048_decode")
    ms_add = ms("so_add_decode")
    ms_rq  = ms("so_rope_q_decode")
    ms_rk  = ms("so_rope_k_decode")
    ms_qn  = ms("so_rmsnorm3d_qnorm_decode")
    ms_kn  = ms("so_rmsnorm3d_knorm_decode")
    small_total = ms_rms*2*NL + ms_add*2*NL + ms_rq*NF + ms_rk*NF + ms_qn*NF + ms_kn*NF
    rows.append(dict(op="SmallOps (norm/rope/add)", kernel="rms/rope/eltwise", single_ms=0,
                     calls=0, total_ms=small_total,
                     gflops=0, gbs=0, eff=0, bound="memory",
                     ai=0, eff_bw=0, eff_xmx=0))

    rows.sort(key=lambda r: -r["total_ms"])
    return rows

# ── Prefill per-op tables ─────────────────────────────────────────────────────
def prefill_op_rows(s):
    rows = []

    def add(name, kernel, ms_val, calls, flops, byts, int8_xmx=False):
        r = roofline(ms_val, flops, byts, int8_xmx)
        rows.append(dict(op=name, kernel=kernel,
                         single_ms=ms_val, calls=calls,
                         total_ms=ms_val*calls,
                         gflops=r['gflops'], gbs=r['gbs'],
                         eff=r['eff'], bound=r['bound'],
                         ai=r['ai'], eff_bw=r['eff_bw'], eff_xmx=r['eff_xmx']))

    sk = f"S{s}"

    # FC_QKV  F16  M=S
    m = ms(f"fc_qkv_prefill_{sk}")
    add("FC_QKV (FP16)", "gemm_kernel", m, NF,
        fc_flops(s, H, 5120), fc_f16_bytes(s, H, 5120))

    # FC_O  F16  M=S
    m = ms(f"fc_o_prefill_{sk}")
    add("FC_O (FP16)", "gemm_kernel", m, NF,
        fc_flops(s, NH*HD, H), fc_f16_bytes(s, NH*HD, H))

    # FC linattn broken-down: qkv, z, a, b, out_proj  (all INT4, prefill uses INT8 XMX)
    m_qkv = ms(f"linattn_qkv_prefill_{sk}")
    add("FC_linattn_qkv (INT4)", "dq+gemm_kernel", m_qkv, NL_LIN,
        fc_flops(s, H, 8192), fc_int4_bytes(s, H, 8192), int8_xmx=True)
    m_z = ms(f"linattn_z_prefill_{sk}")
    add("FC_linattn_z (INT4)", "dq+gemm_kernel", m_z, NL_LIN,
        fc_flops(s, H, 4096), fc_int4_bytes(s, H, 4096), int8_xmx=True)
    m_a = ms(f"linattn_a_prefill_{sk}")
    add("FC_linattn_a (INT4)", "dq+gemm_kernel", m_a, NL_LIN,
        fc_flops(s, H, 32), fc_int4_bytes(s, H, 32), int8_xmx=True)
    m_b = ms(f"linattn_b_prefill_{sk}")
    add("FC_linattn_b (INT4)", "dq+gemm_kernel", m_b, NL_LIN,
        fc_flops(s, H, 32), fc_int4_bytes(s, H, 32), int8_xmx=True)
    m_out = ms(f"linattn_out_prefill_{sk}")
    add("FC_linattn_out (INT4)", "dq+gemm_kernel", m_out, NL_LIN,
        fc_flops(s, 4096, H), fc_int4_bytes(s, 4096, H), int8_xmx=True)

    # MoE routed prefill (fused)
    m = ms(f"moe_routed_prefill_{sk}")
    # Prefill bytes: all NE expert weights read (not just TK)
    fl_p = moe_routed_flops(tk=TK, h=H, i=I) * s  # scaled by S
    # Actually for prefill: GEMM over S tokens, read all weights once
    by_p_gate = s*H*2 + NE*H*I//2 + NE*(H//128)*I*2 + NE*(H//128)*I//2
    by_p_up   = s*H*2 + NE*H*I//2 + NE*(H//128)*I*2 + NE*(H//128)*I//2
    by_p_down = s*I*2 + NE*I*H//2 + NE*(I//128)*H*2 + NE*(I//128)*H//2
    # router
    by_p_router = s*H*2 + NE*H//2 + (H//128)*NE*2 + s*NE*2
    by_p_out  = s*H*2
    by_p_total = by_p_gate + by_p_up + by_p_down + by_p_router + by_p_out
    fl_p = 2 * (2*s*TK*H*I + s*TK*I*H)   # actual compute for TK experts × S tokens
    # Use INT8 XMX for prefill MoE (INT4 weights → INT8 compute)
    add("MoE3GEMM_fused (routed, INT4)", "grouped_micro_gemm×3+scatter+gather", m, NL,
        fl_p, by_p_total, int8_xmx=True)

    # Shared expert: 3 separate F16 FC  (combined)
    mg = ms(f"shared_gate_prefill_{sk}")
    mu = ms(f"shared_up_prefill_{sk}")
    md_val = ms(f"shared_down_prefill_{sk}")
    ms_sh = mg + mu + md_val
    add("SharedExpert_gate+up+down (FP16 FC×3)", "gemm_kernel×3", ms_sh, NL,
        shared_f16_prefill_flops(s), shared_f16_prefill_bytes(s))

    # PA prefill (O(S²) attention)
    m = ms(f"pa_prefill_{sk}")
    pa_b = 2 * NKV * HD * s  # simplified (KV written, ignoring exact sdpa bytes)
    pa_fl = 2 * NH * HD * s * s
    add("PagedAttention (FP16, causal)", "sdpa_micro__prefill", m, NF,
        pa_fl, pa_b)

    # GDN prefill
    m = ms(f"gdn_prefill_{sk}")
    if m == 0 and s == 131072:
        # S131072 failed: extrapolate ×2 from S65536
        m = ms("gdn_prefill_S65536") * 2
        gdn_note = " [extrapolated]"
    else:
        gdn_note = ""
    add(f"GatedDeltaNet{gdn_note}", "gated_delta_net_ref_sa", m, NL_LIN,
        gdn_flops(s), gdn_bytes(s))

    # LM head (same as decode, only 1 token out)
    lm_ms = ms("lm_head_decode_M1")
    rows.append(dict(op="LM_head (INT8, 1 out tok)", kernel="gemm_kernel",
                     single_ms=lm_ms, calls=1, total_ms=lm_ms,
                     gflops=roofline(lm_ms, fc_flops(1,H,VOCAB), fc_int8_bytes(1,H,VOCAB))['gflops'],
                     gbs=roofline(lm_ms, fc_flops(1,H,VOCAB), fc_int8_bytes(1,H,VOCAB))['gbs'],
                     eff=roofline(lm_ms, fc_flops(1,H,VOCAB), fc_int8_bytes(1,H,VOCAB))['eff'],
                     bound="memory", ai=0, eff_bw=0, eff_xmx=0))

    # Small ops for prefill (only have S<=8192; extrapolate if needed)
    if s <= 8192:
        ms_rms_p = ms(f"so_rmsnorm_h2048_prefill_{sk}")
        ms_rq_p  = ms(f"so_rope_q_prefill_{sk}")
    else:
        scale = s / 8192
        ms_rms_p = ms("so_rmsnorm_h2048_prefill_S8192") * scale
        ms_rq_p  = ms("so_rope_q_prefill_S8192") * scale
    small_total = ms_rms_p*2*NL + ms_rq_p*NF
    rows.append(dict(op="SmallOps (norm/rope)", kernel="rms/rope", single_ms=0,
                     calls=0, total_ms=small_total,
                     gflops=0, gbs=0, eff=0, bound="memory",
                     ai=0, eff_bw=0, eff_xmx=0))

    rows.sort(key=lambda r: -r["total_ms"])
    return rows

def total_ms(rows):
    return sum(r["total_ms"] for r in rows)

# ── Build performance_metrics JSON ────────────────────────────────────────────
metrics = {
    "platform": "PTL",
    "platform_desc": "PTL (B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s, FP16 XMX 58.982 TFLOPS)",
    "bw": BW, "fp16_tflops": FP16, "int8_tops": INT8, "ridge_f16": RIDGE,
    "config": "FP16 QKV/O + FP16 shared expert (3 separate FC)",
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
        "# Qwen3.5-MoE-35B-A3B (qwen3_5_moe) — Roofline on PTL (B390 iGPU)",
        "",
        "**Date:** 2026-05-09",
        "**Target:** PTL (Panther Lake B390 iGPU, 12 Xe @ 2400 MHz)",
        "**Config:** FP16 QKV/O + FP16 shared expert (3 separate FC ops)",
        "**Profiler:** cliloader 3.0.6 `--device-performance-timing`, mean kernel time",
        "**Token sweep:** S/kv ∈ {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}",
        "**Bench:** `.github/skills/dev_roofline_profiling/utils/{fc,moe,pa,gdn,small_ops}_bench`",
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
        "## 2. Model Configuration (qwen3_5_moe — F16 QKV/O variant)",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| `vocab_size`                    | {VOCAB:,} |",
        f"| `hidden_size` (H)               | {H:,} |",
        f"| `num_hidden_layers`             | **{NL}** (hybrid attention) |",
        "| Layer pattern                   | `linear_attn × 3 → full_attn × 1`, repeated → **10 full + 30 GDN** |",
        f"| `num_attention_heads`           | {NH} (NH, full-attn) |",
        f"| `num_key_value_heads`           | {NKV} (NKV, GQA group=8) |",
        f"| `head_dim`                      | {HD} |",
        f"| `linear_num_value_heads`        | {HK} (HK in GDN bench) |",
        f"| `linear_value_head_dim`         | {VD} |",
        f"| `moe_intermediate_size`         | {I} |",
        f"| `shared_expert_intermediate_size` | **{SI} (always-on)** |",
        f"| `num_experts`                   | {NE} |",
        f"| `num_experts_per_tok`           | {TK} |",
        "| FC_QKV weight quant             | **FP16 (uncompressed)** |",
        "| FC_O weight quant               | **FP16 (uncompressed)** |",
        "| Shared expert quant             | **FP16 (uncompressed), 3 separate FullyConnected ops** |",
        "| MoE routed expert quant         | INT4 g=128 (asymmetric), fused via MOE3GemmFusedCompressed |",
        "| FC_linattn weight quant         | INT4 g=128 (in_proj_qkv/z/a/b + out_proj) |",
        "| LM-head weight quant            | INT8 g=128 |",
        "| KV cache quant                  | INT8 (asymmetric) |",
        "| Activation dtype                | FP16 |",
        "",
        "",
        "---",
        "",
        "## 3. Weight Size Summary",
        "",
        "| Weight | Shape (K × N) | Quant | Bytes | × Layers | Total MB |",
        "|---|---:|---|---:|---:|---:|",
    ]

    # Weight table
    def mb(b): return b / 1024**2
    # FC_QKV: 10 layers × (2048, 5120) f16
    fcqkv_b = H*5120*2; lines.append(f"| FC_QKV (fused Q+K+V proj) | {H}×5120 | FP16 | {fcqkv_b:,.0f} | {NF} | {mb(fcqkv_b*NF):.1f} |")
    # FC_O: 10 layers × (4096, 2048) f16
    fco_b   = NH*HD*H*2; lines.append(f"| FC_O (attention output)   | {NH*HD}×{H} | FP16 | {fco_b:,.0f} | {NF} | {mb(fco_b*NF):.1f} |")
    # FC_linattn broken-down: qkv(2048,8192), z(2048,4096), a(2048,32), b(2048,32), out(4096,2048)
    def int4_wb(K_, N_, g=128): return N_*K_//2 + N_*(K_//g)*2 + N_*(K_//g)//2
    fcl_qkv_b = int4_wb(H, 8192); lines.append(f"| FC_linattn_qkv (in_proj_qkv)| {H}×8192 | INT4 g=128 | {fcl_qkv_b:,.0f} | {NL_LIN} | {mb(fcl_qkv_b*NL_LIN):.1f} |")
    fcl_z_b   = int4_wb(H, 4096); lines.append(f"| FC_linattn_z (in_proj_z)    | {H}×4096 | INT4 g=128 | {fcl_z_b:,.0f} | {NL_LIN} | {mb(fcl_z_b*NL_LIN):.1f} |")
    fcl_a_b   = int4_wb(H, 32);   lines.append(f"| FC_linattn_a (in_proj_a)    | {H}×32   | INT4 g=128 | {fcl_a_b:,.0f} | {NL_LIN} | {mb(fcl_a_b*NL_LIN):.1f} |")
    fcl_b_b   = int4_wb(H, 32);   lines.append(f"| FC_linattn_b (in_proj_b)    | {H}×32   | INT4 g=128 | {fcl_b_b:,.0f} | {NL_LIN} | {mb(fcl_b_b*NL_LIN):.1f} |")
    fcl_out_b = int4_wb(4096, H);  lines.append(f"| FC_linattn_out (out_proj)   | 4096×{H} | INT4 g=128 | {fcl_out_b:,.0f} | {NL_LIN} | {mb(fcl_out_b*NL_LIN):.1f} |")
    fcl_total = fcl_qkv_b + fcl_z_b + fcl_a_b + fcl_b_b + fcl_out_b
    # MoE routed: 40 layers × 256 experts × gate+up+down INT4
    moe_per_expert = (2*I*H//2 + 2*(H//128)*I*2 + 2*(H//128)*I//2) + (I*H//2 + (I//128)*H*2 + (I//128)*H//2)
    lines.append(f"| MoE Expert gate+up+down   | {H}×{I} / {I}×{H} | INT4 g=128 | {moe_per_expert:,.0f}/expert | {NL}×{NE} | {mb(moe_per_expert*NL*NE):.1f} |")
    # Router: 40 × 256×2048 int4
    router_b = NE*H//2 + (H//128)*NE*2 + (H//128)*NE//2
    lines.append(f"| Router                    | {H}×{NE} | INT4 g=128 | {router_b:,.0f} | {NL} | {mb(router_b*NL):.1f} |")
    # Shared expert: 40 × gate+up+down f16
    shared_per = 2*SI*H*2 + H*SI*2   # gate+up: (H,SI)×2 + down: (SI,H)
    lines.append(f"| Shared Expert gate+up+down | {H}×{SI} / {SI}×{H} | FP16 | {shared_per:,.0f} | {NL} | {mb(shared_per*NL):.1f} |")
    # LM head: 1 × (2048, 248320) int8
    lmh_b = H*VOCAB + (H//128)*VOCAB*2
    lines.append(f"| LM_Head                   | {H}×{VOCAB} | INT8 g=128 | {lmh_b:,.0f} | 1 | {mb(lmh_b):.1f} |")
    total_w = fcqkv_b*NF + fco_b*NF + fcl_total*NL_LIN + moe_per_expert*NL*NE + router_b*NL + shared_per*NL + lmh_b
    lines.append(f"| **Total static weights** | | | | | **{mb(total_w):.0f} MB** |")

    lines += [
        "",
        "---",
        "",
        "## 4. Graph Fusion Notes",
        "",
        "| Op variant | GPU primitive | Fused? | Notes |",
        "|---|---|---|---|",
        "| FC_QKV / FC_O (FP16) | FullyConnected (plain) | Not fused further | Single `gemm_kernel` per call |",
        "| MoE routed experts (INT4) | MOE3GemmFusedCompressed | ✅ Fused | `moe_3gemm_swiglu_mlp_gate_up` + `_down` + scatter/gather |",
        "| Shared expert gate/up (FP16) | 3 × FullyConnected | ❌ NOT fused | FP16 weights cannot match `MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN` |",
        "| Shared expert down (FP16) | FullyConnected | ❌ NOT fused | Same reason |",
        "| FC_linattn_qkv/z/out (INT4) | FullyConnectedCompressed | Partial | decode: 1 `gemm_kernel`; prefill: `dynamic_quantize_gpu_opt`+`gemm_kernel` |",
        "| FC_linattn_a/b (INT4, tiny) | FullyConnectedCompressed | Partial | decode: 1 `gemm_kernel` (3.8µs); prefill: dominated by dynamic_quantize overhead |",
        "| PagedAttention | PagedAttention | ✅ Fused | INT8 KV cache, GQA group=8 |",
        "| GatedDeltaNet | GatedDeltaNet | ✅ Fused | Reference kernel (not optimised) |",
        "",
        "> **Key insight**: Shared expert with FP16 weights runs as **3 separate FullyConnected kernels** per MoE layer × 40 layers.",
        "> This is the **actual model runtime behaviour** — the FP16 plain constant is converted to FullyConnected by",
        "> `ConvertMatMulToFullyConnected` before `FuseMOESharedExpert` can pattern-match it.",
        "",
        "---",
        "",
        "## 5. Decode Performance — Totals",
        "",
        "| kv tokens | MoE routed/L (ms) | Shared F16/L (ms) | PA/L (ms) | GDN/L (ms) | linattn/L (ms) | LM head (ms) | **total ms** | **tok/s** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    decode_totals = {}
    for kv in KV_SIZES:
        rows = metrics["decode"][kv]["rows"]
        def get_total(op_substr):
            return sum(r['total_ms'] for r in rows if op_substr in r['op'])
        moe_r    = get_total("MoE3GEMM_fused")
        shared_r = get_total("SharedExpert")
        pa_r     = get_total("PagedAttention")
        gdn_r    = get_total("GatedDeltaNet")
        lin_r    = get_total("FC_linattn")
        lm_r     = get_total("LM_head")
        tot      = metrics["decode"][kv]["total_ms"]
        toks     = 1000/tot if tot > 0 else 0
        decode_totals[kv] = {"total_ms": tot, "tok_s": toks}
        lines.append(
            f"| {kv:>7,} | {moe_r/NL:.4f} | {shared_r/NL:.4f} | {pa_r/NF:.4f} | {gdn_r/NL_LIN:.4f} | {lin_r/NL_LIN:.4f} | {lm_r:.3f} | **{tot:.2f}** | **{toks:.1f}** |"
        )

    lines += [
        "",
        "---",
        "",
        "## 6. Prefill Performance — Totals (TTFT)",
        "",
        "| S tokens | MoE routed/L (ms) | Shared F16/L (ms) | PA/L (ms) | GDN/L (ms) | linattn/L (ms) | **TTFT (ms)** | **tok/s** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for s in S_SIZES:
        rows = metrics["prefill"][s]["rows"]
        def get_tot(op_substr):
            return sum(r['total_ms'] for r in rows if op_substr in r['op'])
        moe_r    = get_tot("MoE3GEMM_fused")
        shared_r = get_tot("SharedExpert")
        pa_r     = get_tot("PagedAttention")
        gdn_r    = get_tot("GatedDeltaNet")
        lin_r    = get_tot("FC_linattn")
        tot      = metrics["prefill"][s]["total_ms"]
        toks     = s/tot*1000 if tot > 0 else 0
        gdn_note = " ★" if s == 131072 else ""
        lines.append(
            f"| {s:>7,} | {moe_r/NL:.3f} | {shared_r/NL:.3f} | {pa_r/NF:.3f} | {gdn_r/NL_LIN:.3f}{gdn_note} | {lin_r/NL_LIN:.3f} | **{tot:.1f}** | **{toks:.1f}** |"
        )

    lines += [
        "",
        "★ GDN prefill S=131072 kernel log was empty (run failed); value extrapolated ×2 from S=65536.",
        "",
        "---",
        "",
        "## 7. Decode Time-Share (kv=4096)",
        "",
    ]

    kv4_rows = metrics["decode"][4096]["rows"]
    tot4 = metrics["decode"][4096]["total_ms"]
    lines.append(f"**Total decode @ kv=4096: {tot4:.2f} ms → {1000/tot4:.1f} tok/s**\n")
    lines.append("| Group | ms | Share |")
    lines.append("|---|---:|---:|")
    for r in kv4_rows:
        if r['total_ms'] > 0:
            lines.append(f"| {r['op']} (×{r['calls']}) | {r['total_ms']:.3f} | {r['total_ms']/tot4*100:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## 8. Decode Tables (1 query token, KV = context length)",
        "",
        "_Columns: op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound_",
        "_Eff% = GB/s / 110 GB/s × 100 for memory-bound; GFLOPS / XMX_peak × 100 for compute-bound._",
        "",
    ]

    for kv in KV_SIZES:
        rows = metrics["decode"][kv]["rows"]
        tot  = metrics["decode"][kv]["total_ms"]
        lines.append(f"### Decode — KV={kv:,}")
        lines.append("")
        lines.append("| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for r in rows:
            lines.append(md_table_row(r))
        lines.append(f"| **TOTAL** | | | | **{tot:.3f}** | | | | |")
        lines.append(f"| | | | | | | | **{1000/tot:.1f} tok/s** | |")
        lines.append("")

    lines += [
        "---",
        "",
        "## 9. Prefill Tables (single forward over S tokens)",
        "",
        "_Eff% for INT4 FC prefill = GFLOPS / INT8 XMX (117.965 TOPS); for FP16 FC = GFLOPS / FP16 XMX (58.982 TFLOPS)._",
        "",
    ]

    for s in S_SIZES:
        rows = metrics["prefill"][s]["rows"]
        tot  = metrics["prefill"][s]["total_ms"]
        toks = s/tot*1000 if tot > 0 else 0
        lines.append(f"### Prefill — S={s:,}")
        lines.append("")
        lines.append("| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for r in rows:
            lines.append(md_table_row(r))
        lines.append(f"| **TOTAL** | | | | **{tot:.1f}** | | | | |")
        lines.append(f"| | | | | TTFT={tot:.0f} ms | | | **{toks:.1f} tok/s** | |")
        lines.append("")

    lines += [
        "---",
        "",
        "## 10. Roofline Highlights (PTL decode, kv=4096)",
        "",
        "| Op | Single ms | Calls | Total ms | GB/s | Eff% | Bound |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for r in sorted(kv4_rows, key=lambda x: -x.get('gbs', 0)):
        if r.get('gbs', 0) > 0:
            lines.append(
                f"| {r['op']} | {r['single_ms']:.4f} | {r['calls']} | {r['total_ms']:.3f} | {r['gbs']:.1f} | {r['eff']:.1f}% | {r['bound']} |"
            )

    lines += [
        "",
        "---",
        "",
        "## 11. Comparison: F16 QKV/O vs INT4 QKV/O (PTL decode kv=4096)",
        "",
        "| Op | INT4 (prev) ms | FP16 (new) ms | Δ ms | Notes |",
        "|---|---:|---:|---:|---|",
        f"| FC_QKV × {NF} layers | (INT4 ~0.128×10=1.28 ms est.) | {ms('fc_qkv_decode_M1')*NF:.3f} ms | — | 4× more weight bytes, lower arith intensity |",
        f"| FC_O × {NF} layers   | (INT4 ~0.100×10=1.00 ms est.) | {ms('fc_o_decode_M1')*NF:.3f} ms | — | Same pattern |",
        f"| MoE routed + Shared × {NL} layers | (fused 0.174 ms/L) | {(ms('moe_routed_decode_M1')+(ms('shared_gate_decode_M1')+ms('shared_up_decode_M1')+ms('shared_down_decode_M1')))*NL:.3f} ms total | — | Routed fused; shared unfused → 3 kernel launches overhead |",
        "",
        "> FC_QKV and FC_O with FP16 weights are ~3× slower per layer than INT4 variants (higher BW demand).",
        "> The shared expert with FP16 weights cannot be fused into MOE3GemmFusedCompressed, adding 3 separate kernel launches per layer.",
        "",
        "---",
        "",
        "_Generated by `build_report_ptl_f16qkvo.py`_",
    ]

    return "\n".join(lines)

summary = build_summary()
OUT_SUMMARY.write_text(summary)
print(f"Saved {OUT_SUMMARY}")

# ── Quick console summary ─────────────────────────────────────────────────────
print("\n=== DECODE TOTALS (PTL F16 QKV/O) ===")
print(f"{'kv':>8}  {'total ms':>10}  {'tok/s':>8}")
for kv in KV_SIZES:
    tot = metrics["decode"][kv]["total_ms"]
    print(f"{kv:>8,}  {tot:>10.2f}  {1000/tot:>8.1f}")

print("\n=== PREFILL TOTALS ===")
print(f"{'S':>8}  {'TTFT ms':>12}  {'TTFT s':>8}  {'tok/s':>8}")
for s in S_SIZES:
    tot = metrics["prefill"][s]["total_ms"]
    print(f"{s:>8,}  {tot:>12.1f}  {tot/1000:>8.3f}  {s/tot*1000:>8.1f}")
