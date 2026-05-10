#!/usr/bin/env python3
"""Generate SUMMARY_qwen3_5_moe_int4g64_ptl.md for PTL 12Xe (B390 iGPU)
   following SUMMARY_TEMPLATE.md layout.

   v2: Full re-run after OpenVINO code update. ALL ops re-measured.
   Config: ALL FC ops (QKV, O, linear attn projs) and shared expert use INT4 g=64.
   MoE routed experts use INT4 g=64 (shared expert NOT fused, runs as 3 separate FC).
   LM head uses INT8 g=128. KV cache INT8.
"""
import json, math
from pathlib import Path

OUT = Path(__file__).resolve().parent
PARSED_JSON = OUT / "parsed_ptl_int4g64_v3.json"
OUT_METRICS = OUT / "performance_metrics_ptl_int4g64_v3.json"
OUT_SUMMARY = OUT / "SUMMARY_qwen3_5_moe_int4g64_ptl.md"

# ── Hardware peaks ────────────────────────────────────────────────────────────
BW      = 110.0       # GB/s
FP16    = 58.9824     # TFLOPS
INT8    = 117.9648    # TOPS
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
G      = 64    # group_size for all body FC and MoE

KV_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
S_SIZES  = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# ── Load data ─────────────────────────────────────────────────────────────────
with open(PARSED_JSON) as f:
    P = json.load(f)

def ms(key):
    return P.get(key, {}).get("total_kernel_ns", 0) / 1e6

# ── Roofline helpers ──────────────────────────────────────────────────────────
def fc_int4_bytes(M_, K, N, g=G):
    """INT4 g=g FC bytes: input(f16) + weight(u4) + scale(f16) + zp(u4) + output(f16)."""
    return M_*K*2 + N*K//2 + N*(K//g)*2 + N*(K//g)//2 + M_*N*2

def fc_int8_bytes(M_, K, N, g=128):
    """INT8 g=g FC bytes: input(f16) + weight(u8) + scale(f16) + output(f16)."""
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
    """Weight bytes only (no io) for INT4 g=g."""
    return N_*K_//2 + N_*(K_//g)*2 + N_*(K_//g)//2

# ── MoE routed-only bytes/flops (shared expert NOT fused, runs as separate FC) ─
def moe_routed_bytes(tk=TK, ne=NE, h=H, i=I, g=G):
    """Decode MoE bytes: TK experts (routed only, SI=0) + router + io."""
    per_exp_w  = int4_wb(h, i, g) * 2 + int4_wb(i, h, g)  # gate+up+down
    router_w   = h * ne * 2  # FP16 router
    io = h * 2 + h * 2
    return tk * per_exp_w + router_w + io

def moe_routed_flops(tk=TK, h=H, i=I):
    return 2.0 * (2*tk*h*i + tk*i*h)

def moe_routed_prefill_bytes(s, tk=TK, ne=NE, h=H, i=I, g=G):
    """Prefill MoE bytes: all NE expert weights read + router + io."""
    per_exp_w  = int4_wb(h, i, g) * 2 + int4_wb(i, h, g)
    all_exp_w  = ne * per_exp_w
    router_w   = h * ne * 2
    io = s * h * 2 + s * h * 2
    return all_exp_w + router_w + io

def moe_routed_prefill_flops(s, tk=TK, h=H, i=I):
    return 2.0 * (2*s*tk*h*i + s*tk*i*h)

# ── Shared expert bytes/flops (3 separate INT4 g=64 FC ops) ──────────────────
def shared_int4_bytes(M_=1, h=H, si=SI, g=G):
    """Bytes for shared expert gate+up+down (all INT4 g=64)."""
    return fc_int4_bytes(M_, h, si, g) + fc_int4_bytes(M_, h, si, g) + fc_int4_bytes(M_, si, h, g)

def shared_int4_flops(M_=1, h=H, si=SI):
    return fc_flops(M_, h, si)*2 + fc_flops(M_, si, h)

# ── GDN bytes/flops ───────────────────────────────────────────────────────────
def gdn_bytes(t=1, hk=HK, kd=KD, vd=VD):
    state = hk * kd * vd * 2
    return 2 * state + t * H * 2

def gdn_flops(t=1, hk=HK, kd=KD, vd=VD):
    return 2.0 * hk * kd * vd * t

# ── PA decode bytes/flops ─────────────────────────────────────────────────────
def pa_decode_bytes(kv, nh=NH, nkv=NKV, hd=HD):
    return 2 * nkv * hd * kv * 1   # INT8 KV cache

def pa_decode_flops(kv, nh=NH, hd=HD):
    return 2.0 * nh * hd * kv * 2

# ── Decode per-op rows ────────────────────────────────────────────────────────
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

    # FC_QKV INT4 g=64
    m = ms("fc_qkv_decode_M1")
    add("FC_QKV (INT4 g=64)", "gemm_kernel", m, NF,
        fc_flops(1, H, 5120), fc_int4_bytes(1, H, 5120))

    # FC_O INT4 g=64
    m = ms("fc_o_decode_M1")
    add("FC_O (INT4 g=64)", "gemm_kernel", m, NF,
        fc_flops(1, NH*HD, H), fc_int4_bytes(1, NH*HD, H))

    # FC linattn projections (all INT4 g=64)
    m = ms("linattn_qkv_decode_M1")
    add("FC_linattn_qkv (INT4 g=64)", "gemm_kernel", m, NL_LIN,
        fc_flops(1, H, 8192), fc_int4_bytes(1, H, 8192))
    m = ms("linattn_z_decode_M1")
    add("FC_linattn_z (INT4 g=64)", "gemm_kernel", m, NL_LIN,
        fc_flops(1, H, 4096), fc_int4_bytes(1, H, 4096))
    m = ms("linattn_a_decode_M1")
    add("FC_linattn_a (INT4 g=64)", "gemm_kernel", m, NL_LIN,
        fc_flops(1, H, 32), fc_int4_bytes(1, H, 32))
    m = ms("linattn_b_decode_M1")
    add("FC_linattn_b (INT4 g=64)", "gemm_kernel", m, NL_LIN,
        fc_flops(1, H, 32), fc_int4_bytes(1, H, 32))
    m = ms("linattn_out_decode_M1")
    add("FC_linattn_out (INT4 g=64)", "gemm_kernel", m, NL_LIN,
        fc_flops(1, 4096, H), fc_int4_bytes(1, 4096, H))

    # LM head INT8 g=128
    m = ms("lm_head_decode_M1")
    add("LM_head (INT8 g=128)", "gemm_kernel", m, 1,
        fc_flops(1, H, VOCAB), fc_int8_bytes(1, H, VOCAB))

    # MoE routed-only (SI=0, INT4 g=64)
    m = ms("moe_routed_decode_M1")
    add("MoE3GEMM_fused (routed, INT4 g=64)", "moe_3gemm_swiglu_mlp", m, NL,
        moe_routed_flops(), moe_routed_bytes())

    # Shared expert: 3 separate INT4 g=64 FC ops
    mg = ms("shared_gate_decode_M1")
    mu = ms("shared_up_decode_M1")
    md = ms("shared_down_decode_M1")
    ms_sh = mg + mu + md
    add("SharedExpert_gate+up+down (INT4 g=64 FC×3)", "gemm_kernel×3", ms_sh, NL,
        shared_int4_flops(), shared_int4_bytes())

    # PA decode
    m = ms(f"pa_decode_kv{kv}")
    add("PagedAttention (INT8 KV)", "paged_attention_opt__single_token", m, NF,
        pa_decode_flops(kv), pa_decode_bytes(kv))

    # GDN decode
    m = ms("gdn_decode_T1")
    add("GatedDeltaNet", "gated_delta_net_ref_sa", m, NL_LIN,
        gdn_flops(), gdn_bytes())

    # Small ops
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

# ── Prefill per-op rows ──────────────────────────────────────────────────────
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

    # FC_QKV INT4 g=64
    m = ms(f"fc_qkv_prefill_{sk}")
    add("FC_QKV (INT4 g=64)", "dq+gemm_kernel", m, NF,
        fc_flops(s, H, 5120), fc_int4_bytes(s, H, 5120), int8_xmx=True)

    # FC_O INT4 g=64
    m = ms(f"fc_o_prefill_{sk}")
    add("FC_O (INT4 g=64)", "dq+gemm_kernel", m, NF,
        fc_flops(s, NH*HD, H), fc_int4_bytes(s, NH*HD, H), int8_xmx=True)

    # FC linattn (all INT4 g=64)
    m = ms(f"linattn_qkv_prefill_{sk}")
    add("FC_linattn_qkv (INT4 g=64)", "dq+gemm_kernel", m, NL_LIN,
        fc_flops(s, H, 8192), fc_int4_bytes(s, H, 8192), int8_xmx=True)
    m = ms(f"linattn_z_prefill_{sk}")
    add("FC_linattn_z (INT4 g=64)", "dq+gemm_kernel", m, NL_LIN,
        fc_flops(s, H, 4096), fc_int4_bytes(s, H, 4096), int8_xmx=True)
    # linattn_a/b prefill only up to S=8192 (tiny ops)
    if s <= 8192:
        m_a = ms(f"linattn_a_prefill_{sk}")
        m_b = ms(f"linattn_b_prefill_{sk}")
    else:
        # Extrapolate linearly from S=8192
        scale = s / 8192
        m_a = ms("linattn_a_prefill_S8192") * scale
        m_b = ms("linattn_b_prefill_S8192") * scale
    add("FC_linattn_a (INT4 g=64)", "dq+gemm_kernel", m_a, NL_LIN,
        fc_flops(s, H, 32), fc_int4_bytes(s, H, 32), int8_xmx=True)
    add("FC_linattn_b (INT4 g=64)", "dq+gemm_kernel", m_b, NL_LIN,
        fc_flops(s, H, 32), fc_int4_bytes(s, H, 32), int8_xmx=True)
    m = ms(f"linattn_out_prefill_{sk}")
    add("FC_linattn_out (INT4 g=64)", "dq+gemm_kernel", m, NL_LIN,
        fc_flops(s, 4096, H), fc_int4_bytes(s, 4096, H), int8_xmx=True)

    # MoE routed-only (SI=0, INT4 g=64)
    m = ms(f"moe_routed_prefill_{sk}")
    add("MoE3GEMM_fused (routed, INT4 g=64)", "grouped_micro_gemm+scatter/gather", m, NL,
        moe_routed_prefill_flops(s), moe_routed_prefill_bytes(s), int8_xmx=True)

    # Shared expert: 3 separate INT4 g=64 FC ops (combined)
    mg = ms(f"shared_gate_prefill_{sk}")
    mu = ms(f"shared_up_prefill_{sk}")
    md_val = ms(f"shared_down_prefill_{sk}")
    ms_sh = mg + mu + md_val
    add("SharedExpert_gate+up+down (INT4 g=64 FC×3)", "dq+gemm_kernel×3", ms_sh, NL,
        shared_int4_flops(s), shared_int4_bytes(s), int8_xmx=True)

    # PA prefill
    m = ms(f"pa_prefill_{sk}")
    pa_b = 2 * NKV * HD * s
    pa_fl = 2 * NH * HD * s * s
    add("PagedAttention (FP16, causal)", "sdpa_micro__prefill", m, NF,
        pa_fl, pa_b)

    # GDN prefill
    m = ms(f"gdn_prefill_{sk}")
    if m == 0 and s == 131072:
        m = ms("gdn_prefill_S65536") * 2
    add("GatedDeltaNet", "gated_delta_net_ref_sa", m, NL_LIN,
        gdn_flops(s), gdn_bytes(s))

    # LM head (same as decode, 1 token)
    lm_ms = ms("lm_head_decode_M1")
    lm_r = roofline(lm_ms, fc_flops(1,H,VOCAB), fc_int8_bytes(1,H,VOCAB))
    rows.append(dict(op="LM_head (INT8 g=128, 1 out tok)", kernel="gemm_kernel",
                     single_ms=lm_ms, calls=1, total_ms=lm_ms,
                     gflops=lm_r['gflops'], gbs=lm_r['gbs'],
                     eff=lm_r['eff'], bound=lm_r['bound'],
                     ai=lm_r['ai'], eff_bw=lm_r['eff_bw'], eff_xmx=lm_r['eff_xmx']))

    # Small ops
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
    "config": "INT4 asymmetric g=64 all body FC + shared expert; LM head INT8 g=128",
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

# ── Build SUMMARY markdown following SUMMARY_TEMPLATE.md ─────────────────────
def build_summary():
    lines = []

    # ── Title & header ────────────────────────────────────────────────────
    lines += [
        "# Qwen3.5-MoE-35B-A3B (qwen3_5_moe) — Roofline on PTL 12Xe (2026-05-10, v3 g=64)",
        "",
        "**Platform**: PTL (Panther Lake B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s)",
        "**Model**: Qwen/Qwen3.5-MoE-35B-A3B (Hybrid Attention: 10 full-attn + 30 GatedDeltaNet, 256 experts × top-8)",
        "",
        "- hidden_size = 2048, 40 layers (10 full-attn + 30 linear-attn GDN), 16 Q heads, 2 KV heads (GQA=8)",
        "- MoE: 256 experts × top-8, intermediate=512, shared expert intermediate=512",
        "- MatMul weights **INT4 asymmetric g=64** / FP16 act; LM_head INT8 g=128 / FP16 act; KV cache INT8",
        "- SDPA: PagedAttention (INT8 KV cache, block_size=16)",
        "",
    ]

    # ── Model parameters & weight shapes ──────────────────────────────────
    lines += [
        "## Model parameters & weight shapes",
        "",
        "Architecture knobs (parsed from model config):",
        "",
        "| Field | Value | Notes |",
        "|---|---:|---|",
        f"| `hidden_size` | {H} | residual / activation channel |",
        f"| `num_hidden_layers` | {NL} | decoder blocks (hybrid) |",
        f"| `num_attention_heads` (NH) | {NH} | Q heads (full-attn) |",
        f"| `num_key_value_heads` (NKV) | {NKV} | GQA: 8-way Q-per-KV sharing |",
        f"| `head_dim` (HD) | {HD} | Q_dim = {NH}×{HD} = {NH*HD}, KV_dim = {NKV}×{HD} = {NKV*HD} |",
        f"| `linear_num_key_heads` | 16 | GDN Q/K heads |",
        f"| `linear_num_value_heads` | {HK} | GDN V heads |",
        f"| `linear_key_head_dim` | {KD} | GDN key dim per head |",
        f"| `linear_value_head_dim` | {VD} | GDN value dim per head |",
        f"| `moe_intermediate_size` | {I} | per-expert FFN hidden |",
        f"| `shared_expert_intermediate_size` | {SI} | shared expert hidden |",
        f"| `num_experts` | {NE} | total experts |",
        f"| `num_experts_per_tok` | {TK} | active experts per token |",
        f"| `vocab_size` | {VOCAB:,} | LM head N |",
        "| `hidden_act` | SwiGLU | gate·up·down pattern |",
        "| Layer pattern | linear_attn×3 → full_attn×1 | 10 full-attn + 30 GDN layers |",
        "",
    ]

    # Weight table
    def mb(b): return b / 1024**2

    lines += [
        "Per-layer weight matrices and global weights:",
        "",
        "| Weight | Shape (K × N) | Quant | Bytes / instance | × Layers | Total MB |",
        "|---|---:|---|---:|---:|---:|",
    ]

    # FC_QKV: full-attn layers × (2048, 5120) INT4 g=64
    fcqkv_b = int4_wb(H, 5120)
    lines.append(f"| FC_QKV (fused Q+K+V proj) | {H}×5120 | INT4 g={G} | {fcqkv_b:,} | {NF} | {mb(fcqkv_b*NF):.1f} |")
    # FC_O: full-attn layers × (4096, 2048) INT4 g=64
    fco_b = int4_wb(NH*HD, H)
    lines.append(f"| FC_O (attention output) | {NH*HD}×{H} | INT4 g={G} | {fco_b:,} | {NF} | {mb(fco_b*NF):.1f} |")
    # FC linattn projections
    fcl_qkv_b = int4_wb(H, 8192)
    lines.append(f"| FC_linattn_qkv (in_proj_qkv) | {H}×8192 | INT4 g={G} | {fcl_qkv_b:,} | {NL_LIN} | {mb(fcl_qkv_b*NL_LIN):.1f} |")
    fcl_z_b = int4_wb(H, 4096)
    lines.append(f"| FC_linattn_z (in_proj_z) | {H}×4096 | INT4 g={G} | {fcl_z_b:,} | {NL_LIN} | {mb(fcl_z_b*NL_LIN):.1f} |")
    fcl_a_b = int4_wb(H, 32)
    lines.append(f"| FC_linattn_a (in_proj_a) | {H}×32 | INT4 g={G} | {fcl_a_b:,} | {NL_LIN} | {mb(fcl_a_b*NL_LIN):.1f} |")
    fcl_b_b = int4_wb(H, 32)
    lines.append(f"| FC_linattn_b (in_proj_b) | {H}×32 | INT4 g={G} | {fcl_b_b:,} | {NL_LIN} | {mb(fcl_b_b*NL_LIN):.1f} |")
    fcl_out_b = int4_wb(4096, H)
    lines.append(f"| FC_linattn_out (out_proj) | 4096×{H} | INT4 g={G} | {fcl_out_b:,} | {NL_LIN} | {mb(fcl_out_b*NL_LIN):.1f} |")
    # MoE experts
    moe_exp_b = int4_wb(H, I) * 2 + int4_wb(I, H)  # gate + up + down
    lines.append(f"| MoE Expert gate+up+down | {H}×{I} / {I}×{H} | INT4 g={G} | {moe_exp_b:,}/expert | {NL}×{NE} | {mb(moe_exp_b*NL*NE):.1f} |")
    # Router
    router_b = H * NE * 2  # FP16
    lines.append(f"| Router | {H}×{NE} | FP16 | {router_b:,} | {NL} | {mb(router_b*NL):.1f} |")
    # Shared expert
    shared_b = int4_wb(H, SI) * 2 + int4_wb(SI, H)
    lines.append(f"| Shared Expert gate+up+down | {H}×{SI} / {SI}×{H} | INT4 g={G} | {shared_b:,} | {NL} | {mb(shared_b*NL):.1f} |")
    # LM head
    lmh_b = H*VOCAB + (H//128)*VOCAB*2
    lines.append(f"| LM_Head | {H}×{VOCAB:,} | INT8 g=128 | {lmh_b:,} | 1 | {mb(lmh_b):.1f} |")
    # Total
    total_w = (fcqkv_b*NF + fco_b*NF +
               (fcl_qkv_b + fcl_z_b + fcl_a_b + fcl_b_b + fcl_out_b)*NL_LIN +
               moe_exp_b*NL*NE + router_b*NL + shared_b*NL + lmh_b)
    lines.append(f"| **Total static weights** | | | | | **{mb(total_w):.0f} MB** |")

    # KV cache
    kv_per_tok_layer = 2 * NKV * HD * 1  # INT8
    lines += [
        "",
        "Activation / KV-cache shapes (S = sequence length, B = batch=1):",
        "",
        "| Tensor | Shape | dtype | Bytes / token / layer | Bytes / token (all layers) |",
        "|---|---|---|---:|---:|",
        f"| Hidden states | [B, S, {H}] | FP16 | {H*2} | — |",
        f"| Q | [B, S, {NH}, {HD}] | FP16 | {NH*HD*2} | — |",
        f"| K (cache) | [num_blocks, {NKV}, {HD}, block_size] | INT8 | {NKV*HD} | {NKV*HD*NL} |",
        f"| V (cache) | [num_blocks, {NKV}, block_size, {HD}] | INT8 | {NKV*HD} | {NKV*HD*NL} |",
        f"| **KV cache total** | per token | INT8 | {kv_per_tok_layer} B / layer | **{kv_per_tok_layer*NL} B / token ({kv_per_tok_layer*NL/1024:.1f} KB)** |",
        "",
        "> Note: Only 10 full-attention layers use KV cache. GDN layers (30) use recurrent state, not KV cache.",
        "> Effective KV cache per token = {0} B / token ({1:.1f} KB)".format(kv_per_tok_layer*NF, kv_per_tok_layer*NF/1024),
    ]

    # ── Theoretical roofline ──────────────────────────────────────────────
    lines += [
        "",
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

    # ── Data sources ──────────────────────────────────────────────────────
    lines += [
        "",
        "## Data sources",
        "",
        "| Op category | Source | Platform |",
        "|---|---|---|",
        "| FC_QKV, FC_O, FC_linattn_* (all INT4 g=64) | **Measured** (fc_bench, cliloader) | PTL 12Xe |",
        "| MoE routed (INT4 g=64, SI=0) | **Measured** (moe_bench, cliloader) | PTL 12Xe |",
        "| Shared expert (INT4 g=64, 3×FC) | **Measured** (fc_bench, cliloader) | PTL 12Xe |",
        "| LM_head (INT8 g=128) | **Measured** (fc_bench, cliloader) | PTL 12Xe |",
        "| PagedAttention (INT8 KV) | **Measured** (pa_bench, cliloader) | PTL 12Xe |",
        "| GatedDeltaNet | **Measured** (gdn_bench, cliloader) | PTL 12Xe |",
        "| SmallOps (rmsnorm, rope, add) | **Measured** (small_ops_bench, cliloader) | PTL 12Xe |",
        "",
        "> v2: All ops re-measured after OpenVINO code update (2026-05-10).",
    ]

    # ── Graph fusion notes ────────────────────────────────────────────────
    lines += [
        "",
        "## Graph fusion notes",
        "",
        "| Bench row | Real graph behaviour | Fused into | Standalone kernel? |",
        "|---|---|---|---|",
        "| FC_QKV / FC_O (INT4 g=64) | FullyConnectedCompressed | decode: `gemm_kernel`; prefill: `dq+gemm` | Yes |",
        "| FC_linattn_qkv/z/out (INT4 g=64) | FullyConnectedCompressed | decode: `gemm_kernel`; prefill: `dq+gemm` | Yes |",
        "| FC_linattn_a/b (INT4 g=64, tiny) | FullyConnectedCompressed | decode: `gemm_kernel`; prefill: `dq+gemm` | Yes |",
        "| MoE routed experts (INT4 g=64) | MOE3GemmFusedCompressed | **Fused** gate+up+down | Yes |",
        "| Shared Expert (INT4 g=64) | 3 × FullyConnectedCompressed | **NOT fused** | Yes (3 separate kernels) |",
        "| PagedAttention | PagedAttention | INT8 KV cache, GQA=8 | Yes |",
        "| GatedDeltaNet | GatedDeltaNet | Reference kernel | Yes |",
        "| SmallOps | rmsnorm/rope/add | Standalone | Yes |",
        "",
        "> **Shared expert NOT fused**: Even with INT4 g=64 weights, `FuseMOESharedExpert` cannot absorb",
        "> the shared expert into `MOE3GemmFusedCompressed`. The shared expert runs as 3 separate",
        "> `FullyConnectedCompressed` kernels per layer × 40 layers (same behaviour as FP16 variant).",
    ]

    # ── Token latency summary ─────────────────────────────────────────────
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
        toks = s / tot * 1000 if tot > 0 else 0
        per_tok = tot / s if s > 0 else 0
        lines.append(f"| {s:,} | {tot:.1f} | {tot/1000:.3f} | {per_tok:.4f} | {toks:.1f} |")

    lines += [
        "",
        "### Decode — TPOT (per output token)",
        "",
        "| KV (ctx) | TPOT (ms) | tokens/s |",
        "|---:|---:|---:|",
    ]
    for kv in KV_SIZES:
        tot = metrics["decode"][kv]["total_ms"]
        toks = 1000 / tot if tot > 0 else 0
        lines.append(f"| {kv:,} | {tot:.2f} | {toks:.1f} |")

    # ── Decode TPOT per-op breakdown ──────────────────────────────────────
    # Collect all unique op names across KV sizes
    all_ops = []
    for kv in KV_SIZES:
        for r in metrics["decode"][kv]["rows"]:
            if r['op'] not in all_ops:
                all_ops.append(r['op'])

    lines += [
        "",
        "### Decode TPOT — per-op breakdown (ms / % of TPOT)",
        "",
    ]
    kv_headers = " | ".join(f"kv={kv:,}" for kv in KV_SIZES)
    kv_seps    = " | ".join("---:" for _ in KV_SIZES)
    lines.append(f"| op | {kv_headers} |")
    lines.append(f"|---|{kv_seps}|")
    for op_name in all_ops:
        cells = []
        for kv in KV_SIZES:
            tot = metrics["decode"][kv]["total_ms"]
            op_ms = sum(r['total_ms'] for r in metrics["decode"][kv]["rows"] if r['op'] == op_name)
            if op_ms > 0:
                pct = op_ms / tot * 100 if tot > 0 else 0
                cells.append(f"{op_ms:.3f} ({pct:.1f}%)")
            else:
                cells.append("—")
        lines.append(f"| {op_name} | {' | '.join(cells)} |")

    # ── Prefill TTFT per-op breakdown ─────────────────────────────────────
    all_ops_pf = []
    for s in S_SIZES:
        for r in metrics["prefill"][s]["rows"]:
            if r['op'] not in all_ops_pf:
                all_ops_pf.append(r['op'])

    lines += [
        "",
        "### Prefill TTFT — per-op breakdown (ms / % of TTFT)",
        "",
    ]
    s_headers = " | ".join(f"S={s:,}" for s in S_SIZES)
    s_seps    = " | ".join("---:" for _ in S_SIZES)
    lines.append(f"| op | {s_headers} |")
    lines.append(f"|---|{s_seps}|")
    for op_name in all_ops_pf:
        cells = []
        for s in S_SIZES:
            tot = metrics["prefill"][s]["total_ms"]
            op_ms = sum(r['total_ms'] for r in metrics["prefill"][s]["rows"] if r['op'] == op_name)
            if op_ms > 0:
                pct = op_ms / tot * 100 if tot > 0 else 0
                cells.append(f"{op_ms:.1f} ({pct:.1f}%)")
            else:
                cells.append("—")
        lines.append(f"| {op_name} | {' | '.join(cells)} |")

    # ── Decode tables ─────────────────────────────────────────────────────
    lines += [
        "",
        "## Decode tables (1 query token, KV = context length)",
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

    # ── Prefill tables ────────────────────────────────────────────────────
    lines += [
        "## Prefill tables (single forward over S tokens)",
        "",
        "_Eff% for INT4 FC prefill = GFLOPS / INT8 XMX (117.965 TOPS); for FP16 SDPA = GFLOPS / FP16 XMX (58.982 TFLOPS)._",
        "",
    ]
    for s in S_SIZES:
        rows = metrics["prefill"][s]["rows"]
        tot  = metrics["prefill"][s]["total_ms"]
        toks = s / tot * 1000 if tot > 0 else 0
        lines.append(f"### Prefill — S={s:,}")
        lines.append("")
        lines.append("| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for r in rows:
            lines.append(md_table_row(r))
        lines.append(f"| **TOTAL** | | | | **{tot:.1f}** | | | | |")
        lines.append(f"| | | | | TTFT={tot:.0f} ms | | | **{toks:.1f} tok/s** | |")
        lines.append("")

    # ── Top contributors ──────────────────────────────────────────────────
    lines += [
        "## Top contributors (sorted by total ms per inference)",
        "",
        "### Decode",
        "",
        "| KV | top1 (ms,%) | top2 | top3 |",
        "|---:|---|---|---|",
    ]
    for kv in KV_SIZES:
        rows = sorted(metrics["decode"][kv]["rows"], key=lambda x: -x["total_ms"])
        tot  = metrics["decode"][kv]["total_ms"]
        tops = []
        for r in rows[:3]:
            pct = r['total_ms'] / tot * 100 if tot > 0 else 0
            tops.append(f"{r['op']} {r['total_ms']:.2f}ms ({pct:.1f}%)")
        while len(tops) < 3:
            tops.append("—")
        lines.append(f"| {kv:,} | {tops[0]} | {tops[1]} | {tops[2]} |")

    lines += [
        "",
        "### Prefill",
        "",
        "| S | top1 (ms,%) | top2 | top3 |",
        "|---:|---|---|---|",
    ]
    for s in S_SIZES:
        rows = sorted(metrics["prefill"][s]["rows"], key=lambda x: -x["total_ms"])
        tot  = metrics["prefill"][s]["total_ms"]
        tops = []
        for r in rows[:3]:
            pct = r['total_ms'] / tot * 100 if tot > 0 else 0
            tops.append(f"{r['op']} {r['total_ms']:.1f}ms ({pct:.1f}%)")
        while len(tops) < 3:
            tops.append("—")
        lines.append(f"| {s:,} | {tops[0]} | {tops[1]} | {tops[2]} |")

    # ── Comparison with F16 QKV/O variant ─────────────────────────────────
    lines += [
        "",
        "## Comparison with other platforms",
        "",
        "### vs F16 QKV/O variant (same PTL 12Xe platform)",
        "",
        "| Config | kv=1024 (ms) | kv=4096 (ms) | kv=32768 (ms) | kv=131072 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    # Reference F16 QKV/O totals from previous session
    f16_ref = {1024: 27.19, 4096: 28.16, 32768: 32.48, 131072: 49.40}
    int4_vals = {kv: metrics["decode"][kv]["total_ms"] for kv in [1024, 4096, 32768, 131072]}
    f16_cells = " | ".join(f"{f16_ref[kv]:.2f}" for kv in [1024, 4096, 32768, 131072])
    int4_cells = " | ".join(f"{int4_vals[kv]:.2f}" for kv in [1024, 4096, 32768, 131072])
    lines.append(f"| F16 QKV/O + F16 shared (3×FC) | {f16_cells} |")
    lines.append(f"| **INT4 g=64 all (this report)** | {int4_cells} |")
    delta_cells = " | ".join(f"{int4_vals[kv]-f16_ref[kv]:+.2f}" for kv in [1024, 4096, 32768, 131072])
    lines.append(f"| Delta (INT4g64 − F16) | {delta_cells} |")

    # ── Caveats & method ──────────────────────────────────────────────────
    lines += [
        "",
        "## Caveats & method",
        "",
        "- Each op profiled in its own process via cliloader Device Performance Timing; we use mean kernel time per iteration.",
        "- FC weight bytes count INT4 weight + FP16 scale (per group) + INT4 zero-point (per group) + FP16 activations.",
        f"- All body FC weights use INT4 asymmetric g={G}. LM head uses INT8 g=128.",
        "- PA bytes assume INT8 KV cache + FP16 Q, FP16 out.",
        "- Decode FC is treated as **memory-bound** (weights read dominates at M=1); prefill FC is **INT8 XMX compute-bound** (S big enough to hit XMX).",
        "- Prefill PA at S≥2048 is compute-bound (FP16 micro-kernel); decode PA is memory-bound.",
        "- lm_head is run only once per token (last position in prefill, every step in decode).",
        "- v2: All ops (FC, MoE, PA, GDN, small ops) re-measured after OpenVINO code update (2026-05-10).",
        "- Shared expert with INT4 g=64 weights still cannot be fused into MOE3GemmFusedCompressed primitive.",
        "- Shared expert runs as 3 separate FullyConnectedCompressed kernels per layer × 40 layers (same as F16 variant).",
        "- MoE routed experts are benchmarked with SI=0 (no shared expert); shared expert benchmarked separately as 3 FC ops.",
        "- GDN prefill S=131072 may be extrapolated ×2 from S=65536 if the original run failed.",
        "- Target machine: Local_Admin@10.239.132.229 (PTL B390 iGPU, 12 Xe cores)",
    ]

    # ── Reproduction ──────────────────────────────────────────────────────
    lines += [
        "",
        "## Reproduction",
        "",
        "```bat",
        "REM On PTL Windows machine:",
        "set OV_BIN=D:\\river\\moe\\openvino\\release_install\\runtime\\bin\\intel64\\Release",
        "set TBB=D:\\river\\moe\\openvino\\temp\\Windows_AMD64\\tbb\\bin",
        "set CLI=C:\\Users\\Local_Admin\\Downloads\\clintercept-3.0.6-win64\\Release\\cliloader.exe",
        "set BUILD=D:\\river\\moe\\dev_roofline_profiling\\utils\\build\\Release",
        "",
        f"REM FC example: fc_bench M K N group_size iters warmup num_bufs precision flush_mb",
        f'REM   %CLI% -d %BUILD%\\fc_bench.exe 1 2048 5120 {G} 15000 500 8 u4 32',
        f"REM MoE example: moe_bench B S H I NE TK group_size iters warmup num_bufs flush_mb shared_I shared_quant",
        f'REM   %CLI% -d %BUILD%\\moe_bench.exe 1 1 2048 512 256 8 {G} 100 10 4 64 512 u4',
        "",
        "REM Full script: run_qwen3_5_moe_ptl_int4g64_v2.bat",
        "```",
        "",
        "_Generated by `build_report_ptl_int4g64.py`_",
    ]

    return "\n".join(lines)

summary = build_summary()
OUT_SUMMARY.write_text(summary)
print(f"Saved {OUT_SUMMARY}")

# ── Quick console summary ─────────────────────────────────────────────────────
print("\n=== DECODE TOTALS (PTL INT4 g=64) ===")
print(f"{'kv':>8}  {'total ms':>10}  {'tok/s':>8}")
for kv in KV_SIZES:
    tot = metrics["decode"][kv]["total_ms"]
    print(f"{kv:>8,}  {tot:>10.2f}  {1000/tot:>8.1f}")

print("\n=== PREFILL TOTALS ===")
print(f"{'S':>8}  {'TTFT ms':>12}  {'TTFT s':>8}  {'tok/s':>8}")
for s in S_SIZES:
    tot = metrics["prefill"][s]["total_ms"]
    print(f"{s:>8,}  {tot:>12.1f}  {tot/1000:>8.3f}  {s/tot*1000:>8.1f}")
