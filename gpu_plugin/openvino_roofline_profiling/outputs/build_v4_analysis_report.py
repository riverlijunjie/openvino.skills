#!/usr/bin/env python3
"""
Generate v4 analysis report for Qwen3.5-MoE on PTL 12Xe (B390 iGPU).
v4: Shared expert FUSED into MoE kernel (shared_I=512).
Compares g=64 vs g=128, and v4 (fused) vs v3 (unfused) for MoE layer.
"""
import json, math
from pathlib import Path

OUT = Path(__file__).resolve().parent
G64_V4  = json.load(open(OUT / "qwen3_5_moe_int4g64" / "parsed_ptl_int4g64_v4.json"))
G128_V4 = json.load(open(OUT / "qwen3_5_moe_int4g128" / "parsed_ptl_int4g128_v4.json"))
G64_V3  = json.load(open(OUT / "qwen3_5_moe_int4g64" / "parsed_ptl_int4g64_v3.json"))
G128_V3 = json.load(open(OUT / "qwen3_5_moe_int4g128" / "parsed_ptl_int4g128_v3.json"))

# ── Hardware ──────────────────────────────────────────────────────────────────
BW   = 110.0       # GB/s
FP16 = 58.9824     # TFLOPS
INT8 = 117.9648    # TOPS

# ── Model ─────────────────────────────────────────────────────────────────────
H = 2048; NH = 16; NKV = 2; HD = 256; HK = 32; KD = 128; VD = 128
I = 512; SI = 512; NE = 256; TK = 8; VOCAB = 248320
NL = 40; NF = 10; NL_LIN = 30

KV_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
S_SIZES  = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

def ms(data, key):
    return data.get(key, {}).get("total_kernel_ns", 0) / 1e6

def int4_wb(K, N, g):
    return N*K//2 + N*(K//g)*2 + N*(K//g)//2

def fc_int4_bytes(M, K, N, g):
    return M*K*2 + N*K//2 + N*(K//g)*2 + N*(K//g)//2 + M*N*2

def fc_int8_bytes(M, K, N, g=128):
    return M*K*2 + N*K + N*(K//g)*2 + M*N*2

def fc_flops(M, K, N):
    return 2.0 * M * K * N

def moe_fused_bytes_decode(g):
    """Fused MoE (routed+shared) decode: TK routed + 1 shared expert weights + router + io."""
    per_exp_w = int4_wb(H, I, g)*2 + int4_wb(I, H, g)
    shared_w  = int4_wb(H, SI, g)*2 + int4_wb(SI, H, g)  # gate+up+down for shared
    router_w  = H * NE * 2
    io = H*2 + H*2
    return TK * per_exp_w + shared_w + router_w + io

def moe_fused_flops_decode():
    routed = 2.0 * (2*TK*H*I + TK*I*H)
    shared = 2.0 * (2*H*SI + SI*H)
    return routed + shared

def moe_fused_bytes_prefill(s, g):
    per_exp_w = int4_wb(H, I, g)*2 + int4_wb(I, H, g)
    shared_w  = int4_wb(H, SI, g)*2 + int4_wb(SI, H, g)
    all_exp_w = NE * per_exp_w
    router_w  = H * NE * 2
    io = s*H*2 + s*H*2
    return all_exp_w + shared_w + router_w + io

def moe_fused_flops_prefill(s):
    routed = 2.0 * (2*s*TK*H*I + s*TK*I*H)
    shared = 2.0 * (2*s*H*SI + s*SI*H)
    return routed + shared

def roofline(ms_val, flops, byts, int8_xmx=False):
    if ms_val <= 0:
        return dict(gbs=0, gflops=0, eff_bw=0, eff_xmx=0, ai=0, bound="N/A")
    ai     = flops / byts if byts > 0 else 0
    gflops = flops / (ms_val * 1e-3) / 1e9
    gbs    = byts  / (ms_val * 1e-3) / 1e9
    peak   = INT8 * 1e12 if int8_xmx else FP16 * 1e12
    ridge  = (INT8 if int8_xmx else FP16) * 1e12 / (BW * 1e9)
    eff_xmx = gflops * 1e9 / peak * 100
    eff_bw  = gbs / BW * 100
    bound   = "memory" if ai < ridge else "compute"
    eff     = eff_bw if bound == "memory" else eff_xmx
    return dict(gbs=gbs, gflops=gflops, eff_bw=eff_bw, eff_xmx=eff_xmx, ai=ai, bound=bound, eff=eff)


# ── Decode total for a given config ──────────────────────────────────────────
def decode_total(data, g, kv, version="v4"):
    """Compute decode total ms for a given config and KV length."""
    fc_qkv      = ms(data, "fc_qkv_decode_M1")
    fc_o        = ms(data, "fc_o_decode_M1")
    linattn_qkv = ms(data, "linattn_qkv_decode_M1")
    linattn_z   = ms(data, "linattn_z_decode_M1")
    linattn_a   = ms(data, "linattn_a_decode_M1")
    linattn_b   = ms(data, "linattn_b_decode_M1")
    linattn_out = ms(data, "linattn_out_decode_M1")
    lm_head     = ms(data, "lm_head_decode_M1")

    if version == "v4":
        moe = ms(data, "moe_fused_decode_M1")
    else:
        moe_r = ms(data, "moe_routed_decode_M1")
        sg = ms(data, "shared_gate_decode_M1")
        su = ms(data, "shared_up_decode_M1")
        sd = ms(data, "shared_down_decode_M1")
        moe = moe_r + sg + su + sd

    # PA, GDN, small_ops: use g=64 v4 data (group-size independent)
    g64_src = G64_V4 if version == "v4" else G64_V3
    pa    = ms(g64_src, f"pa_decode_kv{kv}")
    gdn   = ms(g64_src, "gdn_decode_T1")
    rms   = ms(g64_src, "so_rmsnorm_h2048_decode")
    add_  = ms(g64_src, "so_add_decode")
    rq    = ms(g64_src, "so_rope_q_decode")
    rk    = ms(g64_src, "so_rope_k_decode")
    qn    = ms(g64_src, "so_rmsnorm3d_qnorm_decode")
    kn    = ms(g64_src, "so_rmsnorm3d_knorm_decode")
    small = rms*2*NL + add_*2*NL + rq*NF + rk*NF + qn*NF + kn*NF

    total = (fc_qkv*NF + fc_o*NF +
             linattn_qkv*NL_LIN + linattn_z*NL_LIN +
             linattn_a*NL_LIN + linattn_b*NL_LIN + linattn_out*NL_LIN +
             lm_head +
             moe*NL +
             pa*NF + gdn*NL_LIN + small)
    return total

# ── Prefill total for a given config ─────────────────────────────────────────
def prefill_total(data, g, s, version="v4"):
    sk = f"S{s}"
    fc_qkv      = ms(data, f"fc_qkv_prefill_{sk}")
    fc_o        = ms(data, f"fc_o_prefill_{sk}")
    linattn_qkv = ms(data, f"linattn_qkv_prefill_{sk}")
    linattn_z   = ms(data, f"linattn_z_prefill_{sk}")
    linattn_out = ms(data, f"linattn_out_prefill_{sk}")

    if s <= 8192:
        linattn_a = ms(data, f"linattn_a_prefill_{sk}")
        linattn_b = ms(data, f"linattn_b_prefill_{sk}")
    else:
        scale = s / 8192
        linattn_a = ms(data, "linattn_a_prefill_S8192") * scale
        linattn_b = ms(data, "linattn_b_prefill_S8192") * scale

    if version == "v4":
        moe = ms(data, f"moe_fused_prefill_{sk}")
    else:
        moe_r = ms(data, f"moe_routed_prefill_{sk}")
        sg = ms(data, f"shared_gate_prefill_{sk}")
        su = ms(data, f"shared_up_prefill_{sk}")
        sd = ms(data, f"shared_down_prefill_{sk}")
        moe = moe_r + sg + su + sd

    g64_src = G64_V4 if version == "v4" else G64_V3
    pa  = ms(g64_src, f"pa_prefill_{sk}")
    gdn = ms(g64_src, f"gdn_prefill_{sk}")
    if gdn == 0 and s == 131072:
        gdn = ms(g64_src, "gdn_prefill_S65536") * 2
    lm  = ms(g64_src, "lm_head_decode_M1")

    if s <= 8192:
        rms_p = ms(g64_src, f"so_rmsnorm_h2048_prefill_{sk}")
        rq_p  = ms(g64_src, f"so_rope_q_prefill_{sk}")
    else:
        scale = s / 8192
        rms_p = ms(g64_src, "so_rmsnorm_h2048_prefill_S8192") * scale
        rq_p  = ms(g64_src, "so_rope_q_prefill_S8192") * scale
    small = rms_p*2*NL + rq_p*NF

    total = (fc_qkv*NF + fc_o*NF +
             linattn_qkv*NL_LIN + linattn_z*NL_LIN +
             linattn_a*NL_LIN + linattn_b*NL_LIN + linattn_out*NL_LIN +
             moe*NL +
             pa*NF + gdn*NL_LIN + lm + small)
    return total


# ── Generate report ──────────────────────────────────────────────────────────
lines = []
def w(s=""): lines.append(s)

w("# Qwen3.5-MoE-35B-A3B — INT4 Performance Analysis Report (v4)")
w()
w("## Platform: PTL 12Xe (Intel Arc B390 iGPU)")
w()
w("| Parameter | Value |")
w("|---|---|")
w("| Xe Cores | 12 |")
w("| EU/Core | 8 |")
w("| Threads/EU | 10 |")
w("| Frequency | 2400 MHz |")
w("| Memory BW | 110 GB/s |")
w("| FP16 XMX peak | 58.98 TFLOPS |")
w("| INT8 XMX peak | 117.96 TOPS |")
w()

w("## Configuration")
w()
w("| Component | Precision | Notes |")
w("|---|---|---|")
w("| Body FC (QKV, O, linattn projs) | INT4 asymmetric | g=64 and g=128 tested |")
w("| MoE (routed + shared expert) | INT4 asymmetric | **Fused** — shared_I=512, shared_quant=u4 |")
w("| LM head | INT8 asymmetric | g=128 (both configs) |")
w("| KV cache | INT8 | |")
w("| PA | FP16 causal SDPA | NH=16, NKV=2, HD=256 |")
w("| GDN | FP16 | HK=32, KD=VD=128 |")
w()
w("> **v4 key change**: Shared expert is now **fused into the MoE kernel** via `FuseMOESharedExpert` transformation (OV commit `2abcdce7f3`). In v3, shared expert ran as 3 separate FC ops.")
w()

# ── Section 1: Decode Latency Comparison ─────────────────────────────────────
w("## 1. Decode Latency — g=64 vs g=128 (v4, fused shared expert)")
w()
w("| KV Length | g=64 (ms) | g=128 (ms) | Delta | g=128 speedup |")
w("|---:|---:|---:|---:|---:|")

for kv in KV_SIZES:
    d64  = decode_total(G64_V4,  64,  kv, "v4")
    d128 = decode_total(G128_V4, 128, kv, "v4")
    delta = d128 - d64
    pct = (d128 - d64) / d64 * 100
    w(f"| {kv:,} | {d64:.2f} | {d128:.2f} | {delta:+.2f} | {pct:+.1f}% |")

w()

# ── Section 2: Prefill Latency Comparison ────────────────────────────────────
w("## 2. Prefill Latency — g=64 vs g=128 (v4, fused shared expert)")
w()
w("| Seq Length | g=64 (ms) | g=128 (ms) | Delta | g=128 speedup |")
w("|---:|---:|---:|---:|---:|")

for s in S_SIZES:
    p64  = prefill_total(G64_V4,  64,  s, "v4")
    p128 = prefill_total(G128_V4, 128, s, "v4")
    delta = p128 - p64
    pct = (p128 - p64) / p64 * 100
    w(f"| {s:,} | {p64:.1f} | {p128:.1f} | {delta:+.1f} | {pct:+.1f}% |")

w()

# ── Section 3: Token throughput ──────────────────────────────────────────────
w("## 3. Token Throughput")
w()
w("### Decode (tok/s)")
w()
w("| KV Length | g=64 tok/s | g=128 tok/s |")
w("|---:|---:|---:|")
for kv in [1024, 4096, 32768, 131072]:
    d64  = decode_total(G64_V4,  64,  kv, "v4")
    d128 = decode_total(G128_V4, 128, kv, "v4")
    w(f"| {kv:,} | {1000/d64:.1f} | {1000/d128:.1f} |")

w()
w("### Prefill (tok/s)")
w()
w("| Seq Length | g=64 tok/s | g=128 tok/s |")
w("|---:|---:|---:|")
for s in [1024, 4096, 32768, 131072]:
    p64  = prefill_total(G64_V4,  64,  s, "v4")
    p128 = prefill_total(G128_V4, 128, s, "v4")
    w(f"| {s:,} | {s*1000/p64:.0f} | {s*1000/p128:.0f} |")

w()

# ── Section 4: Per-Op Decode Breakdown (KV=4096) ────────────────────────────
w("## 4. Per-Op Decode Breakdown (KV=4096)")
w()
w("| Op | g=64 (ms) | g=128 (ms) | Calls | g=64 Total (ms) | g=128 Total (ms) | Delta % |")
w("|---|---:|---:|---:|---:|---:|---:|")

decode_ops = [
    ("FC_QKV",       "fc_qkv_decode_M1",     NF),
    ("FC_O",         "fc_o_decode_M1",        NF),
    ("linattn_qkv",  "linattn_qkv_decode_M1", NL_LIN),
    ("linattn_z",    "linattn_z_decode_M1",   NL_LIN),
    ("linattn_a",    "linattn_a_decode_M1",   NL_LIN),
    ("linattn_b",    "linattn_b_decode_M1",   NL_LIN),
    ("linattn_out",  "linattn_out_decode_M1", NL_LIN),
    ("LM_head",      "lm_head_decode_M1",     1),
    ("MoE_fused",    "moe_fused_decode_M1",   NL),
]

for name, key, calls in decode_ops:
    m64  = ms(G64_V4,  key)
    m128 = ms(G128_V4, key) if key in G128_V4 else m64  # PA/GDN from g64
    if m64 > 0:
        pct = (m128 - m64) / m64 * 100
    else:
        pct = 0
    w(f"| {name} | {m64:.4f} | {m128:.4f} | {calls} | {m64*calls:.3f} | {m128*calls:.3f} | {pct:+.1f}% |")

# Add PA, GDN as g-independent
pa_ms = ms(G64_V4, "pa_decode_kv4096")
gdn_ms = ms(G64_V4, "gdn_decode_T1")
w(f"| PA (KV=4096) | {pa_ms:.4f} | {pa_ms:.4f} | {NF} | {pa_ms*NF:.3f} | {pa_ms*NF:.3f} | 0.0% |")
w(f"| GDN | {gdn_ms:.4f} | {gdn_ms:.4f} | {NL_LIN} | {gdn_ms*NL_LIN:.3f} | {gdn_ms*NL_LIN:.3f} | 0.0% |")

# Small ops
g64_src = G64_V4
rms = ms(g64_src, "so_rmsnorm_h2048_decode")
add_ = ms(g64_src, "so_add_decode")
rq = ms(g64_src, "so_rope_q_decode")
rk = ms(g64_src, "so_rope_k_decode")
qn = ms(g64_src, "so_rmsnorm3d_qnorm_decode")
kn = ms(g64_src, "so_rmsnorm3d_knorm_decode")
small = rms*2*NL + add_*2*NL + rq*NF + rk*NF + qn*NF + kn*NF
w(f"| SmallOps | — | — | — | {small:.3f} | {small:.3f} | 0.0% |")

w()

# ── Section 5: MoE Fusion Impact (v3 unfused vs v4 fused) ───────────────────
w("## 5. MoE Layer: Fused (v4) vs Unfused (v3) Comparison")
w()
w("In v3, the shared expert ran as 3 separate INT4 FC ops (gate, up, down). In v4, it is fused into the MoE kernel.")
w()
w("### Decode (MoE layer only, per-layer)")
w()
w("| Config | v3 MoE routed | v3 shared (3×FC) | v3 Total | v4 Fused | Fusion speedup |")
w("|---|---:|---:|---:|---:|---:|")

for g, v3data, v4data in [(64, G64_V3, G64_V4), (128, G128_V3, G128_V4)]:
    moe_v3 = ms(v3data, "moe_routed_decode_M1")
    sg = ms(v3data, "shared_gate_decode_M1")
    su = ms(v3data, "shared_up_decode_M1")
    sd = ms(v3data, "shared_down_decode_M1")
    shared_v3 = sg + su + sd
    total_v3 = moe_v3 + shared_v3
    fused_v4 = ms(v4data, "moe_fused_decode_M1")
    pct = (fused_v4 - total_v3) / total_v3 * 100
    w(f"| g={g} | {moe_v3:.4f} | {shared_v3:.4f} | {total_v3:.4f} | {fused_v4:.4f} | {pct:+.1f}% |")

w()
w("### Prefill (MoE layer only, per-layer)")
w()
w("| Config | Seq Len | v3 MoE routed | v3 shared (3×FC) | v3 Total | v4 Fused | Fusion Δ% |")
w("|---|---:|---:|---:|---:|---:|---:|")

for g, v3data, v4data in [(64, G64_V3, G64_V4), (128, G128_V3, G128_V4)]:
    for s in S_SIZES:
        sk = f"S{s}"
        moe_v3 = ms(v3data, f"moe_routed_prefill_{sk}")
        sg = ms(v3data, f"shared_gate_prefill_{sk}")
        su = ms(v3data, f"shared_up_prefill_{sk}")
        sd = ms(v3data, f"shared_down_prefill_{sk}")
        shared_v3 = sg + su + sd
        total_v3 = moe_v3 + shared_v3
        fused_v4 = ms(v4data, f"moe_fused_prefill_{sk}")
        if total_v3 > 0:
            pct = (fused_v4 - total_v3) / total_v3 * 100
        else:
            pct = 0
        w(f"| g={g} | {s:,} | {moe_v3:.3f} | {shared_v3:.3f} | {total_v3:.3f} | {fused_v4:.3f} | {pct:+.1f}% |")

w()

# ── Section 6: Full Pipeline v3 vs v4 ───────────────────────────────────────
w("## 6. End-to-End Pipeline: v3 vs v4")
w()
w("### Decode (KV=4096)")
w()
w("| Config | v3 (ms) | v4 (ms) | Delta |")
w("|---|---:|---:|---:|")
for g, v3data, v4data in [(64, G64_V3, G64_V4), (128, G128_V3, G128_V4)]:
    d_v3 = decode_total(v3data, g, 4096, "v3")
    d_v4 = decode_total(v4data, g, 4096, "v4")
    pct = (d_v4 - d_v3) / d_v3 * 100
    w(f"| g={g} | {d_v3:.2f} | {d_v4:.2f} | {pct:+.1f}% |")

w()
w("### Prefill")
w()
w("| Config | Seq Len | v3 (ms) | v4 (ms) | Delta |")
w("|---|---:|---:|---:|---:|")
for g, v3data, v4data in [(64, G64_V3, G64_V4), (128, G128_V3, G128_V4)]:
    for s in [1024, 4096, 32768, 131072]:
        p_v3 = prefill_total(v3data, g, s, "v3")
        p_v4 = prefill_total(v4data, g, s, "v4")
        pct = (p_v4 - p_v3) / p_v3 * 100
        w(f"| g={g} | {s:,} | {p_v3:.1f} | {p_v4:.1f} | {pct:+.1f}% |")

w()

# ── Section 7: Roofline Analysis for Key Ops ────────────────────────────────
w("## 7. Roofline Analysis — Key Ops")
w()
w("### MoE Fused Decode (M=1)")
w()
w("| Config | Latency (ms) | GFLOPS | GB/s | BW Eff | XMX Eff | Bound |")
w("|---|---:|---:|---:|---:|---:|---|")
for g, data in [(64, G64_V4), (128, G128_V4)]:
    m = ms(data, "moe_fused_decode_M1")
    r = roofline(m, moe_fused_flops_decode(), moe_fused_bytes_decode(g))
    w(f"| g={g} | {m:.4f} | {r['gflops']:.1f} | {r['gbs']:.1f} | {r['eff_bw']:.1f}% | {r['eff_xmx']:.2f}% | {r['bound']} |")

w()
w("### FC_QKV Decode (M=1, K=2048, N=5120)")
w()
w("| Config | Latency (ms) | GFLOPS | GB/s | BW Eff | Bound |")
w("|---|---:|---:|---:|---:|---|")
for g, data in [(64, G64_V4), (128, G128_V4)]:
    m = ms(data, "fc_qkv_decode_M1")
    r = roofline(m, fc_flops(1, H, 5120), fc_int4_bytes(1, H, 5120, g))
    w(f"| g={g} | {m:.4f} | {r['gflops']:.1f} | {r['gbs']:.1f} | {r['eff_bw']:.1f}% | {r['bound']} |")

w()
w("### MoE Fused Prefill (S=4096)")
w()
w("| Config | Latency (ms) | GFLOPS | GB/s | XMX Eff | Bound |")
w("|---|---:|---:|---:|---:|---|")
for g, data in [(64, G64_V4), (128, G128_V4)]:
    m = ms(data, "moe_fused_prefill_S4096")
    r = roofline(m, moe_fused_flops_prefill(4096), moe_fused_bytes_prefill(4096, g), int8_xmx=True)
    w(f"| g={g} | {m:.3f} | {r['gflops']:.1f} | {r['gbs']:.1f} | {r['eff_xmx']:.1f}% | {r['bound']} |")

w()

# ── Section 8: Per-Op Prefill Breakdown (S=4096) ────────────────────────────
w("## 8. Per-Op Prefill Breakdown (S=4096)")
w()
w("| Op | g=64 (ms) | g=128 (ms) | Calls | g=64 Total (ms) | g=128 Total (ms) | Delta % |")
w("|---|---:|---:|---:|---:|---:|---:|")

prefill_ops = [
    ("FC_QKV",       "fc_qkv_prefill_S4096",     NF),
    ("FC_O",         "fc_o_prefill_S4096",        NF),
    ("linattn_qkv",  "linattn_qkv_prefill_S4096", NL_LIN),
    ("linattn_z",    "linattn_z_prefill_S4096",   NL_LIN),
    ("linattn_a",    "linattn_a_prefill_S4096",   NL_LIN),
    ("linattn_b",    "linattn_b_prefill_S4096",   NL_LIN),
    ("linattn_out",  "linattn_out_prefill_S4096", NL_LIN),
    ("MoE_fused",    "moe_fused_prefill_S4096",   NL),
]

for name, key, calls in prefill_ops:
    m64  = ms(G64_V4,  key)
    m128 = ms(G128_V4, key) if key in G128_V4 else m64
    if m64 > 0:
        pct = (m128 - m64) / m64 * 100
    else:
        pct = 0
    w(f"| {name} | {m64:.3f} | {m128:.3f} | {calls} | {m64*calls:.2f} | {m128*calls:.2f} | {pct:+.1f}% |")

pa_ms = ms(G64_V4, "pa_prefill_S4096")
gdn_ms = ms(G64_V4, "gdn_prefill_S4096")
w(f"| PA (S=4096) | {pa_ms:.3f} | {pa_ms:.3f} | {NF} | {pa_ms*NF:.2f} | {pa_ms*NF:.2f} | 0.0% |")
w(f"| GDN | {gdn_ms:.3f} | {gdn_ms:.3f} | {NL_LIN} | {gdn_ms*NL_LIN:.2f} | {gdn_ms*NL_LIN:.2f} | 0.0% |")

w()

# ── Section 9: Key Findings ─────────────────────────────────────────────────
w("## 9. Key Findings")
w()

# Calculate key metrics
d64_4k  = decode_total(G64_V4, 64, 4096, "v4")
d128_4k = decode_total(G128_V4, 128, 4096, "v4")
d_pct = (d128_4k - d64_4k) / d64_4k * 100

p64_4k  = prefill_total(G64_V4, 64, 4096, "v4")
p128_4k = prefill_total(G128_V4, 128, 4096, "v4")
p_pct = (p128_4k - p64_4k) / p64_4k * 100

moe_v3_64 = ms(G64_V3, "moe_routed_decode_M1") + ms(G64_V3, "shared_gate_decode_M1") + ms(G64_V3, "shared_up_decode_M1") + ms(G64_V3, "shared_down_decode_M1")
moe_v4_64 = ms(G64_V4, "moe_fused_decode_M1")
moe_fusion_pct = (moe_v4_64 - moe_v3_64) / moe_v3_64 * 100

pf_v3_64_131k = ms(G64_V3, "moe_routed_prefill_S131072") + ms(G64_V3, "shared_gate_prefill_S131072") + ms(G64_V3, "shared_up_prefill_S131072") + ms(G64_V3, "shared_down_prefill_S131072")
pf_v4_64_131k = ms(G64_V4, "moe_fused_prefill_S131072")
pf_fusion_pct = (pf_v4_64_131k - pf_v3_64_131k) / pf_v3_64_131k * 100

w(f"1. **g=128 vs g=64 decode** (KV=4096): g=128 is **{-d_pct:.1f}% faster** ({d64_4k:.2f} ms → {d128_4k:.2f} ms)")
w(f"2. **g=128 vs g=64 prefill** (S=4096): g=128 is **{-p_pct:.1f}% faster** ({p64_4k:.1f} ms → {p128_4k:.1f} ms)")
w(f"3. **Shared expert fusion impact (decode, g=64)**: v4 fused MoE is **{-moe_fusion_pct:.1f}% {'faster' if moe_fusion_pct < 0 else 'slower'}** per layer ({moe_v3_64:.4f} → {moe_v4_64:.4f} ms)")
w(f"4. **Shared expert fusion impact (prefill S=131072, g=64)**: v4 fused is **{-pf_fusion_pct:.1f}% {'faster' if pf_fusion_pct < 0 else 'slower'}** per layer ({pf_v3_64_131k:.2f} → {pf_v4_64_131k:.2f} ms)")
w(f"5. **Decode throughput** (KV=4096): g=64 = {1000/d64_4k:.1f} tok/s, g=128 = {1000/d128_4k:.1f} tok/s")
w(f"6. **Prefill throughput** (S=4096): g=64 = {4096*1000/p64_4k:.0f} tok/s, g=128 = {4096*1000/p128_4k:.0f} tok/s")
w(f"7. **LM head dominates decode**: {ms(G64_V4, 'lm_head_decode_M1'):.2f} ms = {ms(G64_V4, 'lm_head_decode_M1')/d64_4k*100:.0f}% of total decode (KV=4096)")
w(f"8. **PA dominates long-context prefill**: PA S=131072 = {ms(G64_V4, 'pa_prefill_S131072')*NF:.0f} ms = {ms(G64_V4, 'pa_prefill_S131072')*NF / prefill_total(G64_V4, 64, 131072, 'v4') * 100:.0f}% of total")

w()
w("---")
w("*Report generated from v4 benchmark data. Platform: PTL 12Xe (B390 iGPU). OV branch: river/moe_expert_precision_f16_support with FuseMOESharedExpert fix.*")

# ── Write output ─────────────────────────────────────────────────────────────
report_path = OUT / "REPORT_qwen3_5_moe_v4_analysis.md"
report_path.write_text("\n".join(lines) + "\n")
print(f"Wrote {report_path}")
