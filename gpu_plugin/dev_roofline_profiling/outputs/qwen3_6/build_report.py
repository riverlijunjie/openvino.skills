#!/usr/bin/env python3
"""
Qwen3.6-35B-A3B roofline report builder for the dev_roofline_profiling skill.

Renders utils/template/SUMMARY_TEMPLATE.md with measured PTL 12Xe data:
  - ptl_metrics.json   (parse_logs.py output: {log_stem:{total_kernel_ns, per_kernel}})
  - ops_mapping.json   (model config + op map)

Workload: prompt {1024,2048,4096,8192}, generate 512 tokens. Decode tables use
the mid-window KV (P+256) so they reflect the *average* decode token during the
512-token generation window (the focus of this run). PA decode is also reported
across the full window (start P / mean P+256 / end P+512).

Run:  python3 build_report.py [ptl_metrics.json] > SUMMARY_qwen3_6_<date>.md
"""
import json
import sys
import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent
MET_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else (OUT / "ptl_metrics.json")
MET = json.loads(MET_PATH.read_text())
OM = json.loads((OUT / "ops_mapping.json").read_text())
CFG = OM["config"]
DATE = datetime.date.today().strftime("%Y%m%d")

# ---------------------------------------------------------------- PTL 12Xe roofline
# From dev_roofline_profiling/SKILL.md (PTL 12Xe == Intel Arc B390, cliloader id).
# FP16 XMX = 12 Xe * 8 EU * 256 FLOP/cyc * 2.40 GHz = 58.982 TFLOPS
# INT8 XMX = 12 Xe * 8 EU * 512 OP/cyc  * 2.40 GHz = 117.964 TOPS
PLAT_NAME = "Intel Arc B390 (Panther Lake, 12Xe iGPU)"
ARCH = "Xe2"
BW = 105.0          # GB/s achievable streaming read (used for BW efficiency)
BW_SPEC = 110.0     # GB/s spec peak
FP16 = 58.982       # TFLOPS
INT8 = 117.964      # TOPS
RIDGE_FP16 = FP16 * 1e3 / BW   # FLOP/byte
RIDGE_INT8 = INT8 * 1e3 / BW

# ---------------------------------------------------------------- model knobs
NL = CFG["num_hidden_layers"]              # 40
NL_F = CFG["num_full_attention_layers"]    # 10
NL_L = CFG["num_linear_attention_layers"]  # 30
H = CFG["hidden_size"]                      # 2048
NH = CFG["num_attention_heads"]            # 16
NKV = CFG["num_key_value_heads"]           # 2
HD = CFG["head_dim"]                        # 256
GATE_H = NH * HD                            # 4096
QKV = 2 * NH * HD + 2 * NKV * HD           # 9216 (query+gate+K+V fused)
LIN_PROJ = 12288                           # in_proj qkv 8192 + z 4096
LIN_VDIM = CFG["linear_num_value_heads"] * CFG["linear_value_head_dim"]  # 4096
I = CFG["moe_intermediate_size"]           # 512
SI = CFG["shared_expert_intermediate_size"]  # 512
NE = CFG["num_experts"]                    # 256
TK = CFG["num_experts_per_tok"]            # 8
VOCAB = CFG["vocab_size"]                   # 248320
G = 128
BLOCK = 16

PROMPTS = [1024, 2048, 4096, 8192]
GEN = 512
DEC_KV = [P + 256 for P in PROMPTS]        # mid-window KV -> 1280,2304,4352,8448


# ---------------------------------------------------------------- metric helpers
def m(key):
    e = MET.get(key)
    return (e["total_kernel_ns"] / 1e6) if e else 0.0


def subk(key):
    e = MET.get(key)
    if not e:
        return []
    it = max(1, e.get("iters_detected", 1))
    rows = []
    for nm, ns, c in e.get("per_kernel", []):
        rows.append((nm, ns / 1e6, c / it))   # (name, per-iter ms, launches/iter)
    return rows


def top_kernel(key):
    s = subk(key)
    return s[0][0] if s else "-"


def gbps(by, ms):
    return by / (ms * 1e6) if ms else 0.0


def gflops(fl, ms):
    return fl / (ms * 1e6) if ms else 0.0


def eff_mem(by, ms):
    return 100.0 * gbps(by, ms) / BW


def eff_xmx(fl, ms, peak):
    return 100.0 * gflops(fl, ms) / (peak * 1e3)


# ---- byte / flop models -----------------------------------------------------
def fc_bytes_i4(mm, k, n):
    return k * n // 2 + (k // G) * n * 2 + (k // G) * n // 2 + mm * k * 2 + mm * n * 2


def fc_bytes_i8(mm, k, n):
    return k * n + (k // G) * n * 2 + mm * k * 2 + mm * n * 2


def fc_flops(mm, k, n):
    return 2 * mm * k * n


def moe_bytes(mm, n_experts=None):
    """Weight + activation traffic for one MoE layer.

    n_experts = number of DISTINCT routed experts whose INT4 weights must stream
    from VRAM. With top-k routing there are mm*TK token->expert assignments over
    NE experts, so the count saturates at NE:
      - decode (mm=1): only TK experts are hit  -> TK weight matrices read;
      - prefill (mm tokens, mm*TK >= NE): all NE experts are hit -> the grouped
        gemm loads every expert's weights once (NE matrices), i.e. NE/TK x more
        weight traffic than a single token. Using TK here would undercount
        prefill weight reads by 32x and wrongly mark MoE compute-bound.
    The shared expert is always-on (+1)."""
    if n_experts is None:
        n_experts = min(NE, mm * TK)
    pe = lambda inter: 3 * H * inter // 2 + 3 * (H // G) * inter * 2 + 3 * (H // G) * inter // 2
    act = (H + (TK * I + SI) * 2) * 2 * mm
    return n_experts * pe(I) + pe(SI) + act


def moe_flops(mm):
    return (TK + 1) * 3 * H * max(I, SI) * 2 * mm


def pa_bytes_dec(kv):
    return 2 * NKV * HD * kv          # INT8 K+V streamed


def pa_flops_dec(kv):
    return 2 * NH * HD * kv * 2       # QK^T + softmax*V


def pa_bytes_pre(S):
    # causal: K+V streamed once + Q + out (FP16)
    return 2 * NKV * HD * S + 2 * NH * HD * S * 2


def pa_flops_pre(S):
    return 2 * NH * HD * (S * S // 2) * 2   # causal lower-triangle


def wbytes_i4(k, n):
    return k * n // 2 + (k // G) * n * 2 + (k // G) * n // 2


def wbytes_i8(k, n):
    return k * n + (k // G) * n * 2


def mb(b):
    return b / 1e6


# ---------------------------------------------------------------- op catalogs
# Decode (M=1): (key, op-label, kernel, calls, bytes, flops)  — all memory-bound
def decode_ops():
    return [
        ("moe_decode_M1", "MoE 3-gemm (TK=8 + shared, NE=256)", top_kernel("moe_decode_M1"),
            NL, moe_bytes(1), moe_flops(1), "mem"),
        ("lm_head_decode_M1", "LM head (2048->248320, INT8)", top_kernel("lm_head_decode_M1"),
            1, fc_bytes_i8(1, H, VOCAB), fc_flops(1, H, VOCAB), "mem"),
        ("fc_linattn_proj_decode_M1", f"FC linattn in_proj (2048->{LIN_PROJ})", top_kernel("fc_linattn_proj_decode_M1"),
            NL_L, fc_bytes_i4(1, H, LIN_PROJ), fc_flops(1, H, LIN_PROJ), "mem"),
        ("fc_o_decode_M1", "FC o_proj / GDN out (4096->2048)", top_kernel("fc_o_decode_M1"),
            NL, fc_bytes_i4(1, LIN_VDIM, H), fc_flops(1, LIN_VDIM, H), "mem"),
        ("fc_qkv_decode_M1", f"FC qkv+gate (2048->{QKV})", top_kernel("fc_qkv_decode_M1"),
            NL_F, fc_bytes_i4(1, H, QKV), fc_flops(1, H, QKV), "mem"),
        ("gdn_decode_T1", "GatedDeltaNet core (qk=16,v=32,K=V=128)", top_kernel("gdn_decode_T1"),
            NL_L, 0, 0, "recurrent"),  # recurrent state op: no clean analytic roofline
        ("so_rmsnorm_h2048_decode", "rmsnorm (H=2048)", top_kernel("so_rmsnorm_h2048_decode"),
            2 * NL, H * 4, H * 8, "mem"),
        ("so_add_decode", "residual add (H=2048)", top_kernel("so_add_decode"),
            2 * NL, H * 6, H, "mem"),
        ("so_rmsnorm3d_qnorm_decode", "q_norm (16x256)", top_kernel("so_rmsnorm3d_qnorm_decode"),
            NL_F, NH * HD * 4, NH * HD * 8, "mem"),
        ("so_rope_q_decode", "rope_q (16x256)", top_kernel("so_rope_q_decode"),
            NL_F, NH * HD * 4, NH * HD * 10, "mem"),
        ("so_rmsnorm3d_knorm_decode", "k_norm (2x256)", top_kernel("so_rmsnorm3d_knorm_decode"),
            NL_F, NKV * HD * 4, NKV * HD * 8, "mem"),
        ("so_rope_k_decode", "rope_k (2x256)", top_kernel("so_rope_k_decode"),
            NL_F, NKV * HD * 4, NKV * HD * 10, "mem"),
        ("so_gate_decode", "attn gate x*sigmoid(y) (H=4096)", top_kernel("so_gate_decode"),
            NL_F, 3 * GATE_H * 2, GATE_H * 5, "mem"),
    ]


def decode_const_ms():
    return sum(m(k) * c for k, _l, _kn, c, _b, _f, _bd in decode_ops())


def pa_decode_total_ms(kv):
    return m(f"pa_decode_kv{kv}") * NL_F


# Prefill (M=S): (key_fmt, op-label, calls, bytes_fn, flops_fn, peak_kind)
#  peak_kind: 'int8' XMX compute, 'fp16' XMX compute, 'mem'
def prefill_ops(S):
    return [
        (f"moe_prefill_S{S}", "MoE grouped-gemm (TK=8 + shared)", NL,
            moe_bytes(S), moe_flops(S), "int8"),
        (f"pa_prefill_S{S}", f"PagedAttention prefill (causal, NH={NH})", NL_F,
            pa_bytes_pre(S), pa_flops_pre(S), "fp16"),
        (f"gdn_prefill_S{S}", "GatedDeltaNet core", NL_L,
            0, 0, "recurrent"),  # recurrent state op: no clean analytic roofline
        (f"fc_linattn_proj_prefill_S{S}", f"FC linattn in_proj (2048->{LIN_PROJ})", NL_L,
            fc_bytes_i8(S, H, LIN_PROJ), fc_flops(S, H, LIN_PROJ), "int8"),
        (f"fc_qkv_prefill_S{S}", f"FC qkv+gate (2048->{QKV})", NL_F,
            fc_bytes_i8(S, H, QKV), fc_flops(S, H, QKV), "int8"),
        (f"fc_o_prefill_S{S}", "FC o_proj / GDN out (4096->2048)", NL,
            fc_bytes_i8(S, LIN_VDIM, H), fc_flops(S, LIN_VDIM, H), "int8"),
        (f"so_rmsnorm_h2048_prefill_S{S}", "rmsnorm (H=2048)", 2 * NL,
            S * H * 4, S * H * 8, "mem"),
        (f"so_rope_q_prefill_S{S}", "rope_q (16x256)", NL_F,
            S * NH * HD * 4, S * NH * HD * 10, "mem"),
        (f"so_gate_prefill_S{S}", "attn gate x*sigmoid(y) (H=4096)", NL_F,
            S * 3 * GATE_H * 2, S * GATE_H * 5, "mem"),
        ("lm_head_decode_M1", "LM head (last token, 2048->248320)", 1,
            fc_bytes_i8(1, H, VOCAB), fc_flops(1, H, VOCAB), "mem"),
    ]


def prefill_total_ms(S):
    return sum(m(k) * c for k, _l, c, _b, _f, _pk in prefill_ops(S))


# ---- roofline floor (theoretical lower-bound time) --------------------------
def op_theoretical_ms(by, fl, peak_kind, measured_ms):
    """Ideal lower-bound time for ONE op call on this HW = max(bytes/BW, FLOP/XMX-peak).
    Cache-resident micro-ops that measure faster than the DRAM floor are capped at
    their measured time so they stay neutral (achieved=100%) in the roll-up."""
    t_mem = by / (BW * 1e6)
    if peak_kind == "int8":
        t_cmp = fl / (INT8 * 1e9)
    elif peak_kind == "fp16":
        t_cmp = fl / (FP16 * 1e9)
    else:
        t_cmp = 0.0
    t = max(t_mem, t_cmp)
    return min(t, measured_ms) if measured_ms else t


def decode_roofline(kv):
    """(theoretical_ms, measured_ms, recurrent_ms) over one decode token.
    theoretical/measured cover only analytically-modelable ops; recurrent (GDN)
    is summed separately. measured + recurrent == full TPOT."""
    theo = meas = rec = 0.0
    for key, _l, _kn, calls, by, fl, bd in decode_ops():
        ms = m(key)
        if bd == "recurrent" or (by == 0 and fl == 0):
            rec += ms * calls
            continue
        theo += op_theoretical_ms(by, fl, bd, ms) * calls
        meas += ms * calls
    pms = m(f"pa_decode_kv{kv}")
    theo += op_theoretical_ms(pa_bytes_dec(kv), pa_flops_dec(kv), "mem", pms) * NL_F
    meas += pms * NL_F
    return theo, meas, rec


def prefill_roofline(S):
    """(theoretical_ms, measured_ms, recurrent_ms) over a TTFT pass.
    measured + recurrent == full TTFT."""
    theo = meas = rec = 0.0
    for key, _l, calls, by, fl, pk in prefill_ops(S):
        ms = m(key)
        if pk == "recurrent" or (by == 0 and fl == 0):
            rec += ms * calls
            continue
        theo += op_theoretical_ms(by, fl, pk, ms) * calls
        meas += ms * calls
    return theo, meas, rec


# ---------------------------------------------------------------- render helpers
def fnum(v, p=4):
    return f"{v:.{p}f}"


def row_metrics(ms, calls, by, fl, peak_kind):
    """returns (single_ms, total_ms, gflops|None, gbs|None, eff_pct|None, bound_str)."""
    tot = ms * calls
    if peak_kind == "recurrent" or (by == 0 and fl == 0):
        return ms, tot, None, None, None, "recurrent"
    gfs = gflops(fl, ms)
    gbs = gbps(by, ms)
    ai = (fl / by) if by else 0.0
    if peak_kind == "int8":
        bound = "compute" if ai > RIDGE_INT8 else "mem"
        eff = eff_xmx(fl, ms, INT8) if bound == "compute" else eff_mem(by, ms)
    elif peak_kind == "fp16":
        bound = "compute" if ai > RIDGE_FP16 else "mem"
        eff = eff_xmx(fl, ms, FP16) if bound == "compute" else eff_mem(by, ms)
    else:
        bound = "mem"
        eff = eff_mem(by, ms)
    # micro-bench small ops can be L2/L3-resident -> achieved BW exceeds the
    # streaming spec; flag as cache-resident rather than reporting >100% mem eff.
    if bound == "mem" and gbs > BW_SPEC:
        bound = "cache"
    return ms, tot, gfs, gbs, eff, bound


def W(s=""):
    print(s)


def _c(v, p=1, suffix=""):
    """format a numeric cell, or em-dash for None."""
    if v is None:
        return "—"
    if p == 0:
        return f"{v:,.0f}{suffix}"
    return f"{v:.{p}f}{suffix}"


# ============================================================================
def main():
    perf = {"model": "Qwen3.6-35B-A3B", "platform": "PTL_12Xe_B390",
            "gen_tokens": GEN, "prompts": PROMPTS}

    # ----- header --------------------------------------------------------------
    W(f"# Qwen3.6-35B-A3B — Roofline on {PLAT_NAME} ({DATE})")
    W()
    W(f"**Platform**: {PLAT_NAME}, {ARCH}; {BW:g} GB/s achievable read "
      f"(spec {BW_SPEC:g}); FP16 XMX {FP16:g} TFLOPS, INT8 XMX {INT8:g} TOPS.")
    W(f"**Model**: text decoder, {NL} layers = {NL_F} full-attention + {NL_L} "
      f"linear-attention (GatedDeltaNet); MoE every layer.")
    W()
    W(f"- {NL} decoder blocks; full-attn every {NL // NL_F}th layer; "
      f"attn_output_gate=true (fused QKV+gate width = 2·{NH}·{HD} + 2·{NKV}·{HD} = {QKV})")
    W(f"- MoE: {NE} experts, top-{TK}, expert intermediate {I}, always-on shared expert "
      f"intermediate {SI}")
    W(f"- MatMul weights INT4 g{G} / FP16 act; LM_head INT8 g{G} / FP16 act; KV cache INT8")
    W(f"- SDPA: PagedAttention (OpenCL micro-kernel), GQA {NH}/{NKV} = {NH // NKV}-way; "
      "linear-attn via GatedDeltaNet")
    W()

    # ----- model parameters & weight shapes -----------------------------------
    W("## Model parameters & weight shapes")
    W()
    W("Architecture knobs (parsed from model config):")
    W()
    W("| Field | Value | Notes |")
    W("|---|---:|---|")
    W(f"| `hidden_size` | {H} | residual / activation channel |")
    W(f"| `num_hidden_layers` | {NL} | {NL_F} full-attn + {NL_L} linear-attn |")
    W(f"| `num_attention_heads` (NH) | {NH} | full-attn Q heads |")
    W(f"| `num_key_value_heads` (NKV) | {NKV} | GQA: {NH // NKV}-way Q-per-KV sharing |")
    W(f"| `head_dim` (HD) | {HD} | Q_dim = NH·HD = {NH * HD}; partial RoPE factor 0.25 |")
    W(f"| `attn_output_gate` | true | q_proj emits [query\\|gate]; gate width = {GATE_H} |")
    W(f"| linear K/V heads | {CFG['linear_num_key_heads']}/{CFG['linear_num_value_heads']} | "
      f"GDN k/v head_dim {CFG['linear_key_head_dim']}; in_proj = {LIN_PROJ} (qkv 8192 + z 4096) |")
    W(f"| `moe_intermediate_size` | {I} | per-expert SwiGLU hidden |")
    W(f"| `num_experts` / `num_experts_per_tok` | {NE} / {TK} | + 1 always-on shared expert |")
    W(f"| `vocab_size` | {VOCAB} | LM head N |")
    W(f"| `rope_theta` | {CFG['rope_theta']} | — |")
    W()
    W("Per-layer weight matrices (one decoder block) and global weights "
      f"(INT4 = K·N/2 + FP16 scale + INT4 zp at g{G}; INT8 = K·N + FP16 scale):")
    W()
    W("| Weight | Shape (K × N) | Quant | MB / instance | × Count | Total MB |")
    W("|---|---:|---|---:|---:|---:|")
    wtab = []

    def wrow(label, k, n, quant, count):
        b = wbytes_i8(k, n) if quant.startswith("INT8") else wbytes_i4(k, n)
        tot = mb(b) * count
        W(f"| {label} | {k} × {n} | {quant} | {mb(b):.2f} | {count} | {tot:,.1f} |")
        wtab.append((label, tot))
        return tot

    total_w = 0.0
    total_w += wrow("Embedding (gather)", VOCAB, H, "INT8 g128", 1)
    total_w += wrow("FC_QKV+gate (full-attn)", H, QKV, "INT4 g128", NL_F)
    total_w += wrow("FC_O / GDN out_proj", LIN_VDIM, H, "INT4 g128", NL)
    total_w += wrow("Lin-attn in_proj (GDN)", H, LIN_PROJ, "INT4 g128", NL_L)
    total_w += wrow("MoE expert gate+up", H, 2 * I, "INT4 g128", NL * NE)
    total_w += wrow("MoE expert down", I, H, "INT4 g128", NL * NE)
    total_w += wrow("MoE shared gate+up", H, 2 * SI, "INT4 g128", NL)
    total_w += wrow("MoE shared down", SI, H, "INT4 g128", NL)
    total_w += wrow("MoE router", H, NE, "FP16", NL)
    total_w += wrow("LM_Head", H, VOCAB, "INT8 g128", 1)
    W(f"| **Total static weights** |  |  |  |  | **{total_w / 1000:,.2f} GB** |")
    W()
    perf["total_weights_gb"] = total_w / 1000

    # KV cache
    k_pt = NKV * HD                      # elems per token per layer (INT8 -> 1 B each)
    v_pt = NKV * HD
    kv_layer_b = (k_pt + v_pt) * 1       # INT8
    W("KV-cache (INT8) per token:")
    W()
    W("| Tensor | Shape | dtype | Bytes/token/layer | Bytes/token (10 full-attn layers) |")
    W("|---|---|---|---:|---:|")
    W(f"| K cache | [blocks, {NKV}, {HD}, {BLOCK}] | INT8 | {k_pt} | {k_pt * NL_F} |")
    W(f"| V cache | [blocks, {NKV}, {BLOCK}, {HD}] | INT8 | {v_pt} | {v_pt * NL_F} |")
    W(f"| **KV total** | per token | INT8 | {kv_layer_b} B/layer | **{kv_layer_b * NL_F} B/token** |")
    W()

    # ----- theoretical roofline -----------------------------------------------
    W("## Theoretical roofline")
    W()
    W("| Metric | Value |")
    W("|---|---|")
    W(f"| FP16 XMX peak | {FP16:g} TFLOPS |")
    W(f"| INT8 XMX peak | {INT8:g} TOPS |")
    W(f"| Memory BW (achievable read) | {BW:g} GB/s (spec {BW_SPEC:g}) |")
    W(f"| Ridge point (FP16) | {RIDGE_FP16:.0f} FLOP/byte |")
    W(f"| Ridge point (INT8) | {RIDGE_INT8:.0f} OP/byte |")
    W()

    # ----- data sources --------------------------------------------------------
    W("## Data sources")
    W()
    W("All ops measured natively on this platform via cliloader Device Performance "
      "Timing (mean kernel time per iteration); **no cross-platform scaling**. Each op runs "
      "in its own process with a cache flush between iterations so weights stream from VRAM. "
      "PTL 12Xe and Intel Arc B390 share metrics (same silicon; cliloader id difference only).")
    W()

    # ----- graph fusion notes --------------------------------------------------
    W("## Graph fusion notes")
    W()
    W("| Bench row | Real graph behaviour | Standalone kernel in graph? |")
    W("|---|---|---|")
    W("| `moe` | gate/up/down SwiGLU experts fused into `moe_3gemm_swiglu_*`; routed via "
      "`fuse_softmax_topk` | Yes (MOE3GemmFusedCompressed) |")
    W("| shared expert | NOT fused on this build — stays as 3 `gemm_kernel` FCs; timed "
      "together with routed MoE | Yes (3 extra FC) |")
    W("| `gate` | attn_output · sigmoid(gate) of gated attention (full-attn layers) | Yes (eltwise) |")
    W("| `multiply`/`swish` | SwiGLU activation, fused into MoE primitive | No — not benched separately |")
    W("| `add` | residual adds | Yes (eltwise) |")
    W("| `rmsnorm` | pre-attn + pre-MLP RMSNorm | Yes |")
    W("| FC prefill | `dynamic_quantize_gpu_opt` + `gemm_kernel` | Yes (2 kernels) |")
    W("| PA decode | `pa_kv_cache_update` + attention + finalization | Yes (3 kernels) |")
    W()

    # ----- token latency summary ----------------------------------------------
    const_ms = decode_const_ms()
    perf["decode_const_ms"] = const_ms
    W("## Token latency summary")
    W()
    W("### Prefill — TTFT and per-token amortized")
    W()
    W("| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |")
    W("|---:|---:|---:|---:|---:|")
    pre_tot = {}
    for S in PROMPTS:
        t = prefill_total_ms(S)
        pre_tot[S] = t
        W(f"| {S} | {t:,.1f} | {t / 1000:.3f} | {t / S:.3f} | {S / t * 1000:,.0f} |")
    W()
    W(f"### Decode — TPOT (per output token, mid 512-gen window KV = P+256)")
    W()
    W("| prompt P | KV (mid) | TPOT (ms) | tokens/s |")
    W("|---:|---:|---:|---:|")
    for P, kv in zip(PROMPTS, DEC_KV):
        tpot = const_ms + pa_decode_total_ms(kv)
        W(f"| {P} | {kv} | {tpot:.3f} | {1000 / tpot:.1f} |")
    W()
    # full 512-token window (start/mean/end) — the requested generation-phase metric
    W("### Decode — full 512-token generation window (PA grows with KV)")
    W()
    W("`start` = KV at first generated token (P); `mean` = KV P+256; `end` = KV P+512. "
      "Only PagedAttention scales with KV; the other 20.86 ms/token is M=1 constant.")
    W()
    W("| prompt P | TPOT start (ms) | TPOT mean (ms) | TPOT end (ms) | "
      "512-tok decode (ms) | decode tok/s |")
    W("|---:|---:|---:|---:|---:|---:|")
    win_rows = []
    for P in PROMPTS:
        ts = const_ms + pa_decode_total_ms(P)
        tm = const_ms + pa_decode_total_ms(P + 256)
        te = const_ms + pa_decode_total_ms(P + 512)
        d512 = GEN * tm
        W(f"| {P} | {ts:.3f} | {tm:.3f} | {te:.3f} | {d512:,.1f} | {1000 / tm:.1f} |")
        win_rows.append(dict(prompt=P, tpot_start=ts, tpot_mean=tm, tpot_end=te,
                             decode512_ms=d512, decode_toks=1000 / tm))
    perf["decode_window"] = win_rows
    W()

    # decode breakdown (ms / % of TPOT) across the 4 KVs
    W("### Decode TPOT — per-op breakdown (ms / % of TPOT)")
    W()
    hdr = " | ".join(f"P={P} (KV{kv})" for P, kv in zip(PROMPTS, DEC_KV))
    sep = "|".join(["---:"] * len(DEC_KV))
    W(f"| op | {hdr} |")
    W(f"|---|{sep}|")
    tpots = [const_ms + pa_decode_total_ms(kv) for kv in DEC_KV]
    for key, lbl, _kn, calls, _b, _f, _bd in decode_ops():
        cells = []
        ms_tot = m(key) * calls
        for tp in tpots:
            cells.append(f"{ms_tot:.3f} ({100 * ms_tot / tp:.1f}%)")
        W(f"| {lbl} | " + " | ".join(cells) + " |")
    # PA row
    cells = []
    for kv, tp in zip(DEC_KV, tpots):
        pa = pa_decode_total_ms(kv)
        cells.append(f"{pa:.3f} ({100 * pa / tp:.1f}%)")
    W(f"| PagedAttention (×{NL_F}) | " + " | ".join(cells) + " |")
    W()

    # prefill breakdown
    W("### Prefill TTFT — per-op breakdown (ms / % of TTFT)")
    W()
    hdr = " | ".join(f"S={S}" for S in PROMPTS)
    sep = "|".join(["---:"] * len(PROMPTS))
    W(f"| op | {hdr} |")
    W(f"|---|{sep}|")
    # collect op labels from S=PROMPTS[0]
    labels = [(l, k.split("_S")[0] if "_S" in k else k) for k, l, _c, _b, _f, _pk in prefill_ops(PROMPTS[0])]
    for idx, (key0, lbl, c0, _b, _f, _pk) in enumerate(prefill_ops(PROMPTS[0])):
        cells = []
        for S in PROMPTS:
            op = prefill_ops(S)[idx]
            ms_tot = m(op[0]) * op[2]
            cells.append(f"{ms_tot:.2f} ({100 * ms_tot / pre_tot[S]:.1f}%)")
        W(f"| {lbl} | " + " | ".join(cells) + " |")
    W()

    # ----- roofline: theoretical floor vs measured ----------------------------
    W("## Roofline: theoretical floor vs measured")
    W()
    W("**Theoretical floor** = sum over analytically-modelable ops of "
      "max(bytes / BW, FLOP / XMX-peak) - the fastest this HW could run each op given its "
      "memory traffic / compute. **Measured** is the summed cliloader kernel time of the same "
      "ops. **achieved % = theoretical / measured** (how close real kernels get to the roofline "
      "ceiling; 100% = on the roofline). GatedDeltaNet uses a recurrent `*_opt` kernel (measured with "
      "`cache_interval=0`, i.e. a single final state snapshot) with no "
      "analytic model, so it is reported separately as *unmodeled GDN* and excluded from the "
      "ratio; `full` = measured + unmodeled = the real TPOT / TTFT.")
    W()
    W("### Decode (per output token, mid 512-gen window KV = P+256)")
    W()
    W("| prompt P | KV | theoretical (ms) | measured (ms) | achieved % | "
      "unmodeled GDN (ms) | full TPOT (ms) |")
    W("|---:|---:|---:|---:|---:|---:|---:|")
    dec_roof = []
    for P, kv in zip(PROMPTS, DEC_KV):
        theo, meas, rec = decode_roofline(kv)
        full = meas + rec
        ach = 100 * theo / meas if meas else 0.0
        W(f"| {P} | {kv} | {theo:.3f} | {meas:.3f} | {ach:.1f}% | {rec:.3f} | {full:.3f} |")
        dec_roof.append(dict(prompt=P, kv=kv, theoretical_ms=theo, measured_ms=meas,
                             achieved_pct=ach, unmodeled_gdn_ms=rec, full_tpot_ms=full))
    W()
    W("### Prefill (TTFT over S tokens)")
    W()
    W("| S | theoretical (ms) | measured (ms) | achieved % | unmodeled GDN (ms) | full TTFT (ms) |")
    W("|---:|---:|---:|---:|---:|---:|")
    pre_roof = []
    for S in PROMPTS:
        theo, meas, rec = prefill_roofline(S)
        full = meas + rec
        ach = 100 * theo / meas if meas else 0.0
        W(f"| {S} | {theo:,.1f} | {meas:,.1f} | {ach:.1f}% | {rec:,.1f} | {full:,.1f} |")
        pre_roof.append(dict(S=S, theoretical_ms=theo, measured_ms=meas,
                             achieved_pct=ach, unmodeled_gdn_ms=rec, full_ttft_ms=full))
    W()
    perf["roofline_decode"] = dec_roof
    perf["roofline_prefill"] = pre_roof

    # ----- decode tables (one per KV) -----------------------------------------
    W("## Decode tables (1 query token, KV = mid 512-gen window context)")
    W()
    dec_top = {}
    for P, kv in zip(PROMPTS, DEC_KV):
        W(f"### Decode — KV={kv} (prompt {P}, mid 512-gen window)")
        W()
        W("| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |")
        W("|---|---|---:|---:|---:|---:|---:|---:|---|")
        rows = []
        for key, lbl, kn, calls, by, fl, bd in decode_ops():
            ms = m(key)
            if ms == 0:
                continue
            s_ms, tot, gfs, gbs, eff, bound = row_metrics(ms, calls, by, fl, bd)
            rows.append((lbl, kn, s_ms, calls, tot, gfs, gbs, eff, bound))
        # PA row at this KV
        pa_ms = m(f"pa_decode_kv{kv}")
        if pa_ms:
            s_ms, tot, gfs, gbs, eff, bound = row_metrics(
                pa_ms, NL_F, pa_bytes_dec(kv), pa_flops_dec(kv), "mem")
            rows.append((f"PagedAttention (i8 KV={kv})", top_kernel(f"pa_decode_kv{kv}"),
                         s_ms, NL_F, tot, gfs, gbs, eff, bound))
        rows.sort(key=lambda r: r[4], reverse=True)
        total_ms = sum(r[4] for r in rows)
        for lbl, kn, s_ms, calls, tot, gfs, gbs, eff, bound in rows:
            W(f"| {lbl} | `{kn}` | {s_ms:.4f} | {calls} | {tot:.4f} | "
              f"{_c(gfs, 0)} | {_c(gbs, 1)} | {_c(eff, 0, '%')} | {bound} |")
        W(f"| **TOTAL** |  |  |  | **{total_ms:.3f}** |  |  |  |  |")
        W()
        dec_top[kv] = sorted([(r[0], r[4]) for r in rows], key=lambda x: x[1], reverse=True)[:3]

    # ----- prefill tables (one per S) -----------------------------------------
    W("## Prefill tables (single forward over S tokens)")
    W()
    pre_top = {}
    for S in PROMPTS:
        W(f"### Prefill — S={S}")
        W()
        W("| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |")
        W("|---|---|---:|---:|---:|---:|---:|---:|---|")
        rows = []
        for key, lbl, calls, by, fl, pk in prefill_ops(S):
            ms = m(key)
            if ms == 0:
                continue
            s_ms, tot, gfs, gbs, eff, bound = row_metrics(ms, calls, by, fl, pk)
            rows.append((lbl, top_kernel(key), s_ms, calls, tot, gfs, gbs, eff, bound))
        rows.sort(key=lambda r: r[4], reverse=True)
        total_ms = sum(r[4] for r in rows)
        for lbl, kn, s_ms, calls, tot, gfs, gbs, eff, bound in rows:
            W(f"| {lbl} | `{kn}` | {s_ms:.4f} | {calls} | {tot:.4f} | "
              f"{_c(gfs, 0)} | {_c(gbs, 1)} | {_c(eff, 0, '%')} | {bound} |")
        W(f"| **TOTAL** |  |  |  | **{total_ms:,.3f}** |  |  |  |  |")
        W()
        pre_top[S] = sorted([(r[0], r[4]) for r in rows], key=lambda x: x[1], reverse=True)[:3]

    # ----- op -> kernel name map ----------------------------------------------
    W("## Op → kernel names (cliloader)")
    W()
    W("Each logical op and the actual GPU kernel(s) it dispatches (one bench process per "
      "op). Kernels are listed in launch order; `launches/call` is how many times each "
      "kernel fires per single op invocation (per layer). Decode is measured at M=1, prefill "
      "at S=8192 — kernel selection can vary with shape.")
    W()

    def kmap_table(op_list):
        W("| op | kernel name(s) | launches/call |")
        W("|---|---|---:|")
        for lbl, key in op_list:
            sk = subk(key)
            if not sk:
                W(f"| {lbl} | `-` | — |")
                continue
            names = "<br>".join(f"`{nm}`" for nm, _s, _l in sk)
            launches = "<br>".join(f"{lpc:.1f}" for _nm, _s, lpc in sk)
            W(f"| {lbl} | {names} | {launches} |")

    W("### Decode (M=1)")
    W()
    kmap_table([(lbl, key) for key, lbl, _kn, _ca, _b, _f, _bd in decode_ops()]
               + [("PagedAttention (KV=4352)", "pa_decode_kv4352")])
    W()
    W("### Prefill (S=8192)")
    W()
    kmap_table([(lbl, key) for key, lbl, _ca, _b, _f, _pk in prefill_ops(8192)])
    W()

    # ----- per-kernel decomposition -------------------------------------------
    W("## Per-kernel decomposition (cliloader kernel names)")
    W()
    W("### Decode sub-kernels — KV=4352 (prompt 4096, representative)")
    W()
    W("| op | kernel name | single ms | launches/call | calls/inf | total ms | % |")
    W("|---|---|---:|---:|---:|---:|---:|")
    dec_sub = []
    for key, lbl, _kn, calls, _b, _f, _bd in decode_ops():
        for nm, sms, lpc in subk(key):
            dec_sub.append((lbl, nm, sms, lpc, calls, sms * calls))
    for nm, sms, lpc in subk("pa_decode_kv4352"):
        dec_sub.append(("PagedAttention", nm, sms, lpc, NL_F, sms * NL_F))
    dec_sub.sort(key=lambda r: r[5], reverse=True)
    dsub_tot = sum(r[5] for r in dec_sub)
    for lbl, nm, sms, lpc, calls, tot in dec_sub[:18]:
        W(f"| {lbl} | `{nm}` | {sms:.4f} | {lpc:.1f} | {calls} | {tot:.4f} | "
          f"{100 * tot / dsub_tot:.1f}% |")
    W()
    W("### Prefill sub-kernels — S=8192 (representative)")
    W()
    W("| op | kernel name | single ms | launches/call | calls/inf | total ms | % |")
    W("|---|---|---:|---:|---:|---:|---:|")
    pre_sub = []
    for key, lbl, calls, _b, _f, _pk in prefill_ops(8192):
        for nm, sms, lpc in subk(key):
            pre_sub.append((lbl, nm, sms, lpc, calls, sms * calls))
    pre_sub.sort(key=lambda r: r[5], reverse=True)
    psub_tot = sum(r[5] for r in pre_sub)
    for lbl, nm, sms, lpc, calls, tot in pre_sub[:18]:
        W(f"| {lbl} | `{nm}` | {sms:.4f} | {lpc:.1f} | {calls} | {tot:.4f} | "
          f"{100 * tot / psub_tot:.1f}% |")
    W()

    # ----- top contributors ----------------------------------------------------
    W("## Top contributors (sorted by total ms per inference)")
    W()
    W("### Decode")
    W()
    W("| KV | top1 (ms,%) | top2 | top3 |")
    W("|---:|---|---|---|")
    for P, kv in zip(PROMPTS, DEC_KV):
        tp = const_ms + pa_decode_total_ms(kv)
        t = dec_top[kv]
        c = [f"{o} {v:.3f}ms ({100 * v / tp:.0f}%)" for o, v in t]
        W(f"| {kv} | " + " | ".join(c) + " |")
    W()
    W("### Prefill")
    W()
    W("| S | top1 (ms,%) | top2 | top3 |")
    W("|---:|---|---|---|")
    for S in PROMPTS:
        t = pre_top[S]
        c = [f"{o} {v:.1f}ms ({100 * v / pre_tot[S]:.0f}%)" for o, v in t]
        W(f"| {S} | " + " | ".join(c) + " |")
    W()

    # ----- end-to-end ----------------------------------------------------------
    W("## End-to-end (prefill TTFT + 512-token decode)")
    W()
    W("| prompt P | TTFT (ms) | 512-tok decode (ms) | total (ms) | avg decode tok/s |")
    W("|---:|---:|---:|---:|---:|")
    e2e = []
    for w in win_rows:
        ttft = pre_tot[w["prompt"]]
        tot = ttft + w["decode512_ms"]
        W(f"| {w['prompt']} | {ttft:,.1f} | {w['decode512_ms']:,.1f} | {tot:,.1f} | "
          f"{w['decode_toks']:.1f} |")
        e2e.append(dict(prompt=w["prompt"], ttft_ms=ttft,
                        decode512_ms=w["decode512_ms"], total_ms=tot,
                        decode_toks=w["decode_toks"]))
    perf["end_to_end"] = e2e
    W()

    # ----- findings & levers ---------------------------------------------------
    W("## Key findings")
    W()
    W(f"- **Decode throughput is flat at ~{min(w['decode_toks'] for w in win_rows):.0f}–"
      f"{max(w['decode_toks'] for w in win_rows):.0f} tok/s across all prompt lengths.** The "
      f"per-token budget is {const_ms:.1f} ms of KV-independent work; PagedAttention over the "
      "512-token window only moves TPOT by ~0.7–1.7 ms, so prompt length barely affects decode.")
    W("- **Three ops own ~80% of decode:** MoE, LM-head, and linear-attn in_proj — all "
      f"memory-bound INT4/INT8 weight streaming at 90–97% of {BW:g} GB/s.")
    W(f"- **Decode reaches ~{dec_roof[0]['achieved_pct']:.0f}% of the memory roofline** "
      "(modelable ops); the remaining gap is mostly MoE (~74%) and PagedAttention (~50%). "
      f"**Prefill reaches ~{pre_roof[0]['achieved_pct']:.0f}% of its roofline at S=1024, falling "
      f"to ~{pre_roof[-1]['achieved_pct']:.0f}% at S=8192**: MoE grouped-gemm is **memory-bound** "
      "— it streams all 256 experts' INT4 weights every layer — yet hits only ~16–42% of weight "
      "BW (small per-expert token groups, gather/scatter and INT4 dequant overhead).")
    W("- **PA kernel heuristic switch at KV≥4096** (`paged_attention_opt__single_token` → "
      "`gqa_single_token`) makes PA-decode non-monotonic: the 4096-prompt window is *faster* "
      "than the 2048-prompt window.")
    W("- **Prefill (TTFT) is MoE/GDN/PA-bound and grows super-linearly** "
      f"({pre_tot[1024] / 1000:.2f} s → {pre_tot[8192] / 1000:.2f} s for 1K → 8K). PA prefill is "
      "quadratic (causal); MoE grouped-gemm dominates the rest.")
    W()
    W("## Optimization levers (highest decode ROI first)")
    W()
    W("1. **MoE expert streaming** — BW-bound at only ~74% efficiency (M=1 micro-gemms have "
      "launch/scheduling overhead); batching decode requests amortizes weight reads, and the "
      "unfused shared expert (3 extra FCs) is a fusion opportunity.")
    W("2. **LM-head** — 508 MB of INT8 weights streamed every token at ~95% BW; INT4 LM-head "
      "would roughly halve it; speculative decoding removes it from the per-token path.")
    W("3. **Linear-attn in_proj** — already ~97% BW-bound; only lower precision or fewer linear "
      "layers help.")
    W(f"4. **Memory bandwidth is the global ceiling** — every major decode op is mem-bound, so "
      f"the {BW:g} GB/s shared LPDDR5x read BW sets the floor.")
    W()

    # ----- reproduction --------------------------------------------------------
    W("## Reproduction")
    W()
    W("Built on the PTL host (VS2022, `OV_SRC_DIR` for `gdn_bench`); each case run under "
      "cliloader `-d` and parsed from Device Performance Timing. Driver: "
      "`utils/run_qwen3_6_ptl.bat` (61 cases). Representative commands:")
    W()
    W("```bat")
    W(":: decode (M=1) — KV-independent ops")
    W("moe_bench.exe        1 1    2048 512 256 8 128 100 10 4 64 512   :: MoE TK=8 + shared")
    W("fc_bench.exe         1 2048   9216 128 5000 200 8 u4 64          :: qkv+gate fused")
    W("fc_bench.exe         1 2048  12288 128 1500 100 4 u4 64          :: linear-attn in_proj")
    W("fc_bench.exe         1 2048 248320 128  300  30 4 u8 64          :: LM-head (INT8)")
    W("gdn_bench.exe        1 1     16 32 128 4000 150 4 0              :: GatedDeltaNet (paged opt, cache_interval=0)")
    W("small_ops_bench.exe  gate    1 4096 --iters 5000 --warmup 200    :: attn output gate")
    W(":: decode PA — 512-token window sweep (KV = P / P+256 / P+512)")
    W("pa_bench.exe decode  1 <KV> 8000 200 4 i8   (PA_NH=16 PA_NKV=2 PA_HD=256)")
    W(":: prefill — per prompt S in {1024,2048,4096,8192}")
    W("moe_bench.exe 1 S 2048 512 256 8 128 ...    pa_bench.exe prefill S 0 ... i8")
    W("```")
    W()
    W("Parse + report:")
    W("```bash")
    W("python3 ../../utils/parse_logs.py logs ptl_metrics.json")
    W("python3 build_report.py ptl_metrics.json > SUMMARY_qwen3_6_%DATE%.md")
    W("```")
    W()

    # ----- caveats -------------------------------------------------------------
    W("## Caveats & method")
    W()
    W("- Each op profiled in its own process via cliloader Device Performance Timing (mean "
      "kernel time per iteration); cache flush between iters so weights stream from VRAM. Totals "
      "are an upper-bound roofline, not a traced wall-clock.")
    W("- FC weight bytes count INT4/INT8 weight + FP16 scale/zp(g128) + FP16 act + FP16 out.")
    W("- **Shared expert is unfused on this build**; the MoE figure times the routed "
      "`MOE3GemmFusedCompressed` plus 3 shared FCs together — the real per-layer cost here.")
    W("- **GatedDeltaNet bench covers the core op only**; depthwise conv1d (k=4) not modeled "
      "(negligible), in_proj counted as `fc_linattn`, out_proj as `fc_o` (4096→2048). It is a "
      "recurrent state op with no clean analytic byte/flop model, so `bound=recurrent` and "
      "Eff/GFLOPS/GB/s are shown as `—` (only the measured latency is meaningful).")
    W(f"- `bound=cache`: tiny eltwise/norm/rope micro-benches are L2/L3-resident, so their "
      f"achieved BW exceeds the {BW_SPEC:g} GB/s streaming spec; in real inference they are "
      "fused into the adjacent matmul/attention and contribute <2% — only their latency is used.")
    W("- The attention output gate is benched with a `gate` proxy op (x·sigmoid(y), H=4096).")
    W("- Decode FC/MoE/LM-head are memory-bound (weights dominate at M=1). Prefill **MoE is still "
      "memory-bound** — with mm·TK ≥ NE every layer streams all 256 experts' weights once (the byte "
      "model uses min(NE, mm·TK) experts, not TK); prefill **FC** is INT8 XMX compute-bound at large "
      "S; PA prefill (S≥2048) is FP16 micro-kernel compute-bound.")
    W("- PA decode is memory-bound (INT8 KV cache + FP16 Q/out); lm_head runs once per token.")
    W("- q_norm/k_norm and residual-add are <0.1% of TTFT and omitted from prefill totals.")

    (OUT / "performance_metrics.json").write_text(json.dumps(perf, indent=2))
    print(f"\n_performance_metrics.json written to {OUT / 'performance_metrics.json'}_",
          file=sys.stderr)


if __name__ == "__main__":
    main()
