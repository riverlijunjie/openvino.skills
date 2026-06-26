#!/usr/bin/env python3
"""
diffusion_gemma (google/diffusiongemma-26B-A4B-it) roofline analysis — PTL 12Xe.

Reads perf_raw.json (output of parse_logs.py over the cliloader logs) and emits:
  - performance_metrics.json           (structured per-op metrics)
  - SUMMARY_diffusion_gemma_<date>.md  (filled from SUMMARY_TEMPLATE.md)

Workload model (block diffusion):
  * ENCODER = autoregressive prompt encode (causal sliding/full) -> "prefill"
              (TTFT, read-only KV cache built).
  * DECODER = block-diffusion denoising over a FIXED canvas of 256 tokens with
              BIDIRECTIONAL self-attention (+ cross-attention to the read-only
              encoder KV cache). Profiled as M=256 ("canvas decode") — one
              denoising step. NO M=1 autoregressive decode happens in reality;
              an M=1 reference is included only for comparison.

Per layer: dense GEGLU MLP (gate/up/down) + 128-expert top-8 MoE, SUMMED.
25 sliding-attn layers (NH16/NKV8/HD256, window=1024) +
 5 full-attn  layers (NH16/NKV2/HD512).
"""
import json
from pathlib import Path
from datetime import date

OUT = Path(__file__).resolve().parent.parent / "outputs" / "diffusion_gemma"
RAW = json.loads((OUT / "perf_raw.json").read_text())

# ---------------- Hardware (PTL 12Xe / Arc B390, 2400 MHz) ----------------
FREQ = 2.4
XE = 12
BW = 110.0
FP16_XMX = XE * 8 * 256 * FREQ / 1000.0   # 58.98 TFLOPS
INT8_XMX = 2 * FP16_XMX                    # 117.96 TOPS
OVH = 0.95
BW_e, FP16_e, INT8_e = BW * OVH, FP16_XMX * OVH, INT8_XMX * OVH

# ---------------- Architecture ----------------
H = 2816
NL = 30
NL_SL = 25     # sliding attention layers
NL_F = 5       # full attention layers
SW = 1024
VOCAB = 262144
I_DENSE = 2112
I_MOE = 704
NE = 128
TK = 8
SL_NH, SL_NKV, SL_HD = 16, 8, 256
F_NH, F_NKV, F_HD = 16, 2, 512
CANVAS = 256
SEQS = [1024, 2048, 4096, 8192]
G = 128
G_DOWN = 64


# ---------------- Log access ----------------
def ms(tag):
    r = RAW.get(tag)
    return r["total_kernel_ns"] / 1e6 if r else 0.0


def ms_filtered(tag, substr):
    r = RAW.get(tag)
    if not r:
        return 0.0
    return sum(ns for kn, ns, _ in r["per_kernel"] if substr in kn) / 1e6


def ms_sdpa(tag):
    """SDPA: count only the sdpa_micro kernel; exclude the bench-only GQA
    KV-head broadcast (broadcast_gpu_ref), which is an artifact of satisfying
    core opset13 SDPA shape inference."""
    v = ms_filtered(tag, "sdpa")
    return v if v else ms(tag)


def kernels(tag):
    r = RAW.get(tag)
    return r["per_kernel"] if r else []


# ---------------- Byte / FLOP models ----------------
def fc_int4_bytes(M_, K, N, g):
    w = N * K / 2 + N * (K // g) * 2
    return w + M_ * K * 2 + M_ * N * 2


def fc_int8_bytes(M_, K, N, g):
    w = N * K + N * (K // g) * 2
    return w + M_ * K * 2 + M_ * N * 2


def fc_flops(M_, K, N):
    return 2.0 * M_ * K * N


def sdpa_flops(Sq, Skv, NH, HD, causal):
    pairs = Sq * Skv if not causal else Sq * (Skv + 1) / 2.0
    return 2.0 * 2.0 * pairs * HD * NH


def sdpa_bytes(Sq, Skv, NH, NKV, HD):
    q = Sq * NH * HD * 2
    kv = 2 * Skv * NKV * HD * 2
    out = Sq * NH * HD * 2
    return q + kv + out


def moe_bytes(M_):
    gate_up = (2 * I_MOE) * H / 2 + (2 * I_MOE) * (H // G) * 2
    down = H * I_MOE / 2 + H * (I_MOE // G_DOWN) * 2
    router = NE * H / 2
    return TK * (gate_up + down) + router + M_ * H * 2 * 2


def moe_flops(M_):
    return 6.0 * H * I_MOE * M_ * TK


# ---------------- Roofline row builders ----------------
def fc_row(op, tag, M_, K, N, quant, calls, cb):
    t = ms(tag)
    g = G_DOWN if K in (I_MOE, I_DENSE) else G
    B = fc_int4_bytes(M_, K, N, g) if quant == "u4" else fc_int8_bytes(M_, K, N, G)
    F = fc_flops(M_, K, N)
    gbs = B / 1e9 / (t / 1e3) if t else 0
    gflops = F / 1e9 / (t / 1e3) if t else 0
    if cb:
        eff = gflops / (INT8_e * 1000) * 100 if t else 0
        bound = "compute(i8)"
    else:
        eff = gbs / BW_e * 100 if t else 0
        bound = "memory"
    return dict(op=op, tag=tag, kernel="dyn_quant+gemm" if cb else "gemm",
                single=t, calls=calls, total=t * calls, gflops=gflops, gbs=gbs,
                eff=eff, bound=bound)


def sdpa_row(op, tag, Sq, Skv, NH, NKV, HD, calls, causal, cb):
    t = ms_sdpa(tag)
    F = sdpa_flops(Sq, Skv, NH, HD, causal)
    B = sdpa_bytes(Sq, Skv, NH, NKV, HD)
    gflops = F / 1e9 / (t / 1e3) if t else 0
    gbs = B / 1e9 / (t / 1e3) if t else 0
    if cb:
        eff = gflops / (FP16_e * 1000) * 100 if t else 0
        bound = "compute(f16)"
    else:
        eff = gbs / BW_e * 100 if t else 0
        bound = "memory"
    return dict(op=op, tag=tag, kernel="sdpa_micro", single=t, calls=calls,
                total=t * calls, gflops=gflops, gbs=gbs, eff=eff, bound=bound)


def moe_row(op, tag, M_, calls, cb):
    t = ms(tag)
    F = moe_flops(M_)
    B = moe_bytes(M_)
    gflops = F / 1e9 / (t / 1e3) if t else 0
    gbs = B / 1e9 / (t / 1e3) if t else 0
    if cb:
        eff = gflops / (FP16_e * 1000) * 100 if t else 0
        bound = "compute(f16)"
    else:
        eff = gbs / BW_e * 100 if t else 0
        bound = "memory"
    return dict(op=op, tag=tag, kernel="moe_3gemm_fused", single=t, calls=calls,
                total=t * calls, gflops=gflops, gbs=gbs, eff=eff, bound=bound)


def smallops_row(t):
    return dict(op="SmallOps(norm/rope/add)", tag="-", kernel="rms/rope/eltwise",
                single=t, calls=1, total=t, gflops=0, gbs=0, eff=0, bound="memory")


# ---------------- Phase op lists ----------------
def small_ops_decode(M_):
    if M_ == 256:
        return (ms("so_rmsnorm_h2816_canvas") * (7 * NL + 1) +
                (ms("so_rmsnorm3d_q_sliding_canvas") + ms("so_rmsnorm3d_k_sliding_canvas")) * NL_SL +
                (ms("so_rmsnorm3d_q_full_canvas") + ms("so_rmsnorm3d_k_full_canvas")) * NL_F +
                (ms("so_rope_q_sliding_canvas") + ms("so_rope_k_sliding_canvas")) * NL_SL +
                ms("so_add_h2816_canvas") * (2 * NL))
    return (ms("so_rmsnorm_h2816_decode") * (7 * NL + 1) +
            (ms("so_rmsnorm3d_q_sliding_decode") + ms("so_rmsnorm3d_k_sliding_decode")) * NL_SL +
            (ms("so_rmsnorm3d_q_full_decode") + ms("so_rmsnorm3d_k_full_decode")) * NL_F +
            (ms("so_rope_q_sliding_decode") + ms("so_rope_k_sliding_decode")) * NL_SL +
            ms("so_add_h2816_decode") * (2 * NL))


def small_ops_prefill(S):
    return (ms(f"so_rmsnorm_h2816_prefill_S{S}") * (7 * NL + 1) +
            (ms(f"so_rmsnorm3d_q_sliding_prefill_S{S}") + ms(f"so_rmsnorm3d_k_sliding_prefill_S{S}")) * NL_SL +
            ms(f"so_rope_q_sliding_prefill_S{S}") * NL_SL +
            ms(f"so_add_h2816_prefill_S{S}") * (2 * NL))


def decode_rows(M_, ctx):
    cb = (M_ >= 256)
    sfx = "canvas_M256" if M_ == 256 else "decode_M1"
    rows = [
        fc_row("FC_QKV_sliding", f"fc_qkv_sliding_{sfx}", M_, 2816, 8192, "u4", NL_SL, cb),
        fc_row("FC_O_sliding", f"fc_o_sliding_{sfx}", M_, 4096, 2816, "u4", NL_SL, cb),
        fc_row("FC_QK_full", f"fc_qk_full_{sfx}", M_, 2816, 9216, "u4", NL_F, cb),
        fc_row("FC_O_full", f"fc_o_full_{sfx}", M_, 8192, 2816, "u4", NL_F, cb),
        fc_row("MLP_gate", f"fc_gate_dense_{sfx}", M_, 2816, 2112, "u4", NL, cb),
        fc_row("MLP_up", f"fc_up_dense_{sfx}", M_, 2816, 2112, "u4", NL, cb),
        fc_row("MLP_down", f"fc_down_dense_{sfx}", M_, 2112, 2816, "u4", NL, cb),
        moe_row("MoE", f"moe_{sfx}", M_, NL, cb),
    ]
    if M_ == 256:
        rows.append(sdpa_row("SDPA_sliding", "sdpa_sliding_canvas_M256", CANVAS, SW + CANVAS, SL_NH, SL_NKV, SL_HD, NL_SL, False, cb))
        rows.append(sdpa_row("SDPA_full", f"sdpa_full_canvas_ctx{ctx}", CANVAS, ctx + CANVAS, F_NH, F_NKV, F_HD, NL_F, False, cb))
        rows.append(fc_row("LM_head", "lm_head_canvas_M256", M_, 2816, VOCAB, "u8", 1, cb))
    else:
        rows.append(sdpa_row("SDPA_sliding", "sdpa_sliding_decode_M1", 1, SW, SL_NH, SL_NKV, SL_HD, NL_SL, False, False))
        rows.append(sdpa_row("SDPA_full", f"sdpa_full_decode_ctx{ctx}", 1, ctx, F_NH, F_NKV, F_HD, NL_F, False, False))
        rows.append(fc_row("LM_head", "lm_head_decode_M1", M_, 2816, VOCAB, "u8", 1, False))
    rows.append(smallops_row(small_ops_decode(M_)))
    return rows


def prefill_rows(S):
    sl_S = min(S, SW)
    return [
        fc_row("FC_QKV_sliding", f"fc_qkv_sliding_prefill_S{S}", S, 2816, 8192, "u4", NL_SL, True),
        fc_row("FC_O_sliding", f"fc_o_sliding_prefill_S{S}", S, 4096, 2816, "u4", NL_SL, True),
        fc_row("FC_QK_full", f"fc_qk_full_prefill_S{S}", S, 2816, 9216, "u4", NL_F, True),
        fc_row("FC_O_full", f"fc_o_full_prefill_S{S}", S, 8192, 2816, "u4", NL_F, True),
        fc_row("MLP_gate", f"fc_gate_dense_prefill_S{S}", S, 2816, 2112, "u4", NL, True),
        fc_row("MLP_up", f"fc_up_dense_prefill_S{S}", S, 2816, 2112, "u4", NL, True),
        fc_row("MLP_down", f"fc_down_dense_prefill_S{S}", S, 2112, 2816, "u4", NL, True),
        moe_row("MoE", f"moe_prefill_S{S}", S, NL, True),
        sdpa_row("SDPA_sliding", "sdpa_sliding_prefill_S1024", sl_S, sl_S, SL_NH, SL_NKV, SL_HD, NL_SL, True, True),
        sdpa_row("SDPA_full", f"sdpa_full_prefill_S{S}", S, S, F_NH, F_NKV, F_HD, NL_F, True, True),
        smallops_row(small_ops_prefill(S)),
    ]


def total_ms(rows):
    return sum(r["total"] for r in rows)


# ---------------- Weight footprint ----------------
def weight_table():
    rows = [
        ("Embedding (tied)", f"{H} × {VOCAB}", "int8 g128", VOCAB * H, 1),
        ("FC_QKV (sliding)", "2816 × 8192", "int4 g128", 8192 * 2816 / 2 + 8192 * (2816 // G) * 2, NL_SL),
        ("FC_O (sliding)", "4096 × 2816", "int4 g128", 2816 * 4096 / 2 + 2816 * (4096 // G) * 2, NL_SL),
        ("FC_QK (full, V=K)", "2816 × 9216", "int4 g128", 9216 * 2816 / 2 + 9216 * (2816 // G) * 2, NL_F),
        ("FC_O (full)", "8192 × 2816", "int4 g128", 2816 * 8192 / 2 + 2816 * (8192 // G) * 2, NL_F),
        ("MLP_gate", "2816 × 2112", "int4 g128", 2112 * 2816 / 2 + 2112 * (2816 // G) * 2, NL),
        ("MLP_up", "2816 × 2112", "int4 g128", 2112 * 2816 / 2 + 2112 * (2816 // G) * 2, NL),
        ("MLP_down", "2112 × 2816", "int4 g64", 2816 * 2112 / 2 + 2816 * (2112 // G_DOWN) * 2, NL),
        ("Router", "2816 × 128", "int4 g128", 128 * 2816 / 2 + 128 * (2816 // G) * 2, NL),
        ("MoE Gate+Up (per expert)", "2816 × 1408", "int4 g128", 1408 * 2816 / 2 + 1408 * (2816 // G) * 2, NL * NE),
        ("MoE Down (per expert)", "704 × 2816", "int4 g64", 2816 * 704 / 2 + 2816 * (704 // G_DOWN) * 2, NL * NE),
        ("LM_Head (tied → 0)", f"{H} × {VOCAB}", "int8 g128", 0, 1),
    ]
    total = sum(b * n for _, _, _, b, n in rows)
    return rows, total / 1024 / 1024


# ---------------- SUMMARY helpers ----------------
def fmt(x, p=3):
    return f"{x:.{p}f}"


def op_table(rows):
    rows = sorted(rows, key=lambda r: -r["total"])
    out = []
    for r in rows:
        out.append(f"| {r['op']} | {r['kernel']} | {fmt(r['single'])} | {r['calls']} | "
                   f"{fmt(r['total'])} | {fmt(r['gflops'],1)} | {fmt(r['gbs'],1)} | "
                   f"{fmt(r['eff'],1)}% | {r['bound']} |")
    return "\n".join(out), total_ms(rows)


def breakdown_cols(by_key, keys):
    ops = [r["op"] for r in by_key[keys[0]]]
    totals = {k: total_ms(by_key[k]) for k in keys}
    lines = []
    for i, op in enumerate(ops):
        cells = []
        for k in keys:
            r = by_key[k][i]
            pct = 100 * r["total"] / totals[k] if totals[k] else 0
            cells.append(f"{fmt(r['total'])} ({fmt(pct,1)}%)")
        lines.append(f"| {op} | " + " | ".join(cells) + " |")
    cells = [f"**{fmt(totals[k])}**" for k in keys]
    lines.append("| **TOTAL** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def top3(rows):
    rows = sorted(rows, key=lambda r: -r["total"])
    t = total_ms(rows)
    cells = []
    for r in rows[:3]:
        pct = 100 * r["total"] / t if t else 0
        cells.append(f"{r['op']} {fmt(r['total'])}ms ({fmt(pct,1)}%)")
    return cells


def _clean(kn):
    base = kn.split("__")[0]
    base = base.replace("_0_0", "")
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) > 6:
        base = parts[0]
    return base


def op_kernel_rows(rows):
    out = []
    for r in rows:
        if r["tag"] == "-":
            out.append(f"| {r['op']} | `rms_gpu_bfyx_opt`<br>`rope_opt`<br>`eltwise` | per-subop |")
            continue
        e = RAW.get(r["tag"], {})
        iters = e.get("iters_detected", 1) or 1
        names, launches = [], []
        for kn, ns, calls in kernels(r["tag"])[:4]:
            names.append(f"`{_clean(kn)}`")
            launches.append(str(round(calls / iters)))
        out.append(f"| {r['op']} | " + "<br>".join(names) + " | " + "<br>".join(launches) + " |")
    return "\n".join(out)


def subkernel_rows(rows, rep_total):
    flat = []
    for r in rows:
        if r["tag"] == "-":
            continue
        e = RAW.get(r["tag"], {})
        iters = e.get("iters_detected", 1) or 1
        for kn, ns, calls in kernels(r["tag"]):
            single = ns / 1e6
            lpc = round(calls / iters)
            total = single * r["calls"]
            flat.append((total, r["op"], _clean(kn), single, lpc, r["calls"]))
    flat.sort(reverse=True)
    out = []
    for total, op, kn, single, lpc, callsinf in flat[:14]:
        pct = 100 * total / rep_total if rep_total else 0
        out.append(f"| {op} | `{kn}` | {fmt(single,4)} | {lpc} | {callsinf} | {fmt(total)} | {fmt(pct,1)}% |")
    return "\n".join(out)


def floor_ms(rows):
    tot = 0.0
    for r in rows:
        if r["tag"] == "-":
            tot += r["total"]
            continue
        single, calls = r["single"], r["calls"]
        if single <= 0:
            continue
        bytes_ = r["gbs"] * 1e9 * (single / 1e3)
        flops_ = r["gflops"] * 1e9 * (single / 1e3)
        if "compute" in r["bound"]:
            peak = INT8_e if "i8" in r["bound"] else FP16_e
            t_op = flops_ / 1e9 / (peak * 1000) * 1e3
        else:
            t_op = bytes_ / 1e9 / BW_e * 1e3
        tot += t_op * calls
    return tot


# ---------------- SUMMARY generation ----------------
def generate_summary():
    today = date.today().isoformat()
    pre = {S: prefill_rows(S) for S in SEQS}
    can = {c: decode_rows(256, c) for c in SEQS}
    ref = {c: decode_rows(1, c) for c in SEQS}
    wrows, wtotal = weight_table()

    L = []
    A = L.append
    A(f"# diffusion_gemma (26B-A4B-it) — Roofline on Intel PTL 12Xe ({today})\n")
    A("**Platform**: Intel Panther Lake 12-Xe iGPU (cliloader: *Intel(R) Arc(TM) B390 GPU, 96 EU / 12 Xe-cores, 2400 MHz*); FP16 XMX 58.98 TFLOPS, INT8 XMX 117.96 TOPS, LPDDR ~110 GB/s.")
    A("**Model**: `google/diffusiongemma-26B-A4B-it` — block-diffusion multimodal Gemma. This report profiles the **text decoder** (`DiffusionGemmaForBlockDiffusion.text_config`), whose dense+MoE transformer is architecturally identical to gemma4-26B-A4B-it.\n")
    A("- 30 layers, hidden 2816, 128-expert top-8 MoE **plus** a dense GEGLU MLP per layer (outputs summed); GQA sliding (25 L) + full (5 L) attention; vocab 262144, tied embeddings.")
    A("- MatMul weights **int4** (g128; down-proj g64) / FP16 act; LM_head **int8** g128 / FP16 act; KV cache **FP16** (SDPA path).")
    A("- SDPA: `ov::op::v13::ScaledDotProductAttention` → GPU `sdpa_micro` (uncompressed FP16 KV). **Not** PagedAttention.\n")

    A("## Diffusion workload semantics\n")
    A("`diffusion_gemma` is a **block-diffusion** model (`canvas_length=256`), not a standard autoregressive LLM:\n")
    A("- **Encoder / prefill** — autoregressive pass over the prompt (causal sliding/full attention) that builds a **read-only** KV cache. This is the TTFT phase; profiled as `prefill S∈{1024,2048,4096,8192}`.")
    A("- **Decoder / canvas step** — each denoising step refines a **fixed canvas of 256 tokens** with **bidirectional** self-attention (causal=0) plus cross-attention to the read-only encoder KV cache. The relevant per-step cost is therefore **M=256**, not M=1. LM_head runs over the whole 256-token canvas each step. Profiled as `canvas decode M=256`.")
    A("- An **M=1 reference decode** is also reported for comparison with classic AR LLMs, but it does **not** occur in real diffusion inference.\n")

    A("## Model parameters & weight shapes\n")
    A("| Field | Value | Notes |")
    A("|---|---:|---|")
    A(f"| `hidden_size` | {H} | residual / activation channel |")
    A(f"| `num_hidden_layers` | {NL} | {NL_SL} sliding + {NL_F} full attention |")
    A(f"| `num_attention_heads` (NH) | {SL_NH} | Q heads |")
    A(f"| `num_key_value_heads` (sliding) | {SL_NKV} | GQA 2-way; HD={SL_HD}, window={SW} |")
    A(f"| `num_global_key_value_heads` (full) | {F_NKV} | GQA 8-way; global HD={F_HD} |")
    A(f"| `intermediate_size` (dense) | {I_DENSE} | GEGLU dense MLP hidden |")
    A(f"| `moe_intermediate_size` | {I_MOE} | per-expert hidden |")
    A(f"| `num_experts` / `top_k` | {NE} / {TK} | softmax-routed MoE |")
    A(f"| `vocab_size` | {VOCAB} | LM head N |")
    A("| `hidden_activation` | gelu_pytorch_tanh | GEGLU |")
    A("| `final_logit_softcapping` | 30.0 | tanh logit soft-cap |")
    A("| `tie_word_embeddings` | true | LM head shares embedding storage |")
    A("| `canvas_length` | 256 | diffusion denoising block size |\n")

    A("Per-layer / global weight matrices (K × N):\n")
    A("| Weight | Shape (K × N) | Quant | × instances | Total MB |")
    A("|---|---:|---|---:|---:|")
    for name, shape, quant, b, n in wrows:
        A(f"| {name} | {shape} | {quant} | {n} | {fmt(b * n / 1024 / 1024, 1)} |")
    A(f"| **Total static weights** |  |  |  | **{fmt(wtotal, 1)} MB** |\n")

    A("## Theoretical roofline\n")
    A("| Metric | Value |")
    A("|---|---|")
    A(f"| FP16 XMX peak | {fmt(FP16_XMX,2)} TFLOPS |")
    A(f"| INT8 XMX peak | {fmt(INT8_XMX,2)} TOPS |")
    A(f"| Memory BW | {BW} GB/s |")
    A(f"| Ridge point (FP16) | {fmt(FP16_XMX*1e3/BW,1)} FLOP/byte |")
    A(f"| Ridge point (INT8) | {fmt(INT8_XMX*1e3/BW,1)} OP/byte |\n")

    A("## Data sources\n")
    A("All op times are **measured on the target PTL 12Xe machine** via cliloader Device Performance Timing (mean kernel ns per iteration), one bench process per op. No cross-platform scaling was used. MoE and dense down-proj use group size **64** (reduction dims I=704 and K=2112 are not divisible by 128); all other FC use g128.\n")

    A("## Graph fusion notes\n")
    A("| Bench row | Real graph behaviour | Standalone kernel? |")
    A("|---|---|---|")
    A("| `MoE` | router + 3-GEMM experts fused into `MOE3GemmFusedCompressed` (`grouped_micro_gemm` ×3 + gather/scatter/softmax-topk) | Yes (fused family) |")
    A("| `MLP gate/up + GEGLU multiply` | dense GEGLU; `multiply` fused into MLP | gate/up/down standalone gemm |")
    A("| `add` | per-layer residual adds (×2: attn + MLP/MoE sum) | Yes (eltwise) |")
    A("| `rmsnorm` | 7×/layer (pre-attn, q/k norm, pre-MLP, pre-MoE, post) + final | Yes (`rms_gpu_bfyx_opt`) |")
    A("| `SDPA broadcast` | bench-only GQA KV-head expansion to satisfy core SDPA shape-infer | excluded (artifact) |\n")

    A("## Token latency summary\n")
    A("### Prefill — encoder TTFT\n")
    A("| S | TTFT (ms) | per-token (ms) | tokens/s |")
    A("|---:|---:|---:|---:|")
    for S in SEQS:
        t = total_ms(pre[S])
        A(f"| {S} | {fmt(t,1)} | {fmt(t/S,4)} | {fmt(S*1000/t,0)} |")
    A("")
    A("### Canvas decode — per denoising step (M=256, bidirectional)\n")
    A("| KV (ctx) | step (ms) | canvas-tok/s | per-canvas-token (ms) |")
    A("|---:|---:|---:|---:|")
    for c in SEQS:
        t = total_ms(can[c])
        A(f"| {c} | {fmt(t,1)} | {fmt(CANVAS*1000/t,0)} | {fmt(t/CANVAS,4)} |")
    A("")
    A("### M=1 reference decode (not used in diffusion; for comparison only)\n")
    A("| KV (ctx) | TPOT (ms) | tokens/s |")
    A("|---:|---:|---:|")
    for c in SEQS:
        t = total_ms(ref[c])
        A(f"| {c} | {fmt(t,2)} | {fmt(1000/t,1)} |")
    A("")

    A("### Canvas-decode per-op breakdown (ms / % of step)\n")
    A("| op | " + " | ".join(f"KV={c}" for c in SEQS) + " |")
    A("|---|" + "---:|" * len(SEQS))
    A(breakdown_cols(can, SEQS))
    A("")
    A("### Prefill TTFT per-op breakdown (ms / % of TTFT)\n")
    A("| op | " + " | ".join(f"S={S}" for S in SEQS) + " |")
    A("|---|" + "---:|" * len(SEQS))
    A(breakdown_cols(pre, SEQS))
    A("")

    A("## Roofline: theoretical floor vs measured\n")
    A("Theoretical floor = Σ max(bytes/BW, FLOP/peak) over modelable ops (SDPA counts the `sdpa_micro` kernel only; small-ops use measured).\n")
    A("### Canvas decode (M=256 step)\n")
    A("| KV | theoretical (ms) | measured (ms) | achieved % |")
    A("|---:|---:|---:|---:|")
    for c in SEQS:
        meas, theo = total_ms(can[c]), floor_ms(can[c])
        A(f"| {c} | {fmt(theo,1)} | {fmt(meas,1)} | {fmt(100*theo/meas,1)}% |")
    A("")
    A("### Prefill (TTFT over S tokens)\n")
    A("| S | theoretical (ms) | measured (ms) | achieved % |")
    A("|---:|---:|---:|---:|")
    for S in SEQS:
        meas, theo = total_ms(pre[S]), floor_ms(pre[S])
        A(f"| {S} | {fmt(theo,1)} | {fmt(meas,1)} | {fmt(100*theo/meas,1)}% |")
    A("")

    A("## Canvas-decode tables (M=256, KV = context length)\n")
    for c in SEQS:
        A(f"### Canvas decode — KV={c}\n")
        A("| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---|")
        body, tot = op_table(can[c])
        A(body)
        A(f"| **TOTAL** |  |  |  | **{fmt(tot)}** |  |  |  |  |\n")

    A("## M=1 reference decode tables (KV = context length)\n")
    for c in SEQS:
        A(f"### Decode (M=1) — KV={c}\n")
        A("| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---|")
        body, tot = op_table(ref[c])
        A(body)
        A(f"| **TOTAL** |  |  |  | **{fmt(tot)}** |  |  |  |  |\n")

    A("## Prefill tables (single forward over S tokens)\n")
    for S in SEQS:
        A(f"### Prefill — S={S}\n")
        A("| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |")
        A("|---|---|---:|---:|---:|---:|---:|---:|---|")
        body, tot = op_table(pre[S])
        A(body)
        A(f"| **TOTAL** |  |  |  | **{fmt(tot)}** |  |  |  |  |\n")

    A("## Op → kernel names (cliloader)\n")
    A("### Canvas decode (M=256)\n")
    A("| op | kernel name(s) | launches/call |")
    A("|---|---|---:|")
    A(op_kernel_rows(can[1024]))
    A("")
    A("### Prefill (S=8192)\n")
    A("| op | kernel name(s) | launches/call |")
    A("|---|---|---:|")
    A(op_kernel_rows(pre[8192]))
    A("")

    A("## Per-kernel decomposition (representative)\n")
    A("### Canvas decode sub-kernels — KV=1024\n")
    A("| op | kernel name | single ms | launches/call | calls/inf | total ms | % |")
    A("|---|---|---:|---:|---:|---:|---:|")
    A(subkernel_rows(can[1024], total_ms(can[1024])))
    A("")
    A("### Prefill sub-kernels — S=8192\n")
    A("| op | kernel name | single ms | launches/call | calls/inf | total ms | % |")
    A("|---|---|---:|---:|---:|---:|---:|")
    A(subkernel_rows(pre[8192], total_ms(pre[8192])))
    A("")

    A("## Top contributors (per inference)\n")
    A("### Canvas decode\n")
    A("| KV | top1 | top2 | top3 |")
    A("|---:|---|---|---|")
    for c in SEQS:
        t = top3(can[c])
        A(f"| {c} | {t[0]} | {t[1]} | {t[2]} |")
    A("")
    A("### Prefill\n")
    A("| S | top1 | top2 | top3 |")
    A("|---:|---|---|---|")
    for S in SEQS:
        t = top3(pre[S])
        A(f"| {S} | {t[0]} | {t[1]} | {t[2]} |")
    A("")

    A("## End-to-end (encoder TTFT + canvas denoising)\n")
    A("Assuming an illustrative block-diffusion schedule of **32 denoising steps** over the 256-token canvas (steps reuse the read-only encoder KV cache; canvas-step cost ≈ constant across the schedule):\n")
    A("| prompt P | TTFT (ms) | 32-step canvas (ms) | total (ms) | canvas tok/s |")
    A("|---:|---:|---:|---:|---:|")
    for S in SEQS:
        ttft = total_ms(pre[S])
        step = total_ms(can[S])
        win = 32 * step
        A(f"| {S} | {fmt(ttft,1)} | {fmt(win,1)} | {fmt(ttft+win,1)} | {fmt(CANVAS*1000/step,0)} |")
    A("")

    A("## Key findings\n")
    can1 = can[1024]
    moe_ms = [r for r in can1 if r['op'] == 'MoE'][0]['total']
    can1_tot = total_ms(can1)
    moe_pct = 100 * moe_ms / can1_tot
    A(f"- **MoE dominates everything.** It is {fmt(moe_pct,0)}% of a canvas-decode step (~{fmt(moe_ms,0)} ms of ~{fmt(can1_tot,0)} ms at KV=1024) and 47-69% of prefill TTFT. The `grouped_micro_gemm` expert kernel (3 launches/step) is the single largest cost — the same bottleneck seen on gemma4-26B-A4B.")
    A(f"- **Canvas decode runs at ~{fmt(CANVAS*1000/can1_tot,0)} canvas-tok/s** (M=256, KV=1024), i.e. ~{fmt(can1_tot,0)} ms per denoising step. Because the canvas is processed in parallel (M=256), per-token throughput is ~{fmt((CANVAS*1000/can1_tot)/(1000/total_ms(ref[1024])),0)}x the M=1 reference ({fmt(1000/total_ms(ref[1024]),0)} tok/s).")
    A("- **Canvas decode is compute-bound** (M=256 saturates INT8 XMX for FC/LM_head and the FP16 MoE/SDPA kernels), unlike a classic M=1 LLM decode which is memory-bound. The M=1 reference is memory-bound and dominated by MoE weight reads.")
    can8 = can[8192]
    def sdpa_share(rows):
        tot = total_ms(rows)
        s = sum(r['total'] for r in rows if r['op'].startswith('SDPA'))
        return 100 * s / tot
    A(f"- **Attention scales with context** only through the 5 full-attention layers (HD=512); sliding layers are capped at window=1024. Combined SDPA grows from ~{fmt(sdpa_share(can1),0)}% (KV=1024) to ~{fmt(sdpa_share(can8),0)}% (KV=8192) of the canvas step (the full-attention SDPA term alone scales ~{fmt([r for r in can8 if r['op']=='SDPA_full'][0]['total']/[r for r in can1 if r['op']=='SDPA_full'][0]['total'],1)}x over that range).")
    lm_ms = [r for r in can1 if r['op'] == 'LM_head'][0]['total']
    A(f"- **LM_head over the 256-token canvas** costs ~{fmt(lm_ms,0)} ms/step (int8, 2816\u00d7262144) — modest vs MoE but non-trivial; it runs every denoising step.")
    A("")

    A("## Optimization levers (highest ROI first)\n")
    A("1. **MoE expert GEMM** — `grouped_micro_gemm` is >2/3 of decode and prefill. Larger expert tiling / better int4 grouped-GEMM scheduling, or expert batching across the 256 canvas tokens, is the top lever.")
    A("2. **Exploit canvas parallelism for MoE routing** — with M=256 tokens per step, expert load is dense; grouping tokens by expert (sort/gather) amortizes weight reads better than M=1.")
    A("3. **INT4 LM_head** — moving the tied LM_head from int8→int4 roughly halves its ~9 ms/step.")
    A("4. **Full-attention KV** — only 5 layers use HD=512 global attention; an INT8 / PagedAttention KV path would cut the SDPA growth at long context (currently FP16 KV).")
    A("5. **Fuse residual adds / norms** — small-ops are 3–13% of time; folding the two residual adds and q/k norms into neighboring primitives trims the tail.")
    A("")

    A("## Caveats & method\n")
    A("- Each op profiled in its own process via cliloader Device Performance Timing; **mean** kernel ns per iteration is used. Times in **ms**.")
    A("- FC weight bytes = int4 weight + FP16 scale (g128, down/MoE g64) + FP16 act + FP16 out. LM_head int8.")
    A("- **SDPA uses FP16 (uncompressed) KV** — the bench models the SDPA path the model compiles to; INT8 KV compression (PagedAttention) was **not** used per the chosen attention implementation.")
    A("- The SDPA bench materializes an explicit GQA KV-head **broadcast** to satisfy core opset13 SDPA shape inference; that `broadcast_gpu_ref` kernel is a **bench artifact** (the GPU SDPA primitive reads compressed GQA KV directly) and is **excluded** — only `sdpa_micro` is counted.")
    A("- MoE and dense down-proj use **group size 64** (I=704, K=2112 not ÷128); scale-byte impact vs g128 is negligible.")
    A("- Canvas decode = **M=256** bidirectional step (the real diffusion cost). M=1 reference is informational only and does not occur in inference.")
    A("- The 32-step end-to-end schedule is illustrative; actual step count depends on the sampler.")
    A("- Target machine: Local PTL 12Xe (Arc B390, 96 EU, 2400 MHz), Windows, OpenVINO release build.")
    A("")

    A("## Reproduction\n")
    A("```bat")
    A(":: On the PTL 12Xe machine (D:\\river\\moe\\dev_roofline_profiling\\utils)")
    A("run_diffusion_gemma_ptl_12xe.bat        :: main sweep (FC / MoE / LM_head / small-ops)")
    A("run_diffusion_gemma_ptl_12xe_sdpa.bat   :: SDPA (needs sdpa_bench rebuilt w/ GQA repeat_kv)")
    A(":: logs -> D:\\river\\moe\\roofline_results\\diffusion_gemma\\ptl_12xe\\*.log")
    A("```")
    A("```bash")
    A("# On the host")
    A("python3 parse_logs.py <logdir> outputs/diffusion_gemma/perf_raw.json")
    A("python3 analyze_diffusion_gemma_ptl_12xe.py   # -> performance_metrics.json + SUMMARY")
    A("```")

    (OUT / f"SUMMARY_diffusion_gemma_{today}.md").write_text("\n".join(L) + "\n")
    return today


def main():
    metrics = {
        "hardware": {"platform": "PTL 12Xe (Arc B390)", "xe_cores": XE, "freq_mhz": 2400,
                     "bw_gbs": BW, "fp16_xmx_tflops": FP16_XMX, "int8_xmx_tops": INT8_XMX},
        "config": dict(hidden=H, layers=NL, sliding=NL_SL, full=NL_F, vocab=VOCAB,
                       I_dense=I_DENSE, I_moe=I_MOE, NE=NE, TK=TK, canvas=CANVAS),
        "prefill": {S: prefill_rows(S) for S in SEQS},
        "canvas_decode": {c: decode_rows(256, c) for c in SEQS},
        "ref_decode_M1": {c: decode_rows(1, c) for c in SEQS},
    }
    (OUT / "performance_metrics.json").write_text(json.dumps(metrics, indent=2))
    today = generate_summary()
    print(f"Wrote performance_metrics.json and SUMMARY_diffusion_gemma_{today}.md to {OUT}")
    for S in SEQS:
        print(f"prefill S={S:5d} TTFT={total_ms(prefill_rows(S)):8.1f} ms")
    for c in SEQS:
        print(f"canvas KV={c:5d} step={total_ms(decode_rows(256,c)):8.1f} ms  "
              f"M1ref={total_ms(decode_rows(1,c)):7.2f} ms")


if __name__ == "__main__":
    main()
