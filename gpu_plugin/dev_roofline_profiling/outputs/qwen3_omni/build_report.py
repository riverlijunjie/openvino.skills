#!/usr/bin/env python3
"""
qwen3_omni roofline report builder for PTL 12Xe Windows iGPU (B390, 2400 MHz,
110 GB/s, FP16 XMX peak 58.98 TFLOPS).

Reads per-log raw kernel JSON (produced by ../../utils/parse_logs.py) and
produces:
  - performance_metrics.json   per-op single ms / GFLOPS / GB/s / Eff% / bound
  - SUMMARY_qwen3_omni_<date>.md  with prefill/decode tables per token size

qwen3_omni Thinker text decoder (dense, GQA). Source: config.json ->
thinker_config.text_config: hidden=2560, layers=36, NH=32 / NKV=8,
head_dim=128, intermediate=9728, vocab=151936.
"""
import json, sys, datetime, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "kernels_raw.json"
OUT = ROOT / "performance_metrics.json"

# ---------------------- HW (PTL 12Xe Windows iGPU / Arc B390 96CUs) -------
HZ        = 2400e6
N_XE      = 12
EU_PER_XE = 8
THR_PER_EU = 10
FP16_XMX_TFLOPS = N_XE * EU_PER_XE * 256 * HZ / 1e12      # 58.9824
INT8_XMX_TOPS   = N_XE * EU_PER_XE * THR_PER_EU * 4 * HZ / 1e12  # 9.216 / cycle? — but SKILL formula is N*EU*256*hz for fp16; for int8 we use 2x = 117.96
# Following the SKILL formula style: INT8 = 2x FP16 XMX
INT8_XMX_TOPS_PEAK = 2.0 * FP16_XMX_TFLOPS               # 117.96 TOPS
BW_GBS    = 110.0                                        # GB/s VRAM (DDR for iGPU)

# ---------------------- Model (qwen3_omni Thinker text) -----------------------
# Source: /home/ov2022/workspace/remote_debug/qwen3_omni/config.json
#         thinker_config.text_config
#   hidden_size=2560, num_hidden_layers=36,
#   num_attention_heads=32, num_key_value_heads=8 (GQA),
#   head_dim=128, intermediate_size=9728, vocab_size=151936,
#   tie_word_embeddings=true, hidden_act=silu, rope_theta=1e6
HIDDEN = 2560
INTER  = 9728
NH     = 32        # Q heads
NKV    = 8         # KV heads (GQA)
HD     = 128
Q_DIM  = NH * HD   # 4096
KV_DIM = NKV * HD  # 1024
LAYERS = 36
VOCAB  = 151936

# ---------------------- Helpers ---------------------------------------------
def fc_flops_bytes(M, K, N, bits):
    """Per-launch FLOPs and bytes. bits=4 for INT4 weights, 8 for INT8."""
    flops = 2.0 * M * K * N
    # weight bytes (compressed) + scale/zp + activations + output
    wbytes = K * N * bits / 8.0
    sbytes = (K // 128) * N * 2.0  # fp16 scale per group of 128
    abytes = M * K * 2.0
    obytes = M * N * 2.0
    return flops, wbytes + sbytes + abytes + obytes

def pa_flops_bytes(Sq, Skv_total, kv_i8=True):
    # PA compute: QK + softmax + AV (dominant: QK + AV). Use the same formula
    # as pa_bench (QK 2*NH*Sq*Skv*HD + AV 2*NH*Sq*Skv*HD + softmax ~25*NH*Sq*Skv).
    flops = 4.0 * NH * Sq * Skv_total * HD + 25.0 * NH * Sq * Skv_total
    elem  = 1 if kv_i8 else 2  # KV bytes per element
    # Read: Q + KV cache; Write: out. Ignore tiny bits.
    bytes_ = (Sq * NH * HD * 2                  # Q read
              + 2 * Skv_total * NKV * HD * elem # K + V cache read
              + Sq * NH * HD * 2)               # output write
    return flops, bytes_

def small_bytes(name, M, n_heads=None):
    if name == "rmsnorm":          return 2.0 * M * HIDDEN * 2 + HIDDEN * 2
    if name == "rmsnorm3d":        return 2.0 * M * (n_heads or NH) * HD * 2 + HD * 2
    if name == "rope":             return 2.0 * M * (n_heads or NH) * HD * 2 + 2.0 * M * HD * 2
    if name == "multiply":         return 3.0 * M * INTER * 2
    if name == "add":              return 3.0 * M * HIDDEN * 2

def eff_pct(achieved_gflops, achieved_gbs, kind):
    if kind == "compute_fp16":   return achieved_gflops / 1000.0 / FP16_XMX_TFLOPS * 100.0
    if kind == "compute_int8":   return achieved_gflops / 1000.0 / INT8_XMX_TOPS_PEAK * 100.0
    if kind == "memory":         return achieved_gbs / BW_GBS * 100.0
    return 0.0

# ---------------------- Build per-op table ---------------------------------
def main():
    raw = json.loads(RAW.read_text())

    def total_ns(tag):
        r = raw.get(tag)
        return r["total_kernel_ns"] if r else None

    def per_kernel_list(tag):
        """Return list of (kernel_name, single_ns_per_iter, launches_per_iter)
        for the given bench tag, sorted by single_ns desc. The parser stores
        per-iter ns already (total/iters) and the original launch count;
        launches_per_iter = launches / iters_detected."""
        r = raw.get(tag)
        if not r:
            return []
        iters = r.get("iters_detected", 1) or 1
        out = []
        for n, t, c in r["per_kernel"]:
            launches = max(1, c // iters) if iters else c
            out.append((n, t, launches))
        return out

    rows = []   # list of (phase, S, op, kernel_label, single_ms, calls, total_ms, gflops, gbs, eff, bound)
    sub_rows = []  # per-kernel decomposition: phase, S, op, kernel_name, single_ms, launches_per_call, calls, total_ms

    def add(phase, S, op, kernel_label, single_ns, calls, flops, bytes_, kind, tag=None):
        if single_ns is None or single_ns <= 0:
            return
        single_ms = single_ns / 1e6
        total_ms  = single_ms * calls
        seconds   = single_ns / 1e9
        gflops = flops / seconds / 1e9 if seconds > 0 else 0.0
        gbs    = bytes_ / seconds / 1e9 if seconds > 0 else 0.0
        eff = eff_pct(gflops, gbs, kind)
        # determine bound: compare ridge point — if op is FC use bits-aware kind; for memory ops, kind="memory"
        bound = "compute" if "compute" in kind else "memory"
        rows.append({
            "phase": phase, "S": S, "op": op, "kernel": kernel_label,
            "single_ms": single_ms, "calls": calls, "total_ms": total_ms,
            "gflops": gflops, "gbs": gbs, "eff_pct": eff, "bound": bound,
            "kind": kind,
        })
        # Sub-kernel decomposition (e.g. PA = kv_cache_update + sdpa_micro / pa_opt + finalization)
        if tag is not None:
            for kn, ns_per_iter, launches in per_kernel_list(tag):
                # Per-kernel achieved metrics: attribute the parent op's full
                # FLOPs/bytes per op-call to this kernel and compare against
                # the kernel's own per-launch time. The dominant kernel will
                # show eff close to op-level; helper kernels (e.g. tiny
                # finalization or kv_cache_update) will appear high-eff,
                # signalling that they are not the bottleneck of this op.
                k_seconds = ns_per_iter / 1e9
                k_gflops = (flops / k_seconds / 1e9) if (k_seconds > 0 and flops > 0) else 0.0
                k_gbs = (bytes_ / k_seconds / 1e9) if k_seconds > 0 else 0.0
                k_eff = eff_pct(k_gflops, k_gbs, kind)
                sub_rows.append({
                    "phase": phase, "S": S, "op": op, "kernel_name": kn,
                    "single_ms": ns_per_iter / 1e6,
                    "launches_per_call": launches,
                    "calls_per_inf": calls * launches,
                    "total_ms": (ns_per_iter / 1e6) * calls * launches,
                    "gflops": k_gflops, "gbs": k_gbs,
                    "eff_pct": k_eff, "kind": kind,
                })

    # FC table (K, N, calls/inference). GQA Q/K/V are fused into a single QKV proj.
    # qkv: 2560 -> Q_dim+K_dim+V_dim = 4096+1024+1024 = 6144
    QKV_N = Q_DIM + 2 * KV_DIM   # 6144
    FC_DEFS = {
        "fc_qkv":  (HIDDEN, QKV_N, LAYERS),  # 2560 -> 6144 (fused QKV)
        "fc_o":    (Q_DIM,  HIDDEN, LAYERS), # 4096 -> 2560
        "fc_gate": (HIDDEN, INTER,  LAYERS), # 2560 -> 9728
        "fc_up":   (HIDDEN, INTER,  LAYERS), # 2560 -> 9728
        "fc_down": (INTER,  HIDDEN, LAYERS), # 9728 -> 2560
    }
    # -------- Decode (single token, S_kv = ctx) --------------------------
    for kv in [1024, 2048, 4096, 8192]:
        for fc, (K,N,calls) in FC_DEFS.items():
            tag = "fc_decode_" + fc.replace("fc_","")
            ns = total_ns(tag)
            f,b = fc_flops_bytes(1, K, N, bits=4)
            add("decode", kv, fc, "fc_int4_g128", ns, calls, f, b, "memory", tag=tag)
        ns = total_ns("fc_decode_lm_head")
        f,b = fc_flops_bytes(1, HIDDEN, VOCAB, bits=8)
        add("decode", kv, "lm_head", "fc_int8_g128", ns, 1, f, b, "memory", tag="fc_decode_lm_head")

        tag = f"pa_decode_kv{kv}"
        ns = total_ns(tag)
        f,b = pa_flops_bytes(1, kv, kv_i8=True)
        add("decode", kv, "pa", "pa_opencl_micro", ns, LAYERS, f, b, "memory", tag=tag)

        # Small ops (M=1) — separate Q (NH=32) and K (NKV=8) for GQA
        small_decode_defs = [
            ("rmsnorm",     small_bytes("rmsnorm",1),                2*LAYERS+1),
            ("rmsnorm3d_q", small_bytes("rmsnorm3d",1, n_heads=NH),   LAYERS),
            ("rmsnorm3d_k", small_bytes("rmsnorm3d",1, n_heads=NKV),  LAYERS),
            ("rope_q",      small_bytes("rope",   1, n_heads=NH),     LAYERS),
            ("rope_k",      small_bytes("rope",   1, n_heads=NKV),    LAYERS),
            # NOTE: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU
            # primitive in the real graph (glu_fusion + swiglu_with_clamp);
            # do not count it as a standalone kernel.
            ("add",         small_bytes("add",   1),                  2*LAYERS),
        ]
        for op, b_per, calls in small_decode_defs:
            tag = "small_decode_" + op
            ns = total_ns(tag)
            add("decode", kv, op, op, ns, calls, 0.0, b_per, "memory", tag=tag)

    # -------- Prefill (S tokens) -----------------------------------------
    for S in [1024, 2048, 4096, 8192]:
        for fc, (K,N,calls) in FC_DEFS.items():
            tag = "fc_prefill_" + fc.replace("fc_","") + f"_S{S}"
            ns = total_ns(tag)
            f,b = fc_flops_bytes(S, K, N, bits=4)
            add("prefill", S, fc, "fc_int4_g128", ns, calls, f, b, "compute_int8", tag=tag)
        ns = total_ns("fc_prefill_lm_head")
        f,b = fc_flops_bytes(1, HIDDEN, VOCAB, bits=8)
        add("prefill", S, "lm_head", "fc_int8_g128", ns, 1, f, b, "memory", tag="fc_prefill_lm_head")

        tag = f"pa_prefill_S{S}"
        ns = total_ns(tag)
        f,b = pa_flops_bytes(S, S, kv_i8=True)
        kind = "compute_fp16" if S >= 2048 else "memory"
        add("prefill", S, "pa", "pa_opencl_micro_prefill", ns, LAYERS, f, b, kind, tag=tag)

        small_prefill_defs = [
            ("rmsnorm",     small_bytes("rmsnorm",   S),                2*LAYERS+1),
            ("rmsnorm3d_q", small_bytes("rmsnorm3d", S, n_heads=NH),    LAYERS),
            ("rmsnorm3d_k", small_bytes("rmsnorm3d", S, n_heads=NKV),   LAYERS),
            ("rope_q",      small_bytes("rope",      S, n_heads=NH),    LAYERS),
            ("rope_k",      small_bytes("rope",      S, n_heads=NKV),   LAYERS),
            # NOTE: SwiGLU `multiply` is fused into SwiGLU primitive — not standalone.
            ("add",         small_bytes("add",       S),                2*LAYERS),
        ]
        for op, b_per, calls in small_prefill_defs:
            tag = f"small_prefill_{op}_S{S}"
            ns = total_ns(tag)
            add("prefill", S, op, op, ns, calls, 0.0, b_per, "memory", tag=tag)

    # ---------- Save -----------------------------------------------------
    OUT.write_text(json.dumps({"hw":{"FP16_XMX_TFLOPS":FP16_XMX_TFLOPS,
                                     "INT8_XMX_TOPS":INT8_XMX_TOPS_PEAK,
                                     "BW_GBS":BW_GBS},
                               "rows":rows,
                               "sub_rows":sub_rows}, indent=2))
    print(f"Wrote {OUT}  ({len(rows)} rows)")

    # ---------- Markdown report -----------------------------------------
    today = datetime.date.today().isoformat()
    md = []
    md.append(f"# qwen3_omni Thinker text — Roofline on PTL 12Xe Windows ({today})\n")
    md.append(f"**Platform**: Intel PTL 12Xe iGPU (Arc B390 96CUs class, 96 EUs = 12 Xe × 8 EU × 10 thr), {HZ/1e6:.0f} MHz, BW≈{BW_GBS:.0f} GB/s, FP16 XMX peak ≈ {FP16_XMX_TFLOPS:.2f} TFLOPS, INT8 XMX peak ≈ {INT8_XMX_TOPS_PEAK:.2f} TOPS.")
    md.append("**Model**: qwen3_omni Thinker text decoder (dense, GQA). Source: `config.json` -> `thinker_config.text_config`.\n")
    md.append(f"- hidden={HIDDEN}, layers={LAYERS}, **GQA NH={NH} / NKV={NKV}**, head_dim={HD} (Q-dim={Q_DIM}, KV-dim={KV_DIM}), intermediate={INTER}, vocab={VOCAB}")
    md.append("- `tie_word_embeddings=true` (LM head shares storage with embedding), `hidden_act=silu` (SwiGLU), `rope_theta=1e6`, mrope_section=[24,20,20]")
    md.append("- MatMul weights INT4 g128 / FP16 act; LM_head INT8 g128 / FP16 act; KV cache INT8")
    md.append("- SDPA: PagedAttention OpenCL + micro_kernel\n")

    # ---- Detailed model parameters & per-weight shapes -----------------
    md.append("## Model parameters & weight shapes")
    md.append("Architecture knobs (parsed from `thinker_config.text_config`):\n")
    md.append("| Field | Value | Notes |")
    md.append("|---|---:|---|")
    md.append(f"| `hidden_size` | {HIDDEN} | residual / activation channel |")
    md.append(f"| `num_hidden_layers` | {LAYERS} | decoder blocks |")
    md.append(f"| `num_attention_heads` (NH) | {NH} | Q heads |")
    md.append(f"| `num_key_value_heads` (NKV) | {NKV} | GQA: {NH//NKV}-way Q-per-KV sharing |")
    md.append(f"| `head_dim` (HD) | {HD} | Q_dim = NH·HD = {Q_DIM}, KV_dim = NKV·HD = {KV_DIM} |")
    md.append(f"| `intermediate_size` | {INTER} | SwiGLU MLP hidden |")
    md.append(f"| `vocab_size` | {VOCAB} | LM head N |")
    md.append("| `hidden_act` | silu | SwiGLU = silu(gate(x)) ⊙ up(x) |")
    md.append("| `tie_word_embeddings` | true | LM head storage shared with token embedding |")
    md.append("| `rope_theta` | 1e6 | mrope_section=[24,20,20] |\n")

    QKV_N = Q_DIM + 2 * KV_DIM
    # INT4 g128 weight bytes: K*N*0.5 + scales (K/128)*N*2 (fp16); INT8 g128: K*N + (K/128)*N*2
    def w_bytes_int4(K, N):
        return K * N // 2 + (K // 128) * N * 2
    def w_bytes_int8(K, N):
        return K * N     + (K // 128) * N * 2

    weights = [
        ("Embedding (shared w/ LM head)", VOCAB, HIDDEN, "INT8 g128 + FP16 scales", w_bytes_int8(HIDDEN, VOCAB), 1),
        ("FC_QKV  (fused Q+K+V proj)", HIDDEN, QKV_N, "INT4 g128 + FP16 scales", w_bytes_int4(HIDDEN, QKV_N), LAYERS),
        ("FC_O    (attention output)", Q_DIM,  HIDDEN, "INT4 g128 + FP16 scales", w_bytes_int4(Q_DIM, HIDDEN), LAYERS),
        ("FC_Gate (SwiGLU gate)",      HIDDEN, INTER,  "INT4 g128 + FP16 scales", w_bytes_int4(HIDDEN, INTER),  LAYERS),
        ("FC_Up   (SwiGLU up)",        HIDDEN, INTER,  "INT4 g128 + FP16 scales", w_bytes_int4(HIDDEN, INTER),  LAYERS),
        ("FC_Down (SwiGLU down)",      INTER,  HIDDEN, "INT4 g128 + FP16 scales", w_bytes_int4(INTER, HIDDEN),  LAYERS),
        ("LM_Head (tied w/ embedding)", HIDDEN, VOCAB, "INT8 g128 + FP16 scales", w_bytes_int8(HIDDEN, VOCAB), 1),
    ]
    md.append("Per-layer weight matrices (one decoder block) and global weights. "
              "INT4 g128: weight = K·N/2 bytes + (K/128)·N FP16 scales. "
              "INT8 g128: weight = K·N bytes + (K/128)·N FP16 scales.\n")
    md.append("| Weight | Shape (K × N) | Quant | Bytes / instance | × Layers | Total MB |")
    md.append("|---|---:|---|---:|---:|---:|")
    grand_total = 0
    for name, K, N, q, b, L in weights:
        total = b * L
        grand_total += total
        md.append(f"| {name} | {K} × {N} | {q} | {b/1e6:.2f} MB | {L} | {total/1e6:.1f} MB |")
    md.append(f"| **Total static weights** |  |  |  |  | **{grand_total/1e6:.1f} MB** |\n")

    # KV-cache per-layer per-token bytes (INT8): K + V per head
    kv_per_token_layer = 2 * NKV * HD  # int8
    kv_per_token_total = kv_per_token_layer * LAYERS
    md.append("Activation / KV-cache shapes (S = sequence length, B = batch=1):\n")
    md.append("| Tensor | Shape | dtype | Bytes / token / layer | Bytes / token (all layers) |")
    md.append("|---|---|---|---:|---:|")
    md.append(f"| Hidden states | [B, S, {HIDDEN}] | FP16 | {HIDDEN*2} | — |")
    md.append(f"| Q | [B, S, {NH}, {HD}] | FP16 | {NH*HD*2} | — |")
    md.append(f"| K (cache) | [num_blocks, {NKV}, {HD}, 16] | INT8 | {NKV*HD} | {NKV*HD*LAYERS} |")
    md.append(f"| V (cache) | [num_blocks, {NKV}, 16, {HD}] | INT8 | {NKV*HD} | {NKV*HD*LAYERS} |")
    md.append(f"| **KV cache total** | per token | INT8 | {kv_per_token_layer} B / layer | **{kv_per_token_total/1024:.1f} KB / token** |\n")

    md.append("## Theoretical roofline")
    md.append("| Metric | Value |")
    md.append("|---|---|")
    md.append(f"| FP16 XMX peak | {FP16_XMX_TFLOPS:.2f} TFLOPS |")
    md.append(f"| INT8 XMX peak | {INT8_XMX_TOPS_PEAK:.2f} TOPS |")
    md.append(f"| Memory BW | {BW_GBS:.0f} GB/s |")
    md.append(f"| Ridge point (FP16) | {FP16_XMX_TFLOPS*1000/BW_GBS:.0f} FLOP/byte |\n")

    md.append("## Graph fusion notes (what the bench rows actually mean in the compiled model)")
    md.append("Some \"small ops\" in the per-op tables are profiled in isolation by `small_ops_bench` "
              "but **do not correspond to standalone GPU kernels** in the real compiled qwen3_omni "
              "graph because the GPU plugin fuses them at compile time:\n")
    md.append("| Bench row | Real graph behaviour | Fused into | Standalone kernel in graph? |")
    md.append("|---|---|---|---|")
    md.append("| `multiply` | `silu(gate(x)) ⊙ up(x)` of SwiGLU MLP | **SwiGLU primitive** (`glu_fusion.cpp` + `swiglu_with_clamp` op) | **No** — bench-only. Time is absorbed into the gated-MLP path. |")
    md.append("| `add` | Two residual adds per layer (post-attention, post-MLP) | not fused (separate `eltwise`) | Yes (2·LAYERS = 72 calls confirms this) |")
    md.append("| `rmsnorm` | Pre-attention + pre-MLP + final RMSNorm | already a single `RMS` primitive (`rms_fusion.cpp`) | Yes — 2·LAYERS+1 = 73 calls, but **cannot be merged across layers** (different tensors / timesteps) |")
    md.append("| `rmsnorm3d_q` / `rmsnorm3d_k` | Per-head q_norm / k_norm after QKV split | not fused with each other (different shapes — NH vs NKV — and different tensors) | Yes — 1 each per layer |")
    md.append("| `rope_q` / `rope_k` | RoPE on Q (NH=32) and K (NKV=8) | sin/cos/concat/mul/add already fused to single `RoPE` primitive (`fuse_rotary_positional_embeddings.cpp`) | Yes — **two distinct `rope_opt` calls per layer** because Q-RoPE and K-RoPE are two graph nodes with different head counts; not merged |\n")
    md.append("**Implication for the totals**: when computing TPOT / TTFT, the `multiply` row should be "
              "treated as 0 in the real graph (it is part of SwiGLU). All other small ops above remain "
              "as separate kernel launches. Fusion candidates that would help most: "
              "(a) merging q_norm + k_norm into a single fused-RMS-on-QKV kernel (~2 launches/layer saved), "
              "(b) batching rope_q + rope_k into one RoPE kernel over the full QKV head dim (would also "
              "halve launch overhead).\n")

    def write_table(phase, S):
        md.append(f"### {phase.capitalize()} — {'KV' if phase=='decode' else 'S'}={S}")
        md.append("| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
        rs = [r for r in rows if r["phase"]==phase and r["S"]==S]
        rs.sort(key=lambda r: -r["total_ms"])
        tot = sum(r["total_ms"] for r in rs)
        for r in rs:
            md.append(f"| {r['op']} | {r['kernel']} | {r['single_ms']:.4f} | {r['calls']} | "
                      f"{r['total_ms']:.3f} | {r['gflops']:.0f} | {r['gbs']:.1f} | "
                      f"{r['eff_pct']:.1f}% | {r['bound']} |")
        md.append(f"| **TOTAL** |  |  |  | **{tot:.3f}** |  |  |  |  |\n")
        md.append("_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU "
                  "primitive and does not appear here as a standalone kernel; "
                  "see *Graph fusion notes* above._\n")

    # ---------- Token latency summary (TTFT / TPOT) ----------------------
    md.append("## Token latency summary")
    md.append("Aggregated per-token latency derived from per-op profiling. "
              "**Prefill total = sum of per-layer ops × 32 layers + lm_head (1 token)**, "
              "**TTFT** is the same as prefill total (time to first token), "
              "**per-token-in-prefill** = TTFT / S (amortized prefill cost per input token). "
              "**Decode total = TPOT** (Time-Per-Output-Token, second token at given KV).\n")

    md.append("### Prefill — TTFT and per-token amortized")
    md.append("| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |")
    md.append("|---:|---:|---:|---:|---:|")
    for S in [1024, 2048, 4096, 8192]:
        rs = [r for r in rows if r["phase"]=="prefill" and r["S"]==S]
        ttft = sum(r["total_ms"] for r in rs)
        per_tok = ttft / S
        tps = 1000.0 / per_tok if per_tok > 0 else 0
        md.append(f"| {S} | {ttft:.2f} | {ttft/1000:.3f} | {per_tok:.4f} | {tps:.1f} |")
    md.append("")

    md.append("### Decode — TPOT (per output token)")
    md.append("| KV (ctx) | TPOT (ms) | tokens/s |")
    md.append("|---:|---:|---:|")
    for kv in [1024, 2048, 4096, 8192]:
        rs = [r for r in rows if r["phase"]=="decode" and r["S"]==kv]
        tpot = sum(r["total_ms"] for r in rs)
        tps = 1000.0 / tpot if tpot > 0 else 0
        md.append(f"| {kv} | {tpot:.3f} | {tps:.1f} |")
    md.append("")

    # ---------- TPOT breakdown by op category ---------------------------
    md.append("### Decode TPOT — per-op breakdown (ms / % of TPOT)")
    cat_map = {
        "fc_qkv":"FC qkv (fused Q+K+V)", "fc_o":"FC o",
        "fc_gate":"FC gate", "fc_up":"FC up", "fc_down":"FC down",
        "lm_head":"lm_head", "pa":"PagedAttention",
        "rmsnorm":"RMSNorm", "rmsnorm3d_q":"RMSNorm q", "rmsnorm3d_k":"RMSNorm k",
        "rope_q":"RoPE q", "rope_k":"RoPE k",
        "add":"Residual Add",
    }
    md.append("| op | " + " | ".join(f"KV={k} ms (%)" for k in [1024,2048,4096,8192]) + " |")
    md.append("|---|" + "|".join("---:" for _ in range(4)) + "|")
    op_order = ["fc_down","fc_up","fc_gate","fc_qkv","fc_o","lm_head","pa",
                "rmsnorm","rmsnorm3d_q","rmsnorm3d_k","rope_q","rope_k","add"]
    for op in op_order:
        cells = []
        for kv in [1024,2048,4096,8192]:
            rs = [r for r in rows if r["phase"]=="decode" and r["S"]==kv]
            tot = sum(r["total_ms"] for r in rs) or 1
            r = next((x for x in rs if x["op"]==op), None)
            if r:
                cells.append(f"{r['total_ms']:.3f} ({r['total_ms']/tot*100:.1f}%)")
            else:
                cells.append("—")
        md.append(f"| {cat_map.get(op,op)} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("### Prefill TTFT — per-op breakdown (ms / % of TTFT)")
    md.append("| op | " + " | ".join(f"S={s} ms (%)" for s in [1024,2048,4096,8192]) + " |")
    md.append("|---|" + "|".join("---:" for _ in range(4)) + "|")
    for op in op_order:
        cells = []
        for s in [1024,2048,4096,8192]:
            rs = [r for r in rows if r["phase"]=="prefill" and r["S"]==s]
            tot = sum(r["total_ms"] for r in rs) or 1
            r = next((x for x in rs if x["op"]==op), None)
            if r:
                cells.append(f"{r['total_ms']:.2f} ({r['total_ms']/tot*100:.1f}%)")
            else:
                cells.append("—")
        md.append(f"| {cat_map.get(op,op)} | " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Decode tables (1 query token, KV = context length)")
    for kv in [1024, 2048, 4096, 8192]:
        write_table("decode", kv)

    md.append("## Prefill tables (single forward over S tokens)")
    for S in [1024, 2048, 4096, 8192]:
        write_table("prefill", S)

    # ------------ Per-kernel decomposition (cliloader names) -----------
    md.append("## Per-kernel decomposition (cliloader kernel names)")
    md.append("Each op above maps to one or more GPU kernels. Below shows the actual "
              "kernel names captured by cliloader's *Device Performance Timing* section, "
              "with per-launch time, launches per op-call, total ms across one inference, "
              "and per-kernel **Eff%** (peak utilization). "
              "The Eff% column attributes the **parent op's full FLOPs/bytes per op-call** "
              "against the kernel's own per-launch time, so the dominant kernel reports an "
              "Eff% close to the op-level value, while helper kernels (e.g. "
              "`pa_kv_cache_update_ref`, `dynamic_quantize_gpu_opt`, `*_finalization`) "
              "appear with apparently very high Eff% — that is the expected signal that "
              "they are *not* the bottleneck for the op. "
              "PA decomposes into `pa_kv_cache_update_ref` + the attention kernel "
              "(`paged_attention_opt__single_token` / `__gqa_single_token` / "
              "`sdpa_micro__prefill`) + finalization. Prefill FC decomposes into "
              "`dynamic_quantize_gpu_opt` + `gemm_kernel`.\n")

    def write_subkernel_table(phase, S):
        md.append(f"### {phase.capitalize()} sub-kernels — {'KV' if phase=='decode' else 'S'}={S}")
        md.append("| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |")
        md.append("|---|---|---:|---:|---:|---:|---:|---:|")
        srs = [r for r in sub_rows if r["phase"]==phase and r["S"]==S]
        srs.sort(key=lambda r: -r["total_ms"])
        tot = sum(r["total_ms"] for r in srs) or 1.0
        for r in srs:
            kn = r["kernel_name"]
            kn_short = re.sub(r"__?\d{6,}.*$", "", kn) or kn
            eff_str = f"{r['eff_pct']:.1f}%" if r['eff_pct'] > 0 else "—"
            md.append(f"| {r['op']} | `{kn_short}` | {r['single_ms']:.4f} | "
                      f"{r['launches_per_call']} | {r['calls_per_inf']} | "
                      f"{r['total_ms']:.3f} | {r['total_ms']/tot*100:.1f}% | {eff_str} |")
        md.append("")

    md.append("### Decode sub-kernels")
    for kv in [1024, 2048, 4096, 8192]:
        write_subkernel_table("decode", kv)
    md.append("### Prefill sub-kernels")
    for S in [1024, 2048, 4096, 8192]:
        write_subkernel_table("prefill", S)

    # Bottleneck breakdown
    md.append("## Top contributors (sorted by total ms per inference)\n")
    for phase, sizes, label in [("decode",[1024,2048,4096,8192],"KV"),
                                ("prefill",[1024,2048,4096,8192],"S")]:
        md.append(f"### {phase}")
        md.append(f"| {label} | top1 (ms,%) | top2 | top3 |")
        md.append("|---:|---|---|---|")
        for s in sizes:
            rs = [r for r in rows if r["phase"]==phase and r["S"]==s]
            tot = sum(r["total_ms"] for r in rs) or 1
            rs.sort(key=lambda r:-r["total_ms"])
            t = lambda r: f"{r['op']} {r['total_ms']:.2f}ms ({r['total_ms']/tot*100:.0f}%)"
            md.append(f"| {s} | {t(rs[0])} | {t(rs[1])} | {t(rs[2])} |")
        md.append("")

    md.append("## Caveats & method")
    md.append("- Each op profiled in its own process via cliloader Device Performance Timing; "
              "we use mean kernel time (mode-of-call-counts) per iteration.")
    md.append("- FC weight bytes count INT4 weight + FP16 scale/zp(g=128) + FP16 act + FP16 out.")
    md.append("- PA bytes assume INT8 KV cache (1B/elem) + FP16 Q, FP16 out.")
    md.append("- Decode FC is treated as **memory-bound** (per SKILL: weights read dominates at M=1); "
              "prefill FC is **INT8 XMX compute-bound** (S big enough to hit XMX).")
    md.append("- Prefill PA at S≥2048 is compute-bound (FP16 micro-kernel); decode PA is memory-bound.")
    md.append("- swish/multiply/add eltwise are typically fused into matmul/SwiGLU in real inference; "
              "they are listed for visibility. swish standalone could not be measured (the GPU plugin "
              "fuses Parameter→Swish→Result into a noop activation that the parser excludes).")
    md.append("- lm_head is run only once per token (last position in prefill, every step in decode).")
    md.append("- Target machine: Local_Admin@10.239.132.229 (Windows PTL 12Xe / Arc B390).\n")

    md.append("## Reproduction")
    md.append("```cmd")
    md.append("REM On the 12Xe Windows host (Local_Admin@10.239.132.229):")
    md.append("D:\\river\\moe\\dev_roofline_profiling\\utils\\run_qwen3_omni_ptl_12xe.bat")
    md.append("REM Logs land in D:\\river\\moe\\roofline_results\\qwen3_omni\\ptl_12xe\\. Then locally:")
    md.append("python utils/parse_logs.py outputs/qwen3_omni/logs_ptl/ outputs/qwen3_omni/kernels_raw.json")
    md.append("python outputs/qwen3_omni/build_report.py")
    md.append("```\n")

    out_md = ROOT / f"SUMMARY_qwen3_omni_{today}.md"
    out_md.write_text("\n".join(md))
    print(f"Wrote {out_md}")

if __name__ == "__main__":
    main()
