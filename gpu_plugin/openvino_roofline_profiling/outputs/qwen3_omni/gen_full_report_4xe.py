#!/usr/bin/env python3
"""Generate full-model SUMMARY for PTL 4Xe from measured PA data + scaled 12Xe estimates."""

# ── Hardware specs ──────────────────────────────────────────────
XE_CORES_4XE = 4;  EU_PER_CORE = 8;  FREQ_MHZ = 2400
FP16_XMX_PEAK = XE_CORES_4XE * EU_PER_CORE * 256 * FREQ_MHZ / 1e3  # GFLOPS
INT8_XMX_PEAK = FP16_XMX_PEAK * 2  # GOPS
BW_4XE = 97.0    # GB/s measured
BW_12XE = 110.0  # GB/s measured on 12Xe
BW_RATIO = BW_12XE / BW_4XE   # 1.134
XMX_RATIO = 3.0               # 12/4 Xe cores
RIDGE = FP16_XMX_PEAK / BW_4XE

# ── Model ───────────────────────────────────────────────────────
LAYERS = 36;  NH = 32;  NKV = 8;  HD = 128
HIDDEN = 2560;  INTER = 9728;  VOCAB = 151936
KV_LENGTHS = [1024, 2048, 4096, 8192]
SEQ_LENGTHS = [1024, 2048, 4096, 8192]

# ── 12Xe reference data (from SUMMARY_qwen3_omni_2026-05-08.md) ─
# Decode per-call ms (constant across KV for non-PA ops)
ref_decode = {
    "fc_up":       {"ms": 0.1289, "gflops": 386, "gbs": 99.8,  "eff": 90.7, "bound": "memory", "kernel": "fc_int4_g128", "calls": 36},
    "fc_gate":     {"ms": 0.1284, "gflops": 388, "gbs": 100.2, "eff": 91.1, "bound": "memory", "kernel": "fc_int4_g128", "calls": 36},
    "fc_down":     {"ms": 0.1283, "gflops": 388, "gbs": 100.3, "eff": 91.2, "bound": "memory", "kernel": "fc_int4_g128", "calls": 36},
    "lm_head":     {"ms": 3.8339, "gflops": 203, "gbs": 103.1, "eff": 93.7, "bound": "memory", "kernel": "fc_int8_g128", "calls": 1},
    "fc_qkv":      {"ms": 0.0821, "gflops": 383, "gbs": 98.9,  "eff": 90.0, "bound": "memory", "kernel": "fc_int4_g128", "calls": 36},
    "fc_o":        {"ms": 0.0558, "gflops": 376, "gbs": 97.2,  "eff": 88.3, "bound": "memory", "kernel": "fc_int4_g128", "calls": 36},
    "rmsnorm":     {"ms": 0.0029, "gflops": 0,   "gbs": 5.3,   "eff": 4.8,  "bound": "memory", "kernel": "rmsnorm",      "calls": 73},
    "add":         {"ms": 0.0019, "gflops": 0,   "gbs": 7.9,   "eff": 7.2,  "bound": "memory", "kernel": "add",          "calls": 72},
    "rmsnorm3d_q": {"ms": 0.0024, "gflops": 0,   "gbs": 6.9,   "eff": 6.3,  "bound": "memory", "kernel": "rmsnorm3d_q",  "calls": 36},
    "rmsnorm3d_k": {"ms": 0.0024, "gflops": 0,   "gbs": 1.8,   "eff": 1.7,  "bound": "memory", "kernel": "rmsnorm3d_k",  "calls": 36},
    "rope_q":      {"ms": 0.0022, "gflops": 0,   "gbs": 7.8,   "eff": 7.1,  "bound": "memory", "kernel": "rope_q",       "calls": 36},
    "rope_k":      {"ms": 0.0020, "gflops": 0,   "gbs": 2.3,   "eff": 2.1,  "bound": "memory", "kernel": "rope_k",       "calls": 36},
}

# 4Xe MEASURED PA decode data
pa_decode_measured = {
    1024: {"ms": 0.1776, "gflops": 101, "gbs": 12.6, "eff": 13.0, "kernel": "pa_opencl_micro"},
    2048: {"ms": 0.2004, "gflops": 174, "gbs": 21.8, "eff": 22.5, "kernel": "pa_opencl_micro"},
    4096: {"ms": 0.1733, "gflops": 415, "gbs": 51.8, "eff": 53.4, "kernel": "pa_opencl_micro"},
    8192: {"ms": 0.3134, "gflops": 453, "gbs": 56.7, "eff": 58.4, "kernel": "pa_opencl_micro"},
}

# PA decode sub-kernels (measured)
pa_decode_sub = {
    1024: [("pa_kv_cache_update_ref", 0.0075), ("paged_attention_opt__single_token", 0.1667), ("paged_attention_opt__single_token_finalization", 0.0034)],
    2048: [("pa_kv_cache_update_ref", 0.0041), ("paged_attention_opt__single_token", 0.1925), ("paged_attention_opt__single_token_finalization", 0.0039)],
    4096: [("pa_kv_cache_update_ref", 0.0043), ("paged_attention_opt__gqa_single_token", 0.1619), ("paged_attention_opt__single_token_finalization", 0.0071)],
    8192: [("pa_kv_cache_update_ref", 0.0040), ("paged_attention_opt__gqa_single_token", 0.2961), ("paged_attention_opt__single_token_finalization", 0.0133)],
}

# 12Xe prefill per-call data (op level)
ref_prefill_op = {
    1024: {
        "fc_down":  {"ms": 0.9911, "gflops": 51459, "gbs": 38.3, "eff": 43.6},
        "fc_up":    {"ms": 0.8761, "gflops": 58214, "gbs": 43.4, "eff": 49.3},
        "fc_gate":  {"ms": 0.8732, "gflops": 58409, "gbs": 43.5, "eff": 49.5},
        "fc_qkv":   {"ms": 0.5758, "gflops": 55941, "gbs": 45.0, "eff": 47.4},
        "fc_o":     {"ms": 0.4194, "gflops": 51208, "gbs": 45.4, "eff": 43.4},
        "add":      {"ms": 0.1069}, "rmsnorm3d_q": {"ms": 0.1833}, "rmsnorm": {"ms": 0.0764},
        "rope_q":   {"ms": 0.1067}, "rmsnorm3d_k": {"ms": 0.0484}, "rope_k": {"ms": 0.0313},
        "lm_head":  {"ms": 3.8534},
    },
    2048: {
        "fc_down":  {"ms": 1.9427, "gflops": 52506, "gbs": 32.5, "eff": 44.5},
        "fc_up":    {"ms": 1.7450, "gflops": 58457, "gbs": 36.2, "eff": 49.6},
        "fc_gate":  {"ms": 1.7317, "gflops": 58905, "gbs": 36.5, "eff": 49.9},
        "fc_qkv":   {"ms": 1.0939, "gflops": 58892, "gbs": 40.0, "eff": 49.9},
        "fc_o":     {"ms": 0.7732, "gflops": 55547, "gbs": 42.3, "eff": 47.1},
        "add":      {"ms": 0.2472}, "rmsnorm3d_q": {"ms": 0.3760}, "rmsnorm": {"ms": 0.1577},
        "rope_q":   {"ms": 0.2611}, "rmsnorm3d_k": {"ms": 0.0938}, "rope_k": {"ms": 0.0586},
        "lm_head":  {"ms": 3.8534},
    },
    4096: {
        "fc_down":  {"ms": 3.7691, "gflops": 54128, "gbs": 30.1, "eff": 45.9},
        "fc_up":    {"ms": 3.6781, "gflops": 55467, "gbs": 30.9, "eff": 47.0},
        "fc_gate":  {"ms": 3.6157, "gflops": 56424, "gbs": 31.4, "eff": 47.8},
        "fc_qkv":   {"ms": 2.2721, "gflops": 56709, "gbs": 35.0, "eff": 48.1},
        "fc_o":     {"ms": 1.4932, "gflops": 57526, "gbs": 40.1, "eff": 48.8},
        "add":      {"ms": 0.5570}, "rmsnorm3d_q": {"ms": 0.7701}, "rmsnorm": {"ms": 0.3651},
        "rope_q":   {"ms": 0.6176}, "rmsnorm3d_k": {"ms": 0.1875}, "rope_k": {"ms": 0.1184},
        "lm_head":  {"ms": 3.8534},
    },
    8192: {
        "fc_down":  {"ms": 7.5537, "gflops": 54016, "gbs": 28.4, "eff": 45.8},
        "fc_up":    {"ms": 6.8841, "gflops": 59270, "gbs": 31.1, "eff": 50.2},
        "fc_gate":  {"ms": 6.8561, "gflops": 59512, "gbs": 31.2, "eff": 50.4},
        "fc_qkv":   {"ms": 4.5833, "gflops": 56225, "gbs": 32.9, "eff": 47.7},
        "fc_o":     {"ms": 2.9588, "gflops": 58064, "gbs": 38.7, "eff": 49.2},
        "add":      {"ms": 1.1972}, "rmsnorm3d_q": {"ms": 1.6433}, "rmsnorm": {"ms": 0.7911},
        "rope_q":   {"ms": 1.3314}, "rmsnorm3d_k": {"ms": 0.3842}, "rope_k": {"ms": 0.3021},
        "lm_head":  {"ms": 3.8534},
    },
}

# 12Xe prefill sub-kernels (gemm + dq split)
ref_prefill_sub = {
    1024: {"fc_down": (0.7679, 0.2232), "fc_up": (0.8170, 0.0591), "fc_gate": (0.8137, 0.0596),
           "fc_qkv": (0.5169, 0.0590), "fc_o": (0.3295, 0.0899)},
    2048: {"fc_down": (1.4908, 0.4519), "fc_up": (1.6241, 0.1209), "fc_gate": (1.6124, 0.1193),
           "fc_qkv": (0.9780, 0.1159), "fc_o": (0.5845, 0.1887)},
    4096: {"fc_down": (2.8277, 0.9413), "fc_up": (3.4309, 0.2472), "fc_gate": (3.3725, 0.2432),
           "fc_qkv": (2.0395, 0.2326), "fc_o": (1.1179, 0.3753)},
    8192: {"fc_down": (5.4703, 2.0834), "fc_up": (6.3923, 0.4918), "fc_gate": (6.3642, 0.4920),
           "fc_qkv": (4.1182, 0.4652), "fc_o": (2.2137, 0.7451)},
}

# 4Xe MEASURED PA prefill data
pa_prefill_measured = {
    1024: {"sdpa_ms": 8.814,   "kv_ms": 0.088, "gflops": 975.5,  "eff": 4.96},
    2048: {"sdpa_ms": 31.564,  "kv_ms": 0.161, "gflops": 1089.1, "eff": 5.54},
    4096: {"sdpa_ms": 122.821, "kv_ms": 0.299, "gflops": 1119.3, "eff": 5.69},
    8192: {"sdpa_ms": 486.689, "kv_ms": 0.552, "gflops": 1129.7, "eff": 5.75},
}

# Prefill PA FLOPs (causal mask): 4 * NH * (Sq*(Sq+1)/2) * HD
def pa_flops_prefill(sq):
    return 4.0 * NH * (sq * (sq + 1) / 2) * HD

# ── Scaling functions ──────────────────────────────────────────
def scale_mem(ms_12xe):
    return ms_12xe * BW_RATIO

def scale_fc_prefill(gemm_12xe, dq_12xe):
    return gemm_12xe * XMX_RATIO + dq_12xe * BW_RATIO

# ── Compute all 4Xe values ─────────────────────────────────────
def compute_decode(kv):
    """Return list of (op, kernel, single_ms, calls, total_ms, gflops, gbs, eff, bound) sorted by total desc"""
    rows = []
    for op, d in ref_decode.items():
        ms4 = scale_mem(d["ms"])
        gflops4 = d["gflops"] / BW_RATIO if d["gflops"] > 0 else 0
        gbs4 = d["gbs"] * BW_4XE / BW_12XE
        rows.append((op, d["kernel"], ms4, d["calls"], ms4*d["calls"], gflops4, gbs4, d["eff"], d["bound"]))
    # PA measured
    p = pa_decode_measured[kv]
    rows.append(("pa", p["kernel"], p["ms"], 36, p["ms"]*36, p["gflops"], p["gbs"], p["eff"], "memory"))
    rows.sort(key=lambda r: -r[4])
    return rows

def compute_prefill(sq):
    rows = []
    fc_ops = ["fc_down", "fc_up", "fc_gate", "fc_qkv", "fc_o"]
    small_ops = ["add", "rmsnorm3d_q", "rmsnorm", "rope_q", "rmsnorm3d_k", "rope_k"]
    calls_map = {"add": 72, "rmsnorm": 73, "rmsnorm3d_q": 36, "rmsnorm3d_k": 36, "rope_q": 36, "rope_k": 36}
    kernel_map = {"add": "add", "rmsnorm": "rmsnorm", "rmsnorm3d_q": "rmsnorm3d_q",
                  "rmsnorm3d_k": "rmsnorm3d_k", "rope_q": "rope_q", "rope_k": "rope_k"}
    ref = ref_prefill_op[sq]
    sub = ref_prefill_sub[sq]
    for op in fc_ops:
        gemm, dq = sub[op]
        ms4 = scale_fc_prefill(gemm, dq)
        ms12 = ref[op]["ms"]
        flops_per_call = ref[op]["gflops"] * ms12 / 1e3  # GFLOPs per call
        gflops4 = flops_per_call / (ms4 / 1e3)
        gbs4 = ref[op]["gbs"] * (ms12 / ms4)
        eff4 = gflops4 / INT8_XMX_PEAK * 100
        rows.append((op, "fc_int4_g128", ms4, 36, ms4*36, gflops4, gbs4, eff4, "compute"))
    # PA measured
    p = pa_prefill_measured[sq]
    pa_total = p["sdpa_ms"] + p["kv_ms"]
    pa_flops = pa_flops_prefill(sq)
    pa_data_mb = (sq * (NH + 2*NKV) * HD * 2) / 1e6  # Q+K+V fp16 in MB
    pa_gbs = pa_data_mb / (pa_total / 1e3) / 1e3  # convert to GB/s
    rows.append(("pa", "pa_ocl_micro_prefill", pa_total, 36, pa_total*36, p["gflops"], pa_gbs, p["eff"], "compute"))
    # Small ops (memory-bound scale)
    for op in small_ops:
        ms12 = ref[op]["ms"]
        ms4 = scale_mem(ms12)
        calls = calls_map[op]
        r12 = ref_decode.get(op, {})
        gbs4 = r12.get("gbs", 0) * BW_4XE / BW_12XE if r12 else 0
        eff4 = r12.get("eff", 0) if r12 else 0
        # For prefill small ops, eff from 12Xe prefill data (can exceed 100% due to cache)
        gbs_pf = ref_prefill_op[sq].get(op, {}).get("gbs", 0) if "gbs" in ref_prefill_op[sq].get(op, {}) else 0
        eff_pf = ref_prefill_op[sq].get(op, {}).get("eff", 0) if "eff" in ref_prefill_op[sq].get(op, {}) else 0
        # Use 12Xe prefill eff% directly (same arch, same eff%)
        # Need to compute gbs from 12Xe prefill
        if gbs_pf:
            gbs4_pf = gbs_pf * BW_4XE / BW_12XE
            rows.append((op, kernel_map[op], ms4, calls, ms4*calls, 0, gbs4_pf, eff_pf, "memory"))
        else:
            rows.append((op, kernel_map[op], ms4, calls, ms4*calls, 0, gbs4, eff4, "memory"))
    # lm_head (memory-bound, same as decode)
    lm_ms4 = scale_mem(ref["lm_head"]["ms"])
    lm_gflops = ref_decode["lm_head"]["gflops"] / BW_RATIO
    lm_gbs = ref_decode["lm_head"]["gbs"] * BW_4XE / BW_12XE
    rows.append(("lm_head", "fc_int8_g128", lm_ms4, 1, lm_ms4, lm_gflops, lm_gbs, ref_decode["lm_head"]["eff"], "memory"))
    rows.sort(key=lambda r: -r[4])
    return rows

# ── Generate report ────────────────────────────────────────────
lines = []
def w(s=""): lines.append(s)

w("# Qwen3-Omni Thinker text — Roofline on PTL 4Xe Linux (2026-05-09)")
w()
w(f"**Platform**: Intel PTL 4Xe iGPU (32 EUs = 4 Xe × 8 EU × 10 thr), {FREQ_MHZ} MHz, BW ≈ {BW_4XE} GB/s (measured), FP16 XMX peak ≈ {FP16_XMX_PEAK/1e3:.2f} TFLOPS, INT8 XMX peak ≈ {INT8_XMX_PEAK/1e3:.2f} TOPS.")
w("**Model**: Qwen3-Omni Thinker text decoder (dense, GQA). Source: `config.json` → `thinker_config.text_config`.")
w()
w("- hidden=2560, layers=36, **GQA NH=32 / NKV=8**, head_dim=128 (Q-dim=4096, KV-dim=1024), intermediate=9728, vocab=151936")
w("- `tie_word_embeddings=true` (LM head shares storage with embedding), `hidden_act=silu` (SwiGLU), `rope_theta=1e6`, mrope_section=[24,20,20]")
w("- MatMul weights INT4 g128 / FP16 act; LM_head INT8 g128 / FP16 act; KV cache INT8")
w("- SDPA: PagedAttention OpenCL + micro_kernel")
w()

w("## Data sources")
w()
w("| Data category | Source | Method |")
w("|---|---|---|")
w("| PA decode (all KV) | **Measured** on PTL 4Xe | `pa_bench` + cliloader |")
w("| PA prefill (all S) | **Measured** on PTL 4Xe | `pa_bench` + cliloader |")
w("| FC decode (all ops) | Estimated from PTL 12Xe | Memory-bound: latency × BW_ratio (110/97 = 1.134) |")
w("| FC prefill (all ops) | Estimated from PTL 12Xe | Compute-bound: gemm × 3.0 (XMX ratio) + dq × 1.134 (BW ratio) |")
w("| Small ops (RMSNorm, RoPE, Add) | Estimated from PTL 12Xe | Memory-bound: latency × 1.134 |")
w("| LM_head | Estimated from PTL 12Xe | Memory-bound: latency × 1.134 |")
w()
w("> **Caveat**: Only PA kernels were measured on this 4Xe machine. All other ops are estimates")
w("> scaled from PTL 12Xe Windows measurements (SUMMARY_qwen3_omni_2026-05-08.md).")
w("> BW efficiency % is preserved for memory-bound ops; INT8 XMX efficiency shifts slightly")
w("> because dynamic-quantize overhead fraction differs.")
w()

w("## Model parameters & weight shapes")
w()
w("Architecture knobs (parsed from `thinker_config.text_config`):")
w()
w("| Field | Value | Notes |")
w("|---|---:|---|")
w("| `hidden_size` | 2560 | residual / activation channel |")
w("| `num_hidden_layers` | 36 | decoder blocks |")
w("| `num_attention_heads` (NH) | 32 | Q heads |")
w("| `num_key_value_heads` (NKV) | 8 | GQA: 4-way Q-per-KV sharing |")
w("| `head_dim` (HD) | 128 | Q_dim = NH·HD = 4096, KV_dim = NKV·HD = 1024 |")
w("| `intermediate_size` | 9728 | SwiGLU MLP hidden |")
w("| `vocab_size` | 151936 | LM head N |")
w("| `hidden_act` | silu | SwiGLU = silu(gate(x)) ⊙ up(x) |")
w("| `tie_word_embeddings` | true | LM head storage shared with token embedding |")
w("| `rope_theta` | 1e6 | mrope_section=[24,20,20] |")
w()

w("Per-layer weight matrices (one decoder block) and global weights. INT4 g128: weight = K·N/2 bytes + (K/128)·N FP16 scales. INT8 g128: weight = K·N bytes + (K/128)·N FP16 scales.")
w()
w("| Weight | Shape (K × N) | Quant | Bytes / instance | × Layers | Total MB |")
w("|---|---:|---|---:|---:|---:|")
w("| Embedding (shared w/ LM head) | 151936 × 2560 | INT8 g128 + FP16 scales | 395.03 MB | 1 | 395.0 MB |")
w("| FC_QKV (fused Q+K+V proj) | 2560 × 6144 | INT4 g128 + FP16 scales | 8.11 MB | 36 | 292.0 MB |")
w("| FC_O (attention output) | 4096 × 2560 | INT4 g128 + FP16 scales | 5.41 MB | 36 | 194.6 MB |")
w("| FC_Gate (SwiGLU gate) | 2560 × 9728 | INT4 g128 + FP16 scales | 12.84 MB | 36 | 462.3 MB |")
w("| FC_Up (SwiGLU up) | 2560 × 9728 | INT4 g128 + FP16 scales | 12.84 MB | 36 | 462.3 MB |")
w("| FC_Down (SwiGLU down) | 9728 × 2560 | INT4 g128 + FP16 scales | 12.84 MB | 36 | 462.3 MB |")
w("| LM_Head (tied w/ embedding) | 2560 × 151936 | INT8 g128 + FP16 scales | 395.03 MB | 1 | 395.0 MB |")
w("| **Total static weights** |  |  |  |  | **2663.5 MB** |")
w()

w("Activation / KV-cache shapes (S = sequence length, B = batch=1):")
w()
w("| Tensor | Shape | dtype | Bytes / token / layer | Bytes / token (all layers) |")
w("|---|---|---|---:|---:|")
w("| Hidden states | [B, S, 2560] | FP16 | 5120 | — |")
w("| Q | [B, S, 32, 128] | FP16 | 8192 | — |")
w("| K (cache) | [num_blocks, 8, 128, 16] | INT8 | 1024 | 36864 |")
w("| V (cache) | [num_blocks, 8, 16, 128] | INT8 | 1024 | 36864 |")
w("| **KV cache total** | per token | INT8 | 2048 B / layer | **72.0 KB / token** |")
w()

w("## Theoretical roofline")
w()
w("| Metric | Value |")
w("|---|---|")
w(f"| FP16 XMX peak | {FP16_XMX_PEAK/1e3:.2f} TFLOPS |")
w(f"| INT8 XMX peak | {INT8_XMX_PEAK/1e3:.2f} TOPS |")
w(f"| Memory BW | {BW_4XE} GB/s (measured) |")
w(f"| Ridge point (FP16) | {RIDGE:.1f} FLOP/byte |")
w()

w("## Graph fusion notes")
w()
w("| Bench row | Real graph behaviour | Fused into | Standalone kernel in graph? |")
w("|---|---|---|---|")
w("| `multiply` | `silu(gate(x)) ⊙ up(x)` of SwiGLU MLP | SwiGLU primitive | No — bench-only |")
w("| `add` | Residual adds per layer | not fused (separate `eltwise`) | Yes |")
w("| `rmsnorm` | Pre-attention + pre-MLP + final RMSNorm | single `RMS` primitive | Yes |")
w("| `sdpa_micro__prefill` | PA prefill attention (EU ALU, causal mask) | PA primitive | Yes |")
w("| `pa_kv_cache_update_ref` | KV cache write | PA primitive | Yes |")
w()
w("> **Note on PA prefill**: The `sdpa_micro__prefill` kernel runs on **EU ALU only** (not XMX/DPAS).")
w("> Disassembly confirms only 2 dummy `dpas.8x1` instructions (systolic probes). Root cause: ngen")
w("> GEMM microkernel catalog has no entries for `HW::Xe3` — PTL 4Xe (ip_version arch=30) maps to")
w("> `GenericXe3` → `Core::Xe3`, which has no matching `HWTag` in the strategy catalog.")
w()

# ── Token latency summary ─────────────────────────────────────
w("## Token latency summary")
w()
w("### Prefill — TTFT and per-token amortized")
w()
w("| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |")
w("|---:|---:|---:|---:|---:|")
for sq in SEQ_LENGTHS:
    rows = compute_prefill(sq)
    ttft = sum(r[4] for r in rows)
    w(f"| {sq} | {ttft:.2f} | {ttft/1e3:.3f} | {ttft/sq:.4f} | {sq/(ttft/1e3):.1f} |")
w()

w("### Decode — TPOT (per output token)")
w()
w("| KV (ctx) | TPOT (ms) | tokens/s |")
w("|---:|---:|---:|")
for kv in KV_LENGTHS:
    rows = compute_decode(kv)
    tpot = sum(r[4] for r in rows)
    w(f"| {kv} | {tpot:.3f} | {1e3/tpot:.1f} |")
w()

# Decode breakdown
w("### Decode TPOT — per-op breakdown (ms / % of TPOT)")
w()
hdr = " | ".join(f"KV={kv} ms (%)" for kv in KV_LENGTHS)
sep = " | ".join(["---:"] * len(KV_LENGTHS))
w(f"| op | {hdr} |")
w(f"|---|{sep}|")
# Gather all ops
all_decode = {}
for kv in KV_LENGTHS:
    rows = compute_decode(kv)
    tpot = sum(r[4] for r in rows)
    for r in rows:
        op = r[0]
        if op not in all_decode: all_decode[op] = {}
        all_decode[op][kv] = (r[4], r[4]/tpot*100)
# Sort by KV=1024 total desc
op_order = sorted(all_decode.keys(), key=lambda o: -all_decode[o].get(1024, (0,))[0])
for op in op_order:
    cols = []
    for kv in KV_LENGTHS:
        ms, pct = all_decode[op].get(kv, (0, 0))
        cols.append(f"{ms:.3f} ({pct:.1f}%)")
    w(f"| {op} | {' | '.join(cols)} |")
w()

# Prefill breakdown
w("### Prefill TTFT — per-op breakdown (ms / % of TTFT)")
w()
hdr = " | ".join(f"S={sq} ms (%)" for sq in SEQ_LENGTHS)
sep = " | ".join(["---:"] * len(SEQ_LENGTHS))
w(f"| op | {hdr} |")
w(f"|---|{sep}|")
all_prefill = {}
for sq in SEQ_LENGTHS:
    rows = compute_prefill(sq)
    ttft = sum(r[4] for r in rows)
    for r in rows:
        op = r[0]
        if op not in all_prefill: all_prefill[op] = {}
        all_prefill[op][sq] = (r[4], r[4]/ttft*100)
op_order_p = sorted(all_prefill.keys(), key=lambda o: -all_prefill[o].get(8192, (0,))[0])
for op in op_order_p:
    cols = []
    for sq in SEQ_LENGTHS:
        ms, pct = all_prefill[op].get(sq, (0, 0))
        cols.append(f"{ms:.2f} ({pct:.1f}%)")
    w(f"| {op} | {' | '.join(cols)} |")
w()

# ── Decode tables ─────────────────────────────────────────────
w("## Decode tables (1 query token, KV = context length)")
w()
for kv in KV_LENGTHS:
    w(f"### Decode — KV={kv}")
    w()
    w("| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---|")
    rows = compute_decode(kv)
    for r in rows:
        op,k,sms,c,tms,gf,gb,ef,bd = r
        gf_s = f"{gf:.0f}" if gf > 0 else "0"
        gb_s = f"{gb:.1f}"
        w(f"| {op} | {k} | {sms:.4f} | {c} | {tms:.3f} | {gf_s} | {gb_s} | {ef:.1f}% | {bd} |")
    total = sum(r[4] for r in rows)
    w(f"| **TOTAL** |  |  |  | **{total:.3f}** |  |  |  |  |")
    w()
    w("_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._")
    w()

# ── Prefill tables ────────────────────────────────────────────
w("## Prefill tables (single forward over S tokens)")
w()
for sq in SEQ_LENGTHS:
    w(f"### Prefill — S={sq}")
    w()
    w("| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---|")
    rows = compute_prefill(sq)
    for r in rows:
        op,k,sms,c,tms,gf,gb,ef,bd = r
        gf_s = f"{gf:.0f}" if gf > 0 else "0"
        gb_s = f"{gb:.1f}"
        w(f"| {op} | {k} | {sms:.4f} | {c} | {tms:.3f} | {gf_s} | {gb_s} | {ef:.1f}% | {bd} |")
    total = sum(r[4] for r in rows)
    w(f"| **TOTAL** |  |  |  | **{total:.3f}** |  |  |  |  |")
    w()
    w("_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._")
    w()

# ── Per-kernel decomposition ──────────────────────────────────
w("## Per-kernel decomposition (cliloader kernel names)")
w()
w("Each op above maps to one or more GPU kernels. PA decomposes into `pa_kv_cache_update_ref` + attention kernel + finalization. Prefill FC decomposes into `dynamic_quantize_gpu_opt` + `gemm_kernel`. Decode FC is a single `gemm_kernel` call.")
w()

# Decode sub-kernels
w("### Decode sub-kernels")
for kv in KV_LENGTHS:
    w(f"### Decode sub-kernels — KV={kv}")
    w()
    w("| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |")
    w("|---|---|---:|---:|---:|---:|---:|---:|")
    subrows = []
    # FC ops (single gemm_kernel each)
    for op, d in ref_decode.items():
        ms4 = scale_mem(d["ms"])
        subrows.append((op, "gemm_kernel" if "fc" in op or op == "lm_head" else
                        ("rms_gpu_bfyx_opt" if "rmsnorm" in op else
                         ("eltwise_simple_vload8" if op == "add" else "rope_opt")),
                        ms4, 1, d["calls"], ms4*d["calls"], d["eff"]))
    # PA sub-kernels
    for kname, kms in pa_decode_sub[kv]:
        eff_s = pa_decode_measured[kv]["eff"] if "single_token" in kname and "finalization" not in kname else 0
        subrows.append(("pa", kname, kms, 1, 36, kms*36, eff_s))
    subrows.sort(key=lambda r: -r[5])
    total_sub = sum(r[5] for r in subrows)
    for r in subrows:
        op,kn,sms,lpc,ci,tms,ef = r
        pct = tms/total_sub*100
        ef_s = f"{ef:.1f}%" if ef > 0 else "—"
        w(f"| {op} | `{kn}` | {sms:.4f} | {lpc} | {ci} | {tms:.3f} | {pct:.1f}% | {ef_s} |")
    w()

# Prefill sub-kernels
w("### Prefill sub-kernels")
for sq in SEQ_LENGTHS:
    w(f"### Prefill sub-kernels — S={sq}")
    w()
    w("| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |")
    w("|---|---|---:|---:|---:|---:|---:|---:|")
    subrows = []
    sub = ref_prefill_sub[sq]
    ref = ref_prefill_op[sq]
    # FC gemm + dq
    for op in ["fc_down", "fc_up", "fc_gate", "fc_qkv", "fc_o"]:
        gemm12, dq12 = sub[op]
        gemm4 = gemm12 * XMX_RATIO
        dq4 = dq12 * BW_RATIO
        ms12_op = ref[op]["ms"]
        flops_per_call = ref[op]["gflops"] * ms12_op / 1e3
        gemm_gflops = flops_per_call / (gemm4 / 1e3)
        gemm_eff = gemm_gflops / INT8_XMX_PEAK * 100
        subrows.append((op, "gemm_kernel", gemm4, 1, 36, gemm4*36, gemm_eff))
        subrows.append((op, "dynamic_quantize_gpu_opt", dq4, 1, 36, dq4*36, 0))
    # PA
    p = pa_prefill_measured[sq]
    subrows.append(("pa", "sdpa_micro__prefill", p["sdpa_ms"], 1, 36, p["sdpa_ms"]*36, p["eff"]))
    subrows.append(("pa", "pa_kv_cache_update_ref", p["kv_ms"], 1, 36, p["kv_ms"]*36, 0))
    # Small ops
    small_data = {"add": ("eltwise_simple_vload8", 72), "rmsnorm3d_q": ("rms_gpu_bfyx_opt", 36),
                  "rmsnorm": ("rms_gpu_bfyx_opt", 73), "rope_q": ("rope_opt", 36),
                  "rmsnorm3d_k": ("rms_gpu_bfyx_opt", 36), "rope_k": ("rope_opt", 36)}
    for op, (kn, calls) in small_data.items():
        ms12 = ref[op]["ms"]
        ms4 = scale_mem(ms12)
        eff = ref_decode.get(op, {}).get("eff", 0)
        subrows.append((op, kn, ms4, 1, calls, ms4*calls, eff))
    # lm_head
    lm_ms4 = scale_mem(ref["lm_head"]["ms"])
    subrows.append(("lm_head", "gemm_kernel", lm_ms4, 1, 1, lm_ms4, ref_decode["lm_head"]["eff"]))
    subrows.sort(key=lambda r: -r[5])
    total_sub = sum(r[5] for r in subrows)
    for r in subrows:
        op,kn,sms,lpc,ci,tms,ef = r
        pct = tms/total_sub*100
        ef_s = f"{ef:.1f}%" if ef > 0 else "—"
        w(f"| {op} | `{kn}` | {sms:.4f} | {lpc} | {ci} | {tms:.3f} | {pct:.1f}% | {ef_s} |")
    w()

# ── Top contributors ─────────────────────────────────────────
w("## Top contributors (sorted by total ms per inference)")
w()
w("### Decode")
w()
w("| KV | top1 (ms,%) | top2 | top3 |")
w("|---:|---|---|---|")
for kv in KV_LENGTHS:
    rows = compute_decode(kv)
    total = sum(r[4] for r in rows)
    top3 = rows[:3]
    cols = []
    for r in top3:
        cols.append(f"{r[0]} {r[4]:.2f}ms ({r[4]/total*100:.0f}%)")
    w(f"| {kv} | {' | '.join(cols)} |")
w()
w("### Prefill")
w()
w("| S | top1 (ms,%) | top2 | top3 |")
w("|---:|---|---|---|")
for sq in SEQ_LENGTHS:
    rows = compute_prefill(sq)
    total = sum(r[4] for r in rows)
    top3 = rows[:3]
    cols = []
    for r in top3:
        cols.append(f"{r[0]} {r[4]:.2f}ms ({r[4]/total*100:.0f}%)")
    w(f"| {sq} | {' | '.join(cols)} |")
w()

# ── Comparison ────────────────────────────────────────────────
w("## Comparison: PTL 4Xe Linux vs PTL 12Xe Windows")
w()
w("| Metric | PTL 4Xe (this report) | PTL 12Xe (2026-05-08) | Ratio |")
w("|---|---:|---:|---:|")
w(f"| Xe cores | 4 | 12 | 3.0× |")
w(f"| FP16 XMX peak | {FP16_XMX_PEAK/1e3:.2f} TFLOPS | 58.98 TFLOPS | 3.0× |")
w(f"| BW (measured) | {BW_4XE} GB/s | {BW_12XE} GB/s | 1.13× |")
# Compute TTFTs
ttft_4xe = {sq: sum(r[4] for r in compute_prefill(sq)) for sq in SEQ_LENGTHS}
ttft_12xe = {1024: 189.78, 2048: 410.01, 4096: 1001.31, 8192: 2588.45}
tpot_4xe = {kv: sum(r[4] for r in compute_decode(kv)) for kv in KV_LENGTHS}
tpot_12xe = {1024: 25.543, 2048: 26.775, 4096: 27.585, 8192: 31.261}
for sq in SEQ_LENGTHS:
    w(f"| TTFT S={sq} | {ttft_4xe[sq]:.1f} ms | {ttft_12xe[sq]:.1f} ms | {ttft_4xe[sq]/ttft_12xe[sq]:.1f}× |")
for kv in KV_LENGTHS:
    w(f"| TPOT KV={kv} | {tpot_4xe[kv]:.1f} ms | {tpot_12xe[kv]:.1f} ms | {tpot_4xe[kv]/tpot_12xe[kv]:.1f}× |")
w()
w("Key differences:")
w("- **PA prefill**: OCL micro-kernel on 4Xe uses EU ALU only (~975–1130 GFLOPS, 5% XMX eff)")
w("  while 12Xe OCL micro-kernel uses XMX/DPAS (~26000–32000 GFLOPS, 44–55% eff)")
w("- **Root cause**: ngen GEMM catalog has no `HWTagXe3` entries; PTL 4Xe (ip_version arch=30)")
w("  maps to `GenericXe3` → `Core::Xe3` with no matching systolic strategy → falls back to EU ALU")
w("- **12Xe Windows**: ip_version likely maps to Xe2 (arch=20) → catalog match → DPAS used")
w("- **Decode**: dominated by memory-bound FC ops; 4Xe is ~1.3× slower (BW ratio)")
w("- **Prefill**: PA dominates increasingly at large S; 4Xe is 3.8–8.0× slower overall due to EU-only PA")
w()

# ── Analysis ──────────────────────────────────────────────────
w("## Analysis & insights")
w()
w("### Decode (memory-bound)")
w()
w("All decode ops are deeply memory-bound. FC kernels achieve 88–94% BW efficiency (estimated),")
w("consistent with 12Xe behavior. PA decode uses two kernel variants:")
w("- **`single_token`** (Skv ≤ 2048): 13–22% BW efficiency")
w("- **`gqa_single_token`** (Skv ≥ 4096): 53–58% BW efficiency (GQA-optimized tiling)")
w()
w("TPOT ranges from 32.7–37.8 ms, dominated by FC weight reads. PA becomes the largest")
w(f"single op at KV=8192 ({tpot_4xe[8192]-26.49:.1f}/{tpot_4xe[8192]:.1f} = {(tpot_4xe[8192]-26.49)/tpot_4xe[8192]*100:.0f}% of TPOT).")
w()
w("### Prefill (compute-bound)")
w()
w("Prefill FC ops are INT8 XMX compute-bound (~50% INT8 XMX efficiency estimated).")
w("PA prefill uses `sdpa_micro__prefill` (EU ALU, no DPAS) achieving only 975–1130 GFLOPS")
w("(~5% of FP16 XMX peak). PA dominates increasingly:")
for sq in SEQ_LENGTHS:
    pa_total = pa_prefill_measured[sq]["sdpa_ms"] + pa_prefill_measured[sq]["kv_ms"]
    pa_inf = pa_total * 36
    ttft = ttft_4xe[sq]
    w(f"- S={sq}: PA = {pa_inf:.0f} ms / {ttft:.0f} ms = **{pa_inf/ttft*100:.0f}%** of TTFT")
w()
w("### Optimization levers")
w()
w("1. **Enable XMX/DPAS for Xe3 PA**: Add `HWTagXe3` entries to ngen GEMM microkernel catalog")
w("   (or allow Xe3 to fall through to Xe2 entries) → expected ~6× prefill PA speedup")
w("2. **CM PA on Linux**: Install CM frontend (`libclangFEWrapper.so`) for CM-based PA with native XMX")
w("3. **GQA kernel for small Skv**: `single_token` at Skv ≤ 2048 only achieves 13–22% BW —")
w("   GQA-optimized variant should apply at all KV lengths")
w("4. **Small ops fusion**: merge q_norm+k_norm, batch rope_q+rope_k to halve launch overhead")
w()

# ── Caveats ───────────────────────────────────────────────────
w("## Caveats & method")
w()
w("- **PA data measured** on PTL 4Xe via `pa_bench` + cliloader. All other ops **estimated** by scaling from PTL 12Xe.")
w("- Scaling: memory-bound ops × 1.134 (BW ratio 110/97); compute-bound FC gemm × 3.0 (XMX core ratio) + dq × 1.134.")
w("- FC weight bytes count INT4 weight + FP16 scale/zp(g=128) + FP16 act + FP16 out.")
w("- PA bytes assume INT8 KV cache (1B/elem) + FP16 Q, FP16 out. Prefill K/V read as FP16 (raw inputs).")
w("- Prefill FLOPs use causal mask formula: 4 × NH × (Sq×(Sq+1)/2) × HD.")
w("- Decode FC treated as memory-bound (weights dominate at M=1); prefill FC is INT8 XMX compute-bound.")
w("- PA prefill XMX eff% is vs FP16 XMX peak (kernel uses EU ALU, not DPAS).")
w("- swish/multiply fused into SwiGLU; not listed as standalone kernel.")
w("- lm_head run once per inference (last position in prefill, every step in decode).")
w("- Target machine: intel@10.239.152.140 (Linux, PTL 4Xe, driver 26.14.037858).")
w()

# ── Reproduction ──────────────────────────────────────────────
w("## Reproduction")
w()
w("```bash")
w("# On intel@10.239.152.140")
w("export LD_LIBRARY_PATH=~/river/openvino/install_release/runtime/lib/intel64:~/river/openvino/temp/Linux_x86_64/tbb/lib")
w("CLILOADER=~/river/clintercept-3.0.6-Linux/bin/cliloader")
w("BIN=~/river/roofline_test_utils/build/pa_bench")
w()
w("# Decode (measured):")
w("$CLILOADER -d $BIN -- --mode decode --n_head 32 --n_kv_head 8 --head_dim 128 --n_layers 1 --S_kv 8192 --S_q 1 --block_size 16 --pages_per_block 1 --kv_cache_dtype i8")
w()
w("# Prefill (measured):")
w("$CLILOADER -d $BIN -- --mode prefill --n_head 32 --n_kv_head 8 --head_dim 128 --n_layers 1 --S_kv 0 --S_q 4096 --block_size 16 --pages_per_block 1 --kv_cache_dtype i8")
w()
w("# Note: FC, small ops, lm_head benchmarks not yet run on 4Xe.")
w("# Those values are estimated from PTL 12Xe (SUMMARY_qwen3_omni_2026-05-08.md).")
w("```")

# ── Write output ──────────────────────────────────────────────
output = "\n".join(lines) + "\n"
out_path = "/home/ov2022/workspace/remote_debug/openvino/.github/skills/dev_roofline_profiling/outputs/qwen3_omni/SUMMARY_qwen3_omni_PA_OCL_PTL4Xe_2026-05-09.md"
with open(out_path, "w") as f:
    f.write(output)
print(f"Written {len(lines)} lines to {out_path}")
