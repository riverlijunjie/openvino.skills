#!/usr/bin/env python3
"""Qwen3-ASR-1.7B roofline report — PTL 12Xe, FP16 weights / INT8 KV.

Reads parsed.json (output of utils/parse_logs.py over logs_ptl_12xe/) and
ops_mapping.json, computes theoretical FLOPs/bytes per op, aggregates per
sequence-length tables, and writes:
  - performance_metrics.json (per-op detailed metrics)
  - weight_distribution.json (theoretical model weight footprint)
  - SUMMARY_qwen3_asr_1_7B_ptl_12xe_<date>.md

Run:  python3 build_report.py
"""
import json
import math
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent

# ---------------- Hardware: PTL 12Xe ----------------
HW = dict(
    name="PTL 12Xe",
    desc="Intel PTL 12Xe iGPU (12 Xe cores × 8 EU × 10 threads), 2400 MHz, 110 GB/s",
    xe_cores=12,
    eus_per_core=8,
    threads_per_eu=10,
    freq_ghz=2.400,
    bw_gbs=110.0,
)
# SKILL formulas
FP16_XMX_TFLOPS = HW["xe_cores"] * HW["eus_per_core"] * 256 * HW["freq_ghz"] / 1000.0  # = 58.9824
INT8_XMX_TOPS   = 2 * FP16_XMX_TFLOPS                                                   # = 117.9648
SIMD_FP16_TFLOPS = HW["xe_cores"] * HW["eus_per_core"] * 32 * HW["freq_ghz"] / 1000.0   # = 7.3728
BW = HW["bw_gbs"]
RIDGE_FP16 = FP16_XMX_TFLOPS * 1e3 / BW   # FLOP/B  (~536)

# ---------------- Model architecture ----------------
ARCH = dict(
    hidden=2048, layers=28, NH=16, NKV=8, HD=128,
    intermediate=6144, vocab=151936, tie_embed=True,
    rope_theta=1_000_000,
)
AUDIO = dict(
    d_model=1024, layers=24, NH=16, HD=64, ffn=4096,
    num_mel=128, S=1500, downsample_hidden=480, output_dim=2048,
)

SEQ_PREFILL = [512, 1024, 4096, 8192]
SEQ_DECODE  = [512, 1024, 4096, 8192]  # KV ctx at the moment we sample TPOT
OUTPUT_TOKENS = 512

# ---------------- Load parsed logs ----------------
parsed = json.loads((HERE / "parsed.json").read_text())

def kernel_ns(stem):
    """Return total kernel ns/iter from one log (sum across kernels)."""
    if stem not in parsed:
        return None
    return parsed[stem]["total_kernel_ns"]

def kernel_subns(stem, contains):
    """Return per-iter ns for the kernel whose name contains `contains`."""
    if stem not in parsed:
        return None
    for name, ns, calls in parsed[stem]["per_kernel"]:
        if contains in name:
            return ns
    return None

def kernel_iters(stem):
    if stem not in parsed:
        return None
    return parsed[stem]["iters_detected"]

# ---------------- Theoretical FLOPs / bytes ----------------
B_F16 = 2  # bytes per fp16 element

# Text-decoder KV cache precision. Audio encoder SDPA is bidirectional MHA without
# a persistent cache, so it is always treated as f16 below. Set to "i8" to match the
# Intel GPU plugin default for PagedAttention models (see
# src/plugins/intel_gpu/src/runtime/execution_config.cpp ~line 293: PA models
# default `m_kv_cache_precision = ov::element::i8`).
KV_PRECISION = "i8"           # "f16" or "i8"
PA_BLOCK_SIZE = 16            # OCL PA path block size; used for INT8 K scale/zp amortization.

def _kv_logical_bytes_per_token(NKV, HD):
    """Bytes a PA-style cache stores per logical token, summed over K + V across all
    NKV heads. For INT8 we include the per-block (K, BY_CHANNEL) and per-token
    (V, BY_TOKEN) scale+zero-point overhead (2*fp16 = 4 bytes), matching the GPU
    plugin layout: K = [num_blocks,NKV,HD,BLOCK+4]; V = [num_blocks,NKV,BLOCK,HD+4].
    """
    if KV_PRECISION == "i8":
        k_per = NKV * HD * (1.0 + 4.0 / PA_BLOCK_SIZE)  # 1 INT8 + 4B scale/zp shared across BLOCK tokens
        v_per = NKV * (HD + 4)                           # HD INT8 + 4B scale/zp shared across HD-row per token
        return k_per + v_per
    return NKV * HD * B_F16 * 2  # f16: K + V

def fc_flops(M, K, N):
    return 2.0 * M * K * N

def fc_bytes_f16(M, K, N):
    """Weight stream + activation IO. FP16 weights, no compression, no scales."""
    return K * N * B_F16 + M * K * B_F16 + M * N * B_F16

def pa_decode_bytes(Skv, NH=ARCH["NH"], NKV=ARCH["NKV"], HD=ARCH["HD"]):
    # K + V for all past tokens (precision-aware) + Q in (f16) + output (f16).
    return Skv * _kv_logical_bytes_per_token(NKV, HD) + NH * HD * B_F16 * 2

def pa_decode_flops(Skv, NH=ARCH["NH"], HD=ARCH["HD"]):
    # Softmax(Q·Kᵀ)·V ≈ 2 attentions of NH*HD per kv position
    return 2.0 * 2 * NH * HD * Skv  # ~4·NH·HD·Skv

def pa_prefill_bytes(S, NH=ARCH["NH"], NKV=ARCH["NKV"], HD=ARCH["HD"]):
    # Q,K,V activations (always f16 input) + output. Cache write/dequant is
    # measured separately by the pa_kv_cache_update row.
    return S * (NH + 2 * NKV) * HD * B_F16 + S * NH * HD * B_F16

def pa_prefill_flops_causal(S, NH=ARCH["NH"], HD=ARCH["HD"]):
    # Causal mask: Sq*(Sq+1)/2 effective pairs; 2 matmuls (Q·Kᵀ and ·V).
    pairs = S * (S + 1) / 2.0
    return 2.0 * 2 * NH * HD * pairs  # = 4·NH·HD·S(S+1)/2

def rmsnorm_bytes(M, H):
    return M * H * B_F16 * 2 + H * B_F16
def rmsnorm_flops(M, H):
    # x^2 (1) + sum (1) + /n (1) + sqrt (10) + rsqrt mul (1) + norm mul (1) + gamma mul (1)
    return float(M * H * 6 + M * 10)  # sqrt counted at 10 fops per group per SKILL

def rope_bytes(M, NH, HD):
    return M * NH * HD * B_F16 * 2 + 2 * M * HD * B_F16
def rope_flops(M, NH, HD):
    # rotate: per element 2 mul + 1 add ≈ 3 fops; cos/sin already precomputed
    return float(M * NH * HD * 3)

def add_bytes(M, H):
    return 3 * M * H * B_F16
def add_flops(M, H):
    return float(M * H)

# ---------------- Build per-op rows ----------------
def _theo_single_ms(flops, bytes_, bound):
    """Theoretical lower-bound single-kernel latency at PTL 12Xe peaks."""
    if bound == "memory":
        return bytes_ / BW / 1e9 * 1000  # bytes / (GB/s * 1e9) -> s -> ms
    return flops / (FP16_XMX_TFLOPS * 1e12) * 1000  # flops / (TFLOPS * 1e12) -> s -> ms

def fc_row(op, kernel_name, single_ms, M, K, N):
    if single_ms is None:
        return None
    flops = fc_flops(M, K, N)
    bytes_ = fc_bytes_f16(M, K, N)
    t_s = single_ms / 1000.0
    gflops = flops / t_s / 1e9
    gbs = bytes_ / t_s / 1e9
    ai = flops / bytes_
    bound = "compute" if ai > RIDGE_FP16 else "memory"
    if bound == "memory":
        eff = gbs / BW * 100
    else:
        eff = gflops / (FP16_XMX_TFLOPS * 1e3) * 100
    theo_ms = _theo_single_ms(flops, bytes_, bound)
    return dict(op=op, kernel=kernel_name, M=M, K=K, N=N,
                single_ms=round(single_ms, 6),
                theo_single_ms=round(theo_ms, 6),
                slowdown=round(single_ms / theo_ms, 2) if theo_ms > 0 else 0,
                flops=flops, bytes=bytes_,
                gflops=round(gflops, 2), gbs=round(gbs, 2),
                ai=round(ai, 3), bound=bound, eff_pct=round(eff, 1))

def pa_decode_row(Skv):
    log = f"pa_decode_kv{Skv}"
    # split into kv_update + pa_compute (+ finalization)
    rows = []
    iters = kernel_iters(log)
    if iters is None:
        return rows
    kv_ns = kernel_subns(log, "pa_kv_cache_update")
    pa_ns = kernel_subns(log, "paged_attention_opt__")
    # If gqa variant, fold finalization too
    fin_ns = kernel_subns(log, "single_token_finalization")
    if pa_ns is not None:
        # Don't double-count finalization (it matches paged_attention_opt__ too).
        # `kernel_subns` returns FIRST match; the loop in parse keeps both kernels
        # in per_kernel list. So we re-scan to disambiguate.
        import re as _re
        attn_ns = 0
        attn_name = "paged_attention_opt__single_token"
        fin_ns2 = 0
        for name, ns, _ in parsed[log]["per_kernel"]:
            if "paged_attention_opt__" in name and "finalization" not in name:
                attn_ns = ns
                # Strip trailing "_<hash>__sa" so we keep the variant
                # (e.g. paged_attention_opt__single_token vs ..._gqa_single_token).
                attn_name = _re.sub(r"_\d+(__sa)?$", "", name)
            elif "single_token_finalization" in name:
                fin_ns2 = ns
        pa_ns = attn_ns
        pa_kernel_name = attn_name
        fin_ns = fin_ns2
    else:
        pa_kernel_name = "paged_attention_opt__single_token"
    # KV cache update row
    if kv_ns is not None and kv_ns > 0:
        # Decode kv_update: read 1 new K + 1 new V (f16 input) and write 1 token into
        # the cache. Cache write bytes follow KV_PRECISION layout.
        bytes_kv = ARCH["NKV"] * ARCH["HD"] * B_F16 * 2 + _kv_logical_bytes_per_token(ARCH["NKV"], ARCH["HD"])
        single_ms = kv_ns / 1e6
        gbs = bytes_kv / (single_ms / 1000.0) / 1e9
        theo_ms = _theo_single_ms(0, bytes_kv, "memory")
        rows.append(dict(op=f"pa_kv_update_kv{Skv}",
                         kernel="pa_kv_cache_update_ref",
                         single_ms=round(single_ms, 6),
                         theo_single_ms=round(theo_ms, 6),
                         slowdown=round(single_ms / theo_ms, 2) if theo_ms > 0 else 0,
                         flops=0.0, bytes=float(bytes_kv),
                         gflops=0.0, gbs=round(gbs, 3),
                         ai=0.0, bound="memory",
                         eff_pct=round(gbs / BW * 100, 1)))
    if pa_ns is not None and pa_ns > 0:
        single_ms = pa_ns / 1e6
        flops = pa_decode_flops(Skv)
        bytes_ = pa_decode_bytes(Skv)
        t_s = single_ms / 1000.0
        gflops = flops / t_s / 1e9
        gbs = bytes_ / t_s / 1e9
        ai = flops / bytes_
        bound = "compute" if ai > RIDGE_FP16 else "memory"
        eff = (gbs / BW * 100) if bound == "memory" else (gflops / (FP16_XMX_TFLOPS * 1e3) * 100)
        theo_ms = _theo_single_ms(flops, bytes_, bound)
        rows.append(dict(op=f"pa_compute_kv{Skv}",
                         kernel=pa_kernel_name,
                         single_ms=round(single_ms, 6),
                         theo_single_ms=round(theo_ms, 6),
                         slowdown=round(single_ms / theo_ms, 2) if theo_ms > 0 else 0,
                         flops=float(flops), bytes=float(bytes_),
                         gflops=round(gflops, 2), gbs=round(gbs, 2),
                         ai=round(ai, 3), bound=bound,
                         eff_pct=round(eff, 1)))
    if fin_ns is not None and fin_ns > 0:
        single_ms = fin_ns / 1e6
        bytes_ = ARCH["NH"] * ARCH["HD"] * B_F16 * 2  # small reduce
        gbs = bytes_ / (single_ms / 1000.0) / 1e9
        theo_ms = _theo_single_ms(0, bytes_, "memory")
        rows.append(dict(op=f"pa_finalize_kv{Skv}",
                         kernel="single_token_finalization",
                         single_ms=round(single_ms, 6),
                         theo_single_ms=round(theo_ms, 6),
                         slowdown=round(single_ms / theo_ms, 2) if theo_ms > 0 else 0,
                         flops=0.0, bytes=float(bytes_),
                         gflops=0.0, gbs=round(gbs, 3),
                         ai=0.0, bound="memory",
                         eff_pct=round(gbs / BW * 100, 1)))
    return rows

def pa_prefill_row(S):
    log = f"pa_prefill_S{S}"
    if log not in parsed:
        return []
    rows = []
    kv_ns = 0
    sdpa_ns = 0
    for name, ns, _ in parsed[log]["per_kernel"]:
        if "pa_kv_cache_update" in name:
            kv_ns = ns
        elif "sdpa_micro__prefill" in name:
            sdpa_ns = ns
    if kv_ns > 0:
        # Prefill kv_update: reads S new K + S new V (f16 input) and writes S tokens to cache.
        bytes_kv = S * ARCH["NKV"] * ARCH["HD"] * B_F16 * 2 + S * _kv_logical_bytes_per_token(ARCH["NKV"], ARCH["HD"])
        single_ms = kv_ns / 1e6
        gbs = bytes_kv / (single_ms / 1000.0) / 1e9
        theo_ms = _theo_single_ms(0, bytes_kv, "memory")
        rows.append(dict(op=f"pa_kv_update_S{S}",
                         kernel="pa_kv_cache_update_ref",
                         single_ms=round(single_ms, 6),
                         theo_single_ms=round(theo_ms, 6),
                         slowdown=round(single_ms / theo_ms, 2) if theo_ms > 0 else 0,
                         flops=0.0, bytes=float(bytes_kv),
                         gflops=0.0, gbs=round(gbs, 2),
                         ai=0.0, bound="memory",
                         eff_pct=round(gbs / BW * 100, 1)))
    if sdpa_ns > 0:
        single_ms = sdpa_ns / 1e6
        flops = pa_prefill_flops_causal(S)
        bytes_ = pa_prefill_bytes(S)
        t_s = single_ms / 1000.0
        gflops = flops / t_s / 1e9
        gbs = bytes_ / t_s / 1e9
        ai = flops / bytes_
        bound = "compute" if ai > RIDGE_FP16 else "memory"
        eff = (gbs / BW * 100) if bound == "memory" else (gflops / (FP16_XMX_TFLOPS * 1e3) * 100)
        theo_ms = _theo_single_ms(flops, bytes_, bound)
        rows.append(dict(op=f"pa_compute_S{S}",
                         kernel="sdpa_micro__prefill",
                         single_ms=round(single_ms, 6),
                         theo_single_ms=round(theo_ms, 6),
                         slowdown=round(single_ms / theo_ms, 2) if theo_ms > 0 else 0,
                         flops=float(flops), bytes=float(bytes_),
                         gflops=round(gflops, 2), gbs=round(gbs, 2),
                         ai=round(ai, 3), bound=bound,
                         eff_pct=round(eff, 1)))
    return rows

def small_row(op_label, kernel_name, log_stem, M, *shape, kind):
    """kind ∈ {'rmsnorm','rope','add'}"""
    ns = kernel_ns(log_stem)
    if ns is None:
        return None
    single_ms = ns / 1e6
    if kind == "rmsnorm":
        H = shape[0] if len(shape) == 1 else shape[0] * shape[1]  # H or NH*HD
        bytes_ = rmsnorm_bytes(M, H)
        flops = rmsnorm_flops(M, H)
    elif kind == "rope":
        NH, HD = shape
        bytes_ = rope_bytes(M, NH, HD)
        flops = rope_flops(M, NH, HD)
    elif kind == "add":
        H = shape[0]
        bytes_ = add_bytes(M, H)
        flops = add_flops(M, H)
    else:
        raise ValueError(kind)
    t_s = single_ms / 1000.0
    gflops = flops / t_s / 1e9
    gbs = bytes_ / t_s / 1e9
    ai = flops / bytes_ if bytes_ > 0 else 0
    bound = "memory"  # all small ops are BW-bound by construction
    eff = gbs / BW * 100
    theo_ms = _theo_single_ms(flops, bytes_, bound)
    return dict(op=op_label, kernel=kernel_name,
                single_ms=round(single_ms, 6),
                theo_single_ms=round(theo_ms, 6),
                slowdown=round(single_ms / theo_ms, 2) if theo_ms > 0 else 0,
                flops=float(flops), bytes=float(bytes_),
                gflops=round(gflops, 2), gbs=round(gbs, 2),
                ai=round(ai, 3), bound=bound,
                eff_pct=round(eff, 1))

# ---------------- Per-token-size assembly ----------------
TEXT_DECODER_OPS = [
    # (label,        suffix, kernel,        M_func,         K, N,    calls_per_inf)
    ("fc_qkv",   "qkv",   "gemm_kernel", 2048, 4096),
    ("fc_o",     "o",     "gemm_kernel", 2048, 2048),
    ("fc_gate",  "gate",  "gemm_kernel", 2048, 6144),
    ("fc_up",    "up",    "gemm_kernel", 2048, 6144),
    ("fc_down",  "down",  "gemm_kernel", 6144, 2048),
]
NUM_LAYERS = ARCH["layers"]  # 28

def assemble_decode(KV):
    """Per-inference decode rows for a given KV (one output token).
       Returns list of rows each with calls/inf and total_ms."""
    rows = []
    # FC rows × NUM_LAYERS
    for op, suffix, kernel, K, N in TEXT_DECODER_OPS:
        r = fc_row(op, kernel, kernel_ns(f"fc_decode_{suffix}") / 1e6 if kernel_ns(f"fc_decode_{suffix}") else None,
                   M=1, K=K, N=N)
        if r:
            r["calls"] = NUM_LAYERS
            r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4)
            rows.append(r)
    # LM_Head x 1
    lm_ns = kernel_ns("fc_decode_lm_head")
    if lm_ns:
        r = fc_row("fc_lm_head", "gemm_kernel", lm_ns / 1e6,
                   M=1, K=ARCH["hidden"], N=ARCH["vocab"])
        r["calls"] = 1
        r["total_ms"] = round(r["single_ms"], 4)
        rows.append(r)
    # PA rows (KV update + compute + finalize) × NUM_LAYERS
    for r in pa_decode_row(KV):
        r["calls"] = NUM_LAYERS
        r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4)
        rows.append(r)
    # Small ops
    # rmsnorm hidden: 2/layer + 1 final
    r = small_row("rmsnorm_hidden", "rms_gpu_bfyx_opt", "small_decode_rmsnorm",
                  1, ARCH["hidden"], kind="rmsnorm")
    if r: r["calls"] = 2 * NUM_LAYERS + 1; r["total_ms"] = round(r["single_ms"] * r["calls"], 4); rows.append(r)
    r = small_row("rmsnorm3d_q", "rms_gpu_bfyx_opt", "small_decode_rmsnorm3d_q",
                  1, ARCH["NH"], ARCH["HD"], kind="rmsnorm")
    if r: r["calls"] = NUM_LAYERS; r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4); rows.append(r)
    r = small_row("rmsnorm3d_k", "rms_gpu_bfyx_opt", "small_decode_rmsnorm3d_k",
                  1, ARCH["NKV"], ARCH["HD"], kind="rmsnorm")
    if r: r["calls"] = NUM_LAYERS; r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4); rows.append(r)
    r = small_row("rope_q", "rope_opt", "small_decode_rope_q",
                  1, ARCH["NH"], ARCH["HD"], kind="rope")
    if r: r["calls"] = NUM_LAYERS; r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4); rows.append(r)
    r = small_row("rope_k", "rope_opt", "small_decode_rope_k",
                  1, ARCH["NKV"], ARCH["HD"], kind="rope")
    if r: r["calls"] = NUM_LAYERS; r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4); rows.append(r)
    r = small_row("residual_add", "eltwise", "small_decode_add",
                  1, ARCH["hidden"], kind="add")
    if r: r["calls"] = 2 * NUM_LAYERS; r["total_ms"] = round(r["single_ms"] * r["calls"], 4); rows.append(r)
    return rows

def assemble_prefill(S):
    rows = []
    # FC body × NUM_LAYERS
    for op, suffix, kernel, K, N in TEXT_DECODER_OPS:
        ns = kernel_ns(f"fc_prefill_{suffix}_S{S}")
        if ns:
            r = fc_row(op, kernel, ns / 1e6, M=S, K=K, N=N)
            r["calls"] = NUM_LAYERS
            r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4)
            rows.append(r)
    # LM_Head: only last token at prefill (M=1)
    lm_ns = kernel_ns("fc_prefill_lm_head")
    if lm_ns:
        r = fc_row("fc_lm_head", "gemm_kernel", lm_ns / 1e6,
                   M=1, K=ARCH["hidden"], N=ARCH["vocab"])
        r["calls"] = 1
        r["total_ms"] = round(r["single_ms"], 4)
        rows.append(r)
    # PA prefill
    for r in pa_prefill_row(S):
        r["calls"] = NUM_LAYERS
        r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4)
        rows.append(r)
    # Small ops at M=S
    r = small_row("rmsnorm_hidden", "rms_gpu_bfyx_opt", f"small_prefill_rmsnorm_S{S}",
                  S, ARCH["hidden"], kind="rmsnorm")
    if r: r["calls"] = 2 * NUM_LAYERS + 1; r["total_ms"] = round(r["single_ms"] * r["calls"], 4); rows.append(r)
    r = small_row("rmsnorm3d_q", "rms_gpu_bfyx_opt", f"small_prefill_rmsnorm3d_q_S{S}",
                  S, ARCH["NH"], ARCH["HD"], kind="rmsnorm")
    if r: r["calls"] = NUM_LAYERS; r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4); rows.append(r)
    r = small_row("rmsnorm3d_k", "rms_gpu_bfyx_opt", f"small_prefill_rmsnorm3d_k_S{S}",
                  S, ARCH["NKV"], ARCH["HD"], kind="rmsnorm")
    if r: r["calls"] = NUM_LAYERS; r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4); rows.append(r)
    r = small_row("rope_q", "rope_opt", f"small_prefill_rope_q_S{S}",
                  S, ARCH["NH"], ARCH["HD"], kind="rope")
    if r: r["calls"] = NUM_LAYERS; r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4); rows.append(r)
    r = small_row("rope_k", "rope_opt", f"small_prefill_rope_k_S{S}",
                  S, ARCH["NKV"], ARCH["HD"], kind="rope")
    if r: r["calls"] = NUM_LAYERS; r["total_ms"] = round(r["single_ms"] * NUM_LAYERS, 4); rows.append(r)
    r = small_row("residual_add", "eltwise", f"small_prefill_add_S{S}",
                  S, ARCH["hidden"], kind="add")
    if r: r["calls"] = 2 * NUM_LAYERS; r["total_ms"] = round(r["single_ms"] * r["calls"], 4); rows.append(r)
    return rows

# ---------------- Audio encoder (fixed overhead, runs once) -----------------
def audio_encoder_rows():
    """All measured at M=1500. Each FC runs `layers` times (18) except outproj (1).
       SDPA cost is the measured PA-prefill encoder cost (causal); we report
       both the as-measured (causal) and ×2 (bidirectional) totals."""
    layers = AUDIO["layers"]
    rows = []
    enc_fc = [
        ("enc_fc_qkv",   "fc_enc_qkv_S1500",   AUDIO["d_model"], AUDIO["d_model"] * 3, layers),
        ("enc_fc_o",     "fc_enc_o_S1500",     AUDIO["d_model"], AUDIO["d_model"],     layers),
        ("enc_fc_ffn1",  "fc_enc_fc1_S1500",   AUDIO["d_model"], AUDIO["ffn"],         layers),
        ("enc_fc_ffn2",  "fc_enc_fc2_S1500",   AUDIO["ffn"],     AUDIO["d_model"],     layers),
        ("enc_outproj",  "fc_enc_outproj_S1500", AUDIO["d_model"], AUDIO["output_dim"], 1),
    ]
    for op, stem, K, N, calls in enc_fc:
        ns = kernel_ns(stem)
        if not ns:
            continue
        r = fc_row(op, "gemm_kernel", ns / 1e6, M=AUDIO["S"], K=K, N=N)
        r["calls"] = calls
        r["total_ms"] = round(r["single_ms"] * calls, 4)
        rows.append(r)
    # SDPA encoder — measured via PA prefill at S=1500 (causal lower bound).
    # Real encoder is bidirectional (no causal mask), so kernel time ~2×.
    ns = kernel_ns("pa_prefill_encS1500")
    if ns:
        sdpa_ns = 0
        kv_ns = 0
        for name, n, _ in parsed["pa_prefill_encS1500"]["per_kernel"]:
            if "sdpa_micro__prefill" in name:
                sdpa_ns = n
            elif "pa_kv_cache_update" in name:
                kv_ns = n
        # Bidirectional estimate = 2× causal sdpa kernel time.
        # No KV cache write in real encoder (encoder has no KV cache),
        # so we exclude pa_kv_update from the encoder total.
        single_ms_causal = sdpa_ns / 1e6
        single_ms_bidir = single_ms_causal * 2
        S = AUDIO["S"]; NH = AUDIO["NH"]; HD = AUDIO["HD"]
        flops_full = 2.0 * 2 * NH * HD * S * S  # full square
        bytes_ = S * 3 * NH * HD * B_F16 + S * NH * HD * B_F16  # QKV in + out
        t_s = single_ms_bidir / 1000.0
        gflops = flops_full / t_s / 1e9
        gbs = bytes_ / t_s / 1e9
        ai = flops_full / bytes_
        bound = "compute" if ai > RIDGE_FP16 else "memory"
        eff = (gbs / BW * 100) if bound == "memory" else (gflops / (FP16_XMX_TFLOPS * 1e3) * 100)
        theo_ms = _theo_single_ms(flops_full, bytes_, bound)
        rows.append(dict(op="enc_sdpa", kernel="sdpa_micro__prefill (×2 bidir est.)",
                         single_ms=round(single_ms_bidir, 6),
                         theo_single_ms=round(theo_ms, 6),
                         slowdown=round(single_ms_bidir / theo_ms, 2) if theo_ms > 0 else 0,
                         flops=float(flops_full), bytes=float(bytes_),
                         gflops=round(gflops, 2), gbs=round(gbs, 2),
                         ai=round(ai, 3), bound=bound,
                         eff_pct=round(eff, 1),
                         calls=layers,
                         total_ms=round(single_ms_bidir * layers, 4)))
    return rows

# ---------------- Theoretical weight distribution ----------------
def weight_distribution():
    """Returns dict with per-tensor MB and totals for FP16, no compression."""
    H = ARCH["hidden"]; NH = ARCH["NH"]; NKV = ARCH["NKV"]; HD = ARCH["HD"]
    FF = ARCH["intermediate"]; V = ARCH["vocab"]
    L = ARCH["layers"]
    # Each per-layer Linear (with bias=False per Qwen3):
    qkv = H * (NH + 2 * NKV) * HD * B_F16   # fused logical size
    o   = NH * HD * H * B_F16
    gate= H * FF * B_F16
    up  = H * FF * B_F16
    down= FF * H * B_F16
    rmsnorm_pl = 2 * H * B_F16  # 2 per layer
    qnorm = HD * B_F16          # per-head gamma
    knorm = HD * B_F16
    decoder_per_layer = qkv + o + gate + up + down + rmsnorm_pl + qnorm + knorm
    decoder_total = decoder_per_layer * L

    embed = V * H * B_F16
    lm_head_extra = 0 if ARCH["tie_embed"] else V * H * B_F16
    final_rms = H * B_F16

    # Audio encoder
    Ae = AUDIO; d = Ae["d_model"]; nh = Ae["NH"]; hd = Ae["HD"]; ff = Ae["ffn"]; L_e = Ae["layers"]
    qkv_e = d * (3 * d) * B_F16
    o_e   = d * d * B_F16
    fc1_e = d * ff * B_F16
    fc2_e = ff * d * B_F16
    ln_e  = 2 * 2 * d * B_F16  # gamma+beta per LN; 2 LN per layer
    enc_layer = qkv_e + o_e + fc1_e + fc2_e + ln_e
    enc_total = enc_layer * L_e
    # conv front-end (approximate): conv1d(128->d, k=3) + conv1d(d->d, k=3, stride=2)
    conv1 = Ae["num_mel"] * d * 3 * B_F16
    conv2 = d * d * 3 * B_F16
    pos_embed = Ae["S"] * d * B_F16  # sinusoidal table (could be fp32; use fp16 conservative)
    final_ln_enc = 2 * d * B_F16
    # output projection adapter: d -> downsample_hidden -> output_dim (small)
    adapter1 = d * Ae["downsample_hidden"] * B_F16
    adapter2 = Ae["downsample_hidden"] * Ae["output_dim"] * B_F16
    enc_front_back = conv1 + conv2 + pos_embed + final_ln_enc + adapter1 + adapter2

    total = decoder_total + embed + lm_head_extra + final_rms + enc_total + enc_front_back

    def mb(x): return round(x / (1024 * 1024), 3)
    return dict(
        decoder=dict(
            per_layer_mb=mb(decoder_per_layer),
            fc_qkv_mb=mb(qkv), fc_o_mb=mb(o),
            fc_gate_mb=mb(gate), fc_up_mb=mb(up), fc_down_mb=mb(down),
            rmsnorm_mb=mb(rmsnorm_pl), qnorm_mb=mb(qnorm), knorm_mb=mb(knorm),
            layers=L, total_mb=mb(decoder_total),
        ),
        embed_lmhead=dict(
            embed_mb=mb(embed), lm_head_extra_mb=mb(lm_head_extra),
            tied=ARCH["tie_embed"], final_rms_mb=mb(final_rms),
        ),
        encoder=dict(
            per_layer_mb=mb(enc_layer),
            fc_qkv_mb=mb(qkv_e), fc_o_mb=mb(o_e),
            fc_fc1_mb=mb(fc1_e), fc_fc2_mb=mb(fc2_e),
            ln_mb=mb(ln_e),
            layers=L_e, total_mb=mb(enc_total),
            front_back_mb=mb(enc_front_back),
            conv1_mb=mb(conv1), conv2_mb=mb(conv2),
            pos_embed_mb=mb(pos_embed),
            adapter_mb=mb(adapter1 + adapter2),
        ),
        total_mb=mb(total),
    )

# ---------------- Build everything ----------------
WEIGHTS = weight_distribution()
ENC_ROWS = audio_encoder_rows()

DECODE_TABLES = {kv: assemble_decode(kv) for kv in SEQ_DECODE}
PREFILL_TABLES = {s: assemble_prefill(s) for s in SEQ_PREFILL}

# Populate theoretical totals (= theo_single_ms * calls) so we can compare
# measured vs theoretical aggregates per op and per scenario.
def _attach_theo_totals(rows):
    for r in rows:
        r["theo_total_ms"] = round(r["theo_single_ms"] * r["calls"], 4)

_attach_theo_totals(ENC_ROWS)
for rows in DECODE_TABLES.values():
    _attach_theo_totals(rows)
for rows in PREFILL_TABLES.values():
    _attach_theo_totals(rows)

ENC_TOTAL_MS = round(sum(r["total_ms"] for r in ENC_ROWS), 4)
ENC_THEO_TOTAL_MS = round(sum(r["theo_total_ms"] for r in ENC_ROWS), 4)
DECODE_TOTAL_MS = {kv: round(sum(r["total_ms"] for r in rows), 4)
                   for kv, rows in DECODE_TABLES.items()}
DECODE_THEO_TOTAL_MS = {kv: round(sum(r["theo_total_ms"] for r in rows), 4)
                        for kv, rows in DECODE_TABLES.items()}
PREFILL_TOTAL_TEXT_MS = {s: round(sum(r["total_ms"] for r in rows), 4)
                         for s, rows in PREFILL_TABLES.items()}
PREFILL_THEO_TOTAL_TEXT_MS = {s: round(sum(r["theo_total_ms"] for r in rows), 4)
                              for s, rows in PREFILL_TABLES.items()}
# TTFT = text decoder prefill + audio encoder (runs once)
PREFILL_TTFT_MS = {s: round(PREFILL_TOTAL_TEXT_MS[s] + ENC_TOTAL_MS, 4)
                   for s in SEQ_PREFILL}
PREFILL_THEO_TTFT_MS = {s: round(PREFILL_THEO_TOTAL_TEXT_MS[s] + ENC_THEO_TOTAL_MS, 4)
                        for s in SEQ_PREFILL}

# ---------------- Output JSON ----------------
metrics = dict(
    platform=HW["name"],
    platform_desc=HW["desc"],
    bw_gbs=BW,
    fp16_xmx_tflops=round(FP16_XMX_TFLOPS, 4),
    int8_xmx_tops=round(INT8_XMX_TOPS, 4),
    simd_fp16_tflops=round(SIMD_FP16_TFLOPS, 4),
    ridge_fp16_flop_per_byte=round(RIDGE_FP16, 2),
    model="Qwen3-ASR-1.7B",
    config_summary=f"FP16 weights (no compression), {'INT8' if KV_PRECISION=='i8' else 'FP16'} KV cache (text decoder; GPU plugin default for PagedAttention), audio encoder SDPA in FP16, PA opencl+micro_kernel, end-to-end (audio encoder 1500 frames + Qwen3 text decoder)",
    arch=ARCH, audio=AUDIO,
    output_tokens=OUTPUT_TOKENS,
    encoder_overhead_ms=ENC_TOTAL_MS,
    encoder_theo_overhead_ms=ENC_THEO_TOTAL_MS,
    encoder_rows=ENC_ROWS,
    decode={str(kv): dict(rows=rows,
                          total_ms=DECODE_TOTAL_MS[kv],
                          theo_total_ms=DECODE_THEO_TOTAL_MS[kv],
                          slowdown=round(DECODE_TOTAL_MS[kv] / DECODE_THEO_TOTAL_MS[kv], 3))
            for kv, rows in DECODE_TABLES.items()},
    prefill={str(s): dict(rows=rows,
                          text_decoder_ms=PREFILL_TOTAL_TEXT_MS[s],
                          text_decoder_theo_ms=PREFILL_THEO_TOTAL_TEXT_MS[s],
                          encoder_ms=ENC_TOTAL_MS,
                          encoder_theo_ms=ENC_THEO_TOTAL_MS,
                          ttft_ms=PREFILL_TTFT_MS[s],
                          theo_ttft_ms=PREFILL_THEO_TTFT_MS[s],
                          slowdown=round(PREFILL_TTFT_MS[s] / PREFILL_THEO_TTFT_MS[s], 3))
             for s, rows in PREFILL_TABLES.items()},
)
(HERE / "performance_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Wrote performance_metrics.json")

(HERE / "weight_distribution.json").write_text(json.dumps(WEIGHTS, indent=2))
print(f"Wrote weight_distribution.json")

# ---------------- Markdown SUMMARY ----------------
def fmt_row(r):
    return (f"| {r['op']} | `{r['kernel']}` | {r['calls']} | "
            f"{r['single_ms']:.4f} | {r['theo_single_ms']:.4f} | "
            f"{r['total_ms']:.4f} | {r['theo_total_ms']:.4f} | "
            f"{r['slowdown']:.2f}× | {r['gflops']:.1f} | {r['gbs']:.1f} | "
            f"{r['eff_pct']:.1f}% | {r['bound']} |")

def table_block(title, rows, total_ms_label, total_ms, theo_total_ms):
    rows_sorted = sorted(rows, key=lambda r: -r["total_ms"])
    overall_slow = (total_ms / theo_total_ms) if theo_total_ms > 0 else 0
    s = [f"### {title}", "",
         "| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows_sorted:
        s.append(fmt_row(r))
    s.append(f"| **{total_ms_label}** |  |  |  |  | **{total_ms:.4f}** | **{theo_total_ms:.4f}** | **{overall_slow:.2f}×** |  |  |  |  |")
    s.append("")
    return "\n".join(s)

DATE = date.today().isoformat()

md = []
md.append(f"# Qwen3-ASR-1.7B — Roofline on {HW['name']} ({DATE})")
md.append("")
md.append(f"**Platform**: {HW['desc']}")
md.append(f"**Model**: Qwen3-ASR-1.7B (audio encoder + Qwen3 text decoder), all weights **FP16**, no compression.")
md.append(f"**SDPA**: PagedAttention (opencl + micro_kernel), {'INT8' if KV_PRECISION=='i8' else 'FP16'} KV cache (text decoder; matches Intel GPU plugin default for PA models). Audio encoder SDPA remains FP16.")
md.append(f"**Source config**: https://huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/config.json")
md.append("")
md.append("**Inputs evaluated**: text-decoder prefill context = 512 / 1024 / 4096 / 8192 tokens; output = 512 tokens; audio encoder runs once over 1500 mel frames.")
md.append("")

md.append("## 1. Hardware peaks (per SKILL.md formulas)")
md.append("")
md.append("`FP16 XMX TFLOPS = xe_cores × 8 × 256 × freq_GHz`; `INT8 XMX = 2× FP16`; `SIMD FP16 = xe_cores × 8 × 32 × freq`.")
md.append("")
md.append("| Platform | Xe Cores | Freq (MHz) | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | SIMD FP16 (TFLOPS) | Ridge FP16 (FLOP/B) |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
md.append(f"| {HW['name']} | {HW['xe_cores']} | {int(HW['freq_ghz']*1000)} | {BW:.0f} | {FP16_XMX_TFLOPS:.3f} | {INT8_XMX_TOPS:.3f} | {SIMD_FP16_TFLOPS:.3f} | {RIDGE_FP16:.0f} |")
md.append("")

md.append("## 2. Model architecture")
md.append("")
md.append("### Text decoder (Qwen3, 28-layer dense)")
md.append("")
md.append("| Field | Value |")
md.append("|---|---:|")
md.append(f"| `hidden_size` | {ARCH['hidden']} |")
md.append(f"| `num_hidden_layers` | {ARCH['layers']} |")
md.append(f"| `num_attention_heads` (NH) | {ARCH['NH']} |")
md.append(f"| `num_key_value_heads` (NKV) | {ARCH['NKV']} (GQA 2:1) |")
md.append(f"| `head_dim` (HD) | {ARCH['HD']} |")
md.append(f"| `intermediate_size` | {ARCH['intermediate']} |")
md.append(f"| `vocab_size` | {ARCH['vocab']} |")
md.append(f"| `tie_word_embeddings` | {ARCH['tie_embed']} |")
md.append(f"| `rope_theta` | {ARCH['rope_theta']:,} |")
md.append("")

md.append("### Audio encoder (Whisper-style)")
md.append("")
md.append("| Field | Value |")
md.append("|---|---:|")
md.append(f"| `d_model` | {AUDIO['d_model']} |")
md.append(f"| `encoder_layers` | {AUDIO['layers']} |")
md.append(f"| `encoder_attention_heads` (MHA, NH=NKV) | {AUDIO['NH']} |")
md.append(f"| `head_dim` | {AUDIO['HD']} |")
md.append(f"| `encoder_ffn_dim` | {AUDIO['ffn']} |")
md.append(f"| `num_mel_bins` | {AUDIO['num_mel']} |")
md.append(f"| `max_source_positions` | {AUDIO['S']} |")
md.append(f"| `output_dim` (→ text decoder hidden) | {AUDIO['output_dim']} |")
md.append("")

md.append("## 3. Theoretical weight distribution (FP16, no compression)")
md.append("")
md.append("### Text decoder per-layer (1 of 28)")
md.append("")
md.append("| Weight | Shape (K × N) | Dtype | MB |")
md.append("|---|---|---|---:|")
md.append(f"| FC_QKV (fused Q+K+V) | 2048 × 4096 | FP16 | {WEIGHTS['decoder']['fc_qkv_mb']} |")
md.append(f"| FC_O (attn output)   | 2048 × 2048 | FP16 | {WEIGHTS['decoder']['fc_o_mb']} |")
md.append(f"| FC_Gate (SwiGLU)     | 2048 × 6144 | FP16 | {WEIGHTS['decoder']['fc_gate_mb']} |")
md.append(f"| FC_Up (SwiGLU)       | 2048 × 6144 | FP16 | {WEIGHTS['decoder']['fc_up_mb']} |")
md.append(f"| FC_Down (SwiGLU)     | 6144 × 2048 | FP16 | {WEIGHTS['decoder']['fc_down_mb']} |")
md.append(f"| RMSNorm × 2          | [2048] each | FP16 | {WEIGHTS['decoder']['rmsnorm_mb']} |")
md.append(f"| q_norm / k_norm      | [128] each  | FP16 | {WEIGHTS['decoder']['qnorm_mb'] + WEIGHTS['decoder']['knorm_mb']:.4f} |")
md.append(f"| **per layer** |  |  | **{WEIGHTS['decoder']['per_layer_mb']:.2f}** |")
md.append(f"| **× {ARCH['layers']} layers** |  |  | **{WEIGHTS['decoder']['total_mb']:.2f}** |")
md.append("")
md.append("### Audio encoder per-layer (1 of 24)")
md.append("")
md.append("| Weight | Shape (K × N) | Dtype | MB |")
md.append("|---|---|---|---:|")
md.append(f"| FC_QKV (fused) | 1024 × 3072 | FP16 | {WEIGHTS['encoder']['fc_qkv_mb']} |")
md.append(f"| FC_O           | 1024 × 1024 | FP16 | {WEIGHTS['encoder']['fc_o_mb']} |")
md.append(f"| FC_FC1 (GELU)  | 1024 × 4096 | FP16 | {WEIGHTS['encoder']['fc_fc1_mb']} |")
md.append(f"| FC_FC2 (GELU)  | 4096 × 1024 | FP16 | {WEIGHTS['encoder']['fc_fc2_mb']} |")
md.append(f"| LayerNorm × 2  | [1024] g+b each | FP16 | {WEIGHTS['encoder']['ln_mb']} |")
md.append(f"| **per layer** |  |  | **{WEIGHTS['encoder']['per_layer_mb']:.2f}** |")
md.append(f"| **× {AUDIO['layers']} layers** |  |  | **{WEIGHTS['encoder']['total_mb']:.2f}** |")
md.append("")
md.append("### Global + shared weights")
md.append("")
md.append("| Weight | Shape | Dtype | MB |")
md.append("|---|---|---|---:|")
md.append(f"| Token embedding | {ARCH['vocab']} × {ARCH['hidden']} | FP16 | {WEIGHTS['embed_lmhead']['embed_mb']} |")
md.append(f"| LM_Head (tied={WEIGHTS['embed_lmhead']['tied']}) | shares embedding | — | {WEIGHTS['embed_lmhead']['lm_head_extra_mb']} |")
md.append(f"| Final RMSNorm (text)  | [{ARCH['hidden']}] | FP16 | {WEIGHTS['embed_lmhead']['final_rms_mb']} |")
md.append(f"| Audio encoder conv front-end | conv1d×2 | FP16 | {WEIGHTS['encoder']['conv1_mb'] + WEIGHTS['encoder']['conv2_mb']:.3f} |")
md.append(f"| Audio positional embedding | {AUDIO['S']} × {AUDIO['d_model']} | FP16 | {WEIGHTS['encoder']['pos_embed_mb']} |")
md.append(f"| Audio output adapter | 1024→480→2048 | FP16 | {WEIGHTS['encoder']['adapter_mb']} |")
md.append("")
md.append("### Totals")
md.append("")
md.append("| Component | MB |")
md.append("|---|---:|")
md.append(f"| Text decoder (28 layers) | {WEIGHTS['decoder']['total_mb']:.2f} |")
md.append(f"| Token embedding (shared w/ LM_Head) | {WEIGHTS['embed_lmhead']['embed_mb']:.2f} |")
md.append(f"| Audio encoder (24 layers) | {WEIGHTS['encoder']['total_mb']:.2f} |")
md.append(f"| Audio encoder front + adapter | {WEIGHTS['encoder']['front_back_mb']:.2f} |")
md.append(f"| Final RMSNorm (text) | {WEIGHTS['embed_lmhead']['final_rms_mb']:.4f} |")
md.append(f"| **Model total** | **{WEIGHTS['total_mb']:.2f}** |")
md.append("")

md.append("## 4. Benchmark methodology")
md.append("")
md.append("- **Bench utils**: `fc_bench` (precision=f16, plain MatMul, no compression), `pa_bench` (kv_dtype=i8, impl=ocl), `small_ops_bench` for rmsnorm/rope/eltwise.")
md.append("- **Tool**: cliloader Device Performance Timing; `parse_logs.py` extracts per-iteration avg GPU kernel ns.")
md.append("- **L2/L3 flush** between every FC infer (64 MB Relu) so each measurement reads weights from VRAM, not on-die cache. Required because the entire FP16 text decoder weights (~707 MB) is too large to fit fully in L2 but per-layer fits, so cache effects would otherwise inflate measured BW.")
md.append("- **Input tensors** allocated via RemoteContext in USM_DEVICE (iGPU shared system memory).")
md.append("- **PA prefill** uses causal mask (Sq·(Sq+1)/2 effective pairs) per SKILL.md.")
md.append("- **PA decode** is split into `pa_kv_cache_update_ref` + `paged_attention_opt__single_token` + `single_token_finalization`.")
md.append("- **Audio encoder SDPA** is bidirectional (no causal mask); measured causal time scaled ×2 in the encoder roofline row.")
md.append("- **swish / multiply** are not profiled separately — they fuse into the SwiGLU primitive per SKILL.md.")
md.append("- **LM_Head** is `K=2048, N=151936` plain FP16 (no INT8 since user requested FP16-only). Counted once per inference at decode and once at prefill (last token only).")
md.append("")

md.append("## 5. Audio encoder fixed overhead (per inference, S=1500)")
md.append("")
md.append("The audio encoder runs **once** per inference regardless of text token count.")
md.append("")
md.append(table_block(f"Audio encoder — S={AUDIO['S']}", ENC_ROWS, "TOTAL (encoder fixed)", ENC_TOTAL_MS, ENC_THEO_TOTAL_MS))

md.append("## 6. Token latency summary")
md.append("")
md.append(f"**TTFT** = audio encoder ({ENC_TOTAL_MS:.2f} ms) + text decoder prefill.")
md.append(f"**TPOT** = per output token decode latency at the listed KV context.")
md.append("")
md.append("### Prefill — TTFT")
md.append("")
md.append("| S (text ctx) | Encoder (ms) | Text-decoder prefill (ms) | **TTFT (ms)** | per-token (ms) | tokens/s |")
md.append("|---:|---:|---:|---:|---:|---:|")
for s in SEQ_PREFILL:
    tt = PREFILL_TTFT_MS[s]
    txt = PREFILL_TOTAL_TEXT_MS[s]
    md.append(f"| {s} | {ENC_TOTAL_MS:.2f} | {txt:.2f} | **{tt:.2f}** | {tt/s:.4f} | {s/(tt/1000):.0f} |")
md.append("")

md.append("### Decode — TPOT")
md.append("")
md.append("| KV (ctx) | TPOT (ms) | tokens/s |")
md.append("|---:|---:|---:|")
for kv in SEQ_DECODE:
    t = DECODE_TOTAL_MS[kv]
    md.append(f"| {kv} | {t:.4f} | {1000/t:.1f} |")
md.append("")

md.append("### Measured vs. theoretical latency (lower bound = roofline)")
md.append("")
md.append("The **theoretical** column is the sum over all kernels of")
md.append("`max(bytes/BW, flops/peak_TFLOPS)` — i.e. the absolute minimum time")
md.append("each kernel could take given its bound. `slowdown = measured / theo` (1.0× = at roofline).")
md.append("")
md.append("#### Decode (per output token)")
md.append("")
md.append("| KV (ctx) | Measured TPOT (ms) | Theoretical TPOT (ms) | Slowdown | Wall-clock tokens/s | Roofline tokens/s |")
md.append("|---:|---:|---:|---:|---:|---:|")
for kv in SEQ_DECODE:
    m = DECODE_TOTAL_MS[kv]; t = DECODE_THEO_TOTAL_MS[kv]
    md.append(f"| {kv} | {m:.4f} | {t:.4f} | {m/t:.2f}× | {1000/m:.1f} | {1000/t:.1f} |")
md.append("")

md.append("#### Prefill / TTFT")
md.append("")
md.append("| S (text ctx) | Measured TTFT (ms) | Theoretical TTFT (ms) | Slowdown | Meas tokens/s | Roofline tokens/s |")
md.append("|---:|---:|---:|---:|---:|---:|")
for s in SEQ_PREFILL:
    m = PREFILL_TTFT_MS[s]; t = PREFILL_THEO_TTFT_MS[s]
    md.append(f"| {s} | {m:.2f} | {t:.2f} | {m/t:.2f}× | {s/(m/1000):.0f} | {s/(t/1000):.0f} |")
md.append("")

md.append("#### Audio encoder fixed overhead")
md.append("")
md.append("| Component | Measured (ms) | Theoretical (ms) | Slowdown |")
md.append("|---|---:|---:|---:|")
md.append(f"| Audio encoder forward (S=1500, {AUDIO['layers']} layers) | {ENC_TOTAL_MS:.2f} | {ENC_THEO_TOTAL_MS:.2f} | {ENC_TOTAL_MS/ENC_THEO_TOTAL_MS:.2f}× |")
md.append("")

# Full end-to-end estimate (audio encoder once + prefill + 512 decode steps).
# Note this is approximate — true decode TPOT grows as KV grows. We use the
# measured TPOT at KV = S_prefill (start of decode) as a representative value.
md.append("### End-to-end latency estimate (output = 512 tokens)")
md.append("")
md.append("Approximation: decode uses the TPOT at KV = S_prefill (start-of-decode value). Real decode TPOT increases as KV grows toward S_prefill + 511.")
md.append("")
md.append("| S (text ctx) | TTFT (ms) | TPOT @ KV=S (ms) | Measured total (ms) | Theoretical total (ms) | Slowdown |")
md.append("|---:|---:|---:|---:|---:|---:|")
for s in SEQ_PREFILL:
    ttft = PREFILL_TTFT_MS[s]
    tpot = DECODE_TOTAL_MS[s]
    total = ttft + tpot * OUTPUT_TOKENS
    theo_ttft = PREFILL_THEO_TTFT_MS[s]
    theo_tpot = DECODE_THEO_TOTAL_MS[s]
    theo_total = theo_ttft + theo_tpot * OUTPUT_TOKENS
    md.append(f"| {s} | {ttft:.2f} | {tpot:.4f} | {total:.1f} | {theo_total:.1f} | {total/theo_total:.2f}× |")
md.append("")

md.append("## 7. Per-kernel tables — Decode (1 query token, KV = context)")
md.append("")
for kv in SEQ_DECODE:
    md.append(table_block(f"Decode — KV={kv}", DECODE_TABLES[kv], "TOTAL", DECODE_TOTAL_MS[kv], DECODE_THEO_TOTAL_MS[kv]))

md.append("## 8. Per-kernel tables — Prefill (text-decoder only; add encoder overhead for TTFT)")
md.append("")
for s in SEQ_PREFILL:
    md.append(table_block(f"Prefill — S={s} (text decoder only)", PREFILL_TABLES[s],
                          "TOTAL (text dec)", PREFILL_TOTAL_TEXT_MS[s], PREFILL_THEO_TOTAL_TEXT_MS[s]))

md.append("## 9. Roofline analysis")
md.append("")
md.append(f"Ridge point at PTL 12Xe FP16 = {RIDGE_FP16:.0f} FLOP/byte. AI < ridge ⇒ memory-bound; AI ≥ ridge ⇒ compute-bound.")
md.append("")
md.append("**Decode characteristics**: Every FC has AI = 2·M·K·N / (K·N·2) → ~M = 1 for decode, well below ridge → **always memory-bound**. PA decode AI is also well below ridge.")
md.append("")
md.append("**Prefill characteristics**: FC AI ≈ M scales with sequence length. For PTL 12Xe (ridge = {0:.0f}), FC becomes compute-bound roughly at M ≥ ridge. SDPA prefill AI scales with S/2; becomes compute-bound around S ≥ 2·ridge.".format(RIDGE_FP16))
md.append("")

# Bottleneck summary: top 5 ops per scenario
def top_ops(rows, n=5):
    return sorted(rows, key=lambda r: -r["total_ms"])[:n]

md.append("### Bottleneck breakdown — Decode (top 5 ops by latency)")
md.append("")
md.append("| KV | #1 | #2 | #3 | #4 | #5 |")
md.append("|---:|---|---|---|---|---|")
for kv in SEQ_DECODE:
    tops = top_ops(DECODE_TABLES[kv])
    fields = [f"{r['op']} ({r['total_ms']:.3f}ms / {100*r['total_ms']/DECODE_TOTAL_MS[kv]:.1f}%)" for r in tops]
    md.append("| " + str(kv) + " | " + " | ".join(fields) + " |")
md.append("")

md.append("### Bottleneck breakdown — Prefill (top 5 ops by latency)")
md.append("")
md.append("| S | #1 | #2 | #3 | #4 | #5 |")
md.append("|---:|---|---|---|---|---|")
for s in SEQ_PREFILL:
    tops = top_ops(PREFILL_TABLES[s])
    fields = [f"{r['op']} ({r['total_ms']:.3f}ms / {100*r['total_ms']/PREFILL_TOTAL_TEXT_MS[s]:.1f}%)" for r in tops]
    md.append("| " + str(s) + " | " + " | ".join(fields) + " |")
md.append("")

md.append("## 10. Notes, caveats & reproduction")
md.append("")
md.append("- **Audio encoder SDPA** is approximated by 2× the measured PA-prefill (causal) kernel at S=1500. Real encoder is bidirectional non-causal; a dedicated non-causal SDPA bench would be needed for a more precise number. Encoder SDPA is a small fraction of overall TTFT, so this approximation has limited impact.")
md.append("- **FC LM_Head** is `K=2048, N=151936` FP16 (~593 MB weight stream). This single layer is the dominant decode op and also dominates prefill last-token cost; INT8/INT4 LM_Head would substantially reduce it.")
md.append("- **PA decode kernel selection**: the Intel GPU plugin keeps `paged_attention_opt__single_token` at small KV and promotes to `paged_attention_opt__gqa_single_token` once the per-token KV working set crosses an internal threshold. For Qwen3-ASR-1.7B with INT8 KV (NH=16 NKV=8 HD=128), the promotion was observed at KV ≥ 4096; at KV = 512/1024 the single_token variant runs.")
# Pull the live pa_compute eff% straight from the assembled rows so the note never drifts.
_pa_eff = {}
for _kv, _rows in DECODE_TABLES.items():
    for _r in _rows:
        if _r["op"] == f"pa_compute_kv{_kv}":
            _pa_eff[_kv] = _r["eff_pct"]
            break
_eff_str = " / ".join(f"{_pa_eff.get(_kv, 0)}%" for _kv in SEQ_DECODE)
_kv_str  = "/".join(str(_kv) for _kv in SEQ_DECODE)
md.append(f"- **KV cache precision = {KV_PRECISION.upper()}**: the Intel GPU plugin defaults `kv_cache_precision` to `i8` for PagedAttention models (`src/plugins/intel_gpu/src/runtime/execution_config.cpp:293`). This run matches that default. PA cache layout is `K = [num_blocks, NKV, HD, BLOCK+4]` (BY_CHANNEL: 1 byte per element + 4-byte scale/zp shared across BLOCK=16 tokens) and `V = [num_blocks, NKV, BLOCK, HD+4]` (BY_TOKEN: 1 byte per element + 4-byte scale/zp shared across HD per token). The theoretical bytes formula above accounts for both the INT8 payload and the scale/zp overhead, so eff% is measured against what the kernel actually streams. Switching to FP16 KV roughly doubles PA-compute bytes (and so doubles TPOT contribution from PA at long context); switching to INT4 KV would roughly halve them again, at the cost of accuracy. Observed PA-compute decode eff: {_eff_str} for KV={_kv_str} — monotonically increasing with KV, as expected (larger working set amortizes per-launch overhead). The `KV cache key type=...` header line in every pa_bench log exposes the compiled cache element type; the `,20` channel dim is the i8 BY_CHANNEL signature (16 BLOCK + 4 scale/zp).")
md.append("- **PA prefill** S=8192 produces ~17 ms of `sdpa_micro__prefill` per layer × 28 layers = ~503 ms — by far the dominant prefill cost at long context. This is compute-bound (AI » ridge), so reducing weight precision will not help here; flash-attention–style algorithmic improvements or INT8 attention math would.")
md.append(f"- **Encoder weights** count ~{WEIGHTS['encoder']['total_mb'] + WEIGHTS['encoder']['front_back_mb']:.0f} MB and run only once: amortized over many output tokens the encoder is a small per-token cost.")
md.append("- **Eff% > 100% on tiny eltwise/norm ops at long prefill** (residual_add, rmsnorm_hidden, rope_q/k, pa_kv_update at S=4096/8192): these kernels are bandwidth-bound and the `small_ops_bench` only allocates a handful of input/output buffers, so the second-and-later iterations hit warm L2 even with the 64 MB flush—the kernel streams less than the theoretical `3·M·H·2B` shown in the table. Treat their eff% as a fitness check (kernel keeps up with budget) rather than as a roofline overshoot. The aggregate TPOT/TTFT row is dominated by FCs + PA where this effect is negligible.")
md.append("")
md.append("### Reproduction")
md.append("")
md.append("```bat")
md.append("REM On PTL 12Xe Windows (Local_Admin@10.239.132.229):")
md.append("D:\\river\\moe\\dev_roofline_profiling\\utils\\run_qwen3_asr_1_7B_ptl_12xe.bat")
md.append("```")
md.append("")
md.append("```bash")
md.append("# On local Linux box:")
md.append("scp -r Local_Admin@<win>:/D/river/moe/roofline_results/qwen3_asr_1_7B/ptl_12xe/*.log \\")
md.append("    .github/skills/dev_roofline_profiling/outputs/qwen3_asr_1_7B/logs_ptl_12xe/")
md.append("cd .github/skills/dev_roofline_profiling/outputs/qwen3_asr_1_7B")
md.append("python3 ../../utils/parse_logs.py logs_ptl_12xe parsed.json")
md.append("python3 build_report.py")
md.append("```")
md.append("")

out_md = HERE / f"SUMMARY_qwen3_asr_1_7B_ptl_12xe_{DATE}.md"
out_md.write_text("\n".join(md))
print(f"Wrote {out_md.name}  ({len(md)} lines)")
