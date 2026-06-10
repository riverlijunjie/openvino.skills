#!/usr/bin/env python3
"""Qwen3-Omni-4B roofline report — PTL 12Xe.

Computes:
  - Theoretical weight distribution (INT4 g128 matmul + INT8 g128 LM_head + FP16 norms/scales).
  - Theoretical per-op roofline (FLOPs, bytes, lower-bound latency at HW peaks).
  - Measured per-op latency from cliloader logs (parsed.json).
  - eff% (achieved / hardware-peak) for each op, per phase, per token size.

Outputs:
  - weight_distribution.json
  - performance_metrics.json
  - SUMMARY_qwen3_omni_4B_ptl_12xe_<date>.md

Run:
  python3 build_report.py

If parsed.json is missing, the script still emits the theoretical sections of
the SUMMARY (weight distribution + per-op theoretical roofline) so the user
can review what is going to be measured before launching the remote sweep.
"""

import json
import math
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent

# =========================================================================
# Hardware: PTL 12Xe (Intel iGPU)
# =========================================================================
HW = dict(
    name="PTL 12Xe",
    desc="Intel PTL 12Xe iGPU (12 Xe cores × 8 EU × 10 threads), 2400 MHz, 110 GB/s",
    xe_cores=12,
    eus_per_core=8,
    threads_per_eu=10,
    freq_ghz=2.400,
    bw_gbs=110.0,
)
# SKILL.md formulas
FP16_XMX_TFLOPS = HW["xe_cores"] * HW["eus_per_core"] * 256 * HW["freq_ghz"] / 1000.0  # 58.9824
INT8_XMX_TOPS   = 2.0 * FP16_XMX_TFLOPS                                                 # 117.9648
SIMD_FP16_TFLOPS = HW["xe_cores"] * HW["eus_per_core"] * 32 * HW["freq_ghz"] / 1000.0   #  7.3728
BW = HW["bw_gbs"]
RIDGE_FP16 = FP16_XMX_TFLOPS * 1e3 / BW                                                 # ~536 FLOP/B

# =========================================================================
# Model architecture (parsed from config.json)
# =========================================================================
THINKER = dict(
    hidden=2560, layers=36,
    NH=32, NKV=8, HD=128,
    intermediate=9728, vocab=151936,
    tie_embed=True, hidden_act="silu",
    rope_theta=1_000_000,
)
AUDIO = dict(
    d_model=1280, layers=32, NH=20, HD=64, ffn=5120,
    num_mel=128, S=1500, downsample_hidden=480, output_dim=2560,
)
VISION = dict(
    hidden=1152, depth=27, NH=16, HD=72, ffn=4304,
    patch=16, image=768, S_patches=2304, vis_tokens=576,
    spatial_merge=2, out_hidden=2560, deepstack_idx=[8, 16, 24],
)
TALKER = dict(
    hidden=2560, layers=28, NH=16, NKV=8, HD=128,
    intermediate=3072, vocab=3072,
    accept_hidden_layer=14,
)
CODE_PRED = dict(
    hidden=1024, layers=5, NH=16, NKV=8, HD=128,
    intermediate=3072, vocab=2048, num_code_groups=16,
)
CODE2WAV = dict(
    hidden=1024, layers=8, NH=16, HD=64, ffn=3072,
    num_quantizers=16, codebook_size=2048,
    decoder_dim=1536,
    upsample_rates=[8, 5, 4, 3],
    upsampling_ratios=[2, 2],
)

SEQ_PREFILL = [512, 1024, 4096, 8192]
SEQ_DECODE  = [512, 1024, 4096, 8192]
OUTPUT_TOKENS = 512

B_F16 = 2
KV_PRECISION = "i8"      # PA OCL default for PA models on Intel GPU
PA_BLOCK_SIZE = 16
GROUP_SIZE = 128

# =========================================================================
# Load parsed logs (optional)
# =========================================================================
PARSED_PATH = HERE / "parsed.json"
parsed = json.loads(PARSED_PATH.read_text()) if PARSED_PATH.exists() else {}

def kernel_total_ns(stem):
    r = parsed.get(stem)
    return r["total_kernel_ns"] if r else None

def kernel_subns(stem, contains, exclude=None):
    r = parsed.get(stem)
    if not r:
        return None
    for name, ns, _ in r["per_kernel"]:
        if contains in name:
            if exclude and exclude in name:
                continue
            return ns
    return None

# =========================================================================
# Theoretical bytes / FLOPs
# =========================================================================
def _kv_logical_bytes_per_token(NKV, HD):
    """Per-token KV cache footprint (K + V over all NKV heads).
       For INT8 PA OCL: K = [num_blocks,NKV,HD,BLOCK+4]; V = [num_blocks,NKV,BLOCK,HD+4].
       Adds 4B (fp16 scale+zp) amortized."""
    if KV_PRECISION == "i8":
        k_per = NKV * HD * (1.0 + 4.0 / PA_BLOCK_SIZE)
        v_per = NKV * (HD + 4)
        return k_per + v_per
    return NKV * HD * B_F16 * 2

def fc_flops(M, K, N):
    return 2.0 * M * K * N

def fc_bytes_quant(M, K, N, w_bits):
    """Bytes streamed per launch for a quantized FC.
       weight = K*N*(bits/8); scale = (K/G)*N*2 fp16; zp = (K/G)*N*1 (INT8 zp);
       act = M*K*2 (fp16); out = M*N*2 (fp16)."""
    w = K * N * (w_bits / 8.0)
    s = (K // GROUP_SIZE) * N * 2.0     # fp16 scales
    z = (K // GROUP_SIZE) * N * 1.0     # int8 zero-points
    a = M * K * 2.0
    o = M * N * 2.0
    return w + s + z + a + o

def fc_bytes_fp16(M, K, N):
    return K * N * 2.0 + M * K * 2.0 + M * N * 2.0

def pa_decode_bytes(Skv, NH, NKV, HD):
    return Skv * _kv_logical_bytes_per_token(NKV, HD) + NH * HD * B_F16 * 2

def pa_decode_flops(Skv, NH, HD):
    return 4.0 * NH * HD * Skv  # QK + softmax(*) + AV ≈ 4·NH·HD·Skv

def pa_prefill_bytes(S, NH, NKV, HD):
    return S * (NH + 2 * NKV) * HD * B_F16 + S * NH * HD * B_F16

def pa_prefill_flops_causal(S, NH, HD):
    pairs = S * (S + 1) / 2.0
    return 4.0 * NH * HD * pairs

def sdpa_full_flops(S, NH, HD):
    return 4.0 * NH * HD * S * S

def rmsnorm_bytes(M, H):
    return M * H * B_F16 * 2 + H * B_F16

def rmsnorm_flops(M, H):
    # x² + sum + /n + sqrt(10) + rsqrt mul + norm mul + gamma mul
    return float(M * H * 6 + M * 10)

def rope_bytes(M, NH, HD):
    return M * NH * HD * B_F16 * 2 + 2 * M * HD * B_F16

def rope_flops(M, NH, HD):
    return float(M * NH * HD * 3)

def add_bytes(M, H):
    return 3 * M * H * B_F16

def add_flops(M, H):
    return float(M * H)

# =========================================================================
# Roofline classification & theoretical lower bound
# =========================================================================
def classify_and_eff(flops, bytes_, single_ms):
    """Returns (gflops, gbs, ai, bound, eff_pct)."""
    t_s = single_ms / 1000.0
    gflops = flops / t_s / 1e9 if t_s > 0 else 0
    gbs = bytes_ / t_s / 1e9 if t_s > 0 else 0
    ai = flops / bytes_ if bytes_ > 0 else 0
    bound = "compute" if ai > RIDGE_FP16 else "memory"
    if bound == "memory":
        eff = gbs / BW * 100
    else:
        eff = gflops / (FP16_XMX_TFLOPS * 1e3) * 100
    return gflops, gbs, ai, bound, eff

def theo_single_ms(flops, bytes_, bound):
    if bound == "memory":
        return bytes_ / BW / 1e9 * 1000
    return flops / (FP16_XMX_TFLOPS * 1e12) * 1000

def make_row(op, kernel, single_ms, calls, flops, bytes_, *, force_bound=None):
    """Build one report row. If single_ms is None (no measurement), the row
       still includes theoretical fields so the SUMMARY can show pure-theory."""
    if single_ms is None:
        # Pure theoretical row
        ai = flops / bytes_ if bytes_ > 0 else 0
        bound = force_bound or ("compute" if ai > RIDGE_FP16 else "memory")
        tms = theo_single_ms(flops, bytes_, bound)
        return dict(
            op=op, kernel=kernel, calls=calls,
            single_ms=None, theo_single_ms=round(tms, 6),
            total_ms=None, theo_total_ms=round(tms * calls, 4),
            slowdown=None,
            gflops=None, gbs=None, ai=round(ai, 3),
            bound=bound, eff_pct=None,
            flops=float(flops), bytes=float(bytes_),
        )
    gflops, gbs, ai, bound, eff = classify_and_eff(flops, bytes_, single_ms)
    if force_bound:
        bound = force_bound
        if bound == "memory":
            eff = gbs / BW * 100
        else:
            eff = gflops / (FP16_XMX_TFLOPS * 1e3) * 100
    tms = theo_single_ms(flops, bytes_, bound)
    return dict(
        op=op, kernel=kernel, calls=calls,
        single_ms=round(single_ms, 6),
        theo_single_ms=round(tms, 6),
        total_ms=round(single_ms * calls, 4),
        theo_total_ms=round(tms * calls, 4),
        slowdown=round(single_ms / tms, 2) if tms > 0 else 0,
        gflops=round(gflops, 2), gbs=round(gbs, 2),
        ai=round(ai, 3), bound=bound, eff_pct=round(eff, 1),
        flops=float(flops), bytes=float(bytes_),
    )

# =========================================================================
# Thinker text decoder rows
# =========================================================================
TH_FCS_DECODE = [
    ("fc_qkv",  2560, 6144),   # 2560 -> Q_dim(4096)+K_dim(1024)+V_dim(1024)=6144
    ("fc_o",    4096, 2560),
    ("fc_gate", 2560, 9728),
    ("fc_up",   2560, 9728),
    ("fc_down", 9728, 2560),
]

def th_fc_row(op, suffix, M, K, N, w_bits, force_bound=None):
    log = f"fc_{'decode' if M == 1 else 'prefill'}_{suffix}"
    if M > 1:
        log += f"_S{M}"
    ns = kernel_total_ns(log)
    single_ms = ns / 1e6 if ns else None
    flops = fc_flops(M, K, N)
    bytes_ = fc_bytes_quant(M, K, N, w_bits)
    return make_row(op, "fc_int4_g128" if w_bits == 4 else "fc_int8_g128",
                    single_ms, THINKER["layers"] if op != "fc_lm_head" else 1,
                    flops, bytes_, force_bound=force_bound)

def th_pa_decode_rows(KV):
    """Returns up to 3 rows: pa_kv_update, pa_compute, pa_finalize."""
    log = f"pa_decode_kv{KV}"
    if log not in parsed:
        # theoretical only
        flops = pa_decode_flops(KV, THINKER["NH"], THINKER["HD"])
        bytes_ = pa_decode_bytes(KV, THINKER["NH"], THINKER["NKV"], THINKER["HD"])
        kv_bytes = THINKER["NKV"] * THINKER["HD"] * B_F16 * 2 + _kv_logical_bytes_per_token(THINKER["NKV"], THINKER["HD"])
        return [
            make_row(f"pa_kv_update_kv{KV}", "pa_kv_cache_update_ref", None, THINKER["layers"], 0.0, kv_bytes, force_bound="memory"),
            make_row(f"pa_compute_kv{KV}",   "paged_attention_opt__single_token", None, THINKER["layers"], flops, bytes_),
        ]
    rows = []
    kv_ns = kernel_subns(log, "pa_kv_cache_update")
    pa_ns = 0; pa_kernel_name = "paged_attention_opt__single_token"; fin_ns = 0
    import re as _re
    for name, ns, _ in parsed[log]["per_kernel"]:
        if "paged_attention_opt__" in name and "finalization" not in name:
            pa_ns = ns
            pa_kernel_name = _re.sub(r"_\d+(__sa)?$", "", name)
        elif "single_token_finalization" in name:
            fin_ns = ns
    if kv_ns:
        kv_bytes = THINKER["NKV"] * THINKER["HD"] * B_F16 * 2 + _kv_logical_bytes_per_token(THINKER["NKV"], THINKER["HD"])
        rows.append(make_row(f"pa_kv_update_kv{KV}", "pa_kv_cache_update_ref",
                             kv_ns / 1e6, THINKER["layers"], 0.0, kv_bytes, force_bound="memory"))
    if pa_ns:
        flops = pa_decode_flops(KV, THINKER["NH"], THINKER["HD"])
        bytes_ = pa_decode_bytes(KV, THINKER["NH"], THINKER["NKV"], THINKER["HD"])
        rows.append(make_row(f"pa_compute_kv{KV}", pa_kernel_name,
                             pa_ns / 1e6, THINKER["layers"], flops, bytes_))
    if fin_ns:
        fb = THINKER["NH"] * THINKER["HD"] * B_F16 * 2
        rows.append(make_row(f"pa_finalize_kv{KV}", "single_token_finalization",
                             fin_ns / 1e6, THINKER["layers"], 0.0, fb, force_bound="memory"))
    return rows

def th_pa_prefill_rows(S):
    log = f"pa_prefill_S{S}"
    flops = pa_prefill_flops_causal(S, THINKER["NH"], THINKER["HD"])
    bytes_ = pa_prefill_bytes(S, THINKER["NH"], THINKER["NKV"], THINKER["HD"])
    kv_bytes = S * THINKER["NKV"] * THINKER["HD"] * B_F16 * 2 + S * _kv_logical_bytes_per_token(THINKER["NKV"], THINKER["HD"])
    if log not in parsed:
        return [
            make_row(f"pa_kv_update_S{S}", "pa_kv_cache_update_ref", None, THINKER["layers"], 0.0, kv_bytes, force_bound="memory"),
            make_row(f"pa_compute_S{S}",   "sdpa_micro__prefill",    None, THINKER["layers"], flops, bytes_),
        ]
    rows = []
    kv_ns = 0; sdpa_ns = 0
    for name, ns, _ in parsed[log]["per_kernel"]:
        if "pa_kv_cache_update" in name:
            kv_ns = ns
        elif "sdpa_micro__prefill" in name:
            sdpa_ns = ns
    if kv_ns:
        rows.append(make_row(f"pa_kv_update_S{S}", "pa_kv_cache_update_ref",
                             kv_ns / 1e6, THINKER["layers"], 0.0, kv_bytes, force_bound="memory"))
    if sdpa_ns:
        rows.append(make_row(f"pa_compute_S{S}", "sdpa_micro__prefill",
                             sdpa_ns / 1e6, THINKER["layers"], flops, bytes_))
    return rows

def th_small_row(op, kernel, log, M, kind, **kw):
    ns = kernel_total_ns(log)
    single_ms = ns / 1e6 if ns else None
    if kind == "rmsnorm":
        H = kw.get("H") or kw["NH"] * kw["HD"]
        bytes_ = rmsnorm_bytes(M, H); flops = rmsnorm_flops(M, H)
    elif kind == "rope":
        bytes_ = rope_bytes(M, kw["NH"], kw["HD"]); flops = rope_flops(M, kw["NH"], kw["HD"])
    elif kind == "add":
        bytes_ = add_bytes(M, kw["H"]); flops = add_flops(M, kw["H"])
    else:
        raise ValueError(kind)
    return make_row(op, kernel, single_ms, kw.get("calls", THINKER["layers"]),
                    flops, bytes_, force_bound="memory")

def assemble_thinker_decode(KV):
    rows = []
    for op, K, N in TH_FCS_DECODE:
        rows.append(th_fc_row(op, op.replace("fc_", ""), 1, K, N, w_bits=4))
    rows.append(th_fc_row("fc_lm_head", "lm_head", 1, THINKER["hidden"], THINKER["vocab"], w_bits=8))
    rows.extend(th_pa_decode_rows(KV))
    rows.append(th_small_row("rmsnorm_hidden", "rms_gpu_bfyx_opt", "small_decode_rmsnorm",
                             1, "rmsnorm", H=THINKER["hidden"], calls=2 * THINKER["layers"] + 1))
    rows.append(th_small_row("rmsnorm3d_q", "rms_gpu_bfyx_opt", "small_decode_rmsnorm3d_q",
                             1, "rmsnorm", NH=THINKER["NH"], HD=THINKER["HD"], calls=THINKER["layers"]))
    rows.append(th_small_row("rmsnorm3d_k", "rms_gpu_bfyx_opt", "small_decode_rmsnorm3d_k",
                             1, "rmsnorm", NH=THINKER["NKV"], HD=THINKER["HD"], calls=THINKER["layers"]))
    rows.append(th_small_row("rope_q", "rope_opt", "small_decode_rope_q",
                             1, "rope", NH=THINKER["NH"], HD=THINKER["HD"], calls=THINKER["layers"]))
    rows.append(th_small_row("rope_k", "rope_opt", "small_decode_rope_k",
                             1, "rope", NH=THINKER["NKV"], HD=THINKER["HD"], calls=THINKER["layers"]))
    rows.append(th_small_row("residual_add", "eltwise", "small_decode_add",
                             1, "add", H=THINKER["hidden"], calls=2 * THINKER["layers"]))
    return rows

def assemble_thinker_prefill(S):
    rows = []
    for op, K, N in TH_FCS_DECODE:
        rows.append(th_fc_row(op, op.replace("fc_", ""), S, K, N, w_bits=4))
    # LM_head runs once at end of prefill (single row)
    rows.append(th_fc_row("fc_lm_head", "lm_head", 1, THINKER["hidden"], THINKER["vocab"], w_bits=8))
    rows.extend(th_pa_prefill_rows(S))
    rows.append(th_small_row("rmsnorm_hidden", "rms_gpu_bfyx_opt", f"small_prefill_rmsnorm_S{S}",
                             S, "rmsnorm", H=THINKER["hidden"], calls=2 * THINKER["layers"] + 1))
    rows.append(th_small_row("rmsnorm3d_q", "rms_gpu_bfyx_opt", f"small_prefill_rmsnorm3d_q_S{S}",
                             S, "rmsnorm", NH=THINKER["NH"], HD=THINKER["HD"], calls=THINKER["layers"]))
    rows.append(th_small_row("rmsnorm3d_k", "rms_gpu_bfyx_opt", f"small_prefill_rmsnorm3d_k_S{S}",
                             S, "rmsnorm", NH=THINKER["NKV"], HD=THINKER["HD"], calls=THINKER["layers"]))
    rows.append(th_small_row("rope_q", "rope_opt", f"small_prefill_rope_q_S{S}",
                             S, "rope", NH=THINKER["NH"], HD=THINKER["HD"], calls=THINKER["layers"]))
    rows.append(th_small_row("rope_k", "rope_opt", f"small_prefill_rope_k_S{S}",
                             S, "rope", NH=THINKER["NKV"], HD=THINKER["HD"], calls=THINKER["layers"]))
    rows.append(th_small_row("residual_add", "eltwise", f"small_prefill_add_S{S}",
                             S, "add", H=THINKER["hidden"], calls=2 * THINKER["layers"]))
    return rows

# =========================================================================
# Audio encoder rows (S=1500 fixed)
# =========================================================================
def audio_encoder_rows():
    rows = []
    L = AUDIO["layers"]; d = AUDIO["d_model"]; ff = AUDIO["ffn"]; S = AUDIO["S"]
    fc_defs = [
        ("enc_fc_qkv", "fc_enc_qkv_S1500",   d, 3 * d, L),
        ("enc_fc_o",   "fc_enc_o_S1500",     d, d,     L),
        ("enc_fc_fc1", "fc_enc_fc1_S1500",   d, ff,    L),
        ("enc_fc_fc2", "fc_enc_fc2_S1500",   ff, d,    L),
        ("enc_outproj","fc_enc_outproj_S1500", d, AUDIO["output_dim"], 1),
    ]
    for op, stem, K, N, calls in fc_defs:
        ns = kernel_total_ns(stem)
        flops = fc_flops(S, K, N)
        bytes_ = fc_bytes_quant(S, K, N, w_bits=4)
        rows.append(make_row(op, "fc_int4_g128", ns / 1e6 if ns else None,
                             calls, flops, bytes_))
    # SDPA: bidirectional. Bench via PA prefill (causal) × 2.
    ns = None; sdpa_ns = 0
    if "pa_prefill_encS1500" in parsed:
        for name, n, _ in parsed["pa_prefill_encS1500"]["per_kernel"]:
            if "sdpa_micro__prefill" in name:
                sdpa_ns = n
    NH = AUDIO["NH"]; HD = AUDIO["HD"]
    flops_full = sdpa_full_flops(S, NH, HD)
    bytes_ = S * 3 * NH * HD * B_F16 + S * NH * HD * B_F16
    single_ms = (sdpa_ns / 1e6) * 2 if sdpa_ns else None
    rows.append(make_row("enc_sdpa", "sdpa_micro__prefill (×2 bidir est.)",
                         single_ms, L, flops_full, bytes_))
    return rows

# =========================================================================
# Vision encoder rows (S=2304 fixed, image 768×768)
# =========================================================================
def vision_encoder_rows():
    rows = []
    L = VISION["depth"]; d = VISION["hidden"]; ff = VISION["ffn"]
    S = VISION["S_patches"]
    fc_defs = [
        ("vis_fc_qkv", "fc_vis_qkv_S2304", d, 3 * d, L),
        ("vis_fc_o",   "fc_vis_o_S2304",   d, d,     L),
        ("vis_fc_fc1", "fc_vis_fc1_S2304", d, ff,    L),
        ("vis_fc_fc2", "fc_vis_fc2_S2304", ff, d,    L),
        ("vis_merge_proj", "fc_vis_merge_S576",
         d * (VISION["spatial_merge"] ** 2), VISION["out_hidden"], 1),
    ]
    for op, stem, K, N, calls in fc_defs:
        ns = kernel_total_ns(stem)
        # merge proj M=576 (post-merge); others M=2304
        M = VISION["vis_tokens"] if op == "vis_merge_proj" else S
        flops = fc_flops(M, K, N)
        bytes_ = fc_bytes_quant(M, K, N, w_bits=4)
        rows.append(make_row(op, "fc_int4_g128", ns / 1e6 if ns else None,
                             calls, flops, bytes_))
    # SDPA bidirectional, HD=72 → padded to 80 by sdpa_micro in some plugin paths.
    sdpa_ns = 0
    if "pa_prefill_visS2304" in parsed:
        for name, n, _ in parsed["pa_prefill_visS2304"]["per_kernel"]:
            if "sdpa_micro__prefill" in name:
                sdpa_ns = n
    NH = VISION["NH"]; HD = VISION["HD"]
    flops_full = sdpa_full_flops(S, NH, HD)
    bytes_ = S * 3 * NH * HD * B_F16 + S * NH * HD * B_F16
    single_ms = (sdpa_ns / 1e6) * 2 if sdpa_ns else None
    rows.append(make_row("vis_sdpa", "sdpa_micro__prefill (×2 bidir est.)",
                         single_ms, L, flops_full, bytes_))
    return rows

# =========================================================================
# Talker decode rows (per output text token, if audio out enabled)
# =========================================================================
def talker_decode_rows(KV):
    rows = []
    L = TALKER["layers"]
    NH = TALKER["NH"]; NKV = TALKER["NKV"]; HD = TALKER["HD"]
    Q_dim = NH * HD          # 2048
    KV_dim = NKV * HD        # 1024
    H = TALKER["hidden"]; FF = TALKER["intermediate"]
    fc_defs = [
        ("talker_fc_qkv",  "fc_talker_decode_qkv",  H, Q_dim + 2 * KV_dim, L),
        ("talker_fc_o",    "fc_talker_decode_o",    Q_dim, H,              L),
        ("talker_fc_gate", "fc_talker_decode_gate", H, FF,                 L),
        ("talker_fc_up",   "fc_talker_decode_up",   H, FF,                 L),
        ("talker_fc_down", "fc_talker_decode_down", FF, H,                 L),
        ("talker_lm_head", "fc_talker_decode_lmhead", H, TALKER["vocab"],  1),
    ]
    for op, stem, K, N, calls in fc_defs:
        ns = kernel_total_ns(stem)
        flops = fc_flops(1, K, N)
        # talker_lm_head uses f16 (small vocab=3072); others int4 g128
        if op == "talker_lm_head":
            bytes_ = fc_bytes_fp16(1, K, N)
            kernel = "fc_f16"
        else:
            bytes_ = fc_bytes_quant(1, K, N, w_bits=4)
            kernel = "fc_int4_g128"
        rows.append(make_row(op, kernel, ns / 1e6 if ns else None, calls, flops, bytes_))
    # Talker PA decode (GQA 16:8)
    log = f"pa_talker_decode_kv{KV}"
    flops = pa_decode_flops(KV, NH, HD)
    bytes_ = pa_decode_bytes(KV, NH, NKV, HD)
    kv_bytes = NKV * HD * B_F16 * 2 + _kv_logical_bytes_per_token(NKV, HD)
    if log in parsed:
        kv_ns = kernel_subns(log, "pa_kv_cache_update")
        pa_ns = 0
        for name, ns, _ in parsed[log]["per_kernel"]:
            if "paged_attention_opt__" in name and "finalization" not in name:
                pa_ns = ns
        if kv_ns:
            rows.append(make_row(f"talker_pa_kv_update_kv{KV}", "pa_kv_cache_update_ref",
                                 kv_ns / 1e6, L, 0.0, kv_bytes, force_bound="memory"))
        if pa_ns:
            rows.append(make_row(f"talker_pa_compute_kv{KV}", "paged_attention_opt__single_token",
                                 pa_ns / 1e6, L, flops, bytes_))
    else:
        rows.append(make_row(f"talker_pa_kv_update_kv{KV}", "pa_kv_cache_update_ref",
                             None, L, 0.0, kv_bytes, force_bound="memory"))
        rows.append(make_row(f"talker_pa_compute_kv{KV}", "paged_attention_opt__single_token",
                             None, L, flops, bytes_))
    return rows

# =========================================================================
# Weight distribution
# =========================================================================
def _w_int4_g128(K, N):
    # int4 weight + fp16 scales + int8 zp
    return K * N * 0.5 + (K // GROUP_SIZE) * N * 2.0 + (K // GROUP_SIZE) * N * 1.0

def _w_int8_g128(K, N):
    return K * N * 1.0 + (K // GROUP_SIZE) * N * 2.0 + (K // GROUP_SIZE) * N * 1.0

def _w_fp16(K, N):
    return K * N * 2.0

def _mb(x):
    return round(x / (1024 * 1024), 3)

def weight_distribution():
    # ------------------- Thinker text decoder --------------------------
    H = THINKER["hidden"]; NH = THINKER["NH"]; NKV = THINKER["NKV"]; HD = THINKER["HD"]
    FF = THINKER["intermediate"]; V = THINKER["vocab"]; L = THINKER["layers"]
    QKV_N = NH * HD + 2 * NKV * HD  # 6144
    qkv  = _w_int4_g128(H, QKV_N)
    o    = _w_int4_g128(NH * HD, H)
    gate = _w_int4_g128(H, FF)
    up   = _w_int4_g128(H, FF)
    down = _w_int4_g128(FF, H)
    rmsnorm_pl = 2 * H * 2  # fp16
    qnorm = HD * 2; knorm = HD * 2
    thinker_pl = qkv + o + gate + up + down + rmsnorm_pl + qnorm + knorm
    thinker_layers = thinker_pl * L
    final_rms_t = H * 2
    # LM head INT8 g128, shared (tied) — counted once.
    embed = _w_int8_g128(V, H)
    lm_extra = 0 if THINKER["tie_embed"] else _w_int8_g128(V, H)
    thinker_total = thinker_layers + final_rms_t + embed + lm_extra

    # ------------------- Audio encoder --------------------------------
    d = AUDIO["d_model"]; ff_a = AUDIO["ffn"]; L_a = AUDIO["layers"]
    qkv_a = _w_int4_g128(d, 3 * d)
    o_a   = _w_int4_g128(d, d)
    fc1_a = _w_int4_g128(d, ff_a)
    fc2_a = _w_int4_g128(ff_a, d)
    ln_a  = 2 * 2 * d * 2  # 2 LN × (gamma + beta) × fp16
    audio_pl = qkv_a + o_a + fc1_a + fc2_a + ln_a
    audio_layers = audio_pl * L_a
    # conv front-end (mel→d via 2× conv1d k=3, stride1+stride2) + pos_embed + adapter
    conv1 = AUDIO["num_mel"] * d * 3 * 2     # fp16 conv
    conv2 = d * d * 3 * 2
    pos_embed_a = AUDIO["S"] * d * 2
    final_ln_a = 2 * d * 2
    adapter_a = _w_int4_g128(d, AUDIO["downsample_hidden"]) + _w_int4_g128(AUDIO["downsample_hidden"], AUDIO["output_dim"])
    audio_misc = conv1 + conv2 + pos_embed_a + final_ln_a + adapter_a
    audio_total = audio_layers + audio_misc

    # ------------------- Vision encoder -------------------------------
    dv = VISION["hidden"]; ff_v = VISION["ffn"]; L_v = VISION["depth"]
    qkv_v = _w_int4_g128(dv, 3 * dv)
    o_v   = _w_int4_g128(dv, dv)
    fc1_v = _w_int4_g128(dv, ff_v)
    fc2_v = _w_int4_g128(ff_v, dv)
    ln_v  = 2 * 2 * dv * 2
    vis_pl = qkv_v + o_v + fc1_v + fc2_v + ln_v
    vis_layers = vis_pl * L_v
    # patch embed (conv2d 16x16, 3->1152) + abs_pos_embed (2304×1152) + merger
    patch_embed = 3 * dv * VISION["patch"] * VISION["patch"] * VISION["spatial_merge"] * 2  # incl temporal merge factor in conv weight
    pos_embed_v = VISION["S_patches"] * dv * 2
    merge_proj = _w_int4_g128(dv * VISION["spatial_merge"] ** 2, VISION["out_hidden"])
    # deepstack adapters: 3 small projections (1152 -> 2560)
    deepstack = 3 * _w_int4_g128(dv, VISION["out_hidden"])
    final_ln_v = 2 * dv * 2
    vis_misc = patch_embed + pos_embed_v + merge_proj + deepstack + final_ln_v
    vis_total = vis_layers + vis_misc

    # ------------------- Talker ---------------------------------------
    Ht = TALKER["hidden"]; NHt = TALKER["NH"]; NKVt = TALKER["NKV"]; HDt = TALKER["HD"]
    FFt = TALKER["intermediate"]; Vt = TALKER["vocab"]; Lt = TALKER["layers"]
    QKVt = NHt * HDt + 2 * NKVt * HDt
    qkv_t = _w_int4_g128(Ht, QKVt)
    o_t   = _w_int4_g128(NHt * HDt, Ht)
    g_t   = _w_int4_g128(Ht, FFt)
    u_t   = _w_int4_g128(Ht, FFt)
    d_t   = _w_int4_g128(FFt, Ht)
    rms_t = 2 * Ht * 2
    talker_pl = qkv_t + o_t + g_t + u_t + d_t + rms_t + HDt * 2 + HDt * 2
    talker_layers_b = talker_pl * Lt
    talker_lmhead = _w_fp16(Ht, Vt)  # tiny vocab
    talker_embed = _w_fp16(Vt, Ht)
    talker_total = talker_layers_b + talker_lmhead + talker_embed + 2 * Ht  # final norm

    # ------------------- Talker code predictor -------------------------
    Hcp = CODE_PRED["hidden"]; NHcp = CODE_PRED["NH"]; NKVcp = CODE_PRED["NKV"]; HDcp = CODE_PRED["HD"]
    FFcp = CODE_PRED["intermediate"]; Vcp = CODE_PRED["vocab"]; Lcp = CODE_PRED["layers"]
    cp_pl = _w_int4_g128(Hcp, NHcp * HDcp + 2 * NKVcp * HDcp) \
          + _w_int4_g128(NHcp * HDcp, Hcp) \
          + _w_int4_g128(Hcp, FFcp) + _w_int4_g128(Hcp, FFcp) + _w_int4_g128(FFcp, Hcp) \
          + 2 * Hcp * 2
    cp_total = cp_pl * Lcp + _w_fp16(Hcp, Vcp) * CODE_PRED["num_code_groups"]

    # ------------------- Code2wav -------------------------------------
    Hc = CODE2WAV["hidden"]; NHc = CODE2WAV["NH"]; HDc = CODE2WAV["HD"]
    FFc = CODE2WAV["ffn"]; Lc = CODE2WAV["layers"]
    c2w_pl = _w_int4_g128(Hc, 3 * NHc * HDc) \
           + _w_int4_g128(NHc * HDc, Hc) \
           + _w_int4_g128(Hc, FFc) + _w_int4_g128(FFc, Hc) \
           + 2 * 2 * Hc * 2
    c2w_layers = c2w_pl * Lc
    # codebooks: Q × codebook_size × decoder_dim
    codebooks = CODE2WAV["num_quantizers"] * CODE2WAV["codebook_size"] * CODE2WAV["decoder_dim"] * 2  # fp16
    upsample = sum(CODE2WAV["upsample_rates"]) * CODE2WAV["decoder_dim"] * CODE2WAV["decoder_dim"] * 2 // 10
    c2w_total = c2w_layers + codebooks + upsample

    return dict(
        thinker=dict(
            per_layer_mb=_mb(thinker_pl), layers=L,
            fc_qkv_mb=_mb(qkv), fc_o_mb=_mb(o),
            fc_gate_mb=_mb(gate), fc_up_mb=_mb(up), fc_down_mb=_mb(down),
            rmsnorm_mb=_mb(rmsnorm_pl), qknorm_mb=_mb(qnorm + knorm),
            embed_mb=_mb(embed), tie_embed=THINKER["tie_embed"],
            lm_extra_mb=_mb(lm_extra), final_rms_mb=_mb(final_rms_t),
            total_mb=_mb(thinker_total),
        ),
        audio=dict(
            per_layer_mb=_mb(audio_pl), layers=L_a,
            fc_qkv_mb=_mb(qkv_a), fc_o_mb=_mb(o_a),
            fc_fc1_mb=_mb(fc1_a), fc_fc2_mb=_mb(fc2_a),
            ln_mb=_mb(ln_a),
            conv_front_mb=_mb(conv1 + conv2), pos_embed_mb=_mb(pos_embed_a),
            adapter_mb=_mb(adapter_a), misc_mb=_mb(audio_misc),
            total_mb=_mb(audio_total),
        ),
        vision=dict(
            per_layer_mb=_mb(vis_pl), depth=L_v,
            fc_qkv_mb=_mb(qkv_v), fc_o_mb=_mb(o_v),
            fc_fc1_mb=_mb(fc1_v), fc_fc2_mb=_mb(fc2_v),
            ln_mb=_mb(ln_v),
            patch_embed_mb=_mb(patch_embed), pos_embed_mb=_mb(pos_embed_v),
            merge_proj_mb=_mb(merge_proj), deepstack_mb=_mb(deepstack),
            misc_mb=_mb(vis_misc),
            total_mb=_mb(vis_total),
        ),
        talker=dict(
            per_layer_mb=_mb(talker_pl), layers=Lt,
            fc_qkv_mb=_mb(qkv_t), fc_o_mb=_mb(o_t),
            fc_gate_mb=_mb(g_t), fc_up_mb=_mb(u_t), fc_down_mb=_mb(d_t),
            embed_mb=_mb(talker_embed), lmhead_mb=_mb(talker_lmhead),
            total_mb=_mb(talker_total),
        ),
        code_predictor=dict(layers=Lcp, total_mb=_mb(cp_total)),
        code2wav=dict(
            per_layer_mb=_mb(c2w_pl), layers=Lc,
            codebooks_mb=_mb(codebooks), upsample_mb=_mb(upsample),
            total_mb=_mb(c2w_total),
        ),
        grand_total_mb=_mb(thinker_total + audio_total + vis_total +
                           talker_total + cp_total + c2w_total),
    )

# =========================================================================
# Assemble
# =========================================================================
WEIGHTS = weight_distribution()
ENC_AUDIO = audio_encoder_rows()
ENC_VIS = vision_encoder_rows()
DECODE_TABLES = {kv: assemble_thinker_decode(kv) for kv in SEQ_DECODE}
PREFILL_TABLES = {s: assemble_thinker_prefill(s) for s in SEQ_PREFILL}
TALKER_DECODE_TABLES = {kv: talker_decode_rows(kv) for kv in SEQ_DECODE}

def _sum_total(rows, theo=False):
    if theo:
        return round(sum((r["theo_total_ms"] or 0) for r in rows), 4)
    # Fall back to theoretical for rows missing measurements so aggregates
    # don't silently undercount unmeasured ops (e.g. fc_bench failures).
    return round(sum((r["total_ms"] if r["total_ms"] is not None else (r["theo_total_ms"] or 0)) for r in rows), 4)

ENC_AUDIO_TOT      = _sum_total(ENC_AUDIO)
ENC_AUDIO_THEO_TOT = _sum_total(ENC_AUDIO, theo=True)
ENC_VIS_TOT        = _sum_total(ENC_VIS)
ENC_VIS_THEO_TOT   = _sum_total(ENC_VIS, theo=True)
DECODE_TOT      = {kv: _sum_total(rows)            for kv, rows in DECODE_TABLES.items()}
DECODE_THEO_TOT = {kv: _sum_total(rows, theo=True) for kv, rows in DECODE_TABLES.items()}
PREFILL_TXT_TOT      = {s: _sum_total(rows)            for s, rows in PREFILL_TABLES.items()}
PREFILL_TXT_THEO_TOT = {s: _sum_total(rows, theo=True) for s, rows in PREFILL_TABLES.items()}
TALKER_DECODE_TOT      = {kv: _sum_total(rows)            for kv, rows in TALKER_DECODE_TABLES.items()}
TALKER_DECODE_THEO_TOT = {kv: _sum_total(rows, theo=True) for kv, rows in TALKER_DECODE_TABLES.items()}

# TTFT = thinker prefill + audio enc + vision enc (both run once if multimodal input)
def ttft(s, modal):
    base = PREFILL_TXT_TOT[s]
    if modal in ("audio", "av"):
        base = base + ENC_AUDIO_TOT if ENC_AUDIO_TOT else base
    if modal in ("vision", "av"):
        base = base + ENC_VIS_TOT if ENC_VIS_TOT else base
    return round(base, 4)

# =========================================================================
# Persist JSON
# =========================================================================
(HERE / "weight_distribution.json").write_text(json.dumps(WEIGHTS, indent=2))
print(f"Wrote weight_distribution.json")

metrics = dict(
    platform=HW["name"], platform_desc=HW["desc"],
    bw_gbs=BW, fp16_xmx_tflops=round(FP16_XMX_TFLOPS, 4),
    int8_xmx_tops=round(INT8_XMX_TOPS, 4),
    simd_fp16_tflops=round(SIMD_FP16_TFLOPS, 4),
    ridge_fp16_flop_per_byte=round(RIDGE_FP16, 2),
    model="Qwen3-Omni-4B-Instruct-multilingual-int4",
    config_summary=(
        f"INT4 g128 matmul + INT8 g128 LM_head + {'INT8' if KV_PRECISION=='i8' else 'FP16'} KV cache, "
        "PA opencl+micro_kernel; multimodal: Thinker text decoder (36-layer dense GQA), "
        "Audio encoder (32-layer MHA), Vision encoder (27-layer SigLIP MHA), Talker (28-layer dense GQA) + code2wav."
    ),
    arch=dict(thinker=THINKER, audio=AUDIO, vision=VISION, talker=TALKER,
              code_pred=CODE_PRED, code2wav=CODE2WAV),
    output_tokens=OUTPUT_TOKENS,
    enc_audio=dict(total_ms=ENC_AUDIO_TOT, theo_total_ms=ENC_AUDIO_THEO_TOT, rows=ENC_AUDIO),
    enc_vision=dict(total_ms=ENC_VIS_TOT, theo_total_ms=ENC_VIS_THEO_TOT, rows=ENC_VIS),
    decode={str(kv): dict(rows=rows,
                          total_ms=DECODE_TOT[kv],
                          theo_total_ms=DECODE_THEO_TOT[kv],
                          slowdown=round(DECODE_TOT[kv] / DECODE_THEO_TOT[kv], 3) if DECODE_THEO_TOT[kv] else None)
            for kv, rows in DECODE_TABLES.items()},
    prefill={str(s): dict(rows=rows,
                          text_decoder_ms=PREFILL_TXT_TOT[s],
                          text_decoder_theo_ms=PREFILL_TXT_THEO_TOT[s],
                          enc_audio_ms=ENC_AUDIO_TOT, enc_vision_ms=ENC_VIS_TOT,
                          ttft_text_only_ms=ttft(s, "text"),
                          ttft_audio_in_ms=ttft(s, "audio"),
                          ttft_av_in_ms=ttft(s, "av"))
             for s, rows in PREFILL_TABLES.items()},
    talker_decode={str(kv): dict(rows=rows,
                                 total_ms=TALKER_DECODE_TOT[kv],
                                 theo_total_ms=TALKER_DECODE_THEO_TOT[kv])
                   for kv, rows in TALKER_DECODE_TABLES.items()},
    weights=WEIGHTS,
)
(HERE / "performance_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Wrote performance_metrics.json")

# =========================================================================
# Markdown SUMMARY
# =========================================================================
def _fmt(x, prec=4, dash="—"):
    if x is None:
        return dash
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)

def _slow(meas, theo):
    if meas is None or theo is None or theo <= 0:
        return "—"
    return f"{meas / theo:.2f}×"

def fmt_row(r):
    return (f"| {r['op']} | `{r['kernel']}` | {r['calls']} | "
            f"{_fmt(r['single_ms'])} | {_fmt(r['theo_single_ms'])} | "
            f"{_fmt(r['total_ms'])} | {_fmt(r['theo_total_ms'])} | "
            f"{_slow(r['single_ms'], r['theo_single_ms'])} | "
            f"{_fmt(r['gflops'], 1)} | {_fmt(r['gbs'], 1)} | "
            f"{_fmt(r['eff_pct'], 1) if r['eff_pct'] is not None else '—'}{'' if r['eff_pct'] is None else '%'} | {r['bound']} |")

def table_block(title, rows, total_label, total_ms, theo_total_ms):
    rows_sorted = sorted(rows, key=lambda r: -(r["total_ms"] or r["theo_total_ms"] or 0))
    s = [f"### {title}", "",
         "| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows_sorted:
        s.append(fmt_row(r))
    s.append(f"| **{total_label}** |  |  |  |  | **{_fmt(total_ms)}** | **{_fmt(theo_total_ms)}** | **{_slow(total_ms, theo_total_ms)}** |  |  |  |  |")
    s.append("")
    return "\n".join(s)

DATE = date.today().isoformat()

md = []
md.append(f"# Qwen3-Omni-4B — Roofline on {HW['name']} ({DATE})")
md.append("")
md.append(f"**Platform**: {HW['desc']}")
md.append(f"**Model**: Qwen3-Omni-4B-Instruct-multilingual-int4 (Thinker text + Audio encoder + Vision encoder + Talker + code2wav).")
md.append(f"**Quantization**: matmul INT4 g128 / LM_head INT8 g128 / KV cache INT8 / activations FP16. PA opencl + micro_kernel.")
md.append(f"**Source config**: `~/workspace/remote_debug/qwen3_omni/models/Qwen3-Omni-4B-Instruct-multilingual-int4/config.json`")
md.append("")
md.append(f"**Inputs evaluated**: thinker-text prefill context = 512 / 1024 / 4096 / 8192 tokens; output = {OUTPUT_TOKENS} tokens. Audio encoder runs once over S=1500 mel frames. Vision encoder runs once over S=2304 patches (768×768 image).")
md.append("")

# ---------- 1. Hardware peaks ----------
md.append("## 1. Hardware peaks (per SKILL.md formulas)")
md.append("")
md.append("`FP16 XMX TFLOPS = xe_cores × 8 × 256 × freq_GHz`; `INT8 XMX = 2× FP16`; `SIMD FP16 = xe_cores × 8 × 32 × freq`.")
md.append("")
md.append("| Platform | Xe Cores | Freq (MHz) | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | SIMD FP16 (TFLOPS) | Ridge FP16 (FLOP/B) |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
md.append(f"| {HW['name']} | {HW['xe_cores']} | {int(HW['freq_ghz']*1000)} | {BW:.0f} | {FP16_XMX_TFLOPS:.3f} | {INT8_XMX_TOPS:.3f} | {SIMD_FP16_TFLOPS:.3f} | {RIDGE_FP16:.0f} |")
md.append("")

# ---------- 2. Architecture ----------
md.append("## 2. Model architecture")
md.append("")
md.append("### Thinker text decoder (Qwen3VL-style dense, 36 layers, GQA 32:8)")
md.append("")
md.append("| Field | Value |")
md.append("|---|---:|")
for k, v in [("hidden_size", THINKER["hidden"]),
             ("num_hidden_layers", THINKER["layers"]),
             ("num_attention_heads (NH)", THINKER["NH"]),
             ("num_key_value_heads (NKV)", f"{THINKER['NKV']} (GQA 4:1)"),
             ("head_dim (HD)", THINKER["HD"]),
             ("Q_dim / KV_dim", f"{THINKER['NH']*THINKER['HD']} / {THINKER['NKV']*THINKER['HD']}"),
             ("intermediate_size", THINKER["intermediate"]),
             ("vocab_size", THINKER["vocab"]),
             ("tie_word_embeddings", THINKER["tie_embed"]),
             ("hidden_act", THINKER["hidden_act"]),
             ("rope_theta", f"{THINKER['rope_theta']:,}")]:
    md.append(f"| `{k}` | {v} |")
md.append("")
md.append("### Audio encoder (Whisper-style MHA, 32 layers, bidirectional)")
md.append("")
md.append("| Field | Value |")
md.append("|---|---:|")
for k, v in [("d_model", AUDIO["d_model"]),
             ("encoder_layers", AUDIO["layers"]),
             ("encoder_attention_heads (NH=NKV)", AUDIO["NH"]),
             ("head_dim", AUDIO["HD"]),
             ("encoder_ffn_dim", AUDIO["ffn"]),
             ("num_mel_bins", AUDIO["num_mel"]),
             ("max_source_positions / S", AUDIO["S"]),
             ("output_dim (→ thinker hidden)", AUDIO["output_dim"])]:
    md.append(f"| `{k}` | {v} |")
md.append("")
md.append("### Vision encoder (SigLIP-style ViT, 27 layers, bidirectional)")
md.append("")
md.append("| Field | Value |")
md.append("|---|---:|")
for k, v in [("hidden_size", VISION["hidden"]),
             ("depth", VISION["depth"]),
             ("num_heads (NH=NKV)", VISION["NH"]),
             ("head_dim", f"{VISION['HD']} (= 1152/16)"),
             ("intermediate_size", VISION["ffn"]),
             ("patch_size", VISION["patch"]),
             ("image_size", VISION["image"]),
             ("S_patches (input tokens to ViT)", f"{VISION['S_patches']} (= 48×48)"),
             ("vis_tokens after spatial_merge=2", VISION["vis_tokens"]),
             ("out_hidden_size", VISION["out_hidden"]),
             ("deepstack_visual_indexes", VISION["deepstack_idx"])]:
    md.append(f"| `{k}` | {v} |")
md.append("")
md.append("### Talker (audio-output AR decoder, 28 layers, GQA 16:8)")
md.append("")
md.append("| Field | Value |")
md.append("|---|---:|")
for k, v in [("hidden_size", TALKER["hidden"]),
             ("num_hidden_layers", TALKER["layers"]),
             ("NH / NKV / HD", f"{TALKER['NH']} / {TALKER['NKV']} / {TALKER['HD']}"),
             ("intermediate_size", TALKER["intermediate"]),
             ("vocab_size (codec tokens)", TALKER["vocab"]),
             ("accept_hidden_layer", TALKER["accept_hidden_layer"])]:
    md.append(f"| `{k}` | {v} |")
md.append("")

# ---------- 3. Weight distribution ----------
md.append("## 3. Theoretical weight distribution (INT4 g128 matmul, INT8 g128 LM_head, FP16 norms)")
md.append("")
md.append("**Quantization sizing**:")
md.append("- INT4 g128 weight = K·N·0.5 bytes + (K/128)·N·2 fp16 scales + (K/128)·N·1 int8 zero-points")
md.append("- INT8 g128 weight = K·N·1 bytes + (K/128)·N·2 fp16 scales + (K/128)·N·1 int8 zero-points")
md.append("- FP16 = K·N·2 bytes (norms, embed/LM head when un-tied or for tiny vocabs like talker)")
md.append("")
md.append("### Thinker text decoder per-layer (1 of 36)")
md.append("")
md.append("| Weight | Shape (K × N) | Quant | MB |")
md.append("|---|---|---|---:|")
md.append(f"| FC_QKV (fused Q+K+V) | 2560 × 6144 | INT4 g128 | {WEIGHTS['thinker']['fc_qkv_mb']} |")
md.append(f"| FC_O (attn output) | 4096 × 2560 | INT4 g128 | {WEIGHTS['thinker']['fc_o_mb']} |")
md.append(f"| FC_Gate (SwiGLU) | 2560 × 9728 | INT4 g128 | {WEIGHTS['thinker']['fc_gate_mb']} |")
md.append(f"| FC_Up   (SwiGLU) | 2560 × 9728 | INT4 g128 | {WEIGHTS['thinker']['fc_up_mb']} |")
md.append(f"| FC_Down (SwiGLU) | 9728 × 2560 | INT4 g128 | {WEIGHTS['thinker']['fc_down_mb']} |")
md.append(f"| RMSNorm × 2 + q_norm + k_norm | small | FP16 | {WEIGHTS['thinker']['rmsnorm_mb'] + WEIGHTS['thinker']['qknorm_mb']:.4f} |")
md.append(f"| **per layer** |  |  | **{WEIGHTS['thinker']['per_layer_mb']:.2f}** |")
md.append(f"| **× 36 layers** |  |  | **{WEIGHTS['thinker']['per_layer_mb']*36:.2f}** |")
md.append("")
md.append("### Thinker global / shared")
md.append("")
md.append("| Weight | Shape | Quant | MB |")
md.append("|---|---|---|---:|")
md.append(f"| Token embedding (tied w/ LM head) | 151936 × 2560 | INT8 g128 | {WEIGHTS['thinker']['embed_mb']} |")
md.append(f"| LM_head extra (tied={WEIGHTS['thinker']['tie_embed']}) | — | — | {WEIGHTS['thinker']['lm_extra_mb']} |")
md.append(f"| Final RMSNorm | [2560] | FP16 | {WEIGHTS['thinker']['final_rms_mb']} |")
md.append(f"| **Thinker total** |  |  | **{WEIGHTS['thinker']['total_mb']:.2f}** |")
md.append("")
md.append("### Audio encoder per-layer (1 of 32) — INT4 g128")
md.append("")
md.append("| Weight | Shape (K × N) | Quant | MB |")
md.append("|---|---|---|---:|")
md.append(f"| FC_QKV (fused) | 1280 × 3840 | INT4 g128 | {WEIGHTS['audio']['fc_qkv_mb']} |")
md.append(f"| FC_O           | 1280 × 1280 | INT4 g128 | {WEIGHTS['audio']['fc_o_mb']} |")
md.append(f"| FC_FC1         | 1280 × 5120 | INT4 g128 | {WEIGHTS['audio']['fc_fc1_mb']} |")
md.append(f"| FC_FC2         | 5120 × 1280 | INT4 g128 | {WEIGHTS['audio']['fc_fc2_mb']} |")
md.append(f"| LayerNorm × 2  | [1280] g+b each | FP16 | {WEIGHTS['audio']['ln_mb']} |")
md.append(f"| **per layer** |  |  | **{WEIGHTS['audio']['per_layer_mb']:.3f}** |")
md.append(f"| **× 32 layers** |  |  | **{WEIGHTS['audio']['per_layer_mb']*32:.2f}** |")
md.append(f"| Conv front-end + pos_embed + adapter | misc | mixed | {WEIGHTS['audio']['misc_mb']:.3f} |")
md.append(f"| **Audio encoder total** |  |  | **{WEIGHTS['audio']['total_mb']:.2f}** |")
md.append("")
md.append("### Vision encoder per-layer (1 of 27) — INT4 g128")
md.append("")
md.append("| Weight | Shape (K × N) | Quant | MB |")
md.append("|---|---|---|---:|")
md.append(f"| FC_QKV (fused) | 1152 × 3456 | INT4 g128 | {WEIGHTS['vision']['fc_qkv_mb']} |")
md.append(f"| FC_O           | 1152 × 1152 | INT4 g128 | {WEIGHTS['vision']['fc_o_mb']} |")
md.append(f"| FC_FC1         | 1152 × 4304 | INT4 g128 | {WEIGHTS['vision']['fc_fc1_mb']} |")
md.append(f"| FC_FC2         | 4304 × 1152 | INT4 g128 | {WEIGHTS['vision']['fc_fc2_mb']} |")
md.append(f"| LayerNorm × 2  | [1152] g+b each | FP16 | {WEIGHTS['vision']['ln_mb']} |")
md.append(f"| **per layer** |  |  | **{WEIGHTS['vision']['per_layer_mb']:.3f}** |")
md.append(f"| **× 27 layers** |  |  | **{WEIGHTS['vision']['per_layer_mb']*27:.2f}** |")
md.append(f"| Patch embed + pos_embed + merger + deepstack | misc | mixed | {WEIGHTS['vision']['misc_mb']:.3f} |")
md.append(f"| **Vision encoder total** |  |  | **{WEIGHTS['vision']['total_mb']:.2f}** |")
md.append("")
md.append("### Talker (audio-output AR decoder) — INT4 g128 matmul, FP16 embed/lm_head (small vocab)")
md.append("")
md.append("| Weight | MB |")
md.append("|---|---:|")
md.append(f"| Per-layer FC body (× 28) | {WEIGHTS['talker']['per_layer_mb']:.3f} → {WEIGHTS['talker']['per_layer_mb']*28:.2f} |")
md.append(f"| Token embed (codec vocab=3072) | {WEIGHTS['talker']['embed_mb']} |")
md.append(f"| LM head (codec vocab=3072) | {WEIGHTS['talker']['lmhead_mb']} |")
md.append(f"| **Talker total** | **{WEIGHTS['talker']['total_mb']:.2f}** |")
md.append("")
md.append("### Code predictor + code2wav (audio-output supporting nets)")
md.append("")
md.append("| Component | MB |")
md.append("|---|---:|")
md.append(f"| Code predictor (5 layers, hidden=1024, × 16 code-group heads) | {WEIGHTS['code_predictor']['total_mb']:.2f} |")
md.append(f"| Code2wav (8 layers, hidden=1024, + 16 × 2048 codebooks dec_dim=1536) | {WEIGHTS['code2wav']['total_mb']:.2f} |")
md.append("")
md.append("### Grand total")
md.append("")
md.append("| Component | MB |")
md.append("|---|---:|")
md.append(f"| Thinker text decoder | {WEIGHTS['thinker']['total_mb']:.2f} |")
md.append(f"| Audio encoder | {WEIGHTS['audio']['total_mb']:.2f} |")
md.append(f"| Vision encoder | {WEIGHTS['vision']['total_mb']:.2f} |")
md.append(f"| Talker | {WEIGHTS['talker']['total_mb']:.2f} |")
md.append(f"| Code predictor | {WEIGHTS['code_predictor']['total_mb']:.2f} |")
md.append(f"| Code2wav | {WEIGHTS['code2wav']['total_mb']:.2f} |")
md.append(f"| **Grand total (on-disk INT4-quantized)** | **{WEIGHTS['grand_total_mb']:.2f}** |")
md.append("")

# ---------- 4. Methodology ----------
md.append("## 4. Benchmark methodology")
md.append("")
md.append("- **Bench utils**: `fc_bench` (precision=u4 default INT4 g128 / u8 for LM_head / f16 for talker lm_head), `pa_bench` (kv_dtype=i8, impl=ocl), `small_ops_bench`.")
md.append("- **Tool**: cliloader Device Performance Timing; `parse_logs.py` extracts per-iteration GPU kernel ns and aggregates per-iteration totals across split kernels (kv_update + pa_compute + finalization).")
md.append("- **L2/L3 flush** between every FC infer to force VRAM-resident weight reads (per-layer fits in L2, full model does not).")
md.append("- **Input/output tensors** allocated via RemoteContext in USM_DEVICE (iGPU shared system memory) to avoid PCIe transfer artifacts.")
md.append("- **PA prefill** uses causal mask (S·(S+1)/2 effective attention pairs, per SKILL.md).")
md.append("- **PA decode** is decomposed into `pa_kv_cache_update_ref` + `paged_attention_opt__single_token` (+ `single_token_finalization` for the GQA variant).")
md.append("- **Audio / vision encoder SDPA** is bidirectional; reported as `pa_bench prefill` (causal) × 2 (lower bound for the full S² attention).")
md.append("- **SwiGLU `multiply` and `swish`** are fused into the SwiGLU primitive in the real graph (per SKILL.md and glu_fusion); not profiled separately.")
md.append("- **Talker** is profiled at decode (M=1) only because in steady state it runs autoregressively per output text token.")
md.append("- **Code2wav** uses small windows and streaming flow-matching; theoretical-only in this report.")
md.append("")

# ---------- 5. Encoder fixed overhead ----------
md.append("## 5. Encoder fixed overhead (runs once per inference)")
md.append("")
md.append(table_block(f"Audio encoder — S={AUDIO['S']} (INT4 g128, bidirectional MHA)", ENC_AUDIO,
                      "TOTAL (audio encoder)", ENC_AUDIO_TOT, ENC_AUDIO_THEO_TOT))
md.append(table_block(f"Vision encoder — S={VISION['S_patches']} (INT4 g128, bidirectional MHA, 768×768 image)", ENC_VIS,
                      "TOTAL (vision encoder)", ENC_VIS_TOT, ENC_VIS_THEO_TOT))

# ---------- 6. Thinker prefill ----------
md.append("## 6. Thinker text decoder — Prefill")
md.append("")
md.append("Per-token-size table. Each FC body row is multiplied ×36 (layers). LM_head runs once (last-token logits).")
md.append("")
for s in SEQ_PREFILL:
    md.append(table_block(f"Prefill S={s}", PREFILL_TABLES[s],
                          f"TOTAL (prefill text decoder)", PREFILL_TXT_TOT[s], PREFILL_TXT_THEO_TOT[s]))

# ---------- 7. Thinker decode ----------
md.append("## 7. Thinker text decoder — Decode (per output token, TPOT)")
md.append("")
for kv in SEQ_DECODE:
    md.append(table_block(f"Decode KV={kv}", DECODE_TABLES[kv],
                          f"TOTAL (decode per token)", DECODE_TOT[kv], DECODE_THEO_TOT[kv]))

# ---------- 8. Talker (optional audio-output overhead) ----------
md.append("## 8. Talker decode — per output text token (audio-output overhead, optional)")
md.append("")
md.append("Talker runs once per output text token when `enable_audio_output=True`. Below is per-token-size cost.")
md.append("")
for kv in SEQ_DECODE:
    md.append(table_block(f"Talker decode KV={kv}", TALKER_DECODE_TABLES[kv],
                          f"TOTAL (talker decode per token)", TALKER_DECODE_TOT[kv], TALKER_DECODE_THEO_TOT[kv]))

# ---------- 9. Token latency summary ----------
md.append("## 9. End-to-end token latency summary")
md.append("")
md.append("### Prefill — TTFT (text-only / +audio input / +audio+vision input)")
md.append("")
md.append("| S (text ctx) | Thinker prefill (ms) | + Audio enc (ms) | + Vision enc (ms) | **TTFT text-only** | **TTFT +audio** | **TTFT +A+V** |")
md.append("|---:|---:|---:|---:|---:|---:|---:|")
for s in SEQ_PREFILL:
    md.append(f"| {s} | {_fmt(PREFILL_TXT_TOT[s])} | {_fmt(ENC_AUDIO_TOT)} | {_fmt(ENC_VIS_TOT)} | "
              f"**{_fmt(ttft(s, 'text'))}** | **{_fmt(ttft(s, 'audio'))}** | **{_fmt(ttft(s, 'av'))}** |")
md.append("")
md.append(f"### Decode — TPOT (per output token; {OUTPUT_TOKENS} output-token total = 512 × TPOT)")
md.append("")
md.append("| KV (ctx) | Thinker TPOT (ms) | + Talker TPOT (ms) | tokens/s (thinker only) | tokens/s (text+audio out) |")
md.append("|---:|---:|---:|---:|---:|")
for kv in SEQ_DECODE:
    tpot_t = DECODE_TOT[kv]; tpot_talker = TALKER_DECODE_TOT[kv]
    ts_thinker = (1000.0 / tpot_t) if tpot_t else None
    ts_av = (1000.0 / (tpot_t + tpot_talker)) if (tpot_t and tpot_talker) else None
    md.append(f"| {kv} | {_fmt(tpot_t)} | {_fmt(tpot_talker)} | "
              f"{_fmt(ts_thinker, 1)} | {_fmt(ts_av, 1)} |")
md.append("")

# ---------- 10. Theoretical vs measured aggregate ----------
md.append("## 10. Measured vs. theoretical aggregate eff%")
md.append("")
md.append("The aggregate slowdown = measured total / theoretical total at each phase. Eff% is the inverse.")
md.append("")
md.append("### Prefill aggregate")
md.append("")
md.append("| S | Measured (ms) | Theoretical roofline (ms) | Slowdown | Aggregate eff% |")
md.append("|---:|---:|---:|---:|---:|")
for s in SEQ_PREFILL:
    m = PREFILL_TXT_TOT[s]; t = PREFILL_TXT_THEO_TOT[s]
    slow = (m / t) if (m and t) else None
    eff = (1 / slow * 100) if slow else None
    md.append(f"| {s} | {_fmt(m)} | {_fmt(t)} | {_fmt(slow, 2)}× | {_fmt(eff, 1)}% |")
md.append("")
md.append("### Decode aggregate")
md.append("")
md.append("| KV | Measured (ms) | Theoretical roofline (ms) | Slowdown | Aggregate eff% |")
md.append("|---:|---:|---:|---:|---:|")
for kv in SEQ_DECODE:
    m = DECODE_TOT[kv]; t = DECODE_THEO_TOT[kv]
    slow = (m / t) if (m and t) else None
    eff = (1 / slow * 100) if slow else None
    md.append(f"| {kv} | {_fmt(m)} | {_fmt(t)} | {_fmt(slow, 2)}× | {_fmt(eff, 1)}% |")
md.append("")

# ---------- 11. Reproduction commands ----------
md.append("## 11. Reproduction commands")
md.append("")
md.append("**Build benches once on Windows PTL 12Xe target:**")
md.append("```cmd")
md.append("D:\\river\\moe\\dev_roofline_profiling\\utils\\build_remote.bat")
md.append("```")
md.append("")
md.append("**Run full sweep:**")
md.append("```cmd")
md.append("D:\\river\\moe\\dev_roofline_profiling\\utils\\run_qwen3_omni_4B_ptl_12xe.bat")
md.append("```")
md.append("")
md.append("**Pull logs back, then re-run report locally:**")
md.append("```bash")
md.append("python3 ../../utils/parse_logs.py logs_ptl_12xe > parsed.json")
md.append("python3 build_report.py")
md.append("```")
md.append("")

# ---------- 12. Caveats ----------
md.append("## 12. Caveats")
md.append("")
md.append(f"- **KV cache scale/zp accounting**: PA OCL INT8 layout stores 4B fp16 scale+zp per 16-token block (K) and per HD-row (V). Roofline bytes include this.")
md.append(f"- **Talker code-predictor / code2wav** are reported theoretical-only (small additional overhead per output text token). For full streaming-TTS performance measurement, a dedicated bench is required.")
md.append(f"- **Vision encoder HD=72** is non-standard; some SDPA paths may pad to HD=80. The measured row (if present) reflects what the GPU plugin actually launches.")
md.append(f"- **Bidirectional encoder SDPA** is approximated as `2× causal sdpa_micro__prefill`. Real bidirectional kernels may be ~1.5–2× depending on tile reuse.")
md.append(f"- **Theoretical fallback for missing measurements**: a few rows could not be measured on PTL 12Xe and use the theoretical value in aggregates (marked with `—` in the per-row meas columns): `vis_sdpa` (`sdpa_micro` does not support `HD=72`, the bench produced 0 enqueues), `vis_fc_fc2` (`fc_int4_g128` requires `K%group_size==0`; `K=4304` is not a multiple of `g128`; fallback uses the theoretical bandwidth-bound INT4 g128 cost).")
md.append(f"- **Theoretical bytes** for FC use the actual VRAM-streamed footprint (compressed weights + fp16 scales + int8 zp + activations), which is what the BW-roofline must compare against, not the post-decompress fp16 weight size.")
md.append("")

(HERE / f"SUMMARY_qwen3_omni_4B_ptl_12xe_{DATE}.md").write_text("\n".join(md))
print(f"Wrote SUMMARY_qwen3_omni_4B_ptl_12xe_{DATE}.md")
