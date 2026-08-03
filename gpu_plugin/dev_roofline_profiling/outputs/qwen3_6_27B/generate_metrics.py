#!/usr/bin/env python3
"""
Qwen3.6-27B B60 Roofline Metrics Generator
===========================================
Reads cliloader logs from logs/b60/qwen3_6_27B/ directory,
parses them via parse_logs.py logic, and generates:
  - outputs/b60_raw_metrics.json  (raw per-log kernel stats from parse_logs.py)
  - outputs/performance_metrics.json  (per-op roofline metrics with FLOPS/BW/eff%)

Usage:
    python3 generate_metrics.py [log_dir] [output_dir]

Defaults:
    log_dir  = ../logs/b60/qwen3_6_27B
    output_dir = ../outputs
"""
import json
import sys
import re
import math
from pathlib import Path
from collections import Counter

# ============================================================
# Hardware (Intel Arc B60, 0xE210, measured by hw_probe/mem_bw)
# ============================================================
HW = {
    "name": "Intel Arc B60 (Xe2 dGPU, 0xE210)",
    "xe_cores": 20,
    "eus_per_core": 8,
    "threads_per_eu": 8,
    "freq_mhz": 2400,
    "bw_gbs_measured_read": 451.0,   # from mem_bw 2048 15 -> Read BW
    "bw_gbs_measured_copy": 397.0,   # from mem_bw 2048 15 -> Copy BW (R+W)
    "bw_gbs": 451.0,                  # use read BW for decode weight-read analysis
    "l3_cache_mb": 18.0,
    "memory_gb": 24.0,
    "fp16_xmx_tflops": 20 * 8 * 256 * 2.4,   # = 98.304
    "int8_xmx_tops": 20 * 8 * 512 * 2.4,      # = 196.608
    "simd_fp16_tflops": 20 * 8 * 32 * 2.4,    # = 12.288
}
HW["fp16_xmx_tflops"] = round(HW["fp16_xmx_tflops"] / 1000.0, 3)
HW["int8_xmx_tops"] = round(HW["int8_xmx_tops"] / 1000.0, 3)

BW = HW["bw_gbs"]        # for memory-bound ops (decode weight reads)
FP16 = HW["fp16_xmx_tflops"]  # for compute-bound ops
INT8 = HW["int8_xmx_tops"]

# Roofline ridge point
RIDGE_FP16 = FP16 * 1e3 / BW   # FLOPS/byte where compute = memory-bound
RIDGE_INT8 = INT8 * 1e3 / BW

# ============================================================
# Model configuration
# ============================================================
CFG = {
    "model": "Qwen3.6-27B",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "num_full_attention_layers": 16,   # 64/4 = 16 full-attn layers
    "num_linear_attention_layers": 48, # 64 - 16 = 48 linear-attn (GDN) layers
    "mtp_extra_layers": 1,             # 1 MTP layer (treated as +1 full-attn)
    "num_attention_heads": 24,
    "num_kv_heads": 4,
    "head_dim": 256,
    "gqa_ratio": 6,
    "rotary_dim": 64,
    "attn_output_gate": True,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_head_dim": 128,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "weight_group_size": 128,
    "kv_cache_quant": "int8",
    "kv_block_size": 16,
}

NL = CFG["num_hidden_layers"]
NL_F = CFG["num_full_attention_layers"]
NL_L = CFG["num_linear_attention_layers"]
MTP = CFG["mtp_extra_layers"]
H = CFG["hidden_size"]
NH = CFG["num_attention_heads"]
NKV = CFG["num_kv_heads"]
HD = CFG["head_dim"]
I = CFG["intermediate_size"]
VOCAB = CFG["vocab_size"]
G = CFG["weight_group_size"]
BLOCK = CFG["kv_block_size"]
LIN_HK = CFG["linear_num_key_heads"]
LIN_HV = CFG["linear_num_value_heads"]
LIN_HD = CFG["linear_head_dim"]

# Derived shapes
QKV_W = 2 * NH * HD + 2 * NKV * HD   # Q+gate+K+V = 6144+6144+1024+1024 = 14336
O_W = NH * HD                           # output proj width = 6144
LIN_PROJ_W = 2 * LIN_HK * LIN_HD + 2 * LIN_HV * LIN_HD  # 4096+6144+6144=16384

# Number of full-attn + MTP layers
NL_FA = NL_F + MTP   # 16 + 1 = 17

TOKEN_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

# ============================================================
# Log parsing (inline, same as parse_logs.py)
# ============================================================
KERNEL_EXCLUDES = (
    "clEnqueue", "clFinish", "clWait", "clFlush",
    "clRelease", "clRetain", "clSetKernel", "activation",
)

def parse_device_timing(text):
    dev_idx = text.find("Device Performance Timing Results")
    if dev_idx < 0:
        return {}
    section = text[dev_idx:]
    kernels = {}
    for line in section.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        name = parts[0]
        if not name or name.startswith("Function Name"):
            continue
        if any(name.startswith(x) for x in KERNEL_EXCLUDES):
            continue
        try:
            calls     = int(parts[1])
            total_ns  = int(parts[2])
            avg_ns    = int(parts[4])
            min_ns    = int(parts[5])
            max_ns    = int(parts[6])
        except (ValueError, IndexError):
            continue
        kernels[name] = dict(calls=calls, total_ns=total_ns,
                             avg_ns=avg_ns, min_ns=min_ns, max_ns=max_ns)
    return kernels


def summarize_log(path):
    text = path.read_text(errors="ignore")
    kernels = parse_device_timing(text)
    gpu_kernels = {n: k for n, k in kernels.items()
                   if not any(n.startswith(x) for x in KERNEL_EXCLUDES)}
    counts = [k["calls"] for k in gpu_kernels.values() if k["calls"] > 1]
    if counts:
        cnt = Counter(counts)
        max_freq = max(cnt.values())
        iters = min(c for c, f in cnt.items() if f == max_freq)
    else:
        iters = max((k["calls"] for k in gpu_kernels.values()), default=1)
    total_ns = 0
    per_kernel = []
    for name, stat in gpu_kernels.items():
        if stat["calls"] < iters:
            continue
        per_iter_ns = stat["total_ns"] / iters
        total_ns += per_iter_ns
        per_kernel.append((name, int(per_iter_ns), stat["calls"]))
    per_kernel.sort(key=lambda x: -x[1])
    return {"total_kernel_ns": int(total_ns), "iters_detected": iters, "per_kernel": per_kernel}


def parse_all_logs(log_dir):
    results = {}
    for log in sorted(Path(log_dir).glob("*.log")):
        results[log.stem] = summarize_log(log)
    return results


# ============================================================
# Metric helpers
# ============================================================
def ns_to_ms(ns):
    return ns / 1e6

def get_kernel_ms(raw, key):
    e = raw.get(key)
    if not e or not e.get("total_kernel_ns"):
        return 0.0
    return ns_to_ms(e["total_kernel_ns"])


def fc_bytes(M_tokens, K_in, N_out, g=128, is_u8=False):
    """Total memory bytes for a FC with INT4/INT8 compressed weights + FP16 activation + output."""
    bytes_per_w = 1 if is_u8 else 0.5
    w_bytes = N_out * K_in * bytes_per_w
    n_groups = K_in // g
    scale_bytes = N_out * n_groups * 2   # f16 scales
    zp_bytes = N_out * n_groups * 1      # u8 zp
    act_bytes = M_tokens * K_in * 2      # f16 input
    out_bytes = M_tokens * N_out * 2     # f16 output
    return w_bytes + scale_bytes + zp_bytes + act_bytes + out_bytes


def fc_flops(M_tokens, K_in, N_out):
    """FC FLOPs: 2 * M * K * N (multiply-add)."""
    return 2 * M_tokens * K_in * N_out


def pa_decode_bytes(sq, skv, nh, nkv, hd, block_size=16):
    """PA decode memory bytes: read KV cache (INT8) + read Q (FP16) + write output (FP16)."""
    # KV cache: [num_blocks, nkv, hd*block_size] INT8 for K + similar for V
    kv_blocks = math.ceil(skv / block_size)
    k_bytes = kv_blocks * nkv * hd * block_size * 1   # INT8
    v_bytes = kv_blocks * nkv * block_size * hd * 1   # INT8
    q_bytes = sq * nh * hd * 2   # FP16
    out_bytes = sq * nh * hd * 2  # FP16
    return k_bytes + v_bytes + q_bytes + out_bytes


def pa_prefill_flops(sq, nh, hd, causal=True):
    """PA prefill FLOPs with causal mask: effective pairs = sq*(sq+1)/2."""
    effective_pairs = sq * (sq + 1) // 2 if causal else sq * sq
    # QK: sq * effective_pairs * head_size (GQA: all nh attend to same KV)
    qk_flops = 2 * nh * effective_pairs * hd
    # SV: sq * effective_pairs * head_size
    sv_flops = 2 * nh * effective_pairs * hd
    return qk_flops + sv_flops


def pa_prefill_bytes(sq, nh, nkv, hd, block_size=16):
    """PA prefill memory bytes: read Q/K/V (FP16) + write output (FP16) + write KV cache (INT8)."""
    q_bytes = sq * nh * hd * 2
    k_read_bytes = sq * nkv * hd * 2   # FP16 K before caching
    v_read_bytes = sq * nkv * hd * 2
    out_bytes = sq * nh * hd * 2
    # KV cache write: INT8
    kv_blocks = math.ceil(sq / block_size)
    k_write = kv_blocks * nkv * hd * block_size * 1
    v_write = kv_blocks * nkv * block_size * hd * 1
    return q_bytes + k_read_bytes + v_read_bytes + out_bytes + k_write + v_write


def gdn_decode_bytes(v_heads, v_hd, qk_heads, qk_hd):
    """GDN decode: read state (FP16) + read QKV inputs + write output."""
    # Recurrent state: [1, v_heads, v_hd, qk_hd] per sequence
    state_bytes = v_heads * v_hd * qk_hd * 2  # FP16
    # Input: Q(1,qk_heads,qk_hd) + K(1,qk_heads,qk_hd) + V(1,v_heads,v_hd)
    inp_bytes = (2 * qk_heads * qk_hd + v_heads * v_hd) * 2
    out_bytes = v_heads * v_hd * 2
    return state_bytes + inp_bytes + out_bytes


def gdn_prefill_bytes(T, v_heads, v_hd, qk_heads, qk_hd, block=256):
    """GDN prefill: read/write state at each step + process QKV."""
    n_blocks = math.ceil(T / block)
    state_bytes = n_blocks * v_heads * v_hd * qk_hd * 2
    inp_bytes = T * (2 * qk_heads * qk_hd + v_heads * v_hd) * 2
    out_bytes = T * v_heads * v_hd * 2
    return state_bytes + inp_bytes + out_bytes


def bound(arithmetic_intensity_flop_per_byte, ridge_point):
    return "compute" if arithmetic_intensity_flop_per_byte > ridge_point else "memory"


def eff_mem(actual_bytes, ms, bw_gbs=BW):
    if ms <= 0:
        return 0.0
    achieved_gbs = actual_bytes / (ms * 1e6)
    return 100.0 * achieved_gbs / bw_gbs


def eff_compute(actual_flops, ms, peak_tflops=FP16):
    if ms <= 0:
        return 0.0
    achieved_tflops = actual_flops / (ms * 1e9)
    return 100.0 * achieved_tflops / peak_tflops


# ============================================================
# Main metrics builder
# ============================================================
def build_metrics(raw):
    metrics = {"hardware": HW, "config": CFG, "ops": {}}
    ops = metrics["ops"]

    # ---- decode FC ops ----
    fc_ops_decode = {
        "fc_qkv_decode": ("fc_qkv_decode_M1",    1, H,    QKV_W, False, NL_FA,   "FC(1,5120,14336) INT4 g128 - QKV+gate, full-attn"),
        "fc_o_decode":   ("fc_o_decode_M1",       1, O_W,  H,     False, NL_FA,   "FC(1,6144,5120) INT4 g128 - O proj, full-attn"),
        "fc_gate_decode":("fc_gate_decode_M1",    1, H,    I,     False, NL+MTP,  "FC(1,5120,17408) INT4 g128 - FFN gate, all layers"),
        "fc_down_decode":("fc_down_decode_M1",    1, I,    H,     False, NL+MTP,  "FC(1,17408,5120) INT4 g128 - FFN down, all layers"),
        "lm_head_decode":("lm_head_decode_M1",    1, H,    VOCAB, True,  1,        "FC(1,5120,248320) INT8 g128 - LM head"),
        "fc_linattn_proj_decode":("fc_linattn_proj_decode_M1", 1, H, LIN_PROJ_W, False, NL_L, "FC(1,5120,16384) INT4 g128 - linattn proj, 48 layers"),
    }
    for op_name, (log_key, M_tok, K_in, N_out, is_u8, calls, note) in fc_ops_decode.items():
        ms = get_kernel_ms(raw, log_key)
        total_bytes = fc_bytes(M_tok, K_in, N_out, G, is_u8)
        flops = fc_flops(M_tok, K_in, N_out)
        ai = flops / total_bytes if total_bytes else 0
        achieved_gbs = total_bytes / (ms * 1e6) if ms else 0
        achieved_tflops = flops / (ms * 1e9) if ms else 0
        bd = bound(ai, RIDGE_INT8 if is_u8 else RIDGE_FP16)
        peak_for_eff = INT8 if is_u8 else FP16
        eff = eff_mem(total_bytes, ms) if bd == "memory" else eff_compute(flops, ms, peak_for_eff)
        ops[op_name] = {
            "note": note, "phase": "decode", "M": M_tok, "K": K_in, "N": N_out,
            "calls_per_inference": calls, "avg_ms": ms, "total_ms": ms * calls,
            "weight_bytes": N_out * K_in * (1 if is_u8 else 0.5),
            "total_bytes": total_bytes, "flops": flops,
            "arithmetic_intensity": ai, "bound": bd,
            "achieved_gbs": achieved_gbs, "achieved_tflops": achieved_tflops,
            "efficiency_pct": eff,
            "raw_log": log_key,
        }

    # fc_up_decode: same shape as fc_gate, reuse data
    ms_gate = get_kernel_ms(raw, "fc_gate_decode_M1")
    ms_up = get_kernel_ms(raw, "fc_up_decode_M1")
    ms_fc_gate_up = (ms_gate + ms_up) / 2 if ms_up > 0 else ms_gate
    ops["fc_gate_up_avg_decode"] = dict(ops["fc_gate_decode"])
    ops["fc_gate_up_avg_decode"]["note"] = f"FC gate ({ms_gate*1000:.1f}µs) + up ({ms_up*1000:.1f}µs) avg"
    ops["fc_gate_up_avg_decode"]["avg_ms"] = ms_fc_gate_up
    ops["fc_gate_up_avg_decode"]["calls_per_inference"] = (NL + MTP) * 2
    ops["fc_gate_up_avg_decode"]["total_ms"] = ms_fc_gate_up * (NL + MTP) * 2

    # ---- decode PA ----
    for kv in TOKEN_SIZES:
        log_key = f"pa_decode_kv{kv}"
        ms = get_kernel_ms(raw, log_key)
        total_bytes = pa_decode_bytes(1, kv, NH, NKV, HD, BLOCK)
        flops = 2 * NH * 1 * kv * HD  # Q@K and S@V
        ai = flops / total_bytes if total_bytes else 0
        achieved_gbs = total_bytes / (ms * 1e6) if ms else 0
        achieved_tflops = flops / (ms * 1e9) if ms else 0
        bd = bound(ai, RIDGE_FP16)
        eff = eff_mem(total_bytes, ms) if bd == "memory" else eff_compute(flops, ms)
        ops[f"pa_decode_kv{kv}"] = {
            "note": f"PA decode, KV={kv}, NH={NH}, NKV={NKV}, HD={HD}, INT8 KV",
            "phase": "decode", "kv": kv,
            "calls_per_inference": NL_FA,
            "avg_ms": ms, "total_ms": ms * NL_FA,
            "total_bytes": total_bytes, "flops": flops,
            "arithmetic_intensity": ai, "bound": bd,
            "achieved_gbs": achieved_gbs, "achieved_tflops": achieved_tflops,
            "efficiency_pct": eff, "raw_log": log_key,
        }

    # ---- decode GDN ----
    log_key = "gdn_decode_T1"
    ms = get_kernel_ms(raw, log_key)
    total_bytes = gdn_decode_bytes(LIN_HV, LIN_HD, LIN_HK, LIN_HD)
    # GDN compute: ssm core is ~2 * v_heads * qk_heads * v_hd per token
    flops = 2 * LIN_HV * LIN_HK * LIN_HD   # approximate SSM GEMM
    ai = flops / total_bytes if total_bytes else 0
    achieved_gbs = total_bytes / (ms * 1e6) if ms else 0
    bd = bound(ai, RIDGE_FP16)
    eff = eff_mem(total_bytes, ms) if bd == "memory" else eff_compute(flops, ms)
    ops["gdn_decode_T1"] = {
        "note": f"GDN decode T=1, qk_heads={LIN_HK}, v_heads={LIN_HV}, head_dim={LIN_HD}",
        "phase": "decode",
        "calls_per_inference": NL_L,
        "avg_ms": ms, "total_ms": ms * NL_L,
        "total_bytes": total_bytes, "flops": flops,
        "arithmetic_intensity": ai, "bound": bd,
        "achieved_gbs": achieved_gbs, "achieved_tflops": achieved_tflops,
        "efficiency_pct": eff, "raw_log": log_key,
    }

    # ---- prefill FC ops ----
    for S in TOKEN_SIZES:
        fc_ops_prefill = {
            f"fc_qkv_prefill_S{S}": (f"fc_qkv_prefill_S{S}",     S, H,    QKV_W, False, NL_FA,  f"FC({S},5120,14336) INT4 g128 prefill"),
            f"fc_o_prefill_S{S}":   (f"fc_o_prefill_S{S}",        S, O_W,  H,     False, NL_FA,  f"FC({S},6144,5120) INT4 g128 prefill"),
            f"fc_gate_prefill_S{S}":(f"fc_gate_prefill_S{S}",     S, H,    I,     False, NL+MTP, f"FC({S},5120,17408) INT4 g128 FFN gate prefill"),
            f"fc_down_prefill_S{S}":(f"fc_down_prefill_S{S}",     S, I,    H,     False, NL+MTP, f"FC({S},17408,5120) INT4 g128 FFN down prefill"),
            f"fc_linattn_prefill_S{S}":(f"fc_linattn_proj_prefill_S{S}", S, H, LIN_PROJ_W, False, NL_L, f"FC({S},5120,16384) INT4 g128 linattn prefill"),
        }
        for op_name, (log_key, M_tok, K_in, N_out, is_u8, calls, note) in fc_ops_prefill.items():
            ms = get_kernel_ms(raw, log_key)
            total_bytes = fc_bytes(M_tok, K_in, N_out, G, is_u8)
            flops = fc_flops(M_tok, K_in, N_out)
            ai = flops / total_bytes if total_bytes else 0
            achieved_gbs = total_bytes / (ms * 1e6) if ms else 0
            achieved_tflops = flops / (ms * 1e9) if ms else 0
            # prefill uses INT8 XMX (dynamic_quantize path)
            bd = bound(ai, RIDGE_INT8)
            peak_for_eff = INT8
            eff = eff_mem(total_bytes, ms) if bd == "memory" else eff_compute(flops, ms, peak_for_eff)
            ops[op_name] = {
                "note": note, "phase": "prefill", "M": M_tok, "K": K_in, "N": N_out,
                "calls_per_inference": calls, "avg_ms": ms, "total_ms": ms * calls,
                "total_bytes": total_bytes, "flops": flops,
                "arithmetic_intensity": ai, "bound": bd,
                "achieved_gbs": achieved_gbs, "achieved_tflops": achieved_tflops,
                "efficiency_pct": eff, "raw_log": log_key,
            }

    # ---- prefill PA ----
    for S in TOKEN_SIZES:
        log_key = f"pa_prefill_S{S}"
        ms = get_kernel_ms(raw, log_key)
        flops = pa_prefill_flops(S, NH, HD, causal=True)
        total_bytes = pa_prefill_bytes(S, NH, NKV, HD, BLOCK)
        ai = flops / total_bytes if total_bytes else 0
        achieved_gbs = total_bytes / (ms * 1e6) if ms else 0
        achieved_tflops = flops / (ms * 1e9) if ms else 0
        bd = bound(ai, RIDGE_FP16)
        eff = eff_mem(total_bytes, ms) if bd == "memory" else eff_compute(flops, ms)
        ops[f"pa_prefill_S{S}"] = {
            "note": f"PA prefill S={S}, NH={NH}, NKV={NKV}, HD={HD}, INT8 KV, causal",
            "phase": "prefill", "sq": S,
            "calls_per_inference": NL_FA,
            "avg_ms": ms, "total_ms": ms * NL_FA,
            "total_bytes": total_bytes, "flops": flops,
            "arithmetic_intensity": ai, "bound": bd,
            "achieved_gbs": achieved_gbs, "achieved_tflops": achieved_tflops,
            "efficiency_pct": eff, "raw_log": log_key,
        }

    # ---- prefill GDN ----
    for S in TOKEN_SIZES:
        log_key = f"gdn_prefill_S{S}"
        ms = get_kernel_ms(raw, log_key)
        total_bytes = gdn_prefill_bytes(S, LIN_HV, LIN_HD, LIN_HK, LIN_HD)
        flops = 2 * LIN_HV * LIN_HK * LIN_HD * S
        ai = flops / total_bytes if total_bytes else 0
        achieved_gbs = total_bytes / (ms * 1e6) if ms else 0
        achieved_tflops = flops / (ms * 1e9) if ms else 0
        bd = bound(ai, RIDGE_FP16)
        eff = eff_mem(total_bytes, ms) if bd == "memory" else eff_compute(flops, ms)
        ops[f"gdn_prefill_S{S}"] = {
            "note": f"GDN prefill T={S}, qk_heads={LIN_HK}, v_heads={LIN_HV}, head_dim={LIN_HD}",
            "phase": "prefill", "T": S,
            "calls_per_inference": NL_L,
            "avg_ms": ms, "total_ms": ms * NL_L,
            "total_bytes": total_bytes, "flops": flops,
            "arithmetic_intensity": ai, "bound": bd,
            "achieved_gbs": achieved_gbs, "achieved_tflops": achieved_tflops,
            "efficiency_pct": eff, "raw_log": log_key,
        }

    # ---- small ops decode (approximate; use heuristic BW for eltwise) ----
    small_ops_decode = {
        "so_rmsnorm_decode":  ("so_rmsnorm_h5120_decode",  1, H*3*2,   NL*2,     "RMSNorm H=5120 (64layers×2=128 calls/decode)"),
        "so_rope_q_decode":   ("so_rope_q_decode",          1, NH*64*2*2, NL_FA,  "RoPE Q M=1, NH=24, rotary_dim=64 (17 calls)"),
        "so_rope_k_decode":   ("so_rope_k_decode",          1, NKV*64*2*2, NL_FA, "RoPE K M=1, NKV=4, rotary_dim=64 (17 calls)"),
        "so_add_decode":      ("so_add_decode",             1, H*3*2,   NL*2,     "Residual add H=5120 (64layers×2=128 calls)"),
    }
    for op_name, (log_key, M_tok, byte_per_call, calls, note) in small_ops_decode.items():
        ms = get_kernel_ms(raw, log_key)
        total_bytes = byte_per_call * M_tok if M_tok else byte_per_call
        achieved_gbs = total_bytes / (ms * 1e6) if ms else 0
        ops[op_name] = {
            "note": note, "phase": "decode",
            "calls_per_inference": calls, "avg_ms": ms, "total_ms": ms * calls,
            "total_bytes": total_bytes, "bound": "memory",
            "achieved_gbs": achieved_gbs, "efficiency_pct": eff_mem(total_bytes, ms),
            "raw_log": log_key,
        }

    # ---- model-level totals ----
    model_totals = {}

    # Decode at each KV length
    for kv in TOKEN_SIZES:
        fc_ms = (
            ops["fc_qkv_decode"]["total_ms"] +
            ops["fc_o_decode"]["total_ms"] +
            ops["fc_gate_up_avg_decode"]["total_ms"] +
            ops["fc_down_decode"]["total_ms"] +
            ops["fc_linattn_proj_decode"]["total_ms"] +
            ops["lm_head_decode"]["total_ms"]
        )
        pa_ms = ops.get(f"pa_decode_kv{kv}", {}).get("total_ms", 0)
        gdn_ms = ops["gdn_decode_T1"]["total_ms"]
        small_ms = (
            ops.get("so_rmsnorm_decode", {}).get("total_ms", 0) +
            ops.get("so_rope_q_decode", {}).get("total_ms", 0) +
            ops.get("so_rope_k_decode", {}).get("total_ms", 0) +
            ops.get("so_add_decode", {}).get("total_ms", 0)
        )
        total_ms = fc_ms + pa_ms + gdn_ms + small_ms
        model_totals[f"decode_kv{kv}"] = {
            "kv_len": kv,
            "fc_ms": fc_ms, "pa_ms": pa_ms, "gdn_ms": gdn_ms, "small_ms": small_ms,
            "total_ms": total_ms,
            "tokens_per_sec": 1000.0 / total_ms if total_ms else 0,
        }

    # Prefill at each S
    for S in TOKEN_SIZES:
        fc_ms = (
            ops.get(f"fc_qkv_prefill_S{S}", {}).get("total_ms", 0) +
            ops.get(f"fc_o_prefill_S{S}", {}).get("total_ms", 0) +
            ops.get(f"fc_gate_prefill_S{S}", {}).get("total_ms", 0) * 2 +  # gate + up
            ops.get(f"fc_down_prefill_S{S}", {}).get("total_ms", 0) +
            ops.get(f"fc_linattn_prefill_S{S}", {}).get("total_ms", 0)
        )
        pa_ms = ops.get(f"pa_prefill_S{S}", {}).get("total_ms", 0)
        gdn_ms = ops.get(f"gdn_prefill_S{S}", {}).get("total_ms", 0)
        small_ms = ops.get(f"so_rmsnorm_h5120_prefill_S{S}", {}).get("total_ms", 0)
        total_ms = fc_ms + pa_ms + gdn_ms + small_ms
        model_totals[f"prefill_S{S}"] = {
            "S": S,
            "fc_ms": fc_ms, "pa_ms": pa_ms, "gdn_ms": gdn_ms, "small_ms": small_ms,
            "total_ms": total_ms,
            "tokens_per_sec": S * 1000.0 / total_ms if total_ms else 0,
        }

    metrics["model_totals"] = model_totals
    return metrics


def main():
    arg1 = sys.argv[1] if len(sys.argv) > 1 else "../logs/b60/qwen3_6_27B"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accept either a directory of .log files OR a pre-parsed JSON file
    p1 = Path(arg1)
    if p1.is_file() and p1.suffix == ".json":
        print(f"Loading pre-parsed raw metrics from: {arg1}")
        raw = json.loads(p1.read_text())
        print(f"  -> {len(raw)} entries")
    else:
        print(f"Parsing logs from: {arg1}")
        raw = parse_all_logs(arg1)
        raw_path = out_dir / "b60_raw_metrics.json"
        raw_path.write_text(json.dumps(raw, indent=2))
        print(f"  -> Raw metrics: {raw_path} ({len(raw)} logs)")

    print("Building roofline metrics...")
    metrics = build_metrics(raw)
    out_path = out_dir / "performance_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"  -> Performance metrics: {out_path}")

    # Quick summary printout
    print(f"\n{'='*80}")
    print(f"Intel Arc B60 — Qwen3.6-27B Decode Summary (kernel-only)")
    print(f"{'='*80}")
    print(f"HW peaks: FP16={HW['fp16_xmx_tflops']} TFLOPS, INT8={HW['int8_xmx_tops']} TOPS, BW={BW} GB/s")
    print(f"\n{'Op':<35}{'Avg ms':>8}{'Calls':>7}{'Tot ms':>9}{'GB/s':>8}{'Eff%':>7}{'Bound':<8}")
    print(f"{'─'*35}{'─'*8}{'─'*7}{'─'*9}{'─'*8}{'─'*7}{'─'*8}")
    for op_name, op in metrics["ops"].items():
        if op["phase"] == "decode" and "kv" not in op_name:
            ms = op.get("avg_ms", 0)
            calls = op.get("calls_per_inference", 0)
            total = op.get("total_ms", 0)
            gbs = op.get("achieved_gbs", 0)
            eff = op.get("efficiency_pct", 0)
            bd = op.get("bound", "?")[:6]
            print(f"{op_name[:35]:<35}{ms*1000:>7.0f}µ{calls:>7}{total:>9.2f}{gbs:>8.0f}{eff:>6.1f}%{bd:<8}")

    print(f"\n{'='*50} Model decode totals {'='*10}")
    print(f"{'KV':>6}{'FC ms':>8}{'PA ms':>8}{'GDN ms':>8}{'Misc ms':>8}{'TOTAL ms':>10}{'tok/s':>8}")
    for kv in TOKEN_SIZES:
        r = metrics["model_totals"].get(f"decode_kv{kv}", {})
        if r:
            print(f"{kv:>6}{r['fc_ms']:>8.2f}{r['pa_ms']:>8.2f}{r['gdn_ms']:>8.2f}"
                  f"{r['small_ms']:>8.2f}{r['total_ms']:>10.2f}{r['tokens_per_sec']:>8.1f}")

    print(f"\n{'='*50} Model prefill totals {'='*9}")
    print(f"{'S':>6}{'FC ms':>8}{'PA ms':>8}{'GDN ms':>8}{'Misc ms':>8}{'TOTAL ms':>10}{'tok/s':>8}")
    for S in TOKEN_SIZES:
        r = metrics["model_totals"].get(f"prefill_S{S}", {})
        if r and r["total_ms"] > 0:
            print(f"{S:>6}{r['fc_ms']:>8.2f}{r['pa_ms']:>8.2f}{r['gdn_ms']:>8.2f}"
                  f"{r['small_ms']:>8.2f}{r['total_ms']:>10.2f}{r['tokens_per_sec']:>8.0f}")

    return metrics


if __name__ == "__main__":
    main()
