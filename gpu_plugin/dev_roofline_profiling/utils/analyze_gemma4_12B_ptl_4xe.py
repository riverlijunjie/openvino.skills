#!/usr/bin/env python3
"""
Gemma4-12B PTL 4Xe roofline analysis.
Reads ptl_4xe_metrics.json (from parse_logs.py) and emits per-op roofline
tables (decode + prefill), model totals, and efficiency vs. PTL 4Xe peaks.

PTL 4Xe peaks: 4 Xe cores @ 2450 MHz
  FP16 XMX = 4*8*256*2.45e9 = 20.07 TFLOPS
  INT8 XMX = 2x = 40.14 TOPS
  BW = 110 GB/s
"""
import json
from pathlib import Path

OUT = Path("/home/ov2022/workspace/remote_debug/openvino/.github/skills/dev_roofline_profiling/outputs/gemma4_12B")
M = json.loads((OUT / "ptl_4xe_metrics.json").read_text())

def ms(k):
    return M[k]["total_kernel_ns"] / 1e6 if k in M else 0.0

# ---- HW peaks ----
BW = 110.0
FP16_XMX = 4 * 8 * 256 * 2.45 / 1000.0  # 20.07 TFLOPS
INT8_XMX = 2 * FP16_XMX                  # 40.14 TOPS
OVH = 0.95
BW_e, FP16_e, INT8_e = BW * OVH, FP16_XMX * OVH, INT8_XMX * OVH

# ---- Model config ----
NL = 48
NL_SLIDING = 40
NL_FULL = 8
SW = 1024

# ---- Byte / FLOP models ----
def fc_int4_bytes(M_, K, N, g=128):
    w = N * K / 2 + N * (K // g) * 2 + N * (K // g) / 2
    return w + M_ * K * 2 + M_ * N * 2

def fc_int8_bytes(M_, K, N, g=128):
    w = N * K + N * (K // g) * 2
    return w + M_ * K * 2 + M_ * N * 2

def fc_flops(M_, K, N):
    return 2 * M_ * K * N

# PA bytes: INT8 KV cache (read) + FP16 Q + FP16 out
def pa_bytes(Sq, Skv, NKV, HD):
    kv = 2 * Skv * NKV * HD * 1          # K + V int8
    q = Sq * NKV * HD * 2 * (16 // NKV if NKV <= 16 else 1)  # rough; use NH actually
    return kv + q

# Decode FC specs: (name, key, M, K, N, quant, layers)
DECODE_FC = [
    ("FC_QKV_sliding", "fc_qkv_sliding_decode_M1", 1, 3840, 8192, "u4", NL_SLIDING),
    ("FC_O_sliding", "fc_o_sliding_decode_M1", 1, 4096, 3840, "u4", NL_SLIDING),
    ("FC_QK_full", "fc_qk_full_decode_M1", 1, 3840, 8704, "u4", NL_FULL),
    ("FC_O_full", "fc_o_full_decode_M1", 1, 8192, 3840, "u4", NL_FULL),
    ("MLP_gate", "fc_gate_dense_decode_M1", 1, 3840, 15360, "u4", NL),
    ("MLP_up", "fc_up_dense_decode_M1", 1, 3840, 15360, "u4", NL),
    ("MLP_down", "fc_down_dense_decode_M1", 1, 15360, 3840, "u4", NL),
    ("LM_head", "lm_head_decode_M1", 1, 3840, 262144, "u8", 1),
]


def fc_row(name, key, M_, K, N, quant, calls, compute_bound=False, xmx=INT8_e):
    t = ms(key)
    B = fc_int4_bytes(M_, K, N) if quant == "u4" else fc_int8_bytes(M_, K, N)
    F = fc_flops(M_, K, N)
    gbs = B / 1e9 / (t / 1e3) if t else 0
    gflops = F / 1e9 / (t / 1e3) if t else 0
    if compute_bound:
        eff = gflops / 1000.0 / xmx * 100
        bound = "compute"
    else:
        eff = gbs / BW * 100
        bound = "memory"
    return dict(op=name, single=t, calls=calls, total=t * calls,
                gflops=gflops, gbs=gbs, eff=eff, bound=bound)


# ===== DECODE =====
print("=" * 90)
print("DECODE per-op (M=1)")
print("=" * 90)
for name, key, M_, K, N, q, calls in DECODE_FC:
    r = fc_row(name, key, M_, K, N, q, calls)
    print(f"{r['op']:18s} single={r['single']:.4f} calls={r['calls']:3d} "
          f"total={r['total']:.3f} GB/s={r['gbs']:.1f} eff={r['eff']:.1f}% {r['bound']}")

# Decode small ops aggregate
so_decode = (
    ms("small_decode_rmsnorm") * (4 * NL + 1) +
    ms("small_decode_rmsnorm3d_q_sl") * NL_SLIDING +
    ms("small_decode_rmsnorm3d_k_sl") * NL_SLIDING +
    ms("small_decode_rmsnorm3d_q_f") * NL_FULL +
    ms("small_decode_rope_q_sl") * NL_SLIDING +
    ms("small_decode_rope_k_sl") * NL_SLIDING +
    ms("small_decode_rope_q_f") * NL_FULL +
    ms("small_decode_add") * (2 * NL))
print(f"SmallOps decode total = {so_decode:.3f} ms")

# Decode PA + totals
print("\nDECODE model totals:")
decode_tot = {}
for kv in [256, 1024, 2048, 4096, 8192]:
    fc_sl = (ms("fc_qkv_sliding_decode_M1") + ms("fc_o_sliding_decode_M1")) * NL_SLIDING
    fc_f = (ms("fc_qk_full_decode_M1") + ms("fc_o_full_decode_M1")) * NL_FULL
    dense = (ms("fc_gate_dense_decode_M1") + ms("fc_up_dense_decode_M1") + ms("fc_down_dense_decode_M1")) * NL
    eff_kv = min(kv, SW)
    pa_sl = ms(f"pa_sliding_decode_kv{eff_kv}") * NL_SLIDING
    pa_f = ms(f"pa_full_decode_kv{kv}") * NL_FULL
    lm = ms("lm_head_decode_M1")
    tot = fc_sl + fc_f + dense + pa_sl + pa_f + lm + so_decode
    decode_tot[kv] = tot
    print(f"  kv={kv:>5}: {tot:.2f} ms -> {1000/tot:.1f} tok/s "
          f"(fc={fc_sl+fc_f+dense:.2f} pa={pa_sl+pa_f:.3f} lm={lm:.2f} so={so_decode:.3f})")

# ===== PREFILL =====
print("\n" + "=" * 90)
print("PREFILL model totals")
print("=" * 90)
prefill_tot = {}
for S in [256, 1024, 2048, 4096, 8192]:
    fc_sl = (ms(f"fc_qkv_sliding_prefill_S{S}") + ms(f"fc_o_sliding_prefill_S{S}")) * NL_SLIDING
    fc_f = (ms(f"fc_qk_full_prefill_S{S}") + ms(f"fc_o_full_prefill_S{S}")) * NL_FULL
    dense = (ms(f"fc_gate_dense_prefill_S{S}") + ms(f"fc_up_dense_prefill_S{S}") + ms(f"fc_down_dense_prefill_S{S}")) * NL
    pa_sl_base = ms(f"pa_sliding_prefill_S{min(S, SW)}")
    pa_sl = (pa_sl_base if S <= SW else ms("pa_sliding_prefill_S1024") * (S / SW)) * NL_SLIDING
    pa_f = ms(f"pa_full_prefill_S{S}") * NL_FULL
    lm = ms("lm_head_prefill")
    so = (ms(f"small_prefill_rmsnorm_S{S}") * (4 * NL + 1) +
          ms(f"small_prefill_rmsnorm3d_q_sl_S{S}") * NL_SLIDING +
          ms(f"small_prefill_rmsnorm3d_k_sl_S{S}") * NL_SLIDING +
          ms(f"small_prefill_rmsnorm3d_q_f_S{S}") * NL_FULL +
          ms(f"small_prefill_rope_q_sl_S{S}") * NL_SLIDING +
          ms(f"small_prefill_rope_k_sl_S{S}") * NL_SLIDING +
          ms(f"small_prefill_rope_q_f_S{S}") * NL_FULL +
          ms(f"small_prefill_add_S{S}") * (2 * NL))
    tot = fc_sl + fc_f + dense + pa_sl + pa_f + lm + so
    prefill_tot[S] = tot
    print(f"  S={S:>5}: TTFT={tot:.1f} ms -> {S*1000/tot:.0f} tok/s "
          f"(fc={fc_sl+fc_f+dense:.1f} pa_sl={pa_sl:.2f} pa_f={pa_f:.1f} lm={lm:.2f} so={so:.1f})")

print("\nDecode totals dict:", {k: round(v, 2) for k, v in decode_tot.items()})
print("Prefill totals dict:", {k: round(v, 1) for k, v in prefill_tot.items()})
