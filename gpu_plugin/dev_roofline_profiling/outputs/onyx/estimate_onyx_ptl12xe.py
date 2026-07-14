#!/usr/bin/env python3
"""Analytical roofline estimation of the Onyx text decoder on PTL 12Xe (B390 iGPU).

This is an ANALYTICAL ESTIMATE (no on-device run). Per-kernel efficiency factors
are reused from *measured* gemma4-12B kernels on the SAME platform (PTL 12Xe,
INT4 g=128 body + INT8 g=128 LM_head, PagedAttention OpenCL+micro-kernel). Onyx
and gemma4-12B are both dense SwiGLU/GEGLU decoders that dispatch the identical
FullyConnectedCompressed `gemm_kernel` and `paged_attention` primitives, so the
gemma efficiency envelope transfers directly. Bytes/FLOPs are recomputed for
Onyx's own shapes; only the efficiency (t_theo / t_meas) is borrowed.

Differences vs the gemma reference that are modelled here:
  * H=6656, L=52, intermediate=19968, vocab=202048 (Onyx values)
  * uniform GQA 32/2 heads, HD=128 on every layer (Q=4096, KV=256)
  * extra per-layer output_gate_proj FC (6656 -> 4096) from use_attn_output_gate
  * sliding_window = 2048 (39 sliding/RoPE layers, 13 full/NoPE layers)
  * KV cache INT4 g=128 (user request) instead of gemma's INT8
"""
import json
import math
import os

# ----------------------------------------------------------------------------
# Hardware — PTL 12Xe (B390 iGPU) @ 2400 MHz
# ----------------------------------------------------------------------------
BW = 110.0e9                       # bytes/s (LPDDR5x peak)
FP16_XMX = 12 * 8 * 256 * 2.4e9    # 58.9824 TFLOPS
INT8_XMX = 2 * FP16_XMX            # 117.9648 TOPS
OVERHEAD = 0.95                    # deduct 5% for unavoidable real-world loss
BW_A = BW * OVERHEAD
FP16_A = FP16_XMX * OVERHEAD
INT8_A = INT8_XMX * OVERHEAD

# ----------------------------------------------------------------------------
# Onyx text-decoder config
# ----------------------------------------------------------------------------
H = 6656
L = 52
NH, NKV, HD = 32, 2, 128
QDIM = NH * HD          # 4096
KVDIM = NKV * HD        # 256
INTER = 19968
VOCAB = 202048
SW = 2048
G = 128                 # quant group size
# layer split: layer_idx%4==3 -> full/NoPE, else sliding/RoPE
FULL_LAYERS = sum(1 for i in range(L) if i % 4 == 3)     # 13
SLIDING_LAYERS = L - FULL_LAYERS                          # 39

SIZES = [1024, 2048, 4096, 8192, 16384, 32768]

# ----------------------------------------------------------------------------
# Byte models
# ----------------------------------------------------------------------------
def int4_w_bytes(K, N):
    return N * K / 2 + N * (K / G) * 2 + N * (K / G) * 0.5   # int4 + fp16 scale + int4 zp

def int8_w_bytes(K, N):
    return N * K + N * (K / G) * 2                            # int8 + fp16 scale

def fc_decode(K, N, quant="int4"):
    wb = int4_w_bytes(K, N) if quant == "int4" else int8_w_bytes(K, N)
    b = wb + K * 2 + N * 2               # +act(fp16) +out(fp16), M=1
    f = 2 * K * N
    return b, f

def fc_prefill(S, K, N, quant="int4"):
    wb = int4_w_bytes(K, N) if quant == "int4" else int8_w_bytes(K, N)
    b = wb + S * K * 2 + S * N * 2
    f = 2 * S * K * N
    return b, f

# PagedAttention (INT4 KV cache g=128 over HD)
def pa_decode(kv_eff):
    # k+v int4 + fp16 scale + int4 zp, per kv token, both k and v
    kv_bytes = (NKV * HD * 0.5) * 2 \
             + (NKV * (HD / G) * 2) * 2 \
             + (NKV * (HD / G) * 0.5) * 2
    b = kv_bytes * kv_eff + NH * HD * 2 + NH * HD * 2   # +Q read +out write
    f = 4 * NH * HD * kv_eff                            # QK^T + PV
    return b, f

def pa_prefill(S, kv_eff, causal_pairs):
    # KV read (int4) once + Q + out over S rows
    kv_bytes = (NKV * HD * 0.5) * 2 + (NKV * (HD / G) * 2) * 2 + (NKV * (HD / G) * 0.5) * 2
    b = kv_bytes * kv_eff + S * NH * HD * 2 + S * NH * HD * 2
    f = 4 * NH * HD * causal_pairs
    return b, f

# ----------------------------------------------------------------------------
# Efficiency envelope (from measured gemma4-12B on PTL 12Xe)
# ----------------------------------------------------------------------------
EFF_DEC_FC   = 0.92    # INT4 FC decode, memory-bound
EFF_DEC_LM   = 0.92    # INT8 LM_head decode
# PA decode efficiency vs effective-kv (interpolated from gemma PA curve)
def eff_pa_dec_sliding(kv_eff):
    return 0.287 if kv_eff <= 256 else 0.486
def eff_pa_dec_full(kv):
    pts = [(256,0.057),(1024,0.094),(2048,0.111),(4096,0.255),
           (8192,0.259),(16384,0.291),(32768,0.308)]
    for k,e in pts:
        if kv <= k: return e
    return 0.308

# Prefill FC XMX efficiency (INT8 XMX path), from gemma section 7.3 (S-dependent, ~flat)
EFF_PRE = {"qkv":0.575, "outgate":0.575, "o":0.535, "gate":0.61, "up":0.61, "down":0.52}
def eff_pa_pre_sliding(S):
    return 0.33
def eff_pa_pre_full(S):
    return 0.07

# ----------------------------------------------------------------------------
# Small ops (rmsnorm x4/layer + qk_norm + rope + 2 residual add + out-gate)
# memory-bound tiny kernels; efficiency ~0.45 (launch-overhead dominated)
# ----------------------------------------------------------------------------
EFF_SMALL = 0.45
def small_ops_decode():
    # per layer, M=1: 4 rmsnorm(H), qk_norm(Q+K), rope(Q+K sliding), 2 add(H), gate sigmoid+mul(Q)
    per_layer = (4 * H * 2 * 2)          # 4 rmsnorm r/w
    per_layer += (QDIM + KVDIM) * 2 * 2  # qk_norm r/w
    per_layer += (QDIM) * 2 * 2          # out-gate sigmoid+mul
    per_layer += 2 * H * 2 * 2           # 2 residual adds r/w
    rope = (QDIM + KVDIM) * 2 * 2        # rope on sliding layers only
    total_b = per_layer * L + rope * SLIDING_LAYERS + H * 2 * 2  # + final norm
    return total_b / BW_A / EFF_SMALL * 1e3   # ms

def small_ops_prefill(S):
    per_layer = (4 * H * 2 * 2)
    per_layer += (QDIM + KVDIM) * 2 * 2
    per_layer += (QDIM) * 2 * 2
    per_layer += 2 * H * 2 * 2
    rope = (QDIM + KVDIM) * 2 * 2
    total_b = (per_layer * L + rope * SLIDING_LAYERS + H * 2 * 2) * S
    return total_b / BW_A / EFF_SMALL * 1e3

# ----------------------------------------------------------------------------
# Per-op builders
# ----------------------------------------------------------------------------
def t_mem(b, eff):   return b / BW_A / eff * 1e3
def t_cmp(f, peak, eff): return f / (peak * eff) * 1e3

FC_LAYERS = {          # (K, N, per-op-count-across-all-layers, eff_key)
    "FC_QKV":   (H, QDIM + 2 * KVDIM, L, "qkv"),
    "FC_OutGate": (H, QDIM, L, "outgate"),
    "FC_O":     (QDIM, H, L, "o"),
    "MLP_gate": (H, INTER, L, "gate"),
    "MLP_up":   (H, INTER, L, "up"),
    "MLP_down": (INTER, H, L, "down"),
}

def build_decode(kv):
    rows = []
    # FC ops (M=1, memory-bound) — identical every layer
    for name,(K,N,cnt,_) in FC_LAYERS.items():
        b,f = fc_decode(K,N)
        t = t_mem(b, EFF_DEC_FC)
        rows.append(dict(op=name, kernel="gemm_kernel", single_ms=t, calls=cnt,
                         total_ms=t*cnt, gflops=f/(t*1e-3)/1e9, gbs=b/(t*1e-3)/1e9,
                         eff=EFF_DEC_FC*100, bound="memory"))
    # LM_head INT8
    b,f = fc_decode(H, VOCAB, "int8")
    t = t_mem(b, EFF_DEC_LM)
    rows.append(dict(op="LM_head", kernel="gemm_kernel", single_ms=t, calls=1,
                     total_ms=t, gflops=f/(t*1e-3)/1e9, gbs=b/(t*1e-3)/1e9,
                     eff=EFF_DEC_LM*100, bound="memory"))
    # PA sliding
    kv_s = min(kv, SW)
    b,f = pa_decode(kv_s); e = eff_pa_dec_sliding(kv_s); t = t_mem(b,e)
    rows.append(dict(op="PA_sliding", kernel="paged_attention", single_ms=t,
                     calls=SLIDING_LAYERS, total_ms=t*SLIDING_LAYERS,
                     gflops=f/(t*1e-3)/1e9, gbs=b/(t*1e-3)/1e9, eff=e*100, bound="memory"))
    # PA full
    b,f = pa_decode(kv); e = eff_pa_dec_full(kv); t = t_mem(b,e)
    rows.append(dict(op="PA_full", kernel="paged_attention", single_ms=t,
                     calls=FULL_LAYERS, total_ms=t*FULL_LAYERS,
                     gflops=f/(t*1e-3)/1e9, gbs=b/(t*1e-3)/1e9, eff=e*100, bound="memory"))
    # small ops
    so = small_ops_decode()
    rows.append(dict(op="SmallOps", kernel="rms/rope/gate/add", single_ms=0, calls=0,
                     total_ms=so, gflops=0, gbs=0, eff=0, bound="memory"))
    total = sum(r["total_ms"] for r in rows)
    rows.sort(key=lambda r:-r["total_ms"])
    return rows, total

def build_prefill(S):
    rows = []
    for name,(K,N,cnt,ek) in FC_LAYERS.items():
        b,f = fc_prefill(S,K,N)
        e = EFF_PRE[ek]; t = t_cmp(f, INT8_XMX, e)
        rows.append(dict(op=name, kernel="dq+gemm_kernel", single_ms=t, calls=cnt,
                         total_ms=t*cnt, gflops=f/(t*1e-3)/1e9, gbs=b/(t*1e-3)/1e9,
                         eff=e*100, bound="compute"))
    # LM_head only last token in prefill (M=1)
    b,f = fc_decode(H, VOCAB, "int8"); t = t_mem(b, EFF_DEC_LM)
    rows.append(dict(op="LM_head", kernel="gemm_kernel", single_ms=t, calls=1, total_ms=t,
                     gflops=f/(t*1e-3)/1e9, gbs=b/(t*1e-3)/1e9, eff=EFF_DEC_LM*100, bound="memory"))
    # PA sliding prefill: causal band width min(S,SW)
    kv_s = min(S, SW)
    pairs_s = S*kv_s - (kv_s*(kv_s-1)/2 if S>=kv_s else S*(S-1)/2)
    b,f = pa_prefill(S, kv_s, pairs_s); e = eff_pa_pre_sliding(S); t = t_cmp(f, FP16_XMX, e)
    rows.append(dict(op="PA_sliding", kernel="sdpa_micro_prefill", single_ms=t,
                     calls=SLIDING_LAYERS, total_ms=t*SLIDING_LAYERS,
                     gflops=f/(t*1e-3)/1e9, gbs=b/(t*1e-3)/1e9, eff=e*100, bound="compute"))
    # PA full prefill: causal pairs S(S+1)/2
    pairs_f = S*(S+1)/2
    b,f = pa_prefill(S, S, pairs_f); e = eff_pa_pre_full(S); t = t_cmp(f, FP16_XMX, e)
    rows.append(dict(op="PA_full", kernel="sdpa_micro_prefill", single_ms=t,
                     calls=FULL_LAYERS, total_ms=t*FULL_LAYERS,
                     gflops=f/(t*1e-3)/1e9, gbs=b/(t*1e-3)/1e9, eff=e*100, bound="compute"))
    so = small_ops_prefill(S)
    rows.append(dict(op="SmallOps", kernel="rms/rope/gate/add", single_ms=0, calls=0,
                     total_ms=so, gflops=0, gbs=0, eff=0, bound="memory"))
    total = sum(r["total_ms"] for r in rows)
    rows.sort(key=lambda r:-r["total_ms"])
    return rows, total

# ----------------------------------------------------------------------------
# Weight footprint
# ----------------------------------------------------------------------------
def weight_footprint():
    items = []
    def add(name, K, N, quant, layers):
        wb = int4_w_bytes(K,N) if quant=="int4" else int8_w_bytes(K,N)
        items.append((name, f"{K}x{N}", quant, wb, layers, wb*layers/1e6))
    add("Embedding", H, VOCAB, "int8", 1)   # not tied
    add("FC_QKV", H, QDIM+2*KVDIM, "int4", L)
    add("FC_OutGate", H, QDIM, "int4", L)
    add("FC_O", QDIM, H, "int4", L)
    add("MLP_gate", H, INTER, "int4", L)
    add("MLP_up", H, INTER, "int4", L)
    add("MLP_down", INTER, H, "int4", L)
    add("LM_Head", H, VOCAB, "int8", 1)
    total = sum(i[5] for i in items)
    return items, total

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------
out = {"platform":"PTL_12Xe","model":"onyx (text decoder)",
       "config":"INT4 g=128 body + INT8 g=128 LM_head + INT4 g=128 KV cache, FP16 act",
       "bw_gbs":110.0, "fp16_tflops":FP16_XMX/1e12, "int8_tops":INT8_XMX/1e12,
       "ridge_f16":FP16_A/BW_A, "ridge_i8":INT8_A/BW_A,
       "sizes":SIZES, "decode":{}, "prefill":{}}

print("="*72)
print("ONYX text decoder — PTL 12Xe analytical roofline estimate")
print(f"FP16 XMX {FP16_XMX/1e12:.3f} TFLOPS | INT8 XMX {INT8_XMX/1e12:.3f} TOPS | BW {BW/1e9:.0f} GB/s")
print(f"Ridge F16 {FP16_A/BW_A:.1f} | Ridge I8 {INT8_A/BW_A:.1f}")
print(f"Layers: {SLIDING_LAYERS} sliding/RoPE + {FULL_LAYERS} full/NoPE")
items, wtot = weight_footprint()
print(f"\nStatic weights: {wtot:.0f} MB")
for n,sh,q,wb,ly,mb in items:
    print(f"  {n:12s} {sh:14s} {q:5s} x{ly:3d} = {mb:8.1f} MB")

def theo_total(rows):
    t = 0.0
    for r in rows:
        e = (r["eff"]/100.0) if r["eff"] > 0 else EFF_SMALL
        t += r["total_ms"] * e
    return t

print("\n--- DECODE (per output token) ---")
for kv in SIZES:
    rows,total = build_decode(kv)
    theo = theo_total(rows)
    out["decode"][str(kv)] = {"total_ms":total, "theo_ms":theo,
                              "achieved_pct":theo/total*100, "rows":rows}
    print(f"KV={kv:6d}: TPOT {total:7.2f} ms  ({1000/total:5.1f} tok/s)"
          f"  theo {theo:7.2f} ms  achieved {theo/total*100:5.1f}%")

print("\n--- PREFILL (TTFT over S tokens) ---")
for S in SIZES:
    rows,total = build_prefill(S)
    theo = theo_total(rows)
    out["prefill"][str(S)] = {"total_ms":total, "theo_ms":theo,
                              "achieved_pct":theo/total*100, "rows":rows}
    print(f"S={S:6d}: TTFT {total:9.2f} ms  ({S/total*1000:7.1f} tok/s)"
          f"  theo {theo:9.2f} ms  achieved {theo/total*100:5.1f}%")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here,"performance_metrics.json"),"w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {os.path.join(here,'performance_metrics.json')}")
