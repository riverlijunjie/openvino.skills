#!/usr/bin/env python3
"""Qwen3.5-MoE-35B-A3B per-op + end-to-end roofline report.

Mirrors qwen3_moe/build_report.py but accounts for hybrid attention:
  - 10 full-attn layers (PA, NH=16/NKV=2/HD=256, fc_qkv 2048x5120, fc_o 4096x2048)
  - 30 linear-attn layers (GatedDeltaNet, fc_linattn_proj 2048x12288)
  - 40 MoE layers with always-on shared expert (I=512)
  - LM head INT8 vocab=248320
"""
import argparse, json
from pathlib import Path

OUT = Path(__file__).resolve().parent
HW = {
    "BMG": dict(bw=456.0, fp16=116.736, int8=233.472,
                desc="BMG (Arc B580, 2850 MHz, 456 GB/s, FP16 XMX 116.736 TFLOPS, INT8 XMX 233.472 TOPS)"),
    "PTL": dict(bw=110.0, fp16=58.982, int8=117.964,
                desc="PTL (Panther Lake iGPU, 110 GB/s, FP16 XMX 58.982 TFLOPS, INT8 XMX 117.964 TOPS)"),
}

_p = argparse.ArgumentParser()
_p.add_argument("--platform", default="BMG", choices=["BMG","PTL"])
_p.add_argument("--metrics", default=None)
_p.add_argument("--out", default=None)
_a = _p.parse_args()

PLAT = _a.platform; HWD = HW[PLAT]
BW, FP16, INT8 = HWD["bw"], HWD["fp16"], HWD["int8"]
M = json.loads((Path(_a.metrics) if _a.metrics else OUT/f"{PLAT.lower()}_metrics.json").read_text())
CFG = json.loads((OUT/"ops_mapping.json").read_text())["config"]

NL_FULL = CFG["num_full_attention_layers"]   # 10
NL_LIN  = CFG["num_linear_attention_layers"] # 30
NL_TOTAL = CFG["num_hidden_layers"]          # 40
H = CFG["hidden_size"]
NH, NKV, HD = CFG["num_attention_heads"], CFG["num_key_value_heads"], CFG["head_dim"]
LIN_NV, LIN_HD = CFG["linear_num_value_heads"], CFG["linear_value_head_dim"]
I = CFG["moe_intermediate_size"]
SI = CFG["shared_expert_intermediate_size"]
NE, TK = CFG["num_experts"], CFG["num_experts_per_tok"]
VOCAB = CFG["vocab_size"]


def per_iter_ms(log): return M.get(log, {}).get("total_kernel_ns", 0)/1e6


def fc_bytes_int4(M_, K, N, g=128):
    return K*N//2 + (K//g)*N*2 + (K//g)*N//2 + M_*K*2 + M_*N*2

def fc_bytes_int8(M_, K, N, g=128):
    return K*N + (K//g)*N*2 + M_*K*2 + M_*N*2

def fc_flops(M_, K, N): return 2*M_*K*N


def fc_row(name, log, calls, K, N, M_=1, prec="u4"):
    ms = per_iter_ms(log)
    if ms == 0: return None
    b = fc_bytes_int4(M_, K, N) if prec=="u4" else fc_bytes_int8(M_, K, N)
    f = fc_flops(M_, K, N)
    gbs = b/(ms*1e6); tflops = f/(ms*1e9)/1000
    return dict(name=name, ms=ms, calls=calls, total=ms*calls,
                ai=f/b, gbs=gbs, tops=tflops, eff=gbs/BW*100, bound="memory")


def moe_decode_row():
    log = "moe_decode_M1"
    ms = per_iter_ms(log)
    if ms == 0: return None
    # TK active routed experts (INT4) + 1 always-on shared expert (INT4)
    routed_b = TK*(3*H*I//2 + 3*(H//128)*I*2 + 3*(H//128)*I//2)
    shared_b = (3*H*SI//2 + 3*(H//128)*SI*2 + 3*(H//128)*SI//2)
    total_b = routed_b + shared_b + (H + (TK*I + SI)*2)*2
    flops = (TK + 1)*3*H*max(I,SI)*2
    gbs = total_b/(ms*1e6)
    return dict(name=f"moe_3gemm fused (TK={TK}+shared, NE={NE}, I={I}, SI={SI})",
                ms=ms, calls=NL_TOTAL, total=ms*NL_TOTAL,
                ai=flops/total_b, gbs=gbs, tops=flops/(ms*1e9)/1000,
                eff=gbs/BW*100, bound="memory")


def pa_decode_row(kv=4096):
    log = f"pa_decode_kv{kv}"; ms = per_iter_ms(log)
    if ms == 0: return None
    kv_b = 2*NKV*HD*kv
    f = 2*NH*HD*kv*2
    gbs = kv_b/(ms*1e6)
    return dict(name=f"pa_decode (kv={kv}, NH={NH}/NKV={NKV}/HD={HD})",
                ms=ms, calls=NL_FULL, total=ms*NL_FULL,
                ai=f/kv_b, gbs=gbs, tops=f/(ms*1e9)/1000,
                eff=gbs/BW*100, bound="memory")


def gdn_decode_row():
    log = "gdn_decode_T1"; ms = per_iter_ms(log)
    if ms == 0: return None
    # GDN state size dominates: HK*K*V*2 (FP16 state per head); read+write
    state_b = 32*128*128*2*2  # HK*K*V*2 bytes (read+write per token)
    f = 2*32*128*128
    gbs = state_b/(ms*1e6)
    return dict(name="gated_delta_net (HK=32, K=V=128)",
                ms=ms, calls=NL_LIN, total=ms*NL_LIN,
                ai=f/state_b, gbs=gbs, tops=f/(ms*1e9)/1000,
                eff=gbs/BW*100, bound="memory")


def small_op_row(name, log, calls, b, f):
    ms = per_iter_ms(log)
    if ms == 0: return None
    gbs = b/(ms*1e6)
    return dict(name=name, ms=ms, calls=calls, total=ms*calls,
                ai=f/b, gbs=gbs, tops=f/(ms*1e9)/1000, eff=gbs/BW*100, bound="memory")


rows = []
rows.append(moe_decode_row())
rows.append(fc_row("lm_head (INT8 g=128)", "lm_head_decode_M1", 1, H, VOCAB, prec="u8"))
rows.append(fc_row("fc_qkv full-attn (INT4)", "fc_qkv_decode_M1", NL_FULL, H, 5120))
rows.append(fc_row("fc_o full-attn (INT4)", "fc_o_decode_M1", NL_FULL, NH*HD, H))
rows.append(fc_row("fc_linattn_proj (INT4)", "fc_linattn_proj_decode_M1", NL_LIN, H, 12288))
DECODE_KV = 4096
rows.append(pa_decode_row(DECODE_KV))
rows.append(gdn_decode_row())
rows.append(small_op_row("rmsnorm (H=2048)", "so_rmsnorm_h2048_decode", 2*NL_TOTAL, H*4, H*8))
rows.append(small_op_row("rope_q (full_attn)", "so_rope_q_decode", NL_FULL, NH*HD*4, NH*HD*10))
rows.append(small_op_row("rope_k (full_attn)", "so_rope_k_decode", NL_FULL, NKV*HD*4, NKV*HD*10))
rows.append(small_op_row("q_norm (full_attn)", "so_rmsnorm3d_qnorm_decode", NL_FULL, NH*HD*4, NH*HD*8))
rows.append(small_op_row("k_norm (full_attn)", "so_rmsnorm3d_knorm_decode", NL_FULL, NKV*HD*4, NKV*HD*8))
rows.append(small_op_row("add (residual)", "so_add_decode", 2*NL_TOTAL, H*6, H))
rows = [r for r in rows if r]
rows.sort(key=lambda r:-r["total"])

print(f"\n## DECODE per-op (kv=4096, {PLAT})")
print(f"_{HWD['desc']}_")
print()
print("| Op | Avg ms | Calls | Total ms | AI | GB/s | Eff% | Bound |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
total_decode = 0
for r in rows:
    total_decode += r["total"]
    print(f"| {r['name']} | {r['ms']:.4f} | {r['calls']} | {r['total']:.3f} | {r['ai']:.2f} | {r['gbs']:.1f} | {r['eff']:.1f}% | {r['bound']} |")
print(f"\n**Decode total** = {total_decode:.3f} ms → **{1000/total_decode:.1f} tok/s**")

# Decode totals across all kv sizes
print(f"\n## DECODE total across kv ({PLAT})\n")
print("| kv | MoE/L | PA/L | GDN/L | fc_linattn/L | lm_head | total ms | tok/s |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|")
decode_totals = {}
lm_ms = per_iter_ms("lm_head_decode_M1")
moe_ms = per_iter_ms("moe_decode_M1")
fcq_ms = per_iter_ms("fc_qkv_decode_M1")
fco_ms = per_iter_ms("fc_o_decode_M1")
fcl_ms = per_iter_ms("fc_linattn_proj_decode_M1")
gdn_ms = per_iter_ms("gdn_decode_T1")
rms_ms = per_iter_ms("so_rmsnorm_h2048_decode")
add_ms = per_iter_ms("so_add_decode")
rope_q_ms = per_iter_ms("so_rope_q_decode")
rope_k_ms = per_iter_ms("so_rope_k_decode")
qn_ms = per_iter_ms("so_rmsnorm3d_qnorm_decode")
kn_ms = per_iter_ms("so_rmsnorm3d_knorm_decode")
for kv in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
    pa_ms = per_iter_ms(f"pa_decode_kv{kv}")
    if pa_ms == 0: continue
    tot = (NL_TOTAL*moe_ms
           + NL_FULL*(fcq_ms + fco_ms + pa_ms + rope_q_ms + rope_k_ms + qn_ms + kn_ms)
           + NL_LIN*(fcl_ms + gdn_ms)
           + lm_ms
           + 2*NL_TOTAL*(rms_ms + add_ms))
    decode_totals[kv] = tot
    print(f"| {kv} | {moe_ms:.4f} | {pa_ms:.4f} | {gdn_ms:.4f} | {fcl_ms:.4f} | {lm_ms:.3f} | {tot:.3f} | {1000/tot:.1f} |")

# Prefill totals
print(f"\n## PREFILL per-S kernel time (ms, {PLAT})")
print()
print("| S | MoE/L | PA full/L | fc_qkv/L | fc_o/L | GDN/L | linattn_proj/L | total est ms |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|")
prefill = {}
for S in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
    moe = per_iter_ms(f"moe_prefill_S{S}")
    pa  = per_iter_ms(f"pa_prefill_S{S}")
    fcq = per_iter_ms(f"fc_qkv_prefill_S{S}")
    fco = per_iter_ms(f"fc_o_prefill_S{S}")
    gdn = per_iter_ms(f"gdn_prefill_S{S}")
    lin = per_iter_ms(f"fc_linattn_proj_prefill_S{S}")
    if moe == 0: continue
    # OOM detection: if MoE per-layer << 1 ms at large S, the run failed
    if moe < 0.5 and S >= 32768:
        print(f"| {S} | OOM | {pa:.3f} | {fcq:.3f} | {fco:.3f} | {gdn:.3f} | {lin:.3f} | OOM ({PLAT} dGPU VRAM exhausted) |")
        continue
    lm = per_iter_ms("lm_head_decode_M1")
    total = (NL_TOTAL*moe
             + NL_FULL*(pa + fcq + fco)
             + NL_LIN*(gdn + lin)
             + lm)
    prefill[S] = total
    print(f"| {S} | {moe:.3f} | {pa:.3f} | {fcq:.3f} | {fco:.3f} | {gdn:.3f} | {lin:.3f} | {total:.2f} |")

if _a.out:
    Path(_a.out).write_text(json.dumps({
        "platform": PLAT, "platform_desc": HWD["desc"],
        "decode_total_ms": total_decode, "decode_rows": rows,
        "decode_totals_by_kv": decode_totals,
        "prefill_total_ms": prefill,
    }, indent=2, default=str))
    print(f"\nSaved {_a.out}")
