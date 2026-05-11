#!/usr/bin/env python3
"""Compute headline decode/prefill totals from parsed cliloader metrics.

Reads bmg_metrics.json and ptl_metrics.json in this directory, plus
ops_mapping.json for shapes/calls, and prints decode-totals and prefill
per-op kernel-time tables for both platforms in markdown form.
"""
import json, re
from pathlib import Path

OUT = Path(__file__).resolve().parent
B = json.loads((OUT/"bmg_metrics.json").read_text())
P = json.loads((OUT/"ptl_metrics.json").read_text())
CFG = json.loads((OUT/"ops_mapping.json").read_text())["config"]
NL_F=CFG["num_full_attention_layers"]; NL_L=CFG["num_linear_attention_layers"]; NL=CFG["num_hidden_layers"]
H=CFG["hidden_size"]; NH=CFG["num_attention_heads"]; NKV=CFG["num_key_value_heads"]; HD=CFG["head_dim"]
I=CFG["moe_intermediate_size"]; SI=CFG["shared_expert_intermediate_size"]
TK=CFG["num_experts_per_tok"]; NE=CFG["num_experts"]; VOCAB=CFG["vocab_size"]
SIZES=[1024,2048,4096,8192,16384,32768,65536,131072]

def total_ns(metrics, log):
    e = metrics.get(log)
    return (e["total_kernel_ns"] if e else 0) / 1e6

def decode_totals(metrics):
    rows=[]
    moe = total_ns(metrics, "moe_decode_M1")
    lm  = total_ns(metrics, "lm_head_decode_M1")
    fcq = total_ns(metrics, "fc_qkv_decode_M1")
    fco = total_ns(metrics, "fc_o_decode_M1")
    fcl = total_ns(metrics, "fc_linattn_proj_decode_M1")
    gdn = total_ns(metrics, "gdn_decode_T1")
    rms = total_ns(metrics, "so_rmsnorm_h2048_decode")
    add = total_ns(metrics, "so_add_decode")
    rq  = total_ns(metrics, "so_rope_q_decode")
    rk  = total_ns(metrics, "so_rope_k_decode")
    qn  = total_ns(metrics, "so_rmsnorm3d_qnorm_decode")
    kn  = total_ns(metrics, "so_rmsnorm3d_knorm_decode")
    for kv in SIZES:
        pa = total_ns(metrics, f"pa_decode_kv{kv}")
        per_layer_full = pa + fcq + fco + rq + rk + qn + kn
        per_layer_lin  = fcl + gdn
        small = 2*NL*(rms+add)
        total = NL*moe + NL_F*per_layer_full + NL_L*per_layer_lin + lm + small
        rows.append(dict(kv=kv, moe=moe, pa=pa, gdn=gdn, fcl=fcl, lm=lm,
                         fcq=fcq, fco=fco, total=total))
    return rows

def prefill_totals(metrics):
    rows=[]
    lm = total_ns(metrics, "lm_head_decode_M1")  # lm_head reused (single token)
    for S in SIZES:
        moe = total_ns(metrics, f"moe_prefill_S{S}")
        pa  = total_ns(metrics, f"pa_prefill_S{S}")
        fcq = total_ns(metrics, f"fc_qkv_prefill_S{S}")
        fco = total_ns(metrics, f"fc_o_prefill_S{S}")
        fcl = total_ns(metrics, f"fc_linattn_proj_prefill_S{S}")
        gdn = total_ns(metrics, f"gdn_prefill_S{S}")
        rms = total_ns(metrics, f"so_rmsnorm_h2048_prefill_S{S}")
        rq  = total_ns(metrics, f"so_rope_q_prefill_S{S}")
        per_layer_full = pa + fcq + fco + rq
        per_layer_lin  = fcl + gdn
        small = 2*NL*rms
        total = NL*moe + NL_F*per_layer_full + NL_L*per_layer_lin + lm + small
        rows.append(dict(S=S, moe=moe, pa=pa, fcq=fcq, fco=fco, gdn=gdn,
                         fcl=fcl, lm=lm, total=total))
    return rows

def fmt(v, w=8, p=3):
    if v >= 1e4: return f"{v:>{w},.0f}"
    return f"{v:>{w}.{p}f}"

print("## DECODE totals\n")
print("| Platform | kv | MoE/L | PA/L | GDN/L | linattn/L | lm | total ms | tok/s |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for plat,M in [("BMG",B),("PTL",P)]:
    for r in decode_totals(M):
        print(f"| {plat} | {r['kv']:,} | {r['moe']:.4f} | {r['pa']:.4f} | {r['gdn']:.4f} | {r['fcl']:.4f} | {r['lm']:.3f} | **{r['total']:.2f}** | **{1000/r['total']:.1f}** |")

print("\n## PREFILL totals\n")
print("| Platform | S | MoE/L | PA full/L | fc_qkv/L | fc_o/L | GDN/L | linattn_proj/L | total ms |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for plat,M in [("BMG",B),("PTL",P)]:
    for r in prefill_totals(M):
        if r['moe']==0:  # OOM
            print(f"| {plat} | {r['S']:,} | OOM | — | — | — | — | — | OOM |")
        else:
            print(f"| {plat} | {r['S']:,} | {r['moe']:.3f} | {r['pa']:.3f} | {r['fcq']:.3f} | {r['fco']:.3f} | {r['gdn']:.3f} | {r['fcl']:.3f} | **{r['total']:,.0f}** |")
