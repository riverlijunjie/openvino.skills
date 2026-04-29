#!/usr/bin/env python3
"""qwen3_5_moe per-token-size kernel-level tables.

For each prefill S in {1024,2048,4096,8192} and each decode kv in {1024..8192},
emit a Markdown table per (platform, mode, size) listing every significant
kernel with single ms / calls/inference / total ms / GB/s / Eff% / Bound.

Mirrors qwen3_moe/build_kernel_tables.py but accounts for hybrid attention.
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
HW={"BMG":dict(bw=456.0,fp16=116.736e3,int8=233.472e3),
    "PTL":dict(bw=110.0, fp16=58.982e3, int8=117.964e3)}

def fc_b4(M,K,N): return K*N//2 + (K//128)*N*2 + (K//128)*N//2 + M*K*2 + M*N*2
def fc_b8(M,K,N): return K*N + (K//128)*N*2 + M*K*2 + M*N*2
def fc_f(M,K,N): return 2*M*K*N

def make_specs():
    return {
        "decode": {
            "moe_3gemm_fused":
                dict(log="moe_decode_M1", calls=NL,
                     bytes=lambda kv: (TK+1)*(3*H*I//2 + 3*(H//128)*I*2 + 3*(H//128)*I//2) + (H+(TK*I+SI)*2)*2,
                     flops=lambda kv: (TK+1)*3*H*I*2, bound="memory"),
            "fc_qkv_full":
                dict(log="fc_qkv_decode_M1", calls=NL_F,
                     bytes=lambda kv: fc_b4(1,H,5120), flops=lambda kv: fc_f(1,H,5120), bound="memory"),
            "fc_o_full":
                dict(log="fc_o_decode_M1", calls=NL_F,
                     bytes=lambda kv: fc_b4(1,NH*HD,H), flops=lambda kv: fc_f(1,NH*HD,H), bound="memory"),
            "fc_linattn_proj":
                dict(log="fc_linattn_proj_decode_M1", calls=NL_L,
                     bytes=lambda kv: fc_b4(1,H,12288), flops=lambda kv: fc_f(1,H,12288), bound="memory"),
            "lm_head":
                dict(log="lm_head_decode_M1", calls=1,
                     bytes=lambda kv: fc_b8(1,H,VOCAB), flops=lambda kv: fc_f(1,H,VOCAB), bound="memory"),
            "paged_attention":
                dict(log=lambda kv: f"pa_decode_kv{kv}", calls=NL_F,
                     bytes=lambda kv: 2*NKV*HD*kv, flops=lambda kv: 2*NH*HD*kv*2, bound="memory"),
            "gated_delta_net":
                dict(log="gdn_decode_T1", calls=NL_L,
                     bytes=lambda kv: 32*128*128*4, flops=lambda kv: 2*32*128*128, bound="memory"),
            "rmsnorm":
                dict(log="so_rmsnorm_h2048_decode", calls=2*NL,
                     bytes=lambda kv: H*4, flops=lambda kv: H*8, bound="memory"),
            "rope_q":
                dict(log="so_rope_q_decode", calls=NL_F,
                     bytes=lambda kv: NH*HD*4, flops=lambda kv: NH*HD*10, bound="memory"),
            "rope_k":
                dict(log="so_rope_k_decode", calls=NL_F,
                     bytes=lambda kv: NKV*HD*4, flops=lambda kv: NKV*HD*10, bound="memory"),
            "q_norm":
                dict(log="so_rmsnorm3d_qnorm_decode", calls=NL_F,
                     bytes=lambda kv: NH*HD*4, flops=lambda kv: NH*HD*8, bound="memory"),
            "k_norm":
                dict(log="so_rmsnorm3d_knorm_decode", calls=NL_F,
                     bytes=lambda kv: NKV*HD*4, flops=lambda kv: NKV*HD*8, bound="memory"),
            "add":
                dict(log="so_add_decode", calls=2*NL,
                     bytes=lambda kv: H*6, flops=lambda kv: H, bound="memory"),
        },
        "prefill": {
            "moe_3gemm_fused":
                dict(log=lambda S: f"moe_prefill_S{S}", calls=NL,
                     bytes=lambda S: 3*H*I*NE//2 + S*(TK+1)*(3*H+3*I)*2,
                     flops=lambda S: S*(TK+1)*3*H*I*2, bound="compute"),
            "fc_qkv_full":
                dict(log=lambda S: f"fc_qkv_prefill_S{S}", calls=NL_F,
                     bytes=lambda S: fc_b8(S,H,5120), flops=lambda S: fc_f(S,H,5120), bound="compute"),
            "fc_o_full":
                dict(log=lambda S: f"fc_o_prefill_S{S}", calls=NL_F,
                     bytes=lambda S: fc_b8(S,NH*HD,H), flops=lambda S: fc_f(S,NH*HD,H), bound="compute"),
            "fc_linattn_proj":
                dict(log=lambda S: f"fc_linattn_proj_prefill_S{S}", calls=NL_L,
                     bytes=lambda S: fc_b8(S,H,12288), flops=lambda S: fc_f(S,H,12288), bound="compute"),
            "paged_attention":
                dict(log=lambda S: f"pa_prefill_S{S}", calls=NL_F,
                     bytes=lambda S: 4*S*NH*HD*2,
                     flops=lambda S: 4*S*S*NH*HD, bound="compute"),
            "gated_delta_net":
                dict(log=lambda S: f"gdn_prefill_S{S}", calls=NL_L,
                     bytes=lambda S: S*32*128*4,
                     flops=lambda S: S*32*128*128*2, bound="compute"),
            "rmsnorm":
                dict(log=lambda S: f"so_rmsnorm_h2048_prefill_S{S}", calls=2*NL,
                     bytes=lambda S: S*H*4, flops=lambda S: S*H*8, bound="memory"),
            "rope_q":
                dict(log=lambda S: f"so_rope_q_prefill_S{S}", calls=NL_F,
                     bytes=lambda S: S*NH*HD*4, flops=lambda S: S*NH*HD*10, bound="memory"),
        },
    }

SPECS = make_specs()


def get_kernels(metrics, log):
    e = metrics.get(log)
    if not e: return []
    out = []
    for entry in e.get("per_kernel", []):
        if not entry or len(entry) < 3: continue
        n, ns, c = entry
        nn = re.sub(r'_\d{15,}', '', n)
        nn = re.sub(r'_+', '_', nn).strip('_')
        out.append((nn, ns/1e6, c))
    out.sort(key=lambda r:-r[1])
    return out


def emit_table(plat, mode, x, peak_compute):
    bw = HW[plat]["bw"]; metrics = B if plat=="BMG" else P; specs = SPECS[mode]
    rows = []
    for opname, spec in specs.items():
        log = spec["log"](x) if callable(spec["log"]) else spec["log"]
        if log is None: continue
        kernels = get_kernels(metrics, log)
        if not kernels: continue
        op_total = sum(k[1] for k in kernels)
        op_b, op_f = spec["bytes"](x), spec["flops"](x)
        calls = spec["calls"]; bound = spec["bound"]
        cum = 0; sig = []
        for k in kernels:
            sig.append(k); cum += k[1]
            if cum >= 0.95*op_total: break
        for kn, kms, _ in sig:
            sh = kms/op_total if op_total else 0
            kb, kf = op_b*sh, op_f*sh
            gbs = kb/(kms*1e6) if kms else 0
            gflops = kf/(kms*1e6) if kms else 0
            eff = gbs/bw*100 if bound=="memory" else gflops/peak_compute*100
            rows.append((opname, kn, kms, calls, kms*calls, gflops, gbs, eff, bound))
    rows.sort(key=lambda r:-r[4])
    print(f"\n#### {plat} — {mode} {'S' if mode=='prefill' else 'kv'}={x}\n")
    print("| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    tot = 0
    for r in rows:
        tot += r[4]
        print(f"| {r[0]} | `{r[1]}` | {r[2]:.4f} | {r[3]} | {r[4]:.3f} | {r[5]:.0f} | {r[6]:.1f} | {r[7]:.1f}% | {r[8]} |")
    print(f"\n**Total inference time (this stage)** ≈ **{tot:.2f} ms**")


def main():
    SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    print("# qwen3_5_moe — Per-token-size Kernel Tables\n")
    for plat in ["BMG","PTL"]:
        peak_dec = HW[plat]["fp16"]; peak_pre = HW[plat]["int8"]
        print(f"\n## Platform: {plat}\n")
        print("### DECODE — per kv (M=1)\n")
        for kv in SIZES: emit_table(plat, "decode", kv, peak_dec)
        print(f"\n### PREFILL — per S (compute-bound uses INT8 XMX peak = {peak_pre/1e3:.1f} TOPS)\n")
        for S in SIZES: emit_table(plat, "prefill", S, peak_pre)


if __name__ == "__main__": main()
