#!/usr/bin/env python3
"""Generate per-token-size kernel-level tables for qwen3_moe SUMMARY.

For each prefill S in {1024..131072} and each decode kv in {1024..131072},
emit a Markdown table with one row per significant kernel:
  ops_name | kernel_name | single ms | calls/inference | total ms |
  GFLOPS  | GB/s | eff% | bound

Outputs to stdout (redirect into SUMMARY).
"""
import json, re
from pathlib import Path

OUT = Path(__file__).resolve().parent
B = json.loads((OUT / "bmg_metrics.json").read_text())
P = json.loads((OUT / "ptl_metrics.json").read_text())
CFG = json.loads((OUT / "ops_mapping.json").read_text())["config"]

NL = CFG["num_hidden_layers"]
H = CFG["hidden_size"]
NH = CFG["num_attention_heads"]
NKV = CFG["num_key_value_heads"]
HD = CFG["head_dim"]
I = CFG["moe_intermediate_size"]
TK = CFG["num_experts_per_tok"]
NE = CFG["num_experts"]
VOCAB = CFG["vocab_size"]

HW = {
    "BMG": dict(bw=456.0, fp16=116.736e3, int8=233.472e3),  # GFLOPS / GOPS
    "PTL": dict(bw=110.0, fp16=58.982e3, int8=117.964e3),
}

# ----- theoretical bytes / flops per (op, mode) at given M/S/kv -----
def fc_bytes_int4(M, K, N):
    return K*N//2 + (K//128)*N*2 + (K//128)*N//2 + M*K*2 + M*N*2

def fc_bytes_int8(M, K, N):
    return K*N + (K//128)*N*2 + M*K*2 + M*N*2

def fc_flops(M, K, N):
    return 2*M*K*N

# Op spec: ( ops_label, kernel_log_key, calls_per_inference,
#           bytes_func(meta), flops_func(meta), prec_for_compute, model_label )
# meta: dict for each call (M for FC, S for prefill, kv for decode etc.)
def make_specs():
    return {
        # ----- DECODE -----
        "decode": {
            "moe_3gemm_fused":
                dict(log="moe_decode_M1", calls=NL,
                     # active per-token weight bytes (TK active experts × 3 GEMMs)
                     bytes=lambda kv: TK*(3*H*I//2 + 3*(H//128)*I*2 + 3*(H//128)*I//2) + (H+TK*I*2)*2,
                     flops=lambda kv: TK*3*H*I*2,
                     bound="memory"),
            "fc_qkv":
                dict(log="fc_qkv_decode_M1", calls=NL,
                     bytes=lambda kv: fc_bytes_int4(1, H, 5120),
                     flops=lambda kv: fc_flops(1, H, 5120), bound="memory"),
            "fc_o":
                dict(log="fc_o_decode_M1", calls=NL,
                     bytes=lambda kv: fc_bytes_int4(1, NH*HD, H),
                     flops=lambda kv: fc_flops(1, NH*HD, H), bound="memory"),
            "lm_head":
                dict(log="lm_head_decode_M1", calls=1,
                     bytes=lambda kv: fc_bytes_int8(1, H, VOCAB),
                     flops=lambda kv: fc_flops(1, H, VOCAB), bound="memory"),
            "paged_attention":
                dict(log=lambda kv: f"pa_decode_kv{kv}", calls=NL,
                     bytes=lambda kv: 2*NKV*HD*kv,           # KV traffic INT8
                     flops=lambda kv: 2*NH*HD*kv*2, bound="memory"),
            "rmsnorm":
                dict(log="so_rmsnorm_h2048_decode", calls=2*NL,
                     bytes=lambda kv: H*4, flops=lambda kv: H*8, bound="memory"),
            "q_norm":
                dict(log="so_rmsnorm3d_qnorm_decode", calls=NL,
                     bytes=lambda kv: NH*HD*4, flops=lambda kv: NH*HD*8, bound="memory"),
            "k_norm":
                dict(log="so_rmsnorm3d_knorm_decode", calls=NL,
                     bytes=lambda kv: NKV*HD*4, flops=lambda kv: NKV*HD*8, bound="memory"),
            "rope_q":
                dict(log="so_rope_q_decode", calls=NL,
                     bytes=lambda kv: NH*HD*4, flops=lambda kv: NH*HD*10, bound="memory"),
            "rope_k":
                dict(log="so_rope_k_decode", calls=NL,
                     bytes=lambda kv: NKV*HD*4, flops=lambda kv: NKV*HD*10, bound="memory"),
            "add":
                dict(log="so_add_decode", calls=2*NL,
                     bytes=lambda kv: H*6, flops=lambda kv: H, bound="memory"),
        },
        # ----- PREFILL -----
        "prefill": {
            "moe_3gemm_fused":
                dict(log=lambda S: f"moe_prefill_S{S}", calls=NL,
                     # All tokens × TK active × 3 GEMMs activations + weights spread
                     bytes=lambda S: 3*H*I*NE//2 + S*TK*(3*H + 3*I)*2,  # rough
                     flops=lambda S: S*TK*3*H*I*2, bound="compute"),
            "fc_qkv":
                dict(log=lambda S: f"fc_qkv_prefill_S{S}", calls=NL,
                     bytes=lambda S: fc_bytes_int8(S, H, 5120),
                     flops=lambda S: fc_flops(S, H, 5120), bound="compute"),
            "fc_o":
                dict(log=lambda S: f"fc_o_prefill_S{S}", calls=NL,
                     bytes=lambda S: fc_bytes_int8(S, NH*HD, H),
                     flops=lambda S: fc_flops(S, NH*HD, H), bound="compute"),
            "paged_attention":
                dict(log=lambda S: f"pa_prefill_S{S}", calls=NL,
                     # SDPA O(S²): bytes ≈ Q+K+V+O = 4·S·NH·HD·2 + ~kv writes
                     bytes=lambda S: 4*S*NH*HD*2,
                     flops=lambda S: 4*S*S*NH*HD,    # QKᵀ + AV both ~2·S²·NH·HD
                     bound="compute"),
            "rmsnorm":
                dict(log=lambda S: f"so_rmsnorm_h2048_prefill_S{S}" if S<=8192 else None,
                     calls=2*NL,
                     bytes=lambda S: S*H*4, flops=lambda S: S*H*8, bound="memory"),
            "rope_q":
                dict(log=lambda S: f"so_rope_q_prefill_S{S}" if S<=8192 else None,
                     calls=NL,
                     bytes=lambda S: S*NH*HD*4, flops=lambda S: S*NH*HD*10, bound="memory"),
        },
    }

SPECS = make_specs()


def get_kernels(metrics, log):
    """Return list of (kernel_name, ms, calls_in_log) sorted by ms desc, after dedup."""
    e = metrics.get(log)
    if not e: return []
    out = []
    for entry in e.get("per_kernel", []):
        if not entry or len(entry) < 3: continue
        name, per_iter_ns, calls = entry[0], entry[1], entry[2]
        # collapse hashed suffixes for readability
        nn = re.sub(r'_\d{15,}', '', name)
        nn = re.sub(r'_+', '_', nn).strip('_')
        out.append((nn, per_iter_ns/1e6, calls))
    out.sort(key=lambda r: -r[1])
    return out


def emit_table(platform, mode, S_or_kv, peak_compute_gflops):
    bw_peak = HW[platform]["bw"]
    metrics = B if platform == "BMG" else P
    specs = SPECS[mode]

    rows = []
    for opname, spec in specs.items():
        log = spec["log"](S_or_kv) if callable(spec["log"]) else spec["log"]
        if log is None:
            continue
        kernels = get_kernels(metrics, log)
        if not kernels: continue
        op_total_ms = sum(k[1] for k in kernels)
        op_bytes = spec["bytes"](S_or_kv)
        op_flops = spec["flops"](S_or_kv)
        calls_per_inf = spec["calls"]
        bound = spec["bound"]
        # significant kernels = top kernels covering >95 % of op time
        cum = 0; sig = []
        for k in kernels:
            sig.append(k); cum += k[1]
            if cum >= 0.95*op_total_ms: break
        for kname, kms, kcalls_in_log in sig:
            share = kms / op_total_ms if op_total_ms else 0
            single_ms = kms
            total_ms = single_ms * calls_per_inf
            # bytes/flops attributed proportional to time share
            kbytes = op_bytes * share
            kflops = op_flops * share
            gbs = kbytes / (single_ms*1e6) if single_ms else 0
            gflops = kflops / (single_ms*1e6) if single_ms else 0
            if bound == "memory":
                eff = gbs / bw_peak * 100
            else:
                eff = gflops / peak_compute_gflops * 100
            rows.append((opname, kname, single_ms, calls_per_inf, total_ms,
                         gflops, gbs, eff, bound))

    rows.sort(key=lambda r: -r[4])  # sort by total ms

    print(f"\n#### {platform} — {mode} {'S' if mode=='prefill' else 'kv'}={S_or_kv}")
    print()
    print("| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    total_ms_inf = 0
    for r in rows:
        total_ms_inf += r[4]
        print(f"| {r[0]} | `{r[1]}` | {r[2]:.4f} | {r[3]} | {r[4]:.3f} | {r[5]:.0f} | {r[6]:.1f} | {r[7]:.1f}% | {r[8]} |")
    print(f"\n**Total inference time (this stage)** ≈ **{total_ms_inf:.2f} ms**")
    return total_ms_inf


def main():
    SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    print("# Per-token-size Kernel Tables\n")
    for plat in ["BMG", "PTL"]:
        peak = HW[plat]["fp16"]  # decode FP16, prefill INT8 — but use unified
        peak_int8 = HW[plat]["int8"]
        print(f"\n## Platform: {plat}\n")
        print("### DECODE — per-kv-size (M=1)\n")
        for kv in SIZES:
            emit_table(plat, "decode", kv, peak)
        print(f"\n### PREFILL — per-S-size (compute-bound uses INT8 XMX peak = {peak_int8/1e3:.1f} TOPS)\n")
        for S in SIZES:
            emit_table(plat, "prefill", S, peak_int8)


if __name__ == "__main__":
    main()
