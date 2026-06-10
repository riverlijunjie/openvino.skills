#!/usr/bin/env python3
"""Qwen3-MoE (Qwen3-Coder-30B-A3B) per-op + end-to-end roofline report.

Reads <metrics>.json (parse_logs.py output) and ops_mapping.json,
prints a Markdown report aligned with SKILL.md.

Usage:
  build_report.py [--platform BMG|PTL] [--metrics file.json]
                  [--out perf.json] [--bw GB/s] [--fp16 TFLOPS] [--int8 TOPS]
"""
import argparse, json
from pathlib import Path

OUT = Path(__file__).resolve().parent

_HW = {
    "BMG": dict(bw=456.0, fp16=116.736, int8=233.472,
                desc="BMG (Arc B580, 2850 MHz, 456 GB/s, FP16 XMX 116.736 TFLOPS, INT8 XMX 233.472 TOPS)"),
    "PTL": dict(bw=110.0, fp16=58.982, int8=117.964,
                desc="PTL (Panther Lake iGPU, 110 GB/s, FP16 XMX 58.982 TFLOPS, INT8 XMX 117.964 TOPS)"),
}
_p = argparse.ArgumentParser()
_p.add_argument("--platform", default="BMG", choices=["BMG", "PTL"])
_p.add_argument("--metrics", default=None)
_p.add_argument("--out", default=None)
_p.add_argument("--bw", type=float, default=None)
_p.add_argument("--fp16", type=float, default=None)
_p.add_argument("--int8", type=float, default=None)
_a = _p.parse_args()

PLATFORM = _a.platform
_d = _HW[PLATFORM]
BW = _a.bw or _d["bw"]
FP16 = _a.fp16 or _d["fp16"]
INT8 = _a.int8 or _d["int8"]
PDESC = _d["desc"]

mfile = Path(_a.metrics) if _a.metrics else (OUT / f"{PLATFORM.lower()}_metrics.json")
M = json.loads(mfile.read_text())
OPS = json.loads((OUT / "ops_mapping.json").read_text())

NL = OPS["config"]["num_hidden_layers"]
H = OPS["config"]["hidden_size"]
NH = OPS["config"]["num_attention_heads"]
NKV = OPS["config"]["num_key_value_heads"]
HD = OPS["config"]["head_dim"]
I_MOE = OPS["config"]["moe_intermediate_size"]
NE = OPS["config"]["num_experts"]
TK = OPS["config"]["num_experts_per_tok"]
VOCAB = OPS["config"]["vocab_size"]


def total_ns(log):
    e = M.get(log)
    return e["total_kernel_ns"] if e else 0


def per_iter_ms(log):
    return total_ns(log) / 1e6


# Bytes per FC (M,K,N) INT4 g=128 weight + FP16 act/output
def fc_bytes_int4(M_, K, N, g=128):
    # weight: K*N/2 bytes (u4) + (K/g)*N*2 (scales fp16) + (K/g)*N*0.5 (zps u4)
    w = K * N // 2 + (K // g) * N * 2 + (K // g) * N // 2
    a = M_ * K * 2
    o = M_ * N * 2
    return w + a + o


def fc_bytes_int8(M_, K, N, g=128):
    w = K * N + (K // g) * N * 2
    a = M_ * K * 2
    o = M_ * N * 2
    return w + a + o


def fc_flops(M_, K, N):
    return 2 * M_ * K * N


# ---- DECODE table ----
def decode_row(name, log, calls, K, N, M_=1, prec="u4"):
    ms = per_iter_ms(log)
    if ms == 0:
        return None
    bytes_ = fc_bytes_int4(M_, K, N) if prec == "u4" else fc_bytes_int8(M_, K, N)
    flops = fc_flops(M_, K, N)
    ai = flops / bytes_
    gbs = bytes_ / (ms * 1e6)
    tops = flops / (ms * 1e9) / 1000  # TOPS
    eff = gbs / BW * 100
    return dict(name=name, ms=ms, calls=calls, total=ms * calls,
                ai=ai, gbs=gbs, tops=tops, eff=eff, bound="memory")


def moe_decode_row():
    log = "moe_decode_M1"
    ms = per_iter_ms(log)
    if ms == 0:
        return None
    # Active expert weight bytes per token (INT4 g=128, no shared_expert in qwen3_moe)
    per_expert_bytes = (3 * H * I_MOE) // 2 + (3 * (H // 128) * I_MOE * 2) + (3 * (H // 128) * I_MOE // 2)
    total_w = TK * per_expert_bytes
    # plus router (small) + activation traffic
    total_b = total_w + (H + TK * I_MOE * 2) * 2 + 2 * H
    flops = TK * 3 * H * I_MOE * 2
    gbs = total_b / (ms * 1e6)
    eff = gbs / BW * 100
    return dict(name=f"moe_3gemm fused (TK={TK}, NE={NE}, I={I_MOE})", ms=ms, calls=NL, total=ms * NL,
                ai=flops / total_b, gbs=gbs, tops=flops / (ms * 1e9) / 1000, eff=eff, bound="memory")


def pa_decode_row(kv=4096):
    log = f"pa_decode_kv{kv}"
    ms = per_iter_ms(log)
    if ms == 0:
        return None
    # kv cache bytes: 2 * NKV * HD * kv (int8) per token
    kv_bytes = 2 * NKV * HD * kv
    flops = 2 * NH * HD * kv * 2  # QK + AV
    gbs = kv_bytes / (ms * 1e6)
    eff = gbs / BW * 100
    return dict(name=f"pa_decode (kv={kv})", ms=ms, calls=NL, total=ms * NL,
                ai=flops / kv_bytes, gbs=gbs, tops=flops / (ms * 1e9) / 1000, eff=eff, bound="memory")


def small_op_row(name, log, calls, bytes_, flops):
    ms = per_iter_ms(log)
    if ms == 0:
        return None
    gbs = bytes_ / (ms * 1e6)
    return dict(name=name, ms=ms, calls=calls, total=ms * calls,
                ai=flops / bytes_, gbs=gbs, tops=flops / (ms * 1e9) / 1000,
                eff=gbs / BW * 100, bound="memory")


rows = []
rows.append(moe_decode_row())
rows.append(decode_row("lm_head (INT8 g=128)", "lm_head_decode_M1", 1, 2048, VOCAB, prec="u8"))
rows.append(decode_row("fc_qkv (INT4 g=128)", "fc_qkv_decode_M1", NL, 2048, 5120))
rows.append(decode_row("fc_o (INT4 g=128)", "fc_o_decode_M1", NL, 4096, 2048))
rows.append(pa_decode_row(4096))
rows.append(small_op_row("rmsnorm (H=2048)", "so_rmsnorm_h2048_decode", 2 * NL, H * 2 * 2, H))
rows.append(small_op_row("add (residual)", "so_add_decode", 2 * NL, H * 2 * 3, H))
rows.append(small_op_row("rope_q", "so_rope_q_decode", NL, NH * HD * 4, NH * HD * 10))
rows.append(small_op_row("rope_k", "so_rope_k_decode", NL, NKV * HD * 4, NKV * HD * 10))
rows.append(small_op_row("q_norm", "so_rmsnorm3d_qnorm_decode", NL, NH * HD * 4, NH * HD))
rows.append(small_op_row("k_norm", "so_rmsnorm3d_knorm_decode", NL, NKV * HD * 4, NKV * HD))
rows = [r for r in rows if r]
rows.sort(key=lambda r: -r["total"])

print(f"\n## DECODE per-op (kv=4096, {PLATFORM})")
print(f"_{PDESC}_")
print()
print("| Op | Avg ms | Calls | Total ms | AI (FLOP/B) | GB/s | Eff% | Bound |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
total_decode = 0.0
for r in rows:
    total_decode += r["total"]
    print(f"| {r['name']} | {r['ms']:.4f} | {r['calls']} | {r['total']:.3f} | {r['ai']:.2f} | {r['gbs']:.1f} | {r['eff']:.1f}% | {r['bound']} |")
print(f"\n**Decode total** = {total_decode:.3f} ms  →  **{1000/total_decode:.1f} tok/s**")

# ---- PREFILL table ----
print(f"\n## PREFILL per-layer kernel time (ms, {PLATFORM})")
print()
print("| S | MoE/layer | PA/layer | fc_qkv/layer | fc_o/layer | total est (ms) |")
print("|---:|---:|---:|---:|---:|---:|")
prefill_summary = {}
for S in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
    moe = per_iter_ms(f"moe_prefill_S{S}")
    pa = per_iter_ms(f"pa_prefill_S{S}")
    fcq = per_iter_ms(f"fc_qkv_prefill_S{S}")
    fco = per_iter_ms(f"fc_o_prefill_S{S}")
    if moe == 0:
        continue
    lm = per_iter_ms("lm_head_decode_M1")
    total = NL * (moe + pa + fcq + fco) + lm
    prefill_summary[S] = total
    print(f"| {S} | {moe:.3f} | {pa:.3f} | {fcq:.3f} | {fco:.3f} | {total:.2f} |")

if _a.out:
    Path(_a.out).write_text(json.dumps({
        "platform": PLATFORM, "platform_desc": PDESC,
        "decode_total_ms": total_decode, "decode_rows": rows,
        "prefill_total_ms": prefill_summary,
    }, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))
    print(f"\nSaved {_a.out}")
