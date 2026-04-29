#!/usr/bin/env python3
"""Qwen3-8B Roofline Report — reads performance_metrics.json.

Per SKILL.md: swish & multiply are NOT profiled separately (should be fused).
LM_Head uses INT8 g=128. HW peaks computed per SKILL.md formulas.

Usage:
    python3 generate_metrics.py    # parses logs -> performance_metrics.json
    python3 roofline_report.py     # prints human-readable tables
"""
import json
from pathlib import Path

UTILS = Path(__file__).resolve().parent
M = json.loads((UTILS / "performance_metrics.json").read_text())

HW = M["hardware"]
NUM_LAYERS = M["config"]["num_layers"]
TOKEN_SIZES = [1024, 2048, 4096, 8192]


def print_hw_table():
    print("=" * 100)
    print("HARDWARE PEAKS (SKILL.md formulas: XMX=cores*8*256*freq; INT8=2xFP16)")
    print("=" * 100)
    print(f"{'Platform':<10}{'Cores':>6}{'Freq MHz':>10}{'BW GB/s':>10}"
          f"{'FP16 XMX TFLOPS':>18}{'INT8 XMX TOPS':>16}{'SIMD FP16 TFLOPS':>18}")
    for p, h in HW.items():
        print(f"{p:<10}{h['xe_cores']:>6}{h['freq_mhz']:>10}{h['bw_gbs']:>10.0f}"
              f"{h['fp16_xmx_tflops']:>18.3f}{h['int8_xmx_tops']:>16.3f}{h['simd_fp16_tflops']:>18.3f}")


def print_per_op_table(platform, phase):
    rows = M["metrics"][platform][phase]
    print(f"\n--- {platform} {phase} ---")
    print(f"{'Op':<42}{'Bound':<9}{'Avg(ms)':>10}{'Calls/inf':>12}"
          f"{'Total(ms)':>12}{'Ach GB/s':>10}{'Ach TOPS':>10}{'Eff%':>8}")
    for r in rows:
        gbs = r.get("achieved_gbs", 0) or 0
        tops = r.get("achieved_tops", r.get("achieved_tflops", 0)) or 0
        print(f"{r['op'][:42]:<42}{r.get('bound','-'):<9}{r['avg_latency_ms']:>10.4f}"
              f"{r['total_calls_per_inference']:>12}{r['total_latency_ms']:>12.3f}"
              f"{gbs:>10.1f}{tops:>10.2f}"
              f"{r.get('efficiency_pct', 0):>7.1f}%")


def print_model_totals():
    print("\n" + "=" * 100)
    print("MODEL-LEVEL TOTALS (kernel-only; decode tok/s = 1000/model_ms)")
    print("=" * 100)
    for p in ("BMG", "PTL"):
        print(f"\n[{p}] Decode")
        print(f"  {'kv':>6}{'fc_ms':>8}{'pa_ms':>8}{'small_ms':>10}{'lm_head_ms':>12}"
              f"{'layer_ms':>10}{'model_ms':>10}{'tok/s':>8}")
        for kv in TOKEN_SIZES:
            r = M["model_totals"][p].get(f"decode_kv{kv}", {})
            if not r:
                continue
            print(f"  {kv:>6}{r['fc_ms']:>8.3f}{r['pa_ms']:>8.4f}"
                  f"{r['small_ms']:>10.4f}{r['lm_head_ms']:>12.3f}"
                  f"{r['layer_ms']:>10.3f}{r['model_ms']:>10.2f}{r['tokens_per_sec']:>8.1f}")

        print(f"\n[{p}] Prefill")
        print(f"  {'S':>6}{'fc_ms':>10}{'dq_ms':>10}{'pa_ms':>10}{'small_ms':>10}"
              f"{'layer_ms':>10}{'model_ms':>10}{'tok/s':>8}")
        for S in TOKEN_SIZES:
            r = M["model_totals"][p].get(f"prefill_S{S}", {})
            if not r:
                continue
            print(f"  {S:>6}{r['fc_ms']:>10.3f}{r['dyn_quant_ms']:>10.3f}"
                  f"{r['pa_ms']:>10.3f}{r['small_ms']:>10.3f}"
                  f"{r['layer_ms']:>10.3f}{r['model_ms']:>10.1f}{r['tokens_per_sec']:>8.0f}")


def print_cross_platform():
    print("\n" + "=" * 100)
    print("CROSS-PLATFORM COMPARISON")
    print("=" * 100)
    print("\nDecode:")
    print(f"  {'kv':>6}{'BMG ms':>10}{'PTL ms':>10}{'PTL/BMG':>10}{'Winner':>10}")
    for kv in TOKEN_SIZES:
        b = M["model_totals"]["BMG"].get(f"decode_kv{kv}", {}).get("model_ms", 0)
        p = M["model_totals"]["PTL"].get(f"decode_kv{kv}", {}).get("model_ms", 0)
        if b and p:
            r = p / b
            print(f"  {kv:>6}{b:>10.2f}{p:>10.2f}{r:>10.2f}x{('BMG' if r>1 else 'PTL'):>10}")
    print("\nPrefill:")
    print(f"  {'S':>6}{'BMG ms':>10}{'PTL ms':>10}{'PTL/BMG':>10}{'Winner':>10}")
    for S in TOKEN_SIZES:
        b = M["model_totals"]["BMG"].get(f"prefill_S{S}", {}).get("model_ms", 0)
        p = M["model_totals"]["PTL"].get(f"prefill_S{S}", {}).get("model_ms", 0)
        if b and p:
            r = p / b
            print(f"  {S:>6}{b:>10.1f}{p:>10.1f}{r:>10.2f}x{('BMG' if r>1 else 'PTL'):>10}")


def main():
    print_hw_table()
    for p in ("BMG", "PTL"):
        print_per_op_table(p, "decode")
    for p in ("BMG", "PTL"):
        print_per_op_table(p, "prefill_8192")
    print_model_totals()
    print_cross_platform()


if __name__ == "__main__":
    main()
