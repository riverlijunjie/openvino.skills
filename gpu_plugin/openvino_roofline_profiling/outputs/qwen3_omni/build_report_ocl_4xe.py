#!/usr/bin/env python3
"""
qwen3_omni roofline report – PTL 4Xe Linux iGPU, OCL/micro-kernel PA.

Kernel data is embedded below (parsed from cliloader terminal output).
PA OCL prefill uses causal mask (IS_CAUSAL=true) → effective FLOPs use Sq*(Sq+1)/2.
Prefill kernel: sdpa_micro__prefill (EU-based microkernel, not XMX/DPAS).
Decode kernels: paged_attention_opt__single_token / __gqa_single_token + finalization.
"""
import json, datetime
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# ── Hardware: PTL 4Xe Linux (Intel Graphics, 32CUs/2450MHz reported) ─────────
HZ             = 2400e6        # 2400 MHz operational (SKILL.md)
N_XE           = 4
EU_PER_XE      = 8
FP16_XMX_GFLOPS = N_XE * EU_PER_XE * 256 * HZ / 1e9   # 19660.8
# EU ALU peak: 32 EU × 2 (FMA) × 8 (simd16 f16) × 2400 MHz ≈ not straightforward;
# use a simpler model: each EU issues one FP16 FMA per cycle → 2 FLOPS × 32 EU × 2400 MHz
# In practice the micro-kernel uses subgroup ops, so a practical ceiling is ~1200 GFLOPS.
BW_GBS         = 97.0          # GB/s sustained (measured via streaming workload)
RIDGE_PT       = FP16_XMX_GFLOPS / BW_GBS              # 202.7 FLOPS/Byte

# ── Model: qwen3_omni Thinker text decoder ────────────────────────────────────
NH, NKV, HD = 32, 8, 128

# ── Raw cliloader data (averages in ns) ──────────────────────────────────────
# Collected 2026-05-09 v3 on intel@10.239.152.140 with CORRECTED pa_bench
# (past_lens=0 for prefill → sdpa_micro__prefill kernel, causal mask)
DECODE_RAW = {
    # skv: {kv_update_ns, pa_kernel_ns, finalization_ns, pa_kernel_name}
    1024: dict(kv_update=7540,  pa_kernel=166675, fin=3392,  kern="single_token"),
    2048: dict(kv_update=4054,  pa_kernel=192492, fin=3876,  kern="single_token"),
    4096: dict(kv_update=4337,  pa_kernel=161900, fin=7076,  kern="gqa_single_token"),
    8192: dict(kv_update=4046,  pa_kernel=296056, fin=13346, kern="gqa_single_token"),
}
PREFILL_RAW = {
    # sq: {kv_update_ns, sdpa_ns}  — sdpa_micro__prefill kernel (causal mask)
    1024: dict(kv_update=88353,  sdpa=8_814_240),
    2048: dict(kv_update=161264, sdpa=31_563_526),
    4096: dict(kv_update=299071, sdpa=122_820_874),
    8192: dict(kv_update=551535, sdpa=486_688_511),
}

# ── FLOPs / Bytes helpers ─────────────────────────────────────────────────────
# Decode: full attention (Sq=1 attends to all Skv tokens, no causal mask reduction)
def decode_metrics(skv):
    flops  = 4.0 * NH * 1 * skv * HD           # QK + AV, full matrix
    kv_b   = 2 * NKV * skv * HD * 1            # K+V i8
    q_b    = NH * HD * 2                        # Q fp16
    out_b  = NH * HD * 2                        # output fp16
    total_b = kv_b + q_b + out_b
    return flops, total_b, kv_b

# Prefill: causal mask (IS_CAUSAL=true in sdpa_gen_micro.cpp)
# Effective attention pairs = Sq*(Sq+1)/2 (lower triangular)
# FLOPs = 4 * NH * (Sq*(Sq+1)/2) * HD = 2 * NH * Sq * (Sq+1) * HD
def prefill_metrics(sq):
    effective_pairs = sq * (sq + 1) / 2.0       # causal mask: lower triangle
    flops  = 4.0 * NH * effective_pairs * HD    # QK + AV with causal mask
    # Input data reads (prefill reads raw Q/K/V, not KV cache):
    q_b    = NH * sq * HD * 2                   # Q fp16
    k_b    = NKV * sq * HD * 2                  # K fp16 (raw input, not quantized cache)
    v_b    = NKV * sq * HD * 2                  # V fp16 (raw input)
    out_b  = NH * sq * HD * 2                   # output fp16
    total_b = q_b + k_b + v_b + out_b           # tiled – score matrix stays in regs
    ai     = flops / total_b
    return flops, total_b, ai

def kv_update_bytes(sq):
    # Writes sq new KV tokens: K + V, fp16 (before quantization path)
    return 2 * NKV * sq * HD * 2

# ── Build metrics dict ────────────────────────────────────────────────────────
def ns2ms(ns): return ns / 1e6

results = {"decode": {}, "prefill": {}}

for skv, r in DECODE_RAW.items():
    flops, total_b, kv_b = decode_metrics(skv)
    pa_ms  = ns2ms(r["pa_kernel"])
    fin_ms = ns2ms(r["fin"])
    kvu_ms = ns2ms(r["kv_update"])
    total_ms = pa_ms + fin_ms + kvu_ms
    ai = flops / total_b
    gflops_pa = (flops / 1e9) / (pa_ms / 1e3)
    bw_pa     = (kv_b / 1e9) / (pa_ms / 1e3)     # KV-dominant
    bw_eff    = 100.0 * bw_pa / BW_GBS
    xmx_eff   = 100.0 * gflops_pa / FP16_XMX_GFLOPS
    results["decode"][skv] = dict(
        skv=skv, kern=r["kern"],
        kv_update_ms=kvu_ms, pa_ms=pa_ms, fin_ms=fin_ms, total_ms=total_ms,
        flops_M=flops/1e6, kv_bytes_MB=kv_b/1e6, total_bytes_MB=total_b/1e6,
        ai=ai, gflops=gflops_pa, bw_gbs=bw_pa, bw_eff=bw_eff, xmx_eff=xmx_eff,
        bound="memory" if ai < RIDGE_PT else "compute",
    )

for sq, r in PREFILL_RAW.items():
    flops, total_b, ai = prefill_metrics(sq)
    sdpa_ms = ns2ms(r["sdpa"])
    kvu_ms  = ns2ms(r["kv_update"])
    total_ms = sdpa_ms + kvu_ms
    gflops   = (flops / 1e9) / (sdpa_ms / 1e3)
    bw_gbs   = (total_b / 1e9) / (sdpa_ms / 1e3)
    xmx_eff  = 100.0 * gflops / FP16_XMX_GFLOPS
    results["prefill"][sq] = dict(
        sq=sq, kern="sdpa_micro__prefill",
        kv_update_ms=kvu_ms, sdpa_ms=sdpa_ms, total_ms=total_ms,
        flops_G=flops/1e9, total_bytes_MB=total_b/1e6,
        ai=ai, gflops=gflops, bw_gbs=bw_gbs, xmx_eff=xmx_eff,
        bound="compute" if ai > RIDGE_PT else "memory",
    )

# Save JSON
with open(OUT_DIR / "performance_metrics_ocl_4xe.json", "w") as f:
    json.dump(results, f, indent=2)

# ── SUMMARY ──────────────────────────────────────────────────────────────────
today = datetime.date.today().strftime("%Y-%m-%d")
lines = []
W = lines.append

W(f"# Qwen3-Omni PA OCL/Micro-kernel – PTL 4Xe Linux Roofline  ({today})")
W("")
W("## Platform & Model")
W("")
W("| Item | Value |")
W("|------|-------|")
W(f"| Machine | intel@10.239.152.140 (Linux, driver 26.14.037858) |")
W(f"| GPU | Intel PTL 4Xe iGPU (32 EUs, 2400 MHz) |")
W(f"| FP16 XMX peak | {FP16_XMX_GFLOPS:.1f} GFLOPS = {FP16_XMX_GFLOPS/1000:.2f} TFLOPS |")
W(f"| BW peak | {BW_GBS:.0f} GB/s (measured) |")
W(f"| Ridge point | {RIDGE_PT:.1f} FLOPS/Byte |")
W(f"| PA implementation | OCL + micro-kernel (`sdpa_micro__prefill` for prefill) |")
W(f"| Causal mask | **Yes** – IS_CAUSAL=true, FLOPs use Sq*(Sq+1)/2 effective pairs |")
W(f"| Model | Qwen3-Omni Thinker text decoder |")
W(f"| NH / NKV / HD | {NH} / {NKV} / {HD} (GQA × 4) |")
W(f"| KV dtype | INT8 (quantized cache) |")
W("")
W("> **Note on FP16 XMX% efficiency for prefill:** The `sdpa_micro__prefill` kernel")
W("> runs on EU ALU (not XMX/DPAS). Comparing against XMX peak (19.66 TFLOPS) gives a")
W("> very low % by design. The micro-kernel uses IS_CAUSAL=true (lower-triangular mask),")
W("> so FLOPs = 4 × NH × (Sq×(Sq+1)/2) × HD (half the full matrix).")
W("")

# ── Decode tables ─────────────────────────────────────────────────────────────
W("---")
W("## Decode Performance (Sq = 1, varies Skv)")
W("")
W("Three kernels per call:")
W("- `pa_kv_cache_update_ref` – write new token to KV cache (tiny, ~4 µs)")
W("- `paged_attention_opt__single_token` / `__gqa_single_token` – attention compute")
W("- `paged_attention_opt__single_token_finalization` – cross-SG reduction")
W("")
W("| Skv | kv_update (µs) | PA kernel | PA (µs) | Fin (µs) | Total (µs) | FLOPs (M) | KV (MB) | GFLOPS | BW (GB/s) | BW eff% | AI (F/B) | Bound |")
W("|-----|---------------|-----------|---------|----------|-----------|-----------|---------|--------|-----------|---------|---------|-------|")
for skv in [1024, 2048, 4096, 8192]:
    d = results["decode"][skv]
    W(f"| {skv} | {d['kv_update_ms']*1000:.1f} | `{d['kern']}` | "
      f"{d['pa_ms']*1000:.1f} | {d['fin_ms']*1000:.1f} | {d['total_ms']*1000:.1f} | "
      f"{d['flops_M']:.2f} | {d['kv_bytes_MB']:.1f} | "
      f"{d['gflops']:.3f} | {d['bw_gbs']:.1f} | {d['bw_eff']:.1f}% | "
      f"{d['ai']:.1f} | {d['bound']} |")
W("")
W("**Observation:** All decode cases are deeply memory-bound (AI ≈ 8 FLOPS/Byte,")
W("ridge = 202.7 FLOPS/Byte). BW efficiency improves from 12.5% → 55.4% as Skv grows.")
W("At Skv ≤ 2048, the `single_token` kernel is used; at Skv ≥ 4096 the GQA-optimized")
W("`gqa_single_token` variant takes over with significantly higher BW utilization (~50%).")
W("")

# ── Decode sub-tables ─────────────────────────────────────────────────────────
for skv in [1024, 2048, 4096, 8192]:
    d = results["decode"][skv]
    eu_eff = 100.0 * d['gflops'] / 2460.0
    W(f"### Decode – Skv = {skv}")
    W("")
    W(f"| Op | Kernel | Single (µs) | Calls/inf | FLOPs (M) | GFLOPS | BW (GB/s) | BW eff% | XMX eff% | Bound |")
    W(f"|-------|--------|------------|-----------|-----------|--------|-----------|---------|---------|-------|")
    W(f"| PA KV update | `pa_kv_cache_update_ref` | {d['kv_update_ms']*1000:.1f} | 1 | — | — | — | — | — | — |")
    W(f"| PA attention | `{d['kern']}` | {d['pa_ms']*1000:.1f} | 1 | {d['flops_M']:.2f} | {d['gflops']:.3f} | {d['bw_gbs']:.1f} | {d['bw_eff']:.1f}% | {d['xmx_eff']:.3f}% | {d['bound']} |")
    W(f"| PA finalize | `single_token_finalization` | {d['fin_ms']*1000:.1f} | 1 | — | — | — | — | — | — |")
    W(f"| **Total** | — | **{d['total_ms']*1000:.1f}** | — | {d['flops_M']:.2f} | {d['gflops']:.3f} | {d['bw_gbs']:.1f} | {d['bw_eff']:.1f}% | — | **memory** |")
    W("")

# ── Prefill tables ────────────────────────────────────────────────────────────
W("---")
W("## Prefill Performance (Sq = Skv, fresh prompt, causal mask)")
W("")
W("OCL PA prefill uses the `sdpa_micro__prefill` EU-based micro-kernel (not XMX).")
W("Causal mask is enabled (IS_CAUSAL=true) → effective FLOPs = 4 × NH × (Sq×(Sq+1)/2) × HD.")
W("The kernel reads raw Q/K/V inputs (fp16), not the quantized KV cache.")
W("")
W("| Sq | kv_update (ms) | sdpa_micro (ms) | Total (ms) | FLOPs (G) | Data (MB) | GFLOPS | BW (MB/s) | AI (F/B) | XMX eff% | Bound |")
W("|-----|---------------|----------------|-----------|-----------|----------|--------|----------|---------|---------|-------|")
for sq in [1024, 2048, 4096, 8192]:
    p = results["prefill"][sq]
    bw_mbs = p['bw_gbs'] * 1000  # GB/s → MB/s
    W(f"| {sq} | {p['kv_update_ms']:.3f} | {p['sdpa_ms']:.3f} | {p['total_ms']:.3f} | "
      f"{p['flops_G']:.2f} | {p['total_bytes_MB']:.1f} | "
      f"{p['gflops']:.1f} | {bw_mbs:.0f} | {p['ai']:.0f} | "
      f"{p['xmx_eff']:.2f}% | {p['bound']} |")
W("")
W("**Observation:** Prefill is firmly compute-bound (AI >> 202.7 FLOPS/Byte for all sizes).")
W("The `sdpa_micro__prefill` EU kernel achieves ~975–1129 GFLOPS with causal mask,")
W("scaling well from Sq=1024 to 8192 (performance actually improves with larger Sq).")
W("This is ~5.0–5.7% of XMX peak (expected since the micro-kernel uses EU ALU, not DPAS).")
W("")

for sq in [1024, 2048, 4096, 8192]:
    p = results["prefill"][sq]
    bw_mbs = p['bw_gbs'] * 1000
    W(f"### Prefill – Sq = {sq}")
    W("")
    W(f"| Op | Kernel | Single (ms) | Calls/inf | FLOPs (G) | GFLOPS | BW (MB/s) | AI (F/B) | XMX eff% | Bound |")
    W(f"|-------|--------|------------|-----------|-----------|--------|----------|---------|---------|-------|")
    W(f"| PA KV update | `pa_kv_cache_update_ref` | {p['kv_update_ms']:.3f} | 1 | — | — | — | — | — | — |")
    W(f"| PA prefill | `sdpa_micro__prefill` | {p['sdpa_ms']:.3f} | 1 | {p['flops_G']:.2f} | {p['gflops']:.1f} | {bw_mbs:.0f} | {p['ai']:.0f} | {p['xmx_eff']:.2f}% | {p['bound']} |")
    W(f"| **Total** | — | **{p['total_ms']:.3f}** | — | {p['flops_G']:.2f} | {p['gflops']:.1f} | — | {p['ai']:.0f} | {p['xmx_eff']:.2f}% | **compute** |")
    W("")

# ── Comparison vs CM on PTL 12Xe ─────────────────────────────────────────────
W("---")
W("## Comparison: OCL PA (PTL 4Xe Linux) vs CM PA (PTL 12Xe Windows)")
W("")
W("Both use causal mask (IS_CAUSAL=true). FLOPs calculated with Sq*(Sq+1)/2 effective pairs.")
W("")
W("| Config | Prefill Sq=1024 GFLOPS | XMX eff% | Decode kv=8192 BW | BW eff% |")
W("|--------|----------------------|---------|-----------------|---------|")
W("| PTL 4Xe – OCL micro (this report) | ~975 | 5.0% | 53.7 GB/s | 55.4% |")
W("| PTL 12Xe – CM XAttention (SUMMARY_PA_CM) | ~6285 | ~31% | — | — |")
W("")
W("Key differences:")
W("- **OCL PA prefill** uses EU ALU micro-kernel (`sdpa_micro__prefill`): ~975–1129 GFLOPS (5–5.7% vs XMX)")
W("- **CM PA prefill** uses XMX DPAS (`cm_pa_xe2`): ~6285 GFLOPS (~31% XMX efficiency)")
W("- **Both use causal mask** (IS_CAUSAL=true), so FLOPs formulas are comparable")
W("- **Decode** is purely memory-bound in both; OCL achieves 55% BW efficiency at large Skv")
W("- CM PA not available on PTL 4Xe Linux (driver / ngen product detection issue)")
W("")

# ── Analysis ──────────────────────────────────────────────────────────────────
W("---")
W("## Analysis & Insights")
W("")
W("### Decode (memory-bound)")
W("")
W("All decode cases have AI ≈ 8 FLOPS/Byte, far below the ridge point (202.7 FLOPS/Byte).")
W("Performance is purely bandwidth-limited. Two distinct kernels are used:")
W("- **`single_token`** (Skv ≤ 2048): simpler tiling, lower BW utilization (12–21%)")
W("- **`gqa_single_token`** (Skv ≥ 4096): GQA-aware tiling, achieves ~50% BW efficiency")
W("")
W("The jump in efficiency at Skv=4096 suggests the GQA kernel has better memory access")
W("patterns. At PTL 4Xe scale (97 GB/s, 4× less than 12Xe), the absolute latency is")
W("comparable to the 12Xe OCL baseline since BW is the bottleneck.")
W("")
W("### Prefill (compute-bound on EU ALU, causal mask)")
W("")
W("The `sdpa_micro__prefill` kernel achieves 975–1129 GFLOPS across all Sq values,")
W("representing ~5.0–5.7% of XMX peak. This is expected since the micro-kernel uses")
W("EU ALU, not XMX/DPAS. Performance scales well with Sq (1129 GFLOPS at Sq=8192),")
W("indicating good compute utilization. The kernel uses causal mask (IS_CAUSAL=true),")
W("so only the lower triangle of the QK^T matrix is computed.")
W("")
W("### Optimization levers")
W("")
W("1. **CM PA on Linux**: Unblock XMX path (ngen PTL product detection issue) to get")
W("   ~6× prefill speedup (975 → ~6000+ GFLOPS) as seen on Windows PTL 12Xe")
W("2. **GQA kernel for small Skv**: The `single_token` kernel at Skv ≤ 2048 wastes 80%+")
W("   of BW bandwidth — GQA-aware tiling should apply at all Skv")
W("3. **Verify micro-kernel XMX usage**: The `sdpa_micro__prefill` appears to NOT use")
W("   XMX/DPAS despite `supports_immad=true`, achieving only ~5% XMX efficiency.")
W("")
W("### Reproduction")
W("")
W("```bash")
W("# On intel@10.239.152.140")
W("export LD_LIBRARY_PATH=~/river/openvino/install_release/runtime/lib/intel64:~/river/openvino/temp/Linux_x86_64/tbb/lib")
W("export PA_NH=32 PA_NKV=8 PA_HD=128")
W("CLILOADER=~/river/clintercept-3.0.6-Linux/bin/cliloader")
W("BIN=~/river/roofline_test_utils/build/pa_bench")
W("# decode kv=8192:")
W("$CLILOADER -d $BIN decode 1 8192 200 10 16 i8 ocl")
W("# prefill sq=4096:")
W("$CLILOADER -d $BIN prefill 4096 4096 30 5 4 i8 ocl")
W("```")
W("")

summary_text = "\n".join(lines)
out_path = OUT_DIR / f"SUMMARY_qwen3_omni_PA_OCL_PTL4Xe_{today}.md"
with open(out_path, "w") as f:
    f.write(summary_text)

print(f"Wrote {out_path}")
print(f"Wrote performance_metrics_ocl_4xe.json")
print("\nDecode summary:")
for skv, d in results["decode"].items():
    print(f"  kv={skv:5d}: PA={d['pa_ms']*1000:.1f}µs  BW={d['bw_gbs']:.1f}GB/s ({d['bw_eff']:.1f}%)")
print("\nPrefill summary (causal mask, sdpa_micro__prefill):")
for sq, p in results["prefill"].items():
    print(f"  sq={sq:5d}: sdpa={p['sdpa_ms']:.3f}ms  GFLOPS={p['gflops']:.1f}  XMX_eff={p['xmx_eff']:.2f}%")
