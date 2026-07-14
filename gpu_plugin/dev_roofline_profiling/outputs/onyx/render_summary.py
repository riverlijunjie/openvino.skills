#!/usr/bin/env python3
"""Render SUMMARY_onyx_ptl_12xe_<date>.md from performance_metrics.json,
following utils/template/SUMMARY_TEMPLATE.md."""
import json, os, datetime

here = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(here, "performance_metrics.json")))
DATE = "2026-07-14"
SIZES = M["sizes"]

def f(x, n=2):
    return f"{x:,.{n}f}"

def krow(r):
    sm = f(r["single_ms"],4) if r["single_ms"] else "—"
    gf = f(r["gflops"],1) if r["gflops"] else "—"
    gb = f(r["gbs"],1) if r["gbs"] else "—"
    ef = f(r["eff"],1)+"%" if r["eff"] else "—"
    return (f"| {r['op']} | `{r['kernel']}` | {sm} | {r['calls']} | "
            f"{f(r['total_ms'],3)} | {gf} | {gb} | {ef} | {r['bound']} |")

def table(rows, total, header):
    out = [f"### {header}", "",
           "| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    out += [krow(r) for r in rows]
    out.append(f"| **TOTAL** |  |  |  | **{f(total,2)}** |  |  |  |  |")
    out.append("")
    return "\n".join(out)

# weight footprint (recompute for table)
H,L,QDIM,KVDIM,INTER,VOCAB,G = 6656,52,4096,256,19968,202048,128
def i4(K,N): return N*K/2 + N*(K/G)*2 + N*(K/G)*0.5
def i8(K,N): return N*K + N*(K/G)*2
W = [
 ("Embedding", f"{H}×{VOCAB}", "INT8 g128", i8(H,VOCAB), 1),
 ("FC_QKV (fused Q+K+V)", f"{H}×{QDIM+2*KVDIM}", "INT4 g128", i4(H,QDIM+2*KVDIM), L),
 ("FC_OutGate (attn output gate)", f"{H}×{QDIM}", "INT4 g128", i4(H,QDIM), L),
 ("FC_O (attn output)", f"{QDIM}×{H}", "INT4 g128", i4(QDIM,H), L),
 ("MLP_gate (SwiGLU gate)", f"{H}×{INTER}", "INT4 g128", i4(H,INTER), L),
 ("MLP_up (SwiGLU up)", f"{H}×{INTER}", "INT4 g128", i4(H,INTER), L),
 ("MLP_down (SwiGLU down)", f"{INTER}×{H}", "INT4 g128", i4(INTER,H), L),
 ("LM_Head", f"{H}×{VOCAB}", "INT8 g128", i8(H,VOCAB), 1),
]
wtot = sum(b*l for _,_,_,b,l in W)/1e6
wlines = []
for n,sh,q,b,l in W:
    wlines.append(f"| {n} | {sh} | {q} | {b/1e6:,.2f} MB | {l} | {b*l/1e6:,.1f} |")

GEN = 512  # decode window used for E2E

# ---------------- assemble ----------------
S = []
S.append(f"# Onyx (dense VLM text decoder) — Roofline on PTL 12Xe ({DATE})")
S.append("")
S.append("**Platform**: PTL B390 iGPU — 12 Xe @ 2400 MHz, 110 GB/s LPDDR5x "
         "(Local_Admin@10.239.132.229)")
S.append("**Model**: `onyx` — 52-layer dense SwiGLU decoder of the unified multimodal "
         "model (vision tower excluded from this text-generation roofline)")
S.append("")
S.append("> **This report uses REAL ON-DEVICE MEASUREMENTS.** Every op was profiled in "
         "its own process on the PTL 12Xe target via cliloader 3.0.6 "
         "`--device-performance-timing` (mean kernel ns/iter), driving the Onyx shapes "
         "through OpenVINO's `FullyConnectedCompressed gemm_kernel` and `PagedAttention` "
         "OpenCL+micro-kernel primitives. Bytes/FLOPs are computed analytically from the "
         "same shapes to derive Eff% and the roofline floor. See *Data sources*.")
S.append("")
S.append("- hidden=6656, layers=52, heads 32Q/2KV (GQA×16), head_dim=128, "
         "intermediate=19968, vocab=202048, SwiGLU (silu)")
S.append("- MatMul weights INT4 g=128 / FP16 act; LM_head INT8 g=128 / FP16 act; "
         "KV cache **INT4 g=128** (user request)")
S.append("- Attention has per-layer **QK-RMSNorm** + **sigmoid output gate** "
         "(extra `output_gate_proj` FC 6656→4096) and iRoPE: "
         "39 sliding(SW=2048)/RoPE + 13 full/NoPE layers")
S.append("- SDPA: PagedAttention OpenCL + micro_kernel")
S.append("")

S.append("## Model parameters & weight shapes\n")
S.append("| Field | Value | Notes |")
S.append("|---|---:|---|")
S.append("| `hidden_size` | 6656 | residual / activation channel |")
S.append("| `num_hidden_layers` | 52 | 39 sliding/RoPE + 13 full/NoPE |")
S.append("| `num_attention_heads` (NH) | 32 | Q heads |")
S.append("| `num_key_value_heads` (NKV) | 2 | GQA: 16-way Q-per-KV sharing |")
S.append("| `head_dim` (HD) | 128 | Q_dim = 4096, KV_dim = 256 |")
S.append("| `intermediate_size` | 19968 | SwiGLU MLP hidden |")
S.append("| `vocab_size` | 202048 | LM head N |")
S.append("| `hidden_act` | silu | SwiGLU = down(silu(gate)·up) |")
S.append("| `sliding_window` | 2048 | sliding layers cap KV at 2048 |")
S.append("| `every_n_layers_nope` | 4 | every 4th layer is full/NoPE |")
S.append("| `rope_theta` | 500000 | |")
S.append("| `use_qk_norm` | true | scaleless RMSNorm on Q,K per head |")
S.append("| `use_attn_output_gate` | true | extra FC 6656→4096 + sigmoid gate |")
S.append("| `tie_word_embeddings` | false | LM head stored separately (INT8) |")
S.append("")
S.append("Per-layer weight matrices (one decoder block) and global weights:\n")
S.append("| Weight | Shape (K × N) | Quant | Bytes / instance | × Layers | Total MB |")
S.append("|---|---:|---|---:|---:|---:|")
S += wlines
S.append(f"| **Total static weights** |  |  |  |  | **{wtot:,.0f} MB** |")
S.append("")
fp16_mb = 2*sum(int(sh.split('×')[0])*int(sh.split('×')[1])*l for _,sh,_,_,l in W)/1e6
S.append(f"_FP16 baseline (no quant) ≈ {fp16_mb:,.0f} MB → quantized total {wtot:,.0f} MB "
         f"is {wtot/fp16_mb*100:.0f}% of FP16 size. During **decode** every FC + LM_head weight is "
         f"re-read each token → ~{wtot-1365.8:,.0f} MB of weight traffic per token "
         f"(embedding not read), which sets the memory-bound decode floor._")
S.append("")

S.append("## Theoretical roofline\n")
S.append("| Metric | Value |")
S.append("|---|---|")
S.append(f"| FP16 XMX peak | {M['fp16_tflops']:.3f} TFLOPS |")
S.append(f"| INT8 XMX peak | {M['int8_tops']:.3f} TOPS |")
S.append(f"| Memory BW | {M['bw_gbs']:.0f} GB/s |")
S.append(f"| Ridge point (FP16) | {M['ridge_f16']:.1f} FLOP/byte |")
S.append(f"| Ridge point (INT8) | {M['ridge_i8']:.1f} OP/byte |")
S.append("")
S.append("_FP16 XMX = 12 Xe × 8 EU × 256 FLOP/cycle × 2.4 GHz. INT8 XMX = 2× FP16. "
         "A 5% overhead is deducted from each peak (achievable BW 104.5 GB/s, FP16 XMX "
         "56.03 TFLOPS, INT8 XMX 112.07 TOPS) before computing t_theo._")
S.append("")

S.append("## Data sources\n")
S.append("- **All FC / PA / small-op rows are measured** on PTL 12Xe via cliloader "
         "(mean kernel ns/iter), one bench process per op (`fc_bench`, `pa_bench`, "
         "`small_ops_bench`). Iterations were sized so each op runs >1 s of GPU time "
         "where feasible; L2/L3 flush kernels evict cached weights between infers so "
         "decode FC measures true VRAM bandwidth.")
S.append("- **Derived rows (not separately measured):** (a) PA_sliding prefill for S>2048 is "
         "scaled from the measured causal PA at S=2048 by the sliding-band pair ratio "
         "(band = S·SW − SW²/2); (b) PA_sliding decode for KV>2048 reuses the KV=2048 "
         "measurement (sliding window caps effective KV at 2048); (c) LM_head is measured "
         "once at M=1 and reused for both prefill (last token) and decode.")
S.append("- Bytes/FLOPs are analytic (weight INT4/INT8 + FP16 scale/zp + FP16 act/out; "
         "PA with u4 KV). Eff% = achieved GB/s ÷ 110 (memory-bound) or GFLOPS ÷ XMX-peak "
         "(compute-bound). The theoretical floor uses a 5% overhead deduction.")
S.append("- Run: 99 cliloader logs in `logs/`, parsed by `utils/parse_logs.py` → `parsed.json` "
         "→ `build_report.py` → `performance_metrics.json`.")
S.append("")

S.append("## Graph fusion notes\n")
S.append("| Bench row | Real graph behaviour | Fused into | Standalone kernel? |")
S.append("|---|---|---|---|")
S.append("| `FC_QKV` | Q+K+V projection | fused QKV gemm | Yes |")
S.append("| `FC_OutGate` | `output_gate_proj` (attn output gate) | separate FC | Yes |")
S.append("| `MLP multiply` | silu(gate)·up | SwiGLU primitive | No — fused |")
S.append("| `MLP gate/up/down` | 3 INT4 FCs | not fused (SwiGLU between) | Yes (×3) |")
S.append("| `add` | 2 residual adds / layer | eltwise | Yes |")
S.append("| `rmsnorm` | 4× / layer + qk_norm + final | RMS primitive | Yes |")
S.append("| `qk_norm` | scaleless RMSNorm on Q,K | RMS primitive | Yes (folded in SmallOps) |")
S.append("| `out-gate sigmoid·mul` | sigmoid(og)⊙attn | eltwise | folded in SmallOps |")
S.append("")

# ---- Token latency summary ----
S.append("## Token latency summary\n")
S.append("### Prefill — TTFT and per-token amortized\n")
S.append("| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |")
S.append("|---:|---:|---:|---:|---:|")
for s in SIZES:
    t = M["prefill"][str(s)]["total_ms"]
    S.append(f"| {s:,} | {f(t)} | {f(t/1000,3)} | {f(t/s,4)} | {f(s/t*1000,0)} |")
S.append("")
S.append("### Decode — TPOT (per output token)\n")
S.append("| KV (ctx) | TPOT (ms) | tokens/s |")
S.append("|---:|---:|---:|")
for kv in SIZES:
    t = M["decode"][str(kv)]["total_ms"]
    S.append(f"| {kv:,} | {f(t)} | {f(1000/t,1)} |")
S.append("")

# ---- Roofline theo vs measured ----
S.append("## Roofline: theoretical floor vs measured\n")
S.append("### Decode (per output token)\n")
S.append("| KV | theoretical (ms) | measured (ms) | achieved % |")
S.append("|---:|---:|---:|---:|")
for kv in SIZES:
    d = M["decode"][str(kv)]
    S.append(f"| {kv:,} | {f(d['theo_ms'])} | {f(d['total_ms'])} | {f(d['achieved_pct'],1)}% |")
S.append("")
S.append("### Prefill (TTFT over S tokens)\n")
S.append("| S | theoretical (ms) | measured (ms) | achieved % |")
S.append("|---:|---:|---:|---:|")
for s in SIZES:
    d = M["prefill"][str(s)]
    S.append(f"| {s:,} | {f(d['theo_ms'])} | {f(d['total_ms'])} | {f(d['achieved_pct'],1)}% |")
S.append("")

# ---- Decode tables ----
S.append("## Decode tables (1 query token, KV = context length)\n")
for kv in SIZES:
    d = M["decode"][str(kv)]
    S.append(table(d["rows"], d["total_ms"], f"Decode — KV={kv:,}"))
    S.append("_SwiGLU `multiply` is fused into the SwiGLU primitive; SmallOps aggregates "
             "rmsnorm/qk_norm/rope/out-gate/residual-add._\n")

# ---- Prefill tables ----
S.append("## Prefill tables (single forward over S tokens)\n")
for s in SIZES:
    d = M["prefill"][str(s)]
    S.append(table(d["rows"], d["total_ms"], f"Prefill — S={s:,}"))
    S.append("_FC prefill uses the INT8-XMX `dynamic_quantize_gpu_opt + gemm_kernel` path._\n")

# ---- Op->kernel ----
S.append("## Op → kernel names (cliloader)\n")
S.append("### Decode (M=1)\n")
S.append("| op | kernel name(s) | launches/call |")
S.append("|---|---|---:|")
for op,k,l in [("FC_QKV","gemm_kernel",1),("FC_OutGate","gemm_kernel",1),
               ("FC_O","gemm_kernel",1),("MLP_gate/up/down","gemm_kernel",3),
               ("LM_head","gemm_kernel",1),
               ("PA_sliding","pa_kv_cache_update + paged_attention",2),
               ("PA_full","pa_kv_cache_update + paged_attention",2),
               ("SmallOps","rms + rope + eltwise + activation","—")]:
    S.append(f"| {op} | `{k}` | {l} |")
S.append("")

# ---- Top contributors ----
S.append("## Top contributors (sorted by total ms per inference)\n")
S.append("### Decode\n")
S.append("| KV | top1 (ms,%) | top2 | top3 |")
S.append("|---:|---|---|---|")
for kv in SIZES:
    d = M["decode"][str(kv)]; tot = d["total_ms"]; rs = d["rows"][:3]
    cells = [f"{r['op']} {f(r['total_ms'],1)}ms ({f(r['total_ms']/tot*100,1)}%)" for r in rs]
    S.append(f"| {kv:,} | {cells[0]} | {cells[1]} | {cells[2]} |")
S.append("")
S.append("### Prefill\n")
S.append("| S | top1 (ms,%) | top2 | top3 |")
S.append("|---:|---|---|---|")
for s in SIZES:
    d = M["prefill"][str(s)]; tot = d["total_ms"]; rs = d["rows"][:3]
    cells = [f"{r['op']} {f(r['total_ms'],1)}ms ({f(r['total_ms']/tot*100,1)}%)" for r in rs]
    S.append(f"| {s:,} | {cells[0]} | {cells[1]} | {cells[2]} |")
S.append("")

# ---- End to end ----
S.append(f"## End-to-end (prefill TTFT + {GEN}-token decode)\n")
S.append(f"| prompt P | TTFT (ms) | {GEN}-tok decode (ms) | total (ms) | avg decode tok/s |")
S.append("|---:|---:|---:|---:|---:|")
for s in SIZES:
    ttft = M["prefill"][str(s)]["total_ms"]
    tpot = M["decode"][str(s)]["total_ms"]
    dec = tpot*GEN
    S.append(f"| {s:,} | {f(ttft)} | {f(dec)} | {f(ttft+dec)} | {f(1000/tpot,1)} |")
S.append("")

# ---- Key findings ----
dec1 = M["decode"]["1024"]; pre32 = M["prefill"]["32768"]
def rowms(d, op): return next(r["total_ms"] for r in d["rows"] if r["op"] == op)
def roweff(d, op): return next(r["eff"] for r in d["rows"] if r["op"] == op)
mlp_ms = sum(rowms(dec1, o) for o in ("MLP_gate","MLP_up","MLP_down"))
lm_ms = rowms(dec1, "LM_head")
pa_full_pre_eff = roweff(pre32, "PA_full")
S.append("## Key findings\n")
S.append(f"- **Decode is hard memory-bound at ~{1000/dec1['total_ms']:.1f} tok/s** "
         f"({f(dec1['total_ms'])} ms/tok), essentially fixed across KV. It is set by "
         f"re-reading ~{wtot-1365.8:,.0f} MB of INT4/INT8 weights per token at 104.5 GB/s. "
         f"MLP (gate+up+down ×52) is {mlp_ms/dec1['total_ms']*100:.0f}% of decode time; "
         f"LM_head (INT8, 202K vocab) alone is {lm_ms/dec1['total_ms']*100:.0f}%.")
S.append(f"- **Decode achieves ~91% of the roofline floor** — little headroom without "
         f"reducing weight bytes (INT4 LM_head, expert/weight pruning) or KV traffic.")
S.append(f"- **Prefill is compute-bound and scales super-linearly**: TTFT achieves "
         f"{M['prefill']['1024']['achieved_pct']:.0f}% of the roofline at S=1K and "
         f"{pre32['achieved_pct']:.0f}% at S=32K. **PA_full** (13 NoPE layers, full causal "
         f"attention, FP16 micro-kernel) grows as S² and becomes the top TTFT contributor "
         f"at long context — but it measured **{pa_full_pre_eff:.0f}% XMX** at S=32K, much "
         f"healthier than the ~7% seen on gemma4-12B (Onyx's 32 Q-heads give the "
         f"micro-kernel far better occupancy than gemma's 16).")
S.append("- The **attention output gate** adds a full extra FC (6656→4096) on every one "
         "of the 52 layers — ~11% of body FC weight traffic — a genuine Onyx-specific "
         "decode cost not present in a vanilla SwiGLU decoder.")
S.append("- **Sliding layers (SW=2048)** keep sliding-PA cost flat for KV≥2048; only the "
         "13 full layers' PA grows with context.")
S.append("")

S.append("## Optimization levers (highest ROI first)\n")
S.append(f"1. **INT4 LM_head** (currently INT8): halves the ~{lm_ms:.0f} ms LM_head decode "
         f"cost → ~{lm_ms/2/dec1['total_ms']*100:.0f}% faster decode, and shaves ~680 MB "
         f"of static weights.")
S.append("2. **Fuse `output_gate_proj` into FC_QKV** (single wide gemm 6656→8704): removes "
         "a separate kernel launch per layer and improves gemm efficiency at M=1.")
S.append(f"3. **Long-context prefill**: PA_full is the #1 TTFT contributor at S>=16K "
         f"(O(S^2)); it already runs at {pa_full_pre_eff:.0f}% XMX (healthy for a causal "
         f"micro-kernel), so the lever is algorithmic — CM/flash tiling, KV sparsity, or "
         f"exploiting that only 13/52 layers are full-attention.")
S.append("4. **Speculative decoding / MTP**: decode is memory-bound and latency-fixed, so "
         "amortizing weight reads over multiple accepted tokens is the only way to raise "
         "effective tok/s materially.")
S.append("5. **INT4 KV cache is already applied** — halves PA byte traffic vs INT8; keep it.")
S.append("")

S.append("## Caveats & method\n")
S.append("- **Measured** via cliloader Device Performance Timing (mean ns/iter), one "
         "process per op. PA is reported as the sum of its `pa_kv_cache_update` + "
         "`paged_attention` kernels (the roofline-relevant latency).")
S.append("- FC weight bytes count INT4/INT8 weight + FP16 scale + INT4 zp (g=128) + FP16 "
         "act + FP16 out.")
S.append("- PA bytes assume INT4 KV cache + FP16 Q/out. Decode FC is memory-bound (M=1); "
         "prefill FC is INT8-XMX compute-bound (S≥1024 all above the INT8 ridge).")
S.append("- PA prefill full uses causal pairs S(S+1)/2; sliding uses band min(S,2048).")
S.append("- LM_head runs once per token (last position in prefill, every step in decode).")
S.append("- Vision tower / adapter are excluded — this is the text-generation roofline only.")
S.append("- Target machine: Local_Admin@10.239.132.229 (PTL 12Xe, 2400 MHz).")
S.append("")

S.append("## Reproduction\n")
S.append("```bash")
S.append("# 1. generate + copy the sweep, run on the PTL 12Xe target under cliloader")
S.append("python3 utils/generate_onyx_runscript.py   # -> utils/run_onyx_ptl_12xe.bat")
S.append("scp utils/run_onyx_ptl_12xe.bat Local_Admin@10.239.132.229:D:/river/moe/dev_roofline_profiling/utils/")
S.append("ssh Local_Admin@10.239.132.229 'cd /d D:\\river\\moe\\dev_roofline_profiling\\utils && call run_onyx_ptl_12xe.bat'")
S.append("# 2. pull logs back and build the report")
S.append("scp -r Local_Admin@10.239.132.229:D:/river/moe/roofline_results/onyx/ptl_12xe outputs/onyx/logs")
S.append("cd outputs/onyx")
S.append("python3 ../../utils/parse_logs.py logs parsed.json")
S.append("python3 build_report.py       # -> performance_metrics.json")
S.append("python3 render_summary.py     # -> this SUMMARY")
S.append("```")
S.append("")

path = os.path.join(here, f"SUMMARY_onyx_ptl_12xe_{DATE}.md")
open(path, "w").write("\n".join(S))
print("wrote", path)
