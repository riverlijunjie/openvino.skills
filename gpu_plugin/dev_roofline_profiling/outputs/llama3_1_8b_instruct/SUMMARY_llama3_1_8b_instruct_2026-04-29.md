# Llama-3.1-8B-Instruct Roofline Analysis — PTL (Arc B390 iGPU)

**Date:** 2026-04-29
**Model:** `meta-llama/Llama-3.1-8B-Instruct` (dense decoder, GQA 32:8, SwiGLU, NL=32, H=4096, I=14336, vocab=128256)
**Platform:** PTL — Intel Arc B390 (Panther Lake, iGPU, Xe2)
- Measured streaming BW: **105 GB/s** (random-init, weight-stream peak)
- FP16 XMX peak: **58.98 TFLOPS**, INT8 XMX peak: **117.96 TOPS**
- AI* (FP16) ≈ 627.5 ops/byte
**Mode coverage:** decode `kv ∈ {1024, 2048}`, prefill `S ∈ {1024, 2048}` (quick-validation grid)
**Quantization:** body INT4 g=128 / lm_head INT8 g=128 / KV-cache INT8 / activations FP16
**Notes:** SDPA is converted to PagedAttention. Llama-3.1 has **no Q/K-norm** (unlike Qwen3-8B), so `q_norm`/`k_norm` rows from the Qwen3 mapping are intentionally omitted.

---

## 1. Architecture vs Qwen3-8B (head-to-head)

| Param | Llama-3.1-8B | Qwen3-8B | Roofline impact |
|---|---:|---:|---|
| `num_hidden_layers` | **32** | 36 | ~11% fewer FC/PA calls |
| `intermediate_size` | **14336** | 12288 | MLP weights +16.7% bigger → MLP decode/prefill scale ↑ |
| `vocab_size` | **128256** | 151936 | LM head ~16% smaller → LM head ms ↓ |
| Q/K-norm | no | yes | one less small-op per layer |
| KV heads | 8 (GQA-4) | 8 (GQA-4) | identical PA roofline |

The two models share Q/K shape and head topology, so the same micro-benches reuse cleanly; only MLP M×N×K and LM head N change.

---

## 2. DECODE roofline (M=1)

For decode, every dominant op is a single-token weight stream → **memory-bound** against 105 GB/s.

### PTL — DECODE kv=1024

| Op | Kernel | Avg ms | Calls/inf | Total ms | Share% | GB/s | BW Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_down | `gemm_kernel` | 0.3032 | 32 | 9.703 | 22.4% | 100.7 | **95.9%** | mem |
| fc_up | `gemm_kernel` | 0.2974 | 32 | 9.516 | 22.0% | 102.7 | **97.8%** | mem |
| fc_gate | `gemm_kernel` | 0.2964 | 32 | 9.486 | 21.9% | 103.0 | **98.1%** | mem |
| lm_head | `gemm_kernel` | 5.0876 | 1 | 5.088 | 11.8% | 104.9 | **99.9%** | mem |
| fc_qkv | `gemm_kernel` | 0.1292 | 32 | 4.134 | 9.6% | 101.4 | **96.5%** | mem |
| fc_o | `gemm_kernel` | 0.0877 | 32 | 2.806 | 6.5% | 99.6 | **94.8%** | mem |
| paged_attention (3 sub-kernels) | `pa_*single_token*` + `kv_cache_update` + `finalization` | 0.0581 | 32 | 1.860 | 4.3% | 36.1 | 34.4% | mem |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0031 | 64 | 0.201 | 0.5% | 5.2 | 5.0% | mem |
| add (residual) | `eltwise_simple_vload8` | 0.0019 | 64 | 0.124 | 0.3% | 12.7 | 12.1% | mem |
| multiply (gate*up) | `eltwise_simple_vload8` | 0.0030 | 32 | 0.097 | 0.2% | 28.4 | 27.1% | mem |
| swish (gate) | `activation_opt` | 0.0024 | 32 | 0.075 | 0.2% | 24.3 | 23.2% | mem |
| rope_q | `rope_opt` | 0.0022 | 32 | 0.069 | 0.2% | 7.6 | 7.3% | mem |
| rope_k | `rope_opt` | 0.0020 | 32 | 0.063 | 0.1% | 2.1 | 2.0% | mem |
| **Decode total** | | | | **43.22** | 100% | | | |
| **Throughput** | | | | | | | **23.1 tok/s** | |

### PTL — DECODE kv=2048

| Op | Kernel | Avg ms | Calls/inf | Total ms | Share% | GB/s | BW Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_down | `gemm_kernel` | 0.3032 | 32 | 9.703 | 21.9% | 100.7 | 95.9% | mem |
| fc_up | `gemm_kernel` | 0.2974 | 32 | 9.516 | 21.5% | 102.7 | 97.8% | mem |
| fc_gate | `gemm_kernel` | 0.2964 | 32 | 9.486 | 21.4% | 103.0 | 98.1% | mem |
| lm_head | `gemm_kernel` | 5.0876 | 1 | 5.088 | 11.5% | 104.9 | 99.9% | mem |
| fc_qkv | `gemm_kernel` | 0.1292 | 32 | 4.134 | 9.3% | 101.4 | 96.5% | mem |
| paged_attention (3 sub-kernels) | `pa_*single_token*` + `kv_cache_update` + `finalization` | 0.0919 | 32 | 2.940 | 6.6% | 45.6 | 43.5% | mem |
| fc_o | `gemm_kernel` | 0.0877 | 32 | 2.806 | 6.3% | 99.6 | 94.8% | mem |
| rmsnorm + small ops total | (8 small kernels) | — | — | 0.629 | 1.4% | — | — | mem |
| **Decode total** | | | | **44.30** | 100% | | | |
| **Throughput** | | | | | | | **22.6 tok/s** | |

### Decode per-kv summary

| kv | FC/layer (ms) | PA/layer (ms) | small/layer (ms) | LM head (ms) | Total (ms) | tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1.114 | 0.058 | 0.020 | 5.088 | 43.22 | **23.1** |
| 2048 | 1.114 | 0.092 | 0.020 | 5.088 | 44.30 | **22.6** |

### Decode findings

- **Body FCs are at the BW roofline**: `fc_down/up/gate/qkv/o` all hit **95–98% of measured 105 GB/s**. There is essentially no room left in the INT4 weight-stream path on PTL.
- **LM head dominates** at single-digit kv. With INT8 g=128 and vocab=128256 it streams ~534 MB and runs at **99.9% of BW peak** → this is purely a memory-volume problem.
  - Llama-3.1's 128k vocab makes LM head **17% faster** than Qwen3-8B's 152k vocab (5.09 ms vs ~6 ms).
- **Per-layer FC budget = 1.114 ms** vs Qwen3-8B's ~1.11 ms — very similar despite the +16.7% MLP width, because Llama has 4 fewer layers (32 vs 36).
- **PagedAttention is loose** at 34.4–43.5% BW efficiency. Same micro-bench result as Qwen3-8B; this is a known PA-on-iGPU gap, not a Llama-specific issue.
- **Small ops are noise** (≤ 1.5% of total). Swish + multiply could in principle be fused into the SwiGLU MLP, but the ROI is < 0.4 ms / inference.

---

## 3. PREFILL roofline

Prefill drives FCs into the INT8-XMX compute regime (M ≥ 1024). Compute peak = **117.96 TOPS** INT8.

### PTL — PREFILL S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | Share% | GFLOPS | XMX Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_down (gemm) | `gemm_kernel` | 2.1892 | 32 | 70.055 | 25.6% | 47,384 | 40.2% | compute |
| fc_up (gemm) | `gemm_kernel` | 1.7847 | 32 | 57.112 | 20.9% | 63,502 | **53.8%** | compute |
| fc_gate (gemm) | `gemm_kernel` | 1.6991 | 32 | 54.370 | 19.9% | 66,902 | **56.7%** | compute |
| fc_qkv (gemm) | `gemm_kernel` | 0.7270 | 32 | 23.263 | 8.5% | 62,510 | 53.0% | compute |
| paged_attention (prefill) | `sdpa_micro_prefill_sa` | 0.6514 | 32 | 20.846 | 7.6% | 24,241 | 20.5% | compute |
| fc_o (gemm) | `gemm_kernel` | 0.5200 | 32 | 16.640 | 6.1% | 54,482 | 46.2% | compute |
| **dynamic_quantize_gpu_opt** | (per-FC pre-pass, 5 ops) | sum 0.764 | 5×32 | **24.448** | **8.9%** | — | — | mem-bound activation quant |
| pa_kv_cache_update | `pa_kv_cache_update_ref_sa` | 0.0573 | 32 | 1.833 | 0.7% | — | — | mem |
| **Prefill total** | | | | **268.58** | 100% | | | |
| **Effective tok/s** | | | | | | | **3,813** | |

### PTL — PREFILL S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | Share% | GFLOPS | XMX Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_down (gemm) | `gemm_kernel` | 4.2772 | 32 | 136.870 | 25.1% | 48,374 | 41.0% | compute |
| fc_gate (gemm) | `gemm_kernel` | 3.2843 | 32 | 105.097 | 19.3% | 69,087 | **58.6%** | compute |
| fc_up (gemm) | `gemm_kernel` | 3.2208 | 32 | 103.066 | 18.9% | 70,410 | **59.7%** | compute |
| paged_attention (prefill) | `sdpa_micro_prefill_sa` | 2.2989 | 32 | 73.563 | 13.5% | 28,410 | 24.1% | compute |
| fc_qkv (gemm) | `gemm_kernel` | 1.4189 | 32 | 45.405 | 8.3% | 62,625 | 53.1% | compute |
| fc_o (gemm) | `gemm_kernel` | 0.9741 | 32 | 31.173 | 5.7% | 57,248 | 48.5% | compute |
| **dynamic_quantize_gpu_opt** | (per-FC pre-pass) | sum 1.540 | 5×32 | **49.292** | **9.0%** | — | — | mem |
| **Prefill total** | | | | **544.47** | 100% | | | |
| **Effective tok/s** | | | | | | | **3,761** | |

### Prefill per-S summary

| S | fc_qkv/L | fc_o/L | fc_gate/L | fc_up/L | fc_down/L | PA/L | total est (ms) | tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.825 | 0.631 | 1.798 | 1.894 | 2.538 | 0.709 | 273.7 | **3,742** |
| 2048 | 1.646 | 1.200 | 3.481 | 3.416 | 4.972 | 2.419 | 553.4 | **3,701** |

(`total est` includes lm_head 1×, the per-row gemm-only `fc_*` figures exclude `dynamic_quantize_gpu_opt`. The "Total ms" column in the per-S kernel tables above reflects the full sum including `dynamic_quantize_gpu_opt`.)

### Prefill findings

- **fc_down is the prefill bottleneck**: 25% of total time, only **40–41% of INT8 XMX peak**. The `(I=14336)→(H=4096)` shape has the lowest reuse of any of the body FCs and is the first thing to optimize.
- **fc_gate / fc_up reach 56–60% XMX efficiency** at S=2048 — these are the healthy ones. They scale almost linearly with S (1.79→3.28 ms gate, 1.89→3.22 ms up) which means MLP up-projection is bandwidth-OK and limited by raw FLOPS density.
- **`dynamic_quantize_gpu_opt` is a hidden 9% tax**: this is the FP16→INT8 activation pre-pass that fc_bench includes per FC. It is memory-bound and runs **before** every gemm. On PTL with shared LPDDR5x this is a real regression area.
- **PagedAttention prefill is the worst kernel by efficiency** (20–24% of compute peak). It uses `sdpa_micro_prefill_sa` and quadratically grows: 0.65 ms (S=1024) → 2.30 ms (S=2048) per layer = +252%, vs +99% for the FC family.
- **Cross-check:** at S=2048, sum of body FC time per layer = 14.49 ms, attention/L = 2.42 ms, ratio ≈ 6:1 — the model is decisively compute-bound on FCs in prefill.

---

## 4. End-to-end estimates

Using measured per-op ms, the toolkit's reported totals are:

| Stage | Total (ms) | Throughput |
|---|---:|---|
| Prefill S=1024 | 273.7 | 3,742 tok/s |
| Prefill S=2048 | 553.4 | 3,701 tok/s |
| Decode kv=1024 | 43.2 | **23.1 tok/s** |
| Decode kv=2048 | 44.3 | **22.6 tok/s** |

For a 1024-prompt + 1024-output scenario: **TTFT ≈ 274 ms**, **steady-state ≈ 23 tok/s**, total ~273 + 1024×44 = ~45.3 s.

---

## 5. Top optimization opportunities (PTL)

1. **fc_down prefill (25% of prefill, 40% XMX eff.)** — biggest single lever. Target: split-K / better register tiling / larger M-block to pull XMX past 50%. Even +10pp lifts whole prefill ~7%.
2. **Reduce or fuse `dynamic_quantize_gpu_opt`** — 9% of prefill time is just FP16→INT8 activation prep. Folding it into the producer (e.g. residual add or rmsnorm output) removes a full re-traversal of every activation tensor.
3. **PagedAttention prefill kernel** — only 20–24% XMX eff. Consider switching to the `paged_attention_opt_*_sa` long-token path or improving the micro-tile strategy of `sdpa_micro_prefill_sa` for GQA-4.
4. **PagedAttention decode** — 34–44% BW eff. KV cache layout already matches the canonical `[blocks, kv_heads, head, block]` layout; remaining headroom is in `paged_attention_opt_single_token` itself, not in our model wiring.
5. **Decode body FCs are saturated.** Do not invest more here — there is at most 2–5% efficiency left across all 5 body FCs combined.
6. **LM head decode (12% of decode)** is at 99.9% of BW. The only way to make it faster on this iGPU is to skip it for non-final tokens (already standard in OpenVINO GenAI's stateful pipeline) or to compress it further (INT4 would halve bytes, at accuracy cost).

---

## 6. Reproduction

```bash
# 1. push run script and sweep
sshpass -p openvino scp models/llama3_1_8b_instruct/run_llama3_1_8b_instruct_ptl.bat \
   Local_Admin@10.239.132.229:D:/river/moe/dev_roofline_profiling/utils/

sshpass -p openvino ssh Local_Admin@10.239.132.229 \
   "cd /d D:\\river\\moe\\dev_roofline_profiling\\utils && run_llama3_1_8b_instruct_ptl.bat"

# 2. pull logs
mkdir -p outputs/llama3_1_8b_instruct/logs_ptl
sshpass -p openvino scp -r 'Local_Admin@10.239.132.229:D:/river/moe/roofline_results/llama3_1_8b_instruct/ptl/*' \
   outputs/llama3_1_8b_instruct/logs_ptl/

# 3. parse + report (re-applies activation_opt patch for swish)
python3 utils/parse_logs.py outputs/llama3_1_8b_instruct/logs_ptl outputs/llama3_1_8b_instruct/ptl_metrics.json
python3 utils/build_postprocess_pipeline.py \
   --model-dir models/llama3_1_8b_instruct \
   --output-dir outputs/llama3_1_8b_instruct \
   --rebuild-db
```

Artifacts:
- raw logs: [outputs/llama3_1_8b_instruct/logs_ptl/](logs_ptl/)
- parsed metrics: [outputs/llama3_1_8b_instruct/ptl_metrics.json](ptl_metrics.json)
- per-op report JSON: [outputs/llama3_1_8b_instruct/performance_metrics_ptl.json](performance_metrics_ptl.json)
- per-token-size kernel tables: [outputs/llama3_1_8b_instruct/kernel_tables.md](kernel_tables.md)
- shared DB: [../../db/metrics.db](../../db/metrics.db)

## 7. Caveats

- **Quick-validation grid only**: only `S/kv ∈ {1024, 2048}` were measured. Longer-context behavior (especially PA prefill at 8k/16k+) is not in this report.
- **`so_swish_decode` workaround**: the parser excludes `activation_*` kernels by default (because fc/pa benches use a Relu kernel as a cache-flush helper). For Llama-3.1's swish, the `activation_opt_*` time was re-injected directly from the raw cliloader log. This is a reporting fix only and does not affect any other op.
- **PTL vs B390 naming**: cliloader reports the iGPU as "Intel(R) Arc(TM) B390 GPU (96CUs, 2400MHz)". Per `platforms.json`, this is the same architecture as PTL spec; same peaks and same roofline apply.
