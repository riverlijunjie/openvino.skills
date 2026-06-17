# diffusion_gemma (26B-A4B-it) — Roofline on Intel PTL 12Xe (2026-06-15)

**Platform**: Intel Panther Lake 12-Xe iGPU (cliloader: *Intel(R) Arc(TM) B390 GPU, 96 EU / 12 Xe-cores, 2400 MHz*); FP16 XMX 58.98 TFLOPS, INT8 XMX 117.96 TOPS, LPDDR ~110 GB/s.
**Model**: `google/diffusiongemma-26B-A4B-it` — block-diffusion multimodal Gemma. This report profiles the **text decoder** (`DiffusionGemmaForBlockDiffusion.text_config`), whose dense+MoE transformer is architecturally identical to gemma4-26B-A4B-it.

- 30 layers, hidden 2816, 128-expert top-8 MoE **plus** a dense GEGLU MLP per layer (outputs summed); GQA sliding (25 L) + full (5 L) attention; vocab 262144, tied embeddings.
- MatMul weights **int4** (g128; down-proj g64) / FP16 act; LM_head **int8** g128 / FP16 act; KV cache **FP16** (SDPA path).
- SDPA: `ov::op::v13::ScaledDotProductAttention` → GPU `sdpa_micro` (uncompressed FP16 KV). **Not** PagedAttention.

## Diffusion workload semantics

`diffusion_gemma` is a **block-diffusion** model (`canvas_length=256`), not a standard autoregressive LLM:

- **Encoder / prefill** — autoregressive pass over the prompt (causal sliding/full attention) that builds a **read-only** KV cache. This is the TTFT phase; profiled as `prefill S∈{1024,2048,4096,8192}`.
- **Decoder / canvas step** — each denoising step refines a **fixed canvas of 256 tokens** with **bidirectional** self-attention (causal=0) plus cross-attention to the read-only encoder KV cache. The relevant per-step cost is therefore **M=256**, not M=1. LM_head runs over the whole 256-token canvas each step. Profiled as `canvas decode M=256`.
- An **M=1 reference decode** is also reported for comparison with classic AR LLMs, but it does **not** occur in real diffusion inference.

## Model parameters & weight shapes

| Field | Value | Notes |
|---|---:|---|
| `hidden_size` | 2816 | residual / activation channel |
| `num_hidden_layers` | 30 | 25 sliding + 5 full attention |
| `num_attention_heads` (NH) | 16 | Q heads |
| `num_key_value_heads` (sliding) | 8 | GQA 2-way; HD=256, window=1024 |
| `num_global_key_value_heads` (full) | 2 | GQA 8-way; global HD=512 |
| `intermediate_size` (dense) | 2112 | GEGLU dense MLP hidden |
| `moe_intermediate_size` | 704 | per-expert hidden |
| `num_experts` / `top_k` | 128 / 8 | softmax-routed MoE |
| `vocab_size` | 262144 | LM head N |
| `hidden_activation` | gelu_pytorch_tanh | GEGLU |
| `final_logit_softcapping` | 30.0 | tanh logit soft-cap |
| `tie_word_embeddings` | true | LM head shares embedding storage |
| `canvas_length` | 256 | diffusion denoising block size |

Per-layer / global weight matrices (K × N):

| Weight | Shape (K × N) | Quant | × instances | Total MB |
|---|---:|---|---:|---:|
| Embedding (tied) | 2816 × 262144 | int8 g128 | 1 | 704.0 |
| FC_QKV (sliding) | 2816 × 8192 | int4 g128 | 25 | 283.6 |
| FC_O (sliding) | 4096 × 2816 | int4 g128 | 25 | 141.8 |
| FC_QK (full, V=K) | 2816 × 9216 | int4 g128 | 5 | 63.8 |
| FC_O (full) | 8192 × 2816 | int4 g128 | 5 | 56.7 |
| MLP_gate | 2816 × 2112 | int4 g128 | 30 | 87.7 |
| MLP_up | 2816 × 2112 | int4 g128 | 30 | 87.7 |
| MLP_down | 2112 × 2816 | int4 g64 | 30 | 90.4 |
| Router | 2816 × 128 | int4 g128 | 30 | 5.3 |
| MoE Gate+Up (per expert) | 2816 × 1408 | int4 g128 | 3840 | 7486.9 |
| MoE Down (per expert) | 704 × 2816 | int4 g64 | 3840 | 3856.9 |
| LM_Head (tied → 0) | 2816 × 262144 | int8 g128 | 1 | 0.0 |
| **Total static weights** |  |  |  | **12864.9 MB** |

## Theoretical roofline

| Metric | Value |
|---|---|
| FP16 XMX peak | 58.98 TFLOPS |
| INT8 XMX peak | 117.96 TOPS |
| Memory BW | 110.0 GB/s |
| Ridge point (FP16) | 536.2 FLOP/byte |
| Ridge point (INT8) | 1072.4 OP/byte |

## Data sources

All op times are **measured on the target PTL 12Xe machine** via cliloader Device Performance Timing (mean kernel ns per iteration), one bench process per op. No cross-platform scaling was used. MoE and dense down-proj use group size **64** (reduction dims I=704 and K=2112 are not divisible by 128); all other FC use g128.

## Graph fusion notes

| Bench row | Real graph behaviour | Standalone kernel? |
|---|---|---|
| `MoE` | router + 3-GEMM experts fused into `MOE3GemmFusedCompressed` (`grouped_micro_gemm` ×3 + gather/scatter/softmax-topk) | Yes (fused family) |
| `MLP gate/up + GEGLU multiply` | dense GEGLU; `multiply` fused into MLP | gate/up/down standalone gemm |
| `add` | per-layer residual adds (×2: attn + MLP/MoE sum) | Yes (eltwise) |
| `rmsnorm` | 7×/layer (pre-attn, q/k norm, pre-MLP, pre-MoE, post) + final | Yes (`rms_gpu_bfyx_opt`) |
| `SDPA broadcast` | bench-only GQA KV-head expansion to satisfy core SDPA shape-infer | excluded (artifact) |

## Token latency summary

### Prefill — encoder TTFT

| S | TTFT (ms) | per-token (ms) | tokens/s |
|---:|---:|---:|---:|
| 1024 | 445.0 | 0.4345 | 2301 |
| 2048 | 719.7 | 0.3514 | 2846 |
| 4096 | 1331.9 | 0.3252 | 3075 |
| 8192 | 2596.4 | 0.3169 | 3155 |

### Canvas decode — per denoising step (M=256, bidirectional)

| KV (ctx) | step (ms) | canvas-tok/s | per-canvas-token (ms) |
|---:|---:|---:|---:|
| 1024 | 298.5 | 858 | 1.1661 |
| 2048 | 300.8 | 851 | 1.1750 |
| 4096 | 305.5 | 838 | 1.1935 |
| 8192 | 314.8 | 813 | 1.2296 |

### M=1 reference decode (not used in diffusion; for comparison only)

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 1024 | 29.65 | 33.7 |
| 2048 | 31.49 | 31.8 |
| 4096 | 34.79 | 28.7 |
| 8192 | 41.61 | 24.0 |

### Canvas-decode per-op breakdown (ms / % of step)

| op | KV=1024 | KV=2048 | KV=4096 | KV=8192 |
|---|---:|---:|---:|---:|
| FC_QKV_sliding | 6.915 (2.3%) | 6.915 (2.3%) | 6.915 (2.3%) | 6.915 (2.2%) |
| FC_O_sliding | 4.040 (1.4%) | 4.040 (1.3%) | 4.040 (1.3%) | 4.040 (1.3%) |
| FC_QK_full | 1.435 (0.5%) | 1.435 (0.5%) | 1.435 (0.5%) | 1.435 (0.5%) |
| FC_O_full | 1.360 (0.5%) | 1.360 (0.5%) | 1.360 (0.4%) | 1.360 (0.4%) |
| MLP_gate | 3.166 (1.1%) | 3.166 (1.1%) | 3.166 (1.0%) | 3.166 (1.0%) |
| MLP_up | 3.132 (1.0%) | 3.132 (1.0%) | 3.132 (1.0%) | 3.132 (1.0%) |
| MLP_down | 3.730 (1.2%) | 3.730 (1.2%) | 3.730 (1.2%) | 3.730 (1.2%) |
| MoE | 242.820 (81.3%) | 242.820 (80.7%) | 242.820 (79.5%) | 242.820 (77.1%) |
| SDPA_sliding | 8.192 (2.7%) | 8.192 (2.7%) | 8.192 (2.7%) | 8.192 (2.6%) |
| SDPA_full | 3.671 (1.2%) | 5.956 (2.0%) | 10.699 (3.5%) | 19.928 (6.3%) |
| LM_head | 9.192 (3.1%) | 9.192 (3.1%) | 9.192 (3.0%) | 9.192 (2.9%) |
| SmallOps(norm/rope/add) | 10.857 (3.6%) | 10.857 (3.6%) | 10.857 (3.6%) | 10.857 (3.4%) |
| **TOTAL** | **298.511** | **300.796** | **305.539** | **314.769** |

### Prefill TTFT per-op breakdown (ms / % of TTFT)

| op | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---:|---:|---:|---:|
| FC_QKV_sliding | 20.613 (4.6%) | 40.842 (5.7%) | 80.713 (6.1%) | 168.870 (6.5%) |
| FC_O_sliding | 11.774 (2.6%) | 22.596 (3.1%) | 43.427 (3.3%) | 88.458 (3.4%) |
| FC_QK_full | 4.457 (1.0%) | 8.989 (1.2%) | 17.953 (1.3%) | 36.430 (1.4%) |
| FC_O_full | 4.562 (1.0%) | 8.081 (1.1%) | 15.707 (1.2%) | 31.808 (1.2%) |
| MLP_gate | 8.397 (1.9%) | 15.772 (2.2%) | 31.803 (2.4%) | 60.658 (2.3%) |
| MLP_up | 8.525 (1.9%) | 16.465 (2.3%) | 31.745 (2.4%) | 61.909 (2.4%) |
| MLP_down | 11.608 (2.6%) | 21.336 (3.0%) | 43.610 (3.3%) | 87.396 (3.4%) |
| MoE | 313.126 (70.4%) | 460.175 (63.9%) | 761.764 (57.2%) | 1234.198 (47.5%) |
| SDPA_sliding | 16.508 (3.7%) | 16.508 (2.3%) | 16.508 (1.2%) | 16.508 (0.6%) |
| SDPA_full | 8.234 (1.9%) | 31.272 (4.3%) | 121.855 (9.1%) | 463.371 (17.8%) |
| SmallOps(norm/rope/add) | 37.170 (8.4%) | 77.621 (10.8%) | 166.765 (12.5%) | 346.839 (13.4%) |
| **TOTAL** | **444.975** | **719.657** | **1331.851** | **2596.444** |

## Roofline: theoretical floor vs measured

Theoretical floor = Σ max(bytes/BW, FLOP/peak) over modelable ops (SDPA counts the `sdpa_micro` kernel only; small-ops use measured).

### Canvas decode (M=256 step)

| KV | theoretical (ms) | measured (ms) | achieved % |
|---:|---:|---:|---:|
| 1024 | 38.1 | 298.5 | 12.8% |
| 2048 | 38.9 | 300.8 | 12.9% |
| 4096 | 40.4 | 305.5 | 13.2% |
| 8192 | 43.5 | 314.8 | 13.8% |

### Prefill (TTFT over S tokens)

| S | theoretical (ms) | measured (ms) | achieved % |
|---:|---:|---:|---:|
| 1024 | 124.8 | 445.0 | 28.0% |
| 2048 | 252.1 | 719.7 | 35.0% |
| 4096 | 524.1 | 1331.9 | 39.4% |
| 8192 | 1106.7 | 2596.4 | 42.6% |

## Canvas-decode tables (M=256, KV = context length)

### Canvas decode — KV=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE | moe_3gemm_fused | 8.094 | 30 | 242.820 | 3009.7 | 3.4 | 5.4% | compute(f16) |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 10.857 | 1 | 10.857 | 0.0 | 0.0 | 0.0% | memory |
| LM_head | dyn_quant+gemm | 9.192 | 1 | 9.192 | 41117.7 | 96.3 | 36.7% | compute(i8) |
| SDPA_sliding | sdpa_micro | 0.328 | 25 | 8.192 | 16384.8 | 44.8 | 29.2% | compute(f16) |
| FC_QKV_sliding | dyn_quant+gemm | 0.277 | 25 | 6.915 | 42698.6 | 63.4 | 38.1% | compute(i8) |
| FC_O_sliding | dyn_quant+gemm | 0.162 | 25 | 4.040 | 36542.6 | 58.7 | 32.6% | compute(i8) |
| MLP_down | dyn_quant+gemm | 0.124 | 30 | 3.730 | 24494.4 | 45.7 | 21.9% | compute(i8) |
| SDPA_full | sdpa_micro | 0.734 | 5 | 3.671 | 14626.0 | 18.6 | 26.1% | compute(f16) |
| MLP_gate | dyn_quant+gemm | 0.106 | 30 | 3.166 | 28852.2 | 53.0 | 25.7% | compute(i8) |
| MLP_up | dyn_quant+gemm | 0.104 | 30 | 3.132 | 29163.9 | 53.5 | 26.0% | compute(i8) |
| FC_QK_full | dyn_quant+gemm | 0.287 | 5 | 1.435 | 46282.5 | 68.1 | 41.3% | compute(i8) |
| FC_O_full | dyn_quant+gemm | 0.272 | 5 | 1.360 | 43424.0 | 64.5 | 38.7% | compute(i8) |
| **TOTAL** |  |  |  | **298.511** |  |  |  |  |

### Canvas decode — KV=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE | moe_3gemm_fused | 8.094 | 30 | 242.820 | 3009.7 | 3.4 | 5.4% | compute(f16) |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 10.857 | 1 | 10.857 | 0.0 | 0.0 | 0.0% | memory |
| LM_head | dyn_quant+gemm | 9.192 | 1 | 9.192 | 41117.7 | 96.3 | 36.7% | compute(i8) |
| SDPA_sliding | sdpa_micro | 0.328 | 25 | 8.192 | 16384.8 | 44.8 | 29.2% | compute(f16) |
| FC_QKV_sliding | dyn_quant+gemm | 0.277 | 25 | 6.915 | 42698.6 | 63.4 | 38.1% | compute(i8) |
| SDPA_full | sdpa_micro | 1.191 | 5 | 5.956 | 16225.8 | 15.0 | 29.0% | compute(f16) |
| FC_O_sliding | dyn_quant+gemm | 0.162 | 25 | 4.040 | 36542.6 | 58.7 | 32.6% | compute(i8) |
| MLP_down | dyn_quant+gemm | 0.124 | 30 | 3.730 | 24494.4 | 45.7 | 21.9% | compute(i8) |
| MLP_gate | dyn_quant+gemm | 0.106 | 30 | 3.166 | 28852.2 | 53.0 | 25.7% | compute(i8) |
| MLP_up | dyn_quant+gemm | 0.104 | 30 | 3.132 | 29163.9 | 53.5 | 26.0% | compute(i8) |
| FC_QK_full | dyn_quant+gemm | 0.287 | 5 | 1.435 | 46282.5 | 68.1 | 41.3% | compute(i8) |
| FC_O_full | dyn_quant+gemm | 0.272 | 5 | 1.360 | 43424.0 | 64.5 | 38.7% | compute(i8) |
| **TOTAL** |  |  |  | **300.796** |  |  |  |  |

### Canvas decode — KV=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE | moe_3gemm_fused | 8.094 | 30 | 242.820 | 3009.7 | 3.4 | 5.4% | compute(f16) |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 10.857 | 1 | 10.857 | 0.0 | 0.0 | 0.0% | memory |
| SDPA_full | sdpa_micro | 2.140 | 5 | 10.699 | 17061.2 | 12.3 | 30.4% | compute(f16) |
| LM_head | dyn_quant+gemm | 9.192 | 1 | 9.192 | 41117.7 | 96.3 | 36.7% | compute(i8) |
| SDPA_sliding | sdpa_micro | 0.328 | 25 | 8.192 | 16384.8 | 44.8 | 29.2% | compute(f16) |
| FC_QKV_sliding | dyn_quant+gemm | 0.277 | 25 | 6.915 | 42698.6 | 63.4 | 38.1% | compute(i8) |
| FC_O_sliding | dyn_quant+gemm | 0.162 | 25 | 4.040 | 36542.6 | 58.7 | 32.6% | compute(i8) |
| MLP_down | dyn_quant+gemm | 0.124 | 30 | 3.730 | 24494.4 | 45.7 | 21.9% | compute(i8) |
| MLP_gate | dyn_quant+gemm | 0.106 | 30 | 3.166 | 28852.2 | 53.0 | 25.7% | compute(i8) |
| MLP_up | dyn_quant+gemm | 0.104 | 30 | 3.132 | 29163.9 | 53.5 | 26.0% | compute(i8) |
| FC_QK_full | dyn_quant+gemm | 0.287 | 5 | 1.435 | 46282.5 | 68.1 | 41.3% | compute(i8) |
| FC_O_full | dyn_quant+gemm | 0.272 | 5 | 1.360 | 43424.0 | 64.5 | 38.7% | compute(i8) |
| **TOTAL** |  |  |  | **305.539** |  |  |  |  |

### Canvas decode — KV=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE | moe_3gemm_fused | 8.094 | 30 | 242.820 | 3009.7 | 3.4 | 5.4% | compute(f16) |
| SDPA_full | sdpa_micro | 3.986 | 5 | 19.928 | 17780.6 | 10.8 | 31.7% | compute(f16) |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 10.857 | 1 | 10.857 | 0.0 | 0.0 | 0.0% | memory |
| LM_head | dyn_quant+gemm | 9.192 | 1 | 9.192 | 41117.7 | 96.3 | 36.7% | compute(i8) |
| SDPA_sliding | sdpa_micro | 0.328 | 25 | 8.192 | 16384.8 | 44.8 | 29.2% | compute(f16) |
| FC_QKV_sliding | dyn_quant+gemm | 0.277 | 25 | 6.915 | 42698.6 | 63.4 | 38.1% | compute(i8) |
| FC_O_sliding | dyn_quant+gemm | 0.162 | 25 | 4.040 | 36542.6 | 58.7 | 32.6% | compute(i8) |
| MLP_down | dyn_quant+gemm | 0.124 | 30 | 3.730 | 24494.4 | 45.7 | 21.9% | compute(i8) |
| MLP_gate | dyn_quant+gemm | 0.106 | 30 | 3.166 | 28852.2 | 53.0 | 25.7% | compute(i8) |
| MLP_up | dyn_quant+gemm | 0.104 | 30 | 3.132 | 29163.9 | 53.5 | 26.0% | compute(i8) |
| FC_QK_full | dyn_quant+gemm | 0.287 | 5 | 1.435 | 46282.5 | 68.1 | 41.3% | compute(i8) |
| FC_O_full | dyn_quant+gemm | 0.272 | 5 | 1.360 | 43424.0 | 64.5 | 38.7% | compute(i8) |
| **TOTAL** |  |  |  | **314.769** |  |  |  |  |

## M=1 reference decode tables (KV = context length)

### Decode (M=1) — KV=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE | moe_3gemm_fused | 0.279 | 30 | 8.373 | 341.0 | 89.5 | 85.6% | memory |
| LM_head | gemm | 7.304 | 1 | 7.304 | 202.1 | 102.7 | 98.3% | memory |
| FC_QKV_sliding | gemm | 0.121 | 25 | 3.029 | 380.8 | 98.4 | 94.1% | memory |
| SDPA_sliding | sdpa_micro | 0.113 | 25 | 2.835 | 147.9 | 74.1 | 70.9% | memory |
| SDPA_full | sdpa_micro | 0.406 | 5 | 2.028 | 82.7 | 10.4 | 10.0% | memory |
| FC_O_sliding | gemm | 0.061 | 25 | 1.537 | 375.1 | 96.9 | 92.8% | memory |
| MLP_down | gemm | 0.037 | 30 | 1.102 | 323.8 | 86.3 | 82.6% | memory |
| MLP_up | gemm | 0.035 | 30 | 1.046 | 341.1 | 88.2 | 84.4% | memory |
| MLP_gate | gemm | 0.034 | 30 | 1.026 | 347.8 | 90.0 | 86.1% | memory |
| FC_QK_full | gemm | 0.134 | 5 | 0.670 | 387.1 | 100.0 | 95.7% | memory |
| FC_O_full | gemm | 0.117 | 5 | 0.586 | 393.8 | 101.7 | 97.3% | memory |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 0.118 | 1 | 0.118 | 0.0 | 0.0 | 0.0% | memory |
| **TOTAL** |  |  |  | **29.653** |  |  |  |  |

### Decode (M=1) — KV=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE | moe_3gemm_fused | 0.279 | 30 | 8.373 | 341.0 | 89.5 | 85.6% | memory |
| LM_head | gemm | 7.304 | 1 | 7.304 | 202.1 | 102.7 | 98.3% | memory |
| SDPA_full | sdpa_micro | 0.774 | 5 | 3.868 | 86.8 | 10.9 | 10.4% | memory |
| FC_QKV_sliding | gemm | 0.121 | 25 | 3.029 | 380.8 | 98.4 | 94.1% | memory |
| SDPA_sliding | sdpa_micro | 0.113 | 25 | 2.835 | 147.9 | 74.1 | 70.9% | memory |
| FC_O_sliding | gemm | 0.061 | 25 | 1.537 | 375.1 | 96.9 | 92.8% | memory |
| MLP_down | gemm | 0.037 | 30 | 1.102 | 323.8 | 86.3 | 82.6% | memory |
| MLP_up | gemm | 0.035 | 30 | 1.046 | 341.1 | 88.2 | 84.4% | memory |
| MLP_gate | gemm | 0.034 | 30 | 1.026 | 347.8 | 90.0 | 86.1% | memory |
| FC_QK_full | gemm | 0.134 | 5 | 0.670 | 387.1 | 100.0 | 95.7% | memory |
| FC_O_full | gemm | 0.117 | 5 | 0.586 | 393.8 | 101.7 | 97.3% | memory |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 0.118 | 1 | 0.118 | 0.0 | 0.0 | 0.0% | memory |
| **TOTAL** |  |  |  | **31.493** |  |  |  |  |

### Decode (M=1) — KV=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE | moe_3gemm_fused | 0.279 | 30 | 8.373 | 341.0 | 89.5 | 85.6% | memory |
| LM_head | gemm | 7.304 | 1 | 7.304 | 202.1 | 102.7 | 98.3% | memory |
| SDPA_full | sdpa_micro | 1.433 | 5 | 7.166 | 93.7 | 11.7 | 11.2% | memory |
| FC_QKV_sliding | gemm | 0.121 | 25 | 3.029 | 380.8 | 98.4 | 94.1% | memory |
| SDPA_sliding | sdpa_micro | 0.113 | 25 | 2.835 | 147.9 | 74.1 | 70.9% | memory |
| FC_O_sliding | gemm | 0.061 | 25 | 1.537 | 375.1 | 96.9 | 92.8% | memory |
| MLP_down | gemm | 0.037 | 30 | 1.102 | 323.8 | 86.3 | 82.6% | memory |
| MLP_up | gemm | 0.035 | 30 | 1.046 | 341.1 | 88.2 | 84.4% | memory |
| MLP_gate | gemm | 0.034 | 30 | 1.026 | 347.8 | 90.0 | 86.1% | memory |
| FC_QK_full | gemm | 0.134 | 5 | 0.670 | 387.1 | 100.0 | 95.7% | memory |
| FC_O_full | gemm | 0.117 | 5 | 0.586 | 393.8 | 101.7 | 97.3% | memory |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 0.118 | 1 | 0.118 | 0.0 | 0.0 | 0.0% | memory |
| **TOTAL** |  |  |  | **34.791** |  |  |  |  |

### Decode (M=1) — KV=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SDPA_full | sdpa_micro | 2.797 | 5 | 13.985 | 96.0 | 12.0 | 11.5% | memory |
| MoE | moe_3gemm_fused | 0.279 | 30 | 8.373 | 341.0 | 89.5 | 85.6% | memory |
| LM_head | gemm | 7.304 | 1 | 7.304 | 202.1 | 102.7 | 98.3% | memory |
| FC_QKV_sliding | gemm | 0.121 | 25 | 3.029 | 380.8 | 98.4 | 94.1% | memory |
| SDPA_sliding | sdpa_micro | 0.113 | 25 | 2.835 | 147.9 | 74.1 | 70.9% | memory |
| FC_O_sliding | gemm | 0.061 | 25 | 1.537 | 375.1 | 96.9 | 92.8% | memory |
| MLP_down | gemm | 0.037 | 30 | 1.102 | 323.8 | 86.3 | 82.6% | memory |
| MLP_up | gemm | 0.035 | 30 | 1.046 | 341.1 | 88.2 | 84.4% | memory |
| MLP_gate | gemm | 0.034 | 30 | 1.026 | 347.8 | 90.0 | 86.1% | memory |
| FC_QK_full | gemm | 0.134 | 5 | 0.670 | 387.1 | 100.0 | 95.7% | memory |
| FC_O_full | gemm | 0.117 | 5 | 0.586 | 393.8 | 101.7 | 97.3% | memory |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 0.118 | 1 | 0.118 | 0.0 | 0.0 | 0.0% | memory |
| **TOTAL** |  |  |  | **41.610** |  |  |  |  |

## Prefill tables (single forward over S tokens)

### Prefill — S=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE | moe_3gemm_fused | 10.438 | 30 | 313.126 | 9335.7 | 3.5 | 16.7% | compute(f16) |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 37.170 | 1 | 37.170 | 0.0 | 0.0 | 0.0% | memory |
| FC_QKV_sliding | dyn_quant+gemm | 0.825 | 25 | 20.613 | 57299.9 | 41.8 | 51.1% | compute(i8) |
| SDPA_sliding | sdpa_micro | 0.660 | 25 | 16.508 | 13021.2 | 38.1 | 23.2% | compute(f16) |
| FC_O_sliding | dyn_quant+gemm | 0.471 | 25 | 11.774 | 50159.0 | 42.7 | 44.8% | compute(i8) |
| MLP_down | dyn_quant+gemm | 0.387 | 30 | 11.608 | 31479.4 | 34.2 | 28.1% | compute(i8) |
| MLP_up | dyn_quant+gemm | 0.284 | 30 | 8.525 | 42862.0 | 46.3 | 38.2% | compute(i8) |
| MLP_gate | dyn_quant+gemm | 0.280 | 30 | 8.397 | 43514.8 | 47.0 | 38.8% | compute(i8) |
| SDPA_full | sdpa_micro | 1.647 | 5 | 8.234 | 10442.3 | 22.9 | 18.6% | compute(f16) |
| FC_O_full | dyn_quant+gemm | 0.912 | 5 | 4.562 | 51776.0 | 37.7 | 46.2% | compute(i8) |
| FC_QK_full | dyn_quant+gemm | 0.891 | 5 | 4.457 | 59629.0 | 42.7 | 53.2% | compute(i8) |
| **TOTAL** |  |  |  | **444.975** |  |  |  |  |

### Prefill — S=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE | moe_3gemm_fused | 15.339 | 30 | 460.175 | 12705.0 | 3.1 | 22.7% | compute(f16) |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 77.621 | 1 | 77.621 | 0.0 | 0.0 | 0.0% | memory |
| FC_QKV_sliding | dyn_quant+gemm | 1.634 | 25 | 40.842 | 57838.7 | 34.9 | 51.6% | compute(i8) |
| SDPA_full | sdpa_micro | 6.254 | 5 | 31.272 | 10992.7 | 12.1 | 19.6% | compute(f16) |
| FC_O_sliding | dyn_quant+gemm | 0.904 | 25 | 22.596 | 52271.7 | 37.9 | 46.6% | compute(i8) |
| MLP_down | dyn_quant+gemm | 0.711 | 30 | 21.336 | 34252.2 | 32.8 | 30.6% | compute(i8) |
| SDPA_sliding | sdpa_micro | 0.660 | 25 | 16.508 | 13021.2 | 38.1 | 23.2% | compute(f16) |
| MLP_up | dyn_quant+gemm | 0.549 | 30 | 16.465 | 44386.8 | 42.4 | 39.6% | compute(i8) |
| MLP_gate | dyn_quant+gemm | 0.526 | 30 | 15.772 | 46336.0 | 44.2 | 41.3% | compute(i8) |
| FC_QK_full | dyn_quant+gemm | 1.798 | 5 | 8.989 | 59126.9 | 34.9 | 52.8% | compute(i8) |
| FC_O_full | dyn_quant+gemm | 1.616 | 5 | 8.081 | 58465.6 | 35.3 | 52.2% | compute(i8) |
| **TOTAL** |  |  |  | **719.657** |  |  |  |  |

### Prefill — S=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE | moe_3gemm_fused | 25.392 | 30 | 761.764 | 15350.0 | 2.8 | 27.4% | compute(f16) |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 166.765 | 1 | 166.765 | 0.0 | 0.0 | 0.0% | memory |
| SDPA_full | sdpa_micro | 24.371 | 5 | 121.855 | 11281.6 | 6.2 | 20.1% | compute(f16) |
| FC_QKV_sliding | dyn_quant+gemm | 3.229 | 25 | 80.713 | 58534.1 | 31.6 | 52.2% | compute(i8) |
| MLP_down | dyn_quant+gemm | 1.454 | 30 | 43.610 | 33516.1 | 29.9 | 29.9% | compute(i8) |
| FC_O_sliding | dyn_quant+gemm | 1.737 | 25 | 43.427 | 54396.1 | 36.0 | 48.5% | compute(i8) |
| MLP_gate | dyn_quant+gemm | 1.060 | 30 | 31.803 | 45958.3 | 41.0 | 41.0% | compute(i8) |
| MLP_up | dyn_quant+gemm | 1.058 | 30 | 31.745 | 46042.9 | 41.0 | 41.1% | compute(i8) |
| FC_QK_full | dyn_quant+gemm | 3.591 | 5 | 17.953 | 59209.5 | 31.2 | 52.8% | compute(i8) |
| SDPA_sliding | sdpa_micro | 0.660 | 25 | 16.508 | 13021.2 | 38.1 | 23.2% | compute(f16) |
| FC_O_full | dyn_quant+gemm | 3.141 | 5 | 15.707 | 60156.8 | 32.5 | 53.7% | compute(i8) |
| **TOTAL** |  |  |  | **1331.851** |  |  |  |  |

### Prefill — S=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE | moe_3gemm_fused | 41.140 | 30 | 1234.198 | 18948.4 | 2.8 | 33.8% | compute(f16) |
| SDPA_full | sdpa_micro | 92.674 | 5 | 463.371 | 11865.7 | 3.3 | 21.2% | compute(f16) |
| SmallOps(norm/rope/add) | rms/rope/eltwise | 346.839 | 1 | 346.839 | 0.0 | 0.0 | 0.0% | memory |
| FC_QKV_sliding | dyn_quant+gemm | 6.755 | 25 | 168.870 | 55953.8 | 28.5 | 49.9% | compute(i8) |
| FC_O_sliding | dyn_quant+gemm | 3.538 | 25 | 88.458 | 53409.3 | 33.7 | 47.7% | compute(i8) |
| MLP_down | dyn_quant+gemm | 2.913 | 30 | 87.396 | 33448.5 | 28.8 | 29.8% | compute(i8) |
| MLP_up | dyn_quant+gemm | 2.064 | 30 | 61.909 | 47218.8 | 40.6 | 42.1% | compute(i8) |
| MLP_gate | dyn_quant+gemm | 2.022 | 30 | 60.658 | 48192.6 | 41.4 | 43.0% | compute(i8) |
| FC_QK_full | dyn_quant+gemm | 7.286 | 5 | 36.430 | 58358.5 | 28.9 | 52.1% | compute(i8) |
| FC_O_full | dyn_quant+gemm | 6.362 | 5 | 31.808 | 59413.2 | 30.2 | 53.0% | compute(i8) |
| SDPA_sliding | sdpa_micro | 0.660 | 25 | 16.508 | 13021.2 | 38.1 | 23.2% | compute(f16) |
| **TOTAL** |  |  |  | **2596.444** |  |  |  |  |

## Op → kernel names (cliloader)

### Canvas decode (M=256)

| op | kernel name(s) | launches/call |
|---|---|---:|
| FC_QKV_sliding | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| FC_O_sliding | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| FC_QK_full | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| FC_O_full | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| MLP_gate | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| MLP_up | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| MLP_down | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| MoE | `grouped_micro_gemm`<br>`moe_scatter_reduction_opt_moe_scatter_reduction_ref`<br>`moe_3gemm_swiglu_fuse_prefill_swiglu`<br>`moe_gather_ref_prefill_gather` | 3<br>1<br>1<br>1 |
| SDPA_sliding | `broadcast_gpu_ref`<br>`sdpa_micro` | 2<br>1 |
| SDPA_full | `sdpa_micro`<br>`broadcast_gpu_ref` | 1<br>2 |
| LM_head | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| SmallOps(norm/rope/add) | `rms_gpu_bfyx_opt`<br>`rope_opt`<br>`eltwise` | per-subop |

### Prefill (S=8192)

| op | kernel name(s) | launches/call |
|---|---|---:|
| FC_QKV_sliding | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| FC_O_sliding | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| FC_QK_full | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| FC_O_full | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| MLP_gate | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| MLP_up | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| MLP_down | `gemm_kernel`<br>`dynamic_quantize_gpu_opt` | 1<br>1 |
| MoE | `grouped_micro_gemm`<br>`moe_gather_ref_prefill_gather`<br>`moe_scatter_reduction_opt_moe_scatter_reduction_ref`<br>`moe_3gemm_swiglu_fuse_prefill_swiglu` | 3<br>1<br>1<br>1 |
| SDPA_sliding | `sdpa_micro`<br>`broadcast_gpu_ref` | 1<br>2 |
| SDPA_full | `sdpa_micro`<br>`broadcast_gpu_ref` | 1<br>2 |
| SmallOps(norm/rope/add) | `rms_gpu_bfyx_opt`<br>`rope_opt`<br>`eltwise` | per-subop |

## Per-kernel decomposition (representative)

### Canvas decode sub-kernels — KV=1024

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
| MoE | `grouped_micro_gemm` | 7.7545 | 3 | 30 | 232.636 | 77.9% |
| SDPA_sliding | `broadcast_gpu_ref` | 0.3920 | 2 | 25 | 9.801 | 3.3% |
| LM_head | `gemm_kernel` | 9.1665 | 1 | 1 | 9.167 | 3.1% |
| SDPA_sliding | `sdpa_micro` | 0.3277 | 1 | 25 | 8.192 | 2.7% |
| FC_QKV_sliding | `gemm_kernel` | 0.2564 | 1 | 25 | 6.410 | 2.1% |
| MoE | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 0.1546 | 1 | 30 | 4.637 | 1.6% |
| SDPA_full | `sdpa_micro` | 0.7341 | 1 | 5 | 3.671 | 1.2% |
| SDPA_full | `broadcast_gpu_ref` | 0.7017 | 2 | 5 | 3.509 | 1.2% |
| FC_O_sliding | `gemm_kernel` | 0.1363 | 1 | 25 | 3.408 | 1.1% |
| MLP_down | `gemm_kernel` | 0.0870 | 1 | 30 | 2.609 | 0.9% |
| MLP_gate | `gemm_kernel` | 0.0856 | 1 | 30 | 2.568 | 0.9% |
| MLP_up | `gemm_kernel` | 0.0843 | 1 | 30 | 2.530 | 0.8% |
| MoE | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.0546 | 1 | 30 | 1.639 | 0.5% |
| FC_QK_full | `gemm_kernel` | 0.2667 | 1 | 5 | 1.334 | 0.4% |

### Prefill sub-kernels — S=8192

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
| MoE | `grouped_micro_gemm` | 22.9400 | 3 | 30 | 688.201 | 26.5% |
| SDPA_full | `sdpa_micro` | 92.6742 | 1 | 5 | 463.371 | 17.8% |
| MoE | `moe_gather_ref_prefill_gather` | 7.0926 | 1 | 30 | 212.778 | 8.2% |
| FC_QKV_sliding | `gemm_kernel` | 6.0022 | 1 | 25 | 150.054 | 5.8% |
| MoE | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 4.1130 | 1 | 30 | 123.391 | 4.8% |
| MoE | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 2.6838 | 1 | 30 | 80.513 | 3.1% |
| FC_O_sliding | `gemm_kernel` | 2.6315 | 1 | 25 | 65.787 | 2.5% |
| MLP_down | `gemm_kernel` | 1.8648 | 1 | 30 | 55.944 | 2.2% |
| MoE | `reorder_data_fast_b1` | 1.6141 | 1 | 30 | 48.422 | 1.9% |
| MLP_up | `gemm_kernel` | 1.4650 | 1 | 30 | 43.949 | 1.7% |
| MLP_gate | `gemm_kernel` | 1.4456 | 1 | 30 | 43.368 | 1.7% |
| MoE | `reorder_data` | 1.4274 | 1 | 30 | 42.823 | 1.6% |
| FC_QK_full | `gemm_kernel` | 6.6544 | 1 | 5 | 33.272 | 1.3% |
| MLP_down | `dynamic_quantize_gpu_opt` | 1.0484 | 1 | 30 | 31.451 | 1.2% |

## Top contributors (per inference)

### Canvas decode

| KV | top1 | top2 | top3 |
|---:|---|---|---|
| 1024 | MoE 242.820ms (81.3%) | SmallOps(norm/rope/add) 10.857ms (3.6%) | LM_head 9.192ms (3.1%) |
| 2048 | MoE 242.820ms (80.7%) | SmallOps(norm/rope/add) 10.857ms (3.6%) | LM_head 9.192ms (3.1%) |
| 4096 | MoE 242.820ms (79.5%) | SmallOps(norm/rope/add) 10.857ms (3.6%) | SDPA_full 10.699ms (3.5%) |
| 8192 | MoE 242.820ms (77.1%) | SDPA_full 19.928ms (6.3%) | SmallOps(norm/rope/add) 10.857ms (3.4%) |

### Prefill

| S | top1 | top2 | top3 |
|---:|---|---|---|
| 1024 | MoE 313.126ms (70.4%) | SmallOps(norm/rope/add) 37.170ms (8.4%) | FC_QKV_sliding 20.613ms (4.6%) |
| 2048 | MoE 460.175ms (63.9%) | SmallOps(norm/rope/add) 77.621ms (10.8%) | FC_QKV_sliding 40.842ms (5.7%) |
| 4096 | MoE 761.764ms (57.2%) | SmallOps(norm/rope/add) 166.765ms (12.5%) | SDPA_full 121.855ms (9.1%) |
| 8192 | MoE 1234.198ms (47.5%) | SDPA_full 463.371ms (17.8%) | SmallOps(norm/rope/add) 346.839ms (13.4%) |

## End-to-end (encoder TTFT + canvas denoising)

Assuming an illustrative block-diffusion schedule of **32 denoising steps** over the 256-token canvas (steps reuse the read-only encoder KV cache; canvas-step cost ≈ constant across the schedule):

| prompt P | TTFT (ms) | 32-step canvas (ms) | total (ms) | canvas tok/s |
|---:|---:|---:|---:|---:|
| 1024 | 445.0 | 9552.4 | 9997.3 | 858 |
| 2048 | 719.7 | 9625.5 | 10345.1 | 851 |
| 4096 | 1331.9 | 9777.3 | 11109.1 | 838 |
| 8192 | 2596.4 | 10072.6 | 12669.0 | 813 |

## Key findings

- **MoE dominates everything.** It is 81% of a canvas-decode step (~243 ms of ~299 ms at KV=1024) and 47-69% of prefill TTFT. The `grouped_micro_gemm` expert kernel (3 launches/step) is the single largest cost — the same bottleneck seen on gemma4-26B-A4B.
- **Canvas decode runs at ~858 canvas-tok/s** (M=256, KV=1024), i.e. ~299 ms per denoising step. Because the canvas is processed in parallel (M=256), per-token throughput is ~25x the M=1 reference (34 tok/s).
- **Canvas decode is compute-bound** (M=256 saturates INT8 XMX for FC/LM_head and the FP16 MoE/SDPA kernels), unlike a classic M=1 LLM decode which is memory-bound. The M=1 reference is memory-bound and dominated by MoE weight reads.
- **Attention scales with context** only through the 5 full-attention layers (HD=512); sliding layers are capped at window=1024. Combined SDPA grows from ~4% (KV=1024) to ~9% (KV=8192) of the canvas step (the full-attention SDPA term alone scales ~5.4x over that range).
- **LM_head over the 256-token canvas** costs ~9 ms/step (int8, 2816×262144) — modest vs MoE but non-trivial; it runs every denoising step.

## Optimization levers (highest ROI first)

1. **MoE expert GEMM** — `grouped_micro_gemm` is >2/3 of decode and prefill. Larger expert tiling / better int4 grouped-GEMM scheduling, or expert batching across the 256 canvas tokens, is the top lever.
2. **Exploit canvas parallelism for MoE routing** — with M=256 tokens per step, expert load is dense; grouping tokens by expert (sort/gather) amortizes weight reads better than M=1.
3. **INT4 LM_head** — moving the tied LM_head from int8→int4 roughly halves its ~9 ms/step.
4. **Full-attention KV** — only 5 layers use HD=512 global attention; an INT8 / PagedAttention KV path would cut the SDPA growth at long context (currently FP16 KV).
5. **Fuse residual adds / norms** — small-ops are 3–13% of time; folding the two residual adds and q/k norms into neighboring primitives trims the tail.

## Caveats & method

- Each op profiled in its own process via cliloader Device Performance Timing; **mean** kernel ns per iteration is used. Times in **ms**.
- FC weight bytes = int4 weight + FP16 scale (g128, down/MoE g64) + FP16 act + FP16 out. LM_head int8.
- **SDPA uses FP16 (uncompressed) KV** — the bench models the SDPA path the model compiles to; INT8 KV compression (PagedAttention) was **not** used per the chosen attention implementation.
- The SDPA bench materializes an explicit GQA KV-head **broadcast** to satisfy core opset13 SDPA shape inference; that `broadcast_gpu_ref` kernel is a **bench artifact** (the GPU SDPA primitive reads compressed GQA KV directly) and is **excluded** — only `sdpa_micro` is counted.
- MoE and dense down-proj use **group size 64** (I=704, K=2112 not ÷128); scale-byte impact vs g128 is negligible.
- Canvas decode = **M=256** bidirectional step (the real diffusion cost). M=1 reference is informational only and does not occur in inference.
- The 32-step end-to-end schedule is illustrative; actual step count depends on the sampler.
- Target machine: Local PTL 12Xe (Arc B390, 96 EU, 2400 MHz), Windows, OpenVINO release build.

## Reproduction

```bat
:: On the PTL 12Xe machine (D:\river\moe\dev_roofline_profiling\utils)
run_diffusion_gemma_ptl_12xe.bat        :: main sweep (FC / MoE / LM_head / small-ops)
run_diffusion_gemma_ptl_12xe_sdpa.bat   :: SDPA (needs sdpa_bench rebuilt w/ GQA repeat_kv)
:: logs -> D:\river\moe\roofline_results\diffusion_gemma\ptl_12xe\*.log
```
```bash
# On the host
python3 parse_logs.py <logdir> outputs/diffusion_gemma/perf_raw.json
python3 analyze_diffusion_gemma_ptl_12xe.py   # -> performance_metrics.json + SUMMARY
```
