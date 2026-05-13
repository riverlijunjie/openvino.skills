# Gemma4 MoE (26B-A4B) — Roofline on PTL 12Xe (2025-05-12)

**Platform**: Intel Arc B390 iGPU (Panther Lake, 12 Xe cores, 8 EUs/core, 10 threads/EU)
**Model**: google/gemma-4-26b-a4b-it (Gemma4 MoE, 26B total / ~4B active)

- `hidden_size=2816`, `num_hidden_layers=30` (25 sliding + 5 full attention, 5:1 pattern)
- `num_attention_heads=16`, sliding: `num_key_value_heads=8, head_dim=256`; full: `num_key_value_heads=2, head_dim=512`
- `attention_k_eq_v=true` for full attention (V=K, no separate V projection)
- `sliding_window=1024`
- Dense MLP: `intermediate_size=2112`, GEGLU activation (`gelu_pytorch_tanh`)
- Sparse MoE: `moe_intermediate_size=704`, `num_experts=128`, `top_k_experts=8`
- Parallel dense MLP + sparse MoE architecture (both execute per layer, outputs summed)
- MatMul weights INT4 g=64 (MoE/dense MLP: 704,2112 not divisible by 128) or INT4 g=128 (attention FCs) / FP16 act; LM_head INT8 g=128 / FP16 act; KV cache FP16
- `vocab_size=262144`, `tie_word_embeddings=true`
- SDPA: PagedAttention (sliding window for 25 layers, full attention for 5 layers)

## Model parameters & weight shapes

Architecture knobs (parsed from model config):

| Field | Value | Notes |
|---|---:|---|
| `hidden_size` | 2816 | residual / activation channel |
| `num_hidden_layers` | 30 | 25 sliding + 5 full attention |
| `num_attention_heads` (NH) | 16 | Q heads |
| `num_key_value_heads_sliding` (NKV_S) | 8 | GQA: 2-way Q-per-KV sharing |
| `num_key_value_heads_full` (NKV_F) | 2 | GQA: 8-way Q-per-KV sharing |
| `head_dim_sliding` (HD_S) | 256 | Q_dim = 16×256 = 4096, KV_dim = 8×256 = 2048 |
| `head_dim_full` (HD_F) | 512 | Q_dim = 16×512 = 8192, KV_dim = 2×512 = 1024 |
| `intermediate_size` (dense MLP) | 2112 | GEGLU MLP hidden |
| `moe_intermediate_size` | 704 | Per-expert hidden dim |
| `num_experts` | 128 | Total experts in MoE |
| `top_k_experts` | 8 | Active experts per token |
| `vocab_size` | 262144 | LM head N |
| `hidden_act` | gelu_pytorch_tanh | GEGLU = x · GELU(gate(x)) |
| `tie_word_embeddings` | true | LM head shares embedding weights |
| `per_expert_scale` | true | Router has per-expert learned scale (may not fuse with GPU plugin) |

Per-layer weight matrices (one decoder block) and global weights:

| Weight | Shape (K × N) | Quant | Bytes / instance | × Layers | Total MB |
|---|---:|---|---:|---:|---:|
| Embedding | 262144 × 2816 | FP16 | 1,476,395,008 | 1 | 1,408.0 |
| FC_QKV_sliding (Q+K+V) | 2816 × 8192 | INT4 g=128 | 12,006,912 | 25 | 286.1 |
| FC_O_sliding | 4096 × 2816 | INT4 g=128 | 6,006,272 | 25 | 143.1 |
| FC_QKV_full (Q+K+V, V=K) | 2816 × 9216 | INT4 g=128 | 13,507,072 | 5 | 64.4 |
| FC_O_full | 8192 × 2816 | INT4 g=128 | 12,006,912 | 5 | 57.2 |
| FC_Dense_Gate (GEGLU) | 2816 × 2112 | INT4 g=64 | 3,215,872 | 30 | 92.0 |
| FC_Dense_Up (GEGLU) | 2816 × 2112 | INT4 g=64 | 3,215,872 | 30 | 92.0 |
| FC_Dense_Down | 2112 × 2816 | INT4 g=64 | 3,215,872 | 30 | 92.0 |
| MoE Gate (per expert) | 2816 × 704 | INT4 g=64 | 1,072,640 | 30 × 128 | 3,904.6 |
| MoE Up (per expert) | 2816 × 704 | INT4 g=64 | 1,072,640 | 30 × 128 | 3,904.6 |
| MoE Down (per expert) | 704 × 2816 | INT4 g=64 | 1,072,640 | 30 × 128 | 3,904.6 |
| Router | 2816 × 128 | FP16 | 720,896 | 30 | 20.6 |
| LM_Head | 2816 × 262144 | INT8 g=128 | 750,261,760 | 1 | 715.6 |
| **Total static weights** |  |  |  |  | **~14,685 MB** |

Activation / KV-cache shapes (S = sequence length, B = batch=1):

| Tensor | Shape | dtype | Bytes / token / layer | Bytes / token (all layers) |
|---|---|---|---:|---:|
| Hidden states | [B, S, 2816] | FP16 | 5,632 | — |
| Q sliding | [B, S, 16, 256] | FP16 | 8,192 | — |
| Q full | [B, S, 16, 512] | FP16 | 16,384 | — |
| K sliding (cache) | [B, S, 8, 256] | FP16 | 4,096 | 102,400 (25 layers) |
| V sliding (cache) | [B, S, 8, 256] | FP16 | 4,096 | 102,400 (25 layers) |
| K full (cache) | [B, S, 2, 512] | FP16 | 2,048 | 10,240 (5 layers) |
| V full (cache) | [B, S, 2, 512] | FP16 | 2,048 | 10,240 (5 layers) |
| **KV cache total** | per token | FP16 | sliding: 8,192 B, full: 4,096 B | **225,280 B / token** |

## Theoretical roofline

| Metric | Value |
|---|---|
| FP16 XMX peak | 58.982 TFLOPS |
| INT8 XMX peak | 117.964 TOPS |
| Memory BW (spec) | 110 GB/s |
| Memory BW (measured read) | ~105 GB/s |
| Ridge point (FP16) | 536 FLOP/byte |
| Ridge point (INT8) | 1073 OP/byte |

## Data sources

All ops measured directly on the target PTL 12Xe platform via cliloader Device Performance Timing.
67 benchmark log files collected, covering all FC, MoE, PA, and small ops for both decode (M=1) and prefill (S=1024,2048,4096,8192).

Benchmarks used:
- `fc_bench.exe` — Dense MatMul (attention FC and dense MLP)
- `moe_bench.exe` — Fused MoE 3gemm kernel
- `pa_bench.exe` — PagedAttention (sliding and full, decode and prefill)
- `small_ops_bench.exe` — RMSNorm, Add, RoPE

## Graph fusion notes

| Bench row | Real graph behaviour | Fused into | Standalone kernel in graph? |
|---|---|---|---|
| `multiply` | `gelu(gate(x)) ⊙ up(x)` of GEGLU dense MLP | GEGLU primitive | No — bench-only |
| `add` | Residual adds per layer | not fused (separate `eltwise`) | Yes |
| `rmsnorm` | Pre-attention + pre-MLP + post-final RMSNorm | single `RMS` primitive | Yes |
| `rope` | Q/K rotary embedding (sliding layers) | single `rope_opt` primitive | Yes |
| MoE `moe_3gemm_swiglu` | Fused gate+up+down for TK=8 experts | single fused MoE kernel (decode) | Yes |
| MoE `grouped_micro_gemm` | Grouped GEMM for TK=8 experts | grouped GEMM kernel (prefill) | Yes |

**Note on Gemma4 parallel architecture**: Each layer runs both a dense MLP and a sparse MoE in parallel. The dense MLP (gate+up+down via FC) and MoE (128 experts, top-8) outputs are summed. This differs from shared-expert MoE architectures (e.g., Qwen3.5) where the shared expert is folded into the MoE call.

## Token latency summary

### Prefill — TTFT and per-token amortized

| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|
| 1024 | 452.7 | 0.45 | 0.442 | 2,262 |
| 2048 | 812.8 | 0.81 | 0.397 | 2,520 |
| 4096 | 1,784.9 | 1.78 | 0.436 | 2,295 |
| 8192 | 4,549.6 | 4.55 | 0.555 | 1,801 |

### Decode — TPOT (per output token)

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 1024 | 28.60 | 35.0 |
| 2048 | 29.03 | 34.4 |
| 4096 | 28.87 | 34.6 |
| 8192 | 29.60 | 33.8 |
| 16384 | 30.91 | 32.4 |
| 32768 | 33.58 | 29.8 |
| 65536 | 38.97 | 25.7 |
| 131072 | 50.96 | 19.6 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)

| op | kv=1024 | kv=2048 | kv=4096 | kv=8192 | kv=16384 | kv=32768 | kv=65536 | kv=131072 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MoE (×30) | 8.65 (30.3%) | 8.65 (29.9%) | 8.65 (30.0%) | 8.65 (29.3%) | 8.65 (28.1%) | 8.65 (25.8%) | 8.65 (22.2%) | 8.65 (17.0%) |
| LM Head (×1) | 7.42 (26.0%) | 7.42 (25.6%) | 7.42 (25.8%) | 7.42 (25.1%) | 7.42 (24.1%) | 7.42 (22.1%) | 7.42 (19.1%) | 7.42 (14.6%) |
| QKV sliding (×25) | 2.99 (10.5%) | 2.99 (10.3%) | 2.99 (10.4%) | 2.99 (10.1%) | 2.99 (9.7%) | 2.99 (8.9%) | 2.99 (7.7%) | 2.99 (5.9%) |
| Dense gate+up (×60) | 2.12 (7.4%) | 2.12 (7.3%) | 2.12 (7.4%) | 2.12 (7.2%) | 2.12 (6.9%) | 2.12 (6.3%) | 2.12 (5.5%) | 2.12 (4.2%) |
| PA sliding (×25) | 2.09 (7.3%) | 2.09 (7.2%) | 2.09 (7.2%) | 2.09 (7.1%) | 2.09 (6.8%) | 2.09 (6.2%) | 2.09 (5.4%) | 2.09 (4.1%) |
| O sliding (×25) | 1.54 (5.4%) | 1.54 (5.3%) | 1.54 (5.3%) | 1.54 (5.2%) | 1.54 (5.0%) | 1.54 (4.6%) | 1.54 (4.0%) | 1.54 (3.0%) |
| Dense down (×30) | 1.11 (3.9%) | 1.11 (3.8%) | 1.11 (3.9%) | 1.11 (3.8%) | 1.11 (3.6%) | 1.11 (3.3%) | 1.11 (2.9%) | 1.11 (2.2%) |
| RMSNorm (×210) | 0.78 (2.7%) | 0.78 (2.7%) | 0.78 (2.7%) | 0.78 (2.6%) | 0.78 (2.5%) | 0.78 (2.3%) | 0.78 (2.0%) | 0.78 (1.5%) |
| QKV full (×5) | 0.67 (2.3%) | 0.67 (2.3%) | 0.67 (2.3%) | 0.67 (2.3%) | 0.67 (2.2%) | 0.67 (2.0%) | 0.67 (1.7%) | 0.67 (1.3%) |
| O full (×5) | 0.59 (2.1%) | 0.59 (2.0%) | 0.59 (2.0%) | 0.59 (2.0%) | 0.59 (1.9%) | 0.59 (1.8%) | 0.59 (1.5%) | 0.59 (1.2%) |
| PA full (×5) | 0.53 (1.9%) | 0.96 (3.3%) | 0.81 (2.8%) | 1.53 (5.2%) | 2.84 (9.2%) | 5.51 (16.4%) | 10.90 (28.0%) | 22.90 (44.9%) |
| Add (×60) | 0.12 (0.4%) | 0.12 (0.4%) | 0.12 (0.4%) | 0.12 (0.4%) | 0.12 (0.4%) | 0.12 (0.4%) | 0.12 (0.3%) | 0.12 (0.2%) |
| **TOTAL** | **28.60** | **29.03** | **28.87** | **29.60** | **30.91** | **33.58** | **38.97** | **50.96** |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)

| op | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---:|---:|---:|---:|
| MoE (×30) | 300.1 (66.3%) | 445.6 (54.8%) | 767.3 (43.0%) | 1250.4 (27.5%) |
| PA full prefill (×5) | 26.3 (5.8%) | 92.9 (11.4%) | 340.7 (19.1%) | 1376.6 (30.3%) |
| PA sliding prefill (×25) | 21.1 (4.6%) | 71.0 (8.7%) | 262.2 (14.7%) | 1085.2 (23.9%) |
| Dense gate+up (×60) | 25.7 (5.7%) | 46.7 (5.8%) | 94.2 (5.3%) | 179.6 (3.9%) |
| RMSNorm (×210) | 22.1 (4.9%) | 43.5 (5.4%) | 91.5 (5.1%) | 191.2 (4.2%) |
| QKV sliding (×25) | 20.1 (4.4%) | 39.6 (4.9%) | 78.5 (4.4%) | 159.7 (3.5%) |
| Dense down (×30) | 11.4 (2.5%) | 21.3 (2.6%) | 43.5 (2.4%) | 85.8 (1.9%) |
| O sliding (×25) | 11.3 (2.5%) | 21.1 (2.6%) | 41.0 (2.3%) | 82.0 (1.8%) |
| RoPE (×50) | 5.9 (1.3%) | 14.1 (1.7%) | 32.2 (1.8%) | 70.4 (1.5%) |
| QKV full (×5) | 4.4 (1.0%) | 8.9 (1.1%) | 17.9 (1.0%) | 36.1 (0.8%) |
| O full (×5) | 4.4 (1.0%) | 8.1 (1.0%) | 15.9 (0.9%) | 32.6 (0.7%) |
| **TOTAL** | **452.7** | **812.8** | **1784.9** | **4549.6** |

## Decode tables (1 query token, KV = context length)

### Decode — KV=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE (fused) | moe_3gemm_swiglu | 0.2883 | 30 | 8.649 | 326.9 | 89.1 | 81.0% BW | mem |
| LM Head | gemm_kernel | 7.4164 | 1 | 7.416 | 198.8 | 101.2 | 92.0% BW | mem |
| FC QKV sliding | gemm_kernel | 0.1198 | 25 | 2.994 | 384.0 | 100.3 | 91.1% BW | mem |
| FC dense gate+up (×2) | gemm_kernel | 0.0354 | 60 | 2.124 | 335.4 | 90.8 | 82.6% BW | mem |
| PA sliding (kv=1024) | paged_attention_opt | 0.0836 | 25 | 2.089 | — | 50.1 | 45.6% BW | mem |
| FC O sliding | gemm_kernel | 0.0614 | 25 | 1.536 | 375.5 | 97.8 | 88.9% BW | mem |
| FC dense down | gemm_kernel | 0.0370 | 30 | 1.109 | 321.2 | 87.0 | 79.1% BW | mem |
| RMSNorm | rms_gpu_bfyx_opt | 0.0037 | 210 | 0.777 | — | — | — | mem |
| FC QKV full | gemm_kernel | 0.1340 | 5 | 0.670 | 386.5 | 100.8 | 91.6% BW | mem |
| FC O full | gemm_kernel | 0.1172 | 5 | 0.586 | 392.6 | 102.5 | 93.1% BW | mem |
| PA full (kv=1024) | paged_attention_opt | 0.1069 | 5 | 0.535 | — | 19.6 | 17.8% BW | mem |
| Add | eltwise_simple | 0.0020 | 60 | 0.118 | — | — | — | mem |
| **TOTAL** |  |  |  | **28.60** |  |  |  |  |

### Decode — KV=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE (fused) | moe_3gemm_swiglu | 0.2883 | 30 | 8.649 | 326.9 | 89.1 | 81.0% BW | mem |
| LM Head | gemm_kernel | 7.4164 | 1 | 7.416 | 198.8 | 101.2 | 92.0% BW | mem |
| FC QKV sliding | gemm_kernel | 0.1198 | 25 | 2.994 | 384.0 | 100.3 | 91.1% BW | mem |
| FC dense gate+up (×2) | gemm_kernel | 0.0354 | 60 | 2.124 | 335.4 | 90.8 | 82.6% BW | mem |
| PA sliding (kv=1024) | paged_attention_opt | 0.0836 | 25 | 2.089 | — | 50.1 | 45.6% BW | mem |
| FC O sliding | gemm_kernel | 0.0614 | 25 | 1.536 | 375.5 | 97.8 | 88.9% BW | mem |
| FC dense down | gemm_kernel | 0.0370 | 30 | 1.109 | 321.2 | 87.0 | 79.1% BW | mem |
| PA full (kv=4096) | paged_attention_opt | 0.1611 | 5 | 0.805 | — | 52.1 | 47.3% BW | mem |
| RMSNorm | rms_gpu_bfyx_opt | 0.0037 | 210 | 0.777 | — | — | — | mem |
| FC QKV full | gemm_kernel | 0.1340 | 5 | 0.670 | 386.5 | 100.8 | 91.6% BW | mem |
| FC O full | gemm_kernel | 0.1172 | 5 | 0.586 | 392.6 | 102.5 | 93.1% BW | mem |
| Add | eltwise_simple | 0.0020 | 60 | 0.118 | — | — | — | mem |
| **TOTAL** |  |  |  | **28.87** |  |  |  |  |

### Decode — KV=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE (fused) | moe_3gemm_swiglu | 0.2883 | 30 | 8.649 | 326.9 | 89.1 | 81.0% BW | mem |
| LM Head | gemm_kernel | 7.4164 | 1 | 7.416 | 198.8 | 101.2 | 92.0% BW | mem |
| FC QKV sliding | gemm_kernel | 0.1198 | 25 | 2.994 | 384.0 | 100.3 | 91.1% BW | mem |
| FC dense gate+up (×2) | gemm_kernel | 0.0354 | 60 | 2.124 | 335.4 | 90.8 | 82.6% BW | mem |
| PA sliding (kv=1024) | paged_attention_opt | 0.0836 | 25 | 2.089 | — | 50.1 | 45.6% BW | mem |
| FC O sliding | gemm_kernel | 0.0614 | 25 | 1.536 | 375.5 | 97.8 | 88.9% BW | mem |
| PA full (kv=8192) | paged_attention_opt | 0.3054 | 5 | 1.527 | — | 54.9 | 49.9% BW | mem |
| FC dense down | gemm_kernel | 0.0370 | 30 | 1.109 | 321.2 | 87.0 | 79.1% BW | mem |
| RMSNorm | rms_gpu_bfyx_opt | 0.0037 | 210 | 0.777 | — | — | — | mem |
| FC QKV full | gemm_kernel | 0.1340 | 5 | 0.670 | 386.5 | 100.8 | 91.6% BW | mem |
| FC O full | gemm_kernel | 0.1172 | 5 | 0.586 | 392.6 | 102.5 | 93.1% BW | mem |
| Add | eltwise_simple | 0.0020 | 60 | 0.118 | — | — | — | mem |
| **TOTAL** |  |  |  | **29.60** |  |  |  |  |

### Decode — KV=32768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE (fused) | moe_3gemm_swiglu | 0.2883 | 30 | 8.649 | 326.9 | 89.1 | 81.0% BW | mem |
| LM Head | gemm_kernel | 7.4164 | 1 | 7.416 | 198.8 | 101.2 | 92.0% BW | mem |
| PA full (kv=32768) | paged_attention_opt | 1.1028 | 5 | 5.514 | — | 60.9 | 55.3% BW | mem |
| FC QKV sliding | gemm_kernel | 0.1198 | 25 | 2.994 | 384.0 | 100.3 | 91.1% BW | mem |
| FC dense gate+up (×2) | gemm_kernel | 0.0354 | 60 | 2.124 | 335.4 | 90.8 | 82.6% BW | mem |
| PA sliding (kv=1024) | paged_attention_opt | 0.0836 | 25 | 2.089 | — | 50.1 | 45.6% BW | mem |
| FC O sliding | gemm_kernel | 0.0614 | 25 | 1.536 | 375.5 | 97.8 | 88.9% BW | mem |
| FC dense down | gemm_kernel | 0.0370 | 30 | 1.109 | 321.2 | 87.0 | 79.1% BW | mem |
| RMSNorm | rms_gpu_bfyx_opt | 0.0037 | 210 | 0.777 | — | — | — | mem |
| FC QKV full | gemm_kernel | 0.1340 | 5 | 0.670 | 386.5 | 100.8 | 91.6% BW | mem |
| FC O full | gemm_kernel | 0.1172 | 5 | 0.586 | 392.6 | 102.5 | 93.1% BW | mem |
| Add | eltwise_simple | 0.0020 | 60 | 0.118 | — | — | — | mem |
| **TOTAL** |  |  |  | **33.58** |  |  |  |  |

### Decode — KV=131072

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA full (kv=131072) | paged_attention_opt | 4.5791 | 5 | 22.895 | — | 58.6 | 53.3% BW | mem |
| MoE (fused) | moe_3gemm_swiglu | 0.2883 | 30 | 8.649 | 326.9 | 89.1 | 81.0% BW | mem |
| LM Head | gemm_kernel | 7.4164 | 1 | 7.416 | 198.8 | 101.2 | 92.0% BW | mem |
| FC QKV sliding | gemm_kernel | 0.1198 | 25 | 2.994 | 384.0 | 100.3 | 91.1% BW | mem |
| FC dense gate+up (×2) | gemm_kernel | 0.0354 | 60 | 2.124 | 335.4 | 90.8 | 82.6% BW | mem |
| PA sliding (kv=1024) | paged_attention_opt | 0.0836 | 25 | 2.089 | — | 50.1 | 45.6% BW | mem |
| FC O sliding | gemm_kernel | 0.0614 | 25 | 1.536 | 375.5 | 97.8 | 88.9% BW | mem |
| FC dense down | gemm_kernel | 0.0370 | 30 | 1.109 | 321.2 | 87.0 | 79.1% BW | mem |
| RMSNorm | rms_gpu_bfyx_opt | 0.0037 | 210 | 0.777 | — | — | — | mem |
| FC QKV full | gemm_kernel | 0.1340 | 5 | 0.670 | 386.5 | 100.8 | 91.6% BW | mem |
| FC O full | gemm_kernel | 0.1172 | 5 | 0.586 | 392.6 | 102.5 | 93.1% BW | mem |
| Add | eltwise_simple | 0.0020 | 60 | 0.118 | — | — | — | mem |
| **TOTAL** |  |  |  | **50.96** |  |  |  |  |

## Prefill tables (single forward over S tokens)

### Prefill — S=1024

| op | kernel | single ms | calls | total ms | TFLOPS/TOPS | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---|
| MoE (fused) | grouped_micro_gemm | 10.004 | 30 | 300.117 | 9.74 TOPS | 8.3% XMX | compute |
| PA full prefill | sdpa_opt__multi_tokens | 5.267 | 5 | 26.334 | 6.53 TFLOPS | 11.1% XMX | compute |
| FC dense gate+up (×2) | gemm_kernel | 0.428 | 60 | 25.697 | 28.44 TOPS | 24.1% XMX | compute |
| RMSNorm | rms_gpu_bfyx_opt | 0.105 | 210 | 22.070 | — | — | mem |
| FC QKV sliding | gemm_kernel | 0.804 | 25 | 20.088 | 58.80 TOPS | 49.8% XMX | compute |
| PA sliding prefill | sdpa_micro__prefill | 0.842 | 25 | 21.050 | 20.40 TFLOPS | 34.6% XMX | compute |
| FC dense down | gemm_kernel | 0.381 | 30 | 11.429 | 31.97 TOPS | 27.1% XMX | compute |
| FC O sliding | gemm_kernel | 0.451 | 25 | 11.273 | 52.39 TOPS | 44.4% XMX | compute |
| RoPE | rope_opt | 0.117 | 50 | 5.870 | — | — | mem |
| FC QKV full | gemm_kernel | 0.886 | 5 | 4.428 | 60.02 TOPS | 50.9% XMX | compute |
| FC O full | gemm_kernel | 0.876 | 5 | 4.380 | 53.93 TOPS | 45.7% XMX | compute |
| **TOTAL** |  |  |  | **452.7** |  |  |  |

### Prefill — S=2048

| op | kernel | single ms | calls | total ms | TFLOPS/TOPS | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---|
| MoE (fused) | grouped_micro_gemm | 14.855 | 30 | 445.643 | 13.12 TOPS | 11.1% XMX | compute |
| PA full prefill | sdpa_opt__multi_tokens | 18.587 | 5 | 92.933 | 7.40 TFLOPS | 12.5% XMX | compute |
| PA sliding prefill | sdpa_micro__prefill | 2.839 | 25 | 70.976 | 24.21 TFLOPS | 41.0% XMX | compute |
| FC dense gate+up (×2) | gemm_kernel | 0.779 | 60 | 46.741 | 31.27 TOPS | 26.5% XMX | compute |
| RMSNorm | rms_gpu_bfyx_opt | 0.207 | 210 | 43.515 | — | — | mem |
| FC QKV sliding | gemm_kernel | 1.584 | 25 | 39.594 | 59.66 TOPS | 50.6% XMX | compute |
| FC dense down | gemm_kernel | 0.709 | 30 | 21.277 | 34.35 TOPS | 29.1% XMX | compute |
| FC O sliding | gemm_kernel | 0.843 | 25 | 21.075 | 56.04 TOPS | 47.5% XMX | compute |
| RoPE | rope_opt | 0.282 | 50 | 14.107 | — | — | mem |
| FC QKV full | gemm_kernel | 1.774 | 5 | 8.871 | 59.92 TOPS | 50.8% XMX | compute |
| FC O full | gemm_kernel | 1.618 | 5 | 8.089 | 58.41 TOPS | 49.5% XMX | compute |
| **TOTAL** |  |  |  | **812.8** |  |  |  |

### Prefill — S=4096

| op | kernel | single ms | calls | total ms | TFLOPS/TOPS | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---|
| MoE (fused) | grouped_micro_gemm | 25.576 | 30 | 767.266 | 15.24 TOPS | 12.9% XMX | compute |
| PA full prefill | sdpa_opt__multi_tokens | 68.131 | 5 | 340.656 | 8.07 TFLOPS | 13.7% XMX | compute |
| PA sliding prefill | sdpa_micro__prefill | 10.487 | 25 | 262.176 | 26.21 TFLOPS | 44.4% XMX | compute |
| FC dense gate+up (×2) | gemm_kernel | 1.570 | 60 | 94.186 | 31.04 TOPS | 26.3% XMX | compute |
| RMSNorm | rms_gpu_bfyx_opt | 0.436 | 210 | 91.487 | — | — | mem |
| FC QKV sliding | gemm_kernel | 3.141 | 25 | 78.519 | 60.17 TOPS | 51.0% XMX | compute |
| FC dense down | gemm_kernel | 1.451 | 30 | 43.536 | 33.57 TOPS | 28.5% XMX | compute |
| FC O sliding | gemm_kernel | 1.641 | 25 | 41.015 | 57.59 TOPS | 48.8% XMX | compute |
| RoPE | rope_opt | 0.644 | 50 | 32.213 | — | — | mem |
| FC QKV full | gemm_kernel | 3.579 | 5 | 17.896 | 59.40 TOPS | 50.4% XMX | compute |
| FC O full | gemm_kernel | 3.189 | 5 | 15.946 | 59.26 TOPS | 50.2% XMX | compute |
| **TOTAL** |  |  |  | **1784.9** |  |  |  |

### Prefill — S=8192

| op | kernel | single ms | calls | total ms | TFLOPS/TOPS | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---|
| PA full prefill | sdpa_opt__multi_tokens | 275.312 | 5 | 1376.560 | 7.99 TFLOPS | 13.5% XMX | compute |
| MoE (fused) | grouped_micro_gemm | 41.679 | 30 | 1250.380 | 18.70 TOPS | 15.9% XMX | compute |
| PA sliding prefill | sdpa_micro__prefill | 43.410 | 25 | 1085.240 | 25.33 TFLOPS | 42.9% XMX | compute |
| RMSNorm | rms_gpu_bfyx_opt | 0.911 | 210 | 191.237 | — | — | mem |
| FC dense gate+up (×2) | gemm_kernel | 2.993 | 60 | 179.604 | 32.55 TOPS | 27.6% XMX | compute |
| FC QKV sliding | gemm_kernel | 6.388 | 25 | 159.698 | 59.17 TOPS | 50.2% XMX | compute |
| FC dense down | gemm_kernel | 2.861 | 30 | 85.844 | 34.05 TOPS | 28.9% XMX | compute |
| FC O sliding | gemm_kernel | 3.279 | 25 | 81.973 | 57.63 TOPS | 48.9% XMX | compute |
| RoPE | rope_opt | 1.408 | 50 | 70.422 | — | — | mem |
| FC QKV full | gemm_kernel | 7.214 | 5 | 36.069 | 58.94 TOPS | 50.0% XMX | compute |
| FC O full | gemm_kernel | 6.523 | 5 | 32.613 | 57.95 TOPS | 49.1% XMX | compute |
| **TOTAL** |  |  |  | **4549.6** |  |  |  |

## Top contributors (sorted by total ms per inference)

### Decode

| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | MoE 8.65ms (30.2%) | LM Head 7.42ms (25.9%) | QKV sliding 2.99ms (10.5%) |
| 4096 | MoE 8.65ms (30.0%) | LM Head 7.42ms (25.7%) | QKV sliding 2.99ms (10.4%) |
| 8192 | MoE 8.65ms (29.2%) | LM Head 7.42ms (25.1%) | QKV sliding 2.99ms (10.1%) |
| 32768 | MoE 8.65ms (25.8%) | LM Head 7.42ms (22.1%) | PA full 5.51ms (16.4%) |
| 65536 | PA full 10.90ms (28.0%) | MoE 8.65ms (22.2%) | LM Head 7.42ms (19.0%) |
| 131072 | PA full 22.90ms (44.9%) | MoE 8.65ms (17.0%) | LM Head 7.42ms (14.6%) |

### Prefill

| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | MoE 300.1ms (66.3%) | PA full 26.3ms (5.8%) | Dense gate+up 25.7ms (5.7%) |
| 2048 | MoE 445.6ms (54.8%) | PA full 92.9ms (11.4%) | PA sliding 71.0ms (8.7%) |
| 4096 | MoE 767.3ms (43.0%) | PA full 340.7ms (19.1%) | PA sliding 262.2ms (14.7%) |
| 8192 | PA full 1376.6ms (30.3%) | MoE 1250.4ms (27.5%) | PA sliding 1085.2ms (23.9%) |

## Key findings & optimization opportunities

### Decode analysis

1. **MoE is the dominant bottleneck** (30% of TPOT at kv=1024, 8.65ms total). The fused `moe_3gemm_swiglu` kernel achieves 89.1 GB/s = 81% BW utilization. Since it loads weights for 8 active experts out of 128 per token, the memory footprint per call is modest (~25.7 MB), but the sheer number of calls (30 layers) accumulates.

2. **LM Head is the second bottleneck** (7.42ms, 26% of TPOT). At 262144 output classes with INT8 weights (~750 MB), it achieves excellent 101.2 GB/s = 92% BW. This is irreducible at current weight precision — speculative decoding could amortize it.

3. **Attention FC ops achieve excellent BW utilization** (91-93% for QKV/O sliding and full). The larger attention matrices (8192, 9216 columns) benefit from high BW efficiency at INT4 g=128.

4. **Dense MLP gate+up/down have lower BW efficiency** (79-83%). The smaller matrix size (K=2816, N=2112) with g=64 quantization results in higher scale/zp overhead ratio.

5. **PA sliding (kv=1024)** achieves 45.6% BW (50.1 GB/s) — with SW=1024 properly enabled, reads exactly 1024 tokens of KV cache per head.

6. **PA full attention** efficiency increases with context length: 17.8% at kv=1024 → 55.3% at kv=32768. At kv=131072, PA full becomes the dominant op (44.9% of TPOT).

7. **Small ops** (RMSNorm + Add) contribute ~0.9ms total — negligible overhead.

### Prefill analysis

1. **MoE prefill is extremely inefficient**: 8-16% INT8 XMX utilization via `grouped_micro_gemm`. At S=1024, it consumes 66% of total TTFT (300ms). The small expert size (704 intermediate) and high expert count (128 with top-8) create challenging workload shaping for GPU GEMM.

2. **PA full prefill** is the second major bottleneck, scaling quadratically: 26ms at S=1024 → 1377ms at S=8192 (O(S²) scaling). Achieves only 7-8 TFLOPS = 11-14% FP16 XMX. The head_dim=512 for full attention makes each SDPA call heavier.

3. **PA sliding prefill** achieves 20-25 TFLOPS = 35-43% XMX (dispatched FLOPs basis). However, the `sdpa_micro` kernel processes the **full causal attention matrix** and applies sliding window masking element-wise (via `-INF` predication), rather than skipping tile blocks outside the window. At S=8192 with SW=1024, only 23.4% of attention pairs are "useful" — the effective XMX utilization for useful work drops to just 5.0% (vs 42.9% dispatched). This is a **major optimization opportunity**: tile-level window skipping could yield up to 4× speedup for large S.

4. **Attention FC ops** achieve ~50% INT8 XMX utilization (59-60 TOPS for QKV/O sliding/full). This is reasonable for INT4 weight dequantization + INT8 XMX compute.

5. **Dense MLP FC** achieves only 27-35 TOPS (23-30% XMX). The small N=2112 dimension limits parallelism.

### Optimization priorities

| Priority | Op | Current | Potential gain | Approach |
|---:|---|---|---|---|
| 1 | MoE prefill | 8-16% XMX | 3-5× | Better grouped GEMM tiling for small-expert MoE |
| 2 | PA sliding prefill | 35-43% FP16 XMX (dispatched), **5-17% useful** | **3-4×** | **Tile-level sliding window skip in sdpa_micro kernel** |
| 3 | PA full prefill | 11-14% FP16 XMX | 2-4× | FlashAttention-2 with HD=512 support |
| 4 | Dense MLP FC | 24-29% XMX | 1.5-2× | Better GEMM tiling for N=2112 |
| 5 | PA full decode | 47-55% BW | 1.5× | GQA-optimized PA for NKV=2, HD=512 |
| 6 | PA sliding decode | 46% BW | 1.5-2× | Small-window PA kernel tuning |

## Caveats & method

- Each op profiled in its own process via cliloader Device Performance Timing; we use mean kernel time per iteration.
- FC weight bytes count INT4 weight + FP16 scale + U4 zp (g=64 or g=128) + FP16 act + FP16 out. LM Head uses INT8 + FP16 scale (g=128).
- PA bytes assume FP16 KV cache + FP16 Q, FP16 out.
- Decode FC is treated as **memory-bound** (weights read dominates at M=1); prefill FC is **INT8 XMX compute-bound** (S big enough to hit XMX).
- Prefill PA at S≥1024 is compute-bound (FP16 XMX); decode PA is memory-bound.
- GEGLU multiply (gelu(gate)·up) is fused into the GEGLU primitive in real inference; listed separately for visibility.
- MoE in real inference uses per_expert_scale in the router — bench uses standard moe_bench without per_expert_scale, which measures the core 3gemm kernel accurately.
- lm_head is run only once per token (last position in prefill, every step in decode).
- Dense MLP gate and up share the same K×N shape (2816×2112), so a single bench measures both; calls counted as ×2 per layer.
- Sliding PA decode always uses kv=1024 (fixed window), regardless of total context length.
- Target machine: Local_Admin@10.239.132.229 (PTL 12Xe, Windows, Intel Arc B390 iGPU)

## Reproduction

```bat
REM Build bench utilities
cd D:\river\moe\dev_roofline_profiling\utils
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=D:\river\moe\openvino\release_install -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

REM Set environment
set PATH=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release;D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin;%PATH%
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set OUT=D:\river\moe\roofline_results\gemma4_moe\ptl

REM MoE decode
%CLI% -d --device-performance-timing D:\river\moe\dev_roofline_profiling\utils\build\Release\moe_bench.exe --B 1 --S 1 --H 2816 --I 704 --NE 128 --TK 8 --q u4 --g 64 > %OUT%\moe_decode_M1.log

REM MoE prefill (S=1024,2048,4096,8192)
for %%S in (1024 2048 4096 8192) do (
    %CLI% -d --device-performance-timing D:\river\moe\dev_roofline_profiling\utils\build\Release\moe_bench.exe --B 1 --S %%S --H 2816 --I 704 --NE 128 --TK 8 --q u4 --g 64 > %OUT%\moe_prefill_S%%S.log
)

REM FC benches (gate, down, QKV sliding/full, O sliding/full, LM head)
REM PA benches (sliding decode/prefill, full decode at kv=1024..131072, full prefill)
REM Small ops (rmsnorm, add, rope)
REM See run_gemma4_moe_ptl.bat for full script
```
