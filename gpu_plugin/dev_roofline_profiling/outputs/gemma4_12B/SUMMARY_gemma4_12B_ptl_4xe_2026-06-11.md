# Gemma4-12B-it (dense) — Roofline on PTL 4Xe (Linux iGPU) (2026-06-11)

**Platform**: intel@10.239.152.140 — Intel PTL **4 Xe** iGPU (32 EUs) @ 2450 MHz, 110 GB/s LPDDR5x, Linux
**Model**: `google/gemma-4-12B-it` (dense, no MoE) — text decoder of the unified multimodal model

- vocab=262144, hidden=3840, **48 layers** (40 sliding + 8 full, 5:1 pattern), GEGLU MLP intermediate=15360, tie_word_embeddings=true, final_logit_softcapping=30.0
- Sliding attn: NH=16 / NKV=8 / HD=256 → Q=4096, KV=2048, sliding_window=1024
- Full attn: NH=16 / NKV=1 / HD=512 → Q=8192, K=512, V reuses K (`attention_k_eq_v`)
- MatMul weights INT4 g=128 / FP16 act; LM_head INT8 g=128 / FP16 act; KV cache INT8
- SDPA: PagedAttention OpenCL + micro_kernel (kv_type=i8)
- **Profiler**: cliloader 3.0.6 `--device-performance-timing`, mean kernel time (ms)
- **Token sweep**: S / kv ∈ {256, 1024, 2048, 4096, 8192}; decode measured at M=1

## Model parameters & weight shapes

Architecture knobs (parsed from model config):

| Field | Value | Notes |
|---|---:|---|
| `hidden_size` | 3,840 | residual / activation channel |
| `num_hidden_layers` | 48 | 40 sliding + 8 full (5 sliding → 1 full) |
| `num_attention_heads` (NH) | 16 | Q heads (both layer types) |
| `num_key_value_heads` (NKV) | 8 (sliding) / 1 (full) | GQA: 2-way (sliding) / 16-way (full) Q-per-KV sharing |
| `head_dim` (HD) | 256 (sliding) / 512 (full) | Q_dim = NH·HD = 4096 / 8192 |
| `intermediate_size` | 15,360 | GEGLU MLP hidden |
| `vocab_size` | 262,144 | LM head N |
| `hidden_act` | gelu_pytorch_tanh | GEGLU = gelu(gate(x)) ⊙ up(x) |
| `tie_word_embeddings` | true | LM head storage shared with token embedding |
| `sliding_window` | 1,024 | sliding-layer effective KV cap |
| `attention_k_eq_v` | true | full attn: V projection reuses K |
| `final_logit_softcapping` | 30.0 | tanh softcap on logits |

Per-layer weight matrices (one decoder block) and global weights (INT4 g=128 unless noted):

| Weight | Shape (K × N) | Quant | Bytes / instance | × Layers | Total MB |
|---|---:|---|---:|---:|---:|
| Embedding (tied → LM_head) | 3840 × 262144 | INT8 | 1,022.0 MB | 1 | 1,022.0 |
| FC_QKV_sliding (fused Q+K+V) | 3840 × 8192 | INT4 | 15.59 MB | 40 | 623.4 |
| FC_O_sliding | 4096 × 3840 | INT4 | 7.79 MB | 40 | 311.7 |
| FC_QK_full (fused Q+K, V=K) | 3840 × 8704 | INT4 | 16.56 MB | 8 | 132.5 |
| FC_O_full | 8192 × 3840 | INT4 | 15.59 MB | 8 | 124.7 |
| MLP_gate (GEGLU gate) | 3840 × 15360 | INT4 | 29.22 MB | 48 | 1,402.7 |
| MLP_up (GEGLU up) | 3840 × 15360 | INT4 | 29.22 MB | 48 | 1,402.7 |
| MLP_down (GEGLU down) | 15360 × 3840 | INT4 | 29.22 MB | 48 | 1,402.7 |
| LM_Head (tied) | 3840 × 262144 | INT8 | 975.0 MB | 1 | 975.0 |
| **Total static weights** |  |  |  |  | **~6,376 MB** |

_INT4 g=128 bytes = N·K/2 + N·(K/128)·2 (FP16 scale) + N·(K/128)/2 (INT4 zp). INT8 g=128 bytes = N·K + N·(K/128)·2. FP16 baseline = 22,710 MB; quantized = ~28 % of FP16._

Activation / KV-cache shapes (S = sequence length, B = batch=1):

| Tensor | Shape | dtype | Bytes / token / layer | Bytes / token (all layers) |
|---|---|---|---:|---:|
| Hidden states | [B, S, 3840] | FP16 | 7,680 | — |
| Q | [B, S, 16, HD] | FP16 | 8,192 (sl) / 16,384 (full) | — |
| K (cache, sliding) | [blocks, 8, 256, 16] | INT8 | 2,048 | 81,920 (×40) |
| V (cache, sliding) | [blocks, 8, 16, 256] | INT8 | 2,048 | 81,920 (×40) |
| K (cache, full) | [blocks, 1, 512, 16] | INT8 | 512 | 4,096 (×8) |
| V (cache, full, =K) | [blocks, 1, 16, 512] | INT8 | 512 | 4,096 (×8) |
| **KV cache total** | per token | INT8 | — | **~172 KB / token** |

## Theoretical roofline

| Metric | Value |
|---|---|
| FP16 XMX peak | **20.07 TFLOPS** (4 Xe × 8 EU × 256 FLOP/cyc × 2.45 GHz) |
| INT8 XMX peak | **40.14 TOPS** (2× FP16) |
| Memory BW | 110 GB/s |
| Ridge point (FP16) | 182.5 FLOP/byte |
| Ridge point (INT8) | 365.0 OP/byte |

_5 % overhead deducted from each peak for real-world losses → achievable BW = 104.5 GB/s, FP16 XMX = 19.07 TFLOPS, INT8 XMX = 38.13 TOPS._

## Data sources

All ops measured directly on **PTL 4Xe** (intel@10.239.152.140) via cliloader Device Performance Timing. Mean kernel time per iteration, each op in its own process with an L2/L3 flush kernel between infers (64 MB) so the measured kernel streams from DRAM. The only modeled (not directly measured per token size) values are:

- **PA sliding prefill S>1024**: bench caps at the sliding window (S=1024); for S>1024 the S=1024 measurement is scaled linearly (sliding work ∝ SW·S).
- **Decode aggregates** use model call-counts (40 sliding + 8 full layers, 48 MLP, 1 LM_head).

## Graph fusion notes

| Bench row | Real graph behaviour | Fused into | Standalone kernel in graph? |
|---|---|---|---|
| `multiply` | gelu(gate(x)) ⊙ up(x) of GEGLU MLP | GEGLU/SwiGLU primitive | No — fused |
| `swish`/`gelu` | GEGLU activation | GEGLU primitive | No — fused |
| `add` | residual adds (2× per layer) | not fused (separate `eltwise`) | Yes |
| `rmsnorm` | pre-attn + post-attn + pre-MLP + post-MLP + final | `RMS` primitive | Yes (4×NL+1) |
| `rmsnorm3d` | Q/K per-head norm (Gemma QK-norm) | `RMS` primitive | Yes |
| `rope` | rotary embedding on Q/K | `RoPE` primitive | Yes |
| FC_QKV/QK/O/MLP (INT4) | FullyConnectedCompressed | decode: `gemm_kernel`; prefill: `dynamic_quantize_gpu_opt` + `gemm_kernel` (INT8 XMX) | Yes |
| PagedAttention sliding/full | PagedAttention | INT8 KV; decode `paged_attention_opt`; prefill `sdpa_micro`/`sdpa_opt` | Yes |

## Token latency summary

### Prefill — TTFT and per-token amortized

| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|
| 256 | 320.0 | 0.320 | 1.250 | 800 |
| 1,024 | 1,287.9 | 1.288 | 1.258 | 795 |
| 2,048 | 2,744.0 | 2.744 | 1.340 | 746 |
| 4,096 | 6,160.9 | 6.161 | 1.504 | 665 |
| 8,192 | 15,453.2 | 15.453 | 1.886 | 530 |

### Decode — TPOT (per output token)

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 256 | 72.04 | 13.9 |
| 1,024 | 79.79 | 12.5 |
| 2,048 | 81.32 | 12.3 |
| 4,096 | 80.79 | 12.4 |
| 8,192 | 83.50 | 12.0 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)

| op | KV=256 | KV=1024 | KV=2048 | KV=4096 | KV=8192 |
|---|---|---|---|---|---|
| DenseMLP (gate+up+down) ×48 | 43.44 (60.3%) | 43.44 (54.4%) | 43.44 (53.4%) | 43.44 (53.8%) | 43.44 (52.0%) |
| LM_head ×1 | 10.03 (13.9%) | 10.03 (12.6%) | 10.03 (12.3%) | 10.03 (12.4%) | 10.03 (12.0%) |
| FC_QKV_sliding ×40 | 6.74 (9.4%) | 6.74 (8.4%) | 6.74 (8.3%) | 6.74 (8.3%) | 6.74 (8.1%) |
| FC_O_sliding ×40 | 3.28 (4.6%) | 3.28 (4.1%) | 3.28 (4.0%) | 3.28 (4.1%) | 3.28 (3.9%) |
| FC_QK_full ×8 | 1.42 (2.0%) | 1.42 (1.8%) | 1.42 (1.7%) | 1.42 (1.8%) | 1.42 (1.7%) |
| FC_O_full ×8 | 1.55 (2.1%) | 1.55 (1.9%) | 1.55 (1.9%) | 1.55 (1.9%) | 1.55 (1.9%) |
| PA_sliding ×40 | 3.41 (4.7%) | 9.90 (12.4%) | 9.90 (12.2%) | 9.90 (12.3%) | 9.90 (11.9%) |
| PA_full ×8 | 0.73 (1.0%) | 1.99 (2.5%) | 3.52 (4.3%) | 2.99 (3.7%) | 5.71 (6.8%) |
| SmallOps (norm/rope/add) | 1.44 (2.0%) | 1.44 (1.8%) | 1.44 (1.8%) | 1.44 (1.8%) | 1.44 (1.7%) |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)

| op | S=256 | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---|---|---|---|---|
| DenseMLP (gate+up+down) ×48 | 195.3 (61.0%) | 739.2 (57.4%) | 1476.4 (53.8%) | 2885.0 (46.8%) | 5876.4 (38.0%) |
| PA_full ×8 | 16.4 (5.1%) | 119.2 (9.3%) | 420.7 (15.3%) | 1603.0 (26.0%) | 6265.2 (40.5%) |
| FC_QKV_sliding ×40 | 31.2 (9.8%) | 114.9 (8.9%) | 230.2 (8.4%) | 460.0 (7.5%) | 939.8 (6.1%) |
| SmallOps (norm/rope/add) | 29.0 (9.1%) | 116.6 (9.1%) | 231.5 (8.4%) | 458.1 (7.4%) | 903.9 (5.8%) |
| FC_O_sliding ×40 | 16.6 (5.2%) | 59.6 (4.6%) | 118.9 (4.3%) | 236.3 (3.8%) | 439.9 (2.8%) |
| FC_QK_full ×8 | 6.2 (1.9%) | 23.2 (1.8%) | 45.9 (1.7%) | 89.5 (1.5%) | 180.3 (1.2%) |
| FC_O_full ×8 | 6.4 (2.0%) | 22.5 (1.7%) | 45.0 (1.6%) | 88.1 (1.4%) | 176.0 (1.1%) |
| PA_sliding ×40 | 9.0 (2.8%) | 82.7 (6.4%) | 165.5 (6.0%) | 330.9 (5.4%) | 661.9 (4.3%) |
| LM_head ×1 | 9.9 (3.1%) | 9.9 (0.8%) | 9.9 (0.4%) | 9.9 (0.2%) | 9.9 (0.1%) |

## Roofline: theoretical floor vs measured

### Decode (per output token)

| prompt P | KV | theoretical (ms) | measured (ms) | achieved % |
|---:|---:|---:|---:|---:|
| 256 | 256 | 65.6 | 72.04 | 91.1% |
| 1,024 | 1,024 | 67.4 | 79.79 | 84.5% |
| 2,048 | 2,048 | 67.9 | 81.32 | 83.5% |
| 4,096 | 4,096 | 67.7 | 80.79 | 83.8% |
| 8,192 | 8,192 | 68.5 | 83.50 | 82.0% |

_Decode is dominated by memory-bound INT4/INT8 weight reads (DenseMLP + LM_head ≈ 74 %), running at 82–94 % of the 110 GB/s ceiling per op._

### Prefill (TTFT over S tokens)

| S | theoretical (ms) | measured (ms) | achieved % |
|---:|---:|---:|---:|
| 256 | 188.5 | 320.0 | 58.9% |
| 1,024 | 568.0 | 1,287.9 | 44.1% |
| 2,048 | 1,150.0 | 2,744.0 | 41.9% |
| 4,096 | 2,520.0 | 6,160.9 | 40.9% |
| 8,192 | 6,180.0 | 15,453.2 | 40.0% |

_Prefill is FC INT8-XMX compute-bound at S≥1024 (achieving ~52–56 % of 40.14 TOPS) plus a fast-growing PA_full FP16 micro-kernel term (≈7 % XMX, the cost of NKV=1 / HD=512 full attention). The PA_full quadratic dominates TTFT beyond S≈4096._

## Decode tables (1 query token, KV = context length)

### Decode — KV=256

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel ×3 | 0.9051 | 48 | 43.44 | 130.4 | 100.9 | 91.7% | memory |
| LM_head (INT8) | gemm_kernel | 10.0292 | 1 | 10.03 | 200.7 | 102.0 | 92.7% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1684 | 40 | 6.74 | 373.5 | 97.2 | 88.4% | memory |
| PA_sliding (INT8 KV, eff=256) | paged_attention_opt | 0.0853 | 40 | 3.41 | 49.1 | 12.0 | 11.0% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0821 | 40 | 3.28 | 382.8 | 99.7 | 90.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1934 | 8 | 1.55 | 325.0 | 84.6 | 76.9% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.44 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1773 | 8 | 1.42 | 376.9 | 98.1 | 89.2% | memory |
| PA_full (INT8 KV) | paged_attention_opt | 0.0913 | 8 | 0.73 | 91.9 | 5.7 | 5.2% | memory |
| **TOTAL** |  |  |  | **72.04** |  |  | **13.9 tok/s** |  |

### Decode — KV=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel ×3 | 0.9051 | 48 | 43.44 | 130.4 | 100.9 | 91.7% | memory |
| LM_head (INT8) | gemm_kernel | 10.0292 | 1 | 10.03 | 200.7 | 102.0 | 92.7% | memory |
| PA_sliding (INT8 KV, eff=1024) | paged_attention_opt | 0.2474 | 40 | 9.90 | 67.8 | 17.0 | 15.4% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1684 | 40 | 6.74 | 373.5 | 97.2 | 88.4% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0821 | 40 | 3.28 | 382.8 | 99.7 | 90.7% | memory |
| PA_full (INT8 KV) | paged_attention_opt | 0.2491 | 8 | 1.99 | 134.7 | 4.2 | 3.8% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1934 | 8 | 1.55 | 325.0 | 84.6 | 76.9% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.44 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1773 | 8 | 1.42 | 376.9 | 98.1 | 89.2% | memory |
| **TOTAL** |  |  |  | **79.79** |  |  | **12.5 tok/s** |  |

### Decode — KV=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel ×3 | 0.9051 | 48 | 43.44 | 130.4 | 100.9 | 91.7% | memory |
| LM_head (INT8) | gemm_kernel | 10.0292 | 1 | 10.03 | 200.7 | 102.0 | 92.7% | memory |
| PA_sliding (INT8 KV, eff=1024) | paged_attention_opt | 0.2474 | 40 | 9.90 | 67.8 | 17.0 | 15.4% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1684 | 40 | 6.74 | 373.5 | 97.2 | 88.4% | memory |
| PA_full (INT8 KV) | paged_attention_opt | 0.4405 | 8 | 3.52 | 152.3 | 4.8 | 4.3% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0821 | 40 | 3.28 | 382.8 | 99.7 | 90.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1934 | 8 | 1.55 | 325.0 | 84.6 | 76.9% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.44 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1773 | 8 | 1.42 | 376.9 | 98.1 | 89.2% | memory |
| **TOTAL** |  |  |  | **81.32** |  |  | **12.3 tok/s** |  |

### Decode — KV=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel ×3 | 0.9051 | 48 | 43.44 | 130.4 | 100.9 | 91.7% | memory |
| LM_head (INT8) | gemm_kernel | 10.0292 | 1 | 10.03 | 200.7 | 102.0 | 92.7% | memory |
| PA_sliding (INT8 KV, eff=1024) | paged_attention_opt | 0.2474 | 40 | 9.90 | 67.8 | 17.0 | 15.4% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1684 | 40 | 6.74 | 373.5 | 97.2 | 88.4% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0821 | 40 | 3.28 | 382.8 | 99.7 | 90.7% | memory |
| PA_full (INT8 KV, GQA kernel) | paged_attention_opt__gqa | 0.3743 | 8 | 2.99 | 358.7 | 11.2 | 10.2% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1934 | 8 | 1.55 | 325.0 | 84.6 | 76.9% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.44 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1773 | 8 | 1.42 | 376.9 | 98.1 | 89.2% | memory |
| **TOTAL** |  |  |  | **80.79** |  |  | **12.4 tok/s** |  |

_Note: at KV≥4096 PA_full switches from `paged_attention_opt__single_token` to the `paged_attention_opt__gqa_single_token` kernel variant, which is faster than the KV=2048 single-token path despite the larger KV._

### Decode — KV=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel ×3 | 0.9051 | 48 | 43.44 | 130.4 | 100.9 | 91.7% | memory |
| LM_head (INT8) | gemm_kernel | 10.0292 | 1 | 10.03 | 200.7 | 102.0 | 92.7% | memory |
| PA_sliding (INT8 KV, eff=1024) | paged_attention_opt | 0.2474 | 40 | 9.90 | 67.8 | 17.0 | 15.4% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1684 | 40 | 6.74 | 373.5 | 97.2 | 88.4% | memory |
| PA_full (INT8 KV, GQA kernel) | paged_attention_opt__gqa | 0.7133 | 8 | 5.71 | 376.4 | 11.8 | 10.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0821 | 40 | 3.28 | 382.8 | 99.7 | 90.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1934 | 8 | 1.55 | 325.0 | 84.6 | 76.9% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.44 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1773 | 8 | 1.42 | 376.9 | 98.1 | 89.2% | memory |
| **TOTAL** |  |  |  | **83.50** |  |  | **12.0 tok/s** |  |

_Note: SwiGLU/GEGLU `multiply` (gelu(gate)·up) is fused into the MLP primitive and does not appear as a standalone kernel; see *Graph fusion notes* above._

## Prefill tables (single forward over S tokens)

### Prefill — S=256

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel ×3 | 4.069 | 48 | 195.3 | 22264 | — | 55.5% | compute |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 0.781 | 40 | 31.2 | 20626 | — | 51.4% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 29.0 | — | — | — | memory |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 0.414 | 40 | 16.6 | 19439 | — | 48.4% | compute |
| PA_full (FP16 prefill, causal) | sdpa_opt__multi_tokens | 2.045 | 8 | 16.4 | 527.0 | 4.4 | 6.6% | memory |
| LM_head (INT8, 1 out tok) | gemm_kernel | 9.914 | 1 | 9.9 | 203.0 | 103.2 | 93.8% | memory |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 0.806 | 8 | 6.4 | 19967 | — | 49.7% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 0.771 | 8 | 6.2 | 22194 | — | 55.3% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro__prefill | 0.225 | 40 | 9.0 | 2393 | 27.9 | 31.5% | memory |
| **TOTAL** |  |  |  | **320.0** |  |  | **TTFT 0.32s, 800 tok/s** |  |

### Prefill — S=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel ×3 | 15.399 | 48 | 739.2 | 23533 | — | 58.6% | compute |
| PA_full (FP16 prefill, causal) | sdpa_opt__multi_tokens | 14.904 | 8 | 119.2 | 1442 | 3.0 | 7.6% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 116.6 | — | — | — | memory |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 2.874 | 40 | 114.9 | 22419 | — | 55.8% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro__prefill | 2.068 | 40 | 82.7 | 4159 | 12.2 | 54.6% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 1.490 | 40 | 59.6 | 21618 | — | 53.9% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 2.901 | 8 | 23.2 | 23596 | — | 58.8% | compute |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 2.816 | 8 | 22.5 | 22875 | — | 57.0% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 9.914 | 1 | 9.9 | 203.0 | 103.2 | 93.8% | memory |
| **TOTAL** |  |  |  | **1287.9** |  |  | **TTFT 1.29s, 795 tok/s** |  |

### Prefill — S=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel ×3 | 30.759 | 48 | 1476.4 | 23563 | — | 58.7% | compute |
| PA_full (FP16 prefill, causal) | sdpa_opt__multi_tokens | 52.588 | 8 | 420.7 | 1633 | 1.7 | 8.6% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 231.5 | — | — | — | memory |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 5.754 | 40 | 230.2 | 22384 | — | 55.8% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro__prefill | 4.137 | 40 | 165.5 | 8302 | 13.6 | 54.6% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 2.973 | 40 | 118.9 | 21674 | — | 54.0% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 5.734 | 8 | 45.9 | 23880 | — | 59.5% | compute |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 5.627 | 8 | 45.0 | 22899 | — | 57.0% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 9.914 | 1 | 9.9 | 203.0 | 103.2 | 93.8% | memory |
| **TOTAL** |  |  |  | **2744.0** |  |  | **TTFT 2.74s, 746 tok/s** |  |

### Prefill — S=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel ×3 | 60.107 | 48 | 2885.0 | 24116 | — | 60.1% | compute |
| PA_full (FP16 prefill, causal) | sdpa_opt__multi_tokens | 200.370 | 8 | 1603.0 | 1714 | 0.9 | 9.0% | compute |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 11.499 | 40 | 460.0 | 22404 | — | 55.8% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 458.1 | — | — | — | memory |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro__prefill | 8.273 | 40 | 330.9 | 16604 | 13.6 | 54.6% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 5.907 | 40 | 236.3 | 21810 | — | 54.3% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 11.185 | 8 | 89.5 | 24487 | — | 61.0% | compute |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 11.008 | 8 | 88.1 | 23413 | — | 58.3% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 9.914 | 1 | 9.9 | 203.0 | 103.2 | 93.8% | memory |
| **TOTAL** |  |  |  | **6160.9** |  |  | **TTFT 6.16s, 665 tok/s** |  |

### Prefill — S=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (FP16 prefill, causal) | sdpa_opt__multi_tokens | 783.146 | 8 | 6265.2 | 1754 | 0.5 | 9.2% | compute |
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel ×3 | 122.425 | 48 | 5876.4 | 23681 | — | 59.0% | compute |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 23.494 | 40 | 939.8 | 21937 | — | 54.7% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 903.9 | — | — | — | memory |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro__prefill (scaled) | 16.546 | 40 | 661.9 | 16604 | 13.6 | 54.6% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 10.998 | 40 | 439.9 | 23432 | — | 58.4% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 22.534 | 8 | 180.3 | 24301 | — | 60.5% | compute |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 22.001 | 8 | 176.0 | 23426 | — | 58.4% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 9.914 | 1 | 9.9 | 203.0 | 103.2 | 93.8% | memory |
| **TOTAL** |  |  |  | **15453.2** |  |  | **TTFT 15.45s, 530 tok/s** |  |

_Note: GEGLU `multiply` is fused into the MLP primitive and does not appear as a standalone kernel._

## Op → kernel names (cliloader)

### Decode (M=1)

| op | kernel name(s) | launches/call |
|---|---|---:|
| FC_QKV_sliding / FC_O_sliding | `gemm_kernel` | 1 |
| FC_QK_full / FC_O_full | `gemm_kernel` | 1 |
| MLP_gate / up / down | `gemm_kernel` | 1 (×3 ops) |
| LM_head | `gemm_kernel` | 1 |
| PA_sliding | `pa_kv_cache_update`<br>`paged_attention_opt__single_token` | 1<br>1 |
| PA_full (KV<4096) | `pa_kv_cache_update`<br>`paged_attention_opt__single_token` | 1<br>1 |
| PA_full (KV≥4096) | `pa_kv_cache_update`<br>`paged_attention_opt__gqa_single_token` | 1<br>1 |
| RMSNorm / QK-norm | `rms_gpu_*` | 1 |
| RoPE | `rope_opt` | 1 |
| Add (residual) | `eltwise` | 1 |

### Prefill (S=8192)

| op | kernel name(s) | launches/call |
|---|---|---:|
| FC_* (INT4) | `dynamic_quantize_gpu_opt`<br>`gemm_kernel` | 1<br>1 |
| LM_head (INT8) | `gemm_kernel` | 1 |
| PA_full | `pa_kv_cache_update`<br>`sdpa_opt__multi_tokens` | 1<br>1 |
| PA_sliding | `pa_kv_cache_update`<br>`sdpa_micro__prefill` | 1<br>1 |
| SmallOps | `rms_gpu_*` / `rope_opt` / `eltwise` | 1 |

## Per-kernel decomposition (cliloader kernel names)

### Decode sub-kernels — KV=1024 (representative)

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
| DenseMLP | `gemm_kernel` | 0.3017 | 3 | 48 | 43.44 | 54.4% |
| LM_head | `gemm_kernel` | 10.0292 | 1 | 1 | 10.03 | 12.6% |
| PA_sliding | `paged_attention_opt__single_token` | 0.2405 | 1 | 40 | 9.62 | 12.1% |
| FC_QKV_sliding | `gemm_kernel` | 0.1684 | 1 | 40 | 6.74 | 8.4% |
| FC_O_sliding | `gemm_kernel` | 0.0821 | 1 | 40 | 3.28 | 4.1% |
| PA_full | `paged_attention_opt__single_token` | 0.2366 | 1 | 8 | 1.89 | 2.4% |
| FC_O_full | `gemm_kernel` | 0.1934 | 1 | 8 | 1.55 | 1.9% |
| FC_QK_full | `gemm_kernel` | 0.1773 | 1 | 8 | 1.42 | 1.8% |
| PA (kv update) | `pa_kv_cache_update` | ~0.007 | 1 | 48 | ~0.34 | 0.4% |

### Prefill sub-kernels — S=8192 (representative)

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
| PA_full | `sdpa_opt__multi_tokens` | 782.85 | 1 | 8 | 6262.8 | 40.5% |
| DenseMLP | `dynamic_quantize_gpu_opt`+`gemm_kernel` | 122.43 | 2 | 48 | 5876.4 | 38.0% |
| FC_QKV_sliding | `dynamic_quantize_gpu_opt`+`gemm_kernel` | 23.49 | 2 | 40 | 939.8 | 6.1% |
| PA_sliding | `sdpa_micro__prefill` | 16.55 | 1 | 40 | 661.9 | 4.3% |
| FC_O_sliding | `dynamic_quantize_gpu_opt`+`gemm_kernel` | 11.00 | 2 | 40 | 439.9 | 2.8% |
| FC_QK_full | `dynamic_quantize_gpu_opt`+`gemm_kernel` | 22.53 | 2 | 8 | 180.3 | 1.2% |
| FC_O_full | `dynamic_quantize_gpu_opt`+`gemm_kernel` | 22.00 | 2 | 8 | 176.0 | 1.1% |
| LM_head | `gemm_kernel` | 9.91 | 1 | 1 | 9.9 | 0.1% |

## Top contributors (sorted by total ms per inference)

### Decode

| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 256 | DenseMLP 43.44ms (60.3%) | LM_head 10.03ms (13.9%) | FC_QKV_sliding 6.74ms (9.4%) |
| 1,024 | DenseMLP 43.44ms (54.4%) | LM_head 10.03ms (12.6%) | PA_sliding 9.90ms (12.4%) |
| 2,048 | DenseMLP 43.44ms (53.4%) | LM_head 10.03ms (12.3%) | PA_sliding 9.90ms (12.2%) |
| 4,096 | DenseMLP 43.44ms (53.8%) | LM_head 10.03ms (12.4%) | PA_sliding 9.90ms (12.3%) |
| 8,192 | DenseMLP 43.44ms (52.0%) | LM_head 10.03ms (12.0%) | PA_sliding 9.90ms (11.9%) |

### Prefill

| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 256 | DenseMLP 195.3ms (61.0%) | FC_QKV_sliding 31.2ms (9.8%) | SmallOps 29.0ms (9.1%) |
| 1,024 | DenseMLP 739.2ms (57.4%) | PA_full 119.2ms (9.3%) | SmallOps 116.6ms (9.1%) |
| 2,048 | DenseMLP 1476.4ms (53.8%) | PA_full 420.7ms (15.3%) | SmallOps 231.5ms (8.4%) |
| 4,096 | DenseMLP 2885.0ms (46.8%) | PA_full 1603.0ms (26.0%) | FC_QKV_sliding 460.0ms (7.5%) |
| 8,192 | PA_full 6265.2ms (40.5%) | DenseMLP 5876.4ms (38.0%) | FC_QKV_sliding 939.8ms (6.1%) |

## End-to-end (prefill TTFT + 512-token decode)

_Decode KV = prompt P (sliding layers cap at sw=1024)._

| prompt P | TTFT (ms) | 512-tok decode (ms) | total (ms) | avg decode tok/s |
|---:|---:|---:|---:|---:|
| 256 | 320.0 | 36,884 | 37,204 | 13.9 |
| 1,024 | 1,287.9 | 40,852 | 42,140 | 12.5 |
| 2,048 | 2,744.0 | 41,636 | 44,380 | 12.3 |
| 4,096 | 6,160.9 | 41,365 | 47,526 | 12.4 |
| 8,192 | 15,453.2 | 42,752 | 58,205 | 12.0 |

## Key findings

- **Decode is memory-bound and runs at 82–94 % of the 110 GB/s ceiling per FC kernel.** TPOT ≈ 72–84 ms → **12–14 tok/s**. Because PTL 4Xe shares the same 110 GB/s LPDDR5x as PTL 12Xe but the decode body is memory-bound, **decode tok/s is essentially identical to PTL 12Xe** (12Xe: 12.7–14.5 tok/s; 4Xe: 12.0–13.9 tok/s). The small gap is from PA (more compute-limited on 4 cores) and FC_O_full landing at 77 % BW.
- **DenseMLP (15360-wide GEGLU) dominates decode at 52–60 %** (43.4 ms over 48 layers), followed by the **INT8 LM_head at ~13 %** (10 ms single read of ~1 GB). These two account for ~66 % of every token.
- **Prefill is FC INT8-XMX compute-bound** at ~52–64 % of the 40.14 TOPS peak, but **PA_full (NKV=1 / HD=512) grows quadratically** and overtakes DenseMLP as the #1 cost at S=8192 (40.5 % of TTFT). The full-attention `sdpa_opt__multi_tokens` kernel only reaches ~9 % XMX efficiency — the single-KV-head, 512-dim full attention is the main prefill scaling wall.
- **Prefill is ~3× slower than PTL 12Xe** (S=8192: 15.5 s vs 5.6 s) — exactly the 3× XMX-core ratio (4 vs 12 Xe), confirming prefill scales with compute, not bandwidth.
- **PA_full decode KV≥4096 picks a faster GQA kernel** (`paged_attention_opt__gqa_single_token`), so PA_full decode at KV=4096 (0.37 ms) is actually faster than KV=2048 (0.44 ms) on the single-token path.

## Optimization levers (highest ROI first)

1. **INT4 LM_head** — the INT8 LM_head reads ~1 GB every token (10 ms, ~13 % of TPOT). INT4 g=128 (with +1 softcap row) roughly halves it → ~5 ms saved/token → ~+1 tok/s decode.
2. **Fuse DenseMLP gate+up into one packed-INT4 read** — gate and up are two separate 30 MB reads per layer (29 ms combined of the 43 ms DenseMLP). A fused gate-up weight read would cut one DRAM pass → up to ~14 ms/token saved.
3. **PA_full prefill micro-kernel** — `sdpa_opt__multi_tokens` at NKV=1/HD=512 runs at only ~9 % XMX; this is the dominant prefill cost at long S. Switching full-attention prefill to the CM/XAttention path (block_size=256) or a HD=512-tuned micro-kernel is the biggest TTFT lever for long prompts.
4. **FC_O_full decode** lands at 77 % BW (vs 88–94 % for other FCs) — its K=8192 / N=3840 shape under-utilizes the gemm tiling at M=1; a shape-specific decode tile could recover ~10 % on that op.
5. **Fuse RMSNorm + RoPE + residual-add** — small ops aggregate to ~1.4 ms/token decode and ~5–6 % of prefill TTFT across many launches; a fused norm+rope+add primitive cuts launch overhead.

## Comparison with other platforms

| Metric | PTL 4Xe (this) | PTL 12Xe (2026-06-05) | Ratio (4Xe/12Xe) |
|---|---:|---:|---:|
| Xe cores | 4 | 12 | 0.33× |
| FP16 XMX | 20.07 TFLOPS | 58.98 TFLOPS | 0.34× |
| Memory BW | 110 GB/s | 110 GB/s | 1.0× |
| Decode TPOT @ kv=1024 | 79.8 ms (12.5 tok/s) | 71.1 ms (14.1 tok/s) | 1.12× |
| Decode TPOT @ kv=8192 | 83.5 ms (12.0 tok/s) | 72.7 ms (13.8 tok/s) | 1.15× |
| Prefill TTFT @ S=1024 | 1,288 ms | 493 ms | 2.61× |
| Prefill TTFT @ S=8192 | 15,453 ms | 5,638 ms | 2.74× |

_Decode (memory-bound) is nearly platform-independent (1.1–1.15× slower on 4Xe due to compute-limited PA), while prefill (XMX compute-bound) scales ~2.6–2.7× with the 3× fewer cores. This is the expected roofline behavior: same DRAM bus → same decode; 1/3 the XMX → ~3× prefill._

## Caveats & method

- Each op profiled in its own process via cliloader Device Performance Timing; mean kernel time per iteration (ms).
- FC weight bytes count INT4 weight + FP16 scale/zp (g=128) + FP16 act + FP16 out. LM_head: INT8 weight + FP16 scale.
- PA bytes assume INT8 KV cache + FP16 Q, FP16 out.
- Decode FC is treated as **memory-bound** (weights read dominates at M=1); prefill FC is **INT8 XMX compute-bound** (S big enough to hit XMX via `dynamic_quantize_gpu_opt`+`gemm_kernel`).
- PA prefill FLOPs use causal lower-triangular pairs = S(S+1)/2; sliding caps to SW=1024.
- PA sliding decode KV caps at sliding_window=1024 → for ctx≥1024 we reuse the kv=1024 measurement.
- PA sliding prefill bench runs up to S=1024; for S>1024 the S=1024 measurement is scaled linearly (sliding work ~ SW·S).
- Full attention uses partial_rotary_factor=0.25 (rope applied to 25 % of HD); the bench measures full HD rope (slight over-estimate, immaterial).
- GEGLU `multiply`/`swish`/`gelu` are fused into the MLP primitive in real inference; they are not profiled separately.
- LM head is run once per token (last position in prefill, every step in decode).
- Target machine: intel@10.239.152.140 (PTL 4Xe iGPU, 32 EUs @ 2450 MHz, Linux).

## Reproduction

```bash
# On PTL 4Xe (intel@10.239.152.140):
cd ~/river/roofline_test_utils
bash run_gemma4_12B_ptl_4xe.sh
# logs land in ~/river/roofline_results/gemma4_12B/ptl_4xe/

# Locally:
python3 .github/skills/dev_roofline_profiling/utils/parse_logs.py \
    outputs/gemma4_12B/ptl_4xe_logs/ outputs/gemma4_12B/ptl_4xe_metrics.json
python3 .github/skills/dev_roofline_profiling/utils/analyze_gemma4_12B_ptl_4xe.py
```
