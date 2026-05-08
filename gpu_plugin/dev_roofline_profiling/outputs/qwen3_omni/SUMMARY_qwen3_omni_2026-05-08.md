# qwen3_omni Thinker text — Roofline on PTL 12Xe Windows (2026-05-08)

**Platform**: Intel PTL 12Xe iGPU (Arc B390 96CUs class, 96 EUs = 12 Xe × 8 EU × 10 thr), 2400 MHz, BW≈110 GB/s, FP16 XMX peak ≈ 58.98 TFLOPS, INT8 XMX peak ≈ 117.96 TOPS.
**Model**: qwen3_omni Thinker text decoder (dense, GQA). Source: `config.json` -> `thinker_config.text_config`.

- hidden=2560, layers=36, **GQA NH=32 / NKV=8**, head_dim=128 (Q-dim=4096, KV-dim=1024), intermediate=9728, vocab=151936
- `tie_word_embeddings=true` (LM head shares storage with embedding), `hidden_act=silu` (SwiGLU), `rope_theta=1e6`, mrope_section=[24,20,20]
- MatMul weights INT4 g128 / FP16 act; LM_head INT8 g128 / FP16 act; KV cache INT8
- SDPA: PagedAttention OpenCL + micro_kernel

## Model parameters & weight shapes
Architecture knobs (parsed from `thinker_config.text_config`):

| Field | Value | Notes |
|---|---:|---|
| `hidden_size` | 2560 | residual / activation channel |
| `num_hidden_layers` | 36 | decoder blocks |
| `num_attention_heads` (NH) | 32 | Q heads |
| `num_key_value_heads` (NKV) | 8 | GQA: 4-way Q-per-KV sharing |
| `head_dim` (HD) | 128 | Q_dim = NH·HD = 4096, KV_dim = NKV·HD = 1024 |
| `intermediate_size` | 9728 | SwiGLU MLP hidden |
| `vocab_size` | 151936 | LM head N |
| `hidden_act` | silu | SwiGLU = silu(gate(x)) ⊙ up(x) |
| `tie_word_embeddings` | true | LM head storage shared with token embedding |
| `rope_theta` | 1e6 | mrope_section=[24,20,20] |

Per-layer weight matrices (one decoder block) and global weights. INT4 g128: weight = K·N/2 bytes + (K/128)·N FP16 scales. INT8 g128: weight = K·N bytes + (K/128)·N FP16 scales.

| Weight | Shape (K × N) | Quant | Bytes / instance | × Layers | Total MB |
|---|---:|---|---:|---:|---:|
| Embedding (shared w/ LM head) | 151936 × 2560 | INT8 g128 + FP16 scales | 395.03 MB | 1 | 395.0 MB |
| FC_QKV  (fused Q+K+V proj) | 2560 × 6144 | INT4 g128 + FP16 scales | 8.11 MB | 36 | 292.0 MB |
| FC_O    (attention output) | 4096 × 2560 | INT4 g128 + FP16 scales | 5.41 MB | 36 | 194.6 MB |
| FC_Gate (SwiGLU gate) | 2560 × 9728 | INT4 g128 + FP16 scales | 12.84 MB | 36 | 462.3 MB |
| FC_Up   (SwiGLU up) | 2560 × 9728 | INT4 g128 + FP16 scales | 12.84 MB | 36 | 462.3 MB |
| FC_Down (SwiGLU down) | 9728 × 2560 | INT4 g128 + FP16 scales | 12.84 MB | 36 | 462.3 MB |
| LM_Head (tied w/ embedding) | 2560 × 151936 | INT8 g128 + FP16 scales | 395.03 MB | 1 | 395.0 MB |
| **Total static weights** |  |  |  |  | **2663.5 MB** |

Activation / KV-cache shapes (S = sequence length, B = batch=1):

| Tensor | Shape | dtype | Bytes / token / layer | Bytes / token (all layers) |
|---|---|---|---:|---:|
| Hidden states | [B, S, 2560] | FP16 | 5120 | — |
| Q | [B, S, 32, 128] | FP16 | 8192 | — |
| K (cache) | [num_blocks, 8, 128, 16] | INT8 | 1024 | 36864 |
| V (cache) | [num_blocks, 8, 16, 128] | INT8 | 1024 | 36864 |
| **KV cache total** | per token | INT8 | 2048 B / layer | **72.0 KB / token** |

## Theoretical roofline
| Metric | Value |
|---|---|
| FP16 XMX peak | 58.98 TFLOPS |
| INT8 XMX peak | 117.96 TOPS |
| Memory BW | 110 GB/s |
| Ridge point (FP16) | 536 FLOP/byte |

## Graph fusion notes (what the bench rows actually mean in the compiled model)
Some "small ops" in the per-op tables are profiled in isolation by `small_ops_bench` but **do not correspond to standalone GPU kernels** in the real compiled qwen3_omni graph because the GPU plugin fuses them at compile time:

| Bench row | Real graph behaviour | Fused into | Standalone kernel in graph? |
|---|---|---|---|
| `multiply` | `silu(gate(x)) ⊙ up(x)` of SwiGLU MLP | **SwiGLU primitive** (`glu_fusion.cpp` + `swiglu_with_clamp` op) | **No** — bench-only. Time is absorbed into the gated-MLP path. |
| `add` | Two residual adds per layer (post-attention, post-MLP) | not fused (separate `eltwise`) | Yes (2·LAYERS = 72 calls confirms this) |
| `rmsnorm` | Pre-attention + pre-MLP + final RMSNorm | already a single `RMS` primitive (`rms_fusion.cpp`) | Yes — 2·LAYERS+1 = 73 calls, but **cannot be merged across layers** (different tensors / timesteps) |
| `rmsnorm3d_q` / `rmsnorm3d_k` | Per-head q_norm / k_norm after QKV split | not fused with each other (different shapes — NH vs NKV — and different tensors) | Yes — 1 each per layer |
| `rope_q` / `rope_k` | RoPE on Q (NH=32) and K (NKV=8) | sin/cos/concat/mul/add already fused to single `RoPE` primitive (`fuse_rotary_positional_embeddings.cpp`) | Yes — **two distinct `rope_opt` calls per layer** because Q-RoPE and K-RoPE are two graph nodes with different head counts; not merged |

**Implication for the totals**: when computing TPOT / TTFT, the `multiply` row should be treated as 0 in the real graph (it is part of SwiGLU). All other small ops above remain as separate kernel launches. Fusion candidates that would help most: (a) merging q_norm + k_norm into a single fused-RMS-on-QKV kernel (~2 launches/layer saved), (b) batching rope_q + rope_k into one RoPE kernel over the full QKV head dim (would also halve launch overhead).

## Token latency summary
Aggregated per-token latency derived from per-op profiling. **Prefill total = sum of per-layer ops × 32 layers + lm_head (1 token)**, **TTFT** is the same as prefill total (time to first token), **per-token-in-prefill** = TTFT / S (amortized prefill cost per input token). **Decode total = TPOT** (Time-Per-Output-Token, second token at given KV).

### Prefill — TTFT and per-token amortized
| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|
| 1024 | 189.78 | 0.190 | 0.1853 | 5395.7 |
| 2048 | 410.01 | 0.410 | 0.2002 | 4995.0 |
| 4096 | 1001.31 | 1.001 | 0.2445 | 4090.6 |
| 8192 | 2588.45 | 2.588 | 0.3160 | 3164.8 |

### Decode — TPOT (per output token)
| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 1024 | 25.543 | 39.1 |
| 2048 | 26.775 | 37.3 |
| 4096 | 27.585 | 36.3 |
| 8192 | 31.261 | 32.0 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)
| op | KV=1024 ms (%) | KV=2048 ms (%) | KV=4096 ms (%) | KV=8192 ms (%) |
|---|---:|---:|---:|---:|
| FC down | 4.618 (18.1%) | 4.618 (17.2%) | 4.618 (16.7%) | 4.618 (14.8%) |
| FC up | 4.640 (18.2%) | 4.640 (17.3%) | 4.640 (16.8%) | 4.640 (14.8%) |
| FC gate | 4.623 (18.1%) | 4.623 (17.3%) | 4.623 (16.8%) | 4.623 (14.8%) |
| FC qkv (fused Q+K+V) | 2.957 (11.6%) | 2.957 (11.0%) | 2.957 (10.7%) | 2.957 (9.5%) |
| FC o | 2.008 (7.9%) | 2.008 (7.5%) | 2.008 (7.3%) | 2.008 (6.4%) |
| lm_head | 3.834 (15.0%) | 3.834 (14.3%) | 3.834 (13.9%) | 3.834 (12.3%) |
| PagedAttention | 2.190 (8.6%) | 3.422 (12.8%) | 4.232 (15.3%) | 7.908 (25.3%) |
| RMSNorm | 0.213 (0.8%) | 0.213 (0.8%) | 0.213 (0.8%) | 0.213 (0.7%) |
| RMSNorm q | 0.087 (0.3%) | 0.087 (0.3%) | 0.087 (0.3%) | 0.087 (0.3%) |
| RMSNorm k | 0.086 (0.3%) | 0.086 (0.3%) | 0.086 (0.3%) | 0.086 (0.3%) |
| RoPE q | 0.078 (0.3%) | 0.078 (0.3%) | 0.078 (0.3%) | 0.078 (0.2%) |
| RoPE k | 0.071 (0.3%) | 0.071 (0.3%) | 0.071 (0.3%) | 0.071 (0.2%) |
| Residual Add | 0.140 (0.5%) | 0.140 (0.5%) | 0.140 (0.5%) | 0.140 (0.4%) |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)
| op | S=1024 ms (%) | S=2048 ms (%) | S=4096 ms (%) | S=8192 ms (%) |
|---|---:|---:|---:|---:|
| FC down | 35.68 (18.8%) | 69.94 (17.1%) | 135.69 (13.6%) | 271.93 (10.5%) |
| FC up | 31.54 (16.6%) | 62.82 (15.3%) | 132.41 (13.2%) | 247.83 (9.6%) |
| FC gate | 31.44 (16.6%) | 62.34 (15.2%) | 130.16 (13.0%) | 246.82 (9.5%) |
| FC qkv (fused Q+K+V) | 20.73 (10.9%) | 39.38 (9.6%) | 81.80 (8.2%) | 165.00 (6.4%) |
| FC o | 15.10 (8.0%) | 27.84 (6.8%) | 53.76 (5.4%) | 106.52 (4.1%) |
| lm_head | 3.85 (2.0%) | 3.85 (0.9%) | 3.85 (0.4%) | 3.85 (0.1%) |
| PagedAttention | 24.86 (13.1%) | 86.11 (21.0%) | 335.92 (33.5%) | 1270.76 (49.1%) |
| RMSNorm | 5.57 (2.9%) | 11.51 (2.8%) | 26.65 (2.7%) | 57.75 (2.2%) |
| RMSNorm q | 6.60 (3.5%) | 13.54 (3.3%) | 27.72 (2.8%) | 59.16 (2.3%) |
| RMSNorm k | 1.74 (0.9%) | 3.38 (0.8%) | 6.75 (0.7%) | 13.83 (0.5%) |
| RoPE q | 3.84 (2.0%) | 9.40 (2.3%) | 22.23 (2.2%) | 47.93 (1.9%) |
| RoPE k | 1.13 (0.6%) | 2.11 (0.5%) | 4.26 (0.4%) | 10.88 (0.4%) |
| Residual Add | 7.70 (4.1%) | 17.80 (4.3%) | 40.10 (4.0%) | 86.20 (3.3%) |

## Decode tables (1 query token, KV = context length)
### Decode — KV=1024
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_up | fc_int4_g128 | 0.1289 | 36 | 4.640 | 386 | 99.8 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1284 | 36 | 4.623 | 388 | 100.2 | 91.1% | memory |
| fc_down | fc_int4_g128 | 0.1283 | 36 | 4.618 | 388 | 100.3 | 91.2% | memory |
| lm_head | fc_int8_g128 | 3.8339 | 1 | 3.834 | 203 | 103.1 | 93.7% | memory |
| fc_qkv | fc_int4_g128 | 0.0821 | 36 | 2.957 | 383 | 98.9 | 90.0% | memory |
| pa | pa_opencl_micro | 0.0608 | 36 | 2.190 | 289 | 34.7 | 31.6% | memory |
| fc_o | fc_int4_g128 | 0.0558 | 36 | 2.008 | 376 | 97.2 | 88.3% | memory |
| rmsnorm | rmsnorm | 0.0029 | 73 | 0.213 | 0 | 5.3 | 4.8% | memory |
| add | add | 0.0019 | 72 | 0.140 | 0 | 7.9 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0024 | 36 | 0.087 | 0 | 6.9 | 6.3% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0024 | 36 | 0.086 | 0 | 1.8 | 1.7% | memory |
| rope_q | rope_q | 0.0022 | 36 | 0.078 | 0 | 7.8 | 7.1% | memory |
| rope_k | rope_k | 0.0020 | 36 | 0.071 | 0 | 2.3 | 2.1% | memory |
| **TOTAL** |  |  |  | **25.543** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=2048
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_up | fc_int4_g128 | 0.1289 | 36 | 4.640 | 386 | 99.8 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1284 | 36 | 4.623 | 388 | 100.2 | 91.1% | memory |
| fc_down | fc_int4_g128 | 0.1283 | 36 | 4.618 | 388 | 100.3 | 91.2% | memory |
| lm_head | fc_int8_g128 | 3.8339 | 1 | 3.834 | 203 | 103.1 | 93.7% | memory |
| pa | pa_opencl_micro | 0.0951 | 36 | 3.422 | 370 | 44.3 | 40.3% | memory |
| fc_qkv | fc_int4_g128 | 0.0821 | 36 | 2.957 | 383 | 98.9 | 90.0% | memory |
| fc_o | fc_int4_g128 | 0.0558 | 36 | 2.008 | 376 | 97.2 | 88.3% | memory |
| rmsnorm | rmsnorm | 0.0029 | 73 | 0.213 | 0 | 5.3 | 4.8% | memory |
| add | add | 0.0019 | 72 | 0.140 | 0 | 7.9 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0024 | 36 | 0.087 | 0 | 6.9 | 6.3% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0024 | 36 | 0.086 | 0 | 1.8 | 1.7% | memory |
| rope_q | rope_q | 0.0022 | 36 | 0.078 | 0 | 7.8 | 7.1% | memory |
| rope_k | rope_k | 0.0020 | 36 | 0.071 | 0 | 2.3 | 2.1% | memory |
| **TOTAL** |  |  |  | **26.775** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=4096
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_up | fc_int4_g128 | 0.1289 | 36 | 4.640 | 386 | 99.8 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1284 | 36 | 4.623 | 388 | 100.2 | 91.1% | memory |
| fc_down | fc_int4_g128 | 0.1283 | 36 | 4.618 | 388 | 100.3 | 91.2% | memory |
| pa | pa_opencl_micro | 0.1176 | 36 | 4.232 | 599 | 71.5 | 65.0% | memory |
| lm_head | fc_int8_g128 | 3.8339 | 1 | 3.834 | 203 | 103.1 | 93.7% | memory |
| fc_qkv | fc_int4_g128 | 0.0821 | 36 | 2.957 | 383 | 98.9 | 90.0% | memory |
| fc_o | fc_int4_g128 | 0.0558 | 36 | 2.008 | 376 | 97.2 | 88.3% | memory |
| rmsnorm | rmsnorm | 0.0029 | 73 | 0.213 | 0 | 5.3 | 4.8% | memory |
| add | add | 0.0019 | 72 | 0.140 | 0 | 7.9 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0024 | 36 | 0.087 | 0 | 6.9 | 6.3% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0024 | 36 | 0.086 | 0 | 1.8 | 1.7% | memory |
| rope_q | rope_q | 0.0022 | 36 | 0.078 | 0 | 7.8 | 7.1% | memory |
| rope_k | rope_k | 0.0020 | 36 | 0.071 | 0 | 2.3 | 2.1% | memory |
| **TOTAL** |  |  |  | **27.585** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=8192
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro | 0.2197 | 36 | 7.908 | 641 | 76.5 | 69.5% | memory |
| fc_up | fc_int4_g128 | 0.1289 | 36 | 4.640 | 386 | 99.8 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1284 | 36 | 4.623 | 388 | 100.2 | 91.1% | memory |
| fc_down | fc_int4_g128 | 0.1283 | 36 | 4.618 | 388 | 100.3 | 91.2% | memory |
| lm_head | fc_int8_g128 | 3.8339 | 1 | 3.834 | 203 | 103.1 | 93.7% | memory |
| fc_qkv | fc_int4_g128 | 0.0821 | 36 | 2.957 | 383 | 98.9 | 90.0% | memory |
| fc_o | fc_int4_g128 | 0.0558 | 36 | 2.008 | 376 | 97.2 | 88.3% | memory |
| rmsnorm | rmsnorm | 0.0029 | 73 | 0.213 | 0 | 5.3 | 4.8% | memory |
| add | add | 0.0019 | 72 | 0.140 | 0 | 7.9 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0024 | 36 | 0.087 | 0 | 6.9 | 6.3% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0024 | 36 | 0.086 | 0 | 1.8 | 1.7% | memory |
| rope_q | rope_q | 0.0022 | 36 | 0.078 | 0 | 7.8 | 7.1% | memory |
| rope_k | rope_k | 0.0020 | 36 | 0.071 | 0 | 2.3 | 2.1% | memory |
| **TOTAL** |  |  |  | **31.261** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

## Prefill tables (single forward over S tokens)
### Prefill — S=1024
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_down | fc_int4_g128 | 0.9911 | 36 | 35.681 | 51459 | 38.3 | 43.6% | compute |
| fc_up | fc_int4_g128 | 0.8761 | 36 | 31.540 | 58214 | 43.4 | 49.3% | compute |
| fc_gate | fc_int4_g128 | 0.8732 | 36 | 31.435 | 58409 | 43.5 | 49.5% | compute |
| pa | pa_opencl_micro_prefill | 0.6907 | 36 | 24.864 | 26089 | 27.3 | 24.8% | memory |
| fc_qkv | fc_int4_g128 | 0.5758 | 36 | 20.730 | 55941 | 45.0 | 47.4% | compute |
| fc_o | fc_int4_g128 | 0.4194 | 36 | 15.097 | 51208 | 45.4 | 43.4% | compute |
| add | add | 0.1069 | 72 | 7.699 | 0 | 147.1 | 133.7% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.1833 | 36 | 6.598 | 0 | 91.5 | 83.2% | memory |
| rmsnorm | rmsnorm | 0.0764 | 73 | 5.574 | 0 | 137.4 | 124.9% | memory |
| lm_head | fc_int8_g128 | 3.8534 | 1 | 3.853 | 202 | 102.6 | 93.3% | memory |
| rope_q | rope_q | 0.1067 | 36 | 3.840 | 0 | 162.2 | 147.5% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0484 | 36 | 1.741 | 0 | 86.7 | 78.8% | memory |
| rope_k | rope_k | 0.0313 | 36 | 1.127 | 0 | 150.7 | 137.0% | memory |
| **TOTAL** |  |  |  | **189.780** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=2048
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro_prefill | 2.3919 | 36 | 86.110 | 30133 | 15.8 | 51.1% | compute |
| fc_down | fc_int4_g128 | 1.9427 | 36 | 69.938 | 52506 | 32.5 | 44.5% | compute |
| fc_up | fc_int4_g128 | 1.7450 | 36 | 62.819 | 58457 | 36.2 | 49.6% | compute |
| fc_gate | fc_int4_g128 | 1.7317 | 36 | 62.341 | 58905 | 36.5 | 49.9% | compute |
| fc_qkv | fc_int4_g128 | 1.0939 | 36 | 39.382 | 58892 | 40.0 | 49.9% | compute |
| fc_o | fc_int4_g128 | 0.7732 | 36 | 27.836 | 55547 | 42.3 | 47.1% | compute |
| add | add | 0.2472 | 72 | 17.801 | 0 | 127.2 | 115.7% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.3760 | 36 | 13.537 | 0 | 89.2 | 81.1% | memory |
| rmsnorm | rmsnorm | 0.1577 | 73 | 11.509 | 0 | 133.0 | 121.0% | memory |
| rope_q | rope_q | 0.2611 | 36 | 9.401 | 0 | 132.5 | 120.5% | memory |
| lm_head | fc_int8_g128 | 3.8534 | 1 | 3.853 | 202 | 102.6 | 93.3% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0938 | 36 | 3.376 | 0 | 89.5 | 81.3% | memory |
| rope_k | rope_k | 0.0586 | 36 | 2.109 | 0 | 161.1 | 146.4% | memory |
| **TOTAL** |  |  |  | **410.011** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=4096
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro_prefill | 9.3312 | 36 | 335.923 | 30896 | 8.1 | 52.4% | compute |
| fc_down | fc_int4_g128 | 3.7691 | 36 | 135.687 | 54128 | 30.1 | 45.9% | compute |
| fc_up | fc_int4_g128 | 3.6781 | 36 | 132.411 | 55467 | 30.9 | 47.0% | compute |
| fc_gate | fc_int4_g128 | 3.6157 | 36 | 130.164 | 56424 | 31.4 | 47.8% | compute |
| fc_qkv | fc_int4_g128 | 2.2721 | 36 | 81.796 | 56709 | 35.0 | 48.1% | compute |
| fc_o | fc_int4_g128 | 1.4932 | 36 | 53.756 | 57526 | 40.1 | 48.8% | compute |
| add | add | 0.5570 | 72 | 40.103 | 0 | 113.0 | 102.7% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.7701 | 36 | 27.723 | 0 | 87.1 | 79.2% | memory |
| rmsnorm | rmsnorm | 0.3651 | 73 | 26.649 | 0 | 114.9 | 104.5% | memory |
| rope_q | rope_q | 0.6176 | 36 | 22.235 | 0 | 112.0 | 101.9% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.1875 | 36 | 6.752 | 0 | 89.5 | 81.3% | memory |
| rope_k | rope_k | 0.1184 | 36 | 4.263 | 0 | 159.4 | 144.9% | memory |
| lm_head | fc_int8_g128 | 3.8534 | 1 | 3.853 | 202 | 102.6 | 93.3% | memory |
| **TOTAL** |  |  |  | **1001.315** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=8192
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro_prefill | 35.2988 | 36 | 1270.758 | 32670 | 4.3 | 55.4% | compute |
| fc_down | fc_int4_g128 | 7.5537 | 36 | 271.932 | 54016 | 28.4 | 45.8% | compute |
| fc_up | fc_int4_g128 | 6.8841 | 36 | 247.827 | 59270 | 31.1 | 50.2% | compute |
| fc_gate | fc_int4_g128 | 6.8561 | 36 | 246.820 | 59512 | 31.2 | 50.4% | compute |
| fc_qkv | fc_int4_g128 | 4.5833 | 36 | 165.000 | 56225 | 32.9 | 47.7% | compute |
| fc_o | fc_int4_g128 | 2.9588 | 36 | 106.517 | 58064 | 38.7 | 49.2% | compute |
| add | add | 1.1972 | 72 | 86.198 | 0 | 105.1 | 95.5% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 1.6433 | 36 | 59.158 | 0 | 81.7 | 74.3% | memory |
| rmsnorm | rmsnorm | 0.7911 | 73 | 57.751 | 0 | 106.0 | 96.4% | memory |
| rope_q | rope_q | 1.3314 | 36 | 47.930 | 0 | 104.0 | 94.5% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.3842 | 36 | 13.832 | 0 | 87.3 | 79.4% | memory |
| rope_k | rope_k | 0.3021 | 36 | 10.877 | 0 | 124.9 | 113.6% | memory |
| lm_head | fc_int8_g128 | 3.8534 | 1 | 3.853 | 202 | 102.6 | 93.3% | memory |
| **TOTAL** |  |  |  | **2588.454** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

## Per-kernel decomposition (cliloader kernel names)
Each op above maps to one or more GPU kernels. Below shows the actual kernel names captured by cliloader's *Device Performance Timing* section, with per-launch time, launches per op-call, total ms across one inference, and per-kernel **Eff%** (peak utilization). The Eff% column attributes the **parent op's full FLOPs/bytes per op-call** against the kernel's own per-launch time, so the dominant kernel reports an Eff% close to the op-level value, while helper kernels (e.g. `pa_kv_cache_update_ref`, `dynamic_quantize_gpu_opt`, `*_finalization`) appear with apparently very high Eff% — that is the expected signal that they are *not* the bottleneck for the op. PA decomposes into `pa_kv_cache_update_ref` + the attention kernel (`paged_attention_opt__single_token` / `__gqa_single_token` / `sdpa_micro__prefill`) + finalization. Prefill FC decomposes into `dynamic_quantize_gpu_opt` + `gemm_kernel`.

### Decode sub-kernels
### Decode sub-kernels — KV=1024
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.1289 | 1 | 36 | 4.640 | 18.2% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1284 | 1 | 36 | 4.623 | 18.1% | 91.1% |
| fc_down | `gemm_kernel` | 0.1283 | 1 | 36 | 4.618 | 18.1% | 91.2% |
| lm_head | `gemm_kernel` | 3.8339 | 1 | 1 | 3.834 | 15.0% | 93.7% |
| fc_qkv | `gemm_kernel` | 0.0821 | 1 | 36 | 2.957 | 11.6% | 90.0% |
| fc_o | `gemm_kernel` | 0.0558 | 1 | 36 | 2.008 | 7.9% | 88.3% |
| pa | `paged_attention_opt__single_token` | 0.0538 | 1 | 36 | 1.936 | 7.6% | 35.7% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0029 | 1 | 73 | 0.213 | 0.8% | 4.8% |
| pa | `pa_kv_cache_update_ref` | 0.0039 | 1 | 36 | 0.142 | 0.6% | 487.5% |
| add | `eltwise_simple_vload8` | 0.0019 | 1 | 72 | 0.140 | 0.5% | 7.2% |
| pa | `paged_attention_opt__single_token_finalization` | 0.0031 | 1 | 36 | 0.113 | 0.4% | 614.3% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.087 | 0.3% | 6.3% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.086 | 0.3% | 1.7% |
| rope_q | `rope_opt` | 0.0022 | 1 | 36 | 0.078 | 0.3% | 7.1% |
| rope_k | `rope_opt` | 0.0020 | 1 | 36 | 0.071 | 0.3% | 2.1% |

### Decode sub-kernels — KV=2048
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.1289 | 1 | 36 | 4.640 | 17.3% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1284 | 1 | 36 | 4.623 | 17.3% | 91.1% |
| fc_down | `gemm_kernel` | 0.1283 | 1 | 36 | 4.618 | 17.2% | 91.2% |
| lm_head | `gemm_kernel` | 3.8339 | 1 | 1 | 3.834 | 14.3% | 93.7% |
| pa | `paged_attention_opt__single_token` | 0.0871 | 1 | 36 | 3.136 | 11.7% | 43.9% |
| fc_qkv | `gemm_kernel` | 0.0821 | 1 | 36 | 2.957 | 11.0% | 90.0% |
| fc_o | `gemm_kernel` | 0.0558 | 1 | 36 | 2.008 | 7.5% | 88.3% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0029 | 1 | 73 | 0.213 | 0.8% | 4.8% |
| pa | `pa_kv_cache_update_ref` | 0.0041 | 1 | 36 | 0.147 | 0.5% | 937.8% |
| add | `eltwise_simple_vload8` | 0.0019 | 1 | 72 | 0.140 | 0.5% | 7.2% |
| pa | `paged_attention_opt__single_token_finalization` | 0.0039 | 1 | 36 | 0.139 | 0.5% | 994.3% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.087 | 0.3% | 6.3% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.086 | 0.3% | 1.7% |
| rope_q | `rope_opt` | 0.0022 | 1 | 36 | 0.078 | 0.3% | 7.1% |
| rope_k | `rope_opt` | 0.0020 | 1 | 36 | 0.071 | 0.3% | 2.1% |

### Decode sub-kernels — KV=4096
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.1289 | 1 | 36 | 4.640 | 16.8% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1284 | 1 | 36 | 4.623 | 16.8% | 91.1% |
| fc_down | `gemm_kernel` | 0.1283 | 1 | 36 | 4.618 | 16.7% | 91.2% |
| pa | `paged_attention_opt__gqa_single_token` | 0.1081 | 1 | 36 | 3.890 | 14.1% | 70.7% |
| lm_head | `gemm_kernel` | 3.8339 | 1 | 1 | 3.834 | 13.9% | 93.7% |
| fc_qkv | `gemm_kernel` | 0.0821 | 1 | 36 | 2.957 | 10.7% | 90.0% |
| fc_o | `gemm_kernel` | 0.0558 | 1 | 36 | 2.008 | 7.3% | 88.3% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0029 | 1 | 73 | 0.213 | 0.8% | 4.8% |
| pa | `paged_attention_opt__single_token_finalization` | 0.0054 | 1 | 36 | 0.196 | 0.7% | 1406.4% |
| pa | `pa_kv_cache_update_ref` | 0.0041 | 1 | 36 | 0.146 | 0.5% | 1879.2% |
| add | `eltwise_simple_vload8` | 0.0019 | 1 | 72 | 0.140 | 0.5% | 7.2% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.087 | 0.3% | 6.3% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.086 | 0.3% | 1.7% |
| rope_q | `rope_opt` | 0.0022 | 1 | 36 | 0.078 | 0.3% | 7.1% |
| rope_k | `rope_opt` | 0.0020 | 1 | 36 | 0.071 | 0.3% | 2.1% |

### Decode sub-kernels — KV=8192
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `paged_attention_opt__gqa_single_token` | 0.2046 | 1 | 36 | 7.367 | 23.6% | 74.6% |
| fc_up | `gemm_kernel` | 0.1289 | 1 | 36 | 4.640 | 14.8% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1284 | 1 | 36 | 4.623 | 14.8% | 91.1% |
| fc_down | `gemm_kernel` | 0.1283 | 1 | 36 | 4.618 | 14.8% | 91.2% |
| lm_head | `gemm_kernel` | 3.8339 | 1 | 1 | 3.834 | 12.3% | 93.7% |
| fc_qkv | `gemm_kernel` | 0.0821 | 1 | 36 | 2.957 | 9.5% | 90.0% |
| fc_o | `gemm_kernel` | 0.0558 | 1 | 36 | 2.008 | 6.4% | 88.3% |
| pa | `paged_attention_opt__single_token_finalization` | 0.0107 | 1 | 36 | 0.386 | 1.2% | 1423.9% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0029 | 1 | 73 | 0.213 | 0.7% | 4.8% |
| pa | `pa_kv_cache_update_ref` | 0.0043 | 1 | 36 | 0.155 | 0.5% | 3543.9% |
| add | `eltwise_simple_vload8` | 0.0019 | 1 | 72 | 0.140 | 0.4% | 7.2% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.087 | 0.3% | 6.3% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.086 | 0.3% | 1.7% |
| rope_q | `rope_opt` | 0.0022 | 1 | 36 | 0.078 | 0.2% | 7.1% |
| rope_k | `rope_opt` | 0.0020 | 1 | 36 | 0.071 | 0.2% | 2.1% |

### Prefill sub-kernels
### Prefill sub-kernels — S=1024
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.8170 | 1 | 36 | 29.411 | 15.5% | 52.9% |
| fc_gate | `gemm_kernel` | 0.8137 | 1 | 36 | 29.292 | 15.4% | 53.1% |
| fc_down | `gemm_kernel` | 0.7679 | 1 | 36 | 27.644 | 14.6% | 56.3% |
| pa | `sdpa_micro__prefill` | 0.6358 | 1 | 36 | 22.889 | 12.1% | 27.0% |
| fc_qkv | `gemm_kernel` | 0.5169 | 1 | 36 | 18.607 | 9.8% | 52.8% |
| fc_o | `gemm_kernel` | 0.3295 | 1 | 36 | 11.862 | 6.3% | 55.2% |
| fc_down | `dynamic_quantize_gpu_opt` | 0.2232 | 1 | 36 | 8.036 | 4.2% | 193.7% |
| add | `eltwise_simple_vload8` | 0.1069 | 1 | 72 | 7.699 | 4.1% | 133.7% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.1833 | 1 | 36 | 6.598 | 3.5% | 83.2% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0764 | 1 | 73 | 5.574 | 2.9% | 124.9% |
| lm_head | `gemm_kernel` | 3.8534 | 1 | 1 | 3.853 | 2.0% | 93.3% |
| rope_q | `rope_opt` | 0.1067 | 1 | 36 | 3.840 | 2.0% | 147.5% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.0899 | 1 | 36 | 3.235 | 1.7% | 202.6% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.0596 | 1 | 36 | 2.144 | 1.1% | 726.0% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.0591 | 1 | 36 | 2.129 | 1.1% | 731.0% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.0590 | 1 | 36 | 2.122 | 1.1% | 463.2% |
| pa | `pa_kv_cache_update_ref` | 0.0549 | 1 | 36 | 1.975 | 1.0% | 312.7% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0484 | 1 | 36 | 1.741 | 0.9% | 78.8% |
| rope_k | `rope_opt` | 0.0313 | 1 | 36 | 1.127 | 0.6% | 137.0% |

### Prefill sub-kernels — S=2048
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `sdpa_micro__prefill` | 2.2786 | 1 | 36 | 82.031 | 20.0% | 53.6% |
| fc_up | `gemm_kernel` | 1.6241 | 1 | 36 | 58.467 | 14.3% | 53.2% |
| fc_gate | `gemm_kernel` | 1.6124 | 1 | 36 | 58.046 | 14.2% | 53.6% |
| fc_down | `gemm_kernel` | 1.4908 | 1 | 36 | 53.670 | 13.1% | 58.0% |
| fc_qkv | `gemm_kernel` | 0.9780 | 1 | 36 | 35.210 | 8.6% | 55.8% |
| fc_o | `gemm_kernel` | 0.5845 | 1 | 36 | 21.041 | 5.1% | 62.3% |
| add | `eltwise_simple_vload8` | 0.2472 | 1 | 72 | 17.801 | 4.3% | 115.7% |
| fc_down | `dynamic_quantize_gpu_opt` | 0.4519 | 1 | 36 | 16.268 | 4.0% | 191.4% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.3760 | 1 | 36 | 13.537 | 3.3% | 81.1% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.1577 | 1 | 73 | 11.509 | 2.8% | 121.0% |
| rope_q | `rope_opt` | 0.2611 | 1 | 36 | 9.401 | 2.3% | 120.5% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.1887 | 1 | 36 | 6.795 | 1.7% | 192.9% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.1209 | 1 | 36 | 4.351 | 1.1% | 715.4% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.1193 | 1 | 36 | 4.295 | 1.0% | 724.8% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.1159 | 1 | 36 | 4.172 | 1.0% | 471.2% |
| pa | `pa_kv_cache_update_ref` | 0.1133 | 1 | 36 | 4.079 | 1.0% | 1078.5% |
| lm_head | `gemm_kernel` | 3.8534 | 1 | 1 | 3.853 | 0.9% | 93.3% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0938 | 1 | 36 | 3.376 | 0.8% | 81.3% |
| rope_k | `rope_opt` | 0.0586 | 1 | 36 | 2.109 | 0.5% | 146.4% |

### Prefill sub-kernels — S=4096
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `sdpa_micro__prefill` | 9.0720 | 1 | 36 | 326.591 | 32.6% | 53.9% |
| fc_up | `gemm_kernel` | 3.4309 | 1 | 36 | 123.513 | 12.3% | 50.4% |
| fc_gate | `gemm_kernel` | 3.3725 | 1 | 36 | 121.408 | 12.1% | 51.3% |
| fc_down | `gemm_kernel` | 2.8277 | 1 | 36 | 101.799 | 10.2% | 61.2% |
| fc_qkv | `gemm_kernel` | 2.0395 | 1 | 36 | 73.422 | 7.3% | 53.6% |
| fc_o | `gemm_kernel` | 1.1179 | 1 | 36 | 40.246 | 4.0% | 65.1% |
| add | `eltwise_simple_vload8` | 0.5570 | 1 | 72 | 40.103 | 4.0% | 102.7% |
| fc_down | `dynamic_quantize_gpu_opt` | 0.9413 | 1 | 36 | 33.888 | 3.4% | 183.7% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.7701 | 1 | 36 | 27.723 | 2.8% | 79.2% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.3651 | 1 | 73 | 26.649 | 2.7% | 104.5% |
| rope_q | `rope_opt` | 0.6176 | 1 | 36 | 22.235 | 2.2% | 101.9% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.3753 | 1 | 36 | 13.510 | 1.3% | 194.0% |
| pa | `pa_kv_cache_update_ref` | 0.2592 | 1 | 36 | 9.332 | 0.9% | 1885.5% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.2472 | 1 | 36 | 8.898 | 0.9% | 699.7% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.2432 | 1 | 36 | 8.756 | 0.9% | 711.1% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.2326 | 1 | 36 | 8.374 | 0.8% | 469.6% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.1875 | 1 | 36 | 6.752 | 0.7% | 81.3% |
| rope_k | `rope_opt` | 0.1184 | 1 | 36 | 4.263 | 0.4% | 144.9% |
| lm_head | `gemm_kernel` | 3.8534 | 1 | 1 | 3.853 | 0.4% | 93.3% |

### Prefill sub-kernels — S=8192
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `sdpa_micro__prefill` | 34.8059 | 1 | 36 | 1253.012 | 48.4% | 56.2% |
| fc_up | `gemm_kernel` | 6.3923 | 1 | 36 | 230.123 | 8.9% | 54.1% |
| fc_gate | `gemm_kernel` | 6.3642 | 1 | 36 | 229.110 | 8.9% | 54.3% |
| fc_down | `gemm_kernel` | 5.4703 | 1 | 36 | 196.931 | 7.6% | 63.2% |
| fc_qkv | `gemm_kernel` | 4.1182 | 1 | 36 | 148.254 | 5.7% | 53.0% |
| add | `eltwise_simple_vload8` | 1.1972 | 1 | 72 | 86.198 | 3.3% | 95.5% |
| fc_o | `gemm_kernel` | 2.2137 | 1 | 36 | 79.693 | 3.1% | 65.8% |
| fc_down | `dynamic_quantize_gpu_opt` | 2.0834 | 1 | 36 | 75.001 | 2.9% | 166.0% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 1.6433 | 1 | 36 | 59.158 | 2.3% | 74.3% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.7911 | 1 | 73 | 57.751 | 2.2% | 96.4% |
| rope_q | `rope_opt` | 1.3314 | 1 | 36 | 47.930 | 1.9% | 94.5% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.7451 | 1 | 36 | 26.824 | 1.0% | 195.5% |
| pa | `pa_kv_cache_update_ref` | 0.4930 | 1 | 36 | 17.746 | 0.7% | 3966.2% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.4920 | 1 | 36 | 17.711 | 0.7% | 703.1% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.4918 | 1 | 36 | 17.704 | 0.7% | 703.3% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.4652 | 1 | 36 | 16.746 | 0.6% | 469.6% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.3842 | 1 | 36 | 13.832 | 0.5% | 79.4% |
| rope_k | `rope_opt` | 0.3021 | 1 | 36 | 10.877 | 0.4% | 113.6% |
| lm_head | `gemm_kernel` | 3.8534 | 1 | 1 | 3.853 | 0.1% | 93.3% |

## Top contributors (sorted by total ms per inference)

### decode
| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | fc_up 4.64ms (18%) | fc_gate 4.62ms (18%) | fc_down 4.62ms (18%) |
| 2048 | fc_up 4.64ms (17%) | fc_gate 4.62ms (17%) | fc_down 4.62ms (17%) |
| 4096 | fc_up 4.64ms (17%) | fc_gate 4.62ms (17%) | fc_down 4.62ms (17%) |
| 8192 | pa 7.91ms (25%) | fc_up 4.64ms (15%) | fc_gate 4.62ms (15%) |

### prefill
| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | fc_down 35.68ms (19%) | fc_up 31.54ms (17%) | fc_gate 31.44ms (17%) |
| 2048 | pa 86.11ms (21%) | fc_down 69.94ms (17%) | fc_up 62.82ms (15%) |
| 4096 | pa 335.92ms (34%) | fc_down 135.69ms (14%) | fc_up 132.41ms (13%) |
| 8192 | pa 1270.76ms (49%) | fc_down 271.93ms (11%) | fc_up 247.83ms (10%) |

## Caveats & method
- Each op profiled in its own process via cliloader Device Performance Timing; we use mean kernel time (mode-of-call-counts) per iteration.
- FC weight bytes count INT4 weight + FP16 scale/zp(g=128) + FP16 act + FP16 out.
- PA bytes assume INT8 KV cache (1B/elem) + FP16 Q, FP16 out.
- Decode FC is treated as **memory-bound** (per SKILL: weights read dominates at M=1); prefill FC is **INT8 XMX compute-bound** (S big enough to hit XMX).
- Prefill PA at S≥2048 is compute-bound (FP16 micro-kernel); decode PA is memory-bound.
- swish/multiply/add eltwise are typically fused into matmul/SwiGLU in real inference; they are listed for visibility. swish standalone could not be measured (the GPU plugin fuses Parameter→Swish→Result into a noop activation that the parser excludes).
- lm_head is run only once per token (last position in prefill, every step in decode).
- Target machine: Local_Admin@10.239.132.229 (Windows PTL 12Xe / Arc B390).

## Reproduction
```cmd
REM On the 12Xe Windows host (Local_Admin@10.239.132.229):
D:\river\moe\dev_roofline_profiling\utils\run_qwen3_omni_ptl_12xe.bat
REM Logs land in D:\river\moe\roofline_results\qwen3_omni\ptl_12xe\. Then locally:
python utils/parse_logs.py outputs/qwen3_omni/logs_ptl/ outputs/qwen3_omni/kernels_raw.json
python outputs/qwen3_omni/build_report.py
```
