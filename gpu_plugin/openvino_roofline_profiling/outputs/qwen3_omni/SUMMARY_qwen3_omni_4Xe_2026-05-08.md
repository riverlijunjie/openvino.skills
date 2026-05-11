# qwen3_omni Thinker text — Roofline on PTL 4Xe Linux (2026-05-08)

**Platform**: Intel PTL 4Xe iGPU (32 EUs = 4 Xe × 8 EU × 10 thr), 2450 MHz, BW≈110 GB/s, FP16 XMX peak ≈ 20.07 TFLOPS, INT8 XMX peak ≈ 40.14 TOPS.
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
| FP16 XMX peak | 20.07 TFLOPS |
| INT8 XMX peak | 40.14 TOPS |
| Memory BW | 110 GB/s |
| Ridge point (FP16) | 182 FLOP/byte |

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
| 1024 | 691.28 | 0.691 | 0.6751 | 1481.3 |
| 2048 | 1917.18 | 1.917 | 0.9361 | 1068.2 |
| 4096 | 5949.81 | 5.950 | 1.4526 | 688.4 |
| 8192 | 20482.78 | 20.483 | 2.5003 | 399.9 |

### Decode — TPOT (per output token)
| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 1024 | 28.311 | 35.3 |
| 2048 | 30.137 | 33.2 |
| 4096 | 29.449 | 34.0 |
| 8192 | 34.524 | 29.0 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)
| op | KV=1024 ms (%) | KV=2048 ms (%) | KV=4096 ms (%) | KV=8192 ms (%) |
|---|---:|---:|---:|---:|
| FC down | 4.642 (16.4%) | 4.642 (15.4%) | 4.642 (15.8%) | 4.642 (13.4%) |
| FC up | 4.612 (16.3%) | 4.612 (15.3%) | 4.612 (15.7%) | 4.612 (13.4%) |
| FC gate | 4.633 (16.4%) | 4.633 (15.4%) | 4.633 (15.7%) | 4.633 (13.4%) |
| FC qkv (fused Q+K+V) | 3.015 (10.7%) | 3.015 (10.0%) | 3.015 (10.2%) | 3.015 (8.7%) |
| FC o | 2.002 (7.1%) | 2.002 (6.6%) | 2.002 (6.8%) | 2.002 (5.8%) |
| lm_head | 3.783 (13.4%) | 3.783 (12.6%) | 3.783 (12.8%) | 3.783 (11.0%) |
| PagedAttention | 5.014 (17.7%) | 6.839 (22.7%) | 6.151 (20.9%) | 11.227 (32.5%) |
| RMSNorm | 0.202 (0.7%) | 0.202 (0.7%) | 0.202 (0.7%) | 0.202 (0.6%) |
| RMSNorm q | 0.079 (0.3%) | 0.079 (0.3%) | 0.079 (0.3%) | 0.079 (0.2%) |
| RMSNorm k | 0.078 (0.3%) | 0.078 (0.3%) | 0.078 (0.3%) | 0.078 (0.2%) |
| RoPE q | 0.072 (0.3%) | 0.072 (0.2%) | 0.072 (0.2%) | 0.072 (0.2%) |
| RoPE k | 0.066 (0.2%) | 0.066 (0.2%) | 0.066 (0.2%) | 0.066 (0.2%) |
| Residual Add | 0.112 (0.4%) | 0.112 (0.4%) | 0.112 (0.4%) | 0.112 (0.3%) |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)
| op | S=1024 ms (%) | S=2048 ms (%) | S=4096 ms (%) | S=8192 ms (%) |
|---|---:|---:|---:|---:|
| FC down | 85.91 (12.4%) | 171.72 (9.0%) | 342.76 (5.8%) | 683.93 (3.3%) |
| FC up | 78.35 (11.3%) | 156.15 (8.1%) | 301.92 (5.1%) | 569.75 (2.8%) |
| FC gate | 78.22 (11.3%) | 155.94 (8.1%) | 301.83 (5.1%) | 569.53 (2.8%) |
| FC qkv (fused Q+K+V) | 54.12 (7.8%) | 108.39 (5.7%) | 196.14 (3.3%) | 399.16 (1.9%) |
| FC o | 35.08 (5.1%) | 66.83 (3.5%) | 132.35 (2.2%) | 268.79 (1.3%) |
| lm_head | 3.77 (0.5%) | 3.77 (0.2%) | 3.77 (0.1%) | 3.77 (0.0%) |
| PagedAttention | 298.86 (43.2%) | 1138.60 (59.4%) | 4438.59 (74.6%) | 17519.42 (85.5%) |
| RMSNorm | 15.09 (2.2%) | 30.45 (1.6%) | 60.45 (1.0%) | 120.26 (0.6%) |
| RMSNorm q | 18.93 (2.7%) | 37.74 (2.0%) | 75.62 (1.3%) | 150.41 (0.7%) |
| RMSNorm k | 4.87 (0.7%) | 9.53 (0.5%) | 18.91 (0.3%) | 37.64 (0.2%) |
| RoPE q | 6.70 (1.0%) | 14.21 (0.7%) | 28.51 (0.5%) | 59.92 (0.3%) |
| RoPE k | 1.47 (0.2%) | 3.17 (0.2%) | 6.88 (0.1%) | 14.67 (0.1%) |
| Residual Add | 9.91 (1.4%) | 20.66 (1.1%) | 42.07 (0.7%) | 85.52 (0.4%) |

## Decode tables (1 query token, KV = context length)
### Decode — KV=1024
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro | 0.1393 | 36 | 5.014 | 126 | 15.2 | 13.8% | memory |
| fc_down | fc_int4_g128 | 0.1289 | 36 | 4.642 | 386 | 99.8 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1287 | 36 | 4.633 | 387 | 100.0 | 90.9% | memory |
| fc_up | fc_int4_g128 | 0.1281 | 36 | 4.612 | 389 | 100.4 | 91.3% | memory |
| lm_head | fc_int8_g128 | 3.7834 | 1 | 3.783 | 206 | 104.5 | 95.0% | memory |
| fc_qkv | fc_int4_g128 | 0.0838 | 36 | 3.015 | 376 | 97.0 | 88.2% | memory |
| fc_o | fc_int4_g128 | 0.0556 | 36 | 2.002 | 377 | 97.4 | 88.6% | memory |
| rmsnorm | rmsnorm | 0.0028 | 73 | 0.202 | 0 | 5.6 | 5.1% | memory |
| add | add | 0.0016 | 72 | 0.112 | 0 | 9.8 | 8.9% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0022 | 36 | 0.079 | 0 | 7.6 | 6.9% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0022 | 36 | 0.078 | 0 | 2.0 | 1.8% | memory |
| rope_q | rope_q | 0.0020 | 36 | 0.072 | 0 | 8.4 | 7.7% | memory |
| rope_k | rope_k | 0.0018 | 36 | 0.066 | 0 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  | **28.311** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=2048
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro | 0.1900 | 36 | 6.839 | 185 | 22.2 | 20.1% | memory |
| fc_down | fc_int4_g128 | 0.1289 | 36 | 4.642 | 386 | 99.8 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1287 | 36 | 4.633 | 387 | 100.0 | 90.9% | memory |
| fc_up | fc_int4_g128 | 0.1281 | 36 | 4.612 | 389 | 100.4 | 91.3% | memory |
| lm_head | fc_int8_g128 | 3.7834 | 1 | 3.783 | 206 | 104.5 | 95.0% | memory |
| fc_qkv | fc_int4_g128 | 0.0838 | 36 | 3.015 | 376 | 97.0 | 88.2% | memory |
| fc_o | fc_int4_g128 | 0.0556 | 36 | 2.002 | 377 | 97.4 | 88.6% | memory |
| rmsnorm | rmsnorm | 0.0028 | 73 | 0.202 | 0 | 5.6 | 5.1% | memory |
| add | add | 0.0016 | 72 | 0.112 | 0 | 9.8 | 8.9% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0022 | 36 | 0.079 | 0 | 7.6 | 6.9% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0022 | 36 | 0.078 | 0 | 2.0 | 1.8% | memory |
| rope_q | rope_q | 0.0020 | 36 | 0.072 | 0 | 8.4 | 7.7% | memory |
| rope_k | rope_k | 0.0018 | 36 | 0.066 | 0 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  | **30.137** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=4096
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro | 0.1709 | 36 | 6.151 | 412 | 49.2 | 44.7% | memory |
| fc_down | fc_int4_g128 | 0.1289 | 36 | 4.642 | 386 | 99.8 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1287 | 36 | 4.633 | 387 | 100.0 | 90.9% | memory |
| fc_up | fc_int4_g128 | 0.1281 | 36 | 4.612 | 389 | 100.4 | 91.3% | memory |
| lm_head | fc_int8_g128 | 3.7834 | 1 | 3.783 | 206 | 104.5 | 95.0% | memory |
| fc_qkv | fc_int4_g128 | 0.0838 | 36 | 3.015 | 376 | 97.0 | 88.2% | memory |
| fc_o | fc_int4_g128 | 0.0556 | 36 | 2.002 | 377 | 97.4 | 88.6% | memory |
| rmsnorm | rmsnorm | 0.0028 | 73 | 0.202 | 0 | 5.6 | 5.1% | memory |
| add | add | 0.0016 | 72 | 0.112 | 0 | 9.8 | 8.9% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0022 | 36 | 0.079 | 0 | 7.6 | 6.9% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0022 | 36 | 0.078 | 0 | 2.0 | 1.8% | memory |
| rope_q | rope_q | 0.0020 | 36 | 0.072 | 0 | 8.4 | 7.7% | memory |
| rope_k | rope_k | 0.0018 | 36 | 0.066 | 0 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  | **29.449** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=8192
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro | 0.3119 | 36 | 11.227 | 451 | 53.9 | 49.0% | memory |
| fc_down | fc_int4_g128 | 0.1289 | 36 | 4.642 | 386 | 99.8 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1287 | 36 | 4.633 | 387 | 100.0 | 90.9% | memory |
| fc_up | fc_int4_g128 | 0.1281 | 36 | 4.612 | 389 | 100.4 | 91.3% | memory |
| lm_head | fc_int8_g128 | 3.7834 | 1 | 3.783 | 206 | 104.5 | 95.0% | memory |
| fc_qkv | fc_int4_g128 | 0.0838 | 36 | 3.015 | 376 | 97.0 | 88.2% | memory |
| fc_o | fc_int4_g128 | 0.0556 | 36 | 2.002 | 377 | 97.4 | 88.6% | memory |
| rmsnorm | rmsnorm | 0.0028 | 73 | 0.202 | 0 | 5.6 | 5.1% | memory |
| add | add | 0.0016 | 72 | 0.112 | 0 | 9.8 | 8.9% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0022 | 36 | 0.079 | 0 | 7.6 | 6.9% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0022 | 36 | 0.078 | 0 | 2.0 | 1.8% | memory |
| rope_q | rope_q | 0.0020 | 36 | 0.072 | 0 | 8.4 | 7.7% | memory |
| rope_k | rope_k | 0.0018 | 36 | 0.066 | 0 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  | **34.524** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

## Prefill tables (single forward over S tokens)
### Prefill — S=1024
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro_prefill | 8.3017 | 36 | 298.862 | 2170 | 2.3 | 2.1% | memory |
| fc_down | fc_int4_g128 | 2.3863 | 36 | 85.905 | 21374 | 15.9 | 53.2% | compute |
| fc_up | fc_int4_g128 | 2.1764 | 36 | 78.352 | 23434 | 17.5 | 58.4% | compute |
| fc_gate | fc_int4_g128 | 2.1728 | 36 | 78.221 | 23473 | 17.5 | 58.5% | compute |
| fc_qkv | fc_int4_g128 | 1.5034 | 36 | 54.121 | 21427 | 17.3 | 53.4% | compute |
| fc_o | fc_int4_g128 | 0.9744 | 36 | 35.079 | 22039 | 19.5 | 54.9% | compute |
| rmsnorm3d_q | rmsnorm3d_q | 0.5258 | 36 | 18.930 | 0 | 31.9 | 29.0% | memory |
| rmsnorm | rmsnorm | 0.2068 | 73 | 15.095 | 0 | 50.7 | 46.1% | memory |
| add | add | 0.1377 | 72 | 9.912 | 0 | 114.2 | 103.9% | memory |
| rope_q | rope_q | 0.1861 | 36 | 6.699 | 0 | 93.0 | 84.5% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.1352 | 36 | 4.866 | 0 | 31.0 | 28.2% | memory |
| lm_head | fc_int8_g128 | 3.7707 | 1 | 3.771 | 206 | 104.8 | 95.3% | memory |
| rope_k | rope_k | 0.0408 | 36 | 1.469 | 0 | 115.6 | 105.1% | memory |
| **TOTAL** |  |  |  | **691.282** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=2048
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro_prefill | 31.6278 | 36 | 1138.601 | 2279 | 1.2 | 11.4% | compute |
| fc_down | fc_int4_g128 | 4.7700 | 36 | 171.720 | 21385 | 13.2 | 53.3% | compute |
| fc_up | fc_int4_g128 | 4.3376 | 36 | 156.152 | 23517 | 14.6 | 58.6% | compute |
| fc_gate | fc_int4_g128 | 4.3317 | 36 | 155.942 | 23549 | 14.6 | 58.7% | compute |
| fc_qkv | fc_int4_g128 | 3.0110 | 36 | 108.395 | 21397 | 14.5 | 53.3% | compute |
| fc_o | fc_int4_g128 | 1.8563 | 36 | 66.827 | 23137 | 17.6 | 57.6% | compute |
| rmsnorm3d_q | rmsnorm3d_q | 1.0484 | 36 | 37.742 | 0 | 32.0 | 29.1% | memory |
| rmsnorm | rmsnorm | 0.4172 | 73 | 30.453 | 0 | 50.3 | 45.7% | memory |
| add | add | 0.2870 | 72 | 20.664 | 0 | 109.6 | 99.6% | memory |
| rope_q | rope_q | 0.3948 | 36 | 14.214 | 0 | 87.6 | 79.7% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.2648 | 36 | 9.532 | 0 | 31.7 | 28.8% | memory |
| lm_head | fc_int8_g128 | 3.7707 | 1 | 3.771 | 206 | 104.8 | 95.3% | memory |
| rope_k | rope_k | 0.0880 | 36 | 3.168 | 0 | 107.2 | 97.5% | memory |
| **TOTAL** |  |  |  | **1917.180** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=4096
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro_prefill | 123.2943 | 36 | 4438.595 | 2338 | 0.6 | 11.7% | compute |
| fc_down | fc_int4_g128 | 9.5210 | 36 | 342.757 | 21427 | 11.9 | 53.4% | compute |
| fc_up | fc_int4_g128 | 8.3867 | 36 | 301.922 | 24325 | 13.5 | 60.6% | compute |
| fc_gate | fc_int4_g128 | 8.3843 | 36 | 301.833 | 24333 | 13.5 | 60.6% | compute |
| fc_qkv | fc_int4_g128 | 5.4484 | 36 | 196.144 | 23649 | 14.6 | 58.9% | compute |
| fc_o | fc_int4_g128 | 3.6763 | 36 | 132.346 | 23366 | 16.3 | 58.2% | compute |
| rmsnorm3d_q | rmsnorm3d_q | 2.1006 | 36 | 75.621 | 0 | 31.9 | 29.0% | memory |
| rmsnorm | rmsnorm | 0.8281 | 73 | 60.453 | 0 | 50.7 | 46.0% | memory |
| add | add | 0.5843 | 72 | 42.066 | 0 | 107.7 | 97.9% | memory |
| rope_q | rope_q | 0.7920 | 36 | 28.513 | 0 | 87.4 | 79.4% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.5252 | 36 | 18.906 | 0 | 31.9 | 29.0% | memory |
| rope_k | rope_k | 0.1912 | 36 | 6.883 | 0 | 98.7 | 89.7% | memory |
| lm_head | fc_int8_g128 | 3.7707 | 1 | 3.771 | 206 | 104.8 | 95.3% | memory |
| **TOTAL** |  |  |  | **5949.808** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=8192
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro_prefill | 486.6505 | 36 | 17519.418 | 2370 | 0.3 | 11.8% | compute |
| fc_down | fc_int4_g128 | 18.9979 | 36 | 683.925 | 21477 | 11.3 | 53.5% | compute |
| fc_up | fc_int4_g128 | 15.8264 | 36 | 569.750 | 25781 | 13.5 | 64.2% | compute |
| fc_gate | fc_int4_g128 | 15.8203 | 36 | 569.531 | 25791 | 13.5 | 64.3% | compute |
| fc_qkv | fc_int4_g128 | 11.0879 | 36 | 399.165 | 23241 | 13.6 | 57.9% | compute |
| fc_o | fc_int4_g128 | 7.4665 | 36 | 268.793 | 23009 | 15.3 | 57.3% | compute |
| rmsnorm3d_q | rmsnorm3d_q | 4.1780 | 36 | 150.406 | 0 | 32.1 | 29.2% | memory |
| rmsnorm | rmsnorm | 1.6474 | 73 | 120.263 | 0 | 50.9 | 46.3% | memory |
| add | add | 1.1877 | 72 | 85.517 | 0 | 105.9 | 96.3% | memory |
| rope_q | rope_q | 1.6644 | 36 | 59.919 | 0 | 83.2 | 75.6% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 1.0456 | 36 | 37.643 | 0 | 32.1 | 29.2% | memory |
| rope_k | rope_k | 0.4076 | 36 | 14.674 | 0 | 92.6 | 84.2% | memory |
| lm_head | fc_int8_g128 | 3.7707 | 1 | 3.771 | 206 | 104.8 | 95.3% | memory |
| **TOTAL** |  |  |  | **20482.776** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

## Per-kernel decomposition (cliloader kernel names)
Each op above maps to one or more GPU kernels. Below shows the actual kernel names captured by cliloader's *Device Performance Timing* section, with per-launch time, launches per op-call, total ms across one inference, and per-kernel **Eff%** (peak utilization). The Eff% column attributes the **parent op's full FLOPs/bytes per op-call** against the kernel's own per-launch time, so the dominant kernel reports an Eff% close to the op-level value, while helper kernels (e.g. `pa_kv_cache_update_ref`, `dynamic_quantize_gpu_opt`, `*_finalization`) appear with apparently very high Eff% — that is the expected signal that they are *not* the bottleneck for the op. PA decomposes into `pa_kv_cache_update_ref` + the attention kernel (`paged_attention_opt__single_token` / `__gqa_single_token` / `sdpa_micro__prefill`) + finalization. Prefill FC decomposes into `dynamic_quantize_gpu_opt` + `gemm_kernel`.

### Decode sub-kernels
### Decode sub-kernels — KV=1024
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `paged_attention_opt__single_token` | 0.1313 | 1 | 36 | 4.728 | 16.7% | 14.6% |
| fc_down | `gemm_kernel` | 0.1289 | 1 | 36 | 4.642 | 16.4% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1287 | 1 | 36 | 4.633 | 16.4% | 90.9% |
| fc_up | `gemm_kernel` | 0.1281 | 1 | 36 | 4.612 | 16.3% | 91.3% |
| lm_head | `gemm_kernel` | 3.7834 | 1 | 1 | 3.783 | 13.4% | 95.0% |
| fc_qkv | `gemm_kernel` | 0.0838 | 1 | 36 | 3.015 | 10.7% | 88.2% |
| fc_o | `gemm_kernel` | 0.0556 | 1 | 36 | 2.002 | 7.1% | 88.6% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0028 | 1 | 73 | 0.202 | 0.7% | 5.1% |
| pa | `pa_kv_cache_update_ref` | 0.0048 | 1 | 36 | 0.171 | 0.6% | 404.4% |
| pa | `paged_attention_opt__single_token_finalization` | 0.0032 | 1 | 36 | 0.115 | 0.4% | 600.6% |
| add | `eltwise_simple_vload8` | 0.0016 | 1 | 72 | 0.112 | 0.4% | 8.9% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0022 | 1 | 36 | 0.079 | 0.3% | 6.9% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0022 | 1 | 36 | 0.078 | 0.3% | 1.8% |
| rope_q | `rope_opt` | 0.0020 | 1 | 36 | 0.072 | 0.3% | 7.7% |
| rope_k | `rope_opt` | 0.0018 | 1 | 36 | 0.066 | 0.2% | 2.3% |

### Decode sub-kernels — KV=2048
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `paged_attention_opt__single_token` | 0.1826 | 1 | 36 | 6.575 | 21.8% | 21.0% |
| fc_down | `gemm_kernel` | 0.1289 | 1 | 36 | 4.642 | 15.4% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1287 | 1 | 36 | 4.633 | 15.4% | 90.9% |
| fc_up | `gemm_kernel` | 0.1281 | 1 | 36 | 4.612 | 15.3% | 91.3% |
| lm_head | `gemm_kernel` | 3.7834 | 1 | 1 | 3.783 | 12.6% | 95.0% |
| fc_qkv | `gemm_kernel` | 0.0838 | 1 | 36 | 3.015 | 10.0% | 88.2% |
| fc_o | `gemm_kernel` | 0.0556 | 1 | 36 | 2.002 | 6.6% | 88.6% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0028 | 1 | 73 | 0.202 | 0.7% | 5.1% |
| pa | `pa_kv_cache_update_ref` | 0.0038 | 1 | 36 | 0.135 | 0.4% | 1017.5% |
| pa | `paged_attention_opt__single_token_finalization` | 0.0036 | 1 | 36 | 0.129 | 0.4% | 1072.2% |
| add | `eltwise_simple_vload8` | 0.0016 | 1 | 72 | 0.112 | 0.4% | 8.9% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0022 | 1 | 36 | 0.079 | 0.3% | 6.9% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0022 | 1 | 36 | 0.078 | 0.3% | 1.8% |
| rope_q | `rope_opt` | 0.0020 | 1 | 36 | 0.072 | 0.2% | 7.7% |
| rope_k | `rope_opt` | 0.0018 | 1 | 36 | 0.066 | 0.2% | 2.3% |

### Decode sub-kernels — KV=4096
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `paged_attention_opt__gqa_single_token` | 0.1600 | 1 | 36 | 5.761 | 19.6% | 47.7% |
| fc_down | `gemm_kernel` | 0.1289 | 1 | 36 | 4.642 | 15.8% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1287 | 1 | 36 | 4.633 | 15.7% | 90.9% |
| fc_up | `gemm_kernel` | 0.1281 | 1 | 36 | 4.612 | 15.7% | 91.3% |
| lm_head | `gemm_kernel` | 3.7834 | 1 | 1 | 3.783 | 12.8% | 95.0% |
| fc_qkv | `gemm_kernel` | 0.0838 | 1 | 36 | 3.015 | 10.2% | 88.2% |
| fc_o | `gemm_kernel` | 0.0556 | 1 | 36 | 2.002 | 6.8% | 88.6% |
| pa | `paged_attention_opt__single_token_finalization` | 0.0069 | 1 | 36 | 0.250 | 0.8% | 1099.4% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0028 | 1 | 73 | 0.202 | 0.7% | 5.1% |
| pa | `pa_kv_cache_update_ref` | 0.0039 | 1 | 36 | 0.140 | 0.5% | 1967.3% |
| add | `eltwise_simple_vload8` | 0.0016 | 1 | 72 | 0.112 | 0.4% | 8.9% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0022 | 1 | 36 | 0.079 | 0.3% | 6.9% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0022 | 1 | 36 | 0.078 | 0.3% | 1.8% |
| rope_q | `rope_opt` | 0.0020 | 1 | 36 | 0.072 | 0.2% | 7.7% |
| rope_k | `rope_opt` | 0.0018 | 1 | 36 | 0.066 | 0.2% | 2.3% |

### Decode sub-kernels — KV=8192
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `paged_attention_opt__gqa_single_token` | 0.2946 | 1 | 36 | 10.605 | 30.7% | 51.8% |
| fc_down | `gemm_kernel` | 0.1289 | 1 | 36 | 4.642 | 13.4% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1287 | 1 | 36 | 4.633 | 13.4% | 90.9% |
| fc_up | `gemm_kernel` | 0.1281 | 1 | 36 | 4.612 | 13.4% | 91.3% |
| lm_head | `gemm_kernel` | 3.7834 | 1 | 1 | 3.783 | 11.0% | 95.0% |
| fc_qkv | `gemm_kernel` | 0.0838 | 1 | 36 | 3.015 | 8.7% | 88.2% |
| fc_o | `gemm_kernel` | 0.0556 | 1 | 36 | 2.002 | 5.8% | 88.6% |
| pa | `paged_attention_opt__single_token_finalization` | 0.0134 | 1 | 36 | 0.483 | 1.4% | 1137.9% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0028 | 1 | 73 | 0.202 | 0.6% | 5.1% |
| pa | `pa_kv_cache_update_ref` | 0.0039 | 1 | 36 | 0.139 | 0.4% | 3961.3% |
| add | `eltwise_simple_vload8` | 0.0016 | 1 | 72 | 0.112 | 0.3% | 8.9% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0022 | 1 | 36 | 0.079 | 0.2% | 6.9% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0022 | 1 | 36 | 0.078 | 0.2% | 1.8% |
| rope_q | `rope_opt` | 0.0020 | 1 | 36 | 0.072 | 0.2% | 7.7% |
| rope_k | `rope_opt` | 0.0018 | 1 | 36 | 0.066 | 0.2% | 2.3% |

### Prefill sub-kernels
### Prefill sub-kernels — S=1024
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `sdpa_micro__prefill` | 8.2181 | 1 | 36 | 295.852 | 42.8% | 2.1% |
| fc_up | `gemm_kernel` | 2.0762 | 1 | 36 | 74.744 | 10.8% | 61.2% |
| fc_gate | `gemm_kernel` | 2.0723 | 1 | 36 | 74.603 | 10.8% | 61.3% |
| fc_down | `gemm_kernel` | 1.9666 | 1 | 36 | 70.799 | 10.2% | 64.6% |
| fc_qkv | `gemm_kernel` | 1.4031 | 1 | 36 | 50.513 | 7.3% | 57.2% |
| fc_o | `gemm_kernel` | 0.7960 | 1 | 36 | 28.655 | 4.1% | 67.2% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.5258 | 1 | 36 | 18.930 | 2.7% | 29.0% |
| fc_down | `dynamic_quantize_gpu_opt` | 0.4196 | 1 | 36 | 15.107 | 2.2% | 302.8% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.2068 | 1 | 73 | 15.095 | 2.2% | 46.1% |
| add | `eltwise_simple_vload8` | 0.1377 | 1 | 72 | 9.912 | 1.4% | 103.9% |
| rope_q | `rope_opt` | 0.1861 | 1 | 36 | 6.699 | 1.0% | 84.5% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.1784 | 1 | 36 | 6.424 | 0.9% | 299.8% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.1352 | 1 | 36 | 4.866 | 0.7% | 28.2% |
| lm_head | `gemm_kernel` | 3.7707 | 1 | 1 | 3.771 | 0.5% | 95.3% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.1005 | 1 | 36 | 3.618 | 0.5% | 1264.2% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.1002 | 1 | 36 | 3.608 | 0.5% | 1267.9% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.1002 | 1 | 36 | 3.608 | 0.5% | 800.8% |
| pa | `pa_kv_cache_update_ref` | 0.0836 | 1 | 36 | 3.010 | 0.4% | 205.2% |
| rope_k | `rope_opt` | 0.0408 | 1 | 36 | 1.469 | 0.2% | 105.1% |

### Prefill sub-kernels — S=2048
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `sdpa_micro__prefill` | 31.4669 | 1 | 36 | 1132.808 | 59.1% | 11.4% |
| fc_up | `gemm_kernel` | 4.1401 | 1 | 36 | 149.043 | 7.8% | 61.4% |
| fc_gate | `gemm_kernel` | 4.1331 | 1 | 36 | 148.792 | 7.8% | 61.5% |
| fc_down | `gemm_kernel` | 3.9157 | 1 | 36 | 140.965 | 7.4% | 64.9% |
| fc_qkv | `gemm_kernel` | 2.8129 | 1 | 36 | 101.263 | 5.3% | 57.1% |
| fc_o | `gemm_kernel` | 1.4969 | 1 | 36 | 53.890 | 2.8% | 71.5% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 1.0484 | 1 | 36 | 37.742 | 2.0% | 29.1% |
| fc_down | `dynamic_quantize_gpu_opt` | 0.8543 | 1 | 36 | 30.755 | 1.6% | 297.5% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.4172 | 1 | 73 | 30.453 | 1.6% | 45.7% |
| add | `eltwise_simple_vload8` | 0.2870 | 1 | 72 | 20.664 | 1.1% | 99.6% |
| rope_q | `rope_opt` | 0.3948 | 1 | 36 | 14.214 | 0.7% | 79.7% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.3594 | 1 | 36 | 12.937 | 0.7% | 297.7% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.2648 | 1 | 36 | 9.532 | 0.5% | 28.8% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.1986 | 1 | 36 | 7.149 | 0.4% | 1279.6% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.1981 | 1 | 36 | 7.132 | 0.4% | 810.2% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.1975 | 1 | 36 | 7.109 | 0.4% | 1286.9% |
| pa | `pa_kv_cache_update_ref` | 0.1609 | 1 | 36 | 5.793 | 0.3% | 2231.6% |
| lm_head | `gemm_kernel` | 3.7707 | 1 | 1 | 3.771 | 0.2% | 95.3% |
| rope_k | `rope_opt` | 0.0880 | 1 | 36 | 3.168 | 0.2% | 97.5% |

### Prefill sub-kernels — S=4096
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `sdpa_micro__prefill` | 123.0025 | 1 | 36 | 4428.088 | 74.4% | 11.7% |
| fc_up | `gemm_kernel` | 7.9935 | 1 | 36 | 287.767 | 4.8% | 63.6% |
| fc_gate | `gemm_kernel` | 7.9896 | 1 | 36 | 287.626 | 4.8% | 63.6% |
| fc_down | `gemm_kernel` | 7.8116 | 1 | 36 | 281.217 | 4.7% | 65.1% |
| fc_qkv | `gemm_kernel` | 5.0538 | 1 | 36 | 181.935 | 3.1% | 63.5% |
| fc_o | `gemm_kernel` | 2.9545 | 1 | 36 | 106.363 | 1.8% | 72.4% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 2.1006 | 1 | 36 | 75.621 | 1.3% | 29.0% |
| fc_down | `dynamic_quantize_gpu_opt` | 1.7094 | 1 | 36 | 61.540 | 1.0% | 297.3% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.8281 | 1 | 73 | 60.453 | 1.0% | 46.0% |
| add | `eltwise_simple_vload8` | 0.5843 | 1 | 72 | 42.066 | 0.7% | 97.9% |
| rope_q | `rope_opt` | 0.7920 | 1 | 36 | 28.513 | 0.5% | 79.4% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.7217 | 1 | 36 | 25.983 | 0.4% | 296.5% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.5252 | 1 | 36 | 18.906 | 0.3% | 29.0% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.3947 | 1 | 36 | 14.209 | 0.2% | 813.3% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.3946 | 1 | 36 | 14.207 | 0.2% | 1287.9% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.3932 | 1 | 36 | 14.154 | 0.2% | 1292.7% |
| pa | `pa_kv_cache_update_ref` | 0.2918 | 1 | 36 | 10.506 | 0.2% | 4922.0% |
| rope_k | `rope_opt` | 0.1912 | 1 | 36 | 6.883 | 0.1% | 89.7% |
| lm_head | `gemm_kernel` | 3.7707 | 1 | 1 | 3.771 | 0.1% | 95.3% |

### Prefill sub-kernels — S=8192
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `sdpa_micro__prefill` | 486.0948 | 1 | 36 | 17499.414 | 85.4% | 11.8% |
| fc_down | `gemm_kernel` | 15.5533 | 1 | 36 | 559.917 | 2.7% | 65.4% |
| fc_up | `gemm_kernel` | 15.0313 | 1 | 36 | 541.125 | 2.6% | 67.6% |
| fc_gate | `gemm_kernel` | 15.0224 | 1 | 36 | 540.807 | 2.6% | 67.7% |
| fc_qkv | `gemm_kernel` | 10.3006 | 1 | 36 | 370.822 | 1.8% | 62.3% |
| fc_o | `gemm_kernel` | 6.0022 | 1 | 36 | 216.078 | 1.1% | 71.3% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 4.1780 | 1 | 36 | 150.406 | 0.7% | 29.2% |
| fc_down | `dynamic_quantize_gpu_opt` | 3.4447 | 1 | 36 | 124.008 | 0.6% | 295.1% |
| rmsnorm | `rms_gpu_bfyx_opt` | 1.6474 | 1 | 73 | 120.263 | 0.6% | 46.3% |
| add | `eltwise_simple_vload8` | 1.1877 | 1 | 72 | 85.517 | 0.4% | 96.3% |
| rope_q | `rope_opt` | 1.6644 | 1 | 36 | 59.919 | 0.3% | 75.6% |
| fc_o | `dynamic_quantize_gpu_opt` | 1.4643 | 1 | 36 | 52.715 | 0.3% | 292.3% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 1.0456 | 1 | 36 | 37.643 | 0.2% | 29.2% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.7979 | 1 | 36 | 28.725 | 0.1% | 1273.9% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.7951 | 1 | 36 | 28.625 | 0.1% | 1278.4% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.7873 | 1 | 36 | 28.343 | 0.1% | 815.4% |
| pa | `pa_kv_cache_update_ref` | 0.5557 | 1 | 36 | 20.004 | 0.1% | 10340.5% |
| rope_k | `rope_opt` | 0.4076 | 1 | 36 | 14.674 | 0.1% | 84.2% |
| lm_head | `gemm_kernel` | 3.7707 | 1 | 1 | 3.771 | 0.0% | 95.3% |

## Top contributors (sorted by total ms per inference)

### decode
| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | pa 5.01ms (18%) | fc_down 4.64ms (16%) | fc_gate 4.63ms (16%) |
| 2048 | pa 6.84ms (23%) | fc_down 4.64ms (15%) | fc_gate 4.63ms (15%) |
| 4096 | pa 6.15ms (21%) | fc_down 4.64ms (16%) | fc_gate 4.63ms (16%) |
| 8192 | pa 11.23ms (33%) | fc_down 4.64ms (13%) | fc_gate 4.63ms (13%) |

### prefill
| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | pa 298.86ms (43%) | fc_down 85.91ms (12%) | fc_up 78.35ms (11%) |
| 2048 | pa 1138.60ms (59%) | fc_down 171.72ms (9%) | fc_up 156.15ms (8%) |
| 4096 | pa 4438.59ms (75%) | fc_down 342.76ms (6%) | fc_up 301.92ms (5%) |
| 8192 | pa 17519.42ms (86%) | fc_down 683.93ms (3%) | fc_up 569.75ms (3%) |

## Caveats & method
- Each op profiled in its own process via cliloader Device Performance Timing; we use mean kernel time (mode-of-call-counts) per iteration.
- FC weight bytes count INT4 weight + FP16 scale/zp(g=128) + FP16 act + FP16 out.
- PA bytes assume INT8 KV cache (1B/elem) + FP16 Q, FP16 out.
- Decode FC is treated as **memory-bound** (per SKILL: weights read dominates at M=1); prefill FC is **INT8 XMX compute-bound** (S big enough to hit XMX).
- Prefill PA at S≥2048 is compute-bound (FP16 micro-kernel); decode PA is memory-bound.
- swish/multiply/add eltwise are typically fused into matmul/SwiGLU in real inference; they are listed for visibility. swish standalone could not be measured (the GPU plugin fuses Parameter→Swish→Result into a noop activation that the parser excludes).
- lm_head is run only once per token (last position in prefill, every step in decode).
- Target machine: intel@10.239.152.140 (Linux PTL 4Xe).

## Reproduction
```bash
# On the 4Xe Linux host (intel@10.239.152.140):
bash ~/river/roofline_test_utils/run_qwen3_omni_ptl_4xe.sh
# Logs land in ~/river/roofline_results/qwen3_omni/ptl_4xe/. Then locally:
python utils/parse_logs.py outputs/qwen3_omni/logs_ptl_4xe/ outputs/qwen3_omni/kernels_raw_4xe.json
python outputs/qwen3_omni/build_report_4xe.py
```
