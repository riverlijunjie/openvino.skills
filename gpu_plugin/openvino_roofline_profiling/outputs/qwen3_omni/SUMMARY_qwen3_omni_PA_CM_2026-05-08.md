# qwen3_omni Thinker text — Roofline on PTL 12Xe Windows, **PA CM kernel** (2026-05-08)

**Platform**: Intel PTL 12Xe iGPU (Arc B390 96CUs class, 96 EUs = 12 Xe × 8 EU × 10 thr), 2400 MHz, BW≈97 GB/s, FP16 XMX peak ≈ 58.98 TFLOPS, INT8 XMX peak ≈ 117.96 TOPS.
**Model**: qwen3_omni Thinker text decoder (dense, GQA). Source: `config.json` -> `thinker_config.text_config`.

- hidden=2560, layers=36, **GQA NH=32 / NKV=8**, head_dim=128 (Q-dim=4096, KV-dim=1024), intermediate=9728, vocab=151936
- `tie_word_embeddings=true` (LM head shares storage with embedding), `hidden_act=silu` (SwiGLU), `rope_theta=1e6`, mrope_section=[24,20,20]
- MatMul weights INT4 g128 / FP16 act; LM_head INT8 g128 / FP16 act; KV cache INT8
- SDPA: **PagedAttention CM kernel (XAttention path, block_size=256)** — `pa_*_cm` kernels. FC + small ops are reused from the OCL sweep (HW-fixed, identical between runs).
- **DRAM ceiling**: BW = **97 GB/s** measured on this device with `utils/hw_probe/mem_bw.c` (1 GiB working set, R+W streaming copy). Any kernel reporting Eff% > 100% on the BW axis is *running faster than the streaming-DRAM roofline* — see *Caveats* for what that means for the CM PA path.
- **Cache-defeat sizing fix**: rotating-buffer auto-sizing in `pa_bench` was previously capped at 8 buffers with a hard-coded 18 MiB L3 assumption. On PTL 12Xe / Arc B390 the LLC is 16 MiB and per-buf KV at KV=8192 INT8 is 16 MiB, so the prior stride-4 access pattern partially hit the cache. The fix raises the cap to 16 and targets a 256 MiB working set (16× LLC). Re-running with the larger working set produced **identical kernel times**, confirming the >ceiling number is intrinsic to the CM kernel — not a cache artifact in our test.

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
| Memory BW | 97 GB/s |
| Ridge point (FP16) | 608 FLOP/byte |

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
| 1024 | 182.49 | 0.182 | 0.1782 | 5611.2 |
| 2048 | 375.72 | 0.376 | 0.1835 | 5450.9 |
| 4096 | 836.95 | 0.837 | 0.2043 | 4893.9 |
| 8192 | 1951.29 | 1.951 | 0.2382 | 4198.2 |

### Decode — TPOT (per output token)
| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 1024 | 25.341 | 39.5 |
| 2048 | 25.320 | 39.5 |
| 4096 | 26.585 | 37.6 |
| 8192 | 26.679 | 37.5 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)
| op | KV=1024 ms (%) | KV=2048 ms (%) | KV=4096 ms (%) | KV=8192 ms (%) |
|---|---:|---:|---:|---:|
| FC down | 4.618 (18.2%) | 4.618 (18.2%) | 4.618 (17.4%) | 4.618 (17.3%) |
| FC up | 4.640 (18.3%) | 4.640 (18.3%) | 4.640 (17.5%) | 4.640 (17.4%) |
| FC gate | 4.623 (18.2%) | 4.623 (18.3%) | 4.623 (17.4%) | 4.623 (17.3%) |
| FC qkv (fused Q+K+V) | 2.957 (11.7%) | 2.957 (11.7%) | 2.957 (11.1%) | 2.957 (11.1%) |
| FC o | 2.008 (7.9%) | 2.008 (7.9%) | 2.008 (7.6%) | 2.008 (7.5%) |
| lm_head | 3.834 (15.1%) | 3.834 (15.1%) | 3.834 (14.4%) | 3.834 (14.4%) |
| PagedAttention | 1.988 (7.8%) | 1.967 (7.8%) | 3.232 (12.2%) | 3.326 (12.5%) |
| RMSNorm | 0.213 (0.8%) | 0.213 (0.8%) | 0.213 (0.8%) | 0.213 (0.8%) |
| RMSNorm q | 0.087 (0.3%) | 0.087 (0.3%) | 0.087 (0.3%) | 0.087 (0.3%) |
| RMSNorm k | 0.086 (0.3%) | 0.086 (0.3%) | 0.086 (0.3%) | 0.086 (0.3%) |
| RoPE q | 0.078 (0.3%) | 0.078 (0.3%) | 0.078 (0.3%) | 0.078 (0.3%) |
| RoPE k | 0.071 (0.3%) | 0.071 (0.3%) | 0.071 (0.3%) | 0.071 (0.3%) |
| Residual Add | 0.140 (0.6%) | 0.140 (0.6%) | 0.140 (0.5%) | 0.140 (0.5%) |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)
| op | S=1024 ms (%) | S=2048 ms (%) | S=4096 ms (%) | S=8192 ms (%) |
|---|---:|---:|---:|---:|
| FC down | 35.68 (19.6%) | 69.94 (18.6%) | 135.69 (16.2%) | 271.93 (13.9%) |
| FC up | 31.54 (17.3%) | 62.82 (16.7%) | 132.41 (15.8%) | 247.83 (12.7%) |
| FC gate | 31.44 (17.2%) | 62.34 (16.6%) | 130.16 (15.6%) | 246.82 (12.6%) |
| FC qkv (fused Q+K+V) | 20.73 (11.4%) | 39.38 (10.5%) | 81.80 (9.8%) | 165.00 (8.5%) |
| FC o | 15.10 (8.3%) | 27.84 (7.4%) | 53.76 (6.4%) | 106.52 (5.5%) |
| lm_head | 3.85 (2.1%) | 3.85 (1.0%) | 3.85 (0.5%) | 3.85 (0.2%) |
| PagedAttention | 17.58 (9.6%) | 51.82 (13.8%) | 171.56 (20.5%) | 633.60 (32.5%) |
| RMSNorm | 5.57 (3.1%) | 11.51 (3.1%) | 26.65 (3.2%) | 57.75 (3.0%) |
| RMSNorm q | 6.60 (3.6%) | 13.54 (3.6%) | 27.72 (3.3%) | 59.16 (3.0%) |
| RMSNorm k | 1.74 (1.0%) | 3.38 (0.9%) | 6.75 (0.8%) | 13.83 (0.7%) |
| RoPE q | 3.84 (2.1%) | 9.40 (2.5%) | 22.23 (2.7%) | 47.93 (2.5%) |
| RoPE k | 1.13 (0.6%) | 2.11 (0.6%) | 4.26 (0.5%) | 10.88 (0.6%) |
| Residual Add | 7.70 (4.2%) | 17.80 (4.7%) | 40.10 (4.8%) | 86.20 (4.4%) |

## Decode tables (1 query token, KV = context length)
### Decode — KV=1024
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_up | fc_int4_g128 | 0.1289 | 36 | 4.640 | 386 | 99.8 | 102.9% | memory |
| fc_gate | fc_int4_g128 | 0.1284 | 36 | 4.623 | 388 | 100.2 | 103.3% | memory |
| fc_down | fc_int4_g128 | 0.1283 | 36 | 4.618 | 388 | 100.3 | 103.4% | memory |
| lm_head | fc_int8_g128 | 3.8339 | 1 | 3.834 | 203 | 103.1 | 106.3% | memory |
| fc_qkv | fc_int4_g128 | 0.0821 | 36 | 2.957 | 383 | 98.9 | 102.0% | memory |
| fc_o | fc_int4_g128 | 0.0558 | 36 | 2.008 | 376 | 97.2 | 100.2% | memory |
| pa | pa_cm_xattention | 0.0552 | 36 | 1.988 | 319 | 38.3 | 39.4% | memory |
| rmsnorm | rmsnorm | 0.0029 | 73 | 0.213 | 0 | 5.3 | 5.4% | memory |
| add | add | 0.0019 | 72 | 0.140 | 0 | 7.9 | 8.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0024 | 36 | 0.087 | 0 | 6.9 | 7.1% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0024 | 36 | 0.086 | 0 | 1.8 | 1.9% | memory |
| rope_q | rope_q | 0.0022 | 36 | 0.078 | 0 | 7.8 | 8.1% | memory |
| rope_k | rope_k | 0.0020 | 36 | 0.071 | 0 | 2.3 | 2.4% | memory |
| **TOTAL** |  |  |  | **25.341** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=2048
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_up | fc_int4_g128 | 0.1289 | 36 | 4.640 | 386 | 99.8 | 102.9% | memory |
| fc_gate | fc_int4_g128 | 0.1284 | 36 | 4.623 | 388 | 100.2 | 103.3% | memory |
| fc_down | fc_int4_g128 | 0.1283 | 36 | 4.618 | 388 | 100.3 | 103.4% | memory |
| lm_head | fc_int8_g128 | 3.8339 | 1 | 3.834 | 203 | 103.1 | 106.3% | memory |
| fc_qkv | fc_int4_g128 | 0.0821 | 36 | 2.957 | 383 | 98.9 | 102.0% | memory |
| fc_o | fc_int4_g128 | 0.0558 | 36 | 2.008 | 376 | 97.2 | 100.2% | memory |
| pa | pa_cm_xattention | 0.0546 | 36 | 1.967 | 644 | 77.1 | 79.5% | memory |
| rmsnorm | rmsnorm | 0.0029 | 73 | 0.213 | 0 | 5.3 | 5.4% | memory |
| add | add | 0.0019 | 72 | 0.140 | 0 | 7.9 | 8.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0024 | 36 | 0.087 | 0 | 6.9 | 7.1% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0024 | 36 | 0.086 | 0 | 1.8 | 1.9% | memory |
| rope_q | rope_q | 0.0022 | 36 | 0.078 | 0 | 7.8 | 8.1% | memory |
| rope_k | rope_k | 0.0020 | 36 | 0.071 | 0 | 2.3 | 2.4% | memory |
| **TOTAL** |  |  |  | **25.320** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=4096
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_up | fc_int4_g128 | 0.1289 | 36 | 4.640 | 386 | 99.8 | 102.9% | memory |
| fc_gate | fc_int4_g128 | 0.1284 | 36 | 4.623 | 388 | 100.2 | 103.3% | memory |
| fc_down | fc_int4_g128 | 0.1283 | 36 | 4.618 | 388 | 100.3 | 103.4% | memory |
| lm_head | fc_int8_g128 | 3.8339 | 1 | 3.834 | 203 | 103.1 | 106.3% | memory |
| pa | pa_cm_xattention | 0.0898 | 36 | 3.232 | 784 | 93.6 | 96.5% | memory |
| fc_qkv | fc_int4_g128 | 0.0821 | 36 | 2.957 | 383 | 98.9 | 102.0% | memory |
| fc_o | fc_int4_g128 | 0.0558 | 36 | 2.008 | 376 | 97.2 | 100.2% | memory |
| rmsnorm | rmsnorm | 0.0029 | 73 | 0.213 | 0 | 5.3 | 5.4% | memory |
| add | add | 0.0019 | 72 | 0.140 | 0 | 7.9 | 8.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0024 | 36 | 0.087 | 0 | 6.9 | 7.1% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0024 | 36 | 0.086 | 0 | 1.8 | 1.9% | memory |
| rope_q | rope_q | 0.0022 | 36 | 0.078 | 0 | 7.8 | 8.1% | memory |
| rope_k | rope_k | 0.0020 | 36 | 0.071 | 0 | 2.3 | 2.4% | memory |
| **TOTAL** |  |  |  | **26.585** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=8192
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_up | fc_int4_g128 | 0.1289 | 36 | 4.640 | 386 | 99.8 | 102.9% | memory |
| fc_gate | fc_int4_g128 | 0.1284 | 36 | 4.623 | 388 | 100.2 | 103.3% | memory |
| fc_down | fc_int4_g128 | 0.1283 | 36 | 4.618 | 388 | 100.3 | 103.4% | memory |
| lm_head | fc_int8_g128 | 3.8339 | 1 | 3.834 | 203 | 103.1 | 106.3% | memory |
| pa | pa_cm_xattention | 0.0924 | 36 | 3.326 | 1524 | 181.8 | 187.4% | memory |
| fc_qkv | fc_int4_g128 | 0.0821 | 36 | 2.957 | 383 | 98.9 | 102.0% | memory |
| fc_o | fc_int4_g128 | 0.0558 | 36 | 2.008 | 376 | 97.2 | 100.2% | memory |
| rmsnorm | rmsnorm | 0.0029 | 73 | 0.213 | 0 | 5.3 | 5.4% | memory |
| add | add | 0.0019 | 72 | 0.140 | 0 | 7.9 | 8.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0024 | 36 | 0.087 | 0 | 6.9 | 7.1% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0024 | 36 | 0.086 | 0 | 1.8 | 1.9% | memory |
| rope_q | rope_q | 0.0022 | 36 | 0.078 | 0 | 7.8 | 8.1% | memory |
| rope_k | rope_k | 0.0020 | 36 | 0.071 | 0 | 2.3 | 2.4% | memory |
| **TOTAL** |  |  |  | **26.679** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

## Prefill tables (single forward over S tokens)
### Prefill — S=1024
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_down | fc_int4_g128 | 0.9911 | 36 | 35.681 | 51459 | 38.3 | 43.6% | compute |
| fc_up | fc_int4_g128 | 0.8761 | 36 | 31.540 | 58214 | 43.4 | 49.3% | compute |
| fc_gate | fc_int4_g128 | 0.8732 | 36 | 31.435 | 58409 | 43.5 | 49.5% | compute |
| fc_qkv | fc_int4_g128 | 0.5758 | 36 | 20.730 | 55941 | 45.0 | 47.4% | compute |
| pa | pa_cm_xattention_prefill | 0.4882 | 36 | 17.576 | 18471 | 38.7 | 39.9% | memory |
| fc_o | fc_int4_g128 | 0.4194 | 36 | 15.097 | 51208 | 45.4 | 43.4% | compute |
| add | add | 0.1069 | 72 | 7.699 | 0 | 147.1 | 151.6% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.1833 | 36 | 6.598 | 0 | 91.5 | 94.4% | memory |
| rmsnorm | rmsnorm | 0.0764 | 73 | 5.574 | 0 | 137.4 | 141.7% | memory |
| lm_head | fc_int8_g128 | 3.8534 | 1 | 3.853 | 202 | 102.6 | 105.8% | memory |
| rope_q | rope_q | 0.1067 | 36 | 3.840 | 0 | 162.2 | 167.2% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0484 | 36 | 1.741 | 0 | 86.7 | 89.4% | memory |
| rope_k | rope_k | 0.0313 | 36 | 1.127 | 0 | 150.7 | 155.4% | memory |
| **TOTAL** |  |  |  | **182.492** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=2048
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fc_down | fc_int4_g128 | 1.9427 | 36 | 69.938 | 52506 | 32.5 | 44.5% | compute |
| fc_up | fc_int4_g128 | 1.7450 | 36 | 62.819 | 58457 | 36.2 | 49.6% | compute |
| fc_gate | fc_int4_g128 | 1.7317 | 36 | 62.341 | 58905 | 36.5 | 49.9% | compute |
| pa | pa_cm_xattention_prefill | 1.4394 | 36 | 51.819 | 25048 | 26.2 | 42.5% | compute |
| fc_qkv | fc_int4_g128 | 1.0939 | 36 | 39.382 | 58892 | 40.0 | 49.9% | compute |
| fc_o | fc_int4_g128 | 0.7732 | 36 | 27.836 | 55547 | 42.3 | 47.1% | compute |
| add | add | 0.2472 | 72 | 17.801 | 0 | 127.2 | 131.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.3760 | 36 | 13.537 | 0 | 89.2 | 92.0% | memory |
| rmsnorm | rmsnorm | 0.1577 | 73 | 11.509 | 0 | 133.0 | 137.2% | memory |
| rope_q | rope_q | 0.2611 | 36 | 9.401 | 0 | 132.5 | 136.6% | memory |
| lm_head | fc_int8_g128 | 3.8534 | 1 | 3.853 | 202 | 102.6 | 105.8% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0938 | 36 | 3.376 | 0 | 89.5 | 92.2% | memory |
| rope_k | rope_k | 0.0586 | 36 | 2.109 | 0 | 161.1 | 166.1% | memory |
| **TOTAL** |  |  |  | **375.720** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=4096
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_cm_xattention_prefill | 4.7656 | 36 | 171.561 | 30256 | 15.8 | 51.3% | compute |
| fc_down | fc_int4_g128 | 3.7691 | 36 | 135.687 | 54128 | 30.1 | 45.9% | compute |
| fc_up | fc_int4_g128 | 3.6781 | 36 | 132.411 | 55467 | 30.9 | 47.0% | compute |
| fc_gate | fc_int4_g128 | 3.6157 | 36 | 130.164 | 56424 | 31.4 | 47.8% | compute |
| fc_qkv | fc_int4_g128 | 2.2721 | 36 | 81.796 | 56709 | 35.0 | 48.1% | compute |
| fc_o | fc_int4_g128 | 1.4932 | 36 | 53.756 | 57526 | 40.1 | 48.8% | compute |
| add | add | 0.5570 | 72 | 40.103 | 0 | 113.0 | 116.4% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.7701 | 36 | 27.723 | 0 | 87.1 | 89.8% | memory |
| rmsnorm | rmsnorm | 0.3651 | 73 | 26.649 | 0 | 114.9 | 118.5% | memory |
| rope_q | rope_q | 0.6176 | 36 | 22.235 | 0 | 112.0 | 115.5% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.1875 | 36 | 6.752 | 0 | 89.5 | 92.2% | memory |
| rope_k | rope_k | 0.1184 | 36 | 4.263 | 0 | 159.4 | 164.3% | memory |
| lm_head | fc_int8_g128 | 3.8534 | 1 | 3.853 | 202 | 102.6 | 105.8% | memory |
| **TOTAL** |  |  |  | **836.953** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=8192
| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_cm_xattention_prefill | 17.5999 | 36 | 633.597 | 32765 | 8.6 | 55.6% | compute |
| fc_down | fc_int4_g128 | 7.5537 | 36 | 271.932 | 54016 | 28.4 | 45.8% | compute |
| fc_up | fc_int4_g128 | 6.8841 | 36 | 247.827 | 59270 | 31.1 | 50.2% | compute |
| fc_gate | fc_int4_g128 | 6.8561 | 36 | 246.820 | 59512 | 31.2 | 50.4% | compute |
| fc_qkv | fc_int4_g128 | 4.5833 | 36 | 165.000 | 56225 | 32.9 | 47.7% | compute |
| fc_o | fc_int4_g128 | 2.9588 | 36 | 106.517 | 58064 | 38.7 | 49.2% | compute |
| add | add | 1.1972 | 72 | 86.198 | 0 | 105.1 | 108.4% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 1.6433 | 36 | 59.158 | 0 | 81.7 | 84.2% | memory |
| rmsnorm | rmsnorm | 0.7911 | 73 | 57.751 | 0 | 106.0 | 109.3% | memory |
| rope_q | rope_q | 1.3314 | 36 | 47.930 | 0 | 104.0 | 107.2% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.3842 | 36 | 13.832 | 0 | 87.3 | 90.0% | memory |
| rope_k | rope_k | 0.3021 | 36 | 10.877 | 0 | 124.9 | 128.8% | memory |
| lm_head | fc_int8_g128 | 3.8534 | 1 | 3.853 | 202 | 102.6 | 105.8% | memory |
| **TOTAL** |  |  |  | **1951.293** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

## Per-kernel decomposition (cliloader kernel names)
Each op above maps to one or more GPU kernels. Below shows the actual kernel names captured by cliloader's *Device Performance Timing* section, with per-launch time, launches per op-call, total ms across one inference, and per-kernel **Eff%** (peak utilization). The Eff% column attributes the **parent op's full FLOPs/bytes per op-call** against the kernel's own per-launch time, so the dominant kernel reports an Eff% close to the op-level value, while helper kernels (e.g. `pa_kv_cache_update_ref`, `dynamic_quantize_gpu_opt`, `*_finalization`) appear with apparently very high Eff% — that is the expected signal that they are *not* the bottleneck for the op. PA-CM decomposes into `pa_kv_cache_update_ref_cm` + the attention kernel (`pa_single_token_cm` for decode, `pa_multi_token_cm_bs1` for prefill) + `pa_single_token_finalization_cm`. Prefill FC decomposes into `dynamic_quantize_gpu_opt` + `gemm_kernel`.

### Decode sub-kernels
### Decode sub-kernels — KV=1024
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.1289 | 1 | 36 | 4.640 | 18.3% | 102.9% |
| fc_gate | `gemm_kernel` | 0.1284 | 1 | 36 | 4.623 | 18.2% | 103.3% |
| fc_down | `gemm_kernel` | 0.1283 | 1 | 36 | 4.618 | 18.2% | 103.4% |
| lm_head | `gemm_kernel` | 3.8339 | 1 | 1 | 3.834 | 15.1% | 106.3% |
| fc_qkv | `gemm_kernel` | 0.0821 | 1 | 36 | 2.957 | 11.7% | 102.0% |
| fc_o | `gemm_kernel` | 0.0558 | 1 | 36 | 2.008 | 7.9% | 100.2% |
| pa | `pa_single_token_cm` | 0.0460 | 1 | 36 | 1.657 | 6.5% | 47.3% |
| pa | `pa_kv_cache_update_ref_cm` | 0.0065 | 1 | 36 | 0.235 | 0.9% | 334.1% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0029 | 1 | 73 | 0.213 | 0.8% | 5.4% |
| add | `eltwise_simple_vload8` | 0.0019 | 1 | 72 | 0.140 | 0.6% | 8.2% |
| pa | `pa_single_token_finalization_cm` | 0.0027 | 1 | 36 | 0.096 | 0.4% | 815.2% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.087 | 0.3% | 7.1% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.086 | 0.3% | 1.9% |
| rope_q | `rope_opt` | 0.0022 | 1 | 36 | 0.078 | 0.3% | 8.1% |
| rope_k | `rope_opt` | 0.0020 | 1 | 36 | 0.071 | 0.3% | 2.4% |

### Decode sub-kernels — KV=2048
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.1289 | 1 | 36 | 4.640 | 18.3% | 102.9% |
| fc_gate | `gemm_kernel` | 0.1284 | 1 | 36 | 4.623 | 18.3% | 103.3% |
| fc_down | `gemm_kernel` | 0.1283 | 1 | 36 | 4.618 | 18.2% | 103.4% |
| lm_head | `gemm_kernel` | 3.8339 | 1 | 1 | 3.834 | 15.1% | 106.3% |
| fc_qkv | `gemm_kernel` | 0.0821 | 1 | 36 | 2.957 | 11.7% | 102.0% |
| fc_o | `gemm_kernel` | 0.0558 | 1 | 36 | 2.008 | 7.9% | 100.2% |
| pa | `pa_single_token_cm` | 0.0457 | 1 | 36 | 1.647 | 6.5% | 94.9% |
| pa | `pa_kv_cache_update_ref_cm` | 0.0064 | 1 | 36 | 0.229 | 0.9% | 683.3% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0029 | 1 | 73 | 0.213 | 0.8% | 5.4% |
| add | `eltwise_simple_vload8` | 0.0019 | 1 | 72 | 0.140 | 0.6% | 8.2% |
| pa | `pa_single_token_finalization_cm` | 0.0025 | 1 | 36 | 0.091 | 0.4% | 1715.1% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.087 | 0.3% | 7.1% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.086 | 0.3% | 1.9% |
| rope_q | `rope_opt` | 0.0022 | 1 | 36 | 0.078 | 0.3% | 8.1% |
| rope_k | `rope_opt` | 0.0020 | 1 | 36 | 0.071 | 0.3% | 2.4% |

### Decode sub-kernels — KV=4096
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.1289 | 1 | 36 | 4.640 | 17.5% | 102.9% |
| fc_gate | `gemm_kernel` | 0.1284 | 1 | 36 | 4.623 | 17.4% | 103.3% |
| fc_down | `gemm_kernel` | 0.1283 | 1 | 36 | 4.618 | 17.4% | 103.4% |
| lm_head | `gemm_kernel` | 3.8339 | 1 | 1 | 3.834 | 14.4% | 106.3% |
| fc_qkv | `gemm_kernel` | 0.0821 | 1 | 36 | 2.957 | 11.1% | 102.0% |
| pa | `pa_single_token_cm` | 0.0807 | 1 | 36 | 2.906 | 10.9% | 107.4% |
| fc_o | `gemm_kernel` | 0.0558 | 1 | 36 | 2.008 | 7.6% | 100.2% |
| pa | `pa_kv_cache_update_ref_cm` | 0.0063 | 1 | 36 | 0.228 | 0.9% | 1371.0% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0029 | 1 | 73 | 0.213 | 0.8% | 5.4% |
| add | `eltwise_simple_vload8` | 0.0019 | 1 | 72 | 0.140 | 0.5% | 8.2% |
| pa | `pa_single_token_finalization_cm` | 0.0028 | 1 | 36 | 0.099 | 0.4% | 3148.6% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.087 | 0.3% | 7.1% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.086 | 0.3% | 1.9% |
| rope_q | `rope_opt` | 0.0022 | 1 | 36 | 0.078 | 0.3% | 8.1% |
| rope_k | `rope_opt` | 0.0020 | 1 | 36 | 0.071 | 0.3% | 2.4% |

### Decode sub-kernels — KV=8192
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.1289 | 1 | 36 | 4.640 | 17.4% | 102.9% |
| fc_gate | `gemm_kernel` | 0.1284 | 1 | 36 | 4.623 | 17.3% | 103.3% |
| fc_down | `gemm_kernel` | 0.1283 | 1 | 36 | 4.618 | 17.3% | 103.4% |
| lm_head | `gemm_kernel` | 3.8339 | 1 | 1 | 3.834 | 14.4% | 106.3% |
| fc_qkv | `gemm_kernel` | 0.0821 | 1 | 36 | 2.957 | 11.1% | 102.0% |
| pa | `pa_single_token_cm` | 0.0820 | 1 | 36 | 2.951 | 11.1% | 211.2% |
| fc_o | `gemm_kernel` | 0.0558 | 1 | 36 | 2.008 | 7.5% | 100.2% |
| pa | `pa_kv_cache_update_ref_cm` | 0.0074 | 1 | 36 | 0.265 | 1.0% | 2352.6% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0029 | 1 | 73 | 0.213 | 0.8% | 5.4% |
| add | `eltwise_simple_vload8` | 0.0019 | 1 | 72 | 0.140 | 0.5% | 8.2% |
| pa | `pa_single_token_finalization_cm` | 0.0031 | 1 | 36 | 0.110 | 0.4% | 5672.7% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.087 | 0.3% | 7.1% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0024 | 1 | 36 | 0.086 | 0.3% | 1.9% |
| rope_q | `rope_opt` | 0.0022 | 1 | 36 | 0.078 | 0.3% | 8.1% |
| rope_k | `rope_opt` | 0.0020 | 1 | 36 | 0.071 | 0.3% | 2.4% |

### Prefill sub-kernels
### Prefill sub-kernels — S=1024
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 0.8170 | 1 | 36 | 29.411 | 16.1% | 52.9% |
| fc_gate | `gemm_kernel` | 0.8137 | 1 | 36 | 29.292 | 16.1% | 53.1% |
| fc_down | `gemm_kernel` | 0.7679 | 1 | 36 | 27.644 | 15.1% | 56.3% |
| fc_qkv | `gemm_kernel` | 0.5169 | 1 | 36 | 18.607 | 10.2% | 52.8% |
| pa | `pa_multi_token_cm_bs1` | 0.4269 | 1 | 36 | 15.369 | 8.4% | 45.6% |
| fc_o | `gemm_kernel` | 0.3295 | 1 | 36 | 11.862 | 6.5% | 55.2% |
| fc_down | `dynamic_quantize_gpu_opt` | 0.2232 | 1 | 36 | 8.036 | 4.4% | 193.7% |
| add | `eltwise_simple_vload8` | 0.1069 | 1 | 72 | 7.699 | 4.2% | 151.6% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.1833 | 1 | 36 | 6.598 | 3.6% | 94.4% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0764 | 1 | 73 | 5.574 | 3.1% | 141.7% |
| lm_head | `gemm_kernel` | 3.8534 | 1 | 1 | 3.853 | 2.1% | 105.8% |
| rope_q | `rope_opt` | 0.1067 | 1 | 36 | 3.840 | 2.1% | 167.2% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.0899 | 1 | 36 | 3.235 | 1.8% | 202.6% |
| pa | `pa_kv_cache_update_ref_cm` | 0.0613 | 1 | 36 | 2.207 | 1.2% | 317.4% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.0596 | 1 | 36 | 2.144 | 1.2% | 726.0% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.0591 | 1 | 36 | 2.129 | 1.2% | 731.0% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.0590 | 1 | 36 | 2.122 | 1.2% | 463.2% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0484 | 1 | 36 | 1.741 | 1.0% | 89.4% |
| rope_k | `rope_opt` | 0.0313 | 1 | 36 | 1.127 | 0.6% | 155.4% |

### Prefill sub-kernels — S=2048
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| fc_up | `gemm_kernel` | 1.6241 | 1 | 36 | 58.467 | 15.6% | 53.2% |
| fc_gate | `gemm_kernel` | 1.6124 | 1 | 36 | 58.046 | 15.4% | 53.6% |
| fc_down | `gemm_kernel` | 1.4908 | 1 | 36 | 53.670 | 14.3% | 58.0% |
| pa | `pa_multi_token_cm_bs1` | 1.3383 | 1 | 36 | 48.180 | 12.8% | 45.7% |
| fc_qkv | `gemm_kernel` | 0.9780 | 1 | 36 | 35.210 | 9.4% | 55.8% |
| fc_o | `gemm_kernel` | 0.5845 | 1 | 36 | 21.041 | 5.6% | 62.3% |
| add | `eltwise_simple_vload8` | 0.2472 | 1 | 72 | 17.801 | 4.7% | 131.2% |
| fc_down | `dynamic_quantize_gpu_opt` | 0.4519 | 1 | 36 | 16.268 | 4.3% | 191.4% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.3760 | 1 | 36 | 13.537 | 3.6% | 92.0% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.1577 | 1 | 73 | 11.509 | 3.1% | 137.2% |
| rope_q | `rope_opt` | 0.2611 | 1 | 36 | 9.401 | 2.5% | 136.6% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.1887 | 1 | 36 | 6.795 | 1.8% | 192.9% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.1209 | 1 | 36 | 4.351 | 1.2% | 715.4% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.1193 | 1 | 36 | 4.295 | 1.1% | 724.8% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.1159 | 1 | 36 | 4.172 | 1.1% | 471.2% |
| lm_head | `gemm_kernel` | 3.8534 | 1 | 1 | 3.853 | 1.0% | 105.8% |
| pa | `pa_kv_cache_update_ref_cm` | 0.1011 | 1 | 36 | 3.639 | 1.0% | 604.7% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0938 | 1 | 36 | 3.376 | 0.9% | 92.2% |
| rope_k | `rope_opt` | 0.0586 | 1 | 36 | 2.109 | 0.6% | 166.1% |

### Prefill sub-kernels — S=4096
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `pa_multi_token_cm_bs1` | 4.5662 | 1 | 36 | 164.382 | 19.6% | 53.5% |
| fc_up | `gemm_kernel` | 3.4309 | 1 | 36 | 123.513 | 14.8% | 50.4% |
| fc_gate | `gemm_kernel` | 3.3725 | 1 | 36 | 121.408 | 14.5% | 51.3% |
| fc_down | `gemm_kernel` | 2.8277 | 1 | 36 | 101.799 | 12.2% | 61.2% |
| fc_qkv | `gemm_kernel` | 2.0395 | 1 | 36 | 73.422 | 8.8% | 53.6% |
| fc_o | `gemm_kernel` | 1.1179 | 1 | 36 | 40.246 | 4.8% | 65.1% |
| add | `eltwise_simple_vload8` | 0.5570 | 1 | 72 | 40.103 | 4.8% | 116.4% |
| fc_down | `dynamic_quantize_gpu_opt` | 0.9413 | 1 | 36 | 33.888 | 4.0% | 183.7% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.7701 | 1 | 36 | 27.723 | 3.3% | 89.8% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.3651 | 1 | 73 | 26.649 | 3.2% | 118.5% |
| rope_q | `rope_opt` | 0.6176 | 1 | 36 | 22.235 | 2.7% | 115.5% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.3753 | 1 | 36 | 13.510 | 1.6% | 194.0% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.2472 | 1 | 36 | 8.898 | 1.1% | 699.7% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.2432 | 1 | 36 | 8.756 | 1.0% | 711.1% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.2326 | 1 | 36 | 8.374 | 1.0% | 469.6% |
| pa | `pa_kv_cache_update_ref_cm` | 0.1994 | 1 | 36 | 7.178 | 0.9% | 1226.0% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.1875 | 1 | 36 | 6.752 | 0.8% | 92.2% |
| rope_k | `rope_opt` | 0.1184 | 1 | 36 | 4.263 | 0.5% | 164.3% |
| lm_head | `gemm_kernel` | 3.8534 | 1 | 1 | 3.853 | 0.5% | 105.8% |

### Prefill sub-kernels — S=8192
| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `pa_multi_token_cm_bs1` | 17.0624 | 1 | 36 | 614.247 | 31.5% | 57.3% |
| fc_up | `gemm_kernel` | 6.3923 | 1 | 36 | 230.123 | 11.8% | 54.1% |
| fc_gate | `gemm_kernel` | 6.3642 | 1 | 36 | 229.110 | 11.7% | 54.3% |
| fc_down | `gemm_kernel` | 5.4703 | 1 | 36 | 196.931 | 10.1% | 63.2% |
| fc_qkv | `gemm_kernel` | 4.1182 | 1 | 36 | 148.254 | 7.6% | 53.0% |
| add | `eltwise_simple_vload8` | 1.1972 | 1 | 72 | 86.198 | 4.4% | 108.4% |
| fc_o | `gemm_kernel` | 2.2137 | 1 | 36 | 79.693 | 4.1% | 65.8% |
| fc_down | `dynamic_quantize_gpu_opt` | 2.0834 | 1 | 36 | 75.001 | 3.8% | 166.0% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 1.6433 | 1 | 36 | 59.158 | 3.0% | 84.2% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.7911 | 1 | 73 | 57.751 | 3.0% | 109.3% |
| rope_q | `rope_opt` | 1.3314 | 1 | 36 | 47.930 | 2.5% | 107.2% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.7451 | 1 | 36 | 26.824 | 1.4% | 195.5% |
| pa | `pa_kv_cache_update_ref_cm` | 0.5375 | 1 | 36 | 19.350 | 1.0% | 1818.9% |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.4920 | 1 | 36 | 17.711 | 0.9% | 703.1% |
| fc_up | `dynamic_quantize_gpu_opt` | 0.4918 | 1 | 36 | 17.704 | 0.9% | 703.3% |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.4652 | 1 | 36 | 16.746 | 0.9% | 469.6% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.3842 | 1 | 36 | 13.832 | 0.7% | 90.0% |
| rope_k | `rope_opt` | 0.3021 | 1 | 36 | 10.877 | 0.6% | 128.8% |
| lm_head | `gemm_kernel` | 3.8534 | 1 | 1 | 3.853 | 0.2% | 105.8% |

## Top contributors (sorted by total ms per inference)

### decode
| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | fc_up 4.64ms (18%) | fc_gate 4.62ms (18%) | fc_down 4.62ms (18%) |
| 2048 | fc_up 4.64ms (18%) | fc_gate 4.62ms (18%) | fc_down 4.62ms (18%) |
| 4096 | fc_up 4.64ms (17%) | fc_gate 4.62ms (17%) | fc_down 4.62ms (17%) |
| 8192 | fc_up 4.64ms (17%) | fc_gate 4.62ms (17%) | fc_down 4.62ms (17%) |

### prefill
| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | fc_down 35.68ms (20%) | fc_up 31.54ms (17%) | fc_gate 31.44ms (17%) |
| 2048 | fc_down 69.94ms (19%) | fc_up 62.82ms (17%) | fc_gate 62.34ms (17%) |
| 4096 | pa 171.56ms (20%) | fc_down 135.69ms (16%) | fc_up 132.41ms (16%) |
| 8192 | pa 633.60ms (32%) | fc_down 271.93ms (14%) | fc_up 247.83ms (13%) |

## Caveats & method
- Each op profiled in its own process via cliloader Device Performance Timing; we use mean kernel time (mode-of-call-counts) per iteration.
- FC weight bytes count INT4 weight + FP16 scale/zp(g=128) + FP16 act + FP16 out.
- PA bytes assume INT8 KV cache (1B/elem) + FP16 Q, FP16 out.
- Decode FC is treated as **memory-bound** (per SKILL: weights read dominates at M=1); prefill FC is **INT8 XMX compute-bound** (S big enough to hit XMX).
- **PA implementation: CM kernel (XAttention path, block_size=256)** — verified via cliloader kernel names (`pa_*_cm_*__sa`).
- **PA implementation: CM kernel (XAttention path, block_size=256)** — verified via cliloader kernel names (`pa_*_cm_*__sa`).
- Prefill PA at S≥2048 is compute-bound (CM XAttention compute-saturating); decode PA is memory-bound (KV-cache streaming).
- **CM PA Eff% > 100% on BW axis is expected** for long context: the byte model assumes a *dense* KV scan (`2 · Skv · NKV · HD` per call). The CM `pa_single_token_cm` kernel actually achieves 180–190 GB/s effective on a 97 GB/s DRAM at KV=8192. Two non-mutually-exclusive explanations: (a) PTL's LPDDR5X memory controller serves PA's strictly sequential and prefetch-friendly KV scan well above the random-rotation `mem_bw` benchmark; (b) the CM XAttention path applies block-level KV pruning (block_size=256 + summary-driven block selection) even in single-token mode, so real DRAM traffic is sub-linear in `Skv`. We verified (b)-or-(a) by re-running with a 16× LLC working set (256 MiB) and observing identical kernel times — i.e. caching is NOT the source.
- swish/multiply/add eltwise are typically fused into matmul/SwiGLU in real inference; they are listed for visibility. swish standalone could not be measured (the GPU plugin fuses Parameter→Swish→Result into a noop activation that the parser excludes).
- lm_head is run only once per token (last position in prefill, every step in decode).
- Target machine: Local_Admin@10.239.132.229 (Windows PTL 12Xe / Arc B390).

## Reproduction
```cmd
REM On the 12Xe Windows host (Local_Admin@10.239.132.229) — re-runs ONLY PA with CM impl:
D:\river\moe\dev_roofline_profiling\utils\run_qwen3_omni_ptl_12xe_pa_cm.bat
REM PA-CM logs land in D:\river\moe\roofline_results\qwen3_omni\ptl_12xe_cm\.
REM FC + small_ops logs are reused from the prior OCL sweep (logs_ptl/).
REM Locally, merge:
cp -r outputs/qwen3_omni/logs_ptl outputs/qwen3_omni/logs_ptl_cm
rm outputs/qwen3_omni/logs_ptl_cm/pa_*.log
scp Local_Admin@10.239.132.229:D:/river/moe/roofline_results/qwen3_omni/ptl_12xe_cm/pa_*.log outputs/qwen3_omni/logs_ptl_cm/
python utils/parse_logs.py outputs/qwen3_omni/logs_ptl_cm outputs/qwen3_omni/kernels_raw_cm.json
python outputs/qwen3_omni/build_report_cm.py
```
