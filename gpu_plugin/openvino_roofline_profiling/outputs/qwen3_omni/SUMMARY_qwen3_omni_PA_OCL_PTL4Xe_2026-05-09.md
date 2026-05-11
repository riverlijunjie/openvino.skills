# Qwen3-Omni Thinker text — Roofline on PTL 4Xe Linux (2026-05-09)

**Platform**: Intel PTL 4Xe iGPU (32 EUs = 4 Xe × 8 EU × 10 thr), 2400 MHz, BW ≈ 97.0 GB/s (measured), FP16 XMX peak ≈ 19.66 TFLOPS, INT8 XMX peak ≈ 39.32 TOPS.
**Model**: Qwen3-Omni Thinker text decoder (dense, GQA). Source: `config.json` → `thinker_config.text_config`.

- hidden=2560, layers=36, **GQA NH=32 / NKV=8**, head_dim=128 (Q-dim=4096, KV-dim=1024), intermediate=9728, vocab=151936
- `tie_word_embeddings=true` (LM head shares storage with embedding), `hidden_act=silu` (SwiGLU), `rope_theta=1e6`, mrope_section=[24,20,20]
- MatMul weights INT4 g128 / FP16 act; LM_head INT8 g128 / FP16 act; KV cache INT8
- SDPA: PagedAttention OpenCL + micro_kernel

## Data sources

| Data category | Source | Method |
|---|---|---|
| PA decode (all KV) | **Measured** on PTL 4Xe | `pa_bench` + cliloader |
| PA prefill (all S) | **Measured** on PTL 4Xe | `pa_bench` + cliloader |
| FC decode (all ops) | Estimated from PTL 12Xe | Memory-bound: latency × BW_ratio (110/97 = 1.134) |
| FC prefill (all ops) | Estimated from PTL 12Xe | Compute-bound: gemm × 3.0 (XMX ratio) + dq × 1.134 (BW ratio) |
| Small ops (RMSNorm, RoPE, Add) | Estimated from PTL 12Xe | Memory-bound: latency × 1.134 |
| LM_head | Estimated from PTL 12Xe | Memory-bound: latency × 1.134 |

> **Caveat**: Only PA kernels were measured on this 4Xe machine. All other ops are estimates
> scaled from PTL 12Xe Windows measurements (SUMMARY_qwen3_omni_2026-05-08.md).
> BW efficiency % is preserved for memory-bound ops; INT8 XMX efficiency shifts slightly
> because dynamic-quantize overhead fraction differs.

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
| FC_QKV (fused Q+K+V proj) | 2560 × 6144 | INT4 g128 + FP16 scales | 8.11 MB | 36 | 292.0 MB |
| FC_O (attention output) | 4096 × 2560 | INT4 g128 + FP16 scales | 5.41 MB | 36 | 194.6 MB |
| FC_Gate (SwiGLU gate) | 2560 × 9728 | INT4 g128 + FP16 scales | 12.84 MB | 36 | 462.3 MB |
| FC_Up (SwiGLU up) | 2560 × 9728 | INT4 g128 + FP16 scales | 12.84 MB | 36 | 462.3 MB |
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
| FP16 XMX peak | 19.66 TFLOPS |
| INT8 XMX peak | 39.32 TOPS |
| Memory BW | 97.0 GB/s (measured) |
| Ridge point (FP16) | 202.7 FLOP/byte |

## Graph fusion notes

| Bench row | Real graph behaviour | Fused into | Standalone kernel in graph? |
|---|---|---|---|
| `multiply` | `silu(gate(x)) ⊙ up(x)` of SwiGLU MLP | SwiGLU primitive | No — bench-only |
| `add` | Residual adds per layer | not fused (separate `eltwise`) | Yes |
| `rmsnorm` | Pre-attention + pre-MLP + final RMSNorm | single `RMS` primitive | Yes |
| `sdpa_micro__prefill` | PA prefill attention (EU ALU, causal mask) | PA primitive | Yes |
| `pa_kv_cache_update_ref` | KV cache write | PA primitive | Yes |

> **Note on PA prefill**: The `sdpa_micro__prefill` kernel runs on **EU ALU only** (not XMX/DPAS).
> Disassembly confirms only 2 dummy `dpas.8x1` instructions (systolic probes). Root cause: ngen
> GEMM microkernel catalog has no entries for `HW::Xe3` — PTL 4Xe (ip_version arch=30) maps to
> `GenericXe3` → `Core::Xe3`, which has no matching `HWTag` in the strategy catalog.

## Token latency summary

### Prefill — TTFT and per-token amortized

| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|
| 1024 | 725.48 | 0.725 | 0.7085 | 1411.5 |
| 2048 | 1931.93 | 1.932 | 0.9433 | 1060.1 |
| 4096 | 6045.96 | 6.046 | 1.4761 | 677.5 |
| 8192 | 20684.71 | 20.685 | 2.5250 | 396.0 |

### Decode — TPOT (per output token)

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 1024 | 32.876 | 30.4 |
| 2048 | 33.697 | 29.7 |
| 4096 | 32.721 | 30.6 |
| 8192 | 37.765 | 26.5 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)

| op | KV=1024 ms (%) | KV=2048 ms (%) | KV=4096 ms (%) | KV=8192 ms (%) |
|---|---: | ---: | ---: | ---:|
| pa | 6.394 (19.4%) | 7.214 (21.4%) | 6.239 (19.1%) | 11.282 (29.9%) |
| fc_up | 5.262 (16.0%) | 5.262 (15.6%) | 5.262 (16.1%) | 5.262 (13.9%) |
| fc_gate | 5.242 (15.9%) | 5.242 (15.6%) | 5.242 (16.0%) | 5.242 (13.9%) |
| fc_down | 5.238 (15.9%) | 5.238 (15.5%) | 5.238 (16.0%) | 5.238 (13.9%) |
| lm_head | 4.348 (13.2%) | 4.348 (12.9%) | 4.348 (13.3%) | 4.348 (11.5%) |
| fc_qkv | 3.352 (10.2%) | 3.352 (9.9%) | 3.352 (10.2%) | 3.352 (8.9%) |
| fc_o | 2.278 (6.9%) | 2.278 (6.8%) | 2.278 (7.0%) | 2.278 (6.0%) |
| rmsnorm | 0.240 (0.7%) | 0.240 (0.7%) | 0.240 (0.7%) | 0.240 (0.6%) |
| add | 0.155 (0.5%) | 0.155 (0.5%) | 0.155 (0.5%) | 0.155 (0.4%) |
| rmsnorm3d_q | 0.098 (0.3%) | 0.098 (0.3%) | 0.098 (0.3%) | 0.098 (0.3%) |
| rmsnorm3d_k | 0.098 (0.3%) | 0.098 (0.3%) | 0.098 (0.3%) | 0.098 (0.3%) |
| rope_q | 0.090 (0.3%) | 0.090 (0.3%) | 0.090 (0.3%) | 0.090 (0.2%) |
| rope_k | 0.082 (0.2%) | 0.082 (0.2%) | 0.082 (0.2%) | 0.082 (0.2%) |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)

| op | S=1024 ms (%) | S=2048 ms (%) | S=4096 ms (%) | S=8192 ms (%) |
|---|---: | ---: | ---: | ---:|
| pa | 320.47 (44.2%) | 1142.10 (59.1%) | 4432.32 (73.3%) | 17540.68 (84.8%) |
| fc_up | 90.65 (12.5%) | 180.34 (9.3%) | 380.63 (6.3%) | 710.45 (3.4%) |
| fc_gate | 90.31 (12.4%) | 179.01 (9.3%) | 374.16 (6.2%) | 707.42 (3.4%) |
| fc_down | 92.05 (12.7%) | 179.46 (9.3%) | 343.82 (5.7%) | 675.85 (3.3%) |
| fc_qkv | 58.23 (8.0%) | 110.36 (5.7%) | 229.76 (3.8%) | 463.76 (2.2%) |
| fc_o | 39.26 (5.4%) | 70.83 (3.7%) | 136.05 (2.3%) | 269.50 (1.3%) |
| add | 8.73 (1.2%) | 20.18 (1.0%) | 45.48 (0.8%) | 97.75 (0.5%) |
| rmsnorm3d_q | 7.48 (1.0%) | 15.35 (0.8%) | 31.44 (0.5%) | 67.09 (0.3%) |
| rmsnorm | 6.32 (0.9%) | 13.05 (0.7%) | 30.22 (0.5%) | 65.49 (0.3%) |
| rope_q | 4.36 (0.6%) | 10.66 (0.6%) | 25.21 (0.4%) | 54.35 (0.3%) |
| rmsnorm3d_k | 1.98 (0.3%) | 3.83 (0.2%) | 7.65 (0.1%) | 15.68 (0.1%) |
| rope_k | 1.28 (0.2%) | 2.39 (0.1%) | 4.83 (0.1%) | 12.33 (0.1%) |
| lm_head | 4.37 (0.6%) | 4.37 (0.2%) | 4.37 (0.1%) | 4.37 (0.0%) |

## Decode tables (1 query token, KV = context length)

### Decode — KV=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro | 0.1776 | 36 | 6.394 | 101 | 12.6 | 13.0% | memory |
| fc_up | fc_int4_g128 | 0.1462 | 36 | 5.262 | 340 | 88.0 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1456 | 36 | 5.242 | 342 | 88.4 | 91.1% | memory |
| fc_down | fc_int4_g128 | 0.1455 | 36 | 5.238 | 342 | 88.4 | 91.2% | memory |
| lm_head | fc_int8_g128 | 4.3477 | 1 | 4.348 | 179 | 90.9 | 93.7% | memory |
| fc_qkv | fc_int4_g128 | 0.0931 | 36 | 3.352 | 338 | 87.2 | 90.0% | memory |
| fc_o | fc_int4_g128 | 0.0633 | 36 | 2.278 | 332 | 85.7 | 88.3% | memory |
| rmsnorm | rmsnorm | 0.0033 | 73 | 0.240 | 0 | 4.7 | 4.8% | memory |
| add | add | 0.0022 | 72 | 0.155 | 0 | 7.0 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0027 | 36 | 0.098 | 0 | 6.1 | 6.3% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0027 | 36 | 0.098 | 0 | 1.6 | 1.7% | memory |
| rope_q | rope_q | 0.0025 | 36 | 0.090 | 0 | 6.9 | 7.1% | memory |
| rope_k | rope_k | 0.0023 | 36 | 0.082 | 0 | 2.0 | 2.1% | memory |
| **TOTAL** |  |  |  | **32.876** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro | 0.2004 | 36 | 7.214 | 174 | 21.8 | 22.5% | memory |
| fc_up | fc_int4_g128 | 0.1462 | 36 | 5.262 | 340 | 88.0 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1456 | 36 | 5.242 | 342 | 88.4 | 91.1% | memory |
| fc_down | fc_int4_g128 | 0.1455 | 36 | 5.238 | 342 | 88.4 | 91.2% | memory |
| lm_head | fc_int8_g128 | 4.3477 | 1 | 4.348 | 179 | 90.9 | 93.7% | memory |
| fc_qkv | fc_int4_g128 | 0.0931 | 36 | 3.352 | 338 | 87.2 | 90.0% | memory |
| fc_o | fc_int4_g128 | 0.0633 | 36 | 2.278 | 332 | 85.7 | 88.3% | memory |
| rmsnorm | rmsnorm | 0.0033 | 73 | 0.240 | 0 | 4.7 | 4.8% | memory |
| add | add | 0.0022 | 72 | 0.155 | 0 | 7.0 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0027 | 36 | 0.098 | 0 | 6.1 | 6.3% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0027 | 36 | 0.098 | 0 | 1.6 | 1.7% | memory |
| rope_q | rope_q | 0.0025 | 36 | 0.090 | 0 | 6.9 | 7.1% | memory |
| rope_k | rope_k | 0.0023 | 36 | 0.082 | 0 | 2.0 | 2.1% | memory |
| **TOTAL** |  |  |  | **33.697** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro | 0.1733 | 36 | 6.239 | 415 | 51.8 | 53.4% | memory |
| fc_up | fc_int4_g128 | 0.1462 | 36 | 5.262 | 340 | 88.0 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1456 | 36 | 5.242 | 342 | 88.4 | 91.1% | memory |
| fc_down | fc_int4_g128 | 0.1455 | 36 | 5.238 | 342 | 88.4 | 91.2% | memory |
| lm_head | fc_int8_g128 | 4.3477 | 1 | 4.348 | 179 | 90.9 | 93.7% | memory |
| fc_qkv | fc_int4_g128 | 0.0931 | 36 | 3.352 | 338 | 87.2 | 90.0% | memory |
| fc_o | fc_int4_g128 | 0.0633 | 36 | 2.278 | 332 | 85.7 | 88.3% | memory |
| rmsnorm | rmsnorm | 0.0033 | 73 | 0.240 | 0 | 4.7 | 4.8% | memory |
| add | add | 0.0022 | 72 | 0.155 | 0 | 7.0 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0027 | 36 | 0.098 | 0 | 6.1 | 6.3% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0027 | 36 | 0.098 | 0 | 1.6 | 1.7% | memory |
| rope_q | rope_q | 0.0025 | 36 | 0.090 | 0 | 6.9 | 7.1% | memory |
| rope_k | rope_k | 0.0023 | 36 | 0.082 | 0 | 2.0 | 2.1% | memory |
| **TOTAL** |  |  |  | **32.721** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Decode — KV=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_opencl_micro | 0.3134 | 36 | 11.282 | 453 | 56.7 | 58.4% | memory |
| fc_up | fc_int4_g128 | 0.1462 | 36 | 5.262 | 340 | 88.0 | 90.7% | memory |
| fc_gate | fc_int4_g128 | 0.1456 | 36 | 5.242 | 342 | 88.4 | 91.1% | memory |
| fc_down | fc_int4_g128 | 0.1455 | 36 | 5.238 | 342 | 88.4 | 91.2% | memory |
| lm_head | fc_int8_g128 | 4.3477 | 1 | 4.348 | 179 | 90.9 | 93.7% | memory |
| fc_qkv | fc_int4_g128 | 0.0931 | 36 | 3.352 | 338 | 87.2 | 90.0% | memory |
| fc_o | fc_int4_g128 | 0.0633 | 36 | 2.278 | 332 | 85.7 | 88.3% | memory |
| rmsnorm | rmsnorm | 0.0033 | 73 | 0.240 | 0 | 4.7 | 4.8% | memory |
| add | add | 0.0022 | 72 | 0.155 | 0 | 7.0 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.0027 | 36 | 0.098 | 0 | 6.1 | 6.3% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0027 | 36 | 0.098 | 0 | 1.6 | 1.7% | memory |
| rope_q | rope_q | 0.0025 | 36 | 0.090 | 0 | 6.9 | 7.1% | memory |
| rope_k | rope_k | 0.0023 | 36 | 0.082 | 0 | 2.0 | 2.1% | memory |
| **TOTAL** |  |  |  | **37.765** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

## Prefill tables (single forward over S tokens)

### Prefill — S=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_ocl_micro_prefill | 8.9020 | 36 | 320.472 | 976 | 1.4 | 5.0% | compute |
| fc_down | fc_int4_g128 | 2.5568 | 36 | 92.045 | 19947 | 14.8 | 50.7% | compute |
| fc_up | fc_int4_g128 | 2.5180 | 36 | 90.649 | 20255 | 15.1 | 51.5% | compute |
| fc_gate | fc_int4_g128 | 2.5087 | 36 | 90.313 | 20330 | 15.1 | 51.7% | compute |
| fc_qkv | fc_int4_g128 | 1.6176 | 36 | 58.234 | 19913 | 16.0 | 50.6% | compute |
| fc_o | fc_int4_g128 | 1.0904 | 36 | 39.256 | 19695 | 17.5 | 50.1% | compute |
| add | add | 0.1212 | 72 | 8.728 | 0 | 7.0 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.2079 | 36 | 7.483 | 0 | 6.1 | 6.3% | memory |
| rmsnorm | rmsnorm | 0.0866 | 73 | 6.325 | 0 | 4.7 | 4.8% | memory |
| lm_head | fc_int8_g128 | 4.3698 | 1 | 4.370 | 179 | 90.9 | 93.7% | memory |
| rope_q | rope_q | 0.1210 | 36 | 4.356 | 0 | 6.9 | 7.1% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.0549 | 36 | 1.976 | 0 | 1.6 | 1.7% | memory |
| rope_k | rope_k | 0.0355 | 36 | 1.278 | 0 | 2.0 | 2.1% | memory |
| **TOTAL** |  |  |  | **725.485** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_ocl_micro_prefill | 31.7250 | 36 | 1142.100 | 1089 | 0.8 | 5.5% | compute |
| fc_up | fc_int4_g128 | 5.0094 | 36 | 180.339 | 20363 | 12.6 | 51.8% | compute |
| fc_down | fc_int4_g128 | 4.9849 | 36 | 179.455 | 20463 | 12.7 | 52.0% | compute |
| fc_gate | fc_int4_g128 | 4.9725 | 36 | 179.010 | 20514 | 12.7 | 52.2% | compute |
| fc_qkv | fc_int4_g128 | 3.0654 | 36 | 110.356 | 21016 | 14.3 | 53.4% | compute |
| fc_o | fc_int4_g128 | 1.9675 | 36 | 70.830 | 21829 | 16.6 | 55.5% | compute |
| add | add | 0.2803 | 72 | 20.184 | 0 | 7.0 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.4264 | 36 | 15.350 | 0 | 6.1 | 6.3% | memory |
| rmsnorm | rmsnorm | 0.1788 | 73 | 13.055 | 0 | 4.7 | 4.8% | memory |
| rope_q | rope_q | 0.2961 | 36 | 10.659 | 0 | 6.9 | 7.1% | memory |
| lm_head | fc_int8_g128 | 4.3698 | 1 | 4.370 | 179 | 90.9 | 93.7% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.1064 | 36 | 3.829 | 0 | 1.6 | 1.7% | memory |
| rope_k | rope_k | 0.0665 | 36 | 2.392 | 0 | 2.0 | 2.1% | memory |
| **TOTAL** |  |  |  | **1931.928** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_ocl_micro_prefill | 123.1200 | 36 | 4432.320 | 1119 | 0.4 | 5.7% | compute |
| fc_up | fc_int4_g128 | 10.5730 | 36 | 380.629 | 19296 | 10.7 | 49.1% | compute |
| fc_gate | fc_int4_g128 | 10.3933 | 36 | 374.159 | 19629 | 10.9 | 49.9% | compute |
| fc_down | fc_int4_g128 | 9.5506 | 36 | 343.820 | 21361 | 11.9 | 54.3% | compute |
| fc_qkv | fc_int4_g128 | 6.3823 | 36 | 229.762 | 20188 | 12.5 | 51.3% | compute |
| fc_o | fc_int4_g128 | 3.7793 | 36 | 136.055 | 22729 | 15.8 | 57.8% | compute |
| add | add | 0.6316 | 72 | 45.479 | 0 | 7.0 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 0.8733 | 36 | 31.439 | 0 | 6.1 | 6.3% | memory |
| rmsnorm | rmsnorm | 0.4140 | 73 | 30.224 | 0 | 4.7 | 4.8% | memory |
| rope_q | rope_q | 0.7004 | 36 | 25.213 | 0 | 6.9 | 7.1% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.2126 | 36 | 7.655 | 0 | 1.6 | 1.7% | memory |
| rope_k | rope_k | 0.1343 | 36 | 4.834 | 0 | 2.0 | 2.1% | memory |
| lm_head | fc_int8_g128 | 4.3698 | 1 | 4.370 | 179 | 90.9 | 93.7% | memory |
| **TOTAL** |  |  |  | **6045.958** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

### Prefill — S=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| pa | pa_ocl_micro_prefill | 487.2410 | 36 | 17540.676 | 1130 | 0.2 | 5.8% | compute |
| fc_up | fc_int4_g128 | 19.7346 | 36 | 710.446 | 20675 | 10.8 | 52.6% | compute |
| fc_gate | fc_int4_g128 | 19.6505 | 36 | 707.419 | 20764 | 10.9 | 52.8% | compute |
| fc_down | fc_int4_g128 | 18.7735 | 36 | 675.847 | 21734 | 11.4 | 55.3% | compute |
| fc_qkv | fc_int4_g128 | 12.8821 | 36 | 463.757 | 20004 | 11.7 | 50.9% | compute |
| fc_o | fc_int4_g128 | 7.4861 | 36 | 269.498 | 22949 | 15.3 | 58.4% | compute |
| add | add | 1.3576 | 72 | 97.751 | 0 | 7.0 | 7.2% | memory |
| rmsnorm3d_q | rmsnorm3d_q | 1.8635 | 36 | 67.087 | 0 | 6.1 | 6.3% | memory |
| rmsnorm | rmsnorm | 0.8971 | 73 | 65.490 | 0 | 4.7 | 4.8% | memory |
| rope_q | rope_q | 1.5098 | 36 | 54.354 | 0 | 6.9 | 7.1% | memory |
| rmsnorm3d_k | rmsnorm3d_k | 0.4357 | 36 | 15.685 | 0 | 1.6 | 1.7% | memory |
| rope_k | rope_k | 0.3426 | 36 | 12.333 | 0 | 2.0 | 2.1% | memory |
| lm_head | fc_int8_g128 | 4.3698 | 1 | 4.370 | 179 | 90.9 | 93.7% | memory |
| **TOTAL** |  |  |  | **20684.713** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

## Per-kernel decomposition (cliloader kernel names)

Each op above maps to one or more GPU kernels. PA decomposes into `pa_kv_cache_update_ref` + attention kernel + finalization. Prefill FC decomposes into `dynamic_quantize_gpu_opt` + `gemm_kernel`. Decode FC is a single `gemm_kernel` call.

### Decode sub-kernels
### Decode sub-kernels — KV=1024

| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `paged_attention_opt__single_token` | 0.1667 | 1 | 36 | 6.001 | 18.3% | 13.0% |
| fc_up | `gemm_kernel` | 0.1462 | 1 | 36 | 5.262 | 16.0% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1456 | 1 | 36 | 5.242 | 15.9% | 91.1% |
| fc_down | `gemm_kernel` | 0.1455 | 1 | 36 | 5.238 | 15.9% | 91.2% |
| lm_head | `gemm_kernel` | 4.3477 | 1 | 1 | 4.348 | 13.2% | 93.7% |
| fc_qkv | `gemm_kernel` | 0.0931 | 1 | 36 | 3.352 | 10.2% | 90.0% |
| fc_o | `gemm_kernel` | 0.0633 | 1 | 36 | 2.278 | 6.9% | 88.3% |
| pa | `pa_kv_cache_update_ref` | 0.0075 | 1 | 36 | 0.270 | 0.8% | — |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0033 | 1 | 73 | 0.240 | 0.7% | 4.8% |
| add | `eltwise_simple_vload8` | 0.0022 | 1 | 72 | 0.155 | 0.5% | 7.2% |
| pa | `paged_attention_opt__single_token_finalization` | 0.0034 | 1 | 36 | 0.122 | 0.4% | — |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0027 | 1 | 36 | 0.098 | 0.3% | 6.3% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0027 | 1 | 36 | 0.098 | 0.3% | 1.7% |
| rope_q | `rope_opt` | 0.0025 | 1 | 36 | 0.090 | 0.3% | 7.1% |
| rope_k | `rope_opt` | 0.0023 | 1 | 36 | 0.082 | 0.2% | 2.1% |

### Decode sub-kernels — KV=2048

| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `paged_attention_opt__single_token` | 0.1925 | 1 | 36 | 6.930 | 20.6% | 22.5% |
| fc_up | `gemm_kernel` | 0.1462 | 1 | 36 | 5.262 | 15.6% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1456 | 1 | 36 | 5.242 | 15.6% | 91.1% |
| fc_down | `gemm_kernel` | 0.1455 | 1 | 36 | 5.238 | 15.5% | 91.2% |
| lm_head | `gemm_kernel` | 4.3477 | 1 | 1 | 4.348 | 12.9% | 93.7% |
| fc_qkv | `gemm_kernel` | 0.0931 | 1 | 36 | 3.352 | 9.9% | 90.0% |
| fc_o | `gemm_kernel` | 0.0633 | 1 | 36 | 2.278 | 6.8% | 88.3% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0033 | 1 | 73 | 0.240 | 0.7% | 4.8% |
| add | `eltwise_simple_vload8` | 0.0022 | 1 | 72 | 0.155 | 0.5% | 7.2% |
| pa | `pa_kv_cache_update_ref` | 0.0041 | 1 | 36 | 0.148 | 0.4% | — |
| pa | `paged_attention_opt__single_token_finalization` | 0.0039 | 1 | 36 | 0.140 | 0.4% | — |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0027 | 1 | 36 | 0.098 | 0.3% | 6.3% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0027 | 1 | 36 | 0.098 | 0.3% | 1.7% |
| rope_q | `rope_opt` | 0.0025 | 1 | 36 | 0.090 | 0.3% | 7.1% |
| rope_k | `rope_opt` | 0.0023 | 1 | 36 | 0.082 | 0.2% | 2.1% |

### Decode sub-kernels — KV=4096

| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `paged_attention_opt__gqa_single_token` | 0.1619 | 1 | 36 | 5.828 | 17.8% | 53.4% |
| fc_up | `gemm_kernel` | 0.1462 | 1 | 36 | 5.262 | 16.1% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1456 | 1 | 36 | 5.242 | 16.0% | 91.1% |
| fc_down | `gemm_kernel` | 0.1455 | 1 | 36 | 5.238 | 16.0% | 91.2% |
| lm_head | `gemm_kernel` | 4.3477 | 1 | 1 | 4.348 | 13.3% | 93.7% |
| fc_qkv | `gemm_kernel` | 0.0931 | 1 | 36 | 3.352 | 10.2% | 90.0% |
| fc_o | `gemm_kernel` | 0.0633 | 1 | 36 | 2.278 | 7.0% | 88.3% |
| pa | `paged_attention_opt__single_token_finalization` | 0.0071 | 1 | 36 | 0.256 | 0.8% | — |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0033 | 1 | 73 | 0.240 | 0.7% | 4.8% |
| add | `eltwise_simple_vload8` | 0.0022 | 1 | 72 | 0.155 | 0.5% | 7.2% |
| pa | `pa_kv_cache_update_ref` | 0.0043 | 1 | 36 | 0.155 | 0.5% | — |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0027 | 1 | 36 | 0.098 | 0.3% | 6.3% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0027 | 1 | 36 | 0.098 | 0.3% | 1.7% |
| rope_q | `rope_opt` | 0.0025 | 1 | 36 | 0.090 | 0.3% | 7.1% |
| rope_k | `rope_opt` | 0.0023 | 1 | 36 | 0.082 | 0.2% | 2.1% |

### Decode sub-kernels — KV=8192

| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `paged_attention_opt__gqa_single_token` | 0.2961 | 1 | 36 | 10.660 | 28.2% | 58.4% |
| fc_up | `gemm_kernel` | 0.1462 | 1 | 36 | 5.262 | 13.9% | 90.7% |
| fc_gate | `gemm_kernel` | 0.1456 | 1 | 36 | 5.242 | 13.9% | 91.1% |
| fc_down | `gemm_kernel` | 0.1455 | 1 | 36 | 5.238 | 13.9% | 91.2% |
| lm_head | `gemm_kernel` | 4.3477 | 1 | 1 | 4.348 | 11.5% | 93.7% |
| fc_qkv | `gemm_kernel` | 0.0931 | 1 | 36 | 3.352 | 8.9% | 90.0% |
| fc_o | `gemm_kernel` | 0.0633 | 1 | 36 | 2.278 | 6.0% | 88.3% |
| pa | `paged_attention_opt__single_token_finalization` | 0.0133 | 1 | 36 | 0.479 | 1.3% | — |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0033 | 1 | 73 | 0.240 | 0.6% | 4.8% |
| add | `eltwise_simple_vload8` | 0.0022 | 1 | 72 | 0.155 | 0.4% | 7.2% |
| pa | `pa_kv_cache_update_ref` | 0.0040 | 1 | 36 | 0.144 | 0.4% | — |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.0027 | 1 | 36 | 0.098 | 0.3% | 6.3% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0027 | 1 | 36 | 0.098 | 0.3% | 1.7% |
| rope_q | `rope_opt` | 0.0025 | 1 | 36 | 0.090 | 0.2% | 7.1% |
| rope_k | `rope_opt` | 0.0023 | 1 | 36 | 0.082 | 0.2% | 2.1% |

### Prefill sub-kernels
### Prefill sub-kernels — S=1024

| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `sdpa_micro__prefill` | 8.8140 | 1 | 36 | 317.304 | 43.7% | 5.0% |
| fc_up | `gemm_kernel` | 2.4510 | 1 | 36 | 88.236 | 12.2% | 52.9% |
| fc_gate | `gemm_kernel` | 2.4411 | 1 | 36 | 87.880 | 12.1% | 53.1% |
| fc_down | `gemm_kernel` | 2.3037 | 1 | 36 | 82.933 | 11.4% | 56.3% |
| fc_qkv | `gemm_kernel` | 1.5507 | 1 | 36 | 55.825 | 7.7% | 52.8% |
| fc_o | `gemm_kernel` | 0.9885 | 1 | 36 | 35.586 | 4.9% | 55.3% |
| fc_down | `dynamic_quantize_gpu_opt` | 0.2531 | 1 | 36 | 9.112 | 1.3% | — |
| add | `eltwise_simple_vload8` | 0.1212 | 1 | 72 | 8.728 | 1.2% | 7.2% |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.2079 | 1 | 36 | 7.483 | 1.0% | 6.3% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.0866 | 1 | 73 | 6.325 | 0.9% | 4.8% |
| lm_head | `gemm_kernel` | 4.3698 | 1 | 1 | 4.370 | 0.6% | 93.7% |
| rope_q | `rope_opt` | 0.1210 | 1 | 36 | 4.356 | 0.6% | 7.1% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.1019 | 1 | 36 | 3.670 | 0.5% | — |
| pa | `pa_kv_cache_update_ref` | 0.0880 | 1 | 36 | 3.168 | 0.4% | — |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.0676 | 1 | 36 | 2.433 | 0.3% | — |
| fc_up | `dynamic_quantize_gpu_opt` | 0.0670 | 1 | 36 | 2.413 | 0.3% | — |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.0669 | 1 | 36 | 2.409 | 0.3% | — |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.0549 | 1 | 36 | 1.976 | 0.3% | 1.7% |
| rope_k | `rope_opt` | 0.0355 | 1 | 36 | 1.278 | 0.2% | 2.1% |

### Prefill sub-kernels — S=2048

| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `sdpa_micro__prefill` | 31.5640 | 1 | 36 | 1136.304 | 58.8% | 5.5% |
| fc_up | `gemm_kernel` | 4.8723 | 1 | 36 | 175.403 | 9.1% | 53.2% |
| fc_gate | `gemm_kernel` | 4.8372 | 1 | 36 | 174.139 | 9.0% | 53.6% |
| fc_down | `gemm_kernel` | 4.4724 | 1 | 36 | 161.006 | 8.3% | 58.0% |
| fc_qkv | `gemm_kernel` | 2.9340 | 1 | 36 | 105.624 | 5.5% | 55.8% |
| fc_o | `gemm_kernel` | 1.7535 | 1 | 36 | 63.126 | 3.3% | 62.3% |
| add | `eltwise_simple_vload8` | 0.2803 | 1 | 72 | 20.184 | 1.0% | 7.2% |
| fc_down | `dynamic_quantize_gpu_opt` | 0.5125 | 1 | 36 | 18.449 | 1.0% | — |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.4264 | 1 | 36 | 15.350 | 0.8% | 6.3% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.1788 | 1 | 73 | 13.055 | 0.7% | 4.8% |
| rope_q | `rope_opt` | 0.2961 | 1 | 36 | 10.659 | 0.6% | 7.1% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.2140 | 1 | 36 | 7.704 | 0.4% | — |
| pa | `pa_kv_cache_update_ref` | 0.1610 | 1 | 36 | 5.796 | 0.3% | — |
| fc_up | `dynamic_quantize_gpu_opt` | 0.1371 | 1 | 36 | 4.936 | 0.3% | — |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.1353 | 1 | 36 | 4.870 | 0.3% | — |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.1314 | 1 | 36 | 4.732 | 0.2% | — |
| lm_head | `gemm_kernel` | 4.3698 | 1 | 1 | 4.370 | 0.2% | 93.7% |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.1064 | 1 | 36 | 3.829 | 0.2% | 1.7% |
| rope_k | `rope_opt` | 0.0665 | 1 | 36 | 2.392 | 0.1% | 2.1% |

### Prefill sub-kernels — S=4096

| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `sdpa_micro__prefill` | 122.8210 | 1 | 36 | 4421.556 | 73.1% | 5.7% |
| fc_up | `gemm_kernel` | 10.2927 | 1 | 36 | 370.537 | 6.1% | 50.4% |
| fc_gate | `gemm_kernel` | 10.1175 | 1 | 36 | 364.230 | 6.0% | 51.3% |
| fc_down | `gemm_kernel` | 8.4831 | 1 | 36 | 305.392 | 5.1% | 61.2% |
| fc_qkv | `gemm_kernel` | 6.1185 | 1 | 36 | 220.266 | 3.6% | 53.6% |
| fc_o | `gemm_kernel` | 3.3537 | 1 | 36 | 120.733 | 2.0% | 65.1% |
| add | `eltwise_simple_vload8` | 0.6316 | 1 | 72 | 45.479 | 0.8% | 7.2% |
| fc_down | `dynamic_quantize_gpu_opt` | 1.0675 | 1 | 36 | 38.428 | 0.6% | — |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 0.8733 | 1 | 36 | 31.439 | 0.5% | 6.3% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.4140 | 1 | 73 | 30.224 | 0.5% | 4.8% |
| rope_q | `rope_opt` | 0.7004 | 1 | 36 | 25.213 | 0.4% | 7.1% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.4256 | 1 | 36 | 15.322 | 0.3% | — |
| pa | `pa_kv_cache_update_ref` | 0.2990 | 1 | 36 | 10.764 | 0.2% | — |
| fc_up | `dynamic_quantize_gpu_opt` | 0.2803 | 1 | 36 | 10.092 | 0.2% | — |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.2758 | 1 | 36 | 9.929 | 0.2% | — |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.2638 | 1 | 36 | 9.496 | 0.2% | — |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.2126 | 1 | 36 | 7.655 | 0.1% | 1.7% |
| rope_k | `rope_opt` | 0.1343 | 1 | 36 | 4.834 | 0.1% | 2.1% |
| lm_head | `gemm_kernel` | 4.3698 | 1 | 1 | 4.370 | 0.1% | 93.7% |

### Prefill sub-kernels — S=8192

| op | kernel name | single ms | launches/call | calls/inf | total ms | % | Eff% |
|---|---|---:|---:|---:|---:|---:|---:|
| pa | `sdpa_micro__prefill` | 486.6890 | 1 | 36 | 17520.804 | 84.7% | 5.8% |
| fc_up | `gemm_kernel` | 19.1769 | 1 | 36 | 690.368 | 3.3% | 54.1% |
| fc_gate | `gemm_kernel` | 19.0926 | 1 | 36 | 687.334 | 3.3% | 54.3% |
| fc_down | `gemm_kernel` | 16.4109 | 1 | 36 | 590.792 | 2.9% | 63.2% |
| fc_qkv | `gemm_kernel` | 12.3546 | 1 | 36 | 444.766 | 2.2% | 53.0% |
| fc_o | `gemm_kernel` | 6.6411 | 1 | 36 | 239.080 | 1.2% | 65.8% |
| add | `eltwise_simple_vload8` | 1.3576 | 1 | 72 | 97.751 | 0.5% | 7.2% |
| fc_down | `dynamic_quantize_gpu_opt` | 2.3626 | 1 | 36 | 85.054 | 0.4% | — |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 1.8635 | 1 | 36 | 67.087 | 0.3% | 6.3% |
| rmsnorm | `rms_gpu_bfyx_opt` | 0.8971 | 1 | 73 | 65.490 | 0.3% | 4.8% |
| rope_q | `rope_opt` | 1.5098 | 1 | 36 | 54.354 | 0.3% | 7.1% |
| fc_o | `dynamic_quantize_gpu_opt` | 0.8450 | 1 | 36 | 30.419 | 0.1% | — |
| fc_gate | `dynamic_quantize_gpu_opt` | 0.5579 | 1 | 36 | 20.086 | 0.1% | — |
| fc_up | `dynamic_quantize_gpu_opt` | 0.5577 | 1 | 36 | 20.078 | 0.1% | — |
| pa | `pa_kv_cache_update_ref` | 0.5520 | 1 | 36 | 19.872 | 0.1% | — |
| fc_qkv | `dynamic_quantize_gpu_opt` | 0.5275 | 1 | 36 | 18.992 | 0.1% | — |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 0.4357 | 1 | 36 | 15.685 | 0.1% | 1.7% |
| rope_k | `rope_opt` | 0.3426 | 1 | 36 | 12.333 | 0.1% | 2.1% |
| lm_head | `gemm_kernel` | 4.3698 | 1 | 1 | 4.370 | 0.0% | 93.7% |

## Top contributors (sorted by total ms per inference)

### Decode

| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | pa 6.39ms (19%) | fc_up 5.26ms (16%) | fc_gate 5.24ms (16%) |
| 2048 | pa 7.21ms (21%) | fc_up 5.26ms (16%) | fc_gate 5.24ms (16%) |
| 4096 | pa 6.24ms (19%) | fc_up 5.26ms (16%) | fc_gate 5.24ms (16%) |
| 8192 | pa 11.28ms (30%) | fc_up 5.26ms (14%) | fc_gate 5.24ms (14%) |

### Prefill

| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | pa 320.47ms (44%) | fc_down 92.05ms (13%) | fc_up 90.65ms (12%) |
| 2048 | pa 1142.10ms (59%) | fc_up 180.34ms (9%) | fc_down 179.46ms (9%) |
| 4096 | pa 4432.32ms (73%) | fc_up 380.63ms (6%) | fc_gate 374.16ms (6%) |
| 8192 | pa 17540.68ms (85%) | fc_up 710.45ms (3%) | fc_gate 707.42ms (3%) |

## Comparison: PTL 4Xe Linux vs PTL 12Xe Windows

| Metric | PTL 4Xe (this report) | PTL 12Xe (2026-05-08) | Ratio |
|---|---:|---:|---:|
| Xe cores | 4 | 12 | 3.0× |
| FP16 XMX peak | 19.66 TFLOPS | 58.98 TFLOPS | 3.0× |
| BW (measured) | 97.0 GB/s | 110.0 GB/s | 1.13× |
| TTFT S=1024 | 725.5 ms | 189.8 ms | 3.8× |
| TTFT S=2048 | 1931.9 ms | 410.0 ms | 4.7× |
| TTFT S=4096 | 6046.0 ms | 1001.3 ms | 6.0× |
| TTFT S=8192 | 20684.7 ms | 2588.4 ms | 8.0× |
| TPOT KV=1024 | 32.9 ms | 25.5 ms | 1.3× |
| TPOT KV=2048 | 33.7 ms | 26.8 ms | 1.3× |
| TPOT KV=4096 | 32.7 ms | 27.6 ms | 1.2× |
| TPOT KV=8192 | 37.8 ms | 31.3 ms | 1.2× |

Key differences:
- **PA prefill**: OCL micro-kernel on 4Xe uses EU ALU only (~975–1130 GFLOPS, 5% XMX eff)
  while 12Xe OCL micro-kernel uses XMX/DPAS (~26000–32000 GFLOPS, 44–55% eff)
- **Root cause**: ngen GEMM catalog has no `HWTagXe3` entries; PTL 4Xe (ip_version arch=30)
  maps to `GenericXe3` → `Core::Xe3` with no matching systolic strategy → falls back to EU ALU
- **12Xe Windows**: ip_version likely maps to Xe2 (arch=20) → catalog match → DPAS used
- **Decode**: dominated by memory-bound FC ops; 4Xe is ~1.3× slower (BW ratio)
- **Prefill**: PA dominates increasingly at large S; 4Xe is 3.8–8.0× slower overall due to EU-only PA

## Analysis & insights

### Decode (memory-bound)

All decode ops are deeply memory-bound. FC kernels achieve 88–94% BW efficiency (estimated),
consistent with 12Xe behavior. PA decode uses two kernel variants:
- **`single_token`** (Skv ≤ 2048): 13–22% BW efficiency
- **`gqa_single_token`** (Skv ≥ 4096): 53–58% BW efficiency (GQA-optimized tiling)

TPOT ranges from 32.7–37.8 ms, dominated by FC weight reads. PA becomes the largest
single op at KV=8192 (11.3/37.8 = 30% of TPOT).

### Prefill (compute-bound)

Prefill FC ops are INT8 XMX compute-bound (~50% INT8 XMX efficiency estimated).
PA prefill uses `sdpa_micro__prefill` (EU ALU, no DPAS) achieving only 975–1130 GFLOPS
(~5% of FP16 XMX peak). PA dominates increasingly:
- S=1024: PA = 320 ms / 725 ms = **44%** of TTFT
- S=2048: PA = 1142 ms / 1932 ms = **59%** of TTFT
- S=4096: PA = 4432 ms / 6046 ms = **73%** of TTFT
- S=8192: PA = 17541 ms / 20685 ms = **85%** of TTFT

### Optimization levers

1. **Enable XMX/DPAS for Xe3 PA**: Add `HWTagXe3` entries to ngen GEMM microkernel catalog
   (or allow Xe3 to fall through to Xe2 entries) → expected ~6× prefill PA speedup
2. **CM PA on Linux**: Install CM frontend (`libclangFEWrapper.so`) for CM-based PA with native XMX
3. **GQA kernel for small Skv**: `single_token` at Skv ≤ 2048 only achieves 13–22% BW —
   GQA-optimized variant should apply at all KV lengths
4. **Small ops fusion**: merge q_norm+k_norm, batch rope_q+rope_k to halve launch overhead

## Caveats & method

- **PA data measured** on PTL 4Xe via `pa_bench` + cliloader. All other ops **estimated** by scaling from PTL 12Xe.
- Scaling: memory-bound ops × 1.134 (BW ratio 110/97); compute-bound FC gemm × 3.0 (XMX core ratio) + dq × 1.134.
- FC weight bytes count INT4 weight + FP16 scale/zp(g=128) + FP16 act + FP16 out.
- PA bytes assume INT8 KV cache (1B/elem) + FP16 Q, FP16 out. Prefill K/V read as FP16 (raw inputs).
- Prefill FLOPs use causal mask formula: 4 × NH × (Sq×(Sq+1)/2) × HD.
- Decode FC treated as memory-bound (weights dominate at M=1); prefill FC is INT8 XMX compute-bound.
- PA prefill XMX eff% is vs FP16 XMX peak (kernel uses EU ALU, not DPAS).
- swish/multiply fused into SwiGLU; not listed as standalone kernel.
- lm_head run once per inference (last position in prefill, every step in decode).
- Target machine: intel@10.239.152.140 (Linux, PTL 4Xe, driver 26.14.037858).

## Reproduction

```bash
# On intel@10.239.152.140
export LD_LIBRARY_PATH=~/river/openvino/install_release/runtime/lib/intel64:~/river/openvino/temp/Linux_x86_64/tbb/lib
CLILOADER=~/river/clintercept-3.0.6-Linux/bin/cliloader
BIN=~/river/roofline_test_utils/build/pa_bench

# Decode (measured):
$CLILOADER -d $BIN -- --mode decode --n_head 32 --n_kv_head 8 --head_dim 128 --n_layers 1 --S_kv 8192 --S_q 1 --block_size 16 --pages_per_block 1 --kv_cache_dtype i8

# Prefill (measured):
$CLILOADER -d $BIN -- --mode prefill --n_head 32 --n_kv_head 8 --head_dim 128 --n_layers 1 --S_kv 0 --S_q 4096 --block_size 16 --pages_per_block 1 --kv_cache_dtype i8

# Note: FC, small ops, lm_head benchmarks not yet run on 4Xe.
# Those values are estimated from PTL 12Xe (SUMMARY_qwen3_omni_2026-05-08.md).
```
