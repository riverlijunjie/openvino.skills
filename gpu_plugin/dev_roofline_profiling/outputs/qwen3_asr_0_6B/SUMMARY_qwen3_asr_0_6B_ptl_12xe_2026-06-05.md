# Qwen3-ASR-0.6B — Roofline on PTL 12Xe (2026-06-05)

**Platform**: Intel PTL 12Xe iGPU (12 Xe cores × 8 EU × 10 threads), 2400 MHz, 110 GB/s
**Model**: Qwen3-ASR-0.6B (audio encoder + Qwen3 text decoder), Linear+Embedding weights **FP16**.
**SDPA**: PagedAttention (opencl + micro_kernel), INT8 KV cache (text decoder; matches Intel GPU plugin default for PA models). Audio encoder SDPA remains FP16.
**Source config**: https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/main/config.json

**Inputs evaluated**: text-decoder prefill context = 512 / 1024 / 4096 / 8192 tokens; output = 512 tokens; audio encoder runs once over 1500 mel frames.

## 1. Hardware peaks (per SKILL.md formulas)

`FP16 XMX TFLOPS = xe_cores × 8 × 256 × freq_GHz`; `INT8 XMX = 2× FP16`; `SIMD FP16 = xe_cores × 8 × 32 × freq`.

| Platform | Xe Cores | Freq (MHz) | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | SIMD FP16 (TFLOPS) | Ridge FP16 (FLOP/B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PTL 12Xe | 12 | 2400 | 110 | 58.982 | 117.965 | 7.373 | 536 |

## 2. Model architecture

### Text decoder (Qwen3, 28-layer dense)

| Field | Value |
|---|---:|
| `hidden_size` | 1024 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` (NH) | 16 |
| `num_key_value_heads` (NKV) | 8 (GQA 2:1) |
| `head_dim` (HD) | 128 |
| `intermediate_size` | 3072 |
| `vocab_size` | 151936 |
| `tie_word_embeddings` | True |
| `rope_theta` | 1,000,000 |

### Audio encoder (Whisper-style)

| Field | Value |
|---|---:|
| `d_model` | 896 |
| `encoder_layers` | 18 |
| `encoder_attention_heads` (MHA) | 14 |
| `head_dim` | 64 |
| `encoder_ffn_dim` | 3584 |
| `num_mel_bins` | 128 |
| `max_source_positions` | 1500 |
| `output_dim` (→ text decoder hidden) | 1024 |

## 3. Theoretical weight distribution (FP16)

### Text decoder per-layer (1 of 28)

| Weight | Shape (K × N) | Dtype | MB |
|---|---|---|---:|
| FC_QKV (fused Q+K+V) | 1024 × 4096 | FP16 | 8.0 |
| FC_O (attn output)   | 2048 × 1024 | FP16 | 4.0 |
| FC_Gate (SwiGLU)     | 1024 × 3072 | FP16 | 6.0 |
| FC_Up (SwiGLU)       | 1024 × 3072 | FP16 | 6.0 |
| FC_Down (SwiGLU)     | 3072 × 1024 | FP16 | 6.0 |
| RMSNorm × 2          | [1024] each | FP16 | 0.004 |
| q_norm / k_norm      | [128] each  | FP16 | 0.0000 |
| **per layer** |  |  | **30.00** |
| **× 28 layers** |  |  | **840.12** |

### Audio encoder per-layer (1 of 18)

| Weight | Shape (K × N) | Dtype | MB |
|---|---|---|---:|
| FC_QKV (fused) | 896 × 2688 | FP16 | 4.594 |
| FC_O           | 896 × 896  | FP16 | 1.531 |
| FC_FC1 (GELU)  | 896 × 3584 | FP16 | 6.125 |
| FC_FC2 (GELU)  | 3584 × 896 | FP16 | 6.125 |
| LayerNorm × 2  | [896] g+b each | FP16 | 0.007 |
| **per layer** |  |  | **18.38** |
| **× 18 layers** |  |  | **330.87** |

### Global + shared weights

| Weight | Shape | Dtype | MB |
|---|---|---|---:|
| Token embedding | 151936 × 1024 | FP16 | 296.75 |
| LM_Head (tied=True) | shares embedding | — | 0.0 |
| Final RMSNorm (text)  | [1024] | FP16 | 0.002 |
| Audio encoder conv front-end | conv1d×2 | FP16 | 5.250 |
| Audio positional embedding | 1500 × 896 | FP16 | 2.563 |
| Audio output adapter | 896→480→1024 | FP16 | 1.758 |

### Totals

| Component | MB |
|---|---:|
| Text decoder (28 layers) | 840.12 |
| Token embedding (shared w/ LM_Head) | 296.75 |
| Audio encoder (18 layers) | 330.87 |
| Audio encoder front + adapter | 9.57 |
| Final RMSNorm (text) | 0.0020 |
| **Model total** | **1477.32** |

## 4. Benchmark methodology

- **Bench utils**: `fc_bench` (precision=f16, plain MatMul, no compression), `pa_bench` (kv_dtype=i8, impl=ocl), `small_ops_bench` for rmsnorm/rope/eltwise.
- **Tool**: cliloader Device Performance Timing; `parse_logs.py` extracts per-iteration avg GPU kernel ns.
- **L2/L3 flush** between every FC infer (64 MB Relu) so each measurement reads weights from VRAM, not on-die cache. Required because the entire FP16 text decoder weights (~177 MB) is small enough to be mostly cached on iGPU.
- **Input tensors** allocated via RemoteContext in USM_DEVICE (iGPU shared system memory).
- **PA prefill** uses causal mask (Sq·(Sq+1)/2 effective pairs) per SKILL.md.
- **PA decode** is split into `pa_kv_cache_update_ref` + `paged_attention_opt__single_token` + `single_token_finalization`.
- **Audio encoder SDPA** is bidirectional (no causal mask); measured causal time scaled ×2 in the encoder roofline row.
- **swish / multiply** are not profiled separately — they fuse into the SwiGLU primitive per SKILL.md.
- **LM_Head** is `K=1024, N=151936` FP16 (~297 MB weight stream). Counted once per inference at decode and once at prefill (last token only).

## 5. Audio encoder fixed overhead (per inference, S=1500)

The audio encoder runs **once** per inference regardless of text token count.

### Audio encoder — S=1500

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| enc_sdpa | `sdpa_micro__prefill (×2 bidir est.)` | 18 | 0.7643 | 0.1367 | 13.7582 | 2.4609 | 5.59× | 10550.2 | 14.1 | 17.9% | compute |
| enc_fc_ffn2 | `gemm_kernel` | 18 | 0.2594 | 0.1806 | 4.6688 | 3.2502 | 1.44× | 37142.3 | 76.6 | 69.6% | memory |
| enc_fc_ffn1 | `gemm_kernel` | 18 | 0.2163 | 0.1806 | 3.8931 | 3.2502 | 1.20× | 44542.3 | 91.8 | 83.5% | memory |
| enc_fc_qkv | `gemm_kernel` | 18 | 0.1847 | 0.1415 | 3.3247 | 2.5476 | 1.31× | 39118.3 | 84.3 | 76.6% | memory |
| enc_fc_o | `gemm_kernel` | 18 | 0.0718 | 0.0635 | 1.2918 | 1.1424 | 1.13× | 33560.2 | 97.3 | 88.4% | memory |
| enc_outproj | `gemm_kernel` | 1 | 0.0795 | 0.0690 | 0.0795 | 0.0690 | 1.15× | 34631.5 | 95.6 | 86.9% | memory |
| **TOTAL (encoder fixed)** |  |  |  |  | **27.0161** | **12.7203** | **2.12×** |  |  |  |  |

## 6. Token latency summary

**TTFT** = audio encoder (27.02 ms) + text decoder prefill.
**TPOT** = per output token decode latency at the listed KV context.

### Prefill — TTFT

| S (text ctx) | Encoder (ms) | Text-decoder prefill (ms) | **TTFT (ms)** | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 27.02 | 26.40 | **53.42** | 0.1043 | 9585 |
| 1024 | 27.02 | 46.70 | **73.72** | 0.0720 | 13891 |
| 4096 | 27.02 | 259.80 | **286.81** | 0.0700 | 14281 |
| 8192 | 27.02 | 763.96 | **790.97** | 0.0966 | 10357 |

### Decode — TPOT

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 512 | 13.4079 | 74.6 |
| 1024 | 13.9649 | 71.6 |
| 4096 | 15.8902 | 62.9 |
| 8192 | 18.7711 | 53.3 |

### Measured vs. theoretical latency (lower bound = roofline)

The **theoretical** column is the sum over all kernels of
`max(bytes/BW, flops/peak_TFLOPS)` — i.e. the absolute minimum time
each kernel could take given its bound. `slowdown = measured / theo` (1.0× = at roofline).

#### Decode (per output token)

| KV (ctx) | Measured TPOT (ms) | Theoretical TPOT (ms) | Slowdown | Wall-clock tokens/s | Roofline tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 13.4079 | 11.1741 | 1.20× | 74.6 | 89.5 |
| 1024 | 13.9649 | 11.4786 | 1.22× | 71.6 | 87.1 |
| 4096 | 15.8902 | 13.3053 | 1.19× | 62.9 | 75.2 |
| 8192 | 18.7711 | 15.7408 | 1.19× | 53.3 | 63.5 |

#### Prefill / TTFT

| S (text ctx) | Measured TTFT (ms) | Theoretical TTFT (ms) | Slowdown | Meas tokens/s | Roofline tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 53.42 | 37.36 | 1.43× | 9585 | 13703 |
| 1024 | 73.72 | 51.17 | 1.44× | 13891 | 20013 |
| 4096 | 286.81 | 164.25 | 1.75× | 14281 | 24937 |
| 8192 | 790.97 | 378.20 | 2.09× | 10357 | 21661 |

#### Audio encoder fixed overhead

| Component | Measured (ms) | Theoretical (ms) | Slowdown |
|---|---:|---:|---:|
| Audio encoder forward (S=1500, 18 layers) | 27.02 | 12.72 | 2.12× |

### End-to-end latency estimate (output = 512 tokens)

Approximation: decode uses the TPOT at KV = S_prefill (start-of-decode value). Real decode TPOT increases as KV grows toward S_prefill + 511.

| S (text ctx) | TTFT (ms) | TPOT @ KV=S (ms) | Measured total (ms) | Theoretical total (ms) | Slowdown |
|---:|---:|---:|---:|---:|---:|
| 512 | 53.42 | 13.4079 | 6918.3 | 5758.5 | 1.20× |
| 1024 | 73.72 | 13.9649 | 7223.7 | 5928.2 | 1.22× |
| 4096 | 286.81 | 15.8902 | 8422.6 | 6976.6 | 1.21× |
| 8192 | 790.97 | 18.7711 | 10401.8 | 8437.5 | 1.23× |

## 7. Per-kernel tables — Decode (1 query token, KV = context)

### Decode — KV=512

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_lm_head | `gemm_kernel` | 1 | 3.1067 | 2.8316 | 3.1067 | 2.8316 | 1.10× | 100.2 | 100.3 | 91.1% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0854 | 0.0764 | 2.3918 | 2.1379 | 1.12× | 98.2 | 98.3 | 89.4% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0651 | 0.0573 | 1.8234 | 1.6036 | 1.14× | 96.6 | 96.7 | 87.9% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0647 | 0.0573 | 1.8128 | 1.6036 | 1.13× | 97.2 | 97.3 | 88.5% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0634 | 0.0573 | 1.7752 | 1.6036 | 1.11× | 99.2 | 99.4 | 90.3% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0422 | 0.0382 | 1.1812 | 1.0692 | 1.10× | 99.4 | 99.6 | 90.5% | memory |
| pa_compute_kv512 | `paged_attention_opt__single_token` | 28 | 0.0210 | 0.0109 | 0.5884 | 0.3065 | 1.92× | 199.6 | 57.3 | 52.1% | memory |
| pa_kv_update_kv512 | `pa_kv_cache_update_ref` | 28 | 0.0058 | 0.0001 | 0.1616 | 0.0016 | 98.73× | 0.0 | 1.1 | 1.0% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0027 | 0.0001 | 0.1526 | 0.0032 | 47.95× | 2.3 | 2.3 | 2.1% | memory |
| pa_finalize_kv512 | `single_token_finalization` | 28 | 0.0032 | 0.0001 | 0.0905 | 0.0021 | 43.39× | 0.0 | 2.5 | 2.3% | memory |
| residual_add | `eltwise` | 56 | 0.0014 | 0.0001 | 0.0768 | 0.0031 | 24.55× | 0.8 | 4.5 | 4.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0671 | 0.0016 | 42.93× | 2.6 | 2.6 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0669 | 0.0031 | 21.39× | 5.2 | 5.1 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0021 | 0.0001 | 0.0581 | 0.0022 | 26.21× | 3.0 | 4.2 | 3.8% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0548 | 0.0012 | 46.69× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **13.4079** | **11.1741** | **1.20×** |  |  |  |  |

### Decode — KV=1024

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_lm_head | `gemm_kernel` | 1 | 3.1067 | 2.8316 | 3.1067 | 2.8316 | 1.10× | 100.2 | 100.3 | 91.1% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0854 | 0.0764 | 2.3918 | 2.1379 | 1.12× | 98.2 | 98.3 | 89.4% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0651 | 0.0573 | 1.8234 | 1.6036 | 1.14× | 96.6 | 96.7 | 87.9% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0647 | 0.0573 | 1.8128 | 1.6036 | 1.13× | 97.2 | 97.3 | 88.5% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0634 | 0.0573 | 1.7752 | 1.6036 | 1.11× | 99.2 | 99.4 | 90.3% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0422 | 0.0382 | 1.1812 | 1.0692 | 1.10× | 99.4 | 99.6 | 90.5% | memory |
| pa_compute_kv1024 | `paged_attention_opt__single_token` | 28 | 0.0398 | 0.0218 | 1.1148 | 0.6110 | 1.82× | 210.7 | 60.3 | 54.8% | memory |
| pa_kv_update_kv1024 | `pa_kv_cache_update_ref` | 28 | 0.0063 | 0.0001 | 0.1757 | 0.0016 | 107.31× | 0.0 | 1.0 | 0.9% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0027 | 0.0001 | 0.1526 | 0.0032 | 47.95× | 2.3 | 2.3 | 2.1% | memory |
| pa_finalize_kv1024 | `single_token_finalization` | 28 | 0.0038 | 0.0001 | 0.1070 | 0.0021 | 51.29× | 0.0 | 2.1 | 1.9% | memory |
| residual_add | `eltwise` | 56 | 0.0014 | 0.0001 | 0.0768 | 0.0031 | 24.55× | 0.8 | 4.5 | 4.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0671 | 0.0016 | 42.93× | 2.6 | 2.6 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0669 | 0.0031 | 21.39× | 5.2 | 5.1 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0021 | 0.0001 | 0.0581 | 0.0022 | 26.21× | 3.0 | 4.2 | 3.8% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0548 | 0.0012 | 46.69× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **13.9649** | **11.4786** | **1.22×** |  |  |  |  |

### Decode — KV=4096

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_lm_head | `gemm_kernel` | 1 | 3.1067 | 2.8316 | 3.1067 | 2.8316 | 1.10× | 100.2 | 100.3 | 91.1% | memory |
| pa_compute_kv4096 | `paged_attention_opt__single_token` | 28 | 0.1060 | 0.0871 | 2.9685 | 2.4377 | 1.22× | 316.5 | 90.3 | 82.1% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0854 | 0.0764 | 2.3918 | 2.1379 | 1.12× | 98.2 | 98.3 | 89.4% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0651 | 0.0573 | 1.8234 | 1.6036 | 1.14× | 96.6 | 96.7 | 87.9% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0647 | 0.0573 | 1.8128 | 1.6036 | 1.13× | 97.2 | 97.3 | 88.5% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0634 | 0.0573 | 1.7752 | 1.6036 | 1.11× | 99.2 | 99.4 | 90.3% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0422 | 0.0382 | 1.1812 | 1.0692 | 1.10× | 99.4 | 99.6 | 90.5% | memory |
| pa_kv_update_kv4096 | `pa_kv_cache_update_ref` | 28 | 0.0066 | 0.0001 | 0.1852 | 0.0016 | 113.15× | 0.0 | 1.0 | 0.9% | memory |
| pa_finalize_kv4096 | `single_token_finalization` | 28 | 0.0060 | 0.0001 | 0.1691 | 0.0021 | 81.10× | 0.0 | 1.4 | 1.2% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0027 | 0.0001 | 0.1526 | 0.0032 | 47.95× | 2.3 | 2.3 | 2.1% | memory |
| residual_add | `eltwise` | 56 | 0.0014 | 0.0001 | 0.0768 | 0.0031 | 24.55× | 0.8 | 4.5 | 4.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0671 | 0.0016 | 42.93× | 2.6 | 2.6 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0669 | 0.0031 | 21.39× | 5.2 | 5.1 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0021 | 0.0001 | 0.0581 | 0.0022 | 26.21× | 3.0 | 4.2 | 3.8% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0548 | 0.0012 | 46.69× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **15.8902** | **13.3053** | **1.19×** |  |  |  |  |

### Decode — KV=8192

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv8192 | `paged_attention_opt__single_token` | 28 | 0.2032 | 0.1740 | 5.6893 | 4.8732 | 1.17× | 330.3 | 94.2 | 85.7% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 3.1067 | 2.8316 | 3.1067 | 2.8316 | 1.10× | 100.2 | 100.3 | 91.1% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0854 | 0.0764 | 2.3918 | 2.1379 | 1.12× | 98.2 | 98.3 | 89.4% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0651 | 0.0573 | 1.8234 | 1.6036 | 1.14× | 96.6 | 96.7 | 87.9% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0647 | 0.0573 | 1.8128 | 1.6036 | 1.13× | 97.2 | 97.3 | 88.5% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0634 | 0.0573 | 1.7752 | 1.6036 | 1.11× | 99.2 | 99.4 | 90.3% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0422 | 0.0382 | 1.1812 | 1.0692 | 1.10× | 99.4 | 99.6 | 90.5% | memory |
| pa_finalize_kv8192 | `single_token_finalization` | 28 | 0.0110 | 0.0001 | 0.3070 | 0.0021 | 147.22× | 0.0 | 0.7 | 0.7% | memory |
| pa_kv_update_kv8192 | `pa_kv_cache_update_ref` | 28 | 0.0074 | 0.0001 | 0.2074 | 0.0016 | 126.67× | 0.0 | 0.9 | 0.8% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0027 | 0.0001 | 0.1526 | 0.0032 | 47.95× | 2.3 | 2.3 | 2.1% | memory |
| residual_add | `eltwise` | 56 | 0.0014 | 0.0001 | 0.0768 | 0.0031 | 24.55× | 0.8 | 4.5 | 4.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0671 | 0.0016 | 42.93× | 2.6 | 2.6 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0669 | 0.0031 | 21.39× | 5.2 | 5.1 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0021 | 0.0001 | 0.0581 | 0.0022 | 26.21× | 3.0 | 4.2 | 3.8% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0548 | 0.0012 | 46.69× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **18.7711** | **15.7408** | **1.19×** |  |  |  |  |

## 8. Per-kernel tables — Prefill (text-decoder only; add encoder overhead for TTFT)

### Prefill — S=512 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_qkv | `gemm_kernel` | 28 | 0.1242 | 0.1239 | 3.4773 | 3.4698 | 1.00× | 34584.1 | 109.8 | 99.8% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 3.0989 | 2.8316 | 3.0989 | 2.8316 | 1.09× | 100.4 | 100.5 | 91.4% | memory |
| pa_compute_S512 | `sdpa_micro__prefill` | 28 | 0.1073 | 0.0572 | 3.0041 | 1.6015 | 1.88× | 10027.6 | 58.6 | 53.3% | memory |
| fc_down | `gemm_kernel` | 28 | 0.1071 | 0.0953 | 2.9988 | 2.6691 | 1.12× | 30076.8 | 97.9 | 89.0% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0947 | 0.0953 | 2.6527 | 2.6691 | 0.99× | 34001.1 | 110.7 | 100.6% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0943 | 0.0953 | 2.6394 | 2.6691 | 0.99× | 34171.7 | 111.2 | 101.1% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0742 | 0.0667 | 2.0773 | 1.8684 | 1.11× | 28945.7 | 98.9 | 89.9% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0482 | 0.0382 | 1.3508 | 1.0687 | 1.26× | 130.5 | 87.0 | 79.1% | memory |
| residual_add | `eltwise` | 56 | 0.0237 | 0.0286 | 1.3288 | 1.6015 | 0.83× | 22.1 | 132.6 | 120.5% | memory |
| pa_kv_update_S512 | `pa_kv_cache_update_ref` | 28 | 0.0341 | 0.0299 | 0.9553 | 0.8383 | 1.14× | 0.0 | 96.5 | 87.8% | memory |
| rope_q | `rope_opt` | 28 | 0.0294 | 0.0405 | 0.8223 | 1.1344 | 0.72× | 107.1 | 151.8 | 138.0% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0144 | 0.0191 | 0.8199 | 1.0878 | 0.75× | 219.0 | 145.9 | 132.7% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0242 | 0.0191 | 0.6779 | 0.5344 | 1.27× | 130.1 | 86.7 | 78.8% | memory |
| rope_k | `rope_opt` | 28 | 0.0179 | 0.0214 | 0.4999 | 0.6005 | 0.83× | 88.1 | 132.1 | 120.1% | memory |
| **TOTAL (text dec)** |  |  |  |  | **26.4034** | **24.6442** | **1.07×** |  |  |  |  |

### Prefill — S=1024 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S1024 | `sdpa_micro__prefill` | 28 | 0.3209 | 0.1144 | 8.9839 | 3.2029 | 2.80× | 13399.1 | 39.2 | 35.7% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.2058 | 0.1716 | 5.7637 | 4.8044 | 1.20× | 41729.9 | 91.7 | 83.4% | memory |
| fc_down | `gemm_kernel` | 28 | 0.1862 | 0.1335 | 5.2128 | 3.7367 | 1.40× | 34605.2 | 78.8 | 71.7% | memory |
| fc_up | `gemm_kernel` | 28 | 0.1454 | 0.1335 | 4.0711 | 3.7367 | 1.09× | 44309.7 | 101.0 | 91.8% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.1449 | 0.1335 | 4.0572 | 3.7367 | 1.09× | 44461.4 | 101.3 | 92.1% | memory |
| fc_o | `gemm_kernel` | 28 | 0.1339 | 0.0953 | 3.7488 | 2.6691 | 1.40× | 32079.3 | 78.3 | 71.2% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 3.0989 | 2.8316 | 3.0989 | 2.8316 | 1.09× | 100.4 | 100.5 | 91.4% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0931 | 0.0763 | 2.6080 | 2.1363 | 1.22× | 135.2 | 90.1 | 81.9% | memory |
| residual_add | `eltwise` | 56 | 0.0445 | 0.0572 | 2.4922 | 3.2029 | 0.78× | 23.6 | 141.4 | 128.5% | memory |
| pa_kv_update_S1024 | `pa_kv_cache_update_ref` | 28 | 0.0558 | 0.0599 | 1.5615 | 1.6765 | 0.93× | 0.0 | 118.1 | 107.4% | memory |
| rope_q | `rope_opt` | 28 | 0.0522 | 0.0810 | 1.4613 | 2.2687 | 0.64× | 120.5 | 170.8 | 155.2% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0248 | 0.0381 | 1.4141 | 2.1745 | 0.65× | 254.0 | 169.2 | 153.8% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0484 | 0.0381 | 1.3558 | 1.0682 | 1.27× | 130.1 | 86.7 | 78.8% | memory |
| rope_k | `rope_opt` | 28 | 0.0312 | 0.0429 | 0.8735 | 1.2011 | 0.73× | 100.8 | 151.3 | 137.5% | memory |
| **TOTAL (text dec)** |  |  |  |  | **46.7028** | **38.4463** | **1.21×** |  |  |  |  |

### Prefill — S=4096 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S4096 | `sdpa_micro__prefill` | 28 | 4.5149 | 1.1654 | 126.4169 | 32.6303 | 3.87× | 15224.3 | 11.2 | 25.8% | compute |
| fc_qkv | `gemm_kernel` | 28 | 0.7443 | 0.5825 | 20.8400 | 16.3112 | 1.28× | 46164.8 | 67.6 | 78.3% | compute |
| fc_down | `gemm_kernel` | 28 | 0.5772 | 0.4369 | 16.1610 | 12.2334 | 1.32× | 44648.0 | 69.0 | 75.7% | compute |
| fc_up | `gemm_kernel` | 28 | 0.5755 | 0.4369 | 16.1137 | 12.2334 | 1.32× | 44778.8 | 69.2 | 75.9% | compute |
| fc_gate | `gemm_kernel` | 28 | 0.5743 | 0.4369 | 16.0814 | 12.2334 | 1.31× | 44868.9 | 69.4 | 76.1% | compute |
| fc_o | `gemm_kernel` | 28 | 0.4112 | 0.2913 | 11.5145 | 8.1556 | 1.41× | 41776.6 | 71.4 | 70.8% | compute |
| residual_add | `eltwise` | 56 | 0.1974 | 0.2288 | 11.0572 | 12.8117 | 0.86× | 21.2 | 127.5 | 115.9% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.3872 | 0.3051 | 10.8416 | 8.5422 | 1.27× | 130.1 | 86.7 | 78.8% | memory |
| rope_q | `rope_opt` | 28 | 0.2671 | 0.3241 | 7.4802 | 9.0749 | 0.82× | 94.2 | 133.4 | 121.3% | memory |
| pa_kv_update_S4096 | `pa_kv_cache_update_ref` | 28 | 0.2255 | 0.2395 | 6.3135 | 6.7061 | 0.94× | 0.0 | 116.8 | 106.2% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0940 | 0.1525 | 5.3603 | 8.6947 | 0.62× | 268.0 | 178.4 | 162.2% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.1855 | 0.1525 | 5.1944 | 4.2711 | 1.22× | 135.9 | 90.5 | 82.2% | memory |
| rope_k | `rope_opt` | 28 | 0.1188 | 0.1716 | 3.3252 | 4.8044 | 0.69× | 106.0 | 158.9 | 144.5% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 3.0989 | 2.8316 | 3.0989 | 2.8316 | 1.09× | 100.4 | 100.5 | 91.4% | memory |
| **TOTAL (text dec)** |  |  |  |  | **259.7988** | **151.5340** | **1.71×** |  |  |  |  |

### Prefill — S=8192 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S8192 | `sdpa_micro__prefill` | 28 | 17.4613 | 4.6609 | 488.9156 | 130.5054 | 3.75× | 15744.1 | 5.8 | 26.7% | compute |
| fc_qkv | `gemm_kernel` | 28 | 1.5107 | 1.1651 | 42.3010 | 32.6224 | 1.30× | 45487.0 | 61.1 | 77.1% | compute |
| fc_up | `gemm_kernel` | 28 | 1.2233 | 0.8738 | 34.2532 | 24.4668 | 1.40× | 42130.6 | 60.0 | 71.4% | compute |
| fc_gate | `gemm_kernel` | 28 | 1.1884 | 0.8738 | 33.2751 | 24.4668 | 1.36× | 43369.0 | 61.8 | 73.5% | compute |
| fc_down | `gemm_kernel` | 28 | 1.0899 | 0.8738 | 30.5170 | 24.4668 | 1.25× | 47288.7 | 67.3 | 80.2% | compute |
| residual_add | `eltwise` | 56 | 0.4426 | 0.4576 | 24.7864 | 25.6234 | 0.97× | 18.9 | 113.7 | 103.4% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.7724 | 0.6101 | 21.6273 | 17.0833 | 1.27× | 130.4 | 86.9 | 79.0% | memory |
| fc_o | `gemm_kernel` | 28 | 0.7456 | 0.5825 | 20.8779 | 16.3112 | 1.28× | 46080.8 | 73.1 | 78.1% | compute |
| rope_q | `rope_opt` | 28 | 0.6339 | 0.6482 | 17.7493 | 18.1499 | 0.98× | 79.4 | 112.5 | 102.3% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.2462 | 0.3051 | 14.0347 | 17.3884 | 0.81× | 204.8 | 136.3 | 123.9% | memory |
| pa_kv_update_S8192 | `pa_kv_cache_update_ref` | 28 | 0.4736 | 0.4790 | 13.2598 | 13.4123 | 0.99× | 0.0 | 111.3 | 101.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.3853 | 0.3051 | 10.7882 | 8.5417 | 1.26× | 130.8 | 87.1 | 79.2% | memory |
| rope_k | `rope_opt` | 28 | 0.3026 | 0.3432 | 8.4733 | 9.6088 | 0.88× | 83.2 | 124.7 | 113.4% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 3.0989 | 2.8316 | 3.0989 | 2.8316 | 1.09× | 100.4 | 100.5 | 91.4% | memory |
| **TOTAL (text dec)** |  |  |  |  | **763.9577** | **365.4788** | **2.09×** |  |  |  |  |

## 9. Roofline analysis

Ridge point at PTL 12Xe FP16 = 536 FLOP/byte. AI < ridge ⇒ memory-bound; AI ≥ ridge ⇒ compute-bound.

**Decode characteristics**: Every FC has AI = 2·M·K·N / (K·N·2) → ~M = 1 for decode, well below ridge → **always memory-bound**. PA decode AI is also well below ridge.

**Prefill characteristics**: FC AI ≈ M scales with sequence length. For PTL 12Xe (ridge = 536), FC becomes compute-bound roughly at M ≥ ridge. SDPA prefill AI scales with S/2; becomes compute-bound around S ≥ 2·ridge.

### Bottleneck breakdown — Decode (top 5 ops by latency)

| KV | #1 | #2 | #3 | #4 | #5 |
|---:|---|---|---|---|---|
| 512 | fc_lm_head (3.107ms / 23.2%) | fc_qkv (2.392ms / 17.8%) | fc_up (1.823ms / 13.6%) | fc_gate (1.813ms / 13.5%) | fc_down (1.775ms / 13.2%) |
| 1024 | fc_lm_head (3.107ms / 22.2%) | fc_qkv (2.392ms / 17.1%) | fc_up (1.823ms / 13.1%) | fc_gate (1.813ms / 13.0%) | fc_down (1.775ms / 12.7%) |
| 4096 | fc_lm_head (3.107ms / 19.6%) | pa_compute_kv4096 (2.969ms / 18.7%) | fc_qkv (2.392ms / 15.1%) | fc_up (1.823ms / 11.5%) | fc_gate (1.813ms / 11.4%) |
| 8192 | pa_compute_kv8192 (5.689ms / 30.3%) | fc_lm_head (3.107ms / 16.6%) | fc_qkv (2.392ms / 12.7%) | fc_up (1.823ms / 9.7%) | fc_gate (1.813ms / 9.7%) |

### Bottleneck breakdown — Prefill (top 5 ops by latency)

| S | #1 | #2 | #3 | #4 | #5 |
|---:|---|---|---|---|---|
| 512 | fc_qkv (3.477ms / 13.2%) | fc_lm_head (3.099ms / 11.7%) | pa_compute_S512 (3.004ms / 11.4%) | fc_down (2.999ms / 11.4%) | fc_gate (2.653ms / 10.0%) |
| 1024 | pa_compute_S1024 (8.984ms / 19.2%) | fc_qkv (5.764ms / 12.3%) | fc_down (5.213ms / 11.2%) | fc_up (4.071ms / 8.7%) | fc_gate (4.057ms / 8.7%) |
| 4096 | pa_compute_S4096 (126.417ms / 48.7%) | fc_qkv (20.840ms / 8.0%) | fc_down (16.161ms / 6.2%) | fc_up (16.114ms / 6.2%) | fc_gate (16.081ms / 6.2%) |
| 8192 | pa_compute_S8192 (488.916ms / 64.0%) | fc_qkv (42.301ms / 5.5%) | fc_up (34.253ms / 4.5%) | fc_gate (33.275ms / 4.4%) | fc_down (30.517ms / 4.0%) |

## 10. Notes, caveats & reproduction

- **Audio encoder SDPA** is approximated by 2× the measured PA-prefill (causal) kernel at S=1500. Real encoder is bidirectional non-causal; a dedicated non-causal SDPA bench would be needed for a more precise number. Encoder SDPA is a small fraction of overall TTFT, so this approximation has limited impact.
- **FC LM_Head** is `K=1024, N=151936` FP16 (~297 MB weight stream). This single layer is the dominant decode op (~50% of TPOT) and also dominates prefill last-token cost; INT8/INT4 LM_Head would substantially reduce it.
- **PA decode** dispatches `paged_attention_opt__single_token` across all measured KV sizes (512–8192). On some other Qwen variants the runtime promotes to `paged_attention_opt__gqa_single_token` at long context; that did not trigger in this configuration.
- **KV cache precision = I8**: the Intel GPU plugin defaults `kv_cache_precision` to `i8` for PagedAttention models (`src/plugins/intel_gpu/src/runtime/execution_config.cpp:293`). This run matches that default. PA cache layout is `K = [num_blocks, NKV, HD, BLOCK+4]` (BY_CHANNEL: 1 byte per element + 4-byte scale/zp shared across BLOCK=16 tokens) and `V = [num_blocks, NKV, BLOCK, HD+4]` (BY_TOKEN: 1 byte per element + 4-byte scale/zp shared across HD per token). The theoretical bytes formula above accounts for both the INT8 payload and the scale/zp overhead, so eff% is measured against what the kernel actually streams. Switching to FP16 KV roughly doubles PA-compute bytes (and so doubles TPOT contribution from PA at long context); switching to INT4 KV would roughly halve them again, at the cost of accuracy. Observed PA-compute decode eff: 52.1% / 54.8% / 82.1% / 85.7% for KV=512/1024/4096/8192 — monotonically increasing with KV, as expected (larger working set amortizes per-launch overhead). The `KV cache key type=...` header line in every pa_bench log exposes the compiled cache element type; the `,20` channel dim is the i8 BY_CHANNEL signature (16 BLOCK + 4 scale/zp).
- **PA prefill** S=8192 produces ~17 ms of `sdpa_micro__prefill` per layer × 28 layers = ~503 ms — by far the dominant prefill cost at long context. This is compute-bound (AI » ridge), so reducing weight precision will not help here; flash-attention–style algorithmic improvements or INT8 attention math would.
- **Encoder weights** count ~89 MB and run only once: amortized over many output tokens the encoder is a small per-token cost.

### Reproduction

```bat
REM On PTL 12Xe Windows (Local_Admin@10.239.132.229):
D:\river\moe\dev_roofline_profiling\utils\run_qwen3_asr_0_6B_ptl_12xe.bat
```

```bash
# On local Linux box:
scp -r Local_Admin@<win>:/D/river/moe/roofline_results/qwen3_asr_0_6B/ptl_12xe/*.log \
    .github/skills/dev_roofline_profiling/outputs/qwen3_asr_0_6B/logs_ptl_12xe/
cd .github/skills/dev_roofline_profiling/outputs/qwen3_asr_0_6B
python3 ../../utils/parse_logs.py logs_ptl_12xe parsed.json
python3 build_report.py
```
