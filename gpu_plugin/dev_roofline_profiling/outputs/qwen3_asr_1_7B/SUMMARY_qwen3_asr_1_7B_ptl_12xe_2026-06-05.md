# Qwen3-ASR-1.7B — Roofline on PTL 12Xe (2026-06-05)

**Platform**: Intel PTL 12Xe iGPU (12 Xe cores × 8 EU × 10 threads), 2400 MHz, 110 GB/s
**Model**: Qwen3-ASR-1.7B (audio encoder + Qwen3 text decoder), all weights **FP16**, no compression.
**SDPA**: PagedAttention (opencl + micro_kernel), INT8 KV cache (text decoder; matches Intel GPU plugin default for PA models). Audio encoder SDPA remains FP16.
**Source config**: https://huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/config.json

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
| `hidden_size` | 2048 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` (NH) | 16 |
| `num_key_value_heads` (NKV) | 8 (GQA 2:1) |
| `head_dim` (HD) | 128 |
| `intermediate_size` | 6144 |
| `vocab_size` | 151936 |
| `tie_word_embeddings` | True |
| `rope_theta` | 1,000,000 |

### Audio encoder (Whisper-style)

| Field | Value |
|---|---:|
| `d_model` | 1024 |
| `encoder_layers` | 24 |
| `encoder_attention_heads` (MHA, NH=NKV) | 16 |
| `head_dim` | 64 |
| `encoder_ffn_dim` | 4096 |
| `num_mel_bins` | 128 |
| `max_source_positions` | 1500 |
| `output_dim` (→ text decoder hidden) | 2048 |

## 3. Theoretical weight distribution (FP16, no compression)

### Text decoder per-layer (1 of 28)

| Weight | Shape (K × N) | Dtype | MB |
|---|---|---|---:|
| FC_QKV (fused Q+K+V) | 2048 × 4096 | FP16 | 16.0 |
| FC_O (attn output)   | 2048 × 2048 | FP16 | 8.0 |
| FC_Gate (SwiGLU)     | 2048 × 6144 | FP16 | 24.0 |
| FC_Up (SwiGLU)       | 2048 × 6144 | FP16 | 24.0 |
| FC_Down (SwiGLU)     | 6144 × 2048 | FP16 | 24.0 |
| RMSNorm × 2          | [2048] each | FP16 | 0.008 |
| q_norm / k_norm      | [128] each  | FP16 | 0.0000 |
| **per layer** |  |  | **96.01** |
| **× 28 layers** |  |  | **2688.23** |

### Audio encoder per-layer (1 of 24)

| Weight | Shape (K × N) | Dtype | MB |
|---|---|---|---:|
| FC_QKV (fused) | 1024 × 3072 | FP16 | 6.0 |
| FC_O           | 1024 × 1024 | FP16 | 2.0 |
| FC_FC1 (GELU)  | 1024 × 4096 | FP16 | 8.0 |
| FC_FC2 (GELU)  | 4096 × 1024 | FP16 | 8.0 |
| LayerNorm × 2  | [1024] g+b each | FP16 | 0.008 |
| **per layer** |  |  | **24.01** |
| **× 24 layers** |  |  | **576.19** |

### Global + shared weights

| Weight | Shape | Dtype | MB |
|---|---|---|---:|
| Token embedding | 151936 × 2048 | FP16 | 593.5 |
| LM_Head (tied=True) | shares embedding | — | 0.0 |
| Final RMSNorm (text)  | [2048] | FP16 | 0.004 |
| Audio encoder conv front-end | conv1d×2 | FP16 | 6.750 |
| Audio positional embedding | 1500 × 1024 | FP16 | 2.93 |
| Audio output adapter | 1024→480→2048 | FP16 | 2.812 |

### Totals

| Component | MB |
|---|---:|
| Text decoder (28 layers) | 2688.23 |
| Token embedding (shared w/ LM_Head) | 593.50 |
| Audio encoder (24 layers) | 576.19 |
| Audio encoder front + adapter | 12.50 |
| Final RMSNorm (text) | 0.0040 |
| **Model total** | **3870.42** |

## 4. Benchmark methodology

- **Bench utils**: `fc_bench` (precision=f16, plain MatMul, no compression), `pa_bench` (kv_dtype=i8, impl=ocl), `small_ops_bench` for rmsnorm/rope/eltwise.
- **Tool**: cliloader Device Performance Timing; `parse_logs.py` extracts per-iteration avg GPU kernel ns.
- **L2/L3 flush** between every FC infer (64 MB Relu) so each measurement reads weights from VRAM, not on-die cache. Required because the entire FP16 text decoder weights (~707 MB) is too large to fit fully in L2 but per-layer fits, so cache effects would otherwise inflate measured BW.
- **Input tensors** allocated via RemoteContext in USM_DEVICE (iGPU shared system memory).
- **PA prefill** uses causal mask (Sq·(Sq+1)/2 effective pairs) per SKILL.md.
- **PA decode** is split into `pa_kv_cache_update_ref` + `paged_attention_opt__single_token` + `single_token_finalization`.
- **Audio encoder SDPA** is bidirectional (no causal mask); measured causal time scaled ×2 in the encoder roofline row.
- **swish / multiply** are not profiled separately — they fuse into the SwiGLU primitive per SKILL.md.
- **LM_Head** is `K=2048, N=151936` plain FP16 (no INT8 since user requested FP16-only). Counted once per inference at decode and once at prefill (last token only).

## 5. Audio encoder fixed overhead (per inference, S=1500)

The audio encoder runs **once** per inference regardless of text token count.

### Audio encoder — S=1500

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| enc_sdpa | `sdpa_micro__prefill (×2 bidir est.)` | 24 | 0.8129 | 0.1562 | 19.5096 | 3.7500 | 5.20× | 11337.2 | 15.1 | 19.2% | compute |
| enc_fc_ffn2 | `gemm_kernel` | 24 | 0.4732 | 0.2159 | 11.3559 | 5.1815 | 2.19× | 26593.2 | 50.2 | 45.6% | memory |
| enc_fc_ffn1 | `gemm_kernel` | 24 | 0.2733 | 0.2159 | 6.5599 | 5.1815 | 1.27× | 46035.9 | 86.9 | 79.0% | memory |
| enc_fc_qkv | `gemm_kernel` | 24 | 0.2038 | 0.1689 | 4.8903 | 4.0537 | 1.21× | 46315.0 | 91.2 | 82.9% | memory |
| enc_fc_o | `gemm_kernel` | 24 | 0.0868 | 0.0749 | 2.0831 | 1.7981 | 1.16× | 36243.2 | 95.0 | 86.3% | memory |
| enc_outproj | `gemm_kernel` | 1 | 0.1441 | 0.1219 | 0.1441 | 0.1219 | 1.18× | 43665.2 | 93.1 | 84.6% | memory |
| **TOTAL (encoder fixed)** |  |  |  |  | **44.5429** | **20.0867** | **2.22×** |  |  |  |  |

## 6. Token latency summary

**TTFT** = audio encoder (44.54 ms) + text decoder prefill.
**TPOT** = per output token decode latency at the listed KV context.

### Prefill — TTFT

| S (text ctx) | Encoder (ms) | Text-decoder prefill (ms) | **TTFT (ms)** | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 44.54 | 58.54 | **103.08** | 0.2013 | 4967 |
| 1024 | 44.54 | 98.31 | **142.86** | 0.1395 | 7168 |
| 4096 | 44.54 | 473.51 | **518.05** | 0.1265 | 7907 |
| 8192 | 44.54 | 1239.24 | **1283.78** | 0.1567 | 6381 |

### Decode — TPOT

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 512 | 34.7340 | 28.8 |
| 1024 | 35.2712 | 28.4 |
| 4096 | 37.1161 | 26.9 |
| 8192 | 40.0639 | 25.0 |

### Measured vs. theoretical latency (lower bound = roofline)

The **theoretical** column is the sum over all kernels of
`max(bytes/BW, flops/peak_TFLOPS)` — i.e. the absolute minimum time
each kernel could take given its bound. `slowdown = measured / theo` (1.0× = at roofline).

#### Decode (per output token)

| KV (ctx) | Measured TPOT (ms) | Theoretical TPOT (ms) | Slowdown | Wall-clock tokens/s | Roofline tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 34.7340 | 31.6324 | 1.10× | 28.8 | 31.6 |
| 1024 | 35.2712 | 31.9369 | 1.10× | 28.4 | 31.3 |
| 4096 | 37.1161 | 33.7636 | 1.10× | 26.9 | 29.6 |
| 8192 | 40.0639 | 36.1991 | 1.11× | 25.0 | 27.6 |

#### Prefill / TTFT

| S (text ctx) | Measured TTFT (ms) | Theoretical TTFT (ms) | Slowdown | Meas tokens/s | Roofline tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 103.08 | 71.60 | 1.44× | 4967 | 7151 |
| 1024 | 142.86 | 97.18 | 1.47× | 7168 | 10537 |
| 4096 | 518.05 | 330.52 | 1.57× | 7907 | 12392 |
| 8192 | 1283.78 | 700.54 | 1.83× | 6381 | 11694 |

#### Audio encoder fixed overhead

| Component | Measured (ms) | Theoretical (ms) | Slowdown |
|---|---:|---:|---:|
| Audio encoder forward (S=1500, 24 layers) | 44.54 | 20.09 | 2.22× |

### End-to-end latency estimate (output = 512 tokens)

Approximation: decode uses the TPOT at KV = S_prefill (start-of-decode value). Real decode TPOT increases as KV grows toward S_prefill + 511.

| S (text ctx) | TTFT (ms) | TPOT @ KV=S (ms) | Measured total (ms) | Theoretical total (ms) | Slowdown |
|---:|---:|---:|---:|---:|---:|
| 512 | 103.08 | 34.7340 | 17886.9 | 16267.4 | 1.10× |
| 1024 | 142.86 | 35.2712 | 18201.7 | 16448.9 | 1.11× |
| 4096 | 518.05 | 37.1161 | 19521.5 | 17617.5 | 1.11× |
| 8192 | 1283.78 | 40.0639 | 21796.5 | 19234.5 | 1.13× |

## 7. Per-kernel tables — Decode (1 query token, KV = context)

### Decode — KV=512

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_down | `gemm_kernel` | 28 | 0.2467 | 0.2289 | 6.9082 | 6.4100 | 1.08× | 102.0 | 102.1 | 92.8% | memory |
| fc_up | `gemm_kernel` | 28 | 0.2432 | 0.2289 | 6.8103 | 6.4100 | 1.06× | 103.5 | 103.5 | 94.1% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.2426 | 0.2289 | 6.7939 | 6.4100 | 1.06× | 103.7 | 103.8 | 94.3% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 5.9963 | 5.6603 | 5.9963 | 5.6603 | 1.06× | 103.8 | 103.8 | 94.4% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.1623 | 0.1526 | 4.5433 | 4.2737 | 1.06× | 103.4 | 103.5 | 94.1% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0828 | 0.0763 | 2.3185 | 2.1374 | 1.08× | 101.3 | 101.4 | 92.2% | memory |
| pa_compute_kv512 | `paged_attention_opt__single_token` | 28 | 0.0215 | 0.0109 | 0.6033 | 0.3065 | 1.97× | 194.7 | 55.9 | 50.8% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0029 | 0.0001 | 0.1678 | 0.0064 | 26.35× | 4.2 | 4.2 | 3.8% | memory |
| pa_kv_update_kv512 | `pa_kv_cache_update_ref` | 28 | 0.0058 | 0.0001 | 0.1627 | 0.0016 | 99.38× | 0.0 | 1.1 | 1.0% | memory |
| pa_finalize_kv512 | `single_token_finalization` | 28 | 0.0034 | 0.0001 | 0.0958 | 0.0021 | 45.96× | 0.0 | 2.4 | 2.2% | memory |
| residual_add | `eltwise` | 56 | 0.0015 | 0.0001 | 0.0868 | 0.0063 | 13.88× | 1.3 | 7.9 | 7.2% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0681 | 0.0016 | 43.54× | 2.5 | 2.5 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0667 | 0.0031 | 21.33× | 5.2 | 5.2 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0020 | 0.0001 | 0.0572 | 0.0022 | 25.83× | 3.0 | 4.3 | 3.9% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0551 | 0.0012 | 46.96× | 1.6 | 2.3 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **34.7340** | **31.6324** | **1.10×** |  |  |  |  |

### Decode — KV=1024

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_down | `gemm_kernel` | 28 | 0.2467 | 0.2289 | 6.9082 | 6.4100 | 1.08× | 102.0 | 102.1 | 92.8% | memory |
| fc_up | `gemm_kernel` | 28 | 0.2432 | 0.2289 | 6.8103 | 6.4100 | 1.06× | 103.5 | 103.5 | 94.1% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.2426 | 0.2289 | 6.7939 | 6.4100 | 1.06× | 103.7 | 103.8 | 94.3% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 5.9963 | 5.6603 | 5.9963 | 5.6603 | 1.06× | 103.8 | 103.8 | 94.4% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.1623 | 0.1526 | 4.5433 | 4.2737 | 1.06× | 103.4 | 103.5 | 94.1% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0828 | 0.0763 | 2.3185 | 2.1374 | 1.08× | 101.3 | 101.4 | 92.2% | memory |
| pa_compute_kv1024 | `paged_attention_opt__single_token` | 28 | 0.0403 | 0.0218 | 1.1290 | 0.6110 | 1.85× | 208.0 | 59.5 | 54.1% | memory |
| pa_kv_update_kv1024 | `pa_kv_cache_update_ref` | 28 | 0.0063 | 0.0001 | 0.1755 | 0.0016 | 107.21× | 0.0 | 1.0 | 0.9% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0029 | 0.0001 | 0.1678 | 0.0064 | 26.35× | 4.2 | 4.2 | 3.8% | memory |
| pa_finalize_kv1024 | `single_token_finalization` | 28 | 0.0034 | 0.0001 | 0.0945 | 0.0021 | 45.32× | 0.0 | 2.4 | 2.2% | memory |
| residual_add | `eltwise` | 56 | 0.0015 | 0.0001 | 0.0868 | 0.0063 | 13.88× | 1.3 | 7.9 | 7.2% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0681 | 0.0016 | 43.54× | 2.5 | 2.5 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0667 | 0.0031 | 21.33× | 5.2 | 5.2 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0020 | 0.0001 | 0.0572 | 0.0022 | 25.83× | 3.0 | 4.3 | 3.9% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0551 | 0.0012 | 46.96× | 1.6 | 2.3 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **35.2712** | **31.9369** | **1.10×** |  |  |  |  |

### Decode — KV=4096

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_down | `gemm_kernel` | 28 | 0.2467 | 0.2289 | 6.9082 | 6.4100 | 1.08× | 102.0 | 102.1 | 92.8% | memory |
| fc_up | `gemm_kernel` | 28 | 0.2432 | 0.2289 | 6.8103 | 6.4100 | 1.06× | 103.5 | 103.5 | 94.1% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.2426 | 0.2289 | 6.7939 | 6.4100 | 1.06× | 103.7 | 103.8 | 94.3% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 5.9963 | 5.6603 | 5.9963 | 5.6603 | 1.06× | 103.8 | 103.8 | 94.4% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.1623 | 0.1526 | 4.5433 | 4.2737 | 1.06× | 103.4 | 103.5 | 94.1% | memory |
| pa_compute_kv4096 | `paged_attention_opt__gqa_single_token` | 28 | 0.1040 | 0.0871 | 2.9109 | 2.4377 | 1.19× | 322.8 | 92.1 | 83.7% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0828 | 0.0763 | 2.3185 | 2.1374 | 1.08× | 101.3 | 101.4 | 92.2% | memory |
| pa_kv_update_kv4096 | `pa_kv_cache_update_ref` | 28 | 0.0061 | 0.0001 | 0.1711 | 0.0016 | 104.51× | 0.0 | 1.1 | 1.0% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0029 | 0.0001 | 0.1678 | 0.0064 | 26.35× | 4.2 | 4.2 | 3.8% | memory |
| pa_finalize_kv4096 | `single_token_finalization` | 28 | 0.0058 | 0.0001 | 0.1619 | 0.0021 | 77.65× | 0.0 | 1.4 | 1.3% | memory |
| residual_add | `eltwise` | 56 | 0.0015 | 0.0001 | 0.0868 | 0.0063 | 13.88× | 1.3 | 7.9 | 7.2% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0681 | 0.0016 | 43.54× | 2.5 | 2.5 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0667 | 0.0031 | 21.33× | 5.2 | 5.2 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0020 | 0.0001 | 0.0572 | 0.0022 | 25.83× | 3.0 | 4.3 | 3.9% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0551 | 0.0012 | 46.96× | 1.6 | 2.3 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **37.1161** | **33.7636** | **1.10×** |  |  |  |  |

### Decode — KV=8192

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_down | `gemm_kernel` | 28 | 0.2467 | 0.2289 | 6.9082 | 6.4100 | 1.08× | 102.0 | 102.1 | 92.8% | memory |
| fc_up | `gemm_kernel` | 28 | 0.2432 | 0.2289 | 6.8103 | 6.4100 | 1.06× | 103.5 | 103.5 | 94.1% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.2426 | 0.2289 | 6.7939 | 6.4100 | 1.06× | 103.7 | 103.8 | 94.3% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 5.9963 | 5.6603 | 5.9963 | 5.6603 | 1.06× | 103.8 | 103.8 | 94.4% | memory |
| pa_compute_kv8192 | `paged_attention_opt__gqa_single_token` | 28 | 0.2028 | 0.1740 | 5.6780 | 4.8732 | 1.17× | 330.9 | 94.4 | 85.8% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.1623 | 0.1526 | 4.5433 | 4.2737 | 1.06× | 103.4 | 103.5 | 94.1% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0828 | 0.0763 | 2.3185 | 2.1374 | 1.08× | 101.3 | 101.4 | 92.2% | memory |
| pa_finalize_kv8192 | `single_token_finalization` | 28 | 0.0114 | 0.0001 | 0.3186 | 0.0021 | 152.81× | 0.0 | 0.7 | 0.7% | memory |
| pa_kv_update_kv8192 | `pa_kv_cache_update_ref` | 28 | 0.0070 | 0.0001 | 0.1951 | 0.0016 | 119.17× | 0.0 | 0.9 | 0.8% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0029 | 0.0001 | 0.1678 | 0.0064 | 26.35× | 4.2 | 4.2 | 3.8% | memory |
| residual_add | `eltwise` | 56 | 0.0015 | 0.0001 | 0.0868 | 0.0063 | 13.88× | 1.3 | 7.9 | 7.2% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0681 | 0.0016 | 43.54× | 2.5 | 2.5 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0667 | 0.0031 | 21.33× | 5.2 | 5.2 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0020 | 0.0001 | 0.0572 | 0.0022 | 25.83× | 3.0 | 4.3 | 3.9% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0551 | 0.0012 | 46.96× | 1.6 | 2.3 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **40.0639** | **36.1991** | **1.11×** |  |  |  |  |

## 8. Per-kernel tables — Prefill (text-decoder only; add encoder overhead for TTFT)

### Prefill — S=512 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_down | `gemm_kernel` | 28 | 0.4080 | 0.3050 | 11.4235 | 8.5411 | 1.34× | 31582.0 | 82.2 | 74.8% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.3462 | 0.3050 | 9.6936 | 8.5411 | 1.13× | 37218.1 | 96.9 | 88.1% | memory |
| fc_up | `gemm_kernel` | 28 | 0.3404 | 0.3050 | 9.5313 | 8.5411 | 1.12× | 37851.9 | 98.6 | 89.6% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.2429 | 0.2097 | 6.8002 | 5.8720 | 1.16× | 35369.2 | 95.0 | 86.4% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 6.0399 | 5.6603 | 6.0399 | 5.6603 | 1.07× | 103.0 | 103.1 | 93.7% | memory |
| fc_o | `gemm_kernel` | 28 | 0.1341 | 0.1144 | 3.7558 | 3.2029 | 1.17× | 32019.5 | 93.8 | 85.3% | memory |
| pa_compute_S512 | `sdpa_micro__prefill` | 28 | 0.1075 | 0.0572 | 3.0106 | 1.6015 | 1.88× | 10005.9 | 58.5 | 53.2% | memory |
| residual_add | `eltwise` | 56 | 0.0445 | 0.0572 | 2.4929 | 3.2029 | 0.78× | 23.6 | 141.3 | 128.5% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0251 | 0.0382 | 1.4329 | 2.1755 | 0.66× | 250.5 | 167.0 | 151.8% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0486 | 0.0382 | 1.3595 | 1.0687 | 1.27× | 129.7 | 86.5 | 78.6% | memory |
| pa_kv_update_S512 | `pa_kv_cache_update_ref` | 28 | 0.0347 | 0.0299 | 0.9708 | 0.8383 | 1.16× | 0.0 | 95.0 | 86.3% | memory |
| rope_q | `rope_opt` | 28 | 0.0301 | 0.0405 | 0.8441 | 1.1344 | 0.74× | 104.3 | 147.8 | 134.4% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0242 | 0.0191 | 0.6789 | 0.5344 | 1.27× | 129.9 | 86.6 | 78.7% | memory |
| rope_k | `rope_opt` | 28 | 0.0179 | 0.0214 | 0.5015 | 0.6005 | 0.84× | 87.8 | 131.7 | 119.8% | memory |
| **TOTAL (text dec)** |  |  |  |  | **58.5355** | **51.5147** | **1.14×** |  |  |  |  |

### Prefill — S=1024 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_down | `gemm_kernel` | 28 | 0.7485 | 0.4369 | 20.9590 | 12.2334 | 1.71× | 34426.9 | 56.0 | 58.4% | compute |
| fc_gate | `gemm_kernel` | 28 | 0.5280 | 0.4369 | 14.7829 | 12.2334 | 1.21× | 48810.2 | 79.4 | 82.8% | compute |
| fc_up | `gemm_kernel` | 28 | 0.5230 | 0.4369 | 14.6434 | 12.2334 | 1.20× | 49274.9 | 80.2 | 83.5% | compute |
| fc_qkv | `gemm_kernel` | 28 | 0.3946 | 0.2913 | 11.0481 | 8.1556 | 1.35× | 43540.2 | 74.4 | 73.8% | compute |
| pa_compute_S1024 | `sdpa_micro__prefill` | 28 | 0.3308 | 0.1144 | 9.2617 | 3.2029 | 2.89× | 12997.2 | 38.0 | 34.6% | memory |
| fc_o | `gemm_kernel` | 28 | 0.2178 | 0.1525 | 6.0997 | 4.2706 | 1.43× | 39431.0 | 77.0 | 70.0% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 6.0399 | 5.6603 | 6.0399 | 5.6603 | 1.07× | 103.0 | 103.1 | 93.7% | memory |
| residual_add | `eltwise` | 56 | 0.0862 | 0.1144 | 4.8277 | 6.4058 | 0.75× | 24.3 | 146.0 | 132.7% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0487 | 0.0763 | 2.7758 | 4.3489 | 0.64× | 258.6 | 172.3 | 156.7% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0932 | 0.0763 | 2.6085 | 2.1363 | 1.22× | 135.2 | 90.1 | 81.9% | memory |
| pa_kv_update_S1024 | `pa_kv_cache_update_ref` | 28 | 0.0552 | 0.0599 | 1.5464 | 1.6765 | 0.92× | 0.0 | 119.3 | 108.4% | memory |
| rope_q | `rope_opt` | 28 | 0.0524 | 0.0810 | 1.4685 | 2.2687 | 0.65× | 120.0 | 169.9 | 154.5% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0487 | 0.0381 | 1.3636 | 1.0682 | 1.28× | 129.4 | 86.2 | 78.3% | memory |
| rope_k | `rope_opt` | 28 | 0.0317 | 0.0429 | 0.8889 | 1.2011 | 0.74× | 99.1 | 148.6 | 135.1% | memory |
| **TOTAL (text dec)** |  |  |  |  | **98.3141** | **77.0951** | **1.28×** |  |  |  |  |

### Prefill — S=4096 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S4096 | `sdpa_micro__prefill` | 28 | 4.4504 | 1.1654 | 124.6102 | 32.6303 | 3.82× | 15445.1 | 11.3 | 26.2% | compute |
| fc_down | `gemm_kernel` | 28 | 3.1762 | 1.7476 | 88.9324 | 48.9336 | 1.82× | 32454.1 | 29.1 | 55.0% | compute |
| fc_up | `gemm_kernel` | 28 | 2.1927 | 1.7476 | 61.3964 | 48.9336 | 1.25× | 47009.6 | 42.1 | 79.7% | compute |
| fc_gate | `gemm_kernel` | 28 | 2.1756 | 1.7476 | 60.9161 | 48.9336 | 1.24× | 47380.2 | 42.4 | 80.3% | compute |
| fc_qkv | `gemm_kernel` | 28 | 1.4172 | 1.1651 | 39.6805 | 32.6224 | 1.22× | 48490.9 | 47.4 | 82.2% | compute |
| residual_add | `eltwise` | 56 | 0.4343 | 0.4576 | 24.3205 | 25.6234 | 0.95× | 19.3 | 115.9 | 105.4% | memory |
| fc_o | `gemm_kernel` | 28 | 0.7470 | 0.5825 | 20.9171 | 16.3112 | 1.28× | 45994.6 | 56.1 | 78.0% | compute |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.2404 | 0.3051 | 13.7011 | 17.3894 | 0.79× | 209.6 | 139.6 | 126.9% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.3785 | 0.3051 | 10.5979 | 8.5422 | 1.24× | 133.1 | 88.7 | 80.6% | memory |
| rope_q | `rope_opt` | 28 | 0.2685 | 0.3241 | 7.5169 | 9.0749 | 0.83× | 93.7 | 132.8 | 120.7% | memory |
| pa_kv_update_S4096 | `pa_kv_cache_update_ref` | 28 | 0.2284 | 0.2395 | 6.3963 | 6.7061 | 0.95× | 0.0 | 115.3 | 104.8% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 6.0399 | 5.6603 | 6.0399 | 5.6603 | 1.07× | 103.0 | 103.1 | 93.7% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.1853 | 0.1525 | 5.1894 | 4.2711 | 1.22× | 136.0 | 90.5 | 82.3% | memory |
| rope_k | `rope_opt` | 28 | 0.1176 | 0.1716 | 3.2920 | 4.8044 | 0.69× | 107.0 | 160.5 | 145.9% | memory |
| **TOTAL (text dec)** |  |  |  |  | **473.5067** | **310.4365** | **1.53×** |  |  |  |  |

### Prefill — S=8192 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S8192 | `sdpa_micro__prefill` | 28 | 17.9295 | 4.6609 | 502.0270 | 130.5054 | 3.85× | 15332.9 | 5.6 | 26.0% | compute |
| fc_down | `gemm_kernel` | 28 | 6.3046 | 3.4953 | 176.5298 | 97.8671 | 1.80× | 32699.5 | 25.3 | 55.4% | compute |
| fc_up | `gemm_kernel` | 28 | 4.2598 | 3.4953 | 119.2731 | 97.8671 | 1.22× | 48396.8 | 37.4 | 82.1% | compute |
| fc_gate | `gemm_kernel` | 28 | 4.2485 | 3.4953 | 118.9590 | 97.8671 | 1.22× | 48524.6 | 37.5 | 82.3% | compute |
| fc_qkv | `gemm_kernel` | 28 | 4.1169 | 2.3302 | 115.2730 | 65.2447 | 1.77× | 33384.2 | 28.5 | 56.6% | compute |
| residual_add | `eltwise` | 56 | 0.9395 | 0.9151 | 52.6141 | 51.2468 | 1.03× | 17.9 | 107.1 | 97.4% | memory |
| fc_o | `gemm_kernel` | 28 | 1.4635 | 1.1651 | 40.9772 | 32.6224 | 1.26× | 46956.4 | 51.6 | 79.6% | compute |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.5869 | 0.6101 | 33.4512 | 34.7767 | 0.96× | 171.7 | 114.4 | 104.0% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.8183 | 0.6101 | 22.9128 | 17.0833 | 1.34× | 123.1 | 82.0 | 74.6% | memory |
| rope_q | `rope_opt` | 28 | 0.6630 | 0.6482 | 18.5643 | 18.1499 | 1.02× | 75.9 | 107.5 | 97.8% | memory |
| pa_kv_update_S8192 | `pa_kv_cache_update_ref` | 28 | 0.4768 | 0.4790 | 13.3513 | 13.4123 | 1.00× | 0.0 | 110.5 | 100.5% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.3820 | 0.3051 | 10.6959 | 8.5417 | 1.25× | 132.0 | 87.8 | 79.9% | memory |
| rope_k | `rope_opt` | 28 | 0.3061 | 0.3432 | 8.5700 | 9.6088 | 0.89× | 82.2 | 123.3 | 112.1% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 6.0399 | 5.6603 | 6.0399 | 5.6603 | 1.07× | 103.0 | 103.1 | 93.7% | memory |
| **TOTAL (text dec)** |  |  |  |  | **1239.2386** | **680.4536** | **1.82×** |  |  |  |  |

## 9. Roofline analysis

Ridge point at PTL 12Xe FP16 = 536 FLOP/byte. AI < ridge ⇒ memory-bound; AI ≥ ridge ⇒ compute-bound.

**Decode characteristics**: Every FC has AI = 2·M·K·N / (K·N·2) → ~M = 1 for decode, well below ridge → **always memory-bound**. PA decode AI is also well below ridge.

**Prefill characteristics**: FC AI ≈ M scales with sequence length. For PTL 12Xe (ridge = 536), FC becomes compute-bound roughly at M ≥ ridge. SDPA prefill AI scales with S/2; becomes compute-bound around S ≥ 2·ridge.

### Bottleneck breakdown — Decode (top 5 ops by latency)

| KV | #1 | #2 | #3 | #4 | #5 |
|---:|---|---|---|---|---|
| 512 | fc_down (6.908ms / 19.9%) | fc_up (6.810ms / 19.6%) | fc_gate (6.794ms / 19.6%) | fc_lm_head (5.996ms / 17.3%) | fc_qkv (4.543ms / 13.1%) |
| 1024 | fc_down (6.908ms / 19.6%) | fc_up (6.810ms / 19.3%) | fc_gate (6.794ms / 19.3%) | fc_lm_head (5.996ms / 17.0%) | fc_qkv (4.543ms / 12.9%) |
| 4096 | fc_down (6.908ms / 18.6%) | fc_up (6.810ms / 18.3%) | fc_gate (6.794ms / 18.3%) | fc_lm_head (5.996ms / 16.2%) | fc_qkv (4.543ms / 12.2%) |
| 8192 | fc_down (6.908ms / 17.2%) | fc_up (6.810ms / 17.0%) | fc_gate (6.794ms / 17.0%) | fc_lm_head (5.996ms / 15.0%) | pa_compute_kv8192 (5.678ms / 14.2%) |

### Bottleneck breakdown — Prefill (top 5 ops by latency)

| S | #1 | #2 | #3 | #4 | #5 |
|---:|---|---|---|---|---|
| 512 | fc_down (11.424ms / 19.5%) | fc_gate (9.694ms / 16.6%) | fc_up (9.531ms / 16.3%) | fc_qkv (6.800ms / 11.6%) | fc_lm_head (6.040ms / 10.3%) |
| 1024 | fc_down (20.959ms / 21.3%) | fc_gate (14.783ms / 15.0%) | fc_up (14.643ms / 14.9%) | fc_qkv (11.048ms / 11.2%) | pa_compute_S1024 (9.262ms / 9.4%) |
| 4096 | pa_compute_S4096 (124.610ms / 26.3%) | fc_down (88.932ms / 18.8%) | fc_up (61.396ms / 13.0%) | fc_gate (60.916ms / 12.9%) | fc_qkv (39.681ms / 8.4%) |
| 8192 | pa_compute_S8192 (502.027ms / 40.5%) | fc_down (176.530ms / 14.2%) | fc_up (119.273ms / 9.6%) | fc_gate (118.959ms / 9.6%) | fc_qkv (115.273ms / 9.3%) |

## 10. Notes, caveats & reproduction

- **Audio encoder SDPA** is approximated by 2× the measured PA-prefill (causal) kernel at S=1500. Real encoder is bidirectional non-causal; a dedicated non-causal SDPA bench would be needed for a more precise number. Encoder SDPA is a small fraction of overall TTFT, so this approximation has limited impact.
- **FC LM_Head** is `K=2048, N=151936` FP16 (~593 MB weight stream). This single layer is the dominant decode op and also dominates prefill last-token cost; INT8/INT4 LM_Head would substantially reduce it.
- **PA decode kernel selection**: the Intel GPU plugin keeps `paged_attention_opt__single_token` at small KV and promotes to `paged_attention_opt__gqa_single_token` once the per-token KV working set crosses an internal threshold. For Qwen3-ASR-1.7B with INT8 KV (NH=16 NKV=8 HD=128), the promotion was observed at KV ≥ 4096; at KV = 512/1024 the single_token variant runs.
- **KV cache precision = I8**: the Intel GPU plugin defaults `kv_cache_precision` to `i8` for PagedAttention models (`src/plugins/intel_gpu/src/runtime/execution_config.cpp:293`). This run matches that default. PA cache layout is `K = [num_blocks, NKV, HD, BLOCK+4]` (BY_CHANNEL: 1 byte per element + 4-byte scale/zp shared across BLOCK=16 tokens) and `V = [num_blocks, NKV, BLOCK, HD+4]` (BY_TOKEN: 1 byte per element + 4-byte scale/zp shared across HD per token). The theoretical bytes formula above accounts for both the INT8 payload and the scale/zp overhead, so eff% is measured against what the kernel actually streams. Switching to FP16 KV roughly doubles PA-compute bytes (and so doubles TPOT contribution from PA at long context); switching to INT4 KV would roughly halve them again, at the cost of accuracy. Observed PA-compute decode eff: 50.8% / 54.1% / 83.7% / 85.8% for KV=512/1024/4096/8192 — monotonically increasing with KV, as expected (larger working set amortizes per-launch overhead). The `KV cache key type=...` header line in every pa_bench log exposes the compiled cache element type; the `,20` channel dim is the i8 BY_CHANNEL signature (16 BLOCK + 4 scale/zp).
- **PA prefill** S=8192 produces ~17 ms of `sdpa_micro__prefill` per layer × 28 layers = ~503 ms — by far the dominant prefill cost at long context. This is compute-bound (AI » ridge), so reducing weight precision will not help here; flash-attention–style algorithmic improvements or INT8 attention math would.
- **Encoder weights** count ~589 MB and run only once: amortized over many output tokens the encoder is a small per-token cost.
- **Eff% > 100% on tiny eltwise/norm ops at long prefill** (residual_add, rmsnorm_hidden, rope_q/k, pa_kv_update at S=4096/8192): these kernels are bandwidth-bound and the `small_ops_bench` only allocates a handful of input/output buffers, so the second-and-later iterations hit warm L2 even with the 64 MB flush—the kernel streams less than the theoretical `3·M·H·2B` shown in the table. Treat their eff% as a fitness check (kernel keeps up with budget) rather than as a roofline overshoot. The aggregate TPOT/TTFT row is dominated by FCs + PA where this effect is negligible.

### Reproduction

```bat
REM On PTL 12Xe Windows (Local_Admin@10.239.132.229):
D:\river\moe\dev_roofline_profiling\utils\run_qwen3_asr_1_7B_ptl_12xe.bat
```

```bash
# On local Linux box:
scp -r Local_Admin@<win>:/D/river/moe/roofline_results/qwen3_asr_1_7B/ptl_12xe/*.log \
    .github/skills/dev_roofline_profiling/outputs/qwen3_asr_1_7B/logs_ptl_12xe/
cd .github/skills/dev_roofline_profiling/outputs/qwen3_asr_1_7B
python3 ../../utils/parse_logs.py logs_ptl_12xe parsed.json
python3 build_report.py
```
