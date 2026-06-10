# Qwen3-ASR-0.6B — Roofline on PTL 12Xe (2026-06-05)

**Platform**: Intel PTL 12Xe iGPU (12 Xe cores × 8 EU × 10 threads), 2400 MHz, 110 GB/s
**Model**: Qwen3-ASR-0.6B (audio encoder + Qwen3 text decoder), Linear+Embedding weights **INT8 (u8 g128, f16 scale)**.
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

## 3. Theoretical weight distribution (INT8 (u8 g128, f16 scale))

### Text decoder per-layer (1 of 28)

| Weight | Shape (K × N) | Dtype | MB |
|---|---|---|---:|
| FC_QKV (fused Q+K+V) | 1024 × 4096 | INT8 (u8 g128, f16 scale) | 4.062 |
| FC_O (attn output)   | 2048 × 1024 | INT8 (u8 g128, f16 scale) | 2.031 |
| FC_Gate (SwiGLU)     | 1024 × 3072 | INT8 (u8 g128, f16 scale) | 3.047 |
| FC_Up (SwiGLU)       | 1024 × 3072 | INT8 (u8 g128, f16 scale) | 3.047 |
| FC_Down (SwiGLU)     | 3072 × 1024 | INT8 (u8 g128, f16 scale) | 3.047 |
| RMSNorm × 2          | [1024] each | FP16 | 0.004 |
| q_norm / k_norm      | [128] each  | FP16 | 0.0000 |
| **per layer** |  |  | **15.24** |
| **× 28 layers** |  |  | **426.69** |

### Audio encoder per-layer (1 of 18)

| Weight | Shape (K × N) | Dtype | MB |
|---|---|---|---:|
| FC_QKV (fused) | 896 × 2688 | INT8 (u8 g128, f16 scale) | 2.333 |
| FC_O           | 896 × 896  | INT8 (u8 g128, f16 scale) | 0.778 |
| FC_FC1 (GELU)  | 896 × 3584 | INT8 (u8 g128, f16 scale) | 3.11 |
| FC_FC2 (GELU)  | 3584 × 896 | INT8 (u8 g128, f16 scale) | 3.11 |
| LayerNorm × 2  | [896] g+b each | FP16 | 0.007 |
| **per layer** |  |  | **9.34** |
| **× 18 layers** |  |  | **168.08** |

### Global + shared weights

| Weight | Shape | Dtype | MB |
|---|---|---|---:|
| Token embedding | 151936 × 1024 | INT8 (u8 g128, f16 scale) | 150.693 |
| LM_Head (tied=True) | shares embedding | — | 0.0 |
| Final RMSNorm (text)  | [1024] | FP16 | 0.002 |
| Audio encoder conv front-end | conv1d×2 | FP16 | 5.250 |
| Audio positional embedding | 1500 × 896 | FP16 | 2.563 |
| Audio output adapter | 896→480→1024 | INT8 (u8 g128, f16 scale) | 0.891 |

### Totals

| Component | MB |
|---|---:|
| Text decoder (28 layers) | 426.69 |
| Token embedding (shared w/ LM_Head) | 150.69 |
| Audio encoder (18 layers) | 168.08 |
| Audio encoder front + adapter | 8.71 |
| Final RMSNorm (text) | 0.0020 |
| **Model total** | **754.17** |

## 4. Benchmark methodology

- **Bench utils**: `fc_bench` (precision=u8 g128, INT8 XMX dyn-quant on prefill), `pa_bench` (kv_dtype=i8, impl=ocl), `small_ops_bench` for rmsnorm/rope/eltwise.
- **Tool**: cliloader Device Performance Timing; `parse_logs.py` extracts per-iteration avg GPU kernel ns.
- **L2/L3 flush** between every FC infer (64 MB Relu) so each measurement reads weights from VRAM, not on-die cache. Required because the entire FP16 text decoder weights (~177 MB) is small enough to be mostly cached on iGPU.
- **Input tensors** allocated via RemoteContext in USM_DEVICE (iGPU shared system memory).
- **PA prefill** uses causal mask (Sq·(Sq+1)/2 effective pairs) per SKILL.md.
- **PA decode** is split into `pa_kv_cache_update_ref` + `paged_attention_opt__single_token` + `single_token_finalization`.
- **Audio encoder SDPA** is bidirectional (no causal mask); measured causal time scaled ×2 in the encoder roofline row.
- **swish / multiply** are not profiled separately — they fuse into the SwiGLU primitive per SKILL.md.
- **LM_Head** is `K=1024, N=151936` INT8 (u8 g128, f16 scale) (~158 MB weight stream incl. scales). Counted once per inference at decode and once at prefill (last token only).

## 5. Audio encoder fixed overhead (per inference, S=1500)

The audio encoder runs **once** per inference regardless of text token count.

### Audio encoder — S=1500

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| enc_sdpa | `sdpa_micro__prefill (×2 bidir est.)` | 18 | 0.7572 | 0.1367 | 13.6292 | 2.4609 | 5.54× | 10650.0 | 14.2 | 18.1% | compute |
| enc_fc_ffn2 | `gemm_kernel` | 18 | 0.2779 | 0.1633 | 5.0024 | 2.9400 | 1.70× | 34665.2 | 60.1 | 58.8% | compute |
| enc_fc_ffn1 | `gemm_kernel` | 18 | 0.1959 | 0.1633 | 3.5260 | 2.9400 | 1.20× | 49180.3 | 85.3 | 83.4% | compute |
| enc_fc_qkv | `gemm_kernel` | 18 | 0.1599 | 0.1225 | 2.8777 | 2.2050 | 1.31× | 45194.3 | 82.5 | 76.6% | compute |
| enc_fc_o | `gemm_kernel` | 18 | 0.0837 | 0.0563 | 1.5061 | 1.0131 | 1.49× | 28783.4 | 74.0 | 67.3% | memory |
| enc_outproj | `gemm_kernel` | 1 | 0.0854 | 0.0608 | 0.0854 | 0.0608 | 1.40× | 32236.9 | 78.4 | 71.2% | memory |
| **TOTAL (encoder fixed)** |  |  |  |  | **26.6268** | **11.6198** | **2.29×** |  |  |  |  |

## 6. Token latency summary

**TTFT** = audio encoder (26.63 ms) + text decoder prefill.
**TPOT** = per output token decode latency at the listed KV context.

### Prefill — TTFT

| S (text ctx) | Encoder (ms) | Text-decoder prefill (ms) | **TTFT (ms)** | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 26.63 | 23.75 | **50.38** | 0.0984 | 10163 |
| 1024 | 26.63 | 43.84 | **70.47** | 0.0688 | 14531 |
| 4096 | 26.63 | 250.12 | **276.75** | 0.0676 | 14800 |
| 8192 | 26.63 | 764.35 | **790.98** | 0.0966 | 10357 |

### Decode — TPOT

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 512 | 7.6969 | 129.9 |
| 1024 | 8.2502 | 121.2 |
| 4096 | 10.1525 | 98.5 |
| 8192 | 13.1745 | 75.9 |

### Measured vs. theoretical latency (lower bound = roofline)

The **theoretical** column is the sum over all kernels of
`max(bytes/BW, flops/peak_TFLOPS)` — i.e. the absolute minimum time
each kernel could take given its bound. `slowdown = measured / theo` (1.0× = at roofline).

#### Decode (per output token)

| KV (ctx) | Measured TPOT (ms) | Theoretical TPOT (ms) | Slowdown | Wall-clock tokens/s | Roofline tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 7.6969 | 5.8404 | 1.32× | 129.9 | 171.2 |
| 1024 | 8.2502 | 6.1449 | 1.34× | 121.2 | 162.7 |
| 4096 | 10.1525 | 7.9716 | 1.27× | 98.5 | 125.4 |
| 8192 | 13.1745 | 10.4071 | 1.27× | 75.9 | 96.1 |

#### Prefill / TTFT

| S (text ctx) | Measured TTFT (ms) | Theoretical TTFT (ms) | Slowdown | Meas tokens/s | Roofline tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 50.38 | 30.93 | 1.63× | 10163 | 16553 |
| 1024 | 70.47 | 45.39 | 1.55× | 14531 | 22562 |
| 4096 | 276.75 | 161.76 | 1.71× | 14800 | 25321 |
| 8192 | 790.98 | 375.71 | 2.11× | 10357 | 21804 |

#### Audio encoder fixed overhead

| Component | Measured (ms) | Theoretical (ms) | Slowdown |
|---|---:|---:|---:|
| Audio encoder forward (S=1500, 18 layers) | 26.63 | 11.62 | 2.29× |

### End-to-end latency estimate (output = 512 tokens)

Approximation: decode uses the TPOT at KV = S_prefill (start-of-decode value). Real decode TPOT increases as KV grows toward S_prefill + 511.

| S (text ctx) | TTFT (ms) | TPOT @ KV=S (ms) | Measured total (ms) | Theoretical total (ms) | Slowdown |
|---:|---:|---:|---:|---:|---:|
| 512 | 50.38 | 7.6969 | 3991.2 | 3021.2 | 1.32× |
| 1024 | 70.47 | 8.2502 | 4294.6 | 3191.6 | 1.35× |
| 4096 | 276.75 | 10.1525 | 5474.8 | 4243.2 | 1.29× |
| 8192 | 790.98 | 13.1745 | 7536.3 | 5704.1 | 1.32× |

## 7. Per-kernel tables — Decode (1 query token, KV = context)

### Decode — KV=512

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_lm_head | `gemm_kernel` | 1 | 1.5418 | 1.4393 | 1.5418 | 1.4393 | 1.07× | 201.8 | 102.7 | 93.3% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0453 | 0.0388 | 1.2683 | 1.0869 | 1.17× | 185.2 | 94.3 | 85.7% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0355 | 0.0291 | 0.9939 | 0.8153 | 1.22× | 177.2 | 90.2 | 82.0% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0346 | 0.0291 | 0.9680 | 0.8153 | 1.19× | 182.0 | 92.7 | 84.2% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0345 | 0.0291 | 0.9663 | 0.8153 | 1.19× | 182.3 | 92.8 | 84.4% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0242 | 0.0194 | 0.6781 | 0.5437 | 1.25× | 173.2 | 88.2 | 80.2% | memory |
| pa_compute_kv512 | `paged_attention_opt__single_token` | 28 | 0.0198 | 0.0109 | 0.5531 | 0.3065 | 1.80× | 212.3 | 61.0 | 55.4% | memory |
| pa_kv_update_kv512 | `pa_kv_cache_update_ref` | 28 | 0.0054 | 0.0001 | 0.1519 | 0.0016 | 92.80× | 0.0 | 1.2 | 1.1% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0027 | 0.0001 | 0.1517 | 0.0032 | 47.66× | 2.3 | 2.3 | 2.1% | memory |
| pa_finalize_kv512 | `single_token_finalization` | 28 | 0.0036 | 0.0001 | 0.1012 | 0.0021 | 48.53× | 0.0 | 2.3 | 2.1% | memory |
| residual_add | `eltwise` | 56 | 0.0014 | 0.0001 | 0.0767 | 0.0031 | 24.51× | 0.8 | 4.5 | 4.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0671 | 0.0016 | 42.88× | 2.6 | 2.6 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0667 | 0.0031 | 21.33× | 5.2 | 5.2 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0020 | 0.0001 | 0.0572 | 0.0022 | 25.81× | 3.0 | 4.3 | 3.9% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0549 | 0.0012 | 46.81× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **7.6969** | **5.8404** | **1.32×** |  |  |  |  |

### Decode — KV=1024

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_lm_head | `gemm_kernel` | 1 | 1.5418 | 1.4393 | 1.5418 | 1.4393 | 1.07× | 201.8 | 102.7 | 93.3% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0453 | 0.0388 | 1.2683 | 1.0869 | 1.17× | 185.2 | 94.3 | 85.7% | memory |
| pa_compute_kv1024 | `paged_attention_opt__single_token` | 28 | 0.0385 | 0.0218 | 1.0774 | 0.6110 | 1.76× | 218.0 | 62.4 | 56.7% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0355 | 0.0291 | 0.9939 | 0.8153 | 1.22× | 177.2 | 90.2 | 82.0% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0346 | 0.0291 | 0.9680 | 0.8153 | 1.19× | 182.0 | 92.7 | 84.2% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0345 | 0.0291 | 0.9663 | 0.8153 | 1.19× | 182.3 | 92.8 | 84.4% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0242 | 0.0194 | 0.6781 | 0.5437 | 1.25× | 173.2 | 88.2 | 80.2% | memory |
| pa_kv_update_kv1024 | `pa_kv_cache_update_ref` | 28 | 0.0064 | 0.0001 | 0.1783 | 0.0016 | 108.91× | 0.0 | 1.0 | 0.9% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0027 | 0.0001 | 0.1517 | 0.0032 | 47.66× | 2.3 | 2.3 | 2.1% | memory |
| pa_finalize_kv1024 | `single_token_finalization` | 28 | 0.0037 | 0.0001 | 0.1038 | 0.0021 | 49.79× | 0.0 | 2.2 | 2.0% | memory |
| residual_add | `eltwise` | 56 | 0.0014 | 0.0001 | 0.0767 | 0.0031 | 24.51× | 0.8 | 4.5 | 4.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0671 | 0.0016 | 42.88× | 2.6 | 2.6 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0667 | 0.0031 | 21.33× | 5.2 | 5.2 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0020 | 0.0001 | 0.0572 | 0.0022 | 25.81× | 3.0 | 4.3 | 3.9% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0549 | 0.0012 | 46.81× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **8.2502** | **6.1449** | **1.34×** |  |  |  |  |

### Decode — KV=4096

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv4096 | `paged_attention_opt__single_token` | 28 | 0.1042 | 0.0871 | 2.9168 | 2.4377 | 1.20× | 322.1 | 91.9 | 83.6% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5418 | 1.4393 | 1.5418 | 1.4393 | 1.07× | 201.8 | 102.7 | 93.3% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0453 | 0.0388 | 1.2683 | 1.0869 | 1.17× | 185.2 | 94.3 | 85.7% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0355 | 0.0291 | 0.9939 | 0.8153 | 1.22× | 177.2 | 90.2 | 82.0% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0346 | 0.0291 | 0.9680 | 0.8153 | 1.19× | 182.0 | 92.7 | 84.2% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0345 | 0.0291 | 0.9663 | 0.8153 | 1.19× | 182.3 | 92.8 | 84.4% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0242 | 0.0194 | 0.6781 | 0.5437 | 1.25× | 173.2 | 88.2 | 80.2% | memory |
| pa_kv_update_kv4096 | `pa_kv_cache_update_ref` | 28 | 0.0063 | 0.0001 | 0.1778 | 0.0016 | 108.58× | 0.0 | 1.0 | 0.9% | memory |
| pa_finalize_kv4096 | `single_token_finalization` | 28 | 0.0060 | 0.0001 | 0.1672 | 0.0021 | 80.20× | 0.0 | 1.4 | 1.2% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0027 | 0.0001 | 0.1517 | 0.0032 | 47.66× | 2.3 | 2.3 | 2.1% | memory |
| residual_add | `eltwise` | 56 | 0.0014 | 0.0001 | 0.0767 | 0.0031 | 24.51× | 0.8 | 4.5 | 4.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0671 | 0.0016 | 42.88× | 2.6 | 2.6 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0667 | 0.0031 | 21.33× | 5.2 | 5.2 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0020 | 0.0001 | 0.0572 | 0.0022 | 25.81× | 3.0 | 4.3 | 3.9% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0549 | 0.0012 | 46.81× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **10.1525** | **7.9716** | **1.27×** |  |  |  |  |

### Decode — KV=8192

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv8192 | `paged_attention_opt__single_token` | 28 | 0.2043 | 0.1740 | 5.7210 | 4.8732 | 1.17× | 328.4 | 93.7 | 85.2% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5418 | 1.4393 | 1.5418 | 1.4393 | 1.07× | 201.8 | 102.7 | 93.3% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0453 | 0.0388 | 1.2683 | 1.0869 | 1.17× | 185.2 | 94.3 | 85.7% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0355 | 0.0291 | 0.9939 | 0.8153 | 1.22× | 177.2 | 90.2 | 82.0% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0346 | 0.0291 | 0.9680 | 0.8153 | 1.19× | 182.0 | 92.7 | 84.2% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0345 | 0.0291 | 0.9663 | 0.8153 | 1.19× | 182.3 | 92.8 | 84.4% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0242 | 0.0194 | 0.6781 | 0.5437 | 1.25× | 173.2 | 88.2 | 80.2% | memory |
| pa_finalize_kv8192 | `single_token_finalization` | 28 | 0.0124 | 0.0001 | 0.3467 | 0.0021 | 166.28× | 0.0 | 0.7 | 0.6% | memory |
| pa_kv_update_kv8192 | `pa_kv_cache_update_ref` | 28 | 0.0077 | 0.0001 | 0.2161 | 0.0016 | 131.99× | 0.0 | 0.8 | 0.8% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0027 | 0.0001 | 0.1517 | 0.0032 | 47.66× | 2.3 | 2.3 | 2.1% | memory |
| residual_add | `eltwise` | 56 | 0.0014 | 0.0001 | 0.0767 | 0.0031 | 24.51× | 0.8 | 4.5 | 4.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0671 | 0.0016 | 42.88× | 2.6 | 2.6 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0024 | 0.0001 | 0.0667 | 0.0031 | 21.33× | 5.2 | 5.2 | 4.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0020 | 0.0001 | 0.0572 | 0.0022 | 25.81× | 3.0 | 4.3 | 3.9% | memory |
| rope_k | `rope_opt` | 28 | 0.0020 | 0.0000 | 0.0549 | 0.0012 | 46.81× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL** |  |  |  |  | **13.1745** | **10.4071** | **1.27×** |  |  |  |  |

## 8. Per-kernel tables — Prefill (text-decoder only; add encoder overhead for TTFT)

### Prefill — S=512 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_down | `gemm_kernel` | 28 | 0.1187 | 0.0672 | 3.3245 | 1.8809 | 1.77× | 27130.0 | 62.2 | 56.6% | memory |
| pa_compute_S512 | `sdpa_micro__prefill` | 28 | 0.1040 | 0.0572 | 2.9108 | 1.6015 | 1.82× | 10348.8 | 60.5 | 55.0% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.1020 | 0.0864 | 2.8572 | 2.4189 | 1.18× | 42090.2 | 93.1 | 84.7% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0829 | 0.0480 | 2.3221 | 1.3429 | 1.73× | 25894.5 | 63.6 | 57.8% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0783 | 0.0672 | 2.1929 | 1.8809 | 1.17× | 41130.1 | 94.3 | 85.8% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0774 | 0.0672 | 2.1681 | 1.8809 | 1.15× | 41600.7 | 95.4 | 86.8% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5372 | 1.4393 | 1.5372 | 1.4393 | 1.07× | 202.4 | 103.0 | 93.6% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0486 | 0.0382 | 1.3614 | 1.0687 | 1.27× | 129.5 | 86.3 | 78.5% | memory |
| residual_add | `eltwise` | 56 | 0.0234 | 0.0286 | 1.3077 | 1.6015 | 0.82× | 22.4 | 134.7 | 122.5% | memory |
| pa_kv_update_S512 | `pa_kv_cache_update_ref` | 28 | 0.0340 | 0.0299 | 0.9520 | 0.8383 | 1.14× | 0.0 | 96.9 | 88.1% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0144 | 0.0191 | 0.8217 | 1.0878 | 0.76× | 218.6 | 145.6 | 132.4% | memory |
| rope_q | `rope_opt` | 28 | 0.0291 | 0.0405 | 0.8162 | 1.1344 | 0.72× | 107.9 | 152.9 | 139.0% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0243 | 0.0191 | 0.6793 | 0.5344 | 1.27× | 129.9 | 86.5 | 78.7% | memory |
| rope_k | `rope_opt` | 28 | 0.0179 | 0.0214 | 0.5005 | 0.6005 | 0.83× | 88.0 | 132.0 | 120.0% | memory |
| **TOTAL (text dec)** |  |  |  |  | **23.7516** | **19.3109** | **1.23×** |  |  |  |  |

### Prefill — S=1024 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S1024 | `sdpa_micro__prefill` | 28 | 0.3131 | 0.1144 | 8.7675 | 3.2029 | 2.74× | 13729.9 | 40.2 | 36.5% | memory |
| fc_down | `gemm_kernel` | 28 | 0.1884 | 0.1092 | 5.2747 | 3.0584 | 1.72× | 34198.9 | 61.5 | 58.0% | compute |
| fc_qkv | `gemm_kernel` | 28 | 0.1723 | 0.1456 | 4.8235 | 4.0778 | 1.18× | 49864.1 | 85.6 | 84.5% | compute |
| fc_gate | `gemm_kernel` | 28 | 0.1426 | 0.1092 | 3.9934 | 3.0584 | 1.31× | 45171.5 | 81.2 | 76.6% | compute |
| fc_up | `gemm_kernel` | 28 | 0.1397 | 0.1092 | 3.9105 | 3.0584 | 1.28× | 46129.2 | 82.9 | 78.2% | compute |
| fc_o | `gemm_kernel` | 28 | 0.1299 | 0.0766 | 3.6366 | 2.1436 | 1.70× | 33069.5 | 64.8 | 58.9% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0951 | 0.0763 | 2.6640 | 2.1363 | 1.25× | 132.4 | 88.2 | 80.2% | memory |
| residual_add | `eltwise` | 56 | 0.0448 | 0.0572 | 2.5087 | 3.2029 | 0.78× | 23.4 | 140.4 | 127.7% | memory |
| pa_kv_update_S1024 | `pa_kv_cache_update_ref` | 28 | 0.0550 | 0.0599 | 1.5388 | 1.6765 | 0.92× | 0.0 | 119.8 | 109.0% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5372 | 1.4393 | 1.5372 | 1.4393 | 1.07× | 202.4 | 103.0 | 93.6% | memory |
| rope_q | `rope_opt` | 28 | 0.0539 | 0.0810 | 1.5103 | 2.2687 | 0.67× | 116.6 | 165.2 | 150.2% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0251 | 0.0381 | 1.4312 | 2.1745 | 0.66× | 251.0 | 167.1 | 151.9% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0486 | 0.0381 | 1.3596 | 1.0682 | 1.27× | 129.8 | 86.4 | 78.6% | memory |
| rope_k | `rope_opt` | 28 | 0.0317 | 0.0429 | 0.8879 | 1.2011 | 0.74× | 99.2 | 148.8 | 135.3% | memory |
| **TOTAL (text dec)** |  |  |  |  | **43.8439** | **33.7670** | **1.30×** |  |  |  |  |

### Prefill — S=4096 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S4096 | `sdpa_micro__prefill` | 28 | 4.4417 | 1.1654 | 124.3688 | 32.6303 | 3.81× | 15475.1 | 11.3 | 26.2% | compute |
| fc_qkv | `gemm_kernel` | 28 | 0.6518 | 0.5825 | 18.2501 | 16.3112 | 1.12× | 52715.9 | 70.9 | 89.4% | compute |
| fc_down | `gemm_kernel` | 28 | 0.6351 | 0.4369 | 17.7839 | 12.2334 | 1.45× | 40573.4 | 57.9 | 68.8% | compute |
| fc_up | `gemm_kernel` | 28 | 0.4826 | 0.4369 | 13.5129 | 12.2334 | 1.10× | 53397.5 | 76.2 | 90.5% | compute |
| fc_gate | `gemm_kernel` | 28 | 0.4771 | 0.4369 | 13.3591 | 12.2334 | 1.09× | 54012.3 | 77.0 | 91.6% | compute |
| fc_o | `gemm_kernel` | 28 | 0.4327 | 0.2913 | 12.1147 | 8.1556 | 1.49× | 39706.8 | 63.1 | 67.3% | compute |
| residual_add | `eltwise` | 56 | 0.1915 | 0.2288 | 10.7232 | 12.8117 | 0.84× | 21.9 | 131.4 | 119.5% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.3797 | 0.3051 | 10.6323 | 8.5422 | 1.24× | 132.7 | 88.4 | 80.3% | memory |
| rope_q | `rope_opt` | 28 | 0.2715 | 0.3241 | 7.6016 | 9.0749 | 0.84× | 92.7 | 131.3 | 119.4% | memory |
| pa_kv_update_S4096 | `pa_kv_cache_update_ref` | 28 | 0.2230 | 0.2395 | 6.2447 | 6.7061 | 0.93× | 0.0 | 118.1 | 107.4% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.1894 | 0.1525 | 5.3025 | 4.2711 | 1.24× | 133.1 | 88.6 | 80.5% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0923 | 0.1525 | 5.2598 | 8.6947 | 0.60× | 273.2 | 181.8 | 165.3% | memory |
| rope_k | `rope_opt` | 28 | 0.1225 | 0.1716 | 3.4303 | 4.8044 | 0.71× | 102.7 | 154.1 | 140.1% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5372 | 1.4393 | 1.5372 | 1.4393 | 1.07× | 202.4 | 103.0 | 93.6% | memory |
| **TOTAL (text dec)** |  |  |  |  | **250.1211** | **150.1417** | **1.67×** |  |  |  |  |

### Prefill — S=8192 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S8192 | `sdpa_micro__prefill` | 28 | 17.5760 | 4.6609 | 492.1275 | 130.5054 | 3.77× | 15641.3 | 5.7 | 26.5% | compute |
| fc_qkv | `gemm_kernel` | 28 | 1.3104 | 1.1651 | 36.6913 | 32.6224 | 1.12× | 52441.5 | 67.3 | 88.9% | compute |
| fc_down | `gemm_kernel` | 28 | 1.3007 | 0.8738 | 36.4184 | 24.4668 | 1.49× | 39625.8 | 54.0 | 67.2% | compute |
| fc_gate | `gemm_kernel` | 28 | 1.1113 | 0.8738 | 31.1160 | 24.4668 | 1.27× | 46378.4 | 63.3 | 78.6% | compute |
| fc_up | `gemm_kernel` | 28 | 1.1108 | 0.8738 | 31.1030 | 24.4668 | 1.27× | 46397.7 | 63.3 | 78.7% | compute |
| residual_add | `eltwise` | 56 | 0.4455 | 0.4576 | 24.9464 | 25.6234 | 0.97× | 18.8 | 113.0 | 102.7% | memory |
| fc_o | `gemm_kernel` | 28 | 0.8580 | 0.5825 | 24.0240 | 16.3112 | 1.47× | 40046.3 | 61.1 | 67.9% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.7757 | 0.6101 | 21.7184 | 17.0833 | 1.27× | 129.9 | 86.5 | 78.7% | memory |
| rope_q | `rope_opt` | 28 | 0.6354 | 0.6482 | 17.7918 | 18.1499 | 0.98× | 79.2 | 112.2 | 102.0% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.2494 | 0.3051 | 14.2180 | 17.3884 | 0.82× | 202.1 | 134.5 | 122.3% | memory |
| pa_kv_update_S8192 | `pa_kv_cache_update_ref` | 28 | 0.4771 | 0.4790 | 13.3575 | 13.4123 | 1.00× | 0.0 | 110.5 | 100.4% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.3844 | 0.3051 | 10.7627 | 8.5417 | 1.26× | 131.2 | 87.3 | 79.4% | memory |
| rope_k | `rope_opt` | 28 | 0.3049 | 0.3432 | 8.5383 | 9.6088 | 0.89× | 82.5 | 123.8 | 112.5% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5372 | 1.4393 | 1.5372 | 1.4393 | 1.07× | 202.4 | 103.0 | 93.6% | memory |
| **TOTAL (text dec)** |  |  |  |  | **764.3505** | **364.0865** | **2.10×** |  |  |  |  |

## 9. Roofline analysis

Ridge point at PTL 12Xe FP16 = 536 FLOP/byte. AI < ridge ⇒ memory-bound; AI ≥ ridge ⇒ compute-bound.

**Decode characteristics**: Every FC has AI = 2·M·K·N / (K·N·2) → ~M = 1 for decode, well below ridge → **always memory-bound**. PA decode AI is also well below ridge.

**Prefill characteristics**: FC AI ≈ M scales with sequence length. For PTL 12Xe (ridge = 536), FC becomes compute-bound roughly at M ≥ ridge. SDPA prefill AI scales with S/2; becomes compute-bound around S ≥ 2·ridge.

### Bottleneck breakdown — Decode (top 5 ops by latency)

| KV | #1 | #2 | #3 | #4 | #5 |
|---:|---|---|---|---|---|
| 512 | fc_lm_head (1.542ms / 20.0%) | fc_qkv (1.268ms / 16.5%) | fc_down (0.994ms / 12.9%) | fc_gate (0.968ms / 12.6%) | fc_up (0.966ms / 12.6%) |
| 1024 | fc_lm_head (1.542ms / 18.7%) | fc_qkv (1.268ms / 15.4%) | pa_compute_kv1024 (1.077ms / 13.1%) | fc_down (0.994ms / 12.0%) | fc_gate (0.968ms / 11.7%) |
| 4096 | pa_compute_kv4096 (2.917ms / 28.7%) | fc_lm_head (1.542ms / 15.2%) | fc_qkv (1.268ms / 12.5%) | fc_down (0.994ms / 9.8%) | fc_gate (0.968ms / 9.5%) |
| 8192 | pa_compute_kv8192 (5.721ms / 43.4%) | fc_lm_head (1.542ms / 11.7%) | fc_qkv (1.268ms / 9.6%) | fc_down (0.994ms / 7.5%) | fc_gate (0.968ms / 7.3%) |

### Bottleneck breakdown — Prefill (top 5 ops by latency)

| S | #1 | #2 | #3 | #4 | #5 |
|---:|---|---|---|---|---|
| 512 | fc_down (3.325ms / 14.0%) | pa_compute_S512 (2.911ms / 12.3%) | fc_qkv (2.857ms / 12.0%) | fc_o (2.322ms / 9.8%) | fc_gate (2.193ms / 9.2%) |
| 1024 | pa_compute_S1024 (8.768ms / 20.0%) | fc_down (5.275ms / 12.0%) | fc_qkv (4.824ms / 11.0%) | fc_gate (3.993ms / 9.1%) | fc_up (3.910ms / 8.9%) |
| 4096 | pa_compute_S4096 (124.369ms / 49.7%) | fc_qkv (18.250ms / 7.3%) | fc_down (17.784ms / 7.1%) | fc_up (13.513ms / 5.4%) | fc_gate (13.359ms / 5.3%) |
| 8192 | pa_compute_S8192 (492.127ms / 64.4%) | fc_qkv (36.691ms / 4.8%) | fc_down (36.418ms / 4.8%) | fc_gate (31.116ms / 4.1%) | fc_up (31.103ms / 4.1%) |

## 10. Notes, caveats & reproduction

- **Audio encoder SDPA** is approximated by 2× the measured PA-prefill (causal) kernel at S=1500. Real encoder is bidirectional non-causal; a dedicated non-causal SDPA bench would be needed for a more precise number. Encoder SDPA is a small fraction of overall TTFT, so this approximation has limited impact.
- **FC LM_Head** is `K=1024, N=151936` FP16 (~297 MB weight stream). This single layer is the dominant decode op (~50% of TPOT) and also dominates prefill last-token cost; INT8/INT4 LM_Head would substantially reduce it.
- **PA decode** dispatches `paged_attention_opt__single_token` across all measured KV sizes (512–8192). On some other Qwen variants the runtime promotes to `paged_attention_opt__gqa_single_token` at long context; that did not trigger in this configuration.
- **KV cache precision = I8**: the Intel GPU plugin defaults `kv_cache_precision` to `i8` for PagedAttention models (`src/plugins/intel_gpu/src/runtime/execution_config.cpp:293`). This run matches that default. PA cache layout is `K = [num_blocks, NKV, HD, BLOCK+4]` (BY_CHANNEL: 1 byte per element + 4-byte scale/zp shared across BLOCK=16 tokens) and `V = [num_blocks, NKV, BLOCK, HD+4]` (BY_TOKEN: 1 byte per element + 4-byte scale/zp shared across HD per token). The theoretical bytes formula above accounts for both the INT8 payload and the scale/zp overhead, so eff% is measured against what the kernel actually streams. Switching to FP16 KV roughly doubles PA-compute bytes (and so doubles TPOT contribution from PA at long context); switching to INT4 KV would roughly halve them again, at the cost of accuracy. Observed PA-compute decode eff: 55.4% / 56.7% / 83.6% / 85.2% for KV=512/1024/4096/8192 — monotonically increasing with KV, as expected (larger working set amortizes per-launch overhead). The `KV cache key type=...` header line in every pa_bench log exposes the compiled cache element type; the `,20` channel dim is the i8 BY_CHANNEL signature (16 BLOCK + 4 scale/zp).
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
    .github/skills/dev_roofline_profiling/outputs/qwen3_asr_0_6B/logs_ptl_12xe_int8w/
cd .github/skills/dev_roofline_profiling/outputs/qwen3_asr_0_6B
python3 ../../utils/parse_logs.py logs_ptl_12xe_int8w parsed_int8w.json
python3 build_report.py
```
