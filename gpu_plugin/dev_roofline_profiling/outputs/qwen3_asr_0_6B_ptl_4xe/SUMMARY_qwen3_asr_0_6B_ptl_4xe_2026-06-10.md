# Qwen3-ASR-0.6B — Roofline on PTL 4Xe (2026-06-10)

**Platform**: Intel PTL 4Xe iGPU (4 Xe cores × 8 EU × 10 threads), 2450 MHz, 110 GB/s
**Model**: Qwen3-ASR-0.6B (audio encoder + Qwen3 text decoder), Linear+Embedding weights **FP16**.
**SDPA**: PagedAttention (opencl + micro_kernel), INT8 KV cache (text decoder; matches Intel GPU plugin default for PA models). Audio encoder SDPA remains FP16.
**Source config**: https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/main/config.json

**Inputs evaluated**: text-decoder prefill context = 512 / 1024 / 4096 / 8192 tokens; output = 512 tokens; audio encoder runs once over 1500 mel frames.

## 1. Hardware peaks (per SKILL.md formulas)

`FP16 XMX TFLOPS = xe_cores × 8 × 256 × freq_GHz`; `INT8 XMX = 2× FP16`; `SIMD FP16 = xe_cores × 8 × 32 × freq`.

| Platform | Xe Cores | Freq (MHz) | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | SIMD FP16 (TFLOPS) | Ridge FP16 (FLOP/B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PTL 4Xe | 4 | 2450 | 110 | 20.070 | 40.141 | 2.509 | 182 |

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
- **L2/L3 flush** between every FC infer (64 MB Relu) so each measurement reads weights from VRAM, not on-die cache. Even so, the small per-op FP16 weight tensors (8 MB QKV, 6 MB MLP) stay partially resident in the PTL memory-side/SLC cache, which is why the M=1 decode FCs measure effective bandwidth above the 110 GB/s DRAM peak (see caveats §10).
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
| enc_sdpa | `sdpa_micro__prefill (×2 bidir est.)` | 18 | 2.4508 | 0.4018 | 44.1141 | 7.2321 | 6.10× | 3290.4 | 4.4 | 16.4% | compute |
| enc_fc_ffn2 | `gemm_kernel` | 18 | 0.7293 | 0.4800 | 13.1273 | 8.6400 | 1.52× | 13209.8 | 27.2 | 65.8% | compute |
| enc_fc_ffn1 | `gemm_kernel` | 18 | 0.5361 | 0.4800 | 9.6505 | 8.6400 | 1.12× | 17968.8 | 37.0 | 89.5% | compute |
| enc_fc_qkv | `gemm_kernel` | 18 | 0.4208 | 0.3600 | 7.5745 | 6.4800 | 1.17× | 17170.4 | 37.0 | 85.6% | compute |
| enc_fc_o | `gemm_kernel` | 18 | 0.1865 | 0.1200 | 3.3562 | 2.1600 | 1.55× | 12916.8 | 37.4 | 64.4% | compute |
| enc_outproj | `gemm_kernel` | 1 | 0.2038 | 0.1371 | 0.2038 | 0.1371 | 1.49× | 13507.7 | 37.3 | 67.3% | compute |
| **TOTAL (encoder fixed)** |  |  |  |  | **78.0264** | **33.2892** | **2.34×** |  |  |  |  |

## 6. Token latency summary

**TTFT** = audio encoder (78.03 ms) + text decoder prefill.
**TPOT** = per output token decode latency at the listed KV context.

### Prefill — TTFT

| S (text ctx) | Encoder (ms) | Text-decoder prefill (ms) | **TTFT (ms)** | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 78.03 | 48.81 | **126.84** | 0.2477 | 4037 |
| 1024 | 78.03 | 104.25 | **182.27** | 0.1780 | 5618 |
| 4096 | 78.03 | 689.34 | **767.36** | 0.1873 | 5338 |
| 8192 | 78.03 | 2179.62 | **2257.65** | 0.2756 | 3629 |

### Decode — TPOT

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 512 | 5.6102 | 178.2 |
| 1024 | 6.0725 | 164.7 |
| 4096 | 8.7617 | 114.1 |
| 8192 | 12.5838 | 79.5 |

### Measured vs. theoretical latency (lower bound = roofline)

The **theoretical** column is the sum over all kernels of
`max(bytes/BW, flops/peak_TFLOPS)` — i.e. the absolute minimum time
each kernel could take given its bound. `slowdown = measured / theo` (1.0× = at roofline).

#### Decode (per output token)

| KV (ctx) | Measured TPOT (ms) | Theoretical TPOT (ms) | Slowdown | Wall-clock tokens/s | Roofline tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 5.6102 | 11.1741 | 0.50× | 178.2 | 89.5 |
| 1024 | 6.0725 | 11.4786 | 0.53× | 164.7 | 87.1 |
| 4096 | 8.7617 | 13.3053 | 0.66× | 114.1 | 75.2 |
| 8192 | 12.5838 | 15.7408 | 0.80× | 79.5 | 63.5 |

#### Prefill / TTFT

| S (text ctx) | Measured TTFT (ms) | Theoretical TTFT (ms) | Slowdown | Meas tokens/s | Roofline tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 126.84 | 67.06 | 1.89× | 4037 | 7635 |
| 1024 | 182.27 | 100.79 | 1.81× | 5618 | 10160 |
| 4096 | 767.36 | 366.68 | 2.09× | 5338 | 11171 |
| 8192 | 2257.65 | 888.97 | 2.54× | 3629 | 9215 |

#### Audio encoder fixed overhead

| Component | Measured (ms) | Theoretical (ms) | Slowdown |
|---|---:|---:|---:|
| Audio encoder forward (S=1500, 18 layers) | 78.03 | 33.29 | 2.34× |

### End-to-end latency estimate (output = 512 tokens)

Approximation: decode uses the TPOT at KV = S_prefill (start-of-decode value). Real decode TPOT increases as KV grows toward S_prefill + 511.

| S (text ctx) | TTFT (ms) | TPOT @ KV=S (ms) | Measured total (ms) | Theoretical total (ms) | Slowdown |
|---:|---:|---:|---:|---:|---:|
| 512 | 126.84 | 5.6102 | 2999.3 | 5788.2 | 0.52× |
| 1024 | 182.27 | 6.0725 | 3291.4 | 5977.8 | 0.55× |
| 4096 | 767.36 | 8.7617 | 5253.4 | 7179.0 | 0.73× |
| 8192 | 2257.65 | 12.5838 | 8700.6 | 8948.3 | 0.97× |

## 7. Per-kernel tables — Decode (1 query token, KV = context)

### Decode — KV=512

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv512 | `paged_attention_opt__single_token` | 28 | 0.0418 | 0.0109 | 1.1712 | 0.3065 | 3.82× | 100.3 | 28.8 | 26.2% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 0.8681 | 2.8316 | 0.8681 | 2.8316 | 0.31× | 358.4 | 358.8 | 326.2% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0277 | 0.0764 | 0.7745 | 2.1379 | 0.36× | 303.3 | 303.7 | 276.1% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0218 | 0.0573 | 0.6104 | 1.6036 | 0.38× | 288.6 | 289.0 | 262.7% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0216 | 0.0573 | 0.6046 | 1.6036 | 0.38× | 291.4 | 291.8 | 265.2% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0193 | 0.0573 | 0.5412 | 1.6036 | 0.34× | 325.5 | 325.9 | 296.3% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0141 | 0.0382 | 0.3941 | 1.0692 | 0.37× | 298.0 | 298.4 | 271.3% | memory |
| pa_kv_update_kv512 | `pa_kv_cache_update_ref` | 28 | 0.0050 | 0.0001 | 0.1403 | 0.0016 | 85.68× | 0.0 | 1.3 | 1.2% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0024 | 0.0001 | 0.1365 | 0.0032 | 42.88× | 2.6 | 2.6 | 2.3% | memory |
| residual_add | `eltwise` | 56 | 0.0013 | 0.0001 | 0.0717 | 0.0031 | 22.92× | 0.8 | 4.8 | 4.4% | memory |
| pa_finalize_kv512 | `single_token_finalization` | 28 | 0.0025 | 0.0001 | 0.0705 | 0.0021 | 33.81× | 0.0 | 3.3 | 3.0% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0617 | 0.0031 | 19.73× | 5.6 | 5.6 | 5.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0605 | 0.0016 | 38.69× | 2.9 | 2.8 | 2.6% | memory |
| rope_q | `rope_opt` | 28 | 0.0019 | 0.0001 | 0.0534 | 0.0022 | 24.11× | 3.2 | 4.6 | 4.1% | memory |
| rope_k | `rope_opt` | 28 | 0.0018 | 0.0000 | 0.0515 | 0.0012 | 43.95× | 1.7 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  |  | **5.6102** | **11.1741** | **0.50×** |  |  |  |  |

### Decode — KV=1024

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv1024 | `paged_attention_opt__single_token` | 28 | 0.0596 | 0.0218 | 1.6691 | 0.6110 | 2.73× | 140.7 | 40.3 | 36.6% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 0.8681 | 2.8316 | 0.8681 | 2.8316 | 0.31× | 358.4 | 358.8 | 326.2% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0277 | 0.0764 | 0.7745 | 2.1379 | 0.36× | 303.3 | 303.7 | 276.1% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0218 | 0.0573 | 0.6104 | 1.6036 | 0.38× | 288.6 | 289.0 | 262.7% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0216 | 0.0573 | 0.6046 | 1.6036 | 0.38× | 291.4 | 291.8 | 265.2% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0193 | 0.0573 | 0.5412 | 1.6036 | 0.34× | 325.5 | 325.9 | 296.3% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0141 | 0.0382 | 0.3941 | 1.0692 | 0.37× | 298.0 | 298.4 | 271.3% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0024 | 0.0001 | 0.1365 | 0.0032 | 42.88× | 2.6 | 2.6 | 2.3% | memory |
| pa_kv_update_kv1024 | `pa_kv_cache_update_ref` | 28 | 0.0040 | 0.0001 | 0.1124 | 0.0016 | 68.63× | 0.0 | 1.6 | 1.5% | memory |
| residual_add | `eltwise` | 56 | 0.0013 | 0.0001 | 0.0717 | 0.0031 | 22.92× | 0.8 | 4.8 | 4.4% | memory |
| pa_finalize_kv1024 | `single_token_finalization` | 28 | 0.0022 | 0.0001 | 0.0628 | 0.0021 | 30.13× | 0.0 | 3.7 | 3.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0617 | 0.0031 | 19.73× | 5.6 | 5.6 | 5.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0605 | 0.0016 | 38.69× | 2.9 | 2.8 | 2.6% | memory |
| rope_q | `rope_opt` | 28 | 0.0019 | 0.0001 | 0.0534 | 0.0022 | 24.11× | 3.2 | 4.6 | 4.1% | memory |
| rope_k | `rope_opt` | 28 | 0.0018 | 0.0000 | 0.0515 | 0.0012 | 43.95× | 1.7 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  |  | **6.0725** | **11.4786** | **0.53×** |  |  |  |  |

### Decode — KV=4096

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv4096 | `paged_attention_opt__single_token` | 28 | 0.1512 | 0.0871 | 4.2332 | 2.4377 | 1.74× | 221.9 | 63.3 | 57.6% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 0.8681 | 2.8316 | 0.8681 | 2.8316 | 0.31× | 358.4 | 358.8 | 326.2% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0277 | 0.0764 | 0.7745 | 2.1379 | 0.36× | 303.3 | 303.7 | 276.1% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0218 | 0.0573 | 0.6104 | 1.6036 | 0.38× | 288.6 | 289.0 | 262.7% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0216 | 0.0573 | 0.6046 | 1.6036 | 0.38× | 291.4 | 291.8 | 265.2% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0193 | 0.0573 | 0.5412 | 1.6036 | 0.34× | 325.5 | 325.9 | 296.3% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0141 | 0.0382 | 0.3941 | 1.0692 | 0.37× | 298.0 | 298.4 | 271.3% | memory |
| pa_finalize_kv4096 | `single_token_finalization` | 28 | 0.0067 | 0.0001 | 0.1879 | 0.0021 | 90.10× | 0.0 | 1.2 | 1.1% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0024 | 0.0001 | 0.1365 | 0.0032 | 42.88× | 2.6 | 2.6 | 2.3% | memory |
| pa_kv_update_kv4096 | `pa_kv_cache_update_ref` | 28 | 0.0040 | 0.0001 | 0.1124 | 0.0016 | 68.63× | 0.0 | 1.6 | 1.5% | memory |
| residual_add | `eltwise` | 56 | 0.0013 | 0.0001 | 0.0717 | 0.0031 | 22.92× | 0.8 | 4.8 | 4.4% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0617 | 0.0031 | 19.73× | 5.6 | 5.6 | 5.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0605 | 0.0016 | 38.69× | 2.9 | 2.8 | 2.6% | memory |
| rope_q | `rope_opt` | 28 | 0.0019 | 0.0001 | 0.0534 | 0.0022 | 24.11× | 3.2 | 4.6 | 4.1% | memory |
| rope_k | `rope_opt` | 28 | 0.0018 | 0.0000 | 0.0515 | 0.0012 | 43.95× | 1.7 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  |  | **8.7617** | **13.3053** | **0.66×** |  |  |  |  |

### Decode — KV=8192

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv8192 | `paged_attention_opt__single_token` | 28 | 0.2811 | 0.1740 | 7.8704 | 4.8732 | 1.62× | 238.8 | 68.1 | 61.9% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 0.8681 | 2.8316 | 0.8681 | 2.8316 | 0.31× | 358.4 | 358.8 | 326.2% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0277 | 0.0764 | 0.7745 | 2.1379 | 0.36× | 303.3 | 303.7 | 276.1% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0218 | 0.0573 | 0.6104 | 1.6036 | 0.38× | 288.6 | 289.0 | 262.7% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0216 | 0.0573 | 0.6046 | 1.6036 | 0.38× | 291.4 | 291.8 | 265.2% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0193 | 0.0573 | 0.5412 | 1.6036 | 0.34× | 325.5 | 325.9 | 296.3% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0141 | 0.0382 | 0.3941 | 1.0692 | 0.37× | 298.0 | 298.4 | 271.3% | memory |
| pa_finalize_kv8192 | `single_token_finalization` | 28 | 0.0133 | 0.0001 | 0.3723 | 0.0021 | 178.55× | 0.0 | 0.6 | 0.6% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0024 | 0.0001 | 0.1365 | 0.0032 | 42.88× | 2.6 | 2.6 | 2.3% | memory |
| pa_kv_update_kv8192 | `pa_kv_cache_update_ref` | 28 | 0.0040 | 0.0001 | 0.1129 | 0.0016 | 68.96× | 0.0 | 1.6 | 1.5% | memory |
| residual_add | `eltwise` | 56 | 0.0013 | 0.0001 | 0.0717 | 0.0031 | 22.92× | 0.8 | 4.8 | 4.4% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0617 | 0.0031 | 19.73× | 5.6 | 5.6 | 5.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0605 | 0.0016 | 38.69× | 2.9 | 2.8 | 2.6% | memory |
| rope_q | `rope_opt` | 28 | 0.0019 | 0.0001 | 0.0534 | 0.0022 | 24.11× | 3.2 | 4.6 | 4.1% | memory |
| rope_k | `rope_opt` | 28 | 0.0018 | 0.0000 | 0.0515 | 0.0012 | 43.95× | 1.7 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  |  | **12.5838** | **15.7408** | **0.80×** |  |  |  |  |

## 8. Per-kernel tables — Prefill (text-decoder only; add encoder overhead for TTFT)

### Prefill — S=512 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_qkv | `gemm_kernel` | 28 | 0.2892 | 0.2140 | 8.0973 | 5.9919 | 1.35× | 14851.7 | 47.1 | 74.0% | compute |
| pa_compute_S512 | `sdpa_micro__prefill` | 28 | 0.2825 | 0.0572 | 7.9104 | 1.6015 | 4.94× | 3808.1 | 22.3 | 20.2% | memory |
| fc_down | `gemm_kernel` | 28 | 0.2098 | 0.1605 | 5.8739 | 4.4939 | 1.31× | 15355.0 | 50.0 | 76.5% | compute |
| fc_up | `gemm_kernel` | 28 | 0.1955 | 0.1605 | 5.4743 | 4.4939 | 1.22× | 16475.8 | 53.6 | 82.1% | compute |
| fc_gate | `gemm_kernel` | 28 | 0.1947 | 0.1605 | 5.4506 | 4.4939 | 1.21× | 16547.7 | 53.9 | 82.4% | compute |
| fc_o | `gemm_kernel` | 28 | 0.1353 | 0.1070 | 3.7872 | 2.9959 | 1.26× | 15876.9 | 54.3 | 79.1% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.1345 | 0.0382 | 3.7653 | 1.0687 | 3.52× | 46.8 | 31.2 | 28.4% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0689 | 0.0191 | 1.9288 | 0.5344 | 3.61× | 45.7 | 30.5 | 27.7% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0257 | 0.0191 | 1.4621 | 1.0878 | 1.34× | 122.8 | 81.8 | 74.4% | memory |
| residual_add | `eltwise` | 56 | 0.0249 | 0.0286 | 1.3968 | 1.6015 | 0.87× | 21.0 | 126.1 | 114.7% | memory |
| rope_q | `rope_opt` | 28 | 0.0430 | 0.0405 | 1.2041 | 1.1344 | 1.06× | 73.2 | 103.6 | 94.2% | memory |
| pa_kv_update_S512 | `pa_kv_cache_update_ref` | 28 | 0.0353 | 0.0299 | 0.9871 | 0.8383 | 1.18× | 0.0 | 93.4 | 84.9% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 0.8576 | 2.8316 | 0.8576 | 2.8316 | 0.30× | 362.8 | 363.2 | 330.2% | memory |
| rope_k | `rope_opt` | 28 | 0.0219 | 0.0214 | 0.6138 | 0.6005 | 1.02× | 71.8 | 107.6 | 97.8% | memory |
| **TOTAL (text dec)** |  |  |  |  | **48.8093** | **33.7682** | **1.45×** |  |  |  |  |

### Prefill — S=1024 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S1024 | `sdpa_micro__prefill` | 28 | 0.9269 | 0.2142 | 25.9543 | 5.9977 | 4.33× | 4638.0 | 13.6 | 23.1% | compute |
| fc_qkv | `gemm_kernel` | 28 | 0.5469 | 0.4280 | 15.3124 | 11.9837 | 1.28× | 15707.4 | 34.5 | 78.3% | compute |
| fc_down | `gemm_kernel` | 28 | 0.4001 | 0.3210 | 11.2016 | 8.9878 | 1.25× | 16103.8 | 36.7 | 80.2% | compute |
| fc_gate | `gemm_kernel` | 28 | 0.3823 | 0.3210 | 10.7038 | 8.9878 | 1.19× | 16852.7 | 38.4 | 84.0% | compute |
| fc_up | `gemm_kernel` | 28 | 0.3764 | 0.3210 | 10.5404 | 8.9878 | 1.17× | 17114.0 | 39.0 | 85.3% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.2701 | 0.0763 | 7.5621 | 2.1363 | 3.54× | 46.6 | 31.1 | 28.3% | memory |
| fc_o | `gemm_kernel` | 28 | 0.2563 | 0.2140 | 7.1754 | 5.9919 | 1.20× | 16759.8 | 40.9 | 83.5% | compute |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.1354 | 0.0381 | 3.7902 | 1.0682 | 3.55× | 46.5 | 31.0 | 28.2% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0518 | 0.0381 | 2.9516 | 2.1745 | 1.36× | 121.7 | 81.0 | 73.7% | memory |
| residual_add | `eltwise` | 56 | 0.0490 | 0.0572 | 2.7414 | 3.2029 | 0.86× | 21.4 | 128.5 | 116.8% | memory |
| rope_q | `rope_opt` | 28 | 0.0870 | 0.0810 | 2.4361 | 2.2687 | 1.07× | 72.3 | 102.4 | 93.1% | memory |
| pa_kv_update_S1024 | `pa_kv_cache_update_ref` | 28 | 0.0668 | 0.0599 | 1.8709 | 1.6765 | 1.12× | 0.0 | 98.6 | 89.6% | memory |
| rope_k | `rope_opt` | 28 | 0.0410 | 0.0429 | 1.1480 | 1.2011 | 0.96× | 76.7 | 115.1 | 104.6% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 0.8576 | 2.8316 | 0.8576 | 2.8316 | 0.30× | 362.8 | 363.2 | 330.2% | memory |
| **TOTAL (text dec)** |  |  |  |  | **104.2458** | **67.4965** | **1.54×** |  |  |  |  |

### Prefill — S=4096 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S4096 | `sdpa_micro__prefill` | 28 | 13.4932 | 3.4248 | 377.8108 | 95.8932 | 3.94× | 5094.1 | 3.7 | 25.4% | compute |
| fc_qkv | `gemm_kernel` | 28 | 2.1867 | 1.7120 | 61.2273 | 47.9349 | 1.28× | 15713.1 | 23.0 | 78.3% | compute |
| fc_down | `gemm_kernel` | 28 | 1.6698 | 1.2840 | 46.7551 | 35.9512 | 1.30× | 15432.6 | 23.9 | 76.9% | compute |
| fc_gate | `gemm_kernel` | 28 | 1.4345 | 1.2840 | 40.1658 | 35.9512 | 1.12× | 17964.4 | 27.8 | 89.5% | compute |
| fc_up | `gemm_kernel` | 28 | 1.4256 | 1.2840 | 39.9158 | 35.9512 | 1.11× | 18076.9 | 27.9 | 90.1% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 1.0636 | 0.3051 | 29.7811 | 8.5422 | 3.49× | 47.4 | 31.6 | 28.7% | memory |
| fc_o | `gemm_kernel` | 28 | 1.0381 | 0.8560 | 29.0678 | 23.9674 | 1.21× | 16548.8 | 28.3 | 82.5% | compute |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.5238 | 0.1525 | 14.6663 | 4.2711 | 3.43× | 48.1 | 32.0 | 29.1% | memory |
| residual_add | `eltwise` | 56 | 0.2304 | 0.2288 | 12.9001 | 12.8117 | 1.01× | 18.2 | 109.2 | 99.3% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.2033 | 0.1525 | 11.5873 | 8.6947 | 1.33× | 124.0 | 82.5 | 75.0% | memory |
| rope_q | `rope_opt` | 28 | 0.3859 | 0.3241 | 10.8065 | 9.0749 | 1.19× | 65.2 | 92.4 | 84.0% | memory |
| pa_kv_update_S4096 | `pa_kv_cache_update_ref` | 28 | 0.3002 | 0.2395 | 8.4069 | 6.7061 | 1.25× | 0.0 | 87.8 | 79.8% | memory |
| rope_k | `rope_opt` | 28 | 0.1924 | 0.1716 | 5.3868 | 4.8044 | 1.12× | 65.4 | 98.1 | 89.2% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 0.8576 | 2.8316 | 0.8576 | 2.8316 | 0.30× | 362.8 | 363.2 | 330.2% | memory |
| **TOTAL (text dec)** |  |  |  |  | **689.3352** | **333.3858** | **2.07×** |  |  |  |  |

### Prefill — S=8192 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S8192 | `sdpa_micro__prefill` | 28 | 55.6512 | 13.6974 | 1558.2350 | 383.5260 | 4.06× | 4939.9 | 1.8 | 24.6% | compute |
| fc_qkv | `gemm_kernel` | 28 | 4.5025 | 3.4239 | 126.0690 | 95.8698 | 1.32× | 15262.6 | 20.5 | 76.0% | compute |
| fc_down | `gemm_kernel` | 28 | 3.1771 | 2.5679 | 88.9598 | 71.9023 | 1.24× | 16222.0 | 23.1 | 80.8% | compute |
| fc_gate | `gemm_kernel` | 28 | 2.8171 | 2.5679 | 78.8786 | 71.9023 | 1.10× | 18295.3 | 26.1 | 91.2% | compute |
| fc_up | `gemm_kernel` | 28 | 2.8090 | 2.5679 | 78.6517 | 71.9023 | 1.09× | 18348.1 | 26.1 | 91.4% | compute |
| fc_o | `gemm_kernel` | 28 | 2.1359 | 1.7120 | 59.8039 | 47.9349 | 1.25× | 16087.1 | 25.5 | 80.2% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 2.0945 | 0.6101 | 58.6452 | 17.0833 | 3.43× | 48.1 | 32.0 | 29.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 1.0511 | 0.3051 | 29.4315 | 8.5417 | 3.45× | 48.0 | 31.9 | 29.0% | memory |
| residual_add | `eltwise` | 56 | 0.4724 | 0.4576 | 26.4528 | 25.6234 | 1.03× | 17.8 | 106.5 | 96.9% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.4235 | 0.3051 | 24.1400 | 17.3884 | 1.39× | 119.0 | 79.2 | 72.0% | memory |
| rope_q | `rope_opt` | 28 | 0.8031 | 0.6482 | 22.4877 | 18.1499 | 1.24× | 62.7 | 88.8 | 80.7% | memory |
| pa_kv_update_S8192 | `pa_kv_cache_update_ref` | 28 | 0.5536 | 0.4790 | 15.5017 | 13.4123 | 1.16× | 0.0 | 95.2 | 86.5% | memory |
| rope_k | `rope_opt` | 28 | 0.4111 | 0.3432 | 11.5103 | 9.6088 | 1.20× | 61.2 | 91.8 | 83.5% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 0.8576 | 2.8316 | 0.8576 | 2.8316 | 0.30× | 362.8 | 363.2 | 330.2% | memory |
| **TOTAL (text dec)** |  |  |  |  | **2179.6248** | **855.6770** | **2.55×** |  |  |  |  |

## 9. Roofline analysis

Ridge point at PTL 4Xe FP16 = 182 FLOP/byte. AI < ridge ⇒ memory-bound; AI ≥ ridge ⇒ compute-bound.

**Decode characteristics**: Every FC has AI = 2·M·K·N / (K·N·2) → ~M = 1 for decode, well below ridge → **always memory-bound**. PA decode AI is also well below ridge.

**Prefill characteristics**: FC AI ≈ M scales with sequence length. For PTL 4Xe (ridge = 182), FC becomes compute-bound roughly at M ≥ ridge. SDPA prefill AI scales with S/2; becomes compute-bound around S ≥ 2·ridge.

### Bottleneck breakdown — Decode (top 5 ops by latency)

| KV | #1 | #2 | #3 | #4 | #5 |
|---:|---|---|---|---|---|
| 512 | pa_compute_kv512 (1.171ms / 20.9%) | fc_lm_head (0.868ms / 15.5%) | fc_qkv (0.774ms / 13.8%) | fc_gate (0.610ms / 10.9%) | fc_up (0.605ms / 10.8%) |
| 1024 | pa_compute_kv1024 (1.669ms / 27.5%) | fc_lm_head (0.868ms / 14.3%) | fc_qkv (0.774ms / 12.8%) | fc_gate (0.610ms / 10.1%) | fc_up (0.605ms / 10.0%) |
| 4096 | pa_compute_kv4096 (4.233ms / 48.3%) | fc_lm_head (0.868ms / 9.9%) | fc_qkv (0.774ms / 8.8%) | fc_gate (0.610ms / 7.0%) | fc_up (0.605ms / 6.9%) |
| 8192 | pa_compute_kv8192 (7.870ms / 62.5%) | fc_lm_head (0.868ms / 6.9%) | fc_qkv (0.774ms / 6.2%) | fc_gate (0.610ms / 4.9%) | fc_up (0.605ms / 4.8%) |

### Bottleneck breakdown — Prefill (top 5 ops by latency)

| S | #1 | #2 | #3 | #4 | #5 |
|---:|---|---|---|---|---|
| 512 | fc_qkv (8.097ms / 16.6%) | pa_compute_S512 (7.910ms / 16.2%) | fc_down (5.874ms / 12.0%) | fc_up (5.474ms / 11.2%) | fc_gate (5.451ms / 11.2%) |
| 1024 | pa_compute_S1024 (25.954ms / 24.9%) | fc_qkv (15.312ms / 14.7%) | fc_down (11.202ms / 10.7%) | fc_gate (10.704ms / 10.3%) | fc_up (10.540ms / 10.1%) |
| 4096 | pa_compute_S4096 (377.811ms / 54.8%) | fc_qkv (61.227ms / 8.9%) | fc_down (46.755ms / 6.8%) | fc_gate (40.166ms / 5.8%) | fc_up (39.916ms / 5.8%) |
| 8192 | pa_compute_S8192 (1558.235ms / 71.5%) | fc_qkv (126.069ms / 5.8%) | fc_down (88.960ms / 4.1%) | fc_gate (78.879ms / 3.6%) | fc_up (78.652ms / 3.6%) |

## 10. Notes, caveats & reproduction

- **Audio encoder SDPA** is approximated by 2× the measured PA-prefill (causal) kernel at S=1500. Real encoder is bidirectional non-causal; a dedicated non-causal SDPA bench would be needed for a more precise number. Encoder SDPA is a small fraction of overall TTFT, so this approximation has limited impact.
- **FC LM_Head** is `K=1024, N=151936` FP16 (~297 MB weight stream). This single layer is the dominant decode op (~50% of TPOT) and also dominates prefill last-token cost; INT8/INT4 LM_Head would substantially reduce it.
- **Decode FC bandwidth > DRAM peak (measurement caveat)**: the memory-bound decode FCs and LM_Head report effective bandwidth of ~290–360 GB/s (Eff% 260–330%, slowdown 0.3–0.4×) — above the 110 GB/s LPDDR5x peak, which is impossible to sustain from DRAM. The cliloader micro-bench keeps each op's weight buffer partially resident in the PTL memory-side cache despite the 64 MB flush. **Consequence**: the absolute decode TPOT (5.6–12.6 ms) is optimistic for the FC/LM_Head contribution; in a real end-to-end decode the 840 MB of distinct FP16 weights cannot all stay cached, so true TPOT is bounded below by the roofline (11.2–15.7 ms theoretical). The compute-bound prefill GEMMs and PA-compute kernels — which dominate at long context and have working sets too large to cache — are measured reliably.
- **PA decode** dispatches `paged_attention_opt__single_token` across all measured KV sizes (512–8192). On some other Qwen variants the runtime promotes to `paged_attention_opt__gqa_single_token` at long context; that did not trigger in this configuration.
- **KV cache precision = I8**: the Intel GPU plugin defaults `kv_cache_precision` to `i8` for PagedAttention models (`src/plugins/intel_gpu/src/runtime/execution_config.cpp:293`). This run matches that default. PA cache layout is `K = [num_blocks, NKV, HD, BLOCK+4]` (BY_CHANNEL: 1 byte per element + 4-byte scale/zp shared across BLOCK=16 tokens) and `V = [num_blocks, NKV, BLOCK, HD+4]` (BY_TOKEN: 1 byte per element + 4-byte scale/zp shared across HD per token). The theoretical bytes formula above accounts for both the INT8 payload and the scale/zp overhead, so eff% is measured against what the kernel actually streams. Switching to FP16 KV roughly doubles PA-compute bytes (and so doubles TPOT contribution from PA at long context); switching to INT4 KV would roughly halve them again, at the cost of accuracy. Observed PA-compute decode eff: 26.2% / 36.6% / 57.6% / 61.9% for KV=512/1024/4096/8192 — monotonically increasing with KV, as expected (larger working set amortizes per-launch overhead). The `KV cache key type=...` header line in every pa_bench log exposes the compiled cache element type; the `,20` channel dim is the i8 BY_CHANNEL signature (16 BLOCK + 4 scale/zp).
- **PA prefill** S=8192 produces ~55.7 ms of `sdpa_micro__prefill` per layer × 28 layers = ~1558 ms — by far the dominant prefill cost at long context. This is compute-bound (AI » ridge), so reducing weight precision will not help here; flash-attention–style algorithmic improvements or INT8 attention math would. On 4Xe this is ~3× the 12Xe cost because XMX throughput scales with Xe-core count while shared-memory bandwidth does not.
- **Encoder weights** count ~89 MB and run only once: amortized over many output tokens the encoder is a small per-token cost.

### Reproduction

```bash
# On PTL 4Xe Linux (intel@10.239.152.140):
scp utils/run_qwen3_asr_0_6B_ptl_4xe.sh intel@10.239.152.140:~/river/roofline_test_utils/
ssh intel@10.239.152.140 'bash ~/river/roofline_test_utils/run_qwen3_asr_0_6B_ptl_4xe.sh'
```

```bash
# On local Linux box:
scp -r intel@10.239.152.140:~/river/roofline_results/qwen3_asr_0_6B/ptl_4xe/*.log \
    .github/skills/dev_roofline_profiling/outputs/qwen3_asr_0_6B_ptl_4xe/logs_ptl_4xe/
cd .github/skills/dev_roofline_profiling/outputs/qwen3_asr_0_6B_ptl_4xe
python3 ../../utils/parse_logs.py logs_ptl_4xe parsed.json
python3 build_report.py
```
