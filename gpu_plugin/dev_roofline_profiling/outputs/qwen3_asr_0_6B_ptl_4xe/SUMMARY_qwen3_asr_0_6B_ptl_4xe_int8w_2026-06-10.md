# Qwen3-ASR-0.6B — Roofline on PTL 4Xe (2026-06-10)

**Platform**: Intel PTL 4Xe iGPU (4 Xe cores × 8 EU × 10 threads), 2450 MHz, 110 GB/s
**Model**: Qwen3-ASR-0.6B (audio encoder + Qwen3 text decoder), Linear+Embedding weights **INT8 (u8 g128, f16 scale)**.
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
- **L2/L3 flush** between every FC infer (64 MB Relu) so each measurement reads weights from VRAM, not on-die cache. Even so, small per-op weight tensors (QKV ~4.1 MB, MLP ~3.0 MB) can stay partially resident in the PTL memory-side/SLC cache, which may bias M=1 decode FC bandwidth (see caveats §10).
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
| enc_sdpa | `sdpa_micro__prefill (×2 bidir est.)` | 18 | 2.4342 | 0.4018 | 43.8153 | 7.2321 | 6.06× | 3312.8 | 4.4 | 16.5% | compute |
| enc_fc_ffn2 | `gemm_kernel` | 18 | 0.6796 | 0.4800 | 12.2333 | 8.6400 | 1.42× | 14175.1 | 24.6 | 70.6% | compute |
| enc_fc_ffn1 | `gemm_kernel` | 18 | 0.4738 | 0.4800 | 8.5278 | 8.6400 | 0.99× | 20334.6 | 35.2 | 101.3% | compute |
| enc_fc_qkv | `gemm_kernel` | 18 | 0.3841 | 0.3600 | 6.9143 | 6.4800 | 1.07× | 18809.8 | 34.4 | 93.7% | compute |
| enc_fc_o | `gemm_kernel` | 18 | 0.1731 | 0.1200 | 3.1151 | 2.1600 | 1.44× | 13916.8 | 35.8 | 69.3% | compute |
| enc_outproj | `gemm_kernel` | 1 | 0.1737 | 0.1371 | 0.1737 | 0.1371 | 1.27× | 15850.8 | 38.5 | 79.0% | compute |
| **TOTAL (encoder fixed)** |  |  |  |  | **74.7795** | **33.2892** | **2.25×** |  |  |  |  |

## 6. Token latency summary

**TTFT** = audio encoder (74.78 ms) + text decoder prefill.
**TPOT** = per output token decode latency at the listed KV context.

### Prefill — TTFT

| S (text ctx) | Encoder (ms) | Text-decoder prefill (ms) | **TTFT (ms)** | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 74.78 | 46.02 | **120.80** | 0.2359 | 4238 |
| 1024 | 74.78 | 97.69 | **172.47** | 0.1684 | 5937 |
| 4096 | 74.78 | 660.05 | **734.83** | 0.1794 | 5574 |
| 8192 | 74.78 | 2111.69 | **2186.47** | 0.2669 | 3747 |

### Decode — TPOT

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 512 | 8.4325 | 118.6 |
| 1024 | 8.8893 | 112.5 |
| 4096 | 11.5957 | 86.2 |
| 8192 | 15.2983 | 65.4 |

### Measured vs. theoretical latency (lower bound = roofline)

The **theoretical** column is the sum over all kernels of
`max(bytes/BW, flops/peak_TFLOPS)` — i.e. the absolute minimum time
each kernel could take given its bound. `slowdown = measured / theo` (1.0× = at roofline).

#### Decode (per output token)

| KV (ctx) | Measured TPOT (ms) | Theoretical TPOT (ms) | Slowdown | Wall-clock tokens/s | Roofline tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 8.4325 | 5.8404 | 1.44× | 118.6 | 171.2 |
| 1024 | 8.8893 | 6.1449 | 1.45× | 112.5 | 162.7 |
| 4096 | 11.5957 | 7.9716 | 1.45× | 86.2 | 125.4 |
| 8192 | 15.2983 | 10.4071 | 1.47× | 65.4 | 96.1 |

#### Prefill / TTFT

| S (text ctx) | Measured TTFT (ms) | Theoretical TTFT (ms) | Slowdown | Meas tokens/s | Roofline tokens/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 120.80 | 65.67 | 1.84× | 4238 | 7797 |
| 1024 | 172.47 | 99.39 | 1.74× | 5937 | 10302 |
| 4096 | 734.83 | 365.28 | 2.01× | 5574 | 11213 |
| 8192 | 2186.47 | 887.57 | 2.46× | 3747 | 9230 |

#### Audio encoder fixed overhead

| Component | Measured (ms) | Theoretical (ms) | Slowdown |
|---|---:|---:|---:|
| Audio encoder forward (S=1500, 18 layers) | 74.78 | 33.29 | 2.25× |

### End-to-end latency estimate (output = 512 tokens)

Approximation: decode uses the TPOT at KV = S_prefill (start-of-decode value). Real decode TPOT increases as KV grows toward S_prefill + 511.

| S (text ctx) | TTFT (ms) | TPOT @ KV=S (ms) | Measured total (ms) | Theoretical total (ms) | Slowdown |
|---:|---:|---:|---:|---:|---:|
| 512 | 120.80 | 8.4325 | 4438.2 | 3055.9 | 1.45× |
| 1024 | 172.47 | 8.8893 | 4723.8 | 3245.6 | 1.46× |
| 4096 | 734.83 | 11.5957 | 6671.8 | 4446.7 | 1.50× |
| 8192 | 2186.47 | 15.2983 | 10019.2 | 6216.0 | 1.61× |

## 7. Per-kernel tables — Decode (1 query token, KV = context)

### Decode — KV=512

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_lm_head | `gemm_kernel` | 1 | 1.5647 | 1.4393 | 1.5647 | 1.4393 | 1.09× | 198.9 | 101.2 | 92.0% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0473 | 0.0388 | 1.3233 | 1.0869 | 1.22× | 177.5 | 90.3 | 82.1% | memory |
| pa_compute_kv512 | `paged_attention_opt__single_token` | 28 | 0.0416 | 0.0109 | 1.1661 | 0.3065 | 3.80× | 100.7 | 28.9 | 26.3% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0367 | 0.0291 | 1.0281 | 0.8153 | 1.26× | 171.3 | 87.2 | 79.3% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0365 | 0.0291 | 1.0208 | 0.8153 | 1.25× | 172.6 | 87.9 | 79.9% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0353 | 0.0291 | 0.9881 | 0.8153 | 1.21× | 178.3 | 90.8 | 82.5% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0249 | 0.0194 | 0.6960 | 0.5437 | 1.28× | 168.7 | 85.9 | 78.1% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0024 | 0.0001 | 0.1396 | 0.0032 | 43.86× | 2.5 | 2.5 | 2.3% | memory |
| pa_kv_update_kv512 | `pa_kv_cache_update_ref` | 28 | 0.0049 | 0.0001 | 0.1385 | 0.0016 | 84.57× | 0.0 | 1.3 | 1.2% | memory |
| pa_finalize_kv512 | `single_token_finalization` | 28 | 0.0025 | 0.0001 | 0.0699 | 0.0021 | 33.52× | 0.0 | 3.3 | 3.0% | memory |
| residual_add | `eltwise` | 56 | 0.0012 | 0.0001 | 0.0697 | 0.0031 | 22.29× | 0.8 | 4.9 | 4.5% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0616 | 0.0031 | 19.70× | 5.6 | 5.6 | 5.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0610 | 0.0016 | 38.99× | 2.8 | 2.8 | 2.6% | memory |
| rope_q | `rope_opt` | 28 | 0.0019 | 0.0001 | 0.0534 | 0.0022 | 24.09× | 3.2 | 4.6 | 4.2% | memory |
| rope_k | `rope_opt` | 28 | 0.0018 | 0.0000 | 0.0517 | 0.0012 | 44.09× | 1.7 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  |  | **8.4325** | **5.8404** | **1.44×** |  |  |  |  |

### Decode — KV=1024

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv1024 | `paged_attention_opt__single_token` | 28 | 0.0592 | 0.0218 | 1.6572 | 0.6110 | 2.71× | 141.7 | 40.6 | 36.9% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5647 | 1.4393 | 1.5647 | 1.4393 | 1.09× | 198.9 | 101.2 | 92.0% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0473 | 0.0388 | 1.3233 | 1.0869 | 1.22× | 177.5 | 90.3 | 82.1% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0367 | 0.0291 | 1.0281 | 0.8153 | 1.26× | 171.3 | 87.2 | 79.3% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0365 | 0.0291 | 1.0208 | 0.8153 | 1.25× | 172.6 | 87.9 | 79.9% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0353 | 0.0291 | 0.9881 | 0.8153 | 1.21× | 178.3 | 90.8 | 82.5% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0249 | 0.0194 | 0.6960 | 0.5437 | 1.28× | 168.7 | 85.9 | 78.1% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0024 | 0.0001 | 0.1396 | 0.0032 | 43.86× | 2.5 | 2.5 | 2.3% | memory |
| pa_kv_update_kv1024 | `pa_kv_cache_update_ref` | 28 | 0.0040 | 0.0001 | 0.1116 | 0.0016 | 68.19× | 0.0 | 1.6 | 1.5% | memory |
| residual_add | `eltwise` | 56 | 0.0012 | 0.0001 | 0.0697 | 0.0031 | 22.29× | 0.8 | 4.9 | 4.5% | memory |
| pa_finalize_kv1024 | `single_token_finalization` | 28 | 0.0022 | 0.0001 | 0.0625 | 0.0021 | 29.98× | 0.0 | 3.7 | 3.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0616 | 0.0031 | 19.70× | 5.6 | 5.6 | 5.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0610 | 0.0016 | 38.99× | 2.8 | 2.8 | 2.6% | memory |
| rope_q | `rope_opt` | 28 | 0.0019 | 0.0001 | 0.0534 | 0.0022 | 24.09× | 3.2 | 4.6 | 4.2% | memory |
| rope_k | `rope_opt` | 28 | 0.0018 | 0.0000 | 0.0517 | 0.0012 | 44.09× | 1.7 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  |  | **8.8893** | **6.1449** | **1.45×** |  |  |  |  |

### Decode — KV=4096

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv4096 | `paged_attention_opt__single_token` | 28 | 0.1514 | 0.0871 | 4.2398 | 2.4377 | 1.74× | 221.6 | 63.2 | 57.5% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5647 | 1.4393 | 1.5647 | 1.4393 | 1.09× | 198.9 | 101.2 | 92.0% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0473 | 0.0388 | 1.3233 | 1.0869 | 1.22× | 177.5 | 90.3 | 82.1% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0367 | 0.0291 | 1.0281 | 0.8153 | 1.26× | 171.3 | 87.2 | 79.3% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0365 | 0.0291 | 1.0208 | 0.8153 | 1.25× | 172.6 | 87.9 | 79.9% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0353 | 0.0291 | 0.9881 | 0.8153 | 1.21× | 178.3 | 90.8 | 82.5% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0249 | 0.0194 | 0.6960 | 0.5437 | 1.28× | 168.7 | 85.9 | 78.1% | memory |
| pa_finalize_kv4096 | `single_token_finalization` | 28 | 0.0066 | 0.0001 | 0.1842 | 0.0021 | 88.35× | 0.0 | 1.2 | 1.1% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0024 | 0.0001 | 0.1396 | 0.0032 | 43.86× | 2.5 | 2.5 | 2.3% | memory |
| pa_kv_update_kv4096 | `pa_kv_cache_update_ref` | 28 | 0.0041 | 0.0001 | 0.1137 | 0.0016 | 69.43× | 0.0 | 1.6 | 1.4% | memory |
| residual_add | `eltwise` | 56 | 0.0012 | 0.0001 | 0.0697 | 0.0031 | 22.29× | 0.8 | 4.9 | 4.5% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0616 | 0.0031 | 19.70× | 5.6 | 5.6 | 5.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0610 | 0.0016 | 38.99× | 2.8 | 2.8 | 2.6% | memory |
| rope_q | `rope_opt` | 28 | 0.0019 | 0.0001 | 0.0534 | 0.0022 | 24.09× | 3.2 | 4.6 | 4.2% | memory |
| rope_k | `rope_opt` | 28 | 0.0018 | 0.0000 | 0.0517 | 0.0012 | 44.09× | 1.7 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  |  | **11.5957** | **7.9716** | **1.45×** |  |  |  |  |

### Decode — KV=8192

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv8192 | `paged_attention_opt__single_token` | 28 | 0.2776 | 0.1740 | 7.7729 | 4.8732 | 1.60× | 241.7 | 69.0 | 62.7% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5647 | 1.4393 | 1.5647 | 1.4393 | 1.09× | 198.9 | 101.2 | 92.0% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.0473 | 0.0388 | 1.3233 | 1.0869 | 1.22× | 177.5 | 90.3 | 82.1% | memory |
| fc_gate | `gemm_kernel` | 28 | 0.0367 | 0.0291 | 1.0281 | 0.8153 | 1.26× | 171.3 | 87.2 | 79.3% | memory |
| fc_up | `gemm_kernel` | 28 | 0.0365 | 0.0291 | 1.0208 | 0.8153 | 1.25× | 172.6 | 87.9 | 79.9% | memory |
| fc_down | `gemm_kernel` | 28 | 0.0353 | 0.0291 | 0.9881 | 0.8153 | 1.21× | 178.3 | 90.8 | 82.5% | memory |
| fc_o | `gemm_kernel` | 28 | 0.0249 | 0.0194 | 0.6960 | 0.5437 | 1.28× | 168.7 | 85.9 | 78.1% | memory |
| pa_finalize_kv8192 | `single_token_finalization` | 28 | 0.0128 | 0.0001 | 0.3578 | 0.0021 | 171.58× | 0.0 | 0.6 | 0.6% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0024 | 0.0001 | 0.1396 | 0.0032 | 43.86× | 2.5 | 2.5 | 2.3% | memory |
| pa_kv_update_kv8192 | `pa_kv_cache_update_ref` | 28 | 0.0039 | 0.0001 | 0.1096 | 0.0016 | 66.97× | 0.0 | 1.6 | 1.5% | memory |
| residual_add | `eltwise` | 56 | 0.0012 | 0.0001 | 0.0697 | 0.0031 | 22.29× | 0.8 | 4.9 | 4.5% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0616 | 0.0031 | 19.70× | 5.6 | 5.6 | 5.1% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0022 | 0.0001 | 0.0610 | 0.0016 | 38.99× | 2.8 | 2.8 | 2.6% | memory |
| rope_q | `rope_opt` | 28 | 0.0019 | 0.0001 | 0.0534 | 0.0022 | 24.09× | 3.2 | 4.6 | 4.2% | memory |
| rope_k | `rope_opt` | 28 | 0.0018 | 0.0000 | 0.0517 | 0.0012 | 44.09× | 1.7 | 2.5 | 2.3% | memory |
| **TOTAL** |  |  |  |  | **15.2983** | **10.4071** | **1.47×** |  |  |  |  |

## 8. Per-kernel tables — Prefill (text-decoder only; add encoder overhead for TTFT)

### Prefill — S=512 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S512 | `sdpa_micro__prefill` | 28 | 0.2822 | 0.0572 | 7.9027 | 1.6015 | 4.93× | 3811.8 | 22.3 | 20.3% | memory |
| fc_qkv | `gemm_kernel` | 28 | 0.2225 | 0.2140 | 6.2299 | 5.9919 | 1.04× | 19303.7 | 42.7 | 96.2% | compute |
| fc_down | `gemm_kernel` | 28 | 0.2143 | 0.1605 | 5.9992 | 4.4939 | 1.33× | 15034.3 | 34.5 | 74.9% | compute |
| fc_up | `gemm_kernel` | 28 | 0.1688 | 0.1605 | 4.7264 | 4.4939 | 1.05× | 19083.2 | 43.8 | 95.1% | compute |
| fc_gate | `gemm_kernel` | 28 | 0.1686 | 0.1605 | 4.7203 | 4.4939 | 1.05× | 19107.7 | 43.8 | 95.2% | compute |
| fc_o | `gemm_kernel` | 28 | 0.1368 | 0.1070 | 3.8292 | 2.9959 | 1.28× | 15702.8 | 38.6 | 78.2% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.1327 | 0.0382 | 3.7161 | 1.0687 | 3.48× | 47.4 | 31.6 | 28.8% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.0655 | 0.0191 | 1.8343 | 0.5344 | 3.43× | 48.1 | 32.0 | 29.1% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5620 | 1.4393 | 1.5620 | 1.4393 | 1.09× | 199.2 | 101.4 | 92.1% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0243 | 0.0191 | 1.3865 | 1.0878 | 1.27× | 129.5 | 86.3 | 78.5% | memory |
| residual_add | `eltwise` | 56 | 0.0243 | 0.0286 | 1.3582 | 1.6015 | 0.85× | 21.6 | 129.7 | 117.9% | memory |
| rope_q | `rope_opt` | 28 | 0.0418 | 0.0405 | 1.1708 | 1.1344 | 1.03× | 75.2 | 106.6 | 96.9% | memory |
| pa_kv_update_S512 | `pa_kv_cache_update_ref` | 28 | 0.0356 | 0.0299 | 0.9975 | 0.8383 | 1.19× | 0.0 | 92.4 | 84.0% | memory |
| rope_k | `rope_opt` | 28 | 0.0209 | 0.0214 | 0.5858 | 0.6005 | 0.98× | 75.2 | 112.8 | 102.5% | memory |
| **TOTAL (text dec)** |  |  |  |  | **46.0189** | **32.3759** | **1.42×** |  |  |  |  |

### Prefill — S=1024 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S1024 | `sdpa_micro__prefill` | 28 | 0.9244 | 0.2142 | 25.8820 | 5.9977 | 4.32× | 4651.0 | 13.6 | 23.2% | compute |
| fc_qkv | `gemm_kernel` | 28 | 0.4169 | 0.4280 | 11.6734 | 11.9837 | 0.97× | 20603.9 | 35.4 | 102.7% | compute |
| fc_down | `gemm_kernel` | 28 | 0.4053 | 0.3210 | 11.3472 | 8.9878 | 1.26× | 15897.2 | 28.6 | 79.2% | compute |
| fc_gate | `gemm_kernel` | 28 | 0.3191 | 0.3210 | 8.9350 | 8.9878 | 0.99× | 20188.9 | 36.3 | 100.6% | compute |
| fc_up | `gemm_kernel` | 28 | 0.3188 | 0.3210 | 8.9271 | 8.9878 | 0.99× | 20206.8 | 36.3 | 100.7% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 0.2654 | 0.0763 | 7.4323 | 2.1363 | 3.48× | 47.4 | 31.6 | 28.7% | memory |
| fc_o | `gemm_kernel` | 28 | 0.2572 | 0.2140 | 7.2023 | 5.9919 | 1.20× | 16697.2 | 32.7 | 83.2% | compute |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.1358 | 0.0381 | 3.8019 | 1.0682 | 3.56× | 46.4 | 30.9 | 28.1% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.0487 | 0.0381 | 2.7756 | 2.1745 | 1.28× | 129.4 | 86.2 | 78.3% | memory |
| residual_add | `eltwise` | 56 | 0.0488 | 0.0572 | 2.7307 | 3.2029 | 0.85× | 21.5 | 129.0 | 117.3% | memory |
| rope_q | `rope_opt` | 28 | 0.0865 | 0.0810 | 2.4210 | 2.2687 | 1.07× | 72.8 | 103.1 | 93.7% | memory |
| pa_kv_update_S1024 | `pa_kv_cache_update_ref` | 28 | 0.0666 | 0.0599 | 1.8637 | 1.6765 | 1.11× | 0.0 | 99.0 | 90.0% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5620 | 1.4393 | 1.5620 | 1.4393 | 1.09× | 199.2 | 101.4 | 92.1% | memory |
| rope_k | `rope_opt` | 28 | 0.0405 | 0.0429 | 1.1333 | 1.2011 | 0.94× | 77.7 | 116.6 | 106.0% | memory |
| **TOTAL (text dec)** |  |  |  |  | **97.6875** | **66.1042** | **1.48×** |  |  |  |  |

### Prefill — S=4096 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S4096 | `sdpa_micro__prefill` | 28 | 13.4863 | 3.4248 | 377.6166 | 95.8932 | 3.94× | 5096.7 | 3.7 | 25.4% | compute |
| fc_qkv | `gemm_kernel` | 28 | 1.7313 | 1.7120 | 48.4758 | 47.9349 | 1.01× | 19846.4 | 26.7 | 98.9% | compute |
| fc_down | `gemm_kernel` | 28 | 1.5963 | 1.2840 | 44.6955 | 35.9512 | 1.24× | 16143.8 | 23.0 | 80.4% | compute |
| fc_gate | `gemm_kernel` | 28 | 1.2382 | 1.2840 | 34.6682 | 35.9512 | 0.96× | 20813.1 | 29.7 | 103.7% | compute |
| fc_up | `gemm_kernel` | 28 | 1.2326 | 1.2840 | 34.5127 | 35.9512 | 0.96× | 20906.9 | 29.8 | 104.2% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 1.0410 | 0.3051 | 29.1492 | 8.5422 | 3.41× | 48.4 | 32.2 | 29.3% | memory |
| fc_o | `gemm_kernel` | 28 | 0.9541 | 0.8560 | 26.7152 | 23.9674 | 1.11× | 18006.1 | 28.6 | 89.7% | compute |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 0.5222 | 0.1525 | 14.6216 | 4.2711 | 3.42× | 48.3 | 32.1 | 29.2% | memory |
| residual_add | `eltwise` | 56 | 0.2270 | 0.2288 | 12.7110 | 12.8117 | 0.99× | 18.5 | 110.9 | 100.8% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.1913 | 0.1525 | 10.9063 | 8.6947 | 1.25× | 131.7 | 87.7 | 79.7% | memory |
| rope_q | `rope_opt` | 28 | 0.3809 | 0.3241 | 10.6648 | 9.0749 | 1.18× | 66.1 | 93.6 | 85.1% | memory |
| pa_kv_update_S4096 | `pa_kv_cache_update_ref` | 28 | 0.2993 | 0.2395 | 8.3804 | 6.7061 | 1.25× | 0.0 | 88.0 | 80.0% | memory |
| rope_k | `rope_opt` | 28 | 0.1919 | 0.1716 | 5.3732 | 4.8044 | 1.12× | 65.6 | 98.4 | 89.4% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5620 | 1.4393 | 1.5620 | 1.4393 | 1.09× | 199.2 | 101.4 | 92.1% | memory |
| **TOTAL (text dec)** |  |  |  |  | **660.0525** | **331.9935** | **1.99×** |  |  |  |  |

### Prefill — S=8192 (text decoder only)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S8192 | `sdpa_micro__prefill` | 28 | 55.4544 | 13.6974 | 1552.7229 | 383.5260 | 4.05× | 4957.4 | 1.8 | 24.7% | compute |
| fc_qkv | `gemm_kernel` | 28 | 3.3411 | 3.4239 | 93.5515 | 95.8698 | 0.98× | 20567.8 | 26.4 | 102.5% | compute |
| fc_down | `gemm_kernel` | 28 | 3.1467 | 2.5679 | 88.1076 | 71.9023 | 1.23× | 16378.9 | 22.3 | 81.6% | compute |
| fc_up | `gemm_kernel` | 28 | 2.4839 | 2.5679 | 69.5491 | 71.9023 | 0.97× | 20749.5 | 28.3 | 103.4% | compute |
| fc_gate | `gemm_kernel` | 28 | 2.4232 | 2.5679 | 67.8489 | 71.9023 | 0.94× | 21269.4 | 29.0 | 106.0% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 28 | 2.0828 | 0.6101 | 58.3172 | 17.0833 | 3.41× | 48.4 | 32.2 | 29.3% | memory |
| fc_o | `gemm_kernel` | 28 | 1.8971 | 1.7120 | 53.1189 | 47.9349 | 1.11× | 18111.7 | 27.6 | 90.2% | compute |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 28 | 1.0476 | 0.3051 | 29.3324 | 8.5417 | 3.43× | 48.1 | 32.0 | 29.1% | memory |
| residual_add | `eltwise` | 56 | 0.4669 | 0.4576 | 26.1474 | 25.6234 | 1.02× | 18.0 | 107.8 | 98.0% | memory |
| rope_q | `rope_opt` | 28 | 0.8021 | 0.6482 | 22.4579 | 18.1499 | 1.24× | 62.8 | 88.9 | 80.8% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 57 | 0.3872 | 0.3051 | 22.0685 | 17.3884 | 1.27× | 130.2 | 86.7 | 78.8% | memory |
| pa_kv_update_S8192 | `pa_kv_cache_update_ref` | 28 | 0.5520 | 0.4790 | 15.4546 | 13.4123 | 1.15× | 0.0 | 95.5 | 86.8% | memory |
| rope_k | `rope_opt` | 28 | 0.4088 | 0.3432 | 11.4469 | 9.6088 | 1.19× | 61.6 | 92.3 | 83.9% | memory |
| fc_lm_head | `gemm_kernel` | 1 | 1.5620 | 1.4393 | 1.5620 | 1.4393 | 1.09× | 199.2 | 101.4 | 92.1% | memory |
| **TOTAL (text dec)** |  |  |  |  | **2111.6858** | **854.2847** | **2.47×** |  |  |  |  |

## 9. Roofline analysis

Ridge point at PTL 4Xe FP16 = 182 FLOP/byte. AI < ridge ⇒ memory-bound; AI ≥ ridge ⇒ compute-bound.

**Decode characteristics**: Every FC has AI = 2·M·K·N / (K·N·2) → ~M = 1 for decode, well below ridge → **always memory-bound**. PA decode AI is also well below ridge.

**Prefill characteristics**: FC AI ≈ M scales with sequence length. For PTL 4Xe (ridge = 182), FC becomes compute-bound roughly at M ≥ ridge. SDPA prefill AI scales with S/2; becomes compute-bound around S ≥ 2·ridge.

### Bottleneck breakdown — Decode (top 5 ops by latency)

| KV | #1 | #2 | #3 | #4 | #5 |
|---:|---|---|---|---|---|
| 512 | fc_lm_head (1.565ms / 18.6%) | fc_qkv (1.323ms / 15.7%) | pa_compute_kv512 (1.166ms / 13.8%) | fc_gate (1.028ms / 12.2%) | fc_up (1.021ms / 12.1%) |
| 1024 | pa_compute_kv1024 (1.657ms / 18.6%) | fc_lm_head (1.565ms / 17.6%) | fc_qkv (1.323ms / 14.9%) | fc_gate (1.028ms / 11.6%) | fc_up (1.021ms / 11.5%) |
| 4096 | pa_compute_kv4096 (4.240ms / 36.6%) | fc_lm_head (1.565ms / 13.5%) | fc_qkv (1.323ms / 11.4%) | fc_gate (1.028ms / 8.9%) | fc_up (1.021ms / 8.8%) |
| 8192 | pa_compute_kv8192 (7.773ms / 50.8%) | fc_lm_head (1.565ms / 10.2%) | fc_qkv (1.323ms / 8.6%) | fc_gate (1.028ms / 6.7%) | fc_up (1.021ms / 6.7%) |

### Bottleneck breakdown — Prefill (top 5 ops by latency)

| S | #1 | #2 | #3 | #4 | #5 |
|---:|---|---|---|---|---|
| 512 | pa_compute_S512 (7.903ms / 17.2%) | fc_qkv (6.230ms / 13.5%) | fc_down (5.999ms / 13.0%) | fc_up (4.726ms / 10.3%) | fc_gate (4.720ms / 10.3%) |
| 1024 | pa_compute_S1024 (25.882ms / 26.5%) | fc_qkv (11.673ms / 11.9%) | fc_down (11.347ms / 11.6%) | fc_gate (8.935ms / 9.1%) | fc_up (8.927ms / 9.1%) |
| 4096 | pa_compute_S4096 (377.617ms / 57.2%) | fc_qkv (48.476ms / 7.3%) | fc_down (44.696ms / 6.8%) | fc_gate (34.668ms / 5.3%) | fc_up (34.513ms / 5.2%) |
| 8192 | pa_compute_S8192 (1552.723ms / 73.5%) | fc_qkv (93.552ms / 4.4%) | fc_down (88.108ms / 4.2%) | fc_up (69.549ms / 3.3%) | fc_gate (67.849ms / 3.2%) |

## 10. Notes, caveats & reproduction

- **Audio encoder SDPA** is approximated by 2× the measured PA-prefill (causal) kernel at S=1500. Real encoder is bidirectional non-causal; a dedicated non-causal SDPA bench would be needed for a more precise number. Encoder SDPA is a small fraction of overall TTFT, so this approximation has limited impact.
- **FC LM_Head** is `K=1024, N=151936` INT8 (u8 g128, f16 scale) (~158 MB weight stream incl. scales). This single layer is typically a dominant decode op and also dominates prefill last-token cost.
- **Decode FC cache caveat**: in this run, decode FC efficiency stays within DRAM peak (max observed 92.0%). Still, partial cache residency may exist; treat roofline lower bounds as a conservative baseline for end-to-end decode.
- **PA decode** dispatches `paged_attention_opt__single_token` across all measured KV sizes (512–8192). On some other Qwen variants the runtime promotes to `paged_attention_opt__gqa_single_token` at long context; that did not trigger in this configuration.
- **KV cache precision = I8**: the Intel GPU plugin defaults `kv_cache_precision` to `i8` for PagedAttention models (`src/plugins/intel_gpu/src/runtime/execution_config.cpp:293`). This run matches that default. PA cache layout is `K = [num_blocks, NKV, HD, BLOCK+4]` (BY_CHANNEL: 1 byte per element + 4-byte scale/zp shared across BLOCK=16 tokens) and `V = [num_blocks, NKV, BLOCK, HD+4]` (BY_TOKEN: 1 byte per element + 4-byte scale/zp shared across HD per token). The theoretical bytes formula above accounts for both the INT8 payload and the scale/zp overhead, so eff% is measured against what the kernel actually streams. Switching to FP16 KV roughly doubles PA-compute bytes (and so doubles TPOT contribution from PA at long context); switching to INT4 KV would roughly halve them again, at the cost of accuracy. Observed PA-compute decode eff: 26.3% / 36.9% / 57.5% / 62.7% for KV=512/1024/4096/8192 — monotonically increasing with KV, as expected (larger working set amortizes per-launch overhead). The `KV cache key type=...` header line in every pa_bench log exposes the compiled cache element type; the `,20` channel dim is the i8 BY_CHANNEL signature (16 BLOCK + 4 scale/zp).
- **PA prefill** S=8192 produces ~55.5 ms of `sdpa_micro__prefill` per layer × 28 layers = ~1553 ms — by far the dominant prefill cost at long context. This is compute-bound (AI » ridge), so reducing weight precision will not help here; flash-attention–style algorithmic improvements or INT8 attention math would. On 4Xe this is ~3× the 12Xe cost because XMX throughput scales with Xe-core count while shared-memory bandwidth does not.
- **Encoder weights** count ~177 MB and run only once: amortized over many output tokens the encoder is a small per-token cost.

### Reproduction

```bash
# On PTL 4Xe Linux (intel@10.239.152.140):
scp utils/run_qwen3_asr_0_6B_ptl_4xe_int8w.sh intel@10.239.152.140:~/river/roofline_test_utils/
ssh intel@10.239.152.140 'bash ~/river/roofline_test_utils/run_qwen3_asr_0_6B_ptl_4xe_int8w.sh'
```

```bash
# On local Linux box:
scp -r intel@10.239.152.140:~/river/roofline_results/qwen3_asr_0_6B/ptl_4xe_int8w/*.log \
    .github/skills/dev_roofline_profiling/outputs/qwen3_asr_0_6B_ptl_4xe/logs_ptl_4xe_int8w/
cd .github/skills/dev_roofline_profiling/outputs/qwen3_asr_0_6B_ptl_4xe
python3 ../../utils/parse_logs.py logs_ptl_4xe_int8w parsed_int8w.json
python3 build_report.py
```
