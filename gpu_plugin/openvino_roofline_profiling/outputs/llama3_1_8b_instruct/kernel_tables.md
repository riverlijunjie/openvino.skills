# meta-llama/Llama-3.1-8B-Instruct (dense decoder, GQA, SwiGLU) — Per-token-size Kernel Tables

## Platform: PTL

### DECODE — per-kv-size (M=1)

#### PTL — decode kv=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.3032 | 32 | 9.703 | 387 | 100.7 | 95.9% | memory |
| fc_up | `gemm_kernel` | 0.2974 | 32 | 9.516 | 395 | 102.7 | 97.8% | memory |
| fc_gate | `gemm_kernel` | 0.2964 | 32 | 9.486 | 396 | 103.0 | 98.1% | memory |
| lm_head | `gemm_kernel` | 5.0876 | 1 | 5.088 | 207 | 104.9 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 32 | 4.134 | 390 | 101.4 | 96.5% | memory |
| fc_o | `gemm_kernel` | 0.0877 | 32 | 2.806 | 383 | 99.6 | 94.8% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0512 | 32 | 1.639 | 289 | 36.1 | 34.4% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 64 | 0.201 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 64 | 0.124 | 2 | 12.7 | 12.1% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0038 | 32 | 0.122 | 289 | 36.1 | 34.4% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0031 | 32 | 0.099 | 289 | 36.1 | 34.4% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0030 | 32 | 0.097 | 5 | 28.4 | 27.1% | memory |
| swish | `activation_opt_0_0` | 0.0024 | 32 | 0.075 | 24 | 24.3 | 23.2% | memory |
| rope_q | `rope_opt` | 0.0022 | 32 | 0.069 | 19 | 7.6 | 7.3% | memory |
| rope_k | `rope_opt` | 0.0020 | 32 | 0.063 | 5 | 2.1 | 2.0% | memory |

**Total inference time (this stage)** ≈ **43.22 ms**

#### PTL — decode kv=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.3032 | 32 | 9.703 | 387 | 100.7 | 95.9% | memory |
| fc_up | `gemm_kernel` | 0.2974 | 32 | 9.516 | 395 | 102.7 | 97.8% | memory |
| fc_gate | `gemm_kernel` | 0.2964 | 32 | 9.486 | 396 | 103.0 | 98.1% | memory |
| lm_head | `gemm_kernel` | 5.0876 | 1 | 5.088 | 207 | 104.9 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 32 | 4.134 | 390 | 101.4 | 96.5% | memory |
| fc_o | `gemm_kernel` | 0.0877 | 32 | 2.806 | 383 | 99.6 | 94.8% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0842 | 32 | 2.695 | 365 | 45.6 | 43.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 64 | 0.201 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 64 | 0.124 | 2 | 12.7 | 12.1% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0039 | 32 | 0.124 | 365 | 45.6 | 43.5% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0030 | 32 | 0.097 | 5 | 28.4 | 27.1% | memory |
| swish | `activation_opt_0_0` | 0.0024 | 32 | 0.075 | 24 | 24.3 | 23.2% | memory |
| rope_q | `rope_opt` | 0.0022 | 32 | 0.069 | 19 | 7.6 | 7.3% | memory |
| rope_k | `rope_opt` | 0.0020 | 32 | 0.063 | 5 | 2.1 | 2.0% | memory |

**Total inference time (this stage)** ≈ **44.18 ms**

### PREFILL — per-S-size (compute-bound uses INT8 XMX peak = 118.0 TOPS)

#### PTL — prefill S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 2.1892 | 32 | 70.055 | 47384 | 38.4 | 40.2% | compute |
| fc_up | `gemm_kernel` | 1.7847 | 32 | 57.112 | 63502 | 51.4 | 53.8% | compute |
| fc_gate | `gemm_kernel` | 1.6991 | 32 | 54.370 | 66902 | 54.2 | 56.7% | compute |
| fc_qkv | `gemm_kernel` | 0.7270 | 32 | 23.263 | 62510 | 56.4 | 53.0% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 0.6514 | 32 | 20.846 | 24241 | 47.3 | 20.5% | compute |
| fc_o | `gemm_kernel` | 0.5200 | 32 | 16.640 | 54482 | 53.6 | 46.2% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.3487 | 32 | 11.160 | 47384 | 38.4 | 40.2% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.1106 | 32 | 3.541 | 54482 | 53.6 | 46.2% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.1090 | 32 | 3.489 | 63502 | 51.4 | 53.8% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.0985 | 32 | 3.151 | 66902 | 54.2 | 56.7% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.0975 | 32 | 3.121 | 62510 | 56.4 | 53.0% | compute |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0573 | 32 | 1.833 | 24241 | 47.3 | 20.5% | compute |

**Total inference time (this stage)** ≈ **268.58 ms**

#### PTL — prefill S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 4.2772 | 32 | 136.870 | 48374 | 27.2 | 41.0% | compute |
| fc_gate | `gemm_kernel` | 3.2843 | 32 | 105.097 | 69087 | 38.8 | 58.6% | compute |
| fc_up | `gemm_kernel` | 3.2208 | 32 | 103.066 | 70410 | 39.6 | 59.7% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 2.2989 | 32 | 73.563 | 28410 | 27.7 | 24.1% | compute |
| fc_qkv | `gemm_kernel` | 1.4189 | 32 | 45.405 | 62625 | 41.0 | 53.1% | compute |
| fc_o | `gemm_kernel` | 0.9741 | 32 | 31.173 | 57248 | 42.1 | 48.5% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.6948 | 32 | 22.234 | 48374 | 27.2 | 41.0% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.2271 | 32 | 7.267 | 62625 | 41.0 | 53.1% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.2262 | 32 | 7.240 | 57248 | 42.1 | 48.5% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.1971 | 32 | 6.307 | 69087 | 38.8 | 58.6% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.1951 | 32 | 6.244 | 70410 | 39.6 | 59.7% | compute |

**Total inference time (this stage)** ≈ **544.47 ms**

