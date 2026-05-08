# Qwen/Qwen3-8B (dense decoder, GQA, SwiGLU) — Per-token-size Kernel Tables

## Platform: BMG

### DECODE — per-kv-size (M=1)

#### BMG — decode kv=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0155 | 36 | 0.558 | 918 | 114.8 | 25.3% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0016 | 36 | 0.056 | 918 | 114.8 | 25.3% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0012 | 36 | 0.044 | 918 | 114.8 | 25.3% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **12.29 ms**

#### BMG — decode kv=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0295 | 36 | 1.063 | 1017 | 127.2 | 28.1% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0021 | 36 | 0.074 | 1017 | 127.2 | 28.1% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **12.77 ms**

#### BMG — decode kv=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0272 | 36 | 0.980 | 2092 | 261.5 | 57.7% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0033 | 36 | 0.119 | 2092 | 261.5 | 57.7% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **12.73 ms**

#### BMG — decode kv=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0613 | 36 | 2.208 | 1886 | 235.7 | 52.0% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0080 | 36 | 0.286 | 1886 | 235.7 | 52.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **14.13 ms**

#### BMG — decode kv=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1577 | 36 | 5.678 | 1516 | 189.5 | 41.8% | memory |
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0172 | 36 | 0.618 | 1516 | 189.5 | 41.8% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **17.93 ms**

#### BMG — decode kv=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.2749 | 36 | 9.897 | 1707 | 213.4 | 47.1% | memory |
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0371 | 36 | 1.334 | 1707 | 213.4 | 47.1% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **22.87 ms**

#### BMG — decode kv=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.4957 | 36 | 17.846 | 1866 | 233.2 | 51.5% | memory |
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0773 | 36 | 2.782 | 1866 | 233.2 | 51.5% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **32.26 ms**

#### BMG — decode kv=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.9355 | 36 | 33.678 | 1951 | 243.8 | 53.8% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1629 | 36 | 5.865 | 1951 | 243.8 | 53.8% | memory |
| fc_down | `gemm_kernel` | 0.0864 | 36 | 3.110 | 1165 | 303.0 | 66.9% | memory |
| fc_up | `gemm_kernel` | 0.0633 | 36 | 2.278 | 1591 | 413.8 | 91.3% | memory |
| fc_gate | `gemm_kernel` | 0.0632 | 36 | 2.277 | 1592 | 414.0 | 91.4% | memory |
| lm_head | `gemm_kernel` | 1.4069 | 1 | 1.407 | 885 | 449.5 | 99.2% | memory |
| fc_qkv | `gemm_kernel` | 0.0347 | 36 | 1.250 | 1450 | 377.1 | 83.3% | memory |
| fc_o | `gemm_kernel` | 0.0238 | 36 | 0.857 | 1410 | 366.9 | 81.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0016 | 72 | 0.117 | 20 | 10.0 | 2.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0010 | 72 | 0.071 | 4 | 24.8 | 5.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 36 | 0.053 | 28 | 11.2 | 2.5% | memory |
| rope_k | `rope_opt` | 0.0013 | 36 | 0.047 | 8 | 3.1 | 0.7% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0013 | 36 | 0.047 | 9 | 55.9 | 12.3% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 26 | 13.2 | 2.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 36 | 0.045 | 7 | 3.3 | 0.7% | memory |
| swish | `activation_opt_0_0` | 0.0008 | 36 | 0.030 | 58 | 58.4 | 12.9% | memory |

**Total inference time (this stage)** ≈ **51.18 ms**

### PREFILL — per-S-size (compute-bound uses INT8 XMX peak = 233.5 TOPS)

#### BMG — prefill S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.6982 | 36 | 25.136 | 126515 | 103.9 | 54.2% | compute |
| fc_up | `gemm_kernel` | 0.6599 | 36 | 23.756 | 147347 | 121.0 | 63.1% | compute |
| fc_gate | `gemm_kernel` | 0.6595 | 36 | 23.743 | 147438 | 121.1 | 63.2% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 0.4006 | 36 | 14.423 | 40561 | 79.2 | 17.4% | compute |
| fc_qkv | `gemm_kernel` | 0.3525 | 36 | 12.690 | 131245 | 118.5 | 56.2% | compute |
| fc_o | `gemm_kernel` | 0.2340 | 36 | 8.424 | 125642 | 123.7 | 53.8% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.1165 | 36 | 4.196 | 126515 | 103.9 | 54.2% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.0402 | 36 | 1.447 | 131245 | 118.5 | 56.2% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.0397 | 36 | 1.429 | 147347 | 121.0 | 63.1% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.0396 | 36 | 1.426 | 147438 | 121.1 | 63.2% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.0395 | 36 | 1.421 | 125642 | 123.7 | 53.8% | compute |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0229 | 36 | 0.825 | 40561 | 79.2 | 17.4% | compute |

**Total inference time (this stage)** ≈ **118.91 ms**

#### BMG — prefill S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 1.2896 | 36 | 46.427 | 135421 | 77.7 | 58.0% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 1.2685 | 36 | 45.666 | 51600 | 50.4 | 22.1% | compute |
| fc_gate | `gemm_kernel` | 1.2268 | 36 | 44.166 | 158123 | 90.7 | 67.7% | compute |
| fc_up | `gemm_kernel` | 1.2256 | 36 | 44.123 | 158269 | 90.8 | 67.8% | compute |
| fc_qkv | `gemm_kernel` | 0.6180 | 36 | 22.249 | 148264 | 97.1 | 63.5% | compute |
| fc_o | `gemm_kernel` | 0.4204 | 36 | 15.133 | 138196 | 101.7 | 59.2% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.2327 | 36 | 8.377 | 135421 | 77.7 | 58.0% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.0772 | 36 | 2.779 | 148264 | 97.1 | 63.5% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.0769 | 36 | 2.770 | 158123 | 90.7 | 67.7% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.0769 | 36 | 2.770 | 158269 | 90.8 | 67.8% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.0769 | 36 | 2.769 | 138196 | 101.7 | 59.2% | compute |

**Total inference time (this stage)** ≈ **237.23 ms**

#### BMG — prefill S=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 4.7889 | 36 | 172.402 | 55841 | 27.3 | 23.9% | compute |
| fc_down | `gemm_kernel` | 2.4756 | 36 | 89.120 | 140574 | 63.2 | 60.2% | compute |
| fc_gate | `gemm_kernel` | 2.4572 | 36 | 88.459 | 157969 | 71.0 | 67.7% | compute |
| fc_up | `gemm_kernel` | 2.4306 | 36 | 87.502 | 159643 | 71.8 | 68.4% | compute |
| fc_qkv | `gemm_kernel` | 1.2139 | 36 | 43.700 | 150954 | 80.1 | 64.7% | compute |
| fc_o | `gemm_kernel` | 0.8093 | 36 | 29.133 | 143070 | 87.6 | 61.3% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.4575 | 36 | 16.471 | 140574 | 63.2 | 60.2% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.1529 | 36 | 5.506 | 157969 | 71.0 | 67.7% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.1521 | 36 | 5.477 | 159643 | 71.8 | 68.4% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.1518 | 36 | 5.466 | 150954 | 80.1 | 64.7% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.1514 | 36 | 5.450 | 143070 | 87.6 | 61.3% | compute |

**Total inference time (this stage)** ≈ **548.68 ms**

#### BMG — prefill S=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 18.4221 | 36 | 663.197 | 58941 | 14.4 | 25.2% | compute |
| fc_down | `gemm_kernel` | 4.8381 | 36 | 174.173 | 143513 | 55.6 | 61.5% | compute |
| fc_up | `gemm_kernel` | 4.7928 | 36 | 172.541 | 161850 | 62.7 | 69.3% | compute |
| fc_gate | `gemm_kernel` | 4.7758 | 36 | 171.929 | 162381 | 62.9 | 69.6% | compute |
| fc_qkv | `gemm_kernel` | 2.3868 | 36 | 85.925 | 153318 | 71.9 | 65.7% | compute |
| fc_o | `gemm_kernel` | 1.5883 | 36 | 57.180 | 145411 | 80.0 | 62.3% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.9079 | 36 | 32.685 | 143513 | 55.6 | 61.5% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.3026 | 36 | 10.892 | 162381 | 62.9 | 69.6% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.3025 | 36 | 10.890 | 153318 | 71.9 | 65.7% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.3023 | 36 | 10.881 | 161850 | 62.7 | 69.3% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.3020 | 36 | 10.872 | 145411 | 80.0 | 62.3% | compute |

**Total inference time (this stage)** ≈ **1401.17 ms**

#### BMG — prefill S=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 72.7683 | 36 | 2619.660 | 60061 | 7.3 | 25.7% | compute |
| fc_gate | `gemm_kernel` | 9.5848 | 36 | 345.053 | 161841 | 57.7 | 69.3% | compute |
| fc_down | `gemm_kernel` | 9.5400 | 36 | 343.439 | 145245 | 51.8 | 62.2% | compute |
| fc_up | `gemm_kernel` | 9.5379 | 36 | 343.365 | 162603 | 58.0 | 69.6% | compute |
| fc_qkv | `gemm_kernel` | 4.7293 | 36 | 170.254 | 154583 | 67.7 | 66.2% | compute |
| fc_o | `gemm_kernel` | 3.1405 | 36 | 113.060 | 146801 | 76.2 | 62.9% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 1.8151 | 36 | 65.344 | 145245 | 51.8 | 62.2% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.6058 | 36 | 21.811 | 161841 | 57.7 | 69.3% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.6053 | 36 | 21.790 | 154583 | 67.7 | 66.2% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.6050 | 36 | 21.779 | 162603 | 58.0 | 69.6% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.6044 | 36 | 21.757 | 146801 | 76.2 | 62.9% | compute |

**Total inference time (this stage)** ≈ **4087.31 ms**

#### BMG — prefill S=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 302.6424 | 36 | 10895.125 | 57950 | 3.5 | 24.8% | compute |
| fc_down | `gemm_kernel` | 19.0891 | 36 | 687.208 | 145112 | 49.5 | 62.2% | compute |
| fc_up | `gemm_kernel` | 19.0827 | 36 | 686.976 | 162495 | 55.4 | 69.6% | compute |
| fc_gate | `gemm_kernel` | 19.0444 | 36 | 685.597 | 162847 | 55.5 | 69.7% | compute |
| fc_qkv | `gemm_kernel` | 9.4680 | 36 | 340.849 | 154470 | 65.2 | 66.2% | compute |
| fc_o | `gemm_kernel` | 6.2401 | 36 | 224.642 | 147659 | 74.4 | 63.2% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 3.6418 | 36 | 131.104 | 145112 | 49.5 | 62.2% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 1.2167 | 36 | 43.801 | 162495 | 55.4 | 69.6% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 1.2111 | 36 | 43.600 | 162847 | 55.5 | 69.7% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 1.2089 | 36 | 43.520 | 154470 | 65.2 | 66.2% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 1.2062 | 36 | 43.424 | 147659 | 74.4 | 63.2% | compute |

**Total inference time (this stage)** ≈ **13825.85 ms**

## Platform: PTL

### DECODE — per-kv-size (M=1)

#### PTL — decode kv=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.2557 | 36 | 9.204 | 394 | 102.4 | 97.5% | memory |
| fc_gate | `gemm_kernel` | 0.2547 | 36 | 9.170 | 395 | 102.8 | 97.9% | memory |
| lm_head | `gemm_kernel` | 6.0312 | 1 | 6.031 | 206 | 104.8 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.649 | 390 | 101.4 | 96.6% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0520 | 36 | 1.871 | 284 | 35.5 | 33.8% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.226 | 10 | 5.2 | 5.0% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0041 | 36 | 0.147 | 284 | 35.5 | 33.8% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.140 | 2 | 12.7 | 12.1% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0031 | 36 | 0.112 | 284 | 35.5 | 33.8% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.103 | 4 | 25.8 | 24.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.088 | 13 | 6.7 | 6.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0022 | 36 | 0.078 | 19 | 7.6 | 7.2% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.7 | 26.4% | memory |

**Total inference time (this stage)** ≈ **32.04 ms**

#### PTL — decode kv=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.2557 | 36 | 9.204 | 394 | 102.4 | 97.5% | memory |
| fc_gate | `gemm_kernel` | 0.2547 | 36 | 9.170 | 395 | 102.8 | 97.9% | memory |
| lm_head | `gemm_kernel` | 6.0312 | 1 | 6.031 | 206 | 104.8 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.649 | 390 | 101.4 | 96.6% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0841 | 36 | 3.027 | 366 | 45.7 | 43.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.226 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.140 | 2 | 12.7 | 12.1% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0038 | 36 | 0.138 | 366 | 45.7 | 43.6% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.103 | 4 | 25.8 | 24.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.088 | 13 | 6.7 | 6.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0022 | 36 | 0.078 | 19 | 7.6 | 7.2% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.7 | 26.4% | memory |

**Total inference time (this stage)** ≈ **33.07 ms**

#### PTL — decode kv=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.2557 | 36 | 9.204 | 394 | 102.4 | 97.5% | memory |
| fc_gate | `gemm_kernel` | 0.2547 | 36 | 9.170 | 395 | 102.8 | 97.9% | memory |
| lm_head | `gemm_kernel` | 6.0312 | 1 | 6.031 | 206 | 104.8 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.649 | 390 | 101.4 | 96.6% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1010 | 36 | 3.636 | 609 | 76.1 | 72.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.226 | 10 | 5.2 | 5.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0054 | 36 | 0.194 | 609 | 76.1 | 72.5% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.140 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.103 | 4 | 25.8 | 24.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.088 | 13 | 6.7 | 6.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0022 | 36 | 0.078 | 19 | 7.6 | 7.2% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.7 | 26.4% | memory |

**Total inference time (this stage)** ≈ **33.74 ms**

#### PTL — decode kv=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 0.2557 | 36 | 9.204 | 394 | 102.4 | 97.5% | memory |
| fc_gate | `gemm_kernel` | 0.2547 | 36 | 9.170 | 395 | 102.8 | 97.9% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1936 | 36 | 6.970 | 643 | 80.4 | 76.6% | memory |
| lm_head | `gemm_kernel` | 6.0312 | 1 | 6.031 | 206 | 104.8 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.649 | 390 | 101.4 | 96.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0110 | 36 | 0.395 | 643 | 80.4 | 76.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.226 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.140 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.103 | 4 | 25.8 | 24.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.088 | 13 | 6.7 | 6.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0022 | 36 | 0.078 | 19 | 7.6 | 7.2% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.7 | 26.4% | memory |

**Total inference time (this stage)** ≈ **37.27 ms**

#### PTL — decode kv=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.3781 | 36 | 13.613 | 657 | 82.1 | 78.2% | memory |
| fc_down | `gemm_kernel` | 0.2557 | 36 | 9.204 | 394 | 102.4 | 97.5% | memory |
| fc_gate | `gemm_kernel` | 0.2547 | 36 | 9.170 | 395 | 102.8 | 97.9% | memory |
| lm_head | `gemm_kernel` | 6.0312 | 1 | 6.031 | 206 | 104.8 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.649 | 390 | 101.4 | 96.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0235 | 36 | 0.845 | 657 | 82.1 | 78.2% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.226 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.140 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.103 | 4 | 25.8 | 24.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.088 | 13 | 6.7 | 6.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0022 | 36 | 0.078 | 19 | 7.6 | 7.2% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.7 | 26.4% | memory |

**Total inference time (this stage)** ≈ **44.37 ms**

#### PTL — decode kv=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.7640 | 36 | 27.504 | 652 | 81.5 | 77.6% | memory |
| fc_down | `gemm_kernel` | 0.2557 | 36 | 9.204 | 394 | 102.4 | 97.5% | memory |
| fc_gate | `gemm_kernel` | 0.2547 | 36 | 9.170 | 395 | 102.8 | 97.9% | memory |
| lm_head | `gemm_kernel` | 6.0312 | 1 | 6.031 | 206 | 104.8 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.649 | 390 | 101.4 | 96.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0514 | 36 | 1.849 | 652 | 81.5 | 77.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.226 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.140 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.103 | 4 | 25.8 | 24.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.088 | 13 | 6.7 | 6.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0022 | 36 | 0.078 | 19 | 7.6 | 7.2% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.7 | 26.4% | memory |

**Total inference time (this stage)** ≈ **59.26 ms**

#### PTL — decode kv=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 1.5367 | 36 | 55.321 | 651 | 81.4 | 77.5% | memory |
| fc_down | `gemm_kernel` | 0.2557 | 36 | 9.204 | 394 | 102.4 | 97.5% | memory |
| fc_gate | `gemm_kernel` | 0.2547 | 36 | 9.170 | 395 | 102.8 | 97.9% | memory |
| lm_head | `gemm_kernel` | 6.0312 | 1 | 6.031 | 206 | 104.8 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.649 | 390 | 101.4 | 96.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1053 | 36 | 3.791 | 651 | 81.4 | 77.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.226 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.140 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.103 | 4 | 25.8 | 24.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.088 | 13 | 6.7 | 6.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0022 | 36 | 0.078 | 19 | 7.6 | 7.2% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.7 | 26.4% | memory |

**Total inference time (this stage)** ≈ **89.02 ms**

#### PTL — decode kv=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 4.4387 | 36 | 159.794 | 459 | 57.4 | 54.7% | memory |
| fc_down | `gemm_kernel` | 0.2557 | 36 | 9.204 | 394 | 102.4 | 97.5% | memory |
| fc_gate | `gemm_kernel` | 0.2547 | 36 | 9.170 | 395 | 102.8 | 97.9% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.2238 | 36 | 8.058 | 459 | 57.4 | 54.7% | memory |
| lm_head | `gemm_kernel` | 6.0312 | 1 | 6.031 | 206 | 104.8 | 99.9% | memory |
| fc_qkv | `gemm_kernel` | 0.1292 | 36 | 4.649 | 390 | 101.4 | 96.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0031 | 72 | 0.226 | 10 | 5.2 | 5.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0019 | 72 | 0.140 | 2 | 12.7 | 12.1% | memory |
| multiply | `eltwise_simple_vload8_0_0` | 0.0029 | 36 | 0.103 | 4 | 25.8 | 24.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.088 | 13 | 6.7 | 6.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 36 | 0.086 | 3 | 1.7 | 1.6% | memory |
| rope_q | `rope_opt` | 0.0022 | 36 | 0.078 | 19 | 7.6 | 7.2% | memory |
| rope_k | `rope_opt` | 0.0020 | 36 | 0.071 | 5 | 2.1 | 2.0% | memory |
| swish | `activation_opt_0_0` | 0.0018 | 36 | 0.064 | 28 | 27.7 | 26.4% | memory |

**Total inference time (this stage)** ≈ **197.76 ms**

### PREFILL — per-S-size (compute-bound uses INT8 XMX peak = 118.0 TOPS)

#### PTL — prefill S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 1.6040 | 36 | 57.743 | 52412 | 43.1 | 44.4% | compute |
| fc_gate | `gemm_kernel` | 1.4925 | 36 | 53.729 | 64661 | 53.1 | 54.8% | compute |
| fc_up | `gemm_kernel` | 1.4886 | 36 | 53.591 | 64911 | 53.3 | 55.0% | compute |
| fc_qkv | `gemm_kernel` | 0.7497 | 36 | 26.990 | 60396 | 54.5 | 51.2% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 0.6436 | 36 | 23.169 | 23622 | 46.1 | 20.0% | compute |
| fc_o | `gemm_kernel` | 0.5047 | 36 | 18.169 | 56966 | 56.1 | 48.3% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.3628 | 36 | 13.059 | 52412 | 43.1 | 44.4% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.1036 | 36 | 3.731 | 60396 | 54.5 | 51.2% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.1017 | 36 | 3.661 | 64661 | 53.1 | 54.8% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.0994 | 36 | 3.578 | 64911 | 53.3 | 55.0% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.0985 | 36 | 3.545 | 56966 | 56.1 | 48.3% | compute |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0837 | 36 | 3.013 | 23622 | 46.1 | 20.0% | compute |

**Total inference time (this stage)** ≈ **263.98 ms**

#### PTL — prefill S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fc_down | `gemm_kernel` | 3.0258 | 36 | 108.927 | 56390 | 32.3 | 47.8% | compute |
| fc_up | `gemm_kernel` | 2.7906 | 36 | 100.462 | 68875 | 39.5 | 58.4% | compute |
| fc_gate | `gemm_kernel` | 2.7887 | 36 | 100.395 | 68957 | 39.5 | 58.5% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 2.3300 | 36 | 83.879 | 28050 | 27.4 | 23.8% | compute |
| fc_qkv | `gemm_kernel` | 1.4189 | 36 | 51.081 | 63293 | 41.4 | 53.7% | compute |
| fc_o | `gemm_kernel` | 1.0012 | 36 | 36.043 | 55946 | 41.2 | 47.4% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 0.6302 | 36 | 22.687 | 56390 | 32.3 | 47.8% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.2271 | 36 | 8.176 | 55946 | 41.2 | 47.4% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.2097 | 36 | 7.548 | 63293 | 41.4 | 53.7% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.2026 | 36 | 7.295 | 68875 | 39.5 | 58.4% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.2009 | 36 | 7.233 | 68957 | 39.5 | 58.5% | compute |

**Total inference time (this stage)** ≈ **533.73 ms**

#### PTL — prefill S=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 8.8175 | 36 | 317.431 | 30370 | 14.8 | 25.7% | compute |
| fc_down | `gemm_kernel` | 5.8378 | 36 | 210.160 | 57470 | 25.8 | 48.7% | compute |
| fc_gate | `gemm_kernel` | 5.5901 | 36 | 201.242 | 68796 | 30.9 | 58.3% | compute |
| fc_up | `gemm_kernel` | 5.4926 | 36 | 197.733 | 70113 | 31.5 | 59.4% | compute |
| fc_qkv | `gemm_kernel` | 2.7575 | 36 | 99.268 | 65536 | 34.8 | 55.6% | compute |
| fc_o | `gemm_kernel` | 1.8748 | 36 | 67.493 | 60291 | 36.9 | 51.1% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 1.3367 | 36 | 48.122 | 57470 | 25.8 | 48.7% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.4048 | 36 | 14.573 | 60291 | 36.9 | 51.1% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.4033 | 36 | 14.518 | 68796 | 30.9 | 58.3% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.3883 | 36 | 13.978 | 65536 | 34.8 | 55.6% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.3882 | 36 | 13.974 | 70113 | 31.5 | 59.4% | compute |

**Total inference time (this stage)** ≈ **1198.49 ms**

#### PTL — prefill S=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 34.8601 | 36 | 1254.965 | 31092 | 7.6 | 26.4% | compute |
| fc_down | `gemm_kernel` | 11.3879 | 36 | 409.965 | 57735 | 22.4 | 48.9% | compute |
| fc_gate | `gemm_kernel` | 11.1352 | 36 | 400.866 | 68792 | 26.7 | 58.3% | compute |
| fc_up | `gemm_kernel` | 10.7386 | 36 | 386.591 | 71355 | 27.7 | 60.5% | compute |
| fc_qkv | `gemm_kernel` | 5.7864 | 36 | 208.312 | 62180 | 29.2 | 52.7% | compute |
| fc_o | `gemm_kernel` | 3.8241 | 36 | 137.669 | 58238 | 32.0 | 49.4% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 2.8951 | 36 | 104.224 | 57735 | 22.4 | 48.9% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.8957 | 36 | 32.247 | 58238 | 32.0 | 49.4% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 0.8521 | 36 | 30.677 | 68792 | 26.7 | 58.3% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.8446 | 36 | 30.404 | 62180 | 29.2 | 52.7% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 0.8182 | 36 | 29.455 | 71355 | 27.7 | 60.5% | compute |

**Total inference time (this stage)** ≈ **3025.37 ms**

#### PTL — prefill S=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 138.8725 | 36 | 4999.412 | 31438 | 3.8 | 26.7% | compute |
| fc_down | `gemm_kernel` | 22.5206 | 36 | 810.743 | 59115 | 21.1 | 50.1% | compute |
| fc_gate | `gemm_kernel` | 22.0057 | 36 | 792.204 | 68877 | 24.6 | 58.4% | compute |
| fc_up | `gemm_kernel` | 21.6280 | 36 | 778.609 | 70087 | 25.0 | 59.4% | compute |
| fc_qkv | `gemm_kernel` | 11.1694 | 36 | 402.099 | 62922 | 27.6 | 53.3% | compute |
| fc_o | `gemm_kernel` | 7.4975 | 36 | 269.910 | 57579 | 29.9 | 48.8% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 5.3787 | 36 | 193.635 | 59115 | 21.1 | 50.1% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 2.0504 | 36 | 73.814 | 57579 | 29.9 | 48.8% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 1.9396 | 36 | 69.826 | 68877 | 24.6 | 58.4% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 1.9363 | 36 | 69.707 | 62922 | 27.6 | 53.3% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 1.9038 | 36 | 68.537 | 70087 | 25.0 | 59.4% | compute |

**Total inference time (this stage)** ≈ **8528.50 ms**

#### PTL — prefill S=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paged_attention | `sdpa_micro_prefill_sa` | 693.1064 | 36 | 24951.832 | 25305 | 1.5 | 21.5% | compute |
| fc_down | `gemm_kernel` | 50.0179 | 36 | 1800.645 | 54147 | 18.5 | 45.9% | compute |
| fc_gate | `gemm_kernel` | 43.8858 | 36 | 1579.888 | 69157 | 23.6 | 58.6% | compute |
| fc_up | `gemm_kernel` | 43.8781 | 36 | 1579.613 | 69231 | 23.6 | 58.7% | compute |
| fc_qkv | `gemm_kernel` | 21.6138 | 36 | 778.097 | 64983 | 27.4 | 55.1% | compute |
| fc_o | `gemm_kernel` | 14.1071 | 36 | 507.854 | 61641 | 31.1 | 52.3% | compute |
| fc_down | `dynamic_quantize_gpu_opt_0_0` | 10.8997 | 36 | 392.390 | 54147 | 18.5 | 45.9% | compute |
| fc_gate | `dynamic_quantize_gpu_opt_0_0` | 3.8107 | 36 | 137.187 | 69157 | 23.6 | 58.6% | compute |
| fc_up | `dynamic_quantize_gpu_opt_0_0` | 3.7674 | 36 | 135.626 | 69231 | 23.6 | 58.7% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 3.7663 | 36 | 135.585 | 64983 | 27.4 | 55.1% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 3.7304 | 36 | 134.294 | 61641 | 31.1 | 52.3% | compute |

**Total inference time (this stage)** ≈ **32133.01 ms**

