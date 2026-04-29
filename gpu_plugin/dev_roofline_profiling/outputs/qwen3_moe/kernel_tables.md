# Per-token-size Kernel Tables


## Platform: BMG

### DECODE — per-kv-size (M=1)


#### BMG — decode kv=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0441 | 48 | 2.115 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0224 | 48 | 1.076 | 975 | 253.6 | 55.6% | memory |
| fc_qkv | `gemm_kernel` | 0.0155 | 48 | 0.746 | 1349 | 351.3 | 77.0% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0155 | 48 | 0.745 | 917 | 57.3 | 12.6% | memory |
| lm_head | `gemm_kernel` | 0.7067 | 1 | 0.707 | 881 | 447.6 | 98.2% | memory |
| fc_o | `gemm_kernel` | 0.0130 | 48 | 0.626 | 1287 | 335.2 | 73.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0033 | 48 | 0.158 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 48 | 0.148 | 975 | 253.6 | 55.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0016 | 48 | 0.076 | 975 | 253.6 | 55.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0016 | 48 | 0.075 | 917 | 57.3 | 12.6% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 48 | 0.070 | 28 | 11.2 | 2.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.060 | 26 | 13.2 | 2.9% | memory |
| rope_k | `rope_opt` | 0.0012 | 48 | 0.059 | 4 | 1.7 | 0.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.059 | 3 | 1.7 | 0.4% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0012 | 48 | 0.058 | 917 | 57.3 | 12.6% | memory |

**Total inference time (this stage)** ≈ **6.99 ms**

#### BMG — decode kv=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0441 | 48 | 2.115 | 975 | 253.6 | 55.6% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0295 | 48 | 1.417 | 1017 | 63.6 | 13.9% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0224 | 48 | 1.076 | 975 | 253.6 | 55.6% | memory |
| fc_qkv | `gemm_kernel` | 0.0155 | 48 | 0.746 | 1349 | 351.3 | 77.0% | memory |
| lm_head | `gemm_kernel` | 0.7067 | 1 | 0.707 | 881 | 447.6 | 98.2% | memory |
| fc_o | `gemm_kernel` | 0.0130 | 48 | 0.626 | 1287 | 335.2 | 73.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0033 | 48 | 0.158 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 48 | 0.148 | 975 | 253.6 | 55.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0021 | 48 | 0.099 | 1017 | 63.6 | 13.9% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0016 | 48 | 0.076 | 975 | 253.6 | 55.6% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 48 | 0.070 | 28 | 11.2 | 2.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.060 | 26 | 13.2 | 2.9% | memory |
| rope_k | `rope_opt` | 0.0012 | 48 | 0.059 | 4 | 1.7 | 0.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.059 | 3 | 1.7 | 0.4% | memory |

**Total inference time (this stage)** ≈ **7.63 ms**

#### BMG — decode kv=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0441 | 48 | 2.115 | 975 | 253.6 | 55.6% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0271 | 48 | 1.303 | 2098 | 131.2 | 28.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0224 | 48 | 1.076 | 975 | 253.6 | 55.6% | memory |
| fc_qkv | `gemm_kernel` | 0.0155 | 48 | 0.746 | 1349 | 351.3 | 77.0% | memory |
| lm_head | `gemm_kernel` | 0.7067 | 1 | 0.707 | 881 | 447.6 | 98.2% | memory |
| fc_o | `gemm_kernel` | 0.0130 | 48 | 0.626 | 1287 | 335.2 | 73.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0033 | 48 | 0.159 | 2098 | 131.2 | 28.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0033 | 48 | 0.158 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 48 | 0.148 | 975 | 253.6 | 55.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0016 | 48 | 0.076 | 975 | 253.6 | 55.6% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 48 | 0.070 | 28 | 11.2 | 2.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.060 | 26 | 13.2 | 2.9% | memory |
| rope_k | `rope_opt` | 0.0012 | 48 | 0.059 | 4 | 1.7 | 0.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.059 | 3 | 1.7 | 0.4% | memory |

**Total inference time (this stage)** ≈ **7.57 ms**

#### BMG — decode kv=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0609 | 48 | 2.924 | 1899 | 118.7 | 26.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0441 | 48 | 2.115 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0224 | 48 | 1.076 | 975 | 253.6 | 55.6% | memory |
| fc_qkv | `gemm_kernel` | 0.0155 | 48 | 0.746 | 1349 | 351.3 | 77.0% | memory |
| lm_head | `gemm_kernel` | 0.7067 | 1 | 0.707 | 881 | 447.6 | 98.2% | memory |
| fc_o | `gemm_kernel` | 0.0130 | 48 | 0.626 | 1287 | 335.2 | 73.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0079 | 48 | 0.379 | 1899 | 118.7 | 26.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0033 | 48 | 0.158 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 48 | 0.148 | 975 | 253.6 | 55.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0016 | 48 | 0.076 | 975 | 253.6 | 55.6% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 48 | 0.070 | 28 | 11.2 | 2.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.060 | 26 | 13.2 | 2.9% | memory |
| rope_k | `rope_opt` | 0.0012 | 48 | 0.059 | 4 | 1.7 | 0.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.059 | 3 | 1.7 | 0.4% | memory |

**Total inference time (this stage)** ≈ **9.42 ms**

#### BMG — decode kv=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1577 | 48 | 7.571 | 1516 | 94.7 | 20.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0441 | 48 | 2.115 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0224 | 48 | 1.076 | 975 | 253.6 | 55.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0172 | 48 | 0.824 | 1516 | 94.7 | 20.8% | memory |
| fc_qkv | `gemm_kernel` | 0.0155 | 48 | 0.746 | 1349 | 351.3 | 77.0% | memory |
| lm_head | `gemm_kernel` | 0.7067 | 1 | 0.707 | 881 | 447.6 | 98.2% | memory |
| fc_o | `gemm_kernel` | 0.0130 | 48 | 0.626 | 1287 | 335.2 | 73.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0033 | 48 | 0.158 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 48 | 0.148 | 975 | 253.6 | 55.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0016 | 48 | 0.076 | 975 | 253.6 | 55.6% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 48 | 0.070 | 28 | 11.2 | 2.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.060 | 26 | 13.2 | 2.9% | memory |
| rope_k | `rope_opt` | 0.0012 | 48 | 0.059 | 4 | 1.7 | 0.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.059 | 3 | 1.7 | 0.4% | memory |

**Total inference time (this stage)** ≈ **14.51 ms**

#### BMG — decode kv=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.2750 | 48 | 13.199 | 1707 | 106.7 | 23.4% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0441 | 48 | 2.115 | 975 | 253.6 | 55.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0371 | 48 | 1.780 | 1707 | 106.7 | 23.4% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0224 | 48 | 1.076 | 975 | 253.6 | 55.6% | memory |
| fc_qkv | `gemm_kernel` | 0.0155 | 48 | 0.746 | 1349 | 351.3 | 77.0% | memory |
| lm_head | `gemm_kernel` | 0.7067 | 1 | 0.707 | 881 | 447.6 | 98.2% | memory |
| fc_o | `gemm_kernel` | 0.0130 | 48 | 0.626 | 1287 | 335.2 | 73.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0033 | 48 | 0.158 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 48 | 0.148 | 975 | 253.6 | 55.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0016 | 48 | 0.076 | 975 | 253.6 | 55.6% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 48 | 0.070 | 28 | 11.2 | 2.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.060 | 26 | 13.2 | 2.9% | memory |
| rope_k | `rope_opt` | 0.0012 | 48 | 0.059 | 4 | 1.7 | 0.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.059 | 3 | 1.7 | 0.4% | memory |

**Total inference time (this stage)** ≈ **21.09 ms**

#### BMG — decode kv=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.4958 | 48 | 23.797 | 1865 | 116.6 | 25.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0773 | 48 | 3.710 | 1865 | 116.6 | 25.6% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0441 | 48 | 2.115 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0224 | 48 | 1.076 | 975 | 253.6 | 55.6% | memory |
| fc_qkv | `gemm_kernel` | 0.0155 | 48 | 0.746 | 1349 | 351.3 | 77.0% | memory |
| lm_head | `gemm_kernel` | 0.7067 | 1 | 0.707 | 881 | 447.6 | 98.2% | memory |
| fc_o | `gemm_kernel` | 0.0130 | 48 | 0.626 | 1287 | 335.2 | 73.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0033 | 48 | 0.158 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 48 | 0.148 | 975 | 253.6 | 55.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0016 | 48 | 0.076 | 975 | 253.6 | 55.6% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 48 | 0.070 | 28 | 11.2 | 2.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.060 | 26 | 13.2 | 2.9% | memory |
| rope_k | `rope_opt` | 0.0012 | 48 | 0.059 | 4 | 1.7 | 0.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.059 | 3 | 1.7 | 0.4% | memory |

**Total inference time (this stage)** ≈ **33.62 ms**

#### BMG — decode kv=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.9360 | 48 | 44.930 | 1949 | 121.8 | 26.7% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1630 | 48 | 7.824 | 1949 | 121.8 | 26.7% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0441 | 48 | 2.115 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0224 | 48 | 1.076 | 975 | 253.6 | 55.6% | memory |
| fc_qkv | `gemm_kernel` | 0.0155 | 48 | 0.746 | 1349 | 351.3 | 77.0% | memory |
| lm_head | `gemm_kernel` | 0.7067 | 1 | 0.707 | 881 | 447.6 | 98.2% | memory |
| fc_o | `gemm_kernel` | 0.0130 | 48 | 0.626 | 1287 | 335.2 | 73.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0033 | 48 | 0.158 | 975 | 253.6 | 55.6% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 48 | 0.148 | 975 | 253.6 | 55.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0016 | 48 | 0.076 | 975 | 253.6 | 55.6% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 48 | 0.070 | 28 | 11.2 | 2.5% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.060 | 26 | 13.2 | 2.9% | memory |
| rope_k | `rope_opt` | 0.0012 | 48 | 0.059 | 4 | 1.7 | 0.4% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0012 | 48 | 0.059 | 3 | 1.7 | 0.4% | memory |

**Total inference time (this stage)** ≈ **58.87 ms**

### PREFILL — per-S-size (compute-bound uses INT8 XMX peak = 233.5 TOPS)


#### BMG — prefill S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 2.6346 | 48 | 126.461 | 21336 | 121.5 | 9.1% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 0.5735 | 48 | 27.528 | 21336 | 121.5 | 9.1% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 0.3448 | 48 | 16.553 | 47044 | 91.9 | 20.1% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.1661 | 48 | 7.972 | 21336 | 121.5 | 9.1% | compute |
| fc_qkv | `gemm_kernel` | 0.1459 | 48 | 7.006 | 130114 | 153.5 | 55.7% | compute |
| fc_o | `gemm_kernel` | 0.1268 | 48 | 6.086 | 103370 | 127.0 | 44.3% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.0932 | 48 | 4.475 | 21336 | 121.5 | 9.1% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.0394 | 48 | 1.892 | 103370 | 127.0 | 44.3% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0185 | 96 | 1.780 | 905 | 452.5 | 99.2% | memory |
| rope_q | `rope_opt` | 0.0307 | 48 | 1.474 | 1366 | 546.2 | 119.8% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0203 | 48 | 0.976 | 47044 | 91.9 | 20.1% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.0191 | 48 | 0.917 | 130114 | 153.5 | 55.7% | compute |

**Total inference time (this stage)** ≈ **203.12 ms**

#### BMG — prefill S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 2.7481 | 48 | 131.907 | 28723 | 107.5 | 12.3% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 1.7414 | 48 | 83.586 | 28723 | 107.5 | 12.3% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 1.2854 | 48 | 61.699 | 50948 | 49.8 | 21.8% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.3831 | 48 | 18.390 | 28723 | 107.5 | 12.3% | compute |
| fc_qkv | `gemm_kernel` | 0.2723 | 48 | 13.073 | 138149 | 128.7 | 59.2% | compute |
| fc_o | `gemm_kernel` | 0.2174 | 48 | 10.436 | 116886 | 114.6 | 50.1% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.1981 | 48 | 9.507 | 28723 | 107.5 | 12.3% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.1400 | 48 | 6.719 | 28723 | 107.5 | 12.3% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.0765 | 48 | 3.674 | 116886 | 114.6 | 50.1% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0358 | 96 | 3.433 | 938 | 469.1 | 102.9% | memory |
| rope_q | `rope_opt` | 0.0520 | 48 | 2.497 | 1613 | 645.1 | 141.5% | memory |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.0385 | 48 | 1.850 | 138149 | 128.7 | 59.2% | compute |

**Total inference time (this stage)** ≈ **346.77 ms**

#### BMG — prefill S=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 4.7734 | 48 | 229.122 | 56009 | 27.3 | 24.0% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 4.6905 | 48 | 225.145 | 33546 | 92.8 | 14.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 2.5872 | 48 | 124.185 | 33546 | 92.8 | 14.4% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.8783 | 48 | 42.160 | 33546 | 92.8 | 14.4% | compute |
| fc_qkv | `gemm_kernel` | 0.5297 | 48 | 25.426 | 141693 | 114.4 | 60.7% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.4085 | 48 | 19.608 | 33546 | 92.8 | 14.4% | compute |
| fc_o | `gemm_kernel` | 0.4017 | 48 | 19.280 | 124180 | 106.3 | 53.2% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.2747 | 48 | 13.187 | 33546 | 92.8 | 14.4% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.1517 | 48 | 7.283 | 124180 | 106.3 | 53.2% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0697 | 96 | 6.693 | 963 | 481.3 | 105.5% | memory |
| rope_q | `rope_opt` | 0.0944 | 48 | 4.530 | 1778 | 711.1 | 155.9% | memory |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.0765 | 48 | 3.674 | 141693 | 114.4 | 60.7% | compute |

**Total inference time (this stage)** ≈ **720.29 ms**

#### BMG — prefill S=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 18.4627 | 48 | 886.211 | 58814 | 14.4 | 25.2% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 7.9867 | 48 | 383.360 | 30940 | 70.5 | 13.3% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 7.9667 | 48 | 382.402 | 30940 | 70.5 | 13.3% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 1.8867 | 48 | 90.561 | 30940 | 70.5 | 13.3% | compute |
| fc_qkv | `gemm_kernel` | 1.0472 | 48 | 50.267 | 142962 | 106.6 | 61.2% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.8021 | 48 | 38.503 | 30940 | 70.5 | 13.3% | compute |
| fc_o | `gemm_kernel` | 0.7876 | 48 | 37.806 | 126142 | 100.2 | 54.0% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.5445 | 48 | 26.136 | 30940 | 70.5 | 13.3% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.3019 | 48 | 14.493 | 126142 | 100.2 | 54.0% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.1375 | 96 | 13.201 | 976 | 488.0 | 107.0% | memory |
| rope_q | `rope_opt` | 0.1803 | 48 | 8.652 | 1862 | 744.6 | 163.3% | memory |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.1545 | 48 | 7.415 | 142962 | 106.6 | 61.2% | compute |

**Total inference time (this stage)** ≈ **1939.01 ms**

#### BMG — prefill S=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 73.0737 | 48 | 3507.538 | 59810 | 7.3 | 25.6% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 36.6023 | 48 | 1756.909 | 20695 | 42.1 | 8.9% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 14.8865 | 48 | 714.550 | 20695 | 42.1 | 8.9% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 3.9102 | 48 | 187.691 | 20695 | 42.1 | 8.9% | compute |
| fc_qkv | `gemm_kernel` | 2.0756 | 48 | 99.629 | 144132 | 103.0 | 61.7% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 1.6258 | 48 | 78.037 | 20695 | 42.1 | 8.9% | compute |
| fc_o | `gemm_kernel` | 1.5573 | 48 | 74.748 | 127479 | 97.3 | 54.6% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.5990 | 48 | 28.752 | 127479 | 97.3 | 54.6% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.3083 | 48 | 14.799 | 144132 | 103.0 | 61.7% | compute |

**Total inference time (this stage)** ≈ **6462.65 ms**

#### BMG — prefill S=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 302.7775 | 48 | 14533.319 | 57923 | 3.5 | 24.8% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 163.3015 | 48 | 7838.470 | 11921 | 22.8 | 5.1% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 27.7102 | 48 | 1330.089 | 11921 | 22.8 | 5.1% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 7.7729 | 48 | 373.101 | 11921 | 22.8 | 5.1% | compute |
| fc_qkv | `gemm_kernel` | 4.1404 | 48 | 198.739 | 144352 | 100.9 | 61.8% | compute |
| fc_o | `gemm_kernel` | 3.0752 | 48 | 147.610 | 128487 | 96.1 | 55.0% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 1.2035 | 48 | 57.767 | 128487 | 96.1 | 55.0% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.6202 | 48 | 29.768 | 144352 | 100.9 | 61.8% | compute |

**Total inference time (this stage)** ≈ **24508.86 ms**

#### BMG — prefill S=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 1244.9835 | 48 | 59759.206 | 56435 | 1.7 | 24.2% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 702.6156 | 48 | 33725.547 | 6271 | 11.6 | 2.7% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 53.8376 | 48 | 2584.207 | 6271 | 11.6 | 2.7% | compute |
| fc_qkv | `gemm_kernel` | 8.2753 | 48 | 397.214 | 144441 | 99.9 | 61.9% | compute |
| fc_o | `gemm_kernel` | 6.1190 | 48 | 293.713 | 128964 | 95.5 | 55.2% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 2.4067 | 48 | 115.523 | 128964 | 95.5 | 55.2% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 1.2400 | 48 | 59.518 | 144441 | 99.9 | 61.9% | compute |

**Total inference time (this stage)** ≈ **96934.93 ms**

#### BMG — prefill S=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 4973.8017 | 48 | 238742.482 | 56548 | 0.9 | 24.2% | compute |
| fc_qkv | `gemm_kernel` | 16.5909 | 48 | 796.364 | 144067 | 99.0 | 61.7% | compute |
| fc_o | `gemm_kernel` | 12.2069 | 48 | 585.929 | 129181 | 95.1 | 55.3% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 4.8160 | 48 | 231.168 | 129181 | 95.1 | 55.3% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 2.4890 | 48 | 119.471 | 144067 | 99.0 | 61.7% | compute |
| moe_3gemm_fused | `reorder_data_0_0` | 0.0039 | 48 | 0.185 | 2567619266 | 4675332.1 | 1099754.7% | compute |

**Total inference time (this stage)** ≈ **240475.60 ms**

## Platform: PTL

### DECODE — per-kv-size (M=1)


#### PTL — decode kv=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1348 | 48 | 6.473 | 343 | 89.3 | 81.2% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0697 | 48 | 3.347 | 343 | 89.3 | 81.2% | memory |
| lm_head | `gemm_kernel` | 3.1701 | 1 | 3.170 | 196 | 99.8 | 90.7% | memory |
| fc_qkv | `gemm_kernel` | 0.0562 | 48 | 2.695 | 373 | 97.3 | 88.4% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0506 | 48 | 2.426 | 292 | 18.2 | 16.6% | memory |
| fc_o | `gemm_kernel` | 0.0464 | 48 | 2.229 | 361 | 94.1 | 85.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0047 | 48 | 0.227 | 343 | 89.3 | 81.2% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0039 | 48 | 0.186 | 292 | 18.2 | 16.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0031 | 48 | 0.149 | 292 | 18.2 | 16.6% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.115 | 14 | 6.8 | 6.2% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.114 | 2 | 0.9 | 0.8% | memory |
| rope_q | `rope_opt` | 0.0022 | 48 | 0.103 | 19 | 7.6 | 6.9% | memory |
| rope_k | `rope_opt` | 0.0019 | 48 | 0.090 | 3 | 1.1 | 1.0% | memory |

**Total inference time (this stage)** ≈ **21.75 ms**

#### PTL — decode kv=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1348 | 48 | 6.473 | 343 | 89.3 | 81.2% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0832 | 48 | 3.993 | 369 | 23.1 | 21.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0697 | 48 | 3.347 | 343 | 89.3 | 81.2% | memory |
| lm_head | `gemm_kernel` | 3.1701 | 1 | 3.170 | 196 | 99.8 | 90.7% | memory |
| fc_qkv | `gemm_kernel` | 0.0562 | 48 | 2.695 | 373 | 97.3 | 88.4% | memory |
| fc_o | `gemm_kernel` | 0.0464 | 48 | 2.229 | 361 | 94.1 | 85.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0047 | 48 | 0.227 | 343 | 89.3 | 81.2% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0038 | 48 | 0.185 | 369 | 23.1 | 21.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.115 | 14 | 6.8 | 6.2% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.114 | 2 | 0.9 | 0.8% | memory |
| rope_q | `rope_opt` | 0.0022 | 48 | 0.103 | 19 | 7.6 | 6.9% | memory |
| rope_k | `rope_opt` | 0.0019 | 48 | 0.090 | 3 | 1.1 | 1.0% | memory |

**Total inference time (this stage)** ≈ **23.17 ms**

#### PTL — decode kv=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1348 | 48 | 6.473 | 343 | 89.3 | 81.2% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1001 | 48 | 4.806 | 613 | 38.3 | 34.9% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0697 | 48 | 3.347 | 343 | 89.3 | 81.2% | memory |
| lm_head | `gemm_kernel` | 3.1701 | 1 | 3.170 | 196 | 99.8 | 90.7% | memory |
| fc_qkv | `gemm_kernel` | 0.0562 | 48 | 2.695 | 373 | 97.3 | 88.4% | memory |
| fc_o | `gemm_kernel` | 0.0464 | 48 | 2.229 | 361 | 94.1 | 85.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0054 | 48 | 0.258 | 613 | 38.3 | 34.9% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0047 | 48 | 0.227 | 343 | 89.3 | 81.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.115 | 14 | 6.8 | 6.2% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.114 | 2 | 0.9 | 0.8% | memory |
| rope_q | `rope_opt` | 0.0022 | 48 | 0.103 | 19 | 7.6 | 6.9% | memory |
| rope_k | `rope_opt` | 0.0019 | 48 | 0.090 | 3 | 1.1 | 1.0% | memory |

**Total inference time (this stage)** ≈ **24.05 ms**

#### PTL — decode kv=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1944 | 48 | 9.332 | 644 | 40.2 | 36.6% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1348 | 48 | 6.473 | 343 | 89.3 | 81.2% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0697 | 48 | 3.347 | 343 | 89.3 | 81.2% | memory |
| lm_head | `gemm_kernel` | 3.1701 | 1 | 3.170 | 196 | 99.8 | 90.7% | memory |
| fc_qkv | `gemm_kernel` | 0.0562 | 48 | 2.695 | 373 | 97.3 | 88.4% | memory |
| fc_o | `gemm_kernel` | 0.0464 | 48 | 2.229 | 361 | 94.1 | 85.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0100 | 48 | 0.482 | 644 | 40.2 | 36.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0047 | 48 | 0.227 | 343 | 89.3 | 81.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.115 | 14 | 6.8 | 6.2% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.114 | 2 | 0.9 | 0.8% | memory |
| rope_q | `rope_opt` | 0.0022 | 48 | 0.103 | 19 | 7.6 | 6.9% | memory |
| rope_k | `rope_opt` | 0.0019 | 48 | 0.090 | 3 | 1.1 | 1.0% | memory |

**Total inference time (this stage)** ≈ **28.80 ms**

#### PTL — decode kv=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.3805 | 48 | 18.264 | 654 | 40.9 | 37.2% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1348 | 48 | 6.473 | 343 | 89.3 | 81.2% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0697 | 48 | 3.347 | 343 | 89.3 | 81.2% | memory |
| lm_head | `gemm_kernel` | 3.1701 | 1 | 3.170 | 196 | 99.8 | 90.7% | memory |
| fc_qkv | `gemm_kernel` | 0.0562 | 48 | 2.695 | 373 | 97.3 | 88.4% | memory |
| fc_o | `gemm_kernel` | 0.0464 | 48 | 2.229 | 361 | 94.1 | 85.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0238 | 48 | 1.140 | 654 | 40.9 | 37.2% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0047 | 48 | 0.227 | 343 | 89.3 | 81.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.115 | 14 | 6.8 | 6.2% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.114 | 2 | 0.9 | 0.8% | memory |
| rope_q | `rope_opt` | 0.0022 | 48 | 0.103 | 19 | 7.6 | 6.9% | memory |
| rope_k | `rope_opt` | 0.0019 | 48 | 0.090 | 3 | 1.1 | 1.0% | memory |

**Total inference time (this stage)** ≈ **38.39 ms**

#### PTL — decode kv=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.7596 | 48 | 36.462 | 655 | 40.9 | 37.2% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1348 | 48 | 6.473 | 343 | 89.3 | 81.2% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0697 | 48 | 3.347 | 343 | 89.3 | 81.2% | memory |
| lm_head | `gemm_kernel` | 3.1701 | 1 | 3.170 | 196 | 99.8 | 90.7% | memory |
| fc_qkv | `gemm_kernel` | 0.0562 | 48 | 2.695 | 373 | 97.3 | 88.4% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0514 | 48 | 2.468 | 655 | 40.9 | 37.2% | memory |
| fc_o | `gemm_kernel` | 0.0464 | 48 | 2.229 | 361 | 94.1 | 85.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0047 | 48 | 0.227 | 343 | 89.3 | 81.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.115 | 14 | 6.8 | 6.2% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.114 | 2 | 0.9 | 0.8% | memory |
| rope_q | `rope_opt` | 0.0022 | 48 | 0.103 | 19 | 7.6 | 6.9% | memory |
| rope_k | `rope_opt` | 0.0019 | 48 | 0.090 | 3 | 1.1 | 1.0% | memory |

**Total inference time (this stage)** ≈ **57.92 ms**

#### PTL — decode kv=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 1.5275 | 48 | 73.318 | 653 | 40.8 | 37.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1348 | 48 | 6.473 | 343 | 89.3 | 81.2% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1053 | 48 | 5.055 | 653 | 40.8 | 37.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0697 | 48 | 3.347 | 343 | 89.3 | 81.2% | memory |
| lm_head | `gemm_kernel` | 3.1701 | 1 | 3.170 | 196 | 99.8 | 90.7% | memory |
| fc_qkv | `gemm_kernel` | 0.0562 | 48 | 2.695 | 373 | 97.3 | 88.4% | memory |
| fc_o | `gemm_kernel` | 0.0464 | 48 | 2.229 | 361 | 94.1 | 85.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0047 | 48 | 0.227 | 343 | 89.3 | 81.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.115 | 14 | 6.8 | 6.2% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.114 | 2 | 0.9 | 0.8% | memory |
| rope_q | `rope_opt` | 0.0022 | 48 | 0.103 | 19 | 7.6 | 6.9% | memory |
| rope_k | `rope_opt` | 0.0019 | 48 | 0.090 | 3 | 1.1 | 1.0% | memory |

**Total inference time (this stage)** ≈ **97.36 ms**

#### PTL — decode kv=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 3.0774 | 48 | 147.714 | 655 | 41.0 | 37.2% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1941 | 48 | 9.318 | 655 | 41.0 | 37.2% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1348 | 48 | 6.473 | 343 | 89.3 | 81.2% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0697 | 48 | 3.347 | 343 | 89.3 | 81.2% | memory |
| lm_head | `gemm_kernel` | 3.1701 | 1 | 3.170 | 196 | 99.8 | 90.7% | memory |
| fc_qkv | `gemm_kernel` | 0.0562 | 48 | 2.695 | 373 | 97.3 | 88.4% | memory |
| fc_o | `gemm_kernel` | 0.0464 | 48 | 2.229 | 361 | 94.1 | 85.6% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0047 | 48 | 0.227 | 343 | 89.3 | 81.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.115 | 14 | 6.8 | 6.2% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0024 | 48 | 0.114 | 2 | 0.9 | 0.8% | memory |
| rope_q | `rope_opt` | 0.0022 | 48 | 0.103 | 19 | 7.6 | 6.9% | memory |
| rope_k | `rope_opt` | 0.0019 | 48 | 0.090 | 3 | 1.1 | 1.0% | memory |

**Total inference time (this stage)** ≈ **176.02 ms**

### PREFILL — per-S-size (compute-bound uses INT8 XMX peak = 118.0 TOPS)


#### PTL — prefill S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 5.8105 | 48 | 278.903 | 10700 | 61.0 | 9.1% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 0.6218 | 48 | 29.844 | 25292 | 49.4 | 21.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 0.5649 | 48 | 27.118 | 10700 | 61.0 | 9.1% | compute |
| fc_qkv | `gemm_kernel` | 0.3777 | 48 | 18.132 | 49660 | 58.6 | 42.1% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.2987 | 48 | 14.339 | 10700 | 61.0 | 9.1% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.2844 | 48 | 13.650 | 10700 | 61.0 | 9.1% | compute |
| fc_o | `gemm_kernel` | 0.2686 | 48 | 12.892 | 45381 | 55.7 | 38.5% | compute |
| rope_q | `rope_opt` | 0.1112 | 48 | 5.340 | 377 | 150.8 | 137.1% | memory |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.1100 | 48 | 5.279 | 45381 | 55.7 | 38.5% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0492 | 96 | 4.719 | 341 | 170.7 | 155.2% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0575 | 48 | 2.760 | 25292 | 49.4 | 21.4% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.0547 | 48 | 2.625 | 49660 | 58.6 | 42.1% | compute |

**Total inference time (this stage)** ≈ **415.60 ms**

#### PTL — prefill S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 6.7013 | 48 | 321.663 | 14655 | 54.9 | 12.4% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 2.2878 | 48 | 109.817 | 28550 | 27.9 | 24.2% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 1.6512 | 48 | 79.257 | 14655 | 54.9 | 12.4% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.9302 | 48 | 44.649 | 14655 | 54.9 | 12.4% | compute |
| fc_qkv | `gemm_kernel` | 0.7116 | 48 | 34.158 | 53031 | 49.4 | 45.0% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.6613 | 48 | 31.745 | 14655 | 54.9 | 12.4% | compute |
| fc_o | `gemm_kernel` | 0.4910 | 48 | 23.567 | 47534 | 46.6 | 40.3% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.2733 | 48 | 13.118 | 14655 | 54.9 | 12.4% | compute |
| rope_q | `rope_opt` | 0.2616 | 48 | 12.555 | 321 | 128.3 | 116.6% | memory |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.2319 | 48 | 11.130 | 47534 | 46.6 | 40.3% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0937 | 96 | 8.995 | 358 | 179.1 | 162.8% | memory |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.0983 | 48 | 4.717 | 53031 | 49.4 | 45.0% | compute |

**Total inference time (this stage)** ≈ **695.37 ms**

#### PTL — prefill S=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 11.1449 | 48 | 534.953 | 16443 | 45.5 | 13.9% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 8.9292 | 48 | 428.600 | 30000 | 14.6 | 25.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 2.4311 | 48 | 116.695 | 16443 | 45.5 | 13.9% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 2.3239 | 48 | 111.548 | 16443 | 45.5 | 13.9% | compute |
| fc_qkv | `gemm_kernel` | 1.5199 | 48 | 72.956 | 48627 | 39.3 | 41.2% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 1.4706 | 48 | 70.591 | 16443 | 45.5 | 13.9% | compute |
| fc_o | `gemm_kernel` | 0.9369 | 48 | 44.969 | 48845 | 41.8 | 41.4% | compute |
| rope_q | `rope_opt` | 0.6188 | 48 | 29.703 | 271 | 108.4 | 98.6% | memory |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.5555 | 48 | 26.664 | 16443 | 45.5 | 13.9% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.2365 | 96 | 22.704 | 284 | 141.9 | 129.0% | memory |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.4700 | 48 | 22.562 | 48845 | 41.8 | 41.4% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.2466 | 48 | 11.837 | 48627 | 39.3 | 41.2% | compute |

**Total inference time (this stage)** ≈ **1493.78 ms**

#### PTL — prefill S=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 35.0989 | 48 | 1684.750 | 30896 | 7.5 | 26.2% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 19.7640 | 48 | 948.670 | 18064 | 41.2 | 15.3% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 5.1325 | 48 | 246.359 | 18064 | 41.2 | 15.3% | compute |
| fc_qkv | `gemm_kernel` | 3.1444 | 48 | 150.934 | 47800 | 35.6 | 40.5% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 3.1133 | 48 | 149.439 | 18064 | 41.2 | 15.3% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 3.0574 | 48 | 146.756 | 18064 | 41.2 | 15.3% | compute |
| fc_o | `gemm_kernel` | 1.7396 | 48 | 83.502 | 54257 | 43.1 | 46.0% | compute |
| rope_q | `rope_opt` | 1.3189 | 48 | 63.306 | 254 | 101.8 | 92.5% | memory |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 1.1765 | 48 | 56.471 | 18064 | 41.2 | 15.3% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.5675 | 96 | 54.483 | 236 | 118.2 | 107.5% | memory |
| moe_3gemm_fused | `reorder_data_0_0` | 1.0166 | 48 | 48.799 | 18064 | 41.2 | 15.3% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 0.7935 | 48 | 38.087 | 54257 | 43.1 | 46.0% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.4497 | 48 | 21.585 | 47800 | 35.6 | 40.5% | compute |

**Total inference time (this stage)** ≈ **3693.14 ms**

#### PTL — prefill S=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 139.1035 | 48 | 6676.968 | 31376 | 3.8 | 26.6% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 30.7660 | 48 | 1476.769 | 21029 | 42.8 | 17.8% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 10.3727 | 48 | 497.888 | 21029 | 42.8 | 17.8% | compute |
| fc_qkv | `gemm_kernel` | 6.2978 | 48 | 302.296 | 48161 | 34.4 | 40.8% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 5.8251 | 48 | 279.602 | 21029 | 42.8 | 17.8% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 5.8132 | 48 | 279.034 | 21029 | 42.8 | 17.8% | compute |
| fc_o | `gemm_kernel` | 3.4560 | 48 | 165.888 | 53369 | 40.7 | 45.2% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 2.2482 | 48 | 107.916 | 21029 | 42.8 | 17.8% | compute |
| moe_3gemm_fused | `reorder_data_0_0` | 2.0148 | 48 | 96.708 | 21029 | 42.8 | 17.8% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 1.6945 | 48 | 81.337 | 53369 | 40.7 | 45.2% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 0.8366 | 48 | 40.156 | 48161 | 34.4 | 40.8% | compute |

**Total inference time (this stage)** ≈ **10004.56 ms**

#### PTL — prefill S=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 612.4418 | 48 | 29397.207 | 28625 | 1.7 | 24.3% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 61.5405 | 48 | 2953.945 | 20409 | 39.0 | 17.3% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 22.9585 | 48 | 1102.010 | 20409 | 39.0 | 17.3% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 12.2968 | 48 | 590.245 | 20409 | 39.0 | 17.3% | compute |
| fc_qkv | `gemm_kernel` | 11.9661 | 48 | 574.371 | 49890 | 34.9 | 42.3% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 11.8942 | 48 | 570.920 | 20409 | 39.0 | 17.3% | compute |
| fc_o | `gemm_kernel` | 6.8536 | 48 | 328.974 | 53798 | 40.2 | 45.6% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 4.6039 | 48 | 220.989 | 20409 | 39.0 | 17.3% | compute |
| moe_3gemm_fused | `reorder_data_0_0` | 4.2866 | 48 | 205.757 | 20409 | 39.0 | 17.3% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 3.3653 | 48 | 161.536 | 53798 | 40.2 | 45.6% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 1.8080 | 48 | 86.785 | 49890 | 34.9 | 42.3% | compute |

**Total inference time (this stage)** ≈ **36192.74 ms**

#### PTL — prefill S=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 2671.5261 | 48 | 128233.251 | 26297 | 0.8 | 22.3% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 114.6764 | 48 | 5504.468 | 20660 | 38.2 | 17.5% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 46.6739 | 48 | 2240.347 | 20660 | 38.2 | 17.5% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 28.9373 | 48 | 1388.992 | 20660 | 38.2 | 17.5% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 23.8542 | 48 | 1145.002 | 20660 | 38.2 | 17.5% | compute |
| fc_qkv | `gemm_kernel` | 23.3841 | 48 | 1122.436 | 50568 | 35.0 | 42.9% | compute |
| fc_o | `gemm_kernel` | 13.8722 | 48 | 665.866 | 52318 | 38.7 | 44.4% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 9.1289 | 48 | 438.187 | 20660 | 38.2 | 17.5% | compute |
| moe_3gemm_fused | `reorder_data_0_0` | 8.5909 | 48 | 412.363 | 20660 | 38.2 | 17.5% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 7.1439 | 48 | 342.907 | 52318 | 38.7 | 44.4% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 3.7952 | 48 | 182.167 | 50568 | 35.0 | 42.9% | compute |

**Total inference time (this stage)** ≈ **141675.99 ms**

#### PTL — prefill S=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 12044.4721 | 48 | 578134.662 | 23352 | 0.4 | 19.8% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 218.9695 | 48 | 10510.537 | 20051 | 36.5 | 17.0% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 93.5027 | 48 | 4488.128 | 20051 | 36.5 | 17.0% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 82.5984 | 48 | 3964.725 | 20051 | 36.5 | 17.0% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 48.5455 | 48 | 2330.185 | 20051 | 36.5 | 17.0% | compute |
| fc_qkv | `gemm_kernel` | 47.7183 | 48 | 2290.477 | 49898 | 34.3 | 42.3% | compute |
| fc_o | `gemm_kernel` | 27.5093 | 48 | 1320.448 | 52822 | 38.9 | 44.8% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 17.9493 | 48 | 861.565 | 20051 | 36.5 | 17.0% | compute |
| moe_3gemm_fused | `reorder_data_0_0` | 17.1658 | 48 | 823.957 | 20051 | 36.5 | 17.0% | compute |
| fc_o | `dynamic_quantize_gpu_opt_0_0` | 14.1216 | 48 | 677.839 | 52822 | 38.9 | 44.8% | compute |
| fc_qkv | `dynamic_quantize_gpu_opt_0_0` | 7.3694 | 48 | 353.731 | 49898 | 34.3 | 42.3% | compute |

**Total inference time (this stage)** ≈ **605756.25 ms**
