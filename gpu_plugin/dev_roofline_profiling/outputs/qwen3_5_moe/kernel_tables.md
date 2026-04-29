# qwen3_5_moe — Per-token-size Kernel Tables


## Platform: BMG

### DECODE — per kv (M=1)


#### BMG — decode kv=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0289 | 40 | 1.158 | 972 | 252.9 | 55.5% | memory |
| lm_head | `gemm_kernel` | 1.1517 | 1 | 1.152 | 883 | 448.9 | 98.4% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 30 | 0.989 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0178 | 40 | 0.712 | 972 | 252.9 | 55.5% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 30 | 0.386 | 82 | 163.2 | 35.8% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0207 | 10 | 0.207 | 713 | 44.6 | 9.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0042 | 40 | 0.168 | 972 | 252.9 | 55.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 10 | 0.155 | 1349 | 351.3 | 77.0% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 10 | 0.131 | 1286 | 334.9 | 73.4% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 40 | 0.125 | 972 | 252.9 | 55.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 80 | 0.119 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 80 | 0.060 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0015 | 40 | 0.060 | 972 | 252.9 | 55.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0015 | 10 | 0.015 | 713 | 44.6 | 9.8% | memory |
| rope_q | `rope_opt` | 0.0015 | 10 | 0.015 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 10 | 0.013 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 3 | 1.6 | 0.4% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0012 | 10 | 0.012 | 713 | 44.6 | 9.8% | memory |

**Total inference time (this stage)** ≈ **5.50 ms**

#### BMG — decode kv=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0289 | 40 | 1.158 | 972 | 252.9 | 55.5% | memory |
| lm_head | `gemm_kernel` | 1.1517 | 1 | 1.152 | 883 | 448.9 | 98.4% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 30 | 0.989 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0178 | 40 | 0.712 | 972 | 252.9 | 55.5% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0403 | 10 | 0.403 | 771 | 48.2 | 10.6% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 30 | 0.386 | 82 | 163.2 | 35.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0042 | 40 | 0.168 | 972 | 252.9 | 55.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 10 | 0.155 | 1349 | 351.3 | 77.0% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 10 | 0.131 | 1286 | 334.9 | 73.4% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 40 | 0.125 | 972 | 252.9 | 55.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 80 | 0.119 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 80 | 0.060 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0015 | 40 | 0.060 | 972 | 252.9 | 55.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0020 | 10 | 0.020 | 771 | 48.2 | 10.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 10 | 0.015 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 10 | 0.013 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **5.69 ms**

#### BMG — decode kv=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0289 | 40 | 1.158 | 972 | 252.9 | 55.5% | memory |
| lm_head | `gemm_kernel` | 1.1517 | 1 | 1.152 | 883 | 448.9 | 98.4% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 30 | 0.989 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0178 | 40 | 0.712 | 972 | 252.9 | 55.5% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 30 | 0.386 | 82 | 163.2 | 35.8% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0333 | 10 | 0.333 | 1767 | 110.4 | 24.2% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0042 | 40 | 0.168 | 972 | 252.9 | 55.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 10 | 0.155 | 1349 | 351.3 | 77.0% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 10 | 0.131 | 1286 | 334.9 | 73.4% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 40 | 0.125 | 972 | 252.9 | 55.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 80 | 0.119 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 80 | 0.060 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0015 | 40 | 0.060 | 972 | 252.9 | 55.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0033 | 10 | 0.033 | 1767 | 110.4 | 24.2% | memory |
| rope_q | `rope_opt` | 0.0015 | 10 | 0.015 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 10 | 0.013 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **5.63 ms**

#### BMG — decode kv=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0289 | 40 | 1.158 | 972 | 252.9 | 55.5% | memory |
| lm_head | `gemm_kernel` | 1.1517 | 1 | 1.152 | 883 | 448.9 | 98.4% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 30 | 0.989 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0178 | 40 | 0.712 | 972 | 252.9 | 55.5% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0648 | 10 | 0.648 | 1865 | 116.6 | 25.6% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 30 | 0.386 | 82 | 163.2 | 35.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0042 | 40 | 0.168 | 972 | 252.9 | 55.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 10 | 0.155 | 1349 | 351.3 | 77.0% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 10 | 0.131 | 1286 | 334.9 | 73.4% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 40 | 0.125 | 972 | 252.9 | 55.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 80 | 0.119 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 80 | 0.060 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0015 | 40 | 0.060 | 972 | 252.9 | 55.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0055 | 10 | 0.055 | 1865 | 116.6 | 25.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 10 | 0.015 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 10 | 0.013 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **5.97 ms**

#### BMG — decode kv=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1336 | 10 | 1.336 | 1791 | 112.0 | 24.6% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0289 | 40 | 1.158 | 972 | 252.9 | 55.5% | memory |
| lm_head | `gemm_kernel` | 1.1517 | 1 | 1.152 | 883 | 448.9 | 98.4% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 30 | 0.989 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0178 | 40 | 0.712 | 972 | 252.9 | 55.5% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 30 | 0.386 | 82 | 163.2 | 35.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0042 | 40 | 0.168 | 972 | 252.9 | 55.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 10 | 0.155 | 1349 | 351.3 | 77.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0141 | 10 | 0.141 | 1791 | 112.0 | 24.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 10 | 0.131 | 1286 | 334.9 | 73.4% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 40 | 0.125 | 972 | 252.9 | 55.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 80 | 0.119 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 80 | 0.060 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0015 | 40 | 0.060 | 972 | 252.9 | 55.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 10 | 0.015 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 10 | 0.013 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **6.74 ms**

#### BMG — decode kv=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.2893 | 10 | 2.893 | 1657 | 103.6 | 22.7% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0289 | 40 | 1.158 | 972 | 252.9 | 55.5% | memory |
| lm_head | `gemm_kernel` | 1.1517 | 1 | 1.152 | 883 | 448.9 | 98.4% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 30 | 0.989 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0178 | 40 | 0.712 | 972 | 252.9 | 55.5% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 30 | 0.386 | 82 | 163.2 | 35.8% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0323 | 10 | 0.323 | 1657 | 103.6 | 22.7% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0042 | 40 | 0.168 | 972 | 252.9 | 55.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 10 | 0.155 | 1349 | 351.3 | 77.0% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 10 | 0.131 | 1286 | 334.9 | 73.4% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 40 | 0.125 | 972 | 252.9 | 55.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 80 | 0.119 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 80 | 0.060 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0015 | 40 | 0.060 | 972 | 252.9 | 55.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 10 | 0.015 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 10 | 0.013 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **8.48 ms**

#### BMG — decode kv=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.5733 | 10 | 5.733 | 1649 | 103.0 | 22.6% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0289 | 40 | 1.158 | 972 | 252.9 | 55.5% | memory |
| lm_head | `gemm_kernel` | 1.1517 | 1 | 1.152 | 883 | 448.9 | 98.4% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 30 | 0.989 | 1526 | 397.3 | 87.1% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0751 | 10 | 0.751 | 1649 | 103.0 | 22.6% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0178 | 40 | 0.712 | 972 | 252.9 | 55.5% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 30 | 0.386 | 82 | 163.2 | 35.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0042 | 40 | 0.168 | 972 | 252.9 | 55.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 10 | 0.155 | 1349 | 351.3 | 77.0% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 10 | 0.131 | 1286 | 334.9 | 73.4% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 40 | 0.125 | 972 | 252.9 | 55.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 80 | 0.119 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 80 | 0.060 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0015 | 40 | 0.060 | 972 | 252.9 | 55.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 10 | 0.015 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 10 | 0.013 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **11.75 ms**

#### BMG — decode kv=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 1.1244 | 10 | 11.244 | 1663 | 103.9 | 22.8% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1642 | 10 | 1.642 | 1663 | 103.9 | 22.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0289 | 40 | 1.158 | 972 | 252.9 | 55.5% | memory |
| lm_head | `gemm_kernel` | 1.1517 | 1 | 1.152 | 883 | 448.9 | 98.4% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 30 | 0.989 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0178 | 40 | 0.712 | 972 | 252.9 | 55.5% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 30 | 0.386 | 82 | 163.2 | 35.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0042 | 40 | 0.168 | 972 | 252.9 | 55.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 10 | 0.155 | 1349 | 351.3 | 77.0% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 10 | 0.131 | 1286 | 334.9 | 73.4% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0031 | 40 | 0.125 | 972 | 252.9 | 55.5% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 80 | 0.119 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 80 | 0.060 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `dynamic_quantize_gpu_opt_0_0` | 0.0015 | 40 | 0.060 | 972 | 252.9 | 55.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 10 | 0.015 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 10 | 0.013 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 10 | 0.013 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **18.15 ms**

### PREFILL — per S (compute-bound uses INT8 XMX peak = 233.5 TOPS)


#### BMG — prefill S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 3.4896 | 40 | 139.583 | 12862 | 120.7 | 5.5% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 1.0665 | 30 | 31.995 | 1007 | 15.7 | 0.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 0.5256 | 40 | 21.024 | 12862 | 120.7 | 5.5% | compute |
| fc_linattn_proj | `gemm_kernel` | 0.3592 | 30 | 10.776 | 136169 | 145.1 | 58.3% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.1649 | 40 | 6.598 | 12862 | 120.7 | 5.5% | compute |
| moe_3gemm_fused | `gemm_kernel` | 0.1180 | 40 | 4.719 | 12862 | 120.7 | 5.5% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 0.4428 | 10 | 4.428 | 36899 | 72.1 | 15.8% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0188 | 80 | 1.500 | 895 | 447.3 | 98.1% | memory |
| fc_qkv_full | `gemm_kernel` | 0.1461 | 10 | 1.461 | 129811 | 153.1 | 55.6% | compute |
| fc_o_full | `gemm_kernel` | 0.1268 | 10 | 1.268 | 103405 | 127.0 | 44.3% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.0193 | 30 | 0.579 | 136169 | 145.1 | 58.3% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.0394 | 10 | 0.394 | 103405 | 127.0 | 44.3% | compute |
| rope_q | `rope_opt` | 0.0352 | 10 | 0.352 | 1193 | 477.2 | 104.6% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.0194 | 10 | 0.194 | 129811 | 153.1 | 55.6% | compute |

**Total inference time (this stage)** ≈ **224.87 ms**

#### BMG — prefill S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 3.5194 | 40 | 140.775 | 19622 | 116.0 | 8.4% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 2.0296 | 30 | 60.889 | 1058 | 16.5 | 0.5% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 1.3783 | 40 | 55.130 | 19622 | 116.0 | 8.4% | compute |
| fc_linattn_proj | `gemm_kernel` | 0.6650 | 30 | 19.949 | 146645 | 119.9 | 62.8% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.3830 | 40 | 15.321 | 19622 | 116.0 | 8.4% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 1.2564 | 10 | 12.564 | 53344 | 52.1 | 22.8% | compute |
| moe_3gemm_fused | `gemm_kernel` | 0.2026 | 40 | 8.103 | 19622 | 116.0 | 8.4% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.1429 | 40 | 5.718 | 19622 | 116.0 | 8.4% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0357 | 80 | 2.859 | 939 | 469.4 | 102.9% | memory |
| fc_qkv_full | `gemm_kernel` | 0.2725 | 10 | 2.725 | 138036 | 128.6 | 59.1% | compute |
| fc_o_full | `gemm_kernel` | 0.2173 | 10 | 2.173 | 116773 | 114.5 | 50.0% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.0380 | 30 | 1.139 | 146645 | 119.9 | 62.8% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.0769 | 10 | 0.769 | 116773 | 114.5 | 50.0% | compute |
| rope_q | `rope_opt` | 0.0604 | 10 | 0.604 | 1390 | 556.0 | 121.9% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.0386 | 10 | 0.386 | 138036 | 128.6 | 59.1% | compute |

**Total inference time (this stage)** ≈ **329.11 ms**

#### BMG — prefill S=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 4.1506 | 40 | 166.023 | 22232 | 92.9 | 9.5% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 4.0887 | 40 | 163.549 | 22232 | 92.9 | 9.5% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 4.1641 | 30 | 124.923 | 1031 | 16.1 | 0.4% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 4.8439 | 10 | 48.439 | 55935 | 27.3 | 24.0% | compute |
| fc_linattn_proj | `gemm_kernel` | 1.3101 | 30 | 39.304 | 148657 | 103.1 | 63.7% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.8794 | 40 | 35.177 | 22232 | 92.9 | 9.5% | compute |
| moe_3gemm_fused | `gemm_kernel` | 0.4125 | 40 | 16.498 | 22232 | 92.9 | 9.5% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.2779 | 40 | 11.115 | 22232 | 92.9 | 9.5% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.2610 | 40 | 10.439 | 22232 | 92.9 | 9.5% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0697 | 80 | 5.577 | 963 | 481.3 | 105.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.5297 | 10 | 5.297 | 141450 | 114.2 | 60.6% | compute |
| fc_o_full | `gemm_kernel` | 0.4017 | 10 | 4.017 | 124159 | 106.3 | 53.2% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.0767 | 30 | 2.300 | 148657 | 103.1 | 63.7% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.1518 | 10 | 1.518 | 124159 | 106.3 | 53.2% | compute |
| rope_q | `rope_opt` | 0.1076 | 10 | 1.076 | 1559 | 623.7 | 136.8% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.0775 | 10 | 0.775 | 141450 | 114.2 | 60.6% | compute |

**Total inference time (this stage)** ≈ **636.03 ms**

#### BMG — prefill S=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| gated_delta_net | `gated_delta_net_ref_sa` | 8.7010 | 30 | 261.029 | 987 | 15.4 | 0.4% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 6.5200 | 40 | 260.801 | 27140 | 89.8 | 11.6% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 6.0465 | 40 | 241.861 | 27140 | 89.8 | 11.6% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 18.8892 | 10 | 188.892 | 57806 | 14.1 | 24.8% | compute |
| fc_linattn_proj | `gemm_kernel` | 2.5721 | 30 | 77.164 | 151209 | 95.5 | 64.8% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 1.8813 | 40 | 75.250 | 27140 | 89.8 | 11.6% | compute |
| moe_3gemm_fused | `gemm_kernel` | 0.8131 | 40 | 32.523 | 27140 | 89.8 | 11.6% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.5472 | 40 | 21.889 | 27140 | 89.8 | 11.6% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.5340 | 40 | 21.359 | 27140 | 89.8 | 11.6% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.1375 | 80 | 11.002 | 976 | 488.0 | 107.0% | memory |
| fc_qkv_full | `gemm_kernel` | 1.0496 | 10 | 10.496 | 142563 | 106.3 | 61.1% | compute |
| fc_o_full | `gemm_kernel` | 0.7880 | 10 | 7.880 | 125990 | 100.1 | 54.0% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.1547 | 30 | 4.640 | 151209 | 95.5 | 64.8% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.3029 | 10 | 3.029 | 125990 | 100.1 | 54.0% | compute |
| rope_q | `rope_opt` | 0.2015 | 10 | 2.015 | 1665 | 666.0 | 146.1% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.1555 | 10 | 1.555 | 142563 | 106.3 | 61.1% | compute |

**Total inference time (this stage)** ≈ **1221.38 ms**

#### BMG — prefill S=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 20.9374 | 40 | 837.496 | 22451 | 64.6 | 9.6% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 76.6344 | 10 | 766.344 | 57216 | 7.0 | 24.5% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 18.0672 | 30 | 542.015 | 951 | 14.9 | 0.4% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 11.2474 | 40 | 449.895 | 22451 | 64.6 | 9.6% | compute |
| fc_linattn_proj | `gemm_kernel` | 5.1996 | 30 | 155.988 | 149637 | 89.9 | 64.1% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 3.8134 | 40 | 152.536 | 22451 | 64.6 | 9.6% | compute |
| moe_3gemm_fused | `gemm_kernel` | 1.5669 | 40 | 62.674 | 22451 | 64.6 | 9.6% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 1.0909 | 40 | 43.635 | 22451 | 64.6 | 9.6% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 1.0865 | 40 | 43.460 | 22451 | 64.6 | 9.6% | compute |
| fc_qkv_full | `gemm_kernel` | 2.0779 | 10 | 20.779 | 143942 | 102.9 | 61.7% | compute |
| fc_o_full | `gemm_kernel` | 1.5569 | 10 | 15.569 | 127521 | 97.4 | 54.6% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.3113 | 30 | 9.340 | 149637 | 89.9 | 64.1% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.5987 | 10 | 5.987 | 127521 | 97.4 | 54.6% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.3092 | 10 | 3.092 | 143942 | 102.9 | 61.7% | compute |

**Total inference time (this stage)** ≈ **3108.81 ms**

#### BMG — prefill S=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 90.1778 | 40 | 3607.114 | 14421 | 38.3 | 6.2% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 301.9960 | 10 | 3019.960 | 58169 | 3.6 | 24.9% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 38.8448 | 30 | 1165.345 | 885 | 13.8 | 0.4% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 20.4819 | 40 | 819.276 | 14421 | 38.3 | 6.2% | compute |
| fc_linattn_proj | `gemm_kernel` | 10.3285 | 30 | 309.855 | 150622 | 88.1 | 64.5% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 7.5934 | 40 | 303.736 | 14421 | 38.3 | 6.2% | compute |
| moe_3gemm_fused | `gemm_kernel` | 2.9339 | 40 | 117.357 | 14421 | 38.3 | 6.2% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 2.1638 | 40 | 86.551 | 14421 | 38.3 | 6.2% | compute |
| fc_qkv_full | `gemm_kernel` | 4.1417 | 10 | 41.417 | 144247 | 100.8 | 61.8% | compute |
| fc_o_full | `gemm_kernel` | 3.0752 | 10 | 30.752 | 128376 | 96.0 | 55.0% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.6212 | 30 | 18.636 | 150622 | 88.1 | 64.5% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 1.2072 | 10 | 12.072 | 128376 | 96.0 | 55.0% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.6223 | 10 | 6.223 | 144247 | 100.8 | 61.8% | compute |

**Total inference time (this stage)** ≈ **9538.29 ms**

#### BMG — prefill S=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 379.2101 | 40 | 15168.404 | 8180 | 20.9 | 3.5% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 1203.8112 | 10 | 12038.112 | 58415 | 1.8 | 25.0% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 81.5412 | 30 | 2446.237 | 843 | 13.2 | 0.4% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 38.7324 | 40 | 1549.297 | 8180 | 20.9 | 3.5% | compute |
| fc_linattn_proj | `gemm_kernel` | 20.7828 | 30 | 623.483 | 149669 | 86.4 | 64.1% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 14.9241 | 40 | 596.964 | 8180 | 20.9 | 3.5% | compute |
| fc_qkv_full | `gemm_kernel` | 8.2753 | 10 | 82.753 | 144373 | 99.8 | 61.8% | compute |
| fc_o_full | `gemm_kernel` | 6.1210 | 10 | 61.210 | 128950 | 95.4 | 55.2% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 1.2561 | 30 | 37.683 | 149669 | 86.4 | 64.1% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 2.4056 | 10 | 24.056 | 128950 | 95.4 | 55.2% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 1.2445 | 10 | 12.445 | 144373 | 99.8 | 61.8% | compute |

**Total inference time (this stage)** ≈ **32640.64 ms**

#### BMG — prefill S=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 4813.2497 | 10 | 48132.497 | 58459 | 0.9 | 25.0% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 188.9804 | 30 | 5669.412 | 727 | 11.4 | 0.3% | compute |
| fc_linattn_proj | `gemm_kernel` | 41.2725 | 30 | 1238.176 | 150556 | 86.3 | 64.5% | compute |
| fc_qkv_full | `gemm_kernel` | 16.6086 | 10 | 166.086 | 143847 | 98.9 | 61.6% | compute |
| fc_o_full | `gemm_kernel` | 12.2761 | 10 | 122.761 | 128355 | 94.5 | 55.0% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 2.5456 | 30 | 76.367 | 150556 | 86.3 | 64.5% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 4.8562 | 10 | 48.562 | 128355 | 94.5 | 55.0% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 2.5005 | 10 | 25.005 | 143847 | 98.9 | 61.6% | compute |
| moe_3gemm_fused | `reorder_data_0_0` | 0.0043 | 40 | 0.171 | 1738103861 | 4337715.8 | 744459.2% | compute |

**Total inference time (this stage)** ≈ **55479.04 ms**

## Platform: PTL

### DECODE — per kv (M=1)


#### PTL — decode kv=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| lm_head | `gemm_kernel` | 5.1610 | 1 | 5.161 | 197 | 100.2 | 91.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1059 | 40 | 4.235 | 326 | 84.7 | 77.0% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 30 | 3.843 | 393 | 102.3 | 93.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0527 | 40 | 2.108 | 326 | 84.7 | 77.0% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 30 | 0.959 | 33 | 65.6 | 59.6% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0571 | 10 | 0.571 | 260 | 16.3 | 14.8% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 10 | 0.554 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 10 | 0.462 | 363 | 94.6 | 86.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 80 | 0.236 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0059 | 40 | 0.235 | 326 | 84.7 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0048 | 40 | 0.190 | 326 | 84.7 | 77.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 80 | 0.120 | 1 | 8.2 | 7.4% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0042 | 10 | 0.042 | 260 | 16.3 | 14.8% | memory |
| rope_q | `rope_opt` | 0.0025 | 10 | 0.025 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 10 | 0.022 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **18.81 ms**

#### PTL — decode kv=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| lm_head | `gemm_kernel` | 5.1610 | 1 | 5.161 | 197 | 100.2 | 91.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1059 | 40 | 4.235 | 326 | 84.7 | 77.0% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 30 | 3.843 | 393 | 102.3 | 93.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0527 | 40 | 2.108 | 326 | 84.7 | 77.0% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.1029 | 10 | 1.029 | 303 | 18.9 | 17.2% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 30 | 0.959 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 10 | 0.554 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 10 | 0.462 | 363 | 94.6 | 86.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 80 | 0.236 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0059 | 40 | 0.235 | 326 | 84.7 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0048 | 40 | 0.190 | 326 | 84.7 | 77.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 80 | 0.120 | 1 | 8.2 | 7.4% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0040 | 10 | 0.040 | 303 | 18.9 | 17.2% | memory |
| rope_q | `rope_opt` | 0.0025 | 10 | 0.025 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 10 | 0.022 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **19.27 ms**

#### PTL — decode kv=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| lm_head | `gemm_kernel` | 5.1610 | 1 | 5.161 | 197 | 100.2 | 91.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1059 | 40 | 4.235 | 326 | 84.7 | 77.0% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 30 | 3.843 | 393 | 102.3 | 93.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0527 | 40 | 2.108 | 326 | 84.7 | 77.0% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 30 | 0.959 | 33 | 65.6 | 59.6% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0785 | 10 | 0.785 | 764 | 47.8 | 43.4% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 10 | 0.554 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 10 | 0.462 | 363 | 94.6 | 86.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 80 | 0.236 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0059 | 40 | 0.235 | 326 | 84.7 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0048 | 40 | 0.190 | 326 | 84.7 | 77.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 80 | 0.120 | 1 | 8.2 | 7.4% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0053 | 10 | 0.053 | 764 | 47.8 | 43.4% | memory |
| rope_q | `rope_opt` | 0.0025 | 10 | 0.025 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 10 | 0.022 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **19.04 ms**

#### PTL — decode kv=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| lm_head | `gemm_kernel` | 5.1610 | 1 | 5.161 | 197 | 100.2 | 91.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1059 | 40 | 4.235 | 326 | 84.7 | 77.0% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 30 | 3.843 | 393 | 102.3 | 93.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0527 | 40 | 2.108 | 326 | 84.7 | 77.0% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1498 | 10 | 1.498 | 827 | 51.7 | 47.0% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 30 | 0.959 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 10 | 0.554 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 10 | 0.462 | 363 | 94.6 | 86.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 80 | 0.236 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0059 | 40 | 0.235 | 326 | 84.7 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0048 | 40 | 0.190 | 326 | 84.7 | 77.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 80 | 0.120 | 1 | 8.2 | 7.4% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0084 | 10 | 0.084 | 827 | 51.7 | 47.0% | memory |
| rope_q | `rope_opt` | 0.0025 | 10 | 0.025 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 10 | 0.022 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **19.78 ms**

#### PTL — decode kv=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| lm_head | `gemm_kernel` | 5.1610 | 1 | 5.161 | 197 | 100.2 | 91.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1059 | 40 | 4.235 | 326 | 84.7 | 77.0% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 30 | 3.843 | 393 | 102.3 | 93.0% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.2695 | 10 | 2.695 | 911 | 57.0 | 51.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0527 | 40 | 2.108 | 326 | 84.7 | 77.0% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 30 | 0.959 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 10 | 0.554 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 10 | 0.462 | 363 | 94.6 | 86.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 80 | 0.236 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0059 | 40 | 0.235 | 326 | 84.7 | 77.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0208 | 10 | 0.208 | 911 | 57.0 | 51.8% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0048 | 40 | 0.190 | 326 | 84.7 | 77.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 80 | 0.120 | 1 | 8.2 | 7.4% | memory |
| rope_q | `rope_opt` | 0.0025 | 10 | 0.025 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 10 | 0.022 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **21.10 ms**

#### PTL — decode kv=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.5248 | 10 | 5.248 | 930 | 58.1 | 52.8% | memory |
| lm_head | `gemm_kernel` | 5.1610 | 1 | 5.161 | 197 | 100.2 | 91.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1059 | 40 | 4.235 | 326 | 84.7 | 77.0% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 30 | 3.843 | 393 | 102.3 | 93.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0527 | 40 | 2.108 | 326 | 84.7 | 77.0% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 30 | 0.959 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 10 | 0.554 | 379 | 98.6 | 89.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0464 | 10 | 0.464 | 930 | 58.1 | 52.8% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 10 | 0.462 | 363 | 94.6 | 86.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 80 | 0.236 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0059 | 40 | 0.235 | 326 | 84.7 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0048 | 40 | 0.190 | 326 | 84.7 | 77.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 80 | 0.120 | 1 | 8.2 | 7.4% | memory |
| rope_q | `rope_opt` | 0.0025 | 10 | 0.025 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 10 | 0.022 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **23.91 ms**

#### PTL — decode kv=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 1.0266 | 10 | 10.266 | 946 | 59.2 | 53.8% | memory |
| lm_head | `gemm_kernel` | 5.1610 | 1 | 5.161 | 197 | 100.2 | 91.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1059 | 40 | 4.235 | 326 | 84.7 | 77.0% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 30 | 3.843 | 393 | 102.3 | 93.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0527 | 40 | 2.108 | 326 | 84.7 | 77.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0990 | 10 | 0.990 | 946 | 59.2 | 53.8% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 30 | 0.959 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 10 | 0.554 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 10 | 0.462 | 363 | 94.6 | 86.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 80 | 0.236 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0059 | 40 | 0.235 | 326 | 84.7 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0048 | 40 | 0.190 | 326 | 84.7 | 77.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 80 | 0.120 | 1 | 8.2 | 7.4% | memory |
| rope_q | `rope_opt` | 0.0025 | 10 | 0.025 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 10 | 0.022 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **29.46 ms**

#### PTL — decode kv=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 2.0238 | 10 | 20.238 | 966 | 60.4 | 54.9% | memory |
| lm_head | `gemm_kernel` | 5.1610 | 1 | 5.161 | 197 | 100.2 | 91.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1059 | 40 | 4.235 | 326 | 84.7 | 77.0% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 30 | 3.843 | 393 | 102.3 | 93.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0527 | 40 | 2.108 | 326 | 84.7 | 77.0% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1915 | 10 | 1.915 | 966 | 60.4 | 54.9% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 30 | 0.959 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 10 | 0.554 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 10 | 0.462 | 363 | 94.6 | 86.0% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 80 | 0.236 | 6 | 2.8 | 2.5% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0059 | 40 | 0.235 | 326 | 84.7 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0048 | 40 | 0.190 | 326 | 84.7 | 77.0% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 80 | 0.120 | 1 | 8.2 | 7.4% | memory |
| rope_q | `rope_opt` | 0.0025 | 10 | 0.025 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 10 | 0.025 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 10 | 0.022 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **40.35 ms**

### PREFILL — per S (compute-bound uses INT8 XMX peak = 118.0 TOPS)


#### PTL — prefill S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 8.1944 | 40 | 327.776 | 5906 | 55.4 | 5.0% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 2.5503 | 30 | 76.509 | 421 | 6.6 | 0.4% | compute |
| fc_linattn_proj | `gemm_kernel` | 0.9664 | 30 | 28.992 | 50521 | 53.8 | 42.8% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 0.5557 | 40 | 22.229 | 5906 | 55.4 | 5.0% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.3165 | 40 | 12.658 | 5906 | 55.4 | 5.0% | compute |
| moe_3gemm_fused | `gemm_kernel` | 0.2762 | 40 | 11.049 | 5906 | 55.4 | 5.0% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 0.6516 | 10 | 6.516 | 24911 | 48.7 | 21.1% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0495 | 80 | 3.961 | 339 | 169.4 | 154.0% | memory |
| fc_qkv_full | `gemm_kernel` | 0.3617 | 10 | 3.617 | 51502 | 60.7 | 43.7% | compute |
| fc_o_full | `gemm_kernel` | 0.2723 | 10 | 2.723 | 44345 | 54.5 | 37.6% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.0537 | 30 | 1.612 | 50521 | 53.8 | 42.8% | compute |
| rope_q | `rope_opt` | 0.1176 | 10 | 1.176 | 357 | 142.7 | 129.7% | memory |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.1151 | 10 | 1.151 | 44345 | 54.5 | 37.6% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.0553 | 10 | 0.553 | 51502 | 60.7 | 43.7% | compute |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0381 | 10 | 0.381 | 24911 | 48.7 | 21.1% | compute |

**Total inference time (this stage)** ≈ **500.90 ms**

#### PTL — prefill S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 8.4555 | 40 | 338.221 | 9359 | 55.3 | 7.9% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 5.1562 | 30 | 154.685 | 416 | 6.5 | 0.4% | compute |
| fc_linattn_proj | `gemm_kernel` | 1.8297 | 30 | 54.892 | 53465 | 43.7 | 45.3% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 1.3721 | 40 | 54.885 | 9359 | 55.3 | 7.9% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.9208 | 40 | 36.833 | 9359 | 55.3 | 7.9% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 2.4548 | 10 | 24.549 | 27331 | 26.7 | 23.2% | compute |
| moe_3gemm_fused | `gemm_kernel` | 0.5366 | 40 | 21.465 | 9359 | 55.3 | 7.9% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.4123 | 40 | 16.492 | 9359 | 55.3 | 7.9% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.2447 | 40 | 9.788 | 9359 | 55.3 | 7.9% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0892 | 80 | 7.137 | 376 | 188.1 | 171.0% | memory |
| fc_qkv_full | `gemm_kernel` | 0.7055 | 10 | 7.055 | 52527 | 48.9 | 44.5% | compute |
| fc_o_full | `gemm_kernel` | 0.4808 | 10 | 4.808 | 49088 | 48.1 | 41.6% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.0982 | 30 | 2.947 | 53465 | 43.7 | 45.3% | compute |
| rope_q | `rope_opt` | 0.2820 | 10 | 2.820 | 297 | 119.0 | 108.2% | memory |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.2192 | 10 | 2.192 | 49088 | 48.1 | 41.6% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.1122 | 10 | 1.122 | 52527 | 48.9 | 44.5% | compute |

**Total inference time (this stage)** ≈ **739.89 ms**

#### PTL — prefill S=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 10.6281 | 40 | 425.124 | 11328 | 47.3 | 9.6% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 10.3558 | 30 | 310.675 | 415 | 6.5 | 0.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 3.8760 | 40 | 155.039 | 11328 | 47.3 | 9.6% | compute |
| fc_linattn_proj | `gemm_kernel` | 3.7142 | 30 | 111.427 | 52703 | 36.6 | 44.7% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 9.3928 | 10 | 93.928 | 28899 | 14.1 | 24.5% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 2.2736 | 40 | 90.946 | 11328 | 47.3 | 9.6% | compute |
| moe_3gemm_fused | `gemm_kernel` | 1.1716 | 40 | 46.866 | 11328 | 47.3 | 9.6% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.8871 | 40 | 35.484 | 11328 | 47.3 | 9.6% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.5401 | 40 | 21.605 | 11328 | 47.3 | 9.6% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.5233 | 40 | 20.932 | 11328 | 47.3 | 9.6% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.2372 | 80 | 18.978 | 283 | 141.4 | 128.6% | memory |
| fc_qkv_full | `gemm_kernel` | 1.4777 | 10 | 14.777 | 50081 | 40.4 | 42.5% | compute |
| fc_o_full | `gemm_kernel` | 0.9084 | 10 | 9.084 | 49667 | 42.5 | 42.1% | compute |
| rope_q | `rope_opt` | 0.6426 | 10 | 6.426 | 261 | 104.4 | 94.9% | memory |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.1975 | 30 | 5.924 | 52703 | 36.6 | 44.7% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.4752 | 10 | 4.752 | 49667 | 42.5 | 42.1% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.2375 | 10 | 2.375 | 50081 | 40.4 | 42.5% | compute |

**Total inference time (this stage)** ≈ **1374.34 ms**

#### PTL — prefill S=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 16.4004 | 40 | 656.015 | 13443 | 44.5 | 11.4% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 20.7342 | 30 | 622.027 | 414 | 6.5 | 0.4% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 40.0312 | 10 | 400.312 | 27272 | 6.7 | 23.1% | compute |
| fc_linattn_proj | `gemm_kernel` | 7.5070 | 30 | 225.209 | 51855 | 32.8 | 44.0% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 5.3591 | 40 | 214.366 | 13443 | 44.5 | 11.4% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 4.9595 | 40 | 198.378 | 13443 | 44.5 | 11.4% | compute |
| moe_3gemm_fused | `gemm_kernel` | 2.5584 | 40 | 102.338 | 13443 | 44.5 | 11.4% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 1.8611 | 40 | 74.445 | 13443 | 44.5 | 11.4% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.5637 | 80 | 45.094 | 238 | 119.1 | 108.2% | memory |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 1.1005 | 40 | 44.022 | 13443 | 44.5 | 11.4% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 1.0896 | 40 | 43.585 | 13443 | 44.5 | 11.4% | compute |
| fc_qkv_full | `gemm_kernel` | 3.1127 | 10 | 31.127 | 48179 | 35.9 | 40.8% | compute |
| fc_o_full | `gemm_kernel` | 1.7512 | 10 | 17.512 | 52861 | 42.0 | 44.8% | compute |
| rope_q | `rope_opt` | 1.3603 | 10 | 13.604 | 247 | 98.7 | 89.7% | memory |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.4444 | 30 | 13.331 | 51855 | 32.8 | 44.0% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.8488 | 10 | 8.488 | 52861 | 42.0 | 44.8% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.4532 | 10 | 4.532 | 48179 | 35.9 | 40.8% | compute |

**Total inference time (this stage)** ≈ **2714.38 ms**

#### PTL — prefill S=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 149.7208 | 10 | 1497.208 | 29271 | 3.6 | 24.8% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 41.3908 | 30 | 1241.724 | 415 | 6.5 | 0.4% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 27.9265 | 40 | 1117.058 | 15294 | 44.0 | 13.0% | compute |
| fc_linattn_proj | `gemm_kernel` | 15.6773 | 30 | 470.319 | 49339 | 29.6 | 41.8% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 10.3001 | 40 | 412.004 | 15294 | 44.0 | 13.0% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 6.6201 | 40 | 264.806 | 15294 | 44.0 | 13.0% | compute |
| moe_3gemm_fused | `gemm_kernel` | 5.0453 | 40 | 201.811 | 15294 | 44.0 | 13.0% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 3.9067 | 40 | 156.269 | 15294 | 44.0 | 13.0% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 2.4162 | 40 | 96.648 | 15294 | 44.0 | 13.0% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 2.1621 | 40 | 86.486 | 15294 | 44.0 | 13.0% | compute |
| fc_qkv_full | `gemm_kernel` | 6.0713 | 10 | 60.713 | 49719 | 35.5 | 42.1% | compute |
| fc_o_full | `gemm_kernel` | 3.5406 | 10 | 35.406 | 51401 | 39.2 | 43.6% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 1.0362 | 30 | 31.085 | 49339 | 29.6 | 41.8% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 1.8071 | 10 | 18.071 | 51401 | 39.2 | 43.6% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.8395 | 10 | 8.395 | 49719 | 35.5 | 42.1% | compute |

**Total inference time (this stage)** ≈ **5698.00 ms**

#### PTL — prefill S=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 622.9081 | 10 | 6229.082 | 28197 | 1.7 | 23.9% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 83.1275 | 30 | 2493.826 | 413 | 6.5 | 0.4% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 50.4649 | 40 | 2018.597 | 15573 | 41.4 | 13.2% | compute |
| fc_linattn_proj | `gemm_kernel` | 30.0432 | 30 | 901.297 | 51192 | 30.0 | 43.4% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 21.4412 | 40 | 857.650 | 15573 | 41.4 | 13.2% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 14.8997 | 40 | 595.989 | 15573 | 41.4 | 13.2% | compute |
| moe_3gemm_fused | `gemm_kernel` | 10.5479 | 40 | 421.917 | 15573 | 41.4 | 13.2% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 8.0531 | 40 | 322.122 | 15573 | 41.4 | 13.2% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 4.7290 | 40 | 189.158 | 15573 | 41.4 | 13.2% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 4.3826 | 40 | 175.306 | 15573 | 41.4 | 13.2% | compute |
| fc_qkv_full | `gemm_kernel` | 12.1427 | 10 | 121.427 | 48986 | 34.2 | 41.5% | compute |
| fc_o_full | `gemm_kernel` | 6.9279 | 10 | 69.279 | 52736 | 39.4 | 44.7% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 2.1740 | 30 | 65.221 | 51192 | 30.0 | 43.4% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 3.4967 | 10 | 34.967 | 52736 | 39.4 | 44.7% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 1.8857 | 10 | 18.857 | 48986 | 34.2 | 41.5% | compute |

**Total inference time (this stage)** ≈ **14514.69 ms**

#### PTL — prefill S=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 2487.6182 | 10 | 24876.182 | 28264 | 0.9 | 24.0% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 169.1589 | 30 | 5074.768 | 406 | 6.3 | 0.3% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 92.7442 | 40 | 3709.767 | 16212 | 41.3 | 13.7% | compute |
| fc_linattn_proj | `gemm_kernel` | 61.4443 | 30 | 1843.328 | 50098 | 28.9 | 42.5% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 44.8077 | 40 | 1792.308 | 16212 | 41.3 | 13.7% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 28.7470 | 40 | 1149.879 | 16212 | 41.3 | 13.7% | compute |
| moe_3gemm_fused | `gemm_kernel` | 19.2330 | 40 | 769.319 | 16212 | 41.3 | 13.7% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 16.3856 | 40 | 655.425 | 16212 | 41.3 | 13.7% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 9.0600 | 40 | 362.400 | 16212 | 41.3 | 13.7% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 8.5232 | 40 | 340.929 | 16212 | 41.3 | 13.7% | compute |
| fc_qkv_full | `gemm_kernel` | 23.5775 | 10 | 235.775 | 49892 | 34.5 | 42.3% | compute |
| fc_o_full | `gemm_kernel` | 13.6317 | 10 | 136.317 | 52991 | 39.2 | 44.9% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 4.3979 | 30 | 131.937 | 50098 | 28.9 | 42.5% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 7.1172 | 10 | 71.172 | 52991 | 39.2 | 44.9% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 3.9701 | 10 | 39.701 | 49892 | 34.5 | 42.3% | compute |

**Total inference time (this stage)** ≈ **41189.21 ms**

#### PTL — prefill S=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 11297.7628 | 10 | 112977.628 | 24905 | 0.4 | 21.1% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 356.4627 | 30 | 10693.880 | 386 | 6.0 | 0.3% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 184.8534 | 40 | 7394.135 | 15621 | 39.0 | 13.2% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 94.5860 | 40 | 3783.440 | 15621 | 39.0 | 13.2% | compute |
| fc_linattn_proj | `gemm_kernel` | 120.8693 | 30 | 3626.080 | 51280 | 29.4 | 43.5% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 67.4723 | 40 | 2698.892 | 15621 | 39.0 | 13.2% | compute |
| moe_3gemm_fused | `gemm_kernel` | 37.8035 | 40 | 1512.139 | 15621 | 39.0 | 13.2% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 33.7955 | 40 | 1351.821 | 15621 | 39.0 | 13.2% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 19.1011 | 40 | 764.044 | 15621 | 39.0 | 13.2% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 17.7980 | 40 | 711.921 | 15621 | 39.0 | 13.2% | compute |
| fc_qkv_full | `gemm_kernel` | 48.5225 | 10 | 485.225 | 49060 | 33.7 | 41.6% | compute |
| fc_o_full | `gemm_kernel` | 27.7884 | 10 | 277.884 | 52233 | 38.5 | 44.3% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 7.7780 | 30 | 233.341 | 51280 | 29.4 | 43.5% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 14.3116 | 10 | 143.116 | 52233 | 38.5 | 44.3% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 7.5069 | 10 | 75.069 | 49060 | 33.7 | 41.6% | compute |

**Total inference time (this stage)** ≈ **146728.61 ms**
