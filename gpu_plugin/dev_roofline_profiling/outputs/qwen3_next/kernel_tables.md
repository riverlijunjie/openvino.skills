# Qwen/Qwen3-Next-80B-A3B-Instruct (qwen3_next) — Per-token-size Kernel Tables


## Platform: BMG

### DECODE — per kv (M=1)


#### BMG — decode kv=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0445 | 48 | 2.135 | 868 | 225.9 | 49.5% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 36 | 1.187 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0207 | 48 | 0.992 | 868 | 225.9 | 49.5% | memory |
| lm_head | `gemm_kernel` | 0.7066 | 1 | 0.707 | 881 | 447.7 | 98.2% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 36 | 0.463 | 82 | 163.2 | 35.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0064 | 48 | 0.309 | 868 | 225.9 | 49.5% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0207 | 12 | 0.249 | 713 | 44.6 | 9.8% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 12 | 0.187 | 1349 | 351.3 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0035 | 48 | 0.170 | 868 | 225.9 | 49.5% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 12 | 0.157 | 1286 | 334.9 | 73.4% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.0015 | 48 | 0.071 | 868 | 225.9 | 49.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0015 | 12 | 0.018 | 713 | 44.6 | 9.8% | memory |
| rope_q | `rope_opt` | 0.0015 | 12 | 0.018 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.016 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 12 | 0.015 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.015 | 3 | 1.6 | 0.4% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0012 | 12 | 0.015 | 713 | 44.6 | 9.8% | memory |

**Total inference time (this stage)** ≈ **6.94 ms**

#### BMG — decode kv=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0445 | 48 | 2.135 | 868 | 225.9 | 49.5% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 36 | 1.187 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0207 | 48 | 0.992 | 868 | 225.9 | 49.5% | memory |
| lm_head | `gemm_kernel` | 0.7066 | 1 | 0.707 | 881 | 447.7 | 98.2% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0403 | 12 | 0.483 | 771 | 48.2 | 10.6% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 36 | 0.463 | 82 | 163.2 | 35.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0064 | 48 | 0.309 | 868 | 225.9 | 49.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 12 | 0.187 | 1349 | 351.3 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0035 | 48 | 0.170 | 868 | 225.9 | 49.5% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 12 | 0.157 | 1286 | 334.9 | 73.4% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.0015 | 48 | 0.071 | 868 | 225.9 | 49.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0020 | 12 | 0.024 | 771 | 48.2 | 10.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 12 | 0.018 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.016 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 12 | 0.015 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.015 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **7.16 ms**

#### BMG — decode kv=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0445 | 48 | 2.135 | 868 | 225.9 | 49.5% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 36 | 1.187 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0207 | 48 | 0.992 | 868 | 225.9 | 49.5% | memory |
| lm_head | `gemm_kernel` | 0.7066 | 1 | 0.707 | 881 | 447.7 | 98.2% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 36 | 0.463 | 82 | 163.2 | 35.8% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0333 | 12 | 0.400 | 1767 | 110.4 | 24.2% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0064 | 48 | 0.309 | 868 | 225.9 | 49.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 12 | 0.187 | 1349 | 351.3 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0035 | 48 | 0.170 | 868 | 225.9 | 49.5% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 12 | 0.157 | 1286 | 334.9 | 73.4% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.0015 | 48 | 0.071 | 868 | 225.9 | 49.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0033 | 12 | 0.040 | 1767 | 110.4 | 24.2% | memory |
| rope_q | `rope_opt` | 0.0015 | 12 | 0.018 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.016 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 12 | 0.015 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.015 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **7.10 ms**

#### BMG — decode kv=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0445 | 48 | 2.135 | 868 | 225.9 | 49.5% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 36 | 1.187 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0207 | 48 | 0.992 | 868 | 225.9 | 49.5% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0648 | 12 | 0.777 | 1865 | 116.6 | 25.6% | memory |
| lm_head | `gemm_kernel` | 0.7066 | 1 | 0.707 | 881 | 447.7 | 98.2% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 36 | 0.463 | 82 | 163.2 | 35.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0064 | 48 | 0.309 | 868 | 225.9 | 49.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 12 | 0.187 | 1349 | 351.3 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0035 | 48 | 0.170 | 868 | 225.9 | 49.5% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 12 | 0.157 | 1286 | 334.9 | 73.4% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.0015 | 48 | 0.071 | 868 | 225.9 | 49.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0055 | 12 | 0.066 | 1865 | 116.6 | 25.6% | memory |
| rope_q | `rope_opt` | 0.0015 | 12 | 0.018 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.016 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 12 | 0.015 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.015 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **7.50 ms**

#### BMG — decode kv=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0445 | 48 | 2.135 | 868 | 225.9 | 49.5% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1336 | 12 | 1.604 | 1791 | 112.0 | 24.6% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 36 | 1.187 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0207 | 48 | 0.992 | 868 | 225.9 | 49.5% | memory |
| lm_head | `gemm_kernel` | 0.7066 | 1 | 0.707 | 881 | 447.7 | 98.2% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 36 | 0.463 | 82 | 163.2 | 35.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0064 | 48 | 0.309 | 868 | 225.9 | 49.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 12 | 0.187 | 1349 | 351.3 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0035 | 48 | 0.170 | 868 | 225.9 | 49.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0141 | 12 | 0.169 | 1791 | 112.0 | 24.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 12 | 0.157 | 1286 | 334.9 | 73.4% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.0015 | 48 | 0.071 | 868 | 225.9 | 49.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 12 | 0.018 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.016 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 12 | 0.015 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.015 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **8.43 ms**

#### BMG — decode kv=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.2893 | 12 | 3.472 | 1657 | 103.6 | 22.7% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0445 | 48 | 2.135 | 868 | 225.9 | 49.5% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 36 | 1.187 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0207 | 48 | 0.992 | 868 | 225.9 | 49.5% | memory |
| lm_head | `gemm_kernel` | 0.7066 | 1 | 0.707 | 881 | 447.7 | 98.2% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 36 | 0.463 | 82 | 163.2 | 35.8% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0323 | 12 | 0.388 | 1657 | 103.6 | 22.7% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0064 | 48 | 0.309 | 868 | 225.9 | 49.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 12 | 0.187 | 1349 | 351.3 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0035 | 48 | 0.170 | 868 | 225.9 | 49.5% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 12 | 0.157 | 1286 | 334.9 | 73.4% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.0015 | 48 | 0.071 | 868 | 225.9 | 49.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 12 | 0.018 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.016 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 12 | 0.015 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.015 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **10.52 ms**

#### BMG — decode kv=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.5733 | 12 | 6.880 | 1649 | 103.0 | 22.6% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0445 | 48 | 2.135 | 868 | 225.9 | 49.5% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 36 | 1.187 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0207 | 48 | 0.992 | 868 | 225.9 | 49.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0751 | 12 | 0.902 | 1649 | 103.0 | 22.6% | memory |
| lm_head | `gemm_kernel` | 0.7066 | 1 | 0.707 | 881 | 447.7 | 98.2% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 36 | 0.463 | 82 | 163.2 | 35.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0064 | 48 | 0.309 | 868 | 225.9 | 49.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 12 | 0.187 | 1349 | 351.3 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0035 | 48 | 0.170 | 868 | 225.9 | 49.5% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 12 | 0.157 | 1286 | 334.9 | 73.4% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.0015 | 48 | 0.071 | 868 | 225.9 | 49.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 12 | 0.018 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.016 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 12 | 0.015 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.015 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **14.44 ms**

#### BMG — decode kv=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 1.1244 | 12 | 13.493 | 1663 | 103.9 | 22.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.0445 | 48 | 2.135 | 868 | 225.9 | 49.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1642 | 12 | 1.970 | 1663 | 103.9 | 22.8% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.0330 | 36 | 1.187 | 1526 | 397.3 | 87.1% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0207 | 48 | 0.992 | 868 | 225.9 | 49.5% | memory |
| lm_head | `gemm_kernel` | 0.7066 | 1 | 0.707 | 881 | 447.7 | 98.2% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0129 | 36 | 0.463 | 82 | 163.2 | 35.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0064 | 48 | 0.309 | 868 | 225.9 | 49.5% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0155 | 12 | 0.187 | 1349 | 351.3 | 77.0% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0035 | 48 | 0.170 | 868 | 225.9 | 49.5% | memory |
| fc_o_full | `gemm_kernel` | 0.0131 | 12 | 0.157 | 1286 | 334.9 | 73.4% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0015 | 96 | 0.143 | 11 | 5.5 | 1.2% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0007 | 96 | 0.072 | 3 | 16.4 | 3.6% | memory |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 0.0015 | 48 | 0.071 | 868 | 225.9 | 49.5% | memory |
| rope_q | `rope_opt` | 0.0015 | 12 | 0.018 | 27 | 10.8 | 2.4% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.016 | 25 | 12.6 | 2.8% | memory |
| rope_k | `rope_opt` | 0.0013 | 12 | 0.015 | 4 | 1.6 | 0.3% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0013 | 12 | 0.015 | 3 | 1.6 | 0.4% | memory |

**Total inference time (this stage)** ≈ **22.12 ms**

### PREFILL — per S (compute-bound uses INT8 XMX peak = 233.5 TOPS)


#### BMG — prefill S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 6.9361 | 48 | 332.932 | 8536 | 117.8 | 3.7% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 1.0665 | 36 | 38.395 | 1007 | 15.7 | 0.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 0.7433 | 48 | 35.677 | 8536 | 117.8 | 3.7% | compute |
| fc_linattn_proj | `gemm_kernel` | 0.3592 | 36 | 12.932 | 136169 | 145.1 | 58.3% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.2085 | 48 | 10.009 | 8536 | 117.8 | 3.7% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 0.4428 | 12 | 5.313 | 36899 | 72.1 | 15.8% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0188 | 96 | 1.800 | 895 | 447.3 | 98.1% | memory |
| fc_qkv_full | `gemm_kernel` | 0.1461 | 12 | 1.753 | 129811 | 153.1 | 55.6% | compute |
| fc_o_full | `gemm_kernel` | 0.1268 | 12 | 1.521 | 103405 | 127.0 | 44.3% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.0193 | 36 | 0.694 | 136169 | 145.1 | 58.3% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.0394 | 12 | 0.473 | 103405 | 127.0 | 44.3% | compute |
| rope_q | `rope_opt` | 0.0352 | 12 | 0.422 | 1193 | 477.2 | 104.6% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.0194 | 12 | 0.233 | 129811 | 153.1 | 55.6% | compute |

**Total inference time (this stage)** ≈ **442.15 ms**

#### BMG — prefill S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 6.9256 | 48 | 332.427 | 14244 | 115.7 | 6.1% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 1.7297 | 48 | 83.027 | 14244 | 115.7 | 6.1% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 2.0296 | 36 | 73.067 | 1058 | 16.5 | 0.5% | compute |
| fc_linattn_proj | `gemm_kernel` | 0.6650 | 36 | 23.938 | 146645 | 119.9 | 62.8% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.4831 | 48 | 23.187 | 14244 | 115.7 | 6.1% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 1.2564 | 12 | 15.077 | 53344 | 52.1 | 22.8% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.2777 | 48 | 13.331 | 14244 | 115.7 | 6.1% | compute |
| moe_3gemm_fused | `gemm_kernel` | 0.2198 | 48 | 10.551 | 14244 | 115.7 | 6.1% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0357 | 96 | 3.431 | 939 | 469.4 | 102.9% | memory |
| fc_qkv_full | `gemm_kernel` | 0.2725 | 12 | 3.271 | 138036 | 128.6 | 59.1% | compute |
| fc_o_full | `gemm_kernel` | 0.2173 | 12 | 2.608 | 116773 | 114.5 | 50.0% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.0380 | 36 | 1.367 | 146645 | 119.9 | 62.8% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.0769 | 12 | 0.923 | 116773 | 114.5 | 50.0% | compute |
| rope_q | `rope_opt` | 0.0604 | 12 | 0.724 | 1390 | 556.0 | 121.9% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.0386 | 12 | 0.463 | 138036 | 128.6 | 59.1% | compute |

**Total inference time (this stage)** ≈ **587.39 ms**

#### BMG — prefill S=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 7.0735 | 48 | 339.526 | 19670 | 103.9 | 8.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 4.5695 | 48 | 219.338 | 19670 | 103.9 | 8.4% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 4.1641 | 36 | 149.907 | 1031 | 16.1 | 0.4% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 4.8439 | 12 | 58.127 | 55935 | 27.3 | 24.0% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 1.1026 | 48 | 52.924 | 19670 | 103.9 | 8.4% | compute |
| fc_linattn_proj | `gemm_kernel` | 1.3101 | 36 | 47.165 | 148657 | 103.1 | 63.7% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.5546 | 48 | 26.623 | 19670 | 103.9 | 8.4% | compute |
| moe_3gemm_fused | `gemm_kernel` | 0.4270 | 48 | 20.495 | 19670 | 103.9 | 8.4% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0697 | 96 | 6.693 | 963 | 481.3 | 105.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.5297 | 12 | 6.357 | 141450 | 114.2 | 60.6% | compute |
| fc_o_full | `gemm_kernel` | 0.4017 | 12 | 4.820 | 124159 | 106.3 | 53.2% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.0767 | 36 | 2.760 | 148657 | 103.1 | 63.7% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.1518 | 12 | 1.822 | 124159 | 106.3 | 53.2% | compute |
| rope_q | `rope_opt` | 0.1076 | 12 | 1.291 | 1559 | 623.7 | 136.8% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.0775 | 12 | 0.930 | 141450 | 114.2 | 60.6% | compute |

**Total inference time (this stage)** ≈ **938.78 ms**

#### BMG — prefill S=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 11.6781 | 48 | 560.550 | 21312 | 82.3 | 9.1% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 9.2155 | 48 | 442.345 | 21312 | 82.3 | 9.1% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 8.7010 | 36 | 313.234 | 987 | 15.4 | 0.4% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 18.8892 | 12 | 226.670 | 57806 | 14.1 | 24.8% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 2.3558 | 48 | 113.079 | 21312 | 82.3 | 9.1% | compute |
| fc_linattn_proj | `gemm_kernel` | 2.5721 | 36 | 92.597 | 151209 | 95.5 | 64.8% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 1.0942 | 48 | 52.522 | 21312 | 82.3 | 9.1% | compute |
| moe_3gemm_fused | `gemm_kernel` | 0.8487 | 48 | 40.737 | 21312 | 82.3 | 9.1% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.6616 | 48 | 31.759 | 21312 | 82.3 | 9.1% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.1375 | 96 | 13.202 | 976 | 488.0 | 107.0% | memory |
| fc_qkv_full | `gemm_kernel` | 1.0496 | 12 | 12.595 | 142563 | 106.3 | 61.1% | compute |
| fc_o_full | `gemm_kernel` | 0.7880 | 12 | 9.456 | 125990 | 100.1 | 54.0% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.1547 | 36 | 5.568 | 151209 | 95.5 | 64.8% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.3029 | 12 | 3.634 | 125990 | 100.1 | 54.0% | compute |
| rope_q | `rope_opt` | 0.2015 | 12 | 2.418 | 1665 | 666.0 | 146.1% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.1555 | 12 | 1.866 | 142563 | 106.3 | 61.1% | compute |

**Total inference time (this stage)** ≈ **1922.23 ms**

#### BMG — prefill S=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 19.8722 | 48 | 953.864 | 24348 | 76.7 | 10.4% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 76.6344 | 12 | 919.613 | 57216 | 7.0 | 24.5% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 15.1669 | 48 | 728.011 | 24348 | 76.7 | 10.4% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 18.0672 | 36 | 650.418 | 951 | 14.9 | 0.4% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 4.7581 | 48 | 228.387 | 24348 | 76.7 | 10.4% | compute |
| fc_linattn_proj | `gemm_kernel` | 5.1996 | 36 | 187.185 | 149637 | 89.9 | 64.1% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 2.2106 | 48 | 106.108 | 24348 | 76.7 | 10.4% | compute |
| moe_3gemm_fused | `gemm_kernel` | 1.6205 | 48 | 77.785 | 24348 | 76.7 | 10.4% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 1.3567 | 48 | 65.122 | 24348 | 76.7 | 10.4% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.2743 | 96 | 26.334 | 979 | 489.3 | 107.3% | memory |
| fc_qkv_full | `gemm_kernel` | 2.0779 | 12 | 24.934 | 143942 | 102.9 | 61.7% | compute |
| fc_o_full | `gemm_kernel` | 1.5569 | 12 | 18.682 | 127521 | 97.4 | 54.6% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.3113 | 36 | 11.207 | 149637 | 89.9 | 64.1% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.5987 | 12 | 7.184 | 127521 | 97.4 | 54.6% | compute |
| rope_q | `rope_opt` | 0.3956 | 12 | 4.748 | 1696 | 678.5 | 148.8% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.3092 | 12 | 3.710 | 143942 | 102.9 | 61.7% | compute |

**Total inference time (this stage)** ≈ **4013.29 ms**

#### BMG — prefill S=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 76.8044 | 48 | 3686.611 | 17957 | 50.2 | 7.7% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 301.9960 | 12 | 3623.952 | 58169 | 3.6 | 24.9% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 38.8448 | 36 | 1398.414 | 885 | 13.8 | 0.4% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 26.5878 | 48 | 1276.212 | 17957 | 50.2 | 7.7% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 9.4599 | 48 | 454.076 | 17957 | 50.2 | 7.7% | compute |
| fc_linattn_proj | `gemm_kernel` | 10.3285 | 36 | 371.826 | 150622 | 88.1 | 64.5% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 4.4185 | 48 | 212.087 | 17957 | 50.2 | 7.7% | compute |
| moe_3gemm_fused | `gemm_kernel` | 3.0553 | 48 | 146.652 | 17957 | 50.2 | 7.7% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.5469 | 96 | 52.504 | 982 | 490.8 | 107.6% | memory |
| fc_qkv_full | `gemm_kernel` | 4.1417 | 12 | 49.700 | 144247 | 100.8 | 61.8% | compute |
| fc_o_full | `gemm_kernel` | 3.0752 | 12 | 36.903 | 128376 | 96.0 | 55.0% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.6212 | 36 | 22.363 | 150622 | 88.1 | 64.5% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 1.2072 | 12 | 14.486 | 128376 | 96.0 | 55.0% | compute |
| rope_q | `rope_opt` | 0.7864 | 12 | 9.437 | 1707 | 682.7 | 149.7% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.6223 | 12 | 7.468 | 144247 | 100.8 | 61.8% | compute |

**Total inference time (this stage)** ≈ **11362.69 ms**

#### BMG — prefill S=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 314.0420 | 48 | 15074.018 | 11070 | 29.0 | 4.7% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 1203.8112 | 12 | 14445.734 | 58415 | 1.8 | 25.0% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 81.5412 | 36 | 2935.485 | 843 | 13.2 | 0.4% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 50.0767 | 48 | 2403.682 | 11070 | 29.0 | 4.7% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 18.6905 | 48 | 897.145 | 11070 | 29.0 | 4.7% | compute |
| fc_linattn_proj | `gemm_kernel` | 20.7828 | 36 | 748.180 | 149669 | 86.4 | 64.1% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 8.8263 | 48 | 423.662 | 11070 | 29.0 | 4.7% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 1.0892 | 96 | 104.560 | 986 | 492.9 | 108.1% | memory |
| fc_qkv_full | `gemm_kernel` | 8.2753 | 12 | 99.303 | 144373 | 99.8 | 61.8% | compute |
| fc_o_full | `gemm_kernel` | 6.1210 | 12 | 73.452 | 128950 | 95.4 | 55.2% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 1.2561 | 36 | 45.219 | 149669 | 86.4 | 64.1% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 2.4056 | 12 | 28.867 | 128950 | 95.4 | 55.2% | compute |
| rope_q | `rope_opt` | 1.5677 | 12 | 18.813 | 1712 | 684.9 | 150.2% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 1.2445 | 12 | 14.934 | 144373 | 99.8 | 61.8% | compute |

**Total inference time (this stage)** ≈ **37313.05 ms**

#### BMG — prefill S=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 4813.2497 | 12 | 57758.997 | 58459 | 0.9 | 25.0% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 188.9804 | 36 | 6803.294 | 727 | 11.4 | 0.3% | compute |
| fc_linattn_proj | `gemm_kernel` | 41.2725 | 36 | 1485.812 | 150556 | 86.3 | 64.5% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 2.1734 | 96 | 208.649 | 988 | 494.0 | 108.3% | memory |
| fc_qkv_full | `gemm_kernel` | 16.6086 | 12 | 199.303 | 143847 | 98.9 | 61.6% | compute |
| fc_o_full | `gemm_kernel` | 12.2761 | 12 | 147.314 | 128355 | 94.5 | 55.0% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 2.5456 | 36 | 91.641 | 150556 | 86.3 | 64.5% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 4.8562 | 12 | 58.274 | 128355 | 94.5 | 55.0% | compute |
| rope_q | `rope_opt` | 3.1269 | 12 | 37.523 | 1717 | 686.8 | 150.6% | memory |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 2.5005 | 12 | 30.006 | 143847 | 98.9 | 61.6% | compute |
| moe_3gemm_fused | `reorder_data_0_0` | 0.0217 | 48 | 1.040 | 418673079 | 1059320.2 | 179324.7% | compute |

**Total inference time (this stage)** ≈ **66821.85 ms**

## Platform: PTL

### DECODE — per kv (M=1)


#### PTL — decode kv=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1282 | 48 | 6.154 | 320 | 83.3 | 75.8% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 36 | 4.612 | 393 | 102.3 | 93.0% | memory |
| lm_head | `gemm_kernel` | 3.2887 | 1 | 3.289 | 189 | 96.2 | 87.4% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0639 | 48 | 3.067 | 320 | 83.3 | 75.8% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 36 | 1.151 | 33 | 65.6 | 59.6% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.0571 | 12 | 0.685 | 260 | 16.3 | 14.8% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 12 | 0.665 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 12 | 0.554 | 363 | 94.6 | 86.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0093 | 48 | 0.446 | 320 | 83.3 | 75.8% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0080 | 48 | 0.383 | 320 | 83.3 | 75.8% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0042 | 12 | 0.050 | 260 | 16.3 | 14.8% | memory |
| rope_q | `rope_opt` | 0.0025 | 12 | 0.030 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.030 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.029 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 12 | 0.027 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **21.60 ms**

#### PTL — decode kv=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1282 | 48 | 6.154 | 320 | 83.3 | 75.8% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 36 | 4.612 | 393 | 102.3 | 93.0% | memory |
| lm_head | `gemm_kernel` | 3.2887 | 1 | 3.289 | 189 | 96.2 | 87.4% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0639 | 48 | 3.067 | 320 | 83.3 | 75.8% | memory |
| paged_attention | `paged_attention_opt_single_token_sa` | 0.1029 | 12 | 1.235 | 303 | 18.9 | 17.2% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 36 | 1.151 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 12 | 0.665 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 12 | 0.554 | 363 | 94.6 | 86.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0093 | 48 | 0.446 | 320 | 83.3 | 75.8% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0080 | 48 | 0.383 | 320 | 83.3 | 75.8% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0040 | 12 | 0.048 | 303 | 18.9 | 17.2% | memory |
| rope_q | `rope_opt` | 0.0025 | 12 | 0.030 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.030 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.029 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 12 | 0.027 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **22.15 ms**

#### PTL — decode kv=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1282 | 48 | 6.154 | 320 | 83.3 | 75.8% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 36 | 4.612 | 393 | 102.3 | 93.0% | memory |
| lm_head | `gemm_kernel` | 3.2887 | 1 | 3.289 | 189 | 96.2 | 87.4% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0639 | 48 | 3.067 | 320 | 83.3 | 75.8% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 36 | 1.151 | 33 | 65.6 | 59.6% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.0785 | 12 | 0.942 | 764 | 47.8 | 43.4% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 12 | 0.665 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 12 | 0.554 | 363 | 94.6 | 86.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0093 | 48 | 0.446 | 320 | 83.3 | 75.8% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0080 | 48 | 0.383 | 320 | 83.3 | 75.8% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0053 | 12 | 0.064 | 764 | 47.8 | 43.4% | memory |
| rope_q | `rope_opt` | 0.0025 | 12 | 0.030 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.030 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.029 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 12 | 0.027 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **21.87 ms**

#### PTL — decode kv=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1282 | 48 | 6.154 | 320 | 83.3 | 75.8% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 36 | 4.612 | 393 | 102.3 | 93.0% | memory |
| lm_head | `gemm_kernel` | 3.2887 | 1 | 3.289 | 189 | 96.2 | 87.4% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0639 | 48 | 3.067 | 320 | 83.3 | 75.8% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.1498 | 12 | 1.797 | 827 | 51.7 | 47.0% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 36 | 1.151 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 12 | 0.665 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 12 | 0.554 | 363 | 94.6 | 86.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0093 | 48 | 0.446 | 320 | 83.3 | 75.8% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0080 | 48 | 0.383 | 320 | 83.3 | 75.8% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0084 | 12 | 0.100 | 827 | 51.7 | 47.0% | memory |
| rope_q | `rope_opt` | 0.0025 | 12 | 0.030 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.030 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.029 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 12 | 0.027 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **22.76 ms**

#### PTL — decode kv=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1282 | 48 | 6.154 | 320 | 83.3 | 75.8% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 36 | 4.612 | 393 | 102.3 | 93.0% | memory |
| lm_head | `gemm_kernel` | 3.2887 | 1 | 3.289 | 189 | 96.2 | 87.4% | memory |
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.2695 | 12 | 3.234 | 911 | 57.0 | 51.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0639 | 48 | 3.067 | 320 | 83.3 | 75.8% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 36 | 1.151 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 12 | 0.665 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 12 | 0.554 | 363 | 94.6 | 86.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0093 | 48 | 0.446 | 320 | 83.3 | 75.8% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0080 | 48 | 0.383 | 320 | 83.3 | 75.8% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0208 | 12 | 0.250 | 911 | 57.0 | 51.8% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| rope_q | `rope_opt` | 0.0025 | 12 | 0.030 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.030 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.029 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 12 | 0.027 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **24.35 ms**

#### PTL — decode kv=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 0.5248 | 12 | 6.298 | 930 | 58.1 | 52.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1282 | 48 | 6.154 | 320 | 83.3 | 75.8% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 36 | 4.612 | 393 | 102.3 | 93.0% | memory |
| lm_head | `gemm_kernel` | 3.2887 | 1 | 3.289 | 189 | 96.2 | 87.4% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0639 | 48 | 3.067 | 320 | 83.3 | 75.8% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 36 | 1.151 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 12 | 0.665 | 379 | 98.6 | 89.6% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0464 | 12 | 0.557 | 930 | 58.1 | 52.8% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 12 | 0.554 | 363 | 94.6 | 86.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0093 | 48 | 0.446 | 320 | 83.3 | 75.8% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0080 | 48 | 0.383 | 320 | 83.3 | 75.8% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| rope_q | `rope_opt` | 0.0025 | 12 | 0.030 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.030 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.029 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 12 | 0.027 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **27.72 ms**

#### PTL — decode kv=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 1.0266 | 12 | 12.319 | 946 | 59.2 | 53.8% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1282 | 48 | 6.154 | 320 | 83.3 | 75.8% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 36 | 4.612 | 393 | 102.3 | 93.0% | memory |
| lm_head | `gemm_kernel` | 3.2887 | 1 | 3.289 | 189 | 96.2 | 87.4% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0639 | 48 | 3.067 | 320 | 83.3 | 75.8% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.0990 | 12 | 1.188 | 946 | 59.2 | 53.8% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 36 | 1.151 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 12 | 0.665 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 12 | 0.554 | 363 | 94.6 | 86.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0093 | 48 | 0.446 | 320 | 83.3 | 75.8% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0080 | 48 | 0.383 | 320 | 83.3 | 75.8% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| rope_q | `rope_opt` | 0.0025 | 12 | 0.030 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.030 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.029 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 12 | 0.027 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **34.37 ms**

#### PTL — decode kv=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `paged_attention_opt_gqa_single_token_sa` | 2.0238 | 12 | 24.285 | 966 | 60.4 | 54.9% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_gate_up` | 0.1282 | 48 | 6.154 | 320 | 83.3 | 75.8% | memory |
| fc_linattn_proj | `gemm_kernel` | 0.1281 | 36 | 4.612 | 393 | 102.3 | 93.0% | memory |
| lm_head | `gemm_kernel` | 3.2887 | 1 | 3.289 | 189 | 96.2 | 87.4% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_mlp_down` | 0.0639 | 48 | 3.067 | 320 | 83.3 | 75.8% | memory |
| paged_attention | `paged_attention_opt_single_token_finalization_sa` | 0.1915 | 12 | 2.298 | 966 | 60.4 | 54.9% | memory |
| gated_delta_net | `gated_delta_net_ref_sa` | 0.0320 | 36 | 1.151 | 33 | 65.6 | 59.6% | memory |
| fc_qkv_full | `gemm_kernel` | 0.0554 | 12 | 0.665 | 379 | 98.6 | 89.6% | memory |
| fc_o_full | `gemm_kernel` | 0.0462 | 12 | 0.554 | 363 | 94.6 | 86.0% | memory |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.0093 | 48 | 0.446 | 320 | 83.3 | 75.8% | memory |
| moe_3gemm_fused | `gemm_kernel` | 0.0080 | 48 | 0.383 | 320 | 83.3 | 75.8% | memory |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0029 | 96 | 0.283 | 6 | 2.8 | 2.5% | memory |
| add | `eltwise_simple_vload8_0_0` | 0.0015 | 96 | 0.144 | 1 | 8.2 | 7.4% | memory |
| rope_q | `rope_opt` | 0.0025 | 12 | 0.030 | 16 | 6.5 | 5.9% | memory |
| q_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.030 | 13 | 6.5 | 5.9% | memory |
| k_norm | `rms_gpu_bfyx_opt_0_0` | 0.0025 | 12 | 0.029 | 2 | 0.8 | 0.8% | memory |
| rope_k | `rope_opt` | 0.0022 | 12 | 0.027 | 2 | 0.9 | 0.8% | memory |

**Total inference time (this stage)** ≈ **47.45 ms**

### PREFILL — per S (compute-bound uses INT8 XMX peak = 118.0 TOPS)


#### PTL — prefill S=1024

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 15.0257 | 48 | 721.233 | 4095 | 56.5 | 3.5% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 2.5503 | 36 | 91.811 | 421 | 6.6 | 0.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 0.7727 | 48 | 37.091 | 4095 | 56.5 | 3.5% | compute |
| fc_linattn_proj | `gemm_kernel` | 0.9664 | 36 | 34.791 | 50521 | 53.8 | 42.8% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 0.4320 | 48 | 20.737 | 4095 | 56.5 | 3.5% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.3174 | 48 | 15.234 | 4095 | 56.5 | 3.5% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 0.6516 | 12 | 7.819 | 24911 | 48.7 | 21.1% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0495 | 96 | 4.754 | 339 | 169.4 | 154.0% | memory |
| fc_qkv_full | `gemm_kernel` | 0.3617 | 12 | 4.341 | 51502 | 60.7 | 43.7% | compute |
| fc_o_full | `gemm_kernel` | 0.2723 | 12 | 3.268 | 44345 | 54.5 | 37.6% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.0537 | 36 | 1.935 | 50521 | 53.8 | 42.8% | compute |
| rope_q | `rope_opt` | 0.1176 | 12 | 1.411 | 357 | 142.7 | 129.7% | memory |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.1151 | 12 | 1.381 | 44345 | 54.5 | 37.6% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.0553 | 12 | 0.663 | 51502 | 60.7 | 43.7% | compute |
| paged_attention | `pa_kv_cache_update_ref_sa` | 0.0381 | 12 | 0.457 | 24911 | 48.7 | 21.1% | compute |

**Total inference time (this stage)** ≈ **946.92 ms**

#### PTL — prefill S=2048

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 15.2800 | 48 | 733.439 | 6853 | 55.7 | 5.8% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 5.1562 | 36 | 185.622 | 416 | 6.5 | 0.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 1.8402 | 48 | 88.331 | 6853 | 55.7 | 5.8% | compute |
| fc_linattn_proj | `gemm_kernel` | 1.8297 | 36 | 65.871 | 53465 | 43.7 | 45.3% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 1.2232 | 48 | 58.712 | 6853 | 55.7 | 5.8% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 0.6170 | 48 | 29.615 | 6853 | 55.7 | 5.8% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 2.4548 | 12 | 29.458 | 27331 | 26.7 | 23.2% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 0.5754 | 48 | 27.620 | 6853 | 55.7 | 5.8% | compute |
| moe_3gemm_fused | `gemm_kernel` | 0.5738 | 48 | 27.544 | 6853 | 55.7 | 5.8% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.0892 | 96 | 8.565 | 376 | 188.1 | 171.0% | memory |
| fc_qkv_full | `gemm_kernel` | 0.7055 | 12 | 8.466 | 52527 | 48.9 | 44.5% | compute |
| fc_o_full | `gemm_kernel` | 0.4808 | 12 | 5.770 | 49088 | 48.1 | 41.6% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.0982 | 36 | 3.536 | 53465 | 43.7 | 45.3% | compute |
| rope_q | `rope_opt` | 0.2820 | 12 | 3.384 | 297 | 119.0 | 108.2% | memory |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.2192 | 12 | 2.630 | 49088 | 48.1 | 41.6% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.1122 | 12 | 1.346 | 52527 | 48.9 | 44.5% | compute |

**Total inference time (this stage)** ≈ **1279.91 ms**

#### PTL — prefill S=4096

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 16.8756 | 48 | 810.030 | 9676 | 51.1 | 8.2% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 10.3558 | 36 | 372.810 | 415 | 6.5 | 0.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 4.4677 | 48 | 214.450 | 9676 | 51.1 | 8.2% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 2.9621 | 48 | 142.179 | 9676 | 51.1 | 8.2% | compute |
| fc_linattn_proj | `gemm_kernel` | 3.7142 | 36 | 133.712 | 52703 | 36.6 | 44.7% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 9.3928 | 12 | 112.713 | 28899 | 14.1 | 24.5% | compute |
| moe_3gemm_fused | `gemm_kernel` | 1.2269 | 48 | 58.890 | 9676 | 51.1 | 8.2% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 1.2179 | 48 | 58.457 | 9676 | 51.1 | 8.2% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 1.2007 | 48 | 57.635 | 9676 | 51.1 | 8.2% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.2372 | 96 | 22.774 | 283 | 141.4 | 128.6% | memory |
| fc_qkv_full | `gemm_kernel` | 1.4777 | 12 | 17.733 | 50081 | 40.4 | 42.5% | compute |
| fc_o_full | `gemm_kernel` | 0.9084 | 12 | 10.901 | 49667 | 42.5 | 42.1% | compute |
| rope_q | `rope_opt` | 0.6426 | 12 | 7.711 | 261 | 104.4 | 94.9% | memory |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.1975 | 36 | 7.108 | 52703 | 36.6 | 44.7% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.4752 | 12 | 5.703 | 49667 | 42.5 | 42.1% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.2375 | 12 | 2.850 | 50081 | 40.4 | 42.5% | compute |

**Total inference time (this stage)** ≈ **2035.65 ms**

#### PTL — prefill S=8192

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 23.9483 | 48 | 1149.517 | 11021 | 42.6 | 9.3% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 20.7342 | 36 | 746.432 | 414 | 6.5 | 0.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 10.9867 | 48 | 527.362 | 11021 | 42.6 | 9.3% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 40.0312 | 12 | 480.374 | 27272 | 6.7 | 23.1% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 6.2414 | 48 | 299.585 | 11021 | 42.6 | 9.3% | compute |
| fc_linattn_proj | `gemm_kernel` | 7.5070 | 36 | 270.250 | 51855 | 32.8 | 44.0% | compute |
| moe_3gemm_fused | `gemm_kernel` | 2.6134 | 48 | 125.444 | 11021 | 42.6 | 9.3% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 2.4603 | 48 | 118.093 | 11021 | 42.6 | 9.3% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 2.4139 | 48 | 115.868 | 11021 | 42.6 | 9.3% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 1.1359 | 48 | 54.523 | 11021 | 42.6 | 9.3% | compute |
| rmsnorm | `rms_gpu_bfyx_opt_0_0` | 0.5637 | 96 | 54.113 | 238 | 119.1 | 108.2% | memory |
| fc_qkv_full | `gemm_kernel` | 3.1127 | 12 | 37.352 | 48179 | 35.9 | 40.8% | compute |
| fc_o_full | `gemm_kernel` | 1.7512 | 12 | 21.014 | 52861 | 42.0 | 44.8% | compute |
| rope_q | `rope_opt` | 1.3603 | 12 | 16.324 | 247 | 98.7 | 89.7% | memory |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 0.4444 | 36 | 15.997 | 51855 | 32.8 | 44.0% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 0.8488 | 12 | 10.186 | 52861 | 42.0 | 44.8% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.4532 | 12 | 5.438 | 48179 | 35.9 | 40.8% | compute |

**Total inference time (this stage)** ≈ **4047.87 ms**

#### PTL — prefill S=16384

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| moe_3gemm_fused | `grouped_micro_gemm` | 41.4734 | 48 | 1990.724 | 12803 | 40.3 | 10.9% | compute |
| paged_attention | `sdpa_micro_prefill_sa` | 149.7208 | 12 | 1796.650 | 29271 | 3.6 | 24.8% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 41.3908 | 36 | 1490.069 | 415 | 6.5 | 0.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 13.4627 | 48 | 646.211 | 12803 | 40.3 | 10.9% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 13.0489 | 48 | 626.346 | 12803 | 40.3 | 10.9% | compute |
| fc_linattn_proj | `gemm_kernel` | 15.6773 | 36 | 564.383 | 49339 | 29.6 | 41.8% | compute |
| moe_3gemm_fused | `gemm_kernel` | 5.2323 | 48 | 251.152 | 12803 | 40.3 | 10.9% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 4.9830 | 48 | 239.183 | 12803 | 40.3 | 10.9% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 4.7987 | 48 | 230.339 | 12803 | 40.3 | 10.9% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 2.4104 | 48 | 115.700 | 12803 | 40.3 | 10.9% | compute |
| fc_qkv_full | `gemm_kernel` | 6.0713 | 12 | 72.856 | 49719 | 35.5 | 42.1% | compute |
| fc_o_full | `gemm_kernel` | 3.5406 | 12 | 42.487 | 51401 | 39.2 | 43.6% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 1.0362 | 36 | 37.302 | 49339 | 29.6 | 41.8% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 1.8071 | 12 | 21.685 | 51401 | 39.2 | 43.6% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 0.8395 | 12 | 10.074 | 49719 | 35.5 | 42.1% | compute |

**Total inference time (this stage)** ≈ **8135.16 ms**

#### PTL — prefill S=32768

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 622.9081 | 12 | 7474.898 | 28197 | 1.7 | 23.9% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 71.5564 | 48 | 3434.707 | 13877 | 38.8 | 11.8% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 83.1275 | 36 | 2992.591 | 413 | 6.5 | 0.4% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 28.1684 | 48 | 1352.085 | 13877 | 38.8 | 11.8% | compute |
| fc_linattn_proj | `gemm_kernel` | 30.0432 | 36 | 1081.557 | 51192 | 30.0 | 43.4% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 22.3770 | 48 | 1074.095 | 13877 | 38.8 | 11.8% | compute |
| moe_3gemm_fused | `gemm_kernel` | 10.3325 | 48 | 495.962 | 13877 | 38.8 | 11.8% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 10.1094 | 48 | 485.252 | 13877 | 38.8 | 11.8% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 9.5982 | 48 | 460.713 | 13877 | 38.8 | 11.8% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 4.7915 | 48 | 229.990 | 13877 | 38.8 | 11.8% | compute |
| fc_qkv_full | `gemm_kernel` | 12.1427 | 12 | 145.712 | 48986 | 34.2 | 41.5% | compute |
| fc_o_full | `gemm_kernel` | 6.9279 | 12 | 83.135 | 52736 | 39.4 | 44.7% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 2.1740 | 36 | 78.265 | 51192 | 30.0 | 43.4% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 3.4967 | 12 | 41.960 | 52736 | 39.4 | 44.7% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 1.8857 | 12 | 22.629 | 48986 | 34.2 | 41.5% | compute |

**Total inference time (this stage)** ≈ **19453.55 ms**

#### PTL — prefill S=65536

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 2487.6182 | 12 | 29851.418 | 28264 | 0.9 | 24.0% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 130.9159 | 48 | 6283.965 | 14396 | 37.7 | 12.2% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 169.1589 | 36 | 6089.722 | 406 | 6.3 | 0.3% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 55.9287 | 48 | 2684.578 | 14396 | 37.7 | 12.2% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 46.1889 | 48 | 2217.065 | 14396 | 37.7 | 12.2% | compute |
| fc_linattn_proj | `gemm_kernel` | 61.4443 | 36 | 2211.993 | 50098 | 28.9 | 42.5% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 20.6619 | 48 | 991.772 | 14396 | 37.7 | 12.2% | compute |
| moe_3gemm_fused | `gemm_kernel` | 19.6917 | 48 | 945.202 | 14396 | 37.7 | 12.2% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 19.1930 | 48 | 921.265 | 14396 | 37.7 | 12.2% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 9.4974 | 48 | 455.873 | 14396 | 37.7 | 12.2% | compute |
| fc_qkv_full | `gemm_kernel` | 23.5775 | 12 | 282.930 | 49892 | 34.5 | 42.3% | compute |
| fc_o_full | `gemm_kernel` | 13.6317 | 12 | 163.581 | 52991 | 39.2 | 44.9% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 4.3979 | 36 | 158.325 | 50098 | 28.9 | 42.5% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 7.1172 | 12 | 85.406 | 52991 | 39.2 | 44.9% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 3.9701 | 12 | 47.641 | 49892 | 34.5 | 42.3% | compute |

**Total inference time (this stage)** ≈ **53390.73 ms**

#### PTL — prefill S=131072

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| paged_attention | `sdpa_micro_prefill_sa` | 11297.7628 | 12 | 135573.153 | 24905 | 0.4 | 21.1% | compute |
| gated_delta_net | `gated_delta_net_ref_sa` | 356.4627 | 36 | 12832.656 | 386 | 6.0 | 0.3% | compute |
| moe_3gemm_fused | `grouped_micro_gemm` | 245.4055 | 48 | 11779.465 | 14430 | 36.5 | 12.2% | compute |
| moe_3gemm_fused | `moe_gather_ref_prefill_gather` | 118.1317 | 48 | 5670.320 | 14430 | 36.5 | 12.2% | compute |
| moe_3gemm_fused | `moe_scatter_reduction_opt_moe_scatter_reduction_ref` | 100.0781 | 48 | 4803.747 | 14430 | 36.5 | 12.2% | compute |
| fc_linattn_proj | `gemm_kernel` | 120.8693 | 36 | 4351.296 | 51280 | 29.4 | 43.5% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_prefill_swiglu` | 42.0664 | 48 | 2019.185 | 14430 | 36.5 | 12.2% | compute |
| moe_3gemm_fused | `moe_3gemm_swiglu_fuse_softmax_topk` | 38.3740 | 48 | 1841.952 | 14430 | 36.5 | 12.2% | compute |
| moe_3gemm_fused | `gemm_kernel` | 38.2456 | 48 | 1835.790 | 14430 | 36.5 | 12.2% | compute |
| moe_3gemm_fused | `reorder_data_fast_b1_0_0` | 20.1147 | 48 | 965.507 | 14430 | 36.5 | 12.2% | compute |
| fc_qkv_full | `gemm_kernel` | 48.5225 | 12 | 582.270 | 49060 | 33.7 | 41.6% | compute |
| fc_o_full | `gemm_kernel` | 27.7884 | 12 | 333.461 | 52233 | 38.5 | 44.3% | compute |
| fc_linattn_proj | `dynamic_quantize_gpu_opt_0_0` | 7.7780 | 36 | 280.009 | 51280 | 29.4 | 43.5% | compute |
| fc_o_full | `dynamic_quantize_gpu_opt_0_0` | 14.3116 | 12 | 171.739 | 52233 | 38.5 | 44.3% | compute |
| fc_qkv_full | `dynamic_quantize_gpu_opt_0_0` | 7.5069 | 12 | 90.082 | 49060 | 33.7 | 41.6% | compute |

**Total inference time (this stage)** ≈ **183130.63 ms**
