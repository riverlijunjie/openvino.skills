# SeedVR2 (VAE encoder + windowed-attention DiT + VAE decoder, video SR/restoration) -- 2560x1536 variant — Per-submodel Kernel Tables

## Platform: PTL

### VAE_ENCODER

#### PTL — vae_encoder stage=vae_encoder

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| down0.conv1 | `gen_conv` | 342.1681 | 116 | 39691.495 | 26266 | 19.0 | 44.5% | compute |
| down1.conv2 | `gen_conv` | 358.0143 | 87 | 31147.242 | 32949 | 11.9 | 55.9% | compute |
| down0.conv1 | `gen_reorder` | 187.6227 | 116 | 21764.229 | 26266 | 19.0 | 44.5% | compute |
| mid_attn_sdpa | `sdpa_micro_prefill_sa` | 547.0381 | 29 | 15864.106 | 14305 | 0.5 | 24.3% | compute |
| down2.conv2 | `gen_conv` | 170.2357 | 87 | 14810.507 | 36829 | 8.1 | 62.4% | compute |
| gn_c128_h1536w2560 | `group_normalization_zsplit_final_zsplit_plain` | 80.8697 | 145 | 11726.106 | 131 | 65.4 | 62.3% | memory |
| gn_c128_h1536w2560 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 42.2773 | 145 | 6130.215 | 131 | 65.4 | 62.3% | memory |
| down1.conv2 | `gen_reorder` | 64.3250 | 87 | 5596.278 | 32949 | 11.9 | 55.9% | compute |
| down1.conv1 | `gen_conv` | 166.8919 | 29 | 4839.866 | 32298 | 16.4 | 54.8% | compute |
| gn_c256_h768w1280 | `group_normalization_zsplit_final_zsplit_plain` | 40.1402 | 116 | 4656.262 | 131 | 65.7 | 62.6% | memory |
| down3_mid.conv1 | `gen_conv` | 28.4550 | 116 | 3300.784 | 27751 | 8.5 | 47.1% | compute |
| down2.conv1 | `gen_conv` | 85.3090 | 29 | 2473.961 | 34724 | 10.1 | 58.9% | compute |
| gn_c256_h768w1280 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 21.1598 | 116 | 2454.534 | 131 | 65.7 | 62.6% | memory |
| down0.downsampler | `gen_reorder` | 63.6472 | 29 | 1845.768 | 13331 | 57.9 | 22.6% | compute |
| conv_in | `gen_reorder` | 63.6341 | 29 | 1845.390 | 3018 | 38.6 | 5.1% | compute |
| down2.conv2 | `gen_reorder` | 18.6868 | 87 | 1625.750 | 36829 | 8.1 | 62.4% | compute |
| down1.conv1 | `gen_reorder` | 48.5335 | 29 | 1407.470 | 32298 | 16.4 | 54.8% | compute |
| down1.downsampler | `gen_conv` | 44.8610 | 29 | 1300.970 | 22803 | 36.3 | 38.7% | compute |
| down1.shortcut | `gen_reorder` | 42.5303 | 29 | 1233.380 | 5154 | 60.4 | 8.7% | compute |
| conv_in | `gen_conv` | 42.2284 | 29 | 1224.625 | 3018 | 38.6 | 5.1% | compute |
| gn_c512_h384w640 | `group_normalization_zsplit_final_zsplit_plain` | 10.4357 | 116 | 1210.543 | 124 | 62.1 | 59.1% | memory |
| down1.downsampler | `gen_reorder` | 31.4205 | 29 | 911.194 | 22803 | 36.3 | 38.7% | compute |
| down2.downsampler | `gen_conv` | 26.6895 | 29 | 773.997 | 24266 | 23.2 | 41.1% | compute |
| down0.downsampler | `gen_conv` | 23.3409 | 29 | 676.885 | 13331 | 57.9 | 22.6% | compute |
| gn_c512_h384w640 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 5.7851 | 116 | 671.077 | 124 | 62.1 | 59.1% | memory |
| down2.conv1 | `gen_reorder` | 14.8795 | 29 | 431.504 | 34724 | 10.1 | 58.9% | compute |
| gn_c512_h192w320 | `group_normalization_zsplit_final_zsplit_plain` | 1.1681 | 290 | 338.758 | 137 | 68.5 | 65.3% | memory |
| down3_mid.conv1 | `gen_reorder` | 2.8852 | 116 | 334.685 | 27751 | 8.5 | 47.1% | compute |
| down2.shortcut | `gen_reorder` | 10.3682 | 29 | 300.679 | 9885 | 57.9 | 16.8% | compute |
| down2.downsampler | `gen_reorder` | 9.1514 | 29 | 265.391 | 24266 | 23.2 | 41.1% | compute |
| down1.shortcut | `gen_conv` | 7.4657 | 29 | 216.505 | 5154 | 60.4 | 8.7% | compute |
| mid_attn_qkvo | `gemm_kernel` | 1.6710 | 116 | 193.833 | 19277 | 75.6 | 32.7% | compute |
| gn_c512_h192w320 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 0.6666 | 290 | 193.314 | 137 | 68.5 | 65.3% | memory |
| down2.shortcut | `gen_conv` | 2.6671 | 29 | 77.347 | 9885 | 57.9 | 16.8% | compute |
| conv_out | `gen_conv` | 2.0928 | 29 | 60.691 | 13040 | 46.4 | 22.1% | compute |
| conv_out | `gen_reorder` | 2.0758 | 29 | 60.198 | 13040 | 46.4 | 22.1% | compute |

**Total inference time (this stage)** ≈ **181655.54 ms**

### DIT

#### PTL — dit stage=dit

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vid_qkv | `gemm_kernel` | 1324.3537 | 32 | 42379.319 | 14138 | 7.4 | 24.0% | compute |
| vid_mlp_out | `gemm_kernel` | 1151.3068 | 32 | 36841.817 | 14636 | 7.9 | 24.8% | compute |
| vid_mlp_up | `gemm_kernel` | 891.1395 | 32 | 28516.465 | 18910 | 10.2 | 32.1% | compute |
| vid_mlp_gate | `gemm_kernel` | 746.9422 | 32 | 23902.151 | 22560 | 12.1 | 38.2% | compute |
| wattn_window_a | `cm_sdpa_vlen_sa` | 657.4146 | 16 | 10518.634 | 19690 | 14.8 | 33.4% | compute |
| wattn_window_b | `cm_sdpa_vlen_sa` | 454.2602 | 16 | 7268.163 | 18997 | 21.5 | 32.2% | compute |
| vid_attn_out | `gemm_kernel` | 204.0696 | 32 | 6530.228 | 30583 | 24.0 | 51.9% | compute |
| vid_in | `gemm_kernel` | 67.0835 | 1 | 67.083 | 4797 | 38.2 | 8.1% | compute |
| vid_out | `gemm_kernel` | 26.2369 | 1 | 26.237 | 5947 | 95.3 | 10.1% | compute |
| txt_qkv | `gemm_kernel` | 0.4413 | 32 | 14.123 | 5168 | 91.8 | 8.8% | compute |
| txt_mlp_out | `gemm_kernel` | 0.4040 | 31 | 12.525 | 5080 | 90.3 | 8.6% | compute |
| txt_mlp_gate | `gemm_kernel` | 0.3954 | 31 | 12.256 | 5192 | 92.3 | 8.8% | compute |
| txt_mlp_up | `gemm_kernel` | 0.3946 | 31 | 12.233 | 5202 | 92.5 | 8.8% | compute |
| txt_attn_out | `gemm_kernel` | 0.1711 | 31 | 5.304 | 4444 | 80.1 | 7.5% | compute |
| txt_in | `gemm_kernel` | 0.3408 | 1 | 0.341 | 4462 | 79.5 | 7.6% | compute |

**Total inference time (this stage)** ≈ **156106.88 ms**

### VAE_DECODER

#### PTL — vae_decoder stage=vae_decoder

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| up2.post_shuffle_conv | `causal_conv3d_carry_tail` | 2397.8990 | 60 | 143873.940 | 8629 | 3.7 | 14.6% | compute |
| up1.post_shuffle_conv | `causal_conv3d_carry_tail` | 891.2516 | 60 | 53475.095 | 16939 | 3.7 | 28.7% | compute |
| up2.conv2 | `gen_conv` | 168.3202 | 300 | 50496.074 | 25557 | 11.1 | 43.3% | compute |
| up3.conv2 | `gen_conv` | 160.4244 | 300 | 48127.334 | 21470 | 18.6 | 36.4% | compute |
| up1.post_shuffle_conv | `gen_conv` | 658.0548 | 60 | 39483.287 | 16939 | 3.7 | 28.7% | compute |
| up2.post_shuffle_conv | `gen_conv` | 622.1157 | 60 | 37326.942 | 8629 | 3.7 | 14.6% | compute |
| up3.conv2 | `causal_conv3d_carry_tail` | 119.5876 | 300 | 35876.289 | 21470 | 18.6 | 36.4% | compute |
| up1.conv1 | `gen_conv` | 180.6259 | 180 | 32512.658 | 27251 | 6.0 | 46.2% | compute |
| gn_c128_h1536w2560 | `group_normalization_zsplit_final_zsplit_plain` | 40.3976 | 600 | 24238.531 | 130 | 65.1 | 62.0% | memory |
| up2.conv2 | `causal_conv3d_carry_tail` | 78.0804 | 300 | 23424.110 | 25557 | 11.1 | 43.3% | compute |
| up3.conv1 | `gen_conv` | 342.6912 | 60 | 20561.472 | 17815 | 12.9 | 30.2% | compute |
| up2.conv1 | `gen_conv` | 322.0138 | 60 | 19320.828 | 24306 | 8.8 | 41.2% | compute |
| up3.conv1 | `causal_conv3d_carry_tail` | 294.0498 | 60 | 17642.989 | 17815 | 12.9 | 30.2% | compute |
| up3.conv2 | `causal_conv3d_carry_pack` | 44.0489 | 300 | 13214.681 | 21470 | 18.6 | 36.4% | compute |
| gn_c128_h1536w2560 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 21.4065 | 600 | 12843.895 | 130 | 65.1 | 62.0% | memory |
| mid_attn_sdpa | `sdpa_micro_prefill_sa` | 418.5564 | 30 | 12556.692 | 18696 | 0.6 | 31.7% | compute |
| up2.post_shuffle_conv | `causal_conv3d_carry_pack` | 205.2352 | 60 | 12314.111 | 8629 | 3.7 | 14.6% | compute |
| gn_c256_h768w1280 | `group_normalization_zsplit_final_zsplit_plain` | 20.1441 | 600 | 12086.468 | 129 | 64.7 | 61.6% | memory |
| up2.conv1 | `causal_conv3d_carry_tail` | 199.9807 | 60 | 11998.840 | 24306 | 8.8 | 41.2% | compute |
| up1.conv1 | `causal_conv3d_carry_tail` | 56.6834 | 180 | 10203.020 | 27251 | 6.0 | 46.2% | compute |
| up3.conv1 | `causal_conv3d_carry_pack` | 144.3348 | 60 | 8660.090 | 17815 | 12.9 | 30.2% | compute |
| up2.conv2 | `causal_conv3d_carry_pack` | 25.7982 | 300 | 7739.453 | 25557 | 11.1 | 43.3% | compute |
| up1.upscale_conv | `gen_reorder` | 60.5562 | 120 | 7266.743 | 13050 | 28.7 | 22.1% | compute |
| conv_out | `causal_conv3d_carry_tail` | 109.7654 | 60 | 6585.922 | 873 | 21.8 | 1.5% | compute |
| gn_c256_h768w1280 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 10.9573 | 600 | 6574.372 | 129 | 64.7 | 61.6% | memory |
| up3.shortcut | `gen_reorder` | 105.6395 | 60 | 6338.368 | 4425 | 51.9 | 7.5% | compute |
| up1.post_shuffle_conv | `causal_conv3d_carry_pack` | 93.5754 | 60 | 5614.526 | 16939 | 3.7 | 28.7% | compute |
| up0.post_shuffle_conv | `gen_conv` | 184.9298 | 30 | 5547.894 | 26625 | 5.8 | 45.1% | compute |
| up0.conv1 | `gen_conv` | 24.8237 | 180 | 4468.263 | 21190 | 6.5 | 35.9% | compute |
| up1.conv1 | `causal_conv3d_carry_pack` | 17.7990 | 180 | 3203.828 | 27251 | 6.0 | 46.2% | compute |
| up2.conv1 | `causal_conv3d_carry_pack` | 50.4389 | 60 | 3026.332 | 24306 | 8.8 | 41.2% | compute |
| mid.conv1 | `gen_conv` | 24.6287 | 120 | 2955.438 | 21328 | 6.5 | 36.2% | compute |
| conv_out | `causal_conv3d_carry_pack` | 42.2425 | 60 | 2534.550 | 873 | 21.8 | 1.5% | compute |
| up0.conv1 | `causal_conv3d_carry_tail` | 13.4099 | 180 | 2413.785 | 21190 | 6.5 | 35.9% | compute |
| up2.shortcut | `gen_reorder` | 37.6702 | 60 | 2260.213 | 10696 | 62.7 | 18.1% | compute |
| up1.upscale_conv | `gen_conv` | 18.4298 | 120 | 2211.575 | 13050 | 28.7 | 22.1% | compute |
| conv_out | `gen_conv` | 34.7361 | 60 | 2084.167 | 873 | 21.8 | 1.5% | compute |
| gn_c512_h384w640 | `group_normalization_zsplit_final_zsplit_plain` | 10.1470 | 180 | 1826.468 | 127 | 63.6 | 60.6% | memory |
| up0.post_shuffle_conv | `causal_conv3d_carry_tail` | 58.3031 | 30 | 1749.094 | 26625 | 5.8 | 45.1% | compute |
| mid.conv1 | `causal_conv3d_carry_tail` | 13.3292 | 120 | 1599.501 | 21328 | 6.5 | 36.2% | compute |
| gn_c512_h384w640 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 5.6682 | 180 | 1020.273 | 127 | 63.6 | 60.6% | memory |
| up3.shortcut | `gen_conv` | 10.8420 | 60 | 650.519 | 4425 | 51.9 | 7.5% | compute |
| up2.shortcut | `gen_conv` | 10.5146 | 60 | 630.878 | 10696 | 62.7 | 18.1% | compute |
| up0.upscale_conv | `gen_reorder` | 9.0879 | 60 | 545.276 | 18210 | 40.3 | 30.9% | compute |
| up0.post_shuffle_conv | `causal_conv3d_carry_pack` | 17.8871 | 30 | 536.613 | 26625 | 5.8 | 45.1% | compute |
| gn_c512_h192w320 | `group_normalization_zsplit_final_zsplit_plain` | 1.1932 | 420 | 501.162 | 135 | 67.4 | 64.2% | memory |
| up0.conv1 | `causal_conv3d_carry_pack` | 2.5162 | 180 | 452.914 | 21190 | 6.5 | 35.9% | compute |
| up0.upscale_conv | `gen_conv` | 5.0639 | 60 | 303.833 | 18210 | 40.3 | 30.9% | compute |
| mid.conv1 | `causal_conv3d_carry_pack` | 2.5238 | 120 | 302.862 | 21328 | 6.5 | 36.2% | compute |
| gn_c512_h192w320 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 0.6719 | 420 | 282.180 | 135 | 67.4 | 64.2% | memory |
| mid_attn_qkvo | `gemm_kernel` | 1.2471 | 120 | 149.647 | 25831 | 101.3 | 43.8% | compute |
| conv_in | `gen_conv` | 0.7927 | 30 | 23.781 | 30746 | 78.3 | 52.1% | compute |
| conv_in | `causal_conv3d_carry_tail` | 0.0467 | 30 | 1.400 | 30746 | 78.3 | 52.1% | compute |
| conv_in | `causal_conv3d_carry_pack` | 0.0371 | 30 | 1.114 | 30746 | 78.3 | 52.1% | compute |

**Total inference time (this stage)** ≈ **721640.29 ms**

