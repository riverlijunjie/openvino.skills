# SeedVR2 (VAE encoder + windowed-attention DiT + VAE decoder, video SR/restoration) — Per-submodel Kernel Tables

## Platform: PTL

### VAE_ENCODER

#### PTL — vae_encoder stage=vae_encoder

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| down0.conv1 | `gen_conv` | 155.4501 | 116 | 18032.212 | 33215 | 24.0 | 56.3% | compute |
| down1.conv2 | `gen_conv` | 181.8817 | 87 | 15823.712 | 34101 | 12.4 | 57.8% | compute |
| down2.conv2 | `gen_conv` | 92.6367 | 87 | 8059.392 | 35788 | 7.9 | 60.7% | compute |
| down0.conv1 | `gen_reorder` | 67.1199 | 116 | 7785.910 | 33215 | 24.0 | 56.3% | compute |
| gn_c128_h1088w1920 | `group_normalization_zsplit_final_zsplit_plain` | 43.1546 | 145 | 6257.423 | 131 | 65.6 | 62.5% | memory |
| gn_c128_h1088w1920 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 22.0544 | 145 | 3197.883 | 131 | 65.6 | 62.5% | memory |
| down1.conv2 | `gen_reorder` | 34.9083 | 87 | 3037.018 | 34101 | 12.4 | 57.8% | compute |
| mid_attn_sdpa | `sdpa_micro_prefill_sa` | 95.5962 | 29 | 2772.289 | 23103 | 1.4 | 39.2% | compute |
| gn_c256_h544w960 | `group_normalization_zsplit_final_zsplit_plain` | 21.1190 | 116 | 2449.799 | 132 | 66.2 | 63.1% | memory |
| down1.conv1 | `gen_conv` | 81.6422 | 29 | 2367.624 | 34244 | 17.4 | 58.1% | compute |
| gn_c256_h544w960 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 11.1770 | 116 | 1296.527 | 132 | 66.2 | 63.1% | memory |
| down2.conv1 | `gen_conv` | 44.1454 | 29 | 1280.217 | 34959 | 10.2 | 59.3% | compute |
| down3_mid.conv1 | `gen_conv` | 10.7897 | 116 | 1251.601 | 36159 | 11.6 | 61.3% | compute |
| conv_in | `gen_reorder` | 33.4251 | 29 | 969.327 | 3103 | 39.7 | 5.3% | compute |
| down2.conv2 | `gen_reorder` | 10.6486 | 87 | 926.430 | 35788 | 7.9 | 60.7% | compute |
| down0.downsampler | `gen_reorder` | 29.5995 | 29 | 858.387 | 14548 | 63.2 | 24.7% | compute |
| down1.conv1 | `gen_reorder` | 26.2981 | 29 | 762.644 | 34244 | 17.4 | 58.1% | compute |
| down1.shortcut | `gen_reorder` | 22.2078 | 29 | 644.026 | 5243 | 61.4 | 8.9% | compute |
| gn_c512_h272w480 | `group_normalization_zsplit_final_zsplit_plain` | 5.5125 | 116 | 639.449 | 122 | 61.0 | 58.1% | memory |
| down1.downsampler | `gen_conv` | 21.8533 | 29 | 633.745 | 24085 | 38.4 | 40.8% | compute |
| conv_in | `gen_conv` | 21.5242 | 29 | 624.200 | 3103 | 39.7 | 5.3% | compute |
| down1.downsampler | `gen_reorder` | 16.5147 | 29 | 478.926 | 24085 | 38.4 | 40.8% | compute |
| gn_c512_h272w480 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 3.2506 | 116 | 377.073 | 122 | 61.0 | 58.1% | memory |
| down0.downsampler | `gen_conv` | 12.7457 | 29 | 369.625 | 14548 | 63.2 | 24.7% | compute |
| down2.downsampler | `gen_conv` | 10.9294 | 29 | 316.954 | 29432 | 28.6 | 49.9% | compute |
| down2.conv1 | `gen_reorder` | 8.7214 | 29 | 252.921 | 34959 | 10.2 | 59.3% | compute |
| down3_mid.conv1 | `gen_reorder` | 1.9884 | 116 | 230.658 | 36159 | 11.6 | 61.3% | compute |
| gn_c512_h136w240 | `group_normalization_zsplit_final_zsplit_plain` | 0.5813 | 290 | 168.564 | 142 | 71.2 | 67.8% | memory |
| down2.shortcut | `gen_reorder` | 5.4082 | 29 | 156.837 | 10100 | 59.2 | 17.1% | compute |
| down2.downsampler | `gen_reorder` | 4.7692 | 29 | 138.308 | 29432 | 28.6 | 49.9% | compute |
| down1.shortcut | `gen_conv` | 3.9028 | 29 | 113.182 | 5243 | 61.4 | 8.9% | compute |
| gn_c512_h136w240 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 0.3558 | 290 | 103.192 | 142 | 71.2 | 67.8% | memory |
| mid_attn_qkvo | `gemm_kernel` | 0.7202 | 116 | 83.545 | 23761 | 93.5 | 40.3% | compute |
| down2.shortcut | `gen_conv` | 1.3694 | 29 | 39.713 | 10100 | 59.2 | 17.1% | compute |
| conv_out | `gen_reorder` | 0.9935 | 29 | 28.811 | 14930 | 53.4 | 25.3% | compute |
| conv_out | `gen_conv` | 0.9408 | 29 | 27.282 | 14930 | 53.4 | 25.3% | compute |

**Total inference time (this stage)** ≈ **82555.41 ms**

### DIT

#### PTL — dit stage=dit

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vid_mlp_out | `gemm_kernel` | 409.8865 | 32 | 13116.367 | 21840 | 11.8 | 37.0% | compute |
| vid_mlp_gate | `gemm_kernel` | 358.4027 | 32 | 11468.886 | 24978 | 13.5 | 42.3% | compute |
| vid_qkv | `gemm_kernel` | 329.2984 | 32 | 10537.548 | 30206 | 15.9 | 51.2% | compute |
| vid_mlp_up | `gemm_kernel` | 235.6615 | 32 | 7541.168 | 37987 | 20.5 | 64.4% | compute |
| wattn_window_a | `cm_sdpa_vlen_sa` | 310.7147 | 16 | 4971.436 | 22108 | 16.7 | 37.5% | compute |
| wattn_window_b | `cm_sdpa_vlen_sa` | 219.7804 | 16 | 3516.487 | 20836 | 23.6 | 35.3% | compute |
| vid_attn_out | `gemm_kernel` | 66.5283 | 32 | 2128.906 | 49837 | 39.1 | 84.5% | compute |
| vid_in | `gemm_kernel` | 35.6465 | 1 | 35.647 | 4796 | 38.2 | 8.1% | compute |
| vid_out | `gemm_kernel` | 14.2810 | 1 | 14.281 | 5804 | 93.0 | 9.8% | compute |
| txt_qkv | `gemm_kernel` | 0.4444 | 32 | 14.220 | 5132 | 91.2 | 8.7% | compute |
| txt_mlp_out | `gemm_kernel` | 0.4008 | 31 | 12.424 | 5121 | 91.0 | 8.7% | compute |
| txt_mlp_gate | `gemm_kernel` | 0.3966 | 31 | 12.295 | 5175 | 92.0 | 8.8% | compute |
| txt_mlp_up | `gemm_kernel` | 0.3933 | 31 | 12.194 | 5218 | 92.8 | 8.8% | compute |
| txt_attn_out | `gemm_kernel` | 0.1473 | 31 | 4.566 | 5161 | 93.0 | 8.8% | compute |
| txt_in | `gemm_kernel` | 0.3404 | 1 | 0.340 | 4466 | 79.6 | 7.6% | compute |

**Total inference time (this stage)** ≈ **53386.76 ms**

### VAE_DECODER

#### PTL — vae_decoder stage=vae_decoder

| Op | Kernel | Single ms | Calls/inf | Total ms | GFLOPS | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| up2.conv2 | `gen_conv` | 22.7500 | 1200 | 27299.978 | 27501 | 12.0 | 46.6% | compute |
| up3.conv2 | `gen_conv` | 22.5677 | 1200 | 27081.210 | 21299 | 18.5 | 36.1% | compute |
| up1.post_shuffle_conv | `gen_conv` | 99.2093 | 240 | 23810.222 | 30662 | 6.8 | 52.0% | compute |
| up2.post_shuffle_conv | `gen_conv` | 93.1048 | 240 | 22345.145 | 26254 | 11.4 | 44.5% | compute |
| up3.conv2 | `causal_conv3d_carry_tail` | 16.9967 | 1200 | 20396.003 | 21299 | 18.5 | 36.1% | compute |
| up1.conv1 | `gen_conv` | 20.2834 | 720 | 14604.054 | 37811 | 8.7 | 64.1% | compute |
| gn_c128_h576w976 | `group_normalization_zsplit_final_zsplit_plain` | 5.6533 | 2400 | 13567.882 | 128 | 64.1 | 61.1% | memory |
| up2.conv2 | `causal_conv3d_carry_tail` | 9.7658 | 1200 | 11718.994 | 27501 | 12.0 | 46.6% | compute |
| up2.conv1 | `gen_conv` | 45.9923 | 240 | 11038.151 | 26588 | 9.7 | 45.1% | compute |
| up3.conv1 | `gen_conv` | 44.3307 | 240 | 10639.361 | 19888 | 14.4 | 33.7% | compute |
| up2.post_shuffle_conv | `causal_conv3d_carry_tail` | 44.2037 | 240 | 10608.878 | 26254 | 11.4 | 44.5% | compute |
| up3.conv1 | `causal_conv3d_carry_tail` | 41.7694 | 240 | 10024.655 | 19888 | 14.4 | 33.7% | compute |
| up3.conv2 | `causal_conv3d_carry_pack` | 7.1268 | 1200 | 8552.101 | 21299 | 18.5 | 36.1% | compute |
| gn_c128_h576w976 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 3.3187 | 2400 | 7964.964 | 128 | 64.1 | 61.1% | memory |
| gn_c256_h288w488 | `group_normalization_zsplit_final_zsplit_plain` | 2.7997 | 2400 | 6719.174 | 129 | 64.3 | 61.3% | memory |
| up1.post_shuffle_conv | `causal_conv3d_carry_tail` | 21.6712 | 240 | 5201.098 | 30662 | 6.8 | 52.0% | compute |
| up2.conv1 | `causal_conv3d_carry_tail` | 20.2251 | 240 | 4854.029 | 26588 | 9.7 | 45.1% | compute |
| up2.conv2 | `causal_conv3d_carry_pack` | 3.5989 | 1200 | 4318.728 | 27501 | 12.0 | 46.6% | compute |
| gn_c256_h288w488 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 1.6733 | 2400 | 4015.884 | 129 | 64.3 | 61.3% | memory |
| conv_out | `causal_conv3d_carry_tail` | 15.8776 | 240 | 3810.617 | 842 | 21.0 | 1.4% | compute |
| up2.post_shuffle_conv | `causal_conv3d_carry_pack` | 14.2058 | 240 | 3409.403 | 26254 | 11.4 | 44.5% | compute |
| up3.conv1 | `causal_conv3d_carry_pack` | 13.9090 | 240 | 3338.150 | 19888 | 14.4 | 33.7% | compute |
| up1.conv1 | `causal_conv3d_carry_tail` | 4.4861 | 720 | 3229.981 | 37811 | 8.7 | 64.1% | compute |
| up0.post_shuffle_conv | `gen_conv` | 20.3175 | 120 | 2438.097 | 37748 | 8.7 | 64.0% | compute |
| up3.shortcut | `gen_reorder` | 10.0698 | 240 | 2416.750 | 6445 | 75.5 | 10.9% | compute |
| up1.upscale_conv | `gen_reorder` | 4.9396 | 480 | 2371.013 | 18709 | 41.6 | 31.7% | compute |
| up1.post_shuffle_conv | `causal_conv3d_carry_pack` | 8.6913 | 240 | 2085.907 | 30662 | 6.8 | 52.0% | compute |
| up2.conv1 | `causal_conv3d_carry_pack` | 8.5299 | 240 | 2047.183 | 26588 | 9.7 | 45.1% | compute |
| up0.conv1 | `gen_conv` | 2.7356 | 720 | 1969.628 | 29198 | 11.8 | 49.5% | compute |
| conv_out | `causal_conv3d_carry_pack` | 7.3837 | 240 | 1772.100 | 842 | 21.0 | 1.4% | compute |
| up1.upscale_conv | `gen_conv` | 2.9374 | 480 | 1409.963 | 18709 | 41.6 | 31.7% | compute |
| mid.conv1 | `gen_conv` | 2.7569 | 480 | 1323.298 | 29203 | 11.8 | 49.5% | compute |
| up2.shortcut | `gen_reorder` | 4.9780 | 240 | 1194.728 | 11403 | 66.9 | 19.3% | compute |
| conv_out | `gen_conv` | 4.4084 | 240 | 1058.012 | 842 | 21.0 | 1.4% | compute |
| up1.conv1 | `causal_conv3d_carry_pack` | 1.3402 | 720 | 964.971 | 37811 | 8.7 | 64.1% | compute |
| gn_c512_h144w244 | `group_normalization_zsplit_final_zsplit_plain` | 1.3392 | 720 | 964.208 | 137 | 68.5 | 65.3% | memory |
| mid_attn_sdpa | `sdpa_micro_prefill_sa` | 7.1107 | 120 | 853.287 | 22494 | 5.1 | 38.1% | compute |
| up0.conv1 | `causal_conv3d_carry_tail` | 1.0720 | 720 | 771.860 | 29198 | 11.8 | 49.5% | compute |
| gn_c512_h144w244 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 0.7595 | 720 | 546.869 | 137 | 68.5 | 65.3% | memory |
| up0.post_shuffle_conv | `causal_conv3d_carry_tail` | 4.4444 | 120 | 533.331 | 37748 | 8.7 | 64.0% | compute |
| mid.conv1 | `causal_conv3d_carry_tail` | 1.0512 | 480 | 504.581 | 29203 | 11.8 | 49.5% | compute |
| up2.shortcut | `gen_conv` | 1.4842 | 240 | 356.205 | 11403 | 66.9 | 19.3% | compute |
| up3.shortcut | `gen_conv` | 1.3638 | 240 | 327.309 | 6445 | 75.5 | 10.9% | compute |
| up0.upscale_conv | `gen_reorder` | 1.2769 | 240 | 306.465 | 18081 | 41.8 | 30.7% | compute |
| up0.conv1 | `causal_conv3d_carry_pack` | 0.2586 | 720 | 186.204 | 29198 | 11.8 | 49.5% | compute |
| up0.upscale_conv | `gen_conv` | 0.7608 | 240 | 182.581 | 18081 | 41.8 | 30.7% | compute |
| up0.post_shuffle_conv | `causal_conv3d_carry_pack` | 1.3872 | 120 | 166.459 | 37748 | 8.7 | 64.0% | compute |
| gn_c512_h72w122 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 0.0981 | 1680 | 164.852 | 245 | 122.3 | 116.5% | memory |
| mid.conv1 | `causal_conv3d_carry_pack` | 0.2571 | 480 | 123.421 | 29203 | 11.8 | 49.5% | compute |
| gn_c512_h72w122 | `group_normalization_zsplit_final_zsplit_plain` | 0.0476 | 1680 | 80.039 | 245 | 122.3 | 116.5% | memory |
| mid_attn_qkvo | `gemm_kernel` | 0.1441 | 480 | 69.153 | 31966 | 128.5 | 54.2% | compute |
| conv_in | `gen_conv` | 0.1318 | 120 | 15.819 | 25032 | 66.2 | 42.4% | compute |
| conv_in | `causal_conv3d_carry_pack` | 0.0111 | 120 | 1.335 | 25032 | 66.2 | 42.4% | compute |
| conv_in | `conv_reorder` | 0.0074 | 120 | 0.891 | 25032 | 66.2 | 42.4% | compute |

**Total inference time (this stage)** ≈ **285755.18 ms**

