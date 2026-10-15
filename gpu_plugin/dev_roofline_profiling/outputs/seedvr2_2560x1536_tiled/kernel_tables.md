# SeedVR2 (VAE encoder + windowed-attention DiT + VAE decoder, video SR/restoration) -- 2560x1536 variant, decoder TILED (K=4) — Per-submodel Kernel Tables

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
| up3.conv2 | `gen_conv` | 47.1019 | 1200 | 56522.311 | 20324 | 17.7 | 34.5% | compute |
| up2.conv2 | `gen_conv` | 46.5437 | 1200 | 55852.460 | 24549 | 10.7 | 41.6% | compute |
| up1.post_shuffle_conv | `gen_conv` | 198.0120 | 240 | 47522.880 | 28768 | 6.3 | 48.8% | compute |
| up2.post_shuffle_conv | `gen_conv` | 191.6441 | 240 | 45994.593 | 25304 | 11.0 | 42.9% | compute |
| up1.conv1 | `gen_conv` | 57.8646 | 720 | 41662.484 | 25376 | 5.7 | 43.0% | compute |
| up3.conv2 | `causal_conv3d_carry_tail` | 31.6520 | 1200 | 37982.458 | 20324 | 17.7 | 34.5% | compute |
| gn_c128_h800w1296 | `group_normalization_zsplit_final_zsplit_plain` | 10.4919 | 2400 | 25180.598 | 132 | 66.2 | 63.0% | memory |
| up2.conv2 | `causal_conv3d_carry_tail` | 19.9571 | 1200 | 23948.461 | 24549 | 10.7 | 41.6% | compute |
| up2.conv1 | `gen_conv` | 94.2520 | 240 | 22620.488 | 24782 | 9.0 | 42.0% | compute |
| up3.conv1 | `gen_conv` | 91.6041 | 240 | 21984.983 | 19760 | 14.3 | 33.5% | compute |
| up2.post_shuffle_conv | `causal_conv3d_carry_tail` | 74.0076 | 240 | 17761.827 | 25304 | 11.0 | 42.9% | compute |
| up3.conv1 | `causal_conv3d_carry_tail` | 70.2661 | 240 | 16863.870 | 19760 | 14.3 | 33.5% | compute |
| up3.conv2 | `causal_conv3d_carry_pack` | 11.4976 | 1200 | 13797.168 | 20324 | 17.7 | 34.5% | compute |
| gn_c128_h800w1296 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 5.5474 | 2400 | 13313.722 | 132 | 66.2 | 63.0% | memory |
| gn_c256_h400w648 | `group_normalization_zsplit_final_zsplit_plain` | 5.2428 | 2400 | 12582.646 | 126 | 62.8 | 59.8% | memory |
| up1.post_shuffle_conv | `causal_conv3d_carry_tail` | 43.0623 | 240 | 10334.960 | 28768 | 6.3 | 48.8% | compute |
| up2.conv2 | `causal_conv3d_carry_pack` | 8.1867 | 1200 | 9824.024 | 24549 | 10.7 | 41.6% | compute |
| up2.conv1 | `causal_conv3d_carry_tail` | 40.2581 | 240 | 9661.937 | 24782 | 9.0 | 42.0% | compute |
| gn_c256_h400w648 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 3.2135 | 2400 | 7712.489 | 126 | 62.8 | 59.8% | memory |
| up1.conv1 | `causal_conv3d_carry_tail` | 10.2185 | 720 | 7357.330 | 25376 | 5.7 | 43.0% | compute |
| conv_out | `causal_conv3d_carry_tail` | 30.2639 | 240 | 7263.336 | 853 | 21.3 | 1.4% | compute |
| up0.post_shuffle_conv | `gen_conv` | 49.8403 | 120 | 5980.835 | 29612 | 6.7 | 50.2% | compute |
| up2.post_shuffle_conv | `causal_conv3d_carry_pack` | 24.3061 | 240 | 5833.463 | 25304 | 11.0 | 42.9% | compute |
| up3.conv1 | `causal_conv3d_carry_pack` | 23.7889 | 240 | 5709.324 | 19760 | 14.3 | 33.5% | compute |
| up0.conv1 | `gen_conv` | 6.7709 | 720 | 4875.031 | 24152 | 8.5 | 40.9% | compute |
| up3.shortcut | `gen_reorder` | 18.5853 | 240 | 4460.462 | 6324 | 74.1 | 10.7% | compute |
| up1.upscale_conv | `gen_reorder` | 9.2574 | 480 | 4443.532 | 19029 | 42.1 | 32.3% | compute |
| up1.post_shuffle_conv | `causal_conv3d_carry_pack` | 13.8174 | 240 | 3316.184 | 28768 | 6.3 | 48.8% | compute |
| mid.conv1 | `gen_conv` | 6.7733 | 480 | 3251.204 | 24181 | 8.5 | 41.0% | compute |
| up2.conv1 | `causal_conv3d_carry_pack` | 13.4655 | 240 | 3231.720 | 24782 | 9.0 | 42.0% | compute |
| up1.conv1 | `causal_conv3d_carry_pack` | 3.9372 | 720 | 2834.776 | 25376 | 5.7 | 43.0% | compute |
| mid_attn_sdpa | `sdpa_micro_prefill_sa` | 23.4383 | 120 | 2812.596 | 23212 | 2.8 | 39.4% | compute |
| conv_out | `causal_conv3d_carry_pack` | 11.4183 | 240 | 2740.385 | 853 | 21.3 | 1.4% | compute |
| up1.upscale_conv | `gen_conv` | 5.0254 | 480 | 2412.185 | 19029 | 42.1 | 32.3% | compute |
| up2.shortcut | `gen_reorder` | 9.3097 | 240 | 2234.330 | 11481 | 67.3 | 19.5% | compute |
| conv_out | `gen_conv` | 8.7377 | 240 | 2097.046 | 853 | 21.3 | 1.4% | compute |
| gn_c512_h200w324 | `group_normalization_zsplit_final_zsplit_plain` | 2.5835 | 720 | 1860.144 | 131 | 65.4 | 62.3% | memory |
| up0.conv1 | `causal_conv3d_carry_tail` | 1.9282 | 720 | 1388.321 | 24152 | 8.5 | 40.9% | compute |
| up0.post_shuffle_conv | `causal_conv3d_carry_tail` | 8.9259 | 120 | 1071.114 | 29612 | 6.7 | 50.2% | compute |
| gn_c512_h200w324 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 1.4707 | 720 | 1058.928 | 131 | 65.4 | 62.3% | memory |
| mid.conv1 | `causal_conv3d_carry_tail` | 1.9144 | 480 | 918.910 | 24181 | 8.5 | 41.0% | compute |
| up3.shortcut | `gen_conv` | 2.9052 | 240 | 697.256 | 6324 | 74.1 | 10.7% | compute |
| up0.upscale_conv | `gen_reorder` | 2.7025 | 240 | 648.594 | 16286 | 36.8 | 27.6% | compute |
| up2.shortcut | `gen_conv` | 2.5273 | 240 | 606.562 | 11481 | 67.3 | 19.5% | compute |
| up0.conv1 | `causal_conv3d_carry_pack` | 0.6053 | 720 | 435.836 | 24152 | 8.5 | 40.9% | compute |
| up0.post_shuffle_conv | `causal_conv3d_carry_pack` | 2.9573 | 120 | 354.882 | 29612 | 6.7 | 50.2% | compute |
| up0.upscale_conv | `gen_conv` | 1.4696 | 240 | 352.715 | 16286 | 36.8 | 27.6% | compute |
| gn_c512_h100w162 | `group_normalization_zsplit_final_zsplit_plain` | 0.2090 | 1680 | 351.071 | 171 | 85.6 | 81.6% | memory |
| gn_c512_h100w162 | `group_normalization_zsplit_calc_sqr_mean_zsplit_plain` | 0.1770 | 1680 | 297.328 | 171 | 85.6 | 81.6% | memory |
| mid.conv1 | `causal_conv3d_carry_pack` | 0.6028 | 480 | 289.335 | 24181 | 8.5 | 41.0% | compute |
| mid_attn_qkvo | `gemm_kernel` | 0.3340 | 480 | 160.333 | 25427 | 100.9 | 43.1% | compute |
| conv_in | `gen_conv` | 0.3878 | 120 | 46.537 | 16521 | 42.8 | 28.0% | compute |
| conv_in | `causal_conv3d_carry_tail` | 0.0196 | 120 | 2.353 | 16521 | 42.8 | 28.0% | compute |
| conv_in | `causal_conv3d_carry_pack` | 0.0187 | 120 | 2.248 | 16521 | 42.8 | 28.0% | compute |

**Total inference time (this stage)** ≈ **572520.57 ms**

