# Qwen3-Omni-4B — Roofline on PTL 4Xe (2026-06-10)

**Platform**: Intel PTL 4Xe iGPU (4 Xe cores × 8 EU × 10 threads), 2400 MHz, 110 GB/s (clinfo/cliloader report 2450 MHz max clock)
**Model**: Qwen3-Omni-4B-Instruct-multilingual-int4 (Thinker text + Audio encoder + Vision encoder + Talker + code2wav).
**Quantization**: matmul INT4 g128 / LM_head INT8 g128 / KV cache INT8 / activations FP16. PA opencl + micro_kernel.
**Source config**: `~/workspace/remote_debug/qwen3_omni/models/Qwen3-Omni-4B-Instruct-multilingual-int4/config.json`

**Inputs evaluated**: thinker-text prefill context = 1024 / 2048 / 4096 / 8192 tokens; output = 512 tokens. The Thinker text decoder (FC + PA + small ops) is **measured** on PTL 4Xe via cliloader; the Audio encoder (S=1500), Vision encoder (S=2304) and Talker are reported **theoretical-only** (not benched in this run).

## 1. Hardware peaks (per SKILL.md formulas)

`FP16 XMX TFLOPS = xe_cores × 8 × 256 × freq_GHz`; `INT8 XMX = 2× FP16`; `SIMD FP16 = xe_cores × 8 × 32 × freq`.

| Platform | Xe Cores | Freq (MHz) | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | SIMD FP16 (TFLOPS) | Ridge FP16 (FLOP/B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PTL 4Xe | 4 | 2400 | 110 | 19.661 | 39.322 | 2.458 | 179 |

## 2. Model architecture

### Thinker text decoder (Qwen3VL-style dense, 36 layers, GQA 32:8)

| Field | Value |
|---|---:|
| `hidden_size` | 2560 |
| `num_hidden_layers` | 36 |
| `num_attention_heads (NH)` | 32 |
| `num_key_value_heads (NKV)` | 8 (GQA 4:1) |
| `head_dim (HD)` | 128 |
| `Q_dim / KV_dim` | 4096 / 1024 |
| `intermediate_size` | 9728 |
| `vocab_size` | 151936 |
| `tie_word_embeddings` | True |
| `hidden_act` | silu |
| `rope_theta` | 1,000,000 |

### Audio encoder (Whisper-style MHA, 32 layers, bidirectional)

| Field | Value |
|---|---:|
| `d_model` | 1280 |
| `encoder_layers` | 32 |
| `encoder_attention_heads (NH=NKV)` | 20 |
| `head_dim` | 64 |
| `encoder_ffn_dim` | 5120 |
| `num_mel_bins` | 128 |
| `max_source_positions / S` | 1500 |
| `output_dim (→ thinker hidden)` | 2560 |

### Vision encoder (SigLIP-style ViT, 27 layers, bidirectional)

| Field | Value |
|---|---:|
| `hidden_size` | 1152 |
| `depth` | 27 |
| `num_heads (NH=NKV)` | 16 |
| `head_dim` | 72 (= 1152/16) |
| `intermediate_size` | 4304 |
| `patch_size` | 16 |
| `image_size` | 768 |
| `S_patches (input tokens to ViT)` | 2304 (= 48×48) |
| `vis_tokens after spatial_merge=2` | 576 |
| `out_hidden_size` | 2560 |
| `deepstack_visual_indexes` | [8, 16, 24] |

### Talker (audio-output AR decoder, 28 layers, GQA 16:8)

| Field | Value |
|---|---:|
| `hidden_size` | 2560 |
| `num_hidden_layers` | 28 |
| `NH / NKV / HD` | 16 / 8 / 128 |
| `intermediate_size` | 3072 |
| `vocab_size (codec tokens)` | 3072 |
| `accept_hidden_layer` | 14 |

## 3. Theoretical weight distribution (INT4 g128 matmul, INT8 g128 LM_head, FP16 norms)

**Quantization sizing**:
- INT4 g128 weight = K·N·0.5 bytes + (K/128)·N·2 fp16 scales + (K/128)·N·1 int8 zero-points
- INT8 g128 weight = K·N·1 bytes + (K/128)·N·2 fp16 scales + (K/128)·N·1 int8 zero-points
- FP16 = K·N·2 bytes (norms, embed/LM head when un-tied or for tiny vocabs like talker)

### Thinker text decoder per-layer (1 of 36)

| Weight | Shape (K × N) | Quant | MB |
|---|---|---|---:|
| FC_QKV (fused Q+K+V) | 2560 × 6144 | INT4 g128 | 7.852 |
| FC_O (attn output) | 4096 × 2560 | INT4 g128 | 5.234 |
| FC_Gate (SwiGLU) | 2560 × 9728 | INT4 g128 | 12.432 |
| FC_Up   (SwiGLU) | 2560 × 9728 | INT4 g128 | 12.432 |
| FC_Down (SwiGLU) | 9728 × 2560 | INT4 g128 | 12.432 |
| RMSNorm × 2 + q_norm + k_norm | small | FP16 | 0.0100 |
| **per layer** |  |  | **50.39** |
| **× 36 layers** |  |  | **1814.08** |

### Thinker global / shared

| Weight | Shape | Quant | MB |
|---|---|---|---:|
| Token embedding (tied w/ LM head) | 151936 × 2560 | INT8 g128 | 379.631 |
| LM_head extra (tied=True) | — | — | 0.0 |
| Final RMSNorm | [2560] | FP16 | 0.005 |
| **Thinker total** |  |  | **2193.72** |

### Audio encoder per-layer (1 of 32) — INT4 g128

| Weight | Shape (K × N) | Quant | MB |
|---|---|---|---:|
| FC_QKV (fused) | 1280 × 3840 | INT4 g128 | 2.454 |
| FC_O           | 1280 × 1280 | INT4 g128 | 0.818 |
| FC_FC1         | 1280 × 5120 | INT4 g128 | 3.271 |
| FC_FC2         | 5120 × 1280 | INT4 g128 | 3.271 |
| LayerNorm × 2  | [1280] g+b each | FP16 | 0.01 |
| **per layer** |  |  | **9.824** |
| **× 32 layers** |  |  | **314.37** |
| Conv front-end + pos_embed + adapter | misc | mixed | 14.894 |
| **Audio encoder total** |  |  | **329.27** |

### Vision encoder per-layer (1 of 27) — INT4 g128

| Weight | Shape (K × N) | Quant | MB |
|---|---|---|---:|
| FC_QKV (fused) | 1152 × 3456 | INT4 g128 | 1.987 |
| FC_O           | 1152 × 1152 | INT4 g128 | 0.662 |
| FC_FC1         | 1152 × 4304 | INT4 g128 | 2.475 |
| FC_FC2         | 4304 × 1152 | INT4 g128 | 2.473 |
| LayerNorm × 2  | [1152] g+b each | FP16 | 0.009 |
| **per layer** |  |  | **7.607** |
| **× 27 layers** |  |  | **205.39** |
| Patch embed + pos_embed + merger + deepstack | misc | mixed | 18.747 |
| **Vision encoder total** |  |  | **224.13** |

### Talker (audio-output AR decoder) — INT4 g128 matmul, FP16 embed/lm_head (small vocab)

| Weight | MB |
|---|---:|
| Per-layer FC body (× 28) | 19.639 → 549.89 |
| Token embed (codec vocab=3072) | 15.0 |
| LM head (codec vocab=3072) | 15.0 |
| **Talker total** | **579.90** |

### Code predictor + code2wav (audio-output supporting nets)

| Component | MB |
|---|---:|
| Code predictor (5 layers, hidden=1024, × 16 code-group heads) | 103.28 |
| Code2wav (8 layers, hidden=1024, + 16 × 2048 codebooks dec_dim=1536) | 146.94 |

### Grand total

| Component | MB |
|---|---:|
| Thinker text decoder | 2193.72 |
| Audio encoder | 329.27 |
| Vision encoder | 224.13 |
| Talker | 579.90 |
| Code predictor | 103.28 |
| Code2wav | 146.94 |
| **Grand total (on-disk INT4-quantized)** | **3577.23** |

## 4. Benchmark methodology

- **Bench utils**: `fc_bench` (precision=u4 default INT4 g128 / u8 for LM_head / f16 for talker lm_head), `pa_bench` (kv_dtype=i8, impl=ocl), `small_ops_bench`.
- **Tool**: cliloader Device Performance Timing; `parse_logs.py` extracts per-iteration GPU kernel ns and aggregates per-iteration totals across split kernels (kv_update + pa_compute + finalization).
- **L2/L3 flush** between every FC infer to force VRAM-resident weight reads (per-layer fits in L2, full model does not).
- **Input/output tensors** allocated via RemoteContext in USM_DEVICE (iGPU shared system memory) to avoid PCIe transfer artifacts.
- **PA prefill** uses causal mask (S·(S+1)/2 effective attention pairs, per SKILL.md).
- **PA decode** is decomposed into `pa_kv_cache_update_ref` + `paged_attention_opt__single_token` (+ `single_token_finalization` for the GQA variant).
- **Audio / vision encoder SDPA** is bidirectional; reported as `pa_bench prefill` (causal) × 2 (lower bound for the full S² attention).
- **SwiGLU `multiply` and `swish`** are fused into the SwiGLU primitive in the real graph (per SKILL.md and glu_fusion); not profiled separately.
- **Talker** is profiled at decode (M=1) only because in steady state it runs autoregressively per output text token.
- **Code2wav** uses small windows and streaming flow-matching; theoretical-only in this report.

## 5. Encoder fixed overhead (runs once per inference)

### Audio encoder — S=1500 (INT4 g128, bidirectional MHA)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| enc_sdpa | `sdpa_micro__prefill (×2 bidir est.)` | 32 | — | 0.5859 | — | 18.7500 | — | — | — | — | compute |
| enc_fc_fc1 | `fc_int4_g128` | 32 | — | 0.5000 | — | 16.0000 | — | — | — | — | compute |
| enc_fc_fc2 | `fc_int4_g128` | 32 | — | 0.5000 | — | 16.0000 | — | — | — | — | compute |
| enc_fc_qkv | `fc_int4_g128` | 32 | — | 0.3750 | — | 12.0000 | — | — | — | — | compute |
| enc_fc_o | `fc_int4_g128` | 32 | — | 0.1250 | — | 4.0000 | — | — | — | — | compute |
| enc_outproj | `fc_int4_g128` | 1 | — | 0.2500 | — | 0.2500 | — | — | — | — | compute |
| **TOTAL (audio encoder)** |  |  |  |  | **67.0000** | **67.0000** | **1.00×** |  |  |  |  |

### Vision encoder — S=2304 (INT4 g128, bidirectional MHA, 768×768 image)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| vis_sdpa | `sdpa_micro__prefill (×2 bidir est.)` | 27 | — | 1.2442 | — | 33.5923 | — | — | — | — | compute |
| vis_fc_fc1 | `fc_int4_g128` | 27 | — | 0.5810 | — | 15.6881 | — | — | — | — | compute |
| vis_fc_fc2 | `fc_int4_g128` | 27 | — | 0.5810 | — | 15.6881 | — | — | — | — | compute |
| vis_fc_qkv | `fc_int4_g128` | 27 | — | 0.4666 | — | 12.5971 | — | — | — | — | compute |
| vis_fc_o | `fc_int4_g128` | 27 | — | 0.1555 | — | 4.1990 | — | — | — | — | compute |
| vis_merge_proj | `fc_int4_g128` | 1 | — | 0.3456 | — | 0.3456 | — | — | — | — | compute |
| **TOTAL (vision encoder)** |  |  |  |  | **82.1102** | **82.1102** | **1.00×** |  |  |  |  |

## 6. Thinker text decoder — Prefill

Per-token-size table. Each FC body row is multiplied ×36 (layers). LM_head runs once (last-token logits).

### Prefill S=1024

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_down | `fc_int4_g128` | 36 | 2.4311 | 1.2971 | 87.5208 | 46.6944 | 1.87× | 20979.0 | 15.7 | 53.4% | compute |
| fc_gate | `fc_int4_g128` | 36 | 2.1798 | 1.2971 | 78.4731 | 46.6944 | 1.68× | 23397.8 | 17.5 | 59.5% | compute |
| fc_up | `fc_int4_g128` | 36 | 2.1703 | 1.2971 | 78.1321 | 46.6944 | 1.67× | 23499.9 | 17.6 | 59.8% | compute |
| pa_compute_S1024 | `sdpa_micro__prefill` | 36 | 1.8172 | 0.4373 | 65.4182 | 15.7440 | 4.16× | 4731.7 | 11.5 | 24.1% | compute |
| fc_qkv | `fc_int4_g128` | 36 | 1.5228 | 0.8192 | 54.8202 | 29.4912 | 1.86× | 21153.6 | 17.1 | 53.8% | compute |
| fc_o | `fc_int4_g128` | 36 | 0.9980 | 0.5461 | 35.9289 | 19.6608 | 1.83× | 21517.3 | 19.2 | 54.7% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.5273 | 0.1526 | 18.9814 | 5.4934 | 3.46× | 47.8 | 31.8 | 28.9% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.2115 | 0.0954 | 15.4385 | 6.9621 | 2.22× | 74.4 | 49.6 | 45.1% | memory |
| residual_add | `eltwise` | 72 | 0.1368 | 0.1430 | 9.8514 | 10.2951 | 0.96× | 19.2 | 115.0 | 104.5% | memory |
| rope_q | `rope_opt` | 36 | 0.1833 | 0.1573 | 6.5978 | 5.6623 | 1.17× | 68.7 | 94.4 | 85.8% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.1366 | 0.0381 | 4.9194 | 1.3734 | 3.58× | 46.1 | 30.7 | 27.9% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8232 | 3.6216 | 3.8232 | 3.6216 | 1.06× | 203.5 | 104.2 | 94.7% | memory |
| pa_kv_update_S1024 | `pa_kv_cache_update_ref` | 36 | 0.0667 | 0.0599 | 2.4026 | 2.1555 | 1.11× | 0.0 | 98.7 | 89.7% | memory |
| rope_k | `rope_opt` | 36 | 0.0407 | 0.0429 | 1.4638 | 1.5443 | 0.95× | 77.4 | 116.0 | 105.5% | memory |
| **TOTAL (prefill text decoder)** |  |  |  |  | **463.7714** | **242.0869** | **1.92×** |  |  |  |  |

### Prefill S=2048

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S2048 | `sdpa_micro__prefill` | 36 | 6.8649 | 1.7485 | 247.1381 | 62.9453 | 3.93× | 5007.5 | 6.1 | 25.5% | compute |
| fc_down | `fc_int4_g128` | 36 | 4.9086 | 2.5941 | 176.7109 | 93.3888 | 1.89× | 20780.8 | 12.9 | 52.8% | compute |
| fc_gate | `fc_int4_g128` | 36 | 4.3468 | 2.5941 | 156.4849 | 93.3888 | 1.68× | 23466.8 | 14.6 | 59.7% | compute |
| fc_up | `fc_int4_g128` | 36 | 4.3357 | 2.5941 | 156.0854 | 93.3888 | 1.67× | 23526.8 | 14.6 | 59.8% | compute |
| fc_qkv | `fc_int4_g128` | 36 | 3.2965 | 1.6384 | 118.6739 | 58.9824 | 2.01× | 19543.3 | 13.3 | 49.7% | compute |
| fc_o | `fc_int4_g128` | 36 | 1.8962 | 1.0923 | 68.2638 | 39.3216 | 1.74× | 22650.2 | 17.3 | 57.6% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 1.0463 | 0.3051 | 37.6682 | 10.9841 | 3.43× | 48.1 | 32.1 | 29.2% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.4196 | 0.1907 | 30.6277 | 13.9209 | 2.20× | 75.0 | 50.0 | 45.5% | memory |
| residual_add | `eltwise` | 72 | 0.2871 | 0.2860 | 20.6688 | 20.5902 | 1.00× | 18.3 | 109.6 | 99.6% | memory |
| rope_q | `rope_opt` | 36 | 0.3933 | 0.3146 | 14.1572 | 11.3246 | 1.25× | 64.0 | 88.0 | 80.0% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.2759 | 0.0763 | 9.9320 | 2.7460 | 3.62× | 45.7 | 30.4 | 27.6% | memory |
| pa_kv_update_S2048 | `pa_kv_cache_update_ref` | 36 | 0.1613 | 0.1198 | 5.8053 | 4.3111 | 1.35× | 0.0 | 81.7 | 74.3% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8232 | 3.6216 | 3.8232 | 3.6216 | 1.06× | 203.5 | 104.2 | 94.7% | memory |
| rope_k | `rope_opt` | 36 | 0.0876 | 0.0858 | 3.1537 | 3.0885 | 1.02× | 71.8 | 107.7 | 97.9% | memory |
| **TOTAL (prefill text decoder)** |  |  |  |  | **1049.1931** | **512.0027** | **2.05×** |  |  |  |  |

### Prefill S=4096

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S4096 | `sdpa_micro__prefill` | 36 | 26.1448 | 6.9922 | 941.2132 | 251.7197 | 3.74× | 5258.1 | 3.2 | 26.7% | compute |
| fc_down | `fc_int4_g128` | 36 | 9.6433 | 5.1883 | 347.1603 | 186.7776 | 1.86× | 21155.6 | 11.8 | 53.8% | compute |
| fc_gate | `fc_int4_g128` | 36 | 8.3722 | 5.1883 | 301.4000 | 186.7776 | 1.61× | 24367.6 | 13.6 | 62.0% | compute |
| fc_up | `fc_int4_g128` | 36 | 8.3432 | 5.1883 | 300.3562 | 186.7776 | 1.61× | 24452.3 | 13.6 | 62.2% | compute |
| fc_qkv | `fc_int4_g128` | 36 | 5.5907 | 3.2768 | 201.2641 | 117.9648 | 1.71× | 23047.2 | 14.2 | 58.6% | compute |
| fc_o | `fc_int4_g128` | 36 | 4.0430 | 2.1845 | 145.5482 | 78.6432 | 1.85× | 21246.4 | 14.8 | 54.0% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 2.0875 | 0.6102 | 75.1490 | 21.9656 | 3.42× | 48.2 | 32.1 | 29.2% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.8372 | 0.3813 | 61.1190 | 27.8383 | 2.20× | 75.2 | 50.1 | 45.5% | memory |
| residual_add | `eltwise` | 72 | 0.5838 | 0.5720 | 42.0316 | 41.1804 | 1.02× | 18.0 | 107.8 | 98.0% | memory |
| rope_q | `rope_opt` | 36 | 0.7892 | 0.6291 | 28.4119 | 22.6492 | 1.25× | 63.8 | 87.7 | 79.7% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.5414 | 0.1525 | 19.4898 | 5.4914 | 3.55× | 46.6 | 31.0 | 28.2% | memory |
| pa_kv_update_S4096 | `pa_kv_cache_update_ref` | 36 | 0.2989 | 0.2395 | 10.7612 | 8.6222 | 1.25× | 0.0 | 88.1 | 80.1% | memory |
| rope_k | `rope_opt` | 36 | 0.1931 | 0.1716 | 6.9504 | 6.1771 | 1.13× | 65.2 | 97.8 | 88.9% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8232 | 3.6216 | 3.8232 | 3.6216 | 1.06× | 203.5 | 104.2 | 94.7% | memory |
| **TOTAL (prefill text decoder)** |  |  |  |  | **2484.6781** | **1146.2063** | **2.17×** |  |  |  |  |

### Prefill S=8192

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S8192 | `sdpa_micro__prefill` | 36 | 111.9477 | 27.9654 | 4030.1155 | 1006.7558 | 4.00× | 4911.4 | 1.5 | 25.0% | compute |
| fc_down | `fc_int4_g128` | 36 | 19.6010 | 10.3765 | 705.6343 | 373.5552 | 1.89× | 20816.4 | 10.9 | 52.9% | compute |
| fc_gate | `fc_int4_g128` | 36 | 15.8164 | 10.3765 | 569.3909 | 373.5552 | 1.52× | 25797.4 | 13.6 | 65.6% | compute |
| fc_up | `fc_int4_g128` | 36 | 15.8141 | 10.3765 | 569.3084 | 373.5552 | 1.52× | 25801.1 | 13.6 | 65.6% | compute |
| fc_qkv | `fc_int4_g128` | 36 | 11.0885 | 6.5536 | 399.1851 | 235.9296 | 1.69× | 23240.2 | 13.6 | 59.1% | compute |
| fc_o | `fc_int4_g128` | 36 | 7.5348 | 4.3691 | 271.2517 | 157.2864 | 1.72× | 22800.8 | 15.2 | 58.0% | compute |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 4.1984 | 1.2202 | 151.1418 | 43.9285 | 3.44× | 48.0 | 32.0 | 29.1% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 1.7021 | 0.7626 | 124.2567 | 55.6733 | 2.23× | 74.0 | 49.3 | 44.8% | memory |
| residual_add | `eltwise` | 72 | 1.1917 | 1.1439 | 85.8005 | 82.3609 | 1.04× | 17.6 | 105.6 | 96.0% | memory |
| rope_q | `rope_opt` | 36 | 1.7031 | 1.2583 | 61.3106 | 45.2985 | 1.35× | 59.1 | 81.3 | 73.9% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 1.0604 | 0.3051 | 38.1730 | 10.9821 | 3.48× | 47.5 | 31.6 | 28.8% | memory |
| pa_kv_update_S8192 | `pa_kv_cache_update_ref` | 36 | 0.5511 | 0.4790 | 19.8380 | 17.2443 | 1.15× | 0.0 | 95.6 | 86.9% | memory |
| rope_k | `rope_opt` | 36 | 0.4084 | 0.3432 | 14.7011 | 12.3541 | 1.19× | 61.6 | 92.4 | 84.0% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8232 | 3.6216 | 3.8232 | 3.6216 | 1.06× | 203.5 | 104.2 | 94.7% | memory |
| **TOTAL (prefill text decoder)** |  |  |  |  | **7043.9308** | **2792.1007** | **2.52×** |  |  |  |  |

## 7. Thinker text decoder — Decode (per output token, TPOT)

### Decode KV=1024

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_down | `fc_int4_g128` | 36 | 0.1289 | 0.1187 | 4.6409 | 4.2742 | 1.09× | 386.4 | 101.3 | 92.1% | memory |
| pa_compute_kv1024 | `paged_attention_opt__single_token` | 36 | 0.1283 | 0.0219 | 4.6193 | 0.7882 | 5.86× | 130.8 | 18.8 | 17.1% | memory |
| fc_up | `fc_int4_g128` | 36 | 0.1278 | 0.1187 | 4.6002 | 4.2742 | 1.08× | 389.8 | 102.2 | 92.9% | memory |
| fc_gate | `fc_int4_g128` | 36 | 0.1278 | 0.1187 | 4.5998 | 4.2742 | 1.08× | 389.8 | 102.2 | 92.9% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8232 | 3.6216 | 3.8232 | 3.6216 | 1.06× | 203.5 | 104.2 | 94.7% | memory |
| fc_qkv | `fc_int4_g128` | 36 | 0.0847 | 0.0750 | 3.0501 | 2.7001 | 1.13× | 371.3 | 97.4 | 88.5% | memory |
| fc_o | `fc_int4_g128` | 36 | 0.0557 | 0.0500 | 2.0061 | 1.8006 | 1.11× | 376.3 | 98.7 | 89.8% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.0028 | 0.0001 | 0.2029 | 0.0102 | 19.85× | 5.5 | 5.5 | 5.0% | memory |
| pa_kv_update_kv1024 | `pa_kv_cache_update_ref` | 36 | 0.0047 | 0.0001 | 0.1704 | 0.0021 | 81.62× | 0.0 | 1.4 | 1.2% | memory |
| residual_add | `eltwise` | 72 | 0.0016 | 0.0001 | 0.1169 | 0.0101 | 11.59× | 1.6 | 9.5 | 8.6% | memory |
| pa_finalize_kv1024 | `single_token_finalization` | 36 | 0.0032 | 0.0001 | 0.1152 | 0.0054 | 21.48× | 0.0 | 5.1 | 4.7% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.0025 | 0.0001 | 0.0891 | 0.0020 | 44.21× | 2.5 | 2.5 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.0023 | 0.0002 | 0.0843 | 0.0080 | 10.51× | 10.5 | 10.5 | 9.5% | memory |
| rope_q | `rope_opt` | 36 | 0.0020 | 0.0002 | 0.0720 | 0.0055 | 12.99× | 6.1 | 8.4 | 7.7% | memory |
| rope_k | `rope_opt` | 36 | 0.0019 | 0.0000 | 0.0680 | 0.0015 | 44.95× | 1.6 | 2.4 | 2.2% | memory |
| **TOTAL (decode per token)** |  |  |  |  | **28.2584** | **21.7779** | **1.30×** |  |  |  |  |

### Decode KV=2048

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv2048 | `paged_attention_opt__single_token` | 36 | 0.1835 | 0.0436 | 6.6061 | 1.5711 | 4.20× | 182.8 | 26.2 | 23.8% | memory |
| fc_down | `fc_int4_g128` | 36 | 0.1289 | 0.1187 | 4.6409 | 4.2742 | 1.09× | 386.4 | 101.3 | 92.1% | memory |
| fc_up | `fc_int4_g128` | 36 | 0.1278 | 0.1187 | 4.6002 | 4.2742 | 1.08× | 389.8 | 102.2 | 92.9% | memory |
| fc_gate | `fc_int4_g128` | 36 | 0.1278 | 0.1187 | 4.5998 | 4.2742 | 1.08× | 389.8 | 102.2 | 92.9% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8232 | 3.6216 | 3.8232 | 3.6216 | 1.06× | 203.5 | 104.2 | 94.7% | memory |
| fc_qkv | `fc_int4_g128` | 36 | 0.0847 | 0.0750 | 3.0501 | 2.7001 | 1.13× | 371.3 | 97.4 | 88.5% | memory |
| fc_o | `fc_int4_g128` | 36 | 0.0557 | 0.0500 | 2.0061 | 1.8006 | 1.11× | 376.3 | 98.7 | 89.8% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.0028 | 0.0001 | 0.2029 | 0.0102 | 19.85× | 5.5 | 5.5 | 5.0% | memory |
| pa_kv_update_kv2048 | `pa_kv_cache_update_ref` | 36 | 0.0038 | 0.0001 | 0.1379 | 0.0021 | 66.05× | 0.0 | 1.7 | 1.5% | memory |
| pa_finalize_kv2048 | `single_token_finalization` | 36 | 0.0035 | 0.0001 | 0.1254 | 0.0054 | 23.38× | 0.0 | 4.7 | 4.3% | memory |
| residual_add | `eltwise` | 72 | 0.0016 | 0.0001 | 0.1169 | 0.0101 | 11.59× | 1.6 | 9.5 | 8.6% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.0025 | 0.0001 | 0.0891 | 0.0020 | 44.21× | 2.5 | 2.5 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.0023 | 0.0002 | 0.0843 | 0.0080 | 10.51× | 10.5 | 10.5 | 9.5% | memory |
| rope_q | `rope_opt` | 36 | 0.0020 | 0.0002 | 0.0720 | 0.0055 | 12.99× | 6.1 | 8.4 | 7.7% | memory |
| rope_k | `rope_opt` | 36 | 0.0019 | 0.0000 | 0.0680 | 0.0015 | 44.95× | 1.6 | 2.4 | 2.2% | memory |
| **TOTAL (decode per token)** |  |  |  |  | **30.2229** | **22.5608** | **1.34×** |  |  |  |  |

### Decode KV=4096

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv4096 | `paged_attention_opt__gqa_single_token` | 36 | 0.1673 | 0.0871 | 6.0225 | 3.1368 | 1.92× | 401.1 | 57.3 | 52.1% | memory |
| fc_down | `fc_int4_g128` | 36 | 0.1289 | 0.1187 | 4.6409 | 4.2742 | 1.09× | 386.4 | 101.3 | 92.1% | memory |
| fc_up | `fc_int4_g128` | 36 | 0.1278 | 0.1187 | 4.6002 | 4.2742 | 1.08× | 389.8 | 102.2 | 92.9% | memory |
| fc_gate | `fc_int4_g128` | 36 | 0.1278 | 0.1187 | 4.5998 | 4.2742 | 1.08× | 389.8 | 102.2 | 92.9% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8232 | 3.6216 | 3.8232 | 3.6216 | 1.06× | 203.5 | 104.2 | 94.7% | memory |
| fc_qkv | `fc_int4_g128` | 36 | 0.0847 | 0.0750 | 3.0501 | 2.7001 | 1.13× | 371.3 | 97.4 | 88.5% | memory |
| fc_o | `fc_int4_g128` | 36 | 0.0557 | 0.0500 | 2.0061 | 1.8006 | 1.11× | 376.3 | 98.7 | 89.8% | memory |
| pa_finalize_kv4096 | `single_token_finalization` | 36 | 0.0067 | 0.0001 | 0.2419 | 0.0054 | 45.10× | 0.0 | 2.4 | 2.2% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.0028 | 0.0001 | 0.2029 | 0.0102 | 19.85× | 5.5 | 5.5 | 5.0% | memory |
| pa_kv_update_kv4096 | `pa_kv_cache_update_ref` | 36 | 0.0040 | 0.0001 | 0.1451 | 0.0021 | 69.50× | 0.0 | 1.6 | 1.5% | memory |
| residual_add | `eltwise` | 72 | 0.0016 | 0.0001 | 0.1169 | 0.0101 | 11.59× | 1.6 | 9.5 | 8.6% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.0025 | 0.0001 | 0.0891 | 0.0020 | 44.21× | 2.5 | 2.5 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.0023 | 0.0002 | 0.0843 | 0.0080 | 10.51× | 10.5 | 10.5 | 9.5% | memory |
| rope_q | `rope_opt` | 36 | 0.0020 | 0.0002 | 0.0720 | 0.0055 | 12.99× | 6.1 | 8.4 | 7.7% | memory |
| rope_k | `rope_opt` | 36 | 0.0019 | 0.0000 | 0.0680 | 0.0015 | 44.95× | 1.6 | 2.4 | 2.2% | memory |
| **TOTAL (decode per token)** |  |  |  |  | **29.7630** | **24.1265** | **1.23×** |  |  |  |  |

### Decode KV=8192

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv8192 | `paged_attention_opt__gqa_single_token` | 36 | 0.3076 | 0.1741 | 11.0729 | 6.2682 | 1.77× | 436.4 | 62.3 | 56.6% | memory |
| fc_down | `fc_int4_g128` | 36 | 0.1289 | 0.1187 | 4.6409 | 4.2742 | 1.09× | 386.4 | 101.3 | 92.1% | memory |
| fc_up | `fc_int4_g128` | 36 | 0.1278 | 0.1187 | 4.6002 | 4.2742 | 1.08× | 389.8 | 102.2 | 92.9% | memory |
| fc_gate | `fc_int4_g128` | 36 | 0.1278 | 0.1187 | 4.5998 | 4.2742 | 1.08× | 389.8 | 102.2 | 92.9% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8232 | 3.6216 | 3.8232 | 3.6216 | 1.06× | 203.5 | 104.2 | 94.7% | memory |
| fc_qkv | `fc_int4_g128` | 36 | 0.0847 | 0.0750 | 3.0501 | 2.7001 | 1.13× | 371.3 | 97.4 | 88.5% | memory |
| fc_o | `fc_int4_g128` | 36 | 0.0557 | 0.0500 | 2.0061 | 1.8006 | 1.11× | 376.3 | 98.7 | 89.8% | memory |
| pa_finalize_kv8192 | `single_token_finalization` | 36 | 0.0133 | 0.0001 | 0.4800 | 0.0054 | 89.49× | 0.0 | 1.2 | 1.1% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.0028 | 0.0001 | 0.2029 | 0.0102 | 19.85× | 5.5 | 5.5 | 5.0% | memory |
| pa_kv_update_kv8192 | `pa_kv_cache_update_ref` | 36 | 0.0040 | 0.0001 | 0.1447 | 0.0021 | 69.29× | 0.0 | 1.6 | 1.5% | memory |
| residual_add | `eltwise` | 72 | 0.0016 | 0.0001 | 0.1169 | 0.0101 | 11.59× | 1.6 | 9.5 | 8.6% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.0025 | 0.0001 | 0.0891 | 0.0020 | 44.21× | 2.5 | 2.5 | 2.3% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.0023 | 0.0002 | 0.0843 | 0.0080 | 10.51× | 10.5 | 10.5 | 9.5% | memory |
| rope_q | `rope_opt` | 36 | 0.0020 | 0.0002 | 0.0720 | 0.0055 | 12.99× | 6.1 | 8.4 | 7.7% | memory |
| rope_k | `rope_opt` | 36 | 0.0019 | 0.0000 | 0.0680 | 0.0015 | 44.95× | 1.6 | 2.4 | 2.2% | memory |
| **TOTAL (decode per token)** |  |  |  |  | **35.0511** | **27.2579** | **1.29×** |  |  |  |  |

## 8. Talker decode — per output text token (audio-output overhead, optional)

Talker runs once per output text token when `enable_audio_output=True`. Below is per-token-size cost.

### Talker decode KV=1024

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| talker_fc_qkv | `fc_int4_g128` | 28 | — | 0.0500 | — | 1.4005 | — | — | — | — | memory |
| talker_fc_gate | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_up | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_down | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_o | `fc_int4_g128` | 28 | — | 0.0250 | — | 0.7009 | — | — | — | — | memory |
| talker_pa_compute_kv1024 | `paged_attention_opt__single_token` | 28 | — | 0.0218 | — | 0.6110 | — | — | — | — | memory |
| talker_lm_head | `fc_f16` | 1 | — | 0.1431 | — | 0.1431 | — | — | — | — | memory |
| talker_pa_kv_update_kv1024 | `pa_kv_cache_update_ref` | 28 | — | 0.0001 | — | 0.0016 | — | — | — | — | memory |
| **TOTAL (talker decode per token)** |  |  |  |  | **6.0092** | **6.0092** | **1.00×** |  |  |  |  |

### Talker decode KV=2048

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| talker_fc_qkv | `fc_int4_g128` | 28 | — | 0.0500 | — | 1.4005 | — | — | — | — | memory |
| talker_pa_compute_kv2048 | `paged_attention_opt__single_token` | 28 | — | 0.0436 | — | 1.2199 | — | — | — | — | memory |
| talker_fc_gate | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_up | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_down | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_o | `fc_int4_g128` | 28 | — | 0.0250 | — | 0.7009 | — | — | — | — | memory |
| talker_lm_head | `fc_f16` | 1 | — | 0.1431 | — | 0.1431 | — | — | — | — | memory |
| talker_pa_kv_update_kv2048 | `pa_kv_cache_update_ref` | 28 | — | 0.0001 | — | 0.0016 | — | — | — | — | memory |
| **TOTAL (talker decode per token)** |  |  |  |  | **6.6181** | **6.6181** | **1.00×** |  |  |  |  |

### Talker decode KV=4096

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| talker_pa_compute_kv4096 | `paged_attention_opt__single_token` | 28 | — | 0.0871 | — | 2.4376 | — | — | — | — | memory |
| talker_fc_qkv | `fc_int4_g128` | 28 | — | 0.0500 | — | 1.4005 | — | — | — | — | memory |
| talker_fc_gate | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_up | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_down | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_o | `fc_int4_g128` | 28 | — | 0.0250 | — | 0.7009 | — | — | — | — | memory |
| talker_lm_head | `fc_f16` | 1 | — | 0.1431 | — | 0.1431 | — | — | — | — | memory |
| talker_pa_kv_update_kv4096 | `pa_kv_cache_update_ref` | 28 | — | 0.0001 | — | 0.0016 | — | — | — | — | memory |
| **TOTAL (talker decode per token)** |  |  |  |  | **7.8358** | **7.8358** | **1.00×** |  |  |  |  |

### Talker decode KV=8192

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| talker_pa_compute_kv8192 | `paged_attention_opt__single_token` | 28 | — | 0.1740 | — | 4.8732 | — | — | — | — | memory |
| talker_fc_qkv | `fc_int4_g128` | 28 | — | 0.0500 | — | 1.4005 | — | — | — | — | memory |
| talker_fc_gate | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_up | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_down | `fc_int4_g128` | 28 | — | 0.0375 | — | 1.0507 | — | — | — | — | memory |
| talker_fc_o | `fc_int4_g128` | 28 | — | 0.0250 | — | 0.7009 | — | — | — | — | memory |
| talker_lm_head | `fc_f16` | 1 | — | 0.1431 | — | 0.1431 | — | — | — | — | memory |
| talker_pa_kv_update_kv8192 | `pa_kv_cache_update_ref` | 28 | — | 0.0001 | — | 0.0016 | — | — | — | — | memory |
| **TOTAL (talker decode per token)** |  |  |  |  | **10.2714** | **10.2714** | **1.00×** |  |  |  |  |

## 9. End-to-end token latency summary

### Prefill — TTFT (text-only / +audio input / +audio+vision input)

| S (text ctx) | Thinker prefill (ms) | + Audio enc (ms) | + Vision enc (ms) | **TTFT text-only** | **TTFT +audio** | **TTFT +A+V** |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 463.7714 | 67.0000 | 82.1102 | **463.7714** | **530.7714** | **612.8816** |
| 2048 | 1049.1931 | 67.0000 | 82.1102 | **1049.1931** | **1116.1931** | **1198.3033** |
| 4096 | 2484.6781 | 67.0000 | 82.1102 | **2484.6781** | **2551.6781** | **2633.7883** |
| 8192 | 7043.9308 | 67.0000 | 82.1102 | **7043.9308** | **7110.9308** | **7193.0410** |

### Decode — TPOT (per output token; 512 output-token total = 512 × TPOT)

| KV (ctx) | Thinker TPOT (ms) | + Talker TPOT (ms) | tokens/s (thinker only) | tokens/s (text+audio out) |
|---:|---:|---:|---:|---:|
| 1024 | 28.2584 | 6.0092 | 35.4 | 29.2 |
| 2048 | 30.2229 | 6.6181 | 33.1 | 27.1 |
| 4096 | 29.7630 | 7.8358 | 33.6 | 26.6 |
| 8192 | 35.0511 | 10.2714 | 28.5 | 22.1 |

## 10. Measured vs. theoretical aggregate eff%

The aggregate slowdown = measured total / theoretical total at each phase. Eff% is the inverse.

### Prefill aggregate

| S | Measured (ms) | Theoretical roofline (ms) | Slowdown | Aggregate eff% |
|---:|---:|---:|---:|---:|
| 1024 | 463.7714 | 242.0869 | 1.92× | 52.2% |
| 2048 | 1049.1931 | 512.0027 | 2.05× | 48.8% |
| 4096 | 2484.6781 | 1146.2063 | 2.17× | 46.1% |
| 8192 | 7043.9308 | 2792.1007 | 2.52× | 39.6% |

### Decode aggregate

| KV | Measured (ms) | Theoretical roofline (ms) | Slowdown | Aggregate eff% |
|---:|---:|---:|---:|---:|
| 1024 | 28.2584 | 21.7779 | 1.30× | 77.1% |
| 2048 | 30.2229 | 22.5608 | 1.34× | 74.6% |
| 4096 | 29.7630 | 24.1265 | 1.23× | 81.1% |
| 8192 | 35.0511 | 27.2579 | 1.29× | 77.8% |

## 11. Reproduction commands

**Run full sweep on PTL 4Xe Linux target (intel@10.239.152.140):**
```bash
scp utils/run_qwen3_omni_ptl_4xe.sh intel@10.239.152.140:~/river/roofline_test_utils/
ssh intel@10.239.152.140 'cd ~/river/roofline_test_utils && bash run_qwen3_omni_ptl_4xe.sh'
```

**Pull logs back, then re-run report locally:**
```bash
scp -r 'intel@10.239.152.140:~/river/roofline_results/qwen3_omni/ptl_4xe/*' logs_ptl_4xe/
python3 parse_logs.py logs_ptl_4xe parsed.json
python3 build_report.py
```

## 12. Caveats

- **KV cache scale/zp accounting**: PA OCL INT8 layout stores 4B fp16 scale+zp per 16-token block (K) and per HD-row (V). Roofline bytes include this.
- **Talker code-predictor / code2wav** are reported theoretical-only (small additional overhead per output text token). For full streaming-TTS performance measurement, a dedicated bench is required.
- **Vision encoder HD=72** is non-standard; some SDPA paths may pad to HD=80. The measured row (if present) reflects what the GPU plugin actually launches.
- **Bidirectional encoder SDPA** is approximated as `2× causal sdpa_micro__prefill`. Real bidirectional kernels may be ~1.5–2× depending on tile reuse.
- **Encoders & Talker are theoretical-only on this PTL 4Xe run**: only the Thinker text decoder (FC + PA + small ops) was benched. The Audio encoder (S=1500), Vision encoder (S=2304) and Talker rows use the analytic INT4-g128 / PA roofline lower bound (marked `—` in the measured columns). They can be measured later by adding the encoder FC/SDPA cases to the 4Xe sweep.
- **Reported max clock is 2450 MHz** (clinfo/cliloader); this report uses the documented 2400 MHz nominal for the FP16/INT8 XMX peaks to stay consistent with the PTL 12Xe report. Using 2450 MHz would lower every compute-bound eff% by ~2%.
- **Prefill FC compute ceiling = INT8 XMX (39.3 TOPS)**: prefill INT4-g128 matmul dynamically quantizes activations to INT8 and decompresses int4→int8 weights, so it executes on the INT8 XMX pipe (2× FP16). Prefill FC eff% is therefore scored against INT8 peak (measured ~21 TFLOP/s ≈ 53–60% of INT8 peak). Decode FC (M=1) is memory-bound and scored against the 110 GB/s VRAM ceiling.
- **Theoretical bytes** for FC use the actual VRAM-streamed footprint (compressed weights + fp16 scales + int8 zp + activations), which is what the BW-roofline must compare against, not the post-decompress fp16 weight size.
