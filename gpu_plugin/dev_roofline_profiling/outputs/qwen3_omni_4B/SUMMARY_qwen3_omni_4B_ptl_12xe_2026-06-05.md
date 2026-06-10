# Qwen3-Omni-4B — Roofline on PTL 12Xe (2026-06-05)

**Platform**: Intel PTL 12Xe iGPU (12 Xe cores × 8 EU × 10 threads), 2400 MHz, 110 GB/s
**Model**: Qwen3-Omni-4B-Instruct-multilingual-int4 (Thinker text + Audio encoder + Vision encoder + Talker + code2wav).
**Quantization**: matmul INT4 g128 / LM_head INT8 g128 / KV cache INT8 / activations FP16. PA opencl + micro_kernel.
**Source config**: `~/workspace/remote_debug/qwen3_omni/models/Qwen3-Omni-4B-Instruct-multilingual-int4/config.json`

**Inputs evaluated**: thinker-text prefill context = 512 / 1024 / 4096 / 8192 tokens; output = 512 tokens. Audio encoder runs once over S=1500 mel frames. Vision encoder runs once over S=2304 patches (768×768 image).

## 1. Hardware peaks (per SKILL.md formulas)

`FP16 XMX TFLOPS = xe_cores × 8 × 256 × freq_GHz`; `INT8 XMX = 2× FP16`; `SIMD FP16 = xe_cores × 8 × 32 × freq`.

| Platform | Xe Cores | Freq (MHz) | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | SIMD FP16 (TFLOPS) | Ridge FP16 (FLOP/B) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PTL 12Xe | 12 | 2400 | 110 | 58.982 | 117.965 | 7.373 | 536 |

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
| enc_sdpa | `sdpa_micro__prefill (×2 bidir est.)` | 32 | 1.1595 | 0.1953 | 37.1039 | 6.2500 | 5.94× | 9935.4 | 13.2 | 16.8% | compute |
| enc_fc_fc2 | `fc_int4_g128` | 32 | 0.4775 | 0.3333 | 15.2786 | 10.6667 | 1.43× | 41178.2 | 47.4 | 69.8% | compute |
| enc_fc_fc1 | `fc_int4_g128` | 32 | 0.4551 | 0.3333 | 14.5647 | 10.6667 | 1.37× | 43196.7 | 49.7 | 73.2% | compute |
| enc_fc_qkv | `fc_int4_g128` | 32 | 0.3421 | 0.2500 | 10.9486 | 8.0000 | 1.37× | 43097.8 | 52.4 | 73.1% | compute |
| enc_fc_o | `fc_int4_g128` | 32 | 0.1654 | 0.0833 | 5.2930 | 2.6667 | 1.98× | 29715.8 | 51.6 | 50.4% | compute |
| enc_outproj | `fc_int4_g128` | 1 | 0.2346 | 0.1667 | 0.2346 | 0.1667 | 1.41× | 41901.4 | 56.4 | 71.0% | compute |
| **TOTAL (audio encoder)** |  |  |  |  | **83.4234** | **38.4168** | **2.17×** |  |  |  |  |

### Vision encoder — S=2304 (INT4 g128, bidirectional MHA, 768×768 image)

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| vis_fc_fc1 | `fc_int4_g128` | 27 | 0.6009 | 0.3874 | 16.2250 | 10.4587 | 1.55× | 38020.4 | 46.2 | 64.5% | compute |
| vis_fc_qkv | `fc_int4_g128` | 27 | 0.4809 | 0.3110 | 12.9848 | 8.3981 | 1.55× | 38147.7 | 48.5 | 64.7% | compute |
| vis_sdpa | `sdpa_micro__prefill (×2 bidir est.)` | 27 | — | 0.4147 | — | 11.1974 | — | — | — | — | compute |
| vis_fc_fc2 | `fc_int4_g128` | 27 | — | 0.3874 | — | 10.4587 | — | — | — | — | compute |
| vis_fc_o | `fc_int4_g128` | 27 | 0.2103 | 0.1037 | 5.6787 | 2.7994 | 2.03× | 29075.9 | 53.8 | 49.3% | compute |
| vis_merge_proj | `fc_int4_g128` | 1 | 0.3345 | 0.2304 | 0.3345 | 0.2304 | 1.45× | 40624.3 | 43.1 | 68.9% | compute |
| **TOTAL (vision encoder)** |  |  |  |  | **56.8791** | **43.5427** | **1.31×** |  |  |  |  |

## 6. Thinker text decoder — Prefill

Per-token-size table. Each FC body row is multiplied ×36 (layers). LM_head runs once (last-token logits).

### Prefill S=512

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_down | `fc_int4_g128` | 36 | 0.5685 | 0.4324 | 20.4667 | 15.5648 | 1.31× | 44855.8 | 45.1 | 76.0% | compute |
| fc_gate | `fc_int4_g128` | 36 | 0.5443 | 0.4324 | 19.5931 | 15.5648 | 1.26× | 46855.6 | 47.1 | 79.4% | compute |
| fc_up | `fc_int4_g128` | 36 | 0.5437 | 0.4324 | 19.5744 | 15.5648 | 1.26× | 46900.4 | 47.1 | 79.5% | compute |
| fc_qkv | `fc_int4_g128` | 36 | 0.3491 | 0.2731 | 12.5692 | 9.8304 | 1.28× | 46130.3 | 49.1 | 78.2% | compute |
| fc_o | `fc_int4_g128` | 36 | 0.2646 | 0.1820 | 9.5271 | 6.5536 | 1.45× | 40573.5 | 46.5 | 68.8% | compute |
| pa_compute_S512 | `sdpa_micro__prefill` | 36 | 0.1868 | 0.0953 | 6.7236 | 3.4317 | 1.96× | 11520.6 | 56.1 | 51.0% | memory |
| residual_add | `eltwise` | 72 | 0.0557 | 0.0715 | 4.0110 | 5.1476 | 0.78× | 23.5 | 141.2 | 128.3% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8722 | 3.6216 | 3.8722 | 3.6216 | 1.07× | 200.9 | 102.9 | 93.5% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.0932 | 0.0763 | 3.3545 | 2.7480 | 1.22× | 135.1 | 90.1 | 81.9% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.0405 | 0.0477 | 2.9555 | 3.4828 | 0.85× | 194.4 | 129.6 | 117.8% | memory |
| rope_q | `rope_opt` | 36 | 0.0569 | 0.0786 | 2.0484 | 2.8312 | 0.72× | 110.6 | 152.0 | 138.2% | memory |
| pa_kv_update_S512 | `pa_kv_cache_update_ref` | 36 | 0.0338 | 0.0299 | 1.2156 | 1.0778 | 1.13× | 0.0 | 97.5 | 88.7% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.0243 | 0.0191 | 0.8733 | 0.6870 | 1.27× | 129.9 | 86.5 | 78.7% | memory |
| rope_k | `rope_opt` | 36 | 0.0179 | 0.0214 | 0.6453 | 0.7721 | 0.84× | 87.7 | 131.6 | 119.6% | memory |
| **TOTAL (prefill text decoder)** |  |  |  |  | **107.4299** | **86.8782** | **1.24×** |  |  |  |  |

### Prefill S=1024

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_down | `fc_int4_g128` | 36 | 1.0152 | 0.8647 | 36.5466 | 31.1296 | 1.17× | 50239.9 | 37.6 | 85.2% | compute |
| fc_up | `fc_int4_g128` | 36 | 0.9144 | 0.8647 | 32.9182 | 31.1296 | 1.06× | 55777.6 | 41.8 | 94.6% | compute |
| fc_gate | `fc_int4_g128` | 36 | 0.9028 | 0.8647 | 32.5018 | 31.1296 | 1.04× | 56492.1 | 42.3 | 95.8% | compute |
| pa_compute_S1024 | `sdpa_micro__prefill` | 36 | 0.6386 | 0.1906 | 22.9891 | 6.8634 | 3.35× | 13464.6 | 32.8 | 29.9% | memory |
| fc_qkv | `fc_int4_g128` | 36 | 0.5927 | 0.5461 | 21.3357 | 19.6608 | 1.09× | 54352.3 | 44.0 | 92.1% | compute |
| fc_o | `fc_int4_g128` | 36 | 0.4301 | 0.3641 | 15.4854 | 13.1072 | 1.18× | 49924.1 | 44.5 | 84.6% | compute |
| residual_add | `eltwise` | 72 | 0.1085 | 0.1430 | 7.8114 | 10.2951 | 0.76× | 24.2 | 145.0 | 131.8% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.1853 | 0.1526 | 6.6698 | 5.4934 | 1.21× | 135.9 | 90.6 | 82.4% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.0773 | 0.0954 | 5.6395 | 6.9621 | 0.81× | 203.7 | 135.8 | 123.5% | memory |
| rope_q | `rope_opt` | 36 | 0.1093 | 0.1573 | 3.9342 | 5.6623 | 0.69× | 115.1 | 158.3 | 143.9% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8722 | 3.6216 | 3.8722 | 3.6216 | 1.07× | 200.9 | 102.9 | 93.5% | memory |
| pa_kv_update_S1024 | `pa_kv_cache_update_ref` | 36 | 0.0543 | 0.0599 | 1.9536 | 2.1555 | 0.91× | 0.0 | 121.4 | 110.3% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.0485 | 0.0381 | 1.7477 | 1.3734 | 1.27× | 129.8 | 86.4 | 78.6% | memory |
| rope_k | `rope_opt` | 36 | 0.0315 | 0.0429 | 1.1332 | 1.5443 | 0.73× | 99.9 | 149.9 | 136.3% | memory |
| **TOTAL (prefill text decoder)** |  |  |  |  | **194.5384** | **170.1279** | **1.14×** |  |  |  |  |

### Prefill S=4096

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S4096 | `sdpa_micro__prefill` | 36 | 8.9895 | 2.3307 | 323.6225 | 83.9066 | 3.86× | 15292.5 | 9.3 | 25.9% | compute |
| fc_down | `fc_int4_g128` | 36 | 3.7368 | 3.4588 | 134.5264 | 124.5184 | 1.08× | 54594.4 | 30.4 | 92.6% | compute |
| fc_up | `fc_int4_g128` | 36 | 3.5996 | 3.4588 | 129.5865 | 124.5184 | 1.04× | 56675.6 | 31.6 | 96.1% | compute |
| fc_gate | `fc_int4_g128` | 36 | 3.5756 | 3.4588 | 128.7217 | 124.5184 | 1.03× | 57056.4 | 31.8 | 96.7% | compute |
| fc_qkv | `fc_int4_g128` | 36 | 2.2948 | 2.1845 | 82.6141 | 78.6432 | 1.05× | 56147.4 | 34.7 | 95.2% | compute |
| fc_o | `fc_int4_g128` | 36 | 1.5489 | 1.4564 | 55.7587 | 52.4288 | 1.06× | 55460.0 | 38.8 | 94.0% | compute |
| residual_add | `eltwise` | 72 | 0.5721 | 0.5720 | 41.1899 | 41.1804 | 1.00× | 18.3 | 110.0 | 100.0% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.7735 | 0.6102 | 27.8443 | 21.9656 | 1.27× | 130.2 | 86.8 | 78.9% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.3655 | 0.3813 | 26.6821 | 27.8383 | 0.96× | 172.2 | 114.8 | 104.3% | memory |
| rope_q | `rope_opt` | 36 | 0.6158 | 0.6291 | 22.1700 | 22.6492 | 0.98× | 81.7 | 112.4 | 102.2% | memory |
| pa_kv_update_S4096 | `pa_kv_cache_update_ref` | 36 | 0.2377 | 0.2395 | 8.5564 | 8.6222 | 0.99× | 0.0 | 110.8 | 100.8% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.1857 | 0.1525 | 6.6855 | 5.4914 | 1.22× | 135.7 | 90.3 | 82.1% | memory |
| rope_k | `rope_opt` | 36 | 0.1229 | 0.1716 | 4.4244 | 6.1771 | 0.72× | 102.4 | 153.6 | 139.6% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8722 | 3.6216 | 3.8722 | 3.6216 | 1.07× | 200.9 | 102.9 | 93.5% | memory |
| **TOTAL (prefill text decoder)** |  |  |  |  | **996.2547** | **726.0796** | **1.37×** |  |  |  |  |

### Prefill S=8192

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_S8192 | `sdpa_micro__prefill` | 36 | 35.0609 | 9.3218 | 1262.1937 | 335.5853 | 3.76× | 15681.9 | 4.8 | 26.6% | compute |
| fc_down | `fc_int4_g128` | 36 | 7.5063 | 6.9177 | 270.2263 | 249.0368 | 1.09× | 54357.4 | 28.6 | 92.2% | compute |
| fc_up | `fc_int4_g128` | 36 | 6.9981 | 6.9177 | 251.9321 | 249.0368 | 1.01× | 58304.6 | 30.6 | 98.9% | compute |
| fc_gate | `fc_int4_g128` | 36 | 6.9067 | 6.9177 | 248.6398 | 249.0368 | 1.00× | 59076.6 | 31.0 | 100.2% | compute |
| fc_qkv | `fc_int4_g128` | 36 | 4.7312 | 4.3691 | 170.3229 | 157.2864 | 1.08× | 54467.9 | 31.9 | 92.3% | compute |
| fc_o | `fc_int4_g128` | 36 | 2.9298 | 2.9127 | 105.4740 | 104.8576 | 1.01× | 58637.7 | 39.1 | 99.4% | compute |
| residual_add | `eltwise` | 72 | 1.2286 | 1.1439 | 88.4592 | 82.3609 | 1.07× | 17.1 | 102.4 | 93.1% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 1.6838 | 1.2202 | 60.6165 | 43.9285 | 1.38× | 119.6 | 79.7 | 72.5% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.8095 | 0.7626 | 59.0904 | 55.6733 | 1.06× | 155.6 | 103.6 | 94.2% | memory |
| rope_q | `rope_opt` | 36 | 1.3356 | 1.2583 | 48.0815 | 45.2985 | 1.06× | 75.4 | 103.6 | 94.2% | memory |
| pa_kv_update_S8192 | `pa_kv_cache_update_ref` | 36 | 0.4913 | 0.4790 | 17.6854 | 17.2443 | 1.03× | 0.0 | 107.3 | 97.5% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.3845 | 0.3051 | 13.8403 | 10.9821 | 1.26× | 131.1 | 87.3 | 79.3% | memory |
| rope_k | `rope_opt` | 36 | 0.3064 | 0.3432 | 11.0313 | 12.3541 | 0.89× | 82.1 | 123.2 | 112.0% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8722 | 3.6216 | 3.8722 | 3.6216 | 1.07× | 200.9 | 102.9 | 93.5% | memory |
| **TOTAL (prefill text decoder)** |  |  |  |  | **2611.4656** | **1616.3030** | **1.62×** |  |  |  |  |

## 7. Thinker text decoder — Decode (per output token, TPOT)

### Decode KV=512

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_gate | `fc_int4_g128` | 36 | 0.1292 | 0.1187 | 4.6508 | 4.2742 | 1.09× | 385.5 | 101.1 | 91.9% | memory |
| fc_down | `fc_int4_g128` | 36 | 0.1291 | 0.1187 | 4.6489 | 4.2742 | 1.09× | 385.7 | 101.1 | 91.9% | memory |
| fc_up | `fc_int4_g128` | 36 | 0.1289 | 0.1187 | 4.6405 | 4.2742 | 1.09× | 386.4 | 101.3 | 92.1% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8722 | 3.6216 | 3.8722 | 3.6216 | 1.07× | 200.9 | 102.9 | 93.5% | memory |
| fc_qkv | `fc_int4_g128` | 36 | 0.0825 | 0.0750 | 2.9713 | 2.7001 | 1.10× | 381.1 | 100.0 | 90.9% | memory |
| fc_o | `fc_int4_g128` | 36 | 0.0565 | 0.0500 | 2.0339 | 1.8006 | 1.13× | 371.2 | 97.4 | 88.5% | memory |
| pa_compute_kv512 | `paged_attention_opt__single_token` | 36 | 0.0330 | 0.0110 | 1.1884 | 0.3968 | 2.99× | 254.1 | 36.7 | 33.4% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.0029 | 0.0001 | 0.2111 | 0.0102 | 20.66× | 5.3 | 5.3 | 4.8% | memory |
| pa_kv_update_kv512 | `pa_kv_cache_update_ref` | 36 | 0.0058 | 0.0001 | 0.2081 | 0.0021 | 99.67× | 0.0 | 1.1 | 1.0% | memory |
| residual_add | `eltwise` | 72 | 0.0019 | 0.0001 | 0.1392 | 0.0101 | 13.81× | 1.3 | 8.0 | 7.2% | memory |
| pa_finalize_kv512 | `single_token_finalization` | 36 | 0.0034 | 0.0001 | 0.1209 | 0.0054 | 22.53× | 0.0 | 4.9 | 4.4% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.0025 | 0.0001 | 0.0895 | 0.0020 | 44.41× | 2.5 | 2.5 | 2.2% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.0024 | 0.0002 | 0.0861 | 0.0080 | 10.73× | 10.3 | 10.3 | 9.3% | memory |
| rope_q | `rope_opt` | 36 | 0.0021 | 0.0002 | 0.0773 | 0.0055 | 13.94× | 5.7 | 7.9 | 7.2% | memory |
| rope_k | `rope_opt` | 36 | 0.0020 | 0.0000 | 0.0703 | 0.0015 | 46.50× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL (decode per token)** |  |  |  |  | **25.0085** | **21.3865** | **1.17×** |  |  |  |  |

### Decode KV=1024

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_gate | `fc_int4_g128` | 36 | 0.1292 | 0.1187 | 4.6508 | 4.2742 | 1.09× | 385.5 | 101.1 | 91.9% | memory |
| fc_down | `fc_int4_g128` | 36 | 0.1291 | 0.1187 | 4.6489 | 4.2742 | 1.09× | 385.7 | 101.1 | 91.9% | memory |
| fc_up | `fc_int4_g128` | 36 | 0.1289 | 0.1187 | 4.6405 | 4.2742 | 1.09× | 386.4 | 101.3 | 92.1% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8722 | 3.6216 | 3.8722 | 3.6216 | 1.07× | 200.9 | 102.9 | 93.5% | memory |
| fc_qkv | `fc_int4_g128` | 36 | 0.0825 | 0.0750 | 2.9713 | 2.7001 | 1.10× | 381.1 | 100.0 | 90.9% | memory |
| fc_o | `fc_int4_g128` | 36 | 0.0565 | 0.0500 | 2.0339 | 1.8006 | 1.13× | 371.2 | 97.4 | 88.5% | memory |
| pa_compute_kv1024 | `paged_attention_opt__single_token` | 36 | 0.0514 | 0.0219 | 1.8512 | 0.7882 | 2.35× | 326.3 | 46.8 | 42.6% | memory |
| pa_kv_update_kv1024 | `pa_kv_cache_update_ref` | 36 | 0.0062 | 0.0001 | 0.2236 | 0.0021 | 107.10× | 0.0 | 1.0 | 0.9% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.0029 | 0.0001 | 0.2111 | 0.0102 | 20.66× | 5.3 | 5.3 | 4.8% | memory |
| residual_add | `eltwise` | 72 | 0.0019 | 0.0001 | 0.1392 | 0.0101 | 13.81× | 1.3 | 8.0 | 7.2% | memory |
| pa_finalize_kv1024 | `single_token_finalization` | 36 | 0.0036 | 0.0001 | 0.1281 | 0.0054 | 23.87× | 0.0 | 4.6 | 4.2% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.0025 | 0.0001 | 0.0895 | 0.0020 | 44.41× | 2.5 | 2.5 | 2.2% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.0024 | 0.0002 | 0.0861 | 0.0080 | 10.73× | 10.3 | 10.3 | 9.3% | memory |
| rope_q | `rope_opt` | 36 | 0.0021 | 0.0002 | 0.0773 | 0.0055 | 13.94× | 5.7 | 7.9 | 7.2% | memory |
| rope_k | `rope_opt` | 36 | 0.0020 | 0.0000 | 0.0703 | 0.0015 | 46.50× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL (decode per token)** |  |  |  |  | **25.6940** | **21.7779** | **1.18×** |  |  |  |  |

### Decode KV=4096

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fc_gate | `fc_int4_g128` | 36 | 0.1292 | 0.1187 | 4.6508 | 4.2742 | 1.09× | 385.5 | 101.1 | 91.9% | memory |
| fc_down | `fc_int4_g128` | 36 | 0.1291 | 0.1187 | 4.6489 | 4.2742 | 1.09× | 385.7 | 101.1 | 91.9% | memory |
| fc_up | `fc_int4_g128` | 36 | 0.1289 | 0.1187 | 4.6405 | 4.2742 | 1.09× | 386.4 | 101.3 | 92.1% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8722 | 3.6216 | 3.8722 | 3.6216 | 1.07× | 200.9 | 102.9 | 93.5% | memory |
| pa_compute_kv4096 | `paged_attention_opt__gqa_single_token` | 36 | 0.1020 | 0.0871 | 3.6712 | 3.1368 | 1.17× | 658.1 | 94.0 | 85.4% | memory |
| fc_qkv | `fc_int4_g128` | 36 | 0.0825 | 0.0750 | 2.9713 | 2.7001 | 1.10× | 381.1 | 100.0 | 90.9% | memory |
| fc_o | `fc_int4_g128` | 36 | 0.0565 | 0.0500 | 2.0339 | 1.8006 | 1.13× | 371.2 | 97.4 | 88.5% | memory |
| pa_kv_update_kv4096 | `pa_kv_cache_update_ref` | 36 | 0.0064 | 0.0001 | 0.2296 | 0.0021 | 109.98× | 0.0 | 1.0 | 0.9% | memory |
| pa_finalize_kv4096 | `single_token_finalization` | 36 | 0.0059 | 0.0001 | 0.2111 | 0.0054 | 39.35× | 0.0 | 2.8 | 2.5% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.0029 | 0.0001 | 0.2111 | 0.0102 | 20.66× | 5.3 | 5.3 | 4.8% | memory |
| residual_add | `eltwise` | 72 | 0.0019 | 0.0001 | 0.1392 | 0.0101 | 13.81× | 1.3 | 8.0 | 7.2% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.0025 | 0.0001 | 0.0895 | 0.0020 | 44.41× | 2.5 | 2.5 | 2.2% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.0024 | 0.0002 | 0.0861 | 0.0080 | 10.73× | 10.3 | 10.3 | 9.3% | memory |
| rope_q | `rope_opt` | 36 | 0.0021 | 0.0002 | 0.0773 | 0.0055 | 13.94× | 5.7 | 7.9 | 7.2% | memory |
| rope_k | `rope_opt` | 36 | 0.0020 | 0.0000 | 0.0703 | 0.0015 | 46.50× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL (decode per token)** |  |  |  |  | **27.6030** | **24.1265** | **1.14×** |  |  |  |  |

### Decode KV=8192

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| pa_compute_kv8192 | `paged_attention_opt__gqa_single_token` | 36 | 0.1991 | 0.1741 | 7.1687 | 6.2682 | 1.14× | 674.0 | 96.2 | 87.4% | memory |
| fc_gate | `fc_int4_g128` | 36 | 0.1292 | 0.1187 | 4.6508 | 4.2742 | 1.09× | 385.5 | 101.1 | 91.9% | memory |
| fc_down | `fc_int4_g128` | 36 | 0.1291 | 0.1187 | 4.6489 | 4.2742 | 1.09× | 385.7 | 101.1 | 91.9% | memory |
| fc_up | `fc_int4_g128` | 36 | 0.1289 | 0.1187 | 4.6405 | 4.2742 | 1.09× | 386.4 | 101.3 | 92.1% | memory |
| fc_lm_head | `fc_int8_g128` | 1 | 3.8722 | 3.6216 | 3.8722 | 3.6216 | 1.07× | 200.9 | 102.9 | 93.5% | memory |
| fc_qkv | `fc_int4_g128` | 36 | 0.0825 | 0.0750 | 2.9713 | 2.7001 | 1.10× | 381.1 | 100.0 | 90.9% | memory |
| fc_o | `fc_int4_g128` | 36 | 0.0565 | 0.0500 | 2.0339 | 1.8006 | 1.13× | 371.2 | 97.4 | 88.5% | memory |
| pa_finalize_kv8192 | `single_token_finalization` | 36 | 0.0102 | 0.0001 | 0.3667 | 0.0054 | 68.36× | 0.0 | 1.6 | 1.5% | memory |
| pa_kv_update_kv8192 | `pa_kv_cache_update_ref` | 36 | 0.0068 | 0.0001 | 0.2438 | 0.0021 | 116.78× | 0.0 | 0.9 | 0.9% | memory |
| rmsnorm_hidden | `rms_gpu_bfyx_opt` | 73 | 0.0029 | 0.0001 | 0.2111 | 0.0102 | 20.66× | 5.3 | 5.3 | 4.8% | memory |
| residual_add | `eltwise` | 72 | 0.0019 | 0.0001 | 0.1392 | 0.0101 | 13.81× | 1.3 | 8.0 | 7.2% | memory |
| rmsnorm3d_k | `rms_gpu_bfyx_opt` | 36 | 0.0025 | 0.0001 | 0.0895 | 0.0020 | 44.41× | 2.5 | 2.5 | 2.2% | memory |
| rmsnorm3d_q | `rms_gpu_bfyx_opt` | 36 | 0.0024 | 0.0002 | 0.0861 | 0.0080 | 10.73× | 10.3 | 10.3 | 9.3% | memory |
| rope_q | `rope_opt` | 36 | 0.0021 | 0.0002 | 0.0773 | 0.0055 | 13.94× | 5.7 | 7.9 | 7.2% | memory |
| rope_k | `rope_opt` | 36 | 0.0020 | 0.0000 | 0.0703 | 0.0015 | 46.50× | 1.6 | 2.4 | 2.1% | memory |
| **TOTAL (decode per token)** |  |  |  |  | **31.2703** | **27.2579** | **1.15×** |  |  |  |  |

## 8. Talker decode — per output text token (audio-output overhead, optional)

Talker runs once per output text token when `enable_audio_output=True`. Below is per-token-size cost.

### Talker decode KV=512

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| talker_fc_qkv | `fc_int4_g128` | 28 | 0.0564 | 0.0500 | 1.5790 | 1.4005 | 1.13× | 371.9 | 97.6 | 88.7% | memory |
| talker_fc_up | `fc_int4_g128` | 28 | 0.0431 | 0.0375 | 1.2079 | 1.0507 | 1.15× | 364.6 | 95.7 | 87.0% | memory |
| talker_fc_down | `fc_int4_g128` | 28 | 0.0430 | 0.0375 | 1.2034 | 1.0507 | 1.15× | 366.0 | 96.0 | 87.3% | memory |
| talker_fc_gate | `fc_int4_g128` | 28 | 0.0428 | 0.0375 | 1.1984 | 1.0507 | 1.14× | 367.5 | 96.4 | 87.7% | memory |
| talker_fc_o | `fc_int4_g128` | 28 | 0.0295 | 0.0250 | 0.8248 | 0.7009 | 1.18× | 356.0 | 93.5 | 85.0% | memory |
| talker_pa_compute_kv512 | `paged_attention_opt__single_token` | 28 | 0.0205 | 0.0109 | 0.5752 | 0.3065 | 1.88× | 204.2 | 58.6 | 53.3% | memory |
| talker_lm_head | `fc_f16` | 1 | 0.1548 | 0.1431 | 0.1548 | 0.1431 | 1.08× | 101.6 | 101.7 | 92.4% | memory |
| talker_pa_kv_update_kv512 | `pa_kv_cache_update_ref` | 28 | 0.0055 | 0.0001 | 0.1543 | 0.0016 | 95.00× | 0.0 | 1.2 | 1.1% | memory |
| **TOTAL (talker decode per token)** |  |  |  |  | **6.8978** | **5.7047** | **1.21×** |  |  |  |  |

### Talker decode KV=1024

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| talker_fc_qkv | `fc_int4_g128` | 28 | 0.0564 | 0.0500 | 1.5790 | 1.4005 | 1.13× | 371.9 | 97.6 | 88.7% | memory |
| talker_fc_up | `fc_int4_g128` | 28 | 0.0431 | 0.0375 | 1.2079 | 1.0507 | 1.15× | 364.6 | 95.7 | 87.0% | memory |
| talker_fc_down | `fc_int4_g128` | 28 | 0.0430 | 0.0375 | 1.2034 | 1.0507 | 1.15× | 366.0 | 96.0 | 87.3% | memory |
| talker_fc_gate | `fc_int4_g128` | 28 | 0.0428 | 0.0375 | 1.1984 | 1.0507 | 1.14× | 367.5 | 96.4 | 87.7% | memory |
| talker_pa_compute_kv1024 | `paged_attention_opt__single_token` | 28 | 0.0389 | 0.0218 | 1.0903 | 0.6110 | 1.78× | 215.4 | 61.6 | 56.0% | memory |
| talker_fc_o | `fc_int4_g128` | 28 | 0.0295 | 0.0250 | 0.8248 | 0.7009 | 1.18× | 356.0 | 93.5 | 85.0% | memory |
| talker_pa_kv_update_kv1024 | `pa_kv_cache_update_ref` | 28 | 0.0063 | 0.0001 | 0.1761 | 0.0016 | 108.41× | 0.0 | 1.0 | 0.9% | memory |
| talker_lm_head | `fc_f16` | 1 | 0.1548 | 0.1431 | 0.1548 | 0.1431 | 1.08× | 101.6 | 101.7 | 92.4% | memory |
| **TOTAL (talker decode per token)** |  |  |  |  | **7.4347** | **6.0092** | **1.24×** |  |  |  |  |

### Talker decode KV=4096

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| talker_pa_compute_kv4096 | `paged_attention_opt__single_token` | 28 | 0.1040 | 0.0871 | 2.9122 | 2.4376 | 1.19× | 322.6 | 92.1 | 83.7% | memory |
| talker_fc_qkv | `fc_int4_g128` | 28 | 0.0564 | 0.0500 | 1.5790 | 1.4005 | 1.13× | 371.9 | 97.6 | 88.7% | memory |
| talker_fc_up | `fc_int4_g128` | 28 | 0.0431 | 0.0375 | 1.2079 | 1.0507 | 1.15× | 364.6 | 95.7 | 87.0% | memory |
| talker_fc_down | `fc_int4_g128` | 28 | 0.0430 | 0.0375 | 1.2034 | 1.0507 | 1.15× | 366.0 | 96.0 | 87.3% | memory |
| talker_fc_gate | `fc_int4_g128` | 28 | 0.0428 | 0.0375 | 1.1984 | 1.0507 | 1.14× | 367.5 | 96.4 | 87.7% | memory |
| talker_fc_o | `fc_int4_g128` | 28 | 0.0295 | 0.0250 | 0.8248 | 0.7009 | 1.18× | 356.0 | 93.5 | 85.0% | memory |
| talker_pa_kv_update_kv4096 | `pa_kv_cache_update_ref` | 28 | 0.0061 | 0.0001 | 0.1703 | 0.0016 | 104.84× | 0.0 | 1.1 | 1.0% | memory |
| talker_lm_head | `fc_f16` | 1 | 0.1548 | 0.1431 | 0.1548 | 0.1431 | 1.08× | 101.6 | 101.7 | 92.4% | memory |
| **TOTAL (talker decode per token)** |  |  |  |  | **9.2508** | **7.8358** | **1.18×** |  |  |  |  |

### Talker decode KV=8192

| op | kernel | calls | meas single ms | theo single ms | meas total ms | theo total ms | slowdown | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| talker_pa_compute_kv8192 | `paged_attention_opt__single_token` | 28 | 0.1994 | 0.1740 | 5.5833 | 4.8732 | 1.15× | 336.6 | 96.0 | 87.3% | memory |
| talker_fc_qkv | `fc_int4_g128` | 28 | 0.0564 | 0.0500 | 1.5790 | 1.4005 | 1.13× | 371.9 | 97.6 | 88.7% | memory |
| talker_fc_up | `fc_int4_g128` | 28 | 0.0431 | 0.0375 | 1.2079 | 1.0507 | 1.15× | 364.6 | 95.7 | 87.0% | memory |
| talker_fc_down | `fc_int4_g128` | 28 | 0.0430 | 0.0375 | 1.2034 | 1.0507 | 1.15× | 366.0 | 96.0 | 87.3% | memory |
| talker_fc_gate | `fc_int4_g128` | 28 | 0.0428 | 0.0375 | 1.1984 | 1.0507 | 1.14× | 367.5 | 96.4 | 87.7% | memory |
| talker_fc_o | `fc_int4_g128` | 28 | 0.0295 | 0.0250 | 0.8248 | 0.7009 | 1.18× | 356.0 | 93.5 | 85.0% | memory |
| talker_pa_kv_update_kv8192 | `pa_kv_cache_update_ref` | 28 | 0.0067 | 0.0001 | 0.1882 | 0.0016 | 115.91× | 0.0 | 1.0 | 0.9% | memory |
| talker_lm_head | `fc_f16` | 1 | 0.1548 | 0.1431 | 0.1548 | 0.1431 | 1.08× | 101.6 | 101.7 | 92.4% | memory |
| **TOTAL (talker decode per token)** |  |  |  |  | **11.9398** | **10.2714** | **1.16×** |  |  |  |  |

## 9. End-to-end token latency summary

### Prefill — TTFT (text-only / +audio input / +audio+vision input)

| S (text ctx) | Thinker prefill (ms) | + Audio enc (ms) | + Vision enc (ms) | **TTFT text-only** | **TTFT +audio** | **TTFT +A+V** |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 107.4299 | 83.4234 | 56.8791 | **107.4299** | **190.8533** | **247.7324** |
| 1024 | 194.5384 | 83.4234 | 56.8791 | **194.5384** | **277.9618** | **334.8409** |
| 4096 | 996.2547 | 83.4234 | 56.8791 | **996.2547** | **1079.6781** | **1136.5572** |
| 8192 | 2611.4656 | 83.4234 | 56.8791 | **2611.4656** | **2694.8890** | **2751.7681** |

### Decode — TPOT (per output token; 512 output-token total = 512 × TPOT)

| KV (ctx) | Thinker TPOT (ms) | + Talker TPOT (ms) | tokens/s (thinker only) | tokens/s (text+audio out) |
|---:|---:|---:|---:|---:|
| 512 | 25.0085 | 6.8978 | 40.0 | 31.3 |
| 1024 | 25.6940 | 7.4347 | 38.9 | 30.2 |
| 4096 | 27.6030 | 9.2508 | 36.2 | 27.1 |
| 8192 | 31.2703 | 11.9398 | 32.0 | 23.1 |

## 10. Measured vs. theoretical aggregate eff%

The aggregate slowdown = measured total / theoretical total at each phase. Eff% is the inverse.

### Prefill aggregate

| S | Measured (ms) | Theoretical roofline (ms) | Slowdown | Aggregate eff% |
|---:|---:|---:|---:|---:|
| 512 | 107.4299 | 86.8782 | 1.24× | 80.9% |
| 1024 | 194.5384 | 170.1279 | 1.14× | 87.5% |
| 4096 | 996.2547 | 726.0796 | 1.37× | 72.9% |
| 8192 | 2611.4656 | 1616.3030 | 1.62× | 61.9% |

### Decode aggregate

| KV | Measured (ms) | Theoretical roofline (ms) | Slowdown | Aggregate eff% |
|---:|---:|---:|---:|---:|
| 512 | 25.0085 | 21.3865 | 1.17× | 85.5% |
| 1024 | 25.6940 | 21.7779 | 1.18× | 84.8% |
| 4096 | 27.6030 | 24.1265 | 1.14× | 87.4% |
| 8192 | 31.2703 | 27.2579 | 1.15× | 87.2% |

## 11. Reproduction commands

**Build benches once on Windows PTL 12Xe target:**
```cmd
D:\river\moe\dev_roofline_profiling\utils\build_remote.bat
```

**Run full sweep:**
```cmd
D:\river\moe\dev_roofline_profiling\utils\run_qwen3_omni_4B_ptl_12xe.bat
```

**Pull logs back, then re-run report locally:**
```bash
python3 ../../utils/parse_logs.py logs_ptl_12xe > parsed.json
python3 build_report.py
```

## 12. Caveats

- **KV cache scale/zp accounting**: PA OCL INT8 layout stores 4B fp16 scale+zp per 16-token block (K) and per HD-row (V). Roofline bytes include this.
- **Talker code-predictor / code2wav** are reported theoretical-only (small additional overhead per output text token). For full streaming-TTS performance measurement, a dedicated bench is required.
- **Vision encoder HD=72** is non-standard; some SDPA paths may pad to HD=80. The measured row (if present) reflects what the GPU plugin actually launches.
- **Bidirectional encoder SDPA** is approximated as `2× causal sdpa_micro__prefill`. Real bidirectional kernels may be ~1.5–2× depending on tile reuse.
- **Theoretical fallback for missing measurements**: a few rows could not be measured on PTL 12Xe and use the theoretical value in aggregates (marked with `—` in the per-row meas columns): `vis_sdpa` (`sdpa_micro` does not support `HD=72`, the bench produced 0 enqueues), `vis_fc_fc2` (`fc_int4_g128` requires `K%group_size==0`; `K=4304` is not a multiple of `g128`; fallback uses the theoretical bandwidth-bound INT4 g128 cost).
- **Theoretical bytes** for FC use the actual VRAM-streamed footprint (compressed weights + fp16 scales + int8 zp + activations), which is what the BW-roofline must compare against, not the post-decompress fp16 weight size.
