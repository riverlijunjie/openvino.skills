# Qwen3.6-35B-A3B — Roofline on Intel Arc B390 (Panther Lake, 12Xe iGPU) (20260610)

**Platform**: Intel Arc B390 (Panther Lake, 12Xe iGPU), Xe2; 105 GB/s achievable read (spec 110); FP16 XMX 58.982 TFLOPS, INT8 XMX 117.964 TOPS.
**Model**: text decoder, 40 layers = 10 full-attention + 30 linear-attention (GatedDeltaNet); MoE every layer.

- 40 decoder blocks; full-attn every 4th layer; attn_output_gate=true (fused QKV+gate width = 2·16·256 + 2·2·256 = 9216)
- MoE: 256 experts, top-8, expert intermediate 512, always-on shared expert intermediate 512
- MatMul weights INT4 g128 / FP16 act; LM_head INT8 g128 / FP16 act; KV cache INT8
- SDPA: PagedAttention (OpenCL micro-kernel), GQA 16/2 = 8-way; linear-attn via GatedDeltaNet

## Model parameters & weight shapes

Architecture knobs (parsed from model config):

| Field | Value | Notes |
|---|---:|---|
| `hidden_size` | 2048 | residual / activation channel |
| `num_hidden_layers` | 40 | 10 full-attn + 30 linear-attn |
| `num_attention_heads` (NH) | 16 | full-attn Q heads |
| `num_key_value_heads` (NKV) | 2 | GQA: 8-way Q-per-KV sharing |
| `head_dim` (HD) | 256 | Q_dim = NH·HD = 4096; partial RoPE factor 0.25 |
| `attn_output_gate` | true | q_proj emits [query\|gate]; gate width = 4096 |
| linear K/V heads | 16/32 | GDN k/v head_dim 128; in_proj = 12288 (qkv 8192 + z 4096) |
| `moe_intermediate_size` | 512 | per-expert SwiGLU hidden |
| `num_experts` / `num_experts_per_tok` | 256 / 8 | + 1 always-on shared expert |
| `vocab_size` | 248320 | LM head N |
| `rope_theta` | 10000000 | — |

Per-layer weight matrices (one decoder block) and global weights (INT4 = K·N/2 + FP16 scale + INT4 zp at g128; INT8 = K·N + FP16 scale):

| Weight | Shape (K × N) | Quant | MB / instance | × Count | Total MB |
|---|---:|---|---:|---:|---:|
| Embedding (gather) | 248320 × 2048 | INT8 g128 | 516.51 | 1 | 516.5 |
| FC_QKV+gate (full-attn) | 2048 × 9216 | INT4 g128 | 9.81 | 10 | 98.1 |
| FC_O / GDN out_proj | 4096 × 2048 | INT4 g128 | 4.36 | 40 | 174.3 |
| Lin-attn in_proj (GDN) | 2048 × 12288 | INT4 g128 | 13.07 | 30 | 392.2 |
| MoE expert gate+up | 2048 × 1024 | INT4 g128 | 1.09 | 10240 | 11,156.8 |
| MoE expert down | 512 × 2048 | INT4 g128 | 0.54 | 10240 | 5,578.4 |
| MoE shared gate+up | 2048 × 1024 | INT4 g128 | 1.09 | 40 | 43.6 |
| MoE shared down | 512 × 2048 | INT4 g128 | 0.54 | 40 | 21.8 |
| MoE router | 2048 × 256 | FP16 | 0.27 | 40 | 10.9 |
| LM_Head | 2048 × 248320 | INT8 g128 | 516.51 | 1 | 516.5 |
| **Total static weights** |  |  |  |  | **18.51 GB** |

KV-cache (INT8) per token:

| Tensor | Shape | dtype | Bytes/token/layer | Bytes/token (10 full-attn layers) |
|---|---|---|---:|---:|
| K cache | [blocks, 2, 256, 16] | INT8 | 512 | 5120 |
| V cache | [blocks, 2, 16, 256] | INT8 | 512 | 5120 |
| **KV total** | per token | INT8 | 1024 B/layer | **10240 B/token** |

## Theoretical roofline

| Metric | Value |
|---|---|
| FP16 XMX peak | 58.982 TFLOPS |
| INT8 XMX peak | 117.964 TOPS |
| Memory BW (achievable read) | 105 GB/s (spec 110) |
| Ridge point (FP16) | 562 FLOP/byte |
| Ridge point (INT8) | 1123 OP/byte |

## Data sources

All ops measured natively on this platform via cliloader Device Performance Timing (mean kernel time per iteration); **no cross-platform scaling**. Each op runs in its own process with a cache flush between iterations so weights stream from VRAM. PTL 12Xe and Intel Arc B390 share metrics (same silicon; cliloader id difference only).

## Graph fusion notes

| Bench row | Real graph behaviour | Standalone kernel in graph? |
|---|---|---|
| `moe` | gate/up/down SwiGLU experts fused into `moe_3gemm_swiglu_*`; routed via `fuse_softmax_topk` | Yes (MOE3GemmFusedCompressed) |
| shared expert | NOT fused on this build — stays as 3 `gemm_kernel` FCs; timed together with routed MoE | Yes (3 extra FC) |
| `gate` | attn_output · sigmoid(gate) of gated attention (full-attn layers) | Yes (eltwise) |
| `multiply`/`swish` | SwiGLU activation, fused into MoE primitive | No — not benched separately |
| `add` | residual adds | Yes (eltwise) |
| `rmsnorm` | pre-attn + pre-MLP RMSNorm | Yes |
| FC prefill | `dynamic_quantize_gpu_opt` + `gemm_kernel` | Yes (2 kernels) |
| PA decode | `pa_kv_cache_update` + attention + finalization | Yes (3 kernels) |

## Token latency summary

### Prefill — TTFT and per-token amortized

| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|
| 1024 | 551.3 | 0.551 | 0.538 | 1,857 |
| 2048 | 810.2 | 0.810 | 0.396 | 2,528 |
| 4096 | 1,481.4 | 1.481 | 0.362 | 2,765 |
| 8192 | 2,962.2 | 2.962 | 0.362 | 2,766 |

### Decode — TPOT (per output token, mid 512-gen window KV = P+256)

| prompt P | KV (mid) | TPOT (ms) | tokens/s |
|---:|---:|---:|---:|
| 1024 | 1280 | 21.634 | 46.2 |
| 2048 | 2304 | 22.103 | 45.2 |
| 4096 | 4352 | 21.812 | 45.8 |
| 8192 | 8448 | 22.556 | 44.3 |

### Decode — full 512-token generation window (PA grows with KV)

`start` = KV at first generated token (P); `mean` = KV P+256; `end` = KV P+512. Only PagedAttention scales with KV; the other 20.86 ms/token is M=1 constant.

| prompt P | TPOT start (ms) | TPOT mean (ms) | TPOT end (ms) | 512-tok decode (ms) | decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 1024 | 21.512 | 21.634 | 21.745 | 11,076.6 | 46.2 |
| 2048 | 21.990 | 22.103 | 22.209 | 11,316.5 | 45.2 |
| 4096 | 21.777 | 21.812 | 21.920 | 11,167.6 | 45.8 |
| 8192 | 22.569 | 22.556 | 22.580 | 11,548.9 | 44.3 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)

| op | P=1024 (KV1280) | P=2048 (KV2304) | P=4096 (KV4352) | P=8192 (KV8448) |
|---|---:|---:|---:|---:|
| MoE 3-gemm (TK=8 + shared, NE=256) | 7.612 (35.2%) | 7.612 (34.4%) | 7.612 (34.9%) | 7.612 (33.7%) |
| LM head (2048->248320, INT8) | 5.157 (23.8%) | 5.157 (23.3%) | 5.157 (23.6%) | 5.157 (22.9%) |
| FC linattn in_proj (2048->12288) | 3.848 (17.8%) | 3.848 (17.4%) | 3.848 (17.6%) | 3.848 (17.1%) |
| FC o_proj / GDN out (4096->2048) | 1.849 (8.5%) | 1.849 (8.4%) | 1.849 (8.5%) | 1.849 (8.2%) |
| FC qkv+gate (2048->9216) | 0.968 (4.5%) | 0.968 (4.4%) | 0.968 (4.4%) | 0.968 (4.3%) |
| GatedDeltaNet core (HK=32,K=V=128) | 0.956 (4.4%) | 0.956 (4.3%) | 0.956 (4.4%) | 0.956 (4.2%) |
| rmsnorm (H=2048) | 0.236 (1.1%) | 0.236 (1.1%) | 0.236 (1.1%) | 0.236 (1.0%) |
| residual add (H=2048) | 0.120 (0.6%) | 0.120 (0.5%) | 0.120 (0.5%) | 0.120 (0.5%) |
| q_norm (16x256) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) |
| rope_q (16x256) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) |
| k_norm (2x256) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) |
| rope_k (2x256) | 0.022 (0.1%) | 0.022 (0.1%) | 0.022 (0.1%) | 0.022 (0.1%) |
| attn gate x*sigmoid(y) (H=4096) | 0.015 (0.1%) | 0.015 (0.1%) | 0.015 (0.1%) | 0.015 (0.1%) |
| PagedAttention (×10) | 0.774 (3.6%) | 1.243 (5.6%) | 0.952 (4.4%) | 1.697 (7.5%) |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)

| op | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---:|---:|---:|---:|
| MoE grouped-gemm (TK=8 + shared) | 403.53 (73.2%) | 512.90 (63.3%) | 834.12 (56.3%) | 1454.62 (49.1%) |
| PagedAttention prefill (causal, NH=16) | 6.69 (1.2%) | 24.69 (3.0%) | 93.81 (6.3%) | 396.58 (13.4%) |
| GatedDeltaNet core | 77.48 (14.1%) | 153.56 (19.0%) | 312.06 (21.1%) | 621.34 (21.0%) |
| FC linattn in_proj (2048->12288) | 29.91 (5.4%) | 57.32 (7.1%) | 118.27 (8.0%) | 238.03 (8.0%) |
| FC qkv+gate (2048->9216) | 7.34 (1.3%) | 14.21 (1.8%) | 30.64 (2.1%) | 60.94 (2.1%) |
| FC o_proj / GDN out (4096->2048) | 14.70 (2.7%) | 28.69 (3.5%) | 54.26 (3.7%) | 111.05 (3.7%) |
| rmsnorm (H=2048) | 3.97 (0.7%) | 7.26 (0.9%) | 19.16 (1.3%) | 45.33 (1.5%) |
| rope_q (16x256) | 1.17 (0.2%) | 2.89 (0.4%) | 6.49 (0.4%) | 13.70 (0.5%) |
| attn gate x*sigmoid(y) (H=4096) | 1.33 (0.2%) | 3.54 (0.4%) | 7.44 (0.5%) | 15.46 (0.5%) |
| LM head (last token, 2048->248320) | 5.16 (0.9%) | 5.16 (0.6%) | 5.16 (0.3%) | 5.16 (0.2%) |

## Decode tables (1 query token, KV = mid 512-gen window context)

### Decode — KV=1280 (prompt 1024, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_793724151708371423` | 0.1903 | 40 | 7.6118 | 298 | 77.4 | 74% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.1573 | 1 | 5.1573 | 197 | 100.2 | 95% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1283 | 30 | 3.8481 | 392 | 102.2 | 97% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0462 | 40 | 1.8488 | 363 | 94.6 | 90% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0968 | 10 | 0.9685 | 390 | 101.5 | 97% | mem |
| GatedDeltaNet core (HK=32,K=V=128) | `gated_delta_net_ref__8629349849706163749__sa` | 0.0319 | 30 | 0.9565 | — | — | — | recurrent |
| PagedAttention (i8 KV=1280) | `paged_attention_opt__single_token_15856477779756318941__sa` | 0.0774 | 10 | 0.7743 | 271 | 16.9 | 16% | mem |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_12510323107858754751_0_0` | 0.0029 | 80 | 0.2359 | 6 | 2.8 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1198 | 1 | 8.2 | 8% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0255 | 13 | 6.4 | 6% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0254 | 16 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0246 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0222 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **21.634** |  |  |  |  |

### Decode — KV=2304 (prompt 2048, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_793724151708371423` | 0.1903 | 40 | 7.6118 | 298 | 77.4 | 74% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.1573 | 1 | 5.1573 | 197 | 100.2 | 95% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1283 | 30 | 3.8481 | 392 | 102.2 | 97% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0462 | 40 | 1.8488 | 363 | 94.6 | 90% | mem |
| PagedAttention (i8 KV=2304) | `paged_attention_opt__single_token_15856477779756318941__sa` | 0.1243 | 10 | 1.2428 | 304 | 19.0 | 18% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0968 | 10 | 0.9685 | 390 | 101.5 | 97% | mem |
| GatedDeltaNet core (HK=32,K=V=128) | `gated_delta_net_ref__8629349849706163749__sa` | 0.0319 | 30 | 0.9565 | — | — | — | recurrent |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_12510323107858754751_0_0` | 0.0029 | 80 | 0.2359 | 6 | 2.8 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1198 | 1 | 8.2 | 8% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0255 | 13 | 6.4 | 6% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0254 | 16 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0246 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0222 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **22.103** |  |  |  |  |

### Decode — KV=4352 (prompt 4096, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_793724151708371423` | 0.1903 | 40 | 7.6118 | 298 | 77.4 | 74% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.1573 | 1 | 5.1573 | 197 | 100.2 | 95% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1283 | 30 | 3.8481 | 392 | 102.2 | 97% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0462 | 40 | 1.8488 | 363 | 94.6 | 90% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0968 | 10 | 0.9685 | 390 | 101.5 | 97% | mem |
| GatedDeltaNet core (HK=32,K=V=128) | `gated_delta_net_ref__8629349849706163749__sa` | 0.0319 | 30 | 0.9565 | — | — | — | recurrent |
| PagedAttention (i8 KV=4352) | `paged_attention_opt__gqa_single_token_15856477779756318941__sa` | 0.0952 | 10 | 0.9521 | 749 | 46.8 | 45% | mem |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_12510323107858754751_0_0` | 0.0029 | 80 | 0.2359 | 6 | 2.8 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1198 | 1 | 8.2 | 8% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0255 | 13 | 6.4 | 6% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0254 | 16 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0246 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0222 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **21.812** |  |  |  |  |

### Decode — KV=8448 (prompt 8192, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_793724151708371423` | 0.1903 | 40 | 7.6118 | 298 | 77.4 | 74% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.1573 | 1 | 5.1573 | 197 | 100.2 | 95% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1283 | 30 | 3.8481 | 392 | 102.2 | 97% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0462 | 40 | 1.8488 | 363 | 94.6 | 90% | mem |
| PagedAttention (i8 KV=8448) | `paged_attention_opt__gqa_single_token_15856477779756318941__sa` | 0.1697 | 10 | 1.6967 | 816 | 51.0 | 49% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0968 | 10 | 0.9685 | 390 | 101.5 | 97% | mem |
| GatedDeltaNet core (HK=32,K=V=128) | `gated_delta_net_ref__8629349849706163749__sa` | 0.0319 | 30 | 0.9565 | — | — | — | recurrent |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_12510323107858754751_0_0` | 0.0029 | 80 | 0.2359 | 6 | 2.8 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1198 | 1 | 8.2 | 8% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0255 | 13 | 6.4 | 6% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0254 | 16 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0246 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0222 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **22.556** |  |  |  |  |

## Prefill tables (single forward over S tokens)

### Prefill — S=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 10.0884 | 40 | 403.5350 | 5,747 | 3.7 | 5% | compute |
| GatedDeltaNet core | `gated_delta_net_ref__8629349849706163749__sa` | 2.5828 | 30 | 77.4833 | — | — | — | recurrent |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.9971 | 30 | 29.9141 | 51,688 | 55.1 | 52% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.3675 | 40 | 14.7016 | 46,743 | 57.4 | 55% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.7343 | 10 | 7.3433 | 52,639 | 57.5 | 55% | mem |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 0.6686 | 10 | 6.6860 | 12,848 | 26.7 | 25% | mem |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.1573 | 1 | 5.1573 | 197 | 100.2 | 95% | mem |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_3188239531497015022_0_0` | 0.0496 | 80 | 3.9666 | 338 | 169.2 | 161% | cache |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_12951181864104916429_0_0` | 0.1329 | 10 | 1.3293 | 158 | 189.3 | 180% | cache |
| rope_q (16x256) | `rope_opt__1263978976134699825` | 0.1173 | 10 | 1.1731 | 358 | 143.0 | 136% | cache |
| **TOTAL** |  |  |  | **551.290** |  |  |  |  |

### Prefill — S=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 12.8224 | 40 | 512.8950 | 9,044 | 4.7 | 8% | compute |
| GatedDeltaNet core | `gated_delta_net_ref__8629349849706163749__sa` | 5.1185 | 30 | 153.5565 | — | — | — | recurrent |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 1.9107 | 30 | 57.3196 | 53,950 | 44.1 | 46% | compute |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.7174 | 40 | 28.6948 | 47,897 | 47.0 | 45% | mem |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 2.4690 | 10 | 24.6897 | 13,917 | 14.4 | 24% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 1.4209 | 10 | 14.2088 | 54,410 | 46.0 | 46% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_8465929718177529364_0_0` | 0.0908 | 80 | 7.2636 | 370 | 184.8 | 176% | cache |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.1573 | 1 | 5.1573 | 197 | 100.2 | 95% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_13390302583414783927_0_0` | 0.3545 | 10 | 3.5450 | 118 | 142.0 | 135% | cache |
| rope_q (16x256) | `rope_opt__13556955710875957727` | 0.2888 | 10 | 2.8881 | 290 | 116.2 | 111% | cache |
| **TOTAL** |  |  |  | **810.218** |  |  |  |  |

### Prefill — S=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 20.8531 | 40 | 834.1226 | 11,122 | 5.1 | 9% | compute |
| GatedDeltaNet core | `gated_delta_net_ref__8629349849706163749__sa` | 10.4020 | 30 | 312.0606 | — | — | — | recurrent |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 3.9423 | 30 | 118.2700 | 52,294 | 36.3 | 44% | compute |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 9.3809 | 10 | 93.8093 | 14,651 | 7.6 | 25% | compute |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 1.3566 | 40 | 54.2627 | 50,657 | 43.4 | 43% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 3.0636 | 10 | 30.6365 | 50,469 | 36.4 | 43% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_2185109744823398906_0_0` | 0.2395 | 80 | 19.1629 | 280 | 140.1 | 133% | cache |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_1600471834299485859_0_0` | 0.7439 | 10 | 7.4392 | 113 | 135.3 | 129% | cache |
| rope_q (16x256) | `rope_opt__11913003717797137888` | 0.6487 | 10 | 6.4874 | 259 | 103.4 | 99% | mem |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.1573 | 1 | 5.1573 | 197 | 100.2 | 95% | mem |
| **TOTAL** |  |  |  | **1,481.408** |  |  |  |  |

### Prefill — S=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 36.3655 | 40 | 1454.6201 | 12,755 | 5.5 | 11% | compute |
| GatedDeltaNet core | `gated_delta_net_ref__8629349849706163749__sa` | 20.7113 | 30 | 621.3390 | — | — | — | recurrent |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 39.6577 | 10 | 396.5768 | 13,863 | 3.6 | 24% | compute |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 7.9343 | 30 | 238.0303 | 51,966 | 32.8 | 44% | compute |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 2.7763 | 40 | 111.0523 | 49,504 | 39.3 | 42% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 6.0943 | 10 | 60.9431 | 50,742 | 33.4 | 43% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_1416219266873627125_0_0` | 0.5666 | 80 | 45.3272 | 237 | 118.4 | 113% | cache |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_10752814223568689025_0_0` | 1.5455 | 10 | 15.4552 | 109 | 130.3 | 124% | cache |
| rope_q (16x256) | `rope_opt__11688920003865995564` | 1.3705 | 10 | 13.7045 | 245 | 97.9 | 93% | mem |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.1573 | 1 | 5.1573 | 197 | 100.2 | 95% | mem |
| **TOTAL** |  |  |  | **2,962.206** |  |  |  |  |

## Per-kernel decomposition (cliloader kernel names)

### Decode sub-kernels — KV=4352 (prompt 4096, representative)

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.1573 | 1.0 | 1 | 5.1573 | 23.6% |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1283 | 1.0 | 30 | 3.8481 | 17.6% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_793724151708371423` | 0.0954 | 1.0 | 40 | 3.8158 | 17.5% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_down_793724151708371423` | 0.0467 | 1.0 | 40 | 1.8695 | 8.6% |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0462 | 1.0 | 40 | 1.8488 | 8.5% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `gemm_kernel` | 0.0320 | 4.0 | 40 | 1.2782 | 5.9% |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0968 | 1.0 | 10 | 0.9685 | 4.4% |
| GatedDeltaNet core (HK=32,K=V=128) | `gated_delta_net_ref__8629349849706163749__sa` | 0.0319 | 1.0 | 30 | 0.9565 | 4.4% |
| PagedAttention | `paged_attention_opt__gqa_single_token_15856477779756318941__sa` | 0.0834 | 1.0 | 10 | 0.8338 | 3.8% |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_12510323107858754751_0_0` | 0.0029 | 1.0 | 80 | 0.2359 | 1.1% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_fuse_softmax_topk_793724151708371423` | 0.0058 | 1.0 | 40 | 0.2333 | 1.1% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `dynamic_quantize_gpu_opt_2680479405095583548_0_0` | 0.0041 | 3.0 | 40 | 0.1644 | 0.8% |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 1.0 | 80 | 0.1198 | 0.5% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `dynamic_quantize_gpu_opt_11767888948171086441_0_0` | 0.0019 | 1.0 | 40 | 0.0748 | 0.3% |
| PagedAttention | `paged_attention_opt__single_token_finalization_15856477779756318941__sa` | 0.0060 | 1.0 | 10 | 0.0603 | 0.3% |
| PagedAttention | `pa_kv_cache_update_ref__15856477779756318941__sa` | 0.0058 | 1.0 | 10 | 0.0580 | 0.3% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `eltwise_simple_vload8_12929541252905950109_0_0` | 0.0013 | 1.0 | 40 | 0.0527 | 0.2% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `reorder_data_fast_b1_2055831431303789805_0_0` | 0.0012 | 1.0 | 40 | 0.0474 | 0.2% |

### Prefill sub-kernels — S=8192 (representative)

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 16.9996 | 3.0 | 40 | 679.9849 | 23.0% |
| GatedDeltaNet core | `gated_delta_net_ref__8629349849706163749__sa` | 20.7113 | 1.0 | 30 | 621.3390 | 21.0% |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 39.4296 | 1.0 | 10 | 394.2965 | 13.3% |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 7.5081 | 1.0 | 30 | 225.2416 | 7.6% |
| MoE grouped-gemm (TK=8 + shared) | `moe_scatter_reduction_opt_moe_scatter_reduction_ref_12318565695326055078` | 5.3873 | 1.0 | 40 | 215.4929 | 7.3% |
| MoE grouped-gemm (TK=8 + shared) | `moe_gather_ref_prefill_gather_12318565695326055078` | 5.0628 | 1.0 | 40 | 202.5112 | 6.8% |
| MoE grouped-gemm (TK=8 + shared) | `moe_3gemm_swiglu_fuse_prefill_swiglu_12318565695326055078` | 1.9868 | 1.0 | 40 | 79.4737 | 2.7% |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 1.8017 | 1.0 | 40 | 72.0692 | 2.4% |
| MoE grouped-gemm (TK=8 + shared) | `gemm_kernel` | 1.6097 | 4.0 | 40 | 64.3866 | 2.2% |
| MoE grouped-gemm (TK=8 + shared) | `dynamic_quantize_gpu_opt_10707039156225127723_0_0` | 1.4496 | 3.0 | 40 | 57.9824 | 2.0% |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 5.6308 | 1.0 | 10 | 56.3079 | 1.9% |
| MoE grouped-gemm (TK=8 + shared) | `reorder_data_fast_b1_18330753658507001113_0_0` | 1.1525 | 1.0 | 40 | 46.1012 | 1.6% |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_1416219266873627125_0_0` | 0.5666 | 1.0 | 80 | 45.3272 | 1.5% |
| MoE grouped-gemm (TK=8 + shared) | `reorder_data_fast_b1_7898771992857082467_0_0` | 1.0195 | 1.0 | 40 | 40.7796 | 1.4% |
| FC o_proj / GDN out (4096->2048) | `dynamic_quantize_gpu_opt_1534384422443530166_0_0` | 0.9746 | 1.0 | 40 | 38.9831 | 1.3% |
| MoE grouped-gemm (TK=8 + shared) | `eltwise_simple_vload8_7979511875005307881_0_0` | 0.9083 | 1.0 | 40 | 36.3316 | 1.2% |
| MoE grouped-gemm (TK=8 + shared) | `moe_3gemm_swiglu_fuse_softmax_topk_12318565695326055078` | 0.6743 | 1.0 | 40 | 26.9729 | 0.9% |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_10752814223568689025_0_0` | 1.5455 | 1.0 | 10 | 15.4552 | 0.5% |

## Top contributors (sorted by total ms per inference)

### Decode

| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1280 | MoE 3-gemm (TK=8 + shared, NE=256) 7.612ms (35%) | LM head (2048->248320, INT8) 5.157ms (24%) | FC linattn in_proj (2048->12288) 3.848ms (18%) |
| 2304 | MoE 3-gemm (TK=8 + shared, NE=256) 7.612ms (34%) | LM head (2048->248320, INT8) 5.157ms (23%) | FC linattn in_proj (2048->12288) 3.848ms (17%) |
| 4352 | MoE 3-gemm (TK=8 + shared, NE=256) 7.612ms (35%) | LM head (2048->248320, INT8) 5.157ms (24%) | FC linattn in_proj (2048->12288) 3.848ms (18%) |
| 8448 | MoE 3-gemm (TK=8 + shared, NE=256) 7.612ms (34%) | LM head (2048->248320, INT8) 5.157ms (23%) | FC linattn in_proj (2048->12288) 3.848ms (17%) |

### Prefill

| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | MoE grouped-gemm (TK=8 + shared) 403.5ms (73%) | GatedDeltaNet core 77.5ms (14%) | FC linattn in_proj (2048->12288) 29.9ms (5%) |
| 2048 | MoE grouped-gemm (TK=8 + shared) 512.9ms (63%) | GatedDeltaNet core 153.6ms (19%) | FC linattn in_proj (2048->12288) 57.3ms (7%) |
| 4096 | MoE grouped-gemm (TK=8 + shared) 834.1ms (56%) | GatedDeltaNet core 312.1ms (21%) | FC linattn in_proj (2048->12288) 118.3ms (8%) |
| 8192 | MoE grouped-gemm (TK=8 + shared) 1454.6ms (49%) | GatedDeltaNet core 621.3ms (21%) | PagedAttention prefill (causal, NH=16) 396.6ms (13%) |

## End-to-end (prefill TTFT + 512-token decode)

| prompt P | TTFT (ms) | 512-tok decode (ms) | total (ms) | avg decode tok/s |
|---:|---:|---:|---:|---:|
| 1024 | 551.3 | 11,076.6 | 11,627.9 | 46.2 |
| 2048 | 810.2 | 11,316.5 | 12,126.7 | 45.2 |
| 4096 | 1,481.4 | 11,167.6 | 12,649.0 | 45.8 |
| 8192 | 2,962.2 | 11,548.9 | 14,511.1 | 44.3 |

## Key findings

- **Decode throughput is flat at ~44–46 tok/s across all prompt lengths.** The per-token budget is 20.9 ms of KV-independent work; PagedAttention over the 512-token window only moves TPOT by ~0.7–1.7 ms, so prompt length barely affects decode.
- **Three ops own ~80% of decode:** MoE, LM-head, and linear-attn in_proj — all memory-bound INT4/INT8 weight streaming at 90–97% of 105 GB/s.
- **PA kernel heuristic switch at KV≥4096** (`paged_attention_opt__single_token` → `gqa_single_token`) makes PA-decode non-monotonic: the 4096-prompt window is *faster* than the 2048-prompt window.
- **Prefill (TTFT) is MoE/GDN/PA-bound and grows super-linearly** (0.55 s → 2.96 s for 1K → 8K). PA prefill is quadratic (causal); MoE grouped-gemm dominates the rest.

## Optimization levers (highest decode ROI first)

1. **MoE expert streaming** — BW-bound at only ~74% efficiency (M=1 micro-gemms have launch/scheduling overhead); batching decode requests amortizes weight reads, and the unfused shared expert (3 extra FCs) is a fusion opportunity.
2. **LM-head** — 508 MB of INT8 weights streamed every token at ~95% BW; INT4 LM-head would roughly halve it; speculative decoding removes it from the per-token path.
3. **Linear-attn in_proj** — already ~97% BW-bound; only lower precision or fewer linear layers help.
4. **Memory bandwidth is the global ceiling** — every major decode op is mem-bound, so the 105 GB/s shared LPDDR5x read BW sets the floor.

## Reproduction

Built on the PTL host (VS2022, `OV_SRC_DIR` for `gdn_bench`); each case run under cliloader `-d` and parsed from Device Performance Timing. Driver: `utils/run_qwen3_6_ptl.bat` (61 cases). Representative commands:

```bat
:: decode (M=1) — KV-independent ops
moe_bench.exe        1 1    2048 512 256 8 128 100 10 4 64 512   :: MoE TK=8 + shared
fc_bench.exe         1 2048   9216 128 5000 200 8 u4 64          :: qkv+gate fused
fc_bench.exe         1 2048  12288 128 1500 100 4 u4 64          :: linear-attn in_proj
fc_bench.exe         1 2048 248320 128  300  30 4 u8 64          :: LM-head (INT8)
gdn_bench.exe        1 1     32 32 128 4000 150 4                :: GatedDeltaNet core
small_ops_bench.exe  gate    1 4096 --iters 5000 --warmup 200    :: attn output gate
:: decode PA — 512-token window sweep (KV = P / P+256 / P+512)
pa_bench.exe decode  1 <KV> 8000 200 4 i8   (PA_NH=16 PA_NKV=2 PA_HD=256)
:: prefill — per prompt S in {1024,2048,4096,8192}
moe_bench.exe 1 S 2048 512 256 8 128 ...    pa_bench.exe prefill S 0 ... i8
```

Parse + report:
```bash
python3 ../../utils/parse_logs.py logs ptl_metrics.json
python3 build_report.py ptl_metrics.json > SUMMARY_qwen3_6_%DATE%.md
```

## Caveats & method

- Each op profiled in its own process via cliloader Device Performance Timing (mean kernel time per iteration); cache flush between iters so weights stream from VRAM. Totals are an upper-bound roofline, not a traced wall-clock.
- FC weight bytes count INT4/INT8 weight + FP16 scale/zp(g128) + FP16 act + FP16 out.
- **Shared expert is unfused on this build**; the MoE figure times the routed `MOE3GemmFusedCompressed` plus 3 shared FCs together — the real per-layer cost here.
- **GatedDeltaNet bench covers the core op only**; depthwise conv1d (k=4) not modeled (negligible), in_proj counted as `fc_linattn`, out_proj as `fc_o` (4096→2048). It is a recurrent state op with no clean analytic byte/flop model, so `bound=recurrent` and Eff/GFLOPS/GB/s are shown as `—` (only the measured latency is meaningful).
- `bound=cache`: tiny eltwise/norm/rope micro-benches are L2/L3-resident, so their achieved BW exceeds the 110 GB/s streaming spec; in real inference they are fused into the adjacent matmul/attention and contribute <2% — only their latency is used.
- The attention output gate is benched with a `gate` proxy op (x·sigmoid(y), H=4096).
- Decode FC/MoE/LM-head are memory-bound (weights dominate at M=1); prefill FC/MoE are INT8 XMX compute-bound at large S; PA prefill (S≥2048) is FP16 micro-kernel compute-bound.
- PA decode is memory-bound (INT8 KV cache + FP16 Q/out); lm_head runs once per token.
- q_norm/k_norm and residual-add are <0.1% of TTFT and omitted from prefill totals.
