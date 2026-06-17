# Qwen3.6-35B-A3B — Roofline on Intel Arc B390 (Panther Lake, 12Xe iGPU) (20260615)

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
| 1024 | 528.9 | 0.529 | 0.517 | 1,936 |
| 2048 | 784.0 | 0.784 | 0.383 | 2,612 |
| 4096 | 1,477.9 | 1.478 | 0.361 | 2,771 |
| 8192 | 2,964.6 | 2.965 | 0.362 | 2,763 |

### Decode — TPOT (per output token, mid 512-gen window KV = P+256)

| prompt P | KV (mid) | TPOT (ms) | tokens/s |
|---:|---:|---:|---:|
| 1024 | 1280 | 21.178 | 47.2 |
| 2048 | 2304 | 21.651 | 46.2 |
| 4096 | 4352 | 21.348 | 46.8 |
| 8192 | 8448 | 22.099 | 45.3 |

### Decode — full 512-token generation window (PA grows with KV)

`start` = KV at first generated token (P); `mean` = KV P+256; `end` = KV P+512. Only PagedAttention scales with KV; the other 20.86 ms/token is M=1 constant.

| prompt P | TPOT start (ms) | TPOT mean (ms) | TPOT end (ms) | 512-tok decode (ms) | decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 1024 | 21.056 | 21.178 | 21.285 | 10,843.0 | 47.2 |
| 2048 | 21.532 | 21.651 | 21.748 | 11,085.3 | 46.2 |
| 4096 | 21.323 | 21.348 | 21.457 | 10,930.0 | 46.8 |
| 8192 | 22.082 | 22.099 | 22.120 | 11,314.6 | 45.3 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)

| op | P=1024 (KV1280) | P=2048 (KV2304) | P=4096 (KV4352) | P=8192 (KV8448) |
|---|---:|---:|---:|---:|
| MoE 3-gemm (TK=8 + shared, NE=256) | 7.793 (36.8%) | 7.793 (36.0%) | 7.793 (36.5%) | 7.793 (35.3%) |
| LM head (2048->248320, INT8) | 5.154 (24.3%) | 5.154 (23.8%) | 5.154 (24.1%) | 5.154 (23.3%) |
| FC linattn in_proj (2048->12288) | 3.885 (18.3%) | 3.885 (17.9%) | 3.885 (18.2%) | 3.885 (17.6%) |
| FC o_proj / GDN out (4096->2048) | 1.841 (8.7%) | 1.841 (8.5%) | 1.841 (8.6%) | 1.841 (8.3%) |
| FC qkv+gate (2048->9216) | 0.963 (4.5%) | 0.963 (4.4%) | 0.963 (4.5%) | 0.963 (4.4%) |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | 0.534 (2.5%) | 0.534 (2.5%) | 0.534 (2.5%) | 0.534 (2.4%) |
| rmsnorm (H=2048) | 0.000 (0.0%) | 0.000 (0.0%) | 0.000 (0.0%) | 0.000 (0.0%) |
| residual add (H=2048) | 0.120 (0.6%) | 0.120 (0.6%) | 0.120 (0.6%) | 0.120 (0.5%) |
| q_norm (16x256) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) |
| rope_q (16x256) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) |
| k_norm (2x256) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) |
| rope_k (2x256) | 0.022 (0.1%) | 0.022 (0.1%) | 0.022 (0.1%) | 0.022 (0.1%) |
| attn gate x*sigmoid(y) (H=4096) | 0.015 (0.1%) | 0.015 (0.1%) | 0.015 (0.1%) | 0.015 (0.1%) |
| PagedAttention (×10) | 0.774 (3.7%) | 1.247 (5.8%) | 0.944 (4.4%) | 1.695 (7.7%) |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)

| op | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---:|---:|---:|---:|
| MoE grouped-gemm (TK=8 + shared) | 383.64 (72.5%) | 502.70 (64.1%) | 840.59 (56.9%) | 1458.79 (49.2%) |
| PagedAttention prefill (causal, NH=16) | 6.64 (1.3%) | 24.25 (3.1%) | 92.57 (6.3%) | 422.92 (14.3%) |
| GatedDeltaNet core | 75.55 (14.3%) | 137.84 (17.6%) | 301.24 (20.4%) | 589.42 (19.9%) |
| FC linattn in_proj (2048->12288) | 29.58 (5.6%) | 56.64 (7.2%) | 119.11 (8.1%) | 237.51 (8.0%) |
| FC qkv+gate (2048->9216) | 7.27 (1.4%) | 14.36 (1.8%) | 30.25 (2.0%) | 61.04 (2.1%) |
| FC o_proj / GDN out (4096->2048) | 14.85 (2.8%) | 29.02 (3.7%) | 55.96 (3.8%) | 113.61 (3.8%) |
| rmsnorm (H=2048) | 3.81 (0.7%) | 7.68 (1.0%) | 19.15 (1.3%) | 46.93 (1.6%) |
| rope_q (16x256) | 1.14 (0.2%) | 2.83 (0.4%) | 6.44 (0.4%) | 13.77 (0.5%) |
| attn gate x*sigmoid(y) (H=4096) | 1.32 (0.2%) | 3.49 (0.4%) | 7.49 (0.5%) | 15.47 (0.5%) |
| LM head (last token, 2048->248320) | 5.15 (1.0%) | 5.15 (0.7%) | 5.15 (0.3%) | 5.15 (0.2%) |

## Roofline: theoretical floor vs measured

**Theoretical floor** = sum over analytically-modelable ops of max(bytes / BW, FLOP / XMX-peak) - the fastest this HW could run each op given its memory traffic / compute. **Measured** is the summed cliloader kernel time of the same ops. **achieved % = theoretical / measured** (how close real kernels get to the roofline ceiling; 100% = on the roofline). GatedDeltaNet uses a recurrent `*_opt` kernel (measured with `cache_interval=0`, i.e. a single final state snapshot) with no analytic model, so it is reported separately as *unmodeled GDN* and excluded from the ratio; `full` = measured + unmodeled = the real TPOT / TTFT.

### Decode (per output token, mid 512-gen window KV = P+256)

| prompt P | KV | theoretical (ms) | measured (ms) | achieved % | unmodeled GDN (ms) | full TPOT (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1280 | 17.027 | 20.644 | 82.5% | 0.534 | 21.178 |
| 2048 | 2304 | 17.127 | 21.117 | 81.1% | 0.534 | 21.651 |
| 4096 | 4352 | 17.326 | 20.814 | 83.2% | 0.534 | 21.348 |
| 8192 | 8448 | 17.726 | 21.565 | 82.2% | 0.534 | 22.099 |

### Prefill (TTFT over S tokens)

| S | theoretical (ms) | measured (ms) | achieved % | unmodeled GDN (ms) | full TTFT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 209.4 | 453.4 | 46.2% | 75.6 | 528.9 |
| 2048 | 247.9 | 646.1 | 38.4% | 137.8 | 784.0 |
| 4096 | 345.3 | 1,176.7 | 29.3% | 301.2 | 1,477.9 |
| 8192 | 581.3 | 2,375.2 | 24.5% | 589.4 | 2,964.6 |

## Decode tables (1 query token, KV = mid 512-gen window context)

### Decode — KV=1280 (prompt 1024, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_6438761271185273278` | 0.1948 | 40 | 7.7934 | 291 | 75.6 | 72% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.1542 | 1 | 5.1542 | 197 | 100.3 | 96% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1295 | 30 | 3.8853 | 389 | 101.2 | 96% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0460 | 40 | 1.8408 | 365 | 95.0 | 90% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0963 | 10 | 0.9627 | 392 | 102.1 | 97% | mem |
| PagedAttention (i8 KV=1280) | `paged_attention_opt__single_token_15856477779756318941__sa` | 0.0774 | 10 | 0.7742 | 271 | 16.9 | 16% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 0.0178 | 30 | 0.5338 | — | — | — | recurrent |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1199 | 1 | 8.2 | 8% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0253 | 16 | 6.5 | 6% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0253 | 13 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0251 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0224 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **21.178** |  |  |  |  |

### Decode — KV=2304 (prompt 2048, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_6438761271185273278` | 0.1948 | 40 | 7.7934 | 291 | 75.6 | 72% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.1542 | 1 | 5.1542 | 197 | 100.3 | 96% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1295 | 30 | 3.8853 | 389 | 101.2 | 96% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0460 | 40 | 1.8408 | 365 | 95.0 | 90% | mem |
| PagedAttention (i8 KV=2304) | `paged_attention_opt__single_token_15856477779756318941__sa` | 0.1247 | 10 | 1.2473 | 303 | 18.9 | 18% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0963 | 10 | 0.9627 | 392 | 102.1 | 97% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 0.0178 | 30 | 0.5338 | — | — | — | recurrent |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1199 | 1 | 8.2 | 8% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0253 | 16 | 6.5 | 6% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0253 | 13 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0251 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0224 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **21.651** |  |  |  |  |

### Decode — KV=4352 (prompt 4096, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_6438761271185273278` | 0.1948 | 40 | 7.7934 | 291 | 75.6 | 72% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.1542 | 1 | 5.1542 | 197 | 100.3 | 96% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1295 | 30 | 3.8853 | 389 | 101.2 | 96% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0460 | 40 | 1.8408 | 365 | 95.0 | 90% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0963 | 10 | 0.9627 | 392 | 102.1 | 97% | mem |
| PagedAttention (i8 KV=4352) | `paged_attention_opt__gqa_single_token_15856477779756318941__sa` | 0.0944 | 10 | 0.9440 | 755 | 47.2 | 45% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 0.0178 | 30 | 0.5338 | — | — | — | recurrent |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1199 | 1 | 8.2 | 8% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0253 | 16 | 6.5 | 6% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0253 | 13 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0251 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0224 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **21.348** |  |  |  |  |

### Decode — KV=8448 (prompt 8192, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_6438761271185273278` | 0.1948 | 40 | 7.7934 | 291 | 75.6 | 72% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.1542 | 1 | 5.1542 | 197 | 100.3 | 96% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1295 | 30 | 3.8853 | 389 | 101.2 | 96% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0460 | 40 | 1.8408 | 365 | 95.0 | 90% | mem |
| PagedAttention (i8 KV=8448) | `paged_attention_opt__gqa_single_token_15856477779756318941__sa` | 0.1695 | 10 | 1.6953 | 816 | 51.0 | 49% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0963 | 10 | 0.9627 | 392 | 102.1 | 97% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 0.0178 | 30 | 0.5338 | — | — | — | recurrent |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1199 | 1 | 8.2 | 8% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0253 | 16 | 6.5 | 6% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0253 | 13 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0251 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0224 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **22.099** |  |  |  |  |

## Prefill tables (single forward over S tokens)

### Prefill — S=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 9.5911 | 40 | 383.6442 | 6,045 | 46.2 | 44% | mem |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 2.5184 | 30 | 75.5519 | — | — | — | recurrent |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.9861 | 30 | 29.5829 | 52,266 | 55.7 | 53% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.3712 | 40 | 14.8462 | 46,287 | 56.9 | 54% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.7266 | 10 | 7.2663 | 53,197 | 58.1 | 55% | mem |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 0.6640 | 10 | 6.6398 | 12,937 | 26.8 | 26% | mem |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.1542 | 1 | 5.1542 | 197 | 100.3 | 96% | mem |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_3188239531497015022_0_0` | 0.0476 | 80 | 3.8076 | 352 | 176.2 | 168% | cache |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_12951181864104916429_0_0` | 0.1319 | 10 | 1.3187 | 159 | 190.8 | 182% | cache |
| rope_q (16x256) | `rope_opt__1263978976134699825` | 0.1136 | 10 | 1.1360 | 369 | 147.7 | 141% | cache |
| **TOTAL** |  |  |  | **528.948** |  |  |  |  |

### Prefill — S=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 12.5674 | 40 | 502.6974 | 9,227 | 37.1 | 35% | mem |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 4.5947 | 30 | 137.8402 | — | — | — | recurrent |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 1.8881 | 30 | 56.6422 | 54,595 | 44.6 | 46% | compute |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.7255 | 40 | 29.0212 | 47,358 | 46.4 | 44% | mem |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 2.4253 | 10 | 24.2531 | 14,167 | 14.7 | 24% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 1.4364 | 10 | 14.3640 | 53,822 | 45.5 | 46% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_8465929718177529364_0_0` | 0.0960 | 80 | 7.6773 | 350 | 174.8 | 166% | cache |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.1542 | 1 | 5.1542 | 197 | 100.3 | 96% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_13390302583414783927_0_0` | 0.3490 | 10 | 3.4905 | 120 | 144.2 | 137% | cache |
| rope_q (16x256) | `rope_opt__13556955710875957727` | 0.2834 | 10 | 2.8341 | 296 | 118.4 | 113% | cache |
| **TOTAL** |  |  |  | **783.974** |  |  |  |  |

### Prefill — S=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 21.0148 | 40 | 840.5920 | 11,036 | 24.4 | 23% | mem |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 10.0412 | 30 | 301.2356 | — | — | — | recurrent |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 3.9703 | 30 | 119.1085 | 51,925 | 36.0 | 44% | compute |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 9.2574 | 10 | 92.5740 | 14,846 | 7.7 | 25% | compute |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 1.3989 | 40 | 55.9570 | 49,123 | 42.1 | 42% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 3.0246 | 10 | 30.2464 | 51,120 | 36.8 | 43% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_2185109744823398906_0_0` | 0.2393 | 80 | 19.1450 | 280 | 140.2 | 134% | cache |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_1600471834299485859_0_0` | 0.7493 | 10 | 7.4927 | 112 | 134.3 | 128% | cache |
| rope_q (16x256) | `rope_opt__11913003717797137888` | 0.6439 | 10 | 6.4390 | 261 | 104.2 | 99% | mem |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.1542 | 1 | 5.1542 | 197 | 100.3 | 96% | mem |
| **TOTAL** |  |  |  | **1,477.944** |  |  |  |  |

### Prefill — S=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 36.4699 | 40 | 1458.7943 | 12,719 | 16.6 | 16% | mem |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 19.6472 | 30 | 589.4155 | — | — | — | recurrent |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 42.2917 | 10 | 422.9165 | 12,999 | 3.4 | 22% | compute |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 7.9169 | 30 | 237.5081 | 52,080 | 32.9 | 44% | compute |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 2.8402 | 40 | 113.6068 | 48,391 | 38.4 | 41% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 6.1037 | 10 | 61.0366 | 50,664 | 33.4 | 43% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_1416219266873627125_0_0` | 0.5866 | 80 | 46.9273 | 229 | 114.4 | 109% | cache |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_10752814223568689025_0_0` | 1.5467 | 10 | 15.4671 | 108 | 130.2 | 124% | cache |
| rope_q (16x256) | `rope_opt__11688920003865995564` | 1.3770 | 10 | 13.7705 | 244 | 97.5 | 93% | mem |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.1542 | 1 | 5.1542 | 197 | 100.3 | 96% | mem |
| **TOTAL** |  |  |  | **2,964.597** |  |  |  |  |

## Op → kernel names (cliloader)

Each logical op and the actual GPU kernel(s) it dispatches (one bench process per op). Kernels are listed in launch order; `launches/call` is how many times each kernel fires per single op invocation (per layer). Decode is measured at M=1, prefill at S=8192 — kernel selection can vary with shape.

### Decode (M=1)

| op | kernel name(s) | launches/call |
|---|---|---:|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_6438761271185273278`<br>`moe_3gemm_swiglu_mlp_down_6438761271185273278`<br>`gemm_kernel`<br>`moe_3gemm_swiglu_fuse_softmax_topk_2566375436411285507`<br>`dynamic_quantize_gpu_opt_2680479405095583548_0_0`<br>`reorder_data_fast_b1_2055831431303789805_0_0`<br>`dynamic_quantize_gpu_opt_11767888948171086441_0_0`<br>`eltwise_simple_vload8_12929541252905950109_0_0`<br>`reorder_data_fast_b1_17423095617980708425_0_0`<br>`moe_3gemm_swiglu_mlp_reduce_6438761271185273278` | 1.0<br>1.0<br>4.0<br>1.0<br>3.0<br>1.0<br>1.0<br>1.0<br>1.0<br>1.0 |
| LM head (2048->248320, INT8) | `gemm_kernel` | 1.0 |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 1.0 |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 1.0 |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 1.0 |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 1.0 |
| rmsnorm (H=2048) | `-` | — |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 1.0 |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 1.0 |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 1.0 |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 1.0 |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 1.0 |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 1.0 |
| PagedAttention (KV=4352) | `paged_attention_opt__gqa_single_token_15856477779756318941__sa`<br>`paged_attention_opt__single_token_finalization_15856477779756318941__sa`<br>`pa_kv_cache_update_ref__15856477779756318941__sa` | 1.0<br>1.0<br>1.0 |

### Prefill (S=8192)

| op | kernel name(s) | launches/call |
|---|---|---:|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm`<br>`moe_scatter_reduction_opt_moe_scatter_reduction_ref_10772290679749044717`<br>`moe_gather_ref_prefill_gather_10772290679749044717`<br>`moe_3gemm_swiglu_fuse_prefill_swiglu_10772290679749044717`<br>`gemm_kernel`<br>`dynamic_quantize_gpu_opt_10707039156225127723_0_0`<br>`reorder_data_fast_b1_18330753658507001113_0_0`<br>`reorder_data_fast_b1_7898771992857082467_0_0`<br>`eltwise_simple_vload8_7979511875005307881_0_0`<br>`moe_3gemm_swiglu_fuse_softmax_topk_10944893001517410604`<br>`dynamic_quantize_gpu_opt_2327842525564133992_0_0` | 3.0<br>1.0<br>1.0<br>1.0<br>4.0<br>3.0<br>1.0<br>1.0<br>1.0<br>1.0<br>1.0 |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa`<br>`pa_kv_cache_update_ref__15856477779756318941__sa` | 1.0<br>1.0 |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 1.0 |
| FC linattn in_proj (2048->12288) | `gemm_kernel`<br>`dynamic_quantize_gpu_opt_8502581394328941662_0_0` | 1.0<br>1.0 |
| FC qkv+gate (2048->9216) | `gemm_kernel`<br>`dynamic_quantize_gpu_opt_8502581394328941662_0_0` | 1.0<br>1.0 |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel`<br>`dynamic_quantize_gpu_opt_1534384422443530166_0_0` | 1.0<br>1.0 |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_1416219266873627125_0_0` | 1.0 |
| rope_q (16x256) | `rope_opt__11688920003865995564` | 1.0 |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_10752814223568689025_0_0` | 1.0 |
| LM head (last token, 2048->248320) | `gemm_kernel` | 1.0 |

## Per-kernel decomposition (cliloader kernel names)

### Decode sub-kernels — KV=4352 (prompt 4096, representative)

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.1542 | 1.0 | 1 | 5.1542 | 24.1% |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1295 | 1.0 | 30 | 3.8853 | 18.2% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_6438761271185273278` | 0.0965 | 1.0 | 40 | 3.8614 | 18.1% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_down_6438761271185273278` | 0.0463 | 1.0 | 40 | 1.8518 | 8.7% |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0460 | 1.0 | 40 | 1.8408 | 8.6% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `gemm_kernel` | 0.0328 | 4.0 | 40 | 1.3128 | 6.1% |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0963 | 1.0 | 10 | 0.9627 | 4.5% |
| PagedAttention | `paged_attention_opt__gqa_single_token_15856477779756318941__sa` | 0.0826 | 1.0 | 10 | 0.8263 | 3.9% |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 0.0178 | 1.0 | 30 | 0.5338 | 2.5% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_fuse_softmax_topk_2566375436411285507` | 0.0063 | 1.0 | 40 | 0.2528 | 1.2% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `dynamic_quantize_gpu_opt_2680479405095583548_0_0` | 0.0042 | 3.0 | 40 | 0.1664 | 0.8% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `reorder_data_fast_b1_2055831431303789805_0_0` | 0.0030 | 1.0 | 40 | 0.1200 | 0.6% |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 1.0 | 80 | 0.1199 | 0.6% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `dynamic_quantize_gpu_opt_11767888948171086441_0_0` | 0.0019 | 1.0 | 40 | 0.0756 | 0.4% |
| PagedAttention | `paged_attention_opt__single_token_finalization_15856477779756318941__sa` | 0.0060 | 1.0 | 10 | 0.0598 | 0.3% |
| PagedAttention | `pa_kv_cache_update_ref__15856477779756318941__sa` | 0.0058 | 1.0 | 10 | 0.0580 | 0.3% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `eltwise_simple_vload8_12929541252905950109_0_0` | 0.0014 | 1.0 | 40 | 0.0566 | 0.3% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `reorder_data_fast_b1_17423095617980708425_0_0` | 0.0013 | 1.0 | 40 | 0.0504 | 0.2% |

### Prefill sub-kernels — S=8192 (representative)

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 16.9907 | 3.0 | 40 | 679.6295 | 22.9% |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 19.6472 | 1.0 | 30 | 589.4155 | 19.9% |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 42.0609 | 1.0 | 10 | 420.6095 | 14.2% |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 7.5120 | 1.0 | 30 | 225.3597 | 7.6% |
| MoE grouped-gemm (TK=8 + shared) | `moe_scatter_reduction_opt_moe_scatter_reduction_ref_10772290679749044717` | 5.3873 | 1.0 | 40 | 215.4904 | 7.3% |
| MoE grouped-gemm (TK=8 + shared) | `moe_gather_ref_prefill_gather_10772290679749044717` | 5.0500 | 1.0 | 40 | 202.0008 | 6.8% |
| MoE grouped-gemm (TK=8 + shared) | `moe_3gemm_swiglu_fuse_prefill_swiglu_10772290679749044717` | 1.9586 | 1.0 | 40 | 78.3446 | 2.6% |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 1.8035 | 1.0 | 40 | 72.1404 | 2.4% |
| MoE grouped-gemm (TK=8 + shared) | `gemm_kernel` | 1.6336 | 4.0 | 40 | 65.3428 | 2.2% |
| MoE grouped-gemm (TK=8 + shared) | `dynamic_quantize_gpu_opt_10707039156225127723_0_0` | 1.4898 | 3.0 | 40 | 59.5933 | 2.0% |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 5.6475 | 1.0 | 10 | 56.4750 | 1.9% |
| MoE grouped-gemm (TK=8 + shared) | `reorder_data_fast_b1_18330753658507001113_0_0` | 1.1956 | 1.0 | 40 | 47.8233 | 1.6% |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_1416219266873627125_0_0` | 0.5866 | 1.0 | 80 | 46.9273 | 1.6% |
| FC o_proj / GDN out (4096->2048) | `dynamic_quantize_gpu_opt_1534384422443530166_0_0` | 1.0367 | 1.0 | 40 | 41.4664 | 1.4% |
| MoE grouped-gemm (TK=8 + shared) | `reorder_data_fast_b1_7898771992857082467_0_0` | 1.0314 | 1.0 | 40 | 41.2575 | 1.4% |
| MoE grouped-gemm (TK=8 + shared) | `eltwise_simple_vload8_7979511875005307881_0_0` | 0.9183 | 1.0 | 40 | 36.7312 | 1.2% |
| MoE grouped-gemm (TK=8 + shared) | `moe_3gemm_swiglu_fuse_softmax_topk_10944893001517410604` | 0.7001 | 1.0 | 40 | 28.0050 | 0.9% |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_10752814223568689025_0_0` | 1.5467 | 1.0 | 10 | 15.4671 | 0.5% |

## Top contributors (sorted by total ms per inference)

### Decode

| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1280 | MoE 3-gemm (TK=8 + shared, NE=256) 7.793ms (37%) | LM head (2048->248320, INT8) 5.154ms (24%) | FC linattn in_proj (2048->12288) 3.885ms (18%) |
| 2304 | MoE 3-gemm (TK=8 + shared, NE=256) 7.793ms (36%) | LM head (2048->248320, INT8) 5.154ms (24%) | FC linattn in_proj (2048->12288) 3.885ms (18%) |
| 4352 | MoE 3-gemm (TK=8 + shared, NE=256) 7.793ms (37%) | LM head (2048->248320, INT8) 5.154ms (24%) | FC linattn in_proj (2048->12288) 3.885ms (18%) |
| 8448 | MoE 3-gemm (TK=8 + shared, NE=256) 7.793ms (35%) | LM head (2048->248320, INT8) 5.154ms (23%) | FC linattn in_proj (2048->12288) 3.885ms (18%) |

### Prefill

| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | MoE grouped-gemm (TK=8 + shared) 383.6ms (73%) | GatedDeltaNet core 75.6ms (14%) | FC linattn in_proj (2048->12288) 29.6ms (6%) |
| 2048 | MoE grouped-gemm (TK=8 + shared) 502.7ms (64%) | GatedDeltaNet core 137.8ms (18%) | FC linattn in_proj (2048->12288) 56.6ms (7%) |
| 4096 | MoE grouped-gemm (TK=8 + shared) 840.6ms (57%) | GatedDeltaNet core 301.2ms (20%) | FC linattn in_proj (2048->12288) 119.1ms (8%) |
| 8192 | MoE grouped-gemm (TK=8 + shared) 1458.8ms (49%) | GatedDeltaNet core 589.4ms (20%) | PagedAttention prefill (causal, NH=16) 422.9ms (14%) |

## End-to-end (prefill TTFT + 512-token decode)

| prompt P | TTFT (ms) | 512-tok decode (ms) | total (ms) | avg decode tok/s |
|---:|---:|---:|---:|---:|
| 1024 | 528.9 | 10,843.0 | 11,372.0 | 47.2 |
| 2048 | 784.0 | 11,085.3 | 11,869.3 | 46.2 |
| 4096 | 1,477.9 | 10,930.0 | 12,407.9 | 46.8 |
| 8192 | 2,964.6 | 11,314.6 | 14,279.2 | 45.3 |

## Key findings

- **Decode throughput is flat at ~45–47 tok/s across all prompt lengths.** The per-token budget is 20.4 ms of KV-independent work; PagedAttention over the 512-token window only moves TPOT by ~0.7–1.7 ms, so prompt length barely affects decode.
- **Three ops own ~80% of decode:** MoE, LM-head, and linear-attn in_proj — all memory-bound INT4/INT8 weight streaming at 90–97% of 105 GB/s.
- **Decode reaches ~82% of the memory roofline** (modelable ops); the remaining gap is mostly MoE (~74%) and PagedAttention (~50%). **Prefill reaches ~46% of its roofline at S=1024, falling to ~24% at S=8192**: MoE grouped-gemm is **memory-bound** — it streams all 256 experts' INT4 weights every layer — yet hits only ~16–42% of weight BW (small per-expert token groups, gather/scatter and INT4 dequant overhead).
- **PA kernel heuristic switch at KV≥4096** (`paged_attention_opt__single_token` → `gqa_single_token`) makes PA-decode non-monotonic: the 4096-prompt window is *faster* than the 2048-prompt window.
- **Prefill (TTFT) is MoE/GDN/PA-bound and grows super-linearly** (0.53 s → 2.96 s for 1K → 8K). PA prefill is quadratic (causal); MoE grouped-gemm dominates the rest.

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
gdn_bench.exe        1 1     16 32 128 4000 150 4 0              :: GatedDeltaNet (paged opt, cache_interval=0)
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
- Decode FC/MoE/LM-head are memory-bound (weights dominate at M=1). Prefill **MoE is still memory-bound** — with mm·TK ≥ NE every layer streams all 256 experts' weights once (the byte model uses min(NE, mm·TK) experts, not TK); prefill **FC** is INT8 XMX compute-bound at large S; PA prefill (S≥2048) is FP16 micro-kernel compute-bound.
- PA decode is memory-bound (INT8 KV cache + FP16 Q/out); lm_head runs once per token.
- q_norm/k_norm and residual-add are <0.1% of TTFT and omitted from prefill totals.
