# Qwen3.6-35B-A3B (INT4 g64) — Roofline on Intel Arc B390 (Panther Lake, 12Xe iGPU) (20260611)

**Platform**: Intel Arc B390 (Panther Lake, 12Xe iGPU), Xe2; 105 GB/s achievable read (spec 110); FP16 XMX 58.982 TFLOPS, INT8 XMX 117.964 TOPS.
**Model**: text decoder, 40 layers = 10 full-attention + 30 linear-attention (GatedDeltaNet); MoE every layer.

- 40 decoder blocks; full-attn every 4th layer; attn_output_gate=true (fused QKV+gate width = 2·16·256 + 2·2·256 = 9216)
- MoE: 256 experts, top-8, expert intermediate 512, always-on shared expert intermediate 512
- MatMul weights INT4 g64 / FP16 act; LM_head INT8 per-token / FP16 act; KV cache INT8
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

Per-layer weight matrices (one decoder block) and global weights (INT4 = K·N/2 + FP16 scale + INT4 zp at g64; INT8 per-token = K·N + N·FP16 scale):

| Weight | Shape (K × N) | Quant | MB / instance | × Count | Total MB |
|---|---:|---|---:|---:|---:|
| Embedding (gather) | 248320 × 2048 | INT8 per-token | 508.56 | 1 | 508.6 |
| FC_QKV+gate (full-attn) | 2048 × 9216 | INT4 g64 | 10.17 | 10 | 101.7 |
| FC_O / GDN out_proj | 4096 × 2048 | INT4 g64 | 4.52 | 40 | 180.9 |
| Lin-attn in_proj (GDN) | 2048 × 12288 | INT4 g64 | 13.57 | 30 | 407.0 |
| MoE expert gate+up | 2048 × 1024 | INT4 g64 | 1.13 | 10240 | 11,576.3 |
| MoE expert down | 512 × 2048 | INT4 g64 | 0.57 | 10240 | 5,788.1 |
| MoE shared gate+up | 2048 × 1024 | INT4 g64 | 1.13 | 40 | 45.2 |
| MoE shared down | 512 × 2048 | INT4 g64 | 0.57 | 40 | 22.6 |
| MoE router | 2048 × 256 | FP16 | 1.05 | 40 | 41.9 |
| LM_Head | 2048 × 248320 | INT8 per-token | 509.06 | 1 | 509.1 |
| **Total static weights** |  |  |  |  | **19.18 GB** |

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
| 1024 | 612.0 | 0.612 | 0.598 | 1,673 |
| 2048 | 889.1 | 0.889 | 0.434 | 2,303 |
| 4096 | 1,582.1 | 1.582 | 0.386 | 2,589 |
| 8192 | 3,097.7 | 3.098 | 0.378 | 2,645 |

### Decode — TPOT (per output token, mid 512-gen window KV = P+256)

| prompt P | KV (mid) | TPOT (ms) | tokens/s |
|---:|---:|---:|---:|
| 1024 | 1280 | 22.359 | 44.7 |
| 2048 | 2304 | 22.849 | 43.8 |
| 4096 | 4352 | 22.525 | 44.4 |
| 8192 | 8448 | 23.300 | 42.9 |

### Decode — full 512-token generation window (PA grows with KV)

`start` = KV at first generated token (P); `mean` = KV P+256; `end` = KV P+512. Only PagedAttention scales with KV; the other 21.58 ms/token is M=1 constant.

| prompt P | TPOT start (ms) | TPOT mean (ms) | TPOT end (ms) | 512-tok decode (ms) | decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 1024 | 22.244 | 22.359 | 22.501 | 11,448.0 | 44.7 |
| 2048 | 22.722 | 22.849 | 22.954 | 11,698.6 | 43.8 |
| 4096 | 22.516 | 22.525 | 22.634 | 11,532.7 | 44.4 |
| 8192 | 23.265 | 23.300 | 23.321 | 11,929.6 | 42.9 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)

| op | P=1024 (KV1280) | P=2048 (KV2304) | P=4096 (KV4352) | P=8192 (KV8448) |
|---|---:|---:|---:|---:|
| MoE 3-gemm (TK=8 + shared, NE=256) | 8.421 (37.7%) | 8.421 (36.9%) | 8.421 (37.4%) | 8.421 (36.1%) |
| LM head (2048->248320, INT8 per-token) | 5.136 (23.0%) | 5.136 (22.5%) | 5.136 (22.8%) | 5.136 (22.0%) |
| FC linattn in_proj (2048->12288) | 4.059 (18.2%) | 4.059 (17.8%) | 4.059 (18.0%) | 4.059 (17.4%) |
| FC o_proj / GDN out (4096->2048) | 1.935 (8.7%) | 1.935 (8.5%) | 1.935 (8.6%) | 1.935 (8.3%) |
| FC qkv+gate (2048->9216) | 1.039 (4.6%) | 1.039 (4.5%) | 1.039 (4.6%) | 1.039 (4.5%) |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | 0.521 (2.3%) | 0.521 (2.3%) | 0.521 (2.3%) | 0.521 (2.2%) |
| rmsnorm (H=2048) | 0.235 (1.0%) | 0.235 (1.0%) | 0.235 (1.0%) | 0.235 (1.0%) |
| residual add (H=2048) | 0.124 (0.6%) | 0.124 (0.5%) | 0.124 (0.6%) | 0.124 (0.5%) |
| q_norm (16x256) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) |
| rope_q (16x256) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) |
| k_norm (2x256) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) | 0.025 (0.1%) |
| rope_k (2x256) | 0.022 (0.1%) | 0.022 (0.1%) | 0.022 (0.1%) | 0.022 (0.1%) |
| attn gate x*sigmoid(y) (H=4096) | 0.015 (0.1%) | 0.015 (0.1%) | 0.015 (0.1%) | 0.015 (0.1%) |
| PagedAttention (×10) | 0.777 (3.5%) | 1.266 (5.5%) | 0.942 (4.2%) | 1.717 (7.4%) |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)

| op | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---:|---:|---:|---:|
| MoE grouped-gemm (TK=8 + shared) | 451.22 (73.7%) | 572.02 (64.3%) | 891.94 (56.4%) | 1489.67 (48.1%) |
| PagedAttention prefill (causal, NH=16) | 6.68 (1.1%) | 24.35 (2.7%) | 94.23 (6.0%) | 397.76 (12.8%) |
| GatedDeltaNet core | 70.46 (11.5%) | 138.04 (15.5%) | 293.51 (18.6%) | 591.52 (19.1%) |
| FC linattn in_proj (2048->12288) | 38.13 (6.2%) | 71.33 (8.0%) | 140.54 (8.9%) | 284.90 (9.2%) |
| FC qkv+gate (2048->9216) | 9.72 (1.6%) | 18.17 (2.0%) | 36.99 (2.3%) | 73.49 (2.4%) |
| FC o_proj / GDN out (4096->2048) | 24.21 (4.0%) | 46.30 (5.2%) | 86.18 (5.4%) | 180.67 (5.8%) |
| rmsnorm (H=2048) | 3.91 (0.6%) | 7.41 (0.8%) | 19.67 (1.2%) | 45.53 (1.5%) |
| rope_q (16x256) | 1.18 (0.2%) | 2.89 (0.3%) | 6.41 (0.4%) | 13.64 (0.4%) |
| attn gate x*sigmoid(y) (H=4096) | 1.33 (0.2%) | 3.48 (0.4%) | 7.46 (0.5%) | 15.40 (0.5%) |
| LM head (last token, 2048->248320, INT8 per-token) | 5.14 (0.8%) | 5.14 (0.6%) | 5.14 (0.3%) | 5.14 (0.2%) |

## Roofline: theoretical floor vs measured

**Theoretical floor** = sum over analytically-modelable ops of max(bytes / BW, FLOP / XMX-peak) - the fastest this HW could run each op given its memory traffic / compute. **Measured** is the summed cliloader kernel time of the same ops. **achieved % = theoretical / measured** (how close real kernels get to the roofline ceiling; 100% = on the roofline). GatedDeltaNet uses a recurrent `*_opt` kernel (measured with `cache_interval=0`, i.e. a single final state snapshot) with no analytic model, so it is reported separately as *unmodeled GDN* and excluded from the ratio; `full` = measured + unmodeled = the real TPOT / TTFT.

### Decode (per output token, mid 512-gen window KV = P+256)

| prompt P | KV | theoretical (ms) | measured (ms) | achieved % | unmodeled GDN (ms) | full TPOT (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1280 | 17.404 | 21.838 | 79.7% | 0.521 | 22.359 |
| 2048 | 2304 | 17.504 | 22.327 | 78.4% | 0.521 | 22.849 |
| 4096 | 4352 | 17.704 | 22.003 | 80.5% | 0.521 | 22.525 |
| 8192 | 8448 | 18.103 | 22.779 | 79.5% | 0.521 | 23.300 |

### Prefill (TTFT over S tokens)

| S | theoretical (ms) | measured (ms) | achieved % | unmodeled GDN (ms) | full TTFT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 215.7 | 541.5 | 39.8% | 70.5 | 612.0 |
| 2048 | 253.7 | 751.1 | 33.8% | 138.0 | 889.1 |
| 4096 | 351.7 | 1,288.5 | 27.3% | 293.5 | 1,582.1 |
| 8192 | 585.8 | 2,506.2 | 23.4% | 591.5 | 3,097.7 |

## Decode tables (1 query token, KV = mid 512-gen window context)

### Decode — KV=1280 (prompt 1024, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_8996202617093540813` | 0.2105 | 40 | 8.4215 | 269 | 72.6 | 69% | mem |
| LM head (2048->248320, INT8 per-token) | `gemm_kernel` | 5.1356 | 1 | 5.1356 | 198 | 99.2 | 94% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1353 | 30 | 4.0586 | 372 | 100.5 | 96% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0484 | 40 | 1.9354 | 347 | 93.7 | 89% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.1039 | 10 | 1.0389 | 363 | 98.1 | 93% | mem |
| PagedAttention (i8 KV=1280) | `paged_attention_opt__single_token_15856477779756318941__sa` | 0.0777 | 10 | 0.7766 | 270 | 16.9 | 16% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 0.0174 | 30 | 0.5214 | — | — | — | recurrent |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_12510323107858754751_0_0` | 0.0029 | 80 | 0.2347 | 6 | 2.8 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1240 | 1 | 7.9 | 8% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0255 | 16 | 6.4 | 6% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0252 | 13 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0246 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0221 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **22.359** |  |  |  |  |

### Decode — KV=2304 (prompt 2048, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_8996202617093540813` | 0.2105 | 40 | 8.4215 | 269 | 72.6 | 69% | mem |
| LM head (2048->248320, INT8 per-token) | `gemm_kernel` | 5.1356 | 1 | 5.1356 | 198 | 99.2 | 94% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1353 | 30 | 4.0586 | 372 | 100.5 | 96% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0484 | 40 | 1.9354 | 347 | 93.7 | 89% | mem |
| PagedAttention (i8 KV=2304) | `paged_attention_opt__single_token_15856477779756318941__sa` | 0.1266 | 10 | 1.2660 | 298 | 18.6 | 18% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.1039 | 10 | 1.0389 | 363 | 98.1 | 93% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 0.0174 | 30 | 0.5214 | — | — | — | recurrent |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_12510323107858754751_0_0` | 0.0029 | 80 | 0.2347 | 6 | 2.8 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1240 | 1 | 7.9 | 8% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0255 | 16 | 6.4 | 6% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0252 | 13 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0246 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0221 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **22.849** |  |  |  |  |

### Decode — KV=4352 (prompt 4096, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_8996202617093540813` | 0.2105 | 40 | 8.4215 | 269 | 72.6 | 69% | mem |
| LM head (2048->248320, INT8 per-token) | `gemm_kernel` | 5.1356 | 1 | 5.1356 | 198 | 99.2 | 94% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1353 | 30 | 4.0586 | 372 | 100.5 | 96% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0484 | 40 | 1.9354 | 347 | 93.7 | 89% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.1039 | 10 | 1.0389 | 363 | 98.1 | 93% | mem |
| PagedAttention (i8 KV=4352) | `paged_attention_opt__gqa_single_token_15856477779756318941__sa` | 0.0942 | 10 | 0.9420 | 757 | 47.3 | 45% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 0.0174 | 30 | 0.5214 | — | — | — | recurrent |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_12510323107858754751_0_0` | 0.0029 | 80 | 0.2347 | 6 | 2.8 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1240 | 1 | 7.9 | 8% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0255 | 16 | 6.4 | 6% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0252 | 13 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0246 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0221 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **22.525** |  |  |  |  |

### Decode — KV=8448 (prompt 8192, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_8996202617093540813` | 0.2105 | 40 | 8.4215 | 269 | 72.6 | 69% | mem |
| LM head (2048->248320, INT8 per-token) | `gemm_kernel` | 5.1356 | 1 | 5.1356 | 198 | 99.2 | 94% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1353 | 30 | 4.0586 | 372 | 100.5 | 96% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0484 | 40 | 1.9354 | 347 | 93.7 | 89% | mem |
| PagedAttention (i8 KV=8448) | `paged_attention_opt__gqa_single_token_15856477779756318941__sa` | 0.1717 | 10 | 1.7173 | 806 | 50.4 | 48% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.1039 | 10 | 1.0389 | 363 | 98.1 | 93% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 0.0174 | 30 | 0.5214 | — | — | — | recurrent |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_12510323107858754751_0_0` | 0.0029 | 80 | 0.2347 | 6 | 2.8 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 80 | 0.1240 | 1 | 7.9 | 8% | mem |
| rope_q (16x256) | `rope_opt__4161905052046582694` | 0.0025 | 10 | 0.0255 | 16 | 6.4 | 6% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_3614857902334376871_0_0` | 0.0025 | 10 | 0.0252 | 13 | 6.5 | 6% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_14080521285413255093_0_0` | 0.0025 | 10 | 0.0246 | 2 | 0.8 | 1% | mem |
| rope_k (2x256) | `rope_opt__13446196000903736985` | 0.0022 | 10 | 0.0221 | 2 | 0.9 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_18444383029630729382_0_0` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **23.300** |  |  |  |  |

## Prefill tables (single forward over S tokens)

### Prefill — S=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 11.2805 | 40 | 451.2188 | 5,140 | 40.7 | 39% | mem |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 2.3487 | 30 | 70.4620 | — | — | — | recurrent |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 1.2711 | 30 | 38.1331 | 40,547 | 43.5 | 41% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.6053 | 40 | 24.2132 | 28,381 | 35.1 | 33% | mem |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.9722 | 10 | 9.7223 | 39,759 | 43.7 | 42% | mem |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 0.6681 | 10 | 6.6808 | 12,858 | 26.7 | 25% | mem |
| LM head (last token, 2048->248320, INT8 per-token) | `gemm_kernel` | 5.1356 | 1 | 5.1356 | 198 | 99.2 | 94% | mem |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_3188239531497015022_0_0` | 0.0488 | 80 | 3.9058 | 344 | 171.8 | 164% | cache |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_12951181864104916429_0_0` | 0.1331 | 10 | 1.3305 | 158 | 189.1 | 180% | cache |
| rope_q (16x256) | `rope_opt__1263978976134699825` | 0.1179 | 10 | 1.1788 | 356 | 142.3 | 136% | cache |
| **TOTAL** |  |  |  | **611.981** |  |  |  |  |

### Prefill — S=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 14.3006 | 40 | 572.0228 | 8,109 | 33.7 | 32% | mem |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 4.6013 | 30 | 138.0389 | — | — | — | recurrent |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 2.3776 | 30 | 71.3286 | 43,354 | 35.6 | 37% | compute |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 1.1576 | 40 | 46.3038 | 29,682 | 29.2 | 28% | mem |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 2.4346 | 10 | 24.3462 | 14,113 | 14.6 | 24% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 1.8174 | 10 | 18.1735 | 42,540 | 36.1 | 36% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_8465929718177529364_0_0` | 0.0926 | 80 | 7.4115 | 362 | 181.1 | 172% | cache |
| LM head (last token, 2048->248320, INT8 per-token) | `gemm_kernel` | 5.1356 | 1 | 5.1356 | 198 | 99.2 | 94% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_13390302583414783927_0_0` | 0.3480 | 10 | 3.4801 | 121 | 144.6 | 138% | cache |
| rope_q (16x256) | `rope_opt__13556955710875957727` | 0.2888 | 10 | 2.8881 | 290 | 116.2 | 111% | cache |
| **TOTAL** |  |  |  | **889.129** |  |  |  |  |

### Prefill — S=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 22.2985 | 40 | 891.9389 | 10,401 | 23.7 | 23% | mem |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 9.7837 | 30 | 293.5116 | — | — | — | recurrent |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 4.6846 | 30 | 140.5371 | 44,008 | 30.6 | 37% | compute |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 9.4230 | 10 | 94.2303 | 14,585 | 7.6 | 25% | compute |
| FC o_proj / GDN out (4096->2048) | `dynamic_quantize_gpu_opt_13641002643722131644_0_0` | 2.1545 | 40 | 86.1808 | 31,895 | 27.4 | 27% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 3.6993 | 10 | 36.9933 | 41,796 | 30.2 | 35% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_2185109744823398906_0_0` | 0.2458 | 80 | 19.6655 | 273 | 136.5 | 130% | cache |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_1600471834299485859_0_0` | 0.7460 | 10 | 7.4597 | 112 | 134.9 | 129% | cache |
| rope_q (16x256) | `rope_opt__11913003717797137888` | 0.6405 | 10 | 6.4055 | 262 | 104.8 | 100% | mem |
| LM head (last token, 2048->248320, INT8 per-token) | `gemm_kernel` | 5.1356 | 1 | 5.1356 | 198 | 99.2 | 94% | mem |
| **TOTAL** |  |  |  | **1,582.058** |  |  |  |  |

### Prefill — S=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 37.2418 | 40 | 1489.6722 | 12,455 | 16.7 | 16% | mem |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 19.7172 | 30 | 591.5157 | — | — | — | recurrent |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 39.7760 | 10 | 397.7603 | 13,821 | 3.6 | 23% | compute |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 9.4968 | 30 | 284.9042 | 43,416 | 27.5 | 37% | compute |
| FC o_proj / GDN out (4096->2048) | `dynamic_quantize_gpu_opt_1862179630816774010_0_0` | 4.5168 | 40 | 180.6724 | 30,428 | 24.2 | 26% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 7.3491 | 10 | 73.4908 | 42,078 | 27.8 | 36% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_1416219266873627125_0_0` | 0.5691 | 80 | 45.5258 | 236 | 117.9 | 112% | cache |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_10752814223568689025_0_0` | 1.5397 | 10 | 15.3972 | 109 | 130.8 | 125% | cache |
| rope_q (16x256) | `rope_opt__11688920003865995564` | 1.3636 | 10 | 13.6361 | 246 | 98.4 | 94% | mem |
| LM head (last token, 2048->248320, INT8 per-token) | `gemm_kernel` | 5.1356 | 1 | 5.1356 | 198 | 99.2 | 94% | mem |
| **TOTAL** |  |  |  | **3,097.710** |  |  |  |  |

## Op → kernel names (cliloader)

Each logical op and the actual GPU kernel(s) it dispatches (one bench process per op). Kernels are listed in launch order; `launches/call` is how many times each kernel fires per single op invocation (per layer). Decode is measured at M=1, prefill at S=8192 — kernel selection can vary with shape.

### Decode (M=1)

| op | kernel name(s) | launches/call |
|---|---|---:|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_8996202617093540813`<br>`moe_3gemm_swiglu_mlp_down_8996202617093540813`<br>`gemm_kernel`<br>`moe_3gemm_swiglu_fuse_softmax_topk_8996202617093540813`<br>`dynamic_quantize_gpu_opt_2680479405095583548_0_0`<br>`dynamic_quantize_gpu_opt_11767888948171086441_0_0`<br>`reorder_data_fast_b1_2055831431303789805_0_0`<br>`eltwise_simple_vload8_12929541252905950109_0_0`<br>`moe_3gemm_swiglu_mlp_reduce_8996202617093540813`<br>`reorder_data_fast_b1_17423095617980708425_0_0` | 1.0<br>1.0<br>4.0<br>1.0<br>3.0<br>1.0<br>1.0<br>1.0<br>1.0<br>1.0 |
| LM head (2048->248320, INT8 per-token) | `gemm_kernel` | 1.0 |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 1.0 |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 1.0 |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 1.0 |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 1.0 |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_12510323107858754751_0_0` | 1.0 |
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
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm`<br>`moe_scatter_reduction_opt_moe_scatter_reduction_ref_2652845305038675812`<br>`moe_gather_ref_prefill_gather_2652845305038675812`<br>`moe_3gemm_swiglu_fuse_prefill_swiglu_2652845305038675812`<br>`gemm_kernel`<br>`dynamic_quantize_gpu_opt_10707039156225127723_0_0`<br>`reorder_data_fast_b1_18330753658507001113_0_0`<br>`reorder_data_fast_b1_7898771992857082467_0_0`<br>`eltwise_simple_vload8_7979511875005307881_0_0`<br>`moe_3gemm_swiglu_fuse_softmax_topk_2652845305038675812`<br>`dynamic_quantize_gpu_opt_2327842525564133992_0_0` | 3.0<br>1.0<br>1.0<br>1.0<br>4.0<br>3.0<br>1.0<br>1.0<br>1.0<br>1.0<br>1.0 |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa`<br>`pa_kv_cache_update_ref__15856477779756318941__sa` | 1.0<br>1.0 |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 1.0 |
| FC linattn in_proj (2048->12288) | `gemm_kernel`<br>`dynamic_quantize_gpu_opt_6627171168508981115_0_0` | 1.0<br>1.0 |
| FC qkv+gate (2048->9216) | `gemm_kernel`<br>`dynamic_quantize_gpu_opt_6627171168508981115_0_0` | 1.0<br>1.0 |
| FC o_proj / GDN out (4096->2048) | `dynamic_quantize_gpu_opt_1862179630816774010_0_0`<br>`gemm_kernel` | 1.0<br>1.0 |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_1416219266873627125_0_0` | 1.0 |
| rope_q (16x256) | `rope_opt__11688920003865995564` | 1.0 |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_10752814223568689025_0_0` | 1.0 |
| LM head (last token, 2048->248320, INT8 per-token) | `gemm_kernel` | 1.0 |

## Per-kernel decomposition (cliloader kernel names)

### Decode sub-kernels — KV=4352 (prompt 4096, representative)

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
| LM head (2048->248320, INT8 per-token) | `gemm_kernel` | 5.1356 | 1.0 | 1 | 5.1356 | 22.8% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_8996202617093540813` | 0.1063 | 1.0 | 40 | 4.2508 | 18.9% |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1353 | 1.0 | 30 | 4.0586 | 18.0% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_down_8996202617093540813` | 0.0519 | 1.0 | 40 | 2.0760 | 9.2% |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0484 | 1.0 | 40 | 1.9354 | 8.6% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `gemm_kernel` | 0.0358 | 4.0 | 40 | 1.4309 | 6.4% |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.1039 | 1.0 | 10 | 1.0389 | 4.6% |
| PagedAttention | `paged_attention_opt__gqa_single_token_15856477779756318941__sa` | 0.0826 | 1.0 | 10 | 0.8259 | 3.7% |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__13079710268314424383__sa` | 0.0174 | 1.0 | 30 | 0.5214 | 2.3% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_fuse_softmax_topk_8996202617093540813` | 0.0060 | 1.0 | 40 | 0.2401 | 1.1% |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_12510323107858754751_0_0` | 0.0029 | 1.0 | 80 | 0.2347 | 1.0% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `dynamic_quantize_gpu_opt_2680479405095583548_0_0` | 0.0041 | 3.0 | 40 | 0.1626 | 0.7% |
| residual add (H=2048) | `eltwise_simple_vload8_5822205451716121890_0_0` | 0.0015 | 1.0 | 80 | 0.1240 | 0.6% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `dynamic_quantize_gpu_opt_11767888948171086441_0_0` | 0.0019 | 1.0 | 40 | 0.0751 | 0.3% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `reorder_data_fast_b1_2055831431303789805_0_0` | 0.0015 | 1.0 | 40 | 0.0590 | 0.3% |
| PagedAttention | `paged_attention_opt__single_token_finalization_15856477779756318941__sa` | 0.0059 | 1.0 | 10 | 0.0589 | 0.3% |
| PagedAttention | `pa_kv_cache_update_ref__15856477779756318941__sa` | 0.0057 | 1.0 | 10 | 0.0572 | 0.3% |
| MoE 3-gemm (TK=8 + shared, NE=256) | `eltwise_simple_vload8_12929541252905950109_0_0` | 0.0013 | 1.0 | 40 | 0.0526 | 0.2% |

### Prefill sub-kernels — S=8192 (representative)

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 17.6384 | 3.0 | 40 | 705.5370 | 22.8% |
| GatedDeltaNet core | `paged_gated_delta_net_opt__13079710268314424383__sa` | 19.7172 | 1.0 | 30 | 591.5157 | 19.1% |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_15856477779756318941__sa` | 39.5451 | 1.0 | 10 | 395.4508 | 12.8% |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 8.4219 | 1.0 | 30 | 252.6557 | 8.2% |
| MoE grouped-gemm (TK=8 + shared) | `moe_scatter_reduction_opt_moe_scatter_reduction_ref_2652845305038675812` | 5.4942 | 1.0 | 40 | 219.7696 | 7.1% |
| MoE grouped-gemm (TK=8 + shared) | `moe_gather_ref_prefill_gather_2652845305038675812` | 5.0099 | 1.0 | 40 | 200.3946 | 6.5% |
| FC o_proj / GDN out (4096->2048) | `dynamic_quantize_gpu_opt_1862179630816774010_0_0` | 2.3777 | 1.0 | 40 | 95.1085 | 3.1% |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 2.1391 | 1.0 | 40 | 85.5639 | 2.8% |
| MoE grouped-gemm (TK=8 + shared) | `moe_3gemm_swiglu_fuse_prefill_swiglu_2652845305038675812` | 1.9137 | 1.0 | 40 | 76.5479 | 2.5% |
| MoE grouped-gemm (TK=8 + shared) | `gemm_kernel` | 1.7582 | 4.0 | 40 | 70.3278 | 2.3% |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 6.3564 | 1.0 | 10 | 63.5643 | 2.1% |
| MoE grouped-gemm (TK=8 + shared) | `dynamic_quantize_gpu_opt_10707039156225127723_0_0` | 1.4688 | 3.0 | 40 | 58.7533 | 1.9% |
| MoE grouped-gemm (TK=8 + shared) | `reorder_data_fast_b1_18330753658507001113_0_0` | 1.1914 | 1.0 | 40 | 47.6541 | 1.5% |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_1416219266873627125_0_0` | 0.5691 | 1.0 | 80 | 45.5258 | 1.5% |
| MoE grouped-gemm (TK=8 + shared) | `reorder_data_fast_b1_7898771992857082467_0_0` | 1.0496 | 1.0 | 40 | 41.9833 | 1.4% |
| MoE grouped-gemm (TK=8 + shared) | `eltwise_simple_vload8_7979511875005307881_0_0` | 0.9291 | 1.0 | 40 | 37.1646 | 1.2% |
| FC linattn in_proj (2048->12288) | `dynamic_quantize_gpu_opt_6627171168508981115_0_0` | 1.0749 | 1.0 | 30 | 32.2485 | 1.0% |
| MoE grouped-gemm (TK=8 + shared) | `moe_3gemm_swiglu_fuse_softmax_topk_2652845305038675812` | 0.6752 | 1.0 | 40 | 27.0100 | 0.9% |

## Top contributors (sorted by total ms per inference)

### Decode

| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1280 | MoE 3-gemm (TK=8 + shared, NE=256) 8.421ms (38%) | LM head (2048->248320, INT8 per-token) 5.136ms (23%) | FC linattn in_proj (2048->12288) 4.059ms (18%) |
| 2304 | MoE 3-gemm (TK=8 + shared, NE=256) 8.421ms (37%) | LM head (2048->248320, INT8 per-token) 5.136ms (22%) | FC linattn in_proj (2048->12288) 4.059ms (18%) |
| 4352 | MoE 3-gemm (TK=8 + shared, NE=256) 8.421ms (37%) | LM head (2048->248320, INT8 per-token) 5.136ms (23%) | FC linattn in_proj (2048->12288) 4.059ms (18%) |
| 8448 | MoE 3-gemm (TK=8 + shared, NE=256) 8.421ms (36%) | LM head (2048->248320, INT8 per-token) 5.136ms (22%) | FC linattn in_proj (2048->12288) 4.059ms (17%) |

### Prefill

| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | MoE grouped-gemm (TK=8 + shared) 451.2ms (74%) | GatedDeltaNet core 70.5ms (12%) | FC linattn in_proj (2048->12288) 38.1ms (6%) |
| 2048 | MoE grouped-gemm (TK=8 + shared) 572.0ms (64%) | GatedDeltaNet core 138.0ms (16%) | FC linattn in_proj (2048->12288) 71.3ms (8%) |
| 4096 | MoE grouped-gemm (TK=8 + shared) 891.9ms (56%) | GatedDeltaNet core 293.5ms (19%) | FC linattn in_proj (2048->12288) 140.5ms (9%) |
| 8192 | MoE grouped-gemm (TK=8 + shared) 1489.7ms (48%) | GatedDeltaNet core 591.5ms (19%) | PagedAttention prefill (causal, NH=16) 397.8ms (13%) |

## End-to-end (prefill TTFT + 512-token decode)

| prompt P | TTFT (ms) | 512-tok decode (ms) | total (ms) | avg decode tok/s |
|---:|---:|---:|---:|---:|
| 1024 | 612.0 | 11,448.0 | 12,060.0 | 44.7 |
| 2048 | 889.1 | 11,698.6 | 12,587.7 | 43.8 |
| 4096 | 1,582.1 | 11,532.7 | 13,114.7 | 44.4 |
| 8192 | 3,097.7 | 11,929.6 | 15,027.3 | 42.9 |

## Key findings

- **Decode throughput is flat at ~43–45 tok/s across all prompt lengths.** The per-token budget is 21.6 ms of KV-independent work; PagedAttention over the 512-token window only moves TPOT by ~0.7–1.7 ms, so prompt length barely affects decode.
- **Three ops own ~80% of decode:** MoE, LM-head, and linear-attn in_proj — all memory-bound weight streaming. LM-head and in_proj reach 94–96% of 105 GB/s; MoE is limited to ~69% by M=1 micro-gemm overhead.
- **Decode reaches ~80% of the memory roofline** (modelable ops); the remaining gap is mostly MoE (~69%) and PagedAttention (~16–48%). **Prefill reaches ~40% of its roofline at S=1024, falling to ~23% at S=8192**: MoE grouped-gemm is **memory-bound** — it streams all 256 experts' INT4 weights every layer — yet hits only ~16–39% of weight BW (small per-expert token groups, gather/scatter and INT4 dequant overhead).
- **PA kernel heuristic switch at KV≥4096** (`paged_attention_opt__single_token` → `gqa_single_token`) makes PA-decode non-monotonic: the 4096-prompt window is *faster* than the 2048-prompt window.
- **Prefill (TTFT) is MoE/GDN/PA-bound and grows super-linearly** (0.61 s → 3.10 s for 1K → 8K). PA prefill is quadratic (causal); MoE grouped-gemm dominates the rest.

## Optimization levers (highest decode ROI first)

1. **MoE expert streaming** — BW-bound at only ~69% efficiency (M=1 micro-gemms have launch/scheduling overhead); batching decode requests amortizes weight reads, and the unfused shared expert (3 extra FCs) is a fusion opportunity.
2. **LM-head** — 508 MB of INT8 weights streamed every token at ~94% BW; INT4 LM-head would roughly halve it; speculative decoding removes it from the per-token path.
3. **Linear-attn in_proj** — already ~96% BW-bound; only lower precision or fewer linear layers help.
4. **Memory bandwidth is the global ceiling** — every major decode op is mem-bound, so the 105 GB/s shared LPDDR5x read BW sets the floor.

## Reproduction

Built on the PTL host (VS2022, `OV_SRC_DIR` for `gdn_bench`); each case run under cliloader `-d` and parsed from Device Performance Timing. Driver: `utils/run_qwen3_6_ptl_int4g64.bat`. Representative commands:

```bat
:: decode (M=1) — KV-independent ops (INT4 body group_size=64)
moe_bench.exe        1 1    2048 512 256 8 64 100 10 4 64 512   :: MoE TK=8 + shared, g64
fc_bench.exe         1 2048   9216 64 5000 200 8 u4 64          :: qkv+gate fused, g64
fc_bench.exe         1 2048  12288 64 1500 100 4 u4 64          :: linear-attn in_proj, g64
fc_bench.exe         1 2048 248320 2048 300 30 4 u8 64          :: LM-head INT8 per-token (gs=K)
gdn_bench.exe        1 1     16 32 128 4000 150 4 0              :: GatedDeltaNet (paged opt, cache_interval=0)
small_ops_bench.exe  gate    1 4096 --iters 30000 --warmup 300   :: attn output gate
:: decode PA — 512-token window sweep (KV = P / P+256 / P+512)
pa_bench.exe decode  1 <KV> 8000 200 4 i8   (PA_NH=16 PA_NKV=2 PA_HD=256)
:: prefill — per prompt S in {1024,2048,4096,8192}
moe_bench.exe 1 S 2048 512 256 8 64 ...    pa_bench.exe prefill S 0 ... i8
```

Parse + report:
```bash
python3 ../../utils/parse_logs.py logs ptl_metrics.json
python3 build_report.py ptl_metrics.json > SUMMARY_qwen3_6_int4g64_%DATE%.md
```

## Caveats & method

- Each op profiled in its own process via cliloader Device Performance Timing (mean kernel time per iteration); cache flush between iters so weights stream from VRAM. Totals are an upper-bound roofline, not a traced wall-clock.
- FC weight bytes count INT4 weight + FP16 scale/zp at g64 + FP16 act + FP16 out; LM-head is INT8 **per-token** (per-output-channel: one FP16 scale per vocab row).
- **Shared expert is unfused on this build**; the MoE figure times the routed `MOE3GemmFusedCompressed` plus 3 shared FCs together — the real per-layer cost here.
- **GatedDeltaNet bench covers the core op only**; depthwise conv1d (k=4) not modeled (negligible), in_proj counted as `fc_linattn`, out_proj as `fc_o` (4096→2048). It is a recurrent state op with no clean analytic byte/flop model, so `bound=recurrent` and Eff/GFLOPS/GB/s are shown as `—` (only the measured latency is meaningful).
- `bound=cache`: tiny eltwise/norm/rope micro-benches are L2/L3-resident, so their achieved BW exceeds the 110 GB/s streaming spec; in real inference they are fused into the adjacent matmul/attention and contribute <2% — only their latency is used.
- The attention output gate is benched with a `gate` proxy op (x·sigmoid(y), H=4096).
- Decode FC/MoE/LM-head are memory-bound (weights dominate at M=1). Prefill **MoE is still memory-bound** — with mm·TK ≥ NE every layer streams all 256 experts' weights once (the byte model uses min(NE, mm·TK) experts, not TK); prefill **FC** is INT8 XMX compute-bound at large S; PA prefill (S≥2048) is FP16 micro-kernel compute-bound.
- PA decode is memory-bound (INT8 KV cache + FP16 Q/out); lm_head runs once per token.
- q_norm/k_norm and residual-add are <0.1% of TTFT and omitted from prefill totals.
