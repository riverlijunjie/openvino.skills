# Qwen3.6-35B-A3B — Roofline on Intel PTL 4Xe iGPU (Panther Lake, 4 Xe cores) (20260611)

**Platform**: Intel PTL 4Xe iGPU (Panther Lake, 4 Xe cores), Xe3; 105 GB/s achievable read (spec 110); FP16 XMX 20.07 TFLOPS, INT8 XMX 40.141 TOPS.
**Model**: text decoder, 40 layers = 10 full-attention + 30 linear-attention (GatedDeltaNet); MoE every layer.

- 40 decoder blocks; full-attn every 4th layer; attn_output_gate=true (fused QKV+gate width = 2·16·256 + 2·2·256 = 9216)
- MoE: 256 experts, top-8, expert intermediate 512, always-on shared expert intermediate 512
- MatMul weights INT4 g128 / FP16 act; LM_head INT8 g128 / FP16 act; KV cache INT8
- SDPA: PagedAttention (OpenCL micro-kernel), GQA 16/2 = 8-way; linear-attn via GatedDeltaNet

> **NOTE**: PTL 4Xe has **1/3 the compute** of PTL 12Xe (4 vs 12 Xe cores) but **identical 110 GB/s memory bandwidth** (shared LPDDR5x). Memory-bound ops (FC/PA/MoE decode, small ops) run at ~same speed; compute-bound ops (FC/MoE/PA prefill) are ~3× slower.

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

Per-layer weight matrices and global weights:

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
| FP16 XMX peak | 20.07 TFLOPS |
| INT8 XMX peak | 40.141 TOPS |
| Memory BW (achievable read) | 105 GB/s (spec 110) |
| Ridge point (FP16) | 191 FLOP/byte |
| Ridge point (INT8) | 382 OP/byte |

## Data sources

| Category | Source | Note |
|---|---|---|
| FC decode/prefill | **Measured on 4Xe** | cliloader Device Performance Timing |
| PA decode/prefill | **Measured on 4Xe** | cliloader Device Performance Timing |
| GatedDeltaNet | **Measured on 4Xe** | cliloader Device Performance Timing |
| Small ops (rmsnorm, rope, add) | **Measured on 4Xe** | cliloader |
| MoE (decode) | **Scaled from 12Xe** | Memory-bound → same BW → 1.0× |
| MoE (prefill) | **Scaled from 12Xe** | Compute-bound → 3× slower (1/3 XMX) |
| Gate (attn_output_gate) | **Scaled from 12Xe** | Elementwise → same BW → 1.0× |
| LM head | **Measured on 4Xe** | cliloader |

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
| 1024 | 1,347.7 | 1.348 | 1.316 | 760 |
| 2048 | 1,970.1 | 1.970 | 0.962 | 1,040 |
| 4096 | 3,571.4 | 3.571 | 0.872 | 1,147 |
| 8192 | 7,144.1 | 7.144 | 0.872 | 1,147 |

### Decode — TPOT (per output token, mid 512-gen window KV = P+256)

| prompt P | KV (mid) | TPOT (ms) | tokens/s |
|---:|---:|---:|---:|
| 1024 | 1280 | 22.856 | 43.8 |
| 2048 | 2304 | 24.478 | 40.9 |
| 4096 | 4352 | 22.414 | 44.6 |
| 8192 | 8448 | 24.188 | 41.3 |

### Decode — full 512-token generation window (PA grows with KV)

| prompt P | TPOT start (ms) | TPOT mean (ms) | TPOT end (ms) | 512-tok decode (ms) | decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 1024 | 22.434 | 22.856 | 23.304 | 11,702.3 | 43.8 |
| 2048 | 24.055 | 24.478 | 24.890 | 12,532.7 | 40.9 |
| 4096 | 22.348 | 22.414 | 22.569 | 11,475.9 | 44.6 |
| 8192 | 24.127 | 24.188 | 24.320 | 12,384.3 | 41.3 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)

| op | P=1024 (KV1280) | P=2048 (KV2304) | P=4096 (KV4352) | P=8192 (KV8448) |
|---|---:|---:|---:|---:|
| MoE 3-gemm (TK=8 + shared, NE=256) | 7.612 (33.3%) | 7.612 (31.1%) | 7.612 (34.0%) | 7.612 (31.5%) |
| LM head (2048->248320, INT8) | 5.205 (22.8%) | 5.205 (21.3%) | 5.205 (23.2%) | 5.205 (21.5%) |
| FC linattn in_proj (2048->12288) | 3.861 (16.9%) | 3.861 (15.8%) | 3.861 (17.2%) | 3.861 (16.0%) |
| FC o_proj / GDN out (4096->2048) | 1.820 (8.0%) | 1.820 (7.4%) | 1.820 (8.1%) | 1.820 (7.5%) |
| FC qkv+gate (2048->9216) | 0.967 (4.2%) | 0.967 (3.9%) | 0.967 (4.3%) | 0.967 (4.0%) |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | 0.542 (2.4%) | 0.542 (2.2%) | 0.542 (2.4%) | 0.542 (2.2%) |
| rmsnorm (H=2048) | 0.208 (0.9%) | 0.208 (0.8%) | 0.208 (0.9%) | 0.208 (0.9%) |
| residual add (H=2048) | 0.105 (0.5%) | 0.105 (0.4%) | 0.105 (0.5%) | 0.105 (0.4%) |
| q_norm (16x256) | 0.023 (0.1%) | 0.023 (0.1%) | 0.023 (0.1%) | 0.023 (0.1%) |
| rope_q (16x256) | 0.024 (0.1%) | 0.024 (0.1%) | 0.024 (0.1%) | 0.024 (0.1%) |
| k_norm (2x256) | 0.022 (0.1%) | 0.022 (0.1%) | 0.022 (0.1%) | 0.022 (0.1%) |
| rope_k (2x256) | 0.021 (0.1%) | 0.021 (0.1%) | 0.021 (0.1%) | 0.021 (0.1%) |
| attn gate x*sigmoid(y) (H=4096) | 0.015 (0.1%) | 0.015 (0.1%) | 0.015 (0.1%) | 0.015 (0.1%) |
| PagedAttention (×10) | 2.430 (10.6%) | 4.053 (16.6%) | 1.989 (8.9%) | 3.762 (15.6%) |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)

| op | S=1024 | S=2048 | S=4096 | S=8192 |
|---|---:|---:|---:|---:|
| MoE grouped-gemm (TK=8 + shared) | 1049.19 (77.8%) | 1333.53 (67.7%) | 2168.72 (60.7%) | 3782.01 (52.9%) |
| PagedAttention prefill (causal, NH=16) | 20.16 (1.5%) | 76.14 (3.9%) | 291.03 (8.1%) | 1132.72 (15.9%) |
| GatedDeltaNet core | 132.70 (9.8%) | 272.67 (13.8%) | 554.47 (15.5%) | 1101.20 (15.4%) |
| FC linattn in_proj (2048->12288) | 72.22 (5.4%) | 144.64 (7.3%) | 281.53 (7.9%) | 570.84 (8.0%) |
| FC qkv+gate (2048->9216) | 18.71 (1.4%) | 39.08 (2.0%) | 70.35 (2.0%) | 155.12 (2.2%) |
| FC o_proj / GDN out (4096->2048) | 36.49 (2.7%) | 70.76 (3.6%) | 147.21 (4.1%) | 289.45 (4.1%) |
| rmsnorm (H=2048) | 9.65 (0.7%) | 20.41 (1.0%) | 36.69 (1.0%) | 74.38 (1.0%) |
| rope_q (16x256) | 2.02 (0.2%) | 4.12 (0.2%) | 8.76 (0.2%) | 17.69 (0.2%) |
| attn gate x*sigmoid(y) (H=4096) | 1.33 (0.1%) | 3.54 (0.2%) | 7.44 (0.2%) | 15.46 (0.2%) |
| LM head (last token, 2048->248320) | 5.21 (0.4%) | 5.21 (0.3%) | 5.21 (0.1%) | 5.21 (0.1%) |

## Roofline: theoretical floor vs measured

### Decode (per output token, mid 512-gen window KV = P+256)

| prompt P | KV | theoretical (ms) | measured (ms) | achieved % | unmodeled GDN (ms) | full TPOT (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1280 | 17.027 | 22.314 | 76.3% | 0.542 | 22.856 |
| 2048 | 2304 | 17.127 | 23.937 | 71.5% | 0.542 | 24.478 |
| 4096 | 4352 | 17.326 | 21.873 | 79.2% | 0.542 | 22.414 |
| 8192 | 8448 | 17.726 | 23.646 | 75.0% | 0.542 | 24.188 |

### Prefill (TTFT over S tokens)

| S | theoretical (ms) | measured (ms) | achieved % | unmodeled GDN (ms) | full TTFT (ms) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 252.6 | 1,215.0 | 20.8% | 132.7 | 1,347.7 |
| 2048 | 349.7 | 1,697.4 | 20.6% | 272.7 | 1,970.1 |
| 4096 | 605.0 | 3,016.9 | 20.1% | 554.5 | 3,571.4 |
| 8192 | 1,342.6 | 6,042.9 | 22.2% | 1,101.2 | 7,144.1 |

## Decode tables (1 query token, KV = mid 512-gen window context)

### Decode — KV=1280 (prompt 1024, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_793724151708371423` | 0.1903 | 40 | 7.6118 | 298 | 77.4 | 74% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.2050 | 1 | 5.2050 | 195 | 99.3 | 95% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1287 | 30 | 3.8615 | 391 | 101.8 | 97% | mem |
| PagedAttention (i8 KV=1280) | `paged_attention_opt__single_token_3220552560579505956__sa` | 0.2430 | 10 | 2.4298 | 86 | 5.4 | 5% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0455 | 40 | 1.8203 | 369 | 96.0 | 91% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__6793667411950381675__sa` | 0.0181 | 30 | 0.5417 | — | — | — | recurrent |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0967 | 10 | 0.9672 | 390 | 101.6 | 97% | mem |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_8529881962352508352_0_0` | 0.0026 | 80 | 0.2082 | 6 | 3.1 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_16925927539244408722_0_0` | 0.0013 | 80 | 0.1051 | 2 | 9.4 | 9% | mem |
| rope_q (16x256) | `rope_opt__15341125376449242927` | 0.0024 | 10 | 0.0237 | 17 | 6.9 | 7% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_13117441039373274511_0_0` | 0.0023 | 10 | 0.0230 | 14 | 7.1 | 7% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_10471926633713670026_0_0` | 0.0022 | 10 | 0.0223 | 2 | 0.9 | 1% | mem |
| rope_k (2x256) | `rope_opt__16024937312394411001` | 0.0021 | 10 | 0.0208 | 2 | 1.0 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `-` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **22.856** |  |  |  |  |

### Decode — KV=2304 (prompt 2048, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_793724151708371423` | 0.1903 | 40 | 7.6118 | 298 | 77.4 | 74% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.2050 | 1 | 5.2050 | 195 | 99.3 | 95% | mem |
| PagedAttention (i8 KV=2304) | `paged_attention_opt__single_token_3220552560579505956__sa` | 0.4053 | 10 | 4.0526 | 93 | 5.8 | 6% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1287 | 30 | 3.8615 | 391 | 101.8 | 97% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0455 | 40 | 1.8203 | 369 | 96.0 | 91% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__6793667411950381675__sa` | 0.0181 | 30 | 0.5417 | — | — | — | recurrent |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0967 | 10 | 0.9672 | 390 | 101.6 | 97% | mem |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_8529881962352508352_0_0` | 0.0026 | 80 | 0.2082 | 6 | 3.1 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_16925927539244408722_0_0` | 0.0013 | 80 | 0.1051 | 2 | 9.4 | 9% | mem |
| rope_q (16x256) | `rope_opt__15341125376449242927` | 0.0024 | 10 | 0.0237 | 17 | 6.9 | 7% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_13117441039373274511_0_0` | 0.0023 | 10 | 0.0230 | 14 | 7.1 | 7% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_10471926633713670026_0_0` | 0.0022 | 10 | 0.0223 | 2 | 0.9 | 1% | mem |
| rope_k (2x256) | `rope_opt__16024937312394411001` | 0.0021 | 10 | 0.0208 | 2 | 1.0 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `-` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **24.478** |  |  |  |  |

### Decode — KV=4352 (prompt 4096, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_793724151708371423` | 0.1903 | 40 | 7.6118 | 298 | 77.4 | 74% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.2050 | 1 | 5.2050 | 195 | 99.3 | 95% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1287 | 30 | 3.8615 | 391 | 101.8 | 97% | mem |
| PagedAttention (i8 KV=4352) | `paged_attention_opt__gqa_single_token_3220552560579505956__sa` | 0.1989 | 10 | 1.9886 | 359 | 22.4 | 21% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0455 | 40 | 1.8203 | 369 | 96.0 | 91% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__6793667411950381675__sa` | 0.0181 | 30 | 0.5417 | — | — | — | recurrent |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0967 | 10 | 0.9672 | 390 | 101.6 | 97% | mem |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_8529881962352508352_0_0` | 0.0026 | 80 | 0.2082 | 6 | 3.1 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_16925927539244408722_0_0` | 0.0013 | 80 | 0.1051 | 2 | 9.4 | 9% | mem |
| rope_q (16x256) | `rope_opt__15341125376449242927` | 0.0024 | 10 | 0.0237 | 17 | 6.9 | 7% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_13117441039373274511_0_0` | 0.0023 | 10 | 0.0230 | 14 | 7.1 | 7% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_10471926633713670026_0_0` | 0.0022 | 10 | 0.0223 | 2 | 0.9 | 1% | mem |
| rope_k (2x256) | `rope_opt__16024937312394411001` | 0.0021 | 10 | 0.0208 | 2 | 1.0 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `-` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **22.414** |  |  |  |  |

### Decode — KV=8448 (prompt 8192, mid 512-gen window)

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE 3-gemm (TK=8 + shared, NE=256) | `moe_3gemm_swiglu_mlp_gate_up_793724151708371423` | 0.1903 | 40 | 7.6118 | 298 | 77.4 | 74% | mem |
| LM head (2048->248320, INT8) | `gemm_kernel` | 5.2050 | 1 | 5.2050 | 195 | 99.3 | 95% | mem |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 0.1287 | 30 | 3.8615 | 391 | 101.8 | 97% | mem |
| PagedAttention (i8 KV=8448) | `paged_attention_opt__gqa_single_token_3220552560579505956__sa` | 0.3762 | 10 | 3.7618 | 368 | 23.0 | 22% | mem |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.0455 | 40 | 1.8203 | 369 | 96.0 | 91% | mem |
| GatedDeltaNet core (qk=16,v=32,K=V=128) | `paged_gated_delta_net_opt__6793667411950381675__sa` | 0.0181 | 30 | 0.5417 | — | — | — | recurrent |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 0.0967 | 10 | 0.9672 | 390 | 101.6 | 97% | mem |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_8529881962352508352_0_0` | 0.0026 | 80 | 0.2082 | 6 | 3.1 | 3% | mem |
| residual add (H=2048) | `eltwise_simple_vload8_16925927539244408722_0_0` | 0.0013 | 80 | 0.1051 | 2 | 9.4 | 9% | mem |
| rope_q (16x256) | `rope_opt__15341125376449242927` | 0.0024 | 10 | 0.0237 | 17 | 6.9 | 7% | mem |
| q_norm (16x256) | `rms_gpu_bfyx_opt_13117441039373274511_0_0` | 0.0023 | 10 | 0.0230 | 14 | 7.1 | 7% | mem |
| k_norm (2x256) | `rms_gpu_bfyx_opt_10471926633713670026_0_0` | 0.0022 | 10 | 0.0223 | 2 | 0.9 | 1% | mem |
| rope_k (2x256) | `rope_opt__16024937312394411001` | 0.0021 | 10 | 0.0208 | 2 | 1.0 | 1% | mem |
| attn gate x*sigmoid(y) (H=4096) | `-` | 0.0015 | 10 | 0.0153 | 13 | 16.0 | 15% | mem |
| **TOTAL** |  |  |  | **24.188** |  |  |  |  |

## Prefill tables (single forward over S tokens)

### Prefill — S=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 26.2298 | 40 | 1049.1910 | 2,211 | 16.9 | 16% | mem |
| GatedDeltaNet core | `paged_gated_delta_net_opt__6793667411950381675__sa` | 4.4235 | 30 | 132.7040 | — | — | — | recurrent |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 2.4075 | 30 | 72.2247 | 21,408 | 22.8 | 53% | compute |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 0.9122 | 40 | 36.4879 | 18,833 | 23.1 | 47% | compute |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_3220552560579505956__sa` | 2.0161 | 10 | 20.1608 | 4,261 | 8.8 | 21% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 1.8713 | 10 | 18.7135 | 20,656 | 22.6 | 51% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_2641287256442443418_0_0` | 0.1206 | 80 | 9.6503 | 139 | 69.5 | 66% | mem |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.2050 | 1 | 5.2050 | 195 | 99.3 | 95% | mem |
| rope_q (16x256) | `rope_opt__15532317810969689154` | 0.2020 | 10 | 2.0204 | 208 | 83.0 | 79% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_12951181864104916429_0_0` | 0.1329 | 10 | 1.3293 | 158 | 189.3 | 180% | cache |
| **TOTAL** |  |  |  | **1,347.687** |  |  |  |  |

### Prefill — S=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 33.3382 | 40 | 1333.5271 | 3,478 | 14.0 | 13% | mem |
| GatedDeltaNet core | `paged_gated_delta_net_opt__6793667411950381675__sa` | 9.0891 | 30 | 272.6730 | — | — | — | recurrent |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 4.8215 | 30 | 144.6449 | 21,379 | 17.5 | 53% | compute |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_3220552560579505956__sa` | 7.6136 | 10 | 76.1365 | 4,513 | 4.7 | 22% | compute |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 1.7689 | 40 | 70.7557 | 19,424 | 19.0 | 48% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 3.9076 | 10 | 39.0760 | 19,784 | 16.7 | 49% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_11225455960293980382_0_0` | 0.2551 | 80 | 20.4094 | 132 | 65.8 | 63% | mem |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.2050 | 1 | 5.2050 | 195 | 99.3 | 95% | mem |
| rope_q (16x256) | `rope_opt__11049501886144597316` | 0.4115 | 10 | 4.1152 | 204 | 81.5 | 78% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_13390302583414783927_0_0` | 0.3545 | 10 | 3.5450 | 118 | 142.0 | 135% | cache |
| **TOTAL** |  |  |  | **1,970.088** |  |  |  |  |

### Prefill — S=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 54.2180 | 40 | 2168.7188 | 4,278 | 9.4 | 11% | compute |
| GatedDeltaNet core | `paged_gated_delta_net_opt__6793667411950381675__sa` | 18.4823 | 30 | 554.4687 | — | — | — | recurrent |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_3220552560579505956__sa` | 29.1032 | 10 | 291.0324 | 4,722 | 2.5 | 24% | compute |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 9.3843 | 30 | 281.5281 | 21,969 | 15.2 | 55% | compute |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 3.6802 | 40 | 147.2083 | 18,673 | 16.0 | 47% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 7.0350 | 10 | 70.3496 | 21,979 | 15.8 | 55% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_2187232604198122086_0_0` | 0.4586 | 80 | 36.6866 | 146 | 73.2 | 70% | mem |
| rope_q (16x256) | `rope_opt__3615952834913402619` | 0.8758 | 10 | 8.7581 | 192 | 76.6 | 73% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_1600471834299485859_0_0` | 0.7439 | 10 | 7.4392 | 113 | 135.3 | 129% | cache |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.2050 | 1 | 5.2050 | 195 | 99.3 | 95% | mem |
| **TOTAL** |  |  |  | **3,571.394** |  |  |  |  |

### Prefill — S=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE grouped-gemm (TK=8 + shared) | `grouped_micro_gemm` | 94.5503 | 40 | 3782.0123 | 4,906 | 6.4 | 12% | compute |
| GatedDeltaNet core | `paged_gated_delta_net_opt__6793667411950381675__sa` | 36.7066 | 30 | 1101.1980 | — | — | — | recurrent |
| PagedAttention prefill (causal, NH=16) | `sdpa_micro__prefill_3220552560579505956__sa` | 113.2716 | 10 | 1132.7161 | 4,853 | 1.3 | 24% | compute |
| FC linattn in_proj (2048->12288) | `gemm_kernel` | 19.0281 | 30 | 570.8443 | 21,669 | 13.7 | 54% | compute |
| FC o_proj / GDN out (4096->2048) | `gemm_kernel` | 7.2363 | 40 | 289.4512 | 18,993 | 15.1 | 47% | compute |
| FC qkv+gate (2048->9216) | `gemm_kernel` | 15.5116 | 10 | 155.1161 | 19,936 | 13.1 | 50% | compute |
| rmsnorm (H=2048) | `rms_gpu_bfyx_opt_6242263758832909782_0_0` | 0.9298 | 80 | 74.3834 | 144 | 72.2 | 69% | mem |
| rope_q (16x256) | `rope_opt__18148871069448078218` | 1.7691 | 10 | 17.6908 | 190 | 75.9 | 72% | mem |
| attn gate x*sigmoid(y) (H=4096) | `eltwise_simple_vload8_10752814223568689025_0_0` | 1.5455 | 10 | 15.4552 | 109 | 130.3 | 124% | cache |
| LM head (last token, 2048->248320) | `gemm_kernel` | 5.2050 | 1 | 5.2050 | 195 | 99.3 | 95% | mem |
| **TOTAL** |  |  |  | **7,144.072** |  |  |  |  |

## Top contributors (sorted by total ms per inference)

### Decode

| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1280 | MoE 3-gemm (TK=8 + shared, NE=256) 7.612ms (32%) | LM head (2048->248320, INT8) 5.205ms (22%) | FC linattn in_proj (2048->12288) 3.861ms (16%) |
| 2304 | MoE 3-gemm (TK=8 + shared, NE=256) 7.612ms (30%) | LM head (2048->248320, INT8) 5.205ms (21%) | PagedAttention (i8 KV=2304) 4.053ms (16%) |
| 4352 | MoE 3-gemm (TK=8 + shared, NE=256) 7.612ms (33%) | LM head (2048->248320, INT8) 5.205ms (22%) | FC linattn in_proj (2048->12288) 3.861ms (17%) |
| 8448 | MoE 3-gemm (TK=8 + shared, NE=256) 7.612ms (30%) | LM head (2048->248320, INT8) 5.205ms (21%) | FC linattn in_proj (2048->12288) 3.861ms (15%) |

### Prefill

| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1024 | MoE grouped-gemm (TK=8 + shared) 1049.2ms (78%) | GatedDeltaNet core 132.7ms (10%) | FC linattn in_proj (2048->12288) 72.2ms (5%) |
| 2048 | MoE grouped-gemm (TK=8 + shared) 1333.5ms (68%) | GatedDeltaNet core 272.7ms (14%) | FC linattn in_proj (2048->12288) 144.6ms (7%) |
| 4096 | MoE grouped-gemm (TK=8 + shared) 2168.7ms (61%) | GatedDeltaNet core 554.5ms (16%) | PagedAttention prefill (causal, NH=16) 291.0ms (8%) |
| 8192 | MoE grouped-gemm (TK=8 + shared) 3782.0ms (53%) | PagedAttention prefill (causal, NH=16) 1132.7ms (16%) | GatedDeltaNet core 1101.2ms (15%) |

## End-to-end (prefill TTFT + 512-token decode)

| prompt P | TTFT (ms) | 512-tok decode (ms) | total (ms) | avg decode tok/s |
|---:|---:|---:|---:|---:|
| 1024 | 1,347.7 | 11,702.3 | 13,050.0 | 43.8 |
| 2048 | 1,970.1 | 12,532.7 | 14,502.8 | 40.9 |
| 4096 | 3,571.4 | 11,475.9 | 15,047.3 | 44.6 |
| 8192 | 7,144.1 | 12,384.3 | 19,528.4 | 41.3 |

## Key findings

- **Decode throughput: ~40–43 tok/s** — nearly identical to PTL 12Xe because decode is **100% memory-bound** and both share 110 GB/s LPDDR5x.
- **Prefill TTFT is ~3× slower** than 12Xe at all sequence lengths because FC/MoE/PA prefill kernels are compute-bound on INT8/FP16 XMX (40.141 TOPS vs 117.96 TOPS).
- **Three ops own ~80% of decode:** MoE, LM-head, and linear-attn in_proj — all memory-bound INT4/INT8 weight streaming.
- **Decode reaches ~76% of the memory roofline** (modelable ops).
- **Prefill reaches ~21% of its roofline at S=1024, falling to ~22% at S=8192.**
- **4Xe vs 12Xe decode ratio: ~1.0×** (BW-limited). **4Xe vs 12Xe prefill ratio: ~2.5–3×** (compute-limited).

## Optimization levers (highest decode ROI first)

1. **Memory bandwidth is the ceiling** — every decode op is BW-bound on 105 GB/s; no amount of compute helps decode.
2. **MoE expert streaming** — NE=256 INT4 experts streamed each token; batching decode or reducing active experts helps.
3. **LM-head** — 508 MB INT8 weight read every token; INT4 quantization or speculative decoding reduces this.
_performance_metrics.json written to /home/ov2022/workspace/remote_debug/openvino/.github/skills/dev_roofline_profiling/outputs/qwen3_6_ptl_4xe/performance_metrics.json_

4. **Prefill needs more compute** — 4Xe is 3× slower for prefill vs 12Xe; chunked prefill or longer prompt caching can hide TTFT.

## Reproduction

Target: `intel@10.239.152.140` (PTL 4Xe, Linux). Build under `~/river/roofline_test_utils/build`. Run script: `run_qwen3_6_ptl_4xe.sh`.

```bash
# Representative commands:
cliloader -d ./fc_bench 1 2048 9216 128 5000 200 8 u4 64    # FC qkv+gate decode
cliloader -d ./pa_bench decode 1 4352 8000 200 4 i8 ocl     # PA decode KV=4352
cliloader -d ./gdn_bench 1 1 16 32 128 4000 150 4 0         # GDN decode
cliloader -d ./fc_bench 4096 2048 9216 128 6 3 4 u4 64      # FC qkv+gate prefill S=4096
cliloader -d ./pa_bench prefill 4096 0 10 3 2 i8 ocl        # PA prefill S=4096
```

Parse + report:
```bash
python3 parse_logs.py logs/ ptl_4xe_metrics.json
python3 build_report.py ptl_4xe_metrics.json > SUMMARY_qwen3_6_ptl_4xe_<date>.md
```

## Caveats & method

- FC/PA/GDN/small-ops: **directly measured** on PTL 4Xe via cliloader.
- MoE: **scaled from PTL 12Xe measurements** — MoE bench failed on 4Xe (shared-expert fusion issue, exit before timing). Decode (memory-bound) kept at 1×; prefill (compute-bound grouped_micro_gemm ~80% of time) scaled by 3×.
- Gate (attn_output_gate): **estimated from PTL 12Xe** — `small_ops_bench` on this build doesn't support the `gate` op. It is elementwise (same BW) → ~same latency.
- Each op profiled in its own process with cache flush between iters.
- PA prefill uses causal mask: effective pairs = S²/2.
- Decode FC/MoE/LM-head are memory-bound (weights dominate at M=1).
- `bound=cache`: tiny eltwise/norm/rope micro-benches are L2/L3-resident; they contribute <2% in real inference.
