# Qwen3.5-MoE-35B-A3B — Roofline on PTL 12Xe (2026-05-10, v4 INT4 g=64)

**Platform**: PTL (Panther Lake B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s)
**Model**: Qwen/Qwen3.5-MoE-35B-A3B (Hybrid Attention: 10 full-attn + 30 GatedDeltaNet, 256 experts × top-8)

- hidden_size = 2048, 40 layers (10 full-attn + 30 linear-attn GDN), 16 Q heads, 2 KV heads (GQA=8)
- MoE: 256 experts × top-8, intermediate=512, shared expert intermediate=512
- MatMul weights **INT4 asymmetric g=64** / FP16 act; LM_head INT8 g=128 / FP16 act; KV cache INT8
- SDPA: PagedAttention (INT8 KV cache, block_size=16)
- **v4**: Shared expert **fused** into MoE kernel (FuseMOESharedExpert, commit 2abcdce7f3)

## Model parameters & weight shapes

Architecture knobs (parsed from model config):

| Field | Value | Notes |
|---|---:|---|
| `hidden_size` | 2048 | residual / activation channel |
| `num_hidden_layers` | 40 | decoder blocks (hybrid) |
| `num_attention_heads` (NH) | 16 | Q heads (full-attn) |
| `num_key_value_heads` (NKV) | 2 | GQA: 8-way Q-per-KV sharing |
| `head_dim` (HD) | 256 | Q_dim = 16×256 = 4096, KV_dim = 2×256 = 512 |
| `linear_num_key_heads` | 16 | GDN Q/K heads |
| `linear_num_value_heads` | 32 | GDN V heads |
| `linear_key_head_dim` | 128 | GDN key dim per head |
| `linear_value_head_dim` | 128 | GDN value dim per head |
| `moe_intermediate_size` | 512 | per-expert FFN hidden |
| `shared_expert_intermediate_size` | 512 | shared expert hidden (fused into MoE) |
| `num_experts` | 256 | total experts |
| `num_experts_per_tok` | 8 | active experts per token |
| `vocab_size` | 248,320 | LM head N |
| `hidden_act` | SwiGLU | gate·up·down pattern |
| Layer pattern | linear_attn×3 → full_attn×1 | 10 full-attn + 30 GDN layers |

Per-layer weight matrices and global weights:

| Weight | Shape (K × N) | Quant | Bytes / instance | × Layers | Total MB |
|---|---:|---|---:|---:|---:|
| FC_QKV (fused Q+K+V proj) | 2048×5120 | INT4 g=64 | 5,652,480 | 10 | 53.9 |
| FC_O (attention output) | 4096×2048 | INT4 g=64 | 4,521,984 | 10 | 43.1 |
| FC_linattn_qkv (in_proj_qkv) | 2048×8192 | INT4 g=64 | 9,043,968 | 30 | 258.8 |
| FC_linattn_z (in_proj_z) | 2048×4096 | INT4 g=64 | 4,521,984 | 30 | 129.4 |
| FC_linattn_a (in_proj_a) | 2048×32 | INT4 g=64 | 35,328 | 30 | 1.0 |
| FC_linattn_b (in_proj_b) | 2048×32 | INT4 g=64 | 35,328 | 30 | 1.0 |
| FC_linattn_out (out_proj) | 4096×2048 | INT4 g=64 | 4,521,984 | 30 | 129.4 |
| MoE Expert gate+up+down | 2048×512 / 512×2048 | INT4 g=64 | 1,695,744/expert | 40×256 | 16560.0 |
| Router | 2048×256 | FP16 | 1,048,576 | 40 | 40.0 |
| Shared Expert gate+up+down | 2048×512 / 512×2048 | INT4 g=64 | 1,695,744 | 40 | 64.7 |
| LM_Head | 2048×248,320 | INT8 g=128 | 516,505,600 | 1 | 492.6 |
| **Total static weights** | | | | | **17774 MB** |

Activation / KV-cache shapes (S = sequence length, B = batch=1):

| Tensor | Shape | dtype | Bytes / token / layer | Bytes / token (all layers) |
|---|---|---|---:|---:|
| Hidden states | [B, S, 2048] | FP16 | 4096 | — |
| Q | [B, S, 16, 256] | FP16 | 8192 | — |
| K (cache) | [num_blocks, 2, 256, block_size] | INT8 | 512 | 20480 |
| V (cache) | [num_blocks, 2, block_size, 256] | INT8 | 512 | 20480 |
| **KV cache total** | per token | INT8 | 1024 B / layer | **40960 B / token (40.0 KB)** |

> Note: Only 10 full-attention layers use KV cache. GDN layers (30) use recurrent state.
> Effective KV cache per token = 10240 B / token (10.0 KB)

## Theoretical roofline

| Metric | Value |
|---|---|
| FP16 XMX peak | 58.9824 TFLOPS |
| INT8 XMX peak | 117.9648 TOPS |
| Memory BW | 110.0 GB/s |
| Ridge point (FP16) | 536.2 FLOP/byte |
| Ridge point (INT8) | 1072.4 FLOP/byte |

## Data sources

| Op category | Source | Platform |
|---|---|---|
| FC_QKV, FC_O, FC_linattn_* (all INT4 g=64) | **Measured** (fc_bench, cliloader) | PTL 12Xe |
| MoE fused routed+shared (INT4 g=64, SI=512) | **Measured** (moe_bench, cliloader) | PTL 12Xe |
| LM_head (INT8 g=128) | **Measured** (fc_bench, cliloader) | PTL 12Xe |
| PagedAttention (INT8 KV) | **Measured** (pa_bench, cliloader) | PTL 12Xe |
| GatedDeltaNet | **Measured** (gdn_bench, cliloader) | PTL 12Xe |
| SmallOps (rmsnorm, rope, add) | **Measured** (small_ops_bench, cliloader) | PTL 12Xe |

> v4: All ops measured with fused shared expert (FuseMOESharedExpert enabled).

## Graph fusion notes

| Bench row | Real graph behaviour | Fused into | Standalone kernel? |
|---|---|---|---|
| FC_QKV / FC_O (INT4 g=64) | FullyConnectedCompressed | decode: `gemm_kernel`; prefill: `dq+gemm` | Yes |
| FC_linattn_qkv/z/out (INT4 g=64) | FullyConnectedCompressed | decode: `gemm_kernel`; prefill: `dq+gemm` | Yes |
| MoE routed + shared expert | MOE3GemmFusedCompressed | **Fused** gate+up+down + shared expert | Yes (single primitive) |
| PagedAttention | PagedAttention | INT8 KV cache, GQA=8 | Yes |
| GatedDeltaNet | GatedDeltaNet | Reference kernel | Yes |
| SmallOps | rmsnorm/rope/add | Standalone | Yes |

> **v4 change**: Shared expert is now **fused** into MOE3GemmFusedCompressed primitive
> via FuseMOESharedExpert transformation (OV commit 2abcdce7f3).
> Previously (v3) it ran as 3 separate FullyConnectedCompressed kernels.

## Token latency summary

### Prefill — TTFT and per-token amortized

| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|
| 1,024 | 625.0 | 0.625 | 0.6104 | 1638.3 |
| 2,048 | 904.1 | 0.904 | 0.4414 | 2265.3 |
| 4,096 | 1594.7 | 1.595 | 0.3893 | 2568.5 |
| 8,192 | 3088.2 | 3.088 | 0.3770 | 2652.7 |
| 16,384 | 6546.4 | 6.546 | 0.3996 | 2502.7 |
| 32,768 | 16343.3 | 16.343 | 0.4988 | 2005.0 |
| 65,536 | 45084.9 | 45.085 | 0.6879 | 1453.6 |
| 131,072 | 151409.6 | 151.410 | 1.1552 | 865.7 |

### Decode — TPOT (per output token)

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 1,024 | 21.85 | 45.8 |
| 2,048 | 22.32 | 44.8 |
| 4,096 | 22.09 | 45.3 |
| 8,192 | 22.84 | 43.8 |
| 16,384 | 24.17 | 41.4 |
| 32,768 | 27.03 | 37.0 |
| 65,536 | 32.64 | 30.6 |
| 131,072 | 43.47 | 23.0 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)

| op | kv=1,024 | kv=2,048 | kv=4,096 | kv=8,192 | kv=16,384 | kv=32,768 | kv=65,536 | kv=131,072 |
|---|---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:|
| MoE_fused (routed+shared, INT4 g=64) | 7.586 (34.7%) | 7.586 (34.0%) | 7.586 (34.3%) | 7.586 (33.2%) | 7.586 (31.4%) | 7.586 (28.1%) | 7.586 (23.2%) | 7.586 (17.5%) |
| LM_head (INT8 g=128) | 5.236 (24.0%) | 5.236 (23.5%) | 5.236 (23.7%) | 5.236 (22.9%) | 5.236 (21.7%) | 5.236 (19.4%) | 5.236 (16.0%) | 5.236 (12.0%) |
| FC_linattn_qkv (INT4 g=64) | 2.760 (12.6%) | 2.760 (12.4%) | 2.760 (12.5%) | 2.760 (12.1%) | 2.760 (11.4%) | 2.760 (10.2%) | 2.760 (8.5%) | 2.760 (6.3%) |
| FC_linattn_z (INT4 g=64) | 1.452 (6.6%) | 1.452 (6.5%) | 1.452 (6.6%) | 1.452 (6.4%) | 1.452 (6.0%) | 1.452 (5.4%) | 1.452 (4.4%) | 1.452 (3.3%) |
| FC_linattn_out (INT4 g=64) | 1.434 (6.6%) | 1.434 (6.4%) | 1.434 (6.5%) | 1.434 (6.3%) | 1.434 (5.9%) | 1.434 (5.3%) | 1.434 (4.4%) | 1.434 (3.3%) |
| GatedDeltaNet | 0.960 (4.4%) | 0.960 (4.3%) | 0.960 (4.3%) | 0.960 (4.2%) | 0.960 (4.0%) | 0.960 (3.6%) | 0.960 (2.9%) | 0.960 (2.2%) |
| PagedAttention (INT8 KV) | 0.652 (3.0%) | 1.119 (5.0%) | 0.888 (4.0%) | 1.637 (7.2%) | 2.971 (12.3%) | 5.834 (21.6%) | 11.437 (35.0%) | 22.270 (51.2%) |
| FC_QKV (INT4 g=64) | 0.606 (2.8%) | 0.606 (2.7%) | 0.606 (2.7%) | 0.606 (2.7%) | 0.606 (2.5%) | 0.606 (2.2%) | 0.606 (1.9%) | 0.606 (1.4%) |
| FC_O (INT4 g=64) | 0.481 (2.2%) | 0.481 (2.2%) | 0.481 (2.2%) | 0.481 (2.1%) | 0.481 (2.0%) | 0.481 (1.8%) | 0.481 (1.5%) | 0.481 (1.1%) |
| SmallOps (norm/rope/add) | 0.452 (2.1%) | 0.452 (2.0%) | 0.452 (2.0%) | 0.452 (2.0%) | 0.452 (1.9%) | 0.452 (1.7%) | 0.452 (1.4%) | 0.452 (1.0%) |
| FC_linattn_b (INT4 g=64) | 0.115 (0.5%) | 0.115 (0.5%) | 0.115 (0.5%) | 0.115 (0.5%) | 0.115 (0.5%) | 0.115 (0.4%) | 0.115 (0.4%) | 0.115 (0.3%) |
| FC_linattn_a (INT4 g=64) | 0.115 (0.5%) | 0.115 (0.5%) | 0.115 (0.5%) | 0.115 (0.5%) | 0.115 (0.5%) | 0.115 (0.4%) | 0.115 (0.4%) | 0.115 (0.3%) |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)

| op | S=1,024 | S=2,048 | S=4,096 | S=8,192 | S=16,384 | S=32,768 | S=65,536 | S=131,072 |
|---|---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:|
| MoE_fused (routed+shared, INT4 g=64) | 449.3 (71.9%) | 556.5 (61.5%) | 857.0 (53.7%) | 1415.1 (45.8%) | 2436.4 (37.2%) | 4829.1 (29.5%) | 9269.5 (20.6%) | 19008.5 (12.6%) |
| GatedDeltaNet | 78.9 (12.6%) | 158.2 (17.5%) | 311.7 (19.5%) | 621.2 (20.1%) | 1241.5 (19.0%) | 2480.0 (15.2%) | 5063.1 (11.2%) | 10162.1 (6.7%) |
| FC_linattn_qkv (INT4 g=64) | 26.3 (4.2%) | 49.3 (5.5%) | 102.2 (6.4%) | 198.7 (6.4%) | 397.7 (6.1%) | 812.0 (5.0%) | 1754.5 (3.9%) | 3248.0 (2.1%) |
| FC_linattn_out (INT4 g=64) | 17.8 (2.9%) | 34.0 (3.8%) | 64.9 (4.1%) | 130.0 (4.2%) | 276.2 (4.2%) | 657.5 (4.0%) | 1366.5 (3.0%) | 3419.9 (2.3%) |
| FC_linattn_z (INT4 g=64) | 14.9 (2.4%) | 27.3 (3.0%) | 56.7 (3.6%) | 113.0 (3.7%) | 232.7 (3.6%) | 456.7 (2.8%) | 963.2 (2.1%) | 1844.8 (1.2%) |
| PagedAttention (FP16, causal) | 6.9 (1.1%) | 24.9 (2.7%) | 95.3 (6.0%) | 396.6 (12.8%) | 1534.6 (23.4%) | 6215.6 (38.0%) | 24846.7 (55.1%) | 109921.6 (72.6%) |
| FC_QKV (INT4 g=64) | 6.0 (1.0%) | 11.0 (1.2%) | 22.0 (1.4%) | 43.7 (1.4%) | 88.3 (1.3%) | 180.9 (1.1%) | 364.4 (0.8%) | 745.5 (0.5%) |
| FC_O (INT4 g=64) | 5.9 (0.9%) | 11.2 (1.2%) | 22.0 (1.4%) | 43.1 (1.4%) | 90.8 (1.4%) | 220.4 (1.3%) | 479.9 (1.1%) | 1110.4 (0.7%) |
| LM_head (INT8 g=128, 1 out tok) | 5.2 (0.8%) | 5.2 (0.6%) | 5.2 (0.3%) | 5.2 (0.2%) | 5.2 (0.1%) | 5.2 (0.0%) | 5.2 (0.0%) | 5.2 (0.0%) |
| SmallOps (norm/rope) | 5.0 (0.8%) | 10.0 (1.1%) | 25.2 (1.6%) | 58.5 (1.9%) | 117.0 (1.8%) | 234.0 (1.4%) | 468.1 (1.0%) | 936.2 (0.6%) |
| FC_linattn_a (INT4 g=64) | 4.4 (0.7%) | 8.3 (0.9%) | 16.2 (1.0%) | 31.4 (1.0%) | 62.8 (1.0%) | 125.6 (0.8%) | 251.2 (0.6%) | 502.5 (0.3%) |
| FC_linattn_b (INT4 g=64) | 4.4 (0.7%) | 8.2 (0.9%) | 16.2 (1.0%) | 31.6 (1.0%) | 63.1 (1.0%) | 126.2 (0.8%) | 252.4 (0.6%) | 504.8 (0.3%) |

## Decode tables (1 query token, KV = context length)

_Eff% = GB/s / 110 GB/s × 100 for memory-bound; GFLOPS / XMX_peak × 100 for compute-bound._

### Decode — KV=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=64) | moe_3gemm_swiglu_mlp | 0.1897 | 40 | 7.586 | 298.55 | 86.0 | 78.2% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| FC_linattn_qkv (INT4 g=64) | gemm_kernel | 0.0920 | 30 | 2.760 | 364.73 | 98.5 | 89.6% | memory |
| FC_linattn_z (INT4 g=64) | gemm_kernel | 0.0484 | 30 | 1.452 | 346.57 | 93.7 | 85.1% | memory |
| FC_linattn_out (INT4 g=64) | gemm_kernel | 0.0478 | 30 | 1.434 | 350.87 | 94.8 | 86.2% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.0652 | 10 | 0.652 | 257.16 | 16.1 | 14.6% | memory |
| FC_QKV (INT4 g=64) | gemm_kernel | 0.0606 | 10 | 0.606 | 346.14 | 93.5 | 85.0% | memory |
| FC_O (INT4 g=64) | gemm_kernel | 0.0481 | 10 | 0.481 | 348.49 | 94.2 | 85.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.15 | 10.3 | 9.4% | memory |
| FC_linattn_a (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.3 | 9.4% | memory |
| **TOTAL** | | | | **21.851** | | | | |
| | | | | | | | **45.8 tok/s** | |

### Decode — KV=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=64) | moe_3gemm_swiglu_mlp | 0.1897 | 40 | 7.586 | 298.55 | 86.0 | 78.2% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| FC_linattn_qkv (INT4 g=64) | gemm_kernel | 0.0920 | 30 | 2.760 | 364.73 | 98.5 | 89.6% | memory |
| FC_linattn_z (INT4 g=64) | gemm_kernel | 0.0484 | 30 | 1.452 | 346.57 | 93.7 | 85.1% | memory |
| FC_linattn_out (INT4 g=64) | gemm_kernel | 0.0478 | 30 | 1.434 | 350.87 | 94.8 | 86.2% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.1119 | 10 | 1.119 | 299.80 | 18.7 | 17.0% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=64) | gemm_kernel | 0.0606 | 10 | 0.606 | 346.14 | 93.5 | 85.0% | memory |
| FC_O (INT4 g=64) | gemm_kernel | 0.0481 | 10 | 0.481 | 348.49 | 94.2 | 85.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.15 | 10.3 | 9.4% | memory |
| FC_linattn_a (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.3 | 9.4% | memory |
| **TOTAL** | | | | **22.318** | | | | |
| | | | | | | | **44.8 tok/s** | |

### Decode — KV=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=64) | moe_3gemm_swiglu_mlp | 0.1897 | 40 | 7.586 | 298.55 | 86.0 | 78.2% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| FC_linattn_qkv (INT4 g=64) | gemm_kernel | 0.0920 | 30 | 2.760 | 364.73 | 98.5 | 89.6% | memory |
| FC_linattn_z (INT4 g=64) | gemm_kernel | 0.0484 | 30 | 1.452 | 346.57 | 93.7 | 85.1% | memory |
| FC_linattn_out (INT4 g=64) | gemm_kernel | 0.0478 | 30 | 1.434 | 350.87 | 94.8 | 86.2% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.0888 | 10 | 0.888 | 756.05 | 47.3 | 43.0% | memory |
| FC_QKV (INT4 g=64) | gemm_kernel | 0.0606 | 10 | 0.606 | 346.14 | 93.5 | 85.0% | memory |
| FC_O (INT4 g=64) | gemm_kernel | 0.0481 | 10 | 0.481 | 348.49 | 94.2 | 85.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.15 | 10.3 | 9.4% | memory |
| FC_linattn_a (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.3 | 9.4% | memory |
| **TOTAL** | | | | **22.087** | | | | |
| | | | | | | | **45.3 tok/s** | |

### Decode — KV=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=64) | moe_3gemm_swiglu_mlp | 0.1897 | 40 | 7.586 | 298.55 | 86.0 | 78.2% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| FC_linattn_qkv (INT4 g=64) | gemm_kernel | 0.0920 | 30 | 2.760 | 364.73 | 98.5 | 89.6% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.1637 | 10 | 1.637 | 820.15 | 51.3 | 46.6% | memory |
| FC_linattn_z (INT4 g=64) | gemm_kernel | 0.0484 | 30 | 1.452 | 346.57 | 93.7 | 85.1% | memory |
| FC_linattn_out (INT4 g=64) | gemm_kernel | 0.0478 | 30 | 1.434 | 350.87 | 94.8 | 86.2% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=64) | gemm_kernel | 0.0606 | 10 | 0.606 | 346.14 | 93.5 | 85.0% | memory |
| FC_O (INT4 g=64) | gemm_kernel | 0.0481 | 10 | 0.481 | 348.49 | 94.2 | 85.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.15 | 10.3 | 9.4% | memory |
| FC_linattn_a (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.3 | 9.4% | memory |
| **TOTAL** | | | | **22.835** | | | | |
| | | | | | | | **43.8 tok/s** | |

### Decode — KV=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=64) | moe_3gemm_swiglu_mlp | 0.1897 | 40 | 7.586 | 298.55 | 86.0 | 78.2% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.2971 | 10 | 2.971 | 903.48 | 56.5 | 51.3% | memory |
| FC_linattn_qkv (INT4 g=64) | gemm_kernel | 0.0920 | 30 | 2.760 | 364.73 | 98.5 | 89.6% | memory |
| FC_linattn_z (INT4 g=64) | gemm_kernel | 0.0484 | 30 | 1.452 | 346.57 | 93.7 | 85.1% | memory |
| FC_linattn_out (INT4 g=64) | gemm_kernel | 0.0478 | 30 | 1.434 | 350.87 | 94.8 | 86.2% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=64) | gemm_kernel | 0.0606 | 10 | 0.606 | 346.14 | 93.5 | 85.0% | memory |
| FC_O (INT4 g=64) | gemm_kernel | 0.0481 | 10 | 0.481 | 348.49 | 94.2 | 85.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.15 | 10.3 | 9.4% | memory |
| FC_linattn_a (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.3 | 9.4% | memory |
| **TOTAL** | | | | **24.170** | | | | |
| | | | | | | | **41.4 tok/s** | |

### Decode — KV=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=64) | moe_3gemm_swiglu_mlp | 0.1897 | 40 | 7.586 | 298.55 | 86.0 | 78.2% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.5834 | 10 | 5.834 | 920.25 | 57.5 | 52.3% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| FC_linattn_qkv (INT4 g=64) | gemm_kernel | 0.0920 | 30 | 2.760 | 364.73 | 98.5 | 89.6% | memory |
| FC_linattn_z (INT4 g=64) | gemm_kernel | 0.0484 | 30 | 1.452 | 346.57 | 93.7 | 85.1% | memory |
| FC_linattn_out (INT4 g=64) | gemm_kernel | 0.0478 | 30 | 1.434 | 350.87 | 94.8 | 86.2% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=64) | gemm_kernel | 0.0606 | 10 | 0.606 | 346.14 | 93.5 | 85.0% | memory |
| FC_O (INT4 g=64) | gemm_kernel | 0.0481 | 10 | 0.481 | 348.49 | 94.2 | 85.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.15 | 10.3 | 9.4% | memory |
| FC_linattn_a (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.3 | 9.4% | memory |
| **TOTAL** | | | | **27.033** | | | | |
| | | | | | | | **37.0 tok/s** | |

### Decode — KV=65,536

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (INT8 KV) | paged_attention_opt | 1.1437 | 10 | 11.437 | 938.80 | 58.7 | 53.3% | memory |
| MoE_fused (routed+shared, INT4 g=64) | moe_3gemm_swiglu_mlp | 0.1897 | 40 | 7.586 | 298.55 | 86.0 | 78.2% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| FC_linattn_qkv (INT4 g=64) | gemm_kernel | 0.0920 | 30 | 2.760 | 364.73 | 98.5 | 89.6% | memory |
| FC_linattn_z (INT4 g=64) | gemm_kernel | 0.0484 | 30 | 1.452 | 346.57 | 93.7 | 85.1% | memory |
| FC_linattn_out (INT4 g=64) | gemm_kernel | 0.0478 | 30 | 1.434 | 350.87 | 94.8 | 86.2% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=64) | gemm_kernel | 0.0606 | 10 | 0.606 | 346.14 | 93.5 | 85.0% | memory |
| FC_O (INT4 g=64) | gemm_kernel | 0.0481 | 10 | 0.481 | 348.49 | 94.2 | 85.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.15 | 10.3 | 9.4% | memory |
| FC_linattn_a (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.3 | 9.4% | memory |
| **TOTAL** | | | | **32.636** | | | | |
| | | | | | | | **30.6 tok/s** | |

### Decode — KV=131,072

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (INT8 KV) | paged_attention_opt | 2.2270 | 10 | 22.270 | 964.31 | 60.3 | 54.8% | memory |
| MoE_fused (routed+shared, INT4 g=64) | moe_3gemm_swiglu_mlp | 0.1897 | 40 | 7.586 | 298.55 | 86.0 | 78.2% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| FC_linattn_qkv (INT4 g=64) | gemm_kernel | 0.0920 | 30 | 2.760 | 364.73 | 98.5 | 89.6% | memory |
| FC_linattn_z (INT4 g=64) | gemm_kernel | 0.0484 | 30 | 1.452 | 346.57 | 93.7 | 85.1% | memory |
| FC_linattn_out (INT4 g=64) | gemm_kernel | 0.0478 | 30 | 1.434 | 350.87 | 94.8 | 86.2% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=64) | gemm_kernel | 0.0606 | 10 | 0.606 | 346.14 | 93.5 | 85.0% | memory |
| FC_O (INT4 g=64) | gemm_kernel | 0.0481 | 10 | 0.481 | 348.49 | 94.2 | 85.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.15 | 10.3 | 9.4% | memory |
| FC_linattn_a (INT4 g=64) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.3 | 9.4% | memory |
| **TOTAL** | | | | **43.469** | | | | |
| | | | | | | | **23.0 tok/s** | |

## Prefill tables (single forward over S tokens)

_Eff% for INT4 FC prefill = GFLOPS / INT8 XMX (117.965 TOPS); for FP16 SDPA = GFLOPS / FP16 XMX (58.982 TFLOPS)._

### Prefill — S=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=64) | grouped_micro_gemm+scatter/gather | 11.2324 | 40 | 449.297 | 5162.03 | 39.6 | 36.0% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 2.6316 | 30 | 78.949 | 408.01 | 2.4 | 2.2% | memory |
| FC_linattn_qkv (INT4 g=64) | dq+gemm_kernel | 0.8764 | 30 | 26.293 | 39203.35 | 34.2 | 33.2% | compute |
| FC_linattn_out (INT4 g=64) | dq+gemm_kernel | 0.5946 | 30 | 17.838 | 28892.96 | 28.8 | 26.2% | memory |
| FC_linattn_z (INT4 g=64) | dq+gemm_kernel | 0.4975 | 30 | 14.926 | 34529.49 | 34.4 | 31.3% | memory |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 0.6890 | 10 | 6.890 | 12467.34 | 1.5 | 21.1% | compute |
| FC_QKV (INT4 g=64) | dq+gemm_kernel | 0.5952 | 10 | 5.952 | 36081.85 | 34.2 | 31.1% | memory |
| FC_O (INT4 g=64) | dq+gemm_kernel | 0.5913 | 10 | 5.913 | 29054.11 | 28.9 | 26.3% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| SmallOps (norm/rope) | rms/rope | — | — | 5.033 | — | — | — | memory |
| FC_linattn_a (INT4 g=64) | dq+gemm_kernel | 0.1454 | 30 | 4.361 | 923.28 | 29.5 | 26.9% | memory |
| FC_linattn_b (INT4 g=64) | dq+gemm_kernel | 0.1453 | 30 | 4.358 | 924.03 | 29.6 | 26.9% | memory |
| **TOTAL** | | | | **625.0** | | | | |
| | | | | TTFT=625 ms | | | **1638.3 tok/s** | |

### Prefill — S=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=64) | grouped_micro_gemm+scatter/gather | 13.9116 | 40 | 556.463 | 8335.80 | 32.6 | 29.6% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 5.2740 | 30 | 158.221 | 407.18 | 2.0 | 1.8% | memory |
| FC_linattn_qkv (INT4 g=64) | dq+gemm_kernel | 1.6430 | 30 | 49.290 | 41825.25 | 31.0 | 35.5% | compute |
| FC_linattn_out (INT4 g=64) | dq+gemm_kernel | 1.1343 | 30 | 34.028 | 30292.67 | 26.2 | 25.7% | compute |
| FC_linattn_z (INT4 g=64) | dq+gemm_kernel | 0.9112 | 30 | 27.335 | 37710.22 | 32.6 | 32.0% | compute |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 2.4853 | 10 | 24.853 | 13825.18 | 0.8 | 23.4% | compute |
| FC_O (INT4 g=64) | dq+gemm_kernel | 1.1247 | 10 | 11.247 | 30550.03 | 26.4 | 25.9% | compute |
| FC_QKV (INT4 g=64) | dq+gemm_kernel | 1.0953 | 10 | 10.953 | 39211.27 | 32.0 | 33.2% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 9.970 | — | — | — | memory |
| FC_linattn_a (INT4 g=64) | dq+gemm_kernel | 0.2751 | 30 | 8.253 | 975.72 | 31.1 | 28.3% | memory |
| FC_linattn_b (INT4 g=64) | dq+gemm_kernel | 0.2745 | 30 | 8.235 | 977.93 | 31.2 | 28.3% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| **TOTAL** | | | | **904.1** | | | | |
| | | | | TTFT=904 ms | | | **2265.3 tok/s** | |

### Prefill — S=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=64) | grouped_micro_gemm+scatter/gather | 21.4259 | 40 | 857.038 | 10824.65 | 22.0 | 20.0% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 10.3904 | 30 | 311.712 | 413.36 | 1.8 | 1.7% | memory |
| FC_linattn_qkv (INT4 g=64) | dq+gemm_kernel | 3.4081 | 30 | 102.244 | 40326.73 | 27.3 | 34.2% | compute |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 9.5349 | 10 | 95.349 | 14414.33 | 0.4 | 24.4% | compute |
| FC_linattn_out (INT4 g=64) | dq+gemm_kernel | 2.1645 | 30 | 64.935 | 31748.34 | 25.3 | 26.9% | compute |
| FC_linattn_z (INT4 g=64) | dq+gemm_kernel | 1.8886 | 30 | 56.659 | 36385.58 | 29.0 | 30.8% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 25.156 | — | — | — | memory |
| FC_QKV (INT4 g=64) | dq+gemm_kernel | 2.2017 | 10 | 22.017 | 39014.50 | 29.2 | 33.1% | compute |
| FC_O (INT4 g=64) | dq+gemm_kernel | 2.2001 | 10 | 22.001 | 31234.24 | 24.9 | 26.5% | compute |
| FC_linattn_a (INT4 g=64) | dq+gemm_kernel | 0.5398 | 30 | 16.193 | 994.61 | 31.6 | 28.8% | memory |
| FC_linattn_b (INT4 g=64) | dq+gemm_kernel | 0.5397 | 30 | 16.192 | 994.67 | 31.6 | 28.8% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| **TOTAL** | | | | **1594.7** | | | | |
| | | | | TTFT=1595 ms | | | **2568.5 tok/s** | |

### Prefill — S=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=64) | grouped_micro_gemm+scatter/gather | 35.3779 | 40 | 1415.116 | 13111.48 | 14.2 | 13.0% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 20.7070 | 30 | 621.210 | 414.83 | 1.7 | 1.6% | memory |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 39.6626 | 10 | 396.626 | 13860.82 | 0.2 | 23.5% | compute |
| FC_linattn_qkv (INT4 g=64) | dq+gemm_kernel | 6.6246 | 30 | 198.737 | 41493.70 | 26.7 | 35.2% | compute |
| FC_linattn_out (INT4 g=64) | dq+gemm_kernel | 4.3333 | 30 | 130.000 | 31716.77 | 24.3 | 26.9% | compute |
| FC_linattn_z (INT4 g=64) | dq+gemm_kernel | 3.7674 | 30 | 113.021 | 36481.31 | 27.9 | 30.9% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 58.510 | — | — | — | memory |
| FC_QKV (INT4 g=64) | dq+gemm_kernel | 4.3712 | 10 | 43.712 | 39301.97 | 28.2 | 33.3% | compute |
| FC_O (INT4 g=64) | dq+gemm_kernel | 4.3092 | 10 | 43.092 | 31894.18 | 24.4 | 27.0% | compute |
| FC_linattn_b (INT4 g=64) | dq+gemm_kernel | 1.0518 | 30 | 31.553 | 1020.89 | 32.4 | 29.5% | memory |
| FC_linattn_a (INT4 g=64) | dq+gemm_kernel | 1.0468 | 30 | 31.404 | 1025.75 | 32.6 | 29.6% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| **TOTAL** | | | | **3088.2** | | | | |
| | | | | TTFT=3088 ms | | | **2652.7 tok/s** | |

### Prefill — S=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=64) | grouped_micro_gemm+scatter/gather | 60.9102 | 40 | 2436.410 | 15230.82 | 9.4 | 12.9% | compute |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 153.4550 | 10 | 1534.550 | 14330.08 | 0.1 | 24.3% | compute |
| GatedDeltaNet | gated_delta_net_ref_sa | 41.3845 | 30 | 1241.536 | 415.13 | 1.7 | 1.5% | memory |
| FC_linattn_qkv (INT4 g=64) | dq+gemm_kernel | 13.2577 | 30 | 397.731 | 41466.89 | 26.0 | 35.2% | compute |
| FC_linattn_out (INT4 g=64) | dq+gemm_kernel | 9.2071 | 30 | 276.214 | 29854.89 | 22.4 | 25.3% | compute |
| FC_linattn_z (INT4 g=64) | dq+gemm_kernel | 7.7561 | 30 | 232.682 | 35440.30 | 26.5 | 30.0% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 117.021 | — | — | — | memory |
| FC_O (INT4 g=64) | dq+gemm_kernel | 9.0833 | 10 | 90.833 | 30261.79 | 22.7 | 25.7% | compute |
| FC_QKV (INT4 g=64) | dq+gemm_kernel | 8.8299 | 10 | 88.299 | 38913.12 | 27.2 | 33.0% | compute |
| FC_linattn_b (INT4 g=64) | dq+gemm_kernel | 2.1035 | 30 | 63.106 | 1020.89 | 32.4 | 29.5% | memory |
| FC_linattn_a (INT4 g=64) | dq+gemm_kernel | 2.0936 | 30 | 62.807 | 1025.75 | 32.6 | 29.6% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| **TOTAL** | | | | **6546.4** | | | | |
| | | | | TTFT=6546 ms | | | **2502.7 tok/s** | |

### Prefill — S=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 621.5587 | 10 | 6215.587 | 14151.67 | 0.1 | 24.0% | compute |
| MoE_fused (routed+shared, INT4 g=64) | grouped_micro_gemm+scatter/gather | 120.7271 | 40 | 4829.086 | 15368.75 | 5.8 | 13.0% | compute |
| GatedDeltaNet | gated_delta_net_ref_sa | 82.6682 | 30 | 2480.046 | 415.63 | 1.6 | 1.5% | memory |
| FC_linattn_qkv (INT4 g=64) | dq+gemm_kernel | 27.0660 | 30 | 811.981 | 40623.32 | 25.1 | 34.4% | compute |
| FC_linattn_out (INT4 g=64) | dq+gemm_kernel | 21.9182 | 30 | 657.545 | 25082.18 | 18.6 | 21.3% | compute |
| FC_linattn_z (INT4 g=64) | dq+gemm_kernel | 15.2231 | 30 | 456.692 | 36113.32 | 26.7 | 30.6% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 234.041 | — | — | — | memory |
| FC_O (INT4 g=64) | dq+gemm_kernel | 22.0427 | 10 | 220.427 | 24940.50 | 18.5 | 21.1% | compute |
| FC_QKV (INT4 g=64) | dq+gemm_kernel | 18.0861 | 10 | 180.861 | 37995.73 | 26.3 | 32.2% | compute |
| FC_linattn_b (INT4 g=64) | dq+gemm_kernel | 4.2071 | 30 | 126.212 | 1020.89 | 32.4 | 29.5% | memory |
| FC_linattn_a (INT4 g=64) | dq+gemm_kernel | 4.1871 | 30 | 125.614 | 1025.75 | 32.6 | 29.6% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| **TOTAL** | | | | **16343.3** | | | | |
| | | | | TTFT=16343 ms | | | **2005.0 tok/s** | |

### Prefill — S=65,536

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 2484.6714 | 10 | 24846.714 | 14160.57 | 0.0 | 24.0% | compute |
| MoE_fused (routed+shared, INT4 g=64) | grouped_micro_gemm+scatter/gather | 231.7368 | 40 | 9269.473 | 16013.22 | 4.2 | 13.6% | compute |
| GatedDeltaNet | gated_delta_net_ref_sa | 168.7711 | 30 | 5063.133 | 407.18 | 1.6 | 1.5% | memory |
| FC_linattn_qkv (INT4 g=64) | dq+gemm_kernel | 58.4848 | 30 | 1754.543 | 37599.94 | 23.1 | 31.9% | compute |
| FC_linattn_out (INT4 g=64) | dq+gemm_kernel | 45.5515 | 30 | 1366.544 | 24137.78 | 17.8 | 20.5% | compute |
| FC_linattn_z (INT4 g=64) | dq+gemm_kernel | 32.1077 | 30 | 963.230 | 34244.52 | 25.2 | 29.0% | compute |
| FC_O (INT4 g=64) | dq+gemm_kernel | 47.9867 | 10 | 479.867 | 22912.84 | 16.9 | 19.4% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 468.083 | — | — | — | memory |
| FC_QKV (INT4 g=64) | dq+gemm_kernel | 36.4431 | 10 | 364.431 | 37713.31 | 25.9 | 32.0% | compute |
| FC_linattn_b (INT4 g=64) | dq+gemm_kernel | 8.4142 | 30 | 252.425 | 1020.89 | 32.4 | 29.5% | memory |
| FC_linattn_a (INT4 g=64) | dq+gemm_kernel | 8.3743 | 30 | 251.229 | 1025.75 | 32.6 | 29.6% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| **TOTAL** | | | | **45084.9** | | | | |
| | | | | TTFT=45085 ms | | | **1453.6 tok/s** | |

### Prefill — S=131,072

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 10992.1630 | 10 | 109921.630 | 12803.44 | 0.0 | 21.7% | compute |
| MoE_fused (routed+shared, INT4 g=64) | grouped_micro_gemm+scatter/gather | 475.2137 | 40 | 19008.548 | 15617.61 | 3.2 | 13.2% | compute |
| GatedDeltaNet | gated_delta_net_ref_sa | 338.7359 | 30 | 10162.078 | 405.74 | 1.6 | 1.4% | memory |
| FC_linattn_out (INT4 g=64) | dq+gemm_kernel | 113.9966 | 30 | 3419.897 | 19290.26 | 14.2 | 16.4% | compute |
| FC_linattn_qkv (INT4 g=64) | dq+gemm_kernel | 108.2666 | 30 | 3247.998 | 40622.37 | 24.9 | 34.4% | compute |
| FC_linattn_z (INT4 g=64) | dq+gemm_kernel | 61.4924 | 30 | 1844.772 | 35760.90 | 26.3 | 30.3% | compute |
| FC_O (INT4 g=64) | dq+gemm_kernel | 111.0430 | 10 | 1110.429 | 19803.36 | 14.5 | 16.8% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 936.165 | — | — | — | memory |
| FC_QKV (INT4 g=64) | dq+gemm_kernel | 74.5499 | 10 | 745.499 | 36871.64 | 25.3 | 31.3% | compute |
| FC_linattn_b (INT4 g=64) | dq+gemm_kernel | 16.8283 | 30 | 504.849 | 1020.89 | 32.4 | 29.5% | memory |
| FC_linattn_a (INT4 g=64) | dq+gemm_kernel | 16.7486 | 30 | 502.457 | 1025.75 | 32.6 | 29.6% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.2361 | 1 | 5.236 | 194.25 | 98.7 | 89.8% | memory |
| **TOTAL** | | | | **151409.6** | | | | |
| | | | | TTFT=151410 ms | | | **865.7 tok/s** | |

## Top contributors (sorted by total ms per inference)

### Decode

| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1,024 | MoE_fused (routed+shared, INT4 g=64) 7.59ms (34.7%) | LM_head (INT8 g=128) 5.24ms (24.0%) | FC_linattn_qkv (INT4 g=64) 2.76ms (12.6%) |
| 2,048 | MoE_fused (routed+shared, INT4 g=64) 7.59ms (34.0%) | LM_head (INT8 g=128) 5.24ms (23.5%) | FC_linattn_qkv (INT4 g=64) 2.76ms (12.4%) |
| 4,096 | MoE_fused (routed+shared, INT4 g=64) 7.59ms (34.3%) | LM_head (INT8 g=128) 5.24ms (23.7%) | FC_linattn_qkv (INT4 g=64) 2.76ms (12.5%) |
| 8,192 | MoE_fused (routed+shared, INT4 g=64) 7.59ms (33.2%) | LM_head (INT8 g=128) 5.24ms (22.9%) | FC_linattn_qkv (INT4 g=64) 2.76ms (12.1%) |
| 16,384 | MoE_fused (routed+shared, INT4 g=64) 7.59ms (31.4%) | LM_head (INT8 g=128) 5.24ms (21.7%) | PagedAttention (INT8 KV) 2.97ms (12.3%) |
| 32,768 | MoE_fused (routed+shared, INT4 g=64) 7.59ms (28.1%) | PagedAttention (INT8 KV) 5.83ms (21.6%) | LM_head (INT8 g=128) 5.24ms (19.4%) |
| 65,536 | PagedAttention (INT8 KV) 11.44ms (35.0%) | MoE_fused (routed+shared, INT4 g=64) 7.59ms (23.2%) | LM_head (INT8 g=128) 5.24ms (16.0%) |
| 131,072 | PagedAttention (INT8 KV) 22.27ms (51.2%) | MoE_fused (routed+shared, INT4 g=64) 7.59ms (17.5%) | LM_head (INT8 g=128) 5.24ms (12.0%) |

### Prefill

| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1,024 | MoE_fused (routed+shared, INT4 g=64) 449.3ms (71.9%) | GatedDeltaNet 78.9ms (12.6%) | FC_linattn_qkv (INT4 g=64) 26.3ms (4.2%) |
| 2,048 | MoE_fused (routed+shared, INT4 g=64) 556.5ms (61.5%) | GatedDeltaNet 158.2ms (17.5%) | FC_linattn_qkv (INT4 g=64) 49.3ms (5.5%) |
| 4,096 | MoE_fused (routed+shared, INT4 g=64) 857.0ms (53.7%) | GatedDeltaNet 311.7ms (19.5%) | FC_linattn_qkv (INT4 g=64) 102.2ms (6.4%) |
| 8,192 | MoE_fused (routed+shared, INT4 g=64) 1415.1ms (45.8%) | GatedDeltaNet 621.2ms (20.1%) | PagedAttention (FP16, causal) 396.6ms (12.8%) |
| 16,384 | MoE_fused (routed+shared, INT4 g=64) 2436.4ms (37.2%) | PagedAttention (FP16, causal) 1534.6ms (23.4%) | GatedDeltaNet 1241.5ms (19.0%) |
| 32,768 | PagedAttention (FP16, causal) 6215.6ms (38.0%) | MoE_fused (routed+shared, INT4 g=64) 4829.1ms (29.5%) | GatedDeltaNet 2480.0ms (15.2%) |
| 65,536 | PagedAttention (FP16, causal) 24846.7ms (55.1%) | MoE_fused (routed+shared, INT4 g=64) 9269.5ms (20.6%) | GatedDeltaNet 5063.1ms (11.2%) |
| 131,072 | PagedAttention (FP16, causal) 109921.6ms (72.6%) | MoE_fused (routed+shared, INT4 g=64) 19008.5ms (12.6%) | GatedDeltaNet 10162.1ms (6.7%) |

## Caveats & method

- Each op profiled in its own process via cliloader Device Performance Timing; we use mean kernel time per iteration.
- FC weight bytes count INT4 weight + FP16 scale (per group) + INT4 zero-point (per group) + FP16 activations. Group size = 64.
- PA bytes assume INT8 KV cache + FP16 Q, FP16 out.
- Decode FC is treated as **memory-bound** (weights read dominates at M=1); prefill FC is **INT8 XMX compute-bound**.
- Prefill PA at S≥2048 is compute-bound (FP16 micro-kernel); decode PA is memory-bound.
- lm_head is run only once per token (last position in prefill, every step in decode).
- **v4**: Shared expert fused into MOE3GemmFusedCompressed primitive (FuseMOESharedExpert fix, commit 2abcdce7f3).
- MoE benchmark uses shared_I=512, shared_quant=u4 — the fused kernel includes both routed and shared expert computation.
- GDN prefill S=131072 may be extrapolated ×2 from S=65536 if the original run was not available.
- Target machine: Local_Admin@10.239.132.229 (PTL B390 iGPU, 12 Xe cores)

## Reproduction

```bat
REM On PTL Windows machine:
set OV_BIN=D:\river\moe\openvino\release_install\runtime\bin\intel64\Release
set TBB=D:\river\moe\openvino\temp\Windows_AMD64\tbb\bin
set CLI=C:\Users\Local_Admin\Downloads\clintercept-3.0.6-win64\Release\cliloader.exe
set BUILD=D:\river\moe\dev_roofline_profiling\utils\build\Release

REM FC example: fc_bench M K N group_size iters warmup num_bufs precision flush_mb
REM   %CLI% -d %BUILD%\fc_bench.exe 1 2048 5120 64 15000 500 8 u4 32
REM MoE fused example: moe_bench B S H I NE TK group_size iters warmup num_bufs flush_mb shared_I shared_quant
REM   %CLI% -d %BUILD%\moe_bench.exe 1 1 2048 512 256 8 64 100 10 4 64 512 u4

REM Full script: run_qwen3_5_moe_ptl_int4g64_v4.bat
```

_Generated by `build_report_ptl_int4g64_v4.py`_