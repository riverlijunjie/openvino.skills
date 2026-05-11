# Qwen3.5-MoE-35B-A3B — Roofline on PTL 12Xe (2026-05-10, v4 INT4 g=128)

**Platform**: PTL (Panther Lake B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s)
**Model**: Qwen/Qwen3.5-MoE-35B-A3B (Hybrid Attention: 10 full-attn + 30 GatedDeltaNet, 256 experts × top-8)

- hidden_size = 2048, 40 layers (10 full-attn + 30 linear-attn GDN), 16 Q heads, 2 KV heads (GQA=8)
- MoE: 256 experts × top-8, intermediate=512, shared expert intermediate=512
- MatMul weights **INT4 asymmetric g=128** / FP16 act; LM_head INT8 g=128 / FP16 act; KV cache INT8
- SDPA: PagedAttention (INT8 KV cache, block_size=16)
- **v4**: Shared expert **fused** into MoE kernel (FuseMOESharedExpert, commit 2abcdce7f3)
- PA/GDN/SmallOps data reused from g=64 v4 run (group-size independent)

## Model parameters & weight shapes

See g=64 report for full weight table. Key difference: all body FC and MoE weights use **g=128** instead of g=64.

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
| FC_QKV, FC_O, FC_linattn_* (all INT4 g=128) | **Measured** (fc_bench) | PTL 12Xe |
| MoE fused routed+shared (INT4 g=128, SI=512) | **Measured** (moe_bench) | PTL 12Xe |
| LM_head (INT8 g=128) | **Measured** (fc_bench) | PTL 12Xe |
| PagedAttention (INT8 KV) | **Reused** from g=64 v4 | PTL 12Xe |
| GatedDeltaNet | **Reused** from g=64 v4 | PTL 12Xe |
| SmallOps | **Reused** from g=64 v4 | PTL 12Xe |

## Graph fusion notes

| Bench row | Real graph behaviour | Fused into | Standalone kernel? |
|---|---|---|---|
| FC_QKV / FC_O (INT4 g=128) | FullyConnectedCompressed | decode: `gemm_kernel`; prefill: `dq+gemm` | Yes |
| MoE routed + shared expert | MOE3GemmFusedCompressed | **Fused** routed+shared | Yes (single primitive) |
| PagedAttention | PagedAttention | INT8 KV cache, GQA=8 | Yes |
| GatedDeltaNet | GatedDeltaNet | Reference kernel | Yes |

## Token latency summary

### Prefill — TTFT and per-token amortized

| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|
| 1,024 | 541.0 | 0.541 | 0.5283 | 1892.9 |
| 2,048 | 808.0 | 0.808 | 0.3945 | 2534.6 |
| 4,096 | 1467.6 | 1.468 | 0.3583 | 2791.0 |
| 8,192 | 2862.7 | 2.863 | 0.3495 | 2861.6 |
| 16,384 | 6136.0 | 6.136 | 0.3745 | 2670.1 |
| 32,768 | 15441.3 | 15.441 | 0.4712 | 2122.1 |
| 65,536 | 42875.1 | 42.875 | 0.6542 | 1528.5 |
| 131,072 | 146739.4 | 146.739 | 1.1195 | 893.2 |

### Decode — TPOT (per output token)

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 1,024 | 20.79 | 48.1 |
| 2,048 | 21.26 | 47.0 |
| 4,096 | 21.03 | 47.6 |
| 8,192 | 21.77 | 45.9 |
| 16,384 | 23.11 | 43.3 |
| 32,768 | 25.97 | 38.5 |
| 65,536 | 31.58 | 31.7 |
| 131,072 | 42.41 | 23.6 |

### Decode TPOT — per-op breakdown (ms / % of TPOT)

| op | kv=1,024 | kv=2,048 | kv=4,096 | kv=8,192 | kv=16,384 | kv=32,768 | kv=65,536 | kv=131,072 |
|---|---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:|
| MoE_fused (routed+shared, INT4 g=128) | 6.981 (33.6%) | 6.981 (32.8%) | 6.981 (33.2%) | 6.981 (32.1%) | 6.981 (30.2%) | 6.981 (26.9%) | 6.981 (22.1%) | 6.981 (16.5%) |
| LM_head (INT8 g=128) | 5.199 (25.0%) | 5.199 (24.5%) | 5.199 (24.7%) | 5.199 (23.9%) | 5.199 (22.5%) | 5.199 (20.0%) | 5.199 (16.5%) | 5.199 (12.3%) |
| FC_linattn_qkv (INT4 g=128) | 2.585 (12.4%) | 2.585 (12.2%) | 2.585 (12.3%) | 2.585 (11.9%) | 2.585 (11.2%) | 2.585 (10.0%) | 2.585 (8.2%) | 2.585 (6.1%) |
| FC_linattn_out (INT4 g=128) | 1.383 (6.7%) | 1.383 (6.5%) | 1.383 (6.6%) | 1.383 (6.4%) | 1.383 (6.0%) | 1.383 (5.3%) | 1.383 (4.4%) | 1.383 (3.3%) |
| FC_linattn_z (INT4 g=128) | 1.338 (6.4%) | 1.338 (6.3%) | 1.338 (6.4%) | 1.338 (6.1%) | 1.338 (5.8%) | 1.338 (5.2%) | 1.338 (4.2%) | 1.338 (3.2%) |
| GatedDeltaNet | 0.960 (4.6%) | 0.960 (4.5%) | 0.960 (4.6%) | 0.960 (4.4%) | 0.960 (4.2%) | 0.960 (3.7%) | 0.960 (3.0%) | 0.960 (2.3%) |
| PagedAttention (INT8 KV) | 0.652 (3.1%) | 1.119 (5.3%) | 0.888 (4.2%) | 1.637 (7.5%) | 2.971 (12.9%) | 5.834 (22.5%) | 11.437 (36.2%) | 22.270 (52.5%) |
| FC_QKV (INT4 g=128) | 0.548 (2.6%) | 0.548 (2.6%) | 0.548 (2.6%) | 0.548 (2.5%) | 0.548 (2.4%) | 0.548 (2.1%) | 0.548 (1.7%) | 0.548 (1.3%) |
| FC_O (INT4 g=128) | 0.461 (2.2%) | 0.461 (2.2%) | 0.461 (2.2%) | 0.461 (2.1%) | 0.461 (2.0%) | 0.461 (1.8%) | 0.461 (1.5%) | 0.461 (1.1%) |
| SmallOps (norm/rope/add) | 0.452 (2.2%) | 0.452 (2.1%) | 0.452 (2.2%) | 0.452 (2.1%) | 0.452 (2.0%) | 0.452 (1.7%) | 0.452 (1.4%) | 0.452 (1.1%) |
| FC_linattn_b (INT4 g=128) | 0.115 (0.6%) | 0.115 (0.5%) | 0.115 (0.5%) | 0.115 (0.5%) | 0.115 (0.5%) | 0.115 (0.4%) | 0.115 (0.4%) | 0.115 (0.3%) |
| FC_linattn_a (INT4 g=128) | 0.114 (0.6%) | 0.114 (0.5%) | 0.114 (0.5%) | 0.114 (0.5%) | 0.114 (0.5%) | 0.114 (0.4%) | 0.114 (0.4%) | 0.114 (0.3%) |

### Prefill TTFT — per-op breakdown (ms / % of TTFT)

| op | S=1,024 | S=2,048 | S=4,096 | S=8,192 | S=16,384 | S=32,768 | S=65,536 | S=131,072 |
|---|---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:|
| MoE_fused (routed+shared, INT4 g=128) | 391.7 (72.4%) | 508.8 (63.0%) | 823.5 (56.1%) | 1373.4 (48.0%) | 2413.4 (39.3%) | 4788.2 (31.0%) | 9142.8 (21.3%) | 18955.2 (12.9%) |
| GatedDeltaNet | 78.9 (14.6%) | 158.2 (19.6%) | 311.7 (21.2%) | 621.2 (21.7%) | 1241.5 (20.2%) | 2480.0 (16.1%) | 5063.1 (11.8%) | 10162.1 (6.9%) |
| FC_linattn_qkv (INT4 g=128) | 20.7 (3.8%) | 39.6 (4.9%) | 82.0 (5.6%) | 163.0 (5.7%) | 328.8 (5.4%) | 721.2 (4.7%) | 1331.9 (3.1%) | 2673.6 (1.8%) |
| FC_linattn_out (INT4 g=128) | 10.6 (2.0%) | 19.9 (2.5%) | 38.8 (2.6%) | 75.1 (2.6%) | 150.6 (2.5%) | 303.7 (2.0%) | 616.7 (1.4%) | 1268.3 (0.9%) |
| FC_linattn_z (INT4 g=128) | 10.6 (2.0%) | 20.1 (2.5%) | 42.4 (2.9%) | 85.0 (3.0%) | 170.6 (2.8%) | 349.5 (2.3%) | 708.1 (1.7%) | 1426.2 (1.0%) |
| PagedAttention (FP16, causal) | 6.9 (1.3%) | 24.9 (3.1%) | 95.3 (6.5%) | 396.6 (13.9%) | 1534.6 (25.0%) | 6215.6 (40.3%) | 24846.7 (58.0%) | 109921.6 (74.9%) |
| LM_head (INT8 g=128, 1 out tok) | 5.2 (1.0%) | 5.2 (0.6%) | 5.2 (0.4%) | 5.2 (0.2%) | 5.2 (0.1%) | 5.2 (0.0%) | 5.2 (0.0%) | 5.2 (0.0%) |
| SmallOps (norm/rope) | 5.0 (0.9%) | 10.0 (1.2%) | 25.2 (1.7%) | 58.5 (2.0%) | 117.0 (1.9%) | 234.0 (1.5%) | 468.1 (1.1%) | 936.2 (0.6%) |
| FC_QKV (INT4 g=128) | 4.2 (0.8%) | 8.0 (1.0%) | 17.5 (1.2%) | 34.3 (1.2%) | 69.6 (1.1%) | 139.9 (0.9%) | 282.4 (0.7%) | 560.4 (0.4%) |
| FC_O (INT4 g=128) | 3.6 (0.7%) | 6.6 (0.8%) | 12.9 (0.9%) | 24.9 (0.9%) | 53.6 (0.9%) | 101.7 (0.7%) | 205.9 (0.5%) | 422.3 (0.3%) |
| FC_linattn_a (INT4 g=128) | 1.8 (0.3%) | 3.4 (0.4%) | 6.6 (0.4%) | 12.4 (0.4%) | 24.8 (0.4%) | 49.6 (0.3%) | 99.3 (0.2%) | 198.5 (0.1%) |
| FC_linattn_b (INT4 g=128) | 1.8 (0.3%) | 3.4 (0.4%) | 6.6 (0.4%) | 13.1 (0.5%) | 26.2 (0.4%) | 52.5 (0.3%) | 104.9 (0.2%) | 209.8 (0.1%) |

## Decode tables (1 query token, KV = context length)

### Decode — KV=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=128) | moe_3gemm_swiglu_mlp | 0.1745 | 40 | 6.981 | 324.46 | 90.3 | 82.1% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| FC_linattn_qkv (INT4 g=128) | gemm_kernel | 0.0862 | 30 | 2.585 | 389.43 | 101.4 | 92.2% | memory |
| FC_linattn_out (INT4 g=128) | gemm_kernel | 0.0461 | 30 | 1.383 | 363.84 | 94.8 | 86.2% | memory |
| FC_linattn_z (INT4 g=128) | gemm_kernel | 0.0446 | 30 | 1.338 | 376.15 | 98.0 | 89.1% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.0652 | 10 | 0.652 | 257.16 | 16.1 | 14.6% | memory |
| FC_QKV (INT4 g=128) | gemm_kernel | 0.0548 | 10 | 0.548 | 382.38 | 99.6 | 90.5% | memory |
| FC_O (INT4 g=128) | gemm_kernel | 0.0461 | 10 | 0.461 | 363.88 | 94.8 | 86.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.0 | 9.1% | memory |
| FC_linattn_a (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.114 | 34.35 | 10.0 | 9.1% | memory |
| **TOTAL** | | | | **20.790** | | | **48.1 tok/s** | |

### Decode — KV=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=128) | moe_3gemm_swiglu_mlp | 0.1745 | 40 | 6.981 | 324.46 | 90.3 | 82.1% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| FC_linattn_qkv (INT4 g=128) | gemm_kernel | 0.0862 | 30 | 2.585 | 389.43 | 101.4 | 92.2% | memory |
| FC_linattn_out (INT4 g=128) | gemm_kernel | 0.0461 | 30 | 1.383 | 363.84 | 94.8 | 86.2% | memory |
| FC_linattn_z (INT4 g=128) | gemm_kernel | 0.0446 | 30 | 1.338 | 376.15 | 98.0 | 89.1% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.1119 | 10 | 1.119 | 299.80 | 18.7 | 17.0% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=128) | gemm_kernel | 0.0548 | 10 | 0.548 | 382.38 | 99.6 | 90.5% | memory |
| FC_O (INT4 g=128) | gemm_kernel | 0.0461 | 10 | 0.461 | 363.88 | 94.8 | 86.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.0 | 9.1% | memory |
| FC_linattn_a (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.114 | 34.35 | 10.0 | 9.1% | memory |
| **TOTAL** | | | | **21.257** | | | **47.0 tok/s** | |

### Decode — KV=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=128) | moe_3gemm_swiglu_mlp | 0.1745 | 40 | 6.981 | 324.46 | 90.3 | 82.1% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| FC_linattn_qkv (INT4 g=128) | gemm_kernel | 0.0862 | 30 | 2.585 | 389.43 | 101.4 | 92.2% | memory |
| FC_linattn_out (INT4 g=128) | gemm_kernel | 0.0461 | 30 | 1.383 | 363.84 | 94.8 | 86.2% | memory |
| FC_linattn_z (INT4 g=128) | gemm_kernel | 0.0446 | 30 | 1.338 | 376.15 | 98.0 | 89.1% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.0888 | 10 | 0.888 | 756.05 | 47.3 | 43.0% | memory |
| FC_QKV (INT4 g=128) | gemm_kernel | 0.0548 | 10 | 0.548 | 382.38 | 99.6 | 90.5% | memory |
| FC_O (INT4 g=128) | gemm_kernel | 0.0461 | 10 | 0.461 | 363.88 | 94.8 | 86.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.0 | 9.1% | memory |
| FC_linattn_a (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.114 | 34.35 | 10.0 | 9.1% | memory |
| **TOTAL** | | | | **21.025** | | | **47.6 tok/s** | |

### Decode — KV=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=128) | moe_3gemm_swiglu_mlp | 0.1745 | 40 | 6.981 | 324.46 | 90.3 | 82.1% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| FC_linattn_qkv (INT4 g=128) | gemm_kernel | 0.0862 | 30 | 2.585 | 389.43 | 101.4 | 92.2% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.1637 | 10 | 1.637 | 820.15 | 51.3 | 46.6% | memory |
| FC_linattn_out (INT4 g=128) | gemm_kernel | 0.0461 | 30 | 1.383 | 363.84 | 94.8 | 86.2% | memory |
| FC_linattn_z (INT4 g=128) | gemm_kernel | 0.0446 | 30 | 1.338 | 376.15 | 98.0 | 89.1% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=128) | gemm_kernel | 0.0548 | 10 | 0.548 | 382.38 | 99.6 | 90.5% | memory |
| FC_O (INT4 g=128) | gemm_kernel | 0.0461 | 10 | 0.461 | 363.88 | 94.8 | 86.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.0 | 9.1% | memory |
| FC_linattn_a (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.114 | 34.35 | 10.0 | 9.1% | memory |
| **TOTAL** | | | | **21.774** | | | **45.9 tok/s** | |

### Decode — KV=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=128) | moe_3gemm_swiglu_mlp | 0.1745 | 40 | 6.981 | 324.46 | 90.3 | 82.1% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.2971 | 10 | 2.971 | 903.48 | 56.5 | 51.3% | memory |
| FC_linattn_qkv (INT4 g=128) | gemm_kernel | 0.0862 | 30 | 2.585 | 389.43 | 101.4 | 92.2% | memory |
| FC_linattn_out (INT4 g=128) | gemm_kernel | 0.0461 | 30 | 1.383 | 363.84 | 94.8 | 86.2% | memory |
| FC_linattn_z (INT4 g=128) | gemm_kernel | 0.0446 | 30 | 1.338 | 376.15 | 98.0 | 89.1% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=128) | gemm_kernel | 0.0548 | 10 | 0.548 | 382.38 | 99.6 | 90.5% | memory |
| FC_O (INT4 g=128) | gemm_kernel | 0.0461 | 10 | 0.461 | 363.88 | 94.8 | 86.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.0 | 9.1% | memory |
| FC_linattn_a (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.114 | 34.35 | 10.0 | 9.1% | memory |
| **TOTAL** | | | | **23.109** | | | **43.3 tok/s** | |

### Decode — KV=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=128) | moe_3gemm_swiglu_mlp | 0.1745 | 40 | 6.981 | 324.46 | 90.3 | 82.1% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt | 0.5834 | 10 | 5.834 | 920.25 | 57.5 | 52.3% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| FC_linattn_qkv (INT4 g=128) | gemm_kernel | 0.0862 | 30 | 2.585 | 389.43 | 101.4 | 92.2% | memory |
| FC_linattn_out (INT4 g=128) | gemm_kernel | 0.0461 | 30 | 1.383 | 363.84 | 94.8 | 86.2% | memory |
| FC_linattn_z (INT4 g=128) | gemm_kernel | 0.0446 | 30 | 1.338 | 376.15 | 98.0 | 89.1% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=128) | gemm_kernel | 0.0548 | 10 | 0.548 | 382.38 | 99.6 | 90.5% | memory |
| FC_O (INT4 g=128) | gemm_kernel | 0.0461 | 10 | 0.461 | 363.88 | 94.8 | 86.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.0 | 9.1% | memory |
| FC_linattn_a (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.114 | 34.35 | 10.0 | 9.1% | memory |
| **TOTAL** | | | | **25.972** | | | **38.5 tok/s** | |

### Decode — KV=65,536

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (INT8 KV) | paged_attention_opt | 1.1437 | 10 | 11.437 | 938.80 | 58.7 | 53.3% | memory |
| MoE_fused (routed+shared, INT4 g=128) | moe_3gemm_swiglu_mlp | 0.1745 | 40 | 6.981 | 324.46 | 90.3 | 82.1% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| FC_linattn_qkv (INT4 g=128) | gemm_kernel | 0.0862 | 30 | 2.585 | 389.43 | 101.4 | 92.2% | memory |
| FC_linattn_out (INT4 g=128) | gemm_kernel | 0.0461 | 30 | 1.383 | 363.84 | 94.8 | 86.2% | memory |
| FC_linattn_z (INT4 g=128) | gemm_kernel | 0.0446 | 30 | 1.338 | 376.15 | 98.0 | 89.1% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=128) | gemm_kernel | 0.0548 | 10 | 0.548 | 382.38 | 99.6 | 90.5% | memory |
| FC_O (INT4 g=128) | gemm_kernel | 0.0461 | 10 | 0.461 | 363.88 | 94.8 | 86.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.0 | 9.1% | memory |
| FC_linattn_a (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.114 | 34.35 | 10.0 | 9.1% | memory |
| **TOTAL** | | | | **31.575** | | | **31.7 tok/s** | |

### Decode — KV=131,072

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (INT8 KV) | paged_attention_opt | 2.2270 | 10 | 22.270 | 964.31 | 60.3 | 54.8% | memory |
| MoE_fused (routed+shared, INT4 g=128) | moe_3gemm_swiglu_mlp | 0.1745 | 40 | 6.981 | 324.46 | 90.3 | 82.1% | memory |
| LM_head (INT8 g=128) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| FC_linattn_qkv (INT4 g=128) | gemm_kernel | 0.0862 | 30 | 2.585 | 389.43 | 101.4 | 92.2% | memory |
| FC_linattn_out (INT4 g=128) | gemm_kernel | 0.0461 | 30 | 1.383 | 363.84 | 94.8 | 86.2% | memory |
| FC_linattn_z (INT4 g=128) | gemm_kernel | 0.0446 | 30 | 1.338 | 376.15 | 98.0 | 89.1% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0320 | 30 | 0.960 | 32.76 | 65.6 | 59.7% | memory |
| FC_QKV (INT4 g=128) | gemm_kernel | 0.0548 | 10 | 0.548 | 382.38 | 99.6 | 90.5% | memory |
| FC_O (INT4 g=128) | gemm_kernel | 0.0461 | 10 | 0.461 | 363.88 | 94.8 | 86.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.452 | — | — | — | memory |
| FC_linattn_b (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.20 | 10.0 | 9.1% | memory |
| FC_linattn_a (INT4 g=128) | gemm_kernel | 0.0038 | 30 | 0.114 | 34.35 | 10.0 | 9.1% | memory |
| **TOTAL** | | | | **42.407** | | | **23.6 tok/s** | |

## Prefill tables (single forward over S tokens)

### Prefill — S=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=128) | grouped_micro_gemm+scatter/gather | 9.7924 | 40 | 391.695 | 5921.15 | 43.9 | 39.9% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 2.6316 | 30 | 78.949 | 408.01 | 2.4 | 2.2% | memory |
| FC_linattn_qkv (INT4 g=128) | dq+gemm_kernel | 0.6916 | 30 | 20.748 | 49680.87 | 42.9 | 42.1% | compute |
| FC_linattn_out (INT4 g=128) | dq+gemm_kernel | 0.3536 | 30 | 10.608 | 48586.84 | 47.9 | 43.6% | memory |
| FC_linattn_z (INT4 g=128) | dq+gemm_kernel | 0.3520 | 30 | 10.559 | 48810.75 | 48.1 | 43.8% | memory |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 0.6890 | 10 | 6.890 | 12467.34 | 1.5 | 21.1% | compute |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| SmallOps (norm/rope) | rms/rope | — | — | 5.033 | — | — | — | memory |
| FC_QKV (INT4 g=128) | dq+gemm_kernel | 0.4151 | 10 | 4.151 | 51739.11 | 48.5 | 44.1% | memory |
| FC_O (INT4 g=128) | dq+gemm_kernel | 0.3557 | 10 | 3.557 | 48295.37 | 47.6 | 43.3% | memory |
| FC_linattn_a (INT4 g=128) | dq+gemm_kernel | 0.0599 | 30 | 1.796 | 2241.59 | 71.7 | 65.2% | memory |
| FC_linattn_b (INT4 g=128) | dq+gemm_kernel | 0.0596 | 30 | 1.787 | 2252.88 | 72.1 | 65.5% | memory |
| **TOTAL** | | | | **541.0** | | | **1892.9 tok/s** | |

### Prefill — S=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=128) | grouped_micro_gemm+scatter/gather | 12.7212 | 40 | 508.847 | 9115.83 | 34.4 | 31.3% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 5.2740 | 30 | 158.221 | 407.18 | 2.0 | 1.8% | memory |
| FC_linattn_qkv (INT4 g=128) | dq+gemm_kernel | 1.3199 | 30 | 39.596 | 52066.01 | 38.4 | 44.1% | compute |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 2.4853 | 10 | 24.853 | 13825.18 | 0.8 | 23.4% | compute |
| FC_linattn_z (INT4 g=128) | dq+gemm_kernel | 0.6689 | 30 | 20.067 | 51367.99 | 44.1 | 43.5% | compute |
| FC_linattn_out (INT4 g=128) | dq+gemm_kernel | 0.6636 | 30 | 19.908 | 51777.01 | 44.5 | 43.9% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 9.970 | — | — | — | memory |
| FC_QKV (INT4 g=128) | dq+gemm_kernel | 0.8016 | 10 | 8.016 | 53580.73 | 43.4 | 45.4% | compute |
| FC_O (INT4 g=128) | dq+gemm_kernel | 0.6605 | 10 | 6.605 | 52024.50 | 44.7 | 44.1% | compute |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| FC_linattn_b (INT4 g=128) | dq+gemm_kernel | 0.1127 | 30 | 3.381 | 2381.73 | 75.9 | 69.0% | memory |
| FC_linattn_a (INT4 g=128) | dq+gemm_kernel | 0.1122 | 30 | 3.366 | 2392.58 | 76.2 | 69.3% | memory |
| **TOTAL** | | | | **808.0** | | | **2534.6 tok/s** | |

### Prefill — S=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=128) | grouped_micro_gemm+scatter/gather | 20.5871 | 40 | 823.485 | 11265.70 | 22.1 | 20.1% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 10.3904 | 30 | 311.712 | 413.36 | 1.8 | 1.7% | memory |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 9.5349 | 10 | 95.349 | 14414.33 | 0.4 | 24.4% | compute |
| FC_linattn_qkv (INT4 g=128) | dq+gemm_kernel | 2.7317 | 30 | 81.952 | 50311.74 | 33.9 | 42.6% | compute |
| FC_linattn_z (INT4 g=128) | dq+gemm_kernel | 1.4147 | 30 | 42.441 | 48575.51 | 38.7 | 41.2% | compute |
| FC_linattn_out (INT4 g=128) | dq+gemm_kernel | 1.2926 | 30 | 38.779 | 53162.69 | 42.3 | 45.1% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 25.156 | — | — | — | memory |
| FC_QKV (INT4 g=128) | dq+gemm_kernel | 1.7472 | 10 | 17.472 | 49163.55 | 36.7 | 41.7% | compute |
| FC_O (INT4 g=128) | dq+gemm_kernel | 1.2906 | 10 | 12.906 | 53244.54 | 42.4 | 45.1% | compute |
| FC_linattn_b (INT4 g=128) | dq+gemm_kernel | 0.2196 | 30 | 6.589 | 2444.38 | 77.7 | 70.7% | memory |
| FC_linattn_a (INT4 g=128) | dq+gemm_kernel | 0.2186 | 30 | 6.558 | 2455.82 | 78.1 | 71.0% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| **TOTAL** | | | | **1467.6** | | | **2791.0 tok/s** | |

### Prefill — S=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=128) | grouped_micro_gemm+scatter/gather | 34.3338 | 40 | 1373.351 | 13510.20 | 14.2 | 12.9% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 20.7070 | 30 | 621.210 | 414.83 | 1.7 | 1.6% | memory |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 39.6626 | 10 | 396.626 | 13860.82 | 0.2 | 23.5% | compute |
| FC_linattn_qkv (INT4 g=128) | dq+gemm_kernel | 5.4338 | 30 | 163.014 | 50586.70 | 32.5 | 42.9% | compute |
| FC_linattn_z (INT4 g=128) | dq+gemm_kernel | 2.8339 | 30 | 85.016 | 48498.46 | 37.1 | 41.1% | compute |
| FC_linattn_out (INT4 g=128) | dq+gemm_kernel | 2.5031 | 30 | 75.094 | 54906.49 | 42.0 | 46.5% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 58.510 | — | — | — | memory |
| FC_QKV (INT4 g=128) | dq+gemm_kernel | 3.4293 | 10 | 34.293 | 50098.02 | 35.8 | 42.5% | compute |
| FC_O (INT4 g=128) | dq+gemm_kernel | 2.4888 | 10 | 24.888 | 55223.62 | 42.2 | 46.8% | compute |
| FC_linattn_b (INT4 g=128) | dq+gemm_kernel | 0.4371 | 30 | 13.114 | 2456.30 | 78.0 | 70.9% | memory |
| FC_linattn_a (INT4 g=128) | dq+gemm_kernel | 0.4136 | 30 | 12.407 | 2596.25 | 82.5 | 75.0% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| **TOTAL** | | | | **2862.7** | | | **2861.6 tok/s** | |

### Prefill — S=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (routed+shared, INT4 g=128) | grouped_micro_gemm+scatter/gather | 60.3338 | 40 | 2413.351 | 15376.35 | 9.2 | 13.0% | compute |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 153.4550 | 10 | 1534.550 | 14330.08 | 0.1 | 24.3% | compute |
| GatedDeltaNet | gated_delta_net_ref_sa | 41.3845 | 30 | 1241.536 | 415.13 | 1.7 | 1.5% | memory |
| FC_linattn_qkv (INT4 g=128) | dq+gemm_kernel | 10.9609 | 30 | 328.827 | 50156.08 | 31.4 | 42.5% | compute |
| FC_linattn_z (INT4 g=128) | dq+gemm_kernel | 5.6858 | 30 | 170.575 | 48344.36 | 36.2 | 41.0% | compute |
| FC_linattn_out (INT4 g=128) | dq+gemm_kernel | 5.0210 | 30 | 150.631 | 54745.21 | 41.0 | 46.4% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 117.021 | — | — | — | memory |
| FC_QKV (INT4 g=128) | dq+gemm_kernel | 6.9622 | 10 | 69.622 | 49351.69 | 34.5 | 41.8% | compute |
| FC_O (INT4 g=128) | dq+gemm_kernel | 5.3632 | 10 | 53.632 | 51252.62 | 38.4 | 43.4% | compute |
| FC_linattn_b (INT4 g=128) | dq+gemm_kernel | 0.8743 | 30 | 26.228 | 2456.30 | 78.0 | 70.9% | memory |
| FC_linattn_a (INT4 g=128) | dq+gemm_kernel | 0.8271 | 30 | 24.814 | 2596.25 | 82.4 | 74.9% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| **TOTAL** | | | | **6136.0** | | | **2670.1 tok/s** | |

### Prefill — S=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 621.5587 | 10 | 6215.587 | 14151.67 | 0.1 | 24.0% | compute |
| MoE_fused (routed+shared, INT4 g=128) | grouped_micro_gemm+scatter/gather | 119.7048 | 40 | 4788.191 | 15500.01 | 5.8 | 13.1% | compute |
| GatedDeltaNet | gated_delta_net_ref_sa | 82.6682 | 30 | 2480.046 | 415.63 | 1.6 | 1.5% | memory |
| FC_linattn_qkv (INT4 g=128) | dq+gemm_kernel | 24.0414 | 30 | 721.241 | 45734.14 | 28.3 | 38.8% | compute |
| FC_linattn_z (INT4 g=128) | dq+gemm_kernel | 11.6514 | 30 | 349.541 | 47183.77 | 34.9 | 40.0% | compute |
| FC_linattn_out (INT4 g=128) | dq+gemm_kernel | 10.1249 | 30 | 303.747 | 54297.37 | 40.2 | 46.0% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 234.041 | — | — | — | memory |
| FC_QKV (INT4 g=128) | dq+gemm_kernel | 13.9949 | 10 | 139.949 | 49103.18 | 34.0 | 41.6% | compute |
| FC_O (INT4 g=128) | dq+gemm_kernel | 10.1686 | 10 | 101.686 | 54064.26 | 40.0 | 45.8% | compute |
| FC_linattn_b (INT4 g=128) | dq+gemm_kernel | 1.7486 | 30 | 52.457 | 2456.30 | 78.0 | 70.9% | memory |
| FC_linattn_a (INT4 g=128) | dq+gemm_kernel | 1.6543 | 30 | 49.629 | 2596.25 | 82.4 | 74.9% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| **TOTAL** | | | | **15441.3** | | | **2122.1 tok/s** | |

### Prefill — S=65,536

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 2484.6714 | 10 | 24846.714 | 14160.57 | 0.0 | 24.0% | compute |
| MoE_fused (routed+shared, INT4 g=128) | grouped_micro_gemm+scatter/gather | 228.5693 | 40 | 9142.771 | 16235.13 | 4.2 | 13.8% | compute |
| GatedDeltaNet | gated_delta_net_ref_sa | 168.7711 | 30 | 5063.133 | 407.18 | 1.6 | 1.5% | memory |
| FC_linattn_qkv (INT4 g=128) | dq+gemm_kernel | 44.3973 | 30 | 1331.920 | 49530.53 | 30.4 | 42.0% | compute |
| FC_linattn_z (INT4 g=128) | dq+gemm_kernel | 23.6018 | 30 | 708.055 | 46585.86 | 34.3 | 39.5% | compute |
| FC_linattn_out (INT4 g=128) | dq+gemm_kernel | 20.5582 | 30 | 616.746 | 53482.91 | 39.4 | 45.3% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 468.083 | — | — | — | memory |
| FC_QKV (INT4 g=128) | dq+gemm_kernel | 28.2441 | 10 | 282.441 | 48661.07 | 33.5 | 41.3% | compute |
| FC_O (INT4 g=128) | dq+gemm_kernel | 20.5883 | 10 | 205.883 | 53404.60 | 39.3 | 45.3% | compute |
| FC_linattn_b (INT4 g=128) | dq+gemm_kernel | 3.4971 | 30 | 104.913 | 2456.30 | 78.0 | 70.9% | memory |
| FC_linattn_a (INT4 g=128) | dq+gemm_kernel | 3.3086 | 30 | 99.258 | 2596.25 | 82.4 | 74.9% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| **TOTAL** | | | | **42875.1** | | | **1528.5 tok/s** | |

### Prefill — S=131,072

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 10992.1630 | 10 | 109921.630 | 12803.44 | 0.0 | 21.7% | compute |
| MoE_fused (routed+shared, INT4 g=128) | grouped_micro_gemm+scatter/gather | 473.8790 | 40 | 18955.160 | 15661.60 | 3.2 | 13.3% | compute |
| GatedDeltaNet | gated_delta_net_ref_sa | 338.7359 | 30 | 10162.078 | 405.74 | 1.6 | 1.4% | memory |
| FC_linattn_qkv (INT4 g=128) | dq+gemm_kernel | 89.1199 | 30 | 2673.597 | 49349.77 | 30.2 | 41.8% | compute |
| FC_linattn_z (INT4 g=128) | dq+gemm_kernel | 47.5399 | 30 | 1426.197 | 46256.38 | 34.0 | 39.2% | compute |
| FC_linattn_out (INT4 g=128) | dq+gemm_kernel | 42.2774 | 30 | 1268.323 | 52014.12 | 38.2 | 44.1% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 936.165 | — | — | — | memory |
| FC_QKV (INT4 g=128) | dq+gemm_kernel | 56.0433 | 10 | 560.433 | 49047.39 | 33.6 | 41.6% | compute |
| FC_O (INT4 g=128) | dq+gemm_kernel | 42.2261 | 10 | 422.261 | 52077.29 | 38.2 | 44.1% | compute |
| FC_linattn_b (INT4 g=128) | dq+gemm_kernel | 6.9942 | 30 | 209.826 | 2456.30 | 78.0 | 70.9% | memory |
| FC_linattn_a (INT4 g=128) | dq+gemm_kernel | 6.6172 | 30 | 198.516 | 2596.25 | 82.4 | 74.9% | memory |
| LM_head (INT8 g=128, 1 out tok) | gemm_kernel | 5.1994 | 1 | 5.199 | 195.62 | 99.4 | 90.4% | memory |
| **TOTAL** | | | | **146739.4** | | | **893.2 tok/s** | |

## Top contributors

### Decode

| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1,024 | MoE_fused (routed+shared, INT4 g=128) 6.98ms (33.6%) | LM_head (INT8 g=128) 5.20ms (25.0%) | FC_linattn_qkv (INT4 g=128) 2.58ms (12.4%) |
| 2,048 | MoE_fused (routed+shared, INT4 g=128) 6.98ms (32.8%) | LM_head (INT8 g=128) 5.20ms (24.5%) | FC_linattn_qkv (INT4 g=128) 2.58ms (12.2%) |
| 4,096 | MoE_fused (routed+shared, INT4 g=128) 6.98ms (33.2%) | LM_head (INT8 g=128) 5.20ms (24.7%) | FC_linattn_qkv (INT4 g=128) 2.58ms (12.3%) |
| 8,192 | MoE_fused (routed+shared, INT4 g=128) 6.98ms (32.1%) | LM_head (INT8 g=128) 5.20ms (23.9%) | FC_linattn_qkv (INT4 g=128) 2.58ms (11.9%) |
| 16,384 | MoE_fused (routed+shared, INT4 g=128) 6.98ms (30.2%) | LM_head (INT8 g=128) 5.20ms (22.5%) | PagedAttention (INT8 KV) 2.97ms (12.9%) |
| 32,768 | MoE_fused (routed+shared, INT4 g=128) 6.98ms (26.9%) | PagedAttention (INT8 KV) 5.83ms (22.5%) | LM_head (INT8 g=128) 5.20ms (20.0%) |
| 65,536 | PagedAttention (INT8 KV) 11.44ms (36.2%) | MoE_fused (routed+shared, INT4 g=128) 6.98ms (22.1%) | LM_head (INT8 g=128) 5.20ms (16.5%) |
| 131,072 | PagedAttention (INT8 KV) 22.27ms (52.5%) | MoE_fused (routed+shared, INT4 g=128) 6.98ms (16.5%) | LM_head (INT8 g=128) 5.20ms (12.3%) |

### Prefill

| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1,024 | MoE_fused (routed+shared, INT4 g=128) 391.7ms (72.4%) | GatedDeltaNet 78.9ms (14.6%) | FC_linattn_qkv (INT4 g=128) 20.7ms (3.8%) |
| 2,048 | MoE_fused (routed+shared, INT4 g=128) 508.8ms (63.0%) | GatedDeltaNet 158.2ms (19.6%) | FC_linattn_qkv (INT4 g=128) 39.6ms (4.9%) |
| 4,096 | MoE_fused (routed+shared, INT4 g=128) 823.5ms (56.1%) | GatedDeltaNet 311.7ms (21.2%) | PagedAttention (FP16, causal) 95.3ms (6.5%) |
| 8,192 | MoE_fused (routed+shared, INT4 g=128) 1373.4ms (48.0%) | GatedDeltaNet 621.2ms (21.7%) | PagedAttention (FP16, causal) 396.6ms (13.9%) |
| 16,384 | MoE_fused (routed+shared, INT4 g=128) 2413.4ms (39.3%) | PagedAttention (FP16, causal) 1534.6ms (25.0%) | GatedDeltaNet 1241.5ms (20.2%) |
| 32,768 | PagedAttention (FP16, causal) 6215.6ms (40.3%) | MoE_fused (routed+shared, INT4 g=128) 4788.2ms (31.0%) | GatedDeltaNet 2480.0ms (16.1%) |
| 65,536 | PagedAttention (FP16, causal) 24846.7ms (58.0%) | MoE_fused (routed+shared, INT4 g=128) 9142.8ms (21.3%) | GatedDeltaNet 5063.1ms (11.8%) |
| 131,072 | PagedAttention (FP16, causal) 109921.6ms (74.9%) | MoE_fused (routed+shared, INT4 g=128) 18955.2ms (12.9%) | GatedDeltaNet 10162.1ms (6.9%) |

## Caveats & method

- Each op profiled in its own process via cliloader Device Performance Timing; mean kernel time per iteration.
- FC weight bytes count INT4 weight + FP16 scale/zp (g=128) + FP16 activations.
- PA/GDN/SmallOps data reused from g=64 v4 run (group-size independent ops).
- **v4**: Shared expert fused into MOE3GemmFusedCompressed (commit 2abcdce7f3).
- Target machine: Local_Admin@10.239.132.229 (PTL B390 iGPU, 12 Xe cores)

## Reproduction

```bat
REM Full script: run_qwen3_5_moe_ptl_int4g128_v4.bat
```

_Generated by `build_report_ptl_int4g128_v4.py`_