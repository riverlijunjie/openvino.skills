# Qwen3.5-MoE-35B-A3B — INT4 Performance Analysis Report (v4)

## Platform: PTL 12Xe (Intel Arc B390 iGPU)

| Parameter | Value |
|---|---|
| Xe Cores | 12 |
| EU/Core | 8 |
| Threads/EU | 10 |
| Frequency | 2400 MHz |
| Memory BW | 110 GB/s |
| FP16 XMX peak | 58.98 TFLOPS |
| INT8 XMX peak | 117.96 TOPS |

## Configuration

| Component | Precision | Notes |
|---|---|---|
| Body FC (QKV, O, linattn projs) | INT4 asymmetric | g=64 and g=128 tested |
| MoE (routed + shared expert) | INT4 asymmetric | **Fused** — shared_I=512, shared_quant=u4 |
| LM head | INT8 asymmetric | g=128 (both configs) |
| KV cache | INT8 | |
| PA | FP16 causal SDPA | NH=16, NKV=2, HD=256 |
| GDN | FP16 | HK=32, KD=VD=128 |

> **v4 key change**: Shared expert is now **fused into the MoE kernel** via `FuseMOESharedExpert` transformation (OV commit `2abcdce7f3`). In v3, shared expert ran as 3 separate FC ops.

## 1. Decode Latency — g=64 vs g=128 (v4, fused shared expert)

| KV Length | g=64 (ms) | g=128 (ms) | Delta | g=128 speedup |
|---:|---:|---:|---:|---:|
| 1,024 | 21.85 | 20.79 | -1.06 | -4.9% |
| 2,048 | 22.32 | 21.26 | -1.06 | -4.8% |
| 4,096 | 22.09 | 21.03 | -1.06 | -4.8% |
| 8,192 | 22.84 | 21.77 | -1.06 | -4.6% |
| 16,384 | 24.17 | 23.11 | -1.06 | -4.4% |
| 32,768 | 27.03 | 25.97 | -1.06 | -3.9% |
| 65,536 | 32.64 | 31.58 | -1.06 | -3.3% |
| 131,072 | 43.47 | 42.41 | -1.06 | -2.4% |

## 2. Prefill Latency — g=64 vs g=128 (v4, fused shared expert)

| Seq Length | g=64 (ms) | g=128 (ms) | Delta | g=128 speedup |
|---:|---:|---:|---:|---:|
| 1,024 | 625.0 | 541.0 | -84.0 | -13.4% |
| 2,048 | 904.1 | 808.1 | -96.0 | -10.6% |
| 4,096 | 1594.7 | 1467.6 | -127.1 | -8.0% |
| 8,192 | 3088.2 | 2862.8 | -225.5 | -7.3% |
| 16,384 | 6546.4 | 6136.0 | -410.4 | -6.3% |
| 32,768 | 16343.3 | 15441.4 | -902.0 | -5.5% |
| 65,536 | 45084.9 | 42875.2 | -2209.8 | -4.9% |
| 131,072 | 151409.6 | 146739.4 | -4670.1 | -3.1% |

## 3. Token Throughput

### Decode (tok/s)

| KV Length | g=64 tok/s | g=128 tok/s |
|---:|---:|---:|
| 1,024 | 45.8 | 48.1 |
| 4,096 | 45.3 | 47.6 |
| 32,768 | 37.0 | 38.5 |
| 131,072 | 23.0 | 23.6 |

### Prefill (tok/s)

| Seq Length | g=64 tok/s | g=128 tok/s |
|---:|---:|---:|
| 1,024 | 1638 | 1893 |
| 4,096 | 2568 | 2791 |
| 32,768 | 2005 | 2122 |
| 131,072 | 866 | 893 |

## 4. Per-Op Decode Breakdown (KV=4096)

| Op | g=64 (ms) | g=128 (ms) | Calls | g=64 Total (ms) | g=128 Total (ms) | Delta % |
|---|---:|---:|---:|---:|---:|---:|
| FC_QKV | 0.0606 | 0.0548 | 10 | 0.606 | 0.548 | -9.5% |
| FC_O | 0.0481 | 0.0461 | 10 | 0.481 | 0.461 | -4.2% |
| linattn_qkv | 0.0920 | 0.0862 | 30 | 2.760 | 2.585 | -6.3% |
| linattn_z | 0.0484 | 0.0446 | 30 | 1.452 | 1.338 | -7.9% |
| linattn_a | 0.0038 | 0.0038 | 30 | 0.115 | 0.114 | -0.4% |
| linattn_b | 0.0038 | 0.0038 | 30 | 0.115 | 0.115 | -0.1% |
| linattn_out | 0.0478 | 0.0461 | 30 | 1.434 | 1.383 | -3.6% |
| LM_head | 5.2361 | 5.1994 | 1 | 5.236 | 5.199 | -0.7% |
| MoE_fused | 0.1897 | 0.1745 | 40 | 7.586 | 6.981 | -8.0% |
| PA (KV=4096) | 0.0888 | 0.0888 | 10 | 0.888 | 0.888 | 0.0% |
| GDN | 0.0320 | 0.0320 | 30 | 0.960 | 0.960 | 0.0% |
| SmallOps | — | — | — | 0.452 | 0.452 | 0.0% |

## 5. MoE Layer: Fused (v4) vs Unfused (v3) Comparison

In v3, the shared expert ran as 3 separate INT4 FC ops (gate, up, down). In v4, it is fused into the MoE kernel.

### Decode (MoE layer only, per-layer)

| Config | v3 MoE routed | v3 shared (3×FC) | v3 Total | v4 Fused | Fusion speedup |
|---|---:|---:|---:|---:|---:|
| g=64 | 0.1724 | 0.0282 | 0.2006 | 0.1897 | -5.4% |
| g=128 | 0.1556 | 0.0257 | 0.1813 | 0.1745 | -3.7% |

### Prefill (MoE layer only, per-layer)

| Config | Seq Len | v3 MoE routed | v3 shared (3×FC) | v3 Total | v4 Fused | Fusion Δ% |
|---|---:|---:|---:|---:|---:|---:|
| g=64 | 1,024 | 10.757 | 0.528 | 11.284 | 11.232 | -0.5% |
| g=64 | 2,048 | 13.475 | 0.922 | 14.397 | 13.912 | -3.4% |
| g=64 | 4,096 | 20.243 | 1.819 | 22.062 | 21.426 | -2.9% |
| g=64 | 8,192 | 32.715 | 3.603 | 36.317 | 35.378 | -2.6% |
| g=64 | 16,384 | 55.490 | 7.455 | 62.946 | 60.910 | -3.2% |
| g=64 | 32,768 | 106.687 | 15.960 | 122.647 | 120.727 | -1.6% |
| g=64 | 65,536 | 211.941 | 38.759 | 250.699 | 231.737 | -7.6% |
| g=64 | 131,072 | 464.845 | 66.476 | 531.321 | 475.214 | -10.6% |
| g=128 | 1,024 | 9.632 | 0.325 | 9.957 | 9.792 | -1.7% |
| g=128 | 2,048 | 12.198 | 0.545 | 12.744 | 12.721 | -0.2% |
| g=128 | 4,096 | 19.321 | 1.063 | 20.384 | 20.587 | +1.0% |
| g=128 | 8,192 | 31.872 | 2.117 | 33.989 | 34.334 | +1.0% |
| g=128 | 16,384 | 56.211 | 4.401 | 60.613 | 60.334 | -0.5% |
| g=128 | 32,768 | 109.562 | 9.133 | 118.694 | 119.705 | +0.9% |
| g=128 | 65,536 | 207.265 | 19.181 | 226.446 | 228.569 | +0.9% |
| g=128 | 131,072 | 428.996 | 39.474 | 468.470 | 473.879 | +1.2% |

## 6. End-to-End Pipeline: v3 vs v4

### Decode (KV=4096)

| Config | v3 (ms) | v4 (ms) | Delta |
|---|---:|---:|---:|
| g=64 | 22.50 | 22.09 | -1.8% |
| g=128 | 21.30 | 21.03 | -1.3% |

### Prefill

| Config | Seq Len | v3 (ms) | v4 (ms) | Delta |
|---|---:|---:|---:|---:|
| g=64 | 1,024 | 625.7 | 625.0 | -0.1% |
| g=64 | 4,096 | 1615.9 | 1594.7 | -1.3% |
| g=64 | 32,768 | 16447.3 | 16343.3 | -0.6% |
| g=64 | 131,072 | 154667.9 | 151409.6 | -2.1% |
| g=128 | 1,024 | 545.6 | 541.0 | -0.8% |
| g=128 | 4,096 | 1460.3 | 1467.6 | +0.5% |
| g=128 | 32,768 | 15412.3 | 15441.4 | +0.2% |
| g=128 | 131,072 | 147328.3 | 146739.4 | -0.4% |

## 7. Roofline Analysis — Key Ops

### MoE Fused Decode (M=1)

| Config | Latency (ms) | GFLOPS | GB/s | BW Eff | XMX Eff | Bound |
|---|---:|---:|---:|---:|---:|---|
| g=64 | 0.1897 | 298.6 | 86.0 | 78.2% | 0.51% | memory |
| g=128 | 0.1745 | 324.5 | 90.3 | 82.1% | 0.55% | memory |

### FC_QKV Decode (M=1, K=2048, N=5120)

| Config | Latency (ms) | GFLOPS | GB/s | BW Eff | Bound |
|---|---:|---:|---:|---:|---|
| g=64 | 0.0606 | 346.1 | 93.5 | 85.0% | memory |
| g=128 | 0.0548 | 382.4 | 99.6 | 90.5% | memory |

### MoE Fused Prefill (S=4096)

| Config | Latency (ms) | GFLOPS | GB/s | XMX Eff | Bound |
|---|---:|---:|---:|---:|---|
| g=64 | 21.426 | 10824.6 | 22.0 | 9.2% | memory |
| g=128 | 20.587 | 11265.7 | 22.1 | 9.6% | memory |

## 8. Per-Op Prefill Breakdown (S=4096)

| Op | g=64 (ms) | g=128 (ms) | Calls | g=64 Total (ms) | g=128 Total (ms) | Delta % |
|---|---:|---:|---:|---:|---:|---:|
| FC_QKV | 2.202 | 1.747 | 10 | 22.02 | 17.47 | -20.6% |
| FC_O | 2.200 | 1.291 | 10 | 22.00 | 12.91 | -41.3% |
| linattn_qkv | 3.408 | 2.732 | 30 | 102.24 | 81.95 | -19.8% |
| linattn_z | 1.889 | 1.415 | 30 | 56.66 | 42.44 | -25.1% |
| linattn_a | 0.540 | 0.219 | 30 | 16.19 | 6.56 | -59.5% |
| linattn_b | 0.540 | 0.220 | 30 | 16.19 | 6.59 | -59.3% |
| linattn_out | 2.165 | 1.293 | 30 | 64.94 | 38.78 | -40.3% |
| MoE_fused | 21.426 | 20.587 | 40 | 857.04 | 823.48 | -3.9% |
| PA (S=4096) | 9.535 | 9.535 | 10 | 95.35 | 95.35 | 0.0% |
| GDN | 10.390 | 10.390 | 30 | 311.71 | 311.71 | 0.0% |

## 9. Key Findings

1. **g=128 vs g=64 decode** (KV=4096): g=128 is **4.8% faster** (22.09 ms → 21.03 ms)
2. **g=128 vs g=64 prefill** (S=4096): g=128 is **8.0% faster** (1594.7 ms → 1467.6 ms)
3. **Shared expert fusion impact (decode, g=64)**: v4 fused MoE is **5.4% faster** per layer (0.2006 → 0.1897 ms)
4. **Shared expert fusion impact (prefill S=131072, g=64)**: v4 fused is **10.6% faster** per layer (531.32 → 475.21 ms)
5. **Decode throughput** (KV=4096): g=64 = 45.3 tok/s, g=128 = 47.6 tok/s
6. **Prefill throughput** (S=4096): g=64 = 2568 tok/s, g=128 = 2791 tok/s
7. **LM head dominates decode**: 5.24 ms = 24% of total decode (KV=4096)
8. **PA dominates long-context prefill**: PA S=131072 = 109922 ms = 73% of total

---
*Report generated from v4 benchmark data. Platform: PTL 12Xe (B390 iGPU). OV branch: river/moe_expert_precision_f16_support with FuseMOESharedExpert fix.*
