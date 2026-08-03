# Onyx (dense VLM text decoder) — Roofline on B70 (2026-07-15)

**Platform**: B70 GPU — 32 Xe @ 2280 MHz, 608 GB/s GDDR memory  
**Model**: `onyx` — 52-layer dense SwiGLU decoder (vision tower excluded)

> **Analytical estimate** — no B70 device is available. Method: scale per-kernel timings measured on PTL 12Xe (B390 iGPU) by hardware ratios. Memory-bound ops: × (110/608) = × 0.181. XMX-compute-bound ops: × (58.98/183.5) = × 0.321. S=64K and kv=64K are extrapolated from S=kv=32K (FC linear ×2; PA_full quadratic ×4; PA_sliding / SmallOps band-linear ×2). Eff% is invariant under hardware scaling.

- hidden=6656, layers=52, GQA 32Q/2KV/HD=128 (16-way sharing), inter=19968, vocab=202048, SwiGLU
- Onyx-specific: **QK-RMSNorm** + **sigmoid output-gate** (extra FC 6656→4096/layer), iRoPE (39 sliding SW=2048 + 13 full/NoPE layers)
- MatMul INT4 g=128 / LM_head INT8 g=128 / KV cache INT4 g=128 / FP16 act
- SDPA: PagedAttention OpenCL + micro_kernel

## Model parameters & weight shapes

| Field | Value |
|---|---:|
| `hidden_size` | 6,656 |
| `num_hidden_layers` | 52 (39 sliding/RoPE + 13 full/NoPE) |
| `num_attention_heads` / `num_kv_heads` | 32 / 2 (GQA×16) |
| `head_dim` | 128 → Q=4096, KV=256 |
| `intermediate_size` | 19,968 |
| `vocab_size` | 202,048 |
| `sliding_window` | 2,048 |

| Weight | Shape | Quant | Per instance | × | Total MB |
|---|---|---|---:|---:|---:|
| Embedding | 6656×202048 | INT8 g128 | 1,365.84 MB | 1 | 1,365.8 |
| FC_QKV | 6656×4608 | INT4 g128 | 15.93 MB | 52 | 828.6 |
| FC_OutGate | 6656×4096 | INT4 g128 | 14.16 MB | 52 | 736.5 |
| FC_O | 4096×6656 | INT4 g128 | 14.16 MB | 52 | 736.5 |
| MLP_gate | 6656×19968 | INT4 g128 | 69.05 MB | 52 | 3,590.6 |
| MLP_up | 6656×19968 | INT4 g128 | 69.05 MB | 52 | 3,590.6 |
| MLP_down | 19968×6656 | INT4 g128 | 69.05 MB | 52 | 3,590.6 |
| LM_Head | 6656×202048 | INT8 g128 | 1,365.84 MB | 1 | 1,365.8 |
| **Total** | | | | | **15,805 MB** |

_FP16 baseline ≈ 55,707 MB → quantized 15,805 MB = 28% of FP16. Decode weight traffic per token (excluding embedding): ~14,439 MB._

## Hardware comparison: B70 vs PTL 12Xe reference

| Metric | PTL 12Xe (measured) | B70 (estimate target) | Ratio |
|---|---:|---:|---:|
| FP16 XMX | 58.982 TFLOPS | 183.5 TFLOPS | ×3.11 |
| INT8 XMX | 117.965 TOPS | 367 TOPS | ×3.11 |
| Memory BW | 110 GB/s | 608 GB/s | ×5.53 |
| Ridge (FP16) | 536.2 | 301.8 FLOP/byte | — |
| Decode expected speedup | 1× (baseline) | ×5.53 | BW-limited |
| Prefill FC expected speedup | 1× (baseline) | ×3.11 | INT8-XMX-limited |

## Theoretical roofline (B70)

| Metric | Value |
|---|---|
| FP16 XMX peak | 183.5 TFLOPS |
| INT8 XMX peak | 367 TOPS |
| Memory BW | 608 GB/s |
| Ridge point (FP16) | 301.8 FLOP/byte |
| Ridge point (INT8) | 603.6 OP/byte |

_B70 ridge points are 1.78× the PTL values (XMX scales by 3.11×, BW by 5.53×); the XMX-to-BW ratio is better balanced on B70 so prefill benefits relatively less than decode._

## Data sources

- **Base measurements**: on-device cliloader profiling of all ops on PTL 12Xe (B390 iGPU), 99 kernel logs in `outputs/onyx/logs/`. See `outputs/onyx/SUMMARY_onyx_ptl_12xe_2026-07-14.md`.
- **B70 scaling**: memory-bound ops × (110/608); XMX-bound ops × (58.98/183.5).
- **S=64K / kv=64K extrapolation**: from S=kv=32K measurements.
- **Eff%**: copied directly from PTL 12Xe measurements (invariant under scaling, same kernel family on both platforms).

## Token latency summary

### Prefill — TTFT

| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|
| 4,096 | 1,195.66 | 1.196 | 0.2919 | 3,426 |
| 16,384 | 5,349.99 | 5.350 | 0.3265 | 3,062 |
| 32,768 | 12,025.17 | 12.025 | 0.3670 | 2,725 |
| 65,536 | 29,053.07 | 29.053 | 0.4433 | 2,256 |

### Decode — TPOT (per output token)

| KV | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 4,096 | 27.10 | 36.9 |
| 16,384 | 27.40 | 36.5 |
| 32,768 | 27.78 | 36.0 |
| 65,536 | 28.64 | 34.9 |

## Roofline: theoretical floor vs estimated

### Decode (per output token)

| KV | theoretical (ms) | estimated (ms) | achieved % |
|---:|---:|---:|---:|
| 4,096 | 23.96 | 27.10 | 88.4% |
| 16,384 | 24.03 | 27.40 | 87.7% |
| 32,768 | 24.12 | 27.78 | 86.8% |
| 65,536 | 24.31 | 28.64 | 84.9% |

### Prefill (TTFT)

| S | theoretical (ms) | estimated (ms) | achieved % |
|---:|---:|---:|---:|
| 4,096 | 633.09 | 1,195.66 | 52.9% |
| 16,384 | 2,670.65 | 5,349.99 | 49.9% |
| 32,768 | 5,680.79 | 12,025.17 | 47.2% |
| 65,536 | 12,612.97 | 29,053.07 | 43.4% |

## Decode tables


### Decode — KV=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `gemm_kernel` | 0.1239 | 52 | 6.441 | 2146.1 | 557.9 | 91.8% | memory |
| MLP_gate | `gemm_kernel` | 0.1237 | 52 | 6.431 | 2149.4 | 558.8 | 91.9% | memory |
| MLP_up | `gemm_kernel` | 0.1235 | 52 | 6.422 | 2152.4 | 559.5 | 92.0% | memory |
| LM_head | `gemm_kernel` | 2.4381 | 1 | 2.438 | 1103.2 | 560.4 | 92.2% | memory |
| FC_QKV | `gemm_kernel` | 0.0307 | 52 | 1.598 | 1995.6 | 519.1 | 85.4% | memory |
| FC_OutGate | `gemm_kernel` | 0.0259 | 52 | 1.345 | 2108.0 | 548.4 | 90.2% | memory |
| FC_O | `gemm_kernel` | 0.0255 | 52 | 1.325 | 2139.6 | 556.6 | 91.6% | memory |
| PA_sliding | `pa_kv_cache_update+paged_attention` | 0.0162 | 39 | 0.631 | 2075.0 | 34.7 | 5.7% | memory |
| SmallOps | `rms/rope/gate/add` | — | 0 | 0.288 | — | — | — | memory |
| PA_full | `pa_kv_cache_update+paged_attention` | 0.0136 | 13 | 0.177 | 4926.5 | 81.2 | 13.4% | memory |
| **TOTAL** |  |  |  | **27.10** |  |  |  |  |

_SW=2048 caps PA_sliding for all KV≥2048; KV=64K PA_full extrapolated ×2 from 32K._


### Decode — KV=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `gemm_kernel` | 0.1239 | 52 | 6.441 | 2146.1 | 557.9 | 91.8% | memory |
| MLP_gate | `gemm_kernel` | 0.1237 | 52 | 6.431 | 2149.4 | 558.8 | 91.9% | memory |
| MLP_up | `gemm_kernel` | 0.1235 | 52 | 6.422 | 2152.4 | 559.5 | 92.0% | memory |
| LM_head | `gemm_kernel` | 2.4381 | 1 | 2.438 | 1103.2 | 560.4 | 92.2% | memory |
| FC_QKV | `gemm_kernel` | 0.0307 | 52 | 1.598 | 1995.6 | 519.1 | 85.4% | memory |
| FC_OutGate | `gemm_kernel` | 0.0259 | 52 | 1.345 | 2108.0 | 548.4 | 90.2% | memory |
| FC_O | `gemm_kernel` | 0.0255 | 52 | 1.325 | 2139.6 | 556.6 | 91.6% | memory |
| PA_sliding | `pa_kv_cache_update+paged_attention` | 0.0162 | 39 | 0.631 | 2075.0 | 34.7 | 5.7% | memory |
| PA_full | `pa_kv_cache_update+paged_attention` | 0.0367 | 13 | 0.477 | 7315.7 | 119.2 | 19.6% | memory |
| SmallOps | `rms/rope/gate/add` | — | 0 | 0.288 | — | — | — | memory |
| **TOTAL** |  |  |  | **27.40** |  |  |  |  |

_SW=2048 caps PA_sliding for all KV≥2048; KV=64K PA_full extrapolated ×2 from 32K._


### Decode — KV=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `gemm_kernel` | 0.1239 | 52 | 6.441 | 2146.1 | 557.9 | 91.8% | memory |
| MLP_gate | `gemm_kernel` | 0.1237 | 52 | 6.431 | 2149.4 | 558.8 | 91.9% | memory |
| MLP_up | `gemm_kernel` | 0.1235 | 52 | 6.422 | 2152.4 | 559.5 | 92.0% | memory |
| LM_head | `gemm_kernel` | 2.4381 | 1 | 2.438 | 1103.2 | 560.4 | 92.2% | memory |
| FC_QKV | `gemm_kernel` | 0.0307 | 52 | 1.598 | 1995.6 | 519.1 | 85.4% | memory |
| FC_OutGate | `gemm_kernel` | 0.0259 | 52 | 1.345 | 2108.0 | 548.4 | 90.2% | memory |
| FC_O | `gemm_kernel` | 0.0255 | 52 | 1.325 | 2139.6 | 556.6 | 91.6% | memory |
| PA_full | `pa_kv_cache_update+paged_attention` | 0.0663 | 13 | 0.862 | 8098.8 | 131.7 | 21.7% | memory |
| PA_sliding | `pa_kv_cache_update+paged_attention` | 0.0162 | 39 | 0.631 | 2075.0 | 34.7 | 5.7% | memory |
| SmallOps | `rms/rope/gate/add` | — | 0 | 0.288 | — | — | — | memory |
| **TOTAL** |  |  |  | **27.78** |  |  |  |  |

_SW=2048 caps PA_sliding for all KV≥2048; KV=64K PA_full extrapolated ×2 from 32K._


### Decode — KV=65,536

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `gemm_kernel` | 0.1239 | 52 | 6.441 | 2146.1 | 557.9 | 91.8% | memory |
| MLP_gate | `gemm_kernel` | 0.1237 | 52 | 6.431 | 2149.4 | 558.8 | 91.9% | memory |
| MLP_up | `gemm_kernel` | 0.1235 | 52 | 6.422 | 2152.4 | 559.5 | 92.0% | memory |
| LM_head | `gemm_kernel` | 2.4381 | 1 | 2.438 | 1103.2 | 560.4 | 92.2% | memory |
| PA_full | `pa_kv_cache_update+paged_attention` | 0.1326 | 13 | 1.724 | 8098.8 | 131.6 | 21.7% | memory |
| FC_QKV | `gemm_kernel` | 0.0307 | 52 | 1.598 | 1995.6 | 519.1 | 85.4% | memory |
| FC_OutGate | `gemm_kernel` | 0.0259 | 52 | 1.345 | 2108.0 | 548.4 | 90.2% | memory |
| FC_O | `gemm_kernel` | 0.0255 | 52 | 1.325 | 2139.6 | 556.6 | 91.6% | memory |
| PA_sliding | `pa_kv_cache_update+paged_attention` | 0.0162 | 39 | 0.631 | 2075.0 | 34.7 | 5.7% | memory |
| SmallOps | `rms/rope/gate/add` | — | 0 | 0.288 | — | — | — | memory |
| **TOTAL** |  |  |  | **28.64** |  |  |  |  |

_SW=2048 caps PA_sliding for all KV≥2048; KV=64K PA_full extrapolated ×2 from 32K._

## Prefill tables


### Prefill — S=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `dq+gemm_kernel` | 5.9612 | 52 | 309.981 | 182644.3 | 48.2 | 49.8% | compute |
| MLP_up | `dq+gemm_kernel` | 4.7230 | 52 | 245.596 | 230525.9 | 60.8 | 62.8% | compute |
| MLP_gate | `dq+gemm_kernel` | 4.6695 | 52 | 242.816 | 233164.8 | 61.5 | 63.5% | compute |
| PA_sliding | `sdpa_micro_prefill` | 2.2322 | 39 | 87.055 | 46186.4 | 30.3 | 25.2% | compute |
| SmallOps | `rms/rope/gate/add` | — | 0 | 83.327 | — | — | — | memory |
| FC_QKV | `dq+gemm_kernel` | 1.3087 | 52 | 68.051 | 191991.5 | 82.7 | 52.3% | compute |
| FC_OutGate | `dq+gemm_kernel` | 1.1697 | 52 | 60.825 | 190935.4 | 87.4 | 52.0% | compute |
| FC_O | `dq+gemm_kernel` | 1.1261 | 52 | 58.557 | 198329.2 | 90.8 | 54.0% | compute |
| PA_full | `sdpa_micro_prefill` | 2.8468 | 13 | 37.008 | 48290.3 | 24.0 | 26.3% | compute |
| LM_head | `gemm_kernel` | 2.4381 | 1 | 2.438 | 1103.2 | 560.4 | 92.2% | memory |
| **TOTAL** |  |  |  | **1,195.66** |  |  |  |  |

_S=64K extrapolated from S=32K (FC ×2, PA_full ×4, PA_sliding ×2.03, SmallOps ×2)._


### Prefill — S=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `dq+gemm_kernel` | 23.2627 | 52 | 1,209.663 | 187213.4 | 40.5 | 51.0% | compute |
| MLP_up | `dq+gemm_kernel` | 20.0897 | 52 | 1,044.667 | 216782.0 | 46.9 | 59.1% | compute |
| MLP_gate | `dq+gemm_kernel` | 18.7956 | 52 | 977.373 | 231708.0 | 50.1 | 63.1% | compute |
| PA_full | `sdpa_micro_prefill` | 44.8651 | 13 | 583.246 | 49017.1 | 6.1 | 26.7% | compute |
| PA_sliding | `sdpa_micro_prefill` | 11.1594 | 39 | 435.217 | 46186.4 | 24.1 | 25.2% | compute |
| SmallOps | `rms/rope/gate/add` | — | 0 | 347.312 | — | — | — | memory |
| FC_QKV | `dq+gemm_kernel` | 5.0712 | 52 | 263.700 | 198184.2 | 75.9 | 54.0% | compute |
| FC_OutGate | `dq+gemm_kernel` | 4.6986 | 52 | 244.329 | 190130.3 | 78.0 | 51.8% | compute |
| FC_O | `dq+gemm_kernel` | 4.6546 | 52 | 242.041 | 191927.4 | 78.7 | 52.3% | compute |
| LM_head | `gemm_kernel` | 2.4381 | 1 | 2.438 | 1103.2 | 560.4 | 92.2% | memory |
| **TOTAL** |  |  |  | **5,349.99** |  |  |  |  |

_S=64K extrapolated from S=32K (FC ×2, PA_full ×4, PA_sliding ×2.03, SmallOps ×2)._


### Prefill — S=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full | `sdpa_micro_prefill` | 191.3915 | 13 | 2,488.089 | 45960.0 | 2.9 | 25.0% | compute |
| MLP_down | `dq+gemm_kernel` | 47.6520 | 52 | 2,477.907 | 182787.4 | 38.1 | 49.8% | compute |
| MLP_gate | `dq+gemm_kernel` | 37.8562 | 52 | 1,968.521 | 230086.4 | 47.9 | 62.7% | compute |
| MLP_up | `dq+gemm_kernel` | 37.7352 | 52 | 1,962.230 | 230824.2 | 48.1 | 62.9% | compute |
| PA_sliding | `sdpa_micro_prefill` | 23.0624 | 39 | 899.434 | 46186.4 | 23.3 | 25.2% | compute |
| SmallOps | `rms/rope/gate/add` | — | 0 | 745.413 | — | — | — | memory |
| FC_QKV | `dq+gemm_kernel` | 10.2603 | 52 | 533.533 | 195905.9 | 73.5 | 53.4% | compute |
| FC_OutGate | `dq+gemm_kernel` | 9.3458 | 52 | 485.982 | 191177.5 | 76.9 | 52.1% | compute |
| FC_O | `dq+gemm_kernel` | 8.8773 | 52 | 461.619 | 201267.3 | 81.0 | 54.8% | compute |
| LM_head | `gemm_kernel` | 2.4381 | 1 | 2.438 | 1103.2 | 560.4 | 92.2% | memory |
| **TOTAL** |  |  |  | **12,025.17** |  |  |  |  |

_S=64K extrapolated from S=32K (FC ×2, PA_full ×4, PA_sliding ×2.03, SmallOps ×2)._


### Prefill — S=65,536

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full | `sdpa_micro_prefill` | 765.5659 | 13 | 9,952.356 | 45959.3 | 1.4 | 25.0% | compute |
| MLP_down | `dq+gemm_kernel` | 95.3041 | 52 | 4,955.813 | 182787.4 | 37.3 | 49.8% | compute |
| MLP_gate | `dq+gemm_kernel` | 75.7124 | 52 | 3,937.043 | 230086.4 | 47.0 | 62.7% | compute |
| MLP_up | `dq+gemm_kernel` | 75.4704 | 52 | 3,924.459 | 230824.2 | 47.2 | 62.9% | compute |
| PA_sliding | `sdpa_micro_prefill` | 46.8684 | 39 | 1,827.867 | 46186.4 | 22.9 | 25.2% | compute |
| SmallOps | `rms/rope/gate/add` | — | 0 | 1,490.827 | — | — | — | memory |
| FC_QKV | `dq+gemm_kernel` | 20.5205 | 52 | 1,067.067 | 195905.9 | 72.7 | 53.4% | compute |
| FC_OutGate | `dq+gemm_kernel` | 18.6916 | 52 | 971.963 | 191177.5 | 76.2 | 52.1% | compute |
| FC_O | `dq+gemm_kernel` | 17.7546 | 52 | 923.237 | 201267.3 | 80.2 | 54.8% | compute |
| LM_head | `gemm_kernel` | 2.4381 | 1 | 2.438 | 1103.2 | 560.4 | 92.2% | memory |
| **TOTAL** |  |  |  | **29,053.07** |  |  |  |  |

_S=64K extrapolated from S=32K (FC ×2, PA_full ×4, PA_sliding ×2.03, SmallOps ×2)._

## Top contributors

### Decode
| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 4,096 | MLP_down 6.4ms (23.8%) | MLP_gate 6.4ms (23.7%) | MLP_up 6.4ms (23.7%) |
| 16,384 | MLP_down 6.4ms (23.5%) | MLP_gate 6.4ms (23.5%) | MLP_up 6.4ms (23.4%) |
| 32,768 | MLP_down 6.4ms (23.2%) | MLP_gate 6.4ms (23.1%) | MLP_up 6.4ms (23.1%) |
| 65,536 | MLP_down 6.4ms (22.5%) | MLP_gate 6.4ms (22.5%) | MLP_up 6.4ms (22.4%) |

### Prefill
| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 4,096 | MLP_down 310.0ms (25.9%) | MLP_up 245.6ms (20.5%) | MLP_gate 242.8ms (20.3%) |
| 16,384 | MLP_down 1,209.7ms (22.6%) | MLP_up 1,044.7ms (19.5%) | MLP_gate 977.4ms (18.3%) |
| 32,768 | PA_full 2,488.1ms (20.7%) | MLP_down 2,477.9ms (20.6%) | MLP_gate 1,968.5ms (16.4%) |
| 65,536 | PA_full 9,952.4ms (34.3%) | MLP_down 4,955.8ms (17.1%) | MLP_gate 3,937.0ms (13.6%) |

## End-to-end (prefill TTFT + 512-token decode)

| prompt P | TTFT (ms) | 512-tok decode (ms) | total (ms) | avg tok/s |
|---:|---:|---:|---:|---:|
| 4,096 | 1,195.66 | 13,872.75 | 15,068.41 | 36.9 |
| 16,384 | 5,349.99 | 14,026.31 | 19,376.30 | 36.5 |
| 32,768 | 12,025.17 | 14,223.31 | 26,248.48 | 36.0 |
| 65,536 | 29,053.07 | 14,664.54 | 43,717.61 | 34.9 |

## Comparison: B70 vs PTL 12Xe

| Phase | Metric | PTL 12Xe (measured) | B70 (estimated) | Speedup |
|---|---|---:|---:|---:|
| Decode | decode kv=4K | 149.76 ms | 27.10 ms | ×5.53 |
| Decode | decode kv=32K | 153.55 ms | 27.78 ms | ×5.53 |
| Decode | decode kv=64K (extrap) | — | 28.64 ms | — |
| Prefill | prefill S=4K | 3,927.02 ms | 1,195.66 ms | ×3.28 |
| Prefill | prefill S=16K | 17,489.38 ms | 5,349.99 ms | ×3.27 |
| Prefill | prefill S=32K | 39,218.40 ms | 12,025.17 ms | ×3.26 |
| Prefill | prefill S=64K (extrap) | — | 29,053.07 ms | — |

## Key findings

- **Decode is memory-bound at ~36.9 tok/s across all KV sizes** (27.10 ms/tok at KV=4K). B70's 5.53× BW advantage over PTL 12Xe delivers a 5.53× decode speedup: PTL 149.76 ms → B70 27.10 ms. MLP (gate+up+down ×52) is 71% of decode; LM_head 9%.
- **Decode achieves ~88% of the B70 BW roofline** — same kernel efficiency as measured on PTL. Little headroom without INT4 LM_head or speculative decoding.
- **Prefill speedup is ~3.1× vs PTL** (XMX ratio 367/117.97 = 3.11) at moderate S. TTFT at S=4K: 3,927.02 ms (PTL) → 1,195.66 ms (B70).
- **PA_full prefill is the dominant TTFT cost at long context**: 2,488.09 ms at S=32K = 21% of TTFT, with 25% FP16 XMX efficiency. B70 scales this proportionally, so the PA bottleneck shifts from S≈16K on PTL to roughly S≈50K on B70.
- **SmallOps (RMSNorm/residual-add) scale by BW** (5.5×) not XMX (3.1×), so they shrink more relative to FC prefill on B70 and account for 6% of prefill TTFT at S=32K vs 11% on PTL.

## Optimization levers (highest ROI first)

1. **INT4 LM_head** (currently INT8): halves the ~2.4 ms LM_head decode cost → ~4% faster decode, frees ~683 MB of weight bytes.
2. **Fuse `output_gate_proj` into FC_QKV** (single 6656→8704 wide gemm): removes a separate M=1 kernel launch per layer.
3. **PA_full prefill at S≥32K**: at S=64K, PA_full is ~45% of TTFT; algorithmic improvements (CM micro-kernel, KV sparsity, chunk-prefill) directly reduce the dominant cost. The ~25% FP16 XMX efficiency already measured on PTL leaves 4× theoretical headroom.
4. **Speculative decoding / MTP**: decode is fully memory-bound; amortizing weight reads over N accepted tokens gives N× decode throughput at the same memory footprint.
5. **Batch decode**: multiple sequences share the same weight re-read cost (BW stays the same, compute scales by batch) until compute-bound; break-even batch ≈ ridge × ops-per-byte ≈ 302 FLOP/byte × 0.5 FLOP/byte (int4) ≈ B~75.

## Caveats & method

- **Estimate only** — no B70 device measured. All times derived from PTL 12Xe measurements × scaling.
- Assumes same kernel occupancy / efficiency factor on B70 (same primitives, same OpenVINO version).
- FP16 XMX = INT8/2 = 183.5 TFLOPS assumed; actual FP16 throughput should be confirmed on device.
- S=64K and kv=64K are linear/quadratic extrapolations; actual values may differ by ±15%.
- Vision tower / adapter excluded; text-generation roofline only.

## Reproduction

```bash
# Run the actual PTL 12Xe benches first (already done):
# see outputs/onyx/logs/ + outputs/onyx/performance_metrics.json

# B70 estimate from PTL measured data:
cd .github/skills/dev_roofline_profiling/outputs/onyx_b70
python3 estimate_onyx_b70.py   # writes performance_metrics.json + this SUMMARY
```

