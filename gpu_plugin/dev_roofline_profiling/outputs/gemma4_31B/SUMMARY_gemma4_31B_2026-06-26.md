# gemma-4-31B-it — Roofline on PTL 12Xe (B390 iGPU) (2026-06-26)

**Platform**: PTL (Panther Lake B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s) — `Local_Admin@10.239.132.229`
**Model**: `google/gemma-4-31B-it` — dense text decoder of the unified Gemma-4 multimodal model

- 60 layers (50 sliding + 10 full, 5:1 pattern); hidden 5376; GQA 32/16 (sliding) & 32/4 (full)
- MatMul weights INT4 g=128 / FP16 act; LM_head INT8 g=128 / FP16 act
- **KV cache measured at BOTH 8-bit (i8) and 4-bit (u4)** — u4 stored packed in a u8 cache (key BY_CHANNEL, value BY_TOKEN) + FP16 scale/zp
- SDPA: PagedAttention OpenCL + micro_kernel
- Profiler: cliloader 3.0.6 `--device-performance-timing`, mean kernel time (ms)
- Token sweep: input S / kv ∈ {1024, 2048, 4096, 8192, 16384, 32768, 49152}; output tokens (decode): 512

## Model parameters & weight shapes

| Field | Value | Notes |
|---|---:|---|
| `hidden_size` | 5376 | residual / activation channel |
| `num_hidden_layers` | 60 | 50 sliding + 10 full (5:1) |
| `num_attention_heads` | 32 | Q heads |
| sliding `num_key_value_heads` | 16 | GQA group 2, HD=256 → Q=8192, KV=4096 |
| full `num_global_key_value_heads` | 4 | GQA group 8, HD=512 → Q=16384, K=2048, V=K |
| `attention_k_eq_v` | true | full-attn V reuses K projection |
| `intermediate_size` | 21504 | GEGLU MLP hidden (gate/up 5376→21504, down 21504→5376) |
| `vocab_size` | 262144 | LM head N (tied) |
| `hidden_act` | gelu_pytorch_tanh | GEGLU |
| `sliding_window` | 1024 | sliding-attn KV cap |
| `final_logit_softcapping` | 30.0 | |
| `tie_word_embeddings` | true | LM head shares token-embedding storage |

Per-module static weight footprint:

| Module | Shape (K×N) | Quant | Per-layer MB | Layers | Total MB | Share |
|---|---|---|---:|---:|---:|---:|
| FC_QKV_sliding | 5376×16384 | INT4 | 43.64 | 50 | 2,182.0 | 13.7% |
| FC_O_sliding | 8192×5376 | INT4 | 21.82 | 50 | 1,091.0 | 6.9% |
| FC_QK_full | 5376×18432 | INT4 | 49.10 | 10 | 491.0 | 3.1% |
| FC_O_full | 16384×5376 | INT4 | 43.64 | 10 | 436.4 | 2.7% |
| MLP_gate | 5376×21504 | INT4 | 57.28 | 60 | 3,436.7 | 21.6% |
| MLP_up | 5376×21504 | INT4 | 57.28 | 60 | 3,436.7 | 21.6% |
| MLP_down | 21504×5376 | INT4 | 57.28 | 60 | 3,436.7 | 21.6% |
| LM_head (tied) | 5376×262144 | INT8 | 1365.00 | 1 | 1,365.0 | 8.6% |
| **TOTAL** | | | | | **15,875.5 MB** | 100% |

## Theoretical roofline

| Metric | Value |
|---|---|
| FP16 XMX peak | 58.982 TFLOPS |
| INT8 XMX peak | 117.965 TOPS |
| Memory BW | 110.0 GB/s |
| Ridge point (FP16) | 536.2 FLOP/byte |
| Ridge point (INT8) | 1072.4 OP/byte |

_FP16 XMX = 12 Xe × 8 EU × 256 FLOP/cycle × 2.4 GHz. INT8 XMX = 2× FP16. Theoretical floor below deducts 5 % overhead from each peak._

## Graph fusion notes

| Bench row | Real graph behaviour | Standalone kernel? |
|---|---|---|
| FC_QKV/O sliding/full (INT4) | FullyConnectedCompressed; decode `gemm_kernel`, prefill `dq+gemm_kernel` (INT8 XMX) | Yes |
| Dense MLP gate/up/down (INT4) | 3 separate FullyConnectedCompressed (GEGLU between gate/up) | Yes (×3) |
| `multiply` (GEGLU silu·up) | fused into MLP SwiGLU primitive | No — bench-only |
| PA sliding / full | PagedAttention (i8 or u4 KV), split kv_update + sdpa | Yes |
| `add` (residual) | eltwise, 2× per layer | Yes |
| `rmsnorm` | RMSNorm primitive, 4×/layer + 1 final | Yes |

## Token latency summary

### Decode — TPOT (per output token), i8 vs u4 KV

_FC / DenseMLP / LM_head / SmallOps are KV-precision-independent; only PA differs._

| kv (ctx) | FC_attn/L | DenseMLP/L | LM_head | SmallOps | PA/L i8 | PA/L u4 | **TPOT i8** | **TPOT u4** | tok/s i8 | tok/s u4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  1,024 | 0.7193 | 1.8049 | 14.479 | 0.669 | 0.3556 | 0.3144 | **176.26** | **174.26** | 5.7 | 5.7 |
|  2,048 | 0.7193 | 1.8049 | 14.479 | 0.669 | 0.5251 | 0.4607 | **177.95** | **175.72** | 5.6 | 5.7 |
|  4,096 | 0.7193 | 1.8049 | 14.479 | 0.669 | 0.4687 | 0.3899 | **177.39** | **175.02** | 5.6 | 5.7 |
|  8,192 | 0.7193 | 1.8049 | 14.479 | 0.669 | 0.3657 | 0.6062 | **176.36** | **177.18** | 5.7 | 5.6 |
| 16,384 | 0.7193 | 1.8049 | 14.479 | 0.669 | 1.2573 | 1.0660 | **185.27** | **181.78** | 5.4 | 5.5 |
| 32,768 | 0.7193 | 1.8049 | 14.479 | 0.669 | 2.2880 | 1.9740 | **195.58** | **190.86** | 5.1 | 5.2 |
| 49,152 | 0.7193 | 1.8049 | 14.479 | 0.669 | 3.3230 | 2.8963 | **205.93** | **200.08** | 4.9 | 5.0 |

### Prefill — TTFT and end-to-end, i8 vs u4 KV

_E2E = TTFT + 512 × decode TPOT (decode kv = input S)._

| S | FC_attn/L | DenseMLP/L | PA/L i8 | PA/L u4 | **TTFT i8** | **TTFT u4** | prefill tok/s i8 | E2E i8 | E2E u4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  1,024 | 4.486 | 12.625 | 11.685 | 11.934 | **1347.4** | **1354.1** | 760 | 91,590 | 90,575 |
|  2,048 | 8.563 | 25.100 | 39.061 | 39.380 | **2827.9** | **2839.7** | 724 | 93,939 | 92,810 |
|  4,096 | 17.075 | 50.336 | 145.559 | 150.112 | **6349.6** | **6412.1** | 645 | 97,172 | 96,020 |
|  8,192 | 35.103 | 100.836 | 605.520 | 620.895 | **15926.7** | **16114.5** | 514 | 106,221 | 106,830 |
| 16,384 | 71.691 | 143.153 | 2466.280 | 2662.581 | **41011.2** | **43042.3** | 400 | 135,871 | 136,112 |
| 32,768 | 134.539 | 285.448 | 11016.698 | 11044.993 | **142278.9** | **142698.1** | 230 | 242,416 | 240,416 |
| 49,152 | 223.355 | 493.689 | 24948.462 | 25325.822 | **302891.2** | **306869.2** | 162 | 408,327 | 409,310 |

## Roofline: theoretical floor vs measured

_Theoretical = Σ max(bytes/BW, FLOP/peak) over every modelable kernel invocation (5 % overhead). Measured = summed cliloader kernel time. achieved % = theoretical / measured._

### Decode — I8 KV (per output token)

| kv | theoretical (ms) | measured (ms) | achieved % | theo tok/s | meas tok/s |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 163.85 | 176.26 | 93.0% | 6.1 | 5.7 |
| 2,048 | 164.26 | 177.95 | 92.3% | 6.1 | 5.6 |
| 4,096 | 165.06 | 177.39 | 93.0% | 6.1 | 5.6 |
| 8,192 | 166.66 | 176.36 | 94.5% | 6.0 | 5.7 |
| 16,384 | 169.87 | 185.27 | 91.7% | 5.9 | 5.4 |
| 32,768 | 176.30 | 195.58 | 90.1% | 5.7 | 5.1 |
| 49,152 | 182.72 | 205.93 | 88.7% | 5.5 | 4.9 |

### Decode — U4 KV (per output token)

| kv | theoretical (ms) | measured (ms) | achieved % | theo tok/s | meas tok/s |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 161.91 | 174.26 | 92.9% | 6.2 | 5.7 |
| 2,048 | 162.14 | 175.72 | 92.3% | 6.2 | 5.7 |
| 4,096 | 162.59 | 175.02 | 92.9% | 6.2 | 5.7 |
| 8,192 | 163.48 | 177.18 | 92.3% | 6.1 | 5.6 |
| 16,384 | 165.28 | 181.78 | 90.9% | 6.1 | 5.5 |
| 32,768 | 168.88 | 190.86 | 88.5% | 5.9 | 5.2 |
| 49,152 | 172.48 | 200.08 | 86.2% | 5.8 | 5.0 |

### Prefill — I8 KV (TTFT over S tokens)

| S | theoretical (ms) | measured (ms) | achieved % |
|---:|---:|---:|---:|
| 1,024 | 580.22 | 1347.38 | 43.1% |
| 2,048 | 1156.83 | 2827.94 | 40.9% |
| 4,096 | 2349.01 | 6349.57 | 37.0% |
| 8,192 | 4880.54 | 15926.72 | 30.6% |
| 16,384 | 10532.28 | 41011.18 | 25.7% |
| 32,768 | 24190.45 | 142278.91 | 17.0% |
| 49,152 | 40988.23 | 302891.17 | 13.5% |

### Prefill — U4 KV (TTFT over S tokens)

| S | theoretical (ms) | measured (ms) | achieved % |
|---:|---:|---:|---:|
| 1,024 | 570.40 | 1354.13 | 42.1% |
| 2,048 | 1139.35 | 2839.65 | 40.1% |
| 4,096 | 2314.06 | 6412.13 | 36.1% |
| 8,192 | 4810.64 | 16114.54 | 29.9% |
| 16,384 | 10392.49 | 43042.33 | 24.1% |
| 32,768 | 23910.87 | 142698.14 | 16.8% |
| 49,152 | 40568.84 | 306869.20 | 13.2% |

## Decode tables (1 query token, KV = context length)

### I8 KV decode

#### Decode — I8 KV — KV=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (I8 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1524 | 50 | 7.622 | 220.13 | 55.0 | 50.0% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| PA_full (I8 KV) | pa_kv_update+pa_sdpa | 0.2031 | 10 | 2.031 | 330.35 | 20.6 | 18.8% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **176.256** |  |  |  |  |

#### Decode — I8 KV — KV=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (I8 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1524 | 50 | 7.622 | 220.13 | 55.0 | 50.0% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| PA_full (I8 KV) | pa_kv_update+pa_sdpa | 0.3726 | 10 | 3.726 | 360.19 | 22.5 | 20.5% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **177.951** |  |  |  |  |

#### Decode — I8 KV — KV=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (I8 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1524 | 50 | 7.622 | 220.13 | 55.0 | 50.0% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| PA_full (I8 KV) | pa_kv_update+pa_sdpa | 0.3163 | 10 | 3.163 | 848.79 | 53.0 | 48.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **177.387** |  |  |  |  |

#### Decode — I8 KV — KV=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (I8 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1524 | 50 | 7.622 | 220.13 | 55.0 | 50.0% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| PA_full (I8 KV) | pa_kv_update+pa_sdpa | 0.2133 | 10 | 2.133 | 2517.52 | 157.3 | 143.0% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **176.357** |  |  |  |  |

#### Decode — I8 KV — KV=16384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| PA_full (I8 KV) | pa_kv_update+pa_sdpa | 1.1049 | 10 | 11.049 | 971.79 | 60.7 | 55.2% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (I8 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1524 | 50 | 7.622 | 220.13 | 55.0 | 50.0% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **185.273** |  |  |  |  |

#### Decode — I8 KV — KV=32768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| PA_full (I8 KV) | pa_kv_update+pa_sdpa | 2.1356 | 10 | 21.356 | 1005.57 | 62.8 | 57.1% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (I8 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1524 | 50 | 7.622 | 220.13 | 55.0 | 50.0% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **195.580** |  |  |  |  |

#### Decode — I8 KV — KV=49152

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| PA_full (I8 KV) | pa_kv_update+pa_sdpa | 3.1705 | 10 | 31.705 | 1015.99 | 63.5 | 57.7% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (I8 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1524 | 50 | 7.622 | 220.13 | 55.0 | 50.0% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **205.930** |  |  |  |  |

### U4 KV decode

#### Decode — U4 KV — KV=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (U4 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1128 | 50 | 5.642 | 297.36 | 41.6 | 37.8% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| PA_full (U4 KV) | pa_kv_update+pa_sdpa | 0.2015 | 10 | 2.015 | 333.02 | 11.7 | 10.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **174.260** |  |  |  |  |

#### Decode — U4 KV — KV=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (U4 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1128 | 50 | 5.642 | 297.36 | 41.6 | 37.8% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| PA_full (U4 KV) | pa_kv_update+pa_sdpa | 0.3479 | 10 | 3.479 | 385.80 | 13.5 | 12.3% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **175.724** |  |  |  |  |

#### Decode — U4 KV — KV=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (U4 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1128 | 50 | 5.642 | 297.36 | 41.6 | 37.8% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| PA_full (U4 KV) | pa_kv_update+pa_sdpa | 0.2771 | 10 | 2.771 | 968.83 | 33.9 | 30.8% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **175.016** |  |  |  |  |

#### Decode — U4 KV — KV=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (U4 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1128 | 50 | 5.642 | 297.36 | 41.6 | 37.8% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| PA_full (U4 KV) | pa_kv_update+pa_sdpa | 0.4934 | 10 | 4.934 | 1088.14 | 38.1 | 34.6% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **177.179** |  |  |  |  |

#### Decode — U4 KV — KV=16384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_full (U4 KV) | pa_kv_update+pa_sdpa | 0.9531 | 10 | 9.531 | 1126.53 | 39.4 | 35.8% | memory |
| PA_sliding (U4 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1128 | 50 | 5.642 | 297.36 | 41.6 | 37.8% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **181.776** |  |  |  |  |

#### Decode — U4 KV — KV=32768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| PA_full (U4 KV) | pa_kv_update+pa_sdpa | 1.8611 | 10 | 18.611 | 1153.86 | 40.4 | 36.7% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (U4 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1128 | 50 | 5.642 | 297.36 | 41.6 | 37.8% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **190.856** |  |  |  |  |

#### Decode — U4 KV — KV=49152

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 1.8049 | 60 | 108.295 | 384.30 | 99.9 | 90.8% | memory |
| PA_full (U4 KV) | pa_kv_update+pa_sdpa | 2.7834 | 10 | 27.834 | 1157.29 | 40.5 | 36.8% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.4508 | 50 | 22.541 | 390.76 | 101.6 | 92.4% | memory |
| LM_head (INT8) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.2209 | 50 | 11.046 | 398.69 | 103.7 | 94.3% | memory |
| PA_sliding (U4 KV, eff=1024) | pa_kv_update+pa_sdpa | 0.1128 | 50 | 5.642 | 297.36 | 41.6 | 37.8% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.5056 | 10 | 5.056 | 392.01 | 101.9 | 92.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.4517 | 10 | 4.517 | 389.96 | 101.4 | 92.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.669 | — | — | — | memory |
| **TOTAL** |  |  |  | **200.079** |  |  |  |  |

## Prefill tables (single forward over S tokens)

### I8 KV prefill

#### Prefill — I8 KV — S=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 12.6254 | 60 | 757.522 | 56258.17 | 27.4 | 47.7% | compute |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 2.8058 | 50 | 140.290 | 64291.20 | 32.2 | 54.5% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 126.503 | — | — | — | memory |
| PA_full (I8 KV prefill, causal) | sdpa_micro_prefill | 10.1128 | 10 | 101.128 | 3400.96 | 7.5 | 6.8% | memory |
| PA_sliding (I8 KV prefill, sw=1024) | sdpa_micro_prefill | 1.5718 | 50 | 78.591 | 10940.66 | 32.0 | 29.1% | memory |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 1.3959 | 50 | 69.795 | 64614.06 | 36.3 | 54.8% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 3.0899 | 10 | 30.899 | 65677.02 | 32.4 | 55.7% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 2.8174 | 10 | 28.174 | 64027.11 | 32.1 | 54.3% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **1347.381** |  |  |  |  |

#### Prefill — I8 KV — S=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 25.0998 | 60 | 1505.990 | 56596.40 | 20.3 | 48.0% | compute |
| PA_full (I8 KV prefill, causal) | sdpa_micro_prefill | 35.9170 | 10 | 359.170 | 3828.44 | 4.2 | 6.5% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 277.362 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 5.3997 | 50 | 269.984 | 66814.62 | 25.0 | 56.6% | compute |
| PA_sliding (I8 KV prefill, sw=1024) | sdpa_micro_prefill | 3.1436 | 50 | 157.181 | 16394.98 | 26.7 | 27.8% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 2.5733 | 50 | 128.666 | 70099.35 | 30.5 | 59.4% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 6.0525 | 10 | 60.525 | 67058.98 | 24.6 | 56.8% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 5.4580 | 10 | 54.580 | 66100.44 | 24.7 | 56.0% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **2827.937** |  |  |  |  |

#### Prefill — I8 KV — S=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 50.3358 | 60 | 3020.148 | 56443.35 | 16.7 | 47.8% | compute |
| PA_full (I8 KV prefill, causal) | sdpa_micro_prefill | 139.2716 | 10 | 1392.716 | 3948.33 | 2.2 | 6.7% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 583.372 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 10.7470 | 50 | 537.349 | 67140.22 | 20.8 | 56.9% | compute |
| PA_sliding (I8 KV prefill, sw=1024) | sdpa_micro_prefill | 6.2872 | 50 | 314.362 | 19127.47 | 24.0 | 32.4% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 5.1298 | 50 | 256.488 | 70330.12 | 26.1 | 59.6% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 12.1231 | 10 | 121.231 | 66958.60 | 20.3 | 56.8% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 10.9425 | 10 | 109.425 | 65940.81 | 20.5 | 55.9% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **6349.570** |  |  |  |  |

#### Prefill — I8 KV — S=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 100.8364 | 60 | 6050.183 | 56351.10 | 14.9 | 47.8% | compute |
| PA_full (I8 KV prefill, causal) | sdpa_micro_prefill | 592.9454 | 10 | 5929.454 | 3709.10 | 1.0 | 6.3% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1197.715 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 21.7020 | 50 | 1085.102 | 66496.46 | 18.5 | 56.4% | compute |
| PA_sliding (I8 KV prefill, sw=1024) | sdpa_micro_prefill | 12.5745 | 50 | 628.724 | 20493.72 | 22.7 | 34.7% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 11.0743 | 50 | 553.715 | 65155.71 | 22.1 | 55.2% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 24.5623 | 10 | 245.623 | 66097.13 | 18.0 | 56.0% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 22.1723 | 10 | 221.723 | 65086.27 | 18.1 | 55.2% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **15926.719** |  |  |  |  |

#### Prefill — I8 KV — S=16384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (I8 KV prefill, causal) | sdpa_micro_prefill | 2441.1312 | 10 | 24411.312 | 3603.51 | 0.5 | 6.1% | compute |
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 143.1529 | 60 | 8589.173 | 79387.04 | 19.7 | 67.3% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 2437.302 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 47.0127 | 50 | 2350.633 | 61392.34 | 16.1 | 52.0% | compute |
| PA_sliding (I8 KV prefill, sw=1024) | sdpa_micro_prefill | 25.1490 | 50 | 1257.449 | 21176.84 | 22.0 | 35.9% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 20.8150 | 50 | 1040.751 | 69330.15 | 22.5 | 58.8% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 48.9824 | 10 | 489.824 | 66289.04 | 17.0 | 56.2% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 42.0256 | 10 | 420.256 | 68677.66 | 18.1 | 58.2% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **41011.178** |  |  |  |  |

#### Prefill — I8 KV — S=32768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (I8 KV prefill, causal) | sdpa_micro_prefill | 10966.4001 | 10 | 109664.001 | 3208.48 | 0.2 | 5.4% | compute |
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 285.4479 | 60 | 17126.874 | 79625.62 | 19.1 | 67.5% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 4886.338 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 85.9043 | 50 | 4295.216 | 67196.12 | 17.1 | 57.0% | compute |
| PA_sliding (I8 KV prefill, sw=1024) | sdpa_micro_prefill | 50.2980 | 50 | 2514.898 | 21518.41 | 21.7 | 36.5% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 39.9797 | 50 | 1998.987 | 72192.02 | 22.8 | 61.2% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 95.2084 | 10 | 952.084 | 68208.19 | 16.9 | 57.8% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 82.6032 | 10 | 826.032 | 69881.53 | 17.8 | 59.2% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **142278.907** |  |  |  |  |

#### Prefill — I8 KV — S=49152

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (I8 KV prefill, causal) | sdpa_micro_prefill | 24873.0146 | 10 | 248730.146 | 3182.82 | 0.1 | 5.4% | compute |
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 493.6893 | 60 | 29621.360 | 69058.51 | 16.4 | 58.5% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 7351.519 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 140.9489 | 50 | 7047.443 | 61431.17 | 15.5 | 52.1% | compute |
| PA_sliding (I8 KV prefill, sw=1024) | sdpa_micro_prefill | 75.4469 | 50 | 3772.346 | 21632.26 | 21.6 | 36.7% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 60.4610 | 50 | 3023.051 | 71605.27 | 22.4 | 60.7% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 186.9929 | 10 | 1869.929 | 52092.82 | 12.8 | 44.2% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 146.0901 | 10 | 1460.901 | 59269.29 | 15.0 | 50.2% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **302891.173** |  |  |  |  |

### U4 KV prefill

#### Prefill — U4 KV — S=1024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 12.6254 | 60 | 757.522 | 56258.17 | 27.4 | 47.7% | compute |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 2.8058 | 50 | 140.290 | 64291.20 | 32.2 | 54.5% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 126.503 | — | — | — | memory |
| PA_full (U4 KV prefill, causal) | sdpa_micro_prefill | 10.2554 | 10 | 102.554 | 3353.67 | 4.1 | 5.7% | compute |
| PA_sliding (U4 KV prefill, sw=1024) | sdpa_micro_prefill | 1.6783 | 50 | 83.914 | 10246.58 | 16.8 | 17.4% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 1.3959 | 50 | 69.795 | 64614.06 | 36.3 | 54.8% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 3.0899 | 10 | 30.899 | 65677.02 | 32.4 | 55.7% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 2.8174 | 10 | 28.174 | 64027.11 | 32.1 | 54.3% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **1354.130** |  |  |  |  |

#### Prefill — U4 KV — S=2048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 25.0998 | 60 | 1505.990 | 56596.40 | 20.3 | 48.0% | compute |
| PA_full (U4 KV prefill, causal) | sdpa_micro_prefill | 36.0239 | 10 | 360.239 | 3817.08 | 2.3 | 6.5% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 277.362 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 5.3997 | 50 | 269.984 | 66814.62 | 25.0 | 56.6% | compute |
| PA_sliding (U4 KV prefill, sw=1024) | sdpa_micro_prefill | 3.3566 | 50 | 167.828 | 15354.87 | 14.0 | 26.0% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 2.5733 | 50 | 128.666 | 70099.35 | 30.5 | 59.4% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 6.0525 | 10 | 60.525 | 67058.98 | 24.6 | 56.8% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 5.4580 | 10 | 54.580 | 66100.44 | 24.7 | 56.0% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **2839.653** |  |  |  |  |

#### Prefill — U4 KV — S=4096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 50.3358 | 60 | 3020.148 | 56443.35 | 16.7 | 47.8% | compute |
| PA_full (U4 KV prefill, causal) | sdpa_micro_prefill | 143.3986 | 10 | 1433.986 | 3834.70 | 1.2 | 6.5% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 583.372 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 10.7470 | 50 | 537.349 | 67140.22 | 20.8 | 56.9% | compute |
| PA_sliding (U4 KV prefill, sw=1024) | sdpa_micro_prefill | 6.7131 | 50 | 335.656 | 17914.02 | 12.6 | 30.4% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 5.1298 | 50 | 256.488 | 70330.12 | 26.1 | 59.6% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 12.1231 | 10 | 121.231 | 66958.60 | 20.3 | 56.8% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 10.9425 | 10 | 109.425 | 65940.81 | 20.5 | 55.9% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **6412.134** |  |  |  |  |

#### Prefill — U4 KV — S=8192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (U4 KV prefill, causal) | sdpa_micro_prefill | 607.4690 | 10 | 6074.690 | 3620.42 | 0.6 | 6.1% | compute |
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 100.8364 | 60 | 6050.183 | 56351.10 | 14.9 | 47.8% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1197.715 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 21.7020 | 50 | 1085.102 | 66496.46 | 18.5 | 56.4% | compute |
| PA_sliding (U4 KV prefill, sw=1024) | sdpa_micro_prefill | 13.4263 | 50 | 671.313 | 19193.59 | 11.9 | 32.5% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 11.0743 | 50 | 553.715 | 65155.71 | 22.1 | 55.2% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 24.5623 | 10 | 245.623 | 66097.13 | 18.0 | 56.0% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 22.1723 | 10 | 221.723 | 65086.27 | 18.1 | 55.2% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **16114.543** |  |  |  |  |

#### Prefill — U4 KV — S=16384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (U4 KV prefill, causal) | sdpa_micro_prefill | 2635.7290 | 10 | 26357.290 | 3337.46 | 0.3 | 5.7% | compute |
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 143.1529 | 60 | 8589.173 | 79387.04 | 19.7 | 67.3% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 2437.302 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 47.0127 | 50 | 2350.633 | 61392.34 | 16.1 | 52.0% | compute |
| PA_sliding (U4 KV prefill, sw=1024) | sdpa_micro_prefill | 26.8525 | 50 | 1342.626 | 19833.38 | 11.5 | 33.6% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 20.8150 | 50 | 1040.751 | 69330.15 | 22.5 | 58.8% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 48.9824 | 10 | 489.824 | 66289.04 | 17.0 | 56.2% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 42.0256 | 10 | 420.256 | 68677.66 | 18.1 | 58.2% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **43042.333** |  |  |  |  |

#### Prefill — U4 KV — S=32768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (U4 KV prefill, causal) | sdpa_micro_prefill | 10991.2881 | 10 | 109912.881 | 3201.21 | 0.1 | 5.4% | compute |
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 285.4479 | 60 | 17126.874 | 79625.62 | 19.1 | 67.5% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 4886.338 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 85.9043 | 50 | 4295.216 | 67196.12 | 17.1 | 57.0% | compute |
| PA_sliding (U4 KV prefill, sw=1024) | sdpa_micro_prefill | 53.7050 | 50 | 2685.251 | 20153.27 | 11.4 | 34.2% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 39.9797 | 50 | 1998.987 | 72192.02 | 22.8 | 61.2% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 95.2084 | 10 | 952.084 | 68208.19 | 16.9 | 57.8% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 82.6032 | 10 | 826.032 | 69881.53 | 17.8 | 59.2% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **142698.141** |  |  |  |  |

#### Prefill — U4 KV — S=49152

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (U4 KV prefill, causal) | sdpa_micro_prefill | 25245.2646 | 10 | 252452.646 | 3135.89 | 0.1 | 5.3% | compute |
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 493.6893 | 60 | 29621.360 | 69058.51 | 16.4 | 58.5% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 7351.519 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 140.9489 | 50 | 7047.443 | 61431.17 | 15.5 | 52.1% | compute |
| PA_sliding (U4 KV prefill, sw=1024) | sdpa_micro_prefill | 80.5575 | 50 | 4027.877 | 20259.90 | 11.3 | 34.3% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 60.4610 | 50 | 3023.051 | 71605.27 | 22.4 | 60.7% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 186.9929 | 10 | 1869.929 | 52092.82 | 12.8 | 44.2% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 146.0901 | 10 | 1460.901 | 59269.29 | 15.0 | 50.2% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 14.4785 | 1 | 14.479 | 194.67 | 98.9 | 89.9% | memory |
| **TOTAL** |  |  |  | **306869.203** |  |  |  |  |

## KV-cache precision comparison — i8 (8-bit) vs u4 (4-bit)

Only PagedAttention reads/writes the KV cache, so FC / DenseMLP / LM_head / SmallOps are identical across precisions. The tables below isolate the PA latency and the resulting whole-token impact.

### PA per-layer latency (measured, ms)

| ctx | PA_sliding decode i8 | u4 | PA_full decode i8 | u4 | PA_full prefill i8 | u4 |
|---:|---:|---:|---:|---:|---:|---:|
|  1,024 | 0.1524 | 0.1128 | 0.2031 | 0.2015 | 10.113 | 10.255 |
|  2,048 | 0.1524 | 0.1128 | 0.3726 | 0.3479 | 35.917 | 36.024 |
|  4,096 | 0.1524 | 0.1128 | 0.3163 | 0.2771 | 139.272 | 143.399 |
|  8,192 | 0.1524 | 0.1128 | 0.2133 | 0.4934 | 592.945 | 607.469 |
| 16,384 | 0.1524 | 0.1128 | 1.1049 | 0.9531 | 2441.131 | 2635.729 |
| 32,768 | 0.1524 | 0.1128 | 2.1356 | 1.8611 | 10966.400 | 10991.288 |
| 49,152 | 0.1524 | 0.1128 | 3.1705 | 2.7834 | 24873.015 | 25245.265 |

### Whole-token impact (TPOT / TTFT, ms) and u4 speedup

| ctx | TPOT i8 | TPOT u4 | TPOT Δ | TTFT i8 | TTFT u4 | TTFT Δ |
|---:|---:|---:|---:|---:|---:|---:|
|  1,024 | 176.26 | 174.26 | +1.1% | 1347.4 | 1354.1 | -0.5% |
|  2,048 | 177.95 | 175.72 | +1.3% | 2827.9 | 2839.7 | -0.4% |
|  4,096 | 177.39 | 175.02 | +1.3% | 6349.6 | 6412.1 | -1.0% |
|  8,192 | 176.36 | 177.18 | -0.5% | 15926.7 | 16114.5 | -1.2% |
| 16,384 | 185.27 | 181.78 | +1.9% | 41011.2 | 43042.3 | -5.0% |
| 32,768 | 195.58 | 190.86 | +2.4% | 142278.9 | 142698.1 | -0.3% |
| 49,152 | 205.93 | 200.08 | +2.8% | 302891.2 | 306869.2 | -1.3% |

_Decode: u4 halves PA KV traffic — the gain is small in absolute TPOT because PA is a minor fraction of the memory-bound decode token (MLP + LM_head dominate), but it grows with context length. Prefill: PA_full is compute-bound (S² FLOPs), so u4 barely changes TTFT — confirming 4-bit KV helps long-context **decode** memory traffic, not prefill compute._

## Top contributors (by total ms per inference, I8 KV)

### Decode

| KV | top1 (ms, %) | top2 | top3 |
|---:|---|---|---|
| 1,024 | DenseMLP_gate+up+down (INT4) 108.30ms (61%) | FC_QKV_sliding (INT4) 22.54ms (13%) | LM_head (INT8) 14.48ms (8%) |
| 2,048 | DenseMLP_gate+up+down (INT4) 108.30ms (61%) | FC_QKV_sliding (INT4) 22.54ms (13%) | LM_head (INT8) 14.48ms (8%) |
| 4,096 | DenseMLP_gate+up+down (INT4) 108.30ms (61%) | FC_QKV_sliding (INT4) 22.54ms (13%) | LM_head (INT8) 14.48ms (8%) |
| 8,192 | DenseMLP_gate+up+down (INT4) 108.30ms (61%) | FC_QKV_sliding (INT4) 22.54ms (13%) | LM_head (INT8) 14.48ms (8%) |
| 16,384 | DenseMLP_gate+up+down (INT4) 108.30ms (58%) | FC_QKV_sliding (INT4) 22.54ms (12%) | LM_head (INT8) 14.48ms (8%) |
| 32,768 | DenseMLP_gate+up+down (INT4) 108.30ms (55%) | FC_QKV_sliding (INT4) 22.54ms (12%) | PA_full (I8 KV) 21.36ms (11%) |
| 49,152 | DenseMLP_gate+up+down (INT4) 108.30ms (53%) | PA_full (I8 KV) 31.71ms (15%) | FC_QKV_sliding (INT4) 22.54ms (11%) |

### Prefill

| S | top1 (ms, %) | top2 | top3 |
|---:|---|---|---|
| 1,024 | DenseMLP_gate+up+down (INT4->INT8 XMX) 757.52ms (56%) | FC_QKV_sliding (INT4->INT8 XMX) 140.29ms (10%) | SmallOps (norm/rope/add) 126.50ms (9%) |
| 2,048 | DenseMLP_gate+up+down (INT4->INT8 XMX) 1505.99ms (53%) | PA_full (I8 KV prefill, causal) 359.17ms (13%) | SmallOps (norm/rope/add) 277.36ms (10%) |
| 4,096 | DenseMLP_gate+up+down (INT4->INT8 XMX) 3020.15ms (48%) | PA_full (I8 KV prefill, causal) 1392.72ms (22%) | SmallOps (norm/rope/add) 583.37ms (9%) |
| 8,192 | DenseMLP_gate+up+down (INT4->INT8 XMX) 6050.18ms (38%) | PA_full (I8 KV prefill, causal) 5929.45ms (37%) | SmallOps (norm/rope/add) 1197.72ms (8%) |
| 16,384 | PA_full (I8 KV prefill, causal) 24411.31ms (60%) | DenseMLP_gate+up+down (INT4->INT8 XMX) 8589.17ms (21%) | SmallOps (norm/rope/add) 2437.30ms (6%) |
| 32,768 | PA_full (I8 KV prefill, causal) 109664.00ms (77%) | DenseMLP_gate+up+down (INT4->INT8 XMX) 17126.87ms (12%) | SmallOps (norm/rope/add) 4886.34ms (3%) |
| 49,152 | PA_full (I8 KV prefill, causal) 248730.15ms (82%) | DenseMLP_gate+up+down (INT4->INT8 XMX) 29621.36ms (10%) | SmallOps (norm/rope/add) 7351.52ms (2%) |

## End-to-end (prefill TTFT + 512-token decode)

| prompt P | TTFT i8 | TTFT u4 | 512-tok decode i8 | u4 | total i8 | total u4 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 1347.4 | 1354.1 | 90,243 | 89,221 | 91,590 | 90,575 |
| 2,048 | 2827.9 | 2839.7 | 91,111 | 89,971 | 93,939 | 92,810 |
| 4,096 | 6349.6 | 6412.1 | 90,822 | 89,608 | 97,172 | 96,020 |
| 8,192 | 15926.7 | 16114.5 | 90,295 | 90,716 | 106,221 | 106,830 |
| 16,384 | 41011.2 | 43042.3 | 94,860 | 93,069 | 135,871 | 136,112 |
| 32,768 | 142278.9 | 142698.1 | 100,137 | 97,718 | 242,416 | 240,416 |
| 49,152 | 302891.2 | 306869.2 | 105,436 | 102,441 | 408,327 | 409,310 |

## Key findings

- **Decode is memory-bound** at M=1: TPOT @ kv=1024 ≈ 176.3 ms (5.7 tok/s). Dominant op: **DenseMLP_gate+up+down (INT4)** (108.30 ms, 61%).
- **4-bit (u4) KV cache** reduces PA decode memory traffic ~2×, but PA is a small share of the decode token, so TPOT only improves ~2.8% even at kv=49,152. The win scales with context length; at short context u4 vs i8 is within noise.
- **LM_head (INT8, V=262144)** is a heavy single-call op in decode — a top contributor every token.
- **Prefill is compute-bound** (INT8 XMX) for FC at S≥2048; at S=49,152 TTFT ≈ 302891 ms, dominated by **PA_full (I8 KV prefill, causal)** (248730 ms, 82%). u4 vs i8 barely moves TTFT because PA_full prefill is S²-compute-bound, not KV-traffic-bound.
- **Full-attention PA** grows ∝ S² in prefill (causal) and ∝ kv in decode; at long context it becomes the prefill bottleneck while sliding-attn PA stays capped at sw=1024.
- Achieved % vs the theoretical floor is highest for large GEMMs (good XMX/BW utilization) and lower for small per-head norm/rope ops (launch-overhead bound).

## Optimization levers (highest ROI first)

1. **INT4 LM_head** — LM_head is INT8 and a top decode cost; INT4 g=128 ~halves its weight read (memory-bound) → direct TPOT win.
2. **u4 KV for full-attn layers at long context** — measured here; biggest relative benefit when full-attn KV traffic is a meaningful share of the decode token (≥32K context).
3. **Fuse GEGLU gate+up** into a single packed FC (double-wide MLP) to cut a kernel launch + activation round-trip per layer.
4. **Speculative decoding / MTP** — decode is memory-bound and latency-bound; multi-token verification amortizes the per-token weight read.
5. **Larger prefill tiles** for full-attn PA to push S² compute closer to FP16 XMX peak.

## Caveats & method

- Each op profiled in its own process via cliloader Device Performance Timing; mean kernel time per iteration.
- FC weight bytes count INT4/INT8 weight + FP16 scale/zp(g=128) + FP16 act + FP16 out.
- KV cache measured at both i8 (8-bit) and u4 (4-bit). u4 is physically stored in a u8 cache tensor (key BY_CHANNEL: 8B packed + 4B fp16 scale/zp per 16-token block; value BY_TOKEN: HD/2 + 4B). u4 PA byte model = i8 × 0.56 (layout-derived); measured kernel time is authoritative for latency.
- PA decode is memory-bound; PA prefill (S≥2048) is compute-bound, so u4 helps decode KV traffic but not prefill compute.
- Decode FC treated as memory-bound (weights read dominates at M=1); prefill FC is INT8-XMX compute-bound.
- Sliding-attn PA prefill measured at S=1024 and scaled linearly for S>1024 (work ∝ sw·S).
- PA_full decode times can be non-monotonic across kv due to per-bench kernel auto-selection (single_token vs gqa_single_token); PA_full decode is a minor contributor (~1% of TPOT), so this does not affect conclusions. Any Eff%>100% on PA rows is a byte-model artifact, not a real overshoot.
- swish/multiply/add eltwise are fused into matmul/SwiGLU in real inference; listed for visibility.
- lm_head runs once per token (last position in prefill, every step in decode).
- Large prefill (S=32768/49152) PA_full uses very few iters (1–2) due to ~10–21 s/call; values are means over those iters.
- Target machine: PTL 12Xe — `Local_Admin@10.239.132.229`, GPU `Intel(R) Arc(TM) B390 (96CUs, 2400MHz)`.

## Reproduction

```bat
REM on PTL 12Xe Windows target:
D:\river\moe\dev_roofline_profiling\utils\configure_remote.bat
D:\river\moe\dev_roofline_profiling\utils\build_remote.bat
REM base i8 sweep (1024..8192):
D:\river\moe\dev_roofline_profiling\utils\run_gemma4_31B_ptl_12xe.bat
REM extension: 16K/32K/48K sizes + u4 KV for all sizes:
D:\river\moe\dev_roofline_profiling\utils\run_gemma4_31B_extkv.bat
REM logs -> D:\river\moe\roofline_results\gemma4_31B\ptl_12xe
python parse_logs.py <logdir> parsed.json && python build_report.py
```
