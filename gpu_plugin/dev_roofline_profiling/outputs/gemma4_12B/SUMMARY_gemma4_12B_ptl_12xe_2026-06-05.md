# Gemma4-12B-it (dense) — Roofline on PTL 12Xe (B390 iGPU)

**Date:** 2026-06-05
**Target:** Local_Admin@10.239.132.229 — PTL B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s
**Model:** `google/gemma-4-12B-it` (dense, no MoE) — text decoder of the unified multimodal model
**Config:** INT4 g=128 body + INT8 g=128 LM_head + INT8 KV cache, FP16 activation
**SDPA impl:** PagedAttention OpenCL + micro_kernel (kv_type=i8)
**Profiler:** cliloader 3.0.6 `--device-performance-timing`, mean kernel time (ms)
**Token sweep:** input S / kv ∈ {256, 1024, 2048, 4096, 8192, 16384, 32768}; **output tokens (decode): 512**
**Bench:** `.github/skills/dev_roofline_profiling/utils/{fc,pa,small_ops}_bench`

---

## 1. Hardware Peaks

| Platform | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | Ridge (F16) | Ridge (I8) |
|---|---:|---:|---:|---:|---:|
| PTL (B390 iGPU, 12 Xe @ 2400 MHz) | 110.0 | 58.982 | 117.965 | 536.2 | 1072.4 FLOP/byte |

_FP16 XMX = 12 × 8 × 256 FLOP/cycle × 2.4 GHz. INT8 XMX = 2× FP16._

---

## 2. Model Configuration

| Field | Value |
|---|---|
| `vocab_size` | 262,144 |
| `hidden_size` (H) | 3,840 |
| `num_hidden_layers` | **48** (40 sliding + 8 full, 5:1 pattern) |
| Sliding attn (NH/NKV/HD) | 16/8/256 → Q=4096, KV=2048 |
| Full attn (NH/NKV/HD) | 16/1/512 → Q=8192, K=512, V=K |
| `sliding_window` | 1024 |
| `attention_k_eq_v` | true (full attn: V reuses K projection) |
| Dense MLP `intermediate_size` | 15360 (GEGLU: gate+up 3840→15360, down 15360→3840) |
| `hidden_activation` | gelu_pytorch_tanh (GEGLU) |
| `tie_word_embeddings` | true |
| `final_logit_softcapping` | 30.0 |
| Body weight quant | INT4 g=128 (asymmetric) |
| LM head weight quant | INT8 g=128 |
| KV cache | INT8 |
| Activation dtype | FP16 |

---

## 3. Graph Fusion Notes

| Op variant | GPU primitive | Fused? | Notes |
|---|---|---|---|
| FC_QKV/O sliding/full (INT4) | FullyConnectedCompressed | Partial | decode: gemm_kernel; prefill: dq+gemm_kernel (INT8 XMX) |
| Dense MLP gate/up/down (INT4) | FullyConnectedCompressed×3 | ❌ NOT fused (GEGLU between gate/up) | 3 separate FC kernels per layer |
| PagedAttention sliding | PagedAttention | ✅ Fused | INT8 KV, GQA group=2, sw=1024 |
| PagedAttention full | PagedAttention | ✅ Fused | INT8 KV, GQA group=16, V reuses K |
| GEGLU multiply | gelu(gate)·up | SwiGLU primitive | Fused (bench-only) |
| add (residual) | eltwise | Not fused | 2× per layer |
| rmsnorm | RMSNorm primitive | Not fused | 4× per layer + 1 final |

---

## 4. Decode Performance — Totals

_Decode total for 512 output tokens = per-token ms × 512._

| kv tokens | FC_attn/L (ms) | DenseMLP/L (ms) | PA_sliding/L (ms) | PA_full/L (ms) | LM_head (ms) | SmallOps (ms) | **ms/tok** | **tok/s** | **decode 512 tok (ms)** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|   256 | 0.2589 | 0.9008 | 0.0333 | 0.0421 | 10.102 | 1.427 | **68.86** | **14.5** | **35,258** |
| 1,024 | 0.2589 | 0.9008 | 0.0785 | 0.1012 | 10.102 | 1.427 | **71.15** | **14.1** | **36,427** |
| 2,048 | 0.2589 | 0.9008 | 0.0785 | 0.1721 | 10.102 | 1.427 | **71.71** | **13.9** | **36,717** |
| 4,096 | 0.2589 | 0.9008 | 0.0785 | 0.1494 | 10.102 | 1.427 | **71.53** | **14.0** | **36,624** |
| 8,192 | 0.2589 | 0.9008 | 0.0785 | 0.2946 | 10.102 | 1.427 | **72.69** | **13.8** | **37,219** |
| 16,384 | 0.2589 | 0.9008 | 0.0785 | 0.5248 | 10.102 | 1.427 | **74.53** | **13.4** | **38,162** |
| 32,768 | 0.2589 | 0.9008 | 0.0785 | 1.0430 | 10.102 | 1.427 | **78.68** | **12.7** | **40,284** |

---

## 5. Prefill Performance — Totals (TTFT) + End-to-End Latency

_End-to-end = TTFT + 512 × decode_ms (decode KV = input S, capped to sliding window for sliding layers)._

| S tokens | FC_attn/L (ms) | DenseMLP/L (ms) | PA_sliding/L (ms) | PA_full/L (ms) | LM_head (ms) | SmallOps (ms) | **TTFT (ms)** | **prefill tok/s** | **decode ms/tok @ kv=S** | **E2E 512-out (ms)** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|   256 | 0.596 | 1.865 | 0.112 | 0.796 | 10.102 | 14.988 | **154.1** | **1662** | **68.86** | **35,412** |
| 1,024 | 1.676 | 5.719 | 0.769 | 5.185 | 10.102 | 56.239 | **493.5** | **2075** | **71.15** | **36,920** |
| 2,048 | 3.324 | 11.088 | 1.537 | 18.026 | 10.102 | 128.512 | **1036.1** | **1977** | **71.71** | **37,753** |
| 4,096 | 6.355 | 22.177 | 3.075 | 66.966 | 10.102 | 279.897 | **2318.3** | **1767** | **71.53** | **38,942** |
| 8,192 | 12.754 | 44.248 | 6.150 | 257.827 | 10.102 | 583.506 | **5638.3** | **1453** | **72.69** | **42,857** |
| 16,384 | 26.258 | 91.261 | 12.300 | 1033.582 | 10.102 | 1232.672 | **15644.4** | **1047** | **74.53** | **53,806** |
| 32,768 | 54.124 | 180.550 | 24.599 | 4930.469 | 10.102 | 2460.688 | **54162.9** | **605** | **78.68** | **94,447** |

---

## 6. Theoretical Weight Footprint per Module

_INT4 g=128 weight bytes = N·K/2 + N·(K/g)·2 (FP16 scale) + N·(K/g)/2 (INT4 zp). INT8 g=128 weight bytes = N·K + N·(K/g)·2 (FP16 scale). FP16 baseline shown for reference (= 2·N·K).
Params count excludes scale/zero-point overhead.
_

| Module | Shape (K×N) | Quant | Per-layer params | Per-layer storage (MB) | Layers | **Total params** | **Total storage (MB)** | Share |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FC_QKV_sliding | 3840×8192 | INT4 | 31.46 M | 15.59 | 40 | **1258.3 M** | **623.4** | 9.8% |
| FC_O_sliding | 4096×3840 | INT4 | 15.73 M | 7.79 | 40 | **629.1 M** | **311.7** | 4.9% |
| FC_QK_full | 3840×8704 | INT4 | 33.42 M | 16.56 | 8 | **267.4 M** | **132.5** | 2.1% |
| FC_O_full | 8192×3840 | INT4 | 31.46 M | 15.59 | 8 | **251.7 M** | **124.7** | 2.0% |
| MLP_gate | 3840×15360 | INT4 | 58.98 M | 29.22 | 48 | **2831.2 M** | **1,402.7** | 22.0% |
| MLP_up | 3840×15360 | INT4 | 58.98 M | 29.22 | 48 | **2831.2 M** | **1,402.7** | 22.0% |
| MLP_down | 15360×3840 | INT4 | 58.98 M | 29.22 | 48 | **2831.2 M** | **1,402.7** | 22.0% |
| LM_head (tied) | 3840×262144 | INT8 | 1006.63 M | 975.00 | 1 | **1006.6 M** | **975.0** | 15.3% |
| **TOTAL** | | | | | | **11.91 B** | **6,375.5** | 100% |

_FP16 baseline (no quant): 22,710 MB. Quantized total (6,376 MB) = ~28.1% of FP16 size._

---

## 7. Measured vs Theoretical Roofline

### 7.1 How the theoretical roofline is computed

For each kernel we know:

- **Bytes moved (B)** = weight read + activation read + output write. Weight bytes include INT4/INT8 packed weights plus FP16 scales and (for INT4) packed zero points at group-128 granularity.
- **FLOPs (F)** = 2 · M · K · N for a GEMM with shape (M·K) × (K·N).

The classic roofline model gives the lower-bound execution time of a kernel as:

$$ t_{ideal}\ =\ \max\!\left(\frac{B}{\mathrm{BW}_{peak}},\ \frac{F}{\mathrm{Peak}_{compute}}\right) $$

- $\mathrm{BW}_{peak}$ = 110.0 GB/s (PTL B390 LPDDR5x peak).
- $\mathrm{Peak}_{compute}$ = 58.982 TFLOPS (FP16 XMX) for decode (low arithmetic intensity → memory-bound) and **117.965 TOPS (INT8 XMX)** for prefill FC ops, since OpenVINO uses the INT8-XMX `dynamic_quantize_gpu_opt + gemm_kernel` path for compressed-weight matmul at M>1.
- The ratio decides whether the kernel is **memory-bound** ($B/\mathrm{BW} > F/\mathrm{Peak}$) or **compute-bound** (the converse).

**Overhead factor:** we deduct **5%** from each peak to model unavoidable real-world losses (kernel launch, dispatch, memory-bus contention from activations/scales, sub-optimal tile boundaries, refresh / power excursions, etc.). The numbers used in the tables below are therefore:

- Achievable BW = peak × 0.95 = **104.50 GB/s**
- Achievable FP16 XMX = peak × 0.95 = **56.033 TFLOPS**
- Achievable INT8 XMX = peak × 0.95 = **112.067 TOPS**

$$ t_{theo}\ =\ \max\!\left(\frac{B}{0.95\cdot\mathrm{BW}_{peak}},\ \frac{F}{0.95\cdot\mathrm{Peak}_{compute}}\right) $$

$$ \text{Eff\%}\ =\ \frac{t_{theo}}{t_{meas}}\times 100\% $$

Aggregated full-model latencies are simply the sum of $t_{theo}$ over every kernel invocation in the model (per-layer ms × layer count for body modules, plus the LM_head call).

### 7.2 Per-op decode — measured vs theoretical (1 query token)

_FC and LM_head decode are independent of kv (M=1). PA rows are listed per tested kv._

| Module | Bytes (KB) | FLOPs (M) | Bound | Theo ms | **Meas ms** | Eff% |
|---|---:|---:|---|---:|---:|---:|
| FC_QKV_sliding | 15,983.5 | 62.9 | memory | 0.1566 | **0.1623** | 96.5% |
| FC_O_sliding | 7,995.5 | 31.5 | memory | 0.0783 | **0.0823** | 95.2% |
| FC_QK_full | 16,982.0 | 66.8 | memory | 0.1664 | **0.1716** | 97.0% |
| FC_O_full | 15,983.5 | 62.9 | memory | 0.1566 | **0.1583** | 98.9% |
| MLP_gate | 29,962.5 | 118.0 | memory | 0.2936 | **0.3004** | 97.7% |
| MLP_up | 29,962.5 | 118.0 | memory | 0.2936 | **0.3013** | 97.5% |
| MLP_down | 29,962.5 | 118.0 | memory | 0.2936 | **0.2992** | 98.1% |
| LM_head | 998,919.5 | 2013.3 | memory | 9.7885 | **10.1023** | 96.9% |
| PA_sliding kv=256 (eff=256) | 1,024.0 | 4.19 | memory | 0.0100 | **0.0333** | 30.2% |
| PA_sliding kv=1024 (eff=1024) | 4,096.0 | 16.78 | memory | 0.0401 | **0.0785** | 51.1% |
| PA_sliding kv=2048 (eff=1024) | 4,096.0 | 16.78 | memory | 0.0401 | **0.0785** | 51.1% |
| PA_sliding kv=4096 (eff=1024) | 4,096.0 | 16.78 | memory | 0.0401 | **0.0785** | 51.1% |
| PA_sliding kv=8192 (eff=1024) | 4,096.0 | 16.78 | memory | 0.0401 | **0.0785** | 51.1% |
| PA_sliding kv=16384 (eff=1024) | 4,096.0 | 16.78 | memory | 0.0401 | **0.0785** | 51.1% |
| PA_sliding kv=32768 (eff=1024) | 4,096.0 | 16.78 | memory | 0.0401 | **0.0785** | 51.1% |
| PA_full kv=256 | 256.0 | 8.39 | memory | 0.0025 | **0.0421** | 6.0% |
| PA_full kv=1024 | 1,024.0 | 33.55 | memory | 0.0100 | **0.1012** | 9.9% |
| PA_full kv=2048 | 2,048.0 | 67.11 | memory | 0.0201 | **0.1721** | 11.7% |
| PA_full kv=4096 | 4,096.0 | 134.22 | memory | 0.0401 | **0.1494** | 26.9% |
| PA_full kv=8192 | 8,192.0 | 268.44 | memory | 0.0803 | **0.2946** | 27.3% |
| PA_full kv=16384 | 16,384.0 | 536.87 | memory | 0.1605 | **0.5248** | 30.6% |
| PA_full kv=32768 | 32,768.0 | 1073.74 | memory | 0.3211 | **1.0430** | 30.8% |

### 7.3 Per-op prefill — measured vs theoretical (per S)

_¹ PA_sliding bench only runs up to S=SW=1024; for S>SW the measured ms is scaled linearly from the S=1024 baseline (work ∝ SW·S)._

#### S = 256

| Module | Bytes (MB) | FLOPs (G) | Bound | Theo ms | **Meas ms** | Eff% |
|---|---:|---:|---|---:|---:|---:|
| FC_QKV_sliding | 21.46 | 16.11 | memory | 0.2153 | **0.3525** | 61.1% |
| FC_O_sliding | 11.67 | 8.05 | memory | 0.1171 | **0.2164** | 54.1% |
| FC_QK_full | 22.69 | 17.11 | memory | 0.2276 | **0.3519** | 64.7% |
| FC_O_full | 21.46 | 16.11 | memory | 0.2153 | **0.3816** | 56.4% |
| MLP_gate | 38.60 | 30.20 | memory | 0.3873 | **0.5765** | 67.2% |
| MLP_up | 38.60 | 30.20 | memory | 0.3873 | **0.5759** | 67.3% |
| MLP_down | 38.60 | 30.20 | memory | 0.3873 | **0.7121** | 54.4% |
| PA_sliding | 6.00 | 0.54 | memory | 0.0602 | **0.1121** | 53.7% |
| PA_full | 8.50 | 1.08 | memory | 0.0853 | **0.7960** | 10.7% |

#### S = 1,024

| Module | Bytes (MB) | FLOPs (G) | Bound | Theo ms | **Meas ms** | Eff% |
|---|---:|---:|---|---:|---:|---:|
| FC_QKV_sliding | 39.09 | 64.42 | compute | 0.5749 | **1.0304** | 55.8% |
| FC_O_sliding | 23.29 | 32.21 | compute | 0.2874 | **0.5574** | 51.6% |
| FC_QK_full | 41.06 | 68.45 | compute | 0.6108 | **1.0593** | 57.7% |
| FC_O_full | 39.09 | 64.42 | compute | 0.5749 | **1.0566** | 54.4% |
| MLP_gate | 66.72 | 120.80 | compute | 1.0779 | **1.8245** | 59.1% |
| MLP_up | 66.72 | 120.80 | compute | 1.0779 | **1.8095** | 59.6% |
| MLP_down | 66.72 | 120.80 | compute | 1.0779 | **2.0849** | 51.7% |
| PA_sliding | 24.00 | 8.60 | memory | 0.2408 | **0.7687** | 31.3% |
| PA_full | 34.00 | 17.20 | memory | 0.3412 | **5.1854** | 6.6% |

#### S = 2,048

| Module | Bytes (MB) | FLOPs (G) | Bound | Theo ms | **Meas ms** | Eff% |
|---|---:|---:|---|---:|---:|---:|
| FC_QKV_sliding | 62.59 | 128.85 | compute | 1.1498 | **2.0045** | 57.4% |
| FC_O_sliding | 38.79 | 64.42 | compute | 0.5749 | **1.1445** | 50.2% |
| FC_QK_full | 65.56 | 136.90 | compute | 1.2216 | **2.1954** | 55.6% |
| FC_O_full | 62.59 | 128.85 | compute | 1.1498 | **2.0063** | 57.3% |
| MLP_gate | 104.22 | 241.59 | compute | 2.1558 | **3.5000** | 61.6% |
| MLP_up | 104.22 | 241.59 | compute | 2.1558 | **3.4952** | 61.7% |
| MLP_down | 104.22 | 241.59 | compute | 2.1558 | **4.0932** | 52.7% |
| PA_sliding ¹ | 40.00 | 25.77 | compute | 0.4599 | **1.5375** | 29.9% |
| PA_full | 68.00 | 68.75 | compute | 1.2270 | **18.0258** | 6.8% |

#### S = 4,096

| Module | Bytes (MB) | FLOPs (G) | Bound | Theo ms | **Meas ms** | Eff% |
|---|---:|---:|---|---:|---:|---:|
| FC_QKV_sliding | 109.59 | 257.70 | compute | 2.2995 | **3.9052** | 58.9% |
| FC_O_sliding | 69.79 | 128.85 | compute | 1.1498 | **2.1108** | 54.5% |
| FC_QK_full | 114.56 | 273.80 | compute | 2.4432 | **4.1546** | 58.8% |
| FC_O_full | 109.59 | 257.70 | compute | 2.2995 | **3.8938** | 59.1% |
| MLP_gate | 179.22 | 483.18 | compute | 4.3116 | **6.9749** | 61.8% |
| MLP_up | 179.22 | 483.18 | compute | 4.3116 | **6.9117** | 62.4% |
| MLP_down | 179.22 | 483.18 | compute | 4.3116 | **8.2907** | 52.0% |
| PA_sliding ¹ | 72.00 | 60.13 | compute | 1.0731 | **3.0749** | 34.9% |
| PA_full | 136.00 | 274.95 | compute | 4.9068 | **66.9660** | 7.3% |

#### S = 8,192

| Module | Bytes (MB) | FLOPs (G) | Bound | Theo ms | **Meas ms** | Eff% |
|---|---:|---:|---|---:|---:|---:|
| FC_QKV_sliding | 203.59 | 515.40 | compute | 4.5990 | **7.8472** | 58.6% |
| FC_O_sliding | 131.79 | 257.70 | compute | 2.2995 | **4.2095** | 54.6% |
| FC_QK_full | 212.56 | 547.61 | compute | 4.8865 | **8.3471** | 58.5% |
| FC_O_full | 203.59 | 515.40 | compute | 4.5990 | **7.8914** | 58.3% |
| MLP_gate | 329.22 | 966.37 | compute | 8.6232 | **14.0075** | 61.6% |
| MLP_up | 329.22 | 966.37 | compute | 8.6232 | **13.8023** | 62.5% |
| MLP_down | 329.22 | 966.37 | compute | 8.6232 | **16.4377** | 52.5% |
| PA_sliding ¹ | 136.00 | 128.85 | compute | 2.2995 | **6.1499** | 37.4% |
| PA_full | 272.00 | 1099.65 | compute | 19.6249 | **257.8267** | 7.6% |

#### S = 16,384

| Module | Bytes (MB) | FLOPs (G) | Bound | Theo ms | **Meas ms** | Eff% |
|---|---:|---:|---|---:|---:|---:|
| FC_QKV_sliding | 391.59 | 1030.79 | compute | 9.1980 | **16.1084** | 57.1% |
| FC_O_sliding | 255.79 | 515.40 | compute | 4.5990 | **8.7668** | 52.5% |
| FC_QK_full | 408.56 | 1095.22 | compute | 9.7729 | **17.0551** | 57.3% |
| FC_O_full | 391.59 | 1030.79 | compute | 9.1980 | **16.1186** | 57.1% |
| MLP_gate | 629.22 | 1932.74 | compute | 17.2463 | **28.6295** | 60.2% |
| MLP_up | 629.22 | 1932.74 | compute | 17.2463 | **28.9131** | 59.6% |
| MLP_down | 629.22 | 1932.74 | compute | 17.2463 | **33.7186** | 51.1% |
| PA_sliding ¹ | 264.00 | 266.29 | compute | 4.7523 | **12.2997** | 38.6% |
| PA_full | 544.00 | 4398.31 | compute | 78.4947 | **1033.5823** | 7.6% |

#### S = 32,768

| Module | Bytes (MB) | FLOPs (G) | Bound | Theo ms | **Meas ms** | Eff% |
|---|---:|---:|---|---:|---:|---:|
| FC_QKV_sliding | 767.59 | 2061.58 | compute | 18.3961 | **33.7631** | 54.5% |
| FC_O_sliding | 503.79 | 1030.79 | compute | 9.1980 | **17.9741** | 51.2% |
| FC_QK_full | 800.56 | 2190.43 | compute | 19.5458 | **34.0297** | 57.4% |
| FC_O_full | 767.59 | 2061.58 | compute | 18.3961 | **32.0307** | 57.4% |
| MLP_gate | 1229.22 | 3865.47 | compute | 34.4926 | **56.9283** | 60.6% |
| MLP_up | 1229.22 | 3865.47 | compute | 34.4926 | **57.0621** | 60.4% |
| MLP_down | 1229.22 | 3865.47 | compute | 34.4926 | **66.5598** | 51.8% |
| PA_sliding ¹ | 520.00 | 541.17 | compute | 9.6579 | **24.5994** | 39.3% |
| PA_full | 1088.00 | 17592.72 | compute | 313.9692 | **4930.4689** | 6.4% |

### 7.4 Aggregate — full-model latency across all tested sizes

_Theoretical = sum of per-kernel $t_{theo}$ over every invocation in the model (with the 5 % overhead already applied). Decode is for 1 generated token at the given kv; prefill is TTFT over S tokens; E2E adds 512 generated tokens at the matched kv._

| Phase | Size | **Theoretical ms** | Theoretical tok/s | **Measured ms** | Measured tok/s | Eff% |
|---|---:|---:|---:|---:|---:|---:|
| Decode (1 tok) | kv=256 | **64.47** | 15.5 | **68.86** | 14.5 | 93.6% |
| Decode (1 tok) | kv=1,024 | **65.74** | 15.2 | **71.15** | 14.1 | 92.4% |
| Decode (1 tok) | kv=2,048 | **65.82** | 15.2 | **71.71** | 13.9 | 91.8% |
| Decode (1 tok) | kv=4,096 | **65.98** | 15.2 | **71.53** | 14.0 | 92.2% |
| Decode (1 tok) | kv=8,192 | **66.30** | 15.1 | **72.69** | 13.8 | 91.2% |
| Decode (1 tok) | kv=16,384 | **66.94** | 14.9 | **74.53** | 13.4 | 89.8% |
| Decode (1 tok) | kv=32,768 | **68.22** | 14.7 | **78.68** | 12.7 | 86.7% |
| Prefill TTFT | S=256 | **85.5** | 2994 | **154.1** | 1662 | 55.5% |
| Prefill TTFT | S=1,024 | **221.3** | 4626 | **493.5** | 2075 | 44.9% |
| Prefill TTFT | S=2,048 | **437.3** | 4684 | **1036.1** | 1977 | 42.2% |
| Prefill TTFT | S=4,096 | **884.4** | 4632 | **2318.3** | 1767 | 38.1% |
| Prefill TTFT | S=8,192 | **1837.4** | 4458 | **5638.3** | 1453 | 32.6% |
| Prefill TTFT | S=16,384 | **3979.0** | 4118 | **15644.4** | 1047 | 25.4% |
| Prefill TTFT | S=32,768 | **9204.0** | 3560 | **54162.9** | 605 | 17.0% |

_E2E latency for 512 generated tokens (TTFT + 512 × decode_ms @ kv=S):_

| Size (S = kv₀) | **Theoretical E2E (ms)** | **Measured E2E (ms)** | Eff% |
|---:|---:|---:|---:|
| 256 | **33,095** | **35,412** | 93.5% |
| 1,024 | **33,878** | **36,920** | 91.8% |
| 2,048 | **34,135** | **37,753** | 90.4% |
| 4,096 | **34,665** | **38,942** | 89.0% |
| 8,192 | **35,782** | **42,857** | 83.5% |
| 16,384 | **38,252** | **53,806** | 71.1% |
| 32,768 | **44,135** | **94,447** | 46.7% |

---

## 8. Decode Tables (1 query token, KV = context length)

_Eff% = GB/s / 110 for memory-bound, GFLOPS / XMX_peak for compute-bound._

### Decode — KV=256

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 0.9008 | 48 | 43.240 | 392.85 | 102.2 | 92.9% | memory |
| LM_head (INT8) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1623 | 40 | 6.493 | 387.61 | 100.8 | 91.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0823 | 40 | 3.293 | 382.06 | 99.4 | 90.4% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.427 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1716 | 8 | 1.373 | 389.50 | 101.3 | 92.1% | memory |
| PA_sliding (INT8 KV, eff=256) | paged_attn_single_token | 0.0333 | 40 | 1.331 | 126.08 | 31.5 | 28.7% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1583 | 8 | 1.267 | 397.34 | 103.4 | 94.0% | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 0.0421 | 8 | 0.337 | 199.15 | 6.2 | 5.7% | memory |
| **TOTAL** | | | | **68.863** | | | **14.5 tok/s** | |

### Decode — KV=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 0.9008 | 48 | 43.240 | 392.85 | 102.2 | 92.9% | memory |
| LM_head (INT8) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1623 | 40 | 6.493 | 387.61 | 100.8 | 91.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0823 | 40 | 3.293 | 382.06 | 99.4 | 90.4% | memory |
| PA_sliding (INT8 KV, eff=1024) | paged_attn_single_token | 0.0785 | 40 | 3.141 | 213.66 | 53.4 | 48.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.427 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1716 | 8 | 1.373 | 389.50 | 101.3 | 92.1% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1583 | 8 | 1.267 | 397.34 | 103.4 | 94.0% | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 0.1012 | 8 | 0.810 | 331.53 | 10.4 | 9.4% | memory |
| **TOTAL** | | | | **71.146** | | | **14.1 tok/s** | |

### Decode — KV=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 0.9008 | 48 | 43.240 | 392.85 | 102.2 | 92.9% | memory |
| LM_head (INT8) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1623 | 40 | 6.493 | 387.61 | 100.8 | 91.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0823 | 40 | 3.293 | 382.06 | 99.4 | 90.4% | memory |
| PA_sliding (INT8 KV, eff=1024) | paged_attn_single_token | 0.0785 | 40 | 3.141 | 213.66 | 53.4 | 48.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.427 | — | — | — | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 0.1721 | 8 | 1.377 | 389.95 | 12.2 | 11.1% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1716 | 8 | 1.373 | 389.50 | 101.3 | 92.1% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1583 | 8 | 1.267 | 397.34 | 103.4 | 94.0% | memory |
| **TOTAL** | | | | **71.713** | | | **13.9 tok/s** | |

### Decode — KV=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 0.9008 | 48 | 43.240 | 392.85 | 102.2 | 92.9% | memory |
| LM_head (INT8) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1623 | 40 | 6.493 | 387.61 | 100.8 | 91.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0823 | 40 | 3.293 | 382.06 | 99.4 | 90.4% | memory |
| PA_sliding (INT8 KV, eff=1024) | paged_attn_single_token | 0.0785 | 40 | 3.141 | 213.66 | 53.4 | 48.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.427 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1716 | 8 | 1.373 | 389.50 | 101.3 | 92.1% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1583 | 8 | 1.267 | 397.34 | 103.4 | 94.0% | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 0.1494 | 8 | 1.196 | 898.15 | 28.1 | 25.5% | memory |
| **TOTAL** | | | | **71.532** | | | **14.0 tok/s** | |

### Decode — KV=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 0.9008 | 48 | 43.240 | 392.85 | 102.2 | 92.9% | memory |
| LM_head (INT8) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1623 | 40 | 6.493 | 387.61 | 100.8 | 91.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0823 | 40 | 3.293 | 382.06 | 99.4 | 90.4% | memory |
| PA_sliding (INT8 KV, eff=1024) | paged_attn_single_token | 0.0785 | 40 | 3.141 | 213.66 | 53.4 | 48.6% | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 0.2946 | 8 | 2.357 | 911.26 | 28.5 | 25.9% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.427 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1716 | 8 | 1.373 | 389.50 | 101.3 | 92.1% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1583 | 8 | 1.267 | 397.34 | 103.4 | 94.0% | memory |
| **TOTAL** | | | | **72.693** | | | **13.8 tok/s** | |

### Decode — KV=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 0.9008 | 48 | 43.240 | 392.85 | 102.2 | 92.9% | memory |
| LM_head (INT8) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1623 | 40 | 6.493 | 387.61 | 100.8 | 91.7% | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 0.5248 | 8 | 4.198 | 1022.99 | 32.0 | 29.1% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0823 | 40 | 3.293 | 382.06 | 99.4 | 90.4% | memory |
| PA_sliding (INT8 KV, eff=1024) | paged_attn_single_token | 0.0785 | 40 | 3.141 | 213.66 | 53.4 | 48.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.427 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1716 | 8 | 1.373 | 389.50 | 101.3 | 92.1% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1583 | 8 | 1.267 | 397.34 | 103.4 | 94.0% | memory |
| **TOTAL** | | | | **74.535** | | | **13.4 tok/s** | |

### Decode — KV=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4) | gemm_kernel x3 | 0.9008 | 48 | 43.240 | 392.85 | 102.2 | 92.9% | memory |
| LM_head (INT8) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 1.0430 | 8 | 8.344 | 1029.45 | 32.2 | 29.2% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1623 | 40 | 6.493 | 387.61 | 100.8 | 91.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0823 | 40 | 3.293 | 382.06 | 99.4 | 90.4% | memory |
| PA_sliding (INT8 KV, eff=1024) | paged_attn_single_token | 0.0785 | 40 | 3.141 | 213.66 | 53.4 | 48.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1.427 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1716 | 8 | 1.373 | 389.50 | 101.3 | 92.1% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1583 | 8 | 1.267 | 397.34 | 103.4 | 94.0% | memory |
| **TOTAL** | | | | **78.680** | | | **12.7 tok/s** | |

---

## 9. Prefill Tables (single forward over S tokens)


### Prefill — S=256

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 1.8645 | 48 | 89.496 | 48590.36 | 65.1 | 59.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 14.988 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 0.3525 | 40 | 14.099 | 45693.09 | 63.8 | 58.0% | memory |
| LM_head (INT8, 1 out tok) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 0.2164 | 40 | 8.654 | 37222.22 | 56.6 | 51.4% | memory |
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 0.7960 | 8 | 6.368 | 1354.23 | 11.2 | 10.2% | memory |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 0.1121 | 40 | 4.485 | 4806.85 | 56.1 | 51.0% | memory |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 0.3816 | 8 | 3.053 | 42207.28 | 59.0 | 53.6% | memory |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 0.3519 | 8 | 2.815 | 48636.11 | 67.6 | 61.5% | memory |
| **TOTAL** | | | | **154.1** | | | **TTFT 0.15s, 1662 tok/s** | |

### Prefill — S=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 5.7189 | 48 | 274.507 | 63366.64 | 36.7 | 53.7% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 56.239 | — | — | — | memory |
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 5.1854 | 8 | 41.483 | 3316.36 | 6.9 | 6.3% | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 1.0304 | 40 | 41.215 | 62525.85 | 39.8 | 53.0% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 0.7687 | 40 | 30.749 | 11185.07 | 32.7 | 29.8% | memory |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 0.5574 | 40 | 22.297 | 57786.67 | 43.8 | 49.0% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 1.0593 | 8 | 8.474 | 64619.80 | 40.6 | 54.8% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 1.0566 | 8 | 8.453 | 60972.55 | 38.8 | 51.7% | compute |
| **TOTAL** | | | | **493.5** | | | **TTFT 0.49s, 2075 tok/s** | |

### Prefill — S=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 11.0885 | 48 | 532.248 | 65362.85 | 29.6 | 55.4% | compute |
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 18.0258 | 8 | 144.207 | 3814.14 | 4.0 | 6.5% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 128.512 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 2.0045 | 40 | 80.181 | 64279.01 | 32.7 | 54.5% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 1.5375 | 40 | 61.499 | 16761.24 | 27.3 | 28.4% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 1.1445 | 40 | 45.781 | 56289.20 | 35.5 | 47.7% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 2.1954 | 8 | 17.563 | 62359.74 | 31.3 | 52.9% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 2.0063 | 8 | 16.051 | 64221.25 | 32.7 | 54.4% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| **TOTAL** | | | | **1036.1** | | | **TTFT 1.04s, 1977 tok/s** | |

### Prefill — S=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 22.1774 | 48 | 1064.513 | 65361.78 | 25.4 | 55.4% | compute |
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 66.9660 | 8 | 535.728 | 4105.74 | 2.1 | 7.0% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 279.897 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 3.9052 | 40 | 156.208 | 65988.50 | 29.4 | 55.9% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 3.0749 | 40 | 122.997 | 19554.78 | 24.6 | 33.2% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 2.1108 | 40 | 84.431 | 61043.38 | 34.7 | 51.7% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 4.1546 | 8 | 33.237 | 65903.94 | 28.9 | 55.9% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 3.8938 | 8 | 31.150 | 66181.61 | 29.5 | 56.1% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| **TOTAL** | | | | **2318.3** | | | **TTFT 2.32s, 1767 tok/s** | |

### Prefill — S=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 44.2475 | 48 | 2123.880 | 65520.14 | 23.4 | 55.5% | compute |
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 257.8267 | 8 | 2062.614 | 4265.06 | 1.1 | 7.2% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 583.506 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 7.8472 | 40 | 313.890 | 65678.57 | 27.2 | 55.7% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 6.1499 | 40 | 245.994 | 20951.55 | 23.2 | 35.5% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 4.2095 | 40 | 168.378 | 61218.93 | 32.8 | 51.9% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 8.3471 | 8 | 66.777 | 65604.49 | 26.7 | 55.6% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 7.8914 | 8 | 63.131 | 65311.10 | 27.1 | 55.4% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| **TOTAL** | | | | **5638.3** | | | **TTFT 5.64s, 1453 tok/s** | |

### Prefill — S=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 1033.5823 | 8 | 8268.658 | 4255.41 | 0.6 | 7.2% | compute |
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 91.2612 | 48 | 4380.538 | 63534.18 | 21.7 | 53.9% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1232.672 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 16.1084 | 40 | 644.336 | 63990.98 | 25.5 | 54.2% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 12.2997 | 40 | 491.988 | 21649.94 | 22.5 | 36.7% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 8.7668 | 40 | 350.671 | 58789.73 | 30.6 | 49.8% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 17.0551 | 8 | 136.441 | 64216.41 | 25.1 | 54.4% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 16.1186 | 8 | 128.949 | 63950.46 | 25.5 | 54.2% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| **TOTAL** | | | | **15644.4** | | | **TTFT 15.64s, 1047 tok/s** | |

### Prefill — S=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 4930.4689 | 8 | 39443.751 | 3568.16 | 0.2 | 6.0% | compute |
| DenseMLP_gate+up+down (INT4->INT8 XMX) | dq+gemm_kernel x3 | 180.5502 | 48 | 8666.411 | 64228.17 | 21.4 | 54.4% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 2460.688 | — | — | — | memory |
| FC_QKV_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 33.7631 | 40 | 1350.522 | 61060.35 | 23.8 | 51.8% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 24.5994 | 40 | 983.977 | 21999.13 | 22.2 | 37.3% | compute |
| FC_O_sliding (INT4->INT8 XMX) | dq+gemm_kernel | 17.9741 | 40 | 718.962 | 57348.87 | 29.4 | 48.6% | compute |
| FC_QK_full (INT4->INT8 XMX) | dq+gemm_kernel | 34.0297 | 8 | 272.238 | 64368.27 | 24.7 | 54.6% | compute |
| FC_O_full (INT4->INT8 XMX) | dq+gemm_kernel | 32.0307 | 8 | 256.246 | 64362.71 | 25.1 | 54.6% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 10.1023 | 1 | 10.102 | 199.29 | 101.3 | 92.0% | memory |
| **TOTAL** | | | | **54162.9** | | | **TTFT 54.16s, 605 tok/s** | |

---

## 10. Decode Time-Share (kv=1,024)

**Total decode @ kv=1024: 71.15 ms → 14.1 tok/s**

| Group | ms | Share |
|---|---:|---:|
| DenseMLP_gate+up+down (INT4) ×48 | 43.240 | 60.8% |
| LM_head (INT8) ×1 | 10.102 | 14.2% |
| FC_QKV_sliding (INT4) ×40 | 6.493 | 9.1% |
| FC_O_sliding (INT4) ×40 | 3.293 | 4.6% |
| PA_sliding (INT8 KV, eff=1024) ×40 | 3.141 | 4.4% |
| SmallOps (norm/rope/add)  | 1.427 | 2.0% |
| FC_QK_full (INT4) ×8 | 1.373 | 1.9% |
| FC_O_full (INT4) ×8 | 1.267 | 1.8% |
| PA_full (INT8 KV) ×8 | 0.810 | 1.1% |

---

## 11. Caveats & Method

- Each op profiled in its own process via cliloader Device Performance Timing; mean kernel time per iteration (ms).
- FC weight bytes: INT4 weight + FP16 scale + INT4 zp (g=128) + FP16 act + FP16 out.
- LM_head bytes: INT8 weight + FP16 scale (g=128) + FP16 act + FP16 out.
- PA bytes: INT8 KV cache + FP16 Q / out.
- Decode FC is memory-bound (weight-read dominates at M=1).
- Prefill FC is compute-bound (INT8 XMX path via dynamic_quantize_gpu_opt + gemm_kernel).
- PA prefill FLOPs use causal lower-triangular pairs = S(S+1)/2 (not S²); sliding caps to SW for S>SW.
- PA sliding decode KV caps at sliding_window=1024 → for ctx≥1024 we reuse kv=1024 measurement.
- PA sliding prefill bench uses full causal mask; for S>1024 we scale the S=1024 measurement linearly (sliding work ~ SW·S).
- Full attention uses partial_rotary_factor=0.25 (rope applied to 25% of HD); the bench measures full HD rope (slight over-estimate, immaterial).
- LM head is profiled once (single output token) and reused for prefill & decode.
- All RMSNorm/RoPE/Add bench timings are aggregated using model call-counts (4×NL+1 rms, 2×NL adds, per-layer rope/q-norm/k-norm for sliding vs full).
- Target: Local_Admin@10.239.132.229 (PTL 12Xe, B390 iGPU).

### Optimization levers

1. **Dense MLP (15360 wide)** dominates decode bandwidth — fusing gate+up into a single packed-INT4 weight read (or moving to INT4 XMX decompose-on-the-fly) is the highest lever.
2. **LM_head INT8** is a ~Gigabyte read every token; INT4 g=128 (with the softcap +1 LM_head) would roughly halve its time on decode.
3. **PA full** GQA group=16 + V=K halves the KV traffic vs. independent V proj; ensure prefill uses sdpa_micro_prefill (compute-bound, INT8 KV).
4. **rmsnorm / rope / add** aggregate to a non-trivial decode tax (193 + 96 + per-layer rope/qk-norm); a fused norm+rope+add primitive would cut launches.