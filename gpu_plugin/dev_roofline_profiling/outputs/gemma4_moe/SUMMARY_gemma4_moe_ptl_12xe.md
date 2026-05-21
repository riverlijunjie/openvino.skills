# Gemma4-26B-A4B-it — Roofline on PTL 12Xe (B390 iGPU)

**Date:** 2026-05-12
**Target:** PTL (Panther Lake B390 iGPU, 12 Xe @ 2400 MHz, 110 GB/s)
**Config:** INT4 g=128 body (MoE g=64, I=704 not divisible by 128) + INT8 g=128 LM_head + INT8 KV cache
**Profiler:** cliloader 3.0.6 `--device-performance-timing`, mean kernel time
**Token sweep:** S/kv ∈ {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}
**Bench:** `.github/skills/dev_roofline_profiling/utils/{fc,moe,pa,small_ops}_bench`

---

## 1. Hardware Peaks

| Platform | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | Ridge point (F16) |
|---|---:|---:|---:|---:|
| PTL (B390 iGPU, 12 Xe @ 2400 MHz) | 110.0 | 58.982 | 117.965 | 536.2 FLOP/byte |

---

## 2. Model Configuration

| Field | Value |
|---|---|
| `vocab_size` | 262,144 |
| `hidden_size` (H) | 2,816 |
| `num_hidden_layers` | **30** (25 sliding + 5 full, 5:1 pattern) |
| Sliding attn: NH / NKV / HD | 16 / 8 / 256 (Q=4096, KV=2048) |
| Full attn: NH / NKV / HD | 16 / 2 / 512 (Q=8192, K=1024, V=K) |
| `sliding_window` | 1024 |
| `attention_k_eq_v` | true (full attn: V reuses K projection) |
| Dense MLP `intermediate_size` | 2112 (GEGLU: gate+up 2816→2112, down 2112→2816) |
| MoE: `moe_intermediate_size` | 704 per expert |
| MoE: `num_experts` / `top_k` | 128 / 8 |
| `hidden_activation` | gelu_pytorch_tanh (GEGLU) |
| `tie_word_embeddings` | true |
| `final_logit_softcapping` | 30.0 |
| Body weight quant | INT4 g=128 (asymmetric) |
| LM head weight quant | INT8 g=128 |
| KV cache | INT8 |
| Activation dtype | FP16 |

> **Note**: Dense MLP and MoE run in parallel per layer. Both are profiled.
> GPU plugin MoE fusion only supports Swish/SiLU — bench uses Swish as proxy for GEGLU.

---

## 3. Graph Fusion Notes

| Op variant | GPU primitive | Fused? | Notes |
|---|---|---|---|
| FC_QKV/O sliding/full (INT4) | FullyConnectedCompressed | Partial | decode: gemm_kernel; prefill: dq+gemm_kernel |
| Dense MLP gate/up/down (INT4) | FullyConnectedCompressed×3 | ❌ NOT fused | 3 separate FC ops (gate+up need GEGLU activation between) |
| MoE routed experts (INT4) | MOE3GemmFusedCompressed | ✅ Fused | gate_up+down+scatter/gather; Swish proxy |
| PagedAttention sliding | PagedAttention | ✅ Fused | INT8 KV, GQA group=2, sw=1024 |
| PagedAttention full | PagedAttention | ✅ Fused | INT8 KV, GQA group=8 |
| GEGLU multiply | silu(gate)·up in DenseMLP | SwiGLU primitive | Fused — bench-only |
| add (residual) | eltwise | Not fused | 2× per layer |
| rmsnorm | RMSNorm primitive | Not fused | 4× per layer + 1 final |

---

## 4. Decode Performance — Totals

| kv tokens | FC_attn/L (ms) | DenseMLP/L (ms) | MoE/L (ms) | PA_sliding/L (ms) | PA_full/L (ms) | LM head (ms) | SmallOps (ms) | **total ms** | **tok/s** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|   1,024 | 0.1929 | 0.1047 | 0.2846 | 0.0800 | 0.1061 | 7.303 | 0.861 | **28.16** | **35.5** |
|   2,048 | 0.1929 | 0.1047 | 0.2846 | 0.0800 | 0.1947 | 7.303 | 0.861 | **28.61** | **35.0** |
|   4,096 | 0.1929 | 0.1047 | 0.2846 | 0.0800 | 0.1601 | 7.303 | 0.861 | **28.43** | **35.2** |
|   8,192 | 0.1929 | 0.1047 | 0.2846 | 0.0800 | 0.3039 | 7.303 | 0.861 | **29.15** | **34.3** |
|  16,384 | 0.1929 | 0.1047 | 0.2846 | 0.0800 | 0.5650 | 7.303 | 0.861 | **30.46** | **32.8** |
|  32,768 | 0.1929 | 0.1047 | 0.2846 | 0.0800 | 1.0911 | 7.303 | 0.861 | **33.09** | **30.2** |
|  65,536 | 0.1929 | 0.1047 | 0.2846 | 0.0800 | 2.1144 | 7.303 | 0.861 | **38.20** | **26.2** |
| 131,072 | 0.1929 | 0.1047 | 0.2846 | 0.0800 | 4.1909 | 7.303 | 0.861 | **48.59** | **20.6** |

---

## 5. Prefill Performance — Totals (TTFT)

| S tokens | FC_attn/L (ms) | DenseMLP/L (ms) | MoE/L (ms) | PA_sliding/L (ms) | PA_full/L (ms) | **TTFT (ms)** | **tok/s** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|   1,024 | 1.337 | 0.937 | 11.250 | 0.770 | 5.191 | **481.9** | **2125** |
|   2,048 | 2.608 | 1.786 | 16.271 | 1.540 | 17.964 | **805.6** | **2542** |
|   4,096 | 5.202 | 3.450 | 24.560 | 3.080 | 66.655 | **1524.2** | **2687** |
|   8,192 | 10.197 | 7.002 | 37.971 | 6.160 | 299.692 | **3547.0** | **2310** |
|  16,384 | 20.708 | 14.509 | 73.708 | 12.320 | 1294.209 | **10518.3** | **1558** |
|  32,768 | 41.817 | 30.773 | 154.959 | 24.641 | 5327.050 | **35013.4** | **936** |
|  65,536 | 83.081 | 61.068 | 345.424 | 49.282 | 21276.941 | **124168.0** | **528** |
| 131,072 | 163.872 | 122.699 | 717.913 | 98.564 | 96105.312 | **516845.9** | **254** |

---

## 6. Decode Time-Share (kv=4,096)

**Total decode @ kv=4096: 28.43 ms → 35.2 tok/s**

| Group | ms | Share |
|---|---:|---:|
| MoE_fused (INT4, TK=8/128) (×30) | 8.539 | 30.0% |
| LM_head (INT8) (×1) | 7.303 | 25.7% |
| DenseMLP_gate+up+down (INT4) (×30) | 3.142 | 11.0% |
| FC_QKV_sliding (INT4) (×25) | 2.997 | 10.5% |
| PA_sliding (INT8 KV, sw=1024) (×25) | 2.000 | 7.0% |
| FC_O_sliding (INT4) (×25) | 1.540 | 5.4% |
| SmallOps (norm/rope/add) (×0) | 0.861 | 3.0% |
| PA_full (INT8 KV) (×5) | 0.801 | 2.8% |
| FC_QK_full (INT4) (×5) | 0.666 | 2.3% |
| FC_O_full (INT4) (×5) | 0.584 | 2.1% |

---

## 7. Decode Tables (1 query token, KV = context length)

_Columns: op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound_
_Eff% = GB/s / 110 GB/s × 100 for memory-bound; GFLOPS / XMX_peak × 100 for compute-bound._

### Decode — KV=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (INT4, TK=8/128) | moe_3gemm+scatter+gather | 0.2846 | 30 | 8.539 | 334.31 | 90.8 | 82.6% | memory |
| LM_head (INT8) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| DenseMLP_gate+up+down (INT4) | gemm_kernel×3 | 0.1047 | 30 | 3.142 | 340.75 | 89.9 | 81.7% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1199 | 25 | 2.997 | 384.86 | 100.2 | 91.1% | memory |
| PA_sliding (INT8 KV, sw=1024) | paged_attn_single_token | 0.0800 | 25 | 2.000 | 209.72 | 52.4 | 47.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0616 | 25 | 1.540 | 374.51 | 97.5 | 88.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.861 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1332 | 5 | 0.666 | 389.59 | 101.4 | 92.2% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1168 | 5 | 0.584 | 394.93 | 102.8 | 93.4% | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 0.1061 | 5 | 0.530 | 316.40 | 19.8 | 18.0% | memory |
| **TOTAL** | | | | **28.162** | | | | |
| | | | | | | | **35.5 tok/s** | |

### Decode — KV=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (INT4, TK=8/128) | moe_3gemm+scatter+gather | 0.2846 | 30 | 8.539 | 334.31 | 90.8 | 82.6% | memory |
| LM_head (INT8) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| DenseMLP_gate+up+down (INT4) | gemm_kernel×3 | 0.1047 | 30 | 3.142 | 340.75 | 89.9 | 81.7% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1199 | 25 | 2.997 | 384.86 | 100.2 | 91.1% | memory |
| PA_sliding (INT8 KV, sw=1024) | paged_attn_single_token | 0.0800 | 25 | 2.000 | 209.72 | 52.4 | 47.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0616 | 25 | 1.540 | 374.51 | 97.5 | 88.6% | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 0.1947 | 5 | 0.974 | 344.68 | 21.5 | 19.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.861 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1332 | 5 | 0.666 | 389.59 | 101.4 | 92.2% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1168 | 5 | 0.584 | 394.93 | 102.8 | 93.4% | memory |
| **TOTAL** | | | | **28.605** | | | | |
| | | | | | | | **35.0 tok/s** | |

### Decode — KV=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (INT4, TK=8/128) | moe_3gemm+scatter+gather | 0.2846 | 30 | 8.539 | 334.31 | 90.8 | 82.6% | memory |
| LM_head (INT8) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| DenseMLP_gate+up+down (INT4) | gemm_kernel×3 | 0.1047 | 30 | 3.142 | 340.75 | 89.9 | 81.7% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1199 | 25 | 2.997 | 384.86 | 100.2 | 91.1% | memory |
| PA_sliding (INT8 KV, sw=1024) | paged_attn_single_token | 0.0800 | 25 | 2.000 | 209.72 | 52.4 | 47.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0616 | 25 | 1.540 | 374.51 | 97.5 | 88.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.861 | — | — | — | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 0.1601 | 5 | 0.801 | 838.16 | 52.4 | 47.6% | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1332 | 5 | 0.666 | 389.59 | 101.4 | 92.2% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1168 | 5 | 0.584 | 394.93 | 102.8 | 93.4% | memory |
| **TOTAL** | | | | **28.432** | | | | |
| | | | | | | | **35.2 tok/s** | |

### Decode — KV=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (INT4, TK=8/128) | moe_3gemm+scatter+gather | 0.2846 | 30 | 8.539 | 334.31 | 90.8 | 82.6% | memory |
| LM_head (INT8) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| DenseMLP_gate+up+down (INT4) | gemm_kernel×3 | 0.1047 | 30 | 3.142 | 340.75 | 89.9 | 81.7% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1199 | 25 | 2.997 | 384.86 | 100.2 | 91.1% | memory |
| PA_sliding (INT8 KV, sw=1024) | paged_attn_single_token | 0.0800 | 25 | 2.000 | 209.72 | 52.4 | 47.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0616 | 25 | 1.540 | 374.51 | 97.5 | 88.6% | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 0.3039 | 5 | 1.520 | 883.30 | 55.2 | 50.2% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.861 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1332 | 5 | 0.666 | 389.59 | 101.4 | 92.2% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1168 | 5 | 0.584 | 394.93 | 102.8 | 93.4% | memory |
| **TOTAL** | | | | **29.151** | | | | |
| | | | | | | | **34.3 tok/s** | |

### Decode — KV=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (INT4, TK=8/128) | moe_3gemm+scatter+gather | 0.2846 | 30 | 8.539 | 334.31 | 90.8 | 82.6% | memory |
| LM_head (INT8) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| DenseMLP_gate+up+down (INT4) | gemm_kernel×3 | 0.1047 | 30 | 3.142 | 340.75 | 89.9 | 81.7% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1199 | 25 | 2.997 | 384.86 | 100.2 | 91.1% | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 0.5650 | 5 | 2.825 | 950.17 | 59.4 | 54.0% | memory |
| PA_sliding (INT8 KV, sw=1024) | paged_attn_single_token | 0.0800 | 25 | 2.000 | 209.72 | 52.4 | 47.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0616 | 25 | 1.540 | 374.51 | 97.5 | 88.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.861 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1332 | 5 | 0.666 | 389.59 | 101.4 | 92.2% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1168 | 5 | 0.584 | 394.93 | 102.8 | 93.4% | memory |
| **TOTAL** | | | | **30.457** | | | | |
| | | | | | | | **32.8 tok/s** | |

### Decode — KV=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (INT4, TK=8/128) | moe_3gemm+scatter+gather | 0.2846 | 30 | 8.539 | 334.31 | 90.8 | 82.6% | memory |
| LM_head (INT8) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| PA_full (INT8 KV) | paged_attn_single_token | 1.0911 | 5 | 5.455 | 984.13 | 61.5 | 55.9% | memory |
| DenseMLP_gate+up+down (INT4) | gemm_kernel×3 | 0.1047 | 30 | 3.142 | 340.75 | 89.9 | 81.7% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1199 | 25 | 2.997 | 384.86 | 100.2 | 91.1% | memory |
| PA_sliding (INT8 KV, sw=1024) | paged_attn_single_token | 0.0800 | 25 | 2.000 | 209.72 | 52.4 | 47.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0616 | 25 | 1.540 | 374.51 | 97.5 | 88.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.861 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1332 | 5 | 0.666 | 389.59 | 101.4 | 92.2% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1168 | 5 | 0.584 | 394.93 | 102.8 | 93.4% | memory |
| **TOTAL** | | | | **33.087** | | | | |
| | | | | | | | **30.2 tok/s** | |

### Decode — KV=65,536

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (INT8 KV) | paged_attn_single_token | 2.1144 | 5 | 10.572 | 1015.66 | 63.5 | 57.7% | memory |
| MoE_fused (INT4, TK=8/128) | moe_3gemm+scatter+gather | 0.2846 | 30 | 8.539 | 334.31 | 90.8 | 82.6% | memory |
| LM_head (INT8) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| DenseMLP_gate+up+down (INT4) | gemm_kernel×3 | 0.1047 | 30 | 3.142 | 340.75 | 89.9 | 81.7% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1199 | 25 | 2.997 | 384.86 | 100.2 | 91.1% | memory |
| PA_sliding (INT8 KV, sw=1024) | paged_attn_single_token | 0.0800 | 25 | 2.000 | 209.72 | 52.4 | 47.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0616 | 25 | 1.540 | 374.51 | 97.5 | 88.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.861 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1332 | 5 | 0.666 | 389.59 | 101.4 | 92.2% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1168 | 5 | 0.584 | 394.93 | 102.8 | 93.4% | memory |
| **TOTAL** | | | | **38.203** | | | | |
| | | | | | | | **26.2 tok/s** | |

### Decode — KV=131,072

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (INT8 KV) | paged_attn_single_token | 4.1909 | 5 | 20.955 | 1024.82 | 64.1 | 58.2% | memory |
| MoE_fused (INT4, TK=8/128) | moe_3gemm+scatter+gather | 0.2846 | 30 | 8.539 | 334.31 | 90.8 | 82.6% | memory |
| LM_head (INT8) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| DenseMLP_gate+up+down (INT4) | gemm_kernel×3 | 0.1047 | 30 | 3.142 | 340.75 | 89.9 | 81.7% | memory |
| FC_QKV_sliding (INT4) | gemm_kernel | 0.1199 | 25 | 2.997 | 384.86 | 100.2 | 91.1% | memory |
| PA_sliding (INT8 KV, sw=1024) | paged_attn_single_token | 0.0800 | 25 | 2.000 | 209.72 | 52.4 | 47.7% | memory |
| FC_O_sliding (INT4) | gemm_kernel | 0.0616 | 25 | 1.540 | 374.51 | 97.5 | 88.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.861 | — | — | — | memory |
| FC_QK_full (INT4) | gemm_kernel | 0.1332 | 5 | 0.666 | 389.59 | 101.4 | 92.2% | memory |
| FC_O_full (INT4) | gemm_kernel | 0.1168 | 5 | 0.584 | 394.93 | 102.8 | 93.4% | memory |
| **TOTAL** | | | | **48.586** | | | | |
| | | | | | | | **20.6 tok/s** | |

---

## 8. Prefill Tables (single forward over S tokens)


### Prefill — S=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (INT4, TK=8/128) | grouped_micro_gemm×3+scatter+gather | 11.2502 | 30 | 337.506 | 8661.37 | 38.7 | 35.2% | memory |
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel×3 | 0.9367 | 30 | 28.102 | 39009.08 | 42.3 | 38.5% | memory |
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 5.1908 | 5 | 25.954 | 3309.65 | 0.8 | 5.6% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 23.653 | — | — | — | memory |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 0.7940 | 25 | 19.849 | 59505.29 | 43.5 | 50.4% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 0.7700 | 25 | 19.251 | 11155.34 | 10.9 | 18.9% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 0.4593 | 25 | 11.482 | 51435.05 | 43.9 | 43.6% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 0.8791 | 5 | 4.396 | 53741.87 | 39.3 | 45.6% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 0.8770 | 5 | 4.385 | 60604.65 | 43.5 | 51.4% | compute |
| **TOTAL** | | | | **481.9** | | | | |
| | | | | | | | **TTFT 0.48s, 2125 tok/s** | |

### Prefill — S=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (INT4, TK=8/128) | grouped_micro_gemm×3+scatter+gather | 16.2711 | 30 | 488.133 | 11977.31 | 28.3 | 25.7% | memory |
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 17.9644 | 5 | 89.822 | 3825.32 | 0.5 | 6.5% | compute |
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel×3 | 1.7858 | 30 | 53.574 | 40923.61 | 39.2 | 35.6% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 49.985 | — | — | — | memory |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 1.5538 | 25 | 38.845 | 60812.44 | 36.7 | 51.6% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 1.5401 | 25 | 38.501 | 11155.34 | 5.4 | 18.9% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 0.8985 | 25 | 22.462 | 52583.96 | 38.2 | 44.6% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 1.7932 | 5 | 8.966 | 59278.56 | 35.0 | 50.3% | compute |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 1.5959 | 5 | 7.979 | 59207.71 | 35.8 | 50.2% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| **TOTAL** | | | | **805.6** | | | | |
| | | | | | | | **TTFT 0.81s, 2542 tok/s** | |

### Prefill — S=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE_fused (INT4, TK=8/128) | grouped_micro_gemm×3+scatter+gather | 24.5603 | 30 | 736.810 | 15869.83 | 20.8 | 18.9% | memory |
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 66.6551 | 5 | 333.275 | 4123.88 | 0.3 | 7.0% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 110.211 | — | — | — | memory |
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel×3 | 3.4499 | 30 | 103.497 | 42367.44 | 37.8 | 35.9% | compute |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 3.1824 | 25 | 79.560 | 59382.35 | 32.1 | 50.3% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 3.0801 | 25 | 77.003 | 11155.34 | 2.7 | 18.9% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 1.7135 | 25 | 42.839 | 55142.57 | 36.5 | 46.7% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 3.5749 | 5 | 17.874 | 59471.14 | 31.3 | 50.4% | compute |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 3.1591 | 5 | 15.795 | 59821.21 | 32.3 | 50.7% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| **TOTAL** | | | | **1524.2** | | | | |
| | | | | | | | **TTFT 1.52s, 2687 tok/s** | |

### Prefill — S=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 299.6920 | 5 | 1498.460 | 3668.81 | 0.1 | 6.2% | compute |
| MoE_fused (INT4, TK=8/128) | grouped_micro_gemm×3+scatter+gather | 37.9712 | 30 | 1139.135 | 20529.69 | 16.0 | 17.4% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 232.091 | — | — | — | memory |
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel×3 | 7.0017 | 30 | 210.051 | 41750.78 | 35.9 | 35.4% | compute |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 6.2816 | 25 | 157.039 | 60169.30 | 30.6 | 51.0% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 6.1602 | 25 | 154.006 | 11155.34 | 1.4 | 18.9% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 3.2834 | 25 | 82.086 | 57555.03 | 36.3 | 48.8% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 7.0813 | 5 | 35.407 | 60045.47 | 29.7 | 50.9% | compute |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 6.2767 | 5 | 31.384 | 60215.70 | 30.6 | 51.0% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| **TOTAL** | | | | **3547.0** | | | | |
| | | | | | | | **TTFT 3.55s, 2310 tok/s** | |

### Prefill — S=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 1294.2093 | 5 | 6471.046 | 3398.25 | 0.1 | 5.8% | compute |
| MoE_fused (INT4, TK=8/128) | grouped_micro_gemm×3+scatter+gather | 73.7078 | 30 | 2211.234 | 21152.08 | 10.9 | 17.9% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 464.182 | — | — | — | memory |
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel×3 | 14.5089 | 30 | 435.268 | 40296.05 | 34.0 | 34.2% | compute |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 12.8137 | 25 | 320.342 | 58992.77 | 29.1 | 50.0% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 12.3205 | 25 | 308.012 | 11155.34 | 0.7 | 18.9% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 6.7685 | 25 | 169.213 | 55840.41 | 34.3 | 47.3% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 13.9272 | 5 | 69.636 | 61060.42 | 29.3 | 51.8% | compute |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 12.4080 | 5 | 62.040 | 60921.50 | 30.0 | 51.6% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| **TOTAL** | | | | **10518.3** | | | | |
| | | | | | | | **TTFT 10.52s, 1558 tok/s** | |

### Prefill — S=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 5327.0498 | 5 | 26635.249 | 3302.43 | 0.0 | 5.6% | compute |
| MoE_fused (INT4, TK=8/128) | grouped_micro_gemm×3+scatter+gather | 154.9590 | 30 | 4648.770 | 20122.40 | 7.8 | 17.1% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 928.363 | — | — | — | memory |
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel×3 | 30.7725 | 30 | 923.176 | 37998.34 | 31.8 | 32.2% | compute |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 25.3627 | 25 | 634.067 | 59608.36 | 28.9 | 50.5% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 24.6409 | 25 | 616.023 | 11155.34 | 0.3 | 18.9% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 14.2106 | 25 | 355.266 | 53193.55 | 32.3 | 45.1% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 28.0916 | 5 | 140.458 | 60544.96 | 28.5 | 51.3% | compute |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 24.9424 | 5 | 124.712 | 60612.80 | 29.4 | 51.4% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| **TOTAL** | | | | **35013.4** | | | | |
| | | | | | | | **TTFT 35.01s, 936 tok/s** | |

### Prefill — S=65,536

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 21276.9413 | 5 | 106384.706 | 3307.28 | 0.0 | 5.6% | compute |
| MoE_fused (INT4, TK=8/128) | grouped_micro_gemm×3+scatter+gather | 345.4240 | 30 | 10362.721 | 18054.02 | 5.8 | 15.3% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 1856.726 | — | — | — | memory |
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel×3 | 61.0681 | 30 | 1832.044 | 38295.10 | 31.9 | 32.5% | compute |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 51.4616 | 25 | 1286.541 | 58755.55 | 28.3 | 49.8% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 49.2819 | 25 | 1232.046 | 11155.34 | 0.2 | 18.9% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 26.7396 | 25 | 668.490 | 56538.92 | 34.1 | 47.9% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 56.5359 | 5 | 282.679 | 60167.36 | 28.1 | 51.0% | compute |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 50.9459 | 5 | 254.730 | 59350.35 | 28.6 | 50.3% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| **TOTAL** | | | | **124168.0** | | | | |
| | | | | | | | **TTFT 124.17s, 528 tok/s** | |

### Prefill — S=131,072

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full (FP16 prefill, causal) | sdpa_micro_prefill | 96105.3123 | 5 | 480526.561 | 2928.82 | 0.0 | 5.0% | compute |
| MoE_fused (INT4, TK=8/128) | grouped_micro_gemm×3+scatter+gather | 717.9135 | 30 | 21537.404 | 17373.38 | 5.0 | 14.7% | compute |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 3713.453 | — | — | — | memory |
| DenseMLP_gate+up+down (INT4→INT8 XMX) | dq+gemm_kernel×3 | 122.6986 | 30 | 3680.958 | 38119.58 | 31.7 | 32.3% | compute |
| FC_QKV_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 102.0702 | 25 | 2551.755 | 59246.61 | 28.4 | 50.2% | compute |
| PA_sliding (FP16 prefill, sw=1024) | sdpa_micro_prefill | 98.5637 | 25 | 2464.093 | 11155.34 | 0.1 | 18.9% | compute |
| FC_O_sliding (INT4→INT8 XMX) | dq+gemm_kernel | 52.8223 | 25 | 1320.557 | 57242.06 | 34.4 | 48.5% | compute |
| FC_QK_full (INT4→INT8 XMX) | dq+gemm_kernel | 110.7548 | 5 | 553.774 | 61426.02 | 28.6 | 52.1% | compute |
| FC_O_full (INT4→INT8 XMX) | dq+gemm_kernel | 98.0131 | 5 | 490.066 | 61699.02 | 29.6 | 52.3% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 7.3029 | 1 | 7.303 | 202.17 | 102.7 | 93.4% | memory |
| **TOTAL** | | | | **516845.9** | | | | |
| | | | | | | | **TTFT 516.85s, 254 tok/s** | |

---

## Caveats & Method

- Each op profiled in its own process via cliloader Device Performance Timing; mean kernel time per iteration.
- FC weight bytes: INT4 weight + FP16 scale/zp(g=128) + FP16 act + FP16 out.
- PA bytes: INT8 KV cache + FP16 Q/out.
- Decode FC is **memory-bound** (weight-read dominates at M=1).
- Prefill FC is **compute-bound** (INT8 XMX path via dynamic quantize).
- Sliding PA decode always uses kv=1024 (sliding_window cap). Full PA grows with context.
- Dense MLP and MoE run in parallel per layer — model total includes both.
- MoE bench uses Swish activation as proxy for GEGLU (GPU plugin limitation).
- LM head: 1 token only, both decode and prefill.
- Small ops prefill for S>8192: linearly extrapolated from S=8192 measurement.
- Target: Local_Admin@10.239.132.229 (PTL 12Xe, B390 iGPU)