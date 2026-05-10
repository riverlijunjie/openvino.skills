# Qwen3.5-MoE-35B-A3B (qwen3_5_moe) — Roofline on PTL (B390 iGPU)

**Date:** 2026-05-09
**Target:** PTL (Panther Lake B390 iGPU, 12 Xe @ 2400 MHz)
**Config:** FP16 QKV/O + FP16 shared expert (3 separate FC ops)
**Profiler:** cliloader 3.0.6 `--device-performance-timing`, mean kernel time
**Token sweep:** S/kv ∈ {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}
**Bench:** `.github/skills/dev_roofline_profiling/utils/{fc,moe,pa,gdn,small_ops}_bench`

---

## 1. Hardware Peaks

| Platform | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | Ridge point (F16) |
|---|---:|---:|---:|---:|
| PTL (B390 iGPU, 12 Xe @ 2400 MHz) | 110.0 | 58.982 | 117.965 | 536.2 FLOP/byte |

---

## 2. Model Configuration (qwen3_5_moe — F16 QKV/O variant)

| Field | Value |
|---|---|
| `vocab_size`                    | 248,320 |
| `hidden_size` (H)               | 2,048 |
| `num_hidden_layers`             | **40** (hybrid attention) |
| Layer pattern                   | `linear_attn × 3 → full_attn × 1`, repeated → **10 full + 30 GDN** |
| `num_attention_heads`           | 16 (NH, full-attn) |
| `num_key_value_heads`           | 2 (NKV, GQA group=8) |
| `head_dim`                      | 256 |
| `linear_num_value_heads`        | 32 (HK in GDN bench) |
| `linear_value_head_dim`         | 128 |
| `moe_intermediate_size`         | 512 |
| `shared_expert_intermediate_size` | **512 (always-on)** |
| `num_experts`                   | 256 |
| `num_experts_per_tok`           | 8 |
| FC_QKV weight quant             | **FP16 (uncompressed)** |
| FC_O weight quant               | **FP16 (uncompressed)** |
| Shared expert quant             | **FP16 (uncompressed), 3 separate FullyConnected ops** |
| MoE routed expert quant         | INT4 g=128 (asymmetric), fused via MOE3GemmFusedCompressed |
| FC_linattn weight quant         | INT4 g=128 (in_proj_qkv/z/a/b + out_proj) |
| LM-head weight quant            | INT8 g=128 |
| KV cache quant                  | INT8 (asymmetric) |
| Activation dtype                | FP16 |


---

## 3. Weight Size Summary

| Weight | Shape (K × N) | Quant | Bytes | × Layers | Total MB |
|---|---:|---|---:|---:|---:|
| FC_QKV (fused Q+K+V proj) | 2048×5120 | FP16 | 20,971,520 | 10 | 200.0 |
| FC_O (attention output)   | 4096×2048 | FP16 | 16,777,216 | 10 | 160.0 |
| FC_linattn_qkv (in_proj_qkv)| 2048×8192 | INT4 g=128 | 8,716,288 | 30 | 249.4 |
| FC_linattn_z (in_proj_z)    | 2048×4096 | INT4 g=128 | 4,358,144 | 30 | 124.7 |
| FC_linattn_a (in_proj_a)    | 2048×32   | INT4 g=128 | 34,048 | 30 | 1.0 |
| FC_linattn_b (in_proj_b)    | 2048×32   | INT4 g=128 | 34,048 | 30 | 1.0 |
| FC_linattn_out (out_proj)   | 4096×2048 | INT4 g=128 | 4,358,144 | 30 | 124.7 |
| MoE Expert gate+up+down   | 2048×512 / 512×2048 | INT4 g=128 | 1,634,304/expert | 40×256 | 15960.0 |
| Router                    | 2048×256 | INT4 g=128 | 272,384 | 40 | 10.4 |
| Shared Expert gate+up+down | 2048×512 / 512×2048 | FP16 | 6,291,456 | 40 | 240.0 |
| LM_Head                   | 2048×248320 | INT8 g=128 | 516,505,600 | 1 | 492.6 |
| **Total static weights** | | | | | **17564 MB** |

---

## 4. Graph Fusion Notes

| Op variant | GPU primitive | Fused? | Notes |
|---|---|---|---|
| FC_QKV / FC_O (FP16) | FullyConnected (plain) | Not fused further | Single `gemm_kernel` per call |
| MoE routed experts (INT4) | MOE3GemmFusedCompressed | ✅ Fused | `moe_3gemm_swiglu_mlp_gate_up` + `_down` + scatter/gather |
| Shared expert gate/up (FP16) | 3 × FullyConnected | ❌ NOT fused | FP16 weights cannot match `MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN` |
| Shared expert down (FP16) | FullyConnected | ❌ NOT fused | Same reason |
| FC_linattn_qkv/z/out (INT4) | FullyConnectedCompressed | Partial | decode: 1 `gemm_kernel`; prefill: `dynamic_quantize_gpu_opt`+`gemm_kernel` |
| FC_linattn_a/b (INT4, tiny) | FullyConnectedCompressed | Partial | decode: 1 `gemm_kernel` (3.8µs); prefill: dominated by dynamic_quantize overhead |
| PagedAttention | PagedAttention | ✅ Fused | INT8 KV cache, GQA group=8 |
| GatedDeltaNet | GatedDeltaNet | ✅ Fused | Reference kernel (not optimised) |

> **Key insight**: Shared expert with FP16 weights runs as **3 separate FullyConnected kernels** per MoE layer × 40 layers.
> This is the **actual model runtime behaviour** — the FP16 plain constant is converted to FullyConnected by
> `ConvertMatMulToFullyConnected` before `FuseMOESharedExpert` can pattern-match it.

---

## 5. Decode Performance — Totals

| kv tokens | MoE routed/L (ms) | Shared F16/L (ms) | PA/L (ms) | GDN/L (ms) | linattn/L (ms) | LM head (ms) | **total ms** | **tok/s** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|   1,024 | 0.1613 | 0.0801 | 0.0713 | 0.0315 | 0.1836 | 5.190 | **27.19** | **36.8** |
|   2,048 | 0.1613 | 0.0801 | 0.1198 | 0.0315 | 0.1836 | 5.190 | **27.67** | **36.1** |
|   4,096 | 0.1613 | 0.0801 | 0.1684 | 0.0315 | 0.1836 | 5.190 | **28.16** | **35.5** |
|   8,192 | 0.1613 | 0.0801 | 0.2523 | 0.0315 | 0.1836 | 5.190 | **29.00** | **34.5** |
|  16,384 | 0.1613 | 0.0801 | 0.4109 | 0.0315 | 0.1836 | 5.190 | **30.59** | **32.7** |
|  32,768 | 0.1613 | 0.0801 | 0.6004 | 0.0315 | 0.1836 | 5.190 | **32.48** | **30.8** |
|  65,536 | 0.1613 | 0.0801 | 1.1726 | 0.0315 | 0.1836 | 5.190 | **38.20** | **26.2** |
| 131,072 | 0.1613 | 0.0801 | 2.2922 | 0.0315 | 0.1836 | 5.190 | **49.40** | **20.2** |

---

## 6. Prefill Performance — Totals (TTFT)

| S tokens | MoE routed/L (ms) | Shared F16/L (ms) | PA/L (ms) | GDN/L (ms) | linattn/L (ms) | **TTFT (ms)** | **tok/s** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|   1,024 | 14.150 | 0.285 | 0.696 | 2.563 | 1.525 | **728.7** | **1405.2** |
|   2,048 | 20.108 | 0.502 | 2.488 | 5.164 | 2.941 | **1132.3** | **1808.8** |
|   4,096 | 24.905 | 0.896 | 9.421 | 10.379 | 6.030 | **1696.9** | **2413.8** |
|   8,192 | 41.639 | 1.634 | 39.993 | 20.610 | 11.886 | **3278.2** | **2498.9** |
|  16,384 | 56.282 | 3.220 | 153.493 | 41.318 | 22.477 | **6138.9** | **2668.9** |
|  32,768 | 106.343 | 7.823 | 623.609 | 83.299 | 46.615 | **15289.2** | **2143.2** |
|  65,536 | 242.328 | 17.065 | 2500.034 | 168.580 | 96.259 | **44505.8** | **1472.5** |
| 131,072 | 552.693 | 28.706 | 11343.723 | 337.161 ★ | 197.563 | **155088.6** | **845.1** |

★ GDN prefill S=131072 kernel log was empty (run failed); value extrapolated ×2 from S=65536.

---

## 7. Decode Time-Share (kv=4096)

**Total decode @ kv=4096: 28.16 ms → 35.5 tok/s**

| Group | ms | Share |
|---|---:|---:|
| MoE3GEMM_fused (routed, INT4) (×40) | 6.453 | 22.9% |
| LM_head (INT8) (×1) | 5.190 | 18.4% |
| SharedExpert_gate+up+down (FP16 FC×3) (×40) | 3.205 | 11.4% |
| FC_QKV (FP16) (×10) | 3.022 | 10.7% |
| FC_linattn_qkv (INT4) (×30) | 2.569 | 9.1% |
| PagedAttention (INT8 KV) (×10) | 1.684 | 6.0% |
| FC_O (FP16) (×10) | 1.679 | 6.0% |
| FC_linattn_out (INT4) (×30) | 1.390 | 4.9% |
| FC_linattn_z (INT4) (×30) | 1.318 | 4.7% |
| GatedDeltaNet (×30) | 0.945 | 3.4% |
| SmallOps (norm/rope/add) (×0) | 0.473 | 1.7% |
| FC_linattn_a (INT4) (×30) | 0.115 | 0.4% |
| FC_linattn_b (INT4) (×30) | 0.115 | 0.4% |

---

## 8. Decode Tables (1 query token, KV = context length)

_Columns: op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound_
_Eff% = GB/s / 110 GB/s × 100 for memory-bound; GFLOPS / XMX_peak × 100 for compute-bound._

### Decode — KV=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE3GEMM_fused (routed, INT4) | moe_3gemm_swiglu_mlp_gate_up+down | 0.1613 | 40 | 6.453 | 311.97 | 82.8 | 75.3% | memory |
| LM_head (INT8) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 0.0801 | 40 | 3.205 | 78.51 | 78.7 | 71.5% | memory |
| FC_QKV (FP16) | gemm_kernel | 0.3022 | 10 | 3.022 | 69.39 | 69.4 | 63.1% | memory |
| FC_linattn_qkv (INT4) | gemm_kernel | 0.0856 | 30 | 2.569 | 391.82 | 102.0 | 92.7% | memory |
| FC_O (FP16) | gemm_kernel | 0.1679 | 10 | 1.679 | 99.92 | 100.0 | 90.9% | memory |
| FC_linattn_out (INT4) | gemm_kernel | 0.0463 | 30 | 1.390 | 362.02 | 94.3 | 85.7% | memory |
| FC_linattn_z (INT4) | gemm_kernel | 0.0439 | 30 | 1.318 | 381.88 | 99.5 | 90.4% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0315 | 30 | 0.945 | 33.30 | 66.7 | 60.7% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt__single_token | 0.0713 | 10 | 0.713 | 235.30 | 14.7 | 13.4% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.473 | — | — | — | memory |
| FC_linattn_a (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.08 | 9.9 | 9.0% | memory |
| FC_linattn_b (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.09 | 9.9 | 9.0% | memory |
| **TOTAL** | | | | **27.190** | | | | |
| | | | | | | | **36.8 tok/s** | |

### Decode — KV=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE3GEMM_fused (routed, INT4) | moe_3gemm_swiglu_mlp_gate_up+down | 0.1613 | 40 | 6.453 | 311.97 | 82.8 | 75.3% | memory |
| LM_head (INT8) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 0.0801 | 40 | 3.205 | 78.51 | 78.7 | 71.5% | memory |
| FC_QKV (FP16) | gemm_kernel | 0.3022 | 10 | 3.022 | 69.39 | 69.4 | 63.1% | memory |
| FC_linattn_qkv (INT4) | gemm_kernel | 0.0856 | 30 | 2.569 | 391.82 | 102.0 | 92.7% | memory |
| FC_O (FP16) | gemm_kernel | 0.1679 | 10 | 1.679 | 99.92 | 100.0 | 90.9% | memory |
| FC_linattn_out (INT4) | gemm_kernel | 0.0463 | 30 | 1.390 | 362.02 | 94.3 | 85.7% | memory |
| FC_linattn_z (INT4) | gemm_kernel | 0.0439 | 30 | 1.318 | 381.88 | 99.5 | 90.4% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt__single_token | 0.1198 | 10 | 1.198 | 280.18 | 17.5 | 15.9% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0315 | 30 | 0.945 | 33.30 | 66.7 | 60.7% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.473 | — | — | — | memory |
| FC_linattn_a (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.08 | 9.9 | 9.0% | memory |
| FC_linattn_b (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.09 | 9.9 | 9.0% | memory |
| **TOTAL** | | | | **27.674** | | | | |
| | | | | | | | **36.1 tok/s** | |

### Decode — KV=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE3GEMM_fused (routed, INT4) | moe_3gemm_swiglu_mlp_gate_up+down | 0.1613 | 40 | 6.453 | 311.97 | 82.8 | 75.3% | memory |
| LM_head (INT8) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 0.0801 | 40 | 3.205 | 78.51 | 78.7 | 71.5% | memory |
| FC_QKV (FP16) | gemm_kernel | 0.3022 | 10 | 3.022 | 69.39 | 69.4 | 63.1% | memory |
| FC_linattn_qkv (INT4) | gemm_kernel | 0.0856 | 30 | 2.569 | 391.82 | 102.0 | 92.7% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt__single_token | 0.1684 | 10 | 1.684 | 398.56 | 24.9 | 22.6% | memory |
| FC_O (FP16) | gemm_kernel | 0.1679 | 10 | 1.679 | 99.92 | 100.0 | 90.9% | memory |
| FC_linattn_out (INT4) | gemm_kernel | 0.0463 | 30 | 1.390 | 362.02 | 94.3 | 85.7% | memory |
| FC_linattn_z (INT4) | gemm_kernel | 0.0439 | 30 | 1.318 | 381.88 | 99.5 | 90.4% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0315 | 30 | 0.945 | 33.30 | 66.7 | 60.7% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.473 | — | — | — | memory |
| FC_linattn_a (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.08 | 9.9 | 9.0% | memory |
| FC_linattn_b (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.09 | 9.9 | 9.0% | memory |
| **TOTAL** | | | | **28.160** | | | | |
| | | | | | | | **35.5 tok/s** | |

### Decode — KV=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE3GEMM_fused (routed, INT4) | moe_3gemm_swiglu_mlp_gate_up+down | 0.1613 | 40 | 6.453 | 311.97 | 82.8 | 75.3% | memory |
| LM_head (INT8) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 0.0801 | 40 | 3.205 | 78.51 | 78.7 | 71.5% | memory |
| FC_QKV (FP16) | gemm_kernel | 0.3022 | 10 | 3.022 | 69.39 | 69.4 | 63.1% | memory |
| FC_linattn_qkv (INT4) | gemm_kernel | 0.0856 | 30 | 2.569 | 391.82 | 102.0 | 92.7% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt__single_token | 0.2523 | 10 | 2.523 | 532.05 | 33.3 | 30.2% | memory |
| FC_O (FP16) | gemm_kernel | 0.1679 | 10 | 1.679 | 99.92 | 100.0 | 90.9% | memory |
| FC_linattn_out (INT4) | gemm_kernel | 0.0463 | 30 | 1.390 | 362.02 | 94.3 | 85.7% | memory |
| FC_linattn_z (INT4) | gemm_kernel | 0.0439 | 30 | 1.318 | 381.88 | 99.5 | 90.4% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0315 | 30 | 0.945 | 33.30 | 66.7 | 60.7% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.473 | — | — | — | memory |
| FC_linattn_a (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.08 | 9.9 | 9.0% | memory |
| FC_linattn_b (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.09 | 9.9 | 9.0% | memory |
| **TOTAL** | | | | **28.999** | | | | |
| | | | | | | | **34.5 tok/s** | |

### Decode — KV=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE3GEMM_fused (routed, INT4) | moe_3gemm_swiglu_mlp_gate_up+down | 0.1613 | 40 | 6.453 | 311.97 | 82.8 | 75.3% | memory |
| LM_head (INT8) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt__single_token | 0.4109 | 10 | 4.109 | 653.26 | 40.8 | 37.1% | memory |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 0.0801 | 40 | 3.205 | 78.51 | 78.7 | 71.5% | memory |
| FC_QKV (FP16) | gemm_kernel | 0.3022 | 10 | 3.022 | 69.39 | 69.4 | 63.1% | memory |
| FC_linattn_qkv (INT4) | gemm_kernel | 0.0856 | 30 | 2.569 | 391.82 | 102.0 | 92.7% | memory |
| FC_O (FP16) | gemm_kernel | 0.1679 | 10 | 1.679 | 99.92 | 100.0 | 90.9% | memory |
| FC_linattn_out (INT4) | gemm_kernel | 0.0463 | 30 | 1.390 | 362.02 | 94.3 | 85.7% | memory |
| FC_linattn_z (INT4) | gemm_kernel | 0.0439 | 30 | 1.318 | 381.88 | 99.5 | 90.4% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0315 | 30 | 0.945 | 33.30 | 66.7 | 60.7% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.473 | — | — | — | memory |
| FC_linattn_a (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.08 | 9.9 | 9.0% | memory |
| FC_linattn_b (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.09 | 9.9 | 9.0% | memory |
| **TOTAL** | | | | **30.586** | | | | |
| | | | | | | | **32.7 tok/s** | |

### Decode — KV=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE3GEMM_fused (routed, INT4) | moe_3gemm_swiglu_mlp_gate_up+down | 0.1613 | 40 | 6.453 | 311.97 | 82.8 | 75.3% | memory |
| PagedAttention (INT8 KV) | paged_attention_opt__single_token | 0.6004 | 10 | 6.004 | 894.24 | 55.9 | 50.8% | memory |
| LM_head (INT8) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 0.0801 | 40 | 3.205 | 78.51 | 78.7 | 71.5% | memory |
| FC_QKV (FP16) | gemm_kernel | 0.3022 | 10 | 3.022 | 69.39 | 69.4 | 63.1% | memory |
| FC_linattn_qkv (INT4) | gemm_kernel | 0.0856 | 30 | 2.569 | 391.82 | 102.0 | 92.7% | memory |
| FC_O (FP16) | gemm_kernel | 0.1679 | 10 | 1.679 | 99.92 | 100.0 | 90.9% | memory |
| FC_linattn_out (INT4) | gemm_kernel | 0.0463 | 30 | 1.390 | 362.02 | 94.3 | 85.7% | memory |
| FC_linattn_z (INT4) | gemm_kernel | 0.0439 | 30 | 1.318 | 381.88 | 99.5 | 90.4% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0315 | 30 | 0.945 | 33.30 | 66.7 | 60.7% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.473 | — | — | — | memory |
| FC_linattn_a (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.08 | 9.9 | 9.0% | memory |
| FC_linattn_b (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.09 | 9.9 | 9.0% | memory |
| **TOTAL** | | | | **32.480** | | | | |
| | | | | | | | **30.8 tok/s** | |

### Decode — KV=65,536

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (INT8 KV) | paged_attention_opt__single_token | 1.1726 | 10 | 11.726 | 915.68 | 57.2 | 52.0% | memory |
| MoE3GEMM_fused (routed, INT4) | moe_3gemm_swiglu_mlp_gate_up+down | 0.1613 | 40 | 6.453 | 311.97 | 82.8 | 75.3% | memory |
| LM_head (INT8) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 0.0801 | 40 | 3.205 | 78.51 | 78.7 | 71.5% | memory |
| FC_QKV (FP16) | gemm_kernel | 0.3022 | 10 | 3.022 | 69.39 | 69.4 | 63.1% | memory |
| FC_linattn_qkv (INT4) | gemm_kernel | 0.0856 | 30 | 2.569 | 391.82 | 102.0 | 92.7% | memory |
| FC_O (FP16) | gemm_kernel | 0.1679 | 10 | 1.679 | 99.92 | 100.0 | 90.9% | memory |
| FC_linattn_out (INT4) | gemm_kernel | 0.0463 | 30 | 1.390 | 362.02 | 94.3 | 85.7% | memory |
| FC_linattn_z (INT4) | gemm_kernel | 0.0439 | 30 | 1.318 | 381.88 | 99.5 | 90.4% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0315 | 30 | 0.945 | 33.30 | 66.7 | 60.7% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.473 | — | — | — | memory |
| FC_linattn_a (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.08 | 9.9 | 9.0% | memory |
| FC_linattn_b (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.09 | 9.9 | 9.0% | memory |
| **TOTAL** | | | | **38.203** | | | | |
| | | | | | | | **26.2 tok/s** | |

### Decode — KV=131,072

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (INT8 KV) | paged_attention_opt__single_token | 2.2922 | 10 | 22.922 | 936.87 | 58.6 | 53.2% | memory |
| MoE3GEMM_fused (routed, INT4) | moe_3gemm_swiglu_mlp_gate_up+down | 0.1613 | 40 | 6.453 | 311.97 | 82.8 | 75.3% | memory |
| LM_head (INT8) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 0.0801 | 40 | 3.205 | 78.51 | 78.7 | 71.5% | memory |
| FC_QKV (FP16) | gemm_kernel | 0.3022 | 10 | 3.022 | 69.39 | 69.4 | 63.1% | memory |
| FC_linattn_qkv (INT4) | gemm_kernel | 0.0856 | 30 | 2.569 | 391.82 | 102.0 | 92.7% | memory |
| FC_O (FP16) | gemm_kernel | 0.1679 | 10 | 1.679 | 99.92 | 100.0 | 90.9% | memory |
| FC_linattn_out (INT4) | gemm_kernel | 0.0463 | 30 | 1.390 | 362.02 | 94.3 | 85.7% | memory |
| FC_linattn_z (INT4) | gemm_kernel | 0.0439 | 30 | 1.318 | 381.88 | 99.5 | 90.4% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 0.0315 | 30 | 0.945 | 33.30 | 66.7 | 60.7% | memory |
| SmallOps (norm/rope/add) | rms/rope/eltwise | — | — | 0.473 | — | — | — | memory |
| FC_linattn_a (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.08 | 9.9 | 9.0% | memory |
| FC_linattn_b (INT4) | gemm_kernel | 0.0038 | 30 | 0.115 | 34.09 | 9.9 | 9.0% | memory |
| **TOTAL** | | | | **49.398** | | | | |
| | | | | | | | **20.2 tok/s** | |

---

## 9. Prefill Tables (single forward over S tokens)

_Eff% for INT4 FC prefill = GFLOPS / INT8 XMX (117.965 TOPS); for FP16 FC = GFLOPS / FP16 XMX (58.982 TFLOPS)._

### Prefill — S=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE3GEMM_fused (routed, INT4) | grouped_micro_gemm×3+scatter+gather | 14.1495 | 40 | 565.981 | 3642.50 | 30.9 | 28.1% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 2.5627 | 30 | 76.881 | 418.99 | 2.5 | 2.2% | memory |
| FC_linattn_qkv (INT4) | dq+gemm_kernel | 0.6955 | 30 | 20.866 | 49399.73 | 42.7 | 41.9% | compute |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 0.2854 | 40 | 11.418 | 22570.25 | 77.1 | 70.1% | memory |
| FC_linattn_out (INT4) | dq+gemm_kernel | 0.3590 | 30 | 10.771 | 47848.79 | 47.2 | 42.9% | memory |
| FC_linattn_z (INT4) | dq+gemm_kernel | 0.3493 | 30 | 10.478 | 49186.38 | 48.5 | 44.1% | memory |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 0.6961 | 10 | 6.961 | 12339.98 | 1.5 | 20.9% | compute |
| FC_O (FP16) | gemm_kernel | 0.6712 | 10 | 6.712 | 25594.91 | 43.7 | 43.4% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| SmallOps (norm/rope) | rms/rope | — | — | 5.176 | — | — | — | memory |
| FC_QKV (FP16) | gemm_kernel | 0.4673 | 10 | 4.673 | 45955.43 | 76.3 | 77.9% | compute |
| FC_linattn_b (INT4) | dq+gemm_kernel | 0.0613 | 30 | 1.839 | 2190.02 | 70.1 | 63.7% | memory |
| FC_linattn_a (INT4) | dq+gemm_kernel | 0.0599 | 30 | 1.797 | 2240.10 | 71.7 | 65.2% | memory |
| **TOTAL** | | | | **728.7** | | | | |
| | | | | TTFT=729 ms | | | **1405.2 tok/s** | |

### Prefill — S=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE3GEMM_fused (routed, INT4) | grouped_micro_gemm×3+scatter+gather | 20.1084 | 40 | 804.334 | 5126.19 | 22.6 | 20.6% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 5.1638 | 30 | 154.914 | 415.87 | 2.0 | 1.8% | memory |
| FC_linattn_qkv (INT4) | dq+gemm_kernel | 1.3632 | 30 | 40.897 | 50409.16 | 37.2 | 42.7% | compute |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 2.4883 | 10 | 24.883 | 13808.36 | 0.8 | 23.4% | compute |
| FC_linattn_z (INT4) | dq+gemm_kernel | 0.6786 | 30 | 20.357 | 50634.91 | 43.5 | 42.9% | compute |
| FC_linattn_out (INT4) | dq+gemm_kernel | 0.6723 | 30 | 20.169 | 51107.06 | 43.9 | 43.3% | compute |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 0.5020 | 40 | 20.079 | 25668.21 | 75.2 | 68.4% | memory |
| FC_O (FP16) | gemm_kernel | 1.4735 | 10 | 14.735 | 23318.51 | 28.5 | 39.5% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 10.688 | — | — | — | memory |
| FC_QKV (FP16) | gemm_kernel | 0.9196 | 10 | 9.196 | 46705.95 | 54.7 | 79.2% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| FC_linattn_a (INT4) | dq+gemm_kernel | 0.1141 | 30 | 3.423 | 2352.72 | 75.0 | 68.2% | memory |
| FC_linattn_b (INT4) | dq+gemm_kernel | 0.1132 | 30 | 3.396 | 2371.59 | 75.6 | 68.7% | memory |
| **TOTAL** | | | | **1132.3** | | | | |
| | | | | TTFT=1132 ms | | | **1808.8 tok/s** | |

### Prefill — S=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE3GEMM_fused (routed, INT4) | grouped_micro_gemm×3+scatter+gather | 24.9045 | 40 | 996.181 | 8277.95 | 19.8 | 18.0% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 10.3786 | 30 | 311.358 | 413.83 | 1.8 | 1.7% | memory |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 9.4211 | 10 | 94.211 | 14588.40 | 0.4 | 24.7% | compute |
| FC_linattn_qkv (INT4) | dq+gemm_kernel | 2.8727 | 30 | 86.180 | 47843.69 | 32.2 | 40.6% | compute |
| FC_linattn_z (INT4) | dq+gemm_kernel | 1.4117 | 30 | 42.350 | 48679.15 | 38.7 | 41.3% | compute |
| FC_linattn_out (INT4) | dq+gemm_kernel | 1.2898 | 30 | 38.694 | 53279.09 | 42.4 | 45.2% | compute |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 0.8956 | 40 | 35.823 | 28774.98 | 77.3 | 70.3% | memory |
| SmallOps (norm/rope) | rms/rope | — | — | 29.614 | — | — | — | memory |
| FC_O (FP16) | gemm_kernel | 2.6020 | 10 | 26.020 | 26409.99 | 25.8 | 44.8% | compute |
| FC_QKV (FP16) | gemm_kernel | 1.7586 | 10 | 17.586 | 48843.97 | 45.3 | 82.8% | compute |
| FC_linattn_b (INT4) | dq+gemm_kernel | 0.2280 | 30 | 6.840 | 2354.54 | 74.9 | 68.1% | memory |
| FC_linattn_a (INT4) | dq+gemm_kernel | 0.2278 | 30 | 6.834 | 2356.77 | 74.9 | 68.1% | memory |
| LM_head (INT8, 1 out tok) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| **TOTAL** | | | | **1696.9** | | | | |
| | | | | TTFT=1697 ms | | | **2413.8 tok/s** | |

### Prefill — S=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE3GEMM_fused (routed, INT4) | grouped_micro_gemm×3+scatter+gather | 41.6385 | 40 | 1665.541 | 9902.29 | 13.6 | 12.3% | memory |
| GatedDeltaNet | gated_delta_net_ref_sa | 20.6099 | 30 | 618.296 | 416.79 | 1.7 | 1.6% | memory |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 39.9931 | 10 | 399.931 | 13746.28 | 0.2 | 23.3% | compute |
| FC_linattn_qkv (INT4) | dq+gemm_kernel | 5.7023 | 30 | 171.068 | 48205.08 | 31.0 | 40.9% | compute |
| FC_linattn_z (INT4) | dq+gemm_kernel | 2.8334 | 30 | 85.001 | 48507.19 | 37.1 | 41.1% | compute |
| FC_linattn_out (INT4) | dq+gemm_kernel | 2.4987 | 30 | 74.962 | 55003.68 | 42.0 | 46.6% | compute |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 1.6336 | 40 | 65.344 | 31549.85 | 80.9 | 73.5% | memory |
| FC_O (FP16) | gemm_kernel | 6.1808 | 10 | 61.808 | 22236.53 | 19.0 | 37.7% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 60.606 | — | — | — | memory |
| FC_QKV (FP16) | gemm_kernel | 4.4892 | 10 | 44.892 | 38269.50 | 30.8 | 64.9% | compute |
| FC_linattn_b (INT4) | dq+gemm_kernel | 0.4314 | 30 | 12.943 | 2488.86 | 79.1 | 71.9% | memory |
| FC_linattn_a (INT4) | dq+gemm_kernel | 0.4201 | 30 | 12.603 | 2555.88 | 81.2 | 73.8% | memory |
| LM_head (INT8, 1 out tok) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| **TOTAL** | | | | **3278.2** | | | | |
| | | | | TTFT=3278 ms | | | **2498.9 tok/s** | |

### Prefill — S=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoE3GEMM_fused (routed, INT4) | grouped_micro_gemm×3+scatter+gather | 56.2818 | 40 | 2251.271 | 14651.88 | 12.7 | 12.4% | compute |
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 153.4928 | 10 | 1534.928 | 14326.56 | 0.1 | 24.3% | compute |
| GatedDeltaNet | gated_delta_net_ref_sa | 41.3175 | 30 | 1239.525 | 415.80 | 1.7 | 1.5% | memory |
| FC_linattn_qkv (INT4) | dq+gemm_kernel | 11.7560 | 30 | 352.679 | 46763.96 | 29.3 | 39.6% | compute |
| FC_linattn_z (INT4) | dq+gemm_kernel | 5.7411 | 30 | 172.233 | 47879.05 | 35.8 | 40.6% | compute |
| FC_linattn_out (INT4) | dq+gemm_kernel | 4.9798 | 30 | 149.395 | 55198.28 | 41.3 | 46.8% | compute |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 3.2202 | 40 | 128.808 | 32010.22 | 80.1 | 72.8% | memory |
| SmallOps (norm/rope) | rms/rope | — | — | 121.212 | — | — | — | memory |
| FC_O (FP16) | gemm_kernel | 11.6158 | 10 | 116.158 | 23664.23 | 18.8 | 40.1% | compute |
| FC_QKV (FP16) | gemm_kernel | 6.7474 | 10 | 67.474 | 50922.91 | 37.9 | 86.3% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| FC_linattn_a (INT4) | dq+gemm_kernel | — | 30 | 0.000 | — | — | — | N/A |
| FC_linattn_b (INT4) | dq+gemm_kernel | — | 30 | 0.000 | — | — | — | N/A |
| **TOTAL** | | | | **6138.9** | | | | |
| | | | | TTFT=6139 ms | | | **2668.9 tok/s** | |

### Prefill — S=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 623.6092 | 10 | 6236.092 | 14105.14 | 0.1 | 23.9% | compute |
| MoE3GEMM_fused (routed, INT4) | grouped_micro_gemm×3+scatter+gather | 106.3426 | 40 | 4253.702 | 15509.01 | 9.5 | 13.1% | compute |
| GatedDeltaNet | gated_delta_net_ref_sa | 83.2990 | 30 | 2498.971 | 412.49 | 1.6 | 1.5% | memory |
| FC_linattn_qkv (INT4) | dq+gemm_kernel | 24.4830 | 30 | 734.490 | 44909.19 | 27.8 | 38.1% | compute |
| FC_linattn_z (INT4) | dq+gemm_kernel | 11.7634 | 30 | 352.903 | 46734.29 | 34.6 | 39.6% | compute |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 7.8235 | 40 | 312.939 | 26351.22 | 65.1 | 59.2% | memory |
| FC_linattn_out (INT4) | dq+gemm_kernel | 10.3681 | 30 | 311.043 | 53023.83 | 39.3 | 44.9% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 242.424 | — | — | — | memory |
| FC_O (FP16) | gemm_kernel | 20.8616 | 10 | 208.616 | 26352.56 | 20.1 | 44.7% | compute |
| FC_QKV (FP16) | gemm_kernel | 13.2790 | 10 | 132.790 | 51750.65 | 37.0 | 87.7% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| FC_linattn_a (INT4) | dq+gemm_kernel | — | 30 | 0.000 | — | — | — | N/A |
| FC_linattn_b (INT4) | dq+gemm_kernel | — | 30 | 0.000 | — | — | — | N/A |
| **TOTAL** | | | | **15289.2** | | | | |
| | | | | TTFT=15289 ms | | | **2143.2 tok/s** | |

### Prefill — S=65,536

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 2500.0339 | 10 | 25000.339 | 14073.56 | 0.0 | 23.9% | compute |
| MoE3GEMM_fused (routed, INT4) | grouped_micro_gemm×3+scatter+gather | 242.3280 | 40 | 9693.122 | 13611.86 | 6.6 | 11.5% | compute |
| GatedDeltaNet | gated_delta_net_ref_sa | 168.5805 | 30 | 5057.414 | 407.64 | 1.6 | 1.5% | memory |
| FC_linattn_qkv (INT4) | dq+gemm_kernel | 51.3070 | 30 | 1539.209 | 42860.12 | 26.3 | 36.3% | compute |
| FC_linattn_z (INT4) | dq+gemm_kernel | 24.1184 | 30 | 723.552 | 45588.05 | 33.6 | 38.6% | compute |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 17.0649 | 40 | 682.595 | 24161.74 | 59.4 | 54.0% | memory |
| FC_linattn_out (INT4) | dq+gemm_kernel | 20.8332 | 30 | 624.996 | 52776.93 | 38.9 | 44.7% | compute |
| SmallOps (norm/rope) | rms/rope | — | — | 484.848 | — | — | — | memory |
| FC_O (FP16) | gemm_kernel | 43.5265 | 10 | 435.265 | 25260.72 | 18.9 | 42.8% | compute |
| FC_QKV (FP16) | gemm_kernel | 25.9227 | 10 | 259.227 | 53018.77 | 37.1 | 89.9% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| FC_linattn_a (INT4) | dq+gemm_kernel | — | 30 | 0.000 | — | — | — | N/A |
| FC_linattn_b (INT4) | dq+gemm_kernel | — | 30 | 0.000 | — | — | — | N/A |
| **TOTAL** | | | | **44505.8** | | | | |
| | | | | TTFT=44506 ms | | | **1472.5 tok/s** | |

### Prefill — S=131,072

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PagedAttention (FP16, causal) | sdpa_micro__prefill | 11343.7231 | 10 | 113437.231 | 12406.64 | 0.0 | 21.0% | compute |
| MoE3GEMM_fused (routed, INT4) | grouped_micro_gemm×3+scatter+gather | 552.6929 | 40 | 22107.718 | 11936.23 | 5.0 | 10.1% | compute |
| GatedDeltaNet [extrapolated] | gated_delta_net_ref_sa | 337.1609 | 30 | 10114.828 | 407.64 | 1.6 | 1.5% | memory |
| FC_linattn_qkv (INT4) | dq+gemm_kernel | 100.4180 | 30 | 3012.539 | 43797.41 | 26.8 | 37.1% | compute |
| FC_linattn_z (INT4) | dq+gemm_kernel | 51.6707 | 30 | 1550.121 | 42558.42 | 31.3 | 36.1% | compute |
| FC_linattn_out (INT4) | dq+gemm_kernel | 45.4747 | 30 | 1364.242 | 48357.05 | 35.5 | 41.0% | compute |
| SharedExpert_gate+up+down (FP16 FC×3) | gemm_kernel×3 | 28.7059 | 40 | 1148.235 | 28727.01 | 70.4 | 64.0% | memory |
| SmallOps (norm/rope) | rms/rope | — | — | 969.695 | — | — | — | memory |
| FC_O (FP16) | gemm_kernel | 84.1002 | 10 | 841.002 | 26147.65 | 19.4 | 44.3% | compute |
| FC_QKV (FP16) | gemm_kernel | 53.7828 | 10 | 537.828 | 51108.86 | 35.3 | 86.7% | compute |
| LM_head (INT8, 1 out tok) | gemm_kernel | 5.1901 | 1 | 5.190 | 195.97 | 99.6 | 90.6% | memory |
| FC_linattn_a (INT4) | dq+gemm_kernel | — | 30 | 0.000 | — | — | — | N/A |
| FC_linattn_b (INT4) | dq+gemm_kernel | — | 30 | 0.000 | — | — | — | N/A |
| **TOTAL** | | | | **155088.6** | | | | |
| | | | | TTFT=155089 ms | | | **845.1 tok/s** | |

---

## 10. Roofline Highlights (PTL decode, kv=4096)

| Op | Single ms | Calls | Total ms | GB/s | Eff% | Bound |
|---|---:|---:|---:|---:|---:|---|
| FC_linattn_qkv (INT4) | 0.0856 | 30 | 2.569 | 102.0 | 92.7% | memory |
| FC_O (FP16) | 0.1679 | 10 | 1.679 | 100.0 | 90.9% | memory |
| LM_head (INT8) | 5.1901 | 1 | 5.190 | 99.6 | 90.6% | memory |
| FC_linattn_z (INT4) | 0.0439 | 30 | 1.318 | 99.5 | 90.4% | memory |
| FC_linattn_out (INT4) | 0.0463 | 30 | 1.390 | 94.3 | 85.7% | memory |
| MoE3GEMM_fused (routed, INT4) | 0.1613 | 40 | 6.453 | 82.8 | 75.3% | memory |
| SharedExpert_gate+up+down (FP16 FC×3) | 0.0801 | 40 | 3.205 | 78.7 | 71.5% | memory |
| FC_QKV (FP16) | 0.3022 | 10 | 3.022 | 69.4 | 63.1% | memory |
| GatedDeltaNet | 0.0315 | 30 | 0.945 | 66.7 | 60.7% | memory |
| PagedAttention (INT8 KV) | 0.1684 | 10 | 1.684 | 24.9 | 22.6% | memory |
| FC_linattn_b (INT4) | 0.0038 | 30 | 0.115 | 9.9 | 9.0% | memory |
| FC_linattn_a (INT4) | 0.0038 | 30 | 0.115 | 9.9 | 9.0% | memory |

---

## 11. Comparison: F16 QKV/O vs INT4 QKV/O (PTL decode kv=4096)

| Op | INT4 (prev) ms | FP16 (new) ms | Δ ms | Notes |
|---|---:|---:|---:|---|
| FC_QKV × 10 layers | (INT4 ~0.128×10=1.28 ms est.) | 3.022 ms | — | 4× more weight bytes, lower arith intensity |
| FC_O × 10 layers   | (INT4 ~0.100×10=1.00 ms est.) | 1.679 ms | — | Same pattern |
| MoE routed + Shared × 40 layers | (fused 0.174 ms/L) | 9.659 ms total | — | Routed fused; shared unfused → 3 kernel launches overhead |

> FC_QKV and FC_O with FP16 weights are ~3× slower per layer than INT4 variants (higher BW demand).
> The shared expert with FP16 weights cannot be fused into MOE3GemmFusedCompressed, adding 3 separate kernel launches per layer.

---

_Generated by `build_report_ptl_f16qkvo.py`_