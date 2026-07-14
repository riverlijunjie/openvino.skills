# Onyx (dense VLM text decoder) — Roofline on PTL 12Xe (2026-07-14)

**Platform**: PTL B390 iGPU — 12 Xe @ 2400 MHz, 110 GB/s LPDDR5x (Local_Admin@10.239.132.229)
**Model**: `onyx` — 52-layer dense SwiGLU decoder of the unified multimodal model (vision tower excluded from this text-generation roofline)

> **This report uses REAL ON-DEVICE MEASUREMENTS.** Every op was profiled in its own process on the PTL 12Xe target via cliloader 3.0.6 `--device-performance-timing` (mean kernel ns/iter), driving the Onyx shapes through OpenVINO's `FullyConnectedCompressed gemm_kernel` and `PagedAttention` OpenCL+micro-kernel primitives. Bytes/FLOPs are computed analytically from the same shapes to derive Eff% and the roofline floor. See *Data sources*.

- hidden=6656, layers=52, heads 32Q/2KV (GQA×16), head_dim=128, intermediate=19968, vocab=202048, SwiGLU (silu)
- MatMul weights INT4 g=128 / FP16 act; LM_head INT8 g=128 / FP16 act; KV cache **INT4 g=128** (user request)
- Attention has per-layer **QK-RMSNorm** + **sigmoid output gate** (extra `output_gate_proj` FC 6656→4096) and iRoPE: 39 sliding(SW=2048)/RoPE + 13 full/NoPE layers
- SDPA: PagedAttention OpenCL + micro_kernel

## Model parameters & weight shapes

| Field | Value | Notes |
|---|---:|---|
| `hidden_size` | 6656 | residual / activation channel |
| `num_hidden_layers` | 52 | 39 sliding/RoPE + 13 full/NoPE |
| `num_attention_heads` (NH) | 32 | Q heads |
| `num_key_value_heads` (NKV) | 2 | GQA: 16-way Q-per-KV sharing |
| `head_dim` (HD) | 128 | Q_dim = 4096, KV_dim = 256 |
| `intermediate_size` | 19968 | SwiGLU MLP hidden |
| `vocab_size` | 202048 | LM head N |
| `hidden_act` | silu | SwiGLU = down(silu(gate)·up) |
| `sliding_window` | 2048 | sliding layers cap KV at 2048 |
| `every_n_layers_nope` | 4 | every 4th layer is full/NoPE |
| `rope_theta` | 500000 | |
| `use_qk_norm` | true | scaleless RMSNorm on Q,K per head |
| `use_attn_output_gate` | true | extra FC 6656→4096 + sigmoid gate |
| `tie_word_embeddings` | false | LM head stored separately (INT8) |

Per-layer weight matrices (one decoder block) and global weights:

| Weight | Shape (K × N) | Quant | Bytes / instance | × Layers | Total MB |
|---|---:|---|---:|---:|---:|
| Embedding | 6656×202048 | INT8 g128 | 1,365.84 MB | 1 | 1,365.8 |
| FC_QKV (fused Q+K+V) | 6656×4608 | INT4 g128 | 15.93 MB | 52 | 828.6 |
| FC_OutGate (attn output gate) | 6656×4096 | INT4 g128 | 14.16 MB | 52 | 736.5 |
| FC_O (attn output) | 4096×6656 | INT4 g128 | 14.16 MB | 52 | 736.5 |
| MLP_gate (SwiGLU gate) | 6656×19968 | INT4 g128 | 69.05 MB | 52 | 3,590.6 |
| MLP_up (SwiGLU up) | 6656×19968 | INT4 g128 | 69.05 MB | 52 | 3,590.6 |
| MLP_down (SwiGLU down) | 19968×6656 | INT4 g128 | 69.05 MB | 52 | 3,590.6 |
| LM_Head | 6656×202048 | INT8 g128 | 1,365.84 MB | 1 | 1,365.8 |
| **Total static weights** |  |  |  |  | **15,805 MB** |

_FP16 baseline (no quant) ≈ 55,707 MB → quantized total 15,805 MB is 28% of FP16 size. During **decode** every FC + LM_head weight is re-read each token → ~14,439 MB of weight traffic per token (embedding not read), which sets the memory-bound decode floor._

## Theoretical roofline

| Metric | Value |
|---|---|
| FP16 XMX peak | 58.982 TFLOPS |
| INT8 XMX peak | 117.965 TOPS |
| Memory BW | 110 GB/s |
| Ridge point (FP16) | 536.2 FLOP/byte |
| Ridge point (INT8) | 1072.4 OP/byte |

_FP16 XMX = 12 Xe × 8 EU × 256 FLOP/cycle × 2.4 GHz. INT8 XMX = 2× FP16. A 5% overhead is deducted from each peak (achievable BW 104.5 GB/s, FP16 XMX 56.03 TFLOPS, INT8 XMX 112.07 TOPS) before computing t_theo._

## Data sources

- **All FC / PA / small-op rows are measured** on PTL 12Xe via cliloader (mean kernel ns/iter), one bench process per op (`fc_bench`, `pa_bench`, `small_ops_bench`). Iterations were sized so each op runs >1 s of GPU time where feasible; L2/L3 flush kernels evict cached weights between infers so decode FC measures true VRAM bandwidth.
- **Derived rows (not separately measured):** (a) PA_sliding prefill for S>2048 is scaled from the measured causal PA at S=2048 by the sliding-band pair ratio (band = S·SW − SW²/2); (b) PA_sliding decode for KV>2048 reuses the KV=2048 measurement (sliding window caps effective KV at 2048); (c) LM_head is measured once at M=1 and reused for both prefill (last token) and decode.
- Bytes/FLOPs are analytic (weight INT4/INT8 + FP16 scale/zp + FP16 act/out; PA with u4 KV). Eff% = achieved GB/s ÷ 110 (memory-bound) or GFLOPS ÷ XMX-peak (compute-bound). The theoretical floor uses a 5% overhead deduction.
- Run: 99 cliloader logs in `logs/`, parsed by `utils/parse_logs.py` → `parsed.json` → `build_report.py` → `performance_metrics.json`.

## Graph fusion notes

| Bench row | Real graph behaviour | Fused into | Standalone kernel? |
|---|---|---|---|
| `FC_QKV` | Q+K+V projection | fused QKV gemm | Yes |
| `FC_OutGate` | `output_gate_proj` (attn output gate) | separate FC | Yes |
| `MLP multiply` | silu(gate)·up | SwiGLU primitive | No — fused |
| `MLP gate/up/down` | 3 INT4 FCs | not fused (SwiGLU between) | Yes (×3) |
| `add` | 2 residual adds / layer | eltwise | Yes |
| `rmsnorm` | 4× / layer + qk_norm + final | RMS primitive | Yes |
| `qk_norm` | scaleless RMSNorm on Q,K | RMS primitive | Yes (folded in SmallOps) |
| `out-gate sigmoid·mul` | sigmoid(og)⊙attn | eltwise | folded in SmallOps |

## Token latency summary

### Prefill — TTFT and per-token amortized

| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|
| 1,024 | 956.01 | 0.956 | 0.9336 | 1,071 |
| 2,048 | 1,883.47 | 1.883 | 0.9197 | 1,087 |
| 4,096 | 3,927.02 | 3.927 | 0.9587 | 1,043 |
| 8,192 | 8,230.27 | 8.230 | 1.0047 | 995 |
| 16,384 | 17,489.38 | 17.489 | 1.0675 | 937 |
| 32,768 | 39,218.40 | 39.218 | 1.1969 | 836 |

### Decode — TPOT (per output token)

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
| 1,024 | 148.22 | 6.7 |
| 2,048 | 149.95 | 6.7 |
| 4,096 | 149.76 | 6.7 |
| 8,192 | 150.28 | 6.7 |
| 16,384 | 151.42 | 6.6 |
| 32,768 | 153.55 | 6.5 |

## Roofline: theoretical floor vs measured

### Decode (per output token)

| KV | theoretical (ms) | measured (ms) | achieved % |
|---:|---:|---:|---:|
| 1,024 | 139.15 | 148.22 | 93.9% |
| 2,048 | 139.28 | 149.95 | 92.9% |
| 4,096 | 139.35 | 149.76 | 93.0% |
| 8,192 | 139.49 | 150.28 | 92.8% |
| 16,384 | 139.76 | 151.42 | 92.3% |
| 32,768 | 140.30 | 153.55 | 91.4% |

### Prefill (TTFT over S tokens)

| S | theoretical (ms) | measured (ms) | achieved % |
|---:|---:|---:|---:|
| 1,024 | 505.94 | 956.01 | 52.9% |
| 2,048 | 1,018.28 | 1,883.47 | 54.1% |
| 4,096 | 2,066.28 | 3,927.02 | 52.6% |
| 8,192 | 4,218.51 | 8,230.27 | 51.3% |
| 16,384 | 8,692.36 | 17,489.38 | 49.7% |
| 32,768 | 18,490.03 | 39,218.40 | 47.1% |

## Decode tables (1 query token, KV = context length)

### Decode — KV=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `gemm_kernel` | 0.6846 | 52 | 35.599 | 388.3 | 100.9 | 91.8% | memory |
| MLP_gate | `gemm_kernel` | 0.6835 | 52 | 35.545 | 388.9 | 101.1 | 91.9% | memory |
| MLP_up | `gemm_kernel` | 0.6826 | 52 | 35.496 | 389.4 | 101.2 | 92.0% | memory |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| FC_QKV | `gemm_kernel` | 0.1699 | 52 | 8.835 | 361.0 | 93.9 | 85.4% | memory |
| FC_OutGate | `gemm_kernel` | 0.1430 | 52 | 7.434 | 381.4 | 99.2 | 90.2% | memory |
| FC_O | `gemm_kernel` | 0.1409 | 52 | 7.325 | 387.1 | 100.7 | 91.6% | memory |
| PA_sliding | `pa_kv_cache_update+paged_attention` | 0.0562 | 39 | 2.193 | 298.4 | 5.1 | 4.7% | memory |
| SmallOps | `rms/rope/gate/add` | — | 0 | 1.589 | — | — | — | memory |
| PA_full | `pa_kv_cache_update+paged_attention` | 0.0559 | 13 | 0.727 | 299.9 | 5.2 | 4.7% | memory |
| **TOTAL** |  |  |  | **148.22** |  |  |  |  |

_SwiGLU `multiply` is fused into the SwiGLU primitive; SmallOps aggregates rmsnorm/qk_norm/rope/out-gate/residual-add._

### Decode — KV=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `gemm_kernel` | 0.6846 | 52 | 35.599 | 388.3 | 100.9 | 91.8% | memory |
| MLP_gate | `gemm_kernel` | 0.6835 | 52 | 35.545 | 388.9 | 101.1 | 91.9% | memory |
| MLP_up | `gemm_kernel` | 0.6826 | 52 | 35.496 | 389.4 | 101.2 | 92.0% | memory |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| FC_QKV | `gemm_kernel` | 0.1699 | 52 | 8.835 | 361.0 | 93.9 | 85.4% | memory |
| FC_OutGate | `gemm_kernel` | 0.1430 | 52 | 7.434 | 381.4 | 99.2 | 90.2% | memory |
| FC_O | `gemm_kernel` | 0.1409 | 52 | 7.325 | 387.1 | 100.7 | 91.6% | memory |
| PA_sliding | `pa_kv_cache_update+paged_attention` | 0.0894 | 39 | 3.486 | 375.4 | 6.3 | 5.7% | memory |
| SmallOps | `rms/rope/gate/add` | — | 0 | 1.589 | — | — | — | memory |
| PA_full | `pa_kv_cache_update+paged_attention` | 0.0896 | 13 | 1.165 | 374.3 | 6.3 | 5.7% | memory |
| **TOTAL** |  |  |  | **149.95** |  |  |  |  |

_SwiGLU `multiply` is fused into the SwiGLU primitive; SmallOps aggregates rmsnorm/qk_norm/rope/out-gate/residual-add._

### Decode — KV=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `gemm_kernel` | 0.6846 | 52 | 35.599 | 388.3 | 100.9 | 91.8% | memory |
| MLP_gate | `gemm_kernel` | 0.6835 | 52 | 35.545 | 388.9 | 101.1 | 91.9% | memory |
| MLP_up | `gemm_kernel` | 0.6826 | 52 | 35.496 | 389.4 | 101.2 | 92.0% | memory |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| FC_QKV | `gemm_kernel` | 0.1699 | 52 | 8.835 | 361.0 | 93.9 | 85.4% | memory |
| FC_OutGate | `gemm_kernel` | 0.1430 | 52 | 7.434 | 381.4 | 99.2 | 90.2% | memory |
| FC_O | `gemm_kernel` | 0.1409 | 52 | 7.325 | 387.1 | 100.7 | 91.6% | memory |
| PA_sliding | `pa_kv_cache_update+paged_attention` | 0.0894 | 39 | 3.486 | 375.4 | 6.3 | 5.7% | memory |
| SmallOps | `rms/rope/gate/add` | — | 0 | 1.589 | — | — | — | memory |
| PA_full | `pa_kv_cache_update+paged_attention` | 0.0753 | 13 | 0.979 | 891.3 | 14.7 | 13.4% | memory |
| **TOTAL** |  |  |  | **149.76** |  |  |  |  |

_SwiGLU `multiply` is fused into the SwiGLU primitive; SmallOps aggregates rmsnorm/qk_norm/rope/out-gate/residual-add._

### Decode — KV=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `gemm_kernel` | 0.6846 | 52 | 35.599 | 388.3 | 100.9 | 91.8% | memory |
| MLP_gate | `gemm_kernel` | 0.6835 | 52 | 35.545 | 388.9 | 101.1 | 91.9% | memory |
| MLP_up | `gemm_kernel` | 0.6826 | 52 | 35.496 | 389.4 | 101.2 | 92.0% | memory |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| FC_QKV | `gemm_kernel` | 0.1699 | 52 | 8.835 | 361.0 | 93.9 | 85.4% | memory |
| FC_OutGate | `gemm_kernel` | 0.1430 | 52 | 7.434 | 381.4 | 99.2 | 90.2% | memory |
| FC_O | `gemm_kernel` | 0.1409 | 52 | 7.325 | 387.1 | 100.7 | 91.6% | memory |
| PA_sliding | `pa_kv_cache_update+paged_attention` | 0.0894 | 39 | 3.486 | 375.4 | 6.3 | 5.7% | memory |
| SmallOps | `rms/rope/gate/add` | — | 0 | 1.589 | — | — | — | memory |
| PA_full | `pa_kv_cache_update+paged_attention` | 0.1147 | 13 | 1.492 | 1,169.8 | 19.1 | 17.4% | memory |
| **TOTAL** |  |  |  | **150.28** |  |  |  |  |

_SwiGLU `multiply` is fused into the SwiGLU primitive; SmallOps aggregates rmsnorm/qk_norm/rope/out-gate/residual-add._

### Decode — KV=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `gemm_kernel` | 0.6846 | 52 | 35.599 | 388.3 | 100.9 | 91.8% | memory |
| MLP_gate | `gemm_kernel` | 0.6835 | 52 | 35.545 | 388.9 | 101.1 | 91.9% | memory |
| MLP_up | `gemm_kernel` | 0.6826 | 52 | 35.496 | 389.4 | 101.2 | 92.0% | memory |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| FC_QKV | `gemm_kernel` | 0.1699 | 52 | 8.835 | 361.0 | 93.9 | 85.4% | memory |
| FC_OutGate | `gemm_kernel` | 0.1430 | 52 | 7.434 | 381.4 | 99.2 | 90.2% | memory |
| FC_O | `gemm_kernel` | 0.1409 | 52 | 7.325 | 387.1 | 100.7 | 91.6% | memory |
| PA_sliding | `pa_kv_cache_update+paged_attention` | 0.0894 | 39 | 3.486 | 375.4 | 6.3 | 5.7% | memory |
| PA_full | `pa_kv_cache_update+paged_attention` | 0.2028 | 13 | 2.637 | 1,323.6 | 21.6 | 19.6% | memory |
| SmallOps | `rms/rope/gate/add` | — | 0 | 1.589 | — | — | — | memory |
| **TOTAL** |  |  |  | **151.42** |  |  |  |  |

_SwiGLU `multiply` is fused into the SwiGLU primitive; SmallOps aggregates rmsnorm/qk_norm/rope/out-gate/residual-add._

### Decode — KV=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `gemm_kernel` | 0.6846 | 52 | 35.599 | 388.3 | 100.9 | 91.8% | memory |
| MLP_gate | `gemm_kernel` | 0.6835 | 52 | 35.545 | 388.9 | 101.1 | 91.9% | memory |
| MLP_up | `gemm_kernel` | 0.6826 | 52 | 35.496 | 389.4 | 101.2 | 92.0% | memory |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| FC_QKV | `gemm_kernel` | 0.1699 | 52 | 8.835 | 361.0 | 93.9 | 85.4% | memory |
| FC_OutGate | `gemm_kernel` | 0.1430 | 52 | 7.434 | 381.4 | 99.2 | 90.2% | memory |
| FC_O | `gemm_kernel` | 0.1409 | 52 | 7.325 | 387.1 | 100.7 | 91.6% | memory |
| PA_full | `pa_kv_cache_update+paged_attention` | 0.3664 | 13 | 4.763 | 1,465.2 | 23.8 | 21.7% | memory |
| PA_sliding | `pa_kv_cache_update+paged_attention` | 0.0894 | 39 | 3.486 | 375.4 | 6.3 | 5.7% | memory |
| SmallOps | `rms/rope/gate/add` | — | 0 | 1.589 | — | — | — | memory |
| **TOTAL** |  |  |  | **153.55** |  |  |  |  |

_SwiGLU `multiply` is fused into the SwiGLU primitive; SmallOps aggregates rmsnorm/qk_norm/rope/out-gate/residual-add._

## Prefill tables (single forward over S tokens)

### Prefill — S=1,024

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `dq+gemm_kernel` | 4.6634 | 52 | 242.495 | 58,368.5 | 26.5 | 49.5% | compute |
| MLP_gate | `dq+gemm_kernel` | 4.0554 | 52 | 210.880 | 67,118.9 | 30.5 | 56.9% | compute |
| MLP_up | `dq+gemm_kernel` | 3.7902 | 52 | 197.092 | 71,814.5 | 32.6 | 60.9% | compute |
| SmallOps | `rms/rope/gate/add` | — | 0 | 105.589 | — | — | — | memory |
| FC_QKV | `dq+gemm_kernel` | 1.0381 | 52 | 53.979 | 60,510.7 | 37.6 | 51.3% | compute |
| FC_OutGate | `dq+gemm_kernel` | 0.9924 | 52 | 51.602 | 56,264.9 | 36.5 | 47.7% | compute |
| FC_O | `dq+gemm_kernel` | 0.8912 | 52 | 46.341 | 62,652.6 | 40.6 | 53.1% | compute |
| PA_sliding | `sdpa_micro_prefill` | 0.6645 | 39 | 25.915 | 12,939.9 | 25.7 | 21.9% | compute |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| PA_full | `sdpa_micro_prefill` | 0.6645 | 13 | 8.638 | 12,939.9 | 25.7 | 21.9% | compute |
| **TOTAL** |  |  |  | **956.01** |  |  |  |  |

_FC prefill uses the INT8-XMX `dynamic_quantize_gpu_opt + gemm_kernel` path._

### Prefill — S=2,048

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `dq+gemm_kernel` | 9.0497 | 52 | 470.584 | 60,155.3 | 19.7 | 51.0% | compute |
| MLP_up | `dq+gemm_kernel` | 7.3891 | 52 | 384.234 | 73,674.1 | 24.1 | 62.5% | compute |
| MLP_gate | `dq+gemm_kernel` | 7.3056 | 52 | 379.892 | 74,516.2 | 24.4 | 63.2% | compute |
| SmallOps | `rms/rope/gate/add` | — | 0 | 224.805 | — | — | — | memory |
| FC_QKV | `dq+gemm_kernel` | 1.9868 | 52 | 103.312 | 63,232.3 | 31.2 | 53.6% | compute |
| FC_OutGate | `dq+gemm_kernel` | 1.8352 | 52 | 95.428 | 60,850.0 | 31.7 | 51.6% | compute |
| FC_O | `dq+gemm_kernel` | 1.7563 | 52 | 91.329 | 63,581.1 | 33.1 | 53.9% | compute |
| PA_sliding | `sdpa_micro_prefill` | 2.3156 | 39 | 90.308 | 14,845.7 | 14.7 | 25.2% | compute |
| PA_full | `sdpa_micro_prefill` | 2.3156 | 13 | 30.103 | 14,845.7 | 14.7 | 25.2% | compute |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| **TOTAL** |  |  |  | **1,883.47** |  |  |  |  |

_FC prefill uses the INT8-XMX `dynamic_quantize_gpu_opt + gemm_kernel` path._

### Prefill — S=4,096

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `dq+gemm_kernel` | 18.5458 | 52 | 964.381 | 58,707.3 | 15.5 | 49.8% | compute |
| MLP_up | `dq+gemm_kernel` | 14.6937 | 52 | 764.073 | 74,097.9 | 19.5 | 62.8% | compute |
| MLP_gate | `dq+gemm_kernel` | 14.5274 | 52 | 755.426 | 74,946.2 | 19.8 | 63.5% | compute |
| SmallOps | `rms/rope/gate/add` | — | 0 | 460.572 | — | — | — | memory |
| PA_sliding | `sdpa_micro_prefill` | 6.9445 | 39 | 270.836 | 14,845.7 | 9.7 | 25.2% | compute |
| FC_QKV | `dq+gemm_kernel` | 4.0714 | 52 | 211.715 | 61,711.8 | 26.6 | 52.3% | compute |
| FC_OutGate | `dq+gemm_kernel` | 3.6391 | 52 | 189.232 | 61,372.4 | 28.1 | 52.0% | compute |
| FC_O | `dq+gemm_kernel` | 3.5034 | 52 | 182.177 | 63,748.9 | 29.2 | 54.0% | compute |
| PA_full | `sdpa_micro_prefill` | 8.8567 | 13 | 115.137 | 15,521.9 | 7.7 | 26.3% | compute |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| **TOTAL** |  |  |  | **3,927.02** |  |  |  |  |

_FC prefill uses the INT8-XMX `dynamic_quantize_gpu_opt + gemm_kernel` path._

### Prefill — S=8,192

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `dq+gemm_kernel` | 36.1591 | 52 | 1,880.275 | 60,221.3 | 14.0 | 51.1% | compute |
| MLP_gate | `dq+gemm_kernel` | 29.9959 | 52 | 1,559.785 | 72,595.0 | 16.8 | 61.5% | compute |
| MLP_up | `dq+gemm_kernel` | 29.4545 | 52 | 1,531.634 | 73,929.2 | 17.2 | 62.7% | compute |
| SmallOps | `rms/rope/gate/add` | — | 0 | 956.103 | — | — | — | memory |
| PA_sliding | `sdpa_micro_prefill` | 16.2023 | 39 | 631.892 | 14,845.7 | 8.3 | 25.2% | compute |
| FC_OutGate | `dq+gemm_kernel` | 8.7342 | 52 | 454.181 | 51,140.8 | 21.8 | 43.4% | compute |
| PA_full | `sdpa_micro_prefill` | 34.7387 | 13 | 451.604 | 15,827.4 | 3.9 | 26.8% | compute |
| FC_QKV | `dq+gemm_kernel` | 7.7664 | 52 | 403.853 | 64,703.2 | 25.8 | 54.8% | compute |
| FC_O | `dq+gemm_kernel` | 6.6821 | 52 | 347.471 | 66,846.4 | 28.5 | 56.7% | compute |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| **TOTAL** |  |  |  | **8,230.27** |  |  |  |  |

_FC prefill uses the INT8-XMX `dynamic_quantize_gpu_opt + gemm_kernel` path._

### Prefill — S=16,384

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MLP_down | `dq+gemm_kernel` | 72.3727 | 52 | 3,763.378 | 60,176.0 | 13.0 | 51.0% | compute |
| MLP_up | `dq+gemm_kernel` | 62.5012 | 52 | 3,250.061 | 69,680.2 | 15.1 | 59.1% | compute |
| MLP_gate | `dq+gemm_kernel` | 58.4750 | 52 | 3,040.701 | 74,477.9 | 16.1 | 63.1% | compute |
| SmallOps | `rms/rope/gate/add` | — | 0 | 1,919.687 | — | — | — | memory |
| PA_full | `sdpa_micro_prefill` | 139.5797 | 13 | 1,814.536 | 15,755.6 | 2.0 | 26.7% | compute |
| PA_sliding | `sdpa_micro_prefill` | 34.7180 | 39 | 1,354.003 | 14,845.7 | 7.7 | 25.2% | compute |
| FC_QKV | `dq+gemm_kernel` | 15.7768 | 52 | 820.396 | 63,702.4 | 24.4 | 54.0% | compute |
| FC_OutGate | `dq+gemm_kernel` | 14.6179 | 52 | 760.132 | 61,113.6 | 25.1 | 51.8% | compute |
| FC_O | `dq+gemm_kernel` | 14.4810 | 52 | 753.014 | 61,691.2 | 25.3 | 52.3% | compute |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| **TOTAL** |  |  |  | **17,489.38** |  |  |  |  |

_FC prefill uses the INT8-XMX `dynamic_quantize_gpu_opt + gemm_kernel` path._

### Prefill — S=32,768

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PA_full | `sdpa_micro_prefill` | 595.4375 | 13 | 7,740.688 | 14,772.9 | 0.9 | 25.0% | compute |
| MLP_down | `dq+gemm_kernel` | 148.2502 | 52 | 7,709.009 | 58,753.3 | 12.2 | 49.8% | compute |
| MLP_gate | `dq+gemm_kernel` | 117.7743 | 52 | 6,124.262 | 73,956.7 | 15.4 | 62.7% | compute |
| MLP_up | `dq+gemm_kernel` | 117.3978 | 52 | 6,104.688 | 74,193.8 | 15.5 | 62.9% | compute |
| SmallOps | `rms/rope/gate/add` | — | 0 | 4,120.103 | — | — | — | memory |
| PA_sliding | `sdpa_micro_prefill` | 71.7494 | 39 | 2,798.226 | 14,845.7 | 7.5 | 25.2% | compute |
| FC_QKV | `dq+gemm_kernel` | 31.9207 | 52 | 1,659.874 | 62,970.0 | 23.6 | 53.4% | compute |
| FC_OutGate | `dq+gemm_kernel` | 29.0757 | 52 | 1,511.936 | 61,450.2 | 24.7 | 52.1% | compute |
| FC_O | `dq+gemm_kernel` | 27.6181 | 52 | 1,436.141 | 64,693.3 | 26.0 | 54.8% | compute |
| LM_head | `gemm_kernel` | 13.4758 | 1 | 13.476 | 199.6 | 101.4 | 92.2% | memory |
| **TOTAL** |  |  |  | **39,218.40** |  |  |  |  |

_FC prefill uses the INT8-XMX `dynamic_quantize_gpu_opt + gemm_kernel` path._

## Op → kernel names (cliloader)

### Decode (M=1)

| op | kernel name(s) | launches/call |
|---|---|---:|
| FC_QKV | `gemm_kernel` | 1 |
| FC_OutGate | `gemm_kernel` | 1 |
| FC_O | `gemm_kernel` | 1 |
| MLP_gate/up/down | `gemm_kernel` | 3 |
| LM_head | `gemm_kernel` | 1 |
| PA_sliding | `pa_kv_cache_update + paged_attention` | 2 |
| PA_full | `pa_kv_cache_update + paged_attention` | 2 |
| SmallOps | `rms + rope + eltwise + activation` | — |

## Top contributors (sorted by total ms per inference)

### Decode

| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1,024 | MLP_down 35.6ms (24.0%) | MLP_gate 35.5ms (24.0%) | MLP_up 35.5ms (23.9%) |
| 2,048 | MLP_down 35.6ms (23.7%) | MLP_gate 35.5ms (23.7%) | MLP_up 35.5ms (23.7%) |
| 4,096 | MLP_down 35.6ms (23.8%) | MLP_gate 35.5ms (23.7%) | MLP_up 35.5ms (23.7%) |
| 8,192 | MLP_down 35.6ms (23.7%) | MLP_gate 35.5ms (23.7%) | MLP_up 35.5ms (23.6%) |
| 16,384 | MLP_down 35.6ms (23.5%) | MLP_gate 35.5ms (23.5%) | MLP_up 35.5ms (23.4%) |
| 32,768 | MLP_down 35.6ms (23.2%) | MLP_gate 35.5ms (23.1%) | MLP_up 35.5ms (23.1%) |

### Prefill

| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
| 1,024 | MLP_down 242.5ms (25.4%) | MLP_gate 210.9ms (22.1%) | MLP_up 197.1ms (20.6%) |
| 2,048 | MLP_down 470.6ms (25.0%) | MLP_up 384.2ms (20.4%) | MLP_gate 379.9ms (20.2%) |
| 4,096 | MLP_down 964.4ms (24.6%) | MLP_up 764.1ms (19.5%) | MLP_gate 755.4ms (19.2%) |
| 8,192 | MLP_down 1,880.3ms (22.8%) | MLP_gate 1,559.8ms (19.0%) | MLP_up 1,531.6ms (18.6%) |
| 16,384 | MLP_down 3,763.4ms (21.5%) | MLP_up 3,250.1ms (18.6%) | MLP_gate 3,040.7ms (17.4%) |
| 32,768 | PA_full 7,740.7ms (19.7%) | MLP_down 7,709.0ms (19.7%) | MLP_gate 6,124.3ms (15.6%) |

## End-to-end (prefill TTFT + 512-token decode)

| prompt P | TTFT (ms) | 512-tok decode (ms) | total (ms) | avg decode tok/s |
|---:|---:|---:|---:|---:|
| 1,024 | 956.01 | 75,887.75 | 76,843.76 | 6.7 |
| 2,048 | 1,883.47 | 76,773.96 | 78,657.43 | 6.7 |
| 4,096 | 3,927.02 | 76,678.48 | 80,605.51 | 6.7 |
| 8,192 | 8,230.27 | 76,941.03 | 85,171.30 | 6.7 |
| 16,384 | 17,489.38 | 77,527.26 | 95,016.64 | 6.6 |
| 32,768 | 39,218.40 | 78,616.13 | 117,834.53 | 6.5 |

## Key findings

- **Decode is hard memory-bound at ~6.7 tok/s** (148.22 ms/tok), essentially fixed across KV. It is set by re-reading ~14,439 MB of INT4/INT8 weights per token at 104.5 GB/s. MLP (gate+up+down ×52) is 72% of decode time; LM_head (INT8, 202K vocab) alone is 9%.
- **Decode achieves ~91% of the roofline floor** — little headroom without reducing weight bytes (INT4 LM_head, expert/weight pruning) or KV traffic.
- **Prefill is compute-bound and scales super-linearly**: TTFT achieves 53% of the roofline at S=1K and 47% at S=32K. **PA_full** (13 NoPE layers, full causal attention, FP16 micro-kernel) grows as S² and becomes the top TTFT contributor at long context — but it measured **25% XMX** at S=32K, much healthier than the ~7% seen on gemma4-12B (Onyx's 32 Q-heads give the micro-kernel far better occupancy than gemma's 16).
- The **attention output gate** adds a full extra FC (6656→4096) on every one of the 52 layers — ~11% of body FC weight traffic — a genuine Onyx-specific decode cost not present in a vanilla SwiGLU decoder.
- **Sliding layers (SW=2048)** keep sliding-PA cost flat for KV≥2048; only the 13 full layers' PA grows with context.

## Optimization levers (highest ROI first)

1. **INT4 LM_head** (currently INT8): halves the ~13 ms LM_head decode cost → ~5% faster decode, and shaves ~680 MB of static weights.
2. **Fuse `output_gate_proj` into FC_QKV** (single wide gemm 6656→8704): removes a separate kernel launch per layer and improves gemm efficiency at M=1.
3. **Long-context prefill**: PA_full is the #1 TTFT contributor at S>=16K (O(S^2)); it already runs at 25% XMX (healthy for a causal micro-kernel), so the lever is algorithmic — CM/flash tiling, KV sparsity, or exploiting that only 13/52 layers are full-attention.
4. **Speculative decoding / MTP**: decode is memory-bound and latency-fixed, so amortizing weight reads over multiple accepted tokens is the only way to raise effective tok/s materially.
5. **INT4 KV cache is already applied** — halves PA byte traffic vs INT8; keep it.

## Caveats & method

- **Measured** via cliloader Device Performance Timing (mean ns/iter), one process per op. PA is reported as the sum of its `pa_kv_cache_update` + `paged_attention` kernels (the roofline-relevant latency).
- FC weight bytes count INT4/INT8 weight + FP16 scale + INT4 zp (g=128) + FP16 act + FP16 out.
- PA bytes assume INT4 KV cache + FP16 Q/out. Decode FC is memory-bound (M=1); prefill FC is INT8-XMX compute-bound (S≥1024 all above the INT8 ridge).
- PA prefill full uses causal pairs S(S+1)/2; sliding uses band min(S,2048).
- LM_head runs once per token (last position in prefill, every step in decode).
- Vision tower / adapter are excluded — this is the text-generation roofline only.
- Target machine: Local_Admin@10.239.132.229 (PTL 12Xe, 2400 MHz).

## Reproduction

```bash
# 1. generate + copy the sweep, run on the PTL 12Xe target under cliloader
python3 utils/generate_onyx_runscript.py   # -> utils/run_onyx_ptl_12xe.bat
scp utils/run_onyx_ptl_12xe.bat Local_Admin@10.239.132.229:D:/river/moe/dev_roofline_profiling/utils/
ssh Local_Admin@10.239.132.229 'cd /d D:\river\moe\dev_roofline_profiling\utils && call run_onyx_ptl_12xe.bat'
# 2. pull logs back and build the report
scp -r Local_Admin@10.239.132.229:D:/river/moe/roofline_results/onyx/ptl_12xe outputs/onyx/logs
cd outputs/onyx
python3 ../../utils/parse_logs.py logs parsed.json
python3 build_report.py       # -> performance_metrics.json
python3 render_summary.py     # -> this SUMMARY
```
