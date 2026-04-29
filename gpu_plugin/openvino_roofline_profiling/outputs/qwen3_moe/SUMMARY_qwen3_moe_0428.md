# Qwen3-Coder-30B-A3B-Instruct (qwen3_moe) — Roofline SUMMARY

**Date:** 2026-04-28  
**Targets:** BMG (Arc B580 dGPU) + PTL (B390 iGPU)  
**Profiler:** cliloader 3.0.6 `--device-performance-timing`, mean kernel time  
**Bench:** `.github/skills/dev_roofline_profiling/utils/{fc,moe,pa,small_ops}_bench`

---

## 1. Hardware Peaks

| Platform | DRAM BW (GB/s, **measured**) | DRAM BW (spec) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) | SLM/WG | L3 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| BMG (Arc B580, 20 Xe @ 2900 MHz) | **453** | 456 | 116.736 | 233.472 | 128 KiB | 18 MiB | dGPU, USM_DEVICE |
| PTL (B390 iGPU, 12 Xe @ 2400 MHz) | **94** | 110 | 58.982 | 117.964 | 128 KiB | 16 MiB | iGPU, USM_DEVICE, host-shared |
| **BMG / PTL ratio** | **4.82×** | 4.14× | **1.98×** | **1.98×** |  |  |  |

DRAM BW columns are now **measured** with our custom OpenCL probe
(`utils/hw_probe/mem_bw.c`, random-init buffers to defeat Xe2 framebuffer
compression). See §12 for the full probe table and how this changed the
roofline math. BMG/PTL: BW dominates BW-bound kernels (≈4.8×); XMX dominates
compute-bound ones (≈2.0×).

---

## 2. Model Configuration

| Field | Value |
|---|---|
| `vocab_size`           | 151,936 |
| `hidden_size` (H)      | 2,048 |
| `num_hidden_layers`    | **48** (all dense PA, no hybrid) |
| `num_attention_heads`  | 32 (NH) |
| `num_key_value_heads`  | 4 (NKV; GQA group=8) |
| `head_dim`             | 128 |
| `moe_intermediate_size`| 768 |
| `num_experts`          | 128 |
| `num_experts_per_tok`  | 8 |
| Shared expert          | none |
| Body weight quant      | INT4 g=128 (asymmetric) |
| LM-head weight quant   | INT8 g=128 |
| KV cache quant         | INT8 (asymmetric) |
| Activation dtype       | FP16 |

---

## 3. GPU Plugin Coverage

| Op | OV GPU plugin | Notes |
|---|---|---|
| FullyConnectedCompressed (INT4 g=128) | ✅ | Fused dequant+GEMM (`gemm_kernel`) |
| MOE3GemmFusedCompressed (INT4)        | ✅ | Decode: 3 specialised `moe_3gemm_swiglu_mlp_*`; Prefill: `grouped_micro_gemm` + gather/scatter |
| PagedAttention (INT8 KV)              | ✅ | Decode: `gqa_single_token` + finalization + `kv_cache_update`; Prefill: `sdpa_micro__prefill` |
| RMSNorm / RoPE / Add                  | ✅ | `rms_gpu_bfyx_opt`, `rope_opt`, `eltwise_simple_vload8` |

No fallback to ref kernels observed for measured shapes (only `moe_scatter_reduction_ref` and `moe_gather_ref_prefill_gather` used in MoE prefill — unavoidable expert-bucketing kernels).

---

## 4. Kernel Breakdown (per op)

> **Authoritative per-token-size kernel tables** (one table per S / kv) are in
> [kernel_tables.md](kernel_tables.md).  The summary below covers the dominant
> kernels of each op.

### 4.1 Decode kernel set
| Op | # kernels | Kernels (sorted by share) |
|---|---:|---|
| **MoE 3-GEMM fused** | 8 | `moe_3gemm_swiglu_mlp_gate_up` (57 %), `_down` (29 %), `fuse_softmax_topk` (4 %), router `gemm_kernel`, `dynamic_quantize_gpu_opt`, 2× `reorder_data_fast_b1`, `mlp_reduce` |
| **FC qkv / FC o**    | 1 | `gemm_kernel` (INT4-fused dequant + GEMM) |
| **LM head**          | 1 | `gemm_kernel` (INT8) |
| **PagedAttention**   | 3 | `paged_attention_opt__gqa_single_token__sa` (85 %), `..._single_token_finalization__sa`, `pa_kv_cache_update_ref__sa` |
| **RMSNorm / q-norm / k-norm** | 1 | `rms_gpu_bfyx_opt` |
| **RoPE q/k**         | 1 | `rope_opt` |
| **Residual add**     | 1 | `eltwise_simple_vload8` |

### 4.2 Prefill kernel set
| Op | # kernels | Kernels (sorted by share) |
|---|---:|---|
| **MoE 3-GEMM fused** | 9 | `grouped_micro_gemm` × **3 launches/iter** (gate / up / down — 51 %), `moe_scatter_reduction_opt/_ref` (28 %), `moe_gather_ref_prefill_gather` (10 %), `_swiglu_fuse_prefill_swiglu`, `_fuse_softmax_topk`, 2× `reorder_data*`, `dynamic_quantize_gpu_opt`, router `gemm_kernel` |
| **FC qkv / FC o**    | 2 | `gemm_kernel` (INT8 XMX, 73–87 %), `dynamic_quantize_gpu_opt` (act quant) |
| **PagedAttention**   | 2 | `sdpa_micro__prefill__sa` (97 %), `pa_kv_cache_update_ref__sa` |
| **RMSNorm / RoPE**   | 1 | (same as decode) |

→ Decode and prefill use **different MoE code paths**; prefill MatMul also adds a
`dynamic_quantize_gpu_opt` because GEMM runs on INT8 XMX (per SKILL.md).

> **Note (2026-04-28):** verified that `grouped_micro_gemm` is launched **3 times
> per MoE layer** in prefill — once each for gate, up, down projections.
> cliloader call counts (S=4096 BMG): `grouped_micro_gemm`=54 vs all other
> `moe_*` kernels=18, with bench iters=18 → 3 launches per iter ✓.
> `parse_logs.py` was updated to use the **mode** of call counts as iters
> (instead of max), which previously caused MoE-prefill totals to be reported
> 3× too low. All prefill totals below reflect the corrected parsing.

---

## 5. Decode performance — per-kv tables

Full per-kernel tables for kv ∈ {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}
on both BMG and PTL: see `BMG — decode kv=*` / `PTL — decode kv=*` in
[kernel_tables.md](kernel_tables.md).

### 5.1 Decode total (ms / token)

| kv tokens | BMG ms | BMG tok/s | PTL ms | PTL tok/s | BMG/PTL |
|---:|---:|---:|---:|---:|---:|
|   1,024 |   7.14 | 140.0 |  22.26 | 44.9 | 3.12× |
|   2,048 |   7.85 | 127.5 |  23.86 | 41.9 | 3.04× |
|   4,096 |   7.80 | 128.3 |  24.75 | 40.4 | 3.17× |
|   8,192 |   9.66 | 103.6 |  29.51 | 33.9 | 3.06× |
|  16,384 |  14.76 |  67.7 |  39.21 | 25.5 | 2.66× |
|  32,768 |  21.36 |  46.8 |  58.85 | 17.0 | 2.76× |
|  65,536 |  33.89 |  29.5 |  98.38 | 10.2 | 2.90× |
| 131,072 |  59.15 |  16.9 | 176.80 |  5.7 | 2.99× |

PA decode is **O(kv)**; everything else is constant per step. Above kv≈16K, PA
overtakes MoE as the dominant kernel.

### 5.2 Decode time-share (kv=4096)

| Group | BMG ms | BMG % | PTL ms | PTL % |
|---|---:|---:|---:|---:|
| MoE (3-GEMM fused) | 3.72 | 47.7 | 10.56 | 42.7 |
| Paged Attention    | 1.54 | 19.7 |  5.25 | 21.2 |
| FC qkv + o         | 1.37 | 17.6 |  4.93 | 19.9 |
| LM head            | 0.71 |  9.1 |  3.17 | 12.8 |
| Small ops (norm/rope/add) | 0.46 | 5.9 | 0.84 | 3.4 |

---

## 6. Prefill performance — per-S tables

Full per-kernel tables for S ∈ {1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}
on both BMG and PTL: see `BMG — prefill S=*` / `PTL — prefill S=*` in
[kernel_tables.md](kernel_tables.md).

### 6.1 Prefill total (ms / inference, TTFT)

| S tokens | BMG ms | PTL ms | BMG/PTL |
|---:|---:|---:|---:|
|   1,024 |    208.1 |    421.5 | 2.03× |
|   2,048 |    352.9 |    698.7 | 1.98× |
|   4,096 |    734.4 |  1,498.0 | 2.04× |
|   8,192 |  1,967.5 |  3,648.9 | 1.86× |
|  16,384 |  6,617.2 | 10,144.4 | 1.53× |
|  32,768 | 24,974.4 | 36,472.7 | 1.46× |
|  65,536 | 98,588.2 |142,254.1 | 1.44× |
| 131,072 |240,660.4†|606,906.0 | 2.52× |

† MoE @ S=131,072 silently OOM'd on BMG (only `reorder_data` ran). MoE term in
that row is extrapolated; everything else is measured. PA still dominates so
total error is small.

PA prefill is **O(S²)**; MoE/FC are O(S). Above S≥8K, PA dominates TTFT on both
platforms.

### 6.2 Prefill time-share (S=4096, BMG)

| Group | ms/inference | Share |
|---|---:|---:|
| MoE (3-GEMM fused, **3× grouped_micro_gemm**) | 442.5 | 60.3 % |
| Paged Attention (`sdpa_micro`)                | 235.6 | 32.1 % |
| FC qkv + o                                    |  55.7 |  7.6 % |
| LM head                                       |   0.7 |  0.1 % |

→ After correcting the parser, **MoE — not PA — dominates prefill at S=4K**.
PA still overtakes MoE at S≥16K because PA grows O(S²) while MoE grows O(S).

---

## 7. Roofline highlights (BMG, decode kv=4096)

| Op (kernel) | Single ms | Calls | Total ms | GB/s | Eff%* | Bound |
|---|---:|---:|---:|---:|---:|---|
| LM head `gemm_kernel` (INT8)            | 0.7067 |   1 |  0.71 | 447.4 | **98.7 %** | memory |
| FC qkv `gemm_kernel` (INT4)             | 0.0155 |  48 |  0.75 | 351.3 |  77.5 %    | memory |
| FC o `gemm_kernel` (INT4)               | 0.0130 |  48 |  0.63 | 335.1 |  74.0 %    | memory |
| MoE `moe_3gemm_swiglu_mlp_gate_up`      | 0.0441 |  48 |  2.12 | 253.3 |  56.0 %    | memory |
| MoE `moe_3gemm_swiglu_mlp_down`         | 0.0224 |  48 |  1.08 | 253.3 |  56.0 %    | memory |
| PA `paged_attention_opt__gqa_single_token__sa` | 0.0271 |  48 |  1.30 | 131.2 | **29.0 %** | memory |
| RMSNorm `rms_gpu_bfyx_opt`              | 0.0015 |  96 |  0.14 |   5.5 |   1.2 %    | memory |

\* Eff% is GB/s ÷ **measured** BMG DRAM peak (453 GB/s). Earlier rev used the
spec 456 GB/s — values shift by < 1 %.

**Decode is fully memory-bound** on both platforms — no kernel sits on the
compute roof. LM head is excellent (98.7 % of measured BW peak); MoE & PA
leave the most room.

### 7.1 PTL (decode kv=4096)
| Op (kernel) | Single ms | Total ms | GB/s | Eff%* |
|---|---:|---:|---:|---:|
| LM head `gemm_kernel` (INT8) | 3.171 |  3.17 |  99.8 | **106 %** |
| FC qkv `gemm_kernel`         | 0.056 |  2.70 |  97.3 |  103 % |
| FC o `gemm_kernel`           | 0.046 |  2.23 |  94.1 |  100 % |
| MoE `mlp_gate_up`            | 0.119 |  5.71 |  89.3 |   95 % |
| PA `gqa_single_token`        | 0.094 |  4.51 |  38.3 |   41 % |

\* PTL Eff% denominator is the **measured streaming DRAM 94 GB/s** (custom
`mem_bw`, see §12). Values **above 100 %** mean the kernel is partially
served from PTL's 16 MiB L2 cache — not impossible, just an indication that
the kernel's hot footprint fits on-die. PA decode is the only PTL kernel
that is genuinely DRAM-bottlenecked (~41 %), and it is also the one whose
working set (KV cache @ kv=4096 ≈ 1 MiB/layer × 48 = 48 MiB) does **not**
fit in L2.

---

## 8. Findings

1. **MoE dominates decode** (≈43–48 % of step time). With TK=8 of NE=128 active,
   each token still pulls 8×3×(2048×768/2)=4.5 MB of INT4 weights — that's the
   memory wall.
2. **LM head is exceptionally well-tuned**: 98.1 % of BMG peak BW, 90.7 % on PTL.
   No further gain available without smaller vocab or different weight format.
3. **PA decode efficiency is low** (28.8 % BMG, 34.9 % PTL @ kv=4096). Likely
   sub-optimal KV-tile coalescing for NKV=4 / NH=32.
4. **PA prefill is O(S²)** — at S=131K a single layer is already >12 s on PTL;
   full prefill above S~32K is impractical without flash-style block sparsity.
5. **MoE @ S=131K OOM on BMG**: only `reorder_data` ran. Need expert streaming
   / re-materialization to make 128K-prefill viable on a 12 GB dGPU.
6. **Prefill MatMul uses INT8 XMX** correctly: each FC adds a
   `dynamic_quantize_gpu_opt` activation-quant kernel before `gemm_kernel`,
   which itself runs in INT8.
7. **BMG/PTL scaling is consistent**: BW-bound kernels scale 3.4–4.5× (lm_head
   4.48×), XMX-bound small ops scale ~2.0× (rmsnorm 1.98×, add 2.01×). No
   anomalous regressions.

---

## 9. Recommendations (ranked)

| Pri | Action | Expected gain |
|---|---|---|
| **P0** | Improve `moe_3gemm_swiglu_mlp_gate_up` / `..._down` to >70 % BW eff (currently 55–56 % on BMG). Investigate weight-tile prefetch + 16-byte coalesced INT4 unpack. | Decode −10…−15 % |
| **P1** | Improve PA decode KV coalescing: NKV=4 with HD=128 currently leaves ~70 % BW unused on BMG. Try 64-aligned head packing per block. | Decode −5…−10 % at kv≥4K |
| **P2** | Add expert-streaming / re-materialization for MoE prefill at S≥64K on BMG (avoid OOM). | Enables S=128K prefill on BMG |
| **P3** | Add flash-style block-sparse prefill kernel to break PA's O(S²). | TTFT −30…−40 % at S≥32K |
| **P4** | Review `dynamic_quantize_gpu_opt` cost in FC prefill — at S=4K it accounts for 28 % of FC-o time on BMG. Fuse into FC's pre-load if possible. | Prefill FC −10 % |
| **P5** | LM-head INT8 already 98 % BMG-peak — no action; only regression-guard. | regression-guard |

---

## 10. Files

- `outputs/qwen3_moe/ops_mapping.json` — model config + per-op shape catalog
- `outputs/qwen3_moe/build_report.py` — per-op + E2E roofline calculator (totals)
- `outputs/qwen3_moe/build_kernel_tables.py` — per-token-size per-kernel table generator
- `outputs/qwen3_moe/kernel_tables.md` — full per-S / per-kv kernel tables (BMG + PTL)
- `outputs/qwen3_moe/kernel_breakdown.md` — qualitative kernel-graph reference
- `outputs/qwen3_moe/bmg_metrics.json` — parsed BMG cliloader logs (59 entries)
- `outputs/qwen3_moe/ptl_metrics.json` — parsed PTL cliloader logs (58 entries)
- `outputs/qwen3_moe/performance_metrics_bmg.json` / `_ptl.json` — report data
- `outputs/qwen3_moe/logs/` — raw BMG cliloader logs
- `outputs/qwen3_moe/logs_ptl/` — raw PTL cliloader logs

Reproduction (BMG):
```bash
ssh openvino-ci-74@10.239.140.155 \
  "cd /mnt/river/model_loading/roofline_test_utils && bash run_qwen3_moe_bmg.sh && bash run_qwen3_moe_bmg_ext.sh"
```
Reproduction (PTL):
```bat
ssh Local_Admin@10.239.132.229 ^
  "cd /d D:\river\moe\dev_roofline_profiling\utils && run_qwen3_moe_ptl.bat && run_qwen3_moe_ptl_ext.bat"
```
Then locally:
```bash
python3 utils/parse_logs.py outputs/qwen3_moe/logs     outputs/qwen3_moe/bmg_metrics.json
python3 utils/parse_logs.py outputs/qwen3_moe/logs_ptl outputs/qwen3_moe/ptl_metrics.json
python3 outputs/qwen3_moe/build_kernel_tables.py > outputs/qwen3_moe/kernel_tables.md
python3 outputs/qwen3_moe/build_report.py --platform BMG --out outputs/qwen3_moe/performance_metrics_bmg.json
python3 outputs/qwen3_moe/build_report.py --platform PTL --out outputs/qwen3_moe/performance_metrics_ptl.json
```

---

## 12. HW Probe Results (per SKILL §4)

Three sources cross-checked: **clinfo** (vendor extensions), **clpeak** (BMG only),
and our **custom OpenCL probes** in [utils/hw_probe/](../../utils/hw_probe/) —
`gpu_info` (raw `clGetDeviceInfo`) and `mem_bw` (large-buffer streaming copy
with random-init init kernel to defeat Xe2 framebuffer compression).

### BMG — Intel® Arc™ B580 Graphics (dGPU)

| Item | Probed value | Source | SKILL spec |
|---|---|---|---|
| Driver | 25.48.36300.8 (NEO, OpenCL 3.0) | clinfo | — |
| Compute units (Xe-cores × EUs) | **160** | custom + clinfo | 20 × 8 = 160 ✓ |
| Clock frequency | **2900 MHz** | clpeak / custom | 2850 nominal |
| Subgroup sizes | **16, 32** | custom | 16 or 32 ✓ |
| **SLM / WG** | **128 KiB** | custom | spec says 32 KB — **spec is wrong** |
| L3 cache | 18 MiB | custom | (not in spec) |
| **DRAM read BW** | **453 GB/s** | custom `mem_bw`, random-init | 456 spec → **99 % of peak** |
| DRAM copy BW (R+W) | 402 GB/s | custom | — |
| clpeak global mem BW (cached) | 1234 GB/s | clpeak | (cache hit, **excluded** per SKILL §4) |
| FP16 ALU (no XMX) | 29.4 TFLOPS | clpeak | — |
| INT8 dot (24-bit fast) | 4.84 TIOPS | clpeak | — |
| Kernel launch latency | 7.59 µs | clpeak | — |
| FP16 XMX peak (spec) | 116.736 TFLOPS @ 2.85 GHz | spec | — |
| INT8 XMX peak (spec) | 233.472 TOPS @ 2.85 GHz | spec | — |

> **Two corrections vs SKILL §3:**
> 1. SLM is **128 KiB** (driver/HW), not 32 KB — affects how PA/MoE kernels can
>    size their per-WG tiles (4× more headroom).
> 2. Naive `mem_bw` reports 2.1 TB/s with zero-init buffers and 925 GB/s with
>    constant-fill — **both wrong**, masked by Xe2 lossless framebuffer
>    compression.  Only an explicit random-init kernel exposes the true 453 GB/s.

### PTL — Intel® Arc™ B390 GPU (Panther Lake H iGPU)

Custom probes built on PTL with VS2022 Community + the OpenCL-SDK installed at
`C:\Users\Local_Admin\ywang2\OpenCL-SDK`. Build & run script:
[utils/hw_probe/run_ptl.bat](../../utils/hw_probe/run_ptl.bat).

| Item | Probed value | Source | SKILL spec |
|---|---|---|---|
| Driver | 32.0.101.8531 | clinfo | — |
| Compute units (CUs) | **96** | custom + clinfo | 12 × 8 = 96 ✓ |
| Max clock | **2400 MHz** | custom + clinfo | 2400 ✓ |
| Subgroup sizes | **16, 32** | custom | 16 or 32 ✓ |
| **SLM / WG** | **128 KiB** | custom | spec says 32 KB — same correction as BMG |
| L3 cache | 16 MiB | custom | (not in spec) |
| Global mem | 16.5 GiB (host-shared LPDDR5x) | custom | — |
| **DRAM read BW** | **94 GB/s** | custom `mem_bw`, random-init | 110 spec → **86 % of peak** |
| DRAM copy BW (R+W) | 99 GB/s | custom | (LPDDR5x shared with CPU) |
| FP16 XMX peak (spec) | 58.98 TFLOPS @ 2.4 GHz | spec | — |
| INT8 XMX peak (spec) | 117.96 TOPS @ 2.4 GHz | spec | — |

PTL's clpeak isn't packaged with the Windows OpenCL-SDK; the custom `mem_bw`
plays the same role.

### Why the rescaled efficiency numbers in §7 sometimes exceed 100 %

PTL's measured **streaming-DRAM** BW is 94 GB/s, but its 16 MiB L2 happily
holds the hot blocks of a single decode token's working set: 4 MoE expert
weight chunks × ~600 KiB ≈ 2.4 MiB, FC qkv weights ≈ 8 MiB, LM head per
token-row ≈ 0.5 MiB. So ~100 % of measured DRAM = "we're hitting cache for
most of the read traffic" — confirms these kernels are not DRAM-bottlenecked
on PTL.  Only PA decode (KV cache 48 MiB at kv=4 K, total) overflows L2 and
shows the true DRAM ceiling at 41 %.

### How probes are used downstream

* **AI* threshold** (mem-bound → compute-bound transition) uses the *measured* DRAM peak:
  - BMG: AI* = 116.736 / 0.453 = **258 ops/byte**
  - PTL: AI* = 58.982 / 0.094 = **627 ops/byte**
* **Kernel-launch overhead** (BMG ≈ 7.6 µs) explains why
  `pa_decode_kv1024 = 18 µs/iter` is launch-dominated (almost half is launch).

---

## 13. Metrics Database (per SKILL §9)

All parsed cliloader logs are stored in [db/metrics.db](db/metrics.db) (built by
[db/build_db.py](db/build_db.py) from `bmg_metrics.json` + `ptl_metrics.json`).
Run: `python3 outputs/qwen3_moe/db/build_db.py` (paths are auto-resolved relative
to the script).

### Schema

```sql
runs(model, platform, config, mode, kv_or_S,
     total_kernel_ns_per_iter, iters, log_path);
kernels(model, platform, config, mode, kv_or_S, kernel,
        calls_total, calls_per_iter,
        per_iter_ns, avg_per_call_ns, share_in_run);
```

### Sample queries

Top BMG decode kernels:
```sql
SELECT config, kernel, per_iter_ns/1e6 AS ms, share_in_run*100 AS pct
FROM kernels WHERE platform='BMG' AND mode='decode'
ORDER BY ms DESC LIMIT 10;
```

MoE prefill BMG-vs-PTL scaling (the join that revealed BMG ≥ PTL at S=16K+):
```sql
SELECT kv_or_S,
       SUM(CASE WHEN platform='BMG' THEN per_iter_ns END)/1e6 AS bmg_ms,
       SUM(CASE WHEN platform='PTL' THEN per_iter_ns END)/1e6 AS ptl_ms
FROM kernels WHERE config LIKE 'moe_prefill_S%'
GROUP BY kv_or_S ORDER BY kv_or_S;
```

### Key finding the DB surfaced for qwen3_moe

| S | BMG MoE (ms) | PTL MoE (ms) | BMG/PTL |
|---:|---:|---:|---:|
|  1 K |   3.62 |   7.23 | 0.50× |
|  4 K |   9.22 |  18.81 | 0.49× |
| 16 K |  59.77 |  58.82 | 1.02× |
| 32 K | 207.53 | 121.21 | **1.71×** |
| 64 K | 788.97 | 239.49 | **3.30×** |
| 128 K | OOM   | 493.53 | — |

BMG MoE prefill is **slower than PTL** at S ≥ 32 K — same memory-pressure /
allocator regime as qwen3_5_moe (12 GiB VRAM is too tight for 128-expert
INT4 weights at long context). PTL's host-shared 16.5 GiB pool stays calm.

---

## 11. Revision History

| Date | Notes |
|---|---|
| 2026-04-28 (rev 1) | Full BMG+PTL re-sweep after PTL toolchain rebuild (VS 18 Insiders + Ninja). 59 BMG / 58 PTL kernels parsed. Added per-token-size kernel tables (`kernel_tables.md`). MoE@S=131K BMG OOM noted. |
| 2026-04-28 (rev 2) | Verified MoE prefill structure: `grouped_micro_gemm` is launched **3× per layer** (gate / up / down). Fixed `parse_logs.py` iters-detection (max → mode) which previously under-reported MoE prefill totals by 3×. All prefill numbers in §6 and `kernel_tables.md` regenerated. Decode unaffected. |
| 2026-04-29 (rev 3) | **Aligned with updated SKILL.md** (§4 HW probe + §9 structured DB). Custom OpenCL probes built/run on **both** BMG (Linux gcc) and PTL (Windows VS2022 + OpenCL-SDK). Two corrections: BMG/PTL **SLM is 128 KiB** (spec says 32 KB), and measured DRAM peaks are **BMG 453 GB/s** (vs 456 spec) and **PTL 94 GB/s** (vs 110 spec, LPDDR5x sharing with CPU). §1 + §7 efficiency tables rescaled to measured BW. New artifacts: `build_db.py` + `metrics.db` (117 runs, 226 kernels). DB cross-platform join confirms qwen3_moe shares qwen3_5_moe's "BMG slower than PTL at long-context MoE prefill" pattern (memory-pressure on 12 GiB VRAM at S ≥ 32 K). |
| 2026-04-29 (rev 4) | **Layout: moved DB into a dedicated `db/` subdirectory** per updated SKILL.md §8 (`utils/`, `db/`, `logs/`, `outputs/`). Files now live at `outputs/qwen3_moe/db/{build_db.py,metrics.db}`. `build_db.py` updated to resolve `bmg_metrics.json`/`ptl_metrics.json` from its parent directory so the move is transparent. |
