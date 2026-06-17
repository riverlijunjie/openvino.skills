# sdpa_micro__generate (PagedAttention MIXED) — layout-preserving optimization (PTL Xe3)

## 0. Headline result

**Hard constraint: the kernel's input/output data layout and format must not
change.** Only optimizations that leave the PA Q/K/V/output tensor layouts
byte-identical to the production kernel qualify. The optimization space is
therefore the kernel's launch/tiling configuration and its internal code — **not**
the I/O layout.

Using the evolutionary (MAP-Elites parameter trust-region) method, the
layout-preserving search re-derived the optimum for the PA MIXED decode
(`tokens=16`, `history=512`, `D=128`, GQA 4:1) on Panther Lake Xe3.

**Result: config `16,16,16,16,8,1,8,1`.** The true xe3 production dispatch is
`xehpc_h128_pa = {16,16,16,16,8,2,8,2}` (`wg_n=2` → `wg_tile_n=32`), which tiles
**32** query columns when only `q_new=16` are real — so half the KQ/VS micro-GEMM
work and half the loaded Q are masked away. Setting `wg_n=1` (`wg_tile_n=16`,
exactly `q_new`) removes that waste with **no layout change and no register
crash**.

| metric | production `8,2` | **elite `8,1`** |
|---|---|---|
| warm (steady-state) s1 / s8 | 0.0423 / 0.2827 ms | **0.0354 / 0.1660 ms** |
| cold s1 / s8 | 0.0568 / 0.3267 ms | **0.0503 / 0.2220 ms** |
| **warm speedup** s1 / s8 | — | **1.19× / 1.70×** |
| avg DRAM(unique) s1–s10 | 23.4 % | **30.8 % (+31.6 %)** |
| correctness (rel-L2) | 5.76e-4 PASS | 5.76e-4 PASS |
| crash-free s1–s10 | ✅ | ✅ |

Validated on the remote B390/Xe3 (iters=50). The elite wins at **every** batch
size s1–s10 with zero `OpenCL error`; the kernel `.cl` file is left **pristine**
(the win is a pure `--cfg`/dispatch change, no kernel-code edit).

## 1. What was measured

A standalone extraction of the OpenVINO GPU `sdpa_micro__generate` kernel
(`SDPAMicroGenerator(prefill=false, gqa_single_token=false)`) configured for the
**PagedAttention MIXED decode stage**, built and run on the remote Panther Lake
(Xe3) machine against `D:\river\moe\openvino`.

- **Why MIXED.** `get_paged_attention_stage()` selects the micro kernel when
  `num_tokens != num_seqs` **and** some `past_len != 0`. With `tokens` new query
  tokens per sequence (`tokens > 1`) and `seqs` sequences, `num_tokens =
  tokens·seqs ≠ seqs`, and `history > 0`, so every config below dispatches the
  `micro_sdpa` PA generate kernel with `IS_PAGED_ATTENTION=1` — exactly the path
  `paged_attention_opt.cpp` uses for multi-sequence decoding. (Single-token
  decode, `tokens=1`, instead hits the `pa_single_token` GENERATE stage and
  never reaches this micro kernel, so it is intentionally excluded.)

- **Kernel consistency.** The test inlines `sdpa_micro.cl` + its batch headers
  and the gemmstone micro-GEMM shims straight from
  `D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2`
  (`--kernel-dir`). Under `IS_PAGED_ATTENTION=1 && !IS_PREFILL` the body runs all
  **four** microkernels — `ugemm_kq` (cached int8 K·Q), `ugemm_kcq` (new-token
  f16 Kc·Q), `ugemm_vs` (cached int8 V·S), `ugemm_vcs` (new-token f16 Vc·S) — so
  the test emits four shims (`kq=0, vs=1, kcq=2, vcs=3`) and the executed code is
  byte-identical to OpenVINO's PA generate kernel.

- **Faithful PA inputs.** Paged int8 KV cache (block_size 16, per-token f16
  scale+zp interleaved in each block as `ADJUSTED_*_HEAD_SIZE = head+4`), plus the
  real index buffers (`subsequence_begins`, `past_lens`, `block_indices`,
  `block_indices_begins`, `blocked_indexes_start_and_gws_mapping`) and the
  12-argument kernel order, dispatch grid, and `1/sqrt(D)` f16 scale that the
  plugin builds. New tokens are passed as f16 `Kc`/`Vc`; past tokens live only in
  the int8 paged cache.

- **Parameters.** `tokens = 16`, `seqs ∈ {1..10}`, `history = 512`
  (`past_len`, rounded up to the KQ `wg_tile_m = 128`), `head-dim = 128`,
  `kv-heads = 8`, `heads = 32` (GQA 4:1), int8 BY_TOKEN compressed KV.
  `k_total = past_len + tokens = 528`.

- **Cache-cold averaging.** Each config is timed over N iterations (30–50 here)
  and averaged, reporting **both** a cache-cold device time and a warm
  (L3-resident) device time. To stop the small (~1–11 MB) KV working set from
  staying resident in the 16 MB L3 across iterations — which would report L3
  bandwidth instead of the true cold-cache decode time — a **`cache_flush`
  kernel** (a 128 MB read-modify-write, ≥ 4× the LLC) runs *before every timed
  cold iteration* to evict L2/L3. The `micro_sdpa` time itself is taken from its
  own OpenCL profiling event (so the flush is excluded), cross-checked against
  cliloader's per-kernel average (agreement < 1 %). The **`cold/warm` ratio** is
  the key diagnostic: it isolates DRAM traffic (cold-only) vs. compute (both).

## 2. Hardware

| | |
|---|---|
| Device | Intel(R) Arc(TM) B390 GPU (Panther Lake **Xe3**) |
| `gmdid` | `0x07800004` → xe3 (PTL) |
| Xe-cores / EUs | 12 cores × 8 = 96 EU, systolic/XMX = 1 |
| GPU clock | 2400 MHz |
| Memory BW (peak) | 112 GB/s (LPDDR5x, datasheet; read-probe ≈ 155–160 GB/s) |
| FP16 XMX (peak) | ≈ 119.5 TFLOP/s @2.4 GHz |
| LLC (L3) | 16 MB (`CL_DEVICE_GLOBAL_MEM_CACHE_SIZE`) |

## 3. Generated kernel / microkernels (host JIT)

All four micro-GEMMs are generated by gemmstone `selectGEMM` and fused into the
SPIR-V at runtime, exactly mirroring `init_microkernels()`:

| micro-GEMM | role | A ext | A layout | scale/zp | binary | grfMin |
|---|---|---|---|---|---:|---:|
| `kq`  | cached int8 K·Q | s8  | N | yes (aqGroupM=1, aqGroupK=D) | 95 376 B | 76 |
| `vs`  | cached int8 V·S | s8  | N | yes (aqGroupM=D, aqGroupK=1) | 36 576 B | 72 |
| `kcq` | new-token f16 Kc·Q | f16 | T | no | 5 984 B | 64 |
| `vcs` | new-token f16 Vc·S | f16 | N | no | 5 744 B | 64 |

The micro-GEMM tile is configurable via `--cfg` (8 ints: `unroll_m/n_kq,
unroll_m/n_vs, wg_m/n_kq, wg_m/n_vs`). The production default (`xehpc_h128_pa`)
is `unroll={16,16}`, `wg={8,2}` → `wg_tile=(128, 32)`; the **layout-preserving
winner** uses `wg={8,1}` → `wg_tile=(128, 16)`, matching `q_new=16` exactly with
no wasted columns. Dispatched kernel: `micro_sdpa SIMD16 REG128 SLM=17536 B`.

### Config structure constraints (from the kernel tile/RSELECT declarations)

A `--cfg` is structurally valid only if it satisfies all of:

1. `wg_tile_n_kq == wg_tile_n_vs` (`unroll_n_kq·wg_n_kq == unroll_n_vs·wg_n_vs`).
2. `wg_tile_m_vs = unroll_m_vs·wg_m_vs == 128` (= `D`; VS covers the full head).
3. `sg_per_wg_kq == sg_per_wg_vs` (`wg_m_kq·wg_n_kq == wg_m_vs·wg_n_vs`).
4. `unroll_m_kq = 16` (other values give NaN); `unroll_n = 16` (32 → grf ≥ 108 →
   `CL_OUT_OF_RESOURCES`, see §5d).

With all unrolls = 16, the only free knob is `wg_n ∈ {1,2,4}`, i.e. `wg_tile_n ∈
{16,32,64}`. Since `q_new = 16`, `wg_n=1` (`wg_tile_n=16`) is the exact match —
larger tiles only add masked work.

## 4. Results

All numbers are device-side `micro_sdpa` time (own profiling event), cache-cold
unless labelled *warm*. FLOPs = `4·B·heads·tokens·k_total·D` (Q·K + P·V).
**`DRAM(unique)`** = int8 KV read **once per KV head** (the true cold-cache DRAM
traffic under GQA 4:1) as a % of the 112 GB/s peak. **`KV/head`** counts the read
once per *query* head (includes the GQA 4:1 reuse, which is L3-served and can
exceed the DRAM peak — the direct fingerprint of in-kernel GQA L3 reuse).

### 4a. Layout-preserving config sweep (`wg_n` = `wg_tile_n`/16)

iters = 30, DRAM(unique) % of the 112 GB/s roofline. All 40 runs PASS
(rel-L2 ≈ 5.8e-4), zero crashes.

| seqs | **`8,1` (wg_tile_n=16)** | `8,2` production (n=32) | `8,4` (n=64) | `m_vs=32` `4,2,4,2` |
|----:|---:|---:|---:|---:|
| 1  | **20.6 %** | 18.6 % | 9.1 %  | 14.8 % |
| 2  | **30.5 %** | 17.8 % | 11.5 % | 15.4 % |
| 3  | **26.4 %** | 23.2 % | 12.8 % | 20.3 % |
| 4  | **27.8 %** | 22.6 % | 12.7 % | 19.8 % |
| 5  | **30.0 %** | 20.8 % | 13.7 % | 23.2 % |
| 6  | **32.8 %** | 25.0 % | 12.8 % | 23.4 % |
| 7  | **33.1 %** | 25.5 % | 13.5 % | 25.7 % |
| 8  | **36.3 %** | 25.6 % | 14.4 % | 24.1 % |
| 9  | **33.9 %** | 27.1 % | 14.7 % | 27.1 % |
| 10 | **36.4 %** | 27.4 % | 14.5 % | 26.5 % |
| **avg** | **30.8 %** | 23.4 % | 13.0 % | 22.0 % |

`8,1` wins at every batch size; **+31.6 % avg** over the production `8,2`. `8,4`
(`wg_tile_n=64`) wastes 4× the columns and collapses; `m_vs=32` is mid.

### 4b. Occupancy refinement (hold `wg_tile_n=16`, vary `sg_per_wg`)

iters = 30. `A` sg8 (elite), `E` sg4 (`unroll_m_vs=32`), `G` sg2
(`unroll_m_vs=64`). All PASS, zero crashes. (`unroll_m_vs=8` / sg16 fails to
build and is excluded.)

| seqs | `A` sg8 = `8,1` | `E` sg4 = `32,16,4,1,4,1` | `G` sg2 = `64,16,2,1,2,1` |
|----:|---:|---:|---:|
| 1  | **20.4 %** | 15.1 % | 10.8 % |
| 4  | 27.9 % | 27.1 % | **33.8 %** |
| 6  | 32.0 % | 35.5 % | **41.3 %** |
| 8  | **37.2 %** | 34.1 % | 33.8 % |
| 10 | 36.3 % | 39.2 % | 39.2 % |
| **avg** | **~31.7 %** | ~31.7 % | ~30.8 % |

The three tie on average, but `A` (sg8) is the most consistent and clearly best
at low batch (s1: 20.4 % vs 10.8 %), where the others starve the 96-EU GPU. **`A`
= `16,16,16,16,8,1,8,1` is the elite.**

### 4c. Head-to-head vs production (iters = 50, with warm/L3-resident floor)

| seqs | prod `8,2` cold / warm | elite `8,1` cold / warm | warm speedup |
|----:|---:|---:|---:|
| 1 | 0.0568 / 0.0423 ms | **0.0503 / 0.0354 ms** | **1.19×** |
| 8 | 0.3267 / 0.2827 ms | **0.2220 / 0.1660 ms** | **1.70×** |

Correctness identical (rel-L2 = 5.76e-4, both PASS). The warm (steady-state) gain
is the real-inference figure: it is where the production `wg_n=2` pays for its
masked column work and `wg_n=1` does not.

### Methodology cross-check (cliloader)

cliloader reports exactly two *enqueued* kernels — `micro_sdpa` and the
`cache_flush` helper (`ugemm_kq/kcq/vs/vcs` are *inlined*, not separate enqueues).
Its per-kernel average agrees with the profiling-event time to < 1 %. The
`cache_flush` (~1.05 ms, 128 MB RMW) is pure eviction overhead and is **excluded**
from every `micro_sdpa` number above.

## 5. Roofline analysis

### 5a. Which roofline, and why `DRAM(unique)`

The cached int8 KV is stored once per **KV head** but read by all 4 **query**
heads of its GQA group. The honest cold-cache DRAM metric counts each KV byte
**once per KV head** (`DRAM(unique)` = `B·Hkv·past_len·2·(D+4)` ≈ 1.1 MB at s1 →
10.8 MB at s10); the 4× group reuse is served from the 16 MB L3, not DRAM. That
is why `KV/head` (which counts all 4 query-head reads) can exceed **112 GB/s** —
it measures L3, not LPDDR5x. All "% of roofline" figures use `DRAM(unique)` ÷
112 GB/s (datasheet; a read-probe measured ~155 GB/s, so 112 is conservative).

### 5b. The kernel is compute/latency-bound here — `cold/warm` proves it

This short-context decode is **not** bandwidth-bound. The `cold/warm` ratio
(≈ 1.3–1.4 at high batch) separates the two costs:

- **warm (L3-resident)** = the matmul + softmax compute floor. The KQ/VS tiles
  (`128×16×128`) are too small to saturate XMX; the floor is **instruction /
  softmax / barrier latency**, not flops.
- **cold − warm** = the DRAM read of the unique KV, which already runs at
  **~near peak bandwidth** for the bytes it moves. The read and the compute run
  **mostly sequentially** (not overlapped), which is the real cap.

So `DRAM(unique) %` is limited by compute + un-overlapped first-block latency,
not by raw bandwidth: `util = unique_bytes / (compute_time + un-overlapped_read)`.

### 5c. Why `wg_n=1` is the win — and why the kernel is already near its ceiling

The production `8,2` tiles `wg_tile_n=32` but `q_new=16`, so the KQ/VS micro-GEMMs
and the Q load do **2× the necessary column work**, half of it masked. Dropping to
`wg_n=1` (`wg_tile_n=16`) removes that waste — lowering both the warm (compute)
floor and the cold time, which is the 1.19–1.70× warm speedup and the
23.4 % → 30.8 % avg DRAM(unique) lift.

Crucially, at the elite config the kernel already moves its KV at **~99 % of the
achievable per-head bandwidth** (`KV/head` ≈ 100 % of peak at s6+). `DRAM(unique)`
stays at ~31 % only because GQA reads each KV head 4× — and those redundant reads
are **mostly L3 hits** (the 1–11 MB KV set fits the 16 MB L3). Pure config tuning
therefore has **almost no headroom left**: the only structural lever is to cut the
4× redundancy, which is the (rejected) work in §7.

### 5d. The `CL_OUT_OF_RESOURCES` constraint and the crash-free choice

`unroll_n=32` raises the VS micro-GEMM to grf ≥ 108–190 and fails kernel placement
with **`CL_OUT_OF_RESOURCES` (-5)** at some batch sizes (a non-monotonic driver
residency quirk, not a correctness fault). Keeping **every micro-GEMM `unroll=16`
(grf ≈ 76)** removes the failure entirely: all swept `unroll_n=16` configs placed
at every batch size with zero `OpenCL error`. The elite `16,16,16,16,8,1,8,1`
holds `wg_tile_m_kq = wg_tile_m_vs = 128`, `wg_tile_n = 16 = q_new`, at the lowest
register footprint — the **guaranteed-safe optimum** under the zero-crash rule.

## 6. Reproduce

```bat
:: build (MSVC, VS dev prompt) — links D:\river\moe\openvino oneDNN/gemmstone lib
call build_test.bat

:: RECOMMENDED layout-preserving winner (wg_n=1), one batch:
sdpa_micro_generate_test.exe --tokens 16 --seqs 10 --history 512 ^
  --head-dim 128 --kv-heads 8 --heads 32 --cfg 16,16,16,16,8,1,8,1 --iters 50 ^
  --kernel-dir D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2

:: head-to-head vs the production default (wg_n=2):
sdpa_micro_generate_test.exe --tokens 16 --seqs 8 --history 512 ^
  --head-dim 128 --kv-heads 8 --heads 32 --cfg 16,16,16,16,8,2,8,2 --iters 50 ^
  --kernel-dir D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2

:: layout-preserving config sweep + occupancy refinement (s1-s10):
call run_sweep_nolayout.bat & type sweep_nolayout.txt | findstr /C:"DRAM(unique)" /C:"OpenCL error"
call run_sweep_refine.bat   & type sweep_refine.txt   | findstr /C:"DRAM(unique)" /C:"OpenCL error"
```

Every timed iteration runs the 128 MB `cache_flush` kernel first, so each
`micro_sdpa` measurement is cache-cold; the flush is event-excluded. All runs pass
`--kernel-dir D:\river\moe\openvino\src\plugins\intel_gpu\src\graph\impls\ocl_v2`
so the inlined kernel comes from that tree.

## 7. Notes & rejected alternatives

- **Final deliverable.** Layout-preserving config `16,16,16,16,8,1,8,1` — **1.19×
  (s1) – 1.70× (s8) warm speedup**, **+31.6 % avg DRAM(unique)** over the
  production default, identical correctness (rel-L2 = 5.76e-4), crash-free at all
  s1–s10. No I/O layout change; the kernel `.cl` is byte-identical to upstream.

- **Productionizing the win** (out of scope here — no commit/push per task policy):
  in `sdpa_gen_micro.cpp`, have the xe3 PA-decode path select `wg_n=1`
  (`wg_tile_n = q_new`) when `q_new ≤ 16`, instead of the fixed `xehpc_h128_pa`
  `wg_n=2`. Validate across other `q_new` values before shipping.

- **Rejected — in-kernel KV-share (layout-preserving).** Processing all 4 query
  heads of a KV group in one work-group (gathering Q from the native
  `[token][head][D]` layout, reading K/V once) is the only remaining structural
  lever. It is a major DPAS-kernel rewrite (block-outer/head-inner restructure, 4×
  running max/sum/accumulator state) for only **~+11 % cold-only** — in warm
  steady-state the redundant 4× KV reads are already L3 hits (§5c), so it barely
  helps the real-inference case. High risk, marginal reward → not implemented.

- **Rejected — GQA-shared KV via host repack** (`16,16,16,16,8,4,8,4`): reached
  ~34 % avg DRAM(unique) but **host-repacks Q** to
  `[seq][kv_head][token][head_in_group][D]` — a kernel **input-layout change**, so
  it violates the constraint and is disqualified.

- **Rejected — aggressive cfg `16,32,32,32,4,2,4,2`**: peaks ~48 % @s6 but
  `unroll_n=32` (grf=190) triggers **`CL_OUT_OF_RESOURCES` at s4/s8**. Rejected
  under the zero-crash requirement.

- **Why ~37 % is near the ceiling.** The decode is compute/latency-bound with a
  `cold/warm ≈ 1.3–1.4` overlap floor (§5b); the kernel already moves KV at ~99 %
  of achievable per-head bandwidth (§5c). Higher DRAM(unique) needs either the
  rejected KV-share (cold-only) or an algorithmic change (approx/lower-precision
  softmax) that would break faithfulness to the production kernel.

- The `cache_flush` kernel (128 MB RMW, ≥ 4× LLC) is essential for honest numbers
  — without it the small KV set stays L3-resident across iterations and the
  measured bandwidth is fake. Its ~1 ms/iter cost is event-excluded.
