---
name: openvino_roofline_profiling
description: Roofline analysis of an OpenVINO model on Intel GPU. Builds a per-op bench harness, measures each kernel's achieved bandwidth/throughput via cliloader, compares to the theoretical roofline, and produces a per-op time-share breakdown. Use when the user wants to understand where inference time goes, verify a plugin optimization, or find compute/memory bottlenecks. Triggers include phrases like "roofline", "profile the model", "why is this kernel slow", "bandwidth efficiency", "per-op breakdown", "how much of decode is LM_Head / FC / attention".
---

# OpenVINO Roofline Profiling

A reproducible workflow for extracting a model's **per-op roofline** on an Intel GPU: what fraction of HW peak each kernel reaches, and what fraction of inference each op consumes.

## When to apply this skill
- User asks for a bandwidth/FLOPS efficiency report of a transformer on BMG / PTL / LNL / B390 / any Xe2/Xe3.
- User wants a time-share breakdown of decode or prefill.
- User suspects a specific op is the bottleneck and wants quantitative proof.
- User wants to compare two platforms (e.g., dGPU vs. iGPU) on the same model.

## What this skill produces
1. **`ops_mapping.json`** — enumerated ops with shapes, dtypes, and call counts per inference.
2. **Per-op bench binaries** (`fc_bench`, `pa_bench`, `small_ops_bench`) — one OV-level kernel per process, one invocation per shape variant, so kernels never contaminate each other.
3. **`performance_metrics.json`** — avg latency, achieved GB/s, achieved TFLOPS/TOPS, efficiency % for every (platform, op, phase).
4. **`SUMMARY.md`** — per-op and model-level tables, plus bottleneck ranking.

## Hard rules (always)
1. **Do not commit, push, or open PRs.** This is a measurement skill, not a code-change skill.
2. **Do not change weight layouts or plugin transformation defaults.** The goal is to observe what real inference does, not to alter it.
3. **Measure with `cliloader`.** Do not trust `std::chrono`-only measurements on GPU.
4. **Use *mean* kernel time from cliloader**, never min. Min over-estimates by capturing best-case cache state.
5. **All times in milliseconds.**

## Step-by-step workflow

### Step 1 — Collect hardware peaks
For each target GPU, compute:
- **FP16 XMX peak** = Xe_cores × EUs_per_core × FLOPs_per_cycle_FP16_XMX × clock_Hz
- **INT8 XMX peak** = Xe_cores × EUs_per_core × threads_per_EU × 4 × clock_Hz
- **DRAM BW peak** — vendor datasheet value (e.g., BMG B580 = 456 GB/s, PTL ≈ 110 GB/s).

Reference specs for commonly-used Intel GPUs live in the `intel-gpu-hw-info` skill; always cross-check the user's exact SKU before committing to numbers.

Special case: cliloader reports Arc B390 with the same device ID as PTL. Use the same roofline for both.

### Step 2 — Enumerate model ops
1. Obtain the model architecture (HF `transformers` source is the canonical reference).
2. For every significant op (MatMul, PagedAttention, RMSNorm, RoPE, residual Add, LM_Head) record:
   - Input/output shapes at each `S` (prefill sequence length) and each `kv_len` (decode).
   - Dtype (activation / weight / KV).
   - Call count per inference (accounting for per-layer invocations).
3. Save to `ops_mapping.json`. Use the provided `ops_mapping.json.example` as a schema.

Classify each op:
- **Memory-bound** if arithmetic intensity `FLOPs / bytes < DRAM_BW / compute_peak`.
- **Compute-bound** otherwise.

### Step 3 — Build per-op bench apps
Ship four C++ apps (already in `utils/`):

| Bench | Measures | Notes |
|---|---|---|
| `fc_bench M K N g iters warm bufs prec flush_mb` | One FullyConnectedCompressed (INT4 / INT8 per-group weights → dequant → MatMul) | Takes the sole role of LM_Head too (pass `u8`) |
| `pa_bench decode\|prefill M kv iters warm bufs kv_dtype` | One PagedAttention (KV update + attention compute) | Do **not** benchmark with "gemm + softmax + gemm" — the fused kernel's behavior won't match |
| `small_ops_bench <op> <shapes...> --iters --warmup --bufs` | RMSNorm, RoPE, eltwise Add | Do **not** benchmark `swiglu` separately — it fuses into the downstream FC |
| `moe_bench B S H I NE TK [g] [iters] [warm] [bufs] [flush_mb] [shared_I]` | One `moe_3gemm_fused_compressed` primitive (router + gate/up/down experts, optional always-on shared expert) | `shared_I=0` for plain MoE (Qwen3-MoE / Mixtral); `shared_I>0` for shared-expert MoE (Qwen3.5-MoE). Bench fails if fusion does not fire — see L12. |

Rules baked into the benches (see §"Lessons learned"):
- **Input/output tensors in USM_DEVICE on dGPU** (`RemoteContext::create_tensor(type, shape, {})`).
- **Multiple input/output buffers rotate** across infer requests to defeat activation L3 reuse.
- **An independent 64 MB f16 Relu "cache flush" model** runs between every measured infer, so weight constants are evicted from L2/L3 before each iteration. Without this, any kernel whose weight footprint fits the on-die cache reports > 100 % efficiency.

### Step 4 — Run with cliloader
- Set `CLI_DevicePerformanceTiming=1` and `CLI_DevicePerformanceTimingSkipUnmap=1`.
- Run **one bench per (op, shape) combination**, so each log is clean. Do not pack multiple shapes into one process.
- **Iteration budget:** choose iters so that `iters × theoretical_latency ≳ 1 s` of GPU kernel time. For decode kernels in µs, this means 8k–50k iters; for prefill kernels in ms, 50 iters suffices.
- Save one log per variant under `logs/<platform>/<op>_<shape>.log`.

Templates in `utils/run_all.sh.template` and `utils/run_all_ptl.bat.template` show the exact flag ordering and iter budgets.

### Step 5 — Parse and aggregate
Use `parse_logs_v2.py` (unchanged — it's generic):
- Skims the `Device Performance Timing Results` section of each log.
- Excludes host-side `clEnqueue*` and the cache-flush `activation_*` kernel.
- Sums `total_ns / max_calls` across all remaining kernels to get the per-iteration pure-kernel time.
- Writes `logs/<platform>_parsed.json`.

### Step 6 — Compute achieved efficiency
Use `generate_metrics.py.example` as a model for your own metrics producer. For each op compute:

| Quantity | Formula |
|---|---|
| bytes | sum of `sizeof(dtype) × num_elements` for every input/output/weight the kernel touches |
| FLOPs | 2×M×K×N for GEMMs; for PA and small ops, derive from algorithm |
| achieved GB/s | bytes / (avg_kernel_ms × 1e-3) / 1e9 |
| achieved TFLOPS | FLOPs / (avg_kernel_ms × 1e-3) / 1e12 |
| eff% (mem-bound) | achieved_GBs / peak_DRAM_BW × 100 |
| eff% (compute-bound) | achieved_TFLOPS / peak_XMX × 100 |

Write all numbers to `performance_metrics.json`.

Math convention (per SKILL tradition — many in-kernel math ops do not have accurate FLOP counts in literature):
- `exp` = 30 FLOPs
- `sin`, `cos` = 10 FLOPs
- `sqrt` = 10 FLOPs

### Step 7 — Build model totals and per-op time-share
For each phase (prefill at each S, decode at each kv_len), aggregate:
```
fc_ms         = Σ over (all FC ops) avg_latency × calls_per_layer × n_layers
pa_compute_ms = pa_compute_avg × n_layers
lm_head_ms    = lm_head_avg × 1           # once per inference
small_ms      = Σ small ops × n_layers
layer_ms      = fc_ms/n_layers + pa_ms/n_layers + small_ms/n_layers
model_ms      = layer_ms × n_layers + lm_head_ms
tokens_per_sec = 1000 / model_ms * (S for prefill, 1 for decode)
```

Then a per-op *time-share* breakdown table:
```
| op | total ms/inference | % of model_ms |
```
sorted descending. Fold everything < 0.1 % into a single "other tiny ops" row.

### Step 8 — Write SUMMARY.md
Sections, in this order:
1. Hardware peaks (cite SKILL.md formulas).
2. Model configuration (layers, head dims, compression scheme, KV layout).
3. Benchmark methodology — USM_DEVICE, rotating buffers, cache flush, iteration budget, cliloader config.
4. **Per-op metrics** (one table per phase): avg_latency_ms, achieved_gbs, achieved_tflops, efficiency_pct, bound.
5. **Model-level totals** — fc/pa/small/lm_head/model_ms/tokens_per_sec per phase per platform.
6. **Per-op time-share** — one table per (platform, phase, kv_len or S). Use the `kv_filter` pattern (see generate_summary.py.example) so each decode table only shows ops measured at that kv_len.
7. **Cross-platform comparison** — only include metrics that are comparable.
8. **Key findings** (free-form bullet list of the top-3 bottlenecks).
9. **Bench harness fix history** — keep a running log of measurement-correctness fixes so they are never re-introduced by a well-meaning refactor.
10. **Recommendations** — ranked list tied directly to the time-share numbers in §6.

## Lessons learned (must carry over to any new bench)

These were all uncovered through painful false-positive / false-negative measurements; violating any of them silently corrupts the roofline numbers.

### L1. USM_DEVICE for activations on dGPU
Default `InferRequest::get_input_tensor()` followed by CPU writes allocates **USM_HOST**. The kernel then reads activations across PCIe, which dominates per-iter time for decode-sized tensors and makes *every* body-FC look PCIe-bound.

Fix: allocate via `RemoteContext::create_tensor(type, shape, {})` with an empty params map — the plugin allocates USM_DEVICE internally. Leave the data uninitialized; roofline is about kernel timing, not correctness.

This affects dGPU only. On iGPU (PTL, LNL) system RAM *is* GPU RAM so USM_HOST and USM_DEVICE are equivalent, but using USM_DEVICE uniformly keeps one code path.

### L2. Rotate input/output tensors to defeat activation L3 reuse
Create `num_bufs ≥ 4` input+output tensors, one per `InferRequest`, and round-robin across iterations. With a single tensor the activations stay cache-resident after iteration 1 and decode small ops (RoPE, Add, RMSNorm) under-report bytes.

### L3. Flush the GPU cache between infers to defeat weight L2 reuse
Rotating input/output is not enough: the FC weight `Constant` is one shared object in the model graph. After iteration 1 it sits in L2 (BMG B580 has ~18 MB L2; a Qwen3-8B body FC's u4 weights are 8–25 MB, fully cacheable). Subsequent iterations measure *L2 bandwidth* and report > 100 % of DRAM peak.

Fix: compile a second, trivial model `Parameter(f16,[N]) → Relu → Result` with N × 2 B = flush_mb MB (default 64 MB, exceeds BMG L2 and PTL CPU LLC), and `infer()` it right before every measured FC infer. The flush kernel appears as `activation_opt_*` in cliloader and is excluded by `parse_logs_v2.py`'s hard-coded filter.

**Rejected alternative**: compiling `num_bufs` copies of the FC model with distinct weight data also defeats L2 reuse, but it distorts iGPU measurements (row-buffer thrashing across N × weight_MB allocations) and changes one platform's numbers in the act of fixing another's. The cache-flush approach is platform-neutral.

### L4. Choose iters so GPU kernel time ≥ 1 s
Cliloader reports averages over *all* `clEnqueue` invocations it saw. If your bench runs 50 iters of a 20 µs decode kernel you have 1 ms of GPU time — noise dominates. Aim for 1 s: `iters = ceil(1000 ms / avg_latency_ms)`. Typical budgets (see template):
- Body FC decode: 8 000
- LM_Head decode: 1 000 (single op already costs ms)
- PA decode: 20 000 (per-op cost is µs)
- Small ops decode: 50 000
- Prefill ops: 50 (each iter already multi-ms)

### L5. Use mean, not min
The `Average (ns)` column in cliloader is what you want. `Min (ns)` picks the iteration with warmest caches and is optimistic by 5–40 %.

### L6. LM_Head is decode-only
In production, LM_Head runs once per generated token, always on a single vector. Do **not** run LM_Head at S = 1024 / 2048 / etc.; allocating 8 × N_vocab×hidden in VRAM on dGPU blows up memory, and there is no real inference path that hits those shapes.

Run exactly one LM_Head config: `fc_bench 1 hidden_dim N_vocab group_size iters warm bufs u8 flush_mb`. Reuse the same number for "prefill lm_head" totals (it runs once per inference in both phases).

### L7. Don't profile `swiglu_ref` / `multiply` as a standalone op
These fuse into the upstream/downstream kernel (SwiGLU gate+up merges into the Gate FC or a following eltwise-fused kernel in recent plugins). A standalone bench of them reports an artificial BW-bound number that is not in any real graph.

### L8. Split PA into KV-update + compute
The fused PA kernel reports two logical sub-steps:
- `kv_cache_update` — quantize current token's K/V and write to block table.
- `pa_*_opt` — the attention compute over cached blocks.

Report them separately in metrics; they have different bounds (kv_update is memory-bound, compute is balanced).

### L9. Dynamic-quantize kernel is real work
Prefill with activation INT8 XMX path emits `dynamic_quantize_gpu_opt` before each body FC. It consumes measurable time (~1 % at small S, up to ~5 % at S=8192). Include it in the per-op breakdown and the FC time share — don't hide it inside `fc_ms`.

### L10. Efficiency > 100 % is always a bug
A single value > 100 % means the bytes/FLOPs model is wrong, or L2/L3 is hiding traffic, or the kernel is fused with something else. Do **not** publish a report with any eff > 100 %. Either fix the measurement (L1–L3), fix the byte-count model (fused kernel? change-of-dtype path?), or clearly annotate the row as "micro-op, theoretical bytes under-estimate actual traffic".

### L11. Strip tiny one-off kernels from per-iter math
`reorder_data_*` runs once at graph init to transpose weights to the plugin-preferred layout. It appears in cliloader with `calls=1` while the measured kernel shows `calls=iters`. `parse_logs_v2.py` uses `per_iter_ns = total_ns / max_calls` so the reorder naturally contributes `total_ns / iters` ≈ 0. Don't special-case; the division handles it.

### L12. MoE bench: build the *exact* subgraph the plugin's fusion chain expects, and verify fusion fired
The GPU plugin emits **one** primitive `moe_3gemm_fused_compressed` for the MoE block, implemented by 4–6 OpenCL kernels (`moe_3gemm_swiglu_fuse_softmax_topk_*`, `moe_3gemm_swiglu_mlp_gate_up_*`, `moe_3gemm_swiglu_mlp_down_*`, `moe_3gemm_swiglu_mlp_reduce_*`, plus an external router `gemm_kernel`). Per-expert MatMuls do **not** exist at runtime, so they must NOT be profiled as separate FCs. The fusion only fires if the IR has the canonical pattern:

```
input [B, S, H]
  ├ experts_reshape → Tile(NE,1) → Reshape(NE, B*S, H)
  │   ├ gate_matmul (u4-grouped weights, transpose_b=true) → Swish
  │   ├ up_matmul   (u4-grouped weights, transpose_b=true)
  │   │     └ Multiply (SwiGLU)
  │   └ down_matmul (u4-grouped weights, transpose_b=true)
  └ router_matmul → Softmax(axis=1) → TopK → normalize → ScatterElementsUpdate → Transpose → Reshape → Unsqueeze
      → Multiply(end_reshape, unsq_routing) → ReduceSum(axis=0)         ← MatMulExpertsFusion's pattern root
```

u4 weights MUST go through the asymmetric chain `Constant(u4) → Convert(f16) → Subtract(zp_u4→f16) → Multiply(scale_f16) → Reshape(3D) → Convert(f32)`. The trailing `Convert(f32)` is **mandatory** — `ConvertMOEToMOECompressed`'s matcher hard-requires f32 there even though the activation precision is f16.

**Validate fusion at runtime, fail loudly otherwise.** After `compile_model`, walk `compiled.get_runtime_model()` and confirm a node with `layerType == "moe_3gemm_fused_compressed"` exists. If not, the bench is silently measuring the unfused fallback (separate FCs + scatter + reduce) and every later number is wrong. Dump the runtime IR to a tmp file and exit non-zero.

#### L12b. Shared-expert MoE: feed `Add` directly from `ReduceSum`, not from a post-reduce `Reshape`
For models with an always-on shared expert (Qwen3.5-MoE, DeepSeek-V2/V3 style), the plugin runs `FuseMOESharedExpert` *before* `ConvertMOEToMOECompressed` to absorb the shared FFN's gate/up/down weights into the same fused primitive (10–11 inputs become 22–23). Its pattern is `Add(moe_m, shared_down_m)` where `moe_m` matches the **internal `MOE` op** — i.e. the node that `MatMulExpertsFusion` produces by replacing its pattern root, **`ReduceSum`**.

If you put a `Reshape(reduce, ...)` in between (a natural reflex when squashing `[B*S, H]` back to `[B, S, H]`), the `Add` no longer consumes the MOE op directly and `FuseMOESharedExpert` silently bails. The runtime graph then shows separate `SharedGateMatMul`, `SharedUpMatMul`, `SharedDownMatMul` `FullyConnected` nodes alongside the fused MOE — you are double-counting compute.

**Fix:** when `shared_I > 0`, route the `Add` directly off `ReduceSum`. Drop the post-reduce `Reshape`. The plugin handles output shape internally. Validate by asserting **no** runtime nodes have friendly names `SharedGate/Up/DownMatMul` after compile. (See `moe_bench/main.cpp` and the reference unit test `moe_3gemm_compressed_gpu_shared_random` in `src/plugins/intel_gpu/tests/unit/test_cases/moe_3gemm_gpu_test.cpp` for the canonical 23-input layout.)

#### L12c. MoE bench bytes/FLOPs accounting
At decode (M=1) the fused primitive is sparse: only `top_k` of `NE` experts run, plus the (always-on) shared expert if present. So the **active** weight traffic per layer is:
```
bytes_decode  = TK · per_expert_bytes  +  [shared_expert_bytes]  +  router_bytes
              ( + small activation IO)
```
where `per_expert_bytes = 2 · (H·I/2 + (H·I/g)·2 + (H·I/g)·1) + (I·H/2 + (I·H/g)·2 + (I·H/g)·1)` for u4 g=128 asymmetric (gate + up + down).

At prefill (S≫TK) every expert is hit by at least one token, so weight traffic saturates to the full pool:
```
bytes_prefill = NE · per_expert_bytes  +  [shared_expert_bytes]  +  router_bytes
FLOPs_prefill = 6 · S · (TK + has_shared) · H · I
```
The `(TK + has_shared)` factor in compute matters: a shared-expert layer is **9/8 ×** more compute than a plain top-8 layer at the same `(H, I)`. Don't forget it in compute-bound regions.

## Remote-execution pattern

This skill usually targets a remote machine (so the user's workstation is not the profiling target). Typical loop:
1. `scp` updated `*.cpp`, `*.sh`, `*.bat` to the remote.
2. On remote: `cd build && make <target> -j`.
3. On remote: run the `run_all` driver. Capture log dir.
4. `scp` back `logs/<platform>/*.log`.
5. Local: `parse_logs_v2.py` → `generate_metrics.py` → `generate_summary.py`.

Default remote deployment paths are project-specific (store them in a companion SKILL for your lab); do not hard-code them in this generic skill.

## Files in this skill

```
openvino_roofline_profiling/
├── SKILL.md                           — this file (methodology, no model-specific data)
├── README.md                          — quick-start + CLI reference
└── utils/
│   ├── CMakeLists.txt                 — builds the four bench binaries
│   ├── fc_bench/main.cpp              — FullyConnectedCompressed bench (INT4/INT8 weights)
│   ├── pa_bench/main.cpp              — PagedAttention bench
│   ├── small_ops_bench/main.cpp       — RMSNorm, RoPE, Add, …
│   ├── moe_bench/main.cpp             — MOE3GemmFusedCompressed bench (±shared expert)
    ├── parse_logs_v2.py               — cliloader log → kernel stats JSON
    ├── run_all.sh.template            — MODEL-AGNOSTIC Linux driver (do NOT edit per model)
    ├── run_all_ptl.bat.template       — MODEL-AGNOSTIC Windows driver (do NOT edit per model)
    ├── run_config.sh.example          — per-model + per-lab config (paths, op list)
    ├── run_all.sh.example             — fully populated reference driver (Qwen3-8B, BMG)
    ├── run_all_ptl.bat.example        — fully populated reference driver (Qwen3-8B, PTL)
    ├── ops_mapping.json.example       — reference op catalog (Qwen3-8B)
    ├── generate_metrics.py.example    — reference metrics aggregator (Qwen3-8B)
    └── generate_summary.py.example    — reference SUMMARY.md generator (Qwen3-8B)
```

**Clean separation of model-agnostic vs. model-specific:**

| Kind | Files | Edit per model? |
|---|---|---|
| Bench source (C++) | `fc_bench/`, `pa_bench/`, `small_ops_bench/`, `moe_bench/`, `CMakeLists.txt` | No — universal |
| Log parser | `parse_logs_v2.py` | No — universal |
| Runner template | `run_all.sh.template`, `run_all_ptl.bat.template` | No — universal |
| Per-model config | `run_config.sh` (copy from `.example`) | **Yes** — op list + paths |
| Aggregation / rendering | `generate_metrics.py`, `generate_summary.py` | **Yes** — copy from `.example` and adapt to the model's op catalog and HW targets |
| Op catalog | `ops_mapping.json` | **Yes** — copy from `.example` and populate for the model |

## Adapting to a new model

1. **Extract the op catalog.** Read the model's HF `transformers` source + OpenVINO verbose log. List every MatMul / PagedAttention / RMSNorm / RoPE / Add / LM_Head with its `(M, K, N)` or equivalent shape, dtype, call count per layer, and occurrences per inference. Save as `ops_mapping.json`.
2. **Write `run_config.sh`** (Linux) or `run_config.bat` (Windows) from the `.example`, updating:
   - `OPENVINO_SETUPVARS`, `CLILOADER`, `BENCH_DIR`, `RESULTS_DIR`
   - `FC_CONFIGS`, `PA_CONFIGS`, `SM_CONFIGS` entries for this model
   - Iteration budgets (default Qwen-class numbers are fine for most 7B–14B dense transformers)
3. **Run `run_all.sh.template`** (or the `.bat`). Do not modify it.
4. **Adapt `generate_metrics.py.example`** → `generate_metrics.py` to:
   - Enumerate the model's per-phase op list matching your `ops_mapping.json`.
   - Compute bytes/FLOPs using the per-op formulas (FC = `2MKN` compute + quant-weight bytes; PA = KV I/O; small ops = your algorithm).
5. **Adapt `generate_summary.py.example`** → `generate_summary.py` to emit the tables you want.

The agent reading this SKILL is expected to perform steps 1, 2, 4, 5 *programmatically* for the user's model — the user should not have to hand-edit the Python aggregators.

## Reference skills
- `intel-gpu-hw-info` for Xe2 / Xe3 architecture numbers.
- `dev_cliloader_analysis` for interpreting raw cliloader output.

Keep explanations conversational. For complex concepts, use multiple analogies.
