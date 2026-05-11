---
name: openvino_roofline_profiling
description: |
  Standalone OpenVINO roofline profiling toolkit for Intel GPUs and
  HuggingFace models. Use this skill when the user wants end-to-end roofline
  analysis, per-op/per-kernel breakdown, hardware-vs-kernel efficiency,
  bottleneck diagnosis, or optimization prioritization for OpenVINO GPU
  inference. It generalizes across Intel GPU families (iGPU/dGPU,
  Xe-LP/Xe-HPG/Xe2/Xe3, with or without XMX) and across HuggingFace
  architectures by decomposing the model into canonical operator families,
  profiling each family with reusable OpenVINO micro-benches, and aggregating
  the results into model-level reports plus a shared SQLite database. Typical
  triggers: "roofline", "profile this model on Intel GPU", "which kernels
  dominate", "memory-bound vs compute-bound", "HF model GPU bottleneck",
  "MoE roofline", "attention efficiency", "why is decode slow", "TTFT
  analysis", "compare Arc GPUs", "compare models on Intel GPU".
---

# OpenVINO Roofline Profiling

This package is a **standalone profiling method + utility layout**. It is not
bound to one machine, one GPU generation, or one model family.

Think of it as four layers:

1. **Platform layer** — characterize the target Intel GPU and store calibrated
   peaks in `utils/platforms.json`.
2. **Model layer** — describe a HuggingFace/OpenVINO model as canonical
   operator families in `models/<model>/ops_mapping.json`.
3. **Benchmark layer** — run reusable OpenVINO micro-benches under
   `cliloader` to obtain measured kernel timings.
4. **Analysis layer** — aggregate results into summary documents and
   `db/metrics.db` for cross-model and cross-platform queries.

## Required inputs

Before profiling, gather or define:

- **GPU environment**
  - OpenVINO build/install path
  - `cliloader` path (download from https://github.com/intel/opencl-intercept-layer/releases/tag/v3.0.6 if missing)
  - target OS/toolchain/runtime environment
  - one or more Intel GPUs to test
- **Model source**
  - HuggingFace model id, local checkout, exported OpenVINO IR, or GenAI flow
  - architecture/config details (`config.json`, HF `modeling_*.py`, graph dump)
  - Model architecture can be found at: local ~/workspace/transformers/src/transformers/models or https://github.com/huggingface/transformers/blob/main/src/transformers/models
- **Workload points**
  - prefill lengths, decode KV lengths, batch, and any model-specific state axis
  - Default token sizes: 1024, 2048, 4096, 8192, 16K, 32K, 64K, 128K
- **Output root**
  - where to save logs, parsed metrics, reports, and DB rows
  - All files related to roofline analysis should be organized under the model's output directory

## Important constraints

- Don't create new commit, push code to remote, submit PR, or change weights layout
- Don't write files to /temp directory; use outputs/ directory
- Sleep time between test cases should not exceed 30s

## GPU architecture reference

Calibrated values are stored in `utils/platforms.json`. Key reference specs:

| GPU | Xe Cores | EU/core | Threads/EU | Subgroup | SLM | BW (GB/s) | FP16 XMX (TFLOPS) | INT8 XMX (TOPS) |
|---|---|---|---|---|---|---|---|---|
| BMG (B580) | 20 | 8 | 8 | 16/32 | 32KB | 456 | 116.736 | 233.472 |
| PTL 12Xe (B390) | 12 | 8 | 10 | 16/32 | 32KB | 110 | 58.982 | 117.965 |
| PTL 4Xe | 4 | 8 | 10 | 16/32 | 32KB | 110 | varies | varies |
| LNL 8Xe | 8 | 8 | 8 | 16/32 | 32KB | 100 | varies | varies |

Computation peak formulas:
- FP16 XMX peak = xe_cores × 8 × 256 × freq_GHz
- INT8 XMX peak = 2 × FP16 XMX peak

Note: Intel Arc B390 uses the same architecture as PTL for roofline calculations due to cliloader recognition.

## Default model compression assumptions

Unless user specifies otherwise:
- MatMul body weights: INT4 quantization with group size 128, FP16 activation, decompressed to FP16 during computation
- LM_Head weights: INT8 quantization with group size 128, FP16 activation
- KV cache: INT8 quantization, decompressed to FP16 in memory
  - k layout: [num_blocks, num_kv_heads, head_size, block_size(16)]
  - v layout: [num_blocks, num_kv_heads, block_size(16), head_size]
- Prefill MatMul: computed by INT8 XMX (weights stored/read as INT4, decompressed to INT8)
- Decode MatMul: computed by FP16 XMX (weights stored/read as INT4, decompressed to FP16)
- FC_Q, FC_K, FC_V are fused into FC_QKV
- swish and multiply are fused into corresponding kernels (don't profile separately)

## SDPA implementation choices

The model may use one of three SDPA implementations. Ask user which to use; default is PagedAttention:

1. **PagedAttention (default)**: two sub-implementations:
   - OpenCL + micro_kernel (default)
   - CM kernel (requires Xe2/Xe3 GPU + CM-JIT support; Windows supported; Linux needs `export CM_FE_DIR=<path of libclangFEWrapper.so>`)
2. **SDPA kernels**: OpenCL + micro_kernel
3. **VLSDPA kernels**: CM kernel for variable-length attention (ViT models like Qwen2-VL / Qwen2.5-VL)

## MoE handling

- MoE uses INT4 compression for expert weights and FP16 activation
- All MoE computation fused into one primitive with 4~6 kernels (grouped_micro_gemm for gate/up/down + moe_* kernels)
- MoE's MLP executed at FP16 precision
- Shared expert can be fused to MoE primitive (see `moe_bench` for shared expert test cases)
- Shared expert supports both INT4-grouped and uncompressed FP16 weight modes

## Generalization strategy

### Generalize across Intel GPUs

Do **not** hardcode BMG/PTL-style assumptions. For each new GPU:

- add/update one entry in `utils/platforms.json`
- fill measured bandwidth, measured frequency, memory type, subgroup sizes,
  SLM, cache, and available compute peaks
- use **XMX peaks** if the GPU exposes XMX
- otherwise use the best available ALU/vector peak
- mark any spec-only numbers as provisional until measured data exists

Rule of thumb: **probe first, theorize second**.

### Generalize across HuggingFace models

Do **not** special-case one model family. Instead, decompose any model into
operator families that map to reusable benchmarks:

| Family | Typical HF modules | Existing bench |
|---|---|---|
| Dense / QKV / O / MLP FC | `Linear`, `MatMul`, fused FC | `utils/fc_bench/` |
| LM head / vocab projection | logits projection, tied output head | `utils/fc_bench/` or dedicated path |
| Attention | SDPA, GQA/MQA, PagedAttention | `utils/pa_bench/` |
| SDPA (uncompressed) | ScaledDotProductAttention (non-PA) | `utils/sdpa_bench/` |
| Variable-length SDPA | ViT block-diagonal attention (Qwen2-VL) | `utils/vlsdpa_bench/` |
| MoE | routed/shared experts, gating | `utils/moe_bench/` |
| Linear/state-space attention | GDN, delta/recurrent kernels | `utils/gdn_bench/` |
| Norm + eltwise + RoPE + quant | RMSNorm, SwiGLU, Add, RoPE, dyn-quant | `utils/small_ops_bench/` |
| Unsupported/custom op | model-specific block | add a bench or use explicit fallback |

This operator-family abstraction is the main reason the tool can scale from
Llama/Qwen/Gemma/Mistral/Phi to hybrid-attention, MoE, and future HF models.

### Separate model description from benchmark implementation

`models/<model>/ops_mapping.json` is the contract between a model and the
benchmark suite. It should capture:

- logical op name / operator family
- exact shape signature for each workload point
- dtype / quantization / compression scheme
- calls per layer / calls per inference
- mode (`prefill`, `decode`, or other)
- token-axis parameter (`S`, `kv`, state length, image tokens, etc.)

If two cases have materially different shapes, treat them as **different
profile points** even if the logical op name is the same.

`models/<model>/report_config.json` is the matching contract for the reporting
layer. It tells the shared `utils/build_model_report.py` engine how to turn
parsed metrics into decode tables, sweep tables, totals, and report JSON
without re-implementing a bespoke `build_report.py` for every model.

`models/<model>/kernel_table_config.json` is the matching contract for the
kernel-table layer. It tells the shared `utils/build_kernel_tables.py` engine
how to emit per-platform, per-mode, and per-size kernel tables without a
bespoke per-model `build_kernel_tables.py`.

`utils/template/SUMMARY_TEMPLATE.md` is the Jinja2-style report template with
`{{VAR}}` placeholders for generating per-model roofline analysis reports.
All `{{PLACEHOLDER}}` variables are documented in `utils/template/README.md`.

## Standard workflow

1. **Characterize the GPU**
   - use `utils/hw_probe/` and vendor/spec data to populate `platforms.json`
   - use measured bandwidth whenever possible
   - watch for cache/compression artifacts in synthetic tests
2. **Extract model topology**
   - read HF config/modeling files or exported OpenVINO graph
   - identify dominant operator families and call counts
   - split prefill and decode because they have different bottlenecks
3. **Build the operator map**
   - write/update `models/<model>/ops_mapping.json`
   - collapse fused subgraphs into the kernel family OpenVINO actually executes
   - omit tiny fully-fused ops unless they materially affect latency
4. **Select or add micro-benches**
   - reuse benches in `utils/` first
   - add a new bench only for a truly new dominant operator family
   - keep one workload point per measurement unit for clean logs
5. **Run under `cliloader`**
   - prefer mean timings over minima
   - ensure each workload point runs long enough for stable statistics
   - avoid cache reuse and host-transfer noise unless intentionally measuring them
6. **Parse and aggregate**
   - parse logs with `utils/parse_logs.py`
   - rebuild the shared DB with `utils/build_db.py`
   - generate model reports with `utils/build_model_report.py`
   - generate kernel tables with `utils/build_kernel_tables.py`
   - or run the one-shot driver `utils/build_postprocess_pipeline.py`
   - compute latency, share, achieved GB/s or TOPS/TFLOPS, efficiency, and bound
7. **Write the report**
   - store raw logs, parsed metrics, tables, and summary together under `outputs/<model>/`
   - label any spec-only or fallback-derived numbers clearly

## Measurement rules that always apply

- prefer **measured** memory bandwidth over spec bandwidth
- use **average/mean** kernel time, not min
- profile **prefill** and **decode** separately
- split compound kernels when runtime behavior warrants it
- size iterations so each case has enough total runtime for stable numbers
  - Target: each op's total kernel execution time ≥ 1000ms; calculate as 1000ms ÷ theoretical latency per op
- for dGPU, minimize PCIe noise when measuring kernel capability; use usm_device memory
- for iGPU, use usm_host or usm_device; explicitly document shared-memory and CPU-contention effects
- if efficiency exceeds 100%, investigate cache reuse, fusion, compression,
  peak mismatch, or measurement error before drawing conclusions
- change all time units to ms
- PA kernel should be split into kv_cache_update and pa computation for separate profiling
- PA prefill uses causal mask (lower-triangular): effective FLOPs = Sq*(Sq+1)/2 ≈ Sq²/2 (not Sq*Skv); decode (Sq=1) has no causal reduction
- dynamic_quantize_gpu_opt kernel (prefill FC): profile separately as its own op
- Math computation costs: exp = 30 FLOPs, sin/cos = 10 FLOPs, sqrt = 10 FLOPs (use for theoretical roofline)
- lm_head should be tested separately from matmul (different weights compression); only single-token test needed (used for both prefill and decode)

## Benchmark design guidelines

- Each op has its own subdirectory and test app under `utils/` for isolated cliloader profiling
- Run one input token size per test app invocation; don't mix sizes in one run
- Avoid L3 cache reuse: create multiple input/output tensor buffers, rotate per iteration
- Auto-size num_bufs so rotating-buffer set exceeds GPU last-level cache
- For dGPU: allocate usm_device memory to avoid PCIe transfer impact
- MoE contains 4~6 kernels: analyze each kernel separately, then aggregate for overall MoE perf
- MoE shared expert test: reference `moe_3gemm_compressed_gpu_shared_random` pattern
- FC bench supports precision modes: `u4` (INT4), `u8` (INT8), `f16` (plain FP16 MatMul)
- PA bench supports implementation modes: `ocl` (default OpenCL+micro_kernel) and `cm` (XAttention CM kernel)
- Save all cliloader perf logs to a logs/ subdirectory
- Test scripts should be copied to target machine and run remotely

## Reporting and SUMMARY requirements

- Use `utils/template/SUMMARY_TEMPLATE.md` as the report template with Jinja2-style `{{VAR}}` placeholders
- SUMMARY should contain separate prefill and decode analysis (one table per token size)
- Each table includes: op name, kernel name, single latency (ms), calling times, total latency (ms), achieved GFLOPS, achieved GB/s, efficiency %, bound (memory/compute)
- Include per-kernel sub-decomposition with actual cliloader kernel names
- Include top contributors by total ms per inference
- Include comparison with other platforms when data available
- Document which bench rows are standalone kernels vs fused away (graph fusion notes)
- All analysis saved as `SUMMARY_<model_name>_<date>.md`
- Remove outdated data; ensure all information is current

## Additional analysis tools

- `utils/ov_verbose_weight_analyze.py` — analyze OpenVINO GPU verbose logs to extract weight constants and produce comprehensive reports (per-category tables, precision summary, architecture insights)
- `utils/parse_cm_pa_logs.py` — parse cliloader CM PA logs and emit per-kernel roofline tables (PTL/Xe2-specific)

## Required outputs

Each complete profiling pass should produce:

- raw logs in `outputs/<model>/logs*/`
- parsed metrics JSON files
- consolidated rows in `db/metrics.db`
- a human-readable summary containing:
  - total latency by mode
  - top operators/kernels by time share
  - achieved compute or bandwidth
  - efficiency relative to the chosen roofline
  - ranked optimization priorities

## Practical limits

- The workflow supports **all Intel GPUs in method**, but data quality depends
  on how well the target GPU is calibrated in `platforms.json`
- The workflow supports **all HuggingFace models in process**, but unsupported
  operator families still need either a new bench or an explicitly labeled
  fallback estimate
- Never silently mix measured and theoretical data without labeling it

## Files to start from

- `README.md` — full standalone user guide
- `utils/platforms.json` — platform registry
- `models/TEMPLATE/` — model onboarding templates
- `utils/template/SUMMARY_TEMPLATE.md` — Jinja2 report template with all placeholder variables
- `utils/template/README.md` — template variable reference documentation
- `utils/` — reusable probes, micro-benches, parsers, DB tools, shared report engine, and shared kernel-table engine
- `utils/ov_verbose_weight_analyze.py` — weight constant analysis from GPU verbose logs
- `utils/parse_cm_pa_logs.py` — CM PA log parser for roofline tables

Reference skills: intel-gpu-hw-info

Keep explanations conversational. For complex concepts, use multiple analogies.
