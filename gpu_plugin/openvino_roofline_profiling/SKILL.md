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
  - `cliloader` path
  - target OS/toolchain/runtime environment
  - one or more Intel GPUs to test
- **Model source**
  - HuggingFace model id, local checkout, exported OpenVINO IR, or GenAI flow
  - architecture/config details (`config.json`, HF `modeling_*.py`, graph dump)
- **Workload points**
  - prefill lengths, decode KV lengths, batch, and any model-specific state axis
- **Output root**
  - where to save logs, parsed metrics, reports, and DB rows

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
- for dGPU, minimize PCIe noise when measuring kernel capability
- for iGPU, explicitly document shared-memory and CPU-contention effects
- if efficiency exceeds 100%, investigate cache reuse, fusion, compression,
  peak mismatch, or measurement error before drawing conclusions

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
- `utils/` — reusable probes, micro-benches, parsers, DB tools, shared report engine, and shared kernel-table engine
