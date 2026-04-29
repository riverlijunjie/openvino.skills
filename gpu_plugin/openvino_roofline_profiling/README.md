# OpenVINO Roofline Profiling

This directory is a **standalone Intel-GPU roofline profiling toolkit** for
OpenVINO workloads. Its purpose is not merely to hold a few benchmark scripts;
it defines a reusable method that can be applied to:

- **all Intel GPU families** supported by the OpenVINO GPU stack, including
  iGPU and dGPU generations with or without XMX, and
- **all HuggingFace model families**, as long as their execution can be
  decomposed into canonical operator families and the dominant operators are
  either benchmarked directly or estimated with explicitly labeled fallbacks.

In short: this tool should work for Qwen today, Llama tomorrow, Gemma the day
after, and whatever spicy new HF architecture shows up next week.

## Design goals

The toolkit is designed around five principles:

1. **Platform-agnostic calibration** — hardware peaks come from a platform
   registry plus on-device probing, not from one fixed GPU.
2. **Model-agnostic decomposition** — models are described as operator families,
   not as hand-written one-off benchmark stories.
3. **Benchmark reuse** — the same OpenVINO micro-benches should cover large
   swaths of models by parameterization.
4. **Measurement traceability** — raw logs, parsed metrics, DB rows, and
   summary conclusions stay connected.
5. **Optimization usefulness** — the final report must explain where time goes,
   what is bound, and what to optimize first.

## Directory layout

```
openvino_roofline_profiling/
├── SKILL.md                     # activation/operation guide for Copilot
├── README.md                    # this standalone manual
├── db/
│   └── metrics.db               # shared multi-model, multi-platform SQLite DB
├── models/
│   ├── TEMPLATE/                # onboarding templates for a new HF/OpenVINO model
│   ├── qwen3_moe/
│   └── qwen3_5_moe/
├── outputs/
│   ├── <model>/                 # logs, parsed metrics, tables, summaries
│   └── ...
└── utils/
    ├── CMakeLists.txt
    ├── platforms.json           # platform registry and calibrated peaks
    ├── hw_probe/                # generic hardware probes
   ├── parse_logs.py            # cliloader log → structured JSON
   ├── build_db.py              # structured JSON → shared SQLite DB
   ├── build_model_report.py    # generic report engine driven by per-model config
   ├── build_kernel_tables.py   # generic kernel-table engine driven by per-model config
   ├── build_postprocess_pipeline.py # one-shot post-parse pipeline driver
    ├── roofline_report.py       # shared aggregation helper(s)
    ├── fc_bench/
    ├── pa_bench/
    ├── sdpa_bench/
    ├── moe_bench/
    ├── gdn_bench/
    └── small_ops_bench/
```

## Architecture of the tool

The toolkit has four abstraction layers.

### Platform layer

`utils/platforms.json` is the hardware registry. Each GPU entry should capture:

- architecture / product name
- compute-unit structure
- subgroup sizes
- memory type and capacity
- measured or nominal frequency
- measured or spec bandwidth
- measured or spec compute peaks
- notes on shared memory, cache behavior, launch overhead, or quirks

This lets the same analysis pipeline adapt from Arc dGPU to integrated GPUs and
future Intel parts. If the GPU has XMX, use XMX rooflines; if it does not, use
the best available ALU/vector roofline.

### Model layer

`models/<model>/ops_mapping.json` describes a model as **canonical operator
families** at specific workload points. The mapping is the bridge between a
HuggingFace architecture and the reusable micro-benches.

Typical operator families include:

| Family | Examples |
|---|---|
| Dense FC | QKV/O projection, MLP up/down/gate, generic Linear |
| LM head | vocab projection / logits head |
| Attention | SDPA/GQA/MQA/PagedAttention |
| MoE | routed/shared experts + gating |
| Linear attention / recurrent state | GDN, DeltaNet, state-space blocks |
| Small ops | RMSNorm, RoPE, add, multiply, activation glue, dyn-quant |
| Fallback/custom | unsupported or new kernels |

The point of the mapping is to say: “this HF block is not a mysterious dragon;
it is just a few familiar dragons wearing different hats.”

### Benchmark layer

The micro-benches under `utils/` instantiate representative OpenVINO graphs and
run them under `cliloader`. Each bench should isolate one operator family at one
shape/configuration point well enough that the resulting kernel timings are
useful for roofline analysis.

### Analysis layer

The analysis pipeline converts raw logs into:

- parsed per-kernel timing JSON
- per-model derived reports and tables produced by the shared report engine
- a shared SQLite database for cross-model and cross-platform study

## General method for any Intel GPU

### 1. Calibrate the platform

For a new GPU, start with `utils/hw_probe/` and add/update a `platforms.json`
entry. The generic process is:

1. read product/spec data from Intel docs and driver/runtime queries
2. measure memory bandwidth on-device
3. record subgroup, SLM, cache, memory type, and frequency
4. store both **spec** and **measured** values when available
5. prefer measured values in roofline reports

Important rule: **synthetic bandwidth tests must avoid false cache/compression
inflation**. On Xe2 we already observed that zero-filled and constant-filled
buffers can report fantasy-land bandwidth. Randomized buffer initialization is
the safer default. Physics is rude, but consistent.

### 2. Decide the compute roofline

Use the highest compute path actually relevant to the kernel:

- XMX peak for XMX-backed kernels
- ALU/vector peak for non-XMX kernels
- a model-specific compute path if the operator is dominated by transcendental
  math or scalar work rather than matrix engines

If exact peak data is missing, keep the result but label it **provisional**.

## General method for any HuggingFace model

### 1. Read the model structure

Start from one or more of:

- HF `config.json`
- HF `modeling_*.py`
- exported OpenVINO IR / graph dump
- runtime graph or execution graph if available

You want to answer four questions:

1. What are the dominant operator families?
2. What are their shapes and dtypes?
3. How many times are they called in prefill and decode?
4. Which subgraphs are actually fused by OpenVINO GPU?

### 2. Normalize the model into operator families

The critical abstraction is to normalize very different model architectures into
common operator families.

Examples:

- **Llama/Qwen/Gemma/Mistral/Phi dense decoder** → FC + attention + small ops + LM head
- **MoE decoder** → FC + routed/shared MoE + attention + small ops + LM head
- **Hybrid attention** → FC + attention + GDN/state-space + small ops + LM head
- **Novel/custom block** → split into existing families where possible, add a
  new micro-bench only for the truly dominant residual piece

### 3. Separate execution modes

Never merge everything into one giant average. Treat at least these modes
separately:

- **Prefill / TTFT path**
- **Decode / next-token path**

If the model has other meaningful phases (vision encoder, audio frontend,
multimodal projector, state init, speculative decode), give them their own mode
labels and workload axes.

### 4. Build `ops_mapping.json`

Each significant profile point should record:

- `op_name`
- `family`
- `mode`
- token/state axis (`S`, `kv`, `T`, image patches, frames, etc.)
- shapes
- dtype / quant / compression
- calls per inference
- remarks on fusion or fallback handling

Same logical op, different shape, different roofline point. Do not over-merge.

## Benchmark design rules

The micro-benches should follow these rules regardless of GPU or model:

1. **One clear measurement unit** — one operator family, one shape/config point,
   one log record series.
2. **Enough runtime for stability** — choose iterations so total kernel time is
   comfortably above transient-noise territory.
3. **Avoid fake cache wins** — rotate buffers or use multiple tensor instances
   when measuring steady-state kernel capability.
4. **Avoid unrelated transfer overhead** — especially on dGPU, avoid host/device
   transfer effects unless the benchmark explicitly studies them.
5. **Measure what OpenVINO really executes** — if runtime fuses operations,
   benchmark the fused form, not a hand-assembled unfused surrogate.
6. **Split compound behavior when needed** — for example, split attention update
   and attention compute if the runtime does so or if their rooflines differ.

## Current benchmark families

| Bench | Covers | Notes |
|---|---|---|
| `fc_bench` | FC/QKV/O/MLP/LM head-like projection | dense and quantized matrix paths |
| `pa_bench` | PagedAttention (with INT8 KV) | canonical attention roofline; matches the deployed PA path |
| `sdpa_bench` | `ov::op::v13::ScaledDotProductAttention` (uncompressed Q/K/V) | optional comparison against PA on identical shapes; do **not** use as the deployed roofline when the model runs PA |
| `moe_bench` | routed/shared expert MoE | profile fused expert path rather than three disconnected GEMMs |
| `gdn_bench` | GDN / linear-attention/state kernels | template for non-SDPA sequence blocks |
| `small_ops_bench` | norm/rope/eltwise/dyn-quant | catch the “small but everywhere” family |

If a model introduces a new dominant family, add a new bench under `utils/`
instead of contorting an existing one beyond recognition.

## Data pipeline

### Raw logs

Use `cliloader` to generate per-kernel timing logs and keep them under
`outputs/<model>/logs*`.

### Parsed metrics

Use `utils/parse_logs.py` to convert logs into structured JSON. The parsed data
should preserve:

- platform
- config/workload point
- detected iterations
- total kernel time
- per-kernel time and call counts

### Generic report generation

Use `utils/build_model_report.py` to turn parsed metrics into human-readable
Markdown tables plus structured report JSON. The engine is shared; only the
model-specific `report_config.json` changes.

Example:

```bash
python3 utils/build_model_report.py \
   --model-dir models/qwen3_moe \
   --output-dir outputs/qwen3_moe \
   --platform BMG \
   --out outputs/qwen3_moe/performance_metrics_bmg.json
```

The old `outputs/<model>/build_report.py` scripts are now thin compatibility
wrappers around this shared engine.

### Generic kernel-table generation

Use `utils/build_kernel_tables.py` to render per-platform, per-mode,
per-token-size kernel tables from parsed metrics. The engine is shared; only the
model-specific `kernel_table_config.json` changes.

Example:

```bash
python3 utils/build_kernel_tables.py \
   --model-dir models/qwen3_moe \
   --output-dir outputs/qwen3_moe \
   --out outputs/qwen3_moe/kernel_tables.md
```

The old `outputs/<model>/build_kernel_tables.py` scripts are now thin
compatibility wrappers around this shared engine.

### Unified postprocess pipeline

After `parse_logs.py` has produced `<platform>_metrics.json`, you can generate
the rest of the postprocess artifacts with one command:

```bash
python3 utils/build_postprocess_pipeline.py \
   --model-dir models/qwen3_moe \
   --output-dir outputs/qwen3_moe \
   --rebuild-db
```

This generates:

- `performance_metrics_<platform>.json`
- `kernel_tables.md`
- optional `db/metrics.db` refresh for the current model

### Shared database

Use `utils/build_db.py` to ingest every model's metrics into `db/metrics.db`.
The shared DB is the place to answer questions like:

- which kernels dominate decode across models?
- how does the same operator family scale across GPUs?
- which cases are most bandwidth-bound?
- how do prefill and decode bottlenecks differ?

## Roofline rules

At minimum, every significant operator or kernel should report:

- average latency
- total latency contribution per inference
- achieved bandwidth and/or compute throughput
- efficiency relative to the chosen platform peak
- memory-bound vs compute-bound classification

Recommended report logic:

- use **measured** bandwidth where possible
- use **mean** kernel time rather than min
- compute-bound kernels should be judged against the relevant compute engine
- memory-bound kernels should be judged against measured memory bandwidth
- if efficiency exceeds 100%, investigate cache reuse, fusion, compression,
  or peak miscalibration before making optimization claims

## Reporting contract

Each complete profiling run should leave behind:

- raw logs
- parsed metrics JSON
- derived report JSON / tables
- a human-readable `SUMMARY_<model>_<date>.md`

The summary should contain, separately for each mode:

- total latency
- per-operator or per-kernel share
- per-workload tables (for example one table per `S` or `kv` point)
- the main bottlenecks
- the top optimization opportunities

The goal is not to print pretty tables for the sake of pretty tables. The goal
is to tell an engineer where the time went and what to fix first.

## Adding a new GPU

1. add or update the GPU entry in `utils/platforms.json`
2. run or adapt a probe under `utils/hw_probe/`
3. verify measured bandwidth/frequency/cache assumptions
4. record whether the main compute path is XMX or non-XMX
5. label the entry as measured or provisional

## Adding a new model

1. copy `models/TEMPLATE/`
2. build `ops_mapping.json` from the HF architecture/config
3. write `report_config.json` to describe decode rows and sweep totals
4. write `kernel_table_config.json` to describe per-kernel table generation
5. classify each dominant block into an operator family
6. reuse existing benches where possible
7. add a new bench only for a truly new dominant family
8. run the measurement sweep and parse logs
9. ingest the results into the shared DB
10. run `utils/build_postprocess_pipeline.py`
11. write the summary

## Examples included here

The `qwen3_moe` and `qwen3_5_moe` directories are **worked examples**, not hard
requirements for the tool. They demonstrate:

- how to structure `models/<model>/`
- how to store outputs and summaries
- how to connect parsed logs to final reports

Use them as examples, not shackles.

## Practical limitations

- “Supports all Intel GPUs” means the **method** is general; measurement quality
  still depends on platform calibration.
- “Supports all HF models” means the **workflow** is general; new dominant
  operator families still require either a new micro-bench or a clearly labeled
  fallback estimate.
- Do not silently mix measured and theoretical numbers. If a number is guessed,
  say so loudly.

## Recommended next steps for future hardening

If you want to evolve this from a strong toolkit into a nearly push-button
system, the best next investments are:

1. a generic `ops_mapping.json` validator/schema checker
2. a generic report generator that reduces model-specific `build_report.py`
3. platform onboarding helpers that emit `platforms.json` stubs automatically
4. automated extraction of operator families from exported OpenVINO graphs
