# OpenVINO Roofline Profiling — utils

Model-agnostic C++ bench apps, a universal log parser, and a pair of runner templates for building a per-op roofline of any OpenVINO model on Intel GPU.

See `../SKILL.md` for the methodology, rationale, and the 11 lessons learned. This README is a practical quick-start.

## What is generic vs. per-model

```
universal (never edit per model):
  fc_bench/, pa_bench/, small_ops_bench/, CMakeLists.txt,
  parse_logs_v2.py,
  run_all.sh.template, run_all_ptl.bat.template

per-model (copy .example → plain name, then edit):
  run_config.sh          ← paths + op list (Linux)
  run_config.bat         ← paths + op list (Windows)
  ops_mapping.json       ← op catalog used by the metrics generator
  generate_metrics.py    ← bytes/FLOPs aggregator, emits performance_metrics.json
  generate_summary.py    ← renders SUMMARY.md from performance_metrics.json
```

## Build the benches

```bash
cmake -B build -DOpenVINO_DIR=/path/to/openvino/build
cmake --build build -j
```

Produces `build/fc_bench`, `build/pa_bench`, `build/small_ops_bench`. These binaries are universal — one build works for any model.

## Bench CLIs

```text
fc_bench <M> <K> <N> [group_size=128] [iters=100] [warmup=10] [num_bufs=8] [precision=u4|u8] [flush_mb=64]
  - Memory-bound decode (M=1) or compute-bound prefill (M>1) FullyConnectedCompressed.
  - precision: "u4" for INT4/group body FCs; "u8" for per-group INT8 LM_Head.
  - flush_mb MB of VRAM Relu between every measured infer (see SKILL §L3). Set 0 to disable.

pa_bench <mode> <M> <kv_len> <iters> <warmup> <num_bufs> [kv_dtype=i8]
  - mode ∈ {decode, prefill}. M = query tokens. kv_len = already-cached KV tokens.

small_ops_bench <op> <shape_args...> --iters N --warmup N --bufs N
  - op ∈ {rmsnorm, rmsnorm3d, rope, add}. DO NOT bench swiglu_ref (it fuses; see SKILL §L7).
```

## Workflow

1. Build benches (once).
2. `cp run_config.sh.example run_config.sh` and edit paths + `FC_CONFIGS` / `PA_CONFIGS` / `SM_CONFIGS` to match **your** model's op shapes.
3. `./run_all.sh.template` (or Windows: `run_all_ptl.bat.template`). One log per (op, shape) lands in `$RESULTS_DIR/`.
4. `python3 parse_logs_v2.py $RESULTS_DIR $RESULTS_DIR.parsed.json` → kernel stats JSON.
5. Run your (model-specific) `generate_metrics.py` to produce `performance_metrics.json`.
6. Run your (model-specific) `generate_summary.py` to render `SUMMARY.md`.

The runner templates are intentionally short and read their op list from the sourced `run_config.{sh,bat}` — do not edit the templates themselves when porting to a new model.

## Reference (`.example`) files

- `run_all.sh.example`, `run_all_ptl.bat.example` — fully populated Qwen3-8B drivers, kept for historical reference and as a worked example of how a full op list looks.
- `ops_mapping.json.example` — schema + Qwen3-8B example.
- `generate_metrics.py.example`, `generate_summary.py.example` — Qwen3-8B reference implementations. Copy and adapt.

## Things not to touch

- `KERNEL_EXCLUDES` in `parse_logs_v2.py` contains `"activation"` to strip the L2 cache-flush kernel (`activation_opt_*`). No FC/PA/small-op bench produces activation-named kernels, so the global exclude is safe. If you add a new bench whose measured kernel is named `activation_*`, change the filter carefully.
- The cache-flush model in `fc_bench/main.cpp` must remain compiled as a *separate* CompiledModel. Folding it into the FC graph would let the plugin optimize it away.
- `num_bufs` in all benches only rotates activation tensors. Weight rotation was tried and rejected — it distorts iGPU measurements (see SKILL §L3 rejected-alternative note).
