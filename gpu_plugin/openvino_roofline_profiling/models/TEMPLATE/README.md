# Model onboarding template

Copy this directory to `models/<your_model>/` when onboarding a new HuggingFace
or OpenVINO model into the roofline profiling toolkit.

## What belongs here

- `ops_mapping.json` — the canonical operator-family description for the model
- `report_config.json` — the model-specific report recipe consumed by the shared
  `utils/build_model_report.py` engine
- `kernel_table_config.json` — the model-specific kernel-table recipe consumed
  by the shared `utils/build_kernel_tables.py` engine
- run scripts for each target platform/host you care about
- any small model-specific notes that are required to reproduce the profiling
  sweep (token grid, batch size, quantization assumptions, special modes)

## Workflow

1. Read the model architecture from HuggingFace config / modeling files or from
   the exported OpenVINO graph.
2. Decompose the model into canonical operator families such as FC, attention,
   MoE, state-space/linear attention, LM head, and small ops.
3. For each significant workload point, add one entry to `ops_mapping.json`.
4. Write `report_config.json` to describe decode rows, sweep tables, and total
  formulas for the shared report engine.
5. Write `kernel_table_config.json` to describe per-mode per-size kernel-table
  generation without a custom per-model Python script.
6. Reuse existing micro-benches in `utils/` whenever possible.
7. Add run scripts that execute the required sweep on your target machine(s).
8. Save raw logs under `outputs/<your_model>/logs*`, parse them, and ingest them
   into the shared DB.

## Naming guidelines

- Use a stable, lowercase model directory name, e.g. `llama3_8b`, `gemma3_27b`,
  `deepseek_v3`, `phi4_multimodal`.
- Keep run scripts platform-neutral in name when possible, for example:
  - `run_<model>_linux.sh`
  - `run_<model>_windows.bat`
  - `run_<model>_<platform>.sh`

## Notes

- If the model introduces a new dominant operator family, add a new bench under
  `utils/` rather than forcing a poor fit.
- If a value is theoretical or estimated, record that fact explicitly instead of
  letting it masquerade as measured data.
