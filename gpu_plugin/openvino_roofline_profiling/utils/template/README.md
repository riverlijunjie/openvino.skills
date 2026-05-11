# Roofline Report Template

## Usage

`SUMMARY_TEMPLATE.md` is a Jinja2-style (double-brace `{{VAR}}`) template for
generating per-model roofline analysis reports. All `{{PLACEHOLDER}}` variables
should be replaced by the report generator script (`build_report_*.py`).

## Variable Reference

### Header / Platform
| Variable | Example | Description |
|---|---|---|
| `MODEL_NAME` | `qwen3_omni Thinker text` | Model display name |
| `PLATFORM_NAME` | `PTL 12Xe Windows` | GPU + OS |
| `DATE` | `2026-05-08` | Report date |
| `PLATFORM_DESCRIPTION` | `Intel PTL 12Xe iGPU …` | Full platform spec line |
| `MODEL_DESCRIPTION` | `qwen3_omni Thinker text decoder (dense, GQA)` | Model summary |

### Model Config
| Variable | Example |
|---|---|
| `HIDDEN_SIZE` | 2560 |
| `NUM_LAYERS` | 36 |
| `NH` | 32 |
| `NKV` | 8 |
| `HD` | 128 |
| `INTERMEDIATE_SIZE` | 9728 |
| `VOCAB_SIZE` | 151936 |
| `MLP_TYPE` | `SwiGLU` or `MoE` |
| `HIDDEN_ACT` | `silu` |

### Hardware Roofline
| Variable | Example |
|---|---|
| `FP16_XMX_PEAK` | 58.98 |
| `INT8_XMX_PEAK` | 117.96 |
| `MEM_BW` | 110 |
| `RIDGE_POINT` | 536 |

### Repeated Sections
Tables for decode/prefill are repeated per KV length / sequence length.
Use `<!-- REPEAT -->` / `<!-- END REPEAT -->` markers as loop boundaries.

- `{{KV_LENGTHS}}`: list, e.g. `[1024, 2048, 4096, 8192]`
- `{{SEQ_LENGTHS}}`: list, e.g. `[1024, 2048, 4096, 8192]`

### Row-level Variables (inside repeated tables)
| Variable | Description |
|---|---|
| `OP` | Op name (fc_up, pa, rmsnorm, etc.) |
| `KERNEL` | Kernel type (fc_int4_g128, pa_opencl_micro, etc.) |
| `KERNEL_NAME` | Actual cliloader kernel name (gemm_kernel, sdpa_micro__prefill, etc.) |
| `SINGLE_MS` | Per-call latency in ms |
| `CALLS` | Number of calls per inference |
| `TOTAL_MS` | single_ms × calls |
| `GFLOPS` | Achieved GFLOPS |
| `GBS` | Achieved GB/s |
| `EFF_PCT` | Efficiency % (vs XMX peak for compute-bound, vs BW for memory-bound) |
| `BOUND` | `memory` or `compute` |

## Adapting for New Models

1. Copy template and fill in model config variables
2. For MoE models: uncomment the MoE weight table rows, remove dense FC_Gate/Up/Down
3. For models with extra ops (e.g. vision encoder): add rows to the per-kernel tables
4. Adjust fusion notes to match actual graph compilation behavior
5. Set `PREFILL_FC_XMX_TYPE` to `INT8` (default) or `FP16` depending on dynamic quantization
