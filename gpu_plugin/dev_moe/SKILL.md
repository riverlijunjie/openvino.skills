---
name: dev_moe
description: Develop Mixture of Experts (MoE) operations for better performance. Use when working on MoE models or improving MoE operation efficiency.
---

## Workflow

1. **Read code**: Read all MOE-related sources (see `reference.md`) and `SUMMARY.md` for architecture/history
2. **Summarize**: Understand the 3-kernel pipeline (gate_up → down → reduce), JIT constants, and dispatch config
3. **Identify optimizations**: Focus on memory-bound bottlenecks (decode GEMV is ~4 FLOPs/byte, 25× below XE2 compute crossover)
4. **Implement & test**: Build with `make ov_gpu_unit_tests -j$(nproc)`, run `--gtest_filter="*moe*"` (51 tests)
5. **Document**: Update `SUMMARY.md` with results; keep this file as concise reference

## Key Architecture Facts

- **Pipeline**: `softmax_topk → mlp_gate_up(fused gate+up GEMV) → mlp_down(GEMV × routing_weight) → mlp_reduce(cross-expert sum)`
- **Dispatch**: `gws={num_experts, SUBGROUP_SIZE, INTERMEDIATE_SIZE/N_BLOCK}`, `lws={1, SUBGROUP_SIZE, SUBGROUP_NUM}`
- **Constants**: `N_BLOCK=4`, `SUBGROUP_NUM=8`, `SUBGROUP_SIZE=32` (XE2) / `16` (older)
- **Prefill path**: Uses oneDNN micro_gemm (not custom GEMV), separate code path in `exec_prefill_micro_gemm()`
- **Shared expert**: Treated as expert `MAX_TOPK` (extra workgroup), scalar gate via parallel dot-product + sigmoid

## JIT Constants (from `add_common_consts()`)

| Constant | Purpose |
|----------|---------|
| `HAS_ZP` | 0=symmetric (skip xg_sum), 1=asymmetric (compute xg_sum for ZP compensation) |
| `WEIGHT_IS_SIGNED` | 0=u4/u8, 1=i4/i8 (controls sign-extension macros) |
| `WEIGHT_COMPRESSEION_DT` | 0=4bit, 1=8bit, 2=f16 |
| `GATE_UP_GROUP_SIZE` / `DOWN_GROUP_SIZE` | Quantization group size (128/256/per-channel) |
| `SHARED_EXPERT_ENABLE` | 0 or 1 |

## Quantization

- **Asymmetric** (u4/u8): `dequant = (w - zp) * scale`, uses `xg_sum` optimization
- **Symmetric** (i4/i8): `dequant = w * scale`, `HAS_ZP=0` compiles out all ZP code
- **Macros** (not inline functions!): `DEQUANT_4BIT_LO/HI`, `DEQUANT_8BIT`, `ZP_ADJUST_*` — must be macros because `.cl` file is compiled 3× into one OpenCL program

## Implemented Optimizations

| Optimization | Impact | Mechanism |
|-------------|--------|-----------|
| **Fused gate+up GEMV** | **Slower** — reverted to 2-call | 2× register pressure → occupancy drop; cache thrashing across gate/up weights; I-cache pressure from unrolled code |
| **xg_sum skip for symmetric** | Eliminates wasted computation when HAS_ZP=0 | `#if HAS_ZP` guards on accumulation + reduce + SLM write |
| **Parallel shared expert scalar gate** | Eliminates single-thread bottleneck | Workgroup-wide parallel reduction (256 threads) |
| **Shared expert transformation fix** | Enables shared expert fusion | Rejection predicate `!is_type<Add>` on standard matcher, context-aware `replace_node` |

## Key Constraints

- **Memory-bound**: Reducing data movement > reducing ALU ops
- **No inline functions in .cl**: Use macros only (file compiled 3× as sub-kernels into single program)
- **Symmetric quant needs dummy ZP tensor**: All-zero tensor matching weight type to satisfy kernel signature
- **Shared expert must match sparse expert INTERMEDIATE_SIZE**: Current assumption in code

## Build & Test

```bash
cd build-x86_64-release
make ov_gpu_unit_tests -j$(nproc)
./bin/intel64/Release/ov_gpu_unit_tests --gtest_filter="*moe*"
# Debug: ENABLE_GPU_DEBUG_CAPS=ON for kernel build logs
```

## Related Docs

- `SUMMARY.md` — Full work summary with detailed architecture, file map, performance tables, and history
- `reference.md` — Complete file listing of all MOE source files
- `moe_shared_expert.md` — Shared expert operator and GenAI integration details
- `qwen3_moe_i4_sym.md` - Performance analysis and optimization details for Qwen3-30B MoE with i4 symmetric quantization
