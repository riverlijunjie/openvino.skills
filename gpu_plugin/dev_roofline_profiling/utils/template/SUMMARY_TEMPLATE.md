# {{MODEL_NAME}} — Roofline on {{PLATFORM_NAME}} ({{DATE}})

**Platform**: {{PLATFORM_DESCRIPTION}}
**Model**: {{MODEL_DESCRIPTION}}

- {{MODEL_CONFIG_BULLETS}}
- MatMul weights {{MATMUL_QUANT}} / {{ACT_DTYPE}} act; LM_head {{LMHEAD_QUANT}} / {{ACT_DTYPE}} act; KV cache {{KV_CACHE_DTYPE}}
- SDPA: {{SDPA_IMPL}}

## Model parameters & weight shapes

Architecture knobs (parsed from model config):

| Field | Value | Notes |
|---|---:|---|
| `hidden_size` | {{HIDDEN_SIZE}} | residual / activation channel |
| `num_hidden_layers` | {{NUM_LAYERS}} | decoder blocks |
| `num_attention_heads` (NH) | {{NH}} | Q heads |
| `num_key_value_heads` (NKV) | {{NKV}} | GQA: {{GQA_RATIO}}-way Q-per-KV sharing |
| `head_dim` (HD) | {{HD}} | Q_dim = NH·HD = {{Q_DIM}}, KV_dim = NKV·HD = {{KV_DIM}} |
| `intermediate_size` | {{INTERMEDIATE_SIZE}} | {{MLP_TYPE}} MLP hidden |
| `vocab_size` | {{VOCAB_SIZE}} | LM head N |
| `hidden_act` | {{HIDDEN_ACT}} | {{MLP_TYPE}} = {{HIDDEN_ACT_FORMULA}} |
| `tie_word_embeddings` | {{TIE_EMBEDDINGS}} | LM head storage shared with token embedding |
<!-- Add model-specific fields below (e.g. rope_theta, MoE params) -->
{{EXTRA_MODEL_FIELDS}}

Per-layer weight matrices (one decoder block) and global weights:

| Weight | Shape (K × N) | Quant | Bytes / instance | × Layers | Total MB |
|---|---:|---|---:|---:|---:|
| Embedding | {{EMBED_SHAPE}} | {{EMBED_QUANT}} | {{EMBED_BYTES}} | 1 | {{EMBED_TOTAL_MB}} |
| FC_QKV (fused Q+K+V proj) | {{FCQKV_SHAPE}} | {{FCQKV_QUANT}} | {{FCQKV_BYTES}} | {{NUM_LAYERS}} | {{FCQKV_TOTAL_MB}} |
| FC_O (attention output) | {{FCO_SHAPE}} | {{FCO_QUANT}} | {{FCO_BYTES}} | {{NUM_LAYERS}} | {{FCO_TOTAL_MB}} |
<!-- For dense models (SwiGLU): -->
| FC_Gate (SwiGLU gate) | {{FCGATE_SHAPE}} | {{FCGATE_QUANT}} | {{FCGATE_BYTES}} | {{NUM_LAYERS}} | {{FCGATE_TOTAL_MB}} |
| FC_Up (SwiGLU up) | {{FCUP_SHAPE}} | {{FCUP_QUANT}} | {{FCUP_BYTES}} | {{NUM_LAYERS}} | {{FCUP_TOTAL_MB}} |
| FC_Down (SwiGLU down) | {{FCDOWN_SHAPE}} | {{FCDOWN_QUANT}} | {{FCDOWN_BYTES}} | {{NUM_LAYERS}} | {{FCDOWN_TOTAL_MB}} |
<!-- For MoE models, replace FC_Gate/Up/Down with MoE expert weights:
| MoE Gate+Up (per expert) | {{MOE_GU_SHAPE}} | {{MOE_GU_QUANT}} | {{MOE_GU_BYTES}} | {{NUM_LAYERS}} × {{NUM_EXPERTS}} | {{MOE_GU_TOTAL_MB}} |
| MoE Down (per expert) | {{MOE_DOWN_SHAPE}} | {{MOE_DOWN_QUANT}} | {{MOE_DOWN_BYTES}} | {{NUM_LAYERS}} × {{NUM_EXPERTS}} | {{MOE_DOWN_TOTAL_MB}} |
| Router | {{ROUTER_SHAPE}} | {{ROUTER_QUANT}} | {{ROUTER_BYTES}} | {{NUM_LAYERS}} | {{ROUTER_TOTAL_MB}} |
| Shared Expert Gate+Up | ... | ... | ... | ... | ... |
| Shared Expert Down | ... | ... | ... | ... | ... |
-->
| LM_Head | {{LMHEAD_SHAPE}} | {{LMHEAD_QUANT}} | {{LMHEAD_BYTES}} | 1 | {{LMHEAD_TOTAL_MB}} |
| **Total static weights** |  |  |  |  | **{{TOTAL_WEIGHTS_MB}} MB** |

Activation / KV-cache shapes (S = sequence length, B = batch=1):

| Tensor | Shape | dtype | Bytes / token / layer | Bytes / token (all layers) |
|---|---|---|---:|---:|
| Hidden states | [B, S, {{HIDDEN_SIZE}}] | FP16 | {{HIDDEN_BYTES}} | — |
| Q | [B, S, {{NH}}, {{HD}}] | FP16 | {{Q_BYTES}} | — |
| K (cache) | [num_blocks, {{NKV}}, {{HD}}, {{BLOCK_SIZE}}] | {{KV_CACHE_DTYPE}} | {{K_BYTES_PER_TOKEN}} | {{K_BYTES_ALL_LAYERS}} |
| V (cache) | [num_blocks, {{NKV}}, {{BLOCK_SIZE}}, {{HD}}] | {{KV_CACHE_DTYPE}} | {{V_BYTES_PER_TOKEN}} | {{V_BYTES_ALL_LAYERS}} |
| **KV cache total** | per token | {{KV_CACHE_DTYPE}} | {{KV_BYTES_PER_LAYER}} B / layer | **{{KV_BYTES_PER_TOKEN_ALL}} / token** |

## Theoretical roofline

| Metric | Value |
|---|---|
| FP16 XMX peak | {{FP16_XMX_PEAK}} TFLOPS |
| INT8 XMX peak | {{INT8_XMX_PEAK}} TOPS |
| Memory BW | {{MEM_BW}} GB/s |
| Ridge point (FP16) | {{RIDGE_POINT}} FLOP/byte |
| Ridge point (INT8) | {{RIDGE_POINT_INT8}} OP/byte |

## Data sources

<!--
When only partial benchmarks are available on the target platform, other ops
can be estimated by scaling from a reference platform. Document clearly:
- Which ops have measured data on this platform
- Which ops are estimated (and from which reference platform + scaling method)
- Scaling rationale: memory-bound ops scale by BW ratio, compute-bound ops by XMX ratio
-->
{{DATA_SOURCES_NOTE}}

## Graph fusion notes

<!--
Document which bench rows correspond to standalone kernels in the compiled graph
and which are fused away. This is model-specific and should be updated per model.
-->

| Bench row | Real graph behaviour | Fused into | Standalone kernel in graph? |
|---|---|---|---|
| `multiply` | `silu(gate(x)) ⊙ up(x)` of SwiGLU MLP | SwiGLU primitive | No — bench-only |
| `add` | Residual adds per layer | not fused (separate `eltwise`) | Yes |
| `rmsnorm` | Pre-attention + pre-MLP + final RMSNorm | single `RMS` primitive | Yes |
<!-- Add model-specific fusion notes -->
{{EXTRA_FUSION_NOTES}}

## Token latency summary

### Prefill — TTFT and per-token amortized

| S | TTFT (ms) | TTFT (s) | per-token (ms) | tokens/s |
|---:|---:|---:|---:|---:|
<!-- One row per sequence length:
| {{S}} | {{TTFT_MS}} | {{TTFT_S}} | {{PER_TOKEN_MS}} | {{TOKENS_PER_S}} |
-->
{{PREFILL_TTFT_ROWS}}

### Decode — TPOT (per output token)

| KV (ctx) | TPOT (ms) | tokens/s |
|---:|---:|---:|
<!-- One row per KV length:
| {{KV}} | {{TPOT_MS}} | {{TOKENS_PER_S}} |
-->
{{DECODE_TPOT_ROWS}}

<!-- OPTIONAL: include when decode latency must reflect a full generation window
(PagedAttention grows with KV across the GEN-token window). `start` = KV at the
first generated token, `mean` = KV at the mid-window token, `end` = KV at the last.
Only PA scales with KV; the rest is M=1 constant. Omit for fixed-KV decode reports. -->
### Decode — full {{GEN_TOKENS}}-token generation window (PA grows with KV)

| prompt P | TPOT start (ms) | TPOT mean (ms) | TPOT end (ms) | {{GEN_TOKENS}}-tok decode (ms) | decode tok/s |
|---:|---:|---:|---:|---:|---:|
<!-- One row per prompt length:
| {{P}} | {{TPOT_START_MS}} | {{TPOT_MEAN_MS}} | {{TPOT_END_MS}} | {{DECODE_WINDOW_MS}} | {{DECODE_TOKS}} |
-->
{{DECODE_WINDOW_ROWS}}

### Decode TPOT — per-op breakdown (ms / % of TPOT)

| op | {{DECODE_KV_HEADERS}} |
|---|{{DECODE_KV_SEPARATORS}}|
<!-- One row per op, columns for each KV length:
| {{OP_NAME}} | {{KV1_MS}} ({{KV1_PCT}}%) | {{KV2_MS}} ({{KV2_PCT}}%) | ... |
-->
{{DECODE_BREAKDOWN_ROWS}}

### Prefill TTFT — per-op breakdown (ms / % of TTFT)

| op | {{PREFILL_S_HEADERS}} |
|---|{{PREFILL_S_SEPARATORS}}|
<!-- One row per op, columns for each S:
| {{OP_NAME}} | {{S1_MS}} ({{S1_PCT}}%) | {{S2_MS}} ({{S2_PCT}}%) | ... |
-->
{{PREFILL_BREAKDOWN_ROWS}}

## Roofline: theoretical floor vs measured

<!--
Theoretical floor = sum over analytically-modelable ops of max(bytes/BW, FLOP/XMX-peak)
— the fastest this HW could run each op given its memory traffic / compute.
Measured = summed cliloader kernel time of the same ops.
achieved % = theoretical / measured (100% = on the roofline ceiling).
Recurrent ops with no analytic byte model (e.g. GatedDeltaNet `*_ref`) are excluded
from the ratio and reported separately as "unmodeled"; full = measured + unmodeled
= the real TPOT / TTFT.
-->

### Decode (per output token)

| prompt P | KV | theoretical (ms) | measured (ms) | achieved % | unmodeled (ms) | full TPOT (ms) |
|---:|---:|---:|---:|---:|---:|---:|
<!-- One row per KV/prompt:
| {{P}} | {{KV}} | {{THEO_MS}} | {{MEAS_MS}} | {{ACHIEVED_PCT}}% | {{UNMODELED_MS}} | {{FULL_TPOT_MS}} |
-->
{{DECODE_ROOFLINE_ROWS}}

### Prefill (TTFT over S tokens)

| S | theoretical (ms) | measured (ms) | achieved % | unmodeled (ms) | full TTFT (ms) |
|---:|---:|---:|---:|---:|---:|
<!-- One row per S:
| {{S}} | {{THEO_MS}} | {{MEAS_MS}} | {{ACHIEVED_PCT}}% | {{UNMODELED_MS}} | {{FULL_TTFT_MS}} |
-->
{{PREFILL_ROOFLINE_ROWS}}

## Decode tables (1 query token, KV = context length)

<!--
One table per KV length. Sorted by total ms descending.
Template for each KV:
-->
<!-- REPEAT for each KV in {{KV_LENGTHS}}: -->
### Decode — KV={{KV}}

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
<!-- One row per op, sorted by total ms desc:
| {{OP}} | {{KERNEL}} | {{SINGLE_MS}} | {{CALLS}} | {{TOTAL_MS}} | {{GFLOPS}} | {{GBS}} | {{EFF_PCT}}% | {{BOUND}} |
-->
{{DECODE_KV_TABLE_ROWS}}
| **TOTAL** |  |  |  | **{{DECODE_KV_TOTAL_MS}}** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

<!-- END REPEAT -->

## Prefill tables (single forward over S tokens)

<!--
One table per sequence length. Sorted by total ms descending.
Template for each S:
-->
<!-- REPEAT for each S in {{SEQ_LENGTHS}}: -->
### Prefill — S={{S}}

| op | kernel | single ms | calls | total ms | GFLOPS | GB/s | Eff% | bound |
|---|---|---:|---:|---:|---:|---:|---:|---|
<!-- One row per op, sorted by total ms desc:
| {{OP}} | {{KERNEL}} | {{SINGLE_MS}} | {{CALLS}} | {{TOTAL_MS}} | {{GFLOPS}} | {{GBS}} | {{EFF_PCT}}% | {{BOUND}} |
-->
{{PREFILL_S_TABLE_ROWS}}
| **TOTAL** |  |  |  | **{{PREFILL_S_TOTAL_MS}}** |  |  |  |  |

_Note: SwiGLU `multiply` (silu(gate)·up) is fused into the SwiGLU primitive and does not appear here as a standalone kernel; see *Graph fusion notes* above._

<!-- END REPEAT -->

## Op → kernel names (cliloader)

<!--
Each logical op and the actual GPU kernel(s) it dispatches (one bench process per op).
List every kernel the op fires, in launch order; launches/call = times each kernel
fires per single op invocation (per layer). Decode measured at M=1, prefill at the
largest S — kernel selection can vary with shape. When an op dispatches multiple
kernels, join the names (and the launch counts) in one cell with <br>.
Examples: an FC fires `dynamic_quantize_gpu_opt` + `gemm_kernel`; PA decode fires
`pa_kv_cache_update` + attention + finalization; MoE fires the fused
`moe_3gemm_swiglu_*` / `grouped_micro_gemm` family + gather/scatter/reorder kernels.
-->

### Decode (M=1)

| op | kernel name(s) | launches/call |
|---|---|---:|
<!-- One row per op; for multi-kernel ops join with <br>:
| {{OP}} | `{{KERNEL_1}}`<br>`{{KERNEL_2}}` | {{LAUNCHES_1}}<br>{{LAUNCHES_2}} |
-->
{{DECODE_OP_KERNEL_ROWS}}

### Prefill (S={{MAX_S}})

| op | kernel name(s) | launches/call |
|---|---|---:|
<!-- One row per op (see Decode note above for multi-kernel formatting): -->
{{PREFILL_OP_KERNEL_ROWS}}

## Per-kernel decomposition (cliloader kernel names)

<!--
Each op maps to one or more GPU kernels. Show the actual kernel names from cliloader
Device Performance Timing (dynamic_quantize_gpu_opt, pa_kv_cache_update_ref,
finalization kernels, etc.). One representative KV/S is shown (not every length),
sorted by total ms desc and truncated to the top contributors.

For prefill FC: dynamic_quantize_gpu_opt + gemm_kernel
For PA decode: pa_kv_cache_update_ref + attention_kernel + finalization
For PA prefill: pa_kv_cache_update_ref + sdpa_micro__prefill
-->

### Decode sub-kernels — KV={{REP_KV}} (representative)

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
<!-- One row per sub-kernel, sorted by total ms desc (top contributors):
| {{OP}} | `{{KERNEL_NAME}}` | {{SINGLE_MS}} | {{LAUNCHES}} | {{CALLS}} | {{TOTAL_MS}} | {{PCT}}% |
-->
{{DECODE_SUBKERNEL_ROWS}}

### Prefill sub-kernels — S={{REP_S}} (representative)

| op | kernel name | single ms | launches/call | calls/inf | total ms | % |
|---|---|---:|---:|---:|---:|---:|
<!-- One row per sub-kernel, sorted by total ms desc (top contributors):
| {{OP}} | `{{KERNEL_NAME}}` | {{SINGLE_MS}} | {{LAUNCHES}} | {{CALLS}} | {{TOTAL_MS}} | {{PCT}}% |
-->
{{PREFILL_SUBKERNEL_ROWS}}

## Top contributors (sorted by total ms per inference)

### Decode

| KV | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
<!-- One row per KV:
| {{KV}} | {{TOP1_OP}} {{TOP1_MS}}ms ({{TOP1_PCT}}%) | {{TOP2_OP}} {{TOP2_MS}}ms ({{TOP2_PCT}}%) | {{TOP3_OP}} {{TOP3_MS}}ms ({{TOP3_PCT}}%) |
-->
{{DECODE_TOP_ROWS}}

### Prefill

| S | top1 (ms,%) | top2 | top3 |
|---:|---|---|---|
<!-- One row per S:
| {{S}} | {{TOP1_OP}} {{TOP1_MS}}ms ({{TOP1_PCT}}%) | {{TOP2_OP}} {{TOP2_MS}}ms ({{TOP2_PCT}}%) | {{TOP3_OP}} {{TOP3_MS}}ms ({{TOP3_PCT}}%) |
-->
{{PREFILL_TOP_ROWS}}

## End-to-end (prefill TTFT + {{GEN_TOKENS}}-token decode)

| prompt P | TTFT (ms) | {{GEN_TOKENS}}-tok decode (ms) | total (ms) | avg decode tok/s |
|---:|---:|---:|---:|---:|
<!-- One row per prompt length:
| {{P}} | {{TTFT_MS}} | {{DECODE_WINDOW_MS}} | {{TOTAL_MS}} | {{AVG_DECODE_TOKS}} |
-->
{{END_TO_END_ROWS}}

## Key findings

<!--
3–6 bullets summarizing the headline results: decode tok/s and what bounds it,
which ops dominate, achieved % of the roofline (decode vs prefill), and any
kernel-selection / scaling anomalies worth flagging.
-->
{{KEY_FINDINGS}}

## Optimization levers (highest ROI first)

<!--
Ranked, actionable levers tied to the measured bottlenecks (e.g. expert batching,
INT4 LM-head, fusing unfused ops, speculative decoding). Each bullet: what to change
and the expected effect.
-->
{{OPTIMIZATION_LEVERS}}

## Comparison with other platforms

<!--
Optional section. Compare key metrics across platforms when data is available.
Useful for understanding how the same model scales across GPU configurations.
-->
{{COMPARISON_TABLE}}

## Caveats & method

- Each op profiled in its own process via cliloader Device Performance Timing; we use mean kernel time per iteration.
- FC weight bytes count {{MATMUL_QUANT}} weight + FP16 scale/zp(g=128) + FP16 act + FP16 out.
- PA bytes assume {{KV_CACHE_DTYPE}} KV cache + FP16 Q, FP16 out.
- Decode FC is treated as **memory-bound** (weights read dominates at M=1); prefill FC is **{{PREFILL_FC_XMX_TYPE}} XMX compute-bound** (S big enough to hit XMX).
- Prefill PA at S≥2048 is compute-bound (FP16 micro-kernel); decode PA is memory-bound.
- swish/multiply/add eltwise are typically fused into matmul/SwiGLU in real inference; they are listed for visibility.
- lm_head is run only once per token (last position in prefill, every step in decode).
- Target machine: {{TARGET_MACHINE}}

## Reproduction

```{{SHELL_TYPE}}
{{REPRODUCTION_COMMANDS}}
```
