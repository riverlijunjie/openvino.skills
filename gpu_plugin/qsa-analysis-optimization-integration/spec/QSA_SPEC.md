# QSA and QSASDPA Specification for OpenVINO

**Status:** Design-for-review / Intel GPU plugin implementation in progress

**Audience:** OpenVINO operator and plugin maintainers, graph-transformation reviewers, kernel and performance reviewers

**Scope:** Qwen Sparse Attention (QSA), including the paged-cache `QSA` operation, the dense-cache `QSASDPA` operation, the `QSAIndexerCache` state holder, and the associated graph-fusion passes.

## 1. Executive summary

Qwen Sparse Attention (QSA) is a sparse-attention mechanism consisting of an indexer and a selected-block attention stage. The indexer scores compressed key blocks, selects the top-$k$ blocks, and the main attention evaluates only the selected blocks plus the causal tail required for the current position.

This proposal exposes two related GPU operations:

- `ov::intel_gpu::op::QSA`: a paged-cache operation for callers that already manage physical pages and block tables, such as a continuous-batching scheduler.
- `ov::intel_gpu::op::QSASDPA`: an SDPA-style sibling that consumes dense, state-managed cache tensors and hides the physical page table from the model graph.

Both operations share the same `QSA::Config`, numerical semantics, host dispatch logic, and CM kernel families. `QSASDPA` lowers dense cache tensors to a degenerate paged representation at compile time; it does not introduce a second attention implementation.

The design deliberately preserves OpenVINO fallback behavior. A graph transformation is allowed to replace a generic graph only when the required topology, shapes, attributes, cache ownership, and rotary-table representation can be proven. Otherwise the original graph remains valid and is left unchanged.

## 2. Goals and non-goals

### 2.1 Goals

1. Provide a first-class Intel GPU implementation of Qwen Sparse Attention.
2. Keep the existing paged `QSA` contract stable.
3. Provide an SDPA-style interface for dense/KVCache-managed deployments.
4. Reuse the existing QSA Q0-Q3 implementation and validated CM kernels.
5. Preserve causal, ragged-token, GQA/MQA, compressed-indexer, and output-gate semantics.
6. Integrate with OpenVINO graph transformations without requiring model authors to manually construct page tables for the SDPA-style path.
7. Make unsupported cases explicit through validation or conservative graph-pass bail-out.
8. Leave room for dynamic batch, beam reordering, rotary-table repacking, and more efficient dense kernels in later revisions.

### 2.2 Non-goals for version 1

- Replacing standard `ScaledDotProductAttention` for arbitrary models.
- Implementing a general-purpose sparse-attention API for all sparsity patterns.
- Modifying or reusing the implementation of `PagedAttention` or standard `ScaledDotProductAttention`.
- Supporting arbitrary dynamic batch size in the dense `QSASDPA` lowering.
- Supporting beam reordering inside QSA kernels.
- Performing graph-side repacking of separate `full_cos` and `full_sin` rotary tables.
- Changing the semantics of the existing paged `QSA` operation.

## 3. Mathematical and semantic definition

Let:

- $D$ be the hidden size used by the indexer projection;
- $H_q$, $H_{kv}$ be the main-attention query and KV head counts;
- $D_h$, $D_v$ be the main-attention key and value head widths;
- $H_i$ be the number of indexer heads;
- $D_i$ be the indexer head width;
- $R$ be the compression ratio, also called `compress_ratio`;
- $K_b$ be the number of selected complete blocks, `block_topk`;
- $B$ be the runtime batch/sequence-slot count.

The indexer computes a query/key representation, applies the configured RMS normalization and partial rotary embedding, and forms block scores:

$$
S_{b,j} = \frac{1}{\sqrt{D_i}}
\sum_{h=0}^{H_i-1}\operatorname{ReLU}
\left(Q_{b,h} K_{j,h}^{T}\right).
$$

The score path is followed by the causal mask and `TopK`. The selected complete blocks are passed to the sparse main-attention stage. The current incomplete block is handled by the causal-tail path and is not incorrectly discarded merely because it is not complete.

The operation does not own the surrounding model projections. Main `q_proj`, `k_proj`, `v_proj`, `o_proj`, gated residual operations, and unrelated post-processing remain ordinary OpenVINO graph nodes unless a separate, independently proven transformation handles them.

## 4. Operation definitions

### 4.1 `ov::intel_gpu::op::QSA`

`QSA` is the paged-cache form. Its OpenVINO internal base operation is `ov::op::internal::QSA`; the Intel GPU class exposes the same operation under the `gpu_opset` type information.

#### Inputs

| Index | Name | Required shape/meaning |
|---:|---|---|
| 0 | `MIXED` | `[tokens, D]`; output of the gated-residual read/mix stage |
| 1 | `QUERY` | `[tokens, Hq*Dh]`, or `[tokens, 2*Hq*Dh]` when the output gate is carried with query data |
| 2 | `KEY` | Current token keys, `[tokens, Hkv*Dh]` |
| 3 | `VALUE` | Current token values, `[tokens, Hkv*Dv]` |
| 4 | `PAST_LENS` | `[num_seqs]`; history length before the current call for each sequence |
| 5 | `SUBSEQUENCE_BEGINS` | `[num_seqs+1]`; offsets into ragged token-major inputs |
| 6 | `BLOCK_INDICES` | Physical page IDs used by logical compressed blocks |
| 7 | `BLOCK_INDICES_BEGINS` | Per-sequence offsets into `BLOCK_INDICES` |
| 8 | `KEY_CACHE` | `[num_blocks, Hkv, PA_BLOCK_SIZE, Dh]` |
| 9 | `VALUE_CACHE` | `[num_blocks, Hkv, PA_BLOCK_SIZE, Dv]` |
| 10 | `INDEXER_RAW_K_CACHE` | `[num_blocks, 1, PA_BLOCK_SIZE, Di]` |
| 11 | `INDEXER_SUMMARY_CACHE` | `[num_blocks, 1, PA_BLOCK_SIZE/R, Di]` |
| 12 | `INDEXER_QK_WEIGHT` | `[H_i*D_i + D_i, D]`; fused indexer Q/K projection |
| 13 | `INDEXER_Q_NORM_WEIGHT` | `[D_i]` |
| 14 | `INDEXER_K_NORM_WEIGHT` | `[D_i]` |
| 15 | `ROTARY_COS_SIN` | Packed rotary table `[max_positions, indexer_rotary_dim]` |
| 16 | `POSITION_IDS` | `[tokens]`; absolute position or mRoPE rank used by Q1 |

#### Output

The output is the main-attention result. Its token and head dimensions follow the `QUERY`, `Config::heads_num`, `Config::kv_heads_num`, `Config::v_head_size`, and output-gate configuration.

`QSA` does not own persistent state. The caller supplies page tables and cache tensors on every execution. This is intentional: it supports scheduler-controlled physical memory and continuous batching.

### 4.2 `ov::intel_gpu::op::QSASDPA`

`QSASDPA` has the same attention and indexer semantics as `QSA`, but removes physical page-table inputs from the public graph contract. It is intended for a graph in which main K/V state is managed like standard SDPA state and indexer state is managed by `QSAIndexerCache`.

#### Inputs

| Index | Name | Required shape/meaning |
|---:|---|---|
| 0 | `MIXED` | `[tokens, D]` |
| 1 | `QUERY` | `[tokens, Hq*Dh]`, or query plus output-gate width |
| 2 | `PAST_LENS` | `[num_seqs]` |
| 3 | `SUBSEQUENCE_BEGINS` | `[num_seqs+1]` |
| 4 | `KEY_CACHE_DENSE` | `[B, Hkv, S, Dh]`; present output of a dense KV cache |
| 5 | `VALUE_CACHE_DENSE` | `[B, Hkv, S, Dv]`; present output of a dense KV cache |
| 6 | `INDEXER_RAW_K_CACHE_DENSE` | `[B, S, D_i]`; output 0 of `QSAIndexerCache` |
| 7 | `INDEXER_SUMMARY_CACHE_DENSE` | `[B, S/R, D_i]`; output 1 of `QSAIndexerCache` |
| 8 | `INDEXER_QK_WEIGHT` | `[H_i*D_i + D_i, D]` |
| 9 | `INDEXER_Q_NORM_WEIGHT` | `[D_i]` |
| 10 | `INDEXER_K_NORM_WEIGHT` | `[D_i]` |
| 11 | `ROTARY_COS_SIN` | Packed rotary table |
| 12 | `POSITION_IDS` | `[tokens]` |

`KEY` and `VALUE` current-token inputs are intentionally absent: their values have already been incorporated into the dense cache by the upstream cache operation. The QSA lowering therefore uses `skip_kv_write=true`.

#### Dense-to-paged lowering

At compile time, the Intel GPU plugin creates a degenerate page representation:

- one physical page per sequence slot;
- `PA_BLOCK_SIZE = S`, where `S` is the statically known cache capacity;
- `BLOCK_INDICES = [0, 1, ..., B-1]`;
- `BLOCK_INDICES_BEGINS = [0, 1, ..., B]`;
- dense indexer cache tensors receive the size-one indexer-head dimension required by the underlying primitive;
- Q0 is skipped because the upstream dense cache has already appended current K/V.

The Q1, Q2, and Q3 CM stages are the same stages used by `QSA`. No new attention kernel is introduced by `QSASDPA`.

#### Static-shape requirement

`PA_BLOCK_SIZE` is compiled into the CM kernel as a JIT constant. Therefore version 1 requires:

- statically known batch/sequence-slot count `B` at compilation;
- statically known cache capacity `S`;
- `S == Config::pa_block_size`;
- compatible static K/V cache layouts;
- a static `PAST_LENS` batch dimension matching `B`.

Unsupported dynamic shapes must be rejected clearly during lowering rather than silently producing an invalid dispatch.

### 4.3 `QSAIndexerCache`

`QSAIndexerCache` is a GPU-internal stateful operation for the dense SDPA-style path. It owns two independent `Variable` states:

1. raw indexer keys;
2. compressed block summaries.

`VariableExtension` exposes only one variable per node, so raw-K is the variable returned by the generic interface and summary is exposed through explicit accessors:

- `get_variable()` / `get_variable_id()` — raw-K;
- `get_summary_variable()` / `get_summary_variable_id()` — summary.

This distinction is normative for state reset, cloning, lowering, and future beam-reorder logic.

#### Inputs

| Index | Name | Meaning |
|---:|---|---|
| 0 | `MIXED` | `[tokens, D]` |
| 1 | `INDEXER_QK_WEIGHT` | Fused Q/K projection weight |
| 2 | `INDEXER_Q_NORM_WEIGHT` | Indexer Q RMSNorm gain |
| 3 | `INDEXER_K_NORM_WEIGHT` | Indexer K RMSNorm gain |
| 4 | `ROTARY_COS_SIN` | Packed rotary table |
| 5 | `PAST_LENS` | Per-sequence history lengths |
| 6 | `SUBSEQUENCE_BEGINS` | Ragged token offsets |
| 7 | `POSITION_IDS` | Absolute positions/mRoPE ranks |

#### Outputs and state transition

- Output 0: current raw-K state, `[B, S, D_i]`.
- Output 1: current summary state, `[B, S/R, D_i]`.
- The Q1 prepare kernel computes the current indexer data and updates the two state buffers.
- The operation lowers to `cldnn::qsa` with `prepare_only=true`; Q0, Q2, and Q3 are not scheduled.
- `ReadValue` and `Assign` provide persistence across independent `InferRequest::infer()` calls.

The fixed-capacity state is zero-initialized as needed and updated at offsets derived from `PAST_LENS`. `reset_state()` must clear both variables consistently.

## 5. Configuration and validation

`QSASDPA` reuses `QSA::Config` verbatim. Important fields are:

| Field | Meaning |
|---|---|
| `heads_num` | Main query head count, $H_q$ |
| `kv_heads_num` | Main KV head count, $H_{kv}$ |
| `k_head_size` | Main key width, $D_h$ |
| `v_head_size` | Main value width, $D_v$ |
| `hidden_size` | Indexer projection input width, $D$ |
| `indexer_heads` | $H_i$ |
| `indexer_kv_heads` | Indexer KV head count; current implementation requires MQA-compatible value |
| `indexer_head_dim` | $D_i$ |
| `indexer_rotary_dim` | Partial rotary width, no greater than $D_i$ |
| `budget` | Selected-token budget |
| `compress_ratio` | $R$; zero means the layer is not a QSA layer |
| `qsa_block_tokens` | Must be consistent with `compress_ratio` |
| `pa_block_size` | Compiled page/capacity size |
| `block_topk` | Must be consistent with budget and compression ratio |
| `scale` | Main attention scale, normally $1/\sqrt{D_h}$ |
| `indexer_norm_eps` | Indexer RMSNorm epsilon |
| `has_output_gate` | Whether query carries the output-gate component |
| `is_causal` | Causal attention flag |

Validation must reject inconsistent geometry, zero or incompatible dimensions, invalid compression/budget relationships, unsupported page sizes, and unsupported dynamic shapes. The method returns a reason where possible so a transformation can fall back to the original graph.

## 6. Internal kernel fusion and execution stages

The QSA primitive is divided into four ordered stages. The stages are a scheduling and
ownership boundary, not four independent OpenVINO operations: intermediate tensors may be
kept in cache or scratch memory, and the host runtime supplies the stage-specific maps and
offsets. The stages are:

| Stage | Kernel family | Main responsibility | Persistent state written |
|---|---|---|---|
| Q0 | `qsa_kv_cache_update` | Append the current projected K/V to the paged main cache | `KEY_CACHE`, `VALUE_CACHE` |
| Q1 | `qsa_prepare`, optionally `qsa_project_dpas` | Build indexer Q/K, apply normalization and RoPE, append raw K, and produce completed-block summaries | `INDEXER_RAW_K_CACHE`, `INDEXER_SUMMARY_CACHE` |
| Q2 | `qsa_score_*` | Score visible compressed blocks and select TopK candidates | Per-call score/TopK scratch |
| Q3 | `qsa_sparse_attention_*` | Read selected blocks and the causal tail, then evaluate sparse attention | Main output only |

The following subsections define the computation performed by each stage. The formulas use
one sequence slot; the implementation applies the same computation to ragged token ranges
described by `PAST_LENS` and `SUBSEQUENCE_BEGINS`.

### 6.1 Q0: main K/V cache update

Q0 receives the already projected current-token tensors `KEY` and `VALUE`. For sequence
slot $s$, token $t$, logical position $p = P_s + t$, and page size $P_A$, the logical
position is mapped to:

$$
\operatorname{page}(p) = \operatorname{BLOCK\_INDICES}[b_s + \lfloor p/P_A \rfloor],
\qquad
\operatorname{offset}(p) = p \bmod P_A,
$$

where $b_s$ is the offset from `BLOCK_INDICES_BEGINS`. For KV head $h$ and channel $d$:

$$
KCache[\operatorname{page}(p),h,\operatorname{offset}(p),d]
   \leftarrow K_{cur}[s,t,h,d],
$$

$$
VCache[\operatorname{page}(p),h,\operatorname{offset}(p),d]
   \leftarrow V_{cur}[s,t,h,d].
$$

Q0 is intentionally limited to cache update. It does not perform the Q/K/V projection,
attention score computation, or output projection. The page-table indirection is what makes
the ordinary `QSA` path suitable for externally scheduled continuous batching.

For `QSASDPA`, the upstream dense `KVCache` has already appended the current K/V. Therefore
Q0 is not scheduled (`skip_kv_write=true`); the dense cache is reinterpreted as one
degenerate page per sequence slot during lowering.

**Kernel partition:** `qsa_kv_cache_update.cm` owns only the writes above. It is scheduled
before Q1 and Q2/Q3 depend on its completion when `skip_kv_write` is false.

### 6.2 Q1: indexer preparation and summary maintenance

Q1 computes the indexer's current query and key representations and updates the compressed
history. Let the fused projection weight contain $H_iD_i$ query rows followed by $D_i$
key rows. For a mixed hidden state $x \in \mathbb{R}^{D}$:

$$
[q_{raw}, k_{raw}] = x W_{idx}^{T},
$$

where $q_{raw}$ has shape $H_i \times D_i$ and $k_{raw}$ has shape $1 \times D_i$.
The indexer applies RMS normalization independently to the Q and K legs:

$$
\widehat q_{h,j} = \frac{q_{raw,h,j}}{\sqrt{\operatorname{mean}_j(q_{raw,h,j}^{2})+\epsilon}}\,\gamma^q_j,
$$

$$
\widehat k_j = \frac{k_{raw,j}}{\sqrt{\operatorname{mean}_j(k_{raw,j}^{2})+\epsilon}}\,\gamma^k_j.
$$

Partial rotary embedding is then applied to the configured first `indexer_rotary_dim`
channels. For a rotary pair $(2j,2j+1)$ and position $p$:

$$
\operatorname{RoPE}(a,p)_{2j} = a_{2j}\cos\theta_{p,j} - a_{2j+1}\sin\theta_{p,j},
$$

$$
\operatorname{RoPE}(a,p)_{2j+1} = a_{2j}\sin\theta_{p,j} + a_{2j+1}\cos\theta_{p,j}.
$$

The remaining channels pass through unchanged. The resulting tensors are the indexer query
$q_i$ and the current raw key $k_i$.

Q1 appends $k_i$ at the logical history position. For each completed block $m$ of size
`compress_ratio` $R$, it forms the block summary:

$$
\bar{k}_{m} = \frac{1}{R}\sum_{u=0}^{R-1} k_i[mR+u].
$$

The summary is then normalized and, where configured, rotated using the block-start
position. Incomplete blocks are retained in the raw-K state but are not emitted as complete
summary candidates. This permits Q2/Q3 to apply the exact causal-tail rule instead of
rounding a partially visible block into a complete block.

Q1 is incremental: it starts from the cache position specified by `PAST_LENS`, processes
only the current token range, and updates only newly completed summaries. It does not
recompute the entire history on every call.

**Kernel partition:**

- `qsa_prepare.cm` implements projection, RMSNorm, rotary handling, raw-K writes, and the
  scalar/vector preparation path.
- `qsa_project_dpas.cm` is the DPAS projection variant used when device and geometry
  validation permit systolic execution.
- Both variants write the same logical raw-K and summary state and must remain numerically
  interchangeable within the documented tolerance.

`QSAIndexerCache` schedules Q1 only (`prepare_only=true`). `QSA` schedules Q1 as part of
the complete operation. `QSASDPA` also schedules Q1 for the current query-side indexer
data, while its persistent raw-K/summary inputs are supplied by `QSAIndexerCache`.

### 6.3 Q2: block scoring and TopK selection

Q2 scores complete compressed blocks visible at the current causal position. For query head
$h$ and block summary $\bar{k}_m$:

$$
z_{s,h,m} = q_{i,s,h}^{T}\bar{k}_{s,m}.
$$

The QSA indexer applies ReLU before head reduction, then scales by the inverse square root
of the indexer head dimension:

$$
r_{s,h,m} = \operatorname{ReLU}(z_{s,h,m}),
$$

$$
score_{s,m} = \frac{1}{\sqrt{D_i}}
\sum_{h=0}^{H_i-1} r_{s,h,m}.
$$

Only blocks whose complete portion is causally visible are eligible for the complete-block
TopK pool. Depending on the export form, the causal restriction can be represented by a
mask/select before TopK or by the equivalent runtime score map. The implementation must
preserve the order of operations: ReLU precedes head reduction and the $1/\sqrt{D_i}$
scale follows the reduction.

Q2 selects `block_topk` indices, normally related to the token budget by:

$$
block\_topk = \left\lfloor\frac{budget}{R}\right\rfloor,
$$

subject to the operation's configuration validation. The selected physical blocks are
passed to Q3 through per-call score/index scratch and runtime dispatch maps. Q2 does not
modify the persistent main KV cache.

**Kernel partition:** `qsa_score_topk_fused.cm` is the fused score-and-TopK path for
supported shapes. `qsa_score_tile_dpas.cm` provides a tiled DPAS score path, while
`qsa_score_partition.cm` handles partitioned score work when the score matrix or scratch
budget requires chunking. The separate `qsa_topk_radix_wg.cm` kernel performs the radix
work-group TopK step when that implementation is selected. The host runtime selects the
variant and constructs the corresponding `score_tile_map`/partition metadata; the
mathematical result is the same score and TopK set.

### 6.4 Q3: selected-block sparse attention

Q3 evaluates the main attention only over the selected complete blocks and the causally
visible tail. Let $\mathcal{M}_s$ be the blocks selected by Q2, and let $\mathcal{T}_s$
be the visible positions in the current incomplete block. The effective key/value set is:

$$
\mathcal{V}_s = \left(\bigcup_{m\in\mathcal{M}_s} block_m\right)\cup\mathcal{T}_s.
$$

For query head $a$, mapped KV head $g(a)$, query vector $q_a$, key $k_{g(a),p}$, and
value $v_{g(a),p}$, Q3 computes:

$$
\ell_{a,p} = scale\; q_a^{T}k_{g(a),p} + mask_{a,p},
\qquad scale = \frac{1}{\sqrt{D_h}}.
$$

It then performs numerically stable online softmax over $p\in\mathcal{V}_s$:

$$
m_a = \max_{p\in\mathcal{V}_s}\ell_{a,p},
$$

$$
\alpha_{a,p} = \frac{\exp(\ell_{a,p}-m_a)}
{\sum_{u\in\mathcal{V}_s}\exp(\ell_{a,u}-m_a)},
$$

$$
y_a = \sum_{p\in\mathcal{V}_s}\alpha_{a,p}v_{g(a),p}.
$$

The causal tail is essential for prefill and decode correctness: a currently incomplete
block may not be in the summary TopK pool, but its visible token rows must still contribute.
GQA is handled by mapping multiple query heads to the corresponding KV head. If
`has_output_gate` is enabled, Q3 also applies the per-head gate carried by `QUERY` to the
attention result before producing the operation output.

**Kernel partition:** `qsa_sparse_attention_dpas.cm` is the DPAS implementation for the
selected-block path. `qsa_sparse_attention_gqa_dpas.cm` is the GQA-specialized DPAS variant.
`qsa_sparse_attention_split.cm` and `qsa_sparse_attention_split_reduce.cm` handle split
dispatch and reduction of partial online-softmax results when selected blocks or query rows
exceed one dispatch tile. The scalar/vector fallback is `qsa_sparse_attention.cm`. The
host runtime builds `token_map`, selected-block maps, and tail metadata; all variants
implement the same online-softmax and causal-tail semantics.

### 6.5 Stage ordering and synchronization

The normal paged execution order is:

```text
Q0: write current K/V cache
  -> Q1: compute indexer Q/K and update raw-K/summary state
  -> Q2: score visible summaries and select block_topk
  -> Q3: attend to selected blocks plus causal tail
```

Q1 may overlap independent preparation work only when its cache writes cannot race with Q2.
Q2 must observe the completed Q1 summary state, and Q3 must observe Q2's selected-block
maps. The implementation therefore chains stage events and uses internal scratch buffers
whose sizes are derived from the maximum dispatch maps.

For `QSASDPA`, the order is:

```text
upstream KVCache update
  -> Q1/QSAIndexerCache state update
  -> Q1/QSASDPA query preparation
  -> Q2
  -> Q3
```

Q0 is omitted because the upstream dense KV cache owns the main K/V update. For
`QSAIndexerCache`, only the Q1 portion and its state write-back are scheduled.

The design intentionally fuses at the operation/primitive boundary while keeping
surrounding projections as ordinary graph operations. This limits semantic ownership and
allows standard OpenVINO transformations to continue operating on projections, residual
paths, and output projections.

## 7. Graph transformations

### 7.1 `QSAFusion`

`QSAFusion` recognizes the indexer score signature:

```text
MatMul(q_index, k_block, transpose_b=true)
  -> Relu
  -> ReduceSum(axis=head)
  -> scale by 1/sqrt(Di)
  -> optional causal mask/clamp
  -> TopK
```

The pass additionally proves the required cache, projection, rotary, and page-table inputs and constructs `ov::intel_gpu::op::QSA`.

### 7.2 `QSASDPAFusion`

`QSASDPAFusion` recognizes the same indexer score signature and requires:

- Q and K indexer legs derived from one shared projection;
- an indexer raw-K history backed by `ReadValue`/`Variable`;
- a downstream `ScaledDotProductAttention` node;
- main K and V operands backed by distinct cache state;
- statically provable batch and cache capacity;
- one already-packed rotary table in the supported representation.

On success it creates the paired internal operations:

```text
QSAIndexerCache -> raw-K and summary state
                 \
                  -> QSASDPA -> attention output
KVCache --------- /
```

The pass leaves q/k/v/o projections and gated-residual nodes untouched. If any requirement cannot be proven, the pass does not rewrite the graph.

For the current real Qwen export, dynamic K/V cache shapes and separate `full_cos`/`full_sin` parameters are intentional bail-out conditions. Graph-side rotary repacking is a future extension, not an implicit assumption.

## 8. Performance design

### 8.1 Kernel reuse

The primary optimization is reuse of the existing QSA CM implementation. This avoids a second numerical implementation of sparse attention and preserves the already-validated DPAS, online-softmax, and causal-tail code.

### 8.2 Avoided work

`QSAIndexerCache` avoids recomputing the complete raw-K history and its block summaries on every call. Only the new token range and newly completed blocks are processed by Q1.

The current two-operation design may execute Q1 twice per call: once to update persistent indexer state and once inside `QSASDPA` to produce the current query-side indexer data needed by Q2. This is a deliberate version-1 trade-off that avoids changing the validated Q1 kernel. A future split-indexer design may remove this duplicate work.

### 8.3 Degenerate-page cost

The dense lowering reserves cache capacity as one page per sequence slot. Compared with true paging, this can increase memory reservation and Q2 scratch upper bounds for large batches. It is efficient and simple for single-stream and small-batch deployments, while the existing paged `QSA` remains the preferred API for scheduler-driven continuous batching.

### 8.4 Dispatch reuse

`PAST_LENS` and `SUBSEQUENCE_BEGINS` are retained in `QSASDPA`. This preserves the existing `PREFILL`, `GENERATE`, and `MIXED` host dispatch paths, including ragged token offsets within the supported static-capacity envelope.

## 9. OpenVINO conformance and integration requirements

The implementation follows these OpenVINO conventions:

- public/internal operation classes are separate from Intel GPU `gpu_opset` wrappers;
- `OPENVINO_OP` and `OPENVINO_MATCHER_PASS_RTTI` identify operations and passes;
- `validate_and_infer_types()` supplies output type and shape inference;
- `clone_with_new_inputs()` preserves the concrete GPU operation type;
- `visit_attributes()` serializes operation attributes;
- `Config` validation is explicit and reusable;
- graph passes are conservative and preserve generic fallback graphs;
- stateful nodes expose variable identity and support clone/state-management requirements;
- ProgramBuilder factory registration is separate for each operation;
- no undocumented runtime state is hidden in a stateless attention operation;
- ordinary model inputs remain ordinary OpenVINO graph inputs unless a transformation proves ownership transfer.

The operation should remain plugin-internal until API stability, serialization, shape-inference, CPU/reference behavior requirements, and cross-plugin ownership are approved by the OpenVINO maintainers. The current implementation is therefore marked under development in the core declarations.

## 10. Extensibility roadmap

1. **Packed rotary-table normalization:** support graph-side conversion from separate cosine and sine parameters after numerical and shape proofs are available.
2. **Dynamic batch/capacity:** either introduce a true dynamic dense implementation or retain a documented fallback to paged `QSA`.
3. **Unequal per-sequence capacities:** support shared-capacity padding with explicit validation and tests.
4. **Beam reorder:** coordinate main K/V, raw-K, and summary state. The safe first implementation is eager `Gather`; a later kernel-level indirect mode can avoid full-cache copies.
5. **Q1 de-duplication:** split or expose the query-side indexer computation without duplicating persistent-cache preparation.
6. **Memory-budget controls:** add capacity hints, trim/sliding-window behavior, or select paged QSA when dense reservation is not acceptable.
7. **Reference and serialization coverage:** add broader model-import, clone, save/read, and fallback tests before promoting the operation beyond plugin-internal status.

## 11. Verification plan and current evidence

The test plan must include:

- configuration and input-count validation;
- output shape/type inference;
- clone and attribute round trips;
- paged QSA regression tests;
- dense QSASDPA prefill and decode numerical comparisons;
- mixed dispatch tests;
- dynamic-shape rejection tests;
- QSAIndexerCache persistence across two independent inference calls;
- reset-state behavior for both cache variables;
- fusion success and conservative bail-out cases;
- real-model topology checks;
- performance measurements for memory, Q1 duplication, scratch size, and latency.

The current repository contains host-side operation tests, real Intel GPU lowering tests for QSASDPA, persistent indexer-cache tests, and graph-transformation tests. The remaining roadmap items above must be treated as explicit coverage gaps rather than assumed support.

## 12. Review decisions requested

The committee is requested to review and approve:

1. the separation between paged `QSA` and dense `QSASDPA`;
2. reuse of the existing QSA primitive and CM kernel families;
3. `QSAIndexerCache` as a two-variable internal state holder;
4. the static-capacity restriction of the version-1 dense lowering;
5. conservative graph-pass bail-out as the required fallback behavior;
6. the plugin-internal status and proposed extensibility path;
7. the test and performance gates required before broader API exposure.
