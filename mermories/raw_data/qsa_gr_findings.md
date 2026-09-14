# DETAILED FINDINGS: Qwen4-Exp/Qwen3.8-Flash-Next in llama.cpp

## EXECUTIVE SUMMARY

llama.cpp has **COMPREHENSIVE IMPLEMENTATION** of Qwen4-Exp model with:
- ✅ Gated Residual (GR/Hyper-Connection) - fully implemented
- ✅ Qwen Sparse Attention (QSA) - fully implemented with indexer cache
- ✅ Per-Layer Embedding (PLE) - fully implemented with n-gram hashing
- ✅ Hybrid Attention Layers - recurrent + full attention mix

## 1. ARCHITECTURE SUPPORT

### Model Support
- **Qwen4-Exp**: `LLM_ARCH_QWEN4EXP` enum value [llama-arch.h:45]
- **Entry Point**: llama_model_qwen4exp class in src/models/qwen4exp.cpp
- **Instantiation**: llama.cpp line 326 creates llama_model_qwen4exp on QWEN4EXP

## 2. GATED RESIDUAL (GR/HYPER-CONNECTION) - FULLY IMPLEMENTED

### Conceptual Design
- **Structure**: C parallel residual branches `[n_embd, C, T]` instead of single `[n_embd, T]`
- **Default**: C=4 branches (dsv4_hc_mult parameter)
- **Operations**: READ/MIX + WRITE/COMBINE gates around each sub-layer

### Implementation Files
- **Header Definitions**: [llama-arch.h:586-591, 639-651]
  - LLM_TENSOR_HC_HEAD_NORM/DOWN/UP
  - LLM_TENSOR_HC_ATTN_NORM/DOWN/UP/INJECT
  - LLM_TENSOR_HC_FFN_NORM/DOWN/UP/INJECT
  
- **Model Loading**: [qwen4exp.cpp:44-54, 153-162, 205-213]
  - Hyperparameters: dsv4_hc_mult, hc_low_rank
  - Output dimension: n_embd_out_impl = C * n_embd
  - 4 parameter tensors per HC module (norm, down, up, inject)
  - 2 HC modules per layer (before attention, before MLP)

### Computation Graph
**File**: [qwen4exp.cpp:241-278]

**build_hc_mix() function** (READ/MIX phase):
1. Grouped RMSNorm: `ggml_rms_norm(x)` → reshape to `[C*D, T]`
2. Scale by norm weights: `ggml_mul(xn, w_norm)` 
3. Low-rank gating:
   - Down projection: `w_down: [C*D, L]` → low-rank bottleneck
   - SiLU activation with scaling: `ggml_silu(scale(lo, 1/C))`
   - Up projection + sigmoid: `sigmoid(w_up: [L, C*D])`
4. Apply gates: `ggml_mul(xn, gate)`
5. Reshape: `[n_embd, C, T]`
6. Mean reduction over branches: collapse to `[n_embd, T]`
7. Output injection weights (for write gate): `w_inject: [C*D, C]`

**build_hc_combine() function** (WRITE/COMBINE phase):
1. Sigmoid injection weights: `2 * sigmoid(inject / C)` → centers on 1.0
2. Reshape sub-layer output to match branch count
3. Residual add with gated write: `residual + gate * output`
4. Result shape: `[n_embd, C, T]`

### Per-Layer Tensor Layout
- Input to layer: `[n_embd, C, T]` (wide residual)
- Attention output: `[n_embd, T]` (mixed by GR)
- MLP output: `[n_embd, T]` (mixed by GR)
- Output of layer: `[n_embd, C, T]` (written back to branches)

### Graph Integration [qwen4exp.cpp:372-430]
```
res_hc: [n_embd, C, T] initialized as C copies of embedding
└─> for each layer:
    ├─ GR_READ: res_hc → [n_embd, T] mixed
    ├─ ATTENTION: QSA or linear based on layer
    ├─ GR_WRITE: inject gate + residual combine
    ├─ GR_READ: before MLP
    ├─ MLP/FFN
    ├─ GR_WRITE: before next layer
└─ Final GR_MIX: res_hc → output norm + projection
```

## 3. QWEN SPARSE ATTENTION (QSA) - FULLY IMPLEMENTED

### Conceptual Design
- **Indexer**: Lightweight scoring mechanism to select important context blocks
- **Block Compression**: Group tokens into blocks, compress with pooling
- **Top-K Selection**: Select indexer_top_k best blocks at block granularity
- **Main Attention**: Run on selected sparse set of full attention blocks

### Hyperparameters [qwen4exp.cpp:56-61]
| Parameter | Type | Description |
|-----------|------|-------------|
| indexer_n_head | uint32 | Number of indexer heads (typically 1) |
| indexer_head_size | uint32 | Dimension per indexer head (typically 64) |
| indexer_top_k | uint32 | Number of top-k blocks to select |
| dsv4_compress_ratios | array | Per-layer compression ratios (4 or custom) |

### Cache Structure [llama-memory-hybrid-idx.h:9-150]

**Class Hierarchy**:
- `llama_memory_hybrid_idx` extends `llama_memory_hybrid`
  - Adds third cache: indexer key cache (one key per token)
  - Block cell mapping: `[n_kv, n_stream]` indices
  - QSA bias: block-level scoring biases

**QSA-specific Methods**:
- `set_input_qsa()`: [llama-memory-hybrid-idx.h:86, cpp:269]
  - Input: cell_blk (cell→block mapping), blk_cells (block→cells), blk_pos (token positions), bias
  - Output: prepared tensors for QSA graph

**Context Wrapper**: `llama_memory_hybrid_idx_context`
- `get_idx()`: Return indexer cache context
- `set_input_qsa()`: Delegate to underlying memory

### Tensor Projections [qwen4exp.cpp:223-228]
**Full Attention Layers Only** (not recurrent):
```
Per layer:
├─ Main Attention Q/K/V:
│  ├─ wq: [n_embd, D_head*n_head*2] (Q + gate interleaved per head)
│  ├─ wk: [n_embd, D_head*n_kv_heads]
│  ├─ wv: [n_embd, D_head*n_kv_heads]
│  └─ wo: [D_head*n_head, n_embd]
├─ Main Attention Norms:
│  ├─ attn_q_norm: [D_head]
│  ├─ attn_k_norm: [D_head]
└─ Indexer Projections:
   ├─ index_q_proj: [n_embd, indexer_n_head*indexer_head_size]
   ├─ index_k_proj: [n_embd, indexer_head_size]
   ├─ index_q_norm: [indexer_head_size]
   └─ index_k_norm: [indexer_head_size]
```

### QSA Input Structure [qwen4exp.cpp:473-530]

**Class**: `llm_graph_input_qsa : public llm_graph_input_i`

**Tensors**:
- `k_idxs`: `I32[n_tokens]` - indexer key cell indices
- `cell_blk`: `I32[n_kv, n_stream]` - cell to block mapping
- `blk_cells`: `I32[ratio*n_blocks, n_stream]` - block to token cells
- `blk_pos`: `I32[4*n_blocks*n_stream]` - token positions in blocks
- `bias`: `F32[n_blocks or n_kv, n_tokens/n_stream]` - per-block scores

**Validation**: `can_reuse()` checks tensor shape consistency for prefill/decode

### Top-K Selection [qwen4exp.cpp:525-600]
**Function**: `build_qsa_top_k()`
- Input: mctx_hyb (memory context), mixed representation, position tensor
- Compute: Block-wise indexer scores
- Output: Top-k indices and scores for sparse attention

## 4. PER-LAYER EMBEDDING (PLE) - FULLY IMPLEMENTED

### Conceptual Design
- **N-gram Hashing**: Use local context (n-gram) to index embedding table
- **Per-Layer**: Different embedding heads per layer
- **Sparse**: Only specific layers use PLE (configured via is_ple_impl)

### Configuration [qwen4exp.cpp:66-106]

| Parameter | Type | Description |
|-----------|------|-------------|
| ple_n_heads | uint32 | Total number of embedding heads |
| ple_ngram_size | uint32 | Context window for n-gram hashing |
| ple_heads_per_ngram | uint32 | Heads per n-gram position |
| ple_conv_kernel | uint32 | Conv1d kernel size for context |
| ple_eos_token_id | uint32 | EOS token ID |
| ple_image_token_id | uint32 | Image token ID (optional) |

**Computed Values**:
- ple_n_heads = (ple_ngram_size - 1) * ple_heads_per_ngram
- ple_head_dim = n_embd_per_layer

### Tensor Structure [qwen4exp.cpp:164-190]

**Global Tensors**:
- `per_layer_tok_embd`: `F32[ple_head_dim, ple_rows]` - flat n-gram embedding table
  - ple_rows = max(ple_head_offsets[h] + ple_head_vocab_sizes[h])

**Per-Layer Tensors** [qwen4exp.cpp:223-229]:
```
layer.ple_key:        [n_embd, C*D] - n-gram key projection
layer.ple_value:      [n_embd, D]   - value projection
layer.ple_norm_key:   [C*D]         - key norm
layer.ple_norm_query: [C*D]         - query norm
layer.ple_norm_conv:  [C*D]         - conv history norm
layer.ple_conv1d:     [ple_conv_kernel, C*D] - conv1d weights
```

**Per-Head Metadata**:
- `ple_head_offsets[LLAMA_MAX_PLE_HEADS]`: Row offset in embedding table
- `ple_head_vocab_sizes[LLAMA_MAX_PLE_HEADS]`: Vocab size per head
- `ple_layer_multipliers[LLAMA_MAX_PLE_NGRAM]`: Layer scale factors

### Integration [qwen4exp.cpp:376-382, 1192+]
- Only runs on recurrent (linear) attention layers
- Builds PLE embeddings: `build_ple()` function
- Integrated into residual after first HC stage per layer

## 5. HYBRID ATTENTION LAYERS

### Layer Classification [qwen4exp.cpp:100-104]

**Per-layer Flags**:
- `is_recr_impl[layer]`: 1=linear/recurrent, 0=full attention
- `is_ple_impl[layer]`: 1=use PLE embedding

**Default Pattern**: 
```
Every full_attention_interval-th layer (default 4):
├─ Layer 0: Recurrent (linear attention)
├─ Layer 1: Recurrent
├─ Layer 2: Recurrent
├─ Layer 3: Full Attention + QSA
└─ Repeat
```

### Linear Attention Layers [qwen4exp.cpp:217-242]

**SSM-based (State Space Model) Mix**:
```
layer.wqkv:       [n_embd, key_dim*2 + value_dim]
layer.wqkv_gate:  [n_embd, value_dim]
layer.ssm_conv1d: [ssm_d_conv, key_dim*2 + value_dim]
layer.ssm_dt:     [ssm_dt_rank] (bias)
layer.ssm_a:      [ssm_dt_rank]
layer.ssm_beta:   [n_embd, n_v_heads]
layer.ssm_alpha:  [n_embd, n_v_heads]
layer.ssm_norm:   [head_v_dim]
layer.ssm_out:    [value_dim, n_embd]
```

**Build Path**: `build_layer_attn_linear()` [qwen4exp.cpp:847+]

### Full Attention Layers [qwen4exp.cpp:210-228]

**With QSA Sparse Selection**:
```
Main Attention (Q/K/V):
├─ Q: [n_embd, D_head*n_head*2] (gate interleaved)
├─ K: [n_embd, D_head*n_kv_heads]
├─ V: [n_embd, D_head*n_kv_heads]
└─ O: [D_head*n_head, n_embd]

Indexer (for block selection):
├─ Q_proj: [n_embd, indexer_heads*indexer_dim]
├─ K_proj: [n_embd, indexer_dim]
└─ Norms: Q/K RMSNorm
```

**Build Path**: `build_layer_attn()` [qwen4exp.cpp:761+]

## 6. BUILD RUNTIME & EXECUTION PATH

### Graph Building [qwen4exp.cpp:356-430]

**Input Setup**:
- `build_inp_embd()`: Token embedding
- `build_inp_mem_hybrid()`: Create llama_memory_hybrid_idx
- `build_inp_pos()`: Position info
- `build_inp_out_ids()`: Output token selection
- `build_inp_ple()`: PLE embedding computation

**Layer Loop**:
```cpp
for (int il = 0; il < n_layer; ++il) {
  // 1. PLE (if enabled)
  if (hparams.is_ple(il))
    res_hc = build_ple(...)
  
  // 2. Attention Path (GR READ)
  cur = build_hc_mix(res_hc, ...)  // [n_embd, C, T] → [n_embd, T]
  
  if (hparams.is_recr(il))
    cur = build_layer_attn_linear(...)  // SSM
  else
    cur = build_layer_attn(mctx_hyb, ...)  // QSA
  
  // 3. GR WRITE (combine attention output)
  res_hc = build_hc_combine(res_hc, cur, inject, il)
  
  // 4. MLP Path (GR READ)
  cur = build_hc_mix(res_hc, ...)  // [n_embd, C, T] → [n_embd, T]
  cur = build_layer_ffn(cur, il)
  
  // 5. GR WRITE (combine MLP output)
  res_hc = build_hc_combine(res_hc, cur, inject, il)
}

// 6. Output (GR READ for final norm)
cur = build_hc_mix(res_hc, ...)  // No inject
cur = build_lora_mm(model.output, cur)  // Logits
```

### Memory Context Setup [llama-model.cpp:2543-2572]

When building llama_memory for Qwen4-Exp:
1. Detect: indexer_top_k > 0 → needs QSA
2. Create: `llama_memory_hybrid_idx` (instead of standard hybrid)
3. Initialize: Both main KV cache and indexer cache
4. Set up: Context wrapper `llama_memory_hybrid_idx_context`

## 7. MISSING/PARTIAL FEATURES

### NOT Implemented in llama.cpp CPU/GPU Kernels:
- ❌ **Fused QSA Kernels**: Uses standard attention decomposition
- ❌ **Fused HC Kernels**: Uses standard operations (RMSNorm, GEMM, activation)
- ❌ **Fused PLE Kernels**: Uses standard gather/conv1d

### GGML Operations Used:
- `ggml_rms_norm()` - for GR RMSNorm
- `ggml_mul()` - for gate multiplication
- `ggml_sigmoid()`/`ggml_silu()` - for activations
- `ggml_scale()` - for scaling
- `ggml_reshape_*()` - for tensor reshaping
- `ggml_mat_mul()` - standard matmul via LoRA path

### Fused Operations (reference):
- [llama-graph.h:43-50] Lists fused op types:
  - LLM_FUSED_OP_FLASH_ATTN
  - LLM_FUSED_OP_GDN_AR / LLM_FUSED_OP_GDN_CH (for delta-net)
  - LLM_FUSED_OP_LIGHTNING_INDEXER
  - LLM_FUSED_OP_DSV4_HC_* (DeepSeek-V4 HC fusion, not Qwen4)

**DeepSeek-V4 HC Comparison** [deepseek4.cpp:290+]:
- Has fused: `LLM_FUSED_OP_DSV4_HC_PRE`, `_COMB`, `_POST`
- Qwen4-Exp uses unfused path with LoRA-MM building blocks

## 8. KEY STRUCTURAL DIFFERENCES FROM OTHER MODELS

### Vs DeepSeek-V4:
- DeepSeek-V4: Fused HC operations, full Sinkhorn normalization
- Qwen4-Exp: Unfused HC, simple low-rank gating
- Both: Share HC multiplier and QSA indexer concepts

### Vs Standard Transformer:
- No single residual pathway (C branches instead)
- No layer norm after each sub-layer (integrated in GR)
- Sparse attention on selected blocks (not full sequence)
- Hybrid attention/linear (MoE-like but per-layer)

## 9. TENSOR DIMENSION TABLES

### Model Hyperparameters [qwen4exp.cpp:1-5]
```
n_embd = 3584 (Qwen3.8-Flash-Next base)
C (hc_mult) = 4
D (hc_low_rank) ≈ 320 (per design)
n_head = H (full attention layers)
n_kv_heads = KV heads (GQA)
indexer_head_size = 64
indexer_n_head = 1
indexer_top_k = K_top
dsv4_compress_ratio = 4 (default)
```

### Tensor Dimensions
| Tensor | Shape | Description |
|--------|-------|-------------|
| res_hc (residual) | [D, C, T] | Wide residual with 4 branches |
| hc_head_norm | [C*D] | HC output norm (global) |
| hc_attn_norm | [C*D] | HC attention read norm (per layer) |
| hc_attn_down | [C*D, L] | HC attention low-rank projection |
| hc_attn_up | [L, C*D] | HC attention up-projection |
| index_k_proj | [D, 64] | Indexer key projection |
| index_q_proj | [D, 64] | Indexer query projection |
| ssm_conv1d | [K, 2*S + V] | Linear attention conv1d |
| ple_key | [D, C*D] | PLE n-gram key projection |

## 10. WHAT OPENVINO COULD BORROW

### Architecture Patterns:
1. **Gated Residual Design**:
   - Low-rank gate bottleneck for efficiency (HC_DOWN → SiLU → HC_UP)
   - Per-branch RMSNorm for stable multi-stream training
   - Flexible branch count (C parameter)

2. **Sparse Attention Indexer**:
   - Lightweight separate scorer for block selection
   - Decoupled from main attention (reusable pattern)
   - Block-level granularity (vs token-level)

3. **Per-Layer Embeddings**:
   - N-gram hashing for context-aware embedding
   - Sparse activation (only on certain layers)
   - Conv1d history tracking

4. **Hybrid Attention Mix**:
   - Recurrent + full attention per-layer (not all-or-nothing)
   - SSM as alternative to transformer attention
   - Per-layer configuration flexibility

### Implementation Practices:
- Grouped RMSNorm (normalize per-branch, scale all)
- LoRA-compatible projection paths
- Stateless graph building (no layer-internal state)
- Block-based sparse selection (enables CUDA Graph compatibility)

### Integration Points for OV:
1. **GPU Plugin** (Intel GPU): Implement QSA/HC as custom operations
   - Reference: PA cache update kernel pattern in llama-memory-hybrid-idx
   - Block compression logic from DSV4_HCA in llama-kv-cache-dsv4.cpp

2. **CPU Plugin**: Leverage GGML-style operations
   - RMSNorm with grouped variant
   - LoRA matmul fusion

3. **Model Conversion**: Handle tensor layout transformation
   - HC dimensionality encoding [n_embd, C, T]
   - PLE embedding table organization
   - Indexer cache initialization
