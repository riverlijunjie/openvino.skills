# NNCF LoRA QAT — Sub-4-bit Quantization Analysis

> Source: `/home/ov2022/workspace/nncf/examples/llm_compression/torch/distillation_qat_with_lora`  
> Core implementation: `nncf/src/nncf/torch/quantization/layers.py`, `quantize_functions.py`, `strip.py`  
> Official doc: `nncf/docs/usage/training_time_compression/quantization_aware_training_lora/Usage.md`

---

## 一、Overview

NNCF LoRA QAT (`FQ_LORA` format) is a **Quantization-Aware Training** method combining:
- **INT4 asymmetric fake-quantization** (group-wise, default `group_size=64`)
- **Absorbable LoRA adapters** — low-rank matrices A, B added to each weight matrix and **fused back into INT4 weights** at deployment (zero inference overhead)
- **Knowledge distillation** from the uncompressed FP16/BF16 teacher model

This is distinct from LoftQ/PiSSA: the NNCF formulation applies fake-quantize to `W + B×A` in every forward pass, so both the LoRA adapters **and the quantization scales** are learned together. After training, A and B are absorbed back into W before re-quantization — the final deployed model has **no LoRA branch**.

### Key formula

```
Training forward pass:
  W' = W + B × A                     ← sum in bfloat16
  y  = FQ(W') × x                    ← FQ = fake-quantize (quantize + dequantize)
  FQ(W') = dequant(quant(W'))         ← differentiable, STE gradient through round()

Deployment (strip/absorb):
  W_int4 = quantize(W + B × A)       ← re-quantize the fused weight
  y      = dequant(W_int4) × x       ← pure INT4 GEMM, no LoRA overhead
```

---

## 二、Quantization Algorithm (Sub-4-bit Scope)

### 2.1 Supported Precisions

| Mode | Bits | Scheme | Used for LoRA QAT? |
|---|---|---|---|
| `INT4_ASYM` | 4 | Asymmetric (uint4 + zero_point) | ✅ **Primary** (default in example) |
| `INT4_SYM` | 4 | Symmetric (int4, no zero_point) | ✅ Supported |
| `INT8_ASYM` | 8 | Asymmetric | Only if all-8-bit; LoRA added when `is_all_8bit=True` |
| `INT8_SYM` | 8 | Symmetric | Same as above |

**Note**: The `distillation_qat_with_lora` example hard-codes `INT4_ASYM, group_size=64`. There is **no native INT3/INT2 path** in this example — it operates at INT4 only.

### 2.2 Asymmetric INT4 Quantization Math

The per-group asymmetric quantization scheme used:

```
Parameters (learned during QAT):
  input_low:   float16, shape = (num_groups, 1)   ← per-group minimum
  input_range: float16, shape = (num_groups, 1)   ← per-group range (positive)
  lora_A:      bfloat16, shape = (rank, in_features)
  lora_B:      bfloat16, shape = (out_features, rank)

level_low  = 0       (unsigned INT4, asymmetric)
level_high = 15
levels     = 16

Forward pass (asymmetric_quantize_lora):
  W' = (W + B @ A).to(W.dtype)        ← fused weight (float16)
  # TuneRange: adjust input_low/input_range so zero maps exactly to an integer level
  input_low_adj, input_range_adj = TuneRange(input_low, input_range, levels)
  scale_per_group = input_range_adj / level_high          ← float16
  zero_point      = round(-input_low_adj / scale_per_group)  ← clamped to [0, 15]
  # Quantize (round-to-nearest, STE in backward):
  q = clamp(round((W' - input_low_adj) / scale_per_group), 0, 15)   ← uint4 in [0,15]
  # Dequantize (fake-quantize output):
  W_fq = q * scale_per_group + input_low_adj                         ← float16
```

**TuneRange** is a key NNCF-specific step: it adjusts `input_low` and `input_range` so that the float zero (0.0) always maps to an exact integer quantum boundary, preventing asymmetric drift of zero across gradient steps.

### 2.3 Quantization Scale Initialization

Before QAT training starts, the model is initialized using **Post-Training Weight Compression** (default: AWQ + Scale Estimation):

```python
model = nncf.compress_weights(
    model,
    mode=CompressWeightsMode.INT4_ASYM,
    group_size=64,
    awq=True,
    scale_estimation=True,
    compression_format=CompressionFormat.FQ_LORA,
    dataset=dataset,           # 128 calibration samples
)
```

This initializes `input_low`, `input_range` from AWQ-calibrated scales — much better than random init. Then QAT fine-tunes these learned scales further.

### 2.4 LoRA Adapter Initialization

Rather than the standard LoRA init (Gaussian A, zero B), NNCF initializes from **SVD of a small random residual** near the quantization granularity:

```python
# In torch_backend.py:
svd_residual = torch.rand(weight_shape) * scale / 100    # tiny perturbation ~1/100 of quant step
svd_residual = svd_residual.reshape(orig_weight_shape)
U, S, V = torch.linalg.svd(svd_residual, full_matrices=False)
# Absorb sqrt(S) into both factors:
B = U[:, :rank] @ diag(sqrt(S[:rank]))    # shape: (out_features, rank)
A = diag(sqrt(S[:rank])) @ V[:rank, :]   # shape: (rank, in_features)
```

This ensures `B @ A` ≈ 0 at initialization (small magnitude), so training starts near the PTQ baseline. This is simpler than LoftQ's SVD-of-residual approach but achieves similar initialization stability.

---

## 三、Weight Layout (Quantized Storage)

### 3.1 During QAT Training (FQ_LORA format)

During training, weights remain in **full floating-point** (BF16/FP16). The quantization is *fake* — it is applied in the forward pass for gradient computation but the stored weights are not actually quantized:

```
Module weight W:          (out_features, in_features)   float16/bfloat16
                           stored as normal PyTorch parameter

Hook (AsymmetricLoraQuantizer):
  input_low:              (num_groups, 1)               float16  — learnable
  input_range:            (num_groups, 1)               float16  — learnable
  lora_A:                 (rank, in_features)           bfloat16 — learnable
  lora_B:                 (out_features, rank)          bfloat16 — learnable

num_groups = out_features × in_features // group_size
             (for group_size=64, in_features=4096, out_features=4096:
              num_groups = 4096 × 4096 // 64 = 262144)
```

The weight tensor `W` itself is **not modified** during training. Only the hook state (`input_low`, `input_range`, `lora_A`, `lora_B`) is updated.

### 3.2 Group-wise Reshape for Scale Application

The weight shape is **reshaped** for per-group scale computation inside `QuantizeAsymmetricTorch`:

```
Original weight W:    (out_features, in_features)
Reshaped W':          (out_features × in_features // group_size, group_size)
                      = (num_groups, group_size)

input_low:            (num_groups, 1)   — broadcasts over group_size dimension
input_range:          (num_groups, 1)
```

After fake-quantize, output is reshaped back to `(out_features, in_features)`. The `weight_shape` stored in `PTLoraSpec` records the reshaped shape; `orig_weight_shape` records the original shape.

### 3.3 After Strip/Absorb — DQ Format (Deployment)

After training, `nncf.strip(model, strip_format=StripFormat.DQ)` converts `FQ_LORA → DQ`:

```python
# strip.py: replace_quantizer_to_compressed_weight_with_decompressor()
# For AsymmetricLoraQuantizer:

# 1. Compute fused quantized weight (LoRA is absorbed here):
qdq_weight = quantizer.quantize(weight)     # applies B@A fusion internally
q_weight   = round(clamp((qdq_weight - input_low) / scale, 0, 15))  # uint4

# 2. Pack 2 × uint4 into 1 × uint8:
packed = pack_uint4(q_weight.uint8())       # shape: (num_groups × group_size // 2,)

# 3. Store packed INT4 weight back into weight parameter:
weight_param.data = packed                  # replaces float16 weight

# 4. Replace quantizer hook with lightweight INT4AsymmetricWeightsDecompressor:
decompressor = INT4AsymmetricWeightsDecompressor(
    scale=scale,           # float16, shape=(num_groups, 1)
    zero_point=zero_point, # uint4 packed into uint8, shape=(num_groups//2, 1) or similar
    compressed_weight_shape=q_weight.shape,   # (num_groups, group_size)
    result_shape=orig_weight_shape,           # (out_features, in_features)
    result_dtype=float16,
)
```

**Final deployed weight layout (INT4_ASYM, group_size=64):**

```
Packed weight tensor:
  dtype:  torch.uint8
  shape:  (out_features × in_features // 2,)   ← 2 uint4 per byte
  content: two uint4 values [q_even, q_odd] per byte
           lower nibble = q_even, upper nibble = q_odd
           values in [0, 15]

Scale buffer (_scale):
  dtype:  torch.float16
  shape:  (out_features × in_features // group_size, 1)
          = (num_groups, 1)

Zero-point buffer (_zero_point):
  dtype:  torch.uint8 (packed uint4, 2 per byte)
  shape:  (num_groups // 2, 1)
  values: zero_point ∈ [0, 15], meaning W_float_zero = zero_point × scale + input_low
```

**Memory cost for a single Linear(4096, 4096) layer with group_size=64:**

```
Packed weight:     4096 × 4096 / 2 bytes = 8 MiB → 4096 × 4096 × 0.5 = 8,388,608 bytes ≈ 8 MiB
Scale (fp16):      4096 × 4096 / 64 × 2 bytes = 524,288 bytes = 0.5 MiB
Zero-point (uint4):4096 × 4096 / 64 / 2 bytes = 131,072 bytes = 0.125 MiB
Total:             ≈ 8.625 MiB  (vs 32 MiB fp16 baseline, compression ratio ≈ 3.7×)
Effective bpw:     4 + 16/64 + 8/128 = 4.3125 bpw  (weight + scale + zp)
```

---

## 四、Dequantization Computation Flow

### 4.1 FQ_LORA Forward Pass (Training)

```
Input:
  W:           float16,  (out_features, in_features)
  lora_A:      bfloat16, (rank, in_features)
  lora_B:      bfloat16, (out_features, rank)
  input_low:   float16,  (num_groups, 1)
  input_range: float16,  (num_groups, 1)

Step 1 — LoRA fusion:
  W' = (W + lora_B @ lora_A).to(float16)    ← cast bfloat16 result back to float16

Step 2 — Reshape for group-wise quantization:
  W'_grouped = W'.reshape(num_groups, group_size)

Step 3 — TuneRange (ensure zero-alignment):
  input_high = input_range + input_low
  clamp input_low ≤ 0, input_high ≥ 0
  scale = levels / (input_high - input_low)
  zp    = round(-input_low × scale)
  adjust input_low / input_high to align zp exactly to an integer grid

Step 4 — Quantize (STE, differentiable):
  q = clamp(round((W'_grouped - input_low_adj) / scale_adj), 0, 15)    ← uint4 in [0,15]

Step 5 — Dequantize (fake-quantize output):
  W_fq = q × scale_adj + input_low_adj                                  ← float16

Step 6 — Reshape back:
  W_fq = W_fq.reshape(out_features, in_features)

Output: W_fq (float16) — used for matrix multiply with activations
```

### 4.2 DQ Decompressor Forward Pass (Inference)

After `nncf.strip(StripFormat.DQ)`, inference dequantization is:

```
Input:
  packed_weight:  uint8, shape=(out_features × in_features // 2,)
  _scale:         float16, shape=(num_groups, 1)
  _zero_point:    uint8 (packed uint4), shape=(num_groups // 2, 1)

INT4AsymmetricWeightsDecompressor.forward(x):

  Step 1 — Unpack uint4:
    q = unpack_uint4(x)                  ← shape: (num_groups × group_size,)   uint8
    q = q.reshape(compressed_weight_shape) ← shape: (num_groups, group_size)

  Step 2 — Unpack zero_point:
    zp = unpack_uint4(_zero_point)
    zp = zp.reshape(zero_point_shape)    ← shape: (num_groups, 1)

  Step 3 — Dequantize (decompress_asymmetric):
    q  = q.to(float16)                   ← cast to compute dtype
    zp = zp.to(float16)
    W_fp = (q - zp) × _scale            ← float16, shape: (num_groups, group_size)

  Step 4 — Reshape to original weight shape:
    W_fp = W_fp.reshape(result_shape)    ← shape: (out_features, in_features)
    W_fp = W_fp.to(result_dtype)         ← typically float16 or bfloat16

Output: W_fp (float16) — used for GEMM/GEMV with activations
```

**Instruction count per weight (CPU/GPU estimate):**

```
unpack_uint4:   1 shift + 1 AND per weight           = 2 ops
cast to fp16:   1 convert                            = 1 op
subtract zp:    1 sub (broadcast from num_groups)    = 1 op
multiply scale: 1 mul (broadcast from num_groups)    = 1 op
                                              Total  = 5 ops/weight
```

This is identical to standard INT4 ASYM dequant — no extra cost from the (now-absorbed) LoRA.

### 4.3 Symmetric Variant (INT4_SYM)

If `INT4_SYM` is used instead:

```
Training forward (symmetric_quantize_lora):
  W' = (W + lora_B @ lora_A).to(W.dtype)
  q  = clamp(round(W'_grouped / scale), -8, 7)    ← int4 in [-8, 7]
  W_fq = q × scale

Inference decompressor (INT4SymmetricWeightsDecompressor.forward):
  q   = unpack_int4(x)                             ← int4 unpacked to int8
  q   = q.reshape(compressed_weight_shape)
  W_fp = q × _scale                                ← no zero_point needed
  W_fp = W_fp.reshape(result_shape)

Packed storage: int8, 2 × int4 per byte (range [-8, 7] → stored as int4)
```

---

## 五、Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                        │
│                                                                  │
│  1. Load pretrained model (BF16)                                 │
│  2. compress_weights(FQ_LORA, INT4_ASYM, group_size=64,         │
│       awq=True, scale_estimation=True)                           │
│     → Insert AsymmetricLoraQuantizer hooks                       │
│     → Initialize scales from AWQ+ScaleEstimation                │
│     → Initialize lora_A, lora_B from SVD of random residual     │
│  3. Compute teacher hiddens (frozen BF16 model)                  │
│  4. Set trainable: lora_A, lora_B (lr=1e-4), scales (lr=1e-5)   │
│  5. Training loop (10 epochs, AdamW, KL-div distillation loss):  │
│     for each batch:                                              │
│       W' = W + B @ A      ← fusion inside quantizer forward     │
│       W_fq = FQ(W')       ← fake-quantize (STE gradient)        │
│       y = W_fq @ x        ← student forward                     │
│       loss = KL(y_student, y_teacher)                            │
│       loss.backward() → updates lora_A, lora_B, input_low,      │
│                         input_range                              │
│  6. strip(StripFormat.DQ):                                       │
│     → quantize(W + B @ A) → packed uint4 weight                 │
│     → replace FQ hook with INT4AsymmetricWeightsDecompressor    │
│  7. Export to OpenVINO (Optimum Intel)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 六、Comparison with Related Methods

| Feature | NNCF FQ_LORA | LoftQ | PiSSA | LR-QAT (Qualcomm) |
|---|---|---|---|---|
| **Forward formula** | `FQ(W + B@A)` | `FQ(W) + B@A` | `FQ(W - B@A) + B@A` | `FQ(W₀ + (α/r)AB)` |
| **LoRA at inference** | **Absorbed (zero overhead)** | Not absorbed (always present) | Not absorbed | Not absorbed |
| **Scale learning** | ✅ Jointly with LoRA | ❌ Fixed post-PTQ | ❌ Fixed post-PTQ | ✅ Learned step sizes |
| **LoRA init** | SVD of small random residual | SVD of quant residual `W−Q(W)` | SVD of principal W components | Standard Gaussian/zero |
| **Precision** | INT4 (asymmetric or symmetric) | INT4, NF4 | INT4, NF4 | INT4 |
| **Native NNCF** | ✅ | ❌ | ❌ | ❌ |
| **Accuracy gain** | ~50% recovery of PTQ degradation | ~40% recovery | Similar to LoftQ | Similar or slightly better |

---

## 七、OpenVINO Export Layout

After `nncf.strip` + Optimum Intel `export_from_model`:

```
OpenVINO IR (.xml / .bin):
  Each Linear layer compressed to INT4_ASYM becomes:
    Constant node:   packed uint8 weight (2 × uint4 per byte)
    Constant node:   float16 scale       (per-group)
    Constant node:   uint8 zero_point    (per-group, packed)
    Subgraph:        Convert(uint8→int32) → Subtract(zp) → Multiply(scale)
                     → Convert(float16) → [MatMul / GEMM]

  This is identical to standard NNCF INT4_ASYM PTWC output —
  the LoRA adapters leave no trace in the exported model.
```

OpenVINO GPU plugin handles this DQ subgraph via its existing INT4 dequant kernel path (same as post-training INT4 compression output).

---

## 八、Limitations and Notes

1. **Only INT4 in the example** — the `distillation_qat_with_lora` sample is hard-coded to `INT4_ASYM, group_size=64`. Sub-4-bit (INT3/INT2) is **not supported** in `FQ_LORA` format.

2. **Group size = 64** — finer than typical GPTQ (group_size=128), contributing to better accuracy at the cost of slightly more scale storage (+scale overhead: 4+16/64=4.25 bpw).

3. **LoRA rank = 256** (default) — large rank; for 4096×4096 weight: A=(256,4096), B=(4096,256), adds 256×4096×2×2 bytes = 4 MiB per layer during training. This is dropped at inference.

4. **Training cost** — ~25 min on A100 for 1.7B model, ~50 min on 3× RTX 3090. Scales linearly with model size and epochs.

5. **Accuracy recovery** — average ~39% recovery of PTQ-induced perplexity increase across tested models (up to 69% for Gemma-2-2B).

6. **Relation to 3-bit support** — `FQ_LORA` itself does not target 3-bit. For 3-bit MoE GPU support, the relevant path remains SignRoundV2 (INT2/INT4 mixed) or GGUF IQ3_S/Q3_K formats. `FQ_LORA` could theoretically be extended to INT3 if NNCF adds `CompressWeightsMode.INT3` support.

---

## 参考

- `nncf/examples/llm_compression/torch/distillation_qat_with_lora/main.py`
- `nncf/src/nncf/torch/quantization/layers.py` — `AsymmetricLoraQuantizer`, `INT4AsymmetricWeightsDecompressor`
- `nncf/src/nncf/torch/quantization/quantize_functions.py` — `asymmetric_quantize_lora`, `TuneRange`
- `nncf/src/nncf/torch/function_hook/strip.py` — `replace_quantizer_to_compressed_weight_with_decompressor`
- `nncf/src/nncf/torch/quantization/strip.py` — `asym_fq_to_decompressor`
- `nncf/src/nncf/quantization/algorithms/weight_compression/torch_backend.py` — `init_lora_adapters`, `get_fq_insertion_command`
- `nncf/docs/usage/training_time_compression/quantization_aware_training_lora/Usage.md`
