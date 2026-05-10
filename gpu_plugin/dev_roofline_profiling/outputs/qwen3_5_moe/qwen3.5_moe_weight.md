# 🔍 Qwen3.5-35B-A3B Weight Analysis Report
### Source: `qwen3.5_verbose.log` (260MB, 1,086,151 lines scanned via OpenVINO GPU verbose log)

---

## 📋 1. Model Overview

| Property | Value |
|---|---|
| **Model** | Qwen3.5-35B-A3B (Mixture-of-Experts) |
| **Deployment** | OpenVINO on Intel Arc B390 GPU |
| **Architecture** | Hybrid: 40-layer decoder + Vision Encoder (27 blocks) |
| **Total Unique Constants** | 1,577 |
| **Grand Total Weight Size** | **~21,505 MB (~21.0 GB)** |
| **Attention Style** | Mixed: Standard GQA (10 layers) + Linear/RWKV Attention (30 layers) |
| **FFN Style** | MoE with 256 routed experts + 1 shared expert per layer |

---

## 📊 2. Weight Summary Table by Category (Sorted by Total Size)

| # | Weight Category | Count | Precision | Example Shape | Per-Weight Size | Total Size (MB) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | **MoE Routed Expert Weights (INT4)** | ~120 | `u4` | `256×512×2048` or `256×2048×512` | 128 MB | **15,360 MB** | 40 layers × 3 (gate/up/down) packed INT4 |
| 2 | **Linear Attn in_proj_qkv** | 30 | `f32` | `8192×2048` | 64 MB | **1,920 MB** | 30 RWKV-style layers, postponed compression |
| 3 | **Linear Attn in_proj_z** | 30 | `f32` | `4096×2048` | 32 MB | **960 MB** | RWKV z-projection weights |
| 4 | **Linear Attn out_proj** | 30 | `f32` | `2048×4096` | 32 MB | **960 MB** | RWKV output projection |
| 5 | **Standard Attn Q Proj** | 10 | `bf16` | `8192×2048` | 32 MB | **320 MB** | Layers 3,7,11,15,19,23,27,31,35,39 |
| 6 | **MoE Expert Quant Scale (f16)** | ~120 | `f16` | `256×512×16` or `256×2048×4` | 4 MB | **480 MB** | Per-expert INT4 decompression scales |
| 7 | **LM Head** | 3 | `u8` | `248320×2048` | ~486 MB | **485.71 MB** | Vocab size=248320, dim=2048; includes scale |
| 8 | **Vision Encoder MLP FC1** | 27 | `i8` | `4304×1152` | ~4.84 MB | **130.7 MB** | 27 vision blocks INT8 |
| 9 | **Vision Encoder MLP FC2** | 27 | `i8` | `1152×4304` | ~4.84 MB | **130.7 MB** | 27 vision blocks INT8 |
| 10 | **Vision Encoder Attn QKV** | 27 | `i8` | `3456×1152` | ~3.88 MB | **104.7 MB** | INT8 quantized |
| 11 | **Standard Attn O Proj** | 10 | `bf16` | `2048×4096` | 16 MB | **160 MB** | Standard attention output proj |
| 12 | **Vision Merger FC1** | 1 | `i8` | `4608×4608` | ~20.2 MB | **20.2 MB** | Vision-language alignment |
| 13 | **MoE Expert Quant ZP (u4)** | ~120 | `u4` | `256×512×16` or `256×2048×4` | 1 MB | **120 MB** | Per-expert INT4 zero points |
| 14 | **Shared Expert gate_proj** | 40 | `bf16` | `512×2048` | 2 MB | **80 MB** | 1 shared expert per layer |
| 15 | **Shared Expert up_proj** | 40 | `bf16` | `512×2048` | 2 MB | **80 MB** | 1 shared expert per layer |
| 16 | **Shared Expert down_proj** | 40 | `bf16` | `2048×512` | 2 MB | **80 MB** | 1 shared expert per layer |
| 17 | **Vision Proj (Patch Embed)** | 1 | `f16` | `1152×3×2×16×16` | ~3.4 MB | **3.4 MB** | Pixel unshuffle conv weight |
| 18 | **Vision Merger FC2** | 1 | `i8` | `2048×4608` | ~9.0 MB | **9.0 MB** | Vision-language alignment |
| 19 | **Standard Attn K Proj** | 10 | `bf16` | `512×2048` | 2 MB | **20 MB** | GQA: 4 KV heads |
| 20 | **Standard Attn V Proj** | 10 | `bf16` | `512×2048` | 2 MB | **20 MB** | GQA: 4 KV heads |
| 21 | **Vision Encoder Attn Proj** | 27 | `i8` | `1152×1152` | ~1.3 MB | **35.1 MB** | INT8 attention output |
| 22 | **Linear Attn in_proj_a/b** | 60 | `f32` | `32×2048` | 0.25 MB | **15 MB** | Low-rank RWKV params |
| 23 | **MoE Expert Gate** | 40 | `bf16` | `1×2048` | 4 KB | **0.16 MB** | Routing gate per layer |
| 24 | **Quant Scale (LM Head)** | 2 | `f16` | `248320×1` | ~485 KB | **0.95 MB** | Scale + ZP for LM head |
| 25 | **ViT Block Scale/ZP (f16)** | ~216 | `f16` | `4304×1` / `3456×1` / `1152×1` | tiny | **~3 MB** | Per-channel INT8 scales |
| 26 | **RoPE / Misc Reshape Consts** | 30+ | `f16` | `8192×1×4` | 64 KB | ~1.9 MB | RoPE sin/cos tables |

---

## 🏗️ 3. Architecture Dissection

### 3.1 Language Model Layer Pattern (40 Layers)

| Layer Index | Attention Type | FFN Type |
|---|---|---|
| 0, 1, 2 | Linear Attn (RWKV) `in_proj_qkv[8192×2048]f32` | MoE (256 experts, INT4) + Shared Expert (bf16) |
| **3** | **Standard GQA** `q[8192×2048]bf16`, `k/v[512×2048]bf16` | MoE + Shared Expert |
| 4, 5, 6 | Linear Attn (RWKV) | MoE + Shared Expert |
| **7** | **Standard GQA** | MoE + Shared Expert |
| ... | *(every 4th layer)* | MoE + Shared Expert |
| **39** | **Standard GQA** | MoE + Shared Expert |

### 3.2 MoE Expert Weight Structure (INT4 Packed)

| Tensor | Precision | Shape | Bytes | Description |
|---|---|---|---|---|
| Expert gate_proj (batched) | `u4` | `256×512×2048` | 128 MB | All 256 experts' gate weights per layer |
| Expert up_proj (batched) | `u4` | `256×512×2048` | 128 MB | All 256 experts' up weights per layer |
| Expert down_proj (batched) | `u4` | `256×2048×512` | 128 MB | All 256 experts' down weights per layer |
| Expert scale (gate/up) | `f16` | `256×512×16` | 4 MB | Group-wise INT4 dequant scale |
| Expert scale (down) | `f16` | `256×2048×4` | 4 MB | Group-wise INT4 dequant scale |
| Expert ZP (gate/up) | `u4` | `256×512×16` | 1 MB | Zero point for INT4 |
| Expert ZP (down) | `u4` | `256×2048×4` | 1 MB | Zero point for INT4 |

### 3.3 Vision Encoder Summary (27 Blocks, INT8)

| Layer Type | Precision | Shape | Per-Block | 27× Total |
|---|---|---|---|---|
| Attn QKV weight | `i8` | `3456×1152` | 3.88 MB | **104.7 MB** |
| Attn Proj weight | `i8` | `1152×1152` | 1.30 MB | **35.1 MB** |
| MLP FC1 weight | `i8` | `4304×1152` | 4.84 MB | **130.7 MB** |
| MLP FC2 weight | `i8` | `1152×4304` | 4.84 MB | **130.7 MB** |
| Scale (all layers) | `f16` | various | ~0.02 MB | **~0.6 MB** |
| Merger FC1 | `i8` | `4608×4608` | — | **20.2 MB** |
| Merger FC2 | `i8` | `2048×4608` | — | **9.0 MB** |
---

## 💡 4. Precision Summary

| Precision | Usage | Typical Weights |
|---|---|---|
| `u4` (INT4) | MoE routed experts (data + ZP) | Expert gate/up/down (packed 4-bit) |
| `f16` (FP16) | Expert dequant scales, patch embed, RoPE | Scale tensors, vision proj |
| `bf16` | Standard attention + shared experts | Q/K/V/O proj, shared expert MLP, gate |
| `f32` | Linear attention (RWKV layers) | in_proj_qkv, in_proj_z, out_proj |
| `i8` (INT8) | Vision encoder (ViT) | All 27-block vision weights + merger |
| `u8` (UINT8) | LM Head | Token vocab embedding (248320×2048) |
| `i32/i64` | Index/shape constants | Misc reshape/index ops |

---

## 📌 5. Key Observations

1. **Dominant cost is MoE experts**: ~15.4 GB (71.6%) is consumed by INT4-packed routed expert weights across 40 layers × 256 experts.
2. **Hybrid attention**: 10/40 layers use standard GQA (bf16, head_dim=128, 64Q/4KV heads), 30/40 use RWKV-style linear attention in f32 with "postponed compression" — meaning OpenVINO defers weight compression until kernel dispatch.
3. **Activated parameters (~3B)**: With 256 experts per layer but only top-K active per token, the actual parameter footprint matches the "A3B" (Active 3B) naming.
4. **Vision encoder is fully INT8**: All 27 ViT blocks use INT8 with per-channel f16 scales — typical OpenVINO INT8 quantization.
5. **LM Head in UINT8**: The 248,320-token vocabulary head is INT8-quantized, saving ~484 MB vs bf16.
6. **Group-size-16 INT4**: Expert scale shapes like `256×512×16` imply group_size=16 for INT4 quantization (i.e., 1 scale per 16 weights).
