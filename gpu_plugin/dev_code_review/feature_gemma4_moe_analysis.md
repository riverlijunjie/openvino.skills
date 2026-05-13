# Gemma4 MoE 分析报告

## 一、Gemma4 vs Qwen3.5-MoE 的 MoE 架构对比

| 特性 | **Gemma4** | **Qwen3.5-MoE** |
|---|---|---|
| **MoE 层部署** | 部分层启用 (`enable_moe_block`) | **每层都是 MoE** |
| **Dense MLP** | 有，与 MoE **并行执行**，结果相加 | 无单独的 dense MLP |
| **Shared Expert** | **无**（用并行 dense MLP 代替） | **有**，带 sigmoid 门控 (`shared_expert_gate`) |
| **Expert 结构** | Gated MLP (gate\_up fused + down) | Gated MLP (gate\_up fused + down)，**完全相同** |
| **Expert 权重形状** | `[E, 2*inter, hidden]` + `[E, hidden, inter]` | 同上 |
| **Router** | RMSNorm + learnable scale + softmax + topk + **per_expert_scale** | 简单 Linear + softmax + topk |
| **TopK 归一化** | topk weights 归一化后再乘 `per_expert_scale` | topk weights 归一化（无 per-expert scale） |
| **组合方式** | `dense_MLP_output + MoE_output + residual` | `shared_expert * sigmoid(gate) + routed_experts + residual` |
| **典型配置** | num\_experts 未公开默认值 | 256 experts, top\_k=8 |

**核心区别**: Gemma4 采用 **"parallel dense + sparse MoE"** 架构：同一层内 dense MLP 和 MoE 各自独立处理同一输入（residual），两路输出分别经过 post-layernorm 后相加。而 Qwen3.5-MoE 采用经典的 **"shared expert + routed experts"** 架构，shared expert 的输出经 sigmoid 门控后与 routed experts 相加。

## 二、Gemma4 Dense MLP 与 MoE Experts 权重 Shape 对比

### Dense MLP 权重 (`Gemma4TextMLP`)

使用标准 `nn.Linear`，权重是 **2D 张量**：

| 权重 | Shape | 说明 |
|---|---|---|
| `gate_proj.weight` | `[intermediate_size, hidden_size]` | 门控投影 |
| `up_proj.weight` | `[intermediate_size, hidden_size]` | 上投影 |
| `down_proj.weight` | `[hidden_size, intermediate_size]` | 下投影 |

默认配置：`hidden_size=2304`, `intermediate_size=9216`，所以：
- `gate_proj`: `[9216, 2304]`
- `up_proj`: `[9216, 2304]`
- `down_proj`: `[2304, 9216]`

### MoE Experts 权重 (`Gemma4TextExperts`)

将所有 expert 权重打包为 **3D 张量**（`nn.Parameter`，非 `nn.Linear`）：

| 权重 | Shape | 说明 |
|---|---|---|
| `gate_up_proj` | `[num_experts, 2 * moe_intermediate_size, hidden_size]` | gate 和 up 融合 |
| `down_proj` | `[num_experts, hidden_size, moe_intermediate_size]` | 下投影 |

### 关键区别

| 维度 | Dense MLP | MoE Experts |
|---|---|---|
| **权重维度** | 2D（单个网络） | 3D（第 0 维是 expert 索引） |
| **gate/up 是否融合** | **分离**：`gate_proj` 和 `up_proj` 各一个 Linear | **融合**：`gate_up_proj` 合并为一个参数，前半 `[:inter]` 是 gate，后半 `[inter:]` 是 up |
| **intermediate_size** | `config.intermediate_size`（默认 **9216**） | `config.moe_intermediate_size`（独立配置，通常**远小于** 9216） |
| **计算方式** | `nn.Linear`（矩阵乘法） | `nn.functional.linear(x, weight[expert_idx])`（逐 expert 索引切片后做矩阵乘法） |
| **总参数量** | 固定：`3 × hidden × inter` | `num_experts × (2 × moe_inter × hidden + hidden × moe_inter)` |

### 具体 shape 对比示例

假设 `hidden_size=2304, intermediate_size=9216, moe_intermediate_size=1024, num_experts=16`：

```
Dense MLP:
  gate_proj.weight:  [9216, 2304]      ← 21.2M params
  up_proj.weight:    [9216, 2304]      ← 21.2M params
  down_proj.weight:  [2304, 9216]      ← 21.2M params
  总计: 63.7M params

MoE Experts:
  gate_up_proj:      [16, 2048, 2304]  ← 75.5M params (gate+up 融合, 2*1024=2048)
  down_proj:         [16, 2304, 1024]  ← 37.7M params
  总计: 113.2M params（但每个 token 只激活 top_k 个 expert）
```

### forward 中的数据流差异

```python
# Dense MLP — 所有 token 走同一组权重
hidden = act_fn(gate_proj(x)) * up_proj(x)   # gate/up 分别计算
output = down_proj(hidden)

# MoE Expert — 每个 expert 用 gate_up 融合权重
gate, up = F.linear(x, gate_up_proj[expert_idx]).chunk(2, dim=-1)  # 一次乘法, 切半
output = F.linear(act_fn(gate) * up, down_proj[expert_idx])
```

Dense MLP 的 `intermediate_size` 通常远大于 MoE 的 `moe_intermediate_size`（如 9216 vs 1024），因为 MoE 通过多个并行 expert 获得总容量，而 Dense MLP 需要在单个网络中提供足够的表达能力。两者并行运行、各自独立处理同一输入、结果相加后才做 residual connection。

## 三、OpenVINO GPU Plugin 的 MoE 支持现状

GPU plugin 实现了完整的 MoE pipeline，分解为以下 primitives:

| Primitive | 功能 |
|---|---|
| `moe_mask_gen` | 从 TopK indices 生成 routing masks |
| `moe_gather` | 按 expert 分配重排 tokens |
| `moe_gemm` | 选择性 expert GEMM（支持压缩权重） |
| `moe_scatter_reduction` | 加权 scatter-reduce 回原位 |
| `moe_3gemm_fused` | 完全融合的 gate\_up + SwiGLU + down + scatter |

**支持的 routing 类型**: `SOFTMAX`（标准 top-k）和 `SIGMOID_BIAS`（DeepSeek-V3 风格）

**支持的 Expert 类型**: `GEMM3_SWIGLU`（3 个 GEMM: gate/up/down，标准 MoE）和 `GEMM2_BIAS_SWIGLU_CLAMP`

**Shared Expert 融合**: `FuseMOESharedExpert` transformation 可以将 `MOE + SharedExpert(带 sigmoid gate)` 的 Add 模式融合为单个 MOE op。

### MoE 相关文件清单

#### Core Op 定义
- `src/core/dev_api/openvino/op/moe.hpp` — Internal `ov::op::internal::MOE` op
- `src/core/src/op/moe.cpp` — Core op 实现

#### GPU Plugin Op Headers
- `src/plugins/intel_gpu/include/intel_gpu/op/moe_compressed.hpp`
- `src/plugins/intel_gpu/include/intel_gpu/op/moe_3gemm_fused_compressed.hpp`

#### GPU Primitives
- `src/plugins/intel_gpu/include/intel_gpu/primitives/moe_mask_gen.hpp`
- `src/plugins/intel_gpu/include/intel_gpu/primitives/moe_gemm.hpp`
- `src/plugins/intel_gpu/include/intel_gpu/primitives/moe_gather.hpp`
- `src/plugins/intel_gpu/include/intel_gpu/primitives/moe_scatter_reduction.hpp`
- `src/plugins/intel_gpu/include/intel_gpu/primitives/moe_3gemm_fused_compressed.hpp`

#### OCL Kernel 实现
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_mlp.cl` — 主融合 MLP kernel
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_fuse.cl`
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_scatter_reduction_ref.cl`
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_scatter_reduction_opt.cl`
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_gemm.cl`
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_mask_gen.cl`
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_gather_ref.cl`

#### Transformations
- `src/plugins/intel_gpu/src/plugin/transformations/convert_moe_to_compressed.hpp/.cpp`
- `src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.hpp/.cpp`
- `src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_shared_expert.hpp/.cpp`
- `src/plugins/intel_gpu/src/plugin/transformations/keep_moe_3gemm_const_precision.hpp/.cpp`

#### Plugin Op 注册
- `src/plugins/intel_gpu/src/plugin/ops/moe.cpp`

### MOE Op 接口

#### Core Op: `ov::op::internal::MOE`
```cpp
enum class Expert_type { GEMM2_BIAS_SWIGLU_CLAMP, GEMM3_SWIGLU };
struct Config {
    Expert_type expert_type;
    float expert_alpha;  // clamp bounds
    float expert_beta;   // swish beta
};
// Inputs: hidden_states, routing_weights, topk_indices, w0, [w0_bias], w1, [w1_bias], w2, [w2_bias]
```

#### GPU-specific: `MOECompressed`
```cpp
struct Config : public MOE::Config {
    size_t hidden_size, inter_size, num_expert, num_shared_expert, top_k;
    size_t group_size;       // quantization group size
    size_t has_batch_dim;
    bool has_zp;             // has zero point (asymmetric quant)
    ov::element::Type out_type;
    RoutingType routing_type; // SOFTMAX or SIGMOID_BIAS
};
```

### 限制与约束

| 约束 | 详情 |
|---|---|
| **数据类型** | 输入/输出必须为 **f16**（优化 3GEMM 融合 kernel） |
| **权重类型** | 仅支持 **u4, i4, u8, i8** 压缩权重 |
| **Scale 类型** | 仅支持 **f16** scales |
| **Layout** | 仅支持 **bfyx** 格式 |
| **Scatter reduction** | hidden dimension 必须能被 block\_size=4 整除；rank 必须为 3 |
| **oneDNN GEMM** | 需要 **IMMAD** 硬件支持 |
| **Expert 类型** | 仅两种：`GEMM2_BIAS_SWIGLU_CLAMP` 和 `GEMM3_SWIGLU` |
| **Routing** | 仅 **Top-K** routing（无 expert-choice routing） |

## 四、OpenVINO GPU Plugin 能否直接支持 Gemma4 的 MoE？

### Expert 计算部分：✅ 完全兼容

Gemma4 的 expert 结构（gate\_up fused + SwiGLU + down）就是标准的 `GEMM3_SWIGLU` 类型，与 GPU plugin 支持的完全一致。

### Router 部分：❌ 有差异，需要适配

Gemma4 Router 有两个特殊点：

1. **RMSNorm + learnable scale + `hidden_size^(-0.5)` 缩放**：在路由前对 hidden states 做了额外的 norm+scale 处理。GPU plugin 的 `FuseMOE3GemmCompressed` 匹配的 pattern 是 `hidden_state → [Reshape] → MatMul → Softmax → TopK → ...`，不包含 Router 内部的 RMSNorm/Scale 操作。但这些操作会在 MatMul 之前作为独立的 OpenVINO op 存在于图中，**不影响 MoE 融合 pattern 的匹配**（因为 `hidden_state_m = any_input()`，matcher 不关心 MatMul 输入端的前处理）。

2. **`per_expert_scale`**：这是 Gemma4 独有的。TopK 归一化后再乘以一个 learnable 的 per-expert 缩放因子。当前 GPU plugin 的 softmax routing pattern 期望的是：
   ```
   Softmax → TopK → ReduceSum → Divide(归一化) → [Slice] → Broadcast → ScatterElementsUpdate
   ```
   Gemma4 在 Divide 之后多了一步 **Multiply(per_expert_scale[topk_indices])**（即 `Gather + Multiply`），这会打断 `FuseMOE3GemmCompressed` 的 pattern 匹配。

### 并行 Dense MLP 融合：❌ 不直接支持

现有的 `FuseMOESharedExpert` 匹配的是 Qwen 风格的 shared expert pattern：
```
Add(MOE_output, sigmoid(gate) * shared_MLP_output)
```
而 Gemma4 的组合方式是：
```
post_norm_1(dense_MLP_output) + post_norm_2(MoE_output) + residual
```
两路输出分别过了各自的 post-layernorm 后才相加，这与现有的 shared expert fusion pattern **不匹配**。

## 五、支持 Gemma4 MoE 所需的改动

| 方面 | 是否可直接支持 | 说明 |
|---|---|---|
| Expert 计算 (GEMM3\_SWIGLU) | ✅ 完全兼容 | gate\_up\_proj + SwiGLU + down\_proj |
| Softmax TopK routing（基础） | ✅ 大部分兼容 | Softmax → TopK → Normalize 可匹配 |
| `per_expert_scale` | ❌ 需要适配 | 会在 Divide 后引入额外的 Gather+Multiply，打断 fusion pattern |
| 并行 Dense MLP + MoE 融合 | ❌ 不匹配现有 pattern | 与 Qwen 的 shared expert + sigmoid gate 模式不同 |

**若要支持 Gemma4 MoE，需要**：

1. **扩展 `FuseMOE3GemmCompressed`** 的 softmax routing pattern，在 `Divide` 和 `Broadcast/Scatter` 之间允许可选的 `Gather + Multiply`（per\_expert\_scale）
2. **Dense MLP 部分可以不融合**，作为独立子图运行，MoE 部分单独融合后两者结果相加即可——这是最低成本方案
3. 或者编写新的 `FuseMOEParallelDense` transformation 将 dense MLP + MoE 一起融合，但收益有限（dense MLP 本身就能高效执行）
