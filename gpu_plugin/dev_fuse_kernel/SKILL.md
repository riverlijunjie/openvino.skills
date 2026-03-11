---
name: dev_fuse_kernel
description: Develop new fused ops/kernels in OpenVINO GPU plugin — full pipeline from core op to OCL kernel. Case study based on FusedRMSNorm implementation.
---

# FusedRMSNorm — 从零开发 OpenVINO GPU 自定义融合算子的完整流程

## 1. 概述

本文档记录了在 OpenVINO GPU plugin 中实现 `FusedRMSNorm` 和 `FusedRMSNormResidual` 自定义融合算子的完整过程，涵盖 10 个新增文件和 5 个修改文件，形成从 Core IR Op → GPU Primitive → OCL Kernel 的完整链路。

### 1.1 优化动机

Qwen3.5 模型中每个 decoder layer 包含 2 次 RMSNorm 调用（`input_layernorm` + `post_attention_layernorm`），40 层模型共 80 次调用。原始实现将 RMSNorm 分解为多个基础算子：

```
x → Cast(f32) → Power(2) → ReduceMean → Add(eps) → Sqrt → Divide → Cast(f16) → Multiply(weight)
```

每次 RMSNorm 产生 5-8 个独立 kernel launch，80 次调用产生约 400-640 个 kernel，显著增加 dispatch overhead。

### 1.2 目标

将 RMSNorm 的分解子图替换为单一自定义算子 `FusedRMSNorm`，从而：
- 减少 kernel launch 次数（5-8 个 → 1 个）
- 消除中间 tensor 的内存读写
- 对于 residual 变体，进一步融合 `x + residual` 加法

---

## 2. 实现架构 — 10 层管线

OpenVINO GPU plugin 的自定义算子需要穿越一条完整的 10 层管线。每一层都是必需的，缺少任何一层都会导致编译或运行时错误：

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Core Op Header (.hpp)   — IR 节点定义                  │
│  Layer 2: Core Op Impl (.cpp)     — validate_and_infer_types    │
│  Layer 3: GPU Primitive (.hpp)    — cldnn 原语定义               │
│  Layer 4: Plugin Bridge (.cpp)    — ov::Node → cldnn::primitive │
│  Layer 5: Graph Instance (.h)     — typed_program_node 模板      │
│  Layer 6: Graph Layout (.cpp)     — calc_output_layouts          │
│  Layer 7: Registry Entry (.cpp)   — ImplementationManager 注册   │
│  Layer 8: OCL Impl (.hpp/.cpp)    — JIT constants + dispatch    │
│  Layer 9: OCL Kernel (.cl)        — 实际 GPU 计算逻辑            │
│  Layer 10: Global Registration    — primitives_list + registry   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 各层实现详解

### Layer 1: Core Op Header
**文件**: `openvino/src/core/dev_api/openvino/op/fused_rmsnorm.hpp`

定义两个 op 类，注册在 `ov::op::internal` 命名空间：

```cpp
// FusedRMSNorm: 2 inputs (x, weight), 1 output
class OPENVINO_API FusedRMSNorm : public ov::op::Op {
    OPENVINO_OP("FusedRMSNorm", "internal");
    // attribute: float m_eps
};

// FusedRMSNormResidual: 3 inputs (x, residual, weight), 2 outputs
class OPENVINO_API FusedRMSNormResidual : public ov::op::Op {
    OPENVINO_OP("FusedRMSNormResidual", "internal");
    // attribute: float m_eps
};
```

**关键要点**：
- 使用 `OPENVINO_OP` 宏注册，第二参数 `"internal"` 指定 op set namespace
- `visit_attributes` 必须序列化 `m_eps`，否则 model cache 会丢失参数
- Residual 变体有 2 个输出：`output[0]` = normalized, `output[1]` = x + residual

### Layer 2: Core Op Implementation
**文件**: `openvino/src/core/src/op/fused_rmsnorm.cpp`

```cpp
void FusedRMSNorm::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 2);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
```

**关键要点**：
- 构造函数必须调用 `constructor_validate_and_infer_types()`
- 输出 shape/dtype 与 input x 完全一致
- `clone_with_new_inputs` 必须传递 `m_eps` attribute

### Layer 3: GPU Primitive Definition
**文件**: `openvino/src/plugins/intel_gpu/include/intel_gpu/primitives/fused_rmsnorm.hpp`

```cpp
struct fused_rmsnorm : public primitive_base<fused_rmsnorm> {
    CLDNN_DECLARE_PRIMITIVE(fused_rmsnorm)
    float _eps;
    bool _has_residual;  // 统一两种模式到一个 primitive
};
```

**设计决策**：用 `_has_residual` 布尔标志统一两个 op 变体到一个 primitive，而不是创建两个独立的 primitive。这样只需一个 kernel 文件（通过 `#if HAS_RESIDUAL` 区分路径）。

**必须实现**：`hash()`, `operator==()`, `save()`, `load()` — 用于 kernel cache 和序列化。

### Layer 4: Plugin Bridge (ov::Node → cldnn::primitive)
**文件**: `openvino/src/plugins/intel_gpu/src/plugin/ops/fused_rmsnorm.cpp`

```cpp
static void CreateFusedRMSNormOp(ProgramBuilder& p, const std::shared_ptr<FusedRMSNorm>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    cldnn::fused_rmsnorm prim(layerName, inputs, op->get_eps(), /*has_residual=*/false, /*num_outputs=*/1);
    p.add_primitive(*op, prim);
}
REGISTER_FACTORY_IMPL(internal, FusedRMSNorm);

static void CreateFusedRMSNormResidualOp(ProgramBuilder& p, const std::shared_ptr<FusedRMSNormResidual>& op) {
    validate_inputs_count(op, {3});
    auto inputs = p.GetInputInfo(op);
    cldnn::fused_rmsnorm prim(layerName, inputs, op->get_eps(), /*has_residual=*/true, /*num_outputs=*/2);
    p.add_primitive(*op, prim);
}
REGISTER_FACTORY_IMPL(internal, FusedRMSNormResidual);
```

**关键要点**：两个 `REGISTER_FACTORY_IMPL` 宏分别对应两个 core op，但都映射到同一个 `cldnn::fused_rmsnorm` primitive。

### Layer 5: Graph Instance Header
**文件**: `openvino/src/plugins/intel_gpu/src/graph/include/fused_rmsnorm_inst.h`

这是模板特化样板代码：

```cpp
template <>
struct typed_program_node<fused_rmsnorm> : public typed_program_node_base<fused_rmsnorm> {
    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
```

**注意**：`get_shape_infer_dependencies` 返回空，因为 RMSNorm 的输出 shape 完全由 input x 决定，不需要额外的 shape 依赖。

### Layer 6: Graph Layout Computation
**文件**: `openvino/src/plugins/intel_gpu/src/graph/fused_rmsnorm.cpp`

```cpp
GPU_DEFINE_PRIMITIVE_TYPE_ID(fused_rmsnorm)

template <typename ShapeType>
std::vector<layout> fused_rmsnorm_inst::calc_output_layouts(...) {
    auto x_layout = impl_param.get_input_layout(0);
    std::vector<layout> output_layouts;
    output_layouts.emplace_back(x_layout.get_partial_shape(), x_layout.data_type, x_layout.format);
    if (desc->_has_residual) {
        output_layouts.emplace_back(x_layout.get_partial_shape(), x_layout.data_type, x_layout.format);
    }
    return output_layouts;
}
```

**关键要点**：`GPU_DEFINE_PRIMITIVE_TYPE_ID` 宏必须存在，否则运行时报 "primitive type not registered"。

### Layer 7: Registry Entry
**文件**: `openvino/src/plugins/intel_gpu/src/graph/registry/fused_rmsnorm_impls.cpp`

```cpp
const std::vector<std::shared_ptr<ImplementationManager>>& Registry<fused_rmsnorm>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::FusedRMSNormRef, shape_types::any)
    };
    return impls;
}
```

### Layer 8: OCL Implementation (JIT + Dispatch)
**文件**: `openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/fused_rmsnorm_ref.hpp` (验证)
**文件**: `openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/fused_rmsnorm_ref.cpp` (JIT + dispatch)

**ImplementationManager (validate_impl)**:
```cpp
bool validate_impl(const program_node& node) const override {
    // 只支持 bfyx format, f16/f32 数据类型
    static constexpr std::array supported_fmts = { format::bfyx };
    static constexpr std::array supported_types = { ov::element::f16, ov::element::f32 };
    // ... 校验所有 input 和 output
}
```

**KernelGenerator (JIT constants)**:
```cpp
JitConstants get_jit_constants(const RuntimeParams& params) const override {
    jit.make("HIDDEN_SIZE", hidden_size);  // 必须是 static dim
    jit.make("IO_TYPE", io_type);          // 0=f16, 1=f32
    jit.make("EPS", desc->_eps);
    jit.make("HAS_RESIDUAL", desc->_has_residual ? 1 : 0);
}
```

**DispatchDataFunc (workgroup sizing)**:
```cpp
// One workgroup per row, 256 work-items per workgroup
wgs.global = {num_rows * 256, 1, 1};
wgs.local = {256, 1, 1};
```

**⚠️ 关键教训**：`get_jit_constants()` 在编译期调用，此时 dynamic shapes（如 batch/seq_len）尚未具象化。**绝对不能**在 `get_jit_constants()` 中对 dynamic dimension 调用 `get_length()`，否则会抛异常。而 `add_stage()` 会**静默吞掉**该异常，导致 kernel 从未被编译，运行时报 "Kernel not found in kernel cache"。

**正确做法**：dynamic dimension 的计算（如 `num_rows`）放在 `DispatchDataFunc` lambda 中，它在运行时调用，此时所有 shape 都已具象化。

### Layer 9: OpenCL Kernel
**文件**: `openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/fused_rmsnorm_ref.cl`

三步算法：
1. **Sum of squares**: 每个 work-item 按 stride 遍历 hidden dim，累加 `val * val`
2. **Workgroup reduction**: 通过 shared local memory 的 tree reduction 汇总所有 work-item 的结果
3. **Normalize + scale**: `rsqrt(mean_sq + eps) * val * weight`

```cl
// Step 1: partial sum-of-squares
float local_sq_sum = 0.0f;
for (uint i = lid; i < HIDDEN_SIZE; i += LOCAL_SIZE) {
    float val = (float)x[row_offset + i];
    #if HAS_RESIDUAL
    val += (float)residual[row_offset + i];
    #endif
    local_sq_sum += val * val;
}

// Step 2: workgroup reduction in LDS
shared_sum[lid] = local_sq_sum;
barrier(CLK_LOCAL_MEM_FENCE);
for (uint stride = LOCAL_SIZE >> 1; stride > 0; stride >>= 1) { ... }

// Step 3: normalize and write
float rsqrt_val = rsqrt(shared_sum[0] / (float)HIDDEN_SIZE + (float)EPS);
for (uint i = lid; i < HIDDEN_SIZE; i += LOCAL_SIZE) {
    output[row_offset + i] = (OUTPUT_TYPE)(val * rsqrt_val * weight[i]);
}
```

### Layer 10: Global Registration
**文件** (修改): `primitives_list.hpp` + `registry.hpp`

```cpp
// primitives_list.hpp — 让 plugin 识别 core op
REGISTER_FACTORY(internal, FusedRMSNorm);
REGISTER_FACTORY(internal, FusedRMSNormResidual);

// registry.hpp — 让 graph 编译器找到 impl
REGISTER_IMPLS(fused_rmsnorm);
```

**⚠️ 关键教训**：`primitives_list.hpp` 中的 `REGISTER_FACTORY` 是必需的。遗漏它会导致运行时报 "FusedRMSNorm of type FusedRMSNorm(internal) is not supported"，因为 plugin 无法将 core op 转换为 GPU primitive。

---

## 4. GenAI 模型层调用变更

### ops.cpp — 算子构造
**文件** (修改): `openvino.genai/src/cpp/src/modeling/ops/ops.cpp`

原始代码（分解实现）:
```cpp
Tensor rms(const Tensor& x, const Tensor& weight, float eps) {
    auto xf = x.to(f32);
    auto var = xf.pow(2.0f).mean(-1, true);
    auto norm = xf * (var + eps).rsqrt();
    return norm.to(orig_dtype) * weight;
}
```

修改后（融合算子）:
```cpp
Tensor rms(const Tensor& x, const Tensor& weight, float eps) {
    auto* ctx = resolve_context(x, weight);
    auto node = std::make_shared<ov::op::internal::FusedRMSNorm>(
        ov::OutputVector{x.output(), weight.output()}, eps);
    return Tensor(node->output(0), ctx);
}

std::pair<Tensor, Tensor> rms_residual(const Tensor& x, const Tensor& residual,
                                        const Tensor& weight, float eps) {
    auto* ctx = resolve_context(x, weight);
    auto node = std::make_shared<ov::op::internal::FusedRMSNormResidual>(
        ov::OutputVector{x.output(), residual.output(), weight.output()}, eps);
    return {Tensor(node->output(0), ctx), Tensor(node->output(1), ctx)};
}
```

### modeling_qwen3_5_text.cpp — 模型调用
**文件** (修改): `openvino.genai/src/cpp/src/modeling/models/qwen3_5/modeling_qwen3_5_text.cpp`

```cpp
Tensor Qwen3_5RMSNorm::forward(const Tensor& x) const {
    auto w_adjusted = 1.0f + weight().to(f32);  // Qwen3.5 uses (1 + w) scaling
    return ops::rms(x, w_adjusted, eps_);
}

std::pair<Tensor, Tensor> Qwen3_5RMSNorm::forward(const Tensor& x, const Tensor& residual) const {
    auto w_adjusted = 1.0f + weight().to(f32);
    return ops::rms_residual(x, residual, w_adjusted, eps_);
}
```

---

## 5. 遇到的运行时错误及修复

### Bug #1: "FusedRMSNorm of type FusedRMSNorm(internal) is not supported"
- **原因**：遗漏了 `primitives_list.hpp` 中的 `REGISTER_FACTORY` 宏
- **修复**：添加 `REGISTER_FACTORY(internal, FusedRMSNorm)` 和 `REGISTER_FACTORY(internal, FusedRMSNormResidual)`
- **教训**：每个 core op 都需要在 `primitives_list.hpp` 中注册工厂，这是 plugin 将 IR 节点映射到 GPU primitive 的入口

### Bug #2: "Kernel for {fusedrmsnorm:FusedRMSNorm_3053} is not found in the kernel cache!"
- **原因**：`get_jit_constants()` 中对 dynamic batch/seq 维度调用了 `get_length()`，导致抛异常。而 `add_stage()` 内部的 try-catch **静默吞掉了异常**，kernel 从未进入编译队列
- **修复**：将 `NUM_ROWS` 计算从 `get_jit_constants()` 移到 `DispatchDataFunc` lambda（运行时才执行，此时 shape 已具象化）
- **教训**：`get_jit_constants()` 只能使用**静态已知**的维度（如 `hidden_size`）。动态维度必须在 dispatch lambda 中处理。调试时需注意 `add_stage()` 的静默异常吞噬行为

---

## 6. 性能分析 — 为什么没有提升？

### 6.1 分析结果

实测 FusedRMSNorm 后的性能指标：
- TTFT: 2218.86ms, TPOT: 27.25ms/token, Throughput: 36.69 tokens/s
- 与优化前**无明显差异**

### 6.2 根因分析

通过 VTune profiling CSV 分析发现：

| 组件 | GPU 时间 (ms) | 占比 | 实例数 |
|------|-------------|------|--------|
| gemm_kernel | 22.096 | 52.3% | 211 |
| moe_3gemm | 10.599 | 25.1% | 160 |
| **rms_gpu_bfyx_opt** | **0.301** | **0.71%** | **30** |
| 其他 | 9.278 | 21.9% | 1397 |
| **总计** | **42.274** | **100%** | **1798** |

**核心发现**：OpenVINO GPU plugin **已经内置了** `ov::op::internal::RMS` → `rms_gpu_bfyx_opt` 的完整优化路径！

1. **已有 RMS Fusion Pass** (`rms_fusion.cpp`)：自动检测分解的 RMSNorm 子图（Power → ReduceMean → Add → Sqrt → Divide → Multiply），合并为单一 `ov::op::internal::RMS` 节点
2. **已有 GPU RMS Primitive** (`rms.hpp`/`rms.cpp`)：将 `internal::RMS` 映射到 `cldnn::rms` primitive
3. **已有优化 kernel** (`rms_gpu_bfyx_opt.cl`)：使用 subgroup operations（`sub_group_reduce_add`）、block read/write、自适应 SIMD 宽度（1/2/4/8）、动态 workgroup sizing

这意味着：**原始分解代码被 RMSFusion pass 自动捕获，已经运行在高度优化的 `rms_gpu_bfyx_opt` 上了！**

我们的 `FusedRMSNorm` 实质上是用一个 **reference 品质的 kernel**（固定 256 workgroup、标量加载、LDS tree reduction）替换了一个 **production 优化的 kernel**（subgroup intrinsics、向量化 block I/O、自适应 workgroup sizing），不但没有提升，反而可能退化。

### 6.3 内置 `rms_gpu_bfyx_opt` vs 我们的 `fused_rmsnorm_ref` 对比

| 特性 | `rms_gpu_bfyx_opt` (已有) | `fused_rmsnorm_ref` (新增) |
|------|---------------------------|---------------------------|
| Reduction | `sub_group_reduce_add` | LDS tree reduction |
| 数据读取 | `DT_INPUT_BLOCK_READ` (1/2/4/8 宽) | 标量 `x[offset + i]` |
| 数据写入 | `DT_OUTPUT_BLOCK_WRITE` (向量化) | 标量 `output[offset + i]` |
| WG sizing | 自适应（根据 `maxWorkGroupSize`) | 固定 256 |
| SIMD width | `REQD_SUB_GROUP_SIZE(16)` | 无指定 |
| Fused ops 支持 | `HAS_FUSED_OPS` (activation/quantize/eltwise) | 无 |
| 动态 shape | 完整支持 + dynamic padding | 部分支持 |
| 残差融合 | ❌ 无 | ✅ `HAS_RESIDUAL` |

### 6.4 唯一有价值的部分

`FusedRMSNormResidual`（残差融合变体）是**真正的新功能** — 已有的 `rms_gpu_bfyx_opt` **不支持**残差加法融合。但即使完美融合，残差加法只是一个 eltwise add（约 0.007ms/次），40 层节省 ≈ 0.28ms，占总 GPU 时间 42.3ms 的 0.66%，收益极其微小。

### 6.5 总结教训

> **在实现任何自定义融合算子之前，必须先检查 OpenVINO 的已有 transformation passes 和 kernel_selector 是否已经涵盖了该优化路径。**

检查清单：
1. 在 `openvino/src/common/transformations/` 搜索是否有对应的 fusion pass
2. 在 `openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/` 搜索已有 kernel
3. 在 profiling CSV 中确认实际运行的 kernel 名称
4. 真正值得优化的瓶颈是 gemm（52.3%）和 MoE（25.1%），而非 RMSNorm（0.71%）

---

## 7. 新增/修改文件完整清单

### 新增文件 (10 个)
| # | 文件路径 | 管线层级 | 说明 |
|---|---------|---------|------|
| 1 | `openvino/src/core/dev_api/openvino/op/fused_rmsnorm.hpp` | Layer 1 | Core Op 头文件 |
| 2 | `openvino/src/core/src/op/fused_rmsnorm.cpp` | Layer 2 | Core Op 实现 |
| 3 | `openvino/src/plugins/intel_gpu/include/intel_gpu/primitives/fused_rmsnorm.hpp` | Layer 3 | GPU Primitive 定义 |
| 4 | `openvino/src/plugins/intel_gpu/src/plugin/ops/fused_rmsnorm.cpp` | Layer 4 | Plugin Bridge |
| 5 | `openvino/src/plugins/intel_gpu/src/graph/include/fused_rmsnorm_inst.h` | Layer 5 | Graph Instance 头文件 |
| 6 | `openvino/src/plugins/intel_gpu/src/graph/fused_rmsnorm.cpp` | Layer 6 | Layout 计算 |
| 7 | `openvino/src/plugins/intel_gpu/src/graph/registry/fused_rmsnorm_impls.cpp` | Layer 7 | Registry Entry |
| 8 | `openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/fused_rmsnorm_ref.hpp` | Layer 8 | OCL Impl (验证) |
| 9 | `openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/fused_rmsnorm_ref.cpp` | Layer 8 | OCL Impl (JIT+Dispatch) |
| 10 | `openvino/src/plugins/intel_gpu/src/graph/impls/ocl_v2/fused_rmsnorm_ref.cl` | Layer 9 | OpenCL Kernel |

### 修改文件 (5 个)
| # | 文件路径 | 变更说明 |
|---|---------|---------|
| 1 | `openvino.genai/.../ops/ops.cpp` | `rms()` 改为创建 FusedRMSNorm 节点；新增 `rms_residual()` |
| 2 | `openvino.genai/.../ops/ops.hpp` | 新增 `rms_residual()` 声明 |
| 3 | `openvino.genai/.../modeling_qwen3_5_text.cpp` | `Qwen3_5RMSNorm::forward` 改为调用 `ops::rms`/`ops::rms_residual` |
| 4 | `openvino/.../plugin/primitives_list.hpp` | 新增 2 个 `REGISTER_FACTORY` |
| 5 | `openvino/.../registry/registry.hpp` | 新增 `REGISTER_IMPLS(fused_rmsnorm)` |

---

## 8. 开发自定义融合算子的通用流程

基于本次实践，总结出 OpenVINO GPU plugin 自定义算子的标准开发流程：

### Step 1: 可行性评估（最重要！）
- [ ] 搜索已有 transformation passes（`openvino/src/common/transformations/`）
- [ ] 搜索已有 kernel_selector kernels（`kernel_selector/kernels/`）
- [ ] 用 VTune profiling 确认目标 kernel 在总时间中的占比
- [ ] 评估理论收益是否值得实现成本

### Step 2: Core Op 定义
- [ ] 创建 `.hpp` 头文件，定义 Op 类（`OPENVINO_OP` 宏注册）
- [ ] 创建 `.cpp` 实现，`validate_and_infer_types()` + `clone_with_new_inputs()`
- [ ] 确保 `visit_attributes()` 序列化所有属性

### Step 3: GPU Primitive
- [ ] 创建 primitive `.hpp`（继承 `primitive_base`，实现 `hash/==/save/load`）
- [ ] 创建 plugin bridge `.cpp`（`REGISTER_FACTORY_IMPL` 宏）

### Step 4: Graph 层
- [ ] 创建 `_inst.h`（typed_program_node 和 typed_primitive_inst 模板特化）
- [ ] 创建 graph `.cpp`（`GPU_DEFINE_PRIMITIVE_TYPE_ID` + `calc_output_layouts`）

### Step 5: OCL 实现
- [ ] 创建 registry `.cpp`（`Registry<T>::get_implementations()`）
- [ ] 创建 impl `.hpp`（`ImplementationManager` + `validate_impl`）
- [ ] 创建 impl `.cpp`（`KernelGenerator` + `PrimitiveImplOCL`）
- [ ] 创建 `.cl` kernel

### Step 6: 全局注册
- [ ] `primitives_list.hpp` 添加 `REGISTER_FACTORY`
- [ ] `registry.hpp` 添加 `REGISTER_IMPLS`

### Step 7: 模型层集成
- [ ] 修改 ops 构造函数
- [ ] 修改模型 forward 调用

### Step 8: 构建验证
```bash
# 三段式构建
cd openvino/build && make openvino -j$(nproc)
cd openvino/build && make openvino_intel_gpu_plugin -j$(nproc)
cd openvino.genai/build && make modeling_qwen3_5 -j$(nproc)
```

---

## 9. 关键设计原则

1. **JIT constants 只能使用静态维度**：`get_jit_constants()` 在编译期调用，dynamic dimension 的 `get_length()` 会抛异常
2. **Dispatch lambda 处理动态维度**：`DispatchDataFunc` 在运行时调用，此时所有 shape 已具象化
3. **`add_stage()` 静默吞异常**：kernel 编译失败不会报错，只会在运行时报 "Kernel not found in cache"
4. **先查已有路径，再造轮子**：OpenVINO 已内置大量 fusion pass + optimized kernel，重复实现浪费精力
5. **优化要对准瓶颈**：profiling 数据决定优先级，0.71% 的 op 不值得单独优化