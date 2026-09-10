# GGUF Model Builder 迁移设计

## 1. 文档目的

本文档定义 GGUF 模型 builder 从 OpenVINO Native GGUF frontend 迁移到 `openvino.genai` modeling 层的中期方案。目标不是一次性重写 GGUF loader 或立即改变 OpenVINO compressed op 的公共 API，而是在保持现有 GPU 执行路径和 GGUF 权重布局兼容的前提下，逐步完成以下工作：

1. 在 `modeling/ops/ops.cpp/.hpp` 中封装 `FullyConnectedCompressed` 和 `GatherCompressed`；
2. 在 modeling 中构造 Qwen3、Qwen3.5 Dense 和 Qwen3.5 MoE 的 OpenVINO graph；
3. 将 GGUF 文件读取、header/metadata/tensor 信息解析与模型 graph 构造解耦；
4. 在 parity 验证完成后，删除 `thirdparty/openvino/src/frontends/gguf/src/builders/` 中的模型专用 builder；
5. 保持 GPU plugin 对 common compressed node 的匹配、转换、OCL/CM kernel 选择和 GGUF quantized weight ABI 不变。

本文档是迁移设计，不代表所有代码已经实现。删除 frontend builder 必须以分阶段编译、结构校验、logits 校验和真实文本生成校验通过为前提。

## 2. 结论摘要

### 2.1 推荐的职责边界

| 层 | 负责内容 | 不负责内容 |
|---|---|---|
| GGUF container reader | magic、version、header、metadata KV、tensor-info、offset/bounds 检查、mmap/文件生命周期 | Qwen attention/MLP/MoE graph topology |
| GGUF metadata/config adapter | 将通用 metadata 转换为 `ModelConfig` 及 Qwen typed config，做字段校验和默认值处理 | 创建 OpenVINO Node |
| GGUF weight source | 按 tensor name 返回 tensor 描述、quantized `ov::Constant` 或原始映射区间；持有 backing storage | 解释 transformer layer 连接关系 |
| modeling ops | 封装标准 op、common compressed op 和 modeling 专用 op | 解析 GGUF 二进制格式 |
| modeling model builder | 使用 typed config、`WeightSource` 和 modeling ops 构建 graph | 直接依赖 `GGUFReader`、读取 metadata KV 字符串 |
| Pipeline/GenAI backend | 选择 backend、创建 builder、compile model、创建 infer request | 复制 GGUF parser 或 architecture-specific graph builder |
| GPU plugin | 将 common compressed node lowering 到 GPU implementation 和 OCL/CM kernel | 依赖 modeling 内部 class 或修改 GGUF loader |

### 2.2 header 和 metadata 应该放在哪里

推荐将 GGUF header、metadata table、tensor-info table 和 mmap 管理放在独立的 GGUF loader/weight-source 层，而不是放进 Qwen model builder。

- **header/container 二进制解析**是格式层职责，应位于可复用的 `GGUFContainerReader`；
- **metadata 到模型配置的转换**是 loader/config adapter 职责；
- **Qwen3/Qwen3.5 字段到 graph 结构的解释**仍属于 modeling builder，但输入应是 typed config，而不是 metadata key/value 查询；
- **tensor mmap 生命周期**应由 `GGUFWeightSource` 或 reader-backed storage 统一持有，防止 graph 中的 `Constant` 在 reader 销毁后悬空；
- frontend 在迁移期间可以保留兼容 reader/入口，但不再承载模型 graph topology。

## 3. 当前实现与问题

### 3.1 现有 Native GGUF frontend

当前模型 builder 位于：

- `thirdparty/openvino/src/frontends/gguf/src/builders/qwen3_builder.cpp`
- `thirdparty/openvino/src/frontends/gguf/src/builders/qwen35_builder.cpp`
- `thirdparty/openvino/src/frontends/gguf/src/builders/qwen35moe_builder.cpp`
- `thirdparty/openvino/src/frontends/gguf/src/builders/builder.hpp`

这些文件同时完成四类工作：

1. 从 `GGUFReader` 读取 header/metadata；
2. 将 architecture-prefixed metadata 转换为 Qwen 配置；
3. 从 mmap 区域创建带 `gguf_*` element type 的 raw constants；
4. 构造 RMSNorm、RoPE、attention、KV cache、MLP、MoE、Gated DeltaNet 和 logits graph。

这种实现可以工作，但使 model builder 绑定到 GGUF 文件格式、Native frontend 私有类型和 reader 生命周期。结果是：

- modeling 无法直接复用完整 graph builder；
- GenAI 的 `ModelConfig`/`WeightSource` 抽象与 GGUF builder 不一致；
- frontend 和 modeling 需要分别维护 Qwen graph 逻辑；
- 删除或扩展 GGUF 格式时容易触碰模型拓扑代码；
- reader 的 mmap 生命周期容易被 graph 中的 raw `Constant` 隐式依赖。

### 3.2 当前 modeling 设施

主要位置：

- `thirdparty/openvino.genai/src/cpp/src/modeling/ops/ops.cpp`
- `thirdparty/openvino.genai/src/cpp/src/modeling/ops/ops.hpp`
- `thirdparty/openvino.genai/src/cpp/src/modeling/models/qwen3/modeling_qwen3.cpp`
- `thirdparty/openvino.genai/src/cpp/src/modeling/weights/`
- `thirdparty/openvino.genai/src/cpp/src/modeling/builder_context.cpp`

`ops.cpp` 已经使用 `ov::op::internal::MOECompressed`，因此它是封装 `FullyConnectedCompressed` 和 `GatherCompressed` 的直接参考。建模层应使用 common transformations 中的：

```cpp
ov::op::internal::FullyConnectedCompressed
ov::op::internal::GatherCompressed
```

不能直接使用 GPU 私有的：

```cpp
ov::intel_gpu::op::FullyConnectedCompressed
```

后者拥有不同的 plugin-private ABI，会把 modeling 绑定到 GPU plugin 实现细节。

## 4. 目标数据流

目标加载路径如下：

```text
model_path (.gguf)
        |
        v
GGUFContainerReader
  - header/version
  - metadata KV
  - tensor info
  - mmap/storage ownership
        |
        +--> GGUFModelConfigAdapter --> ModelConfig/QwenConfig
        |
        +--> GGUFWeightSource --------> WeightSource
                                             |
                                             v
                              modeling Qwen model builder
                              - topology
                              - modeling ops
                              - compressed weights
                                             |
                                             v
                                       ov::Model
                                             |
                                             v
                                ModelingBackend::compile()
                                             |
                                             v
                                    GPU OCL/CM lowering
```

重要原则是：builder 只看到 `ModelConfig`、typed architecture config、`WeightSource` 和 modeling ops，不看到 `GGUFReader` 的 metadata 查询 API。

## 5. Compressed ops 封装设计

### 5.1 `FullyConnectedCompressed`

common op 的 canonical 五输入顺序必须保持不变：

```text
0 activation
1 compressed weight
2 bias
3 weight decompression scale
4 weight zero point
```

现有 GGUF path 使用 quantized raw weight 和空的 dynamic placeholder 表示可选 scale/zero-point。封装层不得擅自把 `gguf_*` weight 转换为 f16/f32，否则会破坏 GPU plugin 对 quantized layout 的识别和 kernel 选择。

建议在 `ops.hpp` 增加低层接口，实际签名可按当前 `Tensor`/`Output<Node>` 类型约定调整：

```cpp
Tensor fully_connected_compressed(
    const Tensor& activation,
    const Tensor& compressed_weight,
    const Tensor& bias,
    const Tensor& weight_scale,
    const Tensor& weight_zero_point);
```

实现要求：

- 调用 `resolve_context()`，确保所有输入属于同一 `BuilderContext`；
- 构造 `ov::op::internal::FullyConnectedCompressed`；
- 保持输入顺序和空 placeholder 语义；
- 不引入 GPU plugin header；
- 不在 modeling 层实现 dequant；
- 保留 output shape/type 推导交给 OpenVINO op；
- 对 GGUF final `lm_head` 保留 compressed node 类型，使 `find_llm_matmul()` 等 utility 能识别它。

建议提供一个 GGUF 专用高层 helper，避免所有 builder 重复处理 placeholder：

```cpp
Tensor gguf_linear(
    const Tensor& activation,
    const GGUFWeight& weight,
    const std::optional<GGUFWeight>& bias,
    const GGUFWeight& scale_placeholder,
    const GGUFWeight& zero_point_placeholder);
```

`gguf_linear()` 只负责把 `WeightSource` 输出适配成 compressed op 输入，不负责读取 GGUF metadata。

### 5.2 `GatherCompressed`

common op 支持四输入和五输入形式：

```text
0 data
1 indices
2 axis
3 decompression scale
4 optional zero point
```

GGUF embedding path 的关键事实：scale input 不仅是解压参数占位符，还决定 output element type。当前 embedding 方案使用 dummy `f16` scale，目的是让 `GatherCompressed` 输出 f16，同时实际 quantization block 信息仍保留在 GGUF compressed data 中。

建议接口：

```cpp
Tensor gather_compressed(
    const Tensor& data,
    const Tensor& indices,
    const Tensor& axis,
    const Tensor& decompression_scale,
    const std::optional<Tensor>& zero_point = std::nullopt);

Tensor gguf_embedding(
    const Tensor& token_ids,
    const GGUFWeight& compressed_embedding,
    const Tensor& axis,
    ElementType output_type = element::f16);
```

实现要求：

- 四输入时构造四输入 op，五输入时构造五输入 op；
- `gguf_embedding()` 默认创建与输出 precision 一致的 dummy f16 scale；
- 保持 indices shape 加 hidden dimension 的输出语义；
- 不将大尺寸 embedding 预先 dequantize 成 f16 buffer；
- 保留 GPU `gather_gguf` lowering 所需的 `GatherCompressed` node type；
- 仅在 GPU 支持的 Q4_0/Q4_1/Q8_0/Q4_K/Q5_K/Q6_K 类型上走 compressed path，其他类型保留 dequantize + 普通 Gather fallback；
- 注意该 common op 的 `evaluate()` 返回 false，CPU fallback 不能假设它可直接执行。

### 5.3 与 GPU 私有 op 的边界

`ov::intel_gpu::op::FullyConnectedCompressed` 与 common `ov::op::internal::FullyConnectedCompressed` 不是同一个 ABI。modeling 只能构造 common node；GPU plugin 负责后续 transformation/matching 和实现选择。这样可以保持：

```text
modeling common node
  -> convert_gguf_fc_compressed.cpp
  -> OCL/CM GGUF implementation
  -> q4/q5/q6/q40/q41/q80 kernel
```

任何将 GPU-private op header 引入 modeling 的方案都应拒绝。

## 6. GGUF loader 分层

### 6.1 `GGUFContainerReader`

建议从现有 `gguf_reader.hpp/.cpp` 提取格式层对象，职责包括：

- 校验 magic 和 version；
- 读取 header 中的 tensor count 和 metadata count；
- 解析 metadata key/value table；
- 解析 tensor name、shape、GGML/GGUF type、offset；
- 检查整数溢出、offset 越界、物理 byte size 和文件长度；
- 管理 mmap 或等价 backing storage；
- 提供只读的 metadata/tensor-info 查询；
- 对字符串、数组、整数、浮点类型提供明确的类型访问接口。

它不应包含 `build_layer()`、`multi_head_attention()`、`make_fc()` 等模型拓扑函数。

### 6.2 `GGUFMetadata` 与 config adapter

metadata 应分两步处理：

1. **格式无关读取**：reader 返回 typed KV，而不是散落在 builder 中的 `get_u64/get_f64/get_str` 调用；
2. **架构相关转换**：`GGUFModelConfigAdapter` 根据 `general.architecture` 创建 `Qwen3Config`、`Qwen35Config` 或 `Qwen35MoeConfig`。

adapter 应集中处理：

- architecture 名称和版本；
- hidden size、intermediate size、layer count；
- attention head/query head/KV head；
- RoPE theta、context length、sliding window；
- RMSNorm epsilon；
- MLP/MoE router、expert 数量和 top-k；
- Qwen3.5 hybrid layer 类型、Gated DeltaNet 参数；
- tensor name convention 和可选字段默认值；
- 不兼容字段的错误信息。

builder 中不应出现以下模式：

```cpp
reader.get_u64(architecture + ".attention.head_count")
reader.get_str("general.architecture")
```

应改为：

```cpp
const auto config = gguf_config_adapter.load_qwen3(reader.metadata());
build_qwen3_model(config, weight_source, finalizer);
```

### 6.3 `GGUFWeightSource`

`GGUFWeightSource` 是 modeling `WeightSource` 与 GGUF container reader 之间的适配器，建议提供：

- `has(name)`；
- `info(name)`；
- `constant(name)`：返回保留原始 GGUF element type 的 constant；
- `data(name)`：返回只读原始字节范围及长度；
- `shape(name)`、`ggml_type(name)`；
- 可选的 tensor alias/name normalization；
- 共享 backing storage handle。

最重要的生命周期规则：

> graph 中的 compressed `Constant` 只要仍然引用 mmap 区域，`GGUFWeightSource` 就必须保持 storage alive。

可以通过 `shared_ptr<GGUFStorage>`、constant 的 aligned shared allocation，或将 storage owner 放入 `WeightSource`/`Model` 的持有对象中实现。不能让 builder 局部创建 reader，然后返回引用 reader mmap 的 `ov::Model`。

### 6.4 loader 的推荐归属

中期可以将 GGUF loader 放到 `openvino.genai` 的 loader/weights 目录，并由 Pipeline 通过已有 GenAI modeling 入口调用。若现有 `gguflib` 已经是稳定、可复用且不包含模型 topology 的库，也可以把 container reader 放在 `gguflib`，在 GenAI 侧仅实现 `GGUFModelConfigAdapter` 和 `GGUFWeightSource`。

推荐优先级：

1. 复用或扩展已有纯格式 `gguflib`；
2. 若 `gguflib` API 不适合 modeling，则新增 GenAI 内部 `modeling/loaders/gguf/`；
3. 不建议让 modeling builder 直接 include frontend 私有的 `frontends/gguf/src/gguf_reader.hpp`；
4. 过渡阶段可由 frontend compatibility wrapper 创建新的 loader 对象，但依赖方向不能反过来。

## 7. modeling builder 迁移方案

### 7.1 迁移顺序

建议按以下顺序迁移：

1. Qwen3 dense；
2. Qwen3.5 dense hybrid；
3. Qwen3.5 MoE hybrid。

每个 builder 内部继续复用已有 modeling layer/ops，例如 RMSNorm、Gated DeltaNet、LinearAttention、MoE 和 KV-cache helper，但所有权重访问统一改为 `WeightSource`。

### 7.2 builder 的目标接口

GenAI 现有 builder 形态可作为目标：

```cpp
std::shared_ptr<ov::Model> build_qwen3_model(
    const ModelConfig& config,
    WeightSource& weights,
    WeightFinalizer& finalizer);
```

GGUF 适配后，入口只需额外选择 architecture adapter：

```cpp
std::shared_ptr<ov::Model> build_model(
    const ModelConfig& config,
    WeightSource& weights,
    WeightFinalizer& finalizer);
```

其中 `config` 已经是 typed configuration，`weights` 已经能返回 raw GGUF compressed constants。builder 不应感知权重来自 GGUF、Safetensors 还是其他格式。

### 7.3 graph parity 要求

迁移后的 graph 必须与原 frontend 在以下方面保持等价：

- parameter 名称和 element type；
- KV cache 输入/输出顺序及 shape；
- attention mask、position ids、beam indices；
- RMSNorm/RoPE/activation 的计算精度；
- compressed FC/Gather/MoE node 的 input order；
- final logits 的 shape、slice 位置和 dtype；
- Qwen3.5 hybrid layer 的层类型排列；
- MoE router、expert selection 和 capacity 语义。

允许内部 helper 名称和 node friendly name 变化，但不允许改变 GPU plugin 识别所依赖的 op type、quantized element type 和输入 ABI。

## 8. frontend builder 删除策略

不能在 modeling builder 尚未完成时直接删除 `builders/`。建议采用以下过渡：

### 阶段 A：双路径

- 保留现有 Native frontend builder；
- 新增 modeling-side GGUF loader 和 Qwen3 builder；
- 用相同 GGUF 文件分别生成两个 graph；
- 比较 graph signature 和单步 logits。

### 阶段 B：compatibility wrapper

- frontend builder 只保留格式入口和 compatibility glue；
- wrapper 将 GGUF path 交给新的 loader/config adapter/modeling builder；
- frontend 不再实现 `build_layer()`、attention、MLP、MoE 等模型逻辑；
- frontend 只在确有外部 API 兼容需求时保留。

### 阶段 C：删除模型 builder

满足下列条件后删除：

- `qwen3_builder.cpp`、`qwen35_builder.cpp`、`qwen35moe_builder.cpp` 的 graph topology 已迁移；
- frontend CMake 不再编译这些模型 builder；
- 现有 GGUF frontend API 使用方已迁移或有兼容入口；
- Qwen3/Qwen3.5 Dense/MoE 的真实文件验证完成；
- OCL 和 CM GPU path 均通过；
- 文档、测试和错误诊断已更新。

`builder.hpp` 最终应删除模型专用声明，或仅保留明确标注为 deprecated 的兼容函数，避免产生第二套 graph builder。

## 9. CMake 和依赖调整

### 9.1 modeling target

中期继续依赖现有 developer API 路径：

- 有 developer package 时链接 `openvino::runtime::dev`；
- in-tree fallback 时加入 `src/common/transformations/include` 和 `src/core/dev_api`；
- modeling target 需要能 include `ov_ops/fully_connected_compressed.hpp`、`ov_ops/gather_compressed.hpp` 和 `ov_ops/moe_compressed.hpp`；
- 不加入 `plugins/intel_gpu/include`。

长期若 compressed ops 需要稳定公共 API，可另行将其提升到合适的 dev API；这不是本次 builder 迁移的前置条件。

### 9.2 GGUF loader target

- loader 依赖 `gguflib` 或新的 GenAI GGUF loader library；
- loader 不应依赖 frontend builder target；
- 若 `gguflib` 只提供 parser，GenAI 侧实现 config adapter/weight source；
- Pipeline 和 GenAI 的链接方式必须保持单一 owner，避免同一 reader 实现被重复编译。

### 9.3 frontend target

迁移完成后：

- 从 frontend CMake source list 移除模型 builder cpp；
- 保留必要的 frontend registration、format detection 和 compatibility wrapper；
- 删除不再使用的 builder header/include path；
- 确认打包安装不会继续暴露 frontend 私有 builder API。

## 10. 测试计划

### 10.1 ops 单元测试

为 `FullyConnectedCompressed` 和 `GatherCompressed` 增加 node-level 测试：

- 5-input FC 的输入顺序、动态 scale/zp placeholder；
- GatherCompressed 的 4-input 和 5-input clone；
- dummy f16 scale 导致 f16 output；
- GGUF quantized element types 不被转换；
- 动态 shape 和 indices shape 推导；
- context 不一致时给出明确错误；
- common op 与 GPU private op 不混用。

### 10.2 graph structural parity

对每种架构生成 frontend graph 和 modeling graph，比较：

- op type 计数；
- parameter/result 数量和 names；
- compressed FC/Gather/MoE 数量；
- embedding 是否只有一个 `GatherCompressed`，且不存在大尺寸 f16 embedding buffer；
- 每层 attention/MLP/MoE/hybrid block 的结构；
- input/output shape 和 dtype。

### 10.3 真实 GGUF 覆盖

至少覆盖：

- Q4_0；
- Q4_1；
- Q8_0；
- Q4_K；
- Q5_K；
- Q6_K；
- Qwen3 dense；
- Qwen3.5 dense；
- Qwen3.5 MoE。

### 10.4 数值和端到端测试

每个模型至少验证：

1. 固定 prompt 的单步 logits，比较误差和 top-k；
2. 固定 seed 的多步 token sequence；
3. KV cache 开启和关闭路径；
4. greedy 与 sampling（若适用）；
5. GPU OCL 和 CM 实现；
6. embedding lookup、final `FullyConnectedCompressed` 和 MoE compressed path；
7. compile time、模型内存和 embedding memory，确认没有意外的完整 f16 materialization。

最终删除 frontend builder 前，应把这些测试设为 CI 或至少作为迁移验收脚本执行。

## 11. 风险与规避

### 11.1 mmap 生命周期

**风险**：reader 局部销毁后，Constant 仍引用 mmap 地址。

**规避**：使用共享 storage owner，并在 `WeightSource`、Constant backing allocation 或 Model-associated holder 中明确延长生命周期；增加 reader 销毁后仍可 compile/infer 的测试。

### 11.2 compressed op ABI 漂移

**风险**：为了 modeling API 方便而改变 input order、placeholder 或 quantized element type，导致 GPU transformation 不匹配。

**规避**：低层 wrapper 只做 node construction；高层 helper 集中处理占位符；加入 node input audit 和 GPU compile test。

### 11.3 Gather 输出精度错误

**风险**：dummy scale 使用错误 dtype，embedding output 变成 f32 或动态类型，进而触发大内存路径或后续 dtype 不匹配。

**规避**：明确规定 GGUF embedding 默认 dummy f16 scale，并检查 output element type；保持 GPU `gather_gguf` 的 f16 约定。

### 11.4 CPU fallback 误用

**风险**：`GatherCompressed::evaluate()` 不支持直接执行，测试在 CPU 上失败后被误判为 graph 错误。

**规避**：结构测试可只做 shape/type/node 检查；数值测试使用支持该 node 的 GPU backend，或明确提供独立 dequantized fallback graph。

### 11.5 GenAI builder registry 不一致

**风险**：新 builder 已存在，但 `ModelRegistry`、GenAI `ModelBuilder` 或 Pipeline `ModelingBackend` 没有统一注册和 route，导致实际仍走旧 frontend。

**规避**：为 architecture name 建立单一注册入口，打印实际 backend/builder；增加 model path 到 builder 的集成测试，禁止静默回退到另一套 graph。

### 11.6 serialization/deserialization

**风险**：新增 common compressed node 后，IR utility 或 `find_llm_matmul()` 不识别 final projection，或者 MOE extension 注册不完整。

**规避**：保留现有 `MOECompressed` extension 注册方式，确认 `FullyConnectedCompressed` 被 final projection 检测逻辑识别，并增加 serialize/deserialize smoke test。

## 12. 推荐实施拆分

### PR/Change 1：compressed modeling ops

- 添加 `fully_connected_compressed()`；
- 添加 `gather_compressed()`；
- 添加 `gguf_linear()`、`gguf_embedding()`；
- 添加 node-level tests；
- 不改 frontend builder。

### PR/Change 2：GGUF loader adapter

- 提取/复用 `GGUFContainerReader`；
- 添加 `GGUFModelConfigAdapter`；
- 添加 `GGUFWeightSource`；
- 明确 shared storage lifetime；
- 加载和 metadata validation tests。

### PR/Change 3：Qwen3 modeling builder

- 将 Qwen3 graph topology 迁入 modeling；
- 使用 compressed ops 和 GGUFWeightSource；
- 保留 frontend 双路径；
- 完成 structural/logits parity。

### PR/Change 4：Qwen3.5 builders

- 先 Dense，再 MoE；
- 覆盖 hybrid attention、Gated DeltaNet、FusedConv、LinearAttention、MOECompressed；
- 完成真实 GGUF 和 GPU OCL/CM 验证。

### PR/Change 5：frontend cleanup

- 将 frontend builder 替换为 compatibility wrapper 或删除；
- 更新 CMake、安装清单和文档；
- 删除重复 graph topology 和过期 include；
- 运行全量 GGUF regression。

## 13. 验收标准

迁移完成的定义：

- modeling 可以在不 include frontend builder 私有头文件的情况下，从 GGUF path 创建 `ov::Model`；
- header/metadata/tensor parsing 由独立 loader/weight source 负责；
- Qwen3/Qwen3.5 Dense/MoE graph topology 只有一份，位于 modeling；
- `FullyConnectedCompressed`、`GatherCompressed`、`MOECompressed` 的 common node ABI 保持稳定；
- embedding 不发生无意的大规模 f16 materialization；
- GPU OCL/CM kernel 仍能识别并执行 compressed graph；
- 单步 logits、多步 token、KV cache 和生成结果通过 parity 验证；
- frontend model-specific builder 源文件已从编译目标移除并可安全删除；
- CMake、测试和文档不再依赖旧 builder。

## 14. 最终建议

本次中期方案应采用“先分层、再迁移、后删除”的策略：

1. 先在 modeling ops 中复用 common `ov_ops`，不要复制 compressed op 实现；
2. 先建立 GGUF loader/config/weight source 边界，再搬 graph topology；
3. 以 Qwen3 为第一条验证链，确认 `GatherCompressed` embedding、`FullyConnectedCompressed` lm_head 和 mmap 生命周期；
4. 再迁移 Qwen3.5 Dense/MoE；
5. 通过双路径 parity 和 GPU regression 后，删除 frontend 中的模型 builder。

这样可以最大限度复用现有 GPU GGUF kernel 和 OpenVINO node ABI，同时让 modeling 成为唯一的模型 graph 构造位置，GGUF frontend 回归为格式入口和兼容层。