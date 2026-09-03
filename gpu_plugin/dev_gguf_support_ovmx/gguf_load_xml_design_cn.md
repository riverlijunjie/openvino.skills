# GGUF 模型图 XML 常规加载与原始权重重绑定设计

**状态**：设计草案；v2 需求修订见第 16 节，覆盖前文的“cache 目录”方案。**v2 已实现并通过 GPU 端到端验证（见第 18 节）。**
**适用仓库**：`openvino.pipeline.mx`、`thirdparty/openvino`、`thirdparty/openvino.genai`
**目标**：首次加载 GGUF 时在同一目录生成同名 OpenVINO 模型图 XML；后续加载时优先使用“XML 图 + 同目录 GGUF 原始权重”，并支持将这两个文件整体复制到任意机器后直接加载。

---

## 1. 背景与现状

当前 GGUF 原生路径已经具备以下能力：

- `thirdparty/openvino/src/frontends/gguf/` 解析 GGUF header、metadata 和 tensor table。
- 量化权重以 `gguf_*` opaque block element type 保存为 `ov::op::v0::Constant`，不在 FrontEnd 中解量化。
- mmap 模式下，Constant 的 buffer 由 `ov::MappedMemory` 持有，权重字节直接来自 GGUF 文件。
- `openvino.genai/src/cpp/src/utils.cpp::read_model()` 已存在 GGUF IR cache：使用 `ov::save_model()` 保存 XML/BIN，并在下一次直接 `Core::read_model(xml)`。

但是，现有 cache 的语义是“缓存完整 IR”，不是“只缓存图、权重仍来自原始 GGUF”：

1. `ov::save_model(gguf_model, cached_xml_path)` 会把 GGUF 权重写入同名 `.bin`。
2. warm load 读取 XML/BIN 时，权重来自 `.bin`，不会重新读取原始 `.gguf`。
3. cache key 当前主要由规范化路径、文件大小和 mtime 组成，不能严格证明 XML 中的权重引用仍与当前 GGUF 内容一致。
4. 生成 XML 时如果发生了 transformation 或 Constant 复制，必须保证每个 weight Constant 的 GGUF tensor 身份仍可追踪。

因此，本需求不是简单修改 `ov::save_model()` 的调用，而是要增加“**图序列化 + 外部 GGUF 权重引用 + 加载后重绑定**”这一套协议。

---

## 2. 目标与非目标

### 2.1 目标

- 首次加载：GGUF → FrontEnd 构图 → 保存图 XML 和权重引用清单。
- 后续加载：XML → 恢复拓扑和非权重常量 → 根据清单从原始 GGUF mmap 对应 tensor → 替换/绑定 GGUF Constant。
- 不把 GGUF 权重复制到 IR `.bin`；`.bin` 最多保存非 GGUF 小常量，推荐本阶段完全不生成或只生成很小的辅助数据。
- 支持同一 GGUF 模型内 Q4_K/Q5_K/Q6_K/Q8_0/F16/F32 等混合张量类型。
- 发现 GGUF 文件被替换、截断、版本不兼容或 tensor 布局变化时，安全地 cache miss 并回退到完整 GGUF 加载。
- 保持现有 GPU `FCGGUFOpt`、GenAI tokenizer rt-info 和模型图语义不变。

### 2.2 非目标

- 不缓存 `compile_model()` 产生的设备程序；OpenVINO compiled model cache 仍由现有机制负责。
- 不修改 GGUF 文件格式。
- 不在 XML cache load 阶段对 GGUF 权重做 host-side dequantize。
- 不把任意外部二进制文件都抽象成通用权重格式；第一阶段只支持 GGUF tensor reference。
- 不允许通过删除/重排 GGUF Constant 来绕过 `gguf_*` element type 和 transformation 防护。

---

## 3. 推荐总体架构

推荐将 cache 分成三个逻辑对象：

```text
<cache_dir>/gguf_ir/<cache_key>/
├── graph.xml              # 只描述 OpenVINO 图和外部权重引用
├── graph.bin              # 可选；不包含 GGUF 权重，第一阶段可为空/不生成
└── weights.manifest.json  # GGUF 文件身份及每个 Constant 的 tensor 映射
```

加载流程：

```text
                         首次加载
 GGUF ──parse/mmap──> GGUF FrontEnd ──build──> ov::Model
                                      │
                                      ├──写入 graph.xml
                                      └──写入 weights.manifest.json

                         后续加载
 GGUF path ──validate manifest──> Core.read_model(graph.xml)
                                      │
                                      └──GGUFWeightRebinder
                                           ├──打开原始 GGUF
                                           ├──按 tensor name/offset 建立 mmap Constant
                                           ├──替换 XML 中的占位 Constant
                                           └──返回可 compile 的 ov::Model
```

关键原则是：**XML 只保存图结构和引用，不拥有 GGUF 权重的最终数据；原始 GGUF 是权重唯一事实来源。**

---

## 4. 外部权重引用协议

### 4.1 Manifest 文件

建议新增版本化 JSON manifest，例如 `weights.manifest.json`：

```json
{
  "schema_version": 1,
  "format": "openvino.gguf.external_weights",
  "graph_xml": "graph.xml",
  "source": {
    "path": "/models/Qwen3-8B-Q5_K_M.gguf",
    "canonical_path": "/models/Qwen3-8B-Q5_K_M.gguf",
    "size": 5033164800,
    "mtime_ns": 1730000000000000000,
    "sha256": "...",
    "gguf_version": 3,
    "data_start": 4096
  },
  "tensors": [
    {
      "constant_id": "stable-constant-id",
      "tensor_name": "blk.0.attn_q.weight",
      "type": "gguf_q5_k",
      "shape": [4096, 4096],
      "data_offset": 12345678,
      "byte_size": 14417920,
      "block_elem_count": 256,
      "block_byte_size": 176,
      "sha256": "..."
    }
  ]
}
```

字段要求：

- `schema_version`：用于未来兼容；未知版本必须拒绝使用。
- `source`：记录原始 GGUF 文件的规范路径、文件大小、mtime、GGUF version、data section 起点和 SHA-256。
- 每个 tensor 同时记录 `tensor_name` 和 `data_offset`。name 用于可读性及重新解析，offset 用于严格校验。
- `type`、`shape`、`byte_size`、block geometry 必须与 XML 中的 Constant 及当前 GGUF tensor table 一致。
- `constant_id` 必须稳定，不应依赖节点遍历顺序。推荐由 `gguf tensor name + type + shape` 生成 SHA-256；同名 tensor 不允许出现歧义。
- `sha256` 可按 tensor 计算。第一阶段可以只计算整个文件 hash，第二阶段再增加按 tensor hash 以减少校验成本。

### 4.2 XML 中的引用信息

仅依赖外部 manifest 也可以工作，但 XML 与 manifest 分离后容易被误配。推荐在 GGUF Constant 的 `<data>` 或 rt-info 中写入最小引用信息：

- `gguf.external_weight = true`
- `gguf.tensor_name`
- `gguf.data_offset`
- `gguf.byte_size`
- `gguf.source_file_hash`
- `gguf.external_weight_schema_version`

XML 中不保存实际 GGUF block 字节。manifest 是批量校验和加载索引，XML rt-info 是单节点自描述信息；两者不一致时必须 cache miss。

### 4.3 Node identity

不能把 friendly name 当作唯一身份，因为图 pass 可能改变 friendly name。建议：

1. FrontEnd 创建 GGUF Constant 时写入不可变 `gguf.tensor_name` rt-info。
2. 序列化前收集每个 GGUF Constant 的 `tensor_name/type/shape`，计算 `constant_id`。
3. rebinder 以 `constant_id` 为主、`tensor_name` 为辅完成绑定。
4. 任何一个 GGUF Constant 缺少身份信息都视为不可缓存，而不是静默按顺序匹配。

---

## 5. XML 序列化方案选择

### 5.1 方案 A：扩展 OpenVINO IR 原生 external-data 语义（推荐长期方案）

扩展 IR serializer/deserializer，使 Constant 的 `<data>` 支持外部引用，例如：

```xml
<data external="gguf"
      manifest="weights.manifest.json"
      tensor="blk.0.attn_q.weight"
      offset="12345678"
      size="14417920"
      hash="..."/>
```

反序列化时，`Core::read_model(graph.xml, properties)` 提供 GGUF source path，IR deserializer 直接创建绑定到 mmap 的 `Constant`。

需要的工作：

- 扩展 IR XML schema、serializer 和 `ir_deserializer`。
- 增加 `external_weights` 读取属性，例如 `ov::gguf_external_weights_path`。
- 定义外部数据 provider 接口，避免 core 直接依赖 GGUF parser。
- 让 GGUF FrontEnd 注册 `GGUFExternalWeightProvider`。
- 明确外部数据文件不存在、hash 不匹配时的错误码和 fallback 语义。

优点是最终没有“占位 Constant + 二次 patch”窗口，`read_model()` 返回时模型已经完整。缺点是改动跨 Core、IR 和 FrontEnd，review 面较大。

### 5.2 方案 B：标准 XML + 加载后 rebinding pass（推荐第一阶段落地）

第一阶段可以不改通用 IR schema：

1. 保存 graph XML 时，为 GGUF Constant 保存引用 rt-info。
2. 序列化时让 GGUF Constant 写入一个可验证的占位 payload，而不是原始权重。
3. `Core.read_model(graph.xml)` 后立即执行 `GGUFWeightRebinder`：
   - 校验 XML/manifest；
   - 从原始 GGUF mmap 创建新的 Constant；
   - 用新 Constant 替换旧占位 Constant 的所有消费者输入；
   - 重新验证模型。
4. 在 rebinding 完成前，禁止 compile、MOC、ConstantFolding 和 GPU transformation pipeline。

注意：占位 payload 不能采用任意形状/字节数。IR deserializer 可能根据 `gguf_*` block geometry 校验 payload 大小，因此需要一个明确的“external placeholder”表示。优先级如下：

- 若 IR serializer 支持跳过外部 Constant，使用跳过/外部引用标记。
- 否则使用与真实 tensor 相同大小的稀疏占位文件会造成空间浪费，不推荐。
- 不要把完整 GGUF 权重写入 `.bin` 后再覆盖，这会失去本需求的收益。

方案 B 的主要风险是从 XML 读出到替换 Constant 之间存在短暂的非法/不完整模型状态，必须把 rebinder 封装在 GenAI 的 `read_model()` 内部，不能暴露给普通调用者。

### 5.3 方案 C：自定义 GGUF-IR FrontEnd

缓存 XML 不直接交给标准 IR FrontEnd，而是由新的 `gguf_ir` FrontEnd 读取 XML + manifest + 原始 GGUF，再构建/绑定模型。

该方案可避免修改通用 IR，但会重复部分 IR 解析和模型 patch 逻辑；适合作为方案 B 无法满足现有 IR 限制时的过渡方案，不建议作为长期公共 API。

### 5.4 决策

推荐分两步实施：

- **短期**：方案 B，复用现有 `ov::save_model/read_model` 和 GenAI GGUF 入口，增加明确的 rebinder 和 manifest。
- **长期**：方案 A，把 external data provider 下沉到 OpenVINO Core/IR，使任何调用方都可以安全加载“XML 图 + GGUF 外部权重”。

---

## 6. 首次保存流程

新增 `GGUFModelCache` 或等价模块，职责不要继续堆在 `utils.cpp` 的路径判断代码中。

### 6.1 入口

`utils::read_model(model_path, properties)` 检测到 GGUF 后：

1. 解析 cache 相关属性和环境变量。
2. 计算 source identity。
3. 查找并验证 XML + manifest。
4. cache hit 时走 XML + rebinding。
5. cache miss 时调用 native GGUF FrontEnd。
6. 对生成的模型执行 cache preparation：
   - 检查所有 GGUF Constant 的 identity；
   - 检查其 shape/type/byte size 与 reader tensor table 一致；
   - 写入 XML、manifest；
   - 原子提交 cache。
7. 首次调用返回 rebinding 后的模型，保证 cold/warm 两条路径模型形态一致。

### 6.2 写 cache 的原子性

使用临时目录：

```text
<key>.tmp.<pid>/graph.xml
<key>.tmp.<pid>/weights.manifest.json
```

完成写入、fsync（至少关闭并检查文件大小）和重新读取校验后，再 rename 为最终目录。禁止先创建最终 XML 再异步写 manifest，否则并发进程可能读到半成品。

建议使用 lock file 或跨进程文件锁，避免多个进程同时生成相同模型 cache。

### 6.3 保存前的图不变量

保存前必须运行检查：

- 所有 `gguf_*` Constant 都有唯一 `constant_id`。
- Constant 的 element type、shape、byte size 和 source tensor 一致。
- 不存在针对 GGUF Constant 的 `Convert`、`Transpose`、`Concat` 或解量化子图。
- `FullyConnectedCompressed` 的 weight 输入仍指向该 Constant。
- tokenizer 和模型 metadata 已写入 `gguf` rt-info。

---

## 7. 后续加载与权重重绑定

### 7.1 cache hit 前校验

至少执行以下校验：

1. `graph.xml`、manifest 均存在且 schema version 支持。
2. 原始 GGUF 文件存在且可读。
3. canonical path 与 manifest 一致；如果允许模型移动，需要由用户显式指定 source path，不能自动信任 manifest 中旧路径。
4. 文件 size、mtime 和 SHA-256 与 manifest 一致。建议默认 hash 校验；超大模型可提供 `strict_hash=false`，但 size + mtime 只能作为性能优化，不应声称内容一致。
5. GGUF version、tensor 数量、data section 起点一致。
6. 每个 manifest tensor 在当前 GGUF tensor table 中存在，name/type/shape/offset/size 全部一致。
7. XML 中 GGUF Constant 的 identity 集合与 manifest 完全相同。

任意检查失败都应视为 cache miss，删除/隔离损坏 cache 后回退到原始 GGUF FrontEnd；不得使用可能错配的权重继续推理。

### 7.2 rebinding 算法

伪代码：

```text
reader = GGUFReader(source_path, mmap=true)
model = Core.read_model(graph.xml)

for constant in model constants where constant.type.is_gguf_block():
    ref = read_external_ref(constant)
    tensor = reader.find_tensor(ref.tensor_name)
    validate(ref, tensor)
    replacement = reader.tensor_constant(ref.tensor_name)
    copy stable names / rt-info from constant to replacement
    replace_all_consumers(constant.output(0), replacement.output(0))

validate_model(model)
assert no GGUF placeholder remains
return model
```

实现时应注意：

- 不要修改原始 Constant 的 buffer 指针；使用新 Constant 替换节点，避免破坏 immutable tensor 语义。
- replacement 必须持有 `std::shared_ptr<ov::MappedMemory>`，确保模型生命周期覆盖 compile 和 infer。
- 用 `ov::Model::replace_node` 或等效安全 API，保留 output names、friendly name、rt-info 和 control dependencies。
- rebinding 完成后再次检查每个 FC weight 输入是否是正确的 `gguf_*` Constant。
- 如果图中存在 embedding 等需要 dense 权重的路径，必须按现有设计区分：只有 GPU 可消费的 FC GGUF weight 走外部 block rebinding；需要 host dequant 的路径仍按既有契约处理，并在 manifest 中明确标记。

---

## 8. Cache key 与版本管理

当前 path + size + mtime 的 key 可以保留作为目录定位，但不能作为唯一正确性保证。

推荐 key 输入：

```text
cache_schema_version
openvino_graph_schema_version
gguf_frontend_version
gguf_source_sha256
gguf_version
architecture
model builder version
```

推荐：

```text
key = SHA256(canonical_path + source_sha256 + cache_schema_version + builder_version)
```

如果使用 source hash 计算成本过高，可先用 path/size/mtime 查找候选，再在 cache hit 校验阶段计算完整 SHA-256；hash 通过后才是真正 hit。

必须在 manifest 中记录：

- GGUF FrontEnd/build commit 或 ABI version；
- OpenVINO graph schema version；
- GGUF element type table version；
- builder architecture/version；
- 是否启用某些图优化开关。

任何影响图拓扑、Constant identity 或 element type 的改动都应使 cache 失效。

---

## 9. API、属性与环境变量

### 9.1 GenAI 内部属性

建议增加内部配置（名字可在实现阶段与现有属性风格对齐）：

- `ov::genai::gguf_ir_cache_dir`：cache 根目录；可复用现有 `ov::cache_dir`。
- `ov::genai::enable_gguf_ir_cache`：默认 `true`。
- `ov::genai::gguf_ir_cache_strict_hash`：默认 `true`。
- `ov::genai::gguf_ir_source_path`：加载 XML 时指定原始 GGUF 路径。
- `ov::genai::gguf_ir_cache_rebuild_on_mismatch`：默认 `true`。

现有环境变量可继续兼容：

- `OPENVINO_GENAI_GGUF_IR_CACHE=0`：禁用 cache。
- `OPENVINO_GENAI_GGUF_CACHE_RELOAD_AFTER_SAVE=0`：仅作为过渡兼容项；新实现中应改为“保存后执行 XML + rebinding 验证”。

### 9.2 Core 长期 API

长期建议提供通用外部权重 provider，而不是把 GGUF 路径硬编码到 `Core::read_model`：

```cpp
class ExternalWeightProvider {
public:
    virtual ov::Tensor load(const ExternalWeightRef& ref) = 0;
    virtual ~ExternalWeightProvider() = default;
};

core.read_model(xml_path,
                {ov::external_weight_provider(provider),
                 ov::external_weight_source(source_path)});
```

GGUF provider 负责 mmap 和 tensor table 校验，IR deserializer 只负责调用 provider。这样未来可扩展 safetensors、分片权重和其他外部格式。

---

## 10. 错误处理与安全要求

### 10.1 必须回退的情况

- manifest 缺失或 schema 不支持；
- XML 与 manifest 的 Constant 集合不一致；
- 原始 GGUF 不存在、路径未提供或不可读；
- 文件 hash/size/mtime 不一致；
- GGUF header、tensor type、shape、offset、byte size 不一致；
- OpenVINO element type table 或 builder version 不兼容；
- rebinding 后仍有 placeholder；
- XML 读取或模型验证失败。

### 10.2 禁止行为

- 不要只因为 XML 存在就认为 cache hit。
- 不要按 Constant 遍历顺序把权重数组顺序绑定到 GGUF tensor 数组。
- 不要 hash 失败后静默使用旧 `.bin`。
- 不要在校验失败时把 GGUF block 转成 F16 作为兜底。
- 不要让半初始化模型进入 `compile_model`。

### 10.3 并发和权限

- cache 文件以临时目录 + 原子 rename 发布。
- manifest、XML 和目录权限遵循现有 cache 策略，不应暴露模型路径之外的敏感信息。
- 错误日志可以打印 cache key 和失败原因，但避免默认打印完整的本地用户路径到远程 telemetry。

---

## 11. 代码工作项

### Phase 0：确认当前实现边界

- [ ] 梳理 `utils.cpp::read_model()` 当前 GGUF cache 分支。
- [ ] 确认 `ov::save_model()` 对 `gguf_*` Constant 的 XML/BIN 序列化行为。
- [ ] 确认 IR deserializer 对 external/placeholder Constant 的可扩展点。
- [ ] 确认所有 GGUF Constant 都有稳定 tensor name rt-info。

### Phase 1：manifest 与 rebinder（短期可交付）

- [ ] 新增 `gguf_model_cache.{hpp,cpp}`，从 `utils.cpp` 拆出 cache 逻辑。
- [ ] 新增 manifest schema、读写、版本校验和原子提交。
- [ ] 增加 source identity：canonical path、size、mtime、SHA-256、GGUF version。
- [ ] 为每个 GGUF Constant 写入/读取 external reference。
- [ ] 实现 `GGUFWeightRebinder`，使用 `GGUFReader::tensor_constant()` 重新 mmap 原始 tensor。
- [ ] 增加替换节点、保留 names/rt-info、重新验证模型的代码。
- [ ] 将当前“读取 XML/BIN 即返回”改为“读取 XML → 校验 → rebinding → 返回”。
- [ ] 禁止把原始 GGUF 权重写入 cache `.bin`；若现有 serializer 无法跳过，先实现临时的 external placeholder 机制。

### Phase 2：OpenVINO IR external data 支持（长期方案）

- [ ] 扩展 XML schema，定义 `external="gguf"` 和 reference 字段。
- [ ] 扩展 serializer，使 GGUF Constant 只写 reference，不写 block bytes。
- [ ] 扩展 deserializer/provider API，使 `Core::read_model` 直接绑定 mmap Constant。
- [ ] 提供 `external_weight_source` 和 provider 属性。
- [ ] 将 GenAI 的 rebinder 迁移为 Core/IR 通用实现。
- [ ] 增加向后兼容：旧版 XML/BIN 仍可正常读取；旧 cache 自动失效并重建。

### Phase 3：GenAI 与性能验证

- [ ] native FE、cache cold、cache warm 三条路径的 tokenizer/model metadata 一致性测试。
- [ ] 测量首次 GGUF parse/build、XML save、warm XML parse、rebinding、compile 各阶段耗时。
- [ ] 确认 warm path 的 host RSS 不因 `.bin` 保存完整权重而增加。
- [ ] 确认 GPU decode/prefill 仍选中 `FCGGUFOpt`，没有误选 OneDNN 普通 FC。
- [ ] 验证模型文件被替换后不会使用旧 cache。
- [ ] 验证 Q4_K_M、Q5_K_M、Q6_K、Q8_0 以及 mixed-format 模型。

---

## 12. 测试设计

### 12.1 单元测试

1. manifest 序列化/反序列化、schema version 和字段缺失。
2. source hash、size、mtime mismatch 都导致 cache miss。
3. tensor name/type/shape/offset/byte size mismatch 都导致 cache miss。
4. replacement Constant 的 `element_type`、shape、byte bytes 与原始 GGUF reader 完全一致。
5. replacement 后所有 consumer、output names、friendly names 和 rt-info 保留。
6. mixed-format 模型每种 GGUF type 都能正确绑定。
7. 空文件、截断 GGUF、重复 tensor name、越界 offset 均安全失败。

### 12.2 IR round-trip 测试

- GGUF 首次加载并写 cache。
- 关闭原始 GGUF mmap 后重新打开 source。
- XML + manifest + original GGUF warm load。
- 对所有 GGUF Constant 比较：tensor name、type、shape、byte size、SHA-256。
- 比较 cold/warm 图的 input/output names、节点数量、关键 op 类型和 rt-info。

### 12.3 端到端测试

固定 prompt 和 greedy 配置，对以下路径比较 token id 序列：

```text
native GGUF cold load
XML + original GGUF warm load
```

至少覆盖：

- Qwen3 Q4_0/Q4_K_M/Q5_K_M/Q6_K/Q8_0；
- mixed Q5_K + Q6_K + F32；
- M=1 decode；
- 长 prompt prefill（验证 transcode + OneDNN 路径）；
- GPU compile 和 inference；
- CPU/NPU 不支持时仍保持清晰 hard-fail。

### 12.4 破坏性测试

- 修改原始 GGUF 一个字节：必须 cache miss。
- 用同路径替换为同大小、同 mtime 但内容不同的文件：严格 hash 模式必须 miss。
- 删除 manifest 或 XML：必须重新构建。
- 复制 XML 到另一台机器但不提供原始 GGUF：必须失败或按策略回退，不能使用隐藏的 `.bin` 权重。
- 多进程并发首次加载：最终 cache 必须是完整且可验证的一份。

---

## 13. 性能、空间与兼容性评估

### 13.1 性能

warm path 预计主要节省：

- GGUF header/tensor table 解析时间；
- qwen3/qwen35 图 builder 时间；
- graph transformation 前置构建时间。

新增成本：

- manifest 校验；
- 原始 GGUF mmap 和 tensor table 校验；
- Constant replacement 或 external provider 绑定。

因此不能只比较 `Core::read_model(xml)` 时间；必须统计完整的“XML 读取 + 原始 GGUF 权重绑定”时间。

### 13.2 空间

目标是 cache 不再保存第二份完整 GGUF 权重。若 `.bin` 仍包含与 GGUF 等大的数据，则需求没有实现。CI 应检查：

```text
size(graph.bin) << size(source.gguf)
```

或者第一阶段直接不生成 `graph.bin`。

### 13.3 兼容性

- 旧版 cache：检测不到 manifest/external schema 时按旧逻辑读取一次，随后建议删除并重建；若旧逻辑会加载完整 `.bin`，应通过版本开关隔离，避免误认为新 cache。
- 非 GGUF 模型：完全走现有路径，不改变 safetensors/IR 行为。
- 原始 GGUF 路径变化：默认要求显式 source path；允许 relocation 时必须重新计算 hash 并更新 manifest。

---

## 14. 推荐实施顺序与验收标准

推荐依赖顺序：

```text
稳定 Constant identity/rt-info
        ↓
manifest + source validation
        ↓
rebinder + cold/warm parity
        ↓
禁止 GGUF 权重写入 BIN
        ↓
IR external-data provider
        ↓
Core 通用 API
```

第一阶段验收标准：

1. 首次加载后生成 `graph.xml` 和 `weights.manifest.json`。
2. cache 目录不包含完整 GGUF 权重副本。
3. warm load 必须打开原始 `.gguf` 并完成 mmap rebinding。
4. warm model 中每个 GGUF Constant 的字节与原始 GGUF 对应 tensor 完全一致。
5. cold/warm 在 GPU 上输出 token id 序列一致。
6. 修改原始 GGUF 内容后严格 cache 校验失败并自动重建。
7. 不支持的格式/架构/损坏 cache 均不静默使用错误权重。

长期验收标准：

- `ov::Core::read_model(graph.xml, external_weight_source=gguf)` 直接返回完整模型；
- 不需要 GenAI 专用 rebinding 代码；
- IR serializer/deserializer 对 external GGUF weight reference 有稳定、版本化、可测试的协议。

---

## 15. 结论

当前实现已经有“GGUF → XML/BIN → XML/BIN”的缓存雏形，但它缓存的是完整权重，不满足“图保存到 XML、下次从原始 GGUF 读取权重”的要求。

最小可行改造是：新增 manifest、给 GGUF Constant 增加稳定 tensor identity、加载 XML 后通过 `GGUFWeightRebinder` 重新 mmap 原始 GGUF 并替换 Constant，同时严格校验源文件 hash 和 tensor layout。长期应把这一能力下沉到 OpenVINO IR 的 external-data provider，形成通用的“模型图与外部权重分离”机制。

> **v2 修订说明**：第 16 节是当前需求的正式设计，优先级高于本节以及前文所有“cache_dir/cache_key/cache hit/cache miss”表述。实现不应再把该功能设计成依赖 `ov::cache_dir` 的缓存功能。

---

## 16. v2：同目录 XML + GGUF 的常规加载方案

### 16.1 需求定义

该功能是模型格式和加载协议，不是可选的模型 cache：

1. 用户传入 `model.gguf`：
        - 若同一目录存在 `model.xml`，按 `model.xml + model.gguf` 加载；
        - 若不存在 `model.xml`，按原有 GGUF FrontEnd 加载，然后生成 `model.xml`。
2. 用户传入 `model.xml`：
        - 必须在同一目录查找对应的 `model.gguf`；
        - 找到后按 `model.xml + model.gguf` 加载；
        - 找不到时给出明确错误，不得尝试从 XML 的绝对路径或隐藏 `.bin` 恢复权重。
3. `model.xml` 和 `model.gguf` 是一个可搬运模型包。将二者复制到任意机器、任意目录后，只要运行时支持该 GGUF 类型和图版本，即可直接加载。
4. XML 与 GGUF 默认使用同一个 basename：

        ```text
        /path/to/model.gguf
        /path/to/model.xml
        ```

        不要求用户维护 cache key、cache 目录或额外 manifest 文件。

### 16.2 文件发现规则

定义统一的 `resolve_gguf_model_pair(input_path)`：

| 输入 | 查找规则 | 结果 |
|---|---|---|
| `foo.gguf` | 同目录 `foo.xml` | XML 存在则 XML+GGUF；否则原始 GGUF 加载并生成 `foo.xml` |
| `foo.xml` | 同目录 `foo.gguf` | 存在则 XML+GGUF；否则报错 |
| 其他扩展名 | 不触发 GGUF 配对 | 保持现有模型加载逻辑 |

规则细节：

- 扩展名匹配应大小写不敏感，但生成文件统一使用小写 `.xml`。
- XML 和 GGUF 必须是普通文件，不能是目录；符号链接是否允许沿用现有文件读取策略。
- 不使用 XML 内保存的绝对路径定位 GGUF。绝对路径只允许作为诊断信息，不能作为加载依赖。
- 如果目录中存在多个候选文件，不按模糊匹配或遍历顺序选择，直接报错。
- `foo.xml` 可以被复制并重命名，但只有同时将对应的 `foo.gguf` 改成同 basename 后才算有效模型包。

### 16.3 XML 文件应保存什么

XML 保存图结构以及每个 GGUF Constant 的**相对权重引用描述**，不保存 GGUF block 的实际内容：

- `gguf.external_weight = true`；
- `gguf.tensor_name`；
- `gguf.tensor_type`；
- `gguf.tensor_shape`；
- `gguf.data_offset`：相对于 GGUF data section 或文件起点，必须统一定义；
- `gguf.byte_size`；
- `gguf.block_elem_count` / `gguf.block_byte_size`；
- `gguf.source_basename`：例如 `foo.gguf`，仅用于诊断和一致性检查；
- `gguf.source_sha256`：用于验证复制后的 GGUF 未被替换；
- `gguf.external_weight_schema_version`；
- `gguf.frontend_version` / `gguf.builder_version`。

**不保存**：原始 GGUF 的绝对路径、cache key、cache_dir 路径、机器相关设备信息、用户名或 host name。

### 16.3.1 对齐真实 IR XML schema

生成的 graph XML 必须复用标准 OpenVINO IR schema（`<net>`/`<layers>`/`<layer>`/`<edges>`/`<rt_info>`），与真实导出的 IR 完全同构，例如参考文件 `openvino_model.xml`（Qwen3-8B）。这样才能被 `Core::read_model()` 反序列化，并保持工具链兼容。

真实 IR 中普通权重 Const 的写法是（offset/size 指向同名 `.bin`）：

```xml
<layer id="18" name="self.model.embed_tokens.weight" type="Const" version="opset1">
    <data element_type="u8" shape="151936, 4096" offset="16412" size="622329856" />
    <output>
        <port id="0" precision="U8">
            <dim>151936</dim>
            <dim>4096</dim>
        </port>
    </output>
</layer>
```

本方案对 GGUF opaque 权重的唯一改动，是在**同一个 `<data>` 元素**上把数据源从 `.bin` 重定向到同目录 GGUF，其余字段（`element_type`/`shape`/`offset`/`size`）保持 IR 原生语义：

```xml
<layer id="..." name="blk.0.attn_q.weight" type="Const" version="opset1">
    <data element_type="gguf_q4_k" shape="4096, 4096"
          source="foo.gguf"
          tensor="blk.0.attn_q.weight"
          offset="12345678" size="14417920"
          hash="..." />
    <output>
        <port id="0" precision="GGUF_Q4_K">
            <dim>4096</dim>
            <dim>4096</dim>
        </port>
    </output>
</layer>
```

约定：

- `element_type` / `shape` / `offset` / `size` 语义与 IR 原生 `<data>` 一致，只是 `offset` 相对 GGUF data section 起点（对应 reader 的 `m_data_start + info->data_offset`），而非 `.bin`。
- 新增属性 `source`（GGUF 文件名，按 XML 所在目录解析）、`tensor`（GGUF tensor name，作为主绑定锚点）、`hash`（源文件 SHA-256）。
- **没有 `source` 属性的 `<data>` 保持原生 IR 行为**（offset/size 指向 `.bin`），因此 embedding/lm_head 等 host 解量化产物、非 GGUF 小常量仍可正常写入 `.bin`，与第 16.5 节的“大权重不落 .bin”不冲突。
- `<rt_info>` 顶层沿用真实 IR 的 `runtime_options`（如 `ACTIVATIONS_SCALE_FACTOR`），这些 GGUF builder 已通过 `set_rt_info` 写入，序列化后与参考 IR 同构。
- 反序列化器遇到 `source="..."` 时，不从 `.bin` 读，而是走 GGUF external weight provider（方案 A）或加载后由 rebinder 替换占位 Const（方案 B）。

### 16.3.2 rt-info 引用字段

除 `<data>` 属性外，可在该 Const 的 `<rt_info>` 冗余记录以下自描述信息，便于校验：

- `gguf.external_weight = true`；
- `gguf.tensor_name`；
- `gguf.tensor_type`；
- `gguf.tensor_shape`；
- `gguf.data_offset`：相对 GGUF data section 起点（与 `<data>` 的 `offset` 一致）；
- `gguf.byte_size`；
- `gguf.block_elem_count` / `gguf.block_byte_size`；
- `gguf.source_basename`：例如 `foo.gguf`，用于诊断和一致性检查；
- `gguf.source_sha256`：验证复制后的 GGUF 未被替换；
- `gguf.external_weight_schema_version`；
- `gguf.frontend_version` / `gguf.builder_version`。

`source="foo.gguf"` 必须按 XML 所在目录解析。它不是绝对路径，也不允许使用 `../` 跨出模型包目录；若未来需要显式外部路径，应通过加载 API 参数提供，而不是写入 XML。

### 16.4 加载入口和状态机

GenAI 当前 `utils::read_model()` 需要从“cache lookup”改为“model pair resolution”：

```text
read_model(input)
  ├─ input = *.gguf
  │    ├─ sibling *.xml exists → load_graph_xml_and_bind_sibling_gguf()
  │    └─ sibling *.xml absent → load_native_gguf_and_save_sibling_xml()
  ├─ input = *.xml
  │    └─ sibling *.gguf exists → load_graph_xml_and_bind_sibling_gguf()
  └─ other input → existing non-GGUF path
```

推荐新增两个内部函数：

- `load_gguf_graph_with_weights(xml_path, gguf_path, properties)`；
- `load_gguf_direct_and_write_xml(gguf_path, properties)`。

首次加载流程：

1. 使用 native GGUF FrontEnd 解析和 mmap `foo.gguf`；
2. 构建 `ov::Model`；
3. 写入每个 GGUF Constant 的外部引用 rt-info；
4. 以临时文件写 `foo.xml.tmp.<pid>`；
5. 关闭并校验 XML 后原子 rename 为 `foo.xml`；
6. 返回原始模型，或重新走一次 XML+GGUF 加载后返回，二者必须通过 cold/warm parity 测试。

第二次以及显式 XML 加载流程：

1. 由 XML 所在目录推导 sibling GGUF 路径；
2. 读取 XML 图；
3. 打开 sibling GGUF 并校验 header、source hash、tensor table；
4. 为每个引用创建 mmap-backed Constant；
5. 替换 XML 中的占位 Constant；
6. 完成图验证后才能进入 MOC、`compile_model` 和 GPU transformation pipeline。

### 16.5 不再使用 `.bin` 保存 GGUF 权重

常规 XML+GGUF 模型包必须只有两个必需文件：

```text
foo.xml
foo.gguf
```

因此：

- `ov::save_model()` 的默认行为不能直接用于 GGUF 图保存，因为它会生成包含权重的 `foo.bin`；
- 应新增 GGUF external-data serializer，或新增专用 `save_gguf_graph(xml_path, gguf_path, model)`；
- 若 IR serializer 仍要求 `.bin`，`.bin` 必须为空/仅包含非 GGUF 小常量，并且加载协议不得依赖它；
- 生成 XML 后应检查旁路文件大小，禁止产生与 `.gguf` 同量级的权重副本；
- 旧版 `cache_dir/gguf_ir/<key>/openvino_model.xml(.bin)` 只作为兼容迁移输入，不再生成新的 cache 目录。

### 16.6 跨机器搬运要求

模型包从机器 A 复制到机器 B 后，加载行为必须与本地生成时一致：

```text
机器 A: /a/models/foo.xml + /a/models/foo.gguf
复制后
机器 B: /b/models/foo.xml + /b/models/foo.gguf
```

实现约束：

- XML 中不得写入 `/a/models/foo.gguf` 这样的绝对路径；
- source reference 只保留 basename 或包内相对路径；
- B 机器加载时使用 XML 所在目录 `/b/models/` 解析 `foo.gguf`；
- 使用 GGUF 文件 SHA-256 检查复制完整性；
- hash 不匹配时明确报错并要求重新复制/重新生成 XML，不得使用旧 `.bin` 兜底；
- XML/ GGUF 同时重命名时，要求 basename 仍一致，或者通过显式 `source_path` 属性提供新的 GGUF 文件名；默认不支持隐式猜测。

### 16.7 绑定协议的推荐简化

由于不再使用 cache manifest，XML 必须自包含每个权重的绑定信息。建议废弃“manifest 必须存在”的要求，将 manifest 降级为可选诊断导出：

- **必需**：`foo.xml`、`foo.gguf`；XML 内含 source 和 tensor references；
- **可选**：`foo.gguf.manifest.json`，用于调试、离线校验和工具展示；加载不能依赖它；
- XML 与 GGUF tensor table 的 name/type/shape/offset/size 不一致时直接失败；
- 同一目录只允许一个同 basename 的 `.gguf` 作为默认权重源。

### 16.8 代码工作项修订

#### OpenVINO Core / IR

- [ ] 增加 GGUF external-data XML schema 和 serializer/deserializer；
- [ ] 支持相对 `source="foo.gguf"`，禁止把绝对路径写入可搬运 XML；
- [ ] 提供 `GGUFExternalWeightProvider` 或等价 provider；
- [ ] 确保 `read_model(foo.xml)` 在 rebinding 后才向调用方返回完整模型；
- [ ] 保留旧 XML/BIN 的读取兼容，但不再为 GGUF 生成新的 `.bin` 权重副本。

#### OpenVINO GenAI

- [ ] 删除 `ov::cache_dir` 作为 GGUF 图 XML 发现条件；
- [ ] 实现 `foo.gguf ↔ foo.xml` sibling resolution；
- [ ] `foo.gguf` 无 XML 时生成同目录 `foo.xml`；
- [ ] `foo.xml` 无 sibling `foo.gguf` 时给出可操作错误；
- [ ] 将现有 `gguf_cached_xml_path()`、`gguf_cached_ir_exists()` 和 cache key 路径逻辑改为同目录逻辑；
- [ ] 保存 XML 前写入 source basename 和 tensor reference；
- [ ] warm load 使用 sibling GGUF mmap rebinding，不读取同名 `.bin`；
- [ ] 增加 XML+GGUF 跨目录、跨机器复制测试。

#### 测试

- [ ] 仅有 `foo.gguf`：成功加载并生成 `foo.xml`；
- [ ] 已有 `foo.xml + foo.gguf`：不重新构图，直接 XML+GGUF 加载；
- [ ] 仅有 `foo.xml`：失败并提示缺少 `foo.gguf`；
- [ ] 复制到新目录：成功加载且 token 结果一致；
- [ ] 同目录放置不同 basename 的 GGUF：不能错误匹配；
- [ ] 修改 GGUF 一个字节：hash 校验失败；
- [ ] 确认未生成完整权重 `.bin`；
- [ ] Q4_K_M/Q5_K_M/Q6_K/Q8_0 和 mixed-format 模型 cold/warm parity。

### 16.9 最终验收标准

1. 输入 `foo.gguf`，首次运行后同目录生成 `foo.xml`，不要求 `cache_dir` 或 manifest。
2. 再次输入 `foo.gguf`，检测到 `foo.xml` 后只恢复 XML 图并从同目录 `foo.gguf` mmap 权重。
3. 输入 `foo.xml`，能够自动发现同目录 `foo.gguf` 并完成加载。
4. 将 `foo.xml + foo.gguf` 复制到任意目录/机器后，无原机器绝对路径依赖即可加载。
5. XML 与 GGUF 不匹配、文件损坏或缺失时，不会加载错误权重，也不会偷偷使用旧 `.bin`。
6. cold load 与 XML+GGUF load 的图结构、GGUF Constant 字节和生成 token 完全一致。

### 16.10 当前方案结论

本需求的正确抽象是“**可搬运的 GGUF 模型包**”，而不是“按 cache key 保存的模型缓存”。模型包的最小组成是同目录、同 basename 的 `foo.xml + foo.gguf`。XML 保存图和相对外部权重引用，GGUF 保存唯一的原始权重数据；加载器根据输入扩展名和 sibling 文件自动选择首次构图或 XML+GGUF 重绑定路径。这样既能复用现有 GGUF FrontEnd 和 GPU kernel，也能保证跨机器复制后不依赖旧机器路径、cache 目录或 `.bin` 权重副本。

---

## 17. tokenizer / detokenizer 的处理方式

### 17.1 结论先行

tokenizer/detokenizer **不应复用 GGUF 权重**，而应作为可搬运模型包的一部分，生成**独立且自包含**的 `*_tokenizer.xml(.bin)` 与 `*_detokenizer.xml(.bin)`。它们与第 16 节“模型图 XML + GGUF 外部权重”是两种不同机制，不要套用同一套 rebinding 协议。

### 17.2 为什么不能像模型权重那样复用 GGUF

核对当前实现（`gguf_utils/gguf_tokenizer.cpp`、`tokenizer/tokenizer_impl.cpp`）后确认：

- tokenizer 数据来源是 **GGUF metadata KV 头**里的 `tokenizer.ggml.*`（vocab tokens、merges、scores、token_type、bos/eos、chat_template），native FE 已把它们写入模型 rt-info 的 `gguf.tokenizer.ggml.*`。
- 这些数据**不在 GGUF 的 tensor data section**，因此没有可 mmap 的连续权重字节区，无法像 FC 权重那样做 offset/byte_size 外部引用。
- `create_tokenizer_from_config()` 通过 openvino_tokenizers 把这些字符串**编译并打包成 string 常量**写入 tokenizer 的 `.bin`。这是一个非平凡变换（BPE 合并表打包、added/special token 处理），不是对 GGUF 字节的拷贝，无法用“相对 offset 引用 GGUF”表达。
- 唯一“复用 GGUF”的做法是加载时重新运行 `create_tokenizer_from_config()`，而这恰恰是我们希望通过 XML 复用跳过的慢路径。

因此对 tokenizer/detokenizer，正确取舍是：**生成一次、以自包含 XML+BIN 形式复用**，而不是引用 GGUF。

### 17.3 为什么自包含 XML+BIN 是合理的

- 体积小：vocab + merges 打包后通常只有几 MB~几十 MB，相对模型权重（GB 级）可忽略，外部引用带来的空间收益极小，却要引入复杂映射协议。
- 天然可搬运：tokenizer/detokenizer 的 XML/BIN 是标准 IR、无绝对路径，**加载时不需要 GGUF 在场**，复制到任意目录/机器即可直接 `read_model`，比模型图 rebinding 简单得多。
- 现状已具备雏形：当前实现已导出 `openvino_tokenizer.xml(.bin)` + `openvino_detokenizer.xml(.bin)`，并在 xml 与 bin 均存在时 reuse。

### 17.4 可搬运模型包的完整组成

```text
foo.xml                  # 模型图 + GGUF 外部权重引用（第 16 节）
foo.gguf                 # 模型权重唯一来源
foo_tokenizer.xml
foo_tokenizer.bin        # 自包含，小
foo_detokenizer.xml
foo_detokenizer.bin      # 自包含，小
```

说明：

- tokenizer/detokenizer 的 `.bin` 是**必需且自包含**的，不适用第 16.5 节“禁止生成 .bin”的约束；该约束只针对模型图的 GGUF 大权重。
- tokenizer/detokenizer 加载**不依赖 `foo.gguf`**；即使只拷贝这 4 个 tokenizer 文件也能独立工作。

### 17.5 命名与发现

- 现状硬编码 `openvino_tokenizer.xml` / `openvino_detokenizer.xml`。当同一目录存在多个 GGUF 模型时会相互覆盖。
- 可搬运包应改为 **basename 前缀**：`foo_tokenizer.xml` / `foo_detokenizer.xml`，与 `foo.gguf`/`foo.xml` 对齐；或将整包放入独立子目录。
- 发现规则并入第 16.2 的 `resolve_gguf_model_pair()`：解析出 `foo` 基名后，同时推导 tokenizer/detokenizer sibling 路径。
- 若 sibling tokenizer XML/BIN 存在则直接 reuse；否则由 GGUF metadata 构建并写出。

### 17.6 生成时机与开关解耦

- 当前 tokenizer 导出依赖 `cache_dir` 或 `enable_save_ov_model`。可搬运方案应与首次 GGUF 加载对齐：**首次加载即在同目录生成 tokenizer/detokenizer XML+BIN**，不再以 cache 开关为前提。
- 目录只读时的回退策略与第 16 节（P5）一致：回退到 `cache_dir` 或临时目录，保持可加载。
- 写出使用临时文件 + 原子 rename，避免并发或半成品。

### 17.7 一致性绑定

- tokenizer/detokenizer 加载虽不需要 GGUF，但应把与模型图相同的 `source_file_hash`（reader 已计算，写在 `gguf.source_file_hash`）写入其 rt-info。
- 当 `foo.gguf` 被替换导致 hash 变化时，模型图和 tokenizer/detokenizer **作为一个包一起失效并重建**，避免图与 vocab 版本错配。

### 17.8 运行时依赖说明

- 无论是否走 GGUF，tokenizer/detokenizer 加载都依赖目标机器上的 `openvino_tokenizers` 扩展库（`load_shared_object`）。
- “复制到任意机器即可使用”的前提是目标机器安装了匹配版本的 openvino_tokenizers；文档与部署说明需明确该依赖。

### 17.9 工作项与验收补充

- [ ] tokenizer/detokenizer 采用 basename 前缀命名，避免同目录多模型冲突。
- [ ] 首次 GGUF 加载即生成 tokenizer/detokenizer XML+BIN，不依赖 cache 开关。
- [ ] sibling 存在时 reuse，缺失时由 GGUF metadata 构建。
- [ ] 在 tokenizer/detokenizer rt-info 写入 `source_file_hash`，与模型图共享失效条件。
- [ ] 目录只读时回退到 cache_dir/临时目录。
- [ ] 验收：仅拷贝 tokenizer/detokenizer 的 XML+BIN 到新机器可独立加载；替换 GGUF 后整包失效重建；reuse 路径与从 GGUF 构建的 encode/decode 结果一致。

---

## 18. v2 实现记录（已落地并验证）

### 18.1 实际改动的文件

OpenVINO Core / 前端（`thirdparty/openvino`，分支 `river/gguf_moe_support_rebase_26.3`）：

- `src/frontends/gguf/src/gguf_reader.{hpp,cpp}`：记录源文件 basename；`make_constant` 给每个 mmap 权重 Const 注入 rt-info `gguf_ext_source`/`gguf_ext_offset`（相对 GGUF 文件起点的绝对偏移）/`gguf_ext_size`。
- `src/core/dev_api/openvino/xml_util/xml_serialize_util.hpp` + `src/core/src/xml_util/xml_serialize_util.cpp`：`XmlSerializer` 增加外部权重信息与 `set_gguf_external_weight()`；主循环在写 Const 前从节点 rt-info 读取并设置；`AlignedBuffer` value 分支在有外部信息时发出 `<data ... source offset size>` 且**不写入 .bin**。
- `src/core/xml_util/include/openvino/xml_util/xml_deserialize_util.hpp` + `src/core/xml_util/src/xml_deserialize_util.cpp`：`XmlDeserializer` 增加 `m_weights_path` 与共享 mmap 缓存并经 `make_visitor` 向子 visitor 传播；`set_constant_num_buffer` 在 `<data>` 含 `source` 时按 XML 目录解析 GGUF、mmap 并零拷贝绑定，否则走原 .bin 路径。
- `src/frontends/ir/src/input_model.cpp`：`convert()` 把 `m_weights_path` 传给反序列化器以解析外部 GGUF。

GenAI（`thirdparty/openvino.genai`）：

- `src/cpp/src/utils.cpp::read_model()`：GGUF 输入改为同目录 sibling 发现——`foo.gguf` → `foo.xml`。存在则 `read_model(foo.xml)`（反序列化器自动 mmap 同目录 GGUF）；不存在则 native FE 构图后 `save_model(model, foo.xml, compress=false)`，再 reload 返回。移除旧的 `cache_dir`/`cache_key` 发现逻辑及相关辅助函数。

### 18.2 端到端验证结果（Qwen3-8B-Q4_1，GPU）

- 首次加载生成 `Qwen3-8B-Q4_1.xml`（6.5 MB），含 **398 处** `source="Qwen3-8B-Q4_1.gguf"` 外部引用，形如 `element_type="gguf_q4_1" shape="4096, 4096" source=... offset=... size=`。
- `Qwen3-8B-Q4_1.bin` 仅 **1.24 GB**（host 解量化的 embedding f16 等），GGUF 为 5.2 GB —— 约 4 GB 量化 FC 权重**未复制到 .bin**，从 GGUF mmap。
- warm 加载（sibling XML 存在）与 cold 加载输出逐字一致，first-token ~379 ms、~62 tok/s。
- 可搬运：把 `.xml`+`.bin` 复制、`.gguf` 以 symlink 放入新目录，传 `.gguf` 路径即成功加载并正常生成，无原路径依赖。

### 18.3 与设计的差异 / 待办

- embedding/lm_head 等 host 解量化的 f16 常量仍写入 `.bin`（无对应可 mmap 的 GGUF 字节区），符合第 16.5 / P1 的分类结论；GGUF opaque 大权重零副本。
- 尚未做严格 source hash 校验（仅做偏移+size 边界检查）；tokenizer/detokenizer 同目录同名生成（第 17 节）尚未改造，当前仍由既有逻辑处理。
- 构建：`CMAKE_COMPILE_WARNING_AS_ERROR=OFF`；增量 `make openvino openvino_ir_frontend openvino_gguf_frontend && make install` + GenAI 增量构建即可。

---

## 19. 最终方案：消除 embedding f16 `.bin`

### 19.1 问题的准确边界

对测试文件 `/mnt/river/ovmx/Qwen3_GGUF/Qwen3-8B-Q4_1.gguf`，GGUF tensor
表实际为：

| tensor | GGUF 类型 | GGUF shape | 语义 |
|---|---|---|---|
| `token_embd.weight` | `Q4_1` | `[4096, 151936]` | embedding |
| `output.weight` | `Q6_K` | `[4096, 151936]` | lm_head |

因此，embedding 和 lm_head 是两个独立的 GGUF tensor，不能因为形状相同就
将 `output.weight` 替代 `token_embd.weight`，也不能假定模型使用 tied
weights。

当前约 1.24 GB 的 `.bin` 数据来自 Q4_1 embedding 在 Qwen3 builder 中被
全量解量化为 dense f16：

```text
token_embd.weight (Q4_1，GGUF，388,956,160 bytes ≈ 371 MiB)
  ↓ dequantize_to_f16()
f16[151936, 4096]（151936 × 4096 × 2 = 1,244,659,712 bytes ≈ 1.16 GiB）
  ↓ Gather
embedding output
```

> 字节核对（`Qwen3-8B-Q4_1.gguf`，已用 gguf reader 验证）：
> `token_embd.weight` 有 622,329,856 个元素，Q4_1 每 32 元素 20 字节，
> 即 `622329856 / 32 × 20 = 388,956,160` 字节 ≈ 371 MiB。
> **不要**把元素数 622,329,856 当成字节数或 622 MB —— 二者相差约 1.6 倍，
> 会导致 external reference 读越界。`output.weight`（Q6_K，256 元素 210
> 字节）为 `622329856 / 256 × 210 = 510,504,960` 字节 ≈ 487 MiB。

这不是 GGUF 原始权重的第二种存储格式，而是当前普通 `Gather` 路径产生的
持久化副本。最终目标不是把它改成 Q6_K，而是保留原始 `token_embd.weight`
的 Q4_1 格式并直接从 GGUF 消费它。

### 19.2 设计决策

新增 GPU 专用的 `EmbeddingCompressed`（实现名称可为 `GatherGGUF`）节点，
替代 embedding 分支中的普通 `Gather`：

```text
token_embd.weight(Q4_1，external GGUF)
  ↓
EmbeddingCompressed(input_ids, weight)
  ↓
f16/f32 hidden states
```

lm_head 保持独立路径：

```text
output.weight(Q6_K，external GGUF)
  ↓
FullyConnectedCompressed
  ↓
logits
```

该方案明确保留两个 tensor 的独立性：

```text
embedding: token_embd.weight(Q4_1)
lm_head:   output.weight(Q6_K)
```

不得通过“复用 `output.weight`”来规避 embedding 实现，否则会改变模型语义。

### 19.3 图和算子契约

`EmbeddingCompressed` 必须具有以下语义：

```text
output[i, :] = dequantize_q4_1(weight[input_ids[i], :])
```

其中：

- `input_ids` 支持当前 GenAI 的 batch、sequence、beam 和动态长度；
- `weight` 是带 `gguf_*` opaque element type 的 Constant；
- `weight` 必须带 `gguf.tensor_name = "token_embd.weight"` 及 source、
  offset、size 等外部引用信息；
- 输出只能是 activation 类型 `f16` 或 `f32`，不能把 Q4_1 block 传给普通
  算子；
- 算子内部负责 Q4_1 解包和解量化，不能在 FE、Core、GenAI host 代码中
  调用 `dequantize_to_f16()`。

Q4_1 的 block 是 32 个元素。实现时必须特别处理 GGUF 的 shape/layout
约定：该文件的 tensor table 显示为 `[4096, 151936]`，而运行时 embedding
逻辑形状为 `[151936, 4096]`。不能仅交换 XML shape；必须依据 GGUF 的
block 顺序、stride 和 `ggml` 参考解码结果确定 token row 到 Q4_1 block
的映射。算子单测必须覆盖：

1. token id 为 0、1、词表最后一个 id；
2. 多个连续 token 和重复 token；
3. batch/sequence 维度组合；
4. 与 host `dequantize_to_f16()` 参考结果逐行比较。

### 19.4 GPU 实现

新增 GPU primitive，例如 `embedding_compressed`，以及对应 OpenCL kernel：

1. 读取 `input_ids`；
2. 根据 GGUF layout 计算目标 token row 的 Q4_1 block 地址；
3. 读取该 row 对应的 scale、minimum 和 4-bit packed values；
4. 在寄存器或局部变量中解量化；
5. 输出 f16/f32 embedding row。

该 kernel 不需要展开完整 `[151936, 4096]` 矩阵。它只读取本次输入中实际
出现的 token rows，因此可直接避免 1.24 GB f16 展开。

建议的性能优化：

- 对 prefill 将 token rows 按 work-group 合并读取，减少每 row 的 kernel
  启动和地址计算开销；
- 对重复 token 增加 row reuse，但不要默认建立完整的 f16 vocabulary cache；
- decode 使用固定 hidden size 的向量化 Q4_1 解包；
- 根据实际 GPU 的 global/local work size 做单独调优；
- 用 Q4_1 的 GGUF 字节带宽和解量化指令吞吐分别评估，而不是与 f16
  `Gather` 直接比较内存大小。

### 19.5 XML 与 `.bin` 结果

生成的 XML 应让 `token_embd.weight` 使用已有的 external-data 形式：

```xml
<data element_type="gguf_q4_1"
      shape="4096,151936"
      source="Qwen3-8B-Q4_1.gguf"
      tensor="token_embd.weight"
      offset="..."
      size="388956160"
      hash="..." />
```

> `size` 必须是 tensor payload 的**字节数**（Q4_1 embedding = 388,956,160），
> **不是**元素数（622,329,856）。这正是 §20.2 offset/size 语义要统一的原因；
> 若误填元素数，reader 会从 389 MB 的区段读取 622 MB，越界并绑定错误权重。

该 Constant 不得在序列化前先变成 f16。serializer 也不得将其字节写入
`.bin`。因此模型图的最小文件集合变为：

```text
Qwen3-8B-Q4_1.xml
Qwen3-8B-Q4_1.gguf
```

`.bin` 可以不存在；如果标准 IR serializer 因其它小型普通 Constant 仍需
生成 `.bin`，则 `.bin` 只能保存这些小常量，不能包含 embedding 或任何
GGUF block 权重。验收时应检查：

```text
size(model.bin) << size(model.gguf)
model.bin 不包含 shape=[151936,4096] 的 f16 Constant
```

注意：去掉 `.bin` 不等于运行时完全不占用 embedding 内存。GPU 可能需要
将压缩的 Q4_1 字节复制到 device/USM 可访问区域，约为 GGUF 中的 389 MB
（388,956,160 字节）；但不再产生约 1.24 GB 的 dense f16 副本。磁盘、
host RSS 和 device memory 必须分别统计，不能把三者混为“零拷贝”。

### 19.6 不能采用的替代方案

1. **只把 XML 的 `element_type` 改成 `gguf_q4_1`**：普通 `Gather` 不认识
   Q4_1 block，运行时会错误读取或直接失败。
2. **加载时重新 `dequantize_to_f16()`**：可以删除持久化 `.bin`，但仍保留
   1.24 GB host f16 峰值和加载开销，不满足“保持原生 GGUF 权重”的最终
   方案。
3. **one-hot + `FullyConnectedCompressed`**：需要构造长度为 151936 的
   one-hot，计算和内存开销远大于 embedding lookup，不可接受。
4. **使用 `output.weight(Q6_K)` 代替 embedding**：两个 tensor 在 GGUF 中独立，
   会改变模型语义。
5. **普通 CPU/NPU fallback**：如果 fallback 仍需全量 f16 展开，就会重新
   引入问题。未实现 `EmbeddingCompressed` 的设备应在 compile 阶段清晰
   `OPENVINO_NOT_IMPLEMENTED`，而不是静默生成 f16 副本。

### 19.7 分阶段交付

#### Phase E0：先修复保存/加载边界

- 给 `token_embd.weight` 保留 GGUF Constant identity；
- 禁止 embedding builder 调用全量 `dequantize_to_f16()` 后再保存 XML；
- 明确 `.bin` 只允许普通小常量；
- 增加 XML 中 Q4_1 external reference 的 round-trip 和严格校验测试。

E0 只能在已有 dense embedding fallback 的设备上作为诊断/过渡方案使用；
它不能宣称已经实现无 f16 展开。

#### Phase E1：GPU `EmbeddingCompressed` 功能实现

- 新增 op/primitive/kernel；
- 支持 Q4_1 `token_embd.weight`；
- 支持动态 batch/sequence/beam；
- 与 host 参考解码逐元素比较；
- 与现有 Q4_1 embedding + Q6_K lm_head 的 token 输出比较。

#### Phase E2：切换默认路径和删除大 `.bin`

- Qwen3 builder 默认创建 `EmbeddingCompressed`；
- serializer 不再看到 f16 embedding Constant；
- cold/warm、XML direct load、GGUF direct load 均使用同一图语义；
- 删除旧的 embedding f16 fallback，或仅保留显式 debug property，默认不启用。

### 19.8 验收标准

对 `Qwen3-8B-Q4_1.gguf` 必须同时满足：

1. GGUF tensor 表仍为 `token_embd.weight=Q4_1`、`output.weight=Q6_K`；
2. XML 中 embedding Constant 的类型为 `gguf_q4_1`，不是 `f16`；
3. XML 中 embedding 引用 `token_embd.weight`，没有引用 `output.weight`；
4. `.bin` 不存在或不包含约 1.24 GB 的 f16 embedding；
5. GPU kernel 结果与 `dequantize_to_f16()` 参考结果在规定容差内一致；
6. cold GGUF load、warm XML+GGUF load 和显式 XML load 的 token id 序列一致；
7. `output.weight` 继续作为独立的 Q6_K `FullyConnectedCompressed` 权重；
8. 复制 XML+GGUF 到新目录后仍可加载，XML 不依赖原机器绝对路径。

---

## 20. 重新分析后的问题清单与修正措施

embedding 方案落地后，当前设计还存在以下不能忽略的问题。它们不改变
E1/E2 的总体方向，但必须进入实现和验收清单。

### 20.1 source hash 仍未实现

第 18 节记录当前只有 offset/size 边界检查，没有严格的 source SHA-256
验证。XML 和 GGUF 被复制到新机器后，如果 GGUF 被替换成同大小文件，仍有
绑定错误权重的风险。

修正：

- 首次生成 XML 时记录完整 GGUF SHA-256；
- warm/direct XML load 先验证 source hash，再验证 tensor name/type/shape/
  offset/size；
- hash 失败必须报错或回退到重新构图，禁止使用旧 `.bin`；
- 大文件可以提供非 strict 的性能选项，但默认验收必须 strict。

### 20.2 offset 语义不一致

设计正文曾同时出现“相对 data section offset”和“相对文件起点 absolute
offset”。当前实现记录的是**相对 GGUF 文件起点的绝对偏移**。两者必须二选一，
否则不同 reader 会绑定到错误位置。

修正：最终协议统一使用：

```text
offset = GGUF 文件起点到 tensor payload 的绝对字节偏移
```

XML/reader/manifest（如导出）统一采用这一语义，并记录 `offset_kind="file"`
或 schema version；旧的 data-section-relative 描述全部改正。加载时仍必须
用 tensor name 重新解析并比较 offset，不能只相信 XML offset。

### 20.3 GGUF element type 的 IR 兼容性

标准 OpenVINO IR、`Constant`、`ConvertPrecision`、`ConstantFolding`、
`transpose_sinking` 和 GPU transformation 都可能误把 opaque block 当成
普通逐元素 tensor。

修正：

- 所有 GGUF Constant 设置 `disable_constant_folding`；
- 对 GGUF Constant 禁止 Convert/Reshape/Transpose/按元素访问；
- 加入 XML round-trip 后的 type/shape/byte hash gate；
- 对 `EmbeddingCompressed` 和 `FullyConnectedCompressed` 的 weight 输入
  做 compile-time 验证，禁止普通算子消费 GGUF block。

### 20.4 embedding 的 shape/layout 不能靠字符串修正

本测试文件的 GGUF tensor table 为 `[4096,151936]`，而 XML dense fallback
为 `[151936,4096]`。如果 kernel 把这两个 shape 简单交换，可能得到能运行但
数值错误的 embedding。

修正：保存 GGUF 原始 shape、逻辑 embedding shape、row stride 和 block
geometry；用固定 token id 的逐行 oracle 检查地址计算。该问题是 E1 的
阻塞项，不允许只用最终生成文本“看起来正常”作为验收。

### 20.5 direct `.xml` 输入还需要独立验证

之前主要验证了传入 `.gguf` 路径时的 sibling discovery。传入目录或直接传入
`.xml` 的路径解析仍可能走普通 IR directory discovery，导致找不到模型或
没有触发 sibling GGUF 绑定。

修正：增加明确测试：

```text
read_model("foo.gguf")  → foo.xml + foo.gguf
read_model("foo.xml")    → foo.xml + foo.gguf
read_model("/dir")       → 不把目录误当作 GGUF 模型文件
```

XML 输入必须在进入普通 IR directory logic 前被识别为 GGUF paired XML。

### 20.6 tokenizer/detokenizer 是独立产物

tokenizer 使用 GGUF metadata，不是 tensor payload；不能把 tokenizer 当作
`token_embd.weight` 的外部 GGUF tensor，也不能因为模型 `.bin` 被去掉而
删除 tokenizer `.bin`。

最终包应允许：

```text
foo.xml + foo.gguf
foo_tokenizer.xml + foo_tokenizer.bin
foo_detokenizer.xml + foo_detokenizer.bin
```

tokenizer/detokenizer 使用 basename 命名、原子写入，并通过 GGUF source hash
与模型包绑定；目标机器仍需安装匹配版本的 `openvino_tokenizers`。

### 20.7 cold/warm 图的算子集合必须一致

首次 native GGUF 构图和 XML warm load 不能出现：cold 使用
`EmbeddingCompressed`，warm 恢复成普通 `Gather`，或反之。

修正：对两条路径比较：

- embedding op 类型；
- embedding weight element type；
- `token_embd.weight` identity；
- lm_head op 类型和 `output.weight` identity；
- 节点数量、输入输出名称和关键 rt-info；
- 固定 prompt 的 token id 序列。

### 20.8 “去掉 `.bin`”与“去掉所有内存开销”不是同一目标

E1 主要消除 1.24 GB dense f16 的磁盘副本和 host 全量展开；它不保证
GGUF 压缩字节永远不进入 GPU memory，也不保证 embedding lookup 的临时
输出为零字节。性能报告必须分别记录：

```text
GGUF 文件大小
模型 XML/.bin 大小
host RSS / mmap 映射大小
device allocation
embedding kernel latency
```

### 20.9 当前 serializer 对普通小常量的处理

即使 embedding 已 externalize，模型中仍可能存在 f32/f16 小常量、RoPE 常量、
scale 或 KV-cache 配置常量。验收不能简单要求 `.bin` 必须为零，而应要求
`.bin` 不包含任何 GGUF 大权重，且大小与普通辅助常量总量一致。

### 20.10 版本与失败策略

XML 中的 external schema、GGUF element type table、builder 和 GPU primitive
都需要版本字段。目标机器不支持 `EmbeddingCompressed` 或 Q4_1 时，必须给出
清晰的 `OPENVINO_NOT_IMPLEMENTED`，不能静默回退到全量 f16（除非用户显式
打开 debug fallback）。

### 20.11 最终优先级

实现顺序固定为：

```text
修正 offset/hash 协议
  ↓
修正 direct XML 输入与 cold/warm parity
  ↓
保留 token_embd.weight 的 Q4_1 external Constant
  ↓
实现 Q4_1 EmbeddingCompressed GPU kernel
  ↓
删除 f16 embedding serializer/fallback
  ↓
完成 tokenizer basename 产物与整包一致性测试
```

完成上述步骤后，`Qwen3-8B-Q4_1.xml + Qwen3-8B-Q4_1.gguf` 才真正满足“图与
原始 GGUF 权重分离、无 1.24 GB embedding `.bin`、可跨机器搬运”的最终目标。

---

## 21. embedding shape/layout 防错与多格式支持

### 21.1 三层形状契约：物理、逻辑、执行

embedding 最容易出现的错误，是把 GGUF tensor table 中的 shape 当成普通
OpenVINO dense tensor shape，或者通过交换 XML shape 试图修复转置。最终实现
必须同时保存三种信息，不能互相替代：

| 层次 | 示例 | 用途 |
|---|---|---|
| GGUF 物理 shape | `[4096, 151936]` | 描述 GGUF header 中的原始维度和 block 排列，不能修改 |
| 逻辑权重 shape | `[151936, 4096]` | 描述 embedding 的 `vocab × hidden` 语义，供算子契约使用 |
| 执行布局 | `row = token_id, col = hidden_id` | 描述 kernel 如何从 block 字节定位一个输出元素 |

建议在 `EmbeddingCompressed` 的 weight rt-info 中显式记录：

```text
gguf.tensor_name       = "token_embd.weight"
gguf.physical_shape    = [4096, 151936]
gguf.logical_shape     = [151936, 4096]
gguf.embedding_axis    = 1
gguf.row_axis          = 1
gguf.block_elem_count  = 32 或 256
gguf.block_byte_size   = 18、20、144、...
gguf.layout_version    = 1
```

`physical_shape` 必须与 GGUF reader 重新解析出的 tensor metadata 完全一致；
`logical_shape` 必须与 Qwen3 的 vocab size、embedding length 和输入 token
range 一致；`layout_version` 用于阻止旧 kernel 按错误规则解释新布局。

XML 的 `shape` 字段不应承担这三种语义。推荐让 opaque Constant 保留
GGUF physical shape，同时由 `EmbeddingCompressed` 的属性表达 logical shape
和 axis mapping。若当前 IR/primitive 契约必须把 Constant shape 暴露成逻辑
shape，也必须额外保存 physical shape，并在 compile 阶段做一致性验证；不能
只交换一个 shape 字符串。

### 21.2 使用唯一的地址映射 oracle

不要在 FE、serializer、GPU kernel 中分别“猜”地址。实现一个唯一的、可测试的
`EmbeddingLayout` 描述和 host reference mapper，GPU JIT 只消费该描述导出的
常量：

```text
map(token_id, hidden_id, physical_shape, format_traits)
    -> (block_index, byte_in_block, value_in_block)
```

对每个格式，mapper 必须验证以下不变量：

```text
0 <= token_id < vocab_size
0 <= hidden_id < hidden_size
block_index * block_byte_size + byte_in_block < tensor_payload_size
decode(map(token_id, hidden_id)) == ggml_reference(token_id, hidden_id)
```

推荐的防错流程是：

1. 从 GGUF tensor metadata 计算 `vocab_size`、`hidden_size`、block 数量和
   payload size；
2. 用 `ggml`/`llama.cpp` reference dequant 对 token id
   `{0, 1, vocab-1}` 的完整 row 生成黄金数据；
3. 用 host mapper 逐元素比较完整 row；
4. 用 GPU kernel 只比较随机 row 和边界 row；
5. 最后再比较完整 embedding output 和最终 logits。

完整 row oracle 应在每个 GGUF 格式上运行，而不是只在 Q4_1 上运行。这样可以
把“shape 错误”和“某格式 decoder 错误”分离出来。

### 21.3 统一格式 traits，不复制 embedding 算子

`EmbeddingCompressed` 不应为 Q4_0、Q4_1、Q4_K、Q8_0 分别创建四个
public op。建议使用一个 op + 一个格式 traits 表：

```text
EmbeddingCompressed
  ├── Q4_0Traits  (block=32, bytes=18, symmetric scale)
  ├── Q4_1Traits  (block=32, bytes=20, scale + minimum)
  ├── Q4_KTraits  (block=256, bytes=144, super-scale + sub-scales)
  ├── Q5_0Traits  (block=32, bytes=22, high-bit plane)
  ├── Q5_1Traits  (block=32, bytes=24, high-bit + minimum)
  ├── Q5_KTraits  (block=256, bytes=176, nested scales)
  ├── Q6_KTraits  (block=256, bytes=210, 6-bit values + sub-scales)
  ├── Q8_0Traits  (block=32, bytes=34, signed int8 values)
  └── Q8_KTraits  (block=256, bytes=292, only when used as weight)
```

其中 traits 至少提供：

```cpp
struct EmbeddingFormatTraits {
    element::Type_t type;
    uint32_t block_elements;
    uint32_t block_bytes;
    uint32_t row_block_count;
    uint32_t values_per_subblock;
    DecodeKind decode_kind;
    bool has_minimum;
    bool is_supported_as_weight;
};
```

`row_block_count` 必须由 `hidden_size / block_elements` 计算并检查整除；不能
从文件名中的 `Q4_1`、`Q4_K_M` 或 `Q5_K_M` 推断 embedding 格式。`*_M`/`*_S`
是模型级 mixed recipe，实际 embedding 格式必须读取 `token_embd.weight` 的
GGUF tensor type。

### 21.4 推荐的支持分层

不同格式的 block geometry 不同，不能强行让一个 decoder 使用一套位移公式。
但可以让它们共享 dispatch、输入检查、row 复用和测试框架：

| 层 | Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 | Q4_K/Q5_K/Q6_K/Q8_K |
|---|---|---|
| 地址计算 | 32 元素 block，row 内连续 block | 256 元素 super-block，row 内连续 super-block |
| 解码 | `Type-0/1` 专用 decoder | K-quant 专用 decoder，处理 nested scale |
| 优化重点 | 向量化 nibble/high-bit unpack | sub-scale 解包、减少重复 scale 读取 |
| 精度验证 | 与对应 `ggml` block decoder 比较 | 必须逐 super-block 比较，禁止只比较最终 logits |

第一阶段建议按以下顺序落地：

1. `Q4_0`、`Q4_1`：验证 32-element row mapping 和 symmetric/asymmetric 两种
   scale 语义；
2. `Q8_0`：作为 byte-oriented decoder 和高精度基线；
3. `Q4_K`、`Q5_K`、`Q6_K`：验证 256-element super-block 及 nested scale；
4. `Q5_0`、`Q5_1`、`Q8_K`：复用已有 dispatch，增加 high-bit 或 super-block
   decoder；
5. IQ/TQ：沿用同一 op，但单独引入 codebook/ternary traits 和质量门禁。

未实现的格式必须在 compile 阶段报告具体的 GGUF type。不得把不支持的
embedding 自动展开成 f16，也不得把 Q4_K 误当成 Q4_0；这两种行为都会产生
静默的错误或内存回归。

### 21.5 embedding 专用的 shape/layout 校验门

`EmbeddingCompressed::validate_and_infer_types()` 和 GPU
`validate_impl()` 至少检查：

1. weight tensor name 是 `token_embd.weight`；
2. physical shape、logical shape、vocab size、hidden size 彼此一致；
3. hidden size 能被该格式的 block element count 整除；
4. GGUF payload size 等于 `row_count × row_block_count × block_byte_size`；
5. input token id 类型是整数，且 rank/动态维度属于支持范围；
6. 不存在会改变 weight layout 的 `Transpose`、`Reshape` 或 `Convert`；
7. kernel decoder 的 `layout_version` 与 XML/rt-info 一致；
8. GGUF source hash、tensor offset、tensor size 与重新扫描的 GGUF metadata 一致。

任意一项失败都应在 compile/load 阶段报错，并包含 tensor name、GGUF type、
physical shape、logical shape 和 offset。不要等到第一次 inference 才暴露
越界或错误 token row。

### 21.6 性能优化：先稳定 row mapping，再优化访问

shape/layout 正确后，性能优化应按访问模式分开处理：

- **decode / 少量 token**：一个 work-group 处理一个或少量 token row；hidden
  维度并行，block 地址只计算一次；
- **prefill / 连续 token**：检测连续 token id，但不能假定 token id 连续；
  对重复 token 可做小型 row cache，不能创建完整 f16 vocabulary cache；
- **Q4_0/Q4_1/Q5_0/Q5_1**：将 packed values 和 scale/minimum 一起加载，减少
  非合并读取；
- **Q4_K/Q5_K/Q6_K**：按 256-element super-block 对齐 work-group，复用
  super-scale/sub-scale，避免每个 hidden lane 重复解包；
- **Q8_0/Q8_K**：使用向量化 byte load，减少无意义的 nibble 操作；
- **所有格式**：输入 token id 先做边界检查；性能版本可在 compile 阶段确认
  范围后关闭重复检查，但 debug/reference kernel 必须保留检查。

性能报告至少分别记录各格式的：row mapping kernel latency、GGUF bytes read、
effective bytes/token、occupancy、以及与 host reference 的误差。不能只用最终
生成 token 作为正确性和性能指标。

### 21.7 新增验收矩阵

每种实际支持的 embedding 格式都必须覆盖：

```text
format ∈ {Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0, Q8_K}
token_id ∈ {0, 1, vocab-1, repeated, continuous, random}
shape ∈ {batch=1/2, sequence=1/4/128, beam=1/4}
input layout ∈ {static, dynamic batch/sequence}
```

每个组合至少执行：

1. GGUF metadata 与 traits 一致性检查；
2. block/row 地址边界检查；
3. 与 `ggml` reference 的逐元素 embedding row 比较；
4. cold GGUF load 与 warm XML+GGUF load 比较；
5. GPU 结果与最终 logits/token 序列比较。

该矩阵允许实现上先支持一部分格式，但不允许“声明支持全部格式、实际
通过错误 decoder 运行”。

---

## 22. 方案验证：与真实代码/文件核对后的结论

本节记录用 `Qwen3-8B-Q4_1.gguf`（gguf reader）和现有 builder 代码
（[qwen3_builder.cpp](../../../thirdparty/openvino/src/frontends/gguf/src/builders/qwen3_builder.cpp)）
对第 19–21 节做的事实核对结果。核对目的是确认最终方案是否存在问题。

### 22.1 已核对为正确的假设

| 假设 | 核对结果 |
|---|---|
| `token_embd.weight = Q4_1`，`output.weight = Q6_K` | 正确（reader 实测） |
| 两者形状相同（`[4096,151936]`）但类型不同、独立 | 正确，不能互相替代 |
| physical `[4096,151936]` vs logical `[151936,4096]` | 正确 |
| `hidden = 4096` 可被 32 和 256 整除 | 正确（`4096%32=0`、`4096%256=0`），Q4_0/Q4_1/Q4_K/Q5_K/Q6_K 的 block 都不跨 row 边界 |
| 当前 `.bin ≈ 1.24 GB` 来自 embedding 全量 f16 | 正确（`dequantize_to_f16("token_embd.weight")` + `Gather`，见 builder 第 552 行） |
| `payload = row_count × (hidden/block_elem) × block_byte`（§21.5 #4） | 正确：`151936 × 128 × 20 = 388,956,160` |

### 22.2 已修正的事实错误

| 位置 | 原错误 | 修正 |
|---|---|---|
| §19.1 | embedding Q4_1「约 622 MB」 | 实为 388,956,160 字节 ≈ 371 MiB；622,329,856 是**元素数**不是字节 |
| §19.5 XML | `size="622329856"` | 改为字节数 `388956160`；误填元素数会读越界 |
| §19.5 | device copy「约 622 MB」 | 改为约 389 MB |

这类「元素数 vs 字节数」混用正是 §20.2 警告的 offset/size 语义风险，
但示例本身此前就踩了坑，说明该风险是真实且高发的。

### 22.3 方案的最大结构性问题：低估了可复用的 FC 解码路径

现有代码里已存在一个关键事实，第 19–21 节没有利用：

```cpp
// qwen3_builder.cpp:74
bool is_gpu_supported_fc_type(const ov::element::Type& type) {
    return ... || type == ov::element::gguf_q4_0 || type == ov::element::gguf_q4_1
        || type == ov::element::gguf_q4_k || type == ov::element::gguf_q5_k
        || type == ov::element::gguf_q6_k || type == ov::element::gguf_q8_0 || ...;
}
```

也就是说 **GPU 的 `FullyConnectedCompressed` 已经能原生解码 Q4_1（以及
Q4_0/Q4_K/Q5_K/Q6_K/Q8_0）block**。而且当模型 tied weights（无
`output.weight`）时，builder 第 602 行会用 `token_embd.weight` 直接喂
`make_fc → FullyConnectedCompressed`（第 106 行）——**同一份 Q4_1
embedding 张量已经作为压缩 FC 权重在 GPU 上被消费过**。

更重要的是布局完全一致：

- lm_head 的 FC 权重逻辑形状是 `[N=vocab=151936, K=hidden=4096]`；
- GGUF 中 `ne0=4096`（hidden，连续）沿 K 平铺 128 个 Q4_1 block，
  `ne1=151936`（vocab）是 output row；
- embedding lookup 需要的正是「取第 `token_id` 个 output row 的 4096 个
  hidden 值」——**和 FC 解码第 `token_id` 行权重是同一批 block、同一套
  地址公式**。

**结论**：§21 担心的「shape/layout 要从零建 oracle、容易出错」很大程度上
已被现网可用的 Q4_1/Q6_K FC 路径证伪。embedding 的正确落地方式应是
「按行选择的 compressed gather」，直接复用 FC 的 per-row block 解码内核，
而不是从头写一套独立的地址映射。这样：

1. 布局风险大幅下降（复用已验证正确的 FC 布局）；
2. 多格式支持几乎免费（FC 已支持 Q4_0/Q4_1/Q4_K/Q5_K/Q6_K/Q8_0）；
3. §21.3 的 traits 表应与 FC 现有 decoder 复用，而非平行新建。

设计文档应把 §19.2 / §21 的「新建 EmbeddingCompressed + 全新 layout
oracle」修正为「复用 FC 压缩权重 decoder 的 row-gather」，仅在 gather
维度选择上新增逻辑。

### 22.4 仍然成立的待办

以下问题与代码核对无冲突，仍需实现：

1. serializer 仍会把 embedding 落成 f16 `.bin`（builder 第 552 行未改）；
2. 无论复用 FC decoder 与否，都需要一个「按 token_id 取行、只解码所选
   行」的 GPU 执行路径，避免 one-hot × FC（§19.6 #3）的开销；
3. source hash（§20.1）、offset/size 字节语义（§20.2、§22.2）、direct
   XML 输入（§20.5）、cold/warm parity（§20.7）仍未实现；
4. `is_gpu_supported_fc_type` 未含 IQ4/Q5_0/Q5_1/Q8_K，§21.4 若要覆盖这些
   格式，需要同时扩 FC decoder 与 embedding 路径。

### 22.5 验证结论

最终方案**方向正确、无致命逻辑错误**，但存在两类必须修正的问题：

- **事实错误（已在本次修正）**：embedding 字节大小、XML `size` 字段用了
  元素数而非字节数；
- **设计冗余（建议修正）**：把 embedding 当作全新算子 + 全新 layout
  oracle，低估了「Q4_1 已作为压缩 FC 权重在 GPU 正确解码」这一既有能力；
  应改为复用 FC 的 per-row block decoder。

采纳 §22.3 的复用方案后，「shape/layout 风险」和「多格式支持」两项都
能显著简化，与用户本轮关注点一致。
