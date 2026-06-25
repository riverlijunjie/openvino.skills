# GGUF 原生支持 — 设计冻结（PR-0 SPEC）

**版本**：v1.0 — *待人工 review，签字后冻结*
**作者**：(待填)
**Reviewer**：OV Core / Common Transformations / GPU Plugin / GenAI 各一名
**日期**：2026-06-08
**状态**：DRAFT

本 SPEC 是 [SUMMARY.md](SUMMARY.md) §12.2 工作分解中 **PR-0** 阶段的产出，
是 **PR-FE / PR-GPU / PR-GENAI** 三个本地 PR 的**唯一**输入契约。
一旦冻结：

- AI agent **不允许**自主修改本文档定义的任何 schema / API 名称 /
  字段语义；如需变更，必须回到本 SPEC 走人工 review 流程后再展开
  下游 PR。
- 本 SPEC 之外的实现细节（kernel 算法、目录组织、JIT 模板）AI 可
  自主决定，前提是不违反本 SPEC。
- **本期基准架构 = qwen3**（唯一），其他架构留给后续增量 PR。
- **所有代码本地 commit，禁止 push 到 remote**；CI 通过本地脚本
  跑（由 PR-FE 的 `ci-gates` subagent 一并交付）。

---

## 0. 范围 & 非目标

### 0.1 In-scope（按 3 个 PR 分组）

**PR-FE（`openvino` 仓）**：

1. OV Core 新增 23 个 GGUF block element-type（§1，**全集一次加齐**
   以避免后续 ABI 跳号；kernel 是否可用与 element-type 是否定义分离）。
2. Common transformations 加 GGUF 不变量守卫（§5，8 处必须 patch）。
3. `src/frontends/gguf/` 新增原生 GGUF FrontEnd（§3），架构 builder
   **只实现 qwen3**。
4. 本地 CI 脚本：GGUF 字节不变量门 + IR round-trip 门（§7.1 / §7.4）。

**PR-GPU（`openvino` 仓）**：

5. GPU 插件新增 `ocl::FCGGUFOpt` ImplementationManager（§4）。
6. **Baseline 5 种格式** kernel：`Q4_0` / `Q4_K` / `Q5_K` / `Q6_K` / `Q8_0`
   （覆盖 qwen3 GGUF 常见量化配方 Q4_K_M / Q5_K_M / Q6_K / Q8_0）。
7. 其他 18 种 element-type 在 `validate_impl` 中返回 false，错误信息
   清晰指向"本期未覆盖"。
8. `ov::supported_gguf_types` property（§4.4）。

**PR-GENAI（`openvino.genai` 仓）**：

9. 切换 `is_gguf_model()` 走 `Core::read_model`（§6.1）。
10. Tokenizer 输入源迁移到 `model.get_rt_info()`（§6.2）。
11. 删除被 FE 取代的 in-tree dequant 代码（§6.3）。
12. token 一致性 oracle（§7.2，**仅 qwen3**）。

### 0.2 Out-of-scope（明确不做）

- 在 OV Core / FE / GenAI 任何上层路径 dequant GGUF block 到
  `f16`/`f32`（**硬约束**，违反即拒绝）。
- 新增 CPU / NPU / AUTO / HETERO 插件支持 —— 本期只 GPU。
- **llama / qwen2 / mistral / phi 等其他架构**：本期不做，留给后续
  增量 PR（每个一个 builder 文件）。
- **18 种非 baseline element-type 的 GPU kernel**：本期 element-type
  定义齐全但 kernel 不实现，遇到时清晰报错。
- 修改 GGUF 文件格式或定义新的 OpenVINO 私有变体。
- 训练 / 微调路径（read-only inference 才在本期范围）。
- GGUF 中非权重张量类型（`F32`/`F64`/`F16`/`BF16`）的特殊处理
  —— 走现有 element-type，不在本 SPEC 范围。
- **push 到 remote / 走 GitHub PR 流程**：本期纯本地交付。

### 0.3 硬约束（不可协商）

| ID | 约束 | 验证方式 |
|----|------|----------|
| C1 | GGUF 权重以原始 block 字节进入 `ov::Constant`，**不在 FE / Core 做 dequant** | §7.1 byte-equality 本地 gate |
| C2 | dequant 只能发生在 GPU OCL kernel 寄存器内或 OneDNN WOQ 低比特域 | §4.3 contract |
| C3 | 任何 transformation pass 不得改写 GGUF Constant 的 shape/dtype/字节内容 | §5 invariance + §7.1 gate |
| C4 | `Core::read_model("*.gguf")` 必须产出可被 Benchmark App / accuracy checker 直接消费的 `ov::Model` | §3.4 acceptance |
| C5 | 与 `llama.cpp` reference 在 qwen3 greedy decode、seed=0、256 token 内 token 完全一致 | §7.2 token equality oracle |
| C6 | 所有 commit 留本地分支，**禁止 push 到 origin** | 人工 review；git hook 可选 |

---

## 1. Element-type 定义

### 1.1 完整枚举（PR-FE `core-types` subagent 输入）

**全集一次加齐**：23 个 element-type 在 PR-FE 中一次性定义，**与 GPU
kernel 是否实现解耦**。kernel 未实现的 type 在 PR-GPU 的 `validate_impl`
中明确返回 false。这样做的原因：element-type 是 ABI 表面，跳号或
后续增加会破坏二进制兼容；kernel 增加属于内部实现，可以增量。

在 `src/core/include/openvino/core/type/element_type.hpp` 的 `Type_t`
枚举中按如下顺序追加；**不允许重排、不允许跳号**：

```cpp
enum class Type_t {
    // ... existing entries ...

    // GGUF block types (PR-FE). Order matches GGML ggml_type enum
    // (https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h).
    gguf_q4_0,    gguf_q4_1,
    gguf_q5_0,    gguf_q5_1,
    gguf_q8_0,    gguf_q8_1,
    gguf_q2_k,    gguf_q3_k,    gguf_q4_k,    gguf_q5_k,    gguf_q6_k,    gguf_q8_k,
    gguf_iq2_xxs, gguf_iq2_xs,  gguf_iq3_xxs, gguf_iq1_s,
    gguf_iq4_nl,  gguf_iq3_s,   gguf_iq2_s,   gguf_iq4_xs,
    gguf_iq1_m,
    gguf_tq1_0,   gguf_tq2_0,
};
```

**总计 23 个新枚举值**。

### 1.2 `TypeInfo` 字段（每种类型一行）

字段语义：

| 字段 | GGUF 类型一律取值 | 备注 |
|------|--------------------|------|
| `is_real()` | `false` | 必须 false —— `fp16_compression` 等 pass 据此跳过 |
| `is_quantized()` | `true` | 必须 true —— `MarkDequantizationSubgraph` 据此处理 |
| `is_signed()` | 见下表 | 用于 `Convert` carve-out 判定 |
| `bitwidth()` | `ceil(block_byte_size * 8 / block_elem_count)` | fractional bpw 上取整（例：Q4_K = 5；IQ1_S = 2） |
| `size()` (字节/元素) | 始终返回 `0`，调用方必须改用 `block_byte_size()` | 与 `nf4`/`u4` carve-out 模式一致 |
| `c_type_string()` | `"gguf_q4_0"` 等下划线小写 | 序列化用 |

`block_byte_size` / `block_elem_count` 表（必须与此表完全一致）：

| 类型 | block_elem | block_bytes | signed |
|------|-----------:|------------:|:------:|
| `gguf_q4_0`    | 32  | 18  | true  |
| `gguf_q4_1`    | 32  | 20  | false |
| `gguf_q5_0`    | 32  | 22  | true  |
| `gguf_q5_1`    | 32  | 24  | false |
| `gguf_q8_0`    | 32  | 34  | true  |
| `gguf_q8_1`    | 32  | 36  | false |
| `gguf_q2_k`    | 256 | 84  | false |
| `gguf_q3_k`    | 256 | 110 | true  |
| `gguf_q4_k`    | 256 | 144 | false |
| `gguf_q5_k`    | 256 | 176 | false |
| `gguf_q6_k`    | 256 | 210 | true  |
| `gguf_q8_k`    | 256 | 292 | true  |
| `gguf_iq1_s`   | 256 | 50  | true  |
| `gguf_iq1_m`   | 256 | 56  | true  |
| `gguf_iq2_xxs` | 256 | 66  | true  |
| `gguf_iq2_xs`  | 256 | 74  | true  |
| `gguf_iq2_s`   | 256 | 82  | true  |
| `gguf_iq3_xxs` | 256 | 98  | true  |
| `gguf_iq3_s`   | 256 | 110 | true  |
| `gguf_iq4_nl`  | 32  | 18  | true  |
| `gguf_iq4_xs`  | 256 | 136 | true  |
| `gguf_tq1_0`   | 256 | 54  | true  |
| `gguf_tq2_0`   | 256 | 66  | true  |

### 1.3 新增公开访问器

在 `element::Type` 上新增（PR-FE 必须实现）：

```cpp
size_t block_byte_size()  const noexcept;  // 例：gguf_q4_k -> 144；非 GGUF -> 0
size_t block_elem_count() const noexcept;  // 例：gguf_q4_k -> 256；非 GGUF -> 0
bool   is_gguf_block()    const noexcept;  // 23 个枚举返回 true，其他 false
```

同步在 free function 层新增 `inline bool is_gguf_block(element::Type t)`
做无对象判定，供 transformations 使用。

### 1.4 `op::v0::Constant` 行为

- `Constant::get_byte_size()` 对 GGUF 类型返回
  `ceil_div(num_elements, block_elem_count) * block_byte_size`。
- `Constant::get_data_ptr<void>()` 返回原始 GGUF block 字节起始；
  **不**提供 `get_data_ptr<T>()` 的类型化重载（编译期 `static_assert`
  禁止）。
- 反序列化路径（`IRDeserializer`）按 `block_byte_size` 读字节段，
  不做 endian 转换（GGUF 规范固定 little-endian，OpenVINO 现有平台
  全部 little-endian，加 `static_assert`）。
- `element_iterator` 对 GGUF 类型抛 `OPENVINO_NOT_IMPLEMENTED` 并
  提示用户："GGUF block types are opaque; use GPU plugin to consume."

### 1.5 Convert / ConstantFolding carve-out

- `op::v0::Convert::validate_and_infer_types()`：当 input dtype 是
  GGUF block 时抛 `OPENVINO_THROW`（参考 `nf4` carve-out 行内位置）。
- `pass::ConstantFolding`：扫描到 GGUF Constant 时跳过整个子图折叠
  路径，并把 `disable_constant_folding` rt-info 自动打到该 Constant
  上（防御性）。

---

## 2. ~~Constant 字节不变量~~（合并到 §5 invariance）

（占位以保留章节编号；具体内容见 §5。）

---

## 3. Frontend I/O 契约

### 3.1 目录布局

```
src/frontends/gguf/
├── CMakeLists.txt
├── include/openvino/frontend/gguf/frontend.hpp
├── src/
│   ├── frontend.cpp                   # FrontEnd / InputModel 实现
│   ├── gguf_reader.{hpp,cpp}          # mmap + header + tensor table 解析
│   ├── gguf_types.hpp                 # GGUF type enum ↔ element::Type 映射
│   ├── builders/
│   │   ├── builder.hpp                # 架构 builder 抽象基类
│   │   └── qwen3_builder.cpp          # PR-FE 唯一交付的 builder
│   └── rt_info_keys.hpp               # §3.3 schema 字符串常量
└── tests/

# 后续增量（不在本期 3 个 PR 范围）：
#   builders/llama_builder.cpp, qwen2_builder.cpp, mistral_builder.cpp, phi_builder.cpp
```

### 3.2 公开入口

```cpp
namespace ov::frontend::gguf {

class FrontEnd : public ov::frontend::FrontEnd {
public:
    std::string get_name() const override { return "gguf"; }
    bool supported_impl(const std::vector<ov::Any>& vars) const override;
    InputModel::Ptr load_impl(const std::vector<ov::Any>& vars) const override;
    std::shared_ptr<ov::Model> convert(const InputModel::Ptr& model) const override;
};

}  // namespace ov::frontend::gguf
```

`supported_impl()` 判定：

1. 路径以 `.gguf` 结尾，**或**
2. 第一个参数是 `std::string` / `std::filesystem::path` / `std::istream*`
   且前 4 字节为 GGUF magic `0x46554747`（"GGUF"）。

`convert()` 行为：当 `gguf.architecture` 不是 `"qwen3"` 时
`OPENVINO_THROW("GGUF architecture '<X>' not supported in this release (qwen3 only)")`；
rt-info 的 metadata 仍正常填值（§3.3）以便后续增量 PR 复用。

### 3.3 rt-info schema（**冻结后不可改**）

**API 约定**：所有 key 走 `ov::Model::set_rt_info` / `get_rt_info` 的
**variadic 路径形式**，顶层段一律为 `"gguf"`，与 OV 现行约定一致
（参考 CPU `runtime_options.kv_cache_precision`、`intel_cpu_hints_config`，
PyTorch FE `decoder_type_name`）。OV rt-info key 体系**不使用** `::`
分隔，因此不写 `gguf::xxx`；下表中的点号仅为可读路径展示，落到代码
中是 variadic 参数链。

> **为什么不加 `ov::` 前缀**：OV 各 plugin/FE 自身的 rt-info 顶层段都
> 不带 `ov_` 前缀（`runtime_options` / `intel_cpu_hints_config` /
> `decoder_type_name` / `is_new_api`），插件名/格式名直接作为顶层段
> 已经构成命名空间。rt-info dict 是 **per-`ov::Model` 局部**的，不存
> 在跨进程全局表，只要顶层段 `"gguf"` 不与同一模型上其他写入者重名
> 即无冲突；`"gguf"` 由 GGUF FE 独占写入，PR-GENAI 只读，无碰撞风险。

PR-FE 实现写入，PR-GENAI 消费。代码示例：

```cpp
// 写入（PR-FE / src/frontends/gguf）
model->set_rt_info(ov::Any(arch), "gguf", "architecture");
model->set_rt_info(ov::Any(tokens), "gguf", "tokenizer", "ggml", "tokens");

// 读取（PR-GENAI / openvino.genai）
auto arch = model->get_rt_info<std::string>("gguf", "architecture");
auto tokens = model->get_rt_info<std::vector<std::string>>(
    "gguf", "tokenizer", "ggml", "tokens");
```

| 路径（点号仅为展示） | 类型 | 含义 | 必填 |
|---------------------|------|------|------|
| `gguf.version` | `uint32_t` | GGUF 文件版本（>=3） | 是 |
| `gguf.architecture` | `std::string` | `general.architecture`；本期 builder 只接受 `"qwen3"` | 是 |
| `gguf.model_name` | `std::string` | `general.name` | 否 |
| `gguf.file_type` | `uint32_t` | GGUF 文件类型（混合精度配方代码，如 14 = Q4_K_M） | 是 |
| `gguf.context_length` | `uint64_t` | `<arch>.context_length` | 是 |
| `gguf.embedding_length` | `uint64_t` | `<arch>.embedding_length` | 是 |
| `gguf.block_count` | `uint64_t` | `<arch>.block_count` | 是 |
| `gguf.attention.head_count` | `uint64_t` | | 是 |
| `gguf.attention.head_count_kv` | `uint64_t` | | 是 |
| `gguf.attention.layer_norm_rms_epsilon` | `float` | | 是 |
| `gguf.rope.dimension_count` | `uint64_t` | | 是 |
| `gguf.rope.freq_base` | `float` | | 否 |
| `gguf.rope.scaling.type` | `std::string` | `"none"` / `"linear"` / `"yarn"` | 否 |
| `gguf.rope.scaling.factor` | `float` | | 否 |
| `gguf.tokenizer.ggml.model` | `std::string` | `"llama"` / `"gpt2"` 等 | 是 |
| `gguf.tokenizer.ggml.tokens` | `std::vector<std::string>` | 词表 | 是 |
| `gguf.tokenizer.ggml.scores` | `std::vector<float>` | SP 评分 | 否 |
| `gguf.tokenizer.ggml.token_type` | `std::vector<int32_t>` | | 否 |
| `gguf.tokenizer.ggml.merges` | `std::vector<std::string>` | BPE merges | 否 |
| `gguf.tokenizer.ggml.bos_token_id` | `uint32_t` | | 是 |
| `gguf.tokenizer.ggml.eos_token_id` | `uint32_t` | | 是 |
| `gguf.tokenizer.chat_template` | `std::string` | Jinja chat 模板 | 否 |
| `gguf.source_file_hash` | `std::string` | 源 GGUF 文件 SHA256（小写 hex） | 是 |

任何新增 key 都必须先更新本表。

### 3.4 输出 `ov::Model` 契约（acceptance）

- **每个 GGUF 权重张量** → 一个 `op::v0::Constant`，dtype 为 §1.1
  中对应的 `gguf_*` 类型，shape 为 `[N, K]`（行主 = 输出维度先）。
- **每个 FC 层** → `FullyConnectedCompressed` 节点，输入 0 是
  activation，输入 1 是上述 GGUF Constant，`weight_scales` /
  `weight_zero_points` 输入**留空**（impl 从 block 内自取）。
- **不允许**在 FE 内插入任何 `Convert` / `Multiply` / `Subtract`
  形式的 dequant 节点（违反 C1）。
- **mmap**：默认开启；可以通过 `ov::frontend::gguf::FrontEnd::load`
  的额外 `ov::AnyMap` 参数 `{"mmap_enable", false}` 关闭做诊断。
  Constant 持有的字节生命周期通过 `AlignedBuffer` 绑定到 mmap region。
- **stream 输入路径**（非 mmap）必须 `std::move` 到 `AlignedBuffer`，
  不允许额外拷贝。

### 3.5 错误处理

- 未识别架构（含本期不支持的 llama/qwen2/mistral/phi/…）→
  `OPENVINO_THROW("GGUF architecture '<X>' not supported in this release (qwen3 only); see PR-FE follow-ups")`。
- 未识别 GGUF type（GGML 后续可能新增）→ `OPENVINO_THROW`，
  **不允许**回退到"按未知字节读取"。
- 文件截断 / magic 错误 → `OPENVINO_THROW` 含字节偏移。

---

## 4. GPU 插件 `ocl::FCGGUFOpt` 契约

### 4.1 注册位置

[fully_connected_impls.cpp](src/plugins/intel_gpu/src/graph/registry/fully_connected_impls.cpp)
增加一行：

```cpp
OV_GPU_CREATE_INSTANCE_OCL(ocl::FCGGUFOpt, static_shape)
```

不增加 dynamic_shape variant（GGUF 模型权重 shape 在 FE 阶段已定）。

**命名 / 类别约定**（已对齐 `ocl_v2/` 现行规则）：

- 类型为 **`struct`** 而非 `class`，与 `FCCompressedGenerateOpt` /
  `RopeOpt` / `SDPAOpt` / `PagedAttentionOpt` / `SDPARef` /
  `GatedDeltaNetRef` / `GatherMatmulImpl` 完全一致（全部 `struct
  XxxOpt|Ref|Impl : public ImplementationManager`）。
- 命名空间 `ov::intel_gpu::ocl`（与同目录所有 manager 同级）。
- 类名 `FCGGUFOpt` 选择依据：
  - `FC` 前缀对齐姊妹类 `FCCompressedGenerateOpt`（同样作用于
    fully_connected）；
  - `GGUF` 作为格式族标识保持大写缩写，与 `SDPA` / `MoE` / `RoPE`
    缩写惯例一致；
  - `Opt` 后缀表示 optimized static-shape impl（vs `Ref`），
    与 `RopeOpt` / `SDPAOpt` / `PagedAttentionOpt` 一致。
- 文件名 `fc_gguf_opt.{hpp,cpp}`（snake_case 镜像），位于
  `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/`；OCL 源
  `fc_gguf_opt.cl` 放在 `ocl_v2/` 根目录（框架约定）。
- 不沿用 `FCCompressedGGUFOpt` 这一更"完整"的名称：(a) `GGUF` 在本
  插件域内等价于 "compressed weight format"，加 `Compressed` 冗余；
  (b) 名称已 11 字符，再加 4 字符显著超出 `ocl_v2/` 现行长度分布
  （RopeOpt / SDPAOpt / SDPARef 普遍 7–10 字符）。

`validate_impl()` 的 baseline 白名单（PR-GPU 阶段）：

```cpp
static constexpr std::array<element::Type_t, 5> kSupportedBaseline = {
    element::Type_t::gguf_q4_0,
    element::Type_t::gguf_q4_k,
    element::Type_t::gguf_q5_k,
    element::Type_t::gguf_q6_k,
    element::Type_t::gguf_q8_0,
};
```

其他 18 种 GGUF element-type 在 `validate_impl` 中返回 false，并
（可选）通过 fallback path 报错：
`"GGUF type <X> kernel not yet implemented in this release; see PR-GPU follow-ups"`。
element-type 定义本身已在 PR-FE 完整加齐 —— 这是契约。

### 4.2 文件布局

- C++ (`.hpp`/`.cpp` + ImplementationManager + JIT generator)：
  `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/`
- OpenCL kernel (`.cl`)：
  `src/plugins/intel_gpu/src/graph/impls/ocl_v2/`（根目录 —— 框架约定）
  文件名 `fc_gguf_opt.cl`、`fc_gguf_transcode.cl`。

### 4.3 dequant 位置契约（C2 落地）

| 阶段 | 允许的数值表示 |
|------|----------------|
| 进入 `FCGGUFOpt::execute()` 时 | GGUF block 字节，原样 |
| `execute_native_ocl_gemv()` 内寄存器 | `half` / `float`（短暂） |
| `execute_transcode_plus_onednn_woq()` 中间 buffer | `i4`/`u4`/`i8`/`u8` + `f16` scale + 可选 `f16` ZP（**绝不** `f16`/`f32` 权重） |
| 写回 host / 其他节点输入 | 不允许 — GGUF Constant 是 FC 私有 |

### 4.4 capabilities property（PR-GPU 一并交付）

在 GPU plugin properties 新增只读 string list：

```cpp
ov::supported_gguf_types
// PR-GPU 阶段精确等于 ["Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]
```

字符串使用 GGUF 规范的大写格式（`"Q4_K"`、`"IQ4_XS"`），便于
benchmark_app / GenAI 上层枚举。后续增量 PR 扩 kernel 时同步扩此
list；GenAI 应在调用前查询此 property，**不**硬编码列表。

---

## 5. Transformation 不变量

### 5.1 GGUF Constant 不变量（C3 落地）

对任何 `dtype = gguf_*` 的 `op::v0::Constant`，MOC 流程任意阶段**不得**：

1. 改写其 `shape`（含 reshape / squeeze / unsqueeze 插入与移除）。
2. 改写其 `element_type`。
3. 改写其字节内容（含 transpose、permute、ConstantFolding）。
4. 在它与 `FullyConnectedCompressed.input(1)` 之间插入任何节点。

允许的操作：

- 移动 Constant 在 graph 中的位置（不改字节）。
- 给 Constant 加 / 删 rt-info（不改字节）。

### 5.2 必须 patch 的 pass 清单（PR-FE `transformations-guards` subagent 输入）

| # | 文件 | 守卫位置 | 守卫形式 |
|---|------|----------|----------|
| 1 | `src/core/src/op/constant.cpp` | `get_byte_size()`、ctor | block-aware sizing |
| 2 | `src/core/src/op/convert.cpp` | `validate_and_infer_types()` | `if (is_gguf_block(in)) OPENVINO_THROW(...)` |
| 3 | `src/common/transformations/src/transformations/common_optimizations/nop_elimination.cpp` | 所有 reshape/squeeze 处理函数 | 命中 GGUF Constant 即跳过 |
| 4 | `src/common/transformations/src/transformations/transpose_sinking/ts_*.cpp` (全部变体) | `transformation_callback` | 命中 GGUF Constant 即返回 false |
| 5 | `src/common/transformations/src/transformations/op_conversions/convert_fc_to_compressed.cpp` | 入口 pattern | `is_gguf_block(weight.get_element_type())` 时直接 return |
| 6 | `src/plugins/intel_gpu/src/plugin/transformations/convert_fc_to_compressed.cpp` | 同上 | 同上 |
| 7 | `src/common/transformations/src/transformations/convert_precision.cpp` | `fuse_type_to_constant` | GGUF Constant 不做精度转换 |
| 8 | `src/common/transformations/src/transformations/low_precision/mark_dequantization_subgraph.cpp` | 入口 | 命中 GGUF 子图设置 `disable_constant_folding` rt-info 并 return |

每个 patch 必须自带一个 lit-style 单测，输入是构造的 GGUF Constant
+ 触发该 pass 的小图，输出 byte hash 与输入相等。

### 5.3 自动安全 pass 清单（无需 patch，但 PR-FE 必须 e2e 验证）

`group_normalization_fusion`、`random_uniform_fusion`、
`matmul_multiply_fusion`、`fp16_compression`、
`align_mixed_fp32_fp16_types` —— 全部依赖 `is_real()=false`
即可安全跳过。PR-FE `ci-gates` subagent 加 5 个 e2e 测试覆盖。

---

## 6. GenAI 集成契约

### 6.1 入口切换

`openvino.genai/src/cpp/src/utils.cpp::read_model()` 中的 `.gguf`
分支：

```cpp
// 旧路径
if (is_gguf_model(path)) return create_from_gguf(path);

// 新路径（PR-GENAI `genai-entry-switch` subagent）
const bool use_native_fe = utils::env_bool(
    "OPENVINO_GENAI_USE_NATIVE_GGUF_FE", /*default=*/true);
if (is_gguf_model(path)) {
    if (use_native_fe) return core.read_model(path);
    return create_from_gguf(path);  // 回退路径，留 1 个 release cycle 后删
}
```

环境变量 `OPENVINO_GENAI_USE_NATIVE_GGUF_FE`：

- 默认值：`true`（PR-GENAI 合入即默认走新 FE）
- `false` 走旧 in-tree reader（一个 release 周期后删除）

**注意**：PR-GENAI 本身**不**删 in-tree reader，只切入口；删除
动作留给后续增量 PR（避免本期回退路径丢失）。

### 6.2 Tokenizer 输入源切换（PR-GENAI `genai-tokenizer-migrate` subagent）

`gguf_tokenizer.cpp::tokenizer_config_from_meta()`：

- 输入参数从 `GGUFReader&` 改为 `const ov::Model&`。
- 读取路径从 `reader.metadata()` 改为 variadic `model.get_rt_info<...>("gguf", "tokenizer", "ggml", ...)`（见 §3.3 API 约定）。
- key 路径必须与 §3.3 表逐项对齐。

### 6.3 删除 / 保留清单（PR-GENAI 一次性处置）

| 文件 | 处置 |
|------|------|
| `src/cpp/src/gguf_utils/gguf.{hpp,cpp}` | **保留作为回退路径**（`OPENVINO_GENAI_USE_NATIVE_GGUF_FE=0` 时使用）；下一个 release 删除 |
| `src/cpp/src/gguf_utils/gguf_quants.cpp` | 同上，**保留**为回退路径；本期不删 |
| `src/cpp/src/gguf_utils/gguf_modeling.{hpp,cpp}` | 同上，**保留**为回退路径；同时作为 PR-FE qwen3 builder 的 reference 参考 |
| `src/cpp/src/gguf_utils/gguf_tokenizer.{hpp,cpp}` | **保留**，仅改 `tokenizer_config_from_meta()` 输入源（默认走 rt-info，回退路径仍走 in-tree reader） |
| `is_gguf_model()` | **保留**作为"需 tokenizer 后处理"提示 |
| Q6_K WA #2135 | **删除**禁用逻辑（新路径下 Q6_K 由 GPU kernel 原生支持） |
| `samples/cpp/text_generation/gguf_reader.cpp` 等 | 改为 `core.read_model` + `LLMPipeline` 2 行示例 |

**为什么不本期删 in-tree reader**：保留一个 release 周期的回退路径
是给 token 漂移 / 性能回归留逃生通道。删除动作留给后续增量 PR。

---

## 7. CI 测试门

### 7.1 GGUF Constant byte-equality gate（PR-FE `ci-gates` subagent 引入）

```
For each model in GGUF test suite:
  m = core.read_model(model_path)
  hashes_before = { id(c): sha256(bytes(c)) for c in gguf_constants(m) }
  apply_moc_transformations(m, /*cf=*/false)
  hashes_after  = { id(c): sha256(bytes(c)) for c in gguf_constants(m) }
  assert hashes_before == hashes_after
```

测试模型集（本期最小集，全部 qwen3 量化变体）：

- qwen3-0.6B-Q4_K_M.gguf
- qwen3-0.6B-Q5_K_M.gguf
- qwen3-0.6B-Q6_K.gguf
- qwen3-0.6B-Q8_0.gguf
- qwen3-0.6B-Q4_0.gguf

选 0.6B 是因为它体积最小（< 1 GB），适合本地脚本日跑 + commit
仓库 fixture 友好；如有更大的 1.7B / 4B 版本作为 nightly 跑可加。

用于覆盖 §1.1 中其他 18 种 element-type 字节不变量的测试模型
（IQ4_XS / TQ2_0 等）以**合成 Constant** 方式构造，不依赖真实
GGUF 文件 —— 因为本期 GPU kernel 不支持，无法做端到端测试。

### 7.2 Token-equality oracle（PR-GENAI 引入）

仅 qwen3，跑全部 5 种 baseline 量化变体：

```
prompt = "The quick brown fox"
seed = 0
greedy decode, max_new_tokens = 256
expected = llama-cli (llama.cpp) 同模型同参数输出
actual   = OV GenAI（OPENVINO_GENAI_USE_NATIVE_GGUF_FE=1）+ GPU 输出
assert expected == actual  (token id 序列完全一致)
```

5 个组合（qwen3-0.6B 在 Q4_0 / Q4_K_M / Q5_K_M / Q6_K / Q8_0 上）
全部必须 100% 一致；任意一个失败即不接受 PR-GENAI 合入。

### 7.3 Per-format kernel oracle（PR-GPU 引入）

仅 5 种 baseline 格式（Q4_0 / Q4_K / Q5_K / Q6_K / Q8_0）：

```
w_gguf = random_gguf_constant(type, K=4096, N=4096)
w_ref_f16 = ggml_dequantize(w_gguf)  # 用 llama.cpp 的 reference 函数
y_actual = FCGGUFOpt(activation_f16, w_gguf)
y_ref    = matmul_f16(activation_f16, w_ref_f16)
assert allclose(y_actual, y_ref, atol=1e-3, rtol=1e-2)
```

其他 18 种 type 不在本期 oracle 范围；增量 PR 加 kernel 时补对应
oracle。

### 7.4 IR round-trip gate（PR-FE 引入）

`m1 = core.read_model("qwen3-0.6B-Q4_K_M.gguf")`
→ `core.serialize(m1, "/tmp/foo.xml", "/tmp/foo.bin")`
→ `m2 = core.read_model("/tmp/foo.xml")`
→ assert all GGUF Constants in `m2` byte-equal to `m1`.

### 7.5 本地 CI 脚本

PR-FE `ci-gates` subagent 一并交付 `tests/scripts/run_gguf_local.sh`：

```bash
# 用法
bash tests/scripts/run_gguf_local.sh [stage]
# stage = fe-byte-eq | fe-roundtrip | gpu-kernel-oracle | genai-token-eq | all
```

每个 stage 对应 §7.1–§7.3 的一个 gate。三个 PR 在本地开发时都
以此脚本作为 acceptance 入口；**绿之前禁止合入下游 PR**。

---

## 8. 命名 & 风格约定

由 PR-0 同时产出一份 `STYLE.md`（与本 SPEC 同目录），约束：

- 所有 GGUF C++ 标识符前缀 `gguf_` 或 `Gguf` 驼峰。
- OCL kernel 文件名前缀 `fc_gguf_` 或 `gguf_`。
- rt-info key 全部走顶层段 `"gguf"` 的 variadic 路径（§3.3），**不写** `::` 分隔（OV 现行约定）。
- error message 含 GGUF type 字符串时使用 §4.4 大写形式。
- 不允许在 GGUF 相关代码中使用 `using namespace`（OpenVINO header
  规则）。
- git commit message 必须以 `[gguf][PR-FE]` / `[gguf][PR-GPU]` /
  `[gguf][PR-GENAI]` 前缀标注归属，便于后续抓 diff 与回滚。

---

## 9. 不可逆决策记录

以下决策一旦冻结，PR-FE 之后**任何 PR 不允许更改**，更改需回到本 SPEC
重新走人工 review：

| ID | 决策 | 锁定理由 |
|----|------|----------|
| D1 | 23 个 `gguf_*` element-type 一次性加齐（§1.1） | 避免后续 ABI 跳号 |
| D2 | rt-info key 全部走顶层段 `"gguf"` 的 variadic 路径（§3.3），无 `::`、无 `ov_` 前缀 | 跨仓共享，改名等于破 ABI；与 OV 现行约定（`runtime_options` / `intel_cpu_hints_config`）对齐 |
| D3 | `OPENVINO_GENAI_USE_NATIVE_GGUF_FE` 环境变量默认 `true`（§6.1） | PR-GENAI 默认行为切换 |
| D4 | byte-equality gate 是 GGUF 项目的 ground truth（§7.1） | 任何 transformation PR 必须过此门 |
| D5 | dequant 仅允许在 OCL kernel 寄存器 / OneDNN WOQ 域（§4.3） | 硬约束 C1/C2 的工程落地 |
| D6 | **qwen3 是本期唯一支持的架构** | 收敛范围至单一基准；其他架构留增量 PR |
| D7 | **GPU baseline = Q4_0 / Q4_K / Q5_K / Q6_K / Q8_0**（§4.1 白名单） | qwen3 GGUF 常见量化配方全覆盖；其他 18 种 type kernel 留增量 |
| D8 | **本地 commit 不 push 到 remote**（§0.3 C6） | 避免未冻结代码进入公共 history |

---

## 10. 待 review 的开放问题（PR-0 冻结前必须解决）

1. **D2 namespace 顶层段** —— 已在本次 review 收敛：选 variadic 顶层段
   `"gguf"`，**不带** `::`、不带 `ov_` 前缀（§3.3 详细评估，对齐
   `runtime_options` / `intel_cpu_hints_config` / `decoder_type_name` 现行
   约定）。本项可在 PR-0 冻结时直接关闭，不再保留为开放问题。
2. **§3.4 mmap 与 `AlignedBuffer` 生命周期** —— `ov::Model` 持有
   `AlignedBuffer` 的 `shared_ptr` 还是 Constant 各自持有？影响
   是否能 partial-unload。**建议**：`ov::Model` 持有，避免重复 mmap。
3. **§5.2 #4 transpose-sinking 范围** —— 是 patch 所有 `ts_*.cpp`
   还是只在公共基类 `TSGeneralForward::transformation_callback` 做
   一次守卫？**建议**：后者，减少 patch 面，但每个 `ts_*` 单测仍
   单独覆盖。
4. **§6.1 默认值是否 `true`** —— 担心 GenAI 老用户回归；可考虑
   `true` 但加显眼 deprecation warning 引导旧路径使用者切换。
5. **§7.2 token-equality 容忍** —— 是否允许 ≥ 第 N 个 token 后
   出现 1 个 token 漂移？**建议**：第一阶段严格 100% 一致，发现
   不可避免漂移再放宽并 root-cause。
6. **`Q8_K` 是否真要支持**（§1.2 列了但通常是激活类型）—— 决策：
   PR-FE 阶段仅在出现作为权重的真实模型时再开 issue，本期 element-type
   先建好枚举占位。
7. **qwen3 GGUF 0.6B 模型的来源** —— 用 HuggingFace `Qwen/Qwen3-0.6B-GGUF`
   官方仓的哪个 commit？**建议**：在 SPEC 冻结时锁定一个 SHA，避免
   后续上游 reconvert 导致 byte-equality 测试漂移。
8. **本地脚本 `run_gguf_local.sh` 的 GPU 选择** —— 多卡机器上 default
   走 `GPU.0` 还是从环境变量读？**建议**：`OV_GGUF_TEST_DEVICE` 环境
   变量，未设时默认 `GPU.0`。
