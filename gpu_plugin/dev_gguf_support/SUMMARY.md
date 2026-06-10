# OpenVINO GPU 插件支持 GGUF 权重压缩格式 — 设计总结

**范围**：在 OpenVINO GPU 插件中原生支持 `llama.cpp` GGUF 的全部低比特权重族
（Type-0/1、K-quants、IQ-quants、Ternary），复用现有的
`fully_connected` / `FullyConnectedCompressed` 原语，并通过**单一**
GGUF `ImplementationManager` 在内部分派给 OCL 内核（memory-bound 的 decode）
或"格式转换 + OneDNN 直建"路径（compute-bound 的 prefill）。

**日期**：2026-06-08
**分支**：`river/enhance_kernel_scheduler`

---

## 1. 总体设计原则

1. **一个原语，一个 GGUF impl 类。** 复用 `cldnn::fully_connected` 并设置
   `compressed_weights = true`。新增**唯一**一个 manager
   `ocl::FCGGUFOpt`，在
   [fully_connected_impls.cpp](src/plugins/intel_gpu/src/graph/registry/fully_connected_impls.cpp)
   中注册一次。具体格式（Q4_0 / Q4_K / IQ4_XS …）通过 `weights.data_type`
   读取，并在 impl 内部以 JIT 常量进行分支选择 —— 不为每种格式
   创建独立 manager。
2. **registry 层不按 M 做分流。** `FCGGUFOpt::support_shapes()`
   接受**任意 M**（decode、prefill、混合都通过）。short-token vs long-token
   只是 impl 的*内部*执行策略，对 scheduler 不可见，避免多 impl 池
   组合爆炸。
3. **单个 impl 内的两条执行路径**：
   - **`M <= M_MEM_BOUND_THRESHOLD`（默认 8）** → 自研 OCL GEMV 内核，
     直接从 HBM 流式读 GGUF block，在寄存器内 dequant，目标 ≥ 90%
     带宽 roofline。结构上等价于现有的
     [fc_compressed_generate_opt](src/plugins/intel_gpu/src/graph/impls/ocl_v2/gemm/fc_compressed_generate_opt.hpp)
     但泛化到所有 GGUF 格式。文件布局遵循 ocl_v2 框架约定（详见
     §3.7）：C++ 端（`.hpp`/`.cpp`、ImplementationManager、JIT
     generator）放在
     [`src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/)
     子目录下；OpenCL kernel 源文件（`.cl`）**必须**留在
     [`src/plugins/intel_gpu/src/graph/impls/ocl_v2/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/)
     根目录（构建脚本统一从这里扫 `.cl`）。
   - **`M > M_MEM_BOUND_THRESHOLD`** → 先跑一个 **format-transcode**
     的 OCL 小内核，把 GGUF block 重打包成 **OneDNN WOQ 原生消费的、
     精度等价的低比特布局**，绝不退化到 `f16`。映射是**精度保持**的
     （见 §3.3.2 表）：Q4_*/IQ4_* → `u4`/`i4` + `f16` scale (+ 可选 ZP)；
     Q5_*/Q6_*/Q8_*/IQ-mid → `i8` + `f16` scale；
     Q2_K/Q3_K/IQ1–IQ3/TQ\* → `i4` + `f16` scale。然后立刻喂给
     **直接构造的 `dnnl::matmul` 原语**（参考
     [moe_3gemm_swiglu_opt.cpp](src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp)
     里 `onednn_linear::create(...)` 的模式），不再经过
     `onednn::FullyConnectedImplementationManager`。Scratchpad 按 K-tile
     分配且始终保持在低比特域，K-tile 峰值约
     `K_TILE * N * (bits_target/8)` 字节（如 INT4 路径 `K_TILE=256, N=4096`
     约 0.25 MB）。
4. **OV Core 扩展完整的 GGUF element-type 集合**（包括 IQ-quants 与
   ternary，详见 §2）。每个新类型都是不透明的 `block_*` 字节打包
   类型，FC impl 是它**唯一**的消费方。
5. **Q4_K_M / Q5_K_M / Q6_K / IQ\*_M 等不是新原语。** GGUF 中 "*_M" /
   "*_S" 后缀只是**文件级的 mixed-precision 配方**：在同一个模型里
   按张量分配不同的 K-quant 变体（如 `attn_v`/`ffn_down` → Q6_K，
   其余 → Q4_K）。每个 FC 的 `element::Type` 天然区分 —— GPU 插件
   无需额外改动即可支持，详见 §3.6 的 mixed-format 设计。

这把过去的"双 impl 家族 / 按 M 分流"设计折叠成一个 registry 条目，
同时保留运行时选择最优执行策略的能力。

---

## 2. 需要支持的 GGUF 量化类型

GGUF 张量是一组定长 **block** 的扁平数组，每个 block 是自描述结构
（自带 scale、zero-point、打包权重），所以 OV `Constant` 只需要保存
一段不透明字节缓冲和总元素数。下表数据来自
[ggml-quants.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.h)。

### 2.1 Type-0 / Type-1 家族（单层 scale，32 元素/block）

| GGUF 类型 | bits/wt | block 元素 | block 字节 | 每 block 元数据                       |
|-----------|---------|------------|------------|---------------------------------------|
| `Q4_0`    | 4       | 32         | 18         | `f16 d`                               |
| `Q4_1`    | 4       | 32         | 20         | `f16 d, f16 m`                        |
| `Q5_0`    | 5       | 32         | 22         | `f16 d, u32 qh`                       |
| `Q5_1`    | 5       | 32         | 24         | `f16 d, f16 m, u32 qh`                |
| `Q8_0`    | 8       | 32         | 34         | `f16 d`                               |
| `Q8_1`    | 8       | 32         | 36         | `f16 d, f16 s`（s 用于点积辅助）       |

### 2.2 K-quant 家族（256 元素 super-block，嵌套 6-/8-bit 子 scale）

| GGUF 类型 | bits/wt | block 元素 | block 字节 | 结构概要                                                                  |
|-----------|---------|------------|------------|---------------------------------------------------------------------------|
| `Q2_K`    | 2       | 256        | 84         | `u8 scales[16]`（4b sub-scale + 4b sub-min）、`f16 d, f16 dmin, u8 qs[64]` |
| `Q3_K`    | 3       | 256        | 110        | `u8 hmask[32], u8 qs[64], u8 scales[12], f16 d`                            |
| `Q4_K`    | 4       | 256 (8×32) | 144        | `f16 d, f16 dmin, u8 scales[12]`（6b s + 6b m）、`u8 qs[128]`              |
| `Q5_K`    | 5       | 256        | 176        | Q4_K 布局 + `u8 qh[32]`（额外比特平面）                                    |
| `Q6_K`    | 6       | 256 (16×16)| 210        | `u8 ql[128], u8 qh[64], i8 scales[16], f16 d`                              |
| `Q8_K`    | 8       | 256        | 292        | `f32 d, i8 qs[256], i16 bsums[16]`（通常作*激活*类型，列此以求完整）        |

### 2.3 IQ-quant 家族（基于 codebook / importance-weighted）

使用 ROM 查找表（16 或 256 项）编码每个 sub-block。内核需要把表通过
JIT 注入到 `__constant` 段。

| GGUF 类型 | bits/wt | block 元素 | block 字节 | 备注                                       |
|-----------|---------|------------|------------|--------------------------------------------|
| `IQ1_S`   | 1.5625  | 256        | 50         | 1-bit + 4-bit sign-codebook 查找            |
| `IQ1_M`   | 1.75    | 256        | 56         | IQ1_S + 更精细的 scale                      |
| `IQ2_XXS` | 2.0625  | 256        | 66         | 256-项 codebook                            |
| `IQ2_XS`  | 2.3125  | 256        | 74         | 512-项 codebook                            |
| `IQ2_S`   | 2.5     | 256        | 82         | 更大 codebook                              |
| `IQ3_XXS` | 3.0625  | 256        | 98         | 256-项 codebook                            |
| `IQ3_S`   | 3.4375  | 256        | 110        | 512-项 codebook                            |
| `IQ4_NL`  | 4.5     | 32         | 18         | 非线性 16-项 codebook（无 super-block）     |
| `IQ4_XS`  | 4.25    | 256        | 136        | 非线性 codebook + 6-bit 子 scale            |

### 2.4 Ternary（BitNet b1.58 风格）

| GGUF 类型 | bits/wt | block 元素 | block 字节 | 备注                                |
|-----------|---------|------------|------------|-------------------------------------|
| `TQ1_0`   | 1.6875  | 256        | 54         | Ternary {-1, 0, +1}，每字节 5 元素   |
| `TQ2_0`   | 2.0625  | 256        | 66         | Ternary 每字节 4 元素                |

### 2.5 OV Core 上对应的改动

上表每种类型都需要在
[src/core/src/type/element_type.cpp](src/core/src/type/element_type.cpp)
新增一个 `Type_t` 枚举值与对应的 `TypeInfo` 行，并在 `element::Type`
上提供三个访问器：

```cpp
size_t block_byte_size()   const;   // 例如 gguf_q4_k 为 144
size_t block_elem_count()  const;   // 例如 gguf_q4_k 为 256
bool   is_gguf_block()     const;   // 整个族都返回 true
```

`op::v0::Constant` 的存储大小是
`ceil_div(N, block_elem_count) * block_byte_size` 字节 —— 与 `nf4`
及其他 sub-byte 类型当前的处理方式完全一致。

`Convert` / `ConstantFolding` 对 GGUF block 类型必须显式禁用
（参考 [src/core/src/op/convert.cpp](src/core/src/op/convert.cpp#L37)
里 `nf4` 的 carve-out）。

---

## 3. 单一 GGUF Impl 类 — `ocl::FCGGUFOpt`

### 3.1 为什么一个 manager 就够

Q4_0 / Q4_K / IQ2_S / TQ2_0 等之间的差异**完全收敛**在"把 1 个 block
解码成 32 / 256 个 `f16` 值"这一段内层循环里。其余所有工作
—— work-group 形状、dispatch 几何、scratchpad 分配、fusion 处理、
OneDNN 衔接 —— 都是**格式无关**的。用 JIT 驱动内层循环的单一 manager
正好契合这一结构，registry 表保持一行：

> **命名 / 类别**（对齐 `ocl_v2/` 现行约定）：声明形式为
> `struct FCGGUFOpt : public ImplementationManager`，命名空间
> `ov::intel_gpu::ocl`，与姊妹 manager `FCCompressedGenerateOpt` /
> `RopeOpt` / `SDPAOpt` / `PagedAttentionOpt` 完全同构。`FC` 前缀对
> 齐 fully_connected 姊妹类，`GGUF` 大写缩写与 `SDPA` / `MoE` /
> `RoPE` 一致，`Opt` 后缀表示 optimized static-shape impl。文件
> `fc_gguf_opt.{hpp,cpp}` 置于 `ocl_v2/gguf/`；OCL 源 `fc_gguf_opt.cl`
> 按框架约定放在 `ocl_v2/` 根目录。详见 [SPEC §4.1](SPEC.md)。

```cpp
// fully_connected_impls.cpp
OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::FullyConnectedImplementationManager, static_shape)
OV_GPU_CREATE_INSTANCE_OCL  (ocl::FCCompressedGenerateOpt,                  static_shape)  // 现有 W4A16/W4A8
OV_GPU_CREATE_INSTANCE_OCL  (ocl::FCGGUFOpt,                                static_shape)  // 新增 — 覆盖所有 GGUF 格式
OV_GPU_GET_INSTANCE_OCL     (fully_connected,                               static_shape)
OV_GPU_GET_INSTANCE_OCL     (fully_connected,                               dynamic_shape, …)
```

### 3.2 校验入口

```cpp
bool validate_impl(const program_node& node) const override {
    const auto& desc = *node.get_kernel_impl_params()->typed_desc<fully_connected>();
    if (!desc.compressed_weights) return false;
    const auto& w_dt = node.get_input_layout(1).data_type;
    return is_gguf_block(w_dt);             // 覆盖 §2 的全部类型
}

bool support_shapes(const kernel_impl_params& params) const override {
    // 不按 M 卡门 —— 任何静态已知形状都接受。
    // small-M vs large-M 分流在 execute() 内部完成。
    const auto& in0 = params.get_input_layout(0);
    if (in0.is_dynamic()) return false;
    return true;
}

bool raw_sub_byte_weight_compatible() const noexcept override { return true; }
```

### 3.3 内部分派 — memory-bound vs compute-bound

```text
                        FCGGUFOpt::execute(params)
                                  │
                M = activation_shape[rank-2]
                                  │
            ┌─────────────────────┴─────────────────────┐
            │                                           │
   M <= M_MEM_BOUND_THRESHOLD                M > M_MEM_BOUND_THRESHOLD
   (decode / 短 prompt)                      (prefill / 长 prompt)
            │                                           │
            ▼                                           ▼
  ┌──────────────────────────────┐         ┌────────────────────────────────────┐
  │ 自研 OCL GEMV 内核            │         │ Stage A：GGUF → OneDNN 原生         │
  │   - 直接从 HBM 流式读 block   │         │   低比特 transcode OCL 内核         │
  │   - 按格式 JIT dequant        │         │   (Q4_* → u4/i4+scale,             │
  │   - N 并行，SG=16             │         │    Q5_*/Q6_*/Q8_* → i8+scale,      │
  │   - 目标 ≥ 90 % BW roofline  │         │    Q2_K/Q3_K/IQ\*/TQ\* → i4+scale) │
  │                              │         │   per-K-tile，双缓冲                │
  │                              │         │ Stage B：直接构造的 dnnl::matmul   │
  │                              │         │   消费 transcoded INT4/INT8 tile    │
  │                              │         │   (DPAS/XMX，compute-bound)         │
  └──────────────────────────────┘         │ 目标 ≥ 75 % FLOPs roofline          │
                                           └────────────────────────────────────┘
```

伪代码：

```cpp
event::ptr FCGGUFOpt::execute(stream& s, primitive_inst& inst) {
    const size_t M = derive_m(inst);
    if (M <= m_mem_bound_threshold) {
        return execute_native_ocl_gemv(s, inst);          // §3.3.1
    }
    return execute_transcode_plus_onednn_woq(s, inst);    // §3.3.2
}
```

#### 3.3.1 Native OCL GEMV（memory-bound）

- 结构模仿现有的
  [`fc_compressed_generate_opt.cl`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/gemm/)。
- 在 JIT 编译期由 `GGUF_TYPE` 常量挑选
  `gguf_decode_<TYPE>(__global const uchar* w_block, …, half* out32)`
  内联函数。
- OpenCL kernel 源文件 `fc_gguf_opt.cl` 放在
  [`src/plugins/intel_gpu/src/graph/impls/ocl_v2/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/)
  根目录（框架约定，见 §3.7），通过 `#define GGUF_TYPE_*` 选择
  内层解码体；IQ-quants 的 codebook 表以
  `__constant uchar TABLE[…]` 的形式随 JIT 一起注入。

#### 3.3.2 Format-transcode + 直建 OneDNN（compute-bound）

**核心原则**：transcode 内核**绝不解码到浮点**，而是把权重重新发射成
**保留 GGUF block 精度等级的、OneDNN 原生量化布局中位宽最小的那一档**。
matmul 始终走 OneDNN 的 WOQ（weight-only-quantized）路径 ——
WOQ 本身对带宽友好的同时又是 compute-bound，比 `f16 × f16` GEMM 既能
省 4–8 倍 scratchpad 又能用满 DPAS 量化吞吐。

**GGUF → OneDNN 目标格式映射表**（`bits_target` 是 OneDNN 权重位宽；
scale/ZP 精度为 `f16`，匹配当前 OneDNN GPU WOQ 接受的范围）：

| GGUF 源                                                | OneDNN 目标权重    | Scale            | Zero point                   | 备注                                                       |
|--------------------------------------------------------|--------------------|------------------|------------------------------|------------------------------------------------------------|
| `Q4_0`                                                 | `i4`（对称）       | `f16`，group=32  | 无                           | nibble 直接拷贝 + scale 规范化。                            |
| `Q4_1`                                                 | `u4`               | `f16`，group=32  | `f16`，group=32              | `m` 转为 per-group ZP。                                     |
| `Q4_K`                                                 | `u4`               | `f16`，group=32  | `f16`，group=32              | 把 super-scale `d` 预乘进 8 个 sub-scale；`dmin` 同样处理。 |
| `Q5_0` / `Q5_1`                                        | `i8` / `u8`        | `f16`，group=32  | 无 / `f16` group=32          | 4 + 1 bit 拼成 5-bit，sign/zero 扩展到 8-bit。             |
| `Q5_K`                                                 | `u8`               | `f16`，group=32  | `f16`，group=32              | 4 + 1 bit → 5-bit → 零扩展到 8-bit；sub-scale × `d`/`dmin`。|
| `Q6_K`                                                 | `i8`               | `f16`，group=16  | 无                           | 4 + 2 bit → 6-bit → sign 扩展到 8-bit；`scales[16] * d`。   |
| `Q8_0` / `Q8_1`                                        | `i8`               | `f16`，group=32  | 无 / `f16` group=32          | 字节直拷；`Q8_1` 的 `s` transcode 后丢弃。                  |
| `Q8_K`                                                 | `i8`               | `f32→f16`，group=256 | 无                       | 激活类型，仅在它作为权重出现时支持。                         |
| `Q2_K` / `Q3_K`                                        | `i4`               | `f16`，group=16 或 32 | 可选                    | 2-/3-bit 权重 sign 扩展到 4-bit，精度等级匹配。              |
| `IQ1_S` / `IQ1_M`                                      | `i4`               | `f16`，group=16  | 无                           | 在 transcode 内做 codebook 查找，输出 i4。                   |
| `IQ2_XXS` / `IQ2_XS` / `IQ2_S` / `IQ3_XXS` / `IQ3_S`   | `i4`               | `f16`，group=16 或 32 | 无                      | codebook 查找 → 4-bit 重量化。INT4 是 OneDNN GPU WOQ 接受的最低档。 |
| `IQ4_NL` / `IQ4_XS`                                    | `i4`               | `f16`，group=32  | 无                           | codebook value → 4-bit index 重映射 + per-group scale。     |
| `TQ1_0` / `TQ2_0`                                      | `i4`               | `f16`，group=32  | 无                           | Ternary {-1, 0, +1} sign 扩展到 i4。                        |

为什么选这些档而不是 `f16`：

- `Q4_*`、K-quants、以及 ≤ 4.5 bpw 的 IQ-quants 在预乘嵌套 K-quant
  scale 之后都可以**无损放入** `i4`/`u4` —— 结果 per-group scale
  承载了 GGUF super-scale × sub-scale 的全部精度。
- `Q5_*`/`Q6_*`/`Q8_*` 需要 `i8`：5/6/8 等效位无法塞入 4 比特，
  只升一档而不是一路退到 `f16`。
- IQ-quant 在 GGUF 源里 codebook 项虽然是浮点，但每个 group 至多 16
  个不同取值 —— 用 per-group 一个 `f16` scale 重量化到 `i4` 与
  原 codebook 查找在等效精度上一致。

**内核机制**：

- OpenCL kernel 源文件按框架约定放在
  [`src/plugins/intel_gpu/src/graph/impls/ocl_v2/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/)
  根目录：`fc_gguf_transcode_to_int4.cl`、`fc_gguf_transcode_to_int8.cl`
  （按 OneDNN 目标档分文件；GGUF 源格式通过 JIT `GGUF_TYPE_*` 选择）。
- 对每个 `[K_TILE × N_TILE]` tile：
  1. 每个 work-item 处理一个输出元素（INT4 路径则每两个 nibble 一组）。
  2. 在寄存器内完成 per-format 解包 + scale 折叠
     （含 K-quant 嵌套 scale 的乘法）。
  3. 把打包后的 `i4`/`i8` 写入权重 scratchpad，把对应的 `f16`
     per-group scale（与可选 `f16` ZP）写入并行的 scale scratchpad。
- 两个 scratchpad 都登记在 `intermediates_memory` 中（由
  `primitive_inst` 统一管理），每 K-tile 占用：
  - INT4 路径：`K_TILE * N / 2` 权重字节 + `(K_TILE / group) * N * 2` scale 字节。
  - INT8 路径：`K_TILE * N` 权重字节 + `(K_TILE / group) * N * 2` scale 字节。
- Tile 尺寸（`K_TILE`、`N_TILE`）来自一张小型 per-arch 表；默认
  `K_TILE = 256`（正好一个 K-quant super-block 一个 work-group）、
  BMG / LNL 上 `N_TILE = 128`。

#### 3.3.3 直接构造 `dnnl::matmul` 原语（不走 FullyConnectedImplementationManager）

参考
[moe_3gemm_swiglu_opt.cpp](src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp)
里 `_shared_up_proj = std::make_shared<onednn_linear>(onednn_linear::create(...))`
与 `dnnl::matmul` + `dnnl::matmul::primitive_desc` 的直建模式，
`FCGGUFOpt` 内部**自管理一组 OneDNN 原语**，不复用顶层
`onednn::FullyConnectedImplementationManager`：

```cpp
struct gguf_onednn_kernel {
    dnnl::matmul                  prim;
    dnnl::matmul::primitive_desc  pd;
    dnnl::memory::desc            src_md;
    dnnl::memory::desc            wei_md;        // i4 / u4 / i8（按 §3.3.2 映射）
    dnnl::memory::desc            scale_md;      // f16，与 weight 平行
    dnnl::memory::desc            zp_md;         // f16，可选
    dnnl::memory::desc            dst_md;
    int                           group_size;
    bool                          has_zp = false;
};

// 缓存键：(GGUF 源类型, M, K_TILE, N)
// 一个 FC node 上若 K_TILE 不变，缓存里通常只有 1–2 项；
// 若同一个模型内多种 GGUF 格式共存（§3.6），每种格式各有自己的条目。
LruCache<std::tuple<element::Type_t, int, int, int>,
         std::shared_ptr<gguf_onednn_kernel>> _onednn_cache{64};
```

构造流程（首次命中或 LRU miss 时执行）：

1. 从 §3.3.2 的 `constexpr` 映射表取出
   `(target_weight_dt, scale_dt, zp_dt, group_size)`。
2. 用 scratchpad 的 layout 直接构造 `wei_md`（不读原 GGUF Constant）、
   `scale_md`、`zp_md`：

   ```cpp
   wei_md   = dnnl::memory::desc({K, N},      target_weight_dt,
                                 dnnl::memory::format_tag::ba);
   scale_md = dnnl::memory::desc({K/group, N}, dnnl::memory::data_type::f16,
                                 dnnl::memory::format_tag::ab);
   if (has_zp) {
       zp_md = dnnl::memory::desc({K/group, N}, dnnl::memory::data_type::f16,
                                  dnnl::memory::format_tag::ab);
   }
   ```
3. 建 `matmul::primitive_desc`，把 weight scale / zp 通过
   `attr.set_scales_mask(...)` / `attr.set_zero_points_mask(...)` 挂上，
   group axis = 0。
4. `pd` → `dnnl::matmul prim(pd)` 一次性创建，存入 LRU 缓存。

每次 `execute_transcode_plus_onednn_woq` 调用：

1. 取得（或惰性创建）`gguf_onednn_kernel` 实例。
2. 提交下一个 K-tile 的 transcode 内核。
3. 用 scratchpad 内存绑定 `wei_md`/`scale_md`/`zp_md`，
   提交 `prim.execute(dnn_stream, {src, wei, scale, zp, dst})`
   消费上一个 K-tile。
4. 双缓冲让 transcode 与 matmul 重叠。

这样设计的好处：

- **不需要修改 `onednn::FullyConnectedImplementationManager`**。后者的
  `validate_impl` 只需早退拒绝 `is_gguf_block()` 即可（§3.5）。
- **不绕 ImplementationManager 的副作用契约**。
  ImplementationManager 在创建顶层 impl 时会读 weight Constant、做
  re-layout、注册 weights cache 等，对 GGUF 这种"OneDNN 看到的权重是
  scratchpad 里临时产物"的语义并不适用。直建 `dnnl::matmul` 把
  这条边界关在 `FCGGUFOpt` 内部。
- **天然支持 §3.6 的 mixed-format**。LRU key 含 `element::Type_t`，
  Q4_K 与 Q6_K 等的 OneDNN 原语自然分槽缓存。

### 3.4 多 impl 池交互

由于 `FCGGUFOpt::support_shapes()` 接受所有静态形状，GGUF 类型 FC
节点的多 impl 池中只有**唯一一个条目** `FCGGUFOpt` 本身。它在
内部按调用决定最优执行路径。

GGUF FC 节点**没有任何 fallback impl**。当 `weight.data_type` 是
GGUF block 类型时，参考 `fully_connected` OCL impl 与顶层 OneDNN impl
都不会通过校验 —— 它们不知道如何读这些格式，硬选只会算错或崩。
详见 §3.5 hard-fail 策略。

不需要单独的"decode" / "prefill" manager，不需要 scheduler 在
prefill→generate 切换点调用 `switch_impl_to()`，也没有
`impl_types::onednn` ↔ `impl_types::ocl` 抖动风险。GGUF impl 对池
而言是单体。

OneDNN **不**作为独立顶层 impl 注册给 GGUF FC：它在
`FCGGUFOpt::execute_transcode_plus_onednn_woq` 内被直建消费。这绕开
了 weight-IO 契约 —— OneDNN 从不接触原始 GGUF 字节。

### 3.5 无 fallback — 找不到合适 impl 就 hard-fail

GGUF 支持是**按 element type opt-in** 的。契约如下：

> 当 `weight.data_type.is_gguf_block()` 为真，且没有任何 GGUF-capable
> impl 对该节点 + 形状 + 融合算子组合通过校验时，GPU 插件**必须**
> 在 graph compile 阶段抛 `OPENVINO_NOT_IMPLEMENTED`。**禁止**
> 静默选用参考 FC kernel 或顶层 OneDNN impl；**禁止**把 Constant
> 静默 dequant 成 `f16`/`u4` 之类作为"温柔"兜底。

理由：

- 参考 `ocl::fully_connected_impl` 没有 GGUF 入口，被选中要么 dispatch
  时崩，要么直接读出垃圾。
- 透明地"dequant 到 f16 Constant"会把 Llama 级模型权重内存放大
  4–8 倍 —— 用户使用 GGUF 想要的内存收益反成 regression，而且没有
  任何信号。
- Fail-fast 让不支持的组合在加载阶段就暴露，而不是第一次 `infer()`
  时才报错；同时给用户一个可操作的错误（指出具体的 FC + GGUF 类型
  + 形状/融合组合）。

强制点：

- [`onednn::FullyConnectedImplementationManager::validate_impl`](src/plugins/intel_gpu/src/graph/impls/onednn/fully_connected_onednn.hpp)：
  `weight.data_type.is_gguf_block()` 时返回 `false`。
- `ocl::fully_connected_impl` 选择器
  ([src/plugins/intel_gpu/src/graph/impls/ocl/fully_connected.cpp](src/plugins/intel_gpu/src/graph/impls/ocl/fully_connected.cpp))：
  每个非 GGUF 内核的 `validate()` 都拒绝 GGUF 权重类型。
- registry 解析逻辑
  ([src/plugins/intel_gpu/src/graph/primitive_inst.cpp](src/plugins/intel_gpu/src/graph/primitive_inst.cpp))：
  当 `m_available_impls` 中无一通过校验且节点权重是 GGUF 时，
  抛出包含 primitive id、GGUF 类型、不支持的形状/融合组合的
  `OPENVINO_NOT_IMPLEMENTED`（让用户知道是去掉一个融合 post-op、
  调整形状对齐，还是换支持的 GGUF 类型）。

### 3.6 一个模型混用多种 GGUF 格式

真实的 GGUF 文件**经常**在不同张量上混用不同量化格式，典型例子：

- `Q4_K_M`：大部分 FFN 使用 `Q4_K`，`attn_v.weight`/`ffn_down.weight`
  使用 `Q6_K`，少量 token-embedding 用 `Q8_0` 或 `f16`。
- `Q5_K_M`：`Q5_K` + `Q6_K` + `Q8_0` 混编。
- `IQ4_NL` recipe：主体 `IQ4_NL` + embedding `Q5_K` + 输出层 `Q6_K`。

本设计**原生支持**这一点，关键原因：

1. **粒度在 element type，不在 manager。** 每个 `FullyConnectedCompressed`
   节点独立携带自己的 weight `element::Type`。FE（§4.4/§8）会按
   per-tensor 把对应的 GGUF block 类型写到该节点的 weight Constant
   上 —— 节点 A 是 `gguf_q4_k`，节点 B 同时可以是 `gguf_q6_k`。
2. **`FCGGUFOpt` 是 polymorphic 的。** `validate_impl` 只问
   `is_gguf_block(w_dt)` 是否为真；`execute` 内部按 `w_dt` 在
   §3.3.2 映射表上查目标低比特档与 transcode kernel；JIT key 包含
   `element::Type_t`，所以同一类 `FCGGUFOpt` 对每种 GGUF 格式各自
   编译一个 OCL kernel 变体并缓存（与 `FCCompressedGenerateOpt`
   现状一致）。
3. **§3.3.3 的 OneDNN 缓存按格式分槽。** LRU key 第一维就是
   `element::Type_t`，所以 Q4_K 节点的 `dnnl::matmul` 与 Q6_K
   节点的 `dnnl::matmul` 不会互相覆盖，也不会因为一个节点的格式
   触发另一个节点的 OneDNN pd 重建。
4. **Scratchpad 复用按目标档而非源档。** 由于映射表把整族 4-bit
   源都压到 `i4`/`u4`、5/6/8-bit 源压到 `i8`，scratchpad 在
   `primitive_inst::allocate_internal_buffers()` 中按 INT4 / INT8
   两类大小分配；不同 GGUF 源对应的 transcode kernel 写入同一类
   scratchpad，调度与内存预算保持稳定。
5. **不需要任何 "mixed-recipe pass"。** GGUF 配方信息止步于 FE：FE
   按 GGUF 张量表给每个节点指定正确的 `element::Type`，GPU 插件
   只看 element type，完全感知不到"M / S / XS" 等 recipe 标签。

可观察的工程约束：

- FE 需要确保**同一份权重**只对应一个 `element::Type`（不要把
  Q4_K 的张量错标成 Q4_0）。这是 §8.3 F1 / F6 的需求。
- 由于每种 GGUF 类型在首次出现时会触发 OCL JIT 编译 + OneDNN pd
  build，一次 compile_model 在 mixed-recipe 模型上**首次**会有
  3–6 次额外 build；可通过把 transcode kernel 与 OneDNN pd 缓存到
  model cache 来缓解（与现有 kernel cache 机制对齐）。
- `support_shapes` 的判定必须**逐 element type 独立**：例如某 GPU
  上 OneDNN WOQ 接受 `i4 + f16-scale, group=32` 但不接受
  `i4 + f16-scale + f16-ZP, group=32`，那么 Q4_0 节点可以走
  compute-bound 分支，但同模型里的 Q4_1 / Q4_K 节点必须按 §3.5
  hard-fail —— *不要*为求"模型整体能跑"而把 Q4_1 退化到 GEMV
  路径，否则会出现长 prompt 上某些层性能莫名跌一档的隐藏行为。
### 3.7 文件布局约定（`ocl_v2` 框架要求）

`ocl_v2` 后端的构建脚本统一从
[`src/plugins/intel_gpu/src/graph/impls/ocl_v2/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/)
**根目录** 扫 OpenCL kernel 源文件（`*.cl`），同时 C++ 实现可以
按后端分推到子目录（参考现有的 [`moe/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/)、[`pa_sdpa/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/pa_sdpa/)、[`sdpa/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/)、
[`gemm/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/gemm/) 等）。GGUF 后端遵循同样的划分：

| 文件类型 | 位置 | 说明 |
|----------|------|------|
| C++ 实现（`*.hpp`/`*.cpp`） | [`src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/) | `fc_gguf_opt.hpp`、`fc_gguf_opt.cpp`、ImplementationManager、JIT generator、GGUF → OneDNN 映射表都集中在这里。 |
| OpenCL kernel 源（`*.cl`） | [`src/plugins/intel_gpu/src/graph/impls/ocl_v2/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/) **根** | `fc_gguf_opt.cl`、`fc_gguf_decoders.cl`、`fc_gguf_transcode_to_int4.cl`、`fc_gguf_transcode_to_int8.cl` —— “.cl” 只能在这个根位置，否则会不被 JIT 加载。 |
| CMake 集成 | [`src/plugins/intel_gpu/src/graph/impls/ocl_v2/CMakeLists.txt`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/CMakeLists.txt) + 子目录自己的 `CMakeLists.txt` | 根目录负责扫 `*.cl` 并 `add_subdirectory(gguf)`；子目录 `CMakeLists.txt` 负责 C++ 源文件。 |

这一拆分有两个原因：

1. **`.cl` 必须在根。** `ocl_v2` 的 JIT 加载器在构建期以根目录为起点
   生成嵌入资源。把 `.cl` 放进 `gguf/` 子目录会导致 kernel 在
   运行期查不到，报 "kernel not found" 错误。
2. **C++ 必须在子目录。** 代码业主对顶层 `ocl_v2/` 有明确要求：
   不添加新的独立 `.hpp`/`.cpp`，所有新后端都需按业务家族
   （moe / pa_sdpa / sdpa / gemm / gguf …）进子目录，保持顶层目录清爽。
---

## 4. 工作项

### 4.1 OV Core（Phase 0）

- 为 §2 中**每一种**类型新增 `Type_t` 项 + `TypeInfo` 行。
- 在 `element::Type` 上添加 `block_byte_size()`、`block_elem_count()`、
  `is_gguf_block()` 访问器。
- 更新 `Constant::create()` 与 `Constant::get_byte_size()`，在
  `is_gguf_block()` 时按 block 计算尺寸。
- 在 `Convert` 中加入 carve-out（GGUF block 类型禁止做任何 convert）。
- FE 层给 GGUF Constant 打上 `disable_constant_folding` rt-info。

### 4.2 GPU 插件（Phase 1 — 只实现 decode 路径）

- 文件布局严格遵循 §3.7的框架约定：
  - C++ 端集中在
    [`src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/)
    子目录：
    - `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/fc_gguf_opt.hpp`
    - `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/fc_gguf_opt.cpp`
    - `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/CMakeLists.txt`
  - OpenCL kernel 源放在
    [`src/plugins/intel_gpu/src/graph/impls/ocl_v2/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/)
    **根目录**（框架要求，否则 JIT 加载不到）：
    - `src/plugins/intel_gpu/src/graph/impls/ocl_v2/fc_gguf_opt.cl`
    - `src/plugins/intel_gpu/src/graph/impls/ocl_v2/fc_gguf_decoders.cl`
      （header-only `.cl`，提供 `gguf_decode_<TYPE>()` 内联函数）。
- 更新 `src/plugins/intel_gpu/src/graph/impls/ocl_v2/CMakeLists.txt`：
  把新增的 `.cl` 加入资源扫描列表；加上 `add_subdirectory(gguf)`
  以集成 C++ 实现。
- 在
  [src/plugins/intel_gpu/src/graph/registry/fully_connected_impls.cpp](src/plugins/intel_gpu/src/graph/registry/fully_connected_impls.cpp)
  注册 `FCGGUFOpt`。
- 给
  [`onednn::FullyConnectedImplementationManager::validate_impl`](src/plugins/intel_gpu/src/graph/impls/onednn/fully_connected_onednn.hpp)
  加 early-out：`weight.data_type.is_gguf_block()` 时直接返回 false ——
  保证 `is_gguf` 只由新 manager 处理。
- 同样的 early-out 加到所有非 GGUF 的 FC validator：
  `src/plugins/intel_gpu/src/graph/impls/ocl/fully_connected.cpp`
  与 `src/plugins/intel_gpu/src/kernel_selector/kernels/fully_connected/*`，
  确保缺 GGUF impl 时按 §3.5 抛 `OPENVINO_NOT_IMPLEMENTED`，而不是
  被错误选中。
- 单元测试：对 `M ∈ {1, 4}` 与 §2 每种类型，跟 `llama.cpp` 参考
  dequant + `f16` matmul 比对精度。
- **Mixed-format 单测**：构造一个 6 节点的 mini-model，其中
  3 个节点用 Q4_K、2 个用 Q6_K、1 个用 Q8_0，验证 6 个节点全部
  通过 `FCGGUFOpt` 的 `validate_impl`，且 OCL JIT 缓存里出现 3 个
  独立变体（每个 element type 一个）。

### 4.3 GPU 插件（Phase 2 — compute-bound 分支）

- 新增 transcode 内核，**OpenCL 源**同样放在
  [`src/plugins/intel_gpu/src/graph/impls/ocl_v2/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/)
  根目录（框架要求）：
  - `fc_gguf_transcode_to_int4.cl` —— tile 粒度 GGUF →
    `i4`/`u4` + `f16`-scale（+ 可选 ZP）。覆盖
    Q4_\*、Q4_K、Q2_K、Q3_K、IQ1\*、IQ2\*、IQ3\*、IQ4\*、TQ\*。
  - `fc_gguf_transcode_to_int8.cl` —— tile 粒度 GGUF → `i8` +
    `f16`-scale。覆盖 Q5_0/Q5_1、Q5_K、Q6_K、Q8_0/Q8_1、Q8_K。
- transcode 对应的 C++ Stage 类与 JIT generator 放在
  [`src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/`](src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/)
  子目录（例如 `fc_gguf_transcode_stage.hpp/cpp`）。
- §3.3.2 映射表以 `constexpr` 形式落在
  `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/fc_gguf_opt.cpp`，
  按 `element::Type_t` 查表得到 transcode kernel + OneDNN 目标描述符。
- **直建 OneDNN 原语**（按 §3.3.3 的模式，参考
  [moe_3gemm_swiglu_opt.cpp](src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp)
  的 `onednn_linear::create(...)` / `dnnl::matmul::primitive_desc`
  写法）：
  - 不调用 `onednn::FullyConnectedImplementationManager::create_impl()`。
  - 用 scratchpad 的 layout 直接构造 `wei_md`/`scale_md`/`zp_md`，
    通过 `dnnl::primitive_attr` 挂 scale/ZP mask。
  - LRU 缓存键为 `(element::Type_t, M, K_TILE, N)`；同一 FC 节点上
    K_TILE 不变时一般只有 1–2 项；mixed-recipe 模型里每种 GGUF
    类型各占一槽。
- scratchpad 生命周期通过
  `primitive_inst::allocate_internal_buffers()` 接入，每个 active
  K-tile 两块 scratchpad（weight + scale）+ 双缓冲。
- 在
  [src/plugins/intel_gpu/include/intel_gpu/runtime/internal_properties.hpp](src/plugins/intel_gpu/include/intel_gpu/runtime/internal_properties.hpp)
  新增内部属性 `m_mem_bound_threshold`，便于按 workload 调优。
- 如果某 GGUF 类型对应的 `(target_dtype, group, ZP_layout)` 三元组
  OneDNN 无法 build pd，**不允许**回退到 `f16` transcode，也**不允许**
  静默路由到 native GEMV ——按 §3.5 抛 `OPENVINO_NOT_IMPLEMENTED`。
- 单元 + 性能测试覆盖 `M ∈ {32, 128, 1024}`，外加精度门禁：
  transcode + OneDNN 结果与 native OCL GEMV 在同一个 kernel、两个
  M 区间上比对，须在 `f16` 容差内一致。
- **Mixed-format 性能测试**：mini-model 内 Q4_K 与 Q6_K 节点
  同时跑 prefill，确认 OneDNN cache 中两类 pd 共存，且两条路径
  互不污染（无 pd 重建抖动）。

### 4.4 GGUF Frontend（Phase 3，独立工作流）

详见 §8 的 FE 缺口分析。工作项概要：

- 把 OpenVINO.GenAI 中的 GGUF reader
  (`samples/cpp/text_generation/gguf_reader.cpp` 等) 升级为
  `src/frontends/gguf/` 下的真正 OV frontend，让 GGUF 能通过
  `ov::Core::read_model("*.gguf")` 加载，而不只是 GenAI helper。
- 权重 `Constant` 节点直接以 §2 的**新 GGUF `element::Type`** 发射
  —— 不要走 `u4 + f16 scale + Multiply/Subtract` 的 decompose
  子图（当前 GenAI reader 做法）。新元素类型的全部意义就在于
  零拷贝、单 Constant 表达。
- 对 K-quants / IQ-quants 这类自包含格式，
  `FullyConnectedCompressed` 的 `weight_scales` / `weight_zero_points`
  输入留**空**（scale 已经在 weight buffer 内）。
- 给每个 GGUF Constant 标 `disable_constant_folding` rt-info，
  阻止任何 pass 把它实例化到 `f32`。
- 用 `mmap` 把 GGUF 文件映射进 Constant 存储；FE 层**不**拷贝、
  **不**重打包 —— 所有重打包都交给 GPU 插件。
- 支持**per-tensor 混合精度**配方（Q4_K_M、IQ4_NL 等）：FE 必须按
  GGUF 张量表给每个张量发出准确的 `element::Type`，这是 §3.6
  在 GPU 端能 work 的前提。

### 4.5 非 GPU 插件 — 无 fallback

按 §3.5 的 hard-fail 契约，**不**提供 `DecomposeGGUFToInt4`
transformation，也**不**提供任何 plugin 端的 dequant-to-`f16` shim。
没有原生 GGUF 支持的 plugin 在 compile 阶段直接抛
`OPENVINO_NOT_IMPLEMENTED`。这样每个后端对"自己实际支持什么"
保持诚实，避免在不支持新 element types 的后端上把 GGUF Constant
透明展成 `f16` 带来的 4–8 倍内存膨胀。

---

## 5. 性能目标

| 工作负载              | 路径                                                          | Roofline 目标   |
|-----------------------|---------------------------------------------------------------|-----------------|
| `M = 1` decode        | Native OCL GEMV                                               | ≥ 90 % HBM BW   |
| `M = 4` decode        | Native OCL GEMV（N 并行 + K-batch）                            | ≥ 85 % HBM BW   |
| `M = 32` prefill      | Transcode → 直建 OneDNN WOQ INT4/INT8 GEMM                    | ≥ 60 % FLOPs    |
| `M ≥ 128` prefill     | Transcode → 直建 OneDNN WOQ INT4/INT8 GEMM（K-tile 流水）      | ≥ 75 % FLOPs    |

注：prefill roofline 是相对 OneDNN WOQ INT4 / INT8 的等效 FLOPs
（DPAS 量化吞吐）算的，而不是 `f16` FLOPs。能保持在低精度档是这条
路径在 scratchpad 写入仍然是 compute-bound 而非带宽 bound 的关键。

阈值 `M_MEM_BOUND_THRESHOLD` 通过内部属性按 arch 调优；BMG / LNL
XMX 级硬件上的默认值 `8`（对应 bandwidth-bound 与 compute-bound
的拐点）。

---

## 6. 风险与未决问题

| 风险 | 缓解 |
|------|------|
| 分数 bpw（Q4_K = 4.5、IQ4_XS = 4.25 …）破坏 `Type::size()` 计算。 | 所有尺寸查询走新的 `block_byte_size()` / `block_elem_count()`；对 GGUF 类型废弃标量 `bitwidth()`。 |
| IQ-quants 的 codebook 表需要落在 device 上某处。 | 以 JIT `__constant` 数组方式注入 —— 每种类型最多 ~512 项 × 8 B = 4 KB，per-kernel 成本可忽略。 |
| 朴素分配会让 transcode scratchpad 把 GPU 内存峰值翻倍。 | 按 K-tile 而非整矩阵分配，**并**始终保持在低比特档（INT4 / INT8，绝不 `f16`）。`K_TILE = 256` 时 INT4 scratchpad 在 `N = 4096` 上约 0.25 MB、INT8 约 0.5 MB。 |
| OneDNN GPU WOQ 不是每个 arch 都接受全部 `(weight_dtype, scale_group, ZP_layout)` 组合。 | Plugin init 时建一张 per-arch 能力表；某 GGUF 类型若目标描述符被 OneDNN 拒绝，按 §3.5 抛 `OPENVINO_NOT_IMPLEMENTED`。**不**静默退到 `f16` transcode、**不**静默走 native GEMV —— 用户能明确看到 (arch, GGUF 类型, 融合算子) 的不支持组合。 |
| 把 K-quant 嵌套 scale（super × sub）预乘到单一 per-group `f16` scale 可能损精度。 | 必要时用 `bf16` 中间精度做累积；IQ codebook 重量化是唯一存在真实质量损失的步骤 —— 每种类型加精度单测把关。 |
| 滑动窗口 / speculative decoding 下同一 FC 反复在 decode 与 prefill 模式间切。 | 内部分派除一次分支外无 per-call setup；不重新编译 kernel，不切 impl。OCL kernel 与 OneDNN sub-impl 都在 compile 期建好并常驻。 |
| 同一模型混用多种 GGUF 格式时首次 compile 会触发多次 JIT + OneDNN pd build。 | 与现有 kernel cache 对齐，把 transcode kernel 与 OneDNN pd 缓存进 model cache；首次 build 成本一次性付出，之后命中缓存。 |
| GGUF `Q8_K` 主要是*激活*类型。 | 列此求完整；FC 权重侧支持可选 —— 只有真实 GGUF 模型用它做权重时再加。 |

---

## 7. 对原问题的回答

> 是否可以复用 FC primitive，用一个 impl 类支持所有 GGUF 量化？

**可以。** `FullyConnectedCompressed` 已经在图层面表达了一切
（weight + scale + ZP 输入，`compressed_weights` 标志），GGUF 格式
之间唯一的区分维度就是权重 `element::Type`。一个 GPU
`ImplementationManager`（`ocl::FCGGUFOpt`），以
`is_gguf_block(weight.data_type)` 为入口校验，就能覆盖 §2 的整个
低比特族；在**内部**按 M 分派到 OCL GEMV（memory-bound，小 M）
或 transcode-to-INT4/INT8 + 直建 OneDNN WOQ GEMM（compute-bound，
大 M）；并嵌进现有多 impl 池而不需要任何 scheduler 改动。同一模型
混用多种 GGUF 格式的场景由 §3.6 描述，本设计**原生支持**且不需要
额外 pass。

---

## 8. GGUF Frontend 分析

### 8.1 本仓库现状

对 OpenVINO 源码树做 `grep` 可知，今天**没有原生 GGUF frontend**：

- `src/frontends/` 下有 `onnx/`、`tensorflow/`、`tensorflow_lite/`、
  `pytorch/`、`jax/`、`paddle/`、`ir/`、`common/` —— **没有 `gguf/`**。
- in-tree 唯一与 GGUF 相关的代码是
  [src/plugins/intel_cpu/thirdparty/shl/tests/llm/convert/gguf-py/gguf/gguf.py](src/plugins/intel_cpu/thirdparty/shl/tests/llm/convert/gguf-py/gguf/gguf.py)
  —— 这是 SHL llama2 量化测试用的 Python *writer*，与模型加载无关，
  不能用来把 GGUF 读成 `ov::Model`。
- 一个可用的 GGUF reader 存在于**独立**的
  [openvinotoolkit/openvino.genai](https://github.com/openvinotoolkit/openvino.genai)
  仓库（`src/cpp/src/gguf_utils/gguf_reader.cpp` 等），但它只服务
  GenAI pipeline：在 C++ 里直接构 `ov::Model`，从不经过
  `ov::Core::read_model`。

所以 §3 设计**没有上游 FE 能交付带新 GGUF element type 的 Constant**。
FE 工作是 GPU 端收益的关键路径阻塞项。

### 8.2 OpenVINO.GenAI 现有 reader 的问题

现有 GenAI reader 作为起步是合理的，但与 §3 架构有三处结构性不兼容：

1. **FE 时即 dequant。** 它把每个权重以
   `Constant(u4)` + `Convert` + `Multiply`/`Subtract` decompose
   子图发射（靠后续 `FuseFC` pass 折回 `FullyConnectedCompressed`）。
   对 `Q4_0`/`Q8_0` 还能工作，但对 K-quants 和 IQ-quants 这意味着
   FE 端就要**重量化嵌套 scale**、把 **codebook 内联进 IR**，
   既毁掉零拷贝承诺，也抵消 §3.3 里的 per-format 精度工作。
2. **没有注册成 `ov::frontend::FrontEnd`。** 不会被 `FrontEndManager`
   发现，任何非 GenAI 调用方（Benchmark App、accuracy checker、
   用户自定义代码）执行 `read_model("*.gguf")` 都会失败 ——
   CI 覆盖与第三方使用都被卡。
3. **整文件加载进 host 内存。** 没有 `mmap`、没有按需 materialize。
   一个 70B Q4_K_M 模型（~38 GB on disk）开发机直接放不下，也废掉
   GPU 的 USM-host fast path。

### 8.3 §3 设计对 FE 的要求

任何要对接 GPU 插件 GGUF impl 的 FE 都必须满足：

| #   | 要求 | 原因 |
|-----|------|------|
| F1  | 权重 `Constant` 节点的 `element::Type` 必须是 §2 中的新 GGUF 类型 —— 不能是 `u4 + f16-scale + Multiply` 子图。 | §3 设计完全依赖 impl 选择时 `is_gguf_block(weight.data_type)` 为真。 |
| F2  | 用 `mmap` 映射 GGUF 文件，让 Constant 存储指向该 mmap 区（只读共享）。 | 多 GB 权重做到零拷贝、瞬时加载。 |
| F3  | 给每个 GGUF Constant 打 `disable_constant_folding` rt-info。 | 阻止 `ConstantFolding`、`ConvertPrecision` 等 pass 把不透明 block 实例化到 `f32`/`f16`。 |
| F4  | 在 `common_optimizations` 中阻断对 GGUF Constant 的所有 `Convert`/`Reshape`/`Transpose` 重写。 | block 布局非 strided、非按元素寻址，普通张量代数对它不成立。 |
| F5  | 从 GGUF 张量名（`blk.N.attn_q.weight`、`blk.N.ffn_gate.weight` …）与 metadata key（`general.architecture`、`*.attention.head_count`、`*.rope.dimension_count` …）重建 LLM 拓扑。 | GGUF 没有图，只有扁平张量表 + key/value 元数据字典。FE 负责拼出 `Attention`、`FFN`、`RMSNorm`、`RoPE` 等子图。 |
| F6  | 支持**单一模型内 per-tensor 混合 element type**（Q4_K_M = 主体 Q4_K + 部分 Q6_K 输出；IQ4_NL = IQ4_XS + Q5_K embed 等）。 | 真实 GGUF 配方天然异构；一个张量 ≠ 模型级单一 dtype。本设计 §3.6 的工作前提。 |
| F7  | 支持**分片 GGUF 文件**（`model-00001-of-00005.gguf`）。 | llama-cpp 约定 > 50 GB 的文件分片；FE 需要透明拼接张量表。 |
| F8  | 把 GGUF metadata（vocab、tokenizer model、RoPE base/scaling、layer-norm epsilon、sliding-window size）透出，让 GenAI 的 tokenizer / pipeline 能消费。 | GGUF 把 tokenizer + 超参数都内嵌了 —— 这些通常由外部 `tokenizer.json` 承载。 |
| F9  | 支持 GGUF 格式版本 V1、V2、V3（header 布局与对齐不同）。 | 实际野外模型三版都有，不能假定最新。 |
| F10 | 任何 Convert/Cast 都不能以 GGUF 类型为输出。 | GGUF 是 weight-only 编码，激活与中间张量必须留在 `f16`/`bf16`/`f32`。 |

### 8.4 FE 工作流特有的风险

1. **OV Core 依赖顺序。** §4.1 不落地，F1 就无法实现 —— FE 物理上
   发射不出 GPU 想要的 Constant。这让 §4.1 成为 §4.4 的硬阻塞，
   §4.4 又是任何端到端 GGUF 测试的硬阻塞。
2. **图重建（F5）复杂且 arch-aware。** Llama / Mistral / Qwen /
   Falcon / Phi / StableLM … 的 `blk.N.*` 命名都略有差异。FE
   需要一张架构注册表（以 `general.architecture` metadata key
   作 key），每个支持的拓扑配一个 builder。维护压力是持续性的。
3. **GGUF 没有标准 FE 一致性测试。** 现有 FE 测试都在
   `src/frontends/<name>/tests/`，假设有参考框架 runtime（ONNX
   Runtime、TF、PT）做精度比对。GGUF 我们得拿 `llama.cpp` 的
   `main --temp 0 --seed 0` 当参考，这是一个非平凡的 CI 依赖。
4. **许可与再分发。** GGUF 参考向量来自 Llama / Apache / MIT 等
   不同条款下再分发的模型。`tests/test_model_zoo` 加入测试资产
   前需要单独走法务审查。
5. **Tokenizer / GenAI 耦合（F8）。** GGUF 把 tokenizer 内嵌了。
   今天 OV 侧 tokenizer 由独立包 `openvino_tokenizers` 承担。
   FE 必须 (a) 透出 metadata 让 `openvino_tokenizers` 运行时
   构造 tokenizer，或 (b) 预构 tokenizer model 图作为兄弟
   `ov::Model` 挂上。(a) 更可取但需要 `openvino_tokenizers` 上
   一个尚不存在的 API。
6. **与 GenAI reader 的 backward-compat。** 一旦正式 FE 上线，
   OpenVINO.GenAI 的 in-tree reader 必须废弃或转发到新 FE，否则
   同一份 `.gguf` 走两条代码路径产出不同 IR —— 调试灾难。

### 8.5 推荐的 FE rollout 顺序

1. **Phase 3a** — 把 §2 的新 GGUF `element::Type` 落进 OV Core
   （是下面所有项的阻塞前提）。
2. **Phase 3b** — 在 `src/frontends/gguf/` 落一个空 FE，先做
   header 解析 + 张量表扫描，按 F1+F2+F3+F4 发射新类型 Constant，
   不做拓扑重建。验证：加载一个 `.gguf` 权重文件，遍历 Constant，
   核对 `element::Type` 与 mmap 区按字节相等。
3. **Phase 3c** — Llama-family 拓扑 builder（F5 限制在
   `general.architecture == llama`）。验证：Llama-2-7B-Q4_0
   端到端推理在 greedy decode 下与 `llama.cpp` 参考 token 一致。
4. **Phase 3d** — 分片文件（F7）+ V1/V2/V3 版本（F9）+ 混合精度
   （F6 — 这是 §3.6 的 enabler）。
5. **Phase 3e** — 增加其它架构 + tokenizer/metadata 桥
   （F5 拓宽 + F8）。Llama / Mistral / Qwen builder 与
   GenAI reader 持平后即可废弃后者。

在 Phase 3a + 3b 上线前，§4.2 / §4.3 的 GPU 工作可以通过
**用测试 helper 合成 GGUF Constant**（一个小 C++ 函数把
`void*` buffer 包成新类型的 `ov::Tensor`）做单元测试，但无法
在真实模型上做端到端验证。

---

## 9. OpenVINO.GenAI 需要的改动

`/home/ov2022/workspace/openvino.genai` 是独立仓库，目前自带一份
in-tree GGUF reader。下表先盘点现状，再列出新 FE 落地后 GenAI
必须做的迁移。

### 9.1 现状盘点

| 项目 | 现状 | 位置 |
|------|------|------|
| 文件布局 | `src/cpp/src/gguf_utils/` 下：`gguf.{hpp,cpp}`、`gguf_quants.cpp`、`gguf_modeling.{hpp,cpp}`、`gguf_tokenizer.{hpp,cpp}` | `src/cpp/src/gguf_utils/` |
| 触发方式 | `LLMPipeline(models_path)` → `utils::read_model()` → `is_gguf_model()` 按 `.gguf` 扩展名判定 → `create_from_gguf()` | `gguf_tokenizer.cpp:32` |
| 量化类型支持 | Q4_0 / Q4_1 / Q4_K / Q8_0；Q6_K **被禁用**（workaround #2135）；F32/F64/F16/BF16；IQ\* / TQ\* / Q5_\* / Q2_K / Q3_K / Q5_K / Q8_K / Q8_1 **不支持** | `gguf_quants.cpp:210-265` |
| 张量产出形式 | **dequantize 到 `(u32 weights, f16 scales, f16 biases)` 三元组**（decompose-style），下游再装配进 `FullyConnectedCompressed`；不是单一不透明 GGUF Constant | `gguf_quants.cpp` `extract_q4_X_data()` / `extract_q8_0_data()` |
| 加载方式 | **整文件读入 host 内存**；无 mmap、无按需 materialise | `gguf.cpp` |
| 架构支持 | `llama`、`qwen2`、`qwen3`，硬编码 if-else，其他架构 `OPENVINO_THROW` | `gguf_modeling.cpp:123-130` |
| Tokenizer | 从 GGUF metadata 里前缀 `tokenizer.` 的 key 抽取后构造 | `gguf_tokenizer.cpp:36-48` |
| FE 注册 | **没有**注册 `ov::frontend::FrontEnd`；只有 GenAI 内部能消费 | n/a |

### 9.2 与主仓 §3/§4/§8 设计的差距

1. **类型覆盖窄。** 只覆盖 4 种量化类型，缺 Q5_\*、K-quants 大半、
   IQ\* 全部、TQ\* 全部、Q8_1、Q8_K；Q6_K 还因 WA #2135 被禁。本设计
   §2 要求覆盖 GGUF 全族。
2. **decompose 式产出。** 拆成 `(u32, f16-scale, f16-bias)` 等价于
   §8.2 描述的"FE 时即 dequant"。对 K-quants 与 IQ-quants 这意味着
   **嵌套 scale 在 FE 端被预乘掉**、**codebook 被内联进 IR** ——
   零拷贝承诺被破坏，§3.3 的 per-format 精度工作被前置抵消。
3. **没有 mmap。** 70B Q4_K_M (~38 GB) 在 GenAI 现有路径下直接撑爆
   host 内存。
4. **不是 OV FE。** `ov::Core::read_model("*.gguf")` 走不通；
   Benchmark App、accuracy checker、第三方代码都用不上 GGUF 输入。
5. **架构覆盖薄。** 只有 llama / qwen2 / qwen3。新 FE 要在
   `src/frontends/gguf/` 里维护架构注册表（§8.4 风险 2）。
6. **Q6_K WA #2135**：因为现 reader 用 decompose 路径不能无损
   表达 Q6_K 的 6-bit 嵌套 scale 才禁用；本设计的 `i8 + f16 scale,
   group=16` 映射（§3.3.2）天然支持，WA 可同步废止。

### 9.3 GenAI 端必须做的改动汇总

| 文件 / 模块 | 改动 | 阶段 |
|-------------|------|------|
| `src/cpp/src/gguf_utils/gguf.{hpp,cpp}` | **保留**作为 mmap + 头解析 + 张量表扫描 helper；若主仓 FE 直接采用同一份代码，迁移到 `src/frontends/gguf/` 后此处删除。 | Phase 3a 期间共存；Phase 3e 删除。 |
| `src/cpp/src/gguf_utils/gguf_quants.cpp` | **删除** `extract_q4_0_data` / `extract_q4_1_data` / `extract_q4_K_data` / `extract_q8_0_data` 等 dequant 函数；GenAI 不再做量化解包，全部交给 GPU 插件 impl。 | Phase 3b 之后。 |
| `src/cpp/src/gguf_utils/gguf_modeling.{hpp,cpp}` | `create_language_model()` 的构图逻辑要么：(a) 整段**迁移**到 `src/frontends/gguf/` 的架构 builder 里（推荐，与 ONNX/PT FE 对齐）；要么 (b) 暂时保留但调用方式从"接收 (u32, f16-scale, f16-bias) 三元组"改成"接收 mmap'd opaque GGUF Constant"，构造的 `FullyConnectedCompressed` 节点 `weight_scales` / `weight_zero_points` 输入留空。 | Phase 3c：Llama-family 迁移；Phase 3e：Qwen 迁移并废弃此文件。 |
| `src/cpp/src/gguf_utils/gguf_tokenizer.{hpp,cpp}` | `tokenizer_config_from_meta()` 的逻辑**保留**在 GenAI 侧 —— 这是 GenAI tokenizer 包的语义，不属于 FE 职责；只是输入来源从 `GGUFReader` 自带的 metadata dict 改成新 FE 通过 `model->get_rt_info()` 透出的 metadata（对应 §8.3 F8 的 (a) 方案）。 | Phase 3e（同步 F8 API）。 |
| `is_gguf_model()` + `LLMPipeline` 构造 | 一旦主仓 FE 注册了 `gguf` extension，`utils::read_model()` 内部把 `.gguf` 分支**改为直接调** `ov::Core::read_model(models_path)`，不再走 `create_from_gguf()`；GenAI 的 `is_gguf_model()` 仍可保留作为"是否需要附加 tokenizer 后处理"的提示。 | Phase 3b 上线即可切换。 |
| Q6_K WA #2135 | 主仓 FE + GPU impl 上线后**废止**禁用；GenAI 端删除对应跳过逻辑。 | Phase 3b。 |
| `samples/cpp/text_generation/gguf_reader.cpp` 等示例 | 改为 `ov::Core core; auto model = core.read_model("foo.gguf");` + `ov::genai::LLMPipeline(model, tokenizer)` 的 2 行示例；删除手工构图代码。 | Phase 3b。 |
| Python 绑定 | `openvino_genai` 的 Python 侧若有 `read_gguf` / `from_gguf` 入口，去重为统一的 `ov.Core().read_model(path)` 路径；保留 `LLMPipeline(path)` 便捷入口。 | Phase 3b。 |
| 测试 / 模型 zoo | GenAI 现有 GGUF 测试改为以 OV FE 加载产出的 `ov::Model` 为输入；新增 IQ\* / TQ\* / Q5_\* / Q6_K 的端到端 token 一致性测试（与 `llama.cpp` 对齐，§8.4 风险 3）。 | Phase 3c 起逐步扩。 |
| `THIRD-PARTY-PROGRAMS.txt` / NOTICE | 若 mmap + parsing 代码与 `llama.cpp` 同源，需在主仓 FE 注册阶段同步许可声明，GenAI 侧的旧声明可一并清理。 | Phase 3a/3b。 |

### 9.4 过渡期共存策略

不要求"一次性切换"。建议两步走：

1. **过渡期（Phase 3a + 3b 上线，GenAI 暂未迁移完）**：GenAI 现有
   reader 与新 FE 共存。新 FE 走 `read_model` 路径，GenAI 内部
   `is_gguf_model()` 仍可优先选用 in-tree reader，避免一次性
   引入大量 token 差异回归。可以通过 GenAI 侧一个环境变量
   `OPENVINO_GENAI_USE_NATIVE_GGUF_FE=1` 切换。
2. **稳定期（Phase 3c 后）**：单元测试与 token 一致性测试在新 FE
   下绿后，将默认切换到新 FE；in-tree reader 转为兜底分支，
   再下个版本完全删除（避免 §8.4 风险 6 的"双路径产出不同 IR"
   调试问题）。

### 9.5 风险

- **Token 漂移**：dequant 在不同位置发生（FE vs GPU 插件）会带来
  最低位的数值差异，可能导致 greedy decode 在长序列后期出现 token
  分叉。对每个迁移的架构都要拿固定 seed + `llama.cpp` 参考做
  字节级 token 比对。
- **架构 builder 迁移成本**：llama/qwen2/qwen3 三套 builder 从
  GenAI 迁到主仓 FE 期间，两边代码必须用同一份"标准张量名 → OV
  子图"映射，否则会与 GenAI 已部署模型不兼容。
- **Tokenizer API gap**：§8.3 F8 (a) 方案要求 `openvino_tokenizers`
  暴露一个"从 metadata dict 构 tokenizer"的 API，这不是 GenAI 仓
  能独立完成的，需要跨仓协调。

---

## 10. Transformation 对 GGUF 原始文件的支持分析

本节回答：OV Core / 通用 transformation 框架在引入新 GGUF block
element types 之后，哪些 pass 会**自动安全**，哪些**必须打补丁**，
哪些**需要新增**行为。结论可直接驱动 §4.1（OV Core 改动）的工作清单。

### 10.1 总览 — 三类影响

| 分类 | 含义 | 数量（粗略） |
|------|------|--------------|
| 🟢 **自动安全** | 已经通过 `is_real()` / `is_quantized()` 等通用谓词早退；GGUF 设 `is_real() = false`、`is_quantized() = true` 后直接跳过。 | ~5 个家族 |
| 🟠 **必须打补丁** | 没有按 element type 做 gate，对不透明 block Constant 直接做 reshape / convert / fold，会**静默破坏数据**。 | ~8 个明确位置 |
| 🆕 **需要新增行为** | 没有任何现成模式（如 fractional bpw 的 Constant sizing、GGUF 序列化 round-trip）。 | ~3 个位置 |

### 10.2 🟢 自动安全的 pass（设好 TypeInfo 即可）

只要新 GGUF 类型在 [src/core/src/type/element_type.cpp](src/core/src/type/element_type.cpp)
的 `types_info[]` 中正确设置：

- `is_real() = false`（GGUF 是整数打包，不是浮点）
- `is_quantized() = true`（GGUF 是量化权重）
- `is_signed()` 按类型决定（Q4_0/Q5_0/Q6_K/Q8_0 是 signed，
  Q4_1/Q5_1/Q4_K/Q5_K/Q2_K/Q3_K 是 unsigned）

下列 pass 已经按这些谓词早退，**无需 GGUF 专用 carve-out**：

| Pass | 检查 | 结论 |
|------|------|------|
| [`group_normalization_fusion.cpp`](src/common/transformations/src/transformations/common_optimizations/group_normalization_fusion.cpp#L36) | `T.is_real() && !T.is_quantized()` | 自动跳过 GGUF |
| [`random_uniform_fusion.cpp`](src/common/transformations/src/transformations/common_optimizations/random_uniform_fusion.cpp#L49) | `is_real()` | 自动跳过 |
| [`matmul_multiply_fusion.cpp`](src/common/transformations/src/transformations/common_optimizations/matmul_multiply_fusion.cpp#L37) | `weights_el_type.is_real()` | 自动跳过 |
| `fp16_compression/mark_decompression_convert_constant_folding.cpp` 等 | `is_real()` | 自动跳过 |
| `align_mixed_fp32_fp16_types.cpp` | `is_real()` | 自动跳过 |

### 10.3 🟠 必须打补丁的 pass（不打会**静默破坏数据**）

按风险从高到低排序：

#### 1. `Constant::get_byte_size()` 与构造函数 — **CRITICAL**

[src/core/src/op/constant.cpp:394](src/core/src/op/constant.cpp#L394) 当前直接返回
`m_data->size()`。GGUF 是分块布局，需要
`ceil_div(num_elements, block_elem_count) * block_byte_size`。
必须改：

- `Constant` 构造函数：按 `is_gguf_block()` 走块计数公式。
- `Constant::get_byte_size()`：同上。
- `cast_vector<>()` / `convert_value_to_string()`：对 GGUF 抛
  `NOT_IMPLEMENTED`（块类型不能按元素访问）。
- `in_t_range<>()`：参考 `nf4` carve-out（[constant.cpp:150-195](src/core/src/op/constant.cpp#L150)）
  加 GGUF 入口。

#### 2. `Convert` op 元素类型校验 — **CRITICAL**

[src/core/src/op/convert.cpp:33-90](src/core/src/op/convert.cpp#L33)
当前对 `nf4` / `f4e2m1` / `f8e8m0` 做了模板特化，限制能/不能转换
的方向（如 from `nf4` 只能去 `{f16, bf16, f32, nf4}`）。

GGUF 必须**整族加入排除列表**：

- 任何方向的 `Convert(..., gguf_*)` 都拒绝（GGUF 是 weight-only
  编码，作为输出不合法 —— 对应 §8.3 F10）。
- `Convert(gguf_*, ...)` 也拒绝（块布局无法逐元素 cast）。
- 给现有的 `CONVERT_ET_LIST` / `is_to_nf4_supported()` 等宏 +
  helper 加平行的 `is_to_gguf_unsupported()`。

#### 3. `nop_elimination` — **HIGH**

[src/common/transformations/src/transformations/common_optimizations/nop_elimination.cpp](src/common/transformations/src/transformations/common_optimizations/nop_elimination.cpp)
对 Reshape/Squeeze/Unsqueeze 在 input shape == output shape 时
直接消除。对 GGUF 当**逻辑形状不变但块边界被重新解释**时会
**静默删 Reshape**，下游 impl 读到错位的 block。

补丁：在每个 eliminator 入口处加
```cpp
if (node->get_input_element_type(0).is_quantized()) return false;
```

#### 4. `transpose_sinking/ts_*.cpp` — **HIGH**

[src/common/transformations/src/transformations/transpose_sinking/](src/common/transformations/src/transformations/transpose_sinking/)
下 `ts_fuse.cpp`、`ts_gather.cpp`、`ts_reduce.cpp`、`ts_squeeze.cpp`
等会用 `cast_vector<size_t>()` 把 transpose order 取出来，构造
新的置换后 Constant。对 GGUF：

- `cast_vector` 失败（块类型不能按元素访问）。
- 即使不失败，置换会打乱 block 字节布局 → 内核读到乱序。

补丁：每个 matcher callback 早退
`if (in_constant->get_element_type().is_quantized()) return;`。

#### 5. `convert_fc_to_compressed.cpp`（通用 + GPU 各一份）— **HIGH**

[src/common/transformations/src/transformations/op_conversions/convert_fc_to_compressed.cpp:33](src/common/transformations/src/transformations/op_conversions/convert_fc_to_compressed.cpp#L33)
的 `process_compressed_weights()` 会 `std::make_shared<v0::Constant>(*constant, new_shape)`
对权重 Constant **reshape**。

- GGUF Constant 本身**已经是**最终的 compressed 形态，根本不应
  匹配"MatMul + Convert + Multiply + Subtract → FCCompressed"模板。
- 但 pattern matcher 可能因为 FE 先发了 `Convert(gguf_*)`（参见 §10.3 #2
  必须拒绝）而误匹配。

补丁：matcher 的根节点谓词加
`is_quantized() && !is_decompressible_via_convert()` 早退；同时
GPU plugin 自己的 `src/plugins/intel_gpu/src/plugin/transformations/convert_fc_to_compressed.cpp`
对照修复。

#### 6. `ConvertPrecision::fuse_type_to_constant()` — **MEDIUM**

[src/common/transformations/src/transformations/convert_precision.cpp:80-150](src/common/transformations/src/transformations/convert_precision.cpp#L80)
没有按 quantized 做早退，会尝试把任何 Constant 的 dtype 套到
`precisions_map` 上。

补丁：函数入口加
```cpp
if (constant->get_element_type().is_quantized()) return false;
```

#### 7. `MarkDequantizationSubgraph` — **MEDIUM**

[src/common/transformations/src/transformations/low_precision/mark_dequantization_subgraph.cpp](src/common/transformations/src/transformations/low_precision/mark_dequantization_subgraph.cpp)
会把"Constant(int) + Convert + Multiply"标记为 DQ 子图。GGUF 单
Constant 不应被标 —— 它本身就是 compressed payload。

补丁：标记前判断 `!is_quantized()` 或显式排除 `is_gguf_block()`。

#### 8. `ConstantFolding` 的 rt-info 标注 — **MEDIUM**

[src/core/src/pass/constant_folding.cpp:85](src/core/src/pass/constant_folding.cpp#L85)
自身不检查 element type，依赖 `disable_constant_folding(node)`
rt-info。

补丁：FE 在发射 GGUF Constant 时**必须**调
`disable_constant_folding(const_node)`（对应 §8.3 F3）。OV Core 的
ConstantFolding pass 本身**无需改动**，但 MOC pipeline 加一道
"sanity scan"：发现 `is_gguf_block()` 的 Constant 没打标 rt-info
时直接 throw（防御性，捕获 FE bug）。

### 10.4 🆕 需要新增的行为

#### 1. Block-aware Constant sizing — **NEW**

OV Core 当前没有"每元素位宽是分数 + 必须按 block 取整"的概念
（`nf4` 是 4 bpw 整数）。需要为每个 GGUF 类型在 `TypeInfo` 上
新增：

```cpp
struct GGUFBlockTraits {
    size_t block_elem_count;   // 32 或 256
    size_t block_byte_size;    // 18/144/176/…
};
```

以及 `element::Type` 上的访问器
`block_byte_size() / block_elem_count() / is_gguf_block()`。
`bitwidth()` 对 GGUF 类型应**抛错或返回向上取整值**，避免现有
依赖 `bitwidth()` 的代码（如 `MatMulMultiplyFusion` 的
`weights_el_type.size() <= 2`）误用。

#### 2. IR 序列化 round-trip — **NEW**

[src/core/src/pass/serialize.cpp](src/core/src/pass/serialize.cpp)
当前依赖 `get_byte_size()` + `get_data_ptr()`，只要 §10.4 #1 落地
就**应**正确 round-trip。但需要：

- 在 `src/core/dev_api/openvino/core/type/element_iterator.hpp`
  等迭代器/格式化路径上加 GGUF carve-out（避免序列化时 .xml
  端尝试打印元素值）。
- IR `.xml` schema 增加新 element type 字符串：
  `gguf_q4_0`、`gguf_q4_k`、`gguf_q5_k`、`gguf_q6_k` 等。
- [src/frontends/ir/src/ir_deserializer.cpp](src/frontends/ir/src/ir_deserializer.cpp)
  Constant 反序列化按新类型计算字节数。
- 单测：`ov::Core::read_model + serialize + read_model` 字节级
  一致性。

#### 3. Plugin 能力声明 — **NEW**

为支持 §3.5 的 hard-fail 契约，建议在
[src/inference/include/openvino/runtime/properties.hpp](src/inference/include/openvino/runtime/properties.hpp)
新增一个 read-only property，例如
`ov::supported_gguf_types`，让每个 plugin 在 `core.get_property()`
里声明它能消费哪些 GGUF element types。GPU 返回完整列表；CPU/NPU
默认返回空 set；上层工具可在 `compile_model` 前提前判定是否会
走到 §3.5 抛错。

### 10.5 与 §4.1 工作项的对齐

§4.1 当前只列了 `Type_t` 注册、`block_byte_size()` 访问器、
`Convert` carve-out、FE 端 `disable_constant_folding`。结合 §10.3
的补丁清单，§4.1 需要**扩展**：

- 在 `Constant` 构造函数、`get_byte_size`、`cast_vector`、
  `convert_value_to_string`、`in_t_range` 共 5 处加块感知逻辑。
- 在 `nop_elimination`、`transpose_sinking/ts_*`（4 个文件）、
  `convert_fc_to_compressed`（通用 + GPU 各 1）、`convert_precision`、
  `mark_dequantization_subgraph` 共 ~9 个 pass 入口加 quantized
  早退。
- 在 IR 反序列化 / 序列化 / element_iterator 三处加 GGUF 入口
  以保 round-trip。
- 新增 `ov::supported_gguf_types` plugin property。

补丁集本身可拆成一个独立的 Phase 0b（"Transformation 防护层"）
PR，先于 GPU impl（Phase 1/2）合入，并附 unit test 覆盖每条
补丁：构造 GGUF Constant → 跑对应 pass → 验证 Constant 字节不变、
图结构未被改写。这条 PR 是 §4.1 OV Core 改动之外**必须**的二级
依赖，缺它的话 GPU impl 即使写好了也会被上游某个 pass 在 MOC
阶段就**静默改写权重**导致跑错。

### 10.6 风险总结

| 风险 | 严重程度 | 缓解 |
|------|----------|------|
| 漏 patch 某个 `ts_*.cpp` / `nop_elimination` / `convert_fc_to_compressed`，GGUF Constant 被偷偷 reshape/transpose。 | **HIGH**（精度静默回归） | Phase 0b 给每个补丁加单测；CI 加一个 "GGUF byte-equality" gate：MOC 前后对每个 GGUF Constant 做字节哈希比较。 |
| 新增 GGUF type 不在 `is_to_nf4_supported` 平行的排除宏里，`Convert` 被允许生成并在执行期才报错。 | MEDIUM | 在 §10.3 #2 的 helper 上加编译期 `static_assert`：遍历 `element::Type_t` 枚举，每个 `is_gguf_block()` 类型必须在排除集内。 |
| 序列化 round-trip 在字符串化路径（debug print、`graph_to_string`）上崩。 | LOW | `element_iterator` 对 GGUF 抛 `NOT_IMPLEMENTED` + 提示用户用 `disable_constant_folding` rt-info 已经设的特殊路径。 |
| Plugin 能力 property 设计与现有 `supported_properties` 不一致。 | LOW | 复用现有 `device.capabilities` 模式，按 string list 暴露。 |

---

## 11. 两种总体路线的可行性与复杂度对比

到目前为止，业界落地 GGUF 支持有两条互斥的总体路线。本节正式评估
两条路线，并给出推荐。

### 11.1 路线定义

- **路线 A — GenAI 内扩 (in-tree extension)**：保留并扩展
  `openvino.genai/src/cpp/src/gguf_utils/` 现有 reader（§9.1），
  在它里面**继续做 dequant 并下发 `(u32 weights, f16 scales, f16 biases)`
  三元组**给 `FullyConnectedCompressed`。新增量化格式（Q5_\*、Q6_K、
  K-quants 其他、IQ\*、TQ\*）通过新增 `extract_qX_X_data()` 函数
  实现。**OpenVINO 主仓不动**：不引入新 GGUF element-type，不写
  GPU impl，不写 GGUF FE。
- **路线 B — 主仓原生 FE + GPU impl**：按本文档 §1–§10 落地：OV
  Core 新增 `block_*` element-type（§2）、`src/frontends/gguf/`
  新建 FE（§8）、GPU 插件新增 `ocl::FCGGUFOpt` 单一 impl 类（§3）、
  Transformation 防护层补丁（§10）、GenAI 端按 §9 迁移到
  `Core::read_model(*.gguf)`。

### 11.2 可行性

| 维度 | 路线 A | 路线 B |
|------|--------|--------|
| 技术可行性 | ✅ **完全可行**。GenAI reader 已是 production 路径，新增 quant 格式只是按 `ggml-quants.c` 写 dequant 内核。 | ✅ 可行但需要跨多个仓库 / 多个 component owner 协同（OV Core、common transformations、GPU plugin、新 FE、GenAI）。 |
| 涵盖面 | ⚠ 仅限 GenAI 调用链。`ov::Core::read_model("*.gguf")`、Benchmark App、accuracy checker、第三方集成走不通。 | ✅ 全 OpenVINO 通用（CLI、samples、Benchmark App、所有绑定）。 |
| 精度保持 | ❌ 受限。dequant 在 host 端 f16/f32 完成，对 K-quants / IQ-quants 的嵌套 scale **只能预乘**，与 `llama.cpp` 的逐子块运算相比有舍入差。 | ✅ §3.3.2 的精度保持映射在低比特域完成 dequant，与 `llama.cpp` 数值轨迹更接近。 |
| 性能上限 | ⚠ 中等。decompose 之后 `FullyConnectedCompressed` 现有内核已经够好；但 K-quants/IQ-quants 的嵌套 scale 被预乘后**带宽放大**（4-bit 原始 → 实际 ~8-bit 等效压缩），decode roofline ≈ 50%。 | ✅ 高。OCL 内核直接消费 GGUF block，按真实压缩比（如 Q4_K_M ≈ 4.5 bpw）跑 ≥ 90% bandwidth roofline。 |
| Host 内存 | ❌ 整文件读进内存。70B Q4_K_M (~38 GB) 直接撑爆。 | ✅ §3.3 + §8.2 的零拷贝/可选 mmap，`Constant` 直接持有 GGUF 字节段。 |
| Mixed-format `*_M`/`*_S` | ⚠ 实现成本高 —— GenAI 现 reader 走单一类型分支，要为每个张量按类型分派 dequant；做完后仍是放大后的等效压缩比。 | ✅ §3.6 的 5 个机制天然支持，单 impl 内分派。 |
| 上游兼容 | ✅ 不动主仓 → 不影响其他 plugin、不影响现有 transformation/MOC 流。 | ⚠ §10 的 8 处 must-patch 必须先合入，否则 GGUF Constant 会被静默改写。 |

### 11.3 复杂度（工程量）

| 维度 | 路线 A | 路线 B |
|------|--------|--------|
| 涉及代码区 | `openvino.genai/src/cpp/src/gguf_utils/` 1 个目录 | OV Core (element-type)、common transformations (§10 8 处)、新 FE 目录、GPU plugin (`ocl_v2/gguf/` + 根目录 `.cl`)、GenAI 迁移 (§9) |
| 跨仓协调 | 单仓内变更 | 主仓 + GenAI 跨仓 PR 序列 |
| 新增 OV public API | 无 | `element::Type` 新枚举（§2）、`ov::supported_gguf_types` property（§10.4）、`FrontEnd` 注册 |
| Reviewer / Code owner 数量 | 1–2（GenAI GGUF 维护者） | ≥ 5（Core、transformations、GPU plugin、FE、GenAI；每个对应独立 CODEOWNERS） |
| 新增 dequant 函数（按格式数） | **15+**（覆盖剩余全部 GGUF 格式各一个 `extract_qX_X_data`） | 0（GenAI 侧）+ §3.3 描述的 per-format JIT 分支 + §3.3.2 transcode 内核（共享多种格式） |
| 量化感知 transformation 工作 | 0 | §10 的 8 处补丁 + 单测 + CI byte-equality gate |
| 测试矩阵 | 对每个新增格式加一个 GenAI 端到端 token 测试 | OV Core 单测 + FE 单测 + GPU 功能测 + GenAI 端到端 + token 一致性 + IR round-trip |
| 大致工作量量级 | **2–4 人周**（按格式数线性扩展，但路径已通） | **3–6 人月**（5 个子模块串行 + 跨仓 review 周期） |
| 风险面 | 局部、单点：dequant 函数本身写错 | 分布式：§10.6 的 transformation 静默改写、§8.4 的 FE 架构覆盖、§9.5 的 token 漂移 |

### 11.4 路线 A 的隐性瓶颈

路线 A 看似廉价，但有三个**结构性**问题，不会因增加更多 dequant 函数
而消失：

1. **decompose 抵消压缩比。** Q4_K_M 原本 4.5 bpw，dequant 成 f16
   之后 `FullyConnectedCompressed` 实际承载 16 bpw；GenAI 现有路径靠
   `compressed_weights=true` 再压回去 8 bpw 等效，仍比原生 4.5 bpw 高
   ~78%。对 memory-bound 的 decode，这直接吃掉一半 token/s。
2. **嵌套 scale 不可无损表达。** Q6_K / IQ-quants 的子块 scale 是
   6-bit / 4-bit 嵌套，dequant 到 f16 后再让 `FullyConnectedCompressed`
   反压缩，无法等价复原 ——这就是 Q6_K WA #2135 的根因。再加多少
   dequant 函数都不解决这个问题（路线 A 下 Q6_K/IQ\* 只能"近似支持"）。
3. **生态隔离。** `.gguf` 在路线 A 下永远只是 GenAI 的私有输入格式；
   Benchmark App、accuracy checker、validation pipeline、第三方集成
   都用不上，社区 PR/issue 也无法收敛到一个入口。

### 11.5 在"原生承载 GGUF 参数、不做上层 dequant"这一**硬约束**下的结论

> **约束**：模型权重必须以原始 GGUF block 字节进入推理路径，
> 上层（OV Core / FE / GenAI host 代码）**不允许** dequant 到
> f16/f32 或对嵌套 scale 做预乘。dequant 只能发生在 GPU 内核
> 寄存器内（§3.3.1）或 OneDNN WOQ 低比特域（§3.3.2）。

在此约束下：

- **路线 A 直接出局**。GenAI 现有 reader 的整个数据契约就是
  decompose 式 `(u32 weights, f16 scales, f16 biases)` 三元组
  （§9.1）—— 这本身就是"上层 dequant"。即使新增格式按同样模式
  扩展，也违反约束；要改成原生承载，等于把 GenAI reader 整体
  重构成 §3 / §8 描述的 FE，已经退化为路线 B。
- **路线 B 是唯一满足约束的方案**。它的设计目标第一条就是
  "Constant 持有不透明 GGUF 字节，元素数即权重总数，dequant 推
  到内核内"（§1 / §3.3）。

### 11.6 决策矩阵（硬约束生效后）

| 客户需求 | 推荐路线 |
|----------|----------|
| **原生 GGUF 参数承载（无上层 dequant）** | **唯一可行：路线 B** |
| 精度保持 / Q6_K / IQ-quants / TQ-quants | **唯一可行：路线 B** |
| 70B+ 模型 host 内存敏感 / 需要 mmap | **唯一可行：路线 B** |
| `ov::Core::read_model("*.gguf")` / Benchmark App / 第三方集成 | **唯一可行：路线 B** |

路线 A 仅在"放弃硬约束、纯做小补丁"的退化场景下还有意义，不在
本次决策范围内。

---

## 12. AI 驱动开发的执行方案（路线 B 落地）

本节专为"AI agent 实施而非人工实施"的场景设计。AI 在 OpenVINO
这种多 component 大仓的优势/劣势与人类不同，工作分解策略必须
相应调整。

### 12.1 AI 实施的成本结构 vs 人工

| 维度 | 人工 | AI agent |
|------|------|----------|
| §10 的 8 处 transformation 补丁 | 散落 8 个文件，code owner 分布广，PR review 周期长 | **强项**：模式高度同构（统一加 `is_real()/is_quantized()` 守卫），一个 subagent 一轮可完成 |
| §3.3 per-format JIT 分支 / §3.3.2 transcode kernel | 需要逐格式手写 + 调精度 | **强项**：每个格式一个 subagent 并行，比对 `llama.cpp` 参考实现 |
| §2 新增 element-type（含 `TypeInfo`、Python/C/JS 绑定、序列化） | 中等工作量，模板化 | **强项**：跟 `nf4`/`u4`/`f4e2m1` 模式机械复制 |
| §8 FE 主体（GGUF parser、架构 builder） | 中等 | **中等**：parser 部分机械；架构 builder 需要语义判断（哪些张量绑哪个子图） |
| §3 元类型语义决策（`is_real`/`is_quantized`/`bitwidth` 含义） | 设计会议 1–2 次 | **弱项**：必须人工先定 spec，AI 不能自洽判定 |
| §9 GenAI 端的 tokenizer API 跨仓协调 | 中等（跨 team 沟通） | **弱项**：跨仓 review、向后兼容性判断需要人工拍板 |
| §10.6 byte-equality CI gate / token 一致性 harness | 一次性 + 维护 | **强项**：harness 与 reference oracle 由 AI 写最合适 |
| Code review 负担 | 标准 PR review | **更重**：AI 生成代码必须每 PR 跑全套测试 + 人工抽检关键设计点 |

**核心结论**：AI 把"3–6 人月"压成 **2–4 周墙钟时间**，但 **不会**
压低 reviewer 人力。决定速度的不是 implement 工作量，而是
**每个 PR 的合并周期 + 设计冻结决策点**。

### 12.2 PR-级分解（3 个本地 PR）

**范围基线收紧**：本期只支持 **qwen3** 作为唯一基准架构，
**不**做 llama / qwen2 / mistral / phi 等其他架构 —— 它们留给
后续增量 PR（每个架构一个 builder 文件，~半天工作量）。

**交付方式**：所有代码本地 commit，**不 push 到 remote**。
每个 PR 是一个独立的 git branch / commit set，方便人工 review 与
回滚；不走 GitHub PR 流程，不触发 remote CI。

每个 PR 必须满足：① 本地可独立编译 + 通过自带测试；② 自带
oracle（`llama.cpp` / 黄金值）；③ 三个 PR 之间有清晰的依赖顺序，
但内部对外接口冻结后允许并行开发。

| PR # | 标题 | 仓库 | 范围 | 依赖 | 验收 |
|------|------|------|------|------|------|
| **PR-FE** | OV Core element-type + Transformation 守卫 + GGUF FE（qwen3 only） | `openvino` | §1 全部 23 个 element-type + §10 八处守卫 + `src/frontends/gguf/` 骨架 + qwen3 builder + IR round-trip + byte-equality CI gate | PR-0（人工 SPEC 冻结）| 本地 `core.read_model("qwen3-Q4_K_M.gguf")` 拿到合法 `ov::Model` + 全部 ov_core / transformation / FE 单测绿 |
| **PR-GPU** | `ocl::FCGGUFOpt` 单一 impl + qwen3 baseline kernel 集 | `openvino` | `src/plugins/intel_gpu/src/graph/impls/ocl_v2/gguf/` 下 C++ + 根目录 `.cl` + 注册 + **qwen3 实际使用的格式族** kernel：Q4_0/Q4_K/Q5_K/Q6_K/Q8_0（5 种）；其他 18 种 element-type 在 `validate_impl` 中返回 false（等后续 PR 扩） + `ov::supported_gguf_types` property | PR-FE（依赖 element-type 与 FE 产出的 `ov::Model`）| GPU 功能测：qwen3-Q4_K_M / qwen3-Q5_K_M / qwen3-Q6_K / qwen3-Q8_0 四个模型推理 + per-format kernel oracle（vs `ggml_dequantize`）+ §7 byte-equality gate 在 GPU plugin transformations 上仍绿 |
| **PR-GENAI** | GenAI 切换到 `Core::read_model` + tokenizer rt-info 迁移 + 删 in-tree reader | `openvino.genai` | `src/cpp/src/utils.cpp` 切入口 + `gguf_tokenizer.cpp` 改读 `model.get_rt_info` + 删除 `gguf_quants.cpp` / `gguf_modeling.{hpp,cpp}` + 默认 `OPENVINO_GENAI_USE_NATIVE_GGUF_FE=1` 同时保留 `=0` 回退路径 + 示例 + Python 绑定 | PR-FE + PR-GPU（端到端可跑通 qwen3 推理）| GenAI 既有 qwen3 GGUF 端到端测试在新路径下绿 + token 一致性 vs `llama-cli` (qwen3, greedy, seed=0, 256 token) 100% 一致 |

**依赖顺序**：PR-FE → PR-GPU → PR-GENAI。**前一个不绿后一个不开
工**，避免接口未冻结导致返工。

### 12.3 各 PR 的内部 subagent fan-out

#### PR-FE（最大、最关键，~5 个 subagent 并行）

| Subagent | 范围 | 输出 |
|----------|------|------|
| `core-types` | 23 个 `gguf_*` element-type + `TypeInfo` + `block_byte_size()`/`block_elem_count()`/`is_gguf_block()` 访问器 + Python/C/JS 绑定 + `Constant::get_byte_size` 修补 + `Convert` carve-out | `src/core/` |
| `transformations-guards` | §10.3 八处守卫补丁（同构模式 `if (is_gguf_block(t)) return;`）+ 自动安全 5 个 pass 的 e2e 测试 | `src/common/transformations/` 8 个文件 |
| `fe-skeleton` | `src/frontends/gguf/` 目录 + `FrontEnd` / `InputModel` 实现 + GGUF parser (mmap + header + tensor table) + GGUF→element::Type 映射 | `src/frontends/gguf/` |
| `fe-qwen3-builder` | qwen3 架构 builder：embeddings、attention（GQA）、MLP（SwiGLU）、RMSNorm、RoPE、lm_head；rt-info schema 填值 | `src/frontends/gguf/src/builders/qwen3_builder.cpp` |
| `ci-gates` | byte-equality CI gate (§7.1) + IR round-trip gate (§7.4) + qwen3 测试模型集成 | `tests/transformations/` + `src/frontends/gguf/tests/` |

主 agent 工作：定义 5 个 subagent 的接口契约、汇总冲突、跑总集成
测试。

#### PR-GPU（~3 个 subagent 并行）

| Subagent | 范围 | 输出 |
|----------|------|------|
| `gpu-manager` | `FCGGUFOpt` ImplementationManager + `fully_connected_impls.cpp` 注册 + execute 分派骨架（M-threshold） + `validate_impl` 只通过 baseline 5 种 type | `ocl_v2/gguf/fc_gguf_opt.{hpp,cpp}` |
| `gpu-native-gemv` | 5 种 baseline 格式的 native OCL GEMV（memory-bound 路径）+ per-format JIT 分支 | `ocl_v2/fc_gguf_opt.cl` + `ocl_v2/gguf/jit/` |
| `gpu-transcode-onednn` | 5 种 baseline 格式的 transcode kernel (Q4_K/Q5_K → u4/u8 + scale + ZP；Q6_K → i8 + scale；Q4_0/Q8_0 → i4/i8 + scale)；直接构造 `dnnl::matmul` （compute-bound 路径） + `ov::supported_gguf_types` property | `ocl_v2/fc_gguf_transcode.cl` + `ocl_v2/gguf/transcode_onednn.cpp` |

主 agent 工作：把 §3.3 / §3.3.2 的伪代码转化成实际的 5 路 oracle
测试，每个 subagent 必须拿 `ggml-quants.c` 对应 dequant 函数做
ground truth。

#### PR-GENAI（~2 个 subagent，串行）

| Subagent | 范围 | 输出 |
|----------|------|------|
| `genai-entry-switch` | `utils.cpp::read_model()` 切换到 `Core::read_model`；`OPENVINO_GENAI_USE_NATIVE_GGUF_FE` 默认 `true` + 回退路径；更新 `samples/cpp/text_generation/` 示例 + Python 绑定 | `src/cpp/src/utils.cpp`、`samples/`、`python/` |
| `genai-tokenizer-migrate` | `gguf_tokenizer.cpp::tokenizer_config_from_meta()` 输入源改为 `model.get_rt_info()`；删除 `gguf_quants.cpp`、`gguf_modeling.{hpp,cpp}` 中已被 FE 取代的代码 | `src/cpp/src/gguf_utils/` |

主 agent 工作：保证 qwen3 端到端 token 一致性测试不漂移。

### 12.4 AI 执行的强制规范

1. **每个 PR 必须自带测试**。无测试 = 不可合，因为 AI 无法在
   下个 PR 才补的测试上自我验证。
2. **本地开发，禁止 push 到 remote**。所有 commit 留在本地分支，
   review 通过后由人工决定何时打包推送。CI 不在 remote 跑，
   依赖本地脚本（`./tests/run_gguf_local.sh` 由 `ci-gates`
   subagent 一并交付）。
3. **subagent fan-out 用于 PR 内部，不跨 PR**。三个 PR 之间是串行
   依赖；PR 内部可以多个 subagent 并行（PR-FE 5 路、PR-GPU 3 路、
   PR-GENAI 2 路）。
4. **每个 subagent 的 oracle 必须可机器验证**。`llama.cpp` 的
   `llama-cli` / `ggml_dequantize` 作为唯一 ground truth；不允许
   用"模型看起来 reasonable"做验收。
5. **SPEC 冻结后 AI 不允许修改 API 名称、rt-info key、
   element-type 枚举顺序**。任何此类变更必须回到 PR-0 走人工 review。
6. **Constant invariance gate（PR-FE 引入）是整个项目的安全网**。
   只要它在本地脚本里跑绿，"上层不允许 dequant"这一硬约束就有
   自动化保障。后续任何修改都先跑此 gate。
7. **qwen3 builder 用 GenAI 现有代码做 reference**。
   `openvino.genai/src/cpp/src/gguf_utils/gguf_modeling.cpp` 里
   qwen3 的"标准张量名 → OV 子图"映射已经验证过，AI 应直接迁移而
   不是从 GGML 文档重写。

### 12.5 时间估算（AI 实施、本地交付）

| 阶段 | PR | 预计墙钟 |
|------|-----|---------|
| 设计冻结 | PR-0（SPEC.md + STYLE.md，人工 review）| 3–5 天 |
| Core + 守卫 + FE + qwen3 builder | PR-FE | 1.5–2 周（5 路并行；瓶颈在 qwen3 builder 调试）|
| GPU manager + 5 种 baseline kernel | PR-GPU | 1.5–2 周（3 路并行；瓶颈在 K-quant transcode oracle）|
| GenAI 切换 + tokenizer 迁移 | PR-GENAI | 3–5 天 |

**总计：4–5 周墙钟**（人工至少 3–6 个月）。
本地交付路径下，外部 review 不再阻塞合并节奏；瓶颈完全是 AI
迭代 + 本地 oracle 跑通的速度。

### 12.6 风险（3-PR 结构特有）

| 风险 | 缓解 |
|------|------|
| PR-FE 把 element-type 枚举顺序或 rt-info key 写错，PR-GPU/PR-GENAI 才发现 | SPEC §1.1 / §3.3 已冻结；PR-FE 完成后必须先用 PR-GPU 的占位 `validate_impl` 跑过一次，对接通过再开 PR-GPU 实质开发 |
| PR-GPU 只覆盖 5 种 baseline 格式，用户拿到 IQ4_XS qwen3 模型会失败 | PR-GPU 的 `validate_impl` 对其他 18 种 type 明确返回 false 并 fallback 到 onednn 已有 W4A16 路径（如果适用）或报清晰错误；docs 标注"本期支持的格式列表" |
| AI 在 5 种 kernel 里写出"看起来对、数值差几位"的代码 | 强制 `ggml_dequantize` oracle + per-block byte-equality；不允许只比对最终 logits |
| 本地开发分支漂移过远，最终 push 时与 master 冲突 | 每个 PR 起手 `git fetch && git rebase origin/master`；每周一次 rebase；不允许在远期未 rebase 的分支上累加 PR |
| qwen3 builder 与 GenAI 现有实现行为不一致，token 漂移 | PR-FE 必须在自带测试中复现 GenAI 现有 qwen3 推理的 hidden states，allclose 通过才能开 PR-GPU |
| 跨仓 PR-GENAI 依赖 OV main 的 PR-FE 已合，但本地 OV 装的是旧版 | PR-GENAI 必须在 README 中固定 `OPENVINO_GIT_SHA` 参考；本地脚本 check OV 版本 |

### 12.7 最终建议

1. **立刻完成 PR-0（SPEC.md + STYLE.md）人工 review** —— SPEC 已起草，
   见 [SPEC.md](.github/skills/dev_gguf_support/SPEC.md)。
2. **PR-FE → PR-GPU → PR-GENAI 严格串行**：每个 PR 完成自带测试
   并人工抽检后再开下一个。
3. **PR 内部用 subagent fan-out**：PR-FE 5 路、PR-GPU 3 路、
   PR-GENAI 2 路并行；主 agent 只分发与汇总。
4. **本地交付，禁止 push**：所有 commit 留本地分支；何时推送由
   人工决定。
5. **qwen3 作为唯一基准**：本期不做 llama/qwen2/mistral/phi；
   留给后续每个架构一个增量 PR（半天工作量、纯 builder 文件新增）。


