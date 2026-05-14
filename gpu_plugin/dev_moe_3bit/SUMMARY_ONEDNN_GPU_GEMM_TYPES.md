# oneDNN GPU GEMM 数据类型支持分析

> 分析目标：oneDNN GPU GEMM（`src/plugins/intel_gpu/thirdparty/onednn_gpu`）是否支持 U2/I2 权重精度

---

## 一、结论摘要

| 问题 | 结论 |
|------|------|
| oneDNN GPU GEMM 是否支持 **U2/I2（2-bit）** 权重？ | **否，完全不支持** |
| oneDNN GPU GEMM 是否支持 **U4/S4（4-bit）** 权重？ | **是，通过 `wei_decomp` 路径支持** |
| 是否存在 3-bit 权重类型？ | **否** |
| Intel Xe GPU 硬件 ISA 层面是否有 2-bit 寄存器类型？ | **是（u2/s2 in ngen），但 oneDNN 未将其暴露给 GEMM** |
| MoE 3-bit prefill 能否直接使用 oneDNN GPU GEMM？ | **否，需要自定义 kernel** |

---

## 二、API 层面：`dnnl_data_type_t` 枚举

**文件：** `include/oneapi/dnnl/dnnl_common_types.h`

```c
typedef enum {
    dnnl_data_type_undef = 0,
    dnnl_f16 = 1,
    dnnl_bf16 = 2,
    dnnl_f32 = 3,
    dnnl_s32 = 4,
    dnnl_s8  = 5,
    dnnl_u8  = 6,
    dnnl_f64 = 7,
    dnnl_boolean = 8,
    dnnl_f8_e5m2 = 9,
    dnnl_f8_e4m3 = 10,
    dnnl_s4 = 11,   // ← 最小整数类型：有符号 4-bit
    dnnl_u4 = 12,   // ← 最小整数类型：无符号 4-bit
    dnnl_e8m0 = 13,
    dnnl_f4_e2m1 = 14,
    dnnl_f4_e3m0 = 15,
    dnnl_s64 = 16,
    // s2/u2 完全不存在
} dnnl_data_type_t;
```

**关键发现：** oneDNN 公开 API 中最小整数精度为 **s4（有符号 4-bit）和 u4（无符号 4-bit）**，没有任何 2-bit 或 3-bit 整数类型。

---

## 三、JIT GPU GEMM 权重类型校验逻辑

**文件：** `src/gpu/intel/gemm/jit.hpp`（Intel GPU 主 GEMM 实现）

### 3.1 整数激活路径（INT8/INT4 activation）

```cpp
// Line 114-119
if (utils::one_of(d->c_type(), s32, f16, bf16, f32, u8, s8)
        && utils::one_of(d->a_type(), u8, s8, u4, s4)) {
    // A = INT8/INT4 时，B（权重）只允许 u8/s8，或者进入 wei_decomp 路径
    VDISPATCH_GEMM(
            (utils::one_of(d->b_type(), u8, s8) || wei_decomp_),
            VERBOSE_UNSUPPORTED_DT);
}
```

a_type 合法集合 = `{u8, s8, u4, s4}` — **不含 u2/s2**

### 3.2 浮点激活路径（f16/bf16 activation）

```cpp
// Line 123-130
} else if (utils::one_of(d->a_type(), f16, bf16)) {
    VDISPATCH_GEMM(d->b_type() == d->a_type(), ...);   // B 必须与 A 同类型
    VDISPATCH_GEMM(utils::one_of(d->c_type(), d->a_type(), f32,
                           f8_e5m2, f8_e4m3), ...);
}
```

此路径下 B（权重）也只能是 f16/bf16，不存在 2-bit 分支。

### 3.3 `wei_decomp`（权重解压缩）路径 — s4/u4 权重压缩

**文件：** `src/gpu/intel/gemm/jit/pd.cpp`（`wei_decomp()` 函数，Line 208-218）

```cpp
bool pd_t::wei_decomp() {
    using namespace data_type;
    return (dy_quant_enabled()                                        // 动态量化启用
               && utils::one_of(d->a_type(), u8, s8, s4, u4, f8_e4m3, ...)
               && utils::one_of(d->b_type(), f16, f32, bf16, f8_e5m2, f8_e4m3))
        && types::data_type_bits(d->a_type()) < types::data_type_bits(d->b_type())
        && attr()->mayiconvert(d->a_type(), f32);
}
```

**分析：** `wei_decomp` 路径 A 侧只识别 `{u8, s8, s4, u4, f8_e4m3}`，B（权重）侧识别 `{f16, f32, bf16, f8_e5m2, f8_e4m3}`。这是一种**激活 < 权重精度**时的逆向解压路径（不是常规的权重压缩 GEMM）。

### 3.4 INT4 权重检测逻辑

```cpp
// pd.cpp Line 337-338
const bool a_int4 = utils::one_of(desc()->a_type(), s4, u4);
const bool b_int4 = utils::one_of(desc()->b_type(), s4, u4);
```

b_int4 和 a_int4 均只检查 **s4/u4**，完全没有 2-bit 分支。

---

## 四、Reference Matmul 权重类型校验

**文件：** `src/gpu/intel/matmul/ref.hpp`（OCL 参考实现）

```cpp
// Line 83-96
const bool is_f32  = src_dt_ == f32  && utils::one_of(wei_dt_, f32, s8, u8, s4, u4);
const bool is_f16  = src_dt_ == f16  && utils::one_of(wei_dt_, f16, s8, u8, s4, u4);
const bool is_bf16 = src_dt_ == bf16 && utils::one_of(wei_dt_, bf16, s8, u8, s4, u4);
const bool is_int8 = utils::one_of(src_dt_, u8, s8)
                  && utils::one_of(wei_dt_, u8, s8, u4, s4);
```

所有路径下，权重（`wei_dt_`）的最小整数精度均为 **s4/u4**，无 2-bit 选项。

---

## 五、硬件 ISA 层面：ngen u2/s2

**文件：** `third_party/ngen/ngen_core.hpp`

```cpp
// Line 417-418
u2 = 0x3E,
s2 = 0x3F,

// GRF 操作辅助方法（Line 1248-1249）
constexpr14 GRF u2() const { return retype(DataType::u2); }
constexpr14 GRF s2() const { return retype(DataType::s2); }
```

**结论：** Intel Xe GPU 的 ISA（指令集）层面**确实存在 u2/s2 寄存器数据类型**，这是 Intel GPU 微架构对 2-bit 量化的低层支持。但 oneDNN 从未将这个能力暴露到 GEMM/matmul 的计算路径中——没有任何 GEMM kernel 使用 2-bit 权重类型作为输入。

---

## 六、支持的精度矩阵（oneDNN GPU GEMM）

| A（激活）类型 | B（权重）类型 | 输出类型 | 路径 |
|--------------|--------------|---------|------|
| f32 | f32 | f32 | FP 路径 |
| f16 | f16 | f16/f32 | FP 路径 |
| bf16 | bf16 | bf16/f32 | FP 路径 |
| f8_e5m2/f8_e4m3 | f8_e*/f16/f32 | f16/f32 | FP8 路径 |
| f4_e2m1/f4_e3m0 | f4_* | f16/f32 | FP4 路径 |
| u8/s8 | u8/s8 | s32/f32 | INT8 路径 |
| **f16/bf16** | **s4/u4**（weight-only） | f16/bf16/f32 | **WOQ INT4 路径** ✅ |
| u8/s8 | s4/u4（weight-only） | s32/f32 | WOQ INT4+INT8 路径 |
| 任何类型 | **s2/u2** | 任何 | **❌ 不支持** |
| 任何类型 | **s3/u3** | 任何 | **❌ 不支持** |

---

## 七、对 MoE 3-bit 方案的影响

### 7.1 当前状态

由于 oneDNN GPU GEMM 的权重最小精度为 INT4，**3-bit（INT3）权重精度的 MoE GEMM 无法使用 oneDNN**，需要完全自定义 OpenCL/SYCL kernel。

### 7.2 可行的替代路径

| 方案 | 可行性 | 说明 |
|------|--------|------|
| **INT4 权重 + oneDNN WOQ** | ✅ 直接支持 | `b_type = s4/u4`，JIT GEMM 有专用路径 |
| **INT3 权重解包到 INT4/INT8 + oneDNN** | ⚠️ 有额外开销 | 先 dequant 为 f16/bf16，再调 GEMM |
| **自定义 INT3 GEMM kernel（OpenCL）** | ✅ 技术可行 | 利用 ngen ISA 级 u2/u4 类型，手写 kernel |
| **INT2 权重 + oneDNN** | ❌ 不支持 | API 中不存在 u2/s2 类型 |

### 7.3 MoE prefill grouped GEMM 分析

**文件：** `src/gpu/intel/matmul/grouped_micro_gemm.hpp/.cpp`

Grouped micro GEMM 是 MoE prefill 的主路径，其类型约束继承自上层 matmul pd_t，没有为 2-bit 或 3-bit 增加任何特殊路径。最小权重精度同样是 **s4/u4**。

---

## 八、关键代码位置索引

| 关注点 | 文件路径 | 关键行 |
|--------|----------|--------|
| API 数据类型枚举 | `include/oneapi/dnnl/dnnl_common_types.h` | `s4=11, u4=12`，无 s2/u2 |
| JIT GEMM 类型校验 | `src/gpu/intel/gemm/jit.hpp` | L114-L148 |
| wei_decomp 判断 | `src/gpu/intel/gemm/jit/pd.cpp` | L208-L218, L337-L338 |
| Reference matmul 类型检查 | `src/gpu/intel/matmul/ref.hpp` | L83-L97 |
| GPU ISA u2/s2（ngen） | `third_party/ngen/ngen_core.hpp` | L417-418, L1248-1249 |
| 权重 INT4 路径(b_int4) | `src/gpu/intel/gemm/jit/pd.cpp` | L337-L360 |

---

## 九、总结

```
oneDNN GPU GEMM 数据类型支持（权重维度）：

  f64 ──── 支持（原生 f64 GPU）
  f32 ──── 支持（标准路径）
  bf16 ─── 支持（标准路径）
  f16 ──── 支持（标准路径）
  f8  ──── 支持（f8_e5m2, f8_e4m3）
  f4  ──── 支持（f4_e2m1, f4_e3m0）
  u8  ──── 支持（INT8 对称量化）
  s8  ──── 支持（INT8 非对称量化）
  u4  ──── 支持（INT4 Weight-Only Quantization）✅
  s4  ──── 支持（INT4 Weight-Only Quantization）✅
  u3/s3 ── ❌ 不支持（API 中不存在）
  u2/s2 ── ❌ 不支持（API 中不存在，仅在 GPU ISA 层存在）
```

**最终结论：**
- oneDNN GPU GEMM **不支持 U2/I2 权重精度**，API 根本不存在这两种类型
- Intel Xe GPU 硬件 ISA **理论上支持** u2/s2 寄存器类型（ngen_core.hpp 可见），但 oneDNN 没有实现使用这些类型的 GEMM kernel
- 最小支持精度为 **u4/s4（4-bit）**，通过 Weight-Only Quantization（WOQ）路径实现
- 3-bit MoE GEMM 需要**完全自定义** OpenCL/SYCL kernel，不能依赖 oneDNN GPU GEMM
