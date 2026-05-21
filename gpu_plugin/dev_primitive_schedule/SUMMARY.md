# Primitive Scheduler — 实现总结与问题解决记录

> **目的**：完整记录 primitive scheduler 的当前实现状态、遇到的所有问题及其解决方案细节，  
> 以及 W4A16 GEMV OCL/CM kernel 的优化过程和 E2E 验证结果。  
> 作为后续工作的延续基础文档。

---

## 目录

1. [系统架构概览](#1-系统架构概览)
2. [原始问题分析](#2-原始问题分析)
3. [解决方案一：Runtime Implementation Switching](#3-解决方案一runtime-implementation-switching)
4. [解决方案二：Sub-byte Weight 兼容性机制](#4-解决方案二sub-byte-weight-兼容性机制)
5. [解决方案三：M=1 Decode 合成重试](#5-解决方案三m1-decode-合成重试)
6. [解决方案四：W4A8 Dynamic Quantization 处理](#6-解决方案四w4a8-dynamic-quantization-处理)
7. [OCL GEMV Kernel 优化详情](#7-ocl-gemv-kernel-优化详情)
8. [CM GEMV Kernel 实验](#8-cm-gemv-kernel-实验)
9. [E2E 验证结果与分析](#9-e2e-验证结果与分析)
10. [当前未解决问题](#10-当前未解决问题)
11. [文件清单与代码位置索引](#11-文件清单与代码位置索引)
12. [详细流程图](#12-详细流程图)

---

## 1. 系统架构概览

### 1.1 Primitive Scheduler 在 GPU Plugin 中的位置

```
Model IR (.xml/.bin)
    ↓
ov::Core::compile_model("GPU")
    ↓
┌──────────────────────────────────────────────────────┐
│ Graph Compilation Stage                               │
│  ├─ program_node 构建                                 │
│  ├─ layout_optimizer (format 选择)                    │
│  └─ Registry<T>::get_implementations()               │
│      ├─ OneDNN ImplementationManager                  │
│      ├─ OCL FCCompressedGenerateOpt (新增)            │
│      ├─ OCL fully_connected (默认 fallback)          │
│      └─ ... (dynamic shape 等)                       │
│  → 选出 primary impl → _impl                         │
└──────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────┐
│ Runtime Execution Stage                               │
│  ├─ primitive_inst::update_impl()                     │
│  │   └─ enable_multi_impl_mode() ← 首次触发           │
│  │       → 构建 ImplPool {onednn, ocl}               │
│  │                                                    │
│  ├─ primitive_inst::execute()                         │
│  │   ├─ select_best_impl_for_inputs()                │
│  │   │   ├─ shape-cache 快速路径 (O(1))              │
│  │   │   ├─ extract_selection_criteria()              │
│  │   │   └─ evaluate_best_impl_type()                │
│  │   │       ├─ threshold 规则: M*K > T → OneDNN     │
│  │   │       └─ threshold 规则: M*K ≤ T → OCL        │
│  │   ├─ switch_impl_to(best)                         │
│  │   │   ├─ 交换 _impl 指针                          │
│  │   │   ├─ update_weights() (权重同步)              │
│  │   │   └─ set_arguments() (重绑定参数)             │
│  │   └─ _impl->execute()                             │
│  │                                                    │
│  └─ update_impl_statistics() (PROFILING 模式)        │
└──────────────────────────────────────────────────────┘
```

### 1.2 核心数据结构

```cpp
// primitive_inst.h 中新增的结构
struct ImplPool {
    std::unordered_map<impl_types, std::shared_ptr<primitive_impl>> impls;
    impl_types active_impl_type = impl_types::any;
    std::unordered_map<impl_types, ImplStats> stats;    // EMA 统计
    mutable ov::PartialShape cached_shape;               // shape-cache 快速路径
    mutable impl_types cached_impl = impl_types::any;
};

struct WorkloadPredictor {
    static constexpr size_t HISTORY_SIZE = 8;
    std::deque<size_t>     workload_history;   // M*K 历史
    std::deque<impl_types> impl_history;
    bool detect_phase_change() const;          // >2x 变化检测
    impl_types predict_next_impl() const;      // 连续3次相同 → 预测
};

enum class ImplSwitchingPolicy { NONE, AUTO_HEURISTIC, MANUAL, PROFILING };

struct ImplSelectionCriteria {
    size_t batch_size, seq_length, m_dimension, k_dimension;
    size_t compute_workload;   // M * K
    bool   is_prefill;         // seq_length > 1
};
```

### 1.3 环境变量控制

| 变量 | 值 | 说明 |
|------|----|------|
| `OV_GPU_MULTI_IMPL_POLICY` | `NONE` / `AUTO_HEURISTIC` / `PROFILING` / `MANUAL` | 控制切换策略 |
| `OV_GPU_MULTI_IMPL_COMPUTE_THRESHOLD` | 整数, 默认 0 (auto) | 工作量阈值，-1 禁用 |

---

## 2. 原始问题分析

### 问题 1：Registry 静态优先级 — OneDNN 始终获胜

**现象**：在 LLM decode 阶段 (M=1, batch=1)，OneDNN 作为 Registry 中的第一个 impl 始终被选中，  
即使 OCL GEMV kernel 在该场景下更适合。

**根因**：`Registry<fully_connected>::get_implementations()` 中 OneDNN 排在首位，  
`get_primitive_impl_for_params()` 采用 first-match 策略，不考虑运行时工作负载特征。

```cpp
// fully_connected_impls.cpp — 注册顺序 = 选择顺序
OV_GPU_CREATE_INSTANCE_ONEDNN(...)  // ← 始终第一个匹配
OV_GPU_CREATE_INSTANCE_OCL(...)     // ← 永远不会被选中
```

**影响**：decode 延迟 13.36 ms/tok（OneDNN），而专用 OCL GEMV 有潜力更低。

### 问题 2：Binary OneDNN Gate — 缺乏工作负载感知

**现象**：`layout_optimizer` 中的 OneDNN 开关是全局的二值控制，  
没有按 primitive 或按 workload 维度的细粒度控制。

```cpp
bool has_all_enabled_onednn_impls_optimization_attribute() {
    return is_enabled_onednn_for<concatenation>() &&
           is_enabled_onednn_for<convolution>() && ...;  // 全部启用或全部禁用
}
```

### 问题 3：编译期静态决策 — 无运行时反馈

**现象**：impl 选择在 graph compilation 阶段一次性完成，后续推理无论工作负载如何变化，  
都使用同一个 impl。对于 LLM 的 Prefill/Generate 两阶段模式尤其不利。

---

## 3. 解决方案一：Runtime Implementation Switching

### 3.1 设计思路

在 `primitive_inst` 中引入 **ImplPool**，在编译时构建包含多个 impl 的池，  
在执行时根据输入 shape 动态选择最优 impl。

### 3.2 核心函数调用链

```
primitive_inst::update_impl()                           # 首次 impl 更新时
  ├─ _impls_factory->get_primitive_impl_for_params()    # 标准选择 → primary (OneDNN)
  └─ enable_multi_impl_mode(policy)                     # 构建 impl pool
       ├─ Step 1: 确定 primary type (onednn/ocl/sycl/cm)
       ├─ Step 2: 收集候选 type (has_impl_for 校验)
       ├─ Step 3: 逐候选编译 + Weight IO Contract 检查
       │   ├─ create_impl_for_type() 或 M=1 重试
       │   ├─ Rule 1: reorder flag 一致
       │   ├─ Rule 2: reorder source layout 一致
       │   └─ Rule 3: sub-byte 跨后端兼容
       └─ add_impl_to_pool(type, impl)

primitive_inst::execute()                               # 每次推理时
  ├─ select_best_impl_for_inputs(params)
  │   ├─ shape-cache 命中 → 直接返回 (O(1))
  │   ├─ extract_selection_criteria()
  │   │   └─ 从 input shape 提取 {batch, seq, M, K, workload, is_prefill}
  │   └─ evaluate_best_impl_type(criteria)
  │       ├─ PROFILING: EMA 比较 → 选更快的
  │       ├─ is_prefill && workload > threshold → OneDNN
  │       ├─ !is_prefill && workload ≤ threshold → OCL
  │       └─ fallback → OneDNN
  ├─ switch_impl_to(best)
  │   ├─ _impl = pool[best]
  │   ├─ update_weights()       # 权重 reorder 同步
  │   ├─ set_arguments()        # 重绑定 kernel 参数
  │   └─ GPU_DEBUG_TRACE: "impl switch onednn → ocl"
  └─ _impl->execute()
```

### 3.3 Threshold 计算

```cpp
// evaluate_best_impl_type() 中的 threshold 逻辑
if (raw_threshold == 0) {          // auto-detect
    threshold = gpu_fp16_gflops * 10;
    // 例如 B580: ~460 GFLOPS → threshold ≈ 4600
    // generate M=1, K=4096: workload = 4096 ≤ 4600 → OCL ✓
    // prefill  M=32, K=4096: workload = 131072 > 4600 → OneDNN ✓
} else if (raw_threshold < 0) {    // 禁用 threshold
    // 跳过 threshold 规则，fallback → OneDNN
} else {
    threshold = raw_threshold;     // 用户指定
}
```

### 3.4 Shape-Cache 快速路径

```cpp
// select_best_impl_for_inputs() — O(1) decode 路径
if (cached_impl != any && input_shape == cached_shape)
    return cached_impl;    // decode 阶段所有 token shape 相同 → 直接命中
```

这确保 decode 阶段（所有 token 的 M=1 shape 相同）不会重复执行 `extract_selection_criteria` 和 `evaluate_best_impl_type` 的完整逻辑。

---

## 4. 解决方案二：Sub-byte Weight 兼容性机制

### 4.1 问题

OneDNN 和 OCL 都直接读取 raw u4/i4 packed weight bytes（不做 weight reorder）。  
**Weight IO Contract** 的 Rule 3 原本无条件拒绝 sub-byte 跨后端共享：

```
Rule 3 (原始): !reorder + sub-byte dtype + 不同后端 → 拒绝
```

但事实上，OneDNN WOQ FC 和我们的 OCL GEMV kernel 都使用 **低 nibble 优先 (low-nibble-first)** 打包约定，可以安全共享同一 weight buffer。

### 4.2 解决方案

引入虚方法 `raw_sub_byte_weight_compatible()`，允许 impl manager 声明其 sub-byte 读取约定：

```cpp
// implementation_manager.hpp — 基类默认返回 false
virtual bool raw_sub_byte_weight_compatible() const noexcept { return false; }

// fc_compressed_generate_opt.hpp — OCL GEMV 声明兼容
bool raw_sub_byte_weight_compatible() const noexcept override { return true; }

// fully_connected_onednn.hpp — OneDNN FC 声明兼容
bool raw_sub_byte_weight_compatible() const noexcept override { return true; }
```

Rule 3 修改为条件拒绝：

```cpp
// primitive_inst.cpp — enable_multi_impl_mode() 中
} else if (weight_is_sub_byte && t != primary_type) {
    const bool primary_raw_ok = primary_mgr && primary_mgr->raw_sub_byte_weight_compatible();
    const bool alt_raw_ok = alt_impl->m_manager && alt_impl->m_manager->raw_sub_byte_weight_compatible();
    if (!(primary_raw_ok && alt_raw_ok))
        reject_reason = "rule3: sub-byte weight cross-backend packing incompatible";
    // 两边都声明兼容 → Rule 3 通过 ✓
}
```

### 4.3 m_manager 为 null 的边界情况

当 primary impl 通过 legacy 路径创建（graph optimizer 直接选择，未经过 `ImplementationManager::update_impl`），  
`_impl->m_manager` 可能为 nullptr。  

解决方案：fallback 到 `m_available_impls` 中查找同类型的第一个 manager：

```cpp
auto find_first_manager_of_type = [&](impl_types type) -> const ImplementationManager* {
    for (const auto& mgr : _impls_factory->m_available_impls)
        if (mgr->get_impl_type() == type)
            return mgr.get();
    return nullptr;
};
const ImplementationManager* primary_mgr =
    _impl->m_manager ? _impl->m_manager : find_first_manager_of_type(primary_type);
```

---

## 5. 解决方案三：M=1 Decode 合成重试

### 5.1 问题

`enable_multi_impl_mode()` 在 **prefill 阶段**首次触发（M >> 1）。此时：

1. `create_impl_for_type(ocl, params_with_M=2048)` 调用
2. 内部 `get_fake_aligned_params_if_possible()` 可能进一步膨胀 M
3. `FCCompressedGenerateOpt::support_shapes(M=2048)` → **false**（M≠1）
4. Fallthrough 到默认 OCL FC impl
5. 默认 OCL FC 的 `raw_sub_byte_weight_compatible() = false`
6. Rule 3 拒绝

→ 即使修复了 Rule 3，错误的 impl 也会导致检查失败。

### 5.2 解决方案

在 `enable_multi_impl_mode()` 的 Step 3 循环中，对 sub-byte weight + 非 primary type 的候选，  
增加 **M=1 合成重试**：

```
enable_multi_impl_mode() Step 3 循环:
  for each candidate_type:
    ├─ create_impl_for_type(*_impl_params, type) → alt_impl
    ├─ 检查: weight_is_sub_byte && type != primary?
    │   ├─ alt_impl 是否存在且有 raw_sub_byte_weight_compatible()?
    │   │   ├─ 是 → 直接使用 (已找到正确的 OCL GEMV)
    │   │   └─ 否 → 需要 M=1 重试 ↓
    │   │
    │   └─ M=1 合成重试:
    │       ├─ 复制 _impl_params → m1_params
    │       ├─ 将激活 shape 的所有非末维 dim 设为 1
    │       │   [40, 1, 4096] → [1, 1, 4096]
    │       ├─ 同步输出 shape
    │       ├─ 清除 dynamic padding mask
    │       │
    │       ├─ (W4A8 特殊处理)
    │       │   若 dynamic_quantized_activation && 激活为 i8:
    │       │   将激活 dtype 改为 f16 (decode 时 dyn_quant 跳过)
    │       │
    │       ├─ 直接遍历 m_available_impls (绕过 fake alignment):
    │       │   for each manager of type:
    │       │     if validate(node) && support_shapes(m1_params):
    │       │       create → compile → check raw_sub_byte_compatible
    │       │       → if OK: alt_impl = m1_impl; break
    │       │
    │       └─ 找到 → alt_impl 替换为 M=1 编译的 GEMV kernel
    │
    ├─ Weight IO Contract (Rules 1-3) 检查
    └─ add_impl_to_pool(type, alt_impl)
```

### 5.3 为何绕过 `create_impl_for_type`

`create_impl_for_type()` 内部调用 `get_fake_aligned_params_if_possible()`，  
该函数会将 M=1 膨胀为 M=16 等值（用于 OneDNN 对齐优化）。  
这导致 `FCCompressedGenerateOpt::support_shapes(M=16)` 始终失败。

合成重试直接遍历 `m_available_impls`，使用精确的 M=1 参数，完全绕过 fake alignment。

---

## 6. 解决方案四：W4A8 Dynamic Quantization 处理

### 6.1 问题

当 OneDNN 使用 W4A8 模式时：
- Prefill 阶段：上游 `dynamic_quantize` 节点将 f16 激活转换为 i8
- Decode 阶段 (M=1, batch=1)：`dynamic_quantize` 节点**跳过执行**，直接传递原始 f16

Pool 构建发生在 prefill 阶段，此时 `_impl_params` 中激活为 i8。  
如果按 i8 编译 OCL kernel（IS_ACT_INT8=1），在 decode 阶段实际收到 f16 数据时会产生错误结果。

### 6.2 解决方案

在 M=1 合成重试中，检测 `dynamic_quantized_activation` 条件，  
主动将激活 dtype 从 i8 改为 f16：

```cpp
if (get_node().is_type<fully_connected>()) {
    const auto& fc_desc = *m1_params.typed_desc<fully_connected>();
    if (fc_desc.dynamic_quantized_activation &&
        m1_params.input_layouts[0].data_type == data_types::i8) {
        m1_params.input_layouts[0].data_type = data_types::f16;
    }
}
```

这确保 pool 中的 OCL GEMV kernel 以 W4A16 模式编译，  
正确接收 decode 阶段 `dynamic_quantize` 跳过后传递的 f16 激活。

### 6.3 触发条件分离

```cpp
// 两个独立的重试触发条件
const bool need_retry =          // 找到的 impl 不正确
    !alt_impl ||
    !(alt_impl->m_manager && alt_impl->m_manager->raw_sub_byte_weight_compatible());

bool need_dyn_quant_retry =      // 找到了正确 impl，但 dtype 不对
    !need_retry &&
    desc.dynamic_quantized_activation &&
    input_layouts[0].data_type == data_types::i8;
```

---

## 7. OCL GEMV Kernel 优化详情

### 7.1 Kernel 架构

```
gemm_generate_opt.cl — 双分支设计
├─ Branch A: f16/f16 GEMM (#if !IS_WEIGHT_INT4)
│   小型 GEMV，每个 work-item 计算一个输出
│
└─ Branch B: INT4 WOQ GEMV (#if IS_WEIGHT_INT4)
    ├─ B1: W4A16 (#if !IS_ACT_INT8)
    │   ├─ SLM 缓存激活: 整个 K 向量 → __local
    │   ├─ UNPACK8_UINT: uint 读取 + 位操作解包
    │   ├─ 双 float8 累加器 (ILP)
    │   ├─ opencl_unroll_hint(4)
    │   └─ TILE_N=1 (每 WI 一个输出) / TILE_N=2 (每 WI 两个输出)
    │
    └─ B2: W4A8 (#if IS_ACT_INT8)
        ├─ 激活为 i8，追加 per-token activation scale
        └─ 结构类似 B1，末尾乘 act_scale
```

### 7.2 关键优化技术

| 技术 | 效果 | 状态 |
|------|------|------|
| **UNPACK8_UINT**: 单次 uint 读取 + 位移解包 8 个 INT4 | +15% BW vs uchar4 逐字节读取 | ✅ 已部署 |
| **双 float8 累加器 (ILP)**: `acc0`, `acc1` 交替使用 | +8% BW，保持 EU 流水线满载 | ✅ 已部署 |
| **N-parallel 高 occupancy 分派**: 每个 WI 计算 1 个输出通道，无 barrier/SLM | 最大化 EU occupancy, B580 1280 thread slots 满载 | ✅ 最终部署 |
| **SLM 激活缓存**: WG 内共享 K 向量 | 减少全局内存读取，当多个 subgroup 共享激活时收益大 | ✅ 曾部署(后被 N-parallel 取代) |
| **opencl_unroll_hint(4)**: 内层循环 | +5% vs 无提示 | ✅ 已部署 |
| **Two-pass K-split**: 分 K_SPLIT 个 WG + reduce | +38–74% 小 N 场景 | ⚠ 原型阶段 |
| **uchar4 vs uint**: 128-bit 读取对比 | uint 更优（减少 transaction） | ✅ 选择 uint |
| **TILE_N=2**: 每 WI 两个输出 | −30%（寄存器压力） | ❌ 放弃 |
| **half8/half16 累加器**: 减半精度 | −20%（精度相关停顿） | ❌ 放弃 |
| **ulong (8-byte) 读取**: 对齐访问 | 无明显收益 | ❌ 放弃 |
| **Full unroll(8)**: 完全展开 | −10%（寄存器溢出） | ❌ 放弃 |

### 7.3 JIT 常量生成 (fc_compressed_generate_opt.cpp)

```
固定常量:                        动态常量 (从 runtime params 推导):
  SG_SIZE = 16                     K_SIZE, N_SIZE, B_SIZE
  VEC_SIZE = 8                     IS_WEIGHT_INT4 = 1
  WG_SIZE = 256                    WEIGHT_IS_SIGNED (i4=1, u4=0)
                                   HAS_ZP, ZP_IS_U8
                                   GROUP_SIZE, NUM_GROUPS = K/GROUP_SIZE
                                   IS_ACT_INT8 (W4A8=1, W4A16=0)

Dispatch: global = {ceil(N/WG_SIZE) × WG_SIZE, B, 1}
          local  = {WG_SIZE, 1, 1}
```

### 7.4 独立 Kernel Benchmark 结果

**硬件**: Intel Arc B580 (BMG, Xe2) — 20 XC, 456 GB/s GDDR6, SG_SIZE=16

#### 穷举变体对比 (N=6144, K=4096)

| 变体 | 平均 GB/s | 峰值 GB/s | 占 456 GB/s |
|------|----------|----------|------------|
| Raw BW 上限 (最少计算) | 421 | 435 | **95.4%** |
| float4×4 累加器 | 368 | 373 | 81.8% |
| **K16 双 float8 累加器 (已部署)** | **366** | **371** | **81.4%** |
| uint + unroll(4) baseline | 341 | 347 | 74.8% |
| ulong 读取 | 341 | 346 | 75.9% |
| half8 累加器 | 286 | 290 | 63.7% |
| TILE_N=2 | 235 | 251 | 55% |

#### 按 Qwen3-8B FC 层形状拆分

| FC 层 | N | K | WG 数 | 带宽 GB/s | Roofline % | 状态 |
|--------|---|----|------|----------|-----------|------|
| gate_proj / up_proj | 12288 | 4096 | 48 | 391 | **88.6%** | ✅ >85% 达标 |
| qkv_proj | 6144 | 4096 | 24 | 366 | 81.4% | ⚠ 接近 |
| o_proj | 4096 | 4096 | 16 | 248 | 64.8% | ❌ GPU 利用不足 |
| down_proj | 4096 | 12288 | 16 | 206 | 46.0% | ❌ 严重利用不足 |

**根因**：N=4096, WG_SIZE=256 → 仅 16 个 WG，而 B580 有 20 个 XC core → 20% 算力闲置。

#### K-Split 原型结果 (two-pass)

| FC 层 | 配置 | WG 总数 | 带宽 GB/s | Roofline % | 提升 |
|--------|------|---------|----------|-----------|------|
| **down_proj** | K_SPLIT=4, WG128 | 64 | **347** | **76.2%** | **+71%** |
| **o_proj** | K_SPLIT=4, WG128 | 64 | **377** | **82.7%** | **+74%** |
| **qkv_proj** | K_SPLIT=2, WG256 | 48 | **361** | **79.2%** | **+38%** |
| gate_proj | K_SPLIT=1, WG256 | 48 | 387 | 84.8% | baseline |

---

## 8. CM GEMV Kernel 实验

### 8.1 实验动机

OCL N-parallel GEMV (WG_SIZE=256, VEC_SIZE=8) 达到 16.45ms/token，距 OneDNN baseline (15.0ms) 仍有 ~10% 差距。
尝试使用 **CM (C for Metal)** 编译器编写同一 GEMV kernel，测试是否能通过 CM 编译器更紧密的 ISA 生成获得性能提升。

CM 是 Intel GPU 上的高级 kernel 语言，通过 `-cmc` flag 编译，可直接访问:
- LSC (Load/Store Cache) 块读写指令
- DPAS/XMX 系统阵列指令
- 更精细的寄存器和线程控制

### 8.2 实现架构

CM kernel 完全复用 OCL 版本的 N-parallel 框架（PrimitiveImplCM = PrimitiveImplOCL），作为新增 impl 注册在 OneDNN 之后、OCL 之前：

```
Registry 注册顺序:
  OV_GPU_CREATE_INSTANCE_ONEDNN(...)
  OV_GPU_CREATE_INSTANCE_CM(cm::FCCompressedGenerateOptCM, ...)   ← 新增
  OV_GPU_CREATE_INSTANCE_OCL(ocl::FCCompressedGenerateOpt, ...)
```

**关键设计决策**:
- CM impl 注册为 `impl_types::ocl`（非 `impl_types::cm`），因为 `evaluate_best_impl_type()` 只理解 onednn/ocl 两种类型
- `validate_impl()` 增加 `check_cm_jit_support()` + `supports_immad` + `get_use_cm()` 三重门控
- 运行 CM kernel 需设置环境变量 `CM_FE_DIR` 指向 `libclangFEWrapper.so` 所在目录

**文件清单**:

| 文件 | 用途 |
|------|------|
| `impls/cm/fc_compressed_generate_opt.cm` | CM kernel 源码 (W4A16 + W4A8) |
| `impls/cm/fc_compressed_generate_opt.cpp` | Host 端 Generator + Impl 类 |
| `impls/cm/fc_compressed_generate_opt.hpp` | ImplementationManager (validate + support_shapes) |
| `registry/fully_connected_impls.cpp` | 注册 CM impl (`#if OV_GPU_WITH_CM`) |

### 8.3 优化迭代与性能对比

在 B580 上以 Qwen3-8B (INT4) 为基准测试，需预设 `export CM_FE_DIR=/mnt/river`:

| 版本 | WG_SIZE | 内存访问方式 | 2nd token (ms) | FC 分割 | 说明 |
|------|---------|-------------|----------------|---------|------|
| **OneDNN baseline** | — | DPAS 系统阵列 | **15.0–15.1** | 181/0 | 参考基准 |
| OCL N-parallel | 256 | `vload8` + sub-group coalesce | **16.45** | 109/72 | 最优 OCL |
| CM v1 (scalar) | 16 | 逐元素指针解引用 | **76.27** | 109/72 | 5× 慢于 OneDNN |
| CM v2 (block_read, CHUNK=32) | 16 | `cm_svm_block_read<uint,16>` + 宽向量 | **29.76** | 109/72 | 2.5× 改进 |
| CM v3 (block_read, VEC=8) | 64 | `cm_svm_block_read<uint,4>` + 轻量向量 | **46.22** | 109/72 | WG 过大反而更慢 |

#### 关键优化步骤:

**v1 → v2 (scalar → block_read, 2.5× 提升)**:
- 将逐元素的 `A_row[k+j]` 替换为 `cm_svm_block_read<uint, 16>(...)` 批量读取 32 个 f16
- 将逐字节的 weight 读取替换为 `cm_svm_block_read<uint, 4>(...)` 批量读取 32 个 nibble
- 使用 CM 向量运算 `vector<float,32> prod = af * wf` + 层次化归约替代标量循环

**v2 → v3 (测试 WG_SIZE 影响)**:
- 降低到 VEC_SIZE=8 (vector<float,8>) 以减少寄存器压力
- 尝试 WG_SIZE=64，但性能反而从 29.76ms 恶化到 46.22ms
- 原因: CM 编译器 per-thread 寄存器使用远高于 OCL，WG_SIZE=64 导致硬件无法调度足够线程

### 8.4 CM vs OCL 性能差距根因分析

CM kernel 最优结果 (29.76ms) 仍为 OneDNN 的 **2×** 和 OCL 的 **1.8×**，根因:

| 因素 | 影响 | 详细说明 |
|------|------|---------|
| **WG_SIZE 被限制在 16** | 主要 | CM 编译器 per-thread 寄存器占用高 → 硬件无法调度 >16 线程/WG。OCL 可达 256 |
| **无 sub-group coalescing** | 主要 | OCL 的 `vload8` 通过 sub-group 隐式合并跨线程的相邻内存访问。CM 的 `cm_svm_block_read` 是 per-thread 独立 SVM 读取，无法跨线程合并 |
| **JIT 编译开销** | 附加 | CM JIT 首次编译 ~2.5 分钟（72 层 × ~2s/层），不适合生产环境。OCL JIT < 1s |
| **Occupancy 差距** | 次要 | B580 有 1280 thread slots，OCL WG=256 可填满 5 个 WG；CM WG=16 需 80 个 WG 才等效 |

### 8.5 结论

**CM 对 GEMV (M=1) memory-bound 场景没有优势**:
- CM 的核心价值在于 **DPAS/XMX** 指令加速 compute-bound 运算（需 M≥8 的 tile 才能利用系统阵列）
- 对于 M=1 的 GEMV，瓶颈是内存带宽，不是算力。OCL 的 sub-group 隐式 coalescing 比 CM 的显式 SVM block read 更高效
- CM kernel 仅适用于有 CM JIT 支持的平台 (`CM_FE_DIR` + `libclangFEWrapper.so`)
- 后续优化应继续聚焦 OCL kernel 或直接研究 OneDNN GEMV 实现

---

## 9. E2E 验证结果与分析

### 9.1 Qwen3-8B Decode 性能

**配置**: 256 输出 token, 6-token 输入, 3 次迭代 (排除 warmup)

| 配置 | 2nd Token (ms/tok) | 吞吐 (tok/s) | 对比 baseline |
|------|-------------------|-------------|--------------|
| **Baseline (NONE/OneDNN)** | **13.36** | **74.87** | — |
| AUTO_HEURISTIC (OCL for generate) | 14.26–14.35 | 69.70–70.14 | **−6.9%** |
| Force OneDNN (threshold=−1) | 13.30 | 75.19 | +0.4% |
| PROFILING 模式 | 14.32 | 69.81 | −6.7% |
| N 阈值 ≥6144 (已回退) | 14.26 | 70.14 | −6.5% |

**补充测试** (2048 token 输入, 32 输出 token, 1 次迭代):

| 配置 | 2nd Token (ms/tok) | 吞吐 (tok/s) | FC 分割 |
|------|-------------------|-------------|---------|
| **Baseline (NONE/OneDNN)** | **15.14** | **66.04** | 181/0 |
| AUTO_HEURISTIC (OCL) | 16.45–16.57 | ~60 | 109/72 |
| AUTO_HEURISTIC (CM v2 best) | 29.76 | 33.60 | 109/72 |

### 9.2 正确性验证

- ✅ 所有配置输出有效文本 (MD5 校验)
- ✅ 无 ",,," / "!!!" / "???" 伪影
- ✅ "impl switch onednn → ocl" 日志确认切换
- ✅ "Enqueue stage gemm_generate_opt" 确认 kernel 选中
- ✅ Layer 0 q_proj 的 input/output 对比在容差范围内

### 9.3 E2E 回退根因分析

| 因素 | 贡献 | 说明 |
|------|------|------|
| **DPAS/XMX 指令优势** | 主要 | OneDNN 使用 DPAS 系统阵列处理 ~128 INT4/指令 vs OCL 向量 EU ~16/迭代 |
| **Per-dispatch 开销差异** | 次要 | OCL 路径每次 FC dispatch ~8μs 开销 × 180 次 = ~1.4ms/token |
| **小 N GPU 利用不足** | 次要 | N=4096 的 FC 层 (o_proj, down_proj) 仅 16 WG / 20 XC |

**关键发现**：框架切换开销**不是**瓶颈。threshold=−1 测试 (强制 OneDNN 但走 AUTO_HEURISTIC 通路) 给出 13.30ms ≈ NONE baseline，证明 ImplPool 机制本身开销 < 0.5%。

---

## 10. 当前未解决问题

### 10.1 OCL/CM 尚未超越 OneDNN (SKILL.md 要求未达成)

**要求**: "OCL implementation has better performance than OneDNN implementation in generate stage"  
**现状**: OCL 比 OneDNN 慢 ~7%，CM 比 OneDNN 慢 ~100%

**已排除的方案**:

| 方案 | 结果 | 原因 |
|------|------|------|
| **CM GEMV Kernel** | ❌ 29.76ms vs 15.14ms (2× 慢) | CM 无 sub-group coalescing，WG_SIZE 被限制在 16，memory-bound 场景无优势 |

**后续可行方案** (按优先级):

| 方案 | 预期影响 | 复杂度 |
|------|---------|--------|
| **研究 OneDNN GEMV 实现**: 分析 OneDNN 的 W4A16 GEMV 具体 ISA 指令，找出其优于 OCL 的关键技术 | 定位精确的差距来源 | 低 |
| **K-Split 框架集成**: 在 PrimitiveImplOCL 中支持两阶段 dispatch (GEMV + reduce) | 小 N 带宽 +38–74%，可能缩小 E2E 差距但不确定能超过 OneDNN | 中 |
| **DPAS Kernel**: 使用 `cl_intel_subgroup_matrix_multiply_accumulate` 指令重写 | 理论 2–4× 指令数减少 → 可能超越 OneDNN | 高 |
| **Dispatch 开销降低**: kernel fusion / persistent kernel | 减少 ~1.4ms/token 开销 | 高 |

### 10.2 Memory 优化 (SKILL.md Phase 2/3 未实现)

- Hot-Swap Memory Management（仅设计，未编码）
- Resource Pooling（仅概念验证）
- Lazy Compilation（未实现）
- LRU Weights Cache（未实现）

### 10.3 TILE_N 动态选择

当前 JIT 不设置 TILE_N（kernel 默认为 1）。当 N 非常大时，TILE_N=2 可能有收益，  
但当前测试表明寄存器压力导致性能下降 30%。

---

## 11. 文件清单与代码位置索引

### 11.1 Production 代码 (13 files + 3 CM files, +2143 / −2 lines)

| 文件 | 行数变化 | 核心作用 |
|------|---------|---------|
| `primitive_inst.cpp` | +917 | 调度引擎: `enable_multi_impl_mode`, `evaluate_best_impl_type`, `switch_impl_to`, M=1 重试, Weight IO 规则 |
| `primitive_inst.h` | +90 | `ImplPool`, `WorkloadPredictor`, `ImplSelectionCriteria`, API 声明 |
| `gemm_generate_opt.cl` | +384 | GEMV kernel (f16/f16 + W4A16 + W4A8), UNPACK8_UINT, 双累加器, SLM |
| `fc_compressed_generate_opt.cpp` | +281 | JIT 生成器: WG_SIZE=256, SG_SIZE=16, 动态推导 K/N/B/GROUP_SIZE |
| `fc_compressed_generate_opt.hpp` | +177 | `FCCompressedGenerateOpt` IM: validate, support_shapes, raw_sub_byte |
| `gemm_generate_opt.cpp` | +125 | 非压缩 GEMM JIT 路径 |
| `gemm_generate_opt.hpp` | +139 | 非压缩 GEMM 验证 |
| `fully_connected_onednn.hpp` | +5 | `raw_sub_byte_weight_compatible() = true` |
| `cm/fc_compressed_generate_opt.cm` | +230 | CM GEMV kernel (W4A16 + W4A8), cm_svm_block_read 向量化加载 |
| `cm/fc_compressed_generate_opt.cpp` | +214 | CM Generator + PrimitiveImplCM 实现 |
| `cm/fc_compressed_generate_opt.hpp` | +132 | CM ImplementationManager (cm_jit + immad 门控) |
| `fully_connected_impls.cpp` | +9/−2 | FCCompressedGenerateOpt 注册 (OneDNN 之后, CM 和 OCL 之前) |
| `gemm_impls.cpp` | +6 | GEMM OCL impl 注册 |
| `implementation_manager.hpp` | +9 | 虚方法 `raw_sub_byte_weight_compatible()` |
| `internal_properties.hpp` | +3 | 属性声明 |
| `options.inl` | +3 | 选项注册 |

### 11.2 测试代码 (gemv_test/w4a16/ — 24 files)

| 文件 | 用途 |
|------|------|
| `w4a16_test.cpp` | 主测试框架: 500 warmup + 500 timed, wall-clock + CL profiling |
| `w4a16_gemv.cl` | 最佳独立 kernel (与 production 匹配) |
| `bw_bench.cpp` + `bw_test.cl` | Raw 带宽上限测试 |
| `ksplit_bench.cpp` + `w4a16_ksplit.cl` | Two-pass K-split benchmark |
| `ksplit_intra_bench.cpp` + `w4a16_ksplit_intra.cl` | Intra-WG K-split (SLM reduce) |
| 多个 `w4a16_gemv_*.cl` | 各 kernel 变体 (block, f4acc, fullunroll, h16acc, kpar, nounroll, ulong, wg512) |

### 11.3 Git 提交历史

```
380023493b Add FCCompressedGenerateOpt OCL W4A8 kernel implementation
cf386457bb Debug ocl_gemm kernel
c6265d84fe Add i8 support for gemm_ocl
fa43e17dac Fix gemm ocl impl cannot be chosen issue
610b56489a Add ocl type gemm_impl
6c6db79d2a Add_impl_pool policy for different impls
fbfcc26721 Runtime Primitive Implementation Switching Policy
```

---

## 12. 详细流程图

> 详细的流程图以 Mermaid 和 SVG 格式保存在同目录下的独立文件中：
> - `flowchart_enable_multi_impl.md` — enable_multi_impl_mode() 完整流程
> - `flowchart_runtime_switching.md` — 运行时 impl 切换决策流程
> - `flowchart_overall_architecture.md` — 整体架构与数据流

### 12.1 enable_multi_impl_mode() 流程 (简化)

```
enable_multi_impl_mode(policy)
│
├─ _switching_policy != NONE?  ─── Yes → return (已初始化)
│
├─ Step 1: 确定 primary impl type
│   └─ _impl->m_manager ? manager->get_impl_type()
│                         : (is_onednn ? onednn : ocl)
│
├─ Step 2: 收集候选 type
│   for each mgr in m_available_impls:
│     └─ GPU-native? → not seen? → has_impl_for(node, type, static)?
│                                   → candidate_types.push_back(type)
│
├─ MANUAL policy? → parse_manual_impl → no rule? → return
│
├─ OneDNN fused post-ops? → non-OneDNN 无法处理 → return
│
├─ add_impl_to_pool(primary_type, _impl)
│
└─ Step 3: for each candidate_type:
    ├─ create_impl_for_type(params, type)
    │
    ├─ Sub-byte weight + type ≠ primary?
    │   ├─ need_retry? (alt不存在 / 非raw-sub-byte-compatible)
    │   │   └─ YES → M=1 合成重试 (绕过 fake alignment)
    │   ├─ need_dyn_quant_retry? (W4A8 dtype 修正)
    │   │   └─ YES → M=1 合成重试 + f16 dtype
    │   └─ 找到正确的 FCCompressedGenerateOpt
    │
    ├─ Weight IO Contract 检查:
    │   ├─ Rule 1: reorder flag 一致
    │   ├─ Rule 2: reorder source layout 一致
    │   └─ Rule 3: sub-byte 兼容性 (raw_sub_byte_weight_compatible)
    │
    └─ add_impl_to_pool(type, alt_impl)
```

### 12.2 运行时切换决策流程 (简化)

```
execute()
│
├─ _switching_policy != NONE && _impl_pool != null?
│   │
│   YES → select_best_impl_for_inputs(params)
│   │   ├─ shape-cache 命中? → return cached_impl (O(1))
│   │   │
│   │   ├─ extract_selection_criteria():
│   │   │   batch_size, seq_length, M, K, compute_workload, is_prefill
│   │   │
│   │   └─ evaluate_best_impl_type():
│   │       ├─ PROFILING + 足够采样? → 选 EMA 更快的
│   │       ├─ is_prefill && workload > threshold? → OneDNN
│   │       ├─ !is_prefill && workload ≤ threshold? → OCL
│   │       ├─ workload > 2×threshold? → OneDNN
│   │       └─ fallback → OneDNN
│   │
│   ├─ best ≠ current?
│   │   └─ switch_impl_to(best):
│   │       ├─ _impl = pool[best]
│   │       ├─ update_weights()
│   │       ├─ OOO queue? → discard null event
│   │       ├─ set_arguments()
│   │       └─ log: "impl switch prev → best"
│   │
│   └─ OOO queue + actually switched? → stream.finish()
│
├─ PROFILING? → record start time
│
├─ _impl->execute()
│
└─ PROFILING? → record end time → update EMA stats
```
