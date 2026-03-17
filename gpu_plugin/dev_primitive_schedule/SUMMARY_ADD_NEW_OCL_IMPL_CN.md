# 新增 OCL GEMV Impl 及使其进入 impl_pool 的全过程总结

## 目标

在 LLM 推理的 **decode 阶段**（token-by-token, M=1），`fully_connected` 节点（压缩权重 INT4）当前由 OneDNN 执行。OneDNN 在处理小批量矩阵时效率偏低，而专门为 M=1 优化的 GEMV OCL kernel 可以显著降低延迟。

目标：新增 `FCCompressedGenerateOpt`（OCL GEMV WOQ kernel），并使其与 OneDNN impl 共存于同一 `impl_pool` 中，由调度器在 decode 阶段动态切换。

---

## 第一步：新建 OCL GEMV Impl

### 核心设计原则

在 `ocl_v2` 框架中，每个新 impl 由三个部分组成：

| 层次 | 类 | 职责 |
|------|----|------|
| `ImplementationManager` 子类 | `FCCompressedGenerateOpt` | 注册、`validate_impl()`、`support_shapes()`、`create_impl()` |
| `PrimitiveImplOCL` 子类 | `FCCompressedOptImpl` | 持有 Stage、`execute()`、序列化 |
| `KernelGenerator` 子类 | `FCCompressedOptGenerator` | JIT 常量生成、dispatch 维度计算 |

### 文件结构

```
src/plugins/intel_gpu/src/graph/impls/ocl_v2/gemm/
├── gemm_generate_opt.cl          ← 复用的 .cl 模板（双分支设计）
├── gemm_generate_opt.hpp         ← f16/f16 GEMV（gemm 原语）
├── gemm_generate_opt.cpp
├── fc_compressed_generate_opt.hpp ← INT4 WOQ GEMV（fully_connected 原语）
└── fc_compressed_generate_opt.cpp
```

### `validate_impl` vs `support_shapes` 分工

```
validate_impl(node)          ← 编译期，基于 program_node 固定属性
  - compressed_weights == true
  - 激活 dtype == f16
  - 权重 dtype == u4 或 i4
  - scale dtype == f16（bias 存在时 index=3，否则 index=2）
  - !dynamic_quantized_activation
  - 不检查 M 维度（M 随推理请求变化）
  - 不检查激活 format（运行时 update_impl_params 会 reshape 到 bfyx）

support_shapes(params)       ← 运行期，基于实际 kernel_impl_params
  - 激活 shape[rank-2] == 1  （M=1，generate 阶段）
  - K % 128 == 0             （GROUP_SIZE = SG_SIZE(16) × VEC_SIZE(8)）
```

### `gemm_generate_opt.cl` 双分支设计

```c
// 原有 f16/f16 GEMV 分支
#if !IS_WEIGHT_INT4
__kernel void gemm_generate_opt(...) { /* f16 × f16 */ }
#endif

// 新增 INT4 WOQ GEMV 分支
#if IS_WEIGHT_INT4
// UNPACK8 宏：从 uchar4 解包 8 个 nibble → float8
// 外层循环：遍历 NUM_GROUPS 个量化组，加载 scale 和可选 ZP
// 内层循环：vload8 激活 + vload4 权重字节 + UNPACK8 + mad 累加
// 写回：TO_OUTPUT_TYPE(acc)
__kernel void gemm_generate_opt(...Scale, [ZP,] C) { /* INT4 WOQ GEMV */ }
#endif
```

### JIT 常量（`FCCompressedOptGenerator`）

| 常量 | 值 | 说明 |
|------|----|------|
| `IS_WEIGHT_INT4` | 1 | 选择 INT4 分支 |
| `WEIGHT_IS_SIGNED` | 0/1 | i4=1, u4=0 |
| `HAS_ZP` | 0/1 | 是否有 zero point |
| `GROUP_SIZE` | 128 | 量化组大小 |
| `NUM_GROUPS` | K/128 | 组数 |
| `K_SIZE`, `N_SIZE`, `B_SIZE` | 运行时 | 张量维度 |

### 注册（`fully_connected_impls.cpp`）

```cpp
// 注册顺序决定 create_impl_for_type 的查找顺序
OV_GPU_CREATE_INSTANCE_ONEDNN(FullyConnectedImplementationManager, static_shape)
OV_GPU_CREATE_INSTANCE_OCL(ocl::FCCompressedGenerateOpt, static_shape)  // ← 新增，在默认 OCL 之前
OV_GPU_GET_INSTANCE_OCL(fully_connected, static_shape)   // 默认 OCL fallback
OV_GPU_GET_INSTANCE_OCL(fully_connected, dynamic_shape)
```

---

## 第二步：使其进入 impl_pool（问题根因分析）

### impl_pool 机制

```
enable_multi_impl_mode()
  ↓ Step 1: 确定 primary impl type（通过 _impl->m_manager->get_impl_type()）
  ↓ Step 2: 收集候选 type（has_impl_for(node, t, static_shape)）
  ↓ Step 3: 对每个候选 type 调用 create_impl_for_type() 尝试编译
            ↓ 通过"Weight IO Contract"三条规则检查
            ↓ 通过 → add_impl_to_pool(t, impl)
```

### Weight IO Contract 三条规则

```
Rule 1: primary->need_weights_reorder() == alt->need_weights_reorder()
        → 两边都不 reorder 或都 reorder，否则不能共享 weight buffer

Rule 2: 两边都 reorder 时，reorder 的源 layout（dtype + format）必须相同

Rule 3: 两边都不 reorder + weight dtype 是 sub-byte（u4/i4）→
        默认拒绝跨后端共享（nibble packing 可能不兼容）
```

### 问题一：Rule 3 无条件拒绝 OneDNN + OCL 配对

**现象**：
```
type=ocl rejected (weight IO contract: rule3: sub-byte weight cross-backend packing incompatible)
```

**根因**：OneDNN WOQ FC 和 OCL GEMV 都不做 weight reorder（都直接读 raw packed bytes）。Rule 3 发现 `weight_is_sub_byte=true && t != primary_type` 就无条件拒绝，无论两边是否约定了相同的存储格式。

**事实上**：OneDNN 和我们的 OCL kernel 都使用 **低 nibble 优先（low-nibble-first）** 的打包约定，可以安全共享。

**修复**：在 `ImplementationManager` 基类新增虚方法，由双方 override 声明兼容性：

```cpp
// implementation_manager.hpp
virtual bool raw_sub_byte_weight_compatible() const noexcept { return false; }

// fc_compressed_generate_opt.hpp
bool raw_sub_byte_weight_compatible() const noexcept override { return true; }

// fully_connected_onednn.hpp
bool raw_sub_byte_weight_compatible() const noexcept override { return true; }
```

Rule 3 修改为条件拒绝：
```cpp
// primitive_inst.cpp
} else if (weight_is_sub_byte && t != primary_type) {
    const bool primary_raw_ok = _impl->m_manager &&
        _impl->m_manager->raw_sub_byte_weight_compatible();
    const bool alt_raw_ok = alt_impl->m_manager &&
        alt_impl->m_manager->raw_sub_byte_weight_compatible();
    if (!(primary_raw_ok && alt_raw_ok)) {
        reject_reason = "rule3: sub-byte weight cross-backend packing incompatible";
    }
}
```

### 问题二：pool 在 prefill 期间构建，M=1 的 OCL impl 无法加入

**现象**：Rule 3 修复后日志仍显示：
```
type=ocl rejected (weight IO contract: rule3: sub-byte weight cross-backend packing incompatible)
```

**根因**：`enable_multi_impl_mode()` 在**第一次** `update_impl()` 时调用，这发生在 **prefill 阶段**（M=2048 等）。此时：

1. `create_impl_for_type(*this, *_impl_params, ocl)` 在已有 M=2048 的 `_impl_params` 下运行
2. `FCCompressedGenerateOpt::support_shapes(M=2048)` → **false**（M≠1）→ 跳过
3. fallthrough 到默认 OCL FC legacy impl，该 impl 的 `raw_sub_byte_weight_compatible() = false`
4. Rule 3 仍然触发

Rule 3 的 `raw_sub_byte_weight_compatible` 检查相当于是对**错误的 OCL impl**（legacy fallback）查询，而非我们的 GEMV kernel。

**修复：M=1 decode 合成重试**

在 `enable_multi_impl_mode()` 的 Step 3 中，增加专门针对 sub-byte weight 的重试路径：

```cpp
// primitive_inst.cpp — enable_multi_impl_mode() Step 3 循环内
if (weight_is_sub_byte && t != primary_type) {
    const bool need_retry =
        !alt_impl ||
        (alt_impl->m_manager && !alt_impl->m_manager->raw_sub_byte_weight_compatible());

    if (need_retry && /* M > 1 condition */) {
        // 合成 M=1 params：将激活 shape 的倒数第二维折叠为 1
        kernel_impl_params m1_params = *_impl_params;
        m1_ps[arank - 2] = 1;
        m1_params.input_layouts[0].set_partial_shape(m1_ps);
        // 同步折叠输出 shape（保持 JIT 维度一致）
        m1_params.output_layouts[0].set_partial_shape(ops_with_m1);

        auto m1_impl = _impls_factory->create_impl_for_type(*this, m1_params, t);
        // 只接受真正支持 raw sub-byte 的 impl（即 GEMV WOQ kernel）
        if (m1_impl && m1_impl->m_manager &&
            m1_impl->m_manager->raw_sub_byte_weight_compatible()) {
            alt_impl = std::move(m1_impl);  // 采用合成的 decode impl
        }
    }
}
```

**逻辑**：pool 在 prefill 时构建，但其中包含了以 M=1 参数编译的 OCL GEMV kernel。当调度器在 decode 阶段（真实 M=1）决策切换到 `ocl` 时，池中已有该 kernel 可用。

### 问题三（潜在）：`validate_impl` 中的 format 检查阻断

**现象**（通过 diagnostic 打印排查）：prefill 阶段 program_node 的激活 layout format 可能是 `fs_b_yx_fsv32` 等非标准格式，而非 `bfyx`，导致我们的 `validate_impl` 中原有的 format 白名单检查失败。

**根因**：`update_impl_params()` 在运行时会将激活 reshape 为 2D `bfyx`，但 `validate_impl(node)` 使用的是编译期 `node.get_input_layout()` 的 format。

**修复**：移除 `validate_impl` 中的 format 检查，改在 `support_shapes()` 中（运行期）验证：

```cpp
// validate_impl 中删除：
// static constexpr std::array valid_fmts = {format::bfyx, format::any};
// if (!one_of(in0.format, valid_fmts)) return false;

// 运行期校验在 support_shapes() 中已通过 shape 维度隐式保证
```

---

## 最终数据流（修复后）

```
编译/prefill 阶段：
  update_impl() [M=2048]
    → _impl = OneDNN FC   (primary)
    → enable_multi_impl_mode()
        → validate_impl(node): FCCompressedGenerateOpt PASS
        → create_impl_for_type(ocl, M=2048): FCCompressed support_shapes FAIL
        → M=1 retry: create_impl_for_type(ocl, M=1): FCCompressed support_shapes PASS ✓
        → m1_impl.raw_sub_byte_weight_compatible() = true
        → Rule 3: primary_raw_ok=true && alt_raw_ok=true → PASS ✓
        → add_impl_to_pool(ocl, FCCompressedOptImpl[M=1])
      impl_pool = { onednn: FC_OneDNN, ocl: FCCompressedOptImpl }

decode 阶段：
  execute() [M=1]
    → should_switch_impl() → evaluate_best_impl_type() → ocl
    → switch_impl_to(ocl)
    → FCCompressedOptImpl::execute()  ← GEMV INT4 WOQ kernel ✓
```

---

## 涉及文件汇总

| 文件 | 改动内容 |
|------|---------|
| `impls/ocl_v2/gemm/gemm_generate_opt.cl` | 新增 `#if IS_WEIGHT_INT4` INT4 WOQ GEMV 分支 |
| `impls/ocl_v2/gemm/fc_compressed_generate_opt.hpp` | `FCCompressedGenerateOpt` IM，含 `raw_sub_byte_weight_compatible()=true` |
| `impls/ocl_v2/gemm/fc_compressed_generate_opt.cpp` | `FCCompressedOptImpl` + `FCCompressedOptGenerator` |
| `registry/implementation_manager.hpp` | 新增 `virtual raw_sub_byte_weight_compatible()` 虚方法 |
| `impls/onednn/fully_connected_onednn.hpp` | override `raw_sub_byte_weight_compatible()=true` |
| `registry/fully_connected_impls.cpp` | 注册 `FCCompressedGenerateOpt`（OneDNN 之后，默认 OCL 之前） |
| `graph/primitive_inst.cpp` | Rule 3 条件化 + M=1 decode 合成重试 |

---

## 调试方法

编译后的 Release binary 中，可通过观察 `std::cout` 输出排查选择路径：

```bash
# 关键日志模式
# validate_impl 通过：
[FCCmpOpt] <node_id> validate_impl PASS

# M=1 合成重试成功：
<node_id>: multi-impl pool: type=ocl using M=1 decode-synthesised impl for sub-byte WOQ

# pool 建立成功：
<node_id>: add_impl_to_pool(onednn)
<node_id>: add_impl_to_pool(ocl)
<node_id>: multi-impl pool enabled (primary=onednn, alts=1, policy=...)

# decode 阶段切换：
<node_id>: switch_impl_to: onednn → ocl
```

若 M=1 retry 返回 nullptr，说明 `validate_impl` 在 program-node 阶段失败，检查：
1. `compressed_weights` 是否为 true（fused 节点可能已拆分）
2. scale_idx 是否正确（有无 bias）
3. 节点是否有 `dynamic_quantized_activation`
