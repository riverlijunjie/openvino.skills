# 3-Bit (U2) GEMM & MoE Support — 详细工作清单与工作流程

> **修订说明**: 本文档根据 `kernel_design_3bit.md` 中的首选kernel实现策略重新评估。
> 核心变化: Prefill路径采用 **Option B (U2→U4转换kernel + 复用现有GEMM)**, Decode路径采用 **Option A (原生U2 dequant)**。

## 概述

在当前OpenVINO GPU插件中增加3-bit (U2) GEMM和MoE支持。根据 `kernel_design_3bit.md` 首选方案，
**Prefill阶段不需要编写全新GEMM kernel**，而是插入一个轻量U2→U4转换kernel后复用现有4-bit GEMM流水线。
这大幅降低了Prefill路径的实现难度，**将核心开发工作聚焦在Decode阶段的OpenCL kernel上**。

---

## 首选Kernel实现策略对照表

| 路径 | 阶段 | 首选方案 | 实现方式 | 难度评估 |
|------|------|---------|---------|---------|
| Dense GEMM | Prefill | **Option B** | U2→U4转换kernel + 现有oneDNN FC (U4 path) | **中** |
| Dense GEMV | Decode | **Option A** | 新增OpenCL U2 dequant GEMV kernel | **高** |
| MoE GEMM | Prefill | **Option B** | U2→U4转换kernel + 现有grouped_gemm/micro_gemm | **中** |
| MoE GEMV | Decode | **Option A** | 扩展现有 `moe_3gemm_swiglu_mlp.cl` 添加U2 dequant分支 | **高** |

**关键约束 (已通过代码验证):**
- `gemmstone StructuredType::Type` 枚举 **没有 u2/s2** → microkernel融合 (Option C) 不可行
- `oneDNN data_type_t` **没有 u2** → oneDNN GEMM/FC 不能直接处理U2权重
- 但U2→U4扩展后，**现有oneDNN u4 path完全可用**

---

## 架构层次总览

```
Layer 0: Core Type System          ✅ 已支持 (ov::element::u2 已存在)
Layer 1: Frontend (ONNX/PyTorch)   ✅ 已支持 (MatMulNBits 2-bit, BitNet)
Layer 2: Common Transformations    ✅ MOECompressed 无type gate, 可透传U2
Layer 3: GPU Plugin Gating         ❌ is_supported(u2) = false
Layer 4: Graph Transformations     ❌ 多处硬编码仅匹配 u4/i4/u8/i8
Layer 5: Kernel Implementations    ❌ 需新增: U2→U4转换kernel + U2 decode kernel
Layer 6: Tests                     ❌ 需新增U2测试用例
```

---

## 详细工作清单

### Phase 1: GPU Plugin类型系统解锁 (前置条件, 1天)

所有后续Phase都依赖此步骤。改动极小但影响面广。

| # | 文件 | 修改内容 | 难度 |
|---|------|---------|------|
| 1.1 | `src/plugins/intel_gpu/src/plugin/common_utils.cpp:195` | `case ov::element::Type_t::u2: return false;` → `return true;` | 低 |
| 1.2 | `src/plugins/intel_gpu/include/intel_gpu/runtime/layout.hpp` | 在 `data_type_traits` 中新增 `static bool is_u2(data_types dt)` helper (用于后续条件判断) | 低 |

---

### Phase 2: Graph Transformation & 调度修改 (2天)

使U2权重能通过现有的graph transformation pipeline，并正确路由到对应的kernel实现。

#### 2A: Pattern匹配扩展 (使transformation能识别U2)

| # | 文件 | 修改内容 | 难度 |
|---|------|---------|------|
| 2.1 | `compressed_weights_pattern.hpp:10-11` | `compressed_constant` lambda 添加 `\|\| output.get_element_type() == ov::element::u2` | 低 |
| 2.2 | `convert_fc_to_compressed.cpp:60` | 添加 u2 分支: 类似u4处理逻辑, 不设置 `weight_u8=true` | 低 |
| 2.3 | `keep_moe_3gemm_const_precision.cpp:20-48` | 所有 `type_matches(ov::element::u4)` 新增并行的 `u2` 变体, 或改为 `type_matches_any({u4,u2})` | 中 |

#### 2B: Impl路由 (oneDNN排除 + quantized_types扩展)

| # | 文件 | 修改内容 | 难度 |
|---|------|---------|------|
| 2.4 | `fully_connected_onednn.cpp` ImplementationManager | `validate()` 中排除 `weight_bitwidth==2`，使 u2 weight 不走 oneDNN FC | 低 |
| 2.5 | `moe_gemm_onednn.cpp` / `moe_gemm_onednn.hpp` | `validate()` 中排除 u2，使 u2 MoE prefill 不走 oneDNN grouped_gemm | 低 |
| 2.6 | `moe_gemm_gen_opt.hpp:39` | `quantized_types` 列表添加 `data_types::u2` | 低 |
| 2.7 | `moe_3gemm_swiglu_opt.cpp:1016` | `if (!(weight_dt == data_types::u4 \|\| weight_dt == data_types::i4))` 添加 `\|\| weight_dt == data_types::u2` | 低 |
| 2.8 | `moe_3gemm_swiglu_opt.cpp:746-760` | 新增 `else if (weight_dt == ov::element::u2)` JIT常量分支: `WEIGHT_COMPRESSEION_DT=3`, `MOE_WEI_DT="uchar"` | 低 |

---

### Phase 3: Prefill路径 — U2→U4转换kernel (3天)

**核心策略: Option B** — 在GEMM执行前，用一个轻量OpenCL kernel将U2权重原地/离线扩展为U4，然后复用现有4-bit GEMM流水线。

#### 优势分析 (vs 编写全新U2 GEMM kernel)

| 对比项 | Option A (全新U2 GEMM) | Option B (U2→U4转换 + 复用) |
|--------|----------------------|---------------------------|
| 代码量 | ~2000行 OpenCL kernel | ~150行转换kernel + ~100行调度 |
| DPAS利用 | 需要处理U2→INT8对齐 | 复用现有u4→DPAS路径 |
| 测试覆盖 | 全新kernel需全面测试 | 复用已验证的GEMM kernel |
| 性能开销 | 最优 (无额外pass) | 额外 ~0.1-0.5ms/layer (转换pass) |
| 维护负担 | 高 (独立kernel演进) | 低 (跟随现有4-bit path) |

#### 3A: U2→U4 转换 OpenCL Kernel

| # | 工作内容 | 难度 |
|---|---------|------|
| 3.1 | **新增** `u2_to_u4_expand.cl`: 将 [E,N,K/4] U2 packed buffer 扩展为 [E,N,K/2] U4 packed buffer | **中** |
| 3.2 | **新增** 对应 host-side C++ 调度文件: 分配临时U4 buffer, 插入expand kernel到exec pipeline | **中** |
| 3.3 | 在 `moe_3gemm_swiglu_opt.cpp` prefill路径添加: 当 weight==u2 时，先dispatch U2→U4 expand kernel，再调用现有 grouped_gemm/micro_gemm | 中 |

**U2→U4转换kernel设计:**
```opencl
// 输入: U2 packed weight [E, N, K/4] (每字节4个u2值)
// 输出: U4 packed weight [E, N, K/2] (每字节2个u4值)
// 每个work-item处理1字节输入(4个u2值) → 2字节输出(4个u4值)
__kernel void u2_to_u4_expand(
    __global const uchar* src,   // U2 packed: [E*N*K/4]
    __global uchar* dst,          // U4 packed: [E*N*K/2]
    int total_bytes               // E*N*K/4
) {
    int gid = get_global_id(0);
    if (gid >= total_bytes) return;

    uchar packed_u2 = src[gid];
    // 提取4个u2值 (每个2-bit)
    uchar v0 = (packed_u2     ) & 0x03;  // bits [1:0]
    uchar v1 = (packed_u2 >> 2) & 0x03;  // bits [3:2]
    uchar v2 = (packed_u2 >> 4) & 0x03;  // bits [5:4]
    uchar v3 = (packed_u2 >> 6) & 0x03;  // bits [7:6]

    // 打包为2个u4字节 (每字节2个u4值)
    dst[gid * 2    ] = v0 | (v1 << 4);   // low nibble + high nibble
    dst[gid * 2 + 1] = v2 | (v3 << 4);
}
```

**内存开销:** 临时U4 buffer大小 = U2 weight大小 × 2。对于典型MoE模型 (E=64, N=3072, K=7168):
- 单层U2 weight: 64 × 3072 × 7168 / 4 = 352 MB → U4 buffer: 704 MB
- **优化**: 可以逐expert扩展 (每次只扩展top-k个expert), 减少到 top_k × N × K / 2

#### 3B: Dense FC Prefill — 同样使用U2→U4转换

| # | 工作内容 | 难度 |
|---|---------|------|
| 3.4 | 在FC执行路径中，当weight==u2时，插入U2→U4 expand kernel，然后走现有 oneDNN FC u4 path | 中 |
| 3.5 | FC的U4 buffer管理: 可以在 graph compilation 阶段静态分配，或运行时lazy分配 | 中 |

**注意:** Dense FC的U2→U4转换可以**预计算一次并缓存** (权重在推理期间不变)，所以不需要每次推理都转换。这消除了运行时开销。

#### 3C: ZP/Scale 兼容性

U2→U4扩展后，scale和ZP的含义不变 (仍然是per-group参数)。但需要注意:
- U2 dequant: `w_f16 = (u2_val - zp) * scale`, u2_val ∈ {0,1,2,3}
- 扩展到U4后: u4_val 仍然 ∈ {0,1,2,3} (高2位为0), dequant公式不变
- Scale和ZP可以直接传递给现有4-bit GEMM，**不需要转换**

---

### Phase 4: Decode路径 — 原生U2 dequant OpenCL kernel (5天, 核心工作)

**核心策略: Option A** — 直接在OpenCL decode kernel中实现U2解压缩，不需要预转换。
这是因为decode是memory-bound的，从U2直接读取比先转换U4再读取节省50%带宽。

#### 4A: MoE Decode — 扩展现有kernel

这是 **工作量最大的部分**，需要在 `moe_3gemm_swiglu_mlp.cl` 中添加完整的U2 GEMV变体。

| # | 文件 | 修改内容 | 难度 |
|---|------|---------|------|
| 4.1 | `moe_3gemm_swiglu_mlp.cl` | 新增 DEQUANT_2BIT 宏 (4个: 从1字节提取4个u2值) | 低 |
| 4.2 | 同文件 | 新增 `gate_up_gemv_n2x_u2()` 函数 (2-output GEMV, U2 weight) | **高** |
| 4.3 | 同文件 | 新增 `gate_up_gemv_n4x_u2()` 函数 (4-output GEMV, U2 weight) | **高** |
| 4.4 | 同文件 | 新增 `down_gemv_u2()` 函数 (down-projection GEMV, U2 weight) | **高** |
| 4.5 | 同文件 | 在 `gate_up` / `down` / `reduce` entry point 添加 `#if WEIGHT_COMPRESSEION_DT == 3` 分支调用U2变体 | 中 |
| 4.6 | `moe_3gemm_swiglu_fuse.cl` | 如果fuse变体独立实现 (非include), 需要同步添加U2支持 | 中 |

**U2 GEMV内循环核心设计 (与U4对比):**

```
U4 inner loop (现有):
  FAKE_GROUP_SIZE=128, SUBGROUP_SIZE=32 → ELEMS_PER_LANE=4
  每次迭代: 读1字节(2个u4值) → DEQUANT_4BIT_LO/HI → 2个FMA

U2 inner loop (新增):
  FAKE_GROUP_SIZE=128, SUBGROUP_SIZE=32 → ELEMS_PER_LANE=4
  每次迭代: 读1字节(4个u2值) → DEQUANT_2BIT_0/1/2/3 → 4个FMA
  注: 读取量相同(1字节), 但处理4个而非2个值 → K方向前进2倍
```

```opencl
// === U2 DEQUANT 宏 ===
#define DEQUANT_2BIT_0(v) convert_half((v) & 0x03)
#define DEQUANT_2BIT_1(v) convert_half(((v) >> 2) & 0x03)
#define DEQUANT_2BIT_2(v) convert_half(((v) >> 4) & 0x03)
#define DEQUANT_2BIT_3(v) convert_half(((v) >> 6) & 0x03)

// === U2 signed variant (如果需要i2支持) ===
// i2 值域: {-2,-1,0,1}, 需要符号扩展
#define DEQUANT_2BIT_SIGNED_0(v) convert_half((char)(((v) & 0x02) ? ((v) & 0x03) | 0xFC : ((v) & 0x03)))
```

**关键数据流差异:**
- U4: weight地址步进 = K/2 字节/行, scale地址步进 = K/group_size × sizeof(half)
- U2: weight地址步进 = K/4 字节/行, scale地址步进 = K/group_size × sizeof(half) (不变)
- 内循环: U4每gk迭代消耗 `ELEMS_PER_LANE/2` 字节; U2每gk迭代消耗 `ELEMS_PER_LANE/4` 字节
- 所以 **U2的gk循环次数只有U4的一半** (同样的K维度, 数据量减半)

#### 4B: Dense GEMV Decode

| # | 文件 | 修改内容 | 难度 |
|---|------|---------|------|
| 4.7 | FC OCL kernel (kernel_selector) | 新增U2 decompression path。但由于decode只在token=1时使用，也可走MoE decode kernel架构 | **中-高** |
| 4.8 | 或: 独立新增 `fc_gemv_u2.cl` | 专门的U2 FC decode kernel (如果kernel_selector不适合扩展) | **中-高** |

**替代方案:** Dense FC decode 也可以使用 Phase 3 的 U2→U4 预转换 + 现有U4 decode path。
但这会浪费50%的decode带宽 (U4是U2的2倍数据量)。
`kernel_design_3bit.md` 首选 Option A (原生dequant), 所以建议原生实现。

---

### Phase 5: 测试与验证 (2天)

| # | 文件 | 修改内容 | 难度 |
|---|------|---------|------|
| 5.1 | `tests/unit/transformations/fuse_moe_3gemm_compressed_test.cpp` | 添加U2 weight的fusion test | 中 |
| 5.2 | `tests/unit/test_cases/moe_3gemm_gpu_test.cpp` | 添加U2 weight的端到端MoE decode test | 中 |
| 5.3 | `tests/unit/test_cases/moe_gemm_gpu_test.cpp` | 添加U2 weight的MoE GEMM (prefill with U2→U4) test | 中 |
| 5.4 | FC相关测试文件 | 添加U2 weight的FC compressed test (decode + prefill) | 中 |
| 5.5 | 新增: U2→U4 expansion kernel 单元测试 | 验证位操作正确性 (边界值: 0x00, 0x55, 0xAA, 0xFF) | 低 |

---

## 修订后的实施工作流程

```
Week 1 (3天): Phase 1 + Phase 2 (类型解锁 + 调度路由)
  ├── 1.1-1.2  解锁u2 type
  ├── 2.1-2.3  Pattern匹配扩展
  ├── 2.4-2.8  Impl路由修改 (oneDNN排除 + JIT常量)
  └── 验证: U2权重模型能加载到GPU graph, 不crash (即使还不能执行)

Week 2 (3天): Phase 3 (Prefill: U2→U4转换)
  ├── 3.1-3.2  U2→U4 expand kernel实现
  ├── 3.3      MoE prefill调度集成
  ├── 3.4-3.5  Dense FC prefill集成
  └── 验证: Prefill路径输出正确 (对比FP16 reference)

Week 3 (5天): Phase 4 (Decode: 原生U2 dequant) ⭐ 核心工作
  ├── 4.1-4.5  MoE decode kernel U2 GEMV实现
  ├── 4.6      Fuse变体同步
  ├── 4.7-4.8  Dense FC decode kernel U2 GEMV
  └── 验证: Decode路径输出正确 + 性能分析 (vs U4 baseline)

Week 4 (2天): Phase 5 (测试 + 集成验证)
  ├── 5.1-5.5  所有单元测试
  ├── 端到端模型验证 (NNCF 3-bit模型)
  └── 性能报告: roofline efficiency, latency对比
```

---

## 工作量评估对比 (修订前 vs 修订后)

| 路径 | 修订前评估 | 修订后评估 (按kernel_design首选方案) | 变化原因 |
|------|----------|----------------------------------|---------|
| Prefill Dense GEMM | **高** (全新OCL GEMM kernel) | **中** (U2→U4转换kernel, ~150行) | Option B: 复用现有4-bit GEMM |
| Prefill MoE GEMM | **高** (micro_gemm扩展) | **中** (U2→U4转换kernel, 同上复用) | Option B: 复用现有grouped_gemm |
| Decode Dense GEMV | **高** (全新kernel) | **中-高** (新增U2 GEMV) | Option A: 但参考MoE decode架构 |
| Decode MoE GEMV | **高** (新增GEMV函数) | **高** (仍需新函数) | Option A: 核心工作不变 |
| 调度/变换层 | 中 | **低** (改动量减少) | 多处只需添加u2到已有列表 |
| 总体 | ~4周 | **~3周** (减少~25%) | Prefill复用策略显著减少工作量 |

---

## 关键技术决策 (修订)

| 决策点 | 选项 | 首选 (kernel_design_3bit.md) | 依据 |
|--------|------|---------------------------|------|
| Prefill GEMM (dense) | A:全新U2 kernel / **B:U2→U4+复用** | **B** | 复用已验证的oneDNN u4 path, 维护成本低 |
| Prefill GEMM (MoE) | A:全新U2 kernel / **B:U2→U4+复用** | **B** | 同上, 且gemmstone无u2 Type |
| Decode GEMV (MoE) | **A:原生U2 dequant** / B:U2→U4+复用 | **A** | 节省50%带宽 (memory-bound path) |
| Decode GEMV (dense) | **A:原生U2 dequant** / B:U2→U4+复用 | **A** | 同上 |
| gemmstone u2扩展 | 修改thirdparty / 绕过 | **绕过** (Option B) | 不修改oneDNN/gemmstone代码 |
| U2→U4 buffer生命周期 | 每次推理转换 / 预计算缓存 | **预计算缓存** | 权重不变, 只需转换一次 |

---

## 风险与依赖

1. **U2→U4 内存开销**: 临时U4 buffer = 2× U2 weight大小。对于大MoE模型可能增加数百MB显存。
   - 缓解: 逐expert/逐layer转换, 复用buffer; 或在graph compilation阶段做一次性offline扩展
   - 进一步优化: 只为prefill path分配U4 buffer, decode path直接读U2

2. **gemmstone无u2**: 如果未来想在microkernel中直接融合U2 dequant (Option C), 需要向gemmstone贡献u2 Type。
   但这是 **未来优化**, 不阻碍当前实现。

3. **性能预期**:
   - Decode: U2直接dequant, 带宽利用率应 ≥85% roofline (vs U4的同等指标)
   - Prefill: U2→U4转换增加 ~0.1-0.5ms/layer, 但只在首次推理时执行 (缓存后为0)

4. **精度**: 参见kernel_design_3bit.md — U2仅4个离散值, 依赖NNCF的mixed-precision分配 (部分层U2, 部分层I4)

---

## 文件修改汇总 (约17个文件)

**Phase 1-2: 类型解锁 + 调度 (8个文件, 低难度):**
1. `src/plugins/intel_gpu/src/plugin/common_utils.cpp`
2. `src/plugins/intel_gpu/include/intel_gpu/runtime/layout.hpp`
3. `src/plugins/intel_gpu/src/plugin/transformations/compressed_weights_pattern.hpp`
4. `src/plugins/intel_gpu/src/plugin/transformations/convert_fc_to_compressed.cpp`
5. `src/plugins/intel_gpu/src/plugin/transformations/keep_moe_3gemm_const_precision.cpp`
6. `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp`
7. `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_gemm_gen_opt.hpp`
8. `src/plugins/intel_gpu/src/graph/impls/onednn/fully_connected_onednn.cpp`
9. `src/plugins/intel_gpu/src/graph/impls/onednn/moe_gemm_onednn.cpp`

**Phase 3: Prefill U2→U4转换 (2-3个新文件, 中难度):**
10. `src/plugins/intel_gpu/src/graph/impls/ocl_v2/u2_to_u4_expand.cl` ⭐ **新增**
11. `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/u2_expand_dispatch.cpp` ⭐ **新增** (或集成到现有文件)

**Phase 4: Decode原生U2 kernel (2个文件, 高难度):**
12. `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_mlp.cl` ⭐ **核心修改**
13. `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe_3gemm_swiglu_fuse.cl`
14. FC decode kernel (kernel_selector或新增ocl_v2文件)

**Phase 5: 测试 (3-4个文件):**
15-17. 上述Phase 5中列出的测试文件
