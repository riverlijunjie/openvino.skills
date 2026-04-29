# Qwen3-MoE 32K Decode 层间 8ms GPU 空白期根因分析

## 问题描述

Qwen3-30B-A3B 在 32K input tokens 的 decode 阶段，每个 decoder layer 之间存在约 8ms 的 GPU 空白期。VTune 特征：
- **0% EU utilization**（无计算）
- **~40% memory controller active**
- **~0.25 GB/s bandwidth**（约 2MB 总传输量）
- 出现在每层的 `pa_kv_cache_update` 执行之前

### 两种观测视角（同一现象）

| 测试方式 | 观测到的现象 |
|----------|-------------|
| 每次 kernel enqueue 后强制 clFinish | pa_kv_cache_update 前有 8ms 空白，然后 20μs 执行完毕 |
| 使用原有 event 同步（in-order queue） | MoE 第一个 exec_single_token 的 clEnqueueNDRange 阻塞 8ms，后续 2 个恢复正常 30μs |

两者是同一个 8ms，从不同角度观测：in-order queue 中 GPU 命令按序执行，8ms 的等待被"转移"到下一个 enqueue 调用上。

## 模型结构上下文

### Qwen3-MoE Decoder Layer 执行顺序
```
input_layernorm → q/k/v_proj → q_norm/k_norm → PagedAttention(kv_cache_update + pa_single_token)
    → o_proj → residual_add → post_attention_layernorm → MoE(gate_up → down → reduce) → residual_add
```
**PA 在 MoE 之前执行**。8ms 空白期出现在上一层 MoE 结束到下一层 PA kv_cache_update 之间。

### 层间 primitive 数量估算
从 MoE 最后一个 kernel 到下一层 PA 的 kv_cache_update，中间约 8-10 个 primitive：
residual_add → input_layernorm → q_proj → k_proj → v_proj → q_norm → k_norm → PA(kv_cache_update)

这些 primitive 在 decode（1 token）下都是极小 kernel（各 ~10-50μs），总 GPU 时间约 200μs。

## 已排除的 OpenVINO 代码层面原因

### 1. PA prepare_internal_buffers — 不是瓶颈
**文件**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp:1750`
```cpp
if ((stage == GENERATE && !has_scores_output && !use_micro_sdpa))
    return;
```
GENERATE 阶段直接 return，无任何 GPU 操作。

### 2. PA update_shape — 提前退出
**文件**: `src/plugins/intel_gpu/src/graph/primitive_inst.cpp:419`
```cpp
if (!input_shape_changed && _impl_params->get_output_layout().is_static())
    return;
```
Steady-state decode 中所有 input shape 不变，output layout 是 static 的，直接 return。

### 3. PA update/update_rt_params — 不被调用
**文件**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp:1516-1523`
```cpp
bool requires_update(...) { return stage == PagedAttentionStage::MIXED; }
```
GENERATE 阶段返回 false，`update()` 和 `update_rt_params()` 不会被调用。

### 4. clFinish/GPU 同步 — 不会触发
**文件**: `src/plugins/intel_gpu/src/graph/primitive_inst.cpp:442-444`
PA 的 shape_infer_dependencies 全部来自 `get_lockable_input_ids()`，都是 input_layout 类型。在 update_shape 中被显式跳过：
```cpp
if (get_node().is_type<paged_attention>() && dep->get_node().is_type<input_layout>())
    continue;
```
因此 `has_runtime_deps` 保持 false，不会触发 `stream.finish()`。

### 5. 方案1（host-resident buffers）— 已实现
所有 PA 的 metadata buffer（PAST_LENS, SUBSEQUENCE_BEGINS, MAX_CONTEXT_LEN 等）通过 `get_lockable_input_ids()` 机制已分配为 `usm_host`。`mem_lock` 对 usm_host 的 lock() 只返回指针，零开销。

### 6. fill()/output memory reset — 不涉及
`need_reset_output_memory()` 主要针对 convolution with padded layout，PA 不触发。

### 7. realloc_if_needed — 不触发
无 IMPL_CHANGED 和 SHAPE_CHANGED flag，不会进入 realloc 路径。

### 8. CPU 处理开销 — 远不够 8ms
每个 primitive 的 `prepare_primitive()` 在 steady-state 下约 20-80μs（主要是 dependency walk 和条件检查）。8-10 个 primitive 共计 ~200-600μs CPU 时间。

## 关键代码路径分析

### PA execute() 流程（GENERATE 阶段）
```
execute() [paged_attention_opt.cpp:1455]
  ├── update_stages_flags()        — 纯 CPU，读 flag bits
  ├── prepare_internal_buffers()   — GENERATE 时直接 return
  ├── execute_stage(kv_cache_update)  — set_arguments + enqueue_kernel
  └── execute_stage(pa_single_token)  — set_arguments + enqueue_kernel (~1ms GPU)
```

### execute_stage() 流程 [primitive_ocl_base.hpp:225-266]
```
execute_stage()
  ├── if need_dispatch_data_update: update_dispatch_data_func()  — 纯 CPU 计算
  ├── if need_args_update: 
  │     ├── get_arguments()     — 收集所有 input/output memory pointers
  │     └── stream.set_arguments()  — 调用 clSetKernelArgMemPointer (USM)
  └── stream.enqueue_kernel()   — clEnqueueNDRangeKernel
```

### set_arguments 中的 USM 处理 [ocl_stream.cpp:60-73]
```cpp
if (memory_capabilities::is_usm_type(mem->get_allocation_type())) {
    return kernel.setArgUsm(idx, buf);  // → clSetKernelArgMemPointer
}
```
kv_cache_update 有 12 个参数（6 个 INPUT + 2 个 KEY/VALUE_CACHE + 3 个 INTERNAL_BUFFER + 1 个 SCALAR）。

### network 执行循环 [network.cpp:827-864]
```cpp
for (auto& inst : _exec_order) {
    inst->prepare_primitive();
    inst->execute();
    executed_prims++;
    if (needs_flushing && executed_prims % flush_frequency == 0)  // flush_frequency=16
        get_stream().flush();
}
```

## 根因确认：WDDM Shared GPU Memory 预算不足

### 确认的根因：WDDM Video Memory Manager 驻留驱逐

8ms 空白期的根因是 **Windows WDDM Shared GPU Memory 预算不足**。总 GPU 分配量（~19-21 GB）超出 WDDM 设定的 Shared GPU Memory 预算（默认 18 GB），WDDM VidMm 在每层执行时需要驱逐和重新映射 allocation 的 GPU page table entries，产生 ~8-10ms 的开销。

**决定性证据**：

| Shared GPU Memory Budget | Decode Latency (ms/token) | 每层 WDDM 开销 |
|--------------------------|---------------------------|----------------|
| 18 GB（默认） | ~551 ms/token | ~9.6 ms/layer |
| **24.75 GB** | **88.26 ms/token** | **~0 ms** |

**6.2x 加速。84% 的 decode latency 都是 WDDM 驻留管理开销。**

### GPU 内存分配明细（from OV_VERBOSE=4 log）

| 分配类型 | 数量 | 总大小 | 主要内容 |
|----------|------|--------|---------|
| usm_host | 1700 | **15.08 GB** | MoE 权重 (144×96MB=13.5GB), attention 权重, 其他常量 |
| usm_device | 185 | **4.13 GB** | KV cache (2×1GB), attention output, 中间 buffer |
| cl_mem | 96 | **1.71 GB** | PA internal buffers (scores 等) |
| **合计** | 1981 | **~20.92 GB** | 全部在 32GB LPDDR5 中 |

**关键发现**：usm_host 分配 **计入** WDDM Shared GPU Memory 预算。这是因为 iGPU 上 GPU 通过 PPGTT 访问 usm_host 内存，WDDM 必须跟踪所有 GPU 可访问的内存。

### WDDM 驻留机制

当总 GPU 分配超出 Shared GPU Memory 预算时：
1. WDDM VidMm 使用 LRU 策略驱逐不活跃的 allocation 的 GPU page table entries
2. 当 GPU kernel 访问被驱逐的 allocation 时，driver 必须重新建立 page table mapping
3. 重新映射 ~1GB 的 KV cache (256K pages × ~55ns/page) ≈ **~14ms**
4. 实际观测 ~8ms（driver 可能只需 re-map 部分 pages）

### 为什么 kernel-skip 实验无效

| 实验 | 为什么无法消除 gap |
|------|-------------------|
| skip_kv_update | KV cache 仍然 **allocated** → 仍消耗 WDDM 预算 |
| skip_all_pa | 同上 |
| skip_moe | MoE 权重仍然 **allocated** → 仍消耗 WDDM 预算 |

所有 skip 实验只跳过了 kernel **执行**，没有释放任何 **allocation**，WDDM 预算压力不变。

### 8K vs 32K 的差异

| Context | GPU 总分配 | WDDM Budget | 状态 | Gap |
|---------|-----------|-------------|------|-----|
| 8K | ~17.5 GB | 18 GB | 0.5 GB 余量 ✅ | 无/极小 |
| 32K | ~19+ GB | 18 GB | **超出 ~1+ GB** ❌ | 8ms/layer |

32K 比 8K 多 ~1.5 GB KV cache (2×1GB vs 2×256MB)，刚好超出 WDDM 预算阈值。

### ~~之前的 TLB 分析（已修正）~~

之前认为 8ms gap 是 GPU TLB thrashing（硬件 TLB cache 容量不足导致 page table walk）。VTune 中的 TLB miss 数据确实存在，但 TLB miss 是 **WDDM 驻留驱逐的下游效应** — allocation 被驱逐后 page table entries 被清除，GPU 重新访问时自然表现为 TLB miss。增大 WDDM budget 后 TLB miss 同样消失，证明这不是硬件 TLB 容量限制。

## 已尝试的优化方案

### 方案1：改变 KV cache 分配类型（usm_device → usm_host）— 无效
**文件**: `variable_state.cpp:119`
```cpp
// 修改前: usm_device  修改后: usm_host
const auto alloc_type = cldnn::allocation_type::usm_host;
```
**结果**: decode latency 仍为 ~520ms，零改善。
**原因**: PTL iGPU 上 usm_host 和 usm_device 使用同一 PPGTT，TLB domain 相同。

### 方案2：改变常量分配类型（usm_host → usm_device）— 灾难性退化
**文件**: `constant.cpp:104`
```cpp
// 修改: 大于 1MB 的常量使用 usm_device
const auto alloc_type = (constLayout.bytes_count() > 1024*1024)
    ? cldnn::allocation_type::usm_device : lockable_type;
```
**结果**: decode latency 从 ~510ms **暴增到 ~34,000ms**（68x 退化）。
**原因**: ~19GB 总量超出 PTL iGPU ~16GB device aperture，每次 kernel launch 触发 driver 级 page fault + swap。

### 方案3：1-byte copy_to 预热 — 几乎无效
**文件**: `paged_attention_opt.cpp` execute() 中
```cpp
key_cache->copy_to(svc_stream, &m_prewarm_scratch[0], 0, 0, 1, false);
value_cache->copy_to(svc_stream, &m_prewarm_scratch[1], 0, 0, 1, false);
```
**结果**: 几乎无改善。
**原因**: Intel GPU driver 不做 allocation 级 residency check，只验证被访问的具体 page。1-byte copy 只触达 1 个 page（共需 147K pages）。

### 方案4：Strided-read OpenCL kernel 预热（当前实现）— 微弱改善
**文件**: `paged_attention_opt.cpp` execute() 中
```cpp
// 自定义 OpenCL kernel，每个 work item 读取 buf[gid * 4096]
// 在 service_stream（独立队列）上 enqueue，覆盖所有 ~147K pages
__kernel void kv_cache_tlb_prewarm(__global const uchar* buf, ulong buf_size) {
    size_t gid = get_global_id(0);
    size_t offset = gid * 4096;
    if (offset < buf_size) { volatile uchar val = buf[offset]; }
}
```

**Benchmark 结果（PTL iGPU, Qwen3-30B-A3B, 32K context, 32 decode tokens）**:

| 版本 | Decode Latency (ms/token) | Generation Time | 改善 |
|------|--------------------------|-----------------|------|
| Baseline | 593.18 | 110.81s | — |
| TLB prewarm (run1) | 583.28 | 110.61s | -9.9ms (1.7%) |
| TLB prewarm (run2) | 579.71 | — | -13.5ms (2.3%) |

**短上下文结果（无 TLB pressure，作为 regression check）**:

| Context | Baseline | Optimized | 差异 |
|---------|----------|-----------|------|
| 1024 | ~42 ms | ~43 ms | 无退化 |
| 4096 | ~47 ms | ~45 ms | 无退化 |

**改善不足的原因分析**：

1. **时序太晚**: prewarm 在 PA `execute()` 中触发，此时该层 MoE 已执行完毕。prewarm 无法与 MoE 并行，只能与后续 kv_cache_update 竞争同一 TLB
2. **iGPU 硬件限制**: service_stream kernel 和 main stream kernel 可能被 GPU 调度器序列化，而非真正并行
3. **TLB 竞争**: 即使并行执行，prewarm 和 MoE 尾部 kernel 共享 TLB，prewarm 填入的 entries 被 MoE 立即驱逐

### 方案5：提前触发 prewarm（network 层面）— 不可行
原设想在 `network::execute_impl()` 中 MoE 执行时在 service_stream 上提交 prewarm kernel。

**已排除原因**：根据 VTune profiling CSV 数据，MoE 每层 GPU 执行时间仅 ~0.5ms（gate_up ~0.32ms × 4 variants + down ~0.16ms × 4 + reduce ~0.008ms × 4 + softmax_topk ~0.011ms × 4）。而 prewarm 需要 ~8ms 走完 147K pages。即使从 MoE 第一个 kernel 开始触发 prewarm，也只能重叠 0.5ms，剩余 7.5ms 仍会阻塞。

### 方案6：CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL — 无显著改善
**文件**: `paged_attention_opt.cpp` execute() 中
```cpp
// 在 execute_stage(kv_cache_update) 前，通过 clSetKernelExecInfo 注册 KV cache USM 指针
auto* ocl_k = dynamic_cast<cldnn::ocl::ocl_kernel*>(kv_cache_update->kernel.get());
cl_kernel raw_kernel = ocl_k->get_handle().get();
const void* usm_ptrs[2] = { key_ptr, value_ptr };
clSetKernelExecInfo(raw_kernel, CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL, sizeof(usm_ptrs), usm_ptrs);
```

**Benchmark 结果（PTL iGPU, Qwen3-30B-A3B, 32K context, 32 decode tokens）**:

| Run | Baseline (ms/token) | USM_PTRS_INTEL (ms/token) |
|-----|---------------------|---------------------------|
| 1 | 561.80 | 541.78 |
| 2 | 540.59 | 552.06 |
| 3 | 549.78 | 539.86 |
| **Avg** | **550.72** | **544.57** |

差异 -6.15ms (-1.1%)，在测量噪声范围内（~10ms stdev）。**结果不具统计显著性**。

**原因分析**：`CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL` 的作用是告知 driver 该 kernel 会访问哪些 USM 指针，用于 **validation** 目的（确保 kernel 不会越界访问未注册的 USM 内存）。在 Intel iGPU 上，driver 并不会因此提前 prefault page table entries。实际的 page table walk 仍然发生在 kernel 执行时首次访问对应 page 的时刻。

### 方案7：Huge Pages (2MB) — 无法在应用层实现
**分析结论**：Intel OpenCL USM 分配 API（`clHostMemAllocINTEL` / `clDeviceMemAllocINTEL`）的 `properties` 参数仅支持以下 flags：
- `CL_MEM_ALLOC_WRITE_COMBINED_INTEL` (1 << 0)
- `CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL` (1 << 1)
- `CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL` (1 << 2)

**没有** huge page / large page 相关的 allocation flag。在 Windows 上更无法对已分配的 USM 内存做 `madvise(MADV_HUGEPAGE)` 等操作。Huge page 支持需要 Intel OpenCL/Level Zero driver 在底层实现，不在 OpenVINO 应用层可控范围内。

Linux 上可能通过系统级 Transparent Huge Pages (THP) 配置间接影响，但需要验证 OpenCL driver 的 USM 分配是否走 THP 路径。

## Kernel Skip 实验（诊断过程）

通过系统地跳过不同 kernel 来隔离 8ms gap 的来源。所有实验使用 32K context, 32 decode tokens。

| 实验 | Decode Latency (ms/token) | vs Baseline | 分析 |
|------|--------------------------|-------------|------|
| **Baseline** | **~551** | — | 正常执行 |
| skip_kv_update | 563.61 | +2.3% (噪声) | Gap 转移到下一个 kernel |
| skip_all_pa | 561.88 | +2.0% (噪声) | Gap 仍未消失 |
| skip_moe | 559.45 | +1.5% (噪声) | MoE 不执行，gap 仍在 |

所有 skip 实验无效，因为它们只跳过 kernel 执行，不释放 allocation → WDDM 预算压力不变。

## 解决方案

### 已验证：增大 Shared GPU Memory Budget

在 Windows 设置中将 Shared GPU Memory 从 18 GB 增大到 24.75 GB：
- Decode latency: 551 → **88.26 ms/token** (6.2x)
- 总 GPU 分配 (~21 GB) 完全 fit 在 budget 内，无 WDDM 驱逐

### 实际优化方向

1. **用户端**：建议用户在 Windows 设置中增大 Shared GPU Memory
   - 需要 GPU 分配量 + 余量（建议 ≥ 总分配量 × 1.2）
   - 对于 Qwen3-30B-A3B 32K：至少 24 GB

2. **OpenVINO 端**：在 GPU plugin 中检测 WDDM 预算压力并发出警告
   - 通过 `clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE)` 获取 budget
   - 在 compilation 后比对总分配量 vs budget
   - 如果超出则 log warning 建议用户调整设置

3. **减少 GPU 内存占用**：使用户不需要手动调整设置
   - 检查是否有 compilation 阶段的临时 buffer 可在 inference 前释放
   - 检查 cl_mem (1.71 GB PA buffers) 是否可按需分配而非预分配
   - MoE 未使用的 expert 权重是否可 lazy-load（但会影响 prefill）

## 关键文件索引

| 文件 | 关注点 |
|------|--------|
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp` | PA 实现：execute(), TLB prewarm kernel, prepare_internal_buffers() |
| `src/plugins/intel_gpu/src/graph/primitive_inst.cpp` | prepare_primitive(), update_shape(), has_dynamic_dependencies_insts() |
| `src/plugins/intel_gpu/src/graph/include/paged_attention_inst.h` | get_lockable_input_ids(), key_cache_memory_ptr(), value_cache_memory_ptr() |
| `src/plugins/intel_gpu/src/graph/network.cpp` | execute_impl() 主循环，flush_frequency=16 |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/primitive_ocl_base.hpp` | execute_stage(), update_stages_flags() |
| `src/plugins/intel_gpu/src/runtime/ocl/ocl_stream.cpp` | set_arguments_impl(), enqueue_kernel() |
| `src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp` | gpu_usm::lock(), gpu_usm::copy_to() |
| `src/plugins/intel_gpu/src/runtime/ocl/ocl_ext.hpp` | CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL, enqueue_memcpy |
| `src/plugins/intel_gpu/src/runtime/ocl/ocl_kernel_builder.hpp` | build_kernels() — 用于编译自定义 OCL kernel |
| `src/plugins/intel_gpu/include/intel_gpu/runtime/engine.hpp` | create_kernel_builder(), get_service_stream() |
| `src/plugins/intel_gpu/src/plugin/variable_state.cpp` | KV cache 分配类型（line 119） |
| `src/plugins/intel_gpu/src/plugin/ops/constant.cpp` | 常量分配类型（line 104） |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_swiglu_opt.cpp` | MoE exec_single_token(), 3 kernel enqueue |
