---
name: dev_gpu_idle_gap_analysis
description: Diagnose GPU idle gaps (pipeline bubbles) in OpenVINO GPU plugin inference on Intel iGPU. Use when investigating unexplained GPU idle time between kernels, high memory controller activity with low EU utilization, decode latency regression at long context, or WDDM shared GPU memory pressure on Windows.
applyTo: "src/plugins/intel_gpu/**"
---

# GPU Idle Gap Diagnosis for Intel iGPU

Systematic methodology for diagnosing GPU idle gaps (pipeline bubbles) in the OpenVINO GPU plugin execution pipeline on Intel integrated GPUs.

## Step 1: Characterize the Gap

Collect VTune or `ze_tracer` profiling data. Classify the idle gap by:

| EU Util | Mem Controller | Bandwidth | Likely Cause |
|---------|---------------|-----------|-------------|
| 0% | High (~40%) | Low (~MB/s) | **WDDM residency eviction** (most common on Windows iGPU — check budget first) |
| 0% | High | High (~GB/s) | Buffer migration (device↔host) |
| 0% | Low | Low | CPU starvation — GPU waiting for commands |
| Low | High | High | Memory-bound kernel, not truly idle |

Enable debug logging to collect memory allocation data:
```
OV_GPU_VERBOSE=4
```

Key log patterns to grep:
- `MemoryTracker.*Allocate` — all GPU memory allocations with type and cumulative size
- `available device mem size` — WDDM Shared GPU Memory budget
- `~MemoryTracker.*Free` — deallocation with peak watermark (`max=` field)

## Step 2: Check WDDM Shared GPU Memory Budget (Windows iGPU)

**This is the #1 cause of GPU idle gaps on Windows iGPU with large models.**

### Mechanism

On Windows iGPU, **all** GPU-accessible allocations (usm_host + usm_device + cl_mem) count against the WDDM Shared GPU Memory budget. When total allocations exceed this budget, WDDM Video Memory Manager (VidMm) evicts GPU page table entries of inactive allocations using LRU. When a kernel subsequently accesses an evicted allocation, the driver must re-establish the page table mapping, stalling the GPU.

**Key insight**: `usm_host` allocations **DO** count against the WDDM Shared GPU Memory budget on iGPU. Even though usm_host is in CPU address space, the GPU accesses it via PPGTT, so WDDM tracks it.

### Diagnosis

1. Sum total GPU allocations from log:
   ```bash
   grep 'MemoryTracker.*Allocate' log.txt | \
     grep -oP 'Allocate (\d+) bytes of (\w+)' | \
     awk '{sizes[$NF]+=$2; counts[$NF]++} END {
       for (t in sizes) printf "%s: count=%d, total=%.2f GB\n", t, counts[t], sizes[t]/1073741824
     }'
   ```

2. Check WDDM budget from log:
   ```bash
   grep 'available device mem size' log.txt
   ```

3. Compare: if `sum(usm_host + usm_device + cl_mem) > WDDM budget` → **this is the cause**.

4. Check Windows Settings → System → Display → Graphics → change default graphics settings to see/adjust Shared GPU Memory.

### Confirmation Test

Increase WDDM Shared GPU Memory to exceed total allocations by ≥20%. If the idle gap disappears, root cause is confirmed.

**Reference case**: Qwen3-30B-A3B at 32K context on PTL iGPU (32GB RAM):

| Metric | Value |
|--------|-------|
| usm_host | 15.08 GB (MoE experts 13.5GB + attention weights + constants) |
| usm_device | 4.13 GB (KV cache 2×1GB + intermediate buffers) |
| cl_mem | 1.71 GB (PA internal buffers) |
| **Total** | **~20.92 GB** |
| WDDM Budget (default) | 18 GB |
| **Over budget** | **~3 GB** |

| WDDM Budget | Decode Latency | Per-layer WDDM overhead |
|-------------|---------------|------------------------|
| 18 GB | 551 ms/token | ~9.6 ms/layer |
| 24.75 GB | **88.26 ms/token** | ~0 ms |

**6.2x speedup. 84% of decode latency was WDDM eviction overhead.**

### Why Context Length Matters

KV cache grows linearly with context length. A model that fits in WDDM budget at short context may exceed it at long context:

| Context | KV Cache Size | Total GPU Alloc | vs 18GB Budget |
|---------|-------------|-----------------|----------------|
| 8K | ~0.5 GB | ~17.5 GB | Fits ✅ |
| 32K | ~2.0 GB | ~19+ GB | Exceeds ❌ |

The gap appears at the context length where total allocations cross the WDDM budget threshold.

## Step 3: Check Code-Level Sync Points

If WDDM budget is sufficient, inspect OpenVINO code-level causes in priority order:

### 3a. Explicit GPU Synchronization

**File**: `src/plugins/intel_gpu/src/graph/primitive_inst.cpp`

| Sync Point | Location | Trigger |
|-----------|----------|---------|
| `stream.finish()` | `update_shape()` | `has_runtime_deps == true` + in-order queue |
| `stream.wait_for_events()` | `update_shape()` | `has_runtime_deps == true` + OOO queue |
| `output->fill(blocking=true)` | `prepare_primitive()` | `need_reset_output_memory()` |
| `mem_lock` on device mem | various | `gpu_usm::lock()` on `usm_device` does blocking memcpy |

`has_runtime_deps` triggers when `get_shape_infer_dependencies()` includes a non-const, non-CPU, non-shape_of GPU dependency. Exception: PagedAttention skips `input_layout` deps.

### 3b. Memory Allocation Type Mismatch

`mem_lock` cost depends on allocation type:

| Type | lock() | Cost |
|------|--------|------|
| `usm_host` | Returns pointer directly | **Zero** |
| `usm_device` | `enqueue_memcpy(blocking=true)` | **High** — GPU→CPU sync |

**Fix**: Ensure all CPU-accessed buffers use `usm_host` via:
- Add to `get_lockable_input_ids()` for input ports
- Set `BufferDescriptor.m_lockable = true` for internal buffers

### 3c. CPU Overhead in prepare_primitive()

**File**: `primitive_inst.cpp`

Key CPU cost centers:
- `has_dynamic_dependencies_insts()`: Recursive walk of all deps checking `MEMORY_CHANGED` flag
- `update_shape()`: Loops all deps comparing layouts
- `set_arguments()`: Kernel arg binding via `clSetKernelArg`/`clSetKernelArgMemPointer`

Estimated: ~20-80us per primitive in steady state. Unlikely to cause multi-ms gaps alone.

## Step 4: In-Order Queue Effects

OpenVINO GPU defaults to in-order queue. Important implications for diagnosis:

- `clEnqueueNDRangeKernel` CPU time may include **waiting for prior GPU commands** to complete
- A GPU stall (e.g., from WDDM re-mapping) appears as blocking time on the **next** enqueue call, not on the kernel that caused it
- Skipping a kernel just shifts the gap to the next kernel in the queue — the underlying stall remains
- To properly attribute a gap: insert `clFinish()` after each kernel and measure GPU time directly

## Common Patterns & Fixes

| Pattern | Symptom | Root Cause | Fix |
|---------|---------|-----------|-----|
| **WDDM budget exceeded** | Multi-ms gap between layers, worse at long context | Total GPU allocs > Shared GPU Memory | Increase WDDM budget, or reduce GPU memory footprint |
| Shape-infer sync | clFinish at fixed layer positions | `get_shape_infer_dependencies()` includes GPU dep | Add to `get_lockable_input_ids()` → usm_host |
| mem_lock blocking | `prepare_internal_buffers()` slow | Locking `usm_device` memory | Mark buffer as lockable → usm_host |
| Output fill | `fill()` takes ms | Large buffer blocking zero-fill | Skip unnecessary fills or use non-blocking |
| Flush timing | Intermittent gaps, correlated with prim count % 16 | `flush_frequency=16` in `execute_impl()` | Adjust flush_frequency or add manual flush |
| Allocation type trap | Catastrophic regression (>10x) | Device aperture overflow on iGPU | Never exceed ~half of system RAM in usm_device on iGPU |

## Approaches That Do NOT Work for WDDM Eviction

These were evaluated and confirmed ineffective for the WDDM budget problem:

| Approach | Why It Fails |
|----------|-------------|
| Skipping kernel execution | Does not free allocations → budget pressure unchanged |
| Changing allocation type (usm_host ↔ usm_device) | Both count against WDDM budget on iGPU; same PPGTT domain |
| Moving all constants to usm_device | Exceeds device aperture → catastrophic regression |
| `CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL` | Validation hint only, does not affect residency or prefault pages |
| Strided-read prewarm kernel on service_stream | ~2% improvement at best; WDDM re-mapping is at driver level, not GPU EU level |
| Huge pages via OpenCL USM API | No allocation flags exist for large/huge pages |

## PTL iGPU Specifics

- usm_host and usm_device share the **same physical LPDDR memory and GPU PPGTT**
- Device memory aperture is approximately half of system RAM. Exceeding it causes catastrophic driver-level paging
- WDDM Shared GPU Memory default is typically ~50-60% of system RAM (e.g., 18 GB on 32 GB system)
- All GPU-accessible allocations (usm_host + usm_device + cl_mem) consume the WDDM budget
- VTune TLB miss data is a **downstream symptom** of WDDM eviction, not an independent hardware limit

## Key Files

| File | Focus |
|------|-------|
| `src/plugins/intel_gpu/src/graph/primitive_inst.cpp` | prepare_primitive(), update_shape(), sync points |
| `src/plugins/intel_gpu/src/graph/network.cpp` | execute_impl() main loop, flush_frequency |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/primitive_ocl_base.hpp` | execute_stage(), update_stages_flags() |
| `src/plugins/intel_gpu/src/runtime/ocl/ocl_stream.cpp` | set_arguments_impl(), enqueue_kernel() |
| `src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp` | gpu_usm::lock(), gpu_usm::fill(), gpu_usm::copy_to() |
| `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp` | PA execute(), prepare_internal_buffers() |
| `src/plugins/intel_gpu/src/graph/include/paged_attention_inst.h` | get_lockable_input_ids(), key/value_cache_memory_ptr() |
| `src/plugins/intel_gpu/src/plugin/variable_state.cpp` | KV cache allocation type |
| `src/plugins/intel_gpu/src/plugin/ops/constant.cpp` | Constant allocation type |

## Runtime Ad-hoc OpenCL Kernel (For Advanced Debugging)

To build and dispatch a custom OpenCL kernel at runtime for diagnostics:
```cpp
auto builder = engine.create_kernel_builder();
std::vector<cldnn::kernel::ptr> built;
builder->build_kernels(source_code, strlen(source_code),
                       cldnn::KernelFormat::SOURCE, "", built);

cldnn::kernel_arguments_desc args_desc;
args_desc.workGroups.global = {global_size, 1, 1};
args_desc.workGroups.local = {256, 1, 1};
args_desc.arguments.push_back({cldnn::argument_desc::Types::INPUT, 0});
args_desc.arguments.push_back({cldnn::argument_desc::Types::SCALAR, 0});

cldnn::kernel_arguments_data args_data;
args_data.inputs.push_back(buffer_memory);
args_data.scalars = &scalars;

stream.set_arguments(*kernel, args_desc, args_data);
stream.enqueue_kernel(*kernel, args_desc, args_data, {}, false);
stream.flush();
```

Key files: `kernel_builder.hpp` (interface), `ocl_kernel_builder.hpp` (OCL impl), `kernel_args.hpp` (argument types).
