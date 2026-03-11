---
name: dev_primitive_schedule
description: Develop and optimize primitive scheduler for better performance and flexibility. 
---

When developing primitive scheduler feature, always include:
1. **Read all primitive scheduler code**: Read primitive scheduler related code to understand its structure and optimizations. This includes:
    - src/plugins/intel_gpu/src/graph/impls
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2
    - src/plugins/intel_gpu/src/graph/primitive_inst.cpp
2. **Identify optimization opportunities**: Look for areas in the code where performance can be improved, such as inefficient algorithms, redundant computations, or memory bottlenecks.
3. **Implement optimizations**: Apply code changes to optimize the identified areas. This may include algorithmic improvements, better memory management, or leveraging hardware capabilities.
4. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance.
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

Keep explanations conversational. For complex concepts, use multiple analogies.

---

## Core Optimization Strategies (Quick Reference)

### 1. Runtime Implementation Switching
**Goal**: Dynamic selection between OneDNN/OCL based on workload characteristics.

**Key Components**:
- `ImplPool` in `primitive_inst`: Store multiple implementations (OneDNN + OCL)
- Heuristic rules: Prefill (large batch) → OneDNN, Generate (small batch) → OCL
- Switch decision at execution time based on batch size, sequence length, matrix size

**Implementation Points**:
- Add `enable_multi_impl_mode()` to primitive_inst
- Modify `execute()` to check workload and switch if needed
- Track performance stats per implementation type

### 2. Hot-Swap Memory Management
**Goal**: Keep only 1 impl in GPU, offload others to host memory. Target: <5% memory overhead.

**Key Components**:
- `ImplHotSwapManager`: Active impl in GPU, standby as snapshot in host
- `create_snapshot()`: Save kernel binary + metadata to host (free GPU memory)
- `restore_from_snapshot()`: Reload kernel to GPU from cached binary
- Switching overhead: ~15-20ms (acceptable for LLM prefill/generate transition)

**Implementation Points**:
- Store kernel binaries in host memory (`std::vector<uint8_t>`)
- Only keep active impl's GPU resources allocated
- Recompile from cached binary on switch

### 3. Resource Pooling
**Goal**: Share weights and internal buffers between implementations.

**Key Components**:
- `ImplResourcePool`: Manage shared memory resources
- Layout compatibility check: Same data type + size/format compatible
- Scratchpad buffers: Size-based reuse (format-agnostic)
- Weights: Exact format match or compatible transformation

**Implementation Points**:
- Add `get_or_allocate_weights()` that checks for compatible existing memory
- Share intermediate buffers when layout compatible
- Expected savings: 20-35% memory for multi-impl mode

### 4. Lazy Compilation
**Goal**: Defer kernel compilation until first execution. Savings: 20-30% initial memory.

**Key Components**:
- `LazyKernelStub`: Placeholder that compiles real kernel on first `run()`
- Selective strategy: Hot kernels (gemm, conv) compile immediately, others lazy
- Background async compilation thread for predicted kernels

**Implementation Points**:
- Modify `kernels_cache` to support deferred compilation
- Add `should_compile_immediately()` heuristic (critical primitives, prefill stage)
- Store compilation parameters, defer actual `build()` call

### 5. LRU Weights Cache
**Goal**: Limit reordered weights variants to prevent unbounded growth. Savings: 10-15% memory.

**Key Components**:
- `WeightsCacheLRU`: MRU at front, evict LRU when exceeds max_entries (default: 3-5)
- Per-primitive cache limit
- Evict on memory pressure

**Implementation Points**:
- Replace unbounded weights cache with LRU in primitive_inst
- Add `max_weights_cache_entries` config property
- Trigger eviction when memory pressure > threshold (0.85)

### 6. Workload Prediction
**Goal**: Predict next implementation needed to reduce switch latency.

**Key Components**:
- Track sequence length history (circular buffer)
- Detect phase change: Large delta in seq_len (>50%) indicates prefill↔generate transition
- Pattern recognition: Last 3 executions same impl → likely continue

**Implementation Points**:
- Add `WorkloadPredictor` with `seq_length_history` deque
- `detect_phase_change()`: Compare current vs previous seq_len
- Background preload predicted impl to reduce swap time

---

## Implementation Priority

**Phase 1 (Core Functionality)**: Runtime switching + heuristic rules  
**Phase 2 (Memory Optimization)**: Hot-swap manager + resource pooling  
**Phase 3 (Advanced)**: Lazy compilation + LRU cache + prediction  

**Target Metrics**:
- Memory overhead: <5% vs single-impl (with hot-swap)
- Switch latency: <20ms (with background precompile)
- Generate latency improvement: 25-35% for LLM (OneDNN 12ms → OCL 8ms)

**Key Files to Modify**:
- `src/plugins/intel_gpu/src/graph/primitive_inst.cpp`: Add multi-impl pool, switching logic
- `src/plugins/intel_gpu/src/graph/registry/gemm_impls.cpp`: Add heuristic predicates
- `src/plugins/intel_gpu/src/graph/impls/ocl/kernels_cache.cpp`: Add lazy compilation
- `src/plugins/intel_gpu/include/intel_gpu/runtime/properties.hpp`: Add new config properties

**Reference Documentation**: See `PRIMITIVE_SCHEDULER_OPTIMIZATION_EN.md` for detailed design.

