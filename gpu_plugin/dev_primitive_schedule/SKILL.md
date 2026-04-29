---
name: dev_primitive_schedule
description: Develop and optimize primitive scheduler for better performance and flexibility. 
---

When developing primitive scheduler feature, always include:
1. **Read all primitive scheduler code**: Read primitive scheduler related code to understand its structure and optimizations. This includes:
    - src/plugins/intel_gpu/src/graph/impls
    - src/plugins/intel_gpu/src/graph/impls/ocl_v2
    - src/plugins/intel_gpu/src/graph/primitive_inst.cpp

2. **Understand the current implementation history**: 
    - Read PRIMITIVE_SCHEDULER_OPTIMIZATION_EN.md to understand the current implementation and optimization strategies for primitive scheduler.
3. **Identify optimization opportunities**: Look for areas in the code where performance can be improved, such as inefficient algorithms, redundant computations, or memory bottlenecks.
4. **Implement optimizations**: Apply code changes to optimize the identified areas. This may include algorithmic improvements, better memory management, or leveraging hardware capabilities.
    note: code style is snake case


5. **Document optimization**:
    - Summarize the optimizations applied and their impact on performance.
    - Include before-and-after performance metrics to demonstrate improvements.
    - Include the ratio to hardware roofline to show efficiency gains.

6. **How to add a new kernel**:
    - Implement the kernel per requirements: input/output layout and format
    - Optimize the kernel to close hardware roofline
        1) memory bound should be >95% roofline, means more than 95% memory bandwidth
        2) compute bound should be >75% roofline, means more than 75% compute utilization
    - Create a seperated directory to write unit tests for the kernel, and ensure it can pass accuracy test and performance test.
        1) input tensor: f16:bfyx:1x1x4096:nopad
            weight tensor: u4:bfyx:6144x4096:nopad
            weight scale tensor: f16:fbyx:6144x32:nopad
            weight zp tensor: u8:fbyx:6144x32:nopad
            bias: f16:bfyx:0x1x1:nopad
            output tensor: f16:bfyx:1x1x6144:nopad
        2) latency target: <8ms on Intel GPU for M=1 GEMV with INT4 weights
    - Put kernel source code into `src/plugins/intel_gpu/src/graph/impls/ocl_v2` (or other relevant directory).
    - Implement a primitive impl to wrap the kernel and make it available for selection in the primitive scheduler.
    - Ensure the kernel is integrated into the primitive scheduler's implementation pool and can be selected based on workload characteristics.

7. **Remote build and validation**:
    - Don't create new commit
    - Don't push any commit to https://github.com/openvinotoolkit/openvino/
    - Don't change weights layout
    - Remote machine: openvino-ci-74@10.239.140.155, pwd: openvino
    - Remote openvino directory: /mnt/river/moe/openvino
    - Remote copy modified files to remote machine's openvino directory.
    - Remote build command: cd /mnt/river/moe/openvino && source build_release.sh
    - Remote test environment setup: cd /mnt/river/moe/openvino.genai/ && source venv/bin/activate
    - Remote e2e test command: cd /mnt/river/moe/openvino.genai/tools/llm_bench && source runme.sh
    - Validate generated output
        1. The line of "Generated: " should contain many chars but shouldn't contain ",,," or "!!!" or "???", not only warmup but also real runs.
        2. Second token choose ocl implementation, check if it is "impl switch onednn → ocl"
    - Kernel debug steps:
        1. Reference to src/plugins/intel_gpu/docs/gpu_debug_utils.md to check kernel execution details with OV_VERBOSE logs and dump primitive's kernel execution input and output tensor.
        2. Check if fullyconnectedcompressed:__module.model.layers.0.self_attn.q_proj/ov_ext::linear/MatMul_fused_3FCs is using OCL implementation in the generated output.
        3. Search “Enqueue stage gemm_generate_opt” to confirm if gemm_ocl is correctly selected for M=1 GEMV with INT4 weights in the generated output.
        4. Confirm if the generated output shows "impl switch onednn → ocl" when switching from prefill to generate stage, both for warmup and real runs.
        5. Compare the input/output of fullyconnectedcompressed:__module.model.layers.0.self_attn.q_proj/ov_ext::linear/MatMul_fused_3FCs for oneDNN and ocl implementation, check if they are the same (or very close with some tolerance) to make sure the switch is correct.
    - Must confirm before finish task:
        1. e2e test result must be the same with OneDNN implementation, both for warmup and real runs.
        2. Provide e2e performance data for oneDNN vs OCL implementation, and confirm OCL implementation has better performance than OneDNN implementation in generate stage, especially for q_proj and up_proj.

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

