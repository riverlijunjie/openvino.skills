

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Performance optimization techniques to consider:
- Must use Intel OpenCL DPAS instruction(XMX), e.g. intel_sub_group_f16_f16_matrix_mad_k16.
- Double-buffering: Overlap SLM loads with DPAS computation
- Prefetching: Use async_work_group_copy for next K-tile
- Register blocking: Maximize DPAS utilization per register load
- Loop unrolling: Fully unroll K-loop if TILE_K <= 64
- SLM bank conflict avoidance: Pad SLM arrays by +1 column
- Provide explicit launch metadata (GWS/LWS/subgroup hints) in kernel comments.

## Reference code / Task:

This is the reference OCL implementation:
```
// Simple row-major FP16 matmul:
//   C[M,N] = A[M,K] x B[K,N]
// Input/Output dtype: half
// Accumulation dtype: float
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += convert_float(A[row * K + k]) * convert_float(B[k * N + col]);
    }

    C[row * N + col] = convert_half(acc);
}

```

## Previous OCL implementations with scores:

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 7.000):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 16 rows x 64 cols
// 4 subgroups x 16 work-items = 64 work-items per WG
// Each subgroup: 2 vertical 8x16 DPAS tiles = 16 rows x 16 cols
// GWS = (ceil(N/64)*64, ceil(M/16))  LWS = (64, 1)
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_sub_group_id();        // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15

    // Each WG covers 16 rows x 64 cols
    const int col_base = get_group_id(0) * 64 + sg_id * 16;
    const int row_base = get_group_id(1) * 16;

    if (row_base >= M || col_base >= N)
        return;

    // Two 8x16 accumulators (16 rows x 16 cols per subgroup)
    float8 acc0 = 0.0f;  // rows [row_base..row_base+7]
    float8 acc1 = 0.0f;  // rows [row_base+8..row_base+15]

    int k = 0;

    // Double-buffering: load first tile
    short8 a_val0, a_val1;
    int8 b_val;

    // Preload first K-tile
    if (k < K) {
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + r;
            int a_idx = row_idx * K + k + sg_lid;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[a_idx] : (half)0.0h;
            ((short*)&a_val0)[r] = as_short(a_elem);
        }
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + 8 + r;
            int a_idx = row_idx * K + k + sg_lid;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[a_idx] : (half)0.0h;
            ((short*)&a_val1)[r] = as_short(a_elem);
        }
        for (int p = 0; p < 8; p++) {
            int k_row0 = k + 2 * p;
            int k_row1 = k + 2 * p + 1;
            int col_idx = col_base + sg_lid;
            half b0 = (k_row0 < K && col_idx < N) ? B[k_row0 * N + col_idx] : (half)0.0h;
            half b1 = (k_row1 < K && col_idx < N) ? B[k_row1 * N + col_idx] : (half)0.0h;
            short2 packed = (short2)(as_short(b0), as_short(b1));
            ((int*)&b_val)[p] = as_int(packed);
        }
    }

    for (k = 0; k < K; k += 16) {
        short8 next_a0, next_a1;
        int8 next_b;
        int next_k = k + 16;

        // Prefetch next K-tile while computing current
        if (next_k < K) {
            for (int r = 0; r < 8; r++) {
                int row_idx = row_base + r;
                int a_idx = row_idx * K + next_k + sg_lid;
                half a_elem = (row_idx < M && (next_k + sg_lid) < K) ? A[a_idx] : (half)0.0h;
                ((short*)&next_a0)[r] = as_short(a_elem);
            }
            for (int r = 0; r < 8; r++) {
                int row_idx = row_base + 8 + r;
                int a_idx = row_idx * K + next_k + sg_lid;
                half a_elem = (row_idx < M && (next_k + sg_lid) < K) ? A[a_idx] : (half)0.0h;
                ((short*)&next_a1)[r] = as_short(a_elem);
            }
            for (int p = 0; p < 8; p++) {
                int k_row0 = next_k + 2 * p;
                int k_row1 = next_k + 2 * p + 1;
                int col_idx = col_base + sg_lid;
                half b0 = (k_row0 < K && col_idx < N) ? B[k_row0 * N + col_idx] : (half)0.0h;
                half b1 = (k_row1 < K && col_idx < N) ? B[k_row1 * N + col_idx] : (half)0.0h;
                short2 packed = (short2)(as_short(b0), as_short(b1));
                ((int*)&next_b)[p] = as_int(packed);
            }
        }

        // DPAS: two 8x16 tiles reusing B
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val1, b_val, acc1);

        // Swap buffers
        a_val0 = next_a0;
        a_val1 = next_a1;
        b_val = next_b;
    }

    // Store 16x16 result
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc0)[r]);
        }
    }
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 8 + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc1)[r]);
        }
    }
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 5.740):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 64 cols
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();

    const int col_base = get_group_id(0) * 64 + sg_id * 16;
    const int row_base = get_group_id(1) * 32;

    if (row_base >= M || col_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    for (int k = 0; k < K; k += 16) {
        // Load A: 4 blocks of 8 rows x 16 K-elements (each WI loads one K via subgroup)
        short8 a0, a1, a2, a3;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + r;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[row_idx * K + k + sg_lid] : (half)0.0h;
            ((short*)&a0)[r] = as_short(a_elem);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + 8 + r;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[row_idx * K + k + sg_lid] : (half)0.0h;
            ((short*)&a1)[r] = as_short(a_elem);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + 16 + r;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[row_idx * K + k + sg_lid] : (half)0.0h;
            ((short*)&a2)[r] = as_short(a_elem);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + 24 + r;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[row_idx * K + k + sg_lid] : (half)0.0h;
            ((short*)&a3)[r] = as_short(a_elem);
        }

        // Load B: 16 K-rows x 16 cols, packed as int8
        int8 b_val;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int k_row0 = k + 2 * p;
            int k_row1 = k + 2 * p + 1;
            int col_idx = col_base + sg_lid;
            half b0 = (k_row0 < K && col_idx < N) ? B[k_row0 * N + col_idx] : (half)0.0h;
            half b1 = (k_row1 < K && col_idx < N) ? B[k_row1 * N + col_idx] : (half)0.0h;
            ((int*)&b_val)[p] = as_int((short2)(as_short(b0), as_short(b1)));
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
    }

    // Store results
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc0)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 8 + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc1)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 16 + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc2)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 24 + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc3)[r]);
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 118.000):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Launch metadata:
// TILE_M = 32, TILE_N = 64, TILE_K = 16
// LWS = (64, 1, 1) -> 4 subgroups of 16
// GWS = (ceil(N/64)*64, ceil(M/32)*1, 1)
// Each subgroup computes 32 rows x 16 cols = 4 DPAS(8x16) ops
// SLM: A_slm[32][16+1] + B_slm[16][64+1] in half

#define TILE_M 32
#define TILE_N 64
#define TILE_K 16
#define SLM_A_STRIDE (TILE_K + 1)
#define SLM_B_STRIDE (TILE_N + 1)

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    __local half A_slm[TILE_M * SLM_A_STRIDE];
    __local half B_slm[TILE_K * SLM_B_STRIDE];

    const int sg_id = get_sub_group_id();        // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int local_id = get_local_id(0);        // 0..63

    const int col_base = get_group_id(0) * TILE_N + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    // 4 accumulators: rows [0..7], [8..15], [16..23], [24..31] x 16 cols
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    for (int k = 0; k < K; k += TILE_K) {
        // Cooperative load A[TILE_M x TILE_K] into SLM
        // 64 threads load 32*16 = 512 elements -> 8 elements per thread
        for (int i = local_id; i < TILE_M * TILE_K; i += 64) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int global_r = row_base + r;
            int global_c = k + c;
            half val = (global_r < M && global_c < K) ? A[global_r * K + global_c] : (half)0.0h;
            A_slm[r * SLM_A_STRIDE + c] = val;
        }

        // Cooperative load B[TILE_K x TILE_N] into SLM
        // 64 threads load 16*64 = 1024 elements -> 16 elements per thread
        for (int i = local_id; i < TILE_K * TILE_N; i += 64) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int global_r = k + r;
            int global_c = get_group_id(0) * TILE_N + c;
            half val = (global_r < K && global_c < N) ? B[global_r * N + global_c] : (half)0.0h;
            B_slm[r * SLM_B_STRIDE + c] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each subgroup loads its portion from SLM and does DPAS
        // Load B: 16 K-rows x 16 cols for this subgroup from SLM
        int b_col_offset = sg_id * 16;
        int8 b_val;
        for (int p = 0; p < 8; p++) {
            int k_row0 = 2 * p;
            int k_row1 = 2 * p + 1;
            half b0 = B_slm[k_row0 * SLM_B_STRIDE + b_col_offset + sg_lid];
            half b1 = B_slm[k_row1 * SLM_B_STRIDE + b_col_offset + sg_lid];
            short2 packed = (short2)(as_short(b0), as_short(b1));
            ((int*)&b_val)[p] = as_int(packed);
        }

        // Block 0: rows 0..7
        {
            short8 a_val;
            for (int r = 0; r < 8; r++) {
                a_val[r] = as_short(A_slm[(0 + r) * SLM_A_STRIDE + sg_lid]);
            }
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc0);
        }

        // Block 1: rows 8..15
        {
            short8 a_val;
            for (int r = 0; r < 8; r++) {
                a_val[r] = as_short(A_slm[(8 + r) * SLM_A_STRIDE + sg_lid]);
            }
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc1);
        }

        // Block 2: rows 16..23
        {
            short8 a_val;
            for (int r = 0; r < 8; r++) {
                a_val[r] = as_short(A_slm[(16 + r) * SLM_A_STRIDE + sg_lid]);
            }
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc2);
        }

        // Block 3: rows 24..31
        {
            short8 a_val;
            for (int r = 0; r < 8; r++) {
                a_val[r] = as_short(A_slm[(24 + r) * SLM_A_STRIDE + sg_lid]);
            }
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    for (int blk = 0; blk < 4; blk++) {
        float8 acc = (blk == 0) ? acc0 : (blk == 1) ? acc1 : (blk == 2) ? acc2 : acc3;
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + blk * 8 + r;
            int col_idx = col_base + sg_lid;
            if (row_idx < M && col_idx < N) {
                C[row_idx * N + col_idx] = convert_half(((float*)&acc)[r]);
            }
        }
    }
}
```

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] PASSED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] PASSED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] PASSED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] PASSED         [100%]

=============================== warnings summary ===============================
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0]
  /home/openvino-ci-74/miniforge3/envs/kernel_intel/lib/python3.12/site-packages/pyopencl/cache.py:517: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
    _create_built_program_from_source_cached(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 4 passed, 1 deselected, 1 warning in 1.18s ==================
The kernel compiles and is correct, great job!

## Hardware specification:
Your code will run on the following hardware:
**Intel Battlemage** with specs: Xe-cores: 20, Render Slices: 5, Ray Tracing Units: 20, Intel® Xe Matrix Extensions (Intel® XMX) Engines: 160, Xe Vector Engines: 160, Graphics Clock: 2670, GPU Peak TOPS (Int8): 233, TBP: 190, PCI Express Configurations ‡: PCI Express 4.0 x8, Device ID: 0xE20B, Memory: 12 GB GDDR6, Memory Interface: 192 bit, Memory Bandwidth: 456, Memory Speed: 19, ISA_GPU: Xe2-HPG
Please consider the hardware specifications when improving the code. 

## Task:

**Your objectives**:
1. Analyze the previous versions and their results (why does one achieve better results than the other?).
2. Identify any inefficiencies and bottlenecks.
3. Propose specific improvements or options to take the best of all prior versions, explaining your reasoning step by step.

4. Provide a new kernel that achieves better performance **on the target hardware**. Provide the complete, improved code in a code block.

**Optimization strategies**:

Here are some potential strategies to improve the kernel runtime:
1. Exploit Constant Cache: Use __constant address space (limited to ~64KB) for frequently accessed read-only data. Declare as __constant float coeffs[N]. All work-items should access same location simultaneously for best performance.
2. Loop Unrolling: Use #pragma unroll N for small, fixed-iteration loops. Manually unroll critical loops when compiler doesn't optimize. Prefer compile-time loop bounds.

**Critical Requirements:**

1. The kernel must exactly match the reference implementation's functionality.
2. The code must compile and run properly on the GPU.
3. Do not cache or reuse previous results; ensure the code executes fully on each run.
4. Keep all hyperparameters (e.g., batch size, dimensions) unchanged as specified in the reference implementation.
8. Beware of the critical error "Unexpected kernel lambda size. In such cases removing constexpr specifier aligns the captures between the host compiler and the device compiler"! Do not capture constexpr variables in lambda functions passed to kernel launches as this can lead to different lambda sizes between the host and device compiler.

Additional Guidance:

1. Clearly comment on any performance optimizations you implement.
2. If you change the kernel structure, explain why.
3. Anticipate and address possible runtime or compilation errors.

Please structure your response as follows:

1. Analysis:
    * Summarize the issues found in the previous kernel and log.
    * Explain your proposed changes and optimizations.
2. Improved OCL code:
    * Provide the complete, improved OCL code in a code block:
```OCL
Your code here
```


## Required Optimizations

Apply the following optimization techniques in your implementation:

- **Register Blocking**: Each work-item computes a THREAD_M×THREAD_N output block in private register arrays. Use `#pragma unroll` on inner loops. Combine with SLM tiling for multi-level memory hierarchy optimization.
- **Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade recomputation for memory savings (e.g., Flash-Attention style). Maintain running accumulators across blocks with proper rescaling.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.