

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 5.740):
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

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.080):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A tile cached in SLM (shared by 4 SGs), B loaded directly from global
// Fast path skips boundary checks when tile is fully in-bounds
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_A_STRIDE (TILE_K + 2)

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
    const int lid = get_local_id(0);

    const int col_base = get_group_id(0) * TILE_N + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    __local half slm_A[TILE_M * SLM_A_STRIDE];

    if (row_base >= M || col_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const bool col_tile_valid = col_valid; // sg_lid < 16, so if col_base < N, all valid

    // Number of full K tiles and remainder
    const int num_full_k_tiles = K / TILE_K;
    const int k_remainder = K - num_full_k_tiles * TILE_K;

    // Fast path: process full K tiles where no K boundary check is needed
    for (int kt = 0; kt < num_full_k_tiles; kt++) {
        const int k = kt * TILE_K;

        // Cooperative load A into SLM: 32x32 = 1024 elems, 64 WIs => 16 each
        if (row_tile_valid) {
            // No row boundary check needed
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                slm_A[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + k + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                int gr = row_base + r;
                half val = (gr < M) ? A[gr * K + k + c] : (half)0.0h;
                slm_A[r * SLM_A_STRIDE + c] = val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Two k16 DPAS steps per k32 tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load A from SLM
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a0)[r] = as_short(slm_A[r * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a1)[r] = as_short(slm_A[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a2)[r] = as_short(slm_A[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a3)[r] = as_short(slm_A[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);

            // Load B from global: no K boundary check needed for full tiles
            int8 b_val;
            int gk_base = k + kk;
            if (col_valid) {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int k_row0 = gk_base + 2 * p;
                    int k_row1 = k_row0 + 1;
                    short s0 = as_short(B[k_row0 * N + col_idx]);
                    short s1 = as_short(B[k_row1 * N + col_idx]);
                    ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                }
            } else {
                b_val = (int8)(0);
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Handle remainder K tile if any
    if (k_remainder > 0) {
        const int k = num_full_k_tiles * TILE_K;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int elem_id = lid + i * 64;
            int r = elem_id >> 5;
            int c = elem_id & 31;
            int gr = row_base + r;
            int gk = k + c;
            half val = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
            slm_A[r * SLM_A_STRIDE + c] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a0)[r] = as_short(slm_A[r * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a1)[r] = as_short(slm_A[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a2)[r] = as_short(slm_A[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a3)[r] = as_short(slm_A[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);

            int8 b_val;
            int gk_base = k + kk;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_row0 = gk_base + 2 * p;
                int k_row1 = k_row0 + 1;
                short s0 = (k_row0 < K && col_valid) ? as_short(B[k_row0 * N + col_idx]) : (short)0;
                short s1 = (k_row1 < K && col_valid) ? as_short(B[k_row1 * N + col_idx]) : (short)0;
                ((int*)&b_val)[p] = as_int((short2)(s0, s1));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }
    }

    // Store results
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + r;
        if (row_idx < M && col_valid)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc0)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 8 + r;
        if (row_idx < M && col_valid)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc1)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 16 + r;
        if (row_idx < M && col_valid)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc2)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 24 + r;
        if (row_idx < M && col_valid)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc3)[r]);
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 5.720):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 64 cols, K-step: 32, double-buffered A in SLM
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// Double-buffer: load next A tile while computing current
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_A_STRIDE (TILE_K + 2)

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
    const int lid = get_local_id(0);

    const int col_base = get_group_id(0) * TILE_N + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    // Double-buffered SLM for A tile
    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];

    if (row_base >= M || col_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prefetch first A tile into buffer 0
    {
        int buf_off = 0;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int elem_id = lid + i * 64;
            int r = elem_id / TILE_K;
            int c = elem_id % TILE_K;
            int gr = row_base + r;
            int gk = c;
            half val = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
            slm_A[buf_off + r * SLM_A_STRIDE + c] = val;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k = kt * TILE_K;
        int cur_buf = (kt & 1) * TILE_M * SLM_A_STRIDE;
        int nxt_buf = ((kt + 1) & 1) * TILE_M * SLM_A_STRIDE;

        // Process two k16 sub-tiles from current buffer AND prefetch next A tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load A from SLM (current buffer)
            short8 a0, a1, a2, a3;

            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((short*)&a0)[r] = as_short(slm_A[cur_buf + r * SLM_A_STRIDE + kk + sg_lid]);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((short*)&a1)[r] = as_short(slm_A[cur_buf + (8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((short*)&a2)[r] = as_short(slm_A[cur_buf + (16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((short*)&a3)[r] = as_short(slm_A[cur_buf + (24 + r) * SLM_A_STRIDE + kk + sg_lid]);
            }

            // Load B from global: 16 K-rows x 16 cols packed as int8
            // Use subgroup block read pattern for coalesced access
            int8 b_val;
            int bk_base = k + kk;
            int col_idx = col_base + sg_lid;
            bool col_valid = (col_idx < N);

            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_row0 = bk_base + 2 * p;
                int k_row1 = bk_base + 2 * p + 1;
                short s0 = (k_row0 < K && col_valid) ? as_short(B[k_row0 * N + col_idx]) : (short)0;
                short s1 = (k_row1 < K && col_valid) ? as_short(B[k_row1 * N + col_idx]) : (short)0;
                ((int*)&b_val)[p] = as_int((short2)(s0, s1));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // Prefetch next A tile into next buffer (overlaps with barrier latency)
        if (kt + 1 < num_k_tiles) {
            int next_k = (kt + 1) * TILE_K;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id / TILE_K;
                int c = elem_id % TILE_K;
                int gr = row_base + r;
                int gk = next_k + c;
                half val = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
                slm_A[nxt_buf + r * SLM_A_STRIDE + c] = val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results using subgroup-width writes
    #pragma unroll
    for (int blk = 0; blk < 4; blk++) {
        float8 acc = (blk == 0) ? acc0 : (blk == 1) ? acc1 : (blk == 2) ? acc2 : acc3;
        int col_idx = col_base + sg_lid;
        if (col_idx < N) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int row_idx = row_base + blk * 8 + r;
                if (row_idx < M) {
                    C[row_idx * N + col_idx] = convert_half(((float*)&acc)[r]);
                }
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
================== 4 passed, 1 deselected, 1 warning in 0.80s ==================
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
1. Coalesced Access: Ensure work-items in the same wavefront/warp access contiguous memory addresses. For row-major data, have adjacent work-items (consecutive get_global_id(0)) access adjacent memory locations. Use linear indexing that matches your data layout.
2. Use Built-in Functions: Prefer built-in functions (mad, fma, dot, cross, length) over manual implementations. Use fast_* variants (fast_normalize, fast_length) when precision allows. Use native_* for maximum speed with reduced precision.

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