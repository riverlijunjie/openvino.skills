

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
- Current kernel achieves 1.07ms = 20.9% XMX utilization on B580 (96 TFLOPS peak).
    Architecture: 64 WIs (4 SGs), A in SLM (2.2KB), B from global/L2, TILE 32x64x32.
    DO NOT change the fundamental architecture (this was proven best).
    DO NOT add B to SLM (causes regression).
    DO NOT increase WG beyond 64 WIs (causes regression).
    Micro-optimizations to try:
        1. Use intel_sub_group_block_read_us for SLM A reads (vectorized)
        2. Merge paired B scalar reads into vload2 or block reads
        3. Remove K-remainder path (K=2048 divides evenly by 32)
        4. Try TILE_M=48 or 64 (more A rows per WG, same B columns)
        5. Unroll K-loop 2x (reduce loop overhead for 64 tiles)
        6. Add __builtin_prefetch or intel_sub_group_block_prefetch for next B tile
    Hardware: B580 = 20 Xe2 cores, 96 TFLOPS FP16 XMX, 456 GB/s, 24MB L2
    DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
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

## Initial implementation or scaffold:
Here is the initial OCL kernel to start from or a scaffold defining function signatures and code structure:
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM, B loaded directly from global (better for Battlemage)
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

    const int n_base = get_group_id(0) * TILE_N;
    const int col_base = n_base + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    __local half slm_A[TILE_M * SLM_A_STRIDE];

    // Early exit for completely out-of-bounds workgroups
    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const bool col_tile_valid = (n_base + TILE_N <= N);
    const int num_full_k_tiles = K / TILE_K;
    const int k_remainder = K - num_full_k_tiles * TILE_K;

    // Main loop over full K tiles
    for (int kt = 0; kt < num_full_k_tiles; kt++) {
        const int k = kt * TILE_K;

        // Cooperative load A into SLM: 32x32 = 1024 elems, 64 WIs => 16 each
        if (row_tile_valid) {
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
                slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? A[gr * K + k + c] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Two k16 DPAS steps per k32 tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load A from SLM: 4 blocks of 8 rows
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

            // Load B directly from global memory
            int gk_base = k + kk;
            int8 b_val;
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

    // Handle remainder K tile
    if (k_remainder > 0) {
        const int k = num_full_k_tiles * TILE_K;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int elem_id = lid + i * 64;
            int r = elem_id >> 5;
            int c = elem_id & 31;
            int gr = row_base + r;
            int gk = k + c;
            slm_A[r * SLM_A_STRIDE + c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
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

    // Store results with fast path
    if (col_valid) {
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + r) * N + col_idx] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 8 + r) * N + col_idx] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 16 + r) * N + col_idx] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 24 + r) * N + col_idx] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C[(row_base + r) * N + col_idx] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C[(row_base + 8 + r) * N + col_idx] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C[(row_base + 16 + r) * N + col_idx] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C[(row_base + 24 + r) * N + col_idx] = convert_half(((float*)&acc3)[r]);
        }
    }
}

```
Do not change function signatures or argument lists in your improved version.

## Hardware specification:
Your code will run on the following hardware:
**Intel Battlemage** with specs: Xe-cores: 20, Render Slices: 5, Ray Tracing Units: 20, Intel® Xe Matrix Extensions (Intel® XMX) Engines: 160, Xe Vector Engines: 160, Graphics Clock: 2670, GPU Peak TOPS (Int8): 233, TBP: 190, PCI Express Configurations ‡: PCI Express 4.0 x8, Device ID: 0xE20B, Memory: 12 GB GDDR6, Memory Interface: 192 bit, Memory Bandwidth: 456, Memory Speed: 19, ISA_GPU: Xe2-HPG
Please consider the hardware specifications when improving the code. 

## Task:

**Your objectives**:
1. Provide a functional OCL kernel that corresponds to the given reference implementation.
2. Use OCL constructs to efficiently run the code on GPU, explaining your reasoning step by step.
3. Provide the complete OCL code in a code block.

**Critical Requirements:**

1. The kernel must exactly match the reference implementation's functionality.
2. The code must compile and run properly on the GPU.
3. Do not cache or reuse previous results; ensure the code executes fully on each run.
4. Keep all hyperparameters (e.g., batch size, dimensions) unchanged as specified in the reference implementation.
8. Beware of the critical error "Unexpected kernel lambda size. In such cases removing constexpr specifier aligns the captures between the host compiler and the device compiler"! Do not capture constexpr variables in lambda functions passed to kernel launches as this can lead to different lambda sizes between the host and device compiler.

Additional Guidance:

Please structure your response as follows:

1. Analysis:
    * Explain the requirements of writing an efficient OCL kernel for the given task.
    * Explain your proposed kernel structure.
2. OCL code:
    * Provide the complete OCL code in a code block:
```OCL
Your code here
```



## Required Optimizations

Apply the following optimization techniques in your implementation:

- **Register Blocking**: Each work-item computes a THREAD_M×THREAD_N output block in private register arrays. Use `#pragma unroll` on inner loops. Combine with SLM tiling for multi-level memory hierarchy optimization.
- **Kernel Fusion**: Combine sequential operations (e.g., exp → add → activation) into a single kernel. Eliminate intermediate buffers by computing in registers.
- **Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing.