

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

## Previous OCL implementations with scores:

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.910):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM, B loaded directly from global/L2
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
// K must be divisible by 32 (no remainder path needed)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_A_STRIDE 32

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

    // SLM for A tile: 32 rows x 32 cols, no padding needed if we use careful access
    __local ushort slm_A[TILE_M * SLM_A_STRIDE];

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
    const int num_k_tiles = K / TILE_K;  // K divisible by 32, no remainder

    // Precompute B column offset
    const int b_col_off = col_valid ? col_idx : 0;

    // Main loop over K tiles - no remainder needed (K % 32 == 0)
    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k = kt * TILE_K;

        // Cooperative load A into SLM: 32x32 = 1024 elems, 64 WIs => 16 each
        // Each WI loads 16 half values
        if (row_tile_valid) {
            __global const ushort* A_us = (__global const ushort*)A;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;   // elem_id / 32
                int c = elem_id & 31;   // elem_id % 32
                slm_A[r * SLM_A_STRIDE + c] = A_us[(row_base + r) * K + k + c];
            }
        } else {
            __global const ushort* A_us = (__global const ushort*)A;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                int gr = row_base + r;
                slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? A_us[gr * K + k + c] : (ushort)0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Two k16 DPAS steps per k32 tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load A from SLM using intel_sub_group_block_read_us for vectorized access
            // Each sub-group reads 8 rows × 16 cols via block reads
            // block_read_us reads 16 consecutive ushort values (one per lane)
            __local const ushort* slm_ptr;

            short8 a0, a1, a2, a3;

            // Block 0: rows 0-7
            slm_ptr = &slm_A[0 * SLM_A_STRIDE + kk];
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(
                    (__local const ushort*)(slm_ptr + r * SLM_A_STRIDE));
            }

            // Block 1: rows 8-15
            slm_ptr = &slm_A[8 * SLM_A_STRIDE + kk];
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(
                    (__local const ushort*)(slm_ptr + r * SLM_A_STRIDE));
            }

            // Block 2: rows 16-23
            slm_ptr = &slm_A[16 * SLM_A_STRIDE + kk];
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(
                    (__local const ushort*)(slm_ptr + r * SLM_A_STRIDE));
            }

            // Block 3: rows 24-31
            slm_ptr = &slm_A[24 * SLM_A_STRIDE + kk];
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(
                    (__local const ushort*)(slm_ptr + r * SLM_A_STRIDE));
            }

            // Load B directly from global memory
            // Merge paired scalar reads: each DPAS k16 needs 16 rows of B packed as 8 int values
            // Each int packs two consecutive k-row halves: (k_row, k_row+1)
            int gk_base = k + kk;
            int8 b_val;

            if (col_valid) {
                __global const ushort* B_us = (__global const ushort*)B;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int k_row0 = gk_base + 2 * p;
                    // Load two consecutive B values and pack into int
                    ushort s0 = B_us[k_row0 * N + b_col_off];
                    ushort s1 = B_us[(k_row0 + 1) * N + b_col_off];
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
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

    // Store results
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

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 2.970):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM, B loaded directly from global (best for Battlemage)
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
// Subgroup size hint: 16

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_A_STRIDE 32

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

    // SLM for A tile: 32 rows x 32 cols, stride=32 (no padding needed for block reads)
    __local ushort slm_A[TILE_M * SLM_A_STRIDE];

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
    const int num_k_tiles = K / TILE_K;  // K guaranteed divisible by 32

    // Precompute B column offset
    __global const half* B_col = B + col_idx;

    // Main loop over K tiles - unrolled 2x (K/32 = 64 iterations -> 32 iterations of 2)
    int kt = 0;
    for (; kt < num_k_tiles - 1; kt += 2) {
        // ============ First K-tile ============
        {
            const int k = kt * TILE_K;

            // Cooperative load A into SLM: 32x32 = 1024 elems, 64 WIs => 16 each
            if (row_tile_valid) {
                __global const half* A_base = A + row_base * K + k;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;   // / 32
                    int c = elem_id & 31;   // % 32
                    slm_A[r * SLM_A_STRIDE + c] = as_ushort(A_base[r * K + c]);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    int gr = row_base + r;
                    slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? as_ushort(A[(gr) * K + k + c]) : (ushort)0;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Two k16 DPAS steps per k32 tile
            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk += 16) {
                // Load A from SLM using block reads - each block_read_us gets one ushort per lane
                short8 a0, a1, a2, a3;
                __local const ushort* slm_ptr = slm_A + kk;

                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a0)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + r * SLM_A_STRIDE));
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a1)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + (8 + r) * SLM_A_STRIDE));
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a2)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + (16 + r) * SLM_A_STRIDE));
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a3)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + (24 + r) * SLM_A_STRIDE));

                // Load B from global - merge paired reads into single 32-bit loads
                int gk_base = k + kk;
                int8 b_val;
                if (col_valid) {
                    __global const half* B_ptr = B_col + gk_base * N;
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        ushort s0 = as_ushort(B_ptr[2 * p * N]);
                        ushort s1 = as_ushort(B_ptr[(2 * p + 1) * N]);
                        ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
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

        // ============ Second K-tile ============
        {
            const int k = (kt + 1) * TILE_K;

            if (row_tile_valid) {
                __global const half* A_base = A + row_base * K + k;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    slm_A[r * SLM_A_STRIDE + c] = as_ushort(A_base[r * K + c]);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    int gr = row_base + r;
                    slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? as_ushort(A[(gr) * K + k + c]) : (ushort)0;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk += 16) {
                short8 a0, a1, a2, a3;
                __local const ushort* slm_ptr = slm_A + kk;

                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a0)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + r * SLM_A_STRIDE));
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a1)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + (8 + r) * SLM_A_STRIDE));
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a2)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + (16 + r) * SLM_A_STRIDE));
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a3)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + (24 + r) * SLM_A_STRIDE));

                int gk_base = k + kk;
                int8 b_val;
                if (col_valid) {
                    __global const half* B_ptr = B_col + gk_base * N;
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        ushort s0 = as_ushort(B_ptr[2 * p * N]);
                        ushort s1 = as_ushort(B_ptr[(2 * p + 1) * N]);
                        ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
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
    }

    // Handle last tile if num_k_tiles is odd
    if (kt < num_k_tiles) {
        const int k = kt * TILE_K;

        if (row_tile_valid) {
            __global const half* A_base = A + row_base * K + k;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                slm_A[r * SLM_A_STRIDE + c] = as_ushort(A_base[r * K + c]);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                int gr = row_base + r;
                slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? as_ushort(A[(gr) * K + k + c]) : (ushort)0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            short8 a0, a1, a2, a3;
            __local const ushort* slm_ptr = slm_A + kk;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + r * SLM_A_STRIDE));
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + (8 + r) * SLM_A_STRIDE));
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + (16 + r) * SLM_A_STRIDE));
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us((__local const ushort*)(slm_ptr + (24 + r) * SLM_A_STRIDE));

            int gk_base = k + kk;
            int8 b_val;
            if (col_valid) {
                __global const half* B_ptr = B_col + gk_base * N;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = as_ushort(B_ptr[2 * p * N]);
                    ushort s1 = as_ushort(B_ptr[(2 * p + 1) * N]);
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                }
            } else {
                b_val = (int8)(0);
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }
    }

    // Store results
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.230):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM, B loaded directly from global/L2
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
// K must be divisible by 32 (K=2048)

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
    // K is divisible by TILE_K (K=2048, TILE_K=32), no remainder needed
    const int num_k_tiles = K / TILE_K;

    // Precompute A and B base pointers
    __global const half* A_base = A + row_base * K;
    __global const half* B_base = B;

    // Main loop over K tiles — unroll 2x to reduce loop overhead
    int kt = 0;
    for (; kt < (num_k_tiles & ~1); kt += 2) {
        // ---- First K tile of the pair ----
        {
            const int k = kt * TILE_K;

            // Cooperative load A into SLM: 32x32 = 1024 elems, 64 WIs => 16 each
            if (row_tile_valid) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    slm_A[r * SLM_A_STRIDE + c] = A_base[r * K + k + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    int gr = row_base + r;
                    slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? A_base[r * K + k + c] : (half)0.0h;
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

                // Load B from global — merge paired reads
                int gk_base = k + kk;
                int8 b_val;
                if (col_valid) {
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        int k_row0 = gk_base + 2 * p;
                        short s0 = as_short(B_base[k_row0 * N + col_idx]);
                        short s1 = as_short(B_base[(k_row0 + 1) * N + col_idx]);
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

        // ---- Second K tile of the pair ----
        {
            const int k = (kt + 1) * TILE_K;

            if (row_tile_valid) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    slm_A[r * SLM_A_STRIDE + c] = A_base[r * K + k + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    int gr = row_base + r;
                    slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? A_base[r * K + k + c] : (half)0.0h;
                }
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

                int gk_base = k + kk;
                int8 b_val;
                if (col_valid) {
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        int k_row0 = gk_base + 2 * p;
                        short s0 = as_short(B_base[k_row0 * N + col_idx]);
                        short s1 = as_short(B_base[(k_row0 + 1) * N + col_idx]);
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
    }

    // Handle odd remaining tile (if num_k_tiles is odd)
    if (kt < num_k_tiles) {
        const int k = kt * TILE_K;

        if (row_tile_valid) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                slm_A[r * SLM_A_STRIDE + c] = A_base[r * K + k + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                int gr = row_base + r;
                slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? A_base[r * K + k + c] : (half)0.0h;
            }
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

            int gk_base = k + kk;
            int8 b_val;
            if (col_valid) {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int k_row0 = gk_base + 2 * p;
                    short s0 = as_short(B_base[k_row0 * N + col_idx]);
                    short s1 = as_short(B_base[(k_row0 + 1) * N + col_idx]);
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
    }

    // Store results
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
================== 4 passed, 1 deselected, 1 warning in 0.79s ==================
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
1. Avoid Bank Conflicts: Local memory is organized into banks (typically 32 banks). Pad shared arrays to avoid stride conflicts, e.g., __local float tile[TILE_SIZE][TILE_SIZE + 1] for transpose operations. Use sequential access patterns within wavefronts.
2. Use Local Memory Strategically: Use __local memory for data reused across work-items in a work-group. Implement tile-based algorithms where each work-group loads a tile into local memory, processes it, then moves to the next tile. Declare with __local float tile[TILE_SIZE].

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