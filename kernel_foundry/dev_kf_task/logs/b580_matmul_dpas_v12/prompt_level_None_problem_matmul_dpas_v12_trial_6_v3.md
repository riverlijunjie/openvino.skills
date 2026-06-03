

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 0.968):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (2KB), B loaded directly from global/L2
// K-loop unrolled 2x: 32 barriers instead of 64 (K=2048 divides evenly by 64)
// No K-remainder path needed.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32

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

    __local ushort slm_A[TILE_M * TILE_K];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);

    // K=2048 divides by 64, so num_k_pairs = 32, no remainder
    const int num_k_pairs = K >> 6;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int b_col = col_valid ? col_idx : (N - 1);
    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Main K-loop: unrolled 2x
    for (int kp = 0; kp < num_k_pairs; kp++) {
        // Process two K-tiles per iteration
        #pragma unroll 2
        for (int sub = 0; sub < 2; sub++) {
            const int k = (kp * 2 + sub) * TILE_K;

            // Cooperative load A into SLM
            if (row_tile_valid) {
                __global const ushort* A_tile = A_us + a_row_base_K + k;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    slm_A[elem_id] = A_tile[r * K + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    int gr = row_base + r;
                    slm_A[elem_id] = (gr < M) ? A_us[gr * K + k + c] : (ushort)0;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Two k16 DPAS steps per k32 tile
            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk += 16) {
                // Load A from SLM using vectorized subgroup block reads
                short8 a0, a1, a2, a3;
                __local const ushort* slm_base = slm_A + kk;

                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm_base + r * TILE_K);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm_base + (8 + r) * TILE_K);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm_base + (16 + r) * TILE_K);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm_base + (24 + r) * TILE_K);

                // Load B from global — paired reads packed into int
                int b_off = (k + kk) * N + b_col;
                int8 b_val;

                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = B_us[b_off];
                    ushort s1 = B_us[b_off + N];
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                    b_off += N2;
                }

                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Store results
    if (col_valid) {
        __global half* C_col = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 0.969):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (2KB), B loaded directly from global/L2
// K-loop unrolled 2x: process two K-tiles per iteration
// K must be divisible by 64 (K=2048) — no remainder path needed
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32

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

    __local ushort slm_A[TILE_M * TILE_K];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);

    // K=2048 divides evenly by 64, so num_k_pairs = 32
    const int num_k_pairs = K >> 6;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int b_col = col_valid ? col_idx : (N - 1);
    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Main K-loop: unrolled 2x, no remainder (K divisible by 64)
    for (int kp = 0; kp < num_k_pairs; kp++) {
        // ===== First K-tile =====
        {
            const int k = (kp << 6);

            // Cooperative load A into SLM
            if (row_tile_valid) {
                __global const ushort* A_tile = A_us + a_row_base_K + k;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    slm_A[elem_id] = A_tile[r * K + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    int gr = row_base + r;
                    slm_A[elem_id] = (gr < M) ? A_us[gr * K + k + c] : (ushort)0;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Two k16 DPAS steps
            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk += 16) {
                short8 a0, a1, a2, a3;
                __local const ushort* slm_base = slm_A + kk;

                // Vectorized SLM reads for A
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm_base + r * TILE_K);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm_base + (8 + r) * TILE_K);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm_base + (16 + r) * TILE_K);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm_base + (24 + r) * TILE_K);

                // Load B: use vload2 to merge paired scalar reads into single 32-bit access
                int b_off = (k + kk) * N + b_col;
                int8 b_val;

                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = B_us[b_off];
                    ushort s1 = B_us[b_off + N];
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                    b_off += N2;
                }

                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // ===== Second K-tile =====
        {
            const int k = (kp << 6) + TILE_K;

            if (row_tile_valid) {
                __global const ushort* A_tile = A_us + a_row_base_K + k;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    slm_A[elem_id] = A_tile[r * K + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    int gr = row_base + r;
                    slm_A[elem_id] = (gr < M) ? A_us[gr * K + k + c] : (ushort)0;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk += 16) {
                short8 a0, a1, a2, a3;
                __local const ushort* slm_base = slm_A + kk;

                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm_base + r * TILE_K);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm_base + (8 + r) * TILE_K);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm_base + (16 + r) * TILE_K);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm_base + (24 + r) * TILE_K);

                int b_off = (k + kk) * N + b_col;
                int8 b_val;

                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = B_us[b_off];
                    ushort s1 = B_us[b_off + N];
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                    b_off += N2;
                }

                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Store results
    if (col_valid) {
        __global half* C_col = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 0.959):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (double-buffered: 2x 2KB = 4KB), B loaded directly from global/L2
// K=2048 divides evenly by 32, no remainder path needed.
// Double-buffering: load next A tile while computing on current tile.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_TILE (TILE_M * TILE_K)

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

    // Double-buffered SLM for A: overlap load of next tile with compute on current
    __local ushort slm_A[2 * SLM_TILE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const int num_k_tiles = K >> 5;  // K / 32

    const int b_col = col_valid ? col_idx : (N - 1);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Precompute per-WI A-load offsets (16 elements per WI, stride 64)
    // elem_id = lid + i*64, r = elem_id >> 5, c = elem_id & 31
    // These are constant across K-tiles; only the k-offset changes.
    int a_local_off[16];
    int a_r[16];
    int a_c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int elem_id = lid + i * 64;
        a_r[i] = elem_id >> 5;
        a_c[i] = elem_id & 31;
        a_local_off[i] = elem_id;
    }

    // Load first A tile into buffer 0
    {
        __local ushort* dst = slm_A;
        if (row_tile_valid) {
            __global const ushort* A_tile = A_us + a_row_base_K;
            #pragma unroll
            for (int i = 0; i < 16; i++)
                dst[a_local_off[i]] = A_tile[a_r[i] * K + a_c[i]];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int gr = row_base + a_r[i];
                dst[a_local_off[i]] = (gr < M) ? A_us[gr * K + a_c[i]] : (ushort)0;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k = kt * TILE_K;
        const int next_kt = kt + 1;
        const bool has_next = (next_kt < num_k_tiles);

        // Prefetch next A tile into the other buffer (overlaps with compute below)
        // We'll barrier before using it in the next iteration
        __local ushort* cur_slm = slm_A + cur_buf * SLM_TILE;
        __local ushort* next_slm = slm_A + (1 - cur_buf) * SLM_TILE;

        // ---- COMPUTE on current buffer ----
        // k16 step 0
        {
            short8 a0, a1, a2, a3;
            __local const ushort* slm_base = cur_slm;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm_base + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm_base + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm_base + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm_base + (24 + r) * TILE_K);

            int b_off = k * N + b_col;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = B_us[b_off];
                ushort s1 = B_us[b_off + N];
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                b_off += N2;
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // k16 step 1
        {
            short8 a0, a1, a2, a3;
            __local const ushort* slm_base = cur_slm + 16;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm_base + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm_base + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm_base + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm_base + (24 + r) * TILE_K);

            int b_off = (k + 16) * N + b_col;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = B_us[b_off];
                ushort s1 = B_us[b_off + N];
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                b_off += N2;
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ---- LOAD next A tile (after compute to allow overlap) ----
        if (has_next) {
            const int next_k = next_kt * TILE_K;
            // Need barrier before writing to next_slm since previous iteration
            // may have been reading from it. But we've finished reading cur_slm above.
            // We need a barrier to ensure all WIs finished reading cur_slm before
            // we potentially write to next_slm (which was the previous cur_slm's partner).
            barrier(CLK_LOCAL_MEM_FENCE);

            if (row_tile_valid) {
                __global const ushort* A_tile = A_us + a_row_base_K + next_k;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    next_slm[a_local_off[i]] = A_tile[a_r[i] * K + a_c[i]];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int gr = row_base + a_r[i];
                    next_slm[a_local_off[i]] = (gr < M) ? A_us[gr * K + next_k + a_c[i]] : (ushort)0;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = 1 - cur_buf;
        }
    }

    // Store results
    if (col_valid) {
        __global half* C_col = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
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
================== 4 passed, 1 deselected, 1 warning in 0.76s ==================
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
1. Loop Unrolling: Use #pragma unroll N for small, fixed-iteration loops. Manually unroll critical loops when compiler doesn't optimize. Prefer compile-time loop bounds.
2. Use Kernel Attributes: Use __attribute__((reqd_work_group_size(X,Y,Z))) to specify fixed work-group size for compiler optimization. Use __attribute__((vec_type_hint(float4))) to hint vectorization opportunities.

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