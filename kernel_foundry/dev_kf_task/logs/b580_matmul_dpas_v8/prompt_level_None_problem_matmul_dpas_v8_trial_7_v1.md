

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Optimization goals:
- Must use Intel OpenCL DPAS instruction(XMX), e.g. intel_sub_group_f16_f16_matrix_mad_k16.
    CRITICAL: For intel_sub_group_f16_f16_matrix_mad_k16, you MUST use:
        - First operand: short8 (NOT float8)
        - Second operand: int8 (NOT float8)
        - Accumulator: float8 (this one is float8)
    Example:
        short8 a_val = as_short8(intel_sub_group_block_read_us8(...));
        int8 b_val = as_int8(intel_sub_group_block_read_ui8(...));
        float8 acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    Using float8 for the first two operands will compile but will NOT use
    XMX hardware acceleration.
- Improve FLOPS and hide memory latency with tiling + subgroup-friendly mapping.
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

### Version 1 (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Tiled DPAS matmul: C[M,N] = A[M,K] * B[K,N], half precision, float accumulation
// Tile: 32M x 32N per work-group, K stepped by 32 (two DPAS k16 steps)
// 4 subgroups per WG, each subgroup computes 8 rows x 32 cols (two 8x16 DPAS outputs)
// LWS = (16, 4), subgroup_size = 16
// GWS = (ceil(N/32)*16, ceil(M/32)*4)

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define SG_COUNT 4

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);
    const int base_m = wg_m * TILE_M;
    const int base_n = wg_n * TILE_N;

    const int sg_id = get_local_id(1);  // 0..3, which subgroup (row block)
    const int sg_lid = get_sub_group_local_id(); // 0..15

    // Each subgroup handles 8 rows x 32 cols = two 8x16 DPAS blocks
    const int my_row_base = base_m + sg_id * 8;

    // Accumulators: 2 DPAS outputs (left 16 cols, right 16 cols)
    float8 acc0 = 0.0f;  // cols [0..15]
    float8 acc1 = 0.0f;  // cols [16..31]

    // SLM: A tile [32][16+2] padded to avoid bank conflicts, B in VNNI format [8][32] as uint
    // A: 32 rows x 16 cols of half, padded stride = 18
    // B VNNI: 8 pairs x 32 cols as uint = 8*32 uint
    __local ushort slm_a[TILE_M * (TILE_K + 2)];  // 32 * 18 = 576 ushorts
    __local uint slm_b_vnni[8 * TILE_N];           // 8 * 32 = 256 uints

    const int a_stride = TILE_K + 2;  // 18, padded to reduce bank conflicts

    // flat thread id within WG: sg_id * 16 + sg_lid, total 64 threads
    const int flat_id = sg_id * 16 + sg_lid;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        // === Load A[32][16] into slm_a[32][18] ===
        // 64 threads, 512 elements -> 8 elements per thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = flat_id + i * 64;
            int r = idx >> 4;       // idx / 16
            int c = idx & 15;       // idx % 16
            int gr = base_m + r;
            int gc = k0 + c;
            ushort val = 0;
            if (gr < M && gc < K)
                val = as_ushort(A[gr * K + gc]);
            slm_a[r * a_stride + c] = val;
        }

        // === Load B[16][32] and pack into VNNI format in SLM ===
        // VNNI: slm_b_vnni[pair][col] = pack(B[2*pair][col], B[2*pair+1][col])
        // 256 uints to write, 64 threads -> 4 per thread
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = flat_id + i * 64;
            int pair = idx >> 5;    // idx / 32 -> which pair (0..7)
            int col = idx & 31;     // idx % 32 -> which column

            int k_row0 = k0 + pair * 2;
            int k_row1 = k0 + pair * 2 + 1;
            int n_col = base_n + col;

            ushort v0 = 0, v1 = 0;
            if (k_row0 < K && n_col < N)
                v0 = as_ushort(B[k_row0 * N + n_col]);
            if (k_row1 < K && n_col < N)
                v1 = as_ushort(B[k_row1 * N + n_col]);

            slm_b_vnni[pair * TILE_N + col] = ((uint)v1 << 16) | (uint)v0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === DPAS computation ===
        // Load A for this subgroup: 8 rows x 16 cols from slm_a
        // Each WI reads its column (sg_lid) across 8 rows
        __local ushort* a_base = &slm_a[sg_id * 8 * a_stride];

        short8 a_val = as_short8((ushort8)(
            a_base[0 * a_stride + sg_lid],
            a_base[1 * a_stride + sg_lid],
            a_base[2 * a_stride + sg_lid],
            a_base[3 * a_stride + sg_lid],
            a_base[4 * a_stride + sg_lid],
            a_base[5 * a_stride + sg_lid],
            a_base[6 * a_stride + sg_lid],
            a_base[7 * a_stride + sg_lid]));

        // Load B VNNI for cols 0-15: each WI reads its column (sg_lid) from 8 pairs
        __local uint* b_base0 = &slm_b_vnni[0];
        int8 b0 = (int8)(
            as_int(b_base0[0 * TILE_N + sg_lid]),
            as_int(b_base0[1 * TILE_N + sg_lid]),
            as_int(b_base0[2 * TILE_N + sg_lid]),
            as_int(b_base0[3 * TILE_N + sg_lid]),
            as_int(b_base0[4 * TILE_N + sg_lid]),
            as_int(b_base0[5 * TILE_N + sg_lid]),
            as_int(b_base0[6 * TILE_N + sg_lid]),
            as_int(b_base0[7 * TILE_N + sg_lid]));

        // Load B VNNI for cols 16-31
        int8 b1 = (int8)(
            as_int(b_base0[0 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[1 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[2 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[3 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[4 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[5 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[6 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[7 * TILE_N + 16 + sg_lid]));

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b1, acc1);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
    int col0 = base_n + sg_lid;
    int col1 = base_n + 16 + sg_lid;

    float acc0_arr[8], acc1_arr[8];
    vstore8(acc0, 0, acc0_arr);
    vstore8(acc1, 0, acc1_arr);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = my_row_base + i;
        if (row < M) {
            if (col0 < N) C[row * N + col0] = convert_half(acc0_arr[i]);
            if (col1 < N) C[row * N + col1] = convert_half(acc1_arr[i]);
        }
    }
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 44.300):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Tile: 32M x 64N per work-group, K-tile = 32 (two DPAS k=16 steps per SLM load)
// 4 subgroups per WG, each subgroup computes 8M x 64N (4 DPAS calls x 2 K-steps)
// LWS = (16, 4) => 64 WIs, 4 subgroups of 16
// GWS = (ceil(N/64)*16, ceil(M/32)*4)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 4, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    const int sg_id = get_local_id(1);           // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int lid0 = get_local_id(0);

    const int row_base = wg_m + sg_id * 8;

    // Accumulators: 8 rows x 64 cols = 4 DPAS blocks
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // SLM for A[32][32] and B in VNNI[2][8][64]
    __local ushort slm_a[TILE_M * TILE_K];       // 1024 ushorts = 2KB
    __local uint slm_b_vnni[2 * 8 * TILE_N];     // 1024 uints = 4KB

    const int flat_id = sg_id * 16 + lid0;       // 0..63

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    // Check if this WG is fully in-bounds for M and N dimensions
    const int wg_m_end = wg_m + TILE_M;
    const int wg_n_end = wg_n + TILE_N;
    const int m_safe = (wg_m_end <= M);
    const int n_safe = (wg_n_end <= N);

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int k_safe = (k0 + TILE_K <= K);

        // === Cooperative load A[32][32] into SLM ===
        // 1024 elements, 64 threads => 16 elements per thread (strided by 64)
        if (m_safe && k_safe) {
            // Fast path: no bounds checking needed
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 64;
                int r = idx >> 5;
                int c = idx & 31;
                int gr = wg_m + r;
                int gc = k0 + c;
                slm_a[r * TILE_K + c] = A_us[gr * K + gc];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 64;
                int r = idx >> 5;
                int c = idx & 31;
                int gr = wg_m + r;
                int gc = k0 + c;
                ushort val = 0;
                if (gr < M && gc < K)
                    val = A_us[gr * K + gc];
                slm_a[r * TILE_K + c] = val;
            }
        }

        // === Cooperative load B[32][64] into VNNI format ===
        // 1024 uints, 64 threads => 16 per thread (strided by 64)
        if (n_safe && k_safe) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 64;
                int block = idx >> 9;
                int rem = idx & 511;
                int pair = rem >> 6;
                int col = rem & 63;

                int k_base = k0 + block * 16;
                int k_row0 = k_base + pair * 2;
                int k_row1 = k_base + pair * 2 + 1;
                int gcol = wg_n + col;

                ushort v0 = B_us[k_row0 * N + gcol];
                ushort v1 = B_us[k_row1 * N + gcol];
                slm_b_vnni[block * 8 * TILE_N + pair * TILE_N + col] = ((uint)v1 << 16) | (uint)v0;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 64;
                int block = idx >> 9;
                int rem = idx & 511;
                int pair = rem >> 6;
                int col = rem & 63;

                int k_base = k0 + block * 16;
                int k_row0 = k_base + pair * 2;
                int k_row1 = k_base + pair * 2 + 1;
                int gcol = wg_n + col;

                ushort v0 = 0, v1 = 0;
                if (gcol < N) {
                    if (k_row0 < K) v0 = B_us[k_row0 * N + gcol];
                    if (k_row1 < K) v1 = B_us[k_row1 * N + gcol];
                }
                slm_b_vnni[block * 8 * TILE_N + pair * TILE_N + col] = ((uint)v1 << 16) | (uint)v0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === Two K-steps of 16 each ===
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            __local ushort* a_ptr = slm_a + sg_id * 8 * TILE_K + kk * 16;
            short8 a_val = as_short8((ushort8)(
                a_ptr[0 * TILE_K + sg_lid], a_ptr[1 * TILE_K + sg_lid],
                a_ptr[2 * TILE_K + sg_lid], a_ptr[3 * TILE_K + sg_lid],
                a_ptr[4 * TILE_K + sg_lid], a_ptr[5 * TILE_K + sg_lid],
                a_ptr[6 * TILE_K + sg_lid], a_ptr[7 * TILE_K + sg_lid]));

            __local uint* b_base = slm_b_vnni + kk * 8 * TILE_N;

            int8 b0 = (int8)(
                as_int(b_base[0*TILE_N + sg_lid]),      as_int(b_base[1*TILE_N + sg_lid]),
                as_int(b_base[2*TILE_N + sg_lid]),      as_int(b_base[3*TILE_N + sg_lid]),
                as_int(b_base[4*TILE_N + sg_lid]),      as_int(b_base[5*TILE_N + sg_lid]),
                as_int(b_base[6*TILE_N + sg_lid]),      as_int(b_base[7*TILE_N + sg_lid]));

            int8 b1 = (int8)(
                as_int(b_base[0*TILE_N + 16 + sg_lid]), as_int(b_base[1*TILE_N + 16 + sg_lid]),
                as_int(b_base[2*TILE_N + 16 + sg_lid]), as_int(b_base[3*TILE_N + 16 + sg_lid]),
                as_int(b_base[4*TILE_N + 16 + sg_lid]), as_int(b_base[5*TILE_N + 16 + sg_lid]),
                as_int(b_base[6*TILE_N + 16 + sg_lid]), as_int(b_base[7*TILE_N + 16 + sg_lid]));

            int8 b2 = (int8)(
                as_int(b_base[0*TILE_N + 32 + sg_lid]), as_int(b_base[1*TILE_N + 32 + sg_lid]),
                as_int(b_base[2*TILE_N + 32 + sg_lid]), as_int(b_base[3*TILE_N + 32 + sg_lid]),
                as_int(b_base[4*TILE_N + 32 + sg_lid]), as_int(b_base[5*TILE_N + 32 + sg_lid]),
                as_int(b_base[6*TILE_N + 32 + sg_lid]), as_int(b_base[7*TILE_N + 32 + sg_lid]));

            int8 b3 = (int8)(
                as_int(b_base[0*TILE_N + 48 + sg_lid]), as_int(b_base[1*TILE_N + 48 + sg_lid]),
                as_int(b_base[2*TILE_N + 48 + sg_lid]), as_int(b_base[3*TILE_N + 48 + sg_lid]),
                as_int(b_base[4*TILE_N + 48 + sg_lid]), as_int(b_base[5*TILE_N + 48 + sg_lid]),
                as_int(b_base[6*TILE_N + 48 + sg_lid]), as_int(b_base[7*TILE_N + 48 + sg_lid]));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b2, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b3, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = row_base + i;
        if (row < M) {
            float v0 = ((float*)&acc0)[i];
            float v1 = ((float*)&acc1)[i];
            float v2 = ((float*)&acc2)[i];
            float v3 = ((float*)&acc3)[i];

            int col0 = wg_n + sg_lid;
            int col1 = wg_n + 16 + sg_lid;
            int col2 = wg_n + 32 + sg_lid;
            int col3 = wg_n + 48 + sg_lid;

            __global half* out = C + row * N;
            if (col0 < N) out[col0] = convert_half(v0);
            if (col1 < N) out[col1] = convert_half(v1);
            if (col2 < N) out[col2] = convert_half(v2);
            if (col3 < N) out[col3] = convert_half(v3);
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 84.800):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Tile: 32M x 128N per work-group, K-tile = 16
// 4 subgroups per WG, each subgroup computes 8M x 128N (8 DPAS calls per K-step)
// LWS = (16, 4) => 64 WIs, 4 subgroups of 16
// GWS = (ceil(N/128)*16, ceil(M/32)*4)

#define TILE_M 32
#define TILE_N 128
#define TILE_K 16
#define SG_SIZE 16
#define NUM_SG 4

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 4, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    const int sg_id = get_local_id(1);
    const int sg_lid = get_sub_group_local_id();
    const int lid0 = get_local_id(0);

    const int row_base = wg_m + sg_id * 8;

    // 8 accumulators: 8 rows x 128 cols
    float8 acc0 = 0.0f; // cols 0-15
    float8 acc1 = 0.0f; // cols 16-31
    float8 acc2 = 0.0f; // cols 32-47
    float8 acc3 = 0.0f; // cols 48-63
    float8 acc4 = 0.0f; // cols 64-79
    float8 acc5 = 0.0f; // cols 80-95
    float8 acc6 = 0.0f; // cols 96-111
    float8 acc7 = 0.0f; // cols 112-127

    // SLM: A[32][16] = 512 ushorts, B_vnni[8][128] = 1024 uints
    __local ushort slm_a[TILE_M * TILE_K];        // 512 ushorts = 1KB
    __local uint slm_b_vnni[8 * TILE_N];          // 1024 uints = 4KB

    const int flat_id = sg_id * 16 + lid0; // 0..63

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        // === Load A[32][16] into SLM: 512 elems, 64 threads => 8 per thread ===
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = flat_id + i * 64;
            int r = idx >> 4;
            int c = idx & 15;
            int gr = wg_m + r;
            int gc = k0 + c;
            ushort val = 0;
            if (gr < M && gc < K)
                val = A_us[gr * K + gc];
            slm_a[r * TILE_K + c] = val;
        }

        // === Load B[16][128] into VNNI: 1024 uints, 64 threads => 16 per thread ===
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int idx = flat_id + i * 64;
            int pair = idx >> 7;   // / 128
            int col = idx & 127;   // % 128
            int k_row0 = k0 + pair * 2;
            int k_row1 = k0 + pair * 2 + 1;
            int gcol = wg_n + col;
            ushort v0 = 0, v1 = 0;
            if (gcol < N) {
                if (k_row0 < K) v0 = B_us[k_row0 * N + gcol];
                if (k_row1 < K) v1 = B_us[k_row1 * N + gcol];
            }
            slm_b_vnni[pair * TILE_N + col] = ((uint)v1 << 16) | (uint)v0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === Load A for this subgroup ===
        __local ushort* a_ptr = slm_a + sg_id * 8 * TILE_K;
        short8 a_val = as_short8((ushort8)(
            a_ptr[0 * TILE_K + sg_lid], a_ptr[1 * TILE_K + sg_lid],
            a_ptr[2 * TILE_K + sg_lid], a_ptr[3 * TILE_K + sg_lid],
            a_ptr[4 * TILE_K + sg_lid], a_ptr[5 * TILE_K + sg_lid],
            a_ptr[6 * TILE_K + sg_lid], a_ptr[7 * TILE_K + sg_lid]));

        // === Load B VNNI for 8 column blocks ===
        __local uint* b_base = slm_b_vnni;

        #define LOAD_B(offset) (int8)( \
            as_int(b_base[0*TILE_N + (offset) + sg_lid]), as_int(b_base[1*TILE_N + (offset) + sg_lid]), \
            as_int(b_base[2*TILE_N + (offset) + sg_lid]), as_int(b_base[3*TILE_N + (offset) + sg_lid]), \
            as_int(b_base[4*TILE_N + (offset) + sg_lid]), as_int(b_base[5*TILE_N + (offset) + sg_lid]), \
            as_int(b_base[6*TILE_N + (offset) + sg_lid]), as_int(b_base[7*TILE_N + (offset) + sg_lid]))

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, LOAD_B(0),  acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, LOAD_B(16), acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, LOAD_B(32), acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, LOAD_B(48), acc3);
        acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, LOAD_B(64), acc4);
        acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, LOAD_B(80), acc5);
        acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, LOAD_B(96), acc6);
        acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, LOAD_B(112),acc7);

        #undef LOAD_B

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = row_base + i;
        if (row < M) {
            int base_off = row * N + wg_n;
            #define STORE_COL(acc_var, col_off) { \
                int col = wg_n + (col_off) + sg_lid; \
                if (col < N) C[base_off + (col_off) + sg_lid] = convert_half(((float*)&acc_var)[i]); \
            }
            STORE_COL(acc0, 0)
            STORE_COL(acc1, 16)
            STORE_COL(acc2, 32)
            STORE_COL(acc3, 48)
            STORE_COL(acc4, 64)
            STORE_COL(acc5, 80)
            STORE_COL(acc6, 96)
            STORE_COL(acc7, 112)
            #undef STORE_COL
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
================== 4 passed, 1 deselected, 1 warning in 1.03s ==================
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
1. Use Predication: Replace if(cond) x = a; else x = b; with x = select(b, a, cond) for scalar types or x = cond ? a : b. Use select() for vector types: result = select(false_val, true_val, condition).
2. Exploit Sub-groups (OpenCL 2.0+): Use sub-group functions for wavefront-level operations: sub_group_reduce_add(), sub_group_broadcast(), intel_sub_group_shuffle(), intel_sub_group_shuffle_down(). Query sub-group size with get_sub_group_size(). Use __attribute__((intel_reqd_sub_group_size(N))) for Intel GPUs.

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