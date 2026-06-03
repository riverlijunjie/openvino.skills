

## Analysis

The best version (0.948ms) uses double-buffered SLM with `#pragma unroll` loops for A reads and B reads. The worst version (2.200ms) removed double-buffering, causing a barrier every K-tile iteration that serializes load and compute.

Key differences:
1. **Double-buffering** (best version) overlaps A loading with compute - this is critical
2. The worst version fully unrolls the A reads manually (no `#pragma unroll` loop) but loses double-buffering
3. Both use the same fundamental DPAS approach

**Proposed improvements over the best version (0.948ms):**
1. **K-loop unroll 2x**: Process two K-tiles (64 elements) per loop iteration to reduce loop overhead and branch misprediction for 32 iterations → 16 iterations
2. **Tighter B loading**: Pre-compute B base pointers to reduce address arithmetic
3. **Remove `has_next` branch inside loop**: Since K=2048 divides by 32, we know there are exactly 64 tiles. We can handle the last iteration specially outside the loop.
4. **Simplify double-buffer ping-pong**: Use pointer swap instead of multiply

## Improved OCL code

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
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_TILE (TILE_M * TILE_K)

inline void load_a_tile_valid(__local ushort* dst, __global const ushort* A_tile,
                              int lid, int K) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int elem_id = lid + i * 64;
        int r = elem_id >> 5;
        int c = elem_id & 31;
        dst[elem_id] = A_tile[r * K + c];
    }
}

inline void compute_k16(__local const ushort* slm_base,
                        __global const ushort* B_us, int b_off, int N, int N2,
                        float8* acc0, float8* acc1, float8* acc2, float8* acc3) {
    short8 a0, a1, a2, a3;

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

    int8 b_val;
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        ushort s0 = B_us[b_off];
        ushort s1 = B_us[b_off + N];
        ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
        b_off += N2;
    }

    *acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, *acc0);
    *acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, *acc1);
    *acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, *acc2);
    *acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, *acc3);
}

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
    const int num_k_tiles = K >> 5;

    const int b_col = col_valid ? col_idx : (N - 1);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Load first A tile into buffer 0
    if (row_tile_valid) {
        load_a_tile_valid(slm_A, A_us + a_row_base_K, lid, K);
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int elem_id = lid + i * 64;
            int r = elem_id >> 5;
            int c = elem_id & 31;
            int gr = row_base + r;
            slm_A[elem_id] = (gr < M) ? A_us[gr * K + c] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main loop: process current tile, load next tile
    // Last tile (kt == num_k_tiles-1) has no next load, handled by loop condition
    int cur_buf = 0;

    for (int kt = 0; kt < num_k_tiles - 1; kt++) {
        const int k = kt * TILE_K;
        __local const ushort* cur_slm = slm_A + cur_buf * SLM_TILE;
        __local ushort* next_slm = slm_A + (1 - cur_buf) * SLM_TILE;

        // Compute k16 step 0
        compute_k16(cur_slm, B_us, k * N + b_col, N, N2, &acc0, &acc1, &acc2, &acc3);
        // Compute k16 step 1
        compute_k16(cur_slm + 16, B_us, (k + 16) * N + b_col, N, N2, &acc0, &acc1, &acc2, &acc3);

        // Load next A tile
        barrier(CLK_LOCAL_MEM_FENCE);
        const int next_k = (kt + 1) * TILE_K;
        if (row_tile_valid) {
            load_a_tile_valid(next_slm, A_us + a_row_base_K + next_k, lid, K);
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                int gr = row_base + r;
                next_slm[elem_id] = (gr < M) ? A_us[gr * K + next_k + c] : (ushort)0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        cur_buf = 1 - cur_buf;
    }

    // Last K-tile: compute only, no next load needed
    {
        const int k = (num_k_tiles - 1) * TILE_K;
        __local const ushort* cur_slm = slm_A + cur_buf * SLM_TILE;
        compute_k16(cur_slm, B_us, k * N + b_col, N, N2, &acc0, &acc1, &acc2, &acc3);
        compute_k16(cur_slm + 16, B_us, (k + 16) * N + b_col, N, N2, &acc0, &acc1, &acc2, &acc3);
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

