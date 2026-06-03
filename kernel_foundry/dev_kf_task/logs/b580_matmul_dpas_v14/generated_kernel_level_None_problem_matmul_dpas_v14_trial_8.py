## Analysis

Looking at the previous versions (all ~1.03-1.05ms), the main bottleneck is likely B memory access. All versions use scalar paired reads for B which is inefficient. The key optimizations to try:

1. **K-loop 2x unroll**: Process two K-steps (each 32) per iteration, reducing loop overhead and barrier count by half (32 barriers → 16 barriers for K=2048)
2. **Double-buffered SLM**: Load next A tile while computing current one - with 2x K unroll, we can overlap the A load of the next step with DPAS of the current step within the same unrolled iteration
3. **Better B load interleaving with DPAS**: Start DPAS as soon as both A and B for that sub-step are ready, overlap next B load with current DPAS
4. **SLM stride = 36**: Try slightly different padding to see if it helps bank conflicts

The SLM budget for double-buffer: 2 × 32 × 36 × 2 bytes = 4608 bytes, well within 64KB.

Key insight: With K-loop 2x unroll + double buffering, we load A[kt+1] into buffer 1 while computing from buffer 0, then compute from buffer 1 while loading A[kt+2] into buffer 0. This halves the number of barriers.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32, K-loop 2x unrolled
// 4 subgroups x 16 WIs = 64 WIs per WG
// Double-buffered SLM: load next A while computing current
// SLM stride = 34 (padded for bank conflict reduction)
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_STRIDE 34
#define SLM_BUF_SIZE (TILE_M * SLM_STRIDE)

inline void load_A_to_slm(
    __global const ushort* A_us,
    __local ushort* dst,
    int row_base, int k_off, int K, int M,
    int lid, bool row_tile_valid,
    __private const int* a_slm_off,
    __private const int* a_r,
    __private const int* a_c,
    int a_row_base_K)
{
    if (row_tile_valid) {
        __global const ushort* src = A_us + a_row_base_K + k_off;
        #pragma unroll
        for (int i = 0; i < 16; i++)
            dst[a_slm_off[i]] = src[a_r[i] * K + a_c[i]];
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int gr = row_base + a_r[i];
            dst[a_slm_off[i]] = (gr < M) ? A_us[gr * K + k_off + a_c[i]] : (ushort)0;
        }
    }
}

inline void compute_k16_step(
    __local const ushort* s,
    __global const ushort* B_us,
    int b_row_start, int b_col, int N,
    float8* acc0, float8* acc1, float8* acc2, float8* acc3,
    bool forward)
{
    // Load B tile: 16 rows x 1 col (per WI via subgroup)
    int8 bv;
    int boff = b_row_start * N + b_col;
    int N2 = N << 1;
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        ushort b0 = B_us[boff];
        ushort b1 = B_us[boff + N];
        ((int*)&bv)[p] = as_int((ushort2)(b0, b1));
        boff += N2;
    }

    // Load A from SLM: 32 rows x 16 cols
    short8 a0, a1, a2, a3;
    #pragma unroll
    for (int r = 0; r < 8; r++)
        ((ushort*)&a0)[r] = intel_sub_group_block_read_us(s + r * SLM_STRIDE);
    #pragma unroll
    for (int r = 0; r < 8; r++)
        ((ushort*)&a1)[r] = intel_sub_group_block_read_us(s + (8 + r) * SLM_STRIDE);
    #pragma unroll
    for (int r = 0; r < 8; r++)
        ((ushort*)&a2)[r] = intel_sub_group_block_read_us(s + (16 + r) * SLM_STRIDE);
    #pragma unroll
    for (int r = 0; r < 8; r++)
        ((ushort*)&a3)[r] = intel_sub_group_block_read_us(s + (24 + r) * SLM_STRIDE);

    // DPAS with boustrophedon ordering
    if (forward) {
        *acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv, *acc0);
        *acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv, *acc1);
        *acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv, *acc2);
        *acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv, *acc3);
    } else {
        *acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv, *acc3);
        *acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv, *acc2);
        *acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv, *acc1);
        *acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv, *acc0);
    }
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
    const int row_base = get_group_id(1) * TILE_M;

    if (row_base >= M || n_base >= N)
        return;

    // Double-buffered SLM
    __local ushort slm_A[2 * SLM_BUF_SIZE];

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int col_idx = n_base + sg_id * 16 + sg_lid;
    const bool col_valid = col_idx < N;
    const int b_col = col_valid ? col_idx : (N - 1);
    const bool row_tile_valid = (row_base + TILE_M <= M);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;

    // Precompute per-WI A-load mapping
    int a_slm_off[16], a_r[16], a_c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int eid = lid + i * 64;
        a_r[i] = eid >> 5;
        a_c[i] = eid & 31;
        a_slm_off[i] = a_r[i] * SLM_STRIDE + a_c[i];
    }

    const int num_k_tiles = K >> 5;  // K / 32

    // Load first A tile into buffer 0
    load_A_to_slm(A_us, slm_A, row_base, 0, K, M, lid, row_tile_valid,
                  a_slm_off, a_r, a_c, a_row_base_K);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K-loop: 2x unrolled with double buffering
    // Process pairs of K-tiles to reduce barrier count
    int kt = 0;
    for (; kt < num_k_tiles - 1; kt += 2) {
        int k_off0 = kt * TILE_K;
        int k_off1 = (kt + 1) * TILE_K;
        int buf_cur = (kt & 1);       // 0 on first iter
        int buf_nxt = 1 - buf_cur;

        __local ushort* cur_slm = slm_A + buf_cur * SLM_BUF_SIZE;
        __local ushort* nxt_slm = slm_A + buf_nxt * SLM_BUF_SIZE;

        // ---- Tile kt: compute from buf_cur, load kt+1 into buf_nxt ----
        // Start loading next A tile into buf_nxt
        // But we need to compute from cur_slm first (it's already loaded and barrier'd)

        // k16 step 0 of tile kt: A cols [0..15]
        {
            __local const ushort* s = cur_slm;

            // Load B for k16 step 0
            int8 bv0;
            int boff = k_off0 * N + b_col;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort b0 = B_us[boff];
                ushort b1 = B_us[boff + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
                boff += (N << 1);
            }

            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(s + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(s + (8 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(s + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(s + (24 + r) * SLM_STRIDE);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv0, acc3);
        }

        // k16 step 1 of tile kt: A cols [16..31]
        {
            __local const ushort* s16 = cur_slm + 16;

            int8 bv1;
            int boff = (k_off0 + 16) * N + b_col;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort b0 = B_us[boff];
                ushort b1 = B_us[boff + N];
                ((int*)&bv1)[p] = as_int((ushort2)(b0, b1));
                boff += (N << 1);
            }

            short8 a0b, a1b, a2b, a3b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0b)[r] = intel_sub_group_block_read_us(s16 + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1b)[r] = intel_sub_group_block_read_us(s16 + (8 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2b)[r] = intel_sub_group_block_read_us(s16 + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3b)[r] = intel_sub_group_block_read_us(s16 + (24 + r) * SLM_STRIDE);

            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3b, bv1, acc3);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2b, bv1, acc2);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1b, bv1, acc1);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0b, bv1, acc0);
        }

        // Load A tile kt+1 into buf_nxt (done after all SLM reads from buf_cur)
        load_A_to_slm(A_us, nxt_slm, row_base, k_off1, K, M, lid, row_tile_valid,
                      a_slm_off, a_r, a_c, a_row_base_K);
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---- Tile kt+1: compute from buf_nxt, load kt+2 into buf_cur ----
        {
            __local const ushort* s = nxt_slm;

            int8 bv0;
            int boff = k_off1 * N + b_col;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort b0 = B_us[boff];
                ushort b1 = B_us[boff + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
                boff += (N << 1);
            }

            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(s + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(s + (8 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(s + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(s + (24 + r) * SLM_STRIDE);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv0, acc3);
        }

        {
            __local const ushort* s16 = nxt_slm + 16;

            int8 bv1;
            int boff = (k_off1 + 16) * N + b_col;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort b0 = B_us[boff];
                ushort b1 = B_us[boff + N];
                ((int*)&bv1)[p] = as_int((ushort2)(b0, b1));
                boff += (N << 1);
            }

            short8 a0b, a1b, a2b, a3b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0b)[r] = intel_sub_group_block_read_us(s16 + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1b)[r] = intel_sub_group_block_read_us(s16 + (8 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2b)[r] = intel_sub_group_block_read_us(s16 + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3b)[r] = intel_sub_group_block_read_us(s16 + (24 + r) * SLM_STRIDE);

            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3b, bv1, acc3);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2b, bv1, acc2);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1b, bv1, acc1);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0b, bv1, acc0);
        }

        // Load A for kt+2 into buf_cur (if there is a next iteration)
        if (kt + 2 < num_k_tiles) {
            load_A_to_slm(A_us, cur_slm, row_base, (kt + 2) * TILE_K, K, M, lid, row_tile_valid,
                          a_slm_off, a_r, a_c, a_row_base_K);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Handle odd remaining tile (if num_k_tiles is odd, kt == num_k_tiles - 1)
    if (kt < num_k_tiles) {
        int k_off = kt * TILE_K;
        int buf = (kt & 1);
        __local ushort* cur_slm = slm_A + buf * SLM_BUF_SIZE;

        // A is already loaded from the previous iteration's prefetch
        // (or from initial load if num_k_tiles == 1)

        {
            __local const ushort* s = cur_slm;
            int8 bv0;
            int boff = k_off * N + b_col;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort b0 = B_us[boff];
                ushort b1 = B_us[boff + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
                boff += (N << 1);
            }

            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(s + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(s + (8 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(s + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(s + (24 + r) * SLM_STRIDE);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv0, acc3);
        }

        {
            __local const ushort* s16 = cur_slm + 16;
            int8 bv1;
            int boff = (k_off + 16) * N + b_col;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort b0 = B_us[boff];
                ushort b1 = B_us[boff + N];
                ((int*)&bv1)[p] = as_int((ushort2)(b0, b1));
                boff += (N << 1);
            }

            short8 a0b, a1b, a2b, a3b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0b)[r] = intel_sub_group_block_read_us(s16 + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1b)[r] = intel_sub_group_block_read_us(s16 + (8 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2b)[r] = intel_sub_group_block_read_us(s16 + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3b)[r] = intel_sub_group_block_read_us(s16 + (24 + r) * SLM_STRIDE);

            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3b, bv1, acc3);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2b, bv1, acc2);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1b, bv1, acc1);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0b, bv1, acc0);
        }
    }

    // Store results
    if (col_valid) {
        __global half* C_out = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_out[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_out[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_out[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_out[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

