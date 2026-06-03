

## Analysis

All three previous versions achieve identical 0.274ms performance with the same architecture: 32 SGs (4×8), 32×32 per-SG tile, 128×256 WG tile, 128 K-unroll. The differences (boustrophedon ordering, SWIZZLE=2 vs 4) don't measurably help.

The current approach uses 512 work-items per WG (32 subgroups × 16 WIs). Each SG computes a 32×32 tile. This is a very large WG that may limit occupancy on B580's 20 Xe-cores.

Key observations:
1. **Occupancy**: 512 WIs/WG is large. B580 has 20 Xe-cores with 8 threads/EU × 16 EUs = 128 threads per Xe-core. A 512-WI WG needs 32 SGs, but each Xe-core only has limited SG slots.
2. **Register pressure**: 32×32 tile per SG = 16 accumulators (float8) = 128 floats = 512 bytes per SG. Plus A/B data. This is heavy.
3. **The user mentioned a baseline with 64 WIs achieving 0.948ms**. The current 512-WI versions are already 3.5× faster at 0.274ms.

Let me try a different approach: **smaller WG with more WGs for better occupancy**, or refine the current best approach with better prefetching.

Let me try a variant with:
- Same 4×8 SG layout (32 SGs), same 128×256 WG tile
- **Software pipelining**: load iteration N+1's data while computing iteration N
- **Cross-iteration prefetch**: pre-load the first K-block of next iteration during last K-block compute
- **SWIZZLE=8** for potentially better L2 reuse with 20 Xe-cores
- Tighter interleaving of loads between individual DPAS calls

```OCL
// v35: Software-pipelined K-loop + SWIZZLE=8 + tighter DPAS/load interleaving
// Architecture: 32 SGs (4x8), each SG: 32x32 tile, WG tile: 128x256
// K-step: 32 per block, 4 blocks per iteration = 128
// Software pipeline: load next iteration's first block during current iteration's last block
// Launch: GWS = (ceil(M/128)*512, ceil(N/256), 1), LWS = (512, 1, 1)
// Subgroup size: 16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable

#define TILE_M 32
#define TILE_N 32
#define WG_M 4
#define WG_N 8
#define SG_SIZE 16
#define WG_TILE_M (WG_M * TILE_M)
#define WG_TILE_N (WG_N * TILE_N)
#define NUM_SG (WG_M * WG_N)
#define WG_SIZE (NUM_SG * SG_SIZE)

#define SWIZZLE 8

// Forward DPAS: interleave b0/b1 per row for better register reuse
#define COMPUTE_FWD(a_arr, a_off, b_arr, b_off) { \
    int8 b0_ = as_int8(*(__private uint8*)&(b_arr)[b_off]); \
    int8 b1_ = as_int8(*(__private uint8*)&(b_arr)[(b_off)+8]); \
    short8 a0_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)]); \
    acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0_, b0_, acc00); \
    acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0_, b1_, acc01); \
    short8 a1_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+8]); \
    acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1_, b0_, acc10); \
    acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1_, b1_, acc11); \
    short8 a2_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+16]); \
    acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2_, b0_, acc20); \
    acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2_, b1_, acc21); \
    short8 a3_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+24]); \
    acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3_, b0_, acc30); \
    acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3_, b1_, acc31); \
}

// Reverse DPAS order (boustrophedon)
#define COMPUTE_REV(a_arr, a_off, b_arr, b_off) { \
    int8 b0_ = as_int8(*(__private uint8*)&(b_arr)[b_off]); \
    int8 b1_ = as_int8(*(__private uint8*)&(b_arr)[(b_off)+8]); \
    short8 a3_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+24]); \
    acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3_, b0_, acc30); \
    acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3_, b1_, acc31); \
    short8 a2_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+16]); \
    acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2_, b0_, acc20); \
    acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2_, b1_, acc21); \
    short8 a1_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+8]); \
    acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1_, b0_, acc10); \
    acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1_, b1_, acc11); \
    short8 a0_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)]); \
    acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0_, b0_, acc00); \
    acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0_, b1_, acc01); \
}

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    int M, int N, int K)
{
    const int sg_id  = get_sub_group_id();
    const int sg_row = sg_id / WG_N;
    const int sg_col = sg_id % WG_N;

    // L3-aware WG swizzle (SWIZZLE=8 for 20 Xe-cores)
    const int grid_m = get_num_groups(0);
    const int grid_n = get_num_groups(1);
    const int gid0 = get_group_id(0);
    const int gid1 = get_group_id(1);
    const int linear = gid1 * grid_m + gid0;
    const int block_size = SWIZZLE * grid_n;
    const int block_id = linear / block_size;
    const int block_off = linear % block_size;
    const int rem_m = grid_m - block_id * SWIZZLE;
    const int cur_swizzle = (rem_m < SWIZZLE) ? rem_m : SWIZZLE;
    const int gn = block_off / cur_swizzle;
    const int gm = block_id * SWIZZLE + (block_off % cur_swizzle);

    const int tile_m = gm * WG_TILE_M + sg_row * TILE_M;
    const int tile_n = gn * WG_TILE_N + sg_col * TILE_N;

    float8 acc00 = 0, acc10 = 0, acc20 = 0, acc30 = 0;
    float8 acc01 = 0, acc11 = 0, acc21 = 0, acc31 = 0;

    const int A_wb = K * 2, A_pb = K * 2;
    const int B_wb = N * 2, B_pb = N * 2;

    // Software-pipelined loop: pre-load first A/B block before loop
    // Then each iteration loads next block while computing current
    
    // Pre-load iteration 0, K-block 0
    ushort a_cur[64];
    intel_sub_group_2d_block_read_16b_32r16x2c(
        (__global void*)A, A_wb, M, A_pb, (int2)(0, tile_m), a_cur);
    uint b0_cur[16];
    intel_sub_group_2d_block_read_transform_16b_16r16x2c(
        (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, 0), b0_cur);
    uint b1_cur[16];
    intel_sub_group_2d_block_read_transform_16b_16r16x2c(
        (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, 16), b1_cur);

    // Main loop with software pipelining
    // K=2048, step=128, 16 iterations, 4 K-blocks each
    for (int k0 = 0; k0 < K; k0 += 128) {
        // ---- K-block 0: k0..k0+31 (data already loaded) ----
        // Start loading A[1]
        ushort a_next[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k0 + 32, tile_m), a_next);

        // Compute k0..k0+15 (forward)
        COMPUTE_FWD(a_cur, 0, b0_cur, 0);

        // Load B[1] first half
        uint b0_next[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 32), b0_next);

        // Compute k0+16..k0+31 (reverse)
        COMPUTE_REV(a_cur, 32, b1_cur, 0);

        // Load B[1] second half
        uint b1_next[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 48), b1_next);

        // ---- K-block 1: k0+32..k0+63 ----
        // Load A[2]
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k0 + 64, tile_m), a_cur);

        // Compute k0+32..k0+47 (forward)
        COMPUTE_FWD(a_next, 0, b0_next, 0);

        // Load B[2] first half
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 64), b0_cur);

        // Compute k0+48..k0+63 (reverse)
        COMPUTE_REV(a_next, 32, b1_next, 0);

        // Load B[2] second half
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 80), b1_cur);

        // ---- K-block 2: k0+64..k0+95 ----
        // Load A[3]
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k0 + 96, tile_m), a_next);

        // Compute k0+64..k0+79 (forward)
        COMPUTE_FWD(a_cur, 0, b0_cur, 0);

        // Load B[3] first half
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 96), b0_next);

        // Compute k0+80..k0+95 (reverse)
        COMPUTE_REV(a_cur, 32, b1_cur, 0);

        // Load B[3] second half
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 112), b1_next);

        // ---- K-block 3: k0+96..k0+127 ----
        // Pre-load next iteration's A[0] and B[0] (if not last iteration)
        int k_next = k0 + 128;
        int has_next = (k_next < K);
        
        // Compute k0+96..k0+111 (forward)
        COMPUTE_FWD(a_next, 0, b0_next, 0);

        // Conditionally pre-load next iteration data
        if (has_next) {
            intel_sub_group_2d_block_read_16b_32r16x2c(
                (__global void*)A, A_wb, M, A_pb, (int2)(k_next, tile_m), a_cur);
            intel_sub_group_2d_block_read_transform_16b_16r16x2c(
                (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k_next), b0_cur);
        }

        // Compute k0+112..k0+127 (reverse)
        COMPUTE_REV(a_next, 32, b1_next, 0);

        if (has_next) {
            intel_sub_group_2d_block_read_transform_16b_16r16x2c(
                (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k_next + 16), b1_cur);
        }
    }

    // Store C tiles - convert f32 accumulators to f16 and write
    #define STORE_BLOCK(acc, row_off, col_off) { \
        ushort c_data[8]; \
        c_data[0] = as_ushort((half)acc.s0); c_data[1] = as_ushort((half)acc.s1); \
        c_data[2] = as_ushort((half)acc.s2); c_data[3] = as_ushort((half)acc.s3); \
        c_data[4] = as_ushort((half)acc.s4); c_data[5] = as_ushort((half)acc.s5); \
        c_data[6] = as_ushort((half)acc.s6); c_data[7] = as_ushort((half)acc.s7); \
        intel_sub_group_2d_block_write_16b_8r16x1c( \
            (__global void*)C, N * 2, M, N * 2, \
            (int2)(tile_n + col_off, tile_m + row_off), c_data); \
    }
    STORE_BLOCK(acc00, 0, 0);  STORE_BLOCK(acc10, 8, 0);
    STORE_BLOCK(acc20, 16, 0); STORE_BLOCK(acc30, 24, 0);
    STORE_BLOCK(acc01, 0, 16); STORE_BLOCK(acc11, 8, 16);
    STORE_BLOCK(acc21, 16, 16); STORE_BLOCK(acc31, 24, 16);
    #undef STORE_BLOCK
}
```

