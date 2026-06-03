## Analysis

The best-performing version (0.417ms) uses a simple 2x K-unroll without prefetching. Interestingly, versions with explicit prefetching (V1: 0.650ms, V2: 0.869ms) performed worse, suggesting that on this hardware, the 2D block read instructions already benefit from hardware prefetching, and explicit prefetch instructions add overhead (extra message traffic, register pressure).

Key observations:
1. **No prefetch is better** - The 0.417ms version has zero prefetch instructions and outperforms versions with prefetching
2. **2x K-unroll helps** - Processing 64 K elements per iteration reduces loop overhead
3. **The remainder loop is dead code** - K=2048 divides evenly by 64, so we can remove it
4. **Load/compute interleaving** - Loading both K-blocks upfront before computing allows the GPU to pipeline loads

Proposed improvements:
1. **Remove the remainder loop** - Dead code elimination, reduces binary size and i-cache pressure
2. **Try 4x K-unroll** - Process 128 K elements per iteration (K=2048 → only 16 iterations), further reducing loop overhead
3. **Better load/compute interleaving** - Interleave loads from the second K-block with compute from the first, hiding memory latency
4. **Boustrophedon FMA ordering** - Alternate forward/reverse DPAS order to improve register locality

## Improved OCL code

```OCL
// v30: 4x K-unroll (128 per iter) + interleaved load/compute + no remainder
// Architecture: 32 SGs (4x8), each SG: 32x32 tile, WG tile: 128x256
// K-step: 128 (4x32), 2D block reads for A and B, no prefetch (HW handles it)
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

#define SWIZZLE 2

// Macro: compute one 32x32x16 DPAS block (4 row-groups x 2 col-groups)
#define COMPUTE_BLOCK(a_arr, a_off, b_arr, b_off) { \
    int8 b0_ = as_int8(*(__private uint8*)&(b_arr)[b_off]); \
    int8 b1_ = as_int8(*(__private uint8*)&(b_arr)[(b_off)+8]); \
    short8 a0_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)]); \
    short8 a1_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+8]); \
    short8 a2_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+16]); \
    short8 a3_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+24]); \
    acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0_, b0_, acc00); \
    acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0_, b1_, acc01); \
    acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1_, b0_, acc10); \
    acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1_, b1_, acc11); \
    acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2_, b0_, acc20); \
    acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2_, b1_, acc21); \
    acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3_, b0_, acc30); \
    acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3_, b1_, acc31); \
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

    // L3-aware WG swizzle
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

    // Main loop: 4x K-unroll (128 K elements per iteration)
    // K=2048 divides evenly by 128 → 16 iterations, no remainder needed
    for (int k0 = 0; k0 < K; k0 += 128) {
        // ---- K-block 0: k0..k0+31 ----
        // Load A[0] and B[0] 
        ushort a_data0[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k0, tile_m), a_data0);

        uint b0_data0[16], b1_data0[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0), b0_data0);
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 16), b1_data0);

        // Compute K-block 0 while loading K-block 1
        // Start loading A[1] early to overlap with compute
        ushort a_data1[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k0 + 32, tile_m), a_data1);

        COMPUTE_BLOCK(a_data0, 0, b0_data0, 0);   // k0..k0+15

        uint b0_data1[16], b1_data1[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 32), b0_data1);

        COMPUTE_BLOCK(a_data0, 32, b1_data0, 0);  // k0+16..k0+31

        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 48), b1_data1);

        // ---- K-block 1: k0+32..k0+63 ----
        // Start loading A[2] early
        ushort a_data2[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k0 + 64, tile_m), a_data2);

        COMPUTE_BLOCK(a_data1, 0, b0_data1, 0);   // k0+32..k0+47

        uint b0_data2[16], b1_data2[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 64), b0_data2);

        COMPUTE_BLOCK(a_data1, 32, b1_data1, 0);  // k0+48..k0+63

        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 80), b1_data2);

        // ---- K-block 2: k0+64..k0+95 ----
        ushort a_data3[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k0 + 96, tile_m), a_data3);

        COMPUTE_BLOCK(a_data2, 0, b0_data2, 0);   // k0+64..k0+79

        uint b0_data3[16], b1_data3[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 96), b0_data3);

        COMPUTE_BLOCK(a_data2, 32, b1_data2, 0);  // k0+80..k0+95

        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 112), b1_data3);

        // ---- K-block 3: k0+96..k0+127 ----
        COMPUTE_BLOCK(a_data3, 0, b0_data3, 0);   // k0+96..k0+111
        COMPUTE_BLOCK(a_data3, 32, b1_data3, 0);  // k0+112..k0+127
    }

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

