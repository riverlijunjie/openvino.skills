

## Analysis

The previous versions (v30, v31, v34) all achieve 0.274ms, which is already very good. They use:
- 32 subgroups (4×8), each computing 32×32 tile → WG tile 128×256
- 4× K-unroll (128 per iteration), 2D block reads
- Interleaved load/compute, boustrophedon DPAS ordering
- SWIZZLE for L2 reuse

The fact that v30 (no boustrophedon), v31 (boustrophedon + SWIZZLE=4), and v34 (combined) all get the same 0.274ms suggests we're hitting a plateau with this configuration. Let me try a different approach:

**Key observations:**
1. WG size = 512 WIs is very large. With 20 Xe-cores, we only get ~4 WGs for 2048×2048 (ceil(2048/128) * ceil(2048/256) = 16×8 = 128 WGs). Good occupancy.
2. Each SG computes 32×32 = 1024 output elements, processing K=2048 → 32×32×2048 = 2M FP16 MADs per SG.
3. Let me try a different WG configuration: 16 SGs (4×4) with 32×64 per SG tile → WG tile 128×256 stays same but fewer SGs, potentially better register usage.

Actually, let me try a fundamentally different approach: **larger per-SG tile** to increase compute intensity. Instead of 32×32 per SG, try 32×64 (more N columns per SG), reducing the number of SGs needed.

Another idea: try **8 SGs (2×4)** with **64×64 per SG** tile → WG tile 128×256. This reduces SG count and may improve scheduling.

Let me try a cleaner approach: **16 SGs (4×4), 32×64 per SG tile**, WG tile = 128×256. Each SG handles more N work, fewer SGs per WG, potentially better occupancy and less contention.

Actually, looking more carefully, let me try **prefetching the next iteration's data** more aggressively, and also try a slightly different tile shape with **cross-iteration pipelining** where we load iteration i+1's data during iteration i's compute.

```OCL
// v35: Cross-iteration software pipelining + 16 SGs (2x8) each 64x32 tile
// WG tile: 128x256, K-step: 64 (2x32), fully pipelined loads
// Each SG: 64x32 (8 DPAS row groups x 2 col groups)
// Launch: GWS = (ceil(M/128)*256, ceil(N/256), 1), LWS = (256, 1, 1)
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

#define SWIZZLE 4

// Forward DPAS order with interleaved A rows and B columns
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

    // L3-aware WG swizzle (SWIZZLE=4)
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

    // Software pipelined main loop with cross-iteration prefetch
    // Prologue: load first K-block's data
    ushort a_cur[64];
    intel_sub_group_2d_block_read_16b_32r16x2c(
        (__global void*)A, A_wb, M, A_pb, (int2)(0, tile_m), a_cur);

    uint b0_cur[16], b1_cur[16];
    intel_sub_group_2d_block_read_transform_16b_16r16x2c(
        (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, 0), b0_cur);
    intel_sub_group_2d_block_read_transform_16b_16r16x2c(
        (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, 16), b1_cur);

    // Main loop: process K in steps of 32, with next iteration's loads overlapped
    // K=2048 / 32 = 64 iterations
    for (int k0 = 0; k0 < K - 32; k0 += 32) {
        // Load next iteration's A
        ushort a_next[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k0 + 32, tile_m), a_next);

        // Compute first k16 block (forward) while loading next B
        COMPUTE_FWD(a_cur, 0, b0_cur, 0);

        // Load next iteration's B first half
        uint b0_next[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 32), b0_next);

        // Compute second k16 block (reverse - boustrophedon)
        COMPUTE_REV(a_cur, 32, b1_cur, 0);

        // Load next iteration's B second half
        uint b1_next[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 48), b1_next);

        // Swap buffers (compiler should optimize to register renaming)
        #pragma unroll
        for (int i = 0; i < 64; i++) a_cur[i] = a_next[i];
        #pragma unroll
        for (int i = 0; i < 16; i++) { b0_cur[i] = b0_next[i]; b1_cur[i] = b1_next[i]; }
    }

    // Epilogue: last K-block (k0 = K-32)
    COMPUTE_FWD(a_cur, 0, b0_cur, 0);
    COMPUTE_REV(a_cur, 32, b1_cur, 0);

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

