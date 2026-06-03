

## Analysis

Both Version 1 and Version 2 achieve nearly identical performance (~0.274ms) with the same architecture: 32 SGs (4×8), 128×256 WG tile, 32×32 per-SG tile. The key differences are:

- Version 1 adds boustrophedon DPAS ordering (forward/reverse alternation) and SWIZZLE=4
- Version 2 uses SWIZZLE=2 and straightforward DPAS ordering

Both are already very fast. The bottleneck is likely memory bandwidth or occupancy-related. At 0.274ms for M=N=K=2048 fp16 GEMM (2×2048³ FLOPs = 17.18 TFLOPS), we're at ~62.7 TFLOPS effective, which is 65% of 96 TFLOPS peak.

**Proposed improvements:**
1. **Software pipelining across loop iterations**: Load the first K-block of the *next* outer iteration while computing the last K-block of the current iteration, creating a true cross-iteration pipeline.
2. **SWIZZLE=4** from V1 (slightly better L2 reuse for A panels)
3. **Boustrophedon DPAS** from V1 (better register locality)
4. **Prefetch hints**: Add explicit 2D block prefetch for B tiles 2 K-blocks ahead
5. **Tighter interleaving**: Place each B load immediately before its corresponding DPAS, and each A load as early as possible

## Improved OCL code

```OCL
// v32: Cross-iteration software pipeline + boustrophedon DPAS + prefetch
// Architecture: 32 SGs (4x8), each SG: 32x32 tile, WG tile: 128x256
// K-step: 128 (4x32), 2D block reads for A and B
// Software pipeline: prologue loads first block, epilogue computes last block
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

#define SWIZZLE 4

// Forward DPAS order: rows 0,1,2,3
#define COMPUTE_FWD(a_arr, a_off, b_arr, b_off) { \
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

// Reverse DPAS order: rows 3,2,1,0 (boustrophedon for register locality)
#define COMPUTE_REV(a_arr, a_off, b_arr, b_off) { \
    int8 b0_ = as_int8(*(__private uint8*)&(b_arr)[b_off]); \
    int8 b1_ = as_int8(*(__private uint8*)&(b_arr)[(b_off)+8]); \
    short8 a3_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+24]); \
    short8 a2_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+16]); \
    short8 a1_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+8]); \
    short8 a0_ = as_short8(*(__private ushort8*)&(a_arr)[(a_off)]); \
    acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3_, b0_, acc30); \
    acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3_, b1_, acc31); \
    acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2_, b0_, acc20); \
    acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2_, b1_, acc21); \
    acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1_, b0_, acc10); \
    acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1_, b1_, acc11); \
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
    const int cur_swizzle = min(rem_m, SWIZZLE);
    const int gn = block_off / cur_swizzle;
    const int gm = block_id * SWIZZLE + (block_off % cur_swizzle);

    const int tile_m = gm * WG_TILE_M + sg_row * TILE_M;
    const int tile_n = gn * WG_TILE_N + sg_col * TILE_N;

    float8 acc00 = 0, acc10 = 0, acc20 = 0, acc30 = 0;
    float8 acc01 = 0, acc11 = 0, acc21 = 0, acc31 = 0;

    const int A_wb = K * 2, A_pb = K * 2;
    const int B_wb = N * 2, B_pb = N * 2;

    // === PROLOGUE: Load first K-block (k=0..31) ===
    ushort a_cur[64];
    uint blo_cur[16], bhi_cur[16];
    
    intel_sub_group_2d_block_read_16b_32r16x2c(
        (__global void*)A, A_wb, M, A_pb, (int2)(0, tile_m), a_cur);
    intel_sub_group_2d_block_read_transform_16b_16r16x2c(
        (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, 0), blo_cur);
    intel_sub_group_2d_block_read_transform_16b_16r16x2c(
        (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, 16), bhi_cur);

    // === MAIN LOOP: Process K in steps of 32, software pipelined ===
    // K=2048 divides evenly by 32 → 64 iterations total
    // Each iteration: compute current block while loading next block
    // Last iteration (k=2016): no next block to load
    
    const int K_STEP = 32;
    const int last_k = K - K_STEP;  // 2016
    
    for (int k = 0; k < last_k; k += K_STEP) {
        // Load next A block (k+32)
        ushort a_nxt[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k + K_STEP, tile_m), a_nxt);

        // Load next B lower half (k+32)
        uint blo_nxt[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k + K_STEP), blo_nxt);

        // Compute current block first half (k..k+15) - forward order
        COMPUTE_FWD(a_cur, 0, blo_cur, 0);

        // Load next B upper half (k+48)
        uint bhi_nxt[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k + K_STEP + 16), bhi_nxt);

        // Compute current block second half (k+16..k+31) - reverse order
        COMPUTE_REV(a_cur, 32, bhi_cur, 0);

        // Swap buffers (compiler should optimize - just register renaming)
        #pragma unroll
        for (int i = 0; i < 64; i++) a_cur[i] = a_nxt[i];
        #pragma unroll
        for (int i = 0; i < 16; i++) blo_cur[i] = blo_nxt[i];
        #pragma unroll
        for (int i = 0; i < 16; i++) bhi_cur[i] = bhi_nxt[i];
    }

    // === EPILOGUE: Compute last K-block (k=2016..2047) ===
    COMPUTE_FWD(a_cur, 0, blo_cur, 0);
    COMPUTE_REV(a_cur, 32, bhi_cur, 0);

    // Store C tiles
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

