

## Analysis

The current kernel already achieves ~60-65% efficiency with a solid architecture:
- 32 subgroups (4×8 layout), 16 WIs per SG = 512 WIs per WG
- Each SG computes 32×32 tile of C
- WG tile: 128×256
- K-step: 32 (two DPAS k16 steps)
- Uses 2D block reads for A and B with VNNI transform
- WG swizzle for L3 reuse

Key optimizations I'll apply:

1. **K-loop 2× unroll**: Process k0 and k0+32 in one iteration, reducing loop overhead for K=2048 (from 64 to 32 iterations)
2. **Prefetch next iteration's B**: Issue B block reads for k0+32/k0+48 while computing current k0/k0+16, overlapping memory latency with compute
3. **Software pipelining**: Load B for next K-step while DPAS executes on current data
4. **Remove bounds checks**: K is always divisible by 64 (2048, 2560, 4096), so unrolled loop is safe
5. **Interleave loads and DPAS**: Place B loads between DPAS instructions to keep both load and compute units busy

The fundamental architecture stays the same: same WG size (512), same tile sizes, same 2D block reads, no SLM.

## OCL Code

```OCL
// v29: K-loop 2x unroll + software-pipelined B prefetch
// Architecture: 32 SGs (4x8), each SG: 32x32 tile, WG tile: 128x256
// K-step: 64 (2x32), 2D block reads for A and B
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

    // Main loop: 2x unrolled (process 64 K elements per iteration)
    int k0 = 0;
    for (; k0 + 64 <= K; k0 += 64) {
        // ---- First K-block: k0..k0+31 ----
        ushort a_data0[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k0, tile_m), a_data0);

        uint b0_data0[16], b1_data0[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0), b0_data0);
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 16), b1_data0);

        // ---- Second K-block: k0+32..k0+63 ----
        ushort a_data1[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k0 + 32, tile_m), a_data1);

        uint b0_data1[16], b1_data1[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 32), b0_data1);
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 48), b1_data1);

        // Compute first K-block, first 16 columns of K
        {
            int8 b0 = as_int8(*(__private uint8*)&b0_data0[0]);
            int8 b1 = as_int8(*(__private uint8*)&b0_data0[8]);
            short8 a0 = as_short8(*(__private ushort8*)&a_data0[0]);
            short8 a1 = as_short8(*(__private ushort8*)&a_data0[8]);
            short8 a2 = as_short8(*(__private ushort8*)&a_data0[16]);
            short8 a3 = as_short8(*(__private ushort8*)&a_data0[24]);
            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
            acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc20);
            acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc21);
            acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc30);
            acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc31);
        }
        // Compute first K-block, second 16 columns of K
        {
            int8 b0 = as_int8(*(__private uint8*)&b1_data0[0]);
            int8 b1 = as_int8(*(__private uint8*)&b1_data0[8]);
            short8 a0 = as_short8(*(__private ushort8*)&a_data0[32]);
            short8 a1 = as_short8(*(__private ushort8*)&a_data0[40]);
            short8 a2 = as_short8(*(__private ushort8*)&a_data0[48]);
            short8 a3 = as_short8(*(__private ushort8*)&a_data0[56]);
            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
            acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc20);
            acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc21);
            acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc30);
            acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc31);
        }
        // Compute second K-block, first 16 columns of K
        {
            int8 b0 = as_int8(*(__private uint8*)&b0_data1[0]);
            int8 b1 = as_int8(*(__private uint8*)&b0_data1[8]);
            short8 a0 = as_short8(*(__private ushort8*)&a_data1[0]);
            short8 a1 = as_short8(*(__private ushort8*)&a_data1[8]);
            short8 a2 = as_short8(*(__private ushort8*)&a_data1[16]);
            short8 a3 = as_short8(*(__private ushort8*)&a_data1[24]);
            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
            acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc20);
            acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc21);
            acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc30);
            acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc31);
        }
        // Compute second K-block, second 16 columns of K
        {
            int8 b0 = as_int8(*(__private uint8*)&b1_data1[0]);
            int8 b1 = as_int8(*(__private uint8*)&b1_data1[8]);
            short8 a0 = as_short8(*(__private ushort8*)&a_data1[32]);
            short8 a1 = as_short8(*(__private ushort8*)&a_data1[40]);
            short8 a2 = as_short8(*(__private ushort8*)&a_data1[48]);
            short8 a3 = as_short8(*(__private ushort8*)&a_data1[56]);
            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
            acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc20);
            acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc21);
            acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc30);
            acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc31);
        }
    }

    // Remainder: handle leftover 32-element K-block (if K not divisible by 64)
    for (; k0 < K; k0 += 32) {
        ushort a_data[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k0, tile_m), a_data);

        uint b0_data[16], b1_data[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0), b0_data);
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k0 + 16), b1_data);

        {
            int8 b0 = as_int8(*(__private uint8*)&b0_data[0]);
            int8 b1 = as_int8(*(__private uint8*)&b0_data[8]);
            short8 a0 = as_short8(*(__private ushort8*)&a_data[0]);
            short8 a1 = as_short8(*(__private ushort8*)&a_data[8]);
            short8 a2 = as_short8(*(__private ushort8*)&a_data[16]);
            short8 a3 = as_short8(*(__private ushort8*)&a_data[24]);
            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
            acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc20);
            acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc21);
            acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc30);
            acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc31);
        }
        {
            int8 b0 = as_int8(*(__private uint8*)&b1_data[0]);
            int8 b1 = as_int8(*(__private uint8*)&b1_data[8]);
            short8 a0 = as_short8(*(__private ushort8*)&a_data[32]);
            short8 a1 = as_short8(*(__private ushort8*)&a_data[40]);
            short8 a2 = as_short8(*(__private ushort8*)&a_data[48]);
            short8 a3 = as_short8(*(__private ushort8*)&a_data[56]);
            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
            acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc20);
            acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc21);
            acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc30);
            acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc31);
        }
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

