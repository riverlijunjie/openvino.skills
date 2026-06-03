

## Analysis

Both Version 1 and Version 2 achieve ~0.274ms with identical architecture (32 SGs, 4x8, 128x256 WG tile). The boustrophedon DPAS ordering in V1 didn't help. Both use 4x K-unroll with interleaved load/compute.

Key observations:
- At 0.274ms for 2048³ f16 GEMM, we're achieving ~62.8 TFLOPS (~65% of 96 TFLOPS peak)
- The architecture is already quite good with 32 SGs per WG
- The main bottleneck is likely memory bandwidth or load latency hiding

**Proposed improvements:**
1. **Software pipelining across loop iterations**: Load the first K-block of iteration i+1 during the last compute of iteration i, creating a true pipeline prologue/epilogue
2. **SWIZZLE=4** from V1 (slightly better L2 reuse for this matrix size)
3. **Tighter interleaving**: Place individual B loads between individual DPAS calls rather than loading both B halves together
4. **Reduce register pressure in last K-block**: Since we don't need to load next iteration's data in the epilogue, the compiler has more scheduling freedom
5. **Try 2x unroll instead of 4x** to reduce register pressure, allowing compiler to schedule better

Let me try a cleaner software-pipelined approach with 2x K-unroll (64 per iter) and true prologue/epilogue pipelining:

## Improved OCL code

```OCL
// v32: Software-pipelined 2x K-unroll with cross-iteration prefetch
// Architecture: 32 SGs (4x8), each SG: 32x32 tile, WG tile: 128x256
// K-step: 64 (2x32), true software pipeline: load[i+1] overlaps compute[i]
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

// Macro: 8 DPAS ops for one 32x32x16 sub-block
// Interleave A-row pairs with B to reduce RAW stalls
#define DPAS8(a_arr, a_off, b_arr, b_off) { \
    int8 _b0 = as_int8(*(__private uint8*)&(b_arr)[b_off]); \
    int8 _b1 = as_int8(*(__private uint8*)&(b_arr)[(b_off)+8]); \
    short8 _a0 = as_short8(*(__private ushort8*)&(a_arr)[(a_off)]); \
    short8 _a1 = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+8]); \
    acc00 = intel_sub_group_f16_f16_matrix_mad_k16(_a0, _b0, acc00); \
    acc01 = intel_sub_group_f16_f16_matrix_mad_k16(_a0, _b1, acc01); \
    acc10 = intel_sub_group_f16_f16_matrix_mad_k16(_a1, _b0, acc10); \
    acc11 = intel_sub_group_f16_f16_matrix_mad_k16(_a1, _b1, acc11); \
    short8 _a2 = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+16]); \
    short8 _a3 = as_short8(*(__private ushort8*)&(a_arr)[(a_off)+24]); \
    acc20 = intel_sub_group_f16_f16_matrix_mad_k16(_a2, _b0, acc20); \
    acc21 = intel_sub_group_f16_f16_matrix_mad_k16(_a2, _b1, acc21); \
    acc30 = intel_sub_group_f16_f16_matrix_mad_k16(_a3, _b0, acc30); \
    acc31 = intel_sub_group_f16_f16_matrix_mad_k16(_a3, _b1, acc31); \
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

    // L3-aware WG swizzle (boustrophedon with SWIZZLE=4)
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
    intel_sub_group_2d_block_read_16b_32r16x2c(
        (__global void*)A, A_wb, M, A_pb, (int2)(0, tile_m), a_cur);

    uint blo_cur[16], bhi_cur[16];
    intel_sub_group_2d_block_read_transform_16b_16r16x2c(
        (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, 0), blo_cur);
    intel_sub_group_2d_block_read_transform_16b_16r16x2c(
        (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, 16), bhi_cur);

    // === MAIN LOOP: software-pipelined, 32 K per step ===
    // K=2048 divides by 32 → 64 iterations, no remainder
    // We process k_step i while loading k_step i+1
    const int num_steps = K / 32;  // = 64

    for (int step = 0; step < num_steps - 1; step++) {
        int k_next = (step + 1) * 32;

        // Load A for next step (overlaps with current compute)
        ushort a_nxt[64];
        intel_sub_group_2d_block_read_16b_32r16x2c(
            (__global void*)A, A_wb, M, A_pb, (int2)(k_next, tile_m), a_nxt);

        // Compute first half of current step (k16 = 0..15)
        DPAS8(a_cur, 0, blo_cur, 0);

        // Load B-lo for next step between DPAS blocks
        uint blo_nxt[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k_next), blo_nxt);

        // Compute second half of current step (k16 = 16..31)
        DPAS8(a_cur, 32, bhi_cur, 0);

        // Load B-hi for next step
        uint bhi_nxt[16];
        intel_sub_group_2d_block_read_transform_16b_16r16x2c(
            (__global void*)B, B_wb, K, B_pb, (int2)(tile_n, k_next + 16), bhi_nxt);

        // Swap buffers (compiler should optimize to register renaming)
        #pragma unroll
        for (int i = 0; i < 64; i++) a_cur[i] = a_nxt[i];
        #pragma unroll
        for (int i = 0; i < 16; i++) blo_cur[i] = blo_nxt[i];
        #pragma unroll
        for (int i = 0; i < 16; i++) bhi_cur[i] = bhi_nxt[i];
    }

    // === EPILOGUE: compute last K-block (no next load needed) ===
    DPAS8(a_cur, 0, blo_cur, 0);
    DPAS8(a_cur, 32, bhi_cur, 0);

    // === Store C tiles ===
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

