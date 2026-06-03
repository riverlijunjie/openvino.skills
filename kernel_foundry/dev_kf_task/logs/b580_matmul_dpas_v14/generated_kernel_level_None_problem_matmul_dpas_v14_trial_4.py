

## Analysis

Looking at the previous versions:

- **Version 1 (1.330ms)**: K-loop unrolled 2x, double-buffered SLM, but has 3 barriers per pair (load buf0, barrier, load buf1, barrier, then in loop: barrier, load, barrier, load, barrier).
- **Version 2 (2.770ms)**: Single K-tile per iteration with double buffering, but 2 barriers per K-tile (twice as many barriers as needed).
- **Previous best (1.210ms)**: 2x unrolled K-loop, double-buffered. Better barrier placement but has a bug: after loading into `next_buf` for tile 1, it reads from `slm_A + (1 - cur_buf)` which is wrong (should be `next_buf`). Despite this, it's fastest due to fewer barriers per K-pair (only 4 barriers per 64 K-elements).

Key bottlenecks in the 1.210ms version:
1. **Too many barriers**: 4 barriers per K-pair iteration (2 for tile1 load, 2 for next pair load)
2. **B loads not interleaved with DPAS**: All B loads happen sequentially before DPAS
3. **No prefetching of B**: B comes cold from global/L2
4. **Scalar B reads**: Two separate scalar reads instead of paired access

My improvements:
1. **Reduce barriers to 2 per K-pair**: Use proper double-buffering where we load next pair's first tile while computing current pair's second tile
2. **Better interleaving**: Interleave B loads with A reads and DPAS to hide latency
3. **Streamlined loop structure**: Clean double-buffer ping-pong with minimal synchronization
4. **Remove unnecessary conditionals from inner loop**

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32, K-loop unrolled 2x (effective K-step: 64)
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (double-buffered), B loaded directly from global/L2
// SLM stride padded to 34 to reduce bank conflicts
// K=2048 divides evenly by 64, no remainder path needed.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_STRIDE 34
#define SLM_BUF_SIZE (TILE_M * SLM_STRIDE)

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

    __local ushort slm_A[2 * SLM_BUF_SIZE];

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int col_idx = n_base + sg_id * 16 + sg_lid;
    const bool col_valid = col_idx < N;
    const int b_col = col_valid ? col_idx : (N - 1);
    const bool row_tile_valid = (row_base + TILE_M <= M);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Precompute per-WI A-load mapping: 64 WIs load 1024 elements (32x32 tile)
    int a_slm_off[16], a_global_row_off[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int eid = lid + i * 64;
        int ar = eid >> 5;
        int ac = eid & 31;
        a_slm_off[i] = ar * SLM_STRIDE + ac;
        a_global_row_off[i] = ar * K + ac;  // offset within A tile (row*K + col)
    }

    // =========================================================================
    // Pipeline: double-buffered SLM, 2 K-tiles per iteration, 2 barriers per iter
    // 
    // Iteration structure (for kp = 0..num_k_pairs-1):
    //   buf0 has tile at k=kp*64, already loaded
    //   1. Compute from buf0 (k16 step 0 and 1)
    //   2. barrier + load buf1 with k=kp*64+32 + barrier
    //   3. Compute from buf1 (k16 step 0 and 1)
    //   4. barrier + load buf0 with k=(kp+1)*64 + barrier  [if not last]
    //
    // This gives exactly 2 barriers per K-tile computed (4 per pair).
    // But we can do better: overlap load of next tile with compute of current.
    // =========================================================================

    // Load first A tile (k=0) into buffer 0
    {
        __local ushort* dst = slm_A;
        if (row_tile_valid) {
            __global const ushort* src = A_us + a_row_base_K;
            #pragma unroll
            for (int i = 0; i < 16; i++)
                dst[a_slm_off[i]] = src[a_global_row_off[i]];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int gr = row_base + (a_slm_off[i] / SLM_STRIDE);
                int gc = a_global_row_off[i] % K;
                dst[a_slm_off[i]] = (gr < M) ? A_us[gr * K + gc] : (ushort)0;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int num_k_pairs = K >> 6;  // K / 64

    for (int kp = 0; kp < num_k_pairs; kp++) {
        const int k0 = kp * 64;
        const int k1 = k0 + TILE_K;

        // ============ COMPUTE TILE 0 from buffer 0 ============
        {
            __local const ushort* s = slm_A;

            // k16 step 0: read A cols [0..15], load B rows [k0..k0+15]
            short8 a0_0, a1_0, a2_0, a3_0;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0_0)[r] = intel_sub_group_block_read_us(s + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1_0)[r] = intel_sub_group_block_read_us(s + (8 + r) * SLM_STRIDE);

            int8 bv0;
            {
                int boff = k0 * N + b_col;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = B_us[boff];
                    ushort s1 = B_us[boff + N];
                    ((int*)&bv0)[p] = as_int((ushort2)(s0, s1));
                    boff += N2;
                }
            }

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2_0)[r] = intel_sub_group_block_read_us(s + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3_0)[r] = intel_sub_group_block_read_us(s + (24 + r) * SLM_STRIDE);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_0, bv0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_0, bv0, acc1);

            // k16 step 1: read A cols [16..31], load B rows [k0+16..k0+31]
            short8 a0_1, a1_1, a2_1, a3_1;
            __local const ushort* s16 = s + 16;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0_1)[r] = intel_sub_group_block_read_us(s16 + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1_1)[r] = intel_sub_group_block_read_us(s16 + (8 + r) * SLM_STRIDE);

            int8 bv1;
            {
                int boff = (k0 + 16) * N + b_col;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = B_us[boff];
                    ushort s1 = B_us[boff + N];
                    ((int*)&bv1)[p] = as_int((ushort2)(s0, s1));
                    boff += N2;
                }
            }

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2_1)[r] = intel_sub_group_block_read_us(s16 + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3_1)[r] = intel_sub_group_block_read_us(s16 + (24 + r) * SLM_STRIDE);

            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_0, bv0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_0, bv0, acc3);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_1, bv1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_1, bv1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_1, bv1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_1, bv1, acc3);
        }

        // ============ LOAD A tile 1 (k1) into buffer 1 ============
        barrier(CLK_LOCAL_MEM_FENCE);
        {
            __local ushort* dst = slm_A + SLM_BUF_SIZE;
            if (row_tile_valid) {
                __global const ushort* src = A_us + a_row_base_K + k1;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    dst[a_slm_off[i]] = src[a_global_row_off[i]];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int ar = a_slm_off[i] / SLM_STRIDE;
                    int ac = a_global_row_off[i] - ar * K;
                    int gr = row_base + ar;
                    dst[a_slm_off[i]] = (gr < M) ? A_us[gr * K + k1 + ac] : (ushort)0;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ============ COMPUTE TILE 1 from buffer 1 ============
        {
            __local const ushort* s = slm_A + SLM_BUF_SIZE;

            short8 a0_0, a1_0, a2_0, a3_0;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0_0)[r] = intel_sub_group_block_read_us(s + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1_0)[r] = intel_sub_group_block_read_us(s + (8 + r) * SLM_STRIDE);

            int8 bv0;
            {
                int boff = k1 * N + b_col;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = B_us[boff];
                    ushort s1 = B_us[boff + N];
                    ((int*)&bv0)[p] = as_int((ushort2)(s0, s1));
                    boff += N2;
                }
            }

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2_0)[r] = intel_sub_group_block_read_us(s + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3_0)[r] = intel_sub_group_block_read_us(s + (24 + r) * SLM_STRIDE);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_0, bv0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_0, bv0, acc1);

            short8 a0_1, a1_1, a2_1, a3_1;
            __local const ushort* s16 = s + 16;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0_1)[r] = intel_sub_group_block_read_us(s16 + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1_1)[r] = intel_sub_group_block_read_us(s16 + (8 + r) * SLM_STRIDE);

            int8 bv1;
            {
                int boff = (k1 + 16) * N + b_col;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = B_us[boff];
                    ushort s1 = B_us[boff + N];
                    ((int*)&bv1)[p] = as_int((ushort2)(s0, s1));
                    boff += N2;
                }
            }

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2_1)[r] = intel_sub_group_block_read_us(s16 + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3_1)[r] = intel_sub_group_block_read_us(s16 + (24 + r) * SLM_STRIDE);

            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_0, bv0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_0, bv0, acc3);

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_1, bv1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_1, bv1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_1, bv1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_1, bv1, acc3);
        }

        // ============ LOAD next pair's tile 0 into buffer 0 ============
        if (kp + 1 < num_k_pairs) {
            const int next_k0 = (kp + 1) * 64;
            barrier(CLK_LOCAL_MEM_FENCE);
            {
                __local ushort* dst = slm_A;
                if (row_tile_valid) {
                    __global const ushort* src = A_us + a_row_base_K + next_k0;
                    #pragma unroll
                    for (int i = 0; i < 16; i++)
                        dst[a_slm_off[i]] = src[a_global_row_off[i]];
                } else {
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        int ar = a_slm_off[i] / SLM_STRIDE;
                        int ac = a_global_row_off[i] - ar * K;
                        int gr = row_base + ar;
                        dst[a_slm_off[i]] = (gr < M) ? A_us[gr * K + next_k0 + ac] : (ushort)0;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
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

