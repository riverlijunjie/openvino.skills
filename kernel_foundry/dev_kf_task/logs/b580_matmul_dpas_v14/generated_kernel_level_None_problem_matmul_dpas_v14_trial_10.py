## Analysis

The previous kernel failed correctness tests. The bug is in the B load using `vload2`. The issue is that `vload2(0, B_us + b_off)` reads two **consecutive** ushort values at the same address (i.e., `B[k][col]` and `B[k][col+1]`), but we need `B[k][col]` and `B[k+1][col]` (two different rows, same column, stride N apart). The `vload2` reads contiguous memory, not strided memory.

The working versions (V1 and V2 scoring 1.010ms and 1.030ms) use explicit scalar reads: `B_us[b_off]` and `B_us[b_off + N]`, which correctly reads two different rows.

**Fix**: Revert to the proven V1 approach (best at 1.010ms) with the scalar B reads, and apply minor optimizations:
- Better interleaving of B loads and A SLM reads with DPAS
- Boustrophedon DPAS ordering on second k16 step
- Keep single-buffered SLM (simpler, proven faster than double-buffered V3 at 1.210ms)

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (single buffer), B from global/L2
// SLM stride = 34 (padded to reduce bank conflicts)
// K=2048 divides evenly by 32, no remainder path needed.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
//
// Based on V1 (best 1.010ms). Optimizations:
//   - Deep interleaving: issue B loads between A SLM reads and DPAS
//   - Boustrophedon DPAS on k16 step 1 for register locality
//   - Precomputed A load offsets to reduce loop address math

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

    __local ushort slm_A[SLM_BUF_SIZE];

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int col_idx = n_base + sg_id * 16 + sg_lid;
    const bool col_valid = col_idx < N;
    const int b_col = col_valid ? col_idx : (N - 1);
    const bool row_tile_valid = (row_base + TILE_M <= M);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Precompute per-WI A-load mapping: 64 WIs load 1024 elements (32x32)
    int a_slm_off[16], a_r[16], a_c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int eid = lid + i * 64;
        a_r[i] = eid >> 5;
        a_c[i] = eid & 31;
        a_slm_off[i] = a_r[i] * SLM_STRIDE + a_c[i];
    }

    // Precompute global A row offsets
    int a_glob_off[16];
    if (row_tile_valid) {
        #pragma unroll
        for (int i = 0; i < 16; i++)
            a_glob_off[i] = a_row_base_K + a_r[i] * K + a_c[i];
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int gr = row_base + a_r[i];
            a_glob_off[i] = (gr < M) ? (gr * K + a_c[i]) : -1;
        }
    }

    // B base pointer for this subgroup's column
    __global const ushort* B_col = B_us + b_col;

    for (int k_off = 0; k_off < K; k_off += TILE_K) {
        // ==== LOAD A tile into SLM ====
        if (row_tile_valid) {
            #pragma unroll
            for (int i = 0; i < 16; i++)
                slm_A[a_slm_off[i]] = A_us[a_glob_off[i] + k_off];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++)
                slm_A[a_slm_off[i]] = (a_glob_off[i] >= 0) ? A_us[a_glob_off[i] + k_off] : (ushort)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ==== k16 step 0: A cols [0..15], B rows [k_off..k_off+15] ====
        {
            __local const ushort* s = slm_A;
            __global const ushort* Bp = B_col + k_off * N;

            // Read A rows 0-7, interleaved with B load start
            short8 a0;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(s + r * SLM_STRIDE);

            // Load B rows 0-7 (paired into int8)
            int8 bv0;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
            }

            // Read A rows 8-15
            short8 a1;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(s + (8 + r) * SLM_STRIDE);

            // Load B rows 8-15
            #pragma unroll
            for (int p = 4; p < 8; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
            }

            // DPAS for a0, then read a2 while DPAS runs
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv0, acc0);

            short8 a2;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(s + (16 + r) * SLM_STRIDE);

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv0, acc1);

            short8 a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(s + (24 + r) * SLM_STRIDE);

            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv0, acc3);
        }

        // ==== k16 step 1: A cols [16..31], B rows [k_off+16..k_off+31] ====
        {
            __local const ushort* s16 = slm_A + 16;
            __global const ushort* Bp = B_col + (k_off + 16) * N;

            short8 a0b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0b)[r] = intel_sub_group_block_read_us(s16 + r * SLM_STRIDE);

            int8 bv1;
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv1)[p] = as_int((ushort2)(b0, b1));
            }

            short8 a1b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1b)[r] = intel_sub_group_block_read_us(s16 + (8 + r) * SLM_STRIDE);

            #pragma unroll
            for (int p = 4; p < 8; p++) {
                ushort b0 = Bp[p * N2];
                ushort b1 = Bp[p * N2 + N];
                ((int*)&bv1)[p] = as_int((ushort2)(b0, b1));
            }

            // Boustrophedon: DPAS a0b first, then read a2b/a3b interleaved
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0b, bv1, acc0);

            short8 a2b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2b)[r] = intel_sub_group_block_read_us(s16 + (16 + r) * SLM_STRIDE);

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1b, bv1, acc1);

            short8 a3b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3b)[r] = intel_sub_group_block_read_us(s16 + (24 + r) * SLM_STRIDE);

            // Boustrophedon: reverse order for last two
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3b, bv1, acc3);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2b, bv1, acc2);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
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

