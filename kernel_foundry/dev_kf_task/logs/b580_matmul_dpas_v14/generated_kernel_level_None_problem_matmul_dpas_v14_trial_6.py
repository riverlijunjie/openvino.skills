

## Analysis

Looking at the results:
- **V1 (single buffer, simple loop)**: 1.050ms - best so far
- **V2 (double buffer, 2x unroll)**: 1.140ms - worse due to complexity overhead
- **V3 (double buffer, complex)**: 1.210ms - even worse, too much overhead

The key insight is that **V1's simplicity wins**. The double-buffering adds SLM usage and complexity without benefit because A is small (32×32×2B = 2KB) and loads quickly from global. The extra barriers and code bloat hurt more than they help.

**Strategy for improvement over V1:**
1. Keep single-buffer SLM architecture (proven best)
2. Better interleave B loads with DPAS compute (hide B latency)
3. Use `vload2` to merge paired B scalar reads into single operations
4. Remove unnecessary variables and simplify address computation
5. Keep boustrophedon DPAS ordering
6. Prefetch next B tile data earlier

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (single buffer), B from global/L2
// SLM stride padded to 34 to reduce bank conflicts
// K=2048 divides evenly by 32, no remainder path needed.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
//
// Optimizations over V1:
//   - Tighter B load using vload2 to merge paired scalar reads
//   - Better interleaving: start B load before all A reads finish
//   - Boustrophedon DPAS for register locality
//   - Simplified address math, fewer intermediate variables

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

    // Precompute per-WI A-load mapping
    int a_slm_off[16], a_r[16], a_c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int eid = lid + i * 64;
        a_r[i] = eid >> 5;
        a_c[i] = eid & 31;
        a_slm_off[i] = a_r[i] * SLM_STRIDE + a_c[i];
    }

    for (int k_off = 0; k_off < K; k_off += TILE_K) {
        // ==== LOAD A tile into SLM ====
        if (row_tile_valid) {
            __global const ushort* src = A_us + a_row_base_K + k_off;
            #pragma unroll
            for (int i = 0; i < 16; i++)
                slm_A[a_slm_off[i]] = src[a_r[i] * K + a_c[i]];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int gr = row_base + a_r[i];
                slm_A[a_slm_off[i]] = (gr < M) ? A_us[gr * K + k_off + a_c[i]] : (ushort)0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ==== k16 step 0: A cols [0..15], B rows [k_off..k_off+15] ====
        {
            // Start B load early (overlapped with A SLM reads)
            int8 bv0;
            int boff = k_off * N + b_col;
            // Use paired reads: load 2 consecutive B rows at once via explicit pair
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort b0 = B_us[boff];
                ushort b1 = B_us[boff + N];
                ((int*)&bv0)[p] = as_int((ushort2)(b0, b1));
                boff += (N << 1);
            }

            // Read all A from SLM for rows 0-31, cols 0-15
            short8 a0, a1, a2, a3;
            __local const ushort* s = slm_A;
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

            // DPAS forward
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv0, acc3);
        }

        // ==== k16 step 1: A cols [16..31], B rows [k_off+16..k_off+31] ====
        {
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
            __local const ushort* s16 = slm_A + 16;
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

            // DPAS reverse (boustrophedon)
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3b, bv1, acc3);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2b, bv1, acc2);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1b, bv1, acc1);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0b, bv1, acc0);
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

