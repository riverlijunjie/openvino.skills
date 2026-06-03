

## Analysis

Looking at the previous versions:

1. **Version 1 (1.91ms)**: Uses `intel_sub_group_block_read_us` for SLM reads, `ushort` SLM type, no loop unrolling. The block reads from SLM with stride 32 should be efficient.

2. **Version 2 (2.97ms)**: 2x loop unroll but with massive code duplication and extra remainder handling - the overhead hurt performance.

3. **Last version (1.23ms)**: Best so far. Uses `SLM_A_STRIDE = 34` (padding to avoid bank conflicts), scalar SLM reads with `as_short`, paired B reads. Has 2x unroll but also remainder path.

Key observations:
- The padding (`TILE_K + 2 = 34`) in the last version helped significantly vs stride-32 in V1
- The 2x unroll in V2 hurt because of code bloat; in the last version it was slightly better
- Scalar SLM reads (last version) outperformed block reads (V1) - possibly because stride-34 breaks alignment for block reads

**Improvements to try:**
1. Use `SLM_A_STRIDE = 32` (no padding) with `intel_sub_group_block_read_us` - this gives aligned vectorized SLM reads
2. Remove the 2x K-loop unroll (simplify, reduce code size/register pressure)
3. Remove remainder path entirely (K=2048 is always even number of tiles)
4. Keep the simple single-iteration loop structure from V1 but with cleaner code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (2KB), B loaded directly from global/L2
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
// K must be divisible by 32 (K=2048, no remainder)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_A_STRIDE 32

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
    const int col_base = n_base + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    // SLM for A tile: 32 rows x 32 cols = 2KB
    __local ushort slm_A[TILE_M * SLM_A_STRIDE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const int num_k_tiles = K >> 5;  // K / 32

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    // Main K-loop: single iteration, no unrolling of outer loop
    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k = kt * TILE_K;

        // Cooperative load A into SLM: 1024 elems, 64 WIs => 16 each
        if (row_tile_valid) {
            __global const ushort* A_tile = A_us + row_base * K + k;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                slm_A[r * SLM_A_STRIDE + c] = A_tile[r * K + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                int gr = row_base + r;
                slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? A_us[gr * K + k + c] : (ushort)0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Two k16 DPAS steps per k32 tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load A from SLM using vectorized block reads
            // intel_sub_group_block_read_us reads 16 consecutive ushorts (1 per lane)
            short8 a0, a1, a2, a3;

            __local const ushort* slm_base = slm_A + kk;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm_base + r * SLM_A_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm_base + (8 + r) * SLM_A_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm_base + (16 + r) * SLM_A_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm_base + (24 + r) * SLM_A_STRIDE);

            // Load B from global — merge paired scalar reads into packed int
            int gk_base = k + kk;
            int8 b_val;

            if (col_valid) {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int k_row0 = gk_base + 2 * p;
                    ushort s0 = B_us[k_row0 * N + col_idx];
                    ushort s1 = B_us[(k_row0 + 1) * N + col_idx];
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                }
            } else {
                b_val = (int8)(0);
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    if (col_valid) {
        __global half* C_col = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

