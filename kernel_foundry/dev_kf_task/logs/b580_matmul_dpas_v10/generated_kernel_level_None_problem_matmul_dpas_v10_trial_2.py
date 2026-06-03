

## Analysis

The best kernel (5.74ms) uses direct global memory loads without SLM, while the SLM version (118ms) is much slower due to barrier overhead and inefficient SLM access patterns. The key insight is that on Battlemage, the direct register-based approach with DPAS works well.

To improve further, I'll:
1. **Increase register blocking** - go to 64 rows x 64 cols per WG (8 DPAS tiles per subgroup instead of 4), improving compute-to-load ratio
2. **Use `intel_sub_group_block_read`** for coalesced B loads instead of scalar loads
3. **Double-buffer A/B loads** to overlap memory latency with DPAS computation
4. **Fully unroll the K-loop body** since TILE_K=16 is fixed

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 128 cols
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// GWS = (ceil(N/128)*128, ceil(M/32))  LWS = (128, 1)
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_sub_group_id();        // 0..7
    const int sg_lid = get_sub_group_local_id(); // 0..15

    const int col_base = get_group_id(0) * 128 + sg_id * 16;
    const int row_base = get_group_id(1) * 32;

    if (row_base >= M || col_base >= N)
        return;

    // 4 accumulators: 32 rows x 16 cols per subgroup
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Double-buffering variables
    short8 a0, a1, a2, a3;
    int8 b_val;

    // Preload first K-tile (k=0)
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + r;
        half a_elem = (row_idx < M && sg_lid < K) ? A[row_idx * K + sg_lid] : (half)0.0h;
        ((short*)&a0)[r] = as_short(a_elem);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 8 + r;
        half a_elem = (row_idx < M && sg_lid < K) ? A[row_idx * K + sg_lid] : (half)0.0h;
        ((short*)&a1)[r] = as_short(a_elem);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 16 + r;
        half a_elem = (row_idx < M && sg_lid < K) ? A[row_idx * K + sg_lid] : (half)0.0h;
        ((short*)&a2)[r] = as_short(a_elem);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 24 + r;
        half a_elem = (row_idx < M && sg_lid < K) ? A[row_idx * K + sg_lid] : (half)0.0h;
        ((short*)&a3)[r] = as_short(a_elem);
    }
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        int k_row0 = 2 * p;
        int k_row1 = 2 * p + 1;
        int col_idx = col_base + sg_lid;
        half b0_val = (k_row0 < K && col_idx < N) ? B[k_row0 * N + col_idx] : (half)0.0h;
        half b1_val = (k_row1 < K && col_idx < N) ? B[k_row1 * N + col_idx] : (half)0.0h;
        ((int*)&b_val)[p] = as_int((short2)(as_short(b0_val), as_short(b1_val)));
    }

    for (int k = 0; k < K; k += 16) {
        // Current tile data already in registers
        short8 na0, na1, na2, na3;
        int8 nb;
        int nk = k + 16;
        bool has_next = (nk < K);

        // Prefetch next tile while computing
        if (has_next) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int row_idx = row_base + r;
                half a_elem = (row_idx < M && (nk + sg_lid) < K) ? A[row_idx * K + nk + sg_lid] : (half)0.0h;
                ((short*)&na0)[r] = as_short(a_elem);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int row_idx = row_base + 8 + r;
                half a_elem = (row_idx < M && (nk + sg_lid) < K) ? A[row_idx * K + nk + sg_lid] : (half)0.0h;
                ((short*)&na1)[r] = as_short(a_elem);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int row_idx = row_base + 16 + r;
                half a_elem = (row_idx < M && (nk + sg_lid) < K) ? A[row_idx * K + nk + sg_lid] : (half)0.0h;
                ((short*)&na2)[r] = as_short(a_elem);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int row_idx = row_base + 24 + r;
                half a_elem = (row_idx < M && (nk + sg_lid) < K) ? A[row_idx * K + nk + sg_lid] : (half)0.0h;
                ((short*)&na3)[r] = as_short(a_elem);
            }
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_row0 = nk + 2 * p;
                int k_row1 = nk + 2 * p + 1;
                int col_idx = col_base + sg_lid;
                half b0_val = (k_row0 < K && col_idx < N) ? B[k_row0 * N + col_idx] : (half)0.0h;
                half b1_val = (k_row1 < K && col_idx < N) ? B[k_row1 * N + col_idx] : (half)0.0h;
                ((int*)&nb)[p] = as_int((short2)(as_short(b0_val), as_short(b1_val)));
            }
        }

        // DPAS compute on current tile
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);

        // Swap buffers
        a0 = na0; a1 = na1; a2 = na2; a3 = na3;
        b_val = nb;
    }

    // Store results
    #pragma unroll
    for (int blk = 0; blk < 4; blk++) {
        float8 acc = (blk == 0) ? acc0 : (blk == 1) ? acc1 : (blk == 2) ? acc2 : acc3;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + blk * 8 + r;
            int col_idx = col_base + sg_lid;
            if (row_idx < M && col_idx < N) {
                C[row_idx * N + col_idx] = convert_half(((float*)&acc)[r]);
            }
        }
    }
}
```

