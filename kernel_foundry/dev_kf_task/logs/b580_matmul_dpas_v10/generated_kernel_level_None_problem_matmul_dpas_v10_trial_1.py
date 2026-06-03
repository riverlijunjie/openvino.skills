

## Analysis

The best performing version (7ms) uses direct global memory loads with double-buffering, avoiding SLM entirely. The SLM-based versions (92ms, 113ms) are much slower, likely due to SLM load overhead and barriers.

Key observations:
1. **No SLM version wins**: Direct global loads with subgroup reads are faster than SLM-based approaches on this hardware.
2. **Double-buffering helps**: Overlapping next tile loads with current DPAS computation hides latency.
3. **Tile size matters**: The 16x64 tile (2 DPAS blocks per subgroup) was faster than 32x64 (4 blocks), likely due to better occupancy.

To improve further:
- **Increase register blocking**: Use 32x64 tile with 4 subgroups but load directly from global memory (no SLM), combining the larger tile from V1/V2 with the no-SLM approach from the best version.
- **Reduce boundary checks**: For tiles fully within bounds, skip conditionals.
- **Use block reads via subgroup**: `intel_sub_group_block_read` for coalesced loads.
- **Unroll more aggressively**.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 64 cols
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
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

    const int col_base = get_group_id(0) * 64 + sg_id * 16;
    const int row_base = get_group_id(1) * 32;

    if (row_base >= M || col_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    for (int k = 0; k < K; k += 16) {
        // Load A: 4 blocks of 8 rows x 16 K-elements (each WI loads one K via subgroup)
        short8 a0, a1, a2, a3;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + r;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[row_idx * K + k + sg_lid] : (half)0.0h;
            ((short*)&a0)[r] = as_short(a_elem);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + 8 + r;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[row_idx * K + k + sg_lid] : (half)0.0h;
            ((short*)&a1)[r] = as_short(a_elem);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + 16 + r;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[row_idx * K + k + sg_lid] : (half)0.0h;
            ((short*)&a2)[r] = as_short(a_elem);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + 24 + r;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[row_idx * K + k + sg_lid] : (half)0.0h;
            ((short*)&a3)[r] = as_short(a_elem);
        }

        // Load B: 16 K-rows x 16 cols, packed as int8
        int8 b_val;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int k_row0 = k + 2 * p;
            int k_row1 = k + 2 * p + 1;
            int col_idx = col_base + sg_lid;
            half b0 = (k_row0 < K && col_idx < N) ? B[k_row0 * N + col_idx] : (half)0.0h;
            half b1 = (k_row1 < K && col_idx < N) ? B[k_row1 * N + col_idx] : (half)0.0h;
            ((int*)&b_val)[p] = as_int((short2)(as_short(b0), as_short(b1)));
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
    }

    // Store results
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc0)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 8 + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc1)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 16 + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc2)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 24 + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N)
            C[row_idx * N + col_idx] = convert_half(((float*)&acc3)[r]);
    }
}
```

