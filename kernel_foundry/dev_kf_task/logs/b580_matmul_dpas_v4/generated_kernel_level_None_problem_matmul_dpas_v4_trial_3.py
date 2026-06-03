## Analysis

The previous SLM-based kernel produced zeros in the rightmost columns. The issue is the GWS/LWS mapping: with `GWS_x = ceil(N/16)*64` and `LWS_x = 64`, `get_group_id(0)` gives the N-tile index directly. But `n_tile = get_group_id(0)` maps one WG to one 16-col tile. With 4 subgroups all sharing the same 16 columns, only 16 columns are covered per WG. So `GWS_x = ceil(N/16)*64` creates 4x too many WGs along N but each only covers 16 cols — the mapping is correct in that sense, but redundant subgroups write to the same tile. Actually looking more carefully: all 4 subgroups compute different row blocks (sg_id*8) but the same column block. So N coverage = `num_groups_0 * 16`. With `GWS_x = ceil(N/16)*64` and `LWS_x=64`, `num_groups_0 = ceil(N/16)`. That's correct for N coverage. 

Wait, the output shows zeros in rightmost columns. Let me re-examine... The output for 2048x2048 shows the last columns are 0. With `col_base = n_tile * 16` and `n_tile = get_group_id(0)`, and `num_groups = ceil(2048/16) = 128`, `col_base` ranges from 0 to 127*16=2032, covering cols 0..2047. That should be fine.

Actually, looking at the error output more carefully: only the first ~16 columns seem correct, then the rest are zero. This suggests `n_tile` is always 0 or very small. The issue might be that `GWS = (ceil(N/16)*64, ceil(M/32), 1)` — so GWS_x = 128*64 = 8192 with LWS_x = 64, giving 128 groups. That should be correct...

Let me just go with the best working kernel (11.6ms) and try to optimize it further with SLM and multiple subgroups per WG, but more carefully. The best kernel uses 1 subgroup per WG. Let me try 4 subgroups per WG tiling along N (4 x 16 = 64 cols per WG).

## Improved OCL code

```OCL
// Optimized FP16 matmul using Intel XMX DPAS instructions
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Each subgroup: 8x16 output tile via intel_sub_group_f16_f16_matrix_mad_k16
// 4 subgroups per WG, tiling along N: WG covers 8 rows x 64 cols
//
// Launch metadata:
//   Subgroup size: 16
//   LWS: (64, 1, 1) — 4 subgroups per workgroup
//   GWS: (ceil(N/64)*64, ceil(M/8), 1)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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
    const int sg_id = get_sub_group_id();        // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15

    // Each WG covers 8 rows x 64 cols; each subgroup covers 8 rows x 16 cols
    const int col_base = get_group_id(0) * 64 + sg_id * 16;
    const int row_base = get_group_id(1) * 8;

    if (row_base >= M || col_base >= N)
        return;

    float8 acc = 0.0f;

    for (int k = 0; k < K; k += 16) {
        // Load A: 8 rows x 16 K-elements
        short8 a_val;
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + r;
            int a_idx = row_idx * K + k + sg_lid;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[a_idx] : (half)0.0h;
            ((short*)&a_val)[r] = as_short(a_elem);
        }

        // Load B: 16 K-rows x 16 N-cols, packed as int8
        int8 b_val;
        for (int p = 0; p < 8; p++) {
            int k_row0 = k + 2 * p;
            int k_row1 = k + 2 * p + 1;
            int col_idx = col_base + sg_lid;

            half b0 = (k_row0 < K && col_idx < N) ? B[k_row0 * N + col_idx] : (half)0.0h;
            half b1 = (k_row1 < K && col_idx < N) ? B[k_row1 * N + col_idx] : (half)0.0h;

            short2 packed = (short2)(as_short(b0), as_short(b1));
            ((int*)&b_val)[p] = as_int(packed);
        }

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    }

    // Store 8x16 result
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc)[r]);
        }
    }
}
```

