

## Analysis

This is an FP16 matrix multiplication targeting Intel Battlemage (Xe2-HPG) with XMX engines. The key optimization is using `intel_sub_group_f16_f16_matrix_mad_k16` DPAS instructions.

**DPAS instruction details for Xe2-HPG:**
- `intel_sub_group_f16_f16_matrix_mad_k16(half16 a, int8 b, float8 acc)` computes an 8×16 × 16×16 → 8×16 tile per subgroup
- Each subgroup has 16 work-items
- A tile: 8 rows × 16 cols (K), each work-item holds `half16` (one row's K=16 elements, but distributed)
- B tile: 16 rows × 16 cols, packed as `int8` per work-item
- Result: 8 rows × 16 cols of float, each work-item holds `float8`

**Tiling strategy:**
- Tile size per subgroup: 8×16 (M_TILE=8, N_TILE=16)
- K is processed in chunks of K_TILE=16
- Each work-group can contain multiple subgroups for better occupancy
- I'll use 4 subgroups per work-group (processing 4 N-tiles = 64 columns, or 4 M-tiles)

**Memory access:**
- A: each subgroup loads 8×16 half values per K iteration
- B: each subgroup loads 16×16 half values per K iteration (as int8 per WI)
- Use SLM to share A tile across subgroups processing different N tiles

**Launch configuration:**
- WG: 16 WIs/SG × num_subgroups
- GWS/LWS chosen to cover M/8 × N/16 tiles

## OCL Code

```OCL
// Optimized FP16 matmul using Intel XMX DPAS instructions
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Tile: 8 rows x 16 cols per subgroup via intel_sub_group_f16_f16_matrix_mad_k16
// K processed in chunks of 16
//
// Launch metadata:
//   Subgroup size: 16
//   LWS: (16, 1, 1) — 1 subgroup per workgroup (simple mapping)
//   GWS: (ceil(N/16)*16, ceil(M/8), 1)
//   Each subgroup computes one 8x16 output tile
//
// For better occupancy, can use LWS=(16*SG_COUNT, 1, 1) and map multiple N-tiles per WG.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Each subgroup handles an 8x16 output tile
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id(); // 0..15

    // Tile coordinates
    const int n_tile = (get_group_id(0) * get_num_sub_groups() + sg_id);
    const int m_tile = get_group_id(1);

    const int col_base = n_tile * 16;
    const int row_base = m_tile * 8;

    if (row_base >= M || col_base >= N)
        return;

    // Accumulator: 8 floats per work-item = 8 rows x 16 cols
    float8 acc = 0.0f;

    // Loop over K in steps of 16
    for (int k = 0; k < K; k += 16) {
        // Load A tile: 8 rows x 16 cols
        // Each WI loads one column across 8 rows -> half8
        // But for DPAS, A is distributed as: each WI holds elements for the systolic feed
        // For intel_sub_group_f16_f16_matrix_mad_k16:
        //   a: short8 per WI (8 rows, each row has k16 distributed across 16 WIs)
        //   b: int8 per WI (16x16 tile, packed)

        // Load A: 8 rows x 16 K-elements
        // Sub-group block read: each WI gets one K-element per row
        // WI sg_lid reads column sg_lid from each of 8 rows
        short8 a_val;
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + r;
            int a_idx = row_idx * K + k + sg_lid;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[a_idx] : (half)0.0h;
            ((short*)&a_val)[r] = as_short(a_elem);
        }

        // Load B: 16 K-rows x 16 N-cols
        // For DPAS b input: int8 per WI
        // B is 16x16 tile. Each WI (sg_lid = col within tile) reads 16 elements (K rows)
        // Packed as pairs: int = two halfs, so int8 = 16 halfs
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

        // DPAS: 8x16 = 8x16 * 16x16
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    }

    // Store result: each WI writes column sg_lid for 8 rows
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc)[r]);
        }
    }
}
```

**Key design decisions:**

1. **DPAS instruction**: Uses `intel_sub_group_f16_f16_matrix_mad_k16` which computes 8×16 output per subgroup with K=16 reduction, matching XMX engine capabilities on Battlemage.

2. **Tiling**: 8×16 output tile per subgroup, K iterated in steps of 16. This maps directly to the DPAS hardware tile size.

3. **Data packing**: B matrix elements are packed as pairs of FP16 into int32 (short2→int) to match the DPAS `int8` input format. A matrix is loaded as short8 (one element per row per work-item).

4. **Vectorized access**: Each work-item loads 8 elements for A and 16 elements for B per K-step, using vector types (short8, int8) for register efficiency.

5. **Subgroup size 16**: Required for the DPAS instruction on Xe2-HPG architecture.

6. **Boundary handling**: Explicit bounds checks with zero-padding for tiles that extend beyond M, N, or K boundaries.

