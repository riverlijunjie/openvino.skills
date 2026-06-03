

## Analysis

The task is to optimize an FP16 matrix multiplication kernel for Intel Battlemage (Xe2-HPG) using DPAS (XMX) instructions.

**Key considerations:**
- `intel_sub_group_f16_f16_matrix_mad_k16` computes an 8×16 output tile with K=16, requiring `short8` for A (8 rows × 16 K-elements distributed across subgroup) and `int8` for B (16K × 16N packed as pairs of halfs).
- Battlemage has 160 XMX engines across 20 Xe-cores, so we need sufficient parallelism.
- The scaffold already has a correct basic structure. I'll enhance it with:

**Optimizations applied:**
1. **Larger tile per workgroup**: 32 rows × 64 cols (4 subgroups handle 4×1 arrangement of 8×16 tiles → but better: each subgroup handles multiple 8×16 tiles vertically). I'll use 4 subgroups, each computing 2 tiles of 8×16 = 16 rows × 16 cols per subgroup, giving 16×64 per WG.
2. **Double buffering**: Pre-load next K-tile while computing current one.
3. **Loop unrolling**: Unroll the K-loop body.
4. **Register blocking**: Each subgroup computes multiple 8×16 output tiles to increase arithmetic intensity.
5. **Prefetching**: Load next iteration's data before DPAS.

**Structure:**
- WG tile: 32 rows × 64 cols
- 4 subgroups of 16 work-items each (WG size = 64)
- Each subgroup computes 4 tiles of 8×16 (32 rows × 16 cols) — maximizing register reuse of B
- K-loop with K-tile = 16, double-buffered loads

Actually, let me keep it simpler but effective: each subgroup computes 2 vertical 8×16 tiles (16 rows × 16 cols), WG covers 16 rows × 64 cols. This gives 4 `float8` accumulators per subgroup, reusing B across 2 row tiles.

Let me go with: WG = 16 rows × 64 cols, 4 subgroups each handling 16 rows × 16 cols (2 DPAS tiles vertically, reusing B).

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 16 rows x 64 cols
// 4 subgroups x 16 work-items = 64 work-items per WG
// Each subgroup: 2 vertical 8x16 DPAS tiles = 16 rows x 16 cols
// GWS = (ceil(N/64)*64, ceil(M/16))  LWS = (64, 1)
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

    // Each WG covers 16 rows x 64 cols
    const int col_base = get_group_id(0) * 64 + sg_id * 16;
    const int row_base = get_group_id(1) * 16;

    if (row_base >= M || col_base >= N)
        return;

    // Two 8x16 accumulators (16 rows x 16 cols per subgroup)
    float8 acc0 = 0.0f;  // rows [row_base..row_base+7]
    float8 acc1 = 0.0f;  // rows [row_base+8..row_base+15]

    int k = 0;

    // Double-buffering: load first tile
    short8 a_val0, a_val1;
    int8 b_val;

    // Preload first K-tile
    if (k < K) {
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + r;
            int a_idx = row_idx * K + k + sg_lid;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[a_idx] : (half)0.0h;
            ((short*)&a_val0)[r] = as_short(a_elem);
        }
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + 8 + r;
            int a_idx = row_idx * K + k + sg_lid;
            half a_elem = (row_idx < M && (k + sg_lid) < K) ? A[a_idx] : (half)0.0h;
            ((short*)&a_val1)[r] = as_short(a_elem);
        }
        for (int p = 0; p < 8; p++) {
            int k_row0 = k + 2 * p;
            int k_row1 = k + 2 * p + 1;
            int col_idx = col_base + sg_lid;
            half b0 = (k_row0 < K && col_idx < N) ? B[k_row0 * N + col_idx] : (half)0.0h;
            half b1 = (k_row1 < K && col_idx < N) ? B[k_row1 * N + col_idx] : (half)0.0h;
            short2 packed = (short2)(as_short(b0), as_short(b1));
            ((int*)&b_val)[p] = as_int(packed);
        }
    }

    for (k = 0; k < K; k += 16) {
        short8 next_a0, next_a1;
        int8 next_b;
        int next_k = k + 16;

        // Prefetch next K-tile while computing current
        if (next_k < K) {
            for (int r = 0; r < 8; r++) {
                int row_idx = row_base + r;
                int a_idx = row_idx * K + next_k + sg_lid;
                half a_elem = (row_idx < M && (next_k + sg_lid) < K) ? A[a_idx] : (half)0.0h;
                ((short*)&next_a0)[r] = as_short(a_elem);
            }
            for (int r = 0; r < 8; r++) {
                int row_idx = row_base + 8 + r;
                int a_idx = row_idx * K + next_k + sg_lid;
                half a_elem = (row_idx < M && (next_k + sg_lid) < K) ? A[a_idx] : (half)0.0h;
                ((short*)&next_a1)[r] = as_short(a_elem);
            }
            for (int p = 0; p < 8; p++) {
                int k_row0 = next_k + 2 * p;
                int k_row1 = next_k + 2 * p + 1;
                int col_idx = col_base + sg_lid;
                half b0 = (k_row0 < K && col_idx < N) ? B[k_row0 * N + col_idx] : (half)0.0h;
                half b1 = (k_row1 < K && col_idx < N) ? B[k_row1 * N + col_idx] : (half)0.0h;
                short2 packed = (short2)(as_short(b0), as_short(b1));
                ((int*)&next_b)[p] = as_int(packed);
            }
        }

        // DPAS: two 8x16 tiles reusing B
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val1, b_val, acc1);

        // Swap buffers
        a_val0 = next_a0;
        a_val1 = next_a1;
        b_val = next_b;
    }

    // Store 16x16 result
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc0)[r]);
        }
    }
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + 8 + r;
        int col_idx = col_base + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc1)[r]);
        }
    }
}
```

