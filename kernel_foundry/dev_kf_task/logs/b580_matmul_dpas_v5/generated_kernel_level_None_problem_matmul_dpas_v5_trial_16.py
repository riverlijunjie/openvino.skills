

## Analysis

Looking at the previous versions, all achieve ~33.9ms. The key issues identified in the feedback are:

1. **Redundant DPAS lane usage**: Lanes 8-15 load duplicate A rows (clamped), wasting half the subgroup's work per DPAS call.
2. **Scalar B gather from SLM**: 16 individual SLM reads per lane per K-step is expensive.
3. **Tail-K in epilogue**: Repeated global loads per output element.

The DPAS intrinsic `intel_sub_group_f16_f16_matrix_mad_k16` computes an 8×16 output tile: 8 rows (from float8 accumulator) × 16 columns (one per lane), with K=16 reduction. Each lane provides its own `a_vec` (half16), but only lanes 0-7 are used for the A matrix (8 unique rows). The B matrix (half16 per lane) provides 16 columns × K=16.

**Strategy for improved kernel:**
- **Tile size**: 32M × 32N per WG, with 2 subgroups (LWS = {16, 2, 1})
- **Register blocking**: Each subgroup computes 16M × 32N using 4 DPAS calls per K-step (2 row-halves × 2 column-halves), giving 4 float8 accumulators per subgroup
- **Transposed B in SLM**: Store B transposed so `b_vec` can be loaded with `vload16` instead of scalar gather
- **Tail-K inside tiled loop**: Handle tail K with the same SLM buffering, zero-padded
- **Proper DPAS lane mapping**: All 16 lanes load unique A rows; lanes 0-7 feed DPAS, lanes 8-15 just need valid data

```OCL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Launch metadata:
//   LWS = {16, 2, 1}  (32 WIs = 2 subgroups of 16)
//   GWS = {ceil_div(N,32)*16, ceil_div(M,32)*2, 1}
//   reqd_sub_group_size = 16
//
// Tile: 32M x 32N per WG, K in chunks of 16
// Each subgroup: 16 rows x 32 cols = 4 DPAS calls per K-step
// B stored transposed in SLM for vectorized loads
// Tail-K handled inside tiled loop (zero-padded)

#define TILE_M  32
#define TILE_N  32
#define TILE_K  16
#define SG_SIZE 16

__attribute__((reqd_work_group_size(16, 2, 1)))
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lane = get_sub_group_local_id();  // 0..15
    const int sg_id = get_local_id(1);          // 0 or 1
    const int wg_x = get_group_id(0);
    const int wg_y = get_group_id(1);

    const int tile_row = wg_y * TILE_M;
    const int tile_col = wg_x * TILE_N;

    // Subgroup handles 16 rows: sg_id=0 -> rows 0..15, sg_id=1 -> rows 16..31
    const int sg_row0 = tile_row + sg_id * 16;

    // SLM: A [32][16+1], B transposed [32][16+1] (B_t[n][k] = B[k][n])
    __local half A_slm[TILE_M][TILE_K + 1];
    __local half Bt_slm[TILE_N][TILE_K + 1];  // transposed B for vectorized loads

    // 4 accumulators: {rows 0-7, rows 8-15} x {cols 0-15, cols 16-31}
    float8 acc00 = (float8)(0.0f);  // rows 0-7,  cols 0-15
    float8 acc10 = (float8)(0.0f);  // rows 8-15, cols 0-15
    float8 acc01 = (float8)(0.0f);  // rows 0-7,  cols 16-31
    float8 acc11 = (float8)(0.0f);  // rows 8-15, cols 16-31

    const int linear_id = sg_id * 16 + lane;  // 0..31

    // Round K up to include tail in tiled loop
    const int k_end = ((K + TILE_K - 1) / TILE_K) * TILE_K;

    for (int kb = 0; kb < k_end; kb += TILE_K) {
        // === Cooperative load A tile [32 x 16] = 512 halfs, 32 threads => 16 each ===
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            int idx = linear_id + t * 32;
            int ar = idx >> 4;        // idx / 16
            int ak = idx & 15;       // idx % 16
            int gr = tile_row + ar;
            int gk = kb + ak;
            A_slm[ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // === Cooperative load B tile transposed [32 x 16] = 512 halfs ===
        // Bt_slm[n][k] = B[kb+k][tile_col+n]
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            int idx = linear_id + t * 32;
            int bn = idx >> 4;        // col index 0..31
            int bk = idx & 15;       // k index 0..15
            int gk = kb + bk;
            int gc = tile_col + bn;
            Bt_slm[bn][bk] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === DPAS computation ===
        // Load B vectors via vload16 from transposed SLM (contiguous!)
        half16 b_vec0 = vload16(0, &Bt_slm[lane][0]);       // cols 0-15
        half16 b_vec1 = vload16(0, &Bt_slm[lane + 16][0]);  // cols 16-31

        // Rows 0-7: DPAS uses lanes 0-7 for A; lanes 8-15 need valid data
        {
            int rr = (lane < 8) ? lane : lane - 8;  // lanes 8-15 mirror 0-7 (valid data)
            half16 a_vec = vload16(0, &A_slm[sg_id * 16 + rr][0]);
            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec0, acc00);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec1, acc01);
        }

        // Rows 8-15
        {
            int rr = (lane < 8) ? lane + 8 : lane;  // lanes 8-15 mirror 8-15
            half16 a_vec = vload16(0, &A_slm[sg_id * 16 + rr][0]);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec0, acc10);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec1, acc11);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
    // Rows 0-7, cols 0-15
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int gr = sg_row0 + r;
        int gc = tile_col + lane;
        if (gr < M && gc < N)
            C[gr * N + gc] = convert_half(((float*)&acc00)[r]);
    }
    // Rows 0-7, cols 16-31
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int gr = sg_row0 + r;
        int gc = tile_col + lane + 16;
        if (gr < M && gc < N)
            C[gr * N + gc] = convert_half(((float*)&acc01)[r]);
    }
    // Rows 8-15, cols 0-15
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int gr = sg_row0 + 8 + r;
        int gc = tile_col + lane;
        if (gr < M && gc < N)
            C[gr * N + gc] = convert_half(((float*)&acc10)[r]);
    }
    // Rows 8-15, cols 16-31
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int gr = sg_row0 + 8 + r;
        int gc = tile_col + lane + 16;
        if (gr < M && gc < N)
            C[gr * N + gc] = convert_half(((float*)&acc11)[r]);
    }
}
```

Key improvements:
1. **Transposed B in SLM** → `vload16` replaces 16 scalar gathers per DPAS call
2. **4 DPAS calls per K-step** → 16×32 output per subgroup (doubled column coverage)
3. **Tail-K inside tiled loop** → zero-padded, no scalar epilogue
4. **Clean lane mapping** → lanes 8-15 get valid (mirrored) data, no clamping to single row

