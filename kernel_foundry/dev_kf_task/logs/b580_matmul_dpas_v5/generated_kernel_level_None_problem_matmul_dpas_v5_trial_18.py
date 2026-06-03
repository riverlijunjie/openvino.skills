

## Analysis

Looking at the previous versions, all achieved 33.9ms. The key issues:

1. **Version 1 (best)**: Uses 16x16 WG but each thread computes only 1 output element - very low arithmetic intensity per thread.
2. **Last version**: Better DPAS usage (8 rows per call) but wastes cycles on scalar SLM loads and has expensive tail handling.

**Key improvements needed:**
- **Register blocking**: Each work-item should compute multiple output elements (e.g., 8 rows × 1 col via DPAS)
- **Vectorized SLM loads**: Use vector loads instead of scalar loops
- **Transpose B in SLM** so `b_vec` can be loaded with `vload16` instead of scalar gather
- **Fold tail K into main loop** with zero-padding
- **Larger tiles per WG**: Process more output per work-group to improve compute/memory ratio

**Strategy**: Use a single subgroup (16 lanes) per WG computing a 16×16 tile. Each DPAS call produces 8 rows × 16 cols. Two DPAS calls cover 16 rows. Use BK=16 per iteration. Store B transposed in SLM so b_vec is a contiguous vload16. Vectorize A loads. Fold tail into main loop.

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Launch metadata:
//   LWS = {16, 1, 1}   (1 subgroup of 16 lanes per WG)
//   GWS = {ceil_div(N,16)*16, ceil_div(M,16), 1}
//   Each subgroup computes one 16x16 C tile
//   BK=16 per iteration, 2 DPAS calls (rows 0-7, rows 8-15)
//   B stored transposed in SLM for contiguous vload16

#define TILE_M 16
#define TILE_N 16
#define BK 16

__attribute__((reqd_work_group_size(16, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lane = get_sub_group_local_id();
    const int tile_col = get_group_id(0) * TILE_N;
    const int tile_row = get_group_id(1) * TILE_M;

    // Accumulators: 2 x float8 = 16 rows x 16 cols (lane = column)
    float8 acc_lo = (float8)(0.0f);
    float8 acc_hi = (float8)(0.0f);

    // SLM: A[16][16], B transposed as Bt[16][16] (col-major B -> each row is one output column's K slice)
    // Bt[col][k] = B[k][col], so lane can vload16 its column's K values
    __local half Asub[TILE_M][BK];
    __local half Bt[TILE_N][BK];  // transposed B

    // Process all K in chunks of BK, zero-pad tail
    for (int kb = 0; kb < K; kb += BK) {
        // Load A[16][16]: 16 lanes, each loads 16 rows for its column
        // lane = k-column index within tile
        {
            int gk = kb + lane;
            bool k_valid = (gk < K);
            #pragma unroll
            for (int r = 0; r < TILE_M; ++r) {
                int gr = tile_row + r;
                Asub[r][lane] = (k_valid && gr < M) ? A[gr * K + gk] : (half)0.0h;
            }
        }

        // Load B transposed: Bt[col][k] = B[kb+k][tile_col+col]
        // lane = col index; load 16 k-values for that column
        {
            int gc = tile_col + lane;
            bool c_valid = (gc < N);
            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                int gk = kb + kk;
                Bt[lane][kk] = (c_valid && gk < K) ? B[gk * N + gc] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // b_vec: contiguous load from transposed layout
        half16 b_vec = vload16(0, &Bt[lane][0]);

        // DPAS for rows 0-7: lanes 0-7 provide A rows 0-7
        {
            half16 a_vec = vload16(0, &Asub[lane < 8 ? lane : 0][0]);
            acc_lo = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc_lo);
        }

        // DPAS for rows 8-15: lanes 0-7 provide A rows 8-15
        {
            half16 a_vec = vload16(0, &Asub[lane < 8 ? (lane + 8) : 8][0]);
            acc_hi = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc_hi);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    int col = tile_col + lane;
    if (col < N) {
        // Rows 0-7
        float acc_arr[8];
        acc_arr[0] = acc_lo.s0; acc_arr[1] = acc_lo.s1;
        acc_arr[2] = acc_lo.s2; acc_arr[3] = acc_lo.s3;
        acc_arr[4] = acc_lo.s4; acc_arr[5] = acc_lo.s5;
        acc_arr[6] = acc_lo.s6; acc_arr[7] = acc_lo.s7;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int gr = tile_row + r;
            if (gr < M)
                C[gr * N + col] = convert_half(acc_arr[r]);
        }

        // Rows 8-15
        acc_arr[0] = acc_hi.s0; acc_arr[1] = acc_hi.s1;
        acc_arr[2] = acc_hi.s2; acc_arr[3] = acc_hi.s3;
        acc_arr[4] = acc_hi.s4; acc_arr[5] = acc_hi.s5;
        acc_arr[6] = acc_hi.s6; acc_arr[7] = acc_hi.s7;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            int gr = tile_row + 8 + r;
            if (gr < M)
                C[gr * N + col] = convert_half(acc_arr[r]);
        }
    }
}
```

