

## Analysis

The key issues with previous kernels:

1. **DPAS output misuse**: Previous versions extract only `.s0` from the `float8` DPAS result, discarding 7/8 of the computed results. The DPAS intrinsic `intel_sub_group_f16_f16_matrix_mad_k16` computes an 8×16 output tile where each lane gets a column and the float8 contains 8 rows. We need to map the subgroup so that each lane's full float8 is used.

2. **Scalar gather for B operands**: Building `b_vec` element-by-element from SLM is slow. We should use `vload16` with proper layout.

3. **Subgroup-aware mapping**: For DPAS on Xe2, the correct mapping is: 1 subgroup of 16 lanes computes an 8×16 tile. Each lane owns one column, float8 gives 8 rows. To compute 16×16, we need 2 DPAS calls (rows 0-7, rows 8-15). Version 1 had this right but at 33.9ms.

4. **Strategy**: Use LWS={16,1,1} with each subgroup computing a 32×16 C tile (4 DPAS calls), process larger K chunks, and use proper SLM layout so B can be loaded with vload16. Double-buffer to hide latency.

```OCL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Launch metadata:
//   LWS = {16, 1, 1}
//   GWS = {ceil_div(N,16)*16, ceil_div(M,32), 1}
//   Each subgroup computes a 32x16 C tile (4 DPAS calls per K-chunk)
//   DPAS: intel_sub_group_f16_f16_matrix_mad_k16 produces float8 per lane
//     lane i owns column (tile_col + i), float8 = 8 consecutive rows

#define TM 32
#define TN 16
#define TK 16

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
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);

    const int tile_row = gy * TM;
    const int tile_col = gx * TN;
    const int col = tile_col + lane;

    // 4 accumulators: rows [0:7], [8:15], [16:23], [24:31]
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // SLM: A[32][16] stored row-major, B[16][16] stored transposed (column-major for vload)
    // For B: store as B_slm[col_in_tile][k] so each lane can vload16 its column
    __local half Asub[TM * TK];       // 32*16 = 512
    __local half Bsub[TN * TK];       // 16*16 = 256, layout: Bsub[n][k]

    const int k_full_end = (K / TK) * TK;

    for (int kb = 0; kb < k_full_end; kb += TK) {
        // Load A[32][16]: 512 elements, 16 lanes => 32 each
        #pragma unroll
        for (int t = 0; t < 32; ++t) {
            int gr = tile_row + t;
            int gk = kb + lane;
            Asub[t * TK + lane] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // Load B[16][16] into transposed layout Bsub[n][k]:
        // Each lane loads one column (its own), all 16 k-values
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            int gk = kb + kk;
            Bsub[lane * TK + kk] = (gk < K && col < N) ? B[gk * N + col] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // B vector: each lane loads its column's 16 k-values with vload16
        half16 b_vec = vload16(0, &Bsub[lane * TK]);

        // 4 DPAS calls for 32 rows
        {
            half16 a_vec = vload16(0, &Asub[(lane < 8 ? lane : 0) * TK]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc0);
        }
        {
            half16 a_vec = vload16(0, &Asub[(lane < 8 ? lane + 8 : 8) * TK]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc1);
        }
        {
            half16 a_vec = vload16(0, &Asub[(lane < 8 ? lane + 16 : 16) * TK]);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc2);
        }
        {
            half16 a_vec = vload16(0, &Asub[(lane < 8 ? lane + 24 : 24) * TK]);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results + K tail
    if (col < N) {
        float acc_arr[32];
        vstore8(acc0, 0, &acc_arr[0]);
        vstore8(acc1, 0, &acc_arr[8]);
        vstore8(acc2, 0, &acc_arr[16]);
        vstore8(acc3, 0, &acc_arr[24]);

        for (int r = 0; r < 32; ++r) {
            int gr = tile_row + r;
            if (gr < M) {
                float v = acc_arr[r];
                for (int k = k_full_end; k < K; ++k) {
                    v += convert_float(A[gr * K + k]) * convert_float(B[k * N + col]);
                }
                C[gr * N + col] = convert_half(v);
            }
        }
    }
}
```

