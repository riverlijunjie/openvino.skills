## Analysis

The previous kernels all achieve ~33.9ms. Key issues:

1. **Low arithmetic intensity**: 256 threads computing 256 outputs with 1 accumulator each — too little work per thread relative to sync/load overhead.
2. **Scalar B gather from SLM**: 16-iteration loop to build `b_vec` is inefficient.
3. **Two barriers per BK block** with small BK=32.

**Proposed improvements:**
- **Register blocking**: Each thread computes multiple C rows (4 rows), so a 16×4 workgroup (64 threads) computes a 16×16 tile, or better: use 16×1 WG (single subgroup) computing a 32×16 tile with 32 accumulators per lane (4 DPAS calls per k-step).
- **Larger tile**: 32M × 16N per workgroup, single subgroup of 16 lanes. Each DPAS produces 8 rows × 16 cols, so 4 DPAS calls = 32 rows.
- **Transposed B in SLM** for contiguous `vload16`.
- **Larger BK=32** with double-step DPAS to amortize load/barrier cost.
- **Prefetch / unrolled loads** to hide latency.

For the DPAS intrinsic `intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc)`:
- `a`: half16 per lane (broadcast across subgroup for 8 rows)
- `b`: half16 per lane (each lane's column data)
- `acc`: float8 (8 row results for this lane's column)
- Only lanes 0-7 contribute `a` values for 8 output rows.

So for 32 rows: 4 DPAS calls (rows 0-7, 8-15, 16-23, 24-31), each needing different `a` data from lanes 0-7.

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Launch metadata:
//   LWS = {16, 1, 1}   (single subgroup of 16 lanes)
//   GWS = {ceil_div(N,16)*16, ceil_div(M,32), 1}
//   Each WG computes a 32x16 C tile
//   4 DPAS(k16) calls per k-step => 32 rows x 16 cols
//   BK=32 => 8 DPAS calls per SLM load (amortize barrier cost)

#define TILE_M 32
#define TILE_N 16
#define BK     32

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
    const int wg_x = get_group_id(0);
    const int wg_y = get_group_id(1);

    const int tile_row = wg_y * TILE_M;
    const int tile_col = wg_x * TILE_N;

    // 4 accumulators for 32 rows (4 groups of 8 rows) x 16 cols
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // SLM: A[32][BK] and B transposed as Bt[16][BK] for contiguous vload
    // Pad to avoid bank conflicts
    __local half Asub[TILE_M][BK + 2];
    __local half Bt[TILE_N][BK + 2];   // Bt[col][k]

    const int k_full_end = (K / BK) * BK;

    for (int kb = 0; kb < K; kb += BK) {
        // Cooperative load with 16 threads
        // A: 32 x 32 = 1024 elems, 16 threads => 64 each
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            int idx = lane + i * 16;  // 0..1023
            int ar = idx / BK;        // 0..31
            int ak = idx % BK;        // 0..31
            int gr = tile_row + ar;
            int gk = kb + ak;
            Asub[ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // B: need B[kb:kb+BK][tile_col:tile_col+16] => 32x16=512 elems
        // Store transposed: Bt[col][k]
        // 512 / 16 = 32 per thread
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int idx = lane + i * 16;  // 0..511
            int bk = idx / TILE_N;    // 0..31
            int bc = idx % TILE_N;    // 0..15
            int gk = kb + bk;
            int gc = tile_col + bc;
            Bt[bc][bk] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Process BK in two k16 steps
        #pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            // b_vec: each lane loads its column's k-values contiguously from Bt
            half16 b_vec = vload16(0, &Bt[lane][ks]);

            // 4 DPAS calls for rows 0-7, 8-15, 16-23, 24-31
            // For DPAS, lanes 0-7 provide the A data for 8 output rows
            {
                half16 a_vec = vload16(0, &Asub[lane < 8 ? lane : 0][ks]);
                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc0);
            }
            {
                half16 a_vec = vload16(0, &Asub[(lane < 8 ? lane : 0) + 8][ks]);
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc1);
            }
            {
                half16 a_vec = vload16(0, &Asub[(lane < 8 ? lane : 0) + 16][ks]);
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc2);
            }
            {
                half16 a_vec = vload16(0, &Asub[(lane < 8 ? lane : 0) + 24][ks]);
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc3);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    int col = tile_col + lane;
    if (col >= N) return;

    // Helper macro to store 8 rows from a float8 accumulator
    #define STORE_ACC(acc_v, row_off) \
    { \
        float acc_arr[8]; \
        acc_arr[0] = acc_v.s0; acc_arr[1] = acc_v.s1; \
        acc_arr[2] = acc_v.s2; acc_arr[3] = acc_v.s3; \
        acc_arr[4] = acc_v.s4; acc_arr[5] = acc_v.s5; \
        acc_arr[6] = acc_v.s6; acc_arr[7] = acc_v.s7; \
        for (int r = 0; r < 8; ++r) { \
            int gr = tile_row + (row_off) + r; \
            if (gr < M) { \
                float v = acc_arr[r]; \
                for (int k = k_full_end; k < K; ++k) \
                    v += convert_float(A[gr * K + k]) * convert_float(B[k * N + col]); \
                C[gr * N + col] = convert_half(v); \
            } \
        } \
    }

    STORE_ACC(acc0, 0)
    STORE_ACC(acc1, 8)
    STORE_ACC(acc2, 16)
    STORE_ACC(acc3, 24)

    #undef STORE_ACC
}
```

