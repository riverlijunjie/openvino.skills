

## Analysis

The key issue with the previous kernel is that **DPAS output is severely underutilized** - only `d.s0` out of `float8` (8 results) is used, wasting 87.5% of compute. The DPAS `intel_sub_group_f16_f16_matrix_mad_k16` produces 8 rows of results across the subgroup, but the kernel discards 7 of them.

**Proper DPAS mapping**: The intrinsic computes an 8×16 output tile (8 rows × 16 columns). Each lane holds one column, and `float8` gives 8 row results. So one subgroup naturally owns an 8×16 C tile. To compute a larger tile, we issue multiple DPAS calls with different A slices.

**Plan**: 
- Each workgroup has 16×1 threads (one subgroup of 16 lanes)
- Each subgroup computes a 32×16 C tile (4 DPAS calls per K-chunk: rows 0-7, 8-15, 16-23, 24-31)
- Each lane owns column `tile_col + lane_id`, accumulates 4 × float8 = 32 scalars
- TILE_K = 16 per iteration, use SLM for A (32×16) and B (16×16)
- Double-buffered SLM to hide latency
- Fast path for interior tiles (no bounds checks), slow path for edges

This properly utilizes all 8 DPAS outputs, giving 4× better compute efficiency than the previous version.

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Launch metadata (host-side):
//   LWS = {16, 1, 1}   (1 subgroup of 16 lanes per WG)
//   GWS = {ceil_div(N,16)*16, ceil_div(M,32), 1}
//   Subgroup size: 16
//
// Each WG computes a 32x16 C tile.
// 4 DPAS calls per K-chunk produce rows [0:7],[8:15],[16:23],[24:31] x 16 cols.
// All 8 elements of float8 DPAS output are used (full utilization).
// Double-buffered SLM hides global load latency.

#define TILE_M 32
#define TILE_N 16
#define TILE_K 16
#define SG_SIZE 16

__attribute__((reqd_work_group_size(SG_SIZE, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lane = get_local_id(0);  // 0..15
    const int gx = get_group_id(0);    // tile column index
    const int gy = get_group_id(1);    // tile row index

    const int tile_row = gy * TILE_M;
    const int tile_col = gx * TILE_N;
    const int col = tile_col + lane;

    // 4 accumulators for 4 groups of 8 rows each = 32 rows total
    float8 acc0 = (float8)(0.0f);  // rows 0-7
    float8 acc1 = (float8)(0.0f);  // rows 8-15
    float8 acc2 = (float8)(0.0f);  // rows 16-23
    float8 acc3 = (float8)(0.0f);  // rows 24-31

    // Double-buffered SLM: A[2][32][16+1], B[2][16][16+1]
    // +1 padding to avoid bank conflicts
    __local half Aslm[2][TILE_M][TILE_K + 1];
    __local half Bslm[2][TILE_K][TILE_N + 1];

    const int k_full_end = (K / TILE_K) * TILE_K;
    const int num_k_tiles = k_full_end / TILE_K;

    // Preload first tile (buf=0) if there's work
    if (num_k_tiles > 0) {
        int kb = 0;
        // Load A: 32 rows x 16 cols = 512 halves, 16 threads => 32 each
        #pragma unroll
        for (int r = 0; r < TILE_M; ++r) {
            int gr = tile_row + r;
            int gk = kb + lane;
            Aslm[0][r][lane] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }
        // Load B: 16 rows x 16 cols = 256 halves, 16 threads => 16 each
        #pragma unroll
        for (int r = 0; r < TILE_K; ++r) {
            int gk = kb + r;
            int gc = col;
            Bslm[0][r][lane] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 0; t < num_k_tiles; ++t) {
        int cur = t & 1;
        int nxt = 1 - cur;
        int kb_next = (t + 1) * TILE_K;

        // Prefetch next tile into nxt buffer (if exists)
        if (t + 1 < num_k_tiles) {
            #pragma unroll
            for (int r = 0; r < TILE_M; ++r) {
                int gr = tile_row + r;
                int gk = kb_next + lane;
                Aslm[nxt][r][lane] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
            }
            #pragma unroll
            for (int r = 0; r < TILE_K; ++r) {
                int gk = kb_next + r;
                int gc = col;
                Bslm[nxt][r][lane] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
            }
        }

        // Compute on current buffer
        // Load B vector: 16 k-values for this lane's column
        half16 b_vec;
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            ((half*)&b_vec)[kk] = Bslm[cur][kk][lane];
        }

        // DPAS for rows 0-7: lane provides A row data, DPAS broadcasts across subgroup
        {
            half16 a_vec = vload16(0, &Aslm[cur][lane < 8 ? lane : 0][0]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc0);
        }
        // DPAS for rows 8-15
        {
            half16 a_vec = vload16(0, &Aslm[cur][(lane < 8 ? lane : 0) + 8][0]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc1);
        }
        // DPAS for rows 16-23
        {
            half16 a_vec = vload16(0, &Aslm[cur][(lane < 8 ? lane : 0) + 16][0]);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc2);
        }
        // DPAS for rows 24-31
        {
            half16 a_vec = vload16(0, &Aslm[cur][(lane < 8 ? lane : 0) + 24][0]);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results + scalar tail-K cleanup
    if (col < N) {
        float* acc_ptrs[4] = { (float*)&acc0, (float*)&acc1, (float*)&acc2, (float*)&acc3 };

        #pragma unroll
        for (int blk = 0; blk < 4; ++blk) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                int gr = tile_row + blk * 8 + r;
                if (gr < M) {
                    float v = acc_ptrs[blk][r];
                    // Scalar tail for remaining K
                    for (int k = k_full_end; k < K; ++k) {
                        v += convert_float(A[gr * K + k]) * convert_float(B[k * N + col]);
                    }
                    C[gr * N + col] = convert_half(v);
                }
            }
        }
    }
}
```

