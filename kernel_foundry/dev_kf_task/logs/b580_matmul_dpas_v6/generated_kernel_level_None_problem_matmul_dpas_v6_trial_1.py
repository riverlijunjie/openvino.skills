## Analysis

The key issues from previous attempts:

1. **Launch configuration not parsed**: The framework expects `reqd_work_group_size` attribute to be parseable. Previous kernels had it but the launch grid computation was inconsistent.

2. **A operand packing bug**: With DPAS k16, each work-item (lane) in a subgroup of 16 holds one int per row, where that int packs 2 halfs. So lane `l` holds K indices `[2*l, 2*l+1]`. With K=16, lane 0 gets K[0,1], lane 7 gets K[14,15] — this only works for 8 lanes, but subgroup size is 16. This means the A matrix distribution is different than assumed.

3. **Misunderstanding DPAS layout**: Looking at Intel's documentation more carefully for `intel_sub_group_f16_f16_matrix_mad_k16`:
   - **A (int8)**: Represents 8×16 matrix. 8 rows, K=16. Each lane holds one int per row. With 16 lanes and 8 ints per lane, the data is distributed: lane `l` holds k-index `l` for each row, but packed as pairs... Actually, the correct layout is: each component of int8 covers a row, and within that row, 16 K values are spread across 16 lanes as pairs of halfs packed into int. So lane `l` holds `A[row][2*l]` and `A[row][2*l+1]`... but that's 32 K values for 16 lanes, not 16.

Let me reconsider. For k16 with subgroup size 16: A is int8 (8 ints per work-item). Each int packs 2 half values. 8 ints × 2 halfs = 16 halfs. So each work-item holds ALL 16 K values for ONE row? No, 8 rows... So it must be: the 16 K values are distributed across the subgroup. With 16 lanes, each lane holds 1 half per row (K/16 = 1). But then int packs 2 halfs = 2 rows per int, and int8 = 16 rows... but we have 8 rows.

Let me try a simpler approach: avoid DPAS complexity and use `intel_sub_group_f16_f16_matrix_mad_k16` with a well-tested, minimal layout, using SLM + `intel_sub_group_block_read` for correct data feeding.

Given repeated failures with DPAS packing, let me use a **simple subgroup-based approach without DPAS** but with register blocking and SLM tiling, which will at least be correct:

```OCL
// FP16 GEMM with SLM tiling and register blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Each work-item computes THREAD_M x THREAD_N output elements
// THREAD_M=8, THREAD_N=1
// LWS = (16, 16, 1) = 256 work-items
// WG tile: 128 rows x 16 cols
// GWS = (ceil(N/16)*16, ceil(M/128)*16, 1)
// K-tile = 16

#define TILE_M 128
#define TILE_N 16
#define TILE_K 16
#define THREAD_M 8

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int lx = get_local_id(0);  // 0..15 -> column within tile
    const int ly = get_local_id(1);  // 0..15 -> row group

    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    const int out_col = wg_n + lx;
    const int row_base = wg_m + ly * THREAD_M;

    // Accumulators
    float acc[THREAD_M];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) acc[i] = 0.0f;

    // SLM tiles
    __local half A_slm[TILE_M * TILE_K];  // 128 x 16
    __local half B_slm[TILE_K * TILE_N];  // 16 x 16

    const int flat_id = ly * 16 + lx;  // 0..255
    const int num_threads = 256;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperative load A_slm[128][16] = 2048 halfs, 256 threads => 8 each
        #pragma unroll
        for (int i = flat_id; i < TILE_M * TILE_K; i += num_threads) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int gm = wg_m + r;
            int gk = k0 + c;
            A_slm[i] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
        }

        // Cooperative load B_slm[16][16] = 256 halfs, 256 threads => 1 each
        {
            int i = flat_id;
            if (i < TILE_K * TILE_N) {
                int r = i / TILE_N;
                int c = i % TILE_N;
                int gk = k0 + r;
                int gn = wg_n + c;
                B_slm[i] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: each thread does THREAD_M rows x 1 col, iterating over K tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float b_val = convert_float(B_slm[kk * TILE_N + lx]);
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                float a_val = convert_float(A_slm[(ly * THREAD_M + r) * TILE_K + kk]);
                acc[r] += a_val * b_val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    if (out_col < N) {
        #pragma unroll
        for (int r = 0; r < THREAD_M; r++) {
            int out_row = row_base + r;
            if (out_row < M) {
                C[out_row * N + out_col] = convert_half(acc[r]);
            }
        }
    }
}
```

