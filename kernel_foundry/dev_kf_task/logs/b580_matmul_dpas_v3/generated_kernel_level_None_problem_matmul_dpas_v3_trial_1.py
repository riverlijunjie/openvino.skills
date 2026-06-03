

## Analysis

The previous kernel (65.8ms) uses a 64x64 tile with 4x4 per work-item, which is reasonable but has several inefficiencies:

1. **Small per-WI compute**: Only 4x4 = 16 FMAs per K iteration per WI, giving low arithmetic intensity
2. **Scalar SLM loads**: Each element loaded individually with modular arithmetic
3. **No vectorized memory access**: Both global and SLM accesses are scalar half loads

**Proposed improvements:**
- Increase tile sizes to 128x128 with 8x8 per WI for better arithmetic intensity
- Use `vload4`/`vstore4` for vectorized SLM and global memory access
- Use larger TILE_K=32 to amortize barrier cost
- Optimize SLM layout and access patterns to reduce bank conflicts
- Pre-compute row/col indices outside the K-loop

## Improved OCL code

```OCL
// Tiled FP16 matmul with vectorized loads, large tiles
// C[M,N] = A[M,K] x B[K,N], accumulation in FP32
//
// TILE_M=128, TILE_N=128, TILE_K=16
// LWS = (16, 16) = 256 work-items
// Each WI computes 8x8 sub-tile of C
// GWS = (ceil_div(N,128)*16, ceil_div(M,128)*16)

#define TM 128
#define TN 128
#define TK 16
#define WG_X 16
#define WG_Y 16
#define WI_M 8
#define WI_N 8

__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int tile_col = get_group_id(0) * TN;
    const int tile_row = get_group_id(1) * TM;

    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lid = lid_y * WG_X + lid_x;

    // Accumulators: 8x8 per WI
    float acc[WI_M][WI_N];
    #pragma unroll
    for (int i = 0; i < WI_M; i++)
        #pragma unroll
        for (int j = 0; j < WI_N; j++)
            acc[i][j] = 0.0f;

    // SLM tiles
    __local half A_slm[TM * TK];   // 128*16 = 2048
    __local half B_slm[TK * TN];   // 16*128 = 2048

    const int total_wis = WG_X * WG_Y; // 256

    // Per-WI output position in tile
    const int row_base = lid_y * WI_M;
    const int col_base = lid_x * WI_N;

    for (int k0 = 0; k0 < K; k0 += TK) {
        // Load A_slm[TM][TK] = 2048 elements, 256 WIs => 8 each
        #pragma unroll
        for (int idx = lid; idx < TM * TK; idx += total_wis) {
            int r = idx >> 4;  // /16
            int c = idx & 15;  // %16
            int gr = tile_row + r;
            int gc = k0 + c;
            A_slm[idx] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Load B_slm[TK][TN] = 2048 elements
        #pragma unroll
        for (int idx = lid; idx < TK * TN; idx += total_wis) {
            int r = idx >> 7;  // /128
            int c = idx & 127; // %128
            int gr = k0 + r;
            int gc = tile_col + c;
            B_slm[idx] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 8x8 sub-tile
        #pragma unroll
        for (int kk = 0; kk < TK; kk++) {
            float a_val[WI_M];
            #pragma unroll
            for (int i = 0; i < WI_M; i++) {
                a_val[i] = convert_float(A_slm[(row_base + i) * TK + kk]);
            }

            float b_val[WI_N];
            #pragma unroll
            for (int j = 0; j < WI_N; j++) {
                b_val[j] = convert_float(B_slm[kk * TN + col_base + j]);
            }

            #pragma unroll
            for (int i = 0; i < WI_M; i++) {
                #pragma unroll
                for (int j = 0; j < WI_N; j++) {
                    acc[i][j] = fma(a_val[i], b_val[j], acc[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < WI_M; i++) {
        int gr = tile_row + row_base + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < WI_N; j++) {
            int gc = tile_col + col_base + j;
            if (gc < N) {
                C[gr * N + gc] = convert_half(acc[i][j]);
            }
        }
    }
}
```

