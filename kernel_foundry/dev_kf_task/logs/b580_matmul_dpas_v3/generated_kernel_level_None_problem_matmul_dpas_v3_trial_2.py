

## Analysis

The best version so far (65.8ms) uses TILE_M=64, TILE_N=64, TILE_K=16 with 4x4 per work-item. Key observations:

1. **Versions 1&2 (172-177ms)**: Used 128x128 tiles with 8x8 per WI - too much register pressure and SLM, causing spills and poor occupancy.
2. **Version 3 (65.8ms)**: 64x64 tiles with 4x4 per WI - better balance but lacks `#pragma unroll` and vectorized loads.

**Improvements to try:**
- Use vectorized SLM loads (vload4/vload8) for better memory throughput
- Add `#pragma unroll` on all inner loops
- Pad SLM to avoid bank conflicts (TILE_K+1 for A, TILE_N+1 for B would be wasteful, but padding A's K dimension helps)
- Increase work per WI slightly: try 64x64 tile with 4x4 per WI but with TILE_K=32 to reduce barrier overhead
- Try TILE_M=64, TILE_N=64, TILE_K=32 to amortize load overhead
- Use `intel_reqd_sub_group_size(16)` attribute

Let me try a version with TILE_K=32 (double the K tile to halve the number of iterations and barriers), vectorized cooperative loads, and full unrolling.

Actually, let me also try increasing per-WI work to 4x8 with a 64x128 tile using 16x16 WG. That gives better arithmetic intensity. But let me be conservative and stick with what worked, just optimizing the inner loop and loads.

## Improved OCL code

```OCL
// Tiled FP16 matmul: C[M,N] = A[M,K] x B[K,N], accumulation in FP32
// TILE_M=64, TILE_N=64, TILE_K=32
// LWS = (16, 16) = 256 work-items
// Each WI computes a 4x4 sub-tile of C
// GWS = (ceil_div(N,64)*16, ceil_div(M,64)*16)
// SLM: A_tile = 64*32 = 2048 halfs (4KB), B_tile = 32*64 = 2048 halfs (4KB)

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define WG_X 16
#define WG_Y 16
#define ITEMS_M 4
#define ITEMS_N 4
#define TOTAL_WIS 256

// Pad TILE_K by 1 to avoid SLM bank conflicts on A reads
#define A_STRIDE (TILE_K + 1)

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
    const int tile_col = get_group_id(0) * TILE_N;
    const int tile_row = get_group_id(1) * TILE_M;

    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lid = lid_y * WG_X + lid_x;

    // Accumulators
    float acc[ITEMS_M][ITEMS_N];
    #pragma unroll
    for (int i = 0; i < ITEMS_M; i++)
        #pragma unroll
        for (int j = 0; j < ITEMS_N; j++)
            acc[i][j] = 0.0f;

    // SLM tiles - A padded to avoid bank conflicts
    __local half A_tile[TILE_M * A_STRIDE];  // 64 * 33
    __local half B_tile[TILE_K * TILE_N];    // 32 * 64

    // Per-WI sub-tile position
    const int local_row_base = lid_y * ITEMS_M;
    const int local_col_base = lid_x * ITEMS_N;

    // A_tile: 64*32 = 2048 elements, 256 WIs => 8 each
    // B_tile: 32*64 = 2048 elements, 256 WIs => 8 each

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A_tile[TILE_M][TILE_K] into padded SLM
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int idx = lid + t * TOTAL_WIS;
            int r = idx / TILE_K;  // row in [0, TILE_M)
            int c = idx % TILE_K;  // col in [0, TILE_K)
            int gr = tile_row + r;
            int gc = k0 + c;
            A_tile[r * A_STRIDE + c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Load B_tile[TILE_K][TILE_N]
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int idx = lid + t * TOTAL_WIS;
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gr = k0 + r;
            int gc = tile_col + c;
            B_tile[r * TILE_N + c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 4x4 sub-tile with full unrolling
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float a_val[ITEMS_M];
            #pragma unroll
            for (int i = 0; i < ITEMS_M; i++) {
                a_val[i] = convert_float(A_tile[(local_row_base + i) * A_STRIDE + kk]);
            }

            float b_val[ITEMS_N];
            #pragma unroll
            for (int j = 0; j < ITEMS_N; j++) {
                b_val[j] = convert_float(B_tile[kk * TILE_N + local_col_base + j]);
            }

            #pragma unroll
            for (int i = 0; i < ITEMS_M; i++) {
                #pragma unroll
                for (int j = 0; j < ITEMS_N; j++) {
                    acc[i][j] = fma(a_val[i], b_val[j], acc[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < ITEMS_M; i++) {
        int gr = tile_row + local_row_base + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < ITEMS_N; j++) {
            int gc = tile_col + local_col_base + j;
            if (gc < N) {
                C[gr * N + gc] = convert_half(acc[i][j]);
            }
        }
    }
}
```

