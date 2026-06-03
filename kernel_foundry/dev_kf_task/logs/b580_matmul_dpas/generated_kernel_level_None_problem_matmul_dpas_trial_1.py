

## Analysis

The best kernel so far (26ms) uses 32x32 tiles with 2x2 per work-item. The previous attempt (62.3ms) used 64x64 tiles with 4x4 per work-item but was slower, likely due to:

1. **Sequential loading pattern**: `flat_id * 4 + i` causes non-coalesced memory access (strided by 4)
2. **Larger tile with same thread count**: More SLM pressure, more bank conflicts
3. **TILE_K=16 is too small**: More loop iterations, more barrier overhead

For the new kernel, I'll optimize the 32x32 tile approach that works well:
- Increase per-thread work to 4x4 with 64x64 tiles but fix the loading pattern
- Use vectorized loads (vload4) for coalesced access
- Use `fma` consistently
- Better SLM layout to reduce bank conflicts
- Cooperative loading with coalesced access patterns

Let me try a refined approach: **TILE_M=32, TILE_N=64, TILE_K=32** with WG=(16,16) where each thread computes 2x4 elements. This gives better N-dimension coverage and vectorized B loads.

Actually, let me stick closer to what works (32x32 tiles) but optimize the inner loop and memory access patterns.

```OCL
// Tiled FP16 matmul with SLM blocking - optimized
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
// TILE_M=32, TILE_N=64, TILE_K=32
// Each WI computes 2x4 sub-tile
// LWS = (16, 16, 1)
// GWS = (ceil_div(N, 64)*16, ceil_div(M, 32)*16, 1)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define WG_X 16
#define WG_Y 16

__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int tile_col = get_group_id(0) * TILE_N;
    const int tile_row = get_group_id(1) * TILE_M;

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lid = ly * WG_X + lx; // 0..255

    // Each thread computes 2 rows x 4 cols
    const int out_row0 = tile_row + ly * 2;
    const int out_col0 = tile_col + lx * 4;

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;

    __local half As[TILE_M * TILE_K]; // 32x32 = 1024
    __local half Bs[TILE_K * TILE_N]; // 32x64 = 2048

    for (int kk = 0; kk < K; kk += TILE_K) {
        // Load A tile [32x32] = 1024 elements, 256 threads => 4 each
        for (int i = lid; i < TILE_M * TILE_K; i += 256) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int gr = tile_row + r;
            int gc = kk + c;
            As[i] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Load B tile [32x64] = 2048 elements, 256 threads => 8 each
        for (int i = lid; i < TILE_K * TILE_N; i += 256) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int gr = kk + r;
            int gc = tile_col + c;
            Bs[i] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int r0 = ly * 2;
        int r1 = ly * 2 + 1;
        int c0 = lx * 4;

        for (int tk = 0; tk < TILE_K; ++tk) {
            float a0 = convert_float(As[r0 * TILE_K + tk]);
            float a1 = convert_float(As[r1 * TILE_K + tk]);
            float b0 = convert_float(Bs[tk * TILE_N + c0]);
            float b1 = convert_float(Bs[tk * TILE_N + c0 + 1]);
            float b2 = convert_float(Bs[tk * TILE_N + c0 + 2]);
            float b3 = convert_float(Bs[tk * TILE_N + c0 + 3]);

            acc00 = fma(a0, b0, acc00);
            acc01 = fma(a0, b1, acc01);
            acc02 = fma(a0, b2, acc02);
            acc03 = fma(a0, b3, acc03);
            acc10 = fma(a1, b0, acc10);
            acc11 = fma(a1, b1, acc11);
            acc12 = fma(a1, b2, acc12);
            acc13 = fma(a1, b3, acc13);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store 2x4 results
    if (out_row0 < M) {
        if (out_col0 < N)     C[out_row0 * N + out_col0]     = convert_half(acc00);
        if (out_col0 + 1 < N) C[out_row0 * N + out_col0 + 1] = convert_half(acc01);
        if (out_col0 + 2 < N) C[out_row0 * N + out_col0 + 2] = convert_half(acc02);
        if (out_col0 + 3 < N) C[out_row0 * N + out_col0 + 3] = convert_half(acc03);
    }
    if (out_row0 + 1 < M) {
        if (out_col0 < N)     C[(out_row0+1) * N + out_col0]     = convert_half(acc10);
        if (out_col0 + 1 < N) C[(out_row0+1) * N + out_col0 + 1] = convert_half(acc11);
        if (out_col0 + 2 < N) C[(out_row0+1) * N + out_col0 + 2] = convert_half(acc12);
        if (out_col0 + 3 < N) C[(out_row0+1) * N + out_col0 + 3] = convert_half(acc13);
    }
}
```

