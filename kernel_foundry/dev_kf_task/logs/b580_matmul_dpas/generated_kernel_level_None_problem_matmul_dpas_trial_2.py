

## Analysis

Looking at the progression:
- V1 (64x64 tile, 4x4 per WI): 62.3ms
- V2 (32x64 tile, 2x4 per WI): 40.1ms  
- V3 (32x32 tile, 2x2 per WI): 26.0ms - **best so far**

Interestingly, the smallest tile with least work per thread won. This suggests the bottleneck is SLM capacity/bandwidth rather than global memory. The 32x32 tiles fit well in SLM and have less pressure.

Key opportunities:
1. **Increase compute per thread** while keeping SLM small - use register blocking more aggressively
2. **Vectorized loads** (vload4/vload8) for coalesced global memory access
3. **SLM bank conflict avoidance** by padding
4. **Larger TILE_K** to reduce barrier overhead relative to compute
5. **Vectorized SLM reads** in the inner loop

My approach: Keep TILE_M=32, TILE_N=32, but increase TILE_K to 64 to amortize load/barrier cost. Use 2x2 per WI but with vectorized loads. Pad SLM to avoid bank conflicts. Use vload2 for B tile reads in inner loop.

Actually, let me try a different approach: TILE_M=64, TILE_N=32, TILE_K=32, with 4x2 per WI. This increases arithmetic intensity per barrier while keeping SLM reasonable (64*32 + 32*32 = 3072 halfs = 6KB).

Let me try keeping the winning 32x32 structure but with TILE_K=64 and vectorized loads:

```OCL
// Tiled FP16 matmul with SLM blocking + vectorized loads + bank conflict padding
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
// LWS = (16, 16, 1)
// GWS = (ceil_div(N, 32)*16, ceil_div(M, 32)*16, 1)
// Subgroup size: 16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32
#define WG_X 16
#define WG_Y 16
// Pad to avoid bank conflicts (32 banks of 4 bytes = 64 halfs per bank cycle)
#define AS_STRIDE (TILE_K + 2)
#define BS_STRIDE (TILE_N + 2)

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

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lid = ly * WG_X + lx;

    const int out_row0 = tile_row + ly * 2;
    const int out_col0 = tile_col + lx * 2;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    __local half As[TILE_M * AS_STRIDE];
    __local half Bs[TILE_K * BS_STRIDE];

    for (int kk = 0; kk < K; kk += TILE_K) {
        // Load A tile [32 x 32] with padding stride
        for (int i = lid; i < TILE_M * TILE_K; i += 256) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int gr = tile_row + r;
            int gc = kk + c;
            As[r * AS_STRIDE + c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Load B tile [32 x 32] with padding stride
        for (int i = lid; i < TILE_K * TILE_N; i += 256) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int gr = kk + r;
            int gc = tile_col + c;
            Bs[r * BS_STRIDE + c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int r0 = ly * 2;
        int r1 = ly * 2 + 1;
        int c0 = lx * 2;
        int c1 = lx * 2 + 1;

        // Unroll inner loop by 4
        int tk = 0;
        for (; tk + 3 < TILE_K; tk += 4) {
            float a0_0 = convert_float(As[r0 * AS_STRIDE + tk]);
            float a1_0 = convert_float(As[r1 * AS_STRIDE + tk]);
            float b0_0 = convert_float(Bs[tk * BS_STRIDE + c0]);
            float b1_0 = convert_float(Bs[tk * BS_STRIDE + c1]);

            float a0_1 = convert_float(As[r0 * AS_STRIDE + tk + 1]);
            float a1_1 = convert_float(As[r1 * AS_STRIDE + tk + 1]);
            float b0_1 = convert_float(Bs[(tk+1) * BS_STRIDE + c0]);
            float b1_1 = convert_float(Bs[(tk+1) * BS_STRIDE + c1]);

            float a0_2 = convert_float(As[r0 * AS_STRIDE + tk + 2]);
            float a1_2 = convert_float(As[r1 * AS_STRIDE + tk + 2]);
            float b0_2 = convert_float(Bs[(tk+2) * BS_STRIDE + c0]);
            float b1_2 = convert_float(Bs[(tk+2) * BS_STRIDE + c1]);

            float a0_3 = convert_float(As[r0 * AS_STRIDE + tk + 3]);
            float a1_3 = convert_float(As[r1 * AS_STRIDE + tk + 3]);
            float b0_3 = convert_float(Bs[(tk+3) * BS_STRIDE + c0]);
            float b1_3 = convert_float(Bs[(tk+3) * BS_STRIDE + c1]);

            acc00 = fma(a0_0, b0_0, acc00);
            acc01 = fma(a0_0, b1_0, acc01);
            acc10 = fma(a1_0, b0_0, acc10);
            acc11 = fma(a1_0, b1_0, acc11);

            acc00 = fma(a0_1, b0_1, acc00);
            acc01 = fma(a0_1, b1_1, acc01);
            acc10 = fma(a1_1, b0_1, acc10);
            acc11 = fma(a1_1, b1_1, acc11);

            acc00 = fma(a0_2, b0_2, acc00);
            acc01 = fma(a0_2, b1_2, acc01);
            acc10 = fma(a1_2, b0_2, acc10);
            acc11 = fma(a1_2, b1_2, acc11);

            acc00 = fma(a0_3, b0_3, acc00);
            acc01 = fma(a0_3, b1_3, acc01);
            acc10 = fma(a1_3, b0_3, acc10);
            acc11 = fma(a1_3, b1_3, acc11);
        }
        for (; tk < TILE_K; ++tk) {
            float a0 = convert_float(As[r0 * AS_STRIDE + tk]);
            float a1 = convert_float(As[r1 * AS_STRIDE + tk]);
            float b0 = convert_float(Bs[tk * BS_STRIDE + c0]);
            float b1 = convert_float(Bs[tk * BS_STRIDE + c1]);
            acc00 = fma(a0, b0, acc00);
            acc01 = fma(a0, b1, acc01);
            acc10 = fma(a1, b0, acc10);
            acc11 = fma(a1, b1, acc11);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (out_row0 < M && out_col0 < N)
        C[out_row0 * N + out_col0] = convert_half(acc00);
    if (out_row0 < M && (out_col0 + 1) < N)
        C[out_row0 * N + out_col0 + 1] = convert_half(acc01);
    if ((out_row0 + 1) < M && out_col0 < N)
        C[(out_row0 + 1) * N + out_col0] = convert_half(acc10);
    if ((out_row0 + 1) < M && (out_col0 + 1) < N)
        C[(out_row0 + 1) * N + out_col0 + 1] = convert_half(acc11);
}
```

