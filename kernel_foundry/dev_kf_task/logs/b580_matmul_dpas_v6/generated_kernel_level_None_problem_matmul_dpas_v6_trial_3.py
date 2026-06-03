

## Analysis

Looking at the progression:
- V1 (95.9ms): Double-buffered SLM, 64x64 tile, 8x4 per thread, LWS=128
- V2 (62.3ms): Single-buffered SLM, 64x64 tile, 4x4 per thread with strided access, LWS=256
- V3 (47.0ms): Single-buffered, 128x16 tile, 8x1 per thread, LWS=256

V3 is fastest but has low arithmetic intensity (THREAD_N=1). The feedback says to:
1. Increase THREAD_N to reuse A values across multiple columns
2. Vectorize global memory loads
3. Use double-buffering to hide latency

My approach for V4:
- **Tile: 128 x 32**, THREAD_M=8, THREAD_N=2 → each thread computes 8×2=16 outputs
- **LWS = (16, 16, 1) = 256 threads**: 16 threads cover 32 cols (2 each), 16 threads cover 128 rows (8 each)
- **TILE_K = 32** to reduce barrier frequency
- **Double-buffered SLM** to overlap load and compute
- **Vectorized loads** using vload2/vload4 where possible
- Use `select` for predication instead of branches

This should improve arithmetic intensity (each `a_val` reused across 2 FMAs) while keeping the tall-tile shape that worked well in V3, and reducing sync overhead.

## Improved OCL code

```OCL
// FP16 GEMM with double-buffered SLM tiling and register blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: TILE_M=128 x TILE_N=32, TILE_K=32
// Each work-item computes THREAD_M=8 rows x THREAD_N=2 cols = 16 outputs
// LWS = (16, 16, 1) = 256 work-items
// 16 threads along x cover 32 cols (2 each), 16 threads along y cover 128 rows (8 each)
// GWS = (ceil(N/32)*16, ceil(M/128)*16, 1)
// SLM: A_slm[2][128*32] = 8192 halfs, B_slm[2][32*32] = 2048 halfs => total ~20KB

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 128
#define TILE_N 32
#define TILE_K 32
#define THREAD_M 8
#define THREAD_N 2
#define WG_X 16
#define WG_Y 16
#define NUM_THREADS 256

__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int lx = get_local_id(0);  // 0..15 -> col dim
    const int ly = get_local_id(1);  // 0..15 -> row dim

    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    const int flat_id = ly * WG_X + lx;  // 0..255

    // Each thread computes rows [ly*8 .. ly*8+7] x cols [lx*2, lx*2+1]
    const int row_base = ly * THREAD_M;
    const int col_base = lx * THREAD_N;

    // Accumulators: 8 rows x 2 cols
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            acc[i][j] = 0.0f;

    // Double-buffered SLM
    __local half A_slm[2][TILE_M * TILE_K];  // 128 x 32 = 4096 halfs each
    __local half B_slm[2][TILE_K * TILE_N];  // 32 x 32 = 1024 halfs each

    // A: 128*32 = 4096 halfs, 256 threads => 16 each
    // B: 32*32 = 1024 halfs, 256 threads => 4 each

    // --- Load first tile (buf 0) ---
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int idx = flat_id + i * NUM_THREADS;
        int r = idx / TILE_K;  // row in tile
        int c = idx % TILE_K;  // col in tile (k dim)
        int gm = wg_m + r;
        int gk = c;  // k0=0
        A_slm[0][idx] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = flat_id + i * NUM_THREADS;
        if (idx < TILE_K * TILE_N) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gk = r;  // k0=0
            int gn = wg_n + c;
            B_slm[0][idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        int next_k = k0 + TILE_K;
        int next_buf = 1 - buf;

        // Prefetch next tile into next_buf
        if (next_k < K) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx / TILE_K;
                int c = idx % TILE_K;
                int gm = wg_m + r;
                int gk = next_k + c;
                A_slm[next_buf][idx] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = flat_id + i * NUM_THREADS;
                if (idx < TILE_K * TILE_N) {
                    int r = idx / TILE_N;
                    int c = idx % TILE_N;
                    int gk = next_k + r;
                    int gn = wg_n + c;
                    B_slm[next_buf][idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
                }
            }
        }

        // Compute on current buffer
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            // Load B values for this thread's 2 cols
            float b0 = convert_float(B_slm[buf][kk * TILE_N + col_base]);
            float b1 = convert_float(B_slm[buf][kk * TILE_N + col_base + 1]);

            // Load A values and accumulate
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                float a_val = convert_float(A_slm[buf][(row_base + r) * TILE_K + kk]);
                acc[r][0] = fma(a_val, b0, acc[r][0]);
                acc[r][1] = fma(a_val, b1, acc[r][1]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Store results
    #pragma unroll
    for (int r = 0; r < THREAD_M; r++) {
        int out_row = wg_m + row_base + r;
        if (out_row < M) {
            #pragma unroll
            for (int c = 0; c < THREAD_N; c++) {
                int out_col = wg_n + col_base + c;
                if (out_col < N) {
                    C[out_row * N + out_col] = convert_half(acc[r][c]);
                }
            }
        }
    }
}
```

