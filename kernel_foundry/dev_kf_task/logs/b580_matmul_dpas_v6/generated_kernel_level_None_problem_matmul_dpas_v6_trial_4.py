

## Analysis

Looking at the results:
- **Version 1** (64x64 tile, 4x4 per thread): 62.3ms
- **Best so far** (128x16 tile, 8x1 per thread): 47.0ms
- **Last attempt** (128x32 tile, 8x2 per thread, double-buffered): 66.9ms

The best kernel wins because:
1. Simple single-buffered SLM with small B tile (16x16 = 256 halfs) fits well in SLM
2. High row reuse (8 rows per thread) with single column means good A reuse
3. Less SLM pressure and fewer barriers

The double-buffered version was slower because the extra SLM (2x buffers) reduced occupancy and the prefetch wasn't truly overlapped.

**Strategy for improvement:**
- Keep the 128-row tile structure that worked well
- Increase TILE_N to 32 with THREAD_N=2 for better compute density per SLM load
- Use vectorized loads (vload2/vload4) to reduce instruction count
- Use `intel_reqd_sub_group_size(16)` for subgroup-friendly execution
- Single-buffered SLM (simpler, less SLM pressure)
- Split main loop from tail to remove bounds checks in hot path
- Use `fma` for better throughput

## Improved OCL code

```OCL
// FP16 GEMM with SLM tiling and register blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: TILE_M=128 x TILE_N=32, TILE_K=16
// Each work-item computes THREAD_M=8 rows x THREAD_N=2 cols = 16 outputs
// LWS = (16, 16, 1) = 256 work-items
// 16 threads along x cover 32 cols (2 each: lx, lx+16)
// 16 threads along y cover 128 rows (8 each: ly*8..ly*8+7)
// GWS = (ceil(N/32)*16, ceil(M/128)*16, 1)
// SLM: A_slm[128*16] = 2048 halfs (4KB), B_slm[16*32] = 512 halfs (1KB) => ~5KB total

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 128
#define TILE_N 32
#define TILE_K 16
#define THREAD_M 8
#define THREAD_N 2
#define WG_X 16
#define WG_Y 16
#define NUM_THREADS 256

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
    const int lx = get_local_id(0);  // 0..15
    const int ly = get_local_id(1);  // 0..15

    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    const int flat_id = ly * WG_X + lx;  // 0..255

    // Thread output mapping:
    // rows: ly*8 .. ly*8+7 (8 consecutive rows)
    // cols: lx, lx+16 (2 cols, stride 16)
    const int row_base = ly * THREAD_M;
    const int col0 = lx;       // first output col within tile
    const int col1 = lx + 16;  // second output col within tile

    // Accumulators: 8 rows x 2 cols
    float acc0[THREAD_M];  // col0
    float acc1[THREAD_M];  // col1
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        acc0[i] = 0.0f;
        acc1[i] = 0.0f;
    }

    // SLM tiles - single buffered
    __local half A_slm[TILE_M * TILE_K];  // 128 x 16 = 2048 halfs
    __local half B_slm[TILE_K * TILE_N];  // 16 x 32 = 512 halfs

    // A: 2048 halfs / 256 threads = 8 each
    // B: 512 halfs / 256 threads = 2 each

    // Precompute load indices for A (each thread loads 8 elements)
    // flat_id maps to positions: flat_id, flat_id+256, ..., flat_id+7*256
    // But 128*16=2048, so 2048/256=8 loads per thread

    // Determine if this WG is fully interior (no bounds checks needed)
    const int wg_m_end = wg_m + TILE_M;
    const int wg_n_end = wg_n + TILE_N;
    const int interior_mn = (wg_m_end <= M) && (wg_n_end <= N);

    // Main K loop - split into aligned part and tail
    const int K_main = (K / TILE_K) * TILE_K;

    // ========== MAIN LOOP (K-aligned tiles) ==========
    for (int k0 = 0; k0 < K_main; k0 += TILE_K) {
        // --- Load A_slm[128][16] ---
        if (interior_mn && 1) {
            // No M bounds check needed, still need K check only for tail (but k0 < K_main so ok)
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 4;    // idx / 16
                int c = idx & 0xF;   // idx % 16
                A_slm[idx] = A[(wg_m + r) * K + k0 + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 4;
                int c = idx & 0xF;
                int gm = wg_m + r;
                int gk = k0 + c;
                A_slm[idx] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
            }
        }

        // --- Load B_slm[16][32] ---
        if (interior_mn && 1) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 5;    // idx / 32
                int c = idx & 0x1F;  // idx % 32
                B_slm[idx] = B[(k0 + r) * N + wg_n + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int idx = flat_id + i * NUM_THREADS;
                if (idx < TILE_K * TILE_N) {
                    int r = idx >> 5;
                    int c = idx & 0x1F;
                    int gk = k0 + r;
                    int gn = wg_n + c;
                    B_slm[idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Compute ---
        // For each kk step, load 2 B values and 8 A values, do 16 FMAs
        __local const half* a_base = A_slm + row_base * TILE_K;

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            // B values for two columns
            float bv0 = convert_float(B_slm[kk * TILE_N + col0]);
            float bv1 = convert_float(B_slm[kk * TILE_N + col1]);

            // A values for 8 rows - use pointer arithmetic
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                float a_val = convert_float(a_base[r * TILE_K + kk]);
                acc0[r] = fma(a_val, bv0, acc0[r]);
                acc1[r] = fma(a_val, bv1, acc1[r]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ========== TAIL K TILE ==========
    if (K_main < K) {
        int k0 = K_main;
        int k_rem = K - K_main;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = flat_id + i * NUM_THREADS;
            int r = idx >> 4;
            int c = idx & 0xF;
            int gm = wg_m + r;
            int gk = k0 + c;
            A_slm[idx] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = flat_id + i * NUM_THREADS;
            if (idx < TILE_K * TILE_N) {
                int r = idx >> 5;
                int c = idx & 0x1F;
                int gk = k0 + r;
                int gn = wg_n + c;
                B_slm[idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local const half* a_base = A_slm + row_base * TILE_K;

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float bv0 = convert_float(B_slm[kk * TILE_N + col0]);
            float bv1 = convert_float(B_slm[kk * TILE_N + col1]);
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                float a_val = convert_float(a_base[r * TILE_K + kk]);
                acc0[r] = fma(a_val, bv0, acc0[r]);
                acc1[r] = fma(a_val, bv1, acc1[r]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ========== STORE RESULTS ==========
    // col0 = wg_n + lx, col1 = wg_n + lx + 16
    int out_col0 = wg_n + col0;
    int out_col1 = wg_n + col1;

    if (interior_mn) {
        // No bounds checks needed
        #pragma unroll
        for (int r = 0; r < THREAD_M; r++) {
            int out_row = wg_m + row_base + r;
            C[out_row * N + out_col0] = convert_half(acc0[r]);
            C[out_row * N + out_col1] = convert_half(acc1[r]);
        }
    } else {
        #pragma unroll
        for (int r = 0; r < THREAD_M; r++) {
            int out_row = wg_m + row_base + r;
            if (out_row < M) {
                if (out_col0 < N)
                    C[out_row * N + out_col0] = convert_half(acc0[r]);
                if (out_col1 < N)
                    C[out_row * N + out_col1] = convert_half(acc1[r]);
            }
        }
    }
}
```

