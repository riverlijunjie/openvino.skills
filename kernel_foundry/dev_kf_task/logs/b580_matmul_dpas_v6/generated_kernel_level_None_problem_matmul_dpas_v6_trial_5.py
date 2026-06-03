

## Analysis

Looking at the results:
- **Best kernel (47ms)**: Simple 128×16 tile, THREAD_M=8, THREAD_N=1, scalar SLM access
- **Version 1 (57ms)**: 128×32 tile, THREAD_M=8, THREAD_N=2, uses stride-16 column mapping
- **Last kernel (71ms)**: 128×32 tile, THREAD_M=8, THREAD_N=2, vectorized loads but complex branching

The simpler 128×16 kernel wins because:
1. Smaller tile = less SLM pressure, better occupancy
2. Simpler access patterns = fewer bank conflicts
3. Less branching overhead

**Key improvements to try:**
1. **Eliminate per-iteration bounds checks** by splitting into main loop (full K tiles) and tail
2. **Preload A values per kk** into registers before the r-loop to reduce repeated SLM reads
3. **Use `fma`** instead of `+= a * b`
4. **Pad SLM** to reduce bank conflicts (TILE_K+1 padding for A)
5. **Vectorized SLM loads** in compute phase using half8 for A rows
6. Keep the winning 128×16 tile configuration

## Improved OCL code

```OCL
// FP16 GEMM with SLM tiling and register blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: TILE_M=128 x TILE_N=16, TILE_K=32
// THREAD_M=8, THREAD_N=1
// LWS = (16, 16, 1) = 256 work-items
// 16 threads along x cover 16 cols
// 16 threads along y cover 128 rows (8 each)
// GWS = (ceil(N/16)*16, ceil(M/128)*16, 1)
// SLM: A_slm[128*(32+1)] padded, B_slm[32*16]

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 128
#define TILE_N 16
#define TILE_K 32
#define THREAD_M 8
#define A_STRIDE (TILE_K + 1)  // pad to avoid bank conflicts

__attribute__((reqd_work_group_size(16, 16, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
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

    // SLM tiles - A padded to avoid bank conflicts
    __local half A_slm[TILE_M * A_STRIDE];  // 128 x 33
    __local half B_slm[TILE_K * TILE_N];    // 32 x 16

    const int flat_id = ly * 16 + lx;  // 0..255

    // Precompute whether this WG is fully interior in M and N
    const int interior_m = (wg_m + TILE_M <= M);
    const int interior_n = (wg_n + TILE_N <= N);

    // Total elements: A = 128*32 = 4096, each thread loads 16
    //                 B = 32*16 = 512, each thread loads 2

    const int K_main = (K / TILE_K) * TILE_K;

    // ========== MAIN LOOP (full K tiles) ==========
    for (int k0 = 0; k0 < K_main; k0 += TILE_K) {
        // --- Load A_slm[128][32] with padding stride 33 ---
        // 4096 elements / 256 threads = 16 each
        if (interior_m) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 256;
                int r = idx >> 5;      // idx / 32
                int c = idx & 0x1F;    // idx % 32
                A_slm[r * A_STRIDE + c] = A[(wg_m + r) * K + k0 + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 256;
                int r = idx >> 5;
                int c = idx & 0x1F;
                int gm = wg_m + r;
                A_slm[r * A_STRIDE + c] = (gm < M) ? A[gm * K + k0 + c] : (half)0.0h;
            }
        }

        // --- Load B_slm[32][16] = 512 halfs, 256 threads => 2 each ---
        if (interior_n) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int idx = flat_id + i * 256;
                int r = idx >> 4;      // idx / 16
                int c = idx & 0xF;     // idx % 16
                B_slm[idx] = B[(k0 + r) * N + wg_n + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int idx = flat_id + i * 256;
                int r = idx >> 4;
                int c = idx & 0xF;
                int gn = wg_n + c;
                B_slm[idx] = (gn < N) ? B[(k0 + r) * N + gn] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Compute: preload A fragment per kk, then FMA ---
        const int a_row_offset = ly * THREAD_M;

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float b_val = convert_float(B_slm[kk * TILE_N + lx]);

            // Preload all 8 A values for this kk
            float a_frag[THREAD_M];
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                a_frag[r] = convert_float(A_slm[(a_row_offset + r) * A_STRIDE + kk]);
            }

            // Pure FMA accumulation
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                acc[r] = fma(a_frag[r], b_val, acc[r]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ========== TAIL K TILE ==========
    if (K_main < K) {
        int k0 = K_main;
        int k_rem = K - K_main;

        // Load A - with K bounds check
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int idx = flat_id + i * 256;
            int r = idx >> 5;
            int c = idx & 0x1F;
            int gm = wg_m + r;
            int gk = k0 + c;
            A_slm[r * A_STRIDE + c] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
        }

        // Load B - with K bounds check
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int idx = flat_id + i * 256;
            int r = idx >> 4;
            int c = idx & 0xF;
            int gk = k0 + r;
            int gn = wg_n + c;
            B_slm[idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int a_row_offset = ly * THREAD_M;

        // Only iterate over valid K range (but zero-padded, so full TILE_K is safe)
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float b_val = convert_float(B_slm[kk * TILE_N + lx]);
            float a_frag[THREAD_M];
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                a_frag[r] = convert_float(A_slm[(a_row_offset + r) * A_STRIDE + kk]);
            }
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                acc[r] = fma(a_frag[r], b_val, acc[r]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ========== STORE RESULTS ==========
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

