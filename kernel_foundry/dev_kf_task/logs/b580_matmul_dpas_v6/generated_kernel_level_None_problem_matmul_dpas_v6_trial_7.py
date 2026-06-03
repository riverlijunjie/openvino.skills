

## Analysis

Looking at the previous versions:

1. **Version 1 (36.2ms)** - Best performer. Uses TILE_M=128, TILE_N=16, TILE_K=64 with SLM tiling, interior/boundary split, and unrolled k-loop with groups of 4. Large K-tile amortizes barrier cost.

2. **Version 2 (42.7ms)** - Double-buffered but with TILE_K=32. The double buffering has a bug: it overlaps prefetch with compute on the *current* buffer, but writes to the next buffer happen concurrently with reads from the current buffer without proper synchronization.

3. **Best so far (47.0ms)** - Simple TILE_K=16, no interior/boundary split, division-based indexing (slow).

4. **Last tested (212ms)** - TILE_N=32 with padded SLM. The padding and wider N-tile caused more SLM pressure and slower performance.

**Key insights:**
- Version 1 is fastest: large TILE_K=64, interior/boundary fast path, bit-shift indexing, unrolled k-loop with 4-wide FMA
- TILE_N=16 matches subgroup size well on Intel
- Larger TILE_K reduces barrier overhead ratio
- Interior checks help avoid per-element bounds checks

**Strategy for improvement:**
- Start from Version 1's structure (proven fastest)
- Use TILE_K=64 with vectorized SLM loads (vload4/vload8) for A
- Use `intel_reqd_sub_group_size(16)` 
- Vectorize the inner compute loop further
- Use `fma` consistently
- Try to reduce SLM reads by loading A into registers as vectors

```OCL
// FP16 GEMM with SLM tiling, vectorized loads, interior fast-path
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: TILE_M=128 x TILE_N=16, TILE_K=64
// THREAD_M=8, each thread computes 8 rows x 1 col
// LWS = (16, 16, 1) = 256 work-items
// GWS = (ceil(N/16)*16, ceil(M/128)*16, 1)
// SLM: A_slm[128*64]=16KB, B_slm[64*16]=2KB => 18KB total

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 128
#define TILE_N 16
#define TILE_K 64
#define THREAD_M 8
#define NUM_THREADS 256

__attribute__((reqd_work_group_size(16, 16, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lx = get_local_id(0);  // 0..15
    const int ly = get_local_id(1);  // 0..15

    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    const int out_col = wg_n + lx;
    const int row_base = wg_m + ly * THREAD_M;
    const int flat_id = ly * 16 + lx;

    // 8 accumulators
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    __local half A_slm[TILE_M * TILE_K];  // 128*64 = 8192 halfs
    __local half B_slm[TILE_K * TILE_N];  // 64*16  = 1024 halfs

    const int interior_m = (wg_m + TILE_M <= M);
    const int interior_n = (wg_n + TILE_N <= N);
    const int K_main = (K / TILE_K) * TILE_K;

    // A: 8192 / 256 = 32 elements each
    // B: 1024 / 256 = 4 elements each

    // Precompute A load row/col for this thread
    // Each thread loads 32 elements at positions flat_id + i*256

    for (int k0 = 0; k0 < K_main; k0 += TILE_K) {

        // --- Load A_slm[128][64] ---
        if (interior_m) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 6;
                int c = idx & 0x3F;
                A_slm[idx] = A[(wg_m + r) * K + k0 + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 6;
                int c = idx & 0x3F;
                int gm = wg_m + r;
                A_slm[idx] = (gm < M) ? A[gm * K + k0 + c] : (half)0.0h;
            }
        }

        // --- Load B_slm[64][16] ---
        if (interior_n) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 4;
                int c = idx & 0xF;
                B_slm[idx] = B[(k0 + r) * N + wg_n + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 4;
                int c = idx & 0xF;
                int gn = wg_n + c;
                B_slm[idx] = (gn < N) ? B[(k0 + r) * N + gn] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Compute: unroll k by 8, vectorize A loads ---
        const int a_base = ly * THREAD_M * TILE_K;

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 8) {
            // Load 8 B values for this column
            float b0 = convert_float(B_slm[(kk + 0) * TILE_N + lx]);
            float b1 = convert_float(B_slm[(kk + 1) * TILE_N + lx]);
            float b2 = convert_float(B_slm[(kk + 2) * TILE_N + lx]);
            float b3 = convert_float(B_slm[(kk + 3) * TILE_N + lx]);
            float b4 = convert_float(B_slm[(kk + 4) * TILE_N + lx]);
            float b5 = convert_float(B_slm[(kk + 5) * TILE_N + lx]);
            float b6 = convert_float(B_slm[(kk + 6) * TILE_N + lx]);
            float b7 = convert_float(B_slm[(kk + 7) * TILE_N + lx]);

            #define DO_ROW(R, ACC) { \
                int aoff = a_base + (R) * TILE_K + kk; \
                ACC = fma(convert_float(A_slm[aoff + 0]), b0, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 1]), b1, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 2]), b2, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 3]), b3, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 4]), b4, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 5]), b5, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 6]), b6, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 7]), b7, ACC); \
            }

            DO_ROW(0, acc0);
            DO_ROW(1, acc1);
            DO_ROW(2, acc2);
            DO_ROW(3, acc3);
            DO_ROW(4, acc4);
            DO_ROW(5, acc5);
            DO_ROW(6, acc6);
            DO_ROW(7, acc7);

            #undef DO_ROW
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ========== TAIL K TILE ==========
    if (K_main < K) {
        int k0 = K_main;
        int k_rem = K - K_main;

        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int idx = flat_id + i * NUM_THREADS;
            int r = idx >> 6;
            int c = idx & 0x3F;
            int gm = wg_m + r;
            int gk = k0 + c;
            A_slm[idx] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
        }

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = flat_id + i * NUM_THREADS;
            int r = idx >> 4;
            int c = idx & 0xF;
            int gk = k0 + r;
            int gn = wg_n + c;
            B_slm[idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int a_base = ly * THREAD_M * TILE_K;

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 8) {
            float b0 = convert_float(B_slm[(kk + 0) * TILE_N + lx]);
            float b1 = convert_float(B_slm[(kk + 1) * TILE_N + lx]);
            float b2 = convert_float(B_slm[(kk + 2) * TILE_N + lx]);
            float b3 = convert_float(B_slm[(kk + 3) * TILE_N + lx]);
            float b4 = convert_float(B_slm[(kk + 4) * TILE_N + lx]);
            float b5 = convert_float(B_slm[(kk + 5) * TILE_N + lx]);
            float b6 = convert_float(B_slm[(kk + 6) * TILE_N + lx]);
            float b7 = convert_float(B_slm[(kk + 7) * TILE_N + lx]);

            #define DO_ROW(R, ACC) { \
                int aoff = a_base + (R) * TILE_K + kk; \
                ACC = fma(convert_float(A_slm[aoff + 0]), b0, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 1]), b1, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 2]), b2, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 3]), b3, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 4]), b4, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 5]), b5, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 6]), b6, ACC); \
                ACC = fma(convert_float(A_slm[aoff + 7]), b7, ACC); \
            }

            DO_ROW(0, acc0);
            DO_ROW(1, acc1);
            DO_ROW(2, acc2);
            DO_ROW(3, acc3);
            DO_ROW(4, acc4);
            DO_ROW(5, acc5);
            DO_ROW(6, acc6);
            DO_ROW(7, acc7);

            #undef DO_ROW
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ========== STORE RESULTS ==========
    if (out_col < N) {
        if (interior_m) {
            C[(row_base + 0) * N + out_col] = convert_half(acc0);
            C[(row_base + 1) * N + out_col] = convert_half(acc1);
            C[(row_base + 2) * N + out_col] = convert_half(acc2);
            C[(row_base + 3) * N + out_col] = convert_half(acc3);
            C[(row_base + 4) * N + out_col] = convert_half(acc4);
            C[(row_base + 5) * N + out_col] = convert_half(acc5);
            C[(row_base + 6) * N + out_col] = convert_half(acc6);
            C[(row_base + 7) * N + out_col] = convert_half(acc7);
        } else {
            float accs[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                int out_row = row_base + r;
                if (out_row < M)
                    C[out_row * N + out_col] = convert_half(accs[r]);
            }
        }
    }
}
```

