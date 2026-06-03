

## Analysis

### Version Comparison
- **Version 1** (48.2ms): TILE_M=128, TILE_N=16, TILE_K=16. Simple SLM tiling with interior/edge splitting. Good but small K tile.
- **Best** (43.3ms): TILE_M=128, TILE_N=16, TILE_K=32. Larger K tile reduces barrier overhead. Separate interior checks for M and N dimensions.
- **Last attempt** (91.0ms): TILE_M=128, TILE_N=64, TILE_K=16. Tried wider N tile with THREAD_N=4, but the fake double-buffering added overhead without benefit, and increased SLM/register pressure hurt occupancy.

### Key Insights
1. The 128x16 tile with TILE_K=32 works best - it's well-matched to the 16-wide subgroups
2. Double buffering without async copies is counterproductive
3. Wider TILE_N=64 caused too much SLM pressure and bank conflicts
4. The best version already has good structure; we should optimize the inner compute loop

### Proposed Improvements
1. **Increase TILE_K to 64** to reduce barrier frequency further (amortize load cost)
2. **Vectorized SLM loads** using vload4/vload8 for coalesced global memory access
3. **Vectorized B loads from SLM** - load as half and convert in bulk
4. **Remove SLM padding for A** and instead use a power-of-2 friendly layout with careful indexing
5. **Prefetch with __builtin_prefetch-style hints** where possible
6. **Unroll the K loop more aggressively** with register-cached A fragments

```OCL
// FP16 GEMM with SLM tiling and optimized inner loop
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: TILE_M=128 x TILE_N=16, TILE_K=64
// THREAD_M=8, THREAD_N=1
// LWS = (16, 16, 1) = 256 work-items
// 16 threads along x cover 16 cols (subgroup width)
// 16 threads along y cover 128 rows (8 each)
// GWS = (ceil(N/16)*16, ceil(M/128)*16, 1)
// SLM: A_slm[128*64] = 8192 halfs (16KB), B_slm[64*16] = 1024 halfs (2KB) => 18KB total

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 128
#define TILE_N 16
#define TILE_K 64
#define THREAD_M 8
#define A_STRIDE (TILE_K)
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
    const int lx = get_local_id(0);  // 0..15 -> column within tile
    const int ly = get_local_id(1);  // 0..15 -> row group

    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    const int out_col = wg_n + lx;
    const int row_base = wg_m + ly * THREAD_M;

    // Accumulators: 8 rows
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    // SLM tiles
    __local half A_slm[TILE_M * TILE_K];   // 128 x 64 = 8192 halfs
    __local half B_slm[TILE_K * TILE_N];   // 64 x 16 = 1024 halfs

    const int flat_id = ly * 16 + lx;  // 0..255

    const int interior_m = (wg_m + TILE_M <= M);
    const int interior_n = (wg_n + TILE_N <= N);

    // A: 128*64 = 8192 elements / 256 threads = 32 each
    // B: 64*16 = 1024 elements / 256 threads = 4 each

    const int K_main = (K / TILE_K) * TILE_K;

    // ========== MAIN LOOP ==========
    for (int k0 = 0; k0 < K_main; k0 += TILE_K) {

        // --- Load A_slm[128][64] ---
        if (interior_m) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 6;      // idx / 64
                int c = idx & 0x3F;    // idx % 64
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
                int r = idx >> 4;      // idx / 16
                int c = idx & 0xF;     // idx % 16
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

        // --- Compute ---
        // Base pointer for this thread's A rows
        const int a_base = ly * THREAD_M * TILE_K;

        // Process in chunks of 4 for better register reuse
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 4) {
            // Load 4 B values
            float b0 = convert_float(B_slm[(kk + 0) * TILE_N + lx]);
            float b1 = convert_float(B_slm[(kk + 1) * TILE_N + lx]);
            float b2 = convert_float(B_slm[(kk + 2) * TILE_N + lx]);
            float b3 = convert_float(B_slm[(kk + 3) * TILE_N + lx]);

            // For each row, load 4 A values and FMA
            #define DO_ROW(R, ACC) { \
                int aoff = a_base + (R) * TILE_K + kk; \
                float a0 = convert_float(A_slm[aoff + 0]); \
                float a1 = convert_float(A_slm[aoff + 1]); \
                float a2 = convert_float(A_slm[aoff + 2]); \
                float a3 = convert_float(A_slm[aoff + 3]); \
                ACC = fma(a0, b0, ACC); \
                ACC = fma(a1, b1, ACC); \
                ACC = fma(a2, b2, ACC); \
                ACC = fma(a3, b3, ACC); \
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
        for (int kk = 0; kk < TILE_K; kk += 4) {
            float b0 = convert_float(B_slm[(kk + 0) * TILE_N + lx]);
            float b1 = convert_float(B_slm[(kk + 1) * TILE_N + lx]);
            float b2 = convert_float(B_slm[(kk + 2) * TILE_N + lx]);
            float b3 = convert_float(B_slm[(kk + 3) * TILE_N + lx]);

            #define DO_ROW(R, ACC) { \
                int aoff = a_base + (R) * TILE_K + kk; \
                float a0 = convert_float(A_slm[aoff + 0]); \
                float a1 = convert_float(A_slm[aoff + 1]); \
                float a2 = convert_float(A_slm[aoff + 2]); \
                float a3 = convert_float(A_slm[aoff + 3]); \
                ACC = fma(a0, b0, ACC); \
                ACC = fma(a1, b1, ACC); \
                ACC = fma(a2, b2, ACC); \
                ACC = fma(a3, b3, ACC); \
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
                if (out_row < M) {
                    C[out_row * N + out_col] = convert_half(accs[r]);
                }
            }
        }
    }
}
```

