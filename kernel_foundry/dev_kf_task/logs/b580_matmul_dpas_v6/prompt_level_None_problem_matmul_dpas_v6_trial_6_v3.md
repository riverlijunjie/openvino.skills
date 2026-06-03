

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Optimization goals:
- Must use Intel OpenCL DPAS instruction, e.g. intel_sub_group_f16_f16_matrix_mad_k16.
- Improve FLOPS and hide memory latency with tiling + subgroup-friendly mapping.
- Provide explicit launch metadata (GWS/LWS/subgroup hints) in kernel comments.

## Reference code / Task:

This is the reference OCL implementation:
```
// Simple row-major FP16 matmul:
//   C[M,N] = A[M,K] x B[K,N]
// Input/Output dtype: half
// Accumulation dtype: float
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += convert_float(A[row * K + k]) * convert_float(B[k * N + col]);
    }

    C[row * N + col] = convert_half(acc);
}

```

## Previous OCL implementations with scores:

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 48.200):
```OCL
// FP16 GEMM with SLM tiling, bank-conflict avoidance, interior/edge splitting
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: TILE_M=128 rows x TILE_N=16 cols, TILE_K=16
// Each work-item computes THREAD_M=8 rows x 1 col
// LWS = (16, 16, 1) = 256 work-items
// GWS = (ceil(N/16)*16, ceil(M/128)*16, 1)
//
// SLM: A_slm[128*(16+1)] padded to avoid bank conflicts = 2176 halfs (~4.25KB)
//       B_slm[16*16] = 256 halfs (0.5KB)
// Total SLM ~5KB

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 128
#define TILE_N 16
#define TILE_K 16
#define THREAD_M 8
#define A_STRIDE (TILE_K + 1)  // pad by 1 to avoid SLM bank conflicts
#define NUM_THREADS 256

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

    // Accumulators: 8 rows x 1 col
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    // SLM tiles - A is padded to avoid bank conflicts
    __local half A_slm[TILE_M * A_STRIDE];  // 128 x 17
    __local half B_slm[TILE_K * TILE_N];    // 16 x 16

    const int flat_id = ly * 16 + lx;  // 0..255

    // Determine if this WG is fully interior (no bounds checks needed for M and N)
    const int wg_interior_mn = (wg_m + TILE_M <= M) && (wg_n + TILE_N <= N);

    // Aligned K tiles
    const int K_aligned = (K / TILE_K) * TILE_K;

    // ========== MAIN LOOP (interior K tiles) ==========
    if (wg_interior_mn) {
        // Fast path: no M/N bounds checks
        for (int k0 = 0; k0 < K_aligned; k0 += TILE_K) {
            // Load A_slm[128][16] -> stored as [128][17] with padding
            // 2048 halfs, 256 threads => 8 each
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 4;     // idx / 16
                int c = idx & 0xF;    // idx % 16
                A_slm[r * A_STRIDE + c] = A[(wg_m + r) * K + k0 + c];
            }

            // Load B_slm[16][16] = 256 halfs, 256 threads => 1 each
            {
                int r = flat_id >> 4;     // flat_id / 16
                int c = flat_id & 0xF;    // flat_id % 16
                B_slm[flat_id] = B[(k0 + r) * N + wg_n + c];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Compute: unrolled inner loop
            __local const half* a_ptr = A_slm + ly * THREAD_M * A_STRIDE;

            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk++) {
                float b_val = convert_float(B_slm[kk * TILE_N + lx]);

                float a0 = convert_float(a_ptr[0 * A_STRIDE + kk]);
                float a1 = convert_float(a_ptr[1 * A_STRIDE + kk]);
                float a2 = convert_float(a_ptr[2 * A_STRIDE + kk]);
                float a3 = convert_float(a_ptr[3 * A_STRIDE + kk]);
                float a4 = convert_float(a_ptr[4 * A_STRIDE + kk]);
                float a5 = convert_float(a_ptr[5 * A_STRIDE + kk]);
                float a6 = convert_float(a_ptr[6 * A_STRIDE + kk]);
                float a7 = convert_float(a_ptr[7 * A_STRIDE + kk]);

                acc0 = fma(a0, b_val, acc0);
                acc1 = fma(a1, b_val, acc1);
                acc2 = fma(a2, b_val, acc2);
                acc3 = fma(a3, b_val, acc3);
                acc4 = fma(a4, b_val, acc4);
                acc5 = fma(a5, b_val, acc5);
                acc6 = fma(a6, b_val, acc6);
                acc7 = fma(a7, b_val, acc7);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // K tail (still interior for M/N)
        if (K_aligned < K) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 4;
                int c = idx & 0xF;
                int gk = K_aligned + c;
                A_slm[r * A_STRIDE + c] = (gk < K) ? A[(wg_m + r) * K + gk] : (half)0.0h;
            }

            {
                int r = flat_id >> 4;
                int c = flat_id & 0xF;
                int gk = K_aligned + r;
                B_slm[flat_id] = (gk < K) ? B[gk * N + wg_n + c] : (half)0.0h;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            __local const half* a_ptr = A_slm + ly * THREAD_M * A_STRIDE;

            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk++) {
                float b_val = convert_float(B_slm[kk * TILE_N + lx]);
                acc0 = fma(convert_float(a_ptr[0 * A_STRIDE + kk]), b_val, acc0);
                acc1 = fma(convert_float(a_ptr[1 * A_STRIDE + kk]), b_val, acc1);
                acc2 = fma(convert_float(a_ptr[2 * A_STRIDE + kk]), b_val, acc2);
                acc3 = fma(convert_float(a_ptr[3 * A_STRIDE + kk]), b_val, acc3);
                acc4 = fma(convert_float(a_ptr[4 * A_STRIDE + kk]), b_val, acc4);
                acc5 = fma(convert_float(a_ptr[5 * A_STRIDE + kk]), b_val, acc5);
                acc6 = fma(convert_float(a_ptr[6 * A_STRIDE + kk]), b_val, acc6);
                acc7 = fma(convert_float(a_ptr[7 * A_STRIDE + kk]), b_val, acc7);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store - no bounds checks
        #pragma unroll
        for (int r = 0; r < THREAD_M; r++) {
            float val;
            switch(r) {
                case 0: val = acc0; break;
                case 1: val = acc1; break;
                case 2: val = acc2; break;
                case 3: val = acc3; break;
                case 4: val = acc4; break;
                case 5: val = acc5; break;
                case 6: val = acc6; break;
                case 7: val = acc7; break;
            }
            C[(row_base + r) * N + out_col] = convert_half(val);
        }

    } else {
        // Edge path: full bounds checking
        for (int k0 = 0; k0 < K; k0 += TILE_K) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 4;
                int c = idx & 0xF;
                int gm = wg_m + r;
                int gk = k0 + c;
                A_slm[r * A_STRIDE + c] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
            }

            {
                int r = flat_id >> 4;
                int c = flat_id & 0xF;
                int gk = k0 + r;
                int gn = wg_n + c;
                B_slm[flat_id] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            __local const half* a_ptr = A_slm + ly * THREAD_M * A_STRIDE;

            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk++) {
                float b_val = convert_float(B_slm[kk * TILE_N + lx]);
                acc0 = fma(convert_float(a_ptr[0 * A_STRIDE + kk]), b_val, acc0);
                acc1 = fma(convert_float(a_ptr[1 * A_STRIDE + kk]), b_val, acc1);
                acc2 = fma(convert_float(a_ptr[2 * A_STRIDE + kk]), b_val, acc2);
                acc3 = fma(convert_float(a_ptr[3 * A_STRIDE + kk]), b_val, acc3);
                acc4 = fma(convert_float(a_ptr[4 * A_STRIDE + kk]), b_val, acc4);
                acc5 = fma(convert_float(a_ptr[5 * A_STRIDE + kk]), b_val, acc5);
                acc6 = fma(convert_float(a_ptr[6 * A_STRIDE + kk]), b_val, acc6);
                acc7 = fma(convert_float(a_ptr[7 * A_STRIDE + kk]), b_val, acc7);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store with bounds checks
        if (out_col < N) {
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

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 43.300):
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 91.000):
```OCL
// FP16 GEMM: C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: TILE_M=128 x TILE_N=64, TILE_K=16
// Each work-item: THREAD_M=8 rows x THREAD_N=4 cols = 32 outputs
// LWS = (16, 16, 1) = 256 work-items
// 16 threads along x cover 64 cols (4 each: lx*4..lx*4+3)
// 16 threads along y cover 128 rows (8 each: ly*8..ly*8+7)
// GWS = (ceil(N/64)*16, ceil(M/128)*16, 1)
// SLM: A_slm[128*16]*2 = 4096 halfs (8KB), B_slm[16*64]*2 = 2048 halfs (4KB) => 12KB total (double buffered)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 128
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 8
#define THREAD_N 4
#define WG_X 16
#define WG_Y 16
#define NUM_THREADS 256

// Double buffer size
#define A_TILE_SIZE (TILE_M * TILE_K)   // 2048
#define B_TILE_SIZE (TILE_K * TILE_N)   // 1024

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
    const int lx = get_local_id(0);  // 0..15
    const int ly = get_local_id(1);  // 0..15

    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    const int flat_id = ly * WG_X + lx;

    // Thread output mapping
    const int row_base = ly * THREAD_M;  // ly*8
    const int col_base = lx * THREAD_N;  // lx*4

    // Accumulators: 8 rows x 4 cols = 32 floats
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            acc[i][j] = 0.0f;

    // Double-buffered SLM
    __local half A_slm[2][A_TILE_SIZE];
    __local half B_slm[2][B_TILE_SIZE];

    // Check if this WG is fully interior
    const int interior_m = (wg_m + TILE_M <= M);
    const int interior_n = (wg_n + TILE_N <= N);

    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    int buf = 0;

    // ===== Preload first tile into buf=0 =====
    {
        int k0 = 0;
        // Load A: 2048 halfs / 256 threads = 8 each
        if (interior_m && k0 + TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 4;    // /16
                int c = idx & 0xF;   // %16
                A_slm[0][idx] = A[(wg_m + r) * K + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 4;
                int c = idx & 0xF;
                int gm = wg_m + r;
                int gk = k0 + c;
                A_slm[0][idx] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
            }
        }

        // Load B: 1024 halfs / 256 threads = 4 each
        if (interior_n && k0 + TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 6;    // /64
                int c = idx & 0x3F;  // %64
                B_slm[0][idx] = B[(k0 + r) * N + wg_n + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = flat_id + i * NUM_THREADS;
                if (idx < B_TILE_SIZE) {
                    int r = idx >> 6;
                    int c = idx & 0x3F;
                    int gk = k0 + r;
                    int gn = wg_n + c;
                    B_slm[0][idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
                }
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // ===== Main loop with double buffering =====
    for (int t = 0; t < num_k_tiles; t++) {
        int cur = t & 1;
        int nxt = 1 - cur;
        int k_next = (t + 1) * TILE_K;

        // Prefetch next tile into nxt buffer (if there is one)
        if (t + 1 < num_k_tiles) {
            if (interior_m && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int idx = flat_id + i * NUM_THREADS;
                    int r = idx >> 4;
                    int c = idx & 0xF;
                    A_slm[nxt][idx] = A[(wg_m + r) * K + k_next + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int idx = flat_id + i * NUM_THREADS;
                    int r = idx >> 4;
                    int c = idx & 0xF;
                    int gm = wg_m + r;
                    int gk = k_next + c;
                    A_slm[nxt][idx] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
                }
            }

            if (interior_n && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int idx = flat_id + i * NUM_THREADS;
                    int r = idx >> 6;
                    int c = idx & 0x3F;
                    B_slm[nxt][idx] = B[(k_next + r) * N + wg_n + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int idx = flat_id + i * NUM_THREADS;
                    if (idx < B_TILE_SIZE) {
                        int r = idx >> 6;
                        int c = idx & 0x3F;
                        int gk = k_next + r;
                        int gn = wg_n + c;
                        B_slm[nxt][idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
                    }
                }
            }
        }

        // Compute on current buffer
        __local const half* a_ptr = A_slm[cur] + row_base * TILE_K;
        __local const half* b_ptr = B_slm[cur];

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            // Load 4 B values for this thread's columns
            float b_reg[THREAD_N];
            #pragma unroll
            for (int c = 0; c < THREAD_N; c++) {
                b_reg[c] = convert_float(b_ptr[kk * TILE_N + col_base + c]);
            }

            // Load 8 A values and accumulate
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                float a_val = convert_float(a_ptr[r * TILE_K + kk]);
                #pragma unroll
                for (int c = 0; c < THREAD_N; c++) {
                    acc[r][c] = fma(a_val, b_reg[c], acc[r][c]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ===== Store results =====
    const int out_col_base = wg_n + col_base;

    if (interior_m && interior_n) {
        #pragma unroll
        for (int r = 0; r < THREAD_M; r++) {
            int out_row = wg_m + row_base + r;
            __global half* out_ptr = C + out_row * N + out_col_base;
            half4 result = (half4)(
                convert_half(acc[r][0]),
                convert_half(acc[r][1]),
                convert_half(acc[r][2]),
                convert_half(acc[r][3])
            );
            vstore4(result, 0, out_ptr);
        }
    } else {
        #pragma unroll
        for (int r = 0; r < THREAD_M; r++) {
            int out_row = wg_m + row_base + r;
            if (out_row < M) {
                #pragma unroll
                for (int c = 0; c < THREAD_N; c++) {
                    int out_col = out_col_base + c;
                    if (out_col < N) {
                        C[out_row * N + out_col] = convert_half(acc[r][c]);
                    }
                }
            }
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.

(II) Bottlenecks and suggestions for improvement:
1. `for (int t = 0; t < num_k_tiles; t++) { ... // Prefetch next tile into nxt buffer ... // Compute on current buffer ... barrier(CLK_LOCAL_MEM_FENCE); }`: your “double buffering” is not actually overlapping load+compute, because prefetch and compute are both separated by full-workgroup barriers and executed serially by the same threads.  
   **How to improve:** switch to true async local-memory copies (Intel OpenCL supports `async_work_group_copy`) and pipeline them:
   - issue async copy for tile `t+1` into `nxt`,
   - compute tile `t` from `cur`,
   - `wait_group_events` only when you actually need `nxt`.
   This removes a large part of barrier stall time and gives real latency hiding. If async copy is unavailable on your target, reduce synchronization frequency by increasing per-load chunk and minimizing per-iteration control flow around loads.

2. `float b_reg[THREAD_N]; ... b_reg[c] = convert_float(b_ptr[kk * TILE_N + col_base + c]);` and `float a_val = convert_float(a_ptr[r * TILE_K + kk]);`: you are doing scalar half→float conversions in the innermost MAC path, which is very expensive relative to FMAs.  
   **How to improve:** vectorize loads/conversions and reuse converted values:
   - load B as one `half4` per `kk` (`vload4`) then `float4 b = convert_float4(...)`;
   - for A, unroll rows in pairs/quads and use `half2/half4` loads where layout allows (or pre-convert one `kk` column of 8 A values into float regs once, then consume across 4 cols).
   This cuts conversion instruction count and improves SIMD utilization. Keep accumulation in `float`, but don’t repeatedly scalar-convert from local memory.

3. `__local half A_slm[2][A_TILE_SIZE]; __local half B_slm[2][B_TILE_SIZE];` with `B_slm[cur][kk * TILE_N + col_base + c]` access by all lanes: B local-memory access pattern causes heavy bank conflicts/broadcast pressure (many threads in subgroup hit nearby same-bank addresses each `kk`).  
   **How to improve:** pad and/or transpose B tile in SLM during load:
   - allocate `B_slm[2][TILE_K][TILE_N + PAD]` (e.g., `PAD=1..4`) and index with padded stride, or
   - store as transposed `[TILE_N][TILE_K]` to make per-thread `kk` fetches more bank-friendly for your subgroup layout.
   Also consider subgroup-cooperative B loads (one lane loads packed vector, subgroup shuffle/broadcast) to reduce redundant SLM reads. This usually gives a noticeable speedup on Intel GPUs for this GEMM shape.

## Hardware specification:
Your code will run on the following hardware:
**Intel Battlemage** with specs: Xe-cores: 20, Render Slices: 5, Ray Tracing Units: 20, Intel® Xe Matrix Extensions (Intel® XMX) Engines: 160, Xe Vector Engines: 160, Graphics Clock: 2670, GPU Peak TOPS (Int8): 233, TBP: 190, PCI Express Configurations ‡: PCI Express 4.0 x8, Device ID: 0xE20B, Memory: 12 GB GDDR6, Memory Interface: 192 bit, Memory Bandwidth: 456, Memory Speed: 19, ISA_GPU: Xe2-HPG
Please consider the hardware specifications when improving the code. 

## Task:

**Your objectives**:
1. Analyze the previous versions and their results (why does one achieve better results than the other?).
2. Identify any inefficiencies and bottlenecks.
3. Propose specific improvements or options to take the best of all prior versions, explaining your reasoning step by step.

4. Provide a new kernel that achieves better performance **on the target hardware**. Provide the complete, improved code in a code block.

**Optimization strategies**:

Here are some potential strategies to improve the kernel runtime:
1. Use Appropriate Data Types: Use half (FP16) when precision allows - doubles throughput on many GPUs. Use native_* math functions (native_sqrt, native_divide) for faster, less precise operations. Use mad() for fused multiply-add. Size integer types appropriately (char, short, int).
2. Use Local Memory Strategically: Use __local memory for data reused across work-items in a work-group. Implement tile-based algorithms where each work-group loads a tile into local memory, processes it, then moves to the next tile. Declare with __local float tile[TILE_SIZE].

**Critical Requirements:**

1. The kernel must exactly match the reference implementation's functionality.
2. The code must compile and run properly on the GPU.
3. Do not cache or reuse previous results; ensure the code executes fully on each run.
4. Keep all hyperparameters (e.g., batch size, dimensions) unchanged as specified in the reference implementation.
8. Beware of the critical error "Unexpected kernel lambda size. In such cases removing constexpr specifier aligns the captures between the host compiler and the device compiler"! Do not capture constexpr variables in lambda functions passed to kernel launches as this can lead to different lambda sizes between the host and device compiler.

Additional Guidance:

1. Clearly comment on any performance optimizations you implement.
2. If you change the kernel structure, explain why.
3. Anticipate and address possible runtime or compilation errors.

Please structure your response as follows:

1. Analysis:
    * Summarize the issues found in the previous kernel and log.
    * Explain your proposed changes and optimizations.
2. Improved OCL code:
    * Provide the complete, improved OCL code in a code block:
```OCL
Your code here
```


## Required Optimizations

Apply the following optimization techniques in your implementation:

- **Register Blocking**: Each work-item computes a THREAD_M×THREAD_N output block in private register arrays. Use `#pragma unroll` on inner loops. Combine with SLM tiling for multi-level memory hierarchy optimization.
- **Kernel Fusion**: Combine sequential operations (e.g., exp → add → activation) into a single kernel. Eliminate intermediate buffers by computing in registers.
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.