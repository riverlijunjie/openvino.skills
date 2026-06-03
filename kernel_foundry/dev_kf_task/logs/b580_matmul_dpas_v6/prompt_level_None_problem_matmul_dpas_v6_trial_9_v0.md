

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 38.900):
```OCL
// FP16 GEMM with SLM tiling, bank-conflict-free layout, vectorized loads
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: TILE_M=128 x TILE_N=16, TILE_K=64
// THREAD_M=8, each thread computes 8 rows x 1 col
// LWS = (16, 16, 1) = 256 work-items
// GWS = (ceil(N/16)*16, ceil(M/128)*16, 1)
// SLM: A_slm[128*64]=16KB, B_slm[64*17]=2.125KB => ~18KB total
// B_slm padded to stride 17 to avoid bank conflicts

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 128
#define TILE_N 16
#define TILE_K 64
#define THREAD_M 8
#define NUM_THREADS 256
#define B_STRIDE 17  // padded to avoid bank conflicts

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

    // 8 accumulators - use individual variables for register allocation
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    // SLM tiles - B padded to avoid bank conflicts
    __local half A_slm[TILE_M * TILE_K];       // 128*64 = 8192 halfs = 16KB
    __local half B_slm[TILE_K * B_STRIDE];     // 64*17 = 1088 halfs ~2.1KB

    const int interior_m = (wg_m + TILE_M <= M);
    const int interior_n = (wg_n + TILE_N <= N);
    const int K_main = (K / TILE_K) * TILE_K;

    // A: 8192 / 256 = 32 elements each
    // B: 1024 / 256 = 4 elements each

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int k_valid = (k0 + TILE_K <= K);

        // --- Load A_slm[128][64] cooperatively ---
        if (interior_m && k_valid) {
            // Fast path: no bounds checks needed
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
                int gk = k0 + c;
                A_slm[idx] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
            }
        }

        // --- Load B_slm[64][16] with padding to stride 17 ---
        if (interior_n && k_valid) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 4;      // idx / 16 -> k index
                int c = idx & 0xF;     // idx % 16 -> n index
                B_slm[r * B_STRIDE + c] = B[(k0 + r) * N + wg_n + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx >> 4;
                int c = idx & 0xF;
                int gk = k0 + r;
                int gn = wg_n + c;
                B_slm[r * B_STRIDE + c] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Compute: unroll k by 8 ---
        const int a_base = ly * THREAD_M * TILE_K;

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 8) {
            // Load 8 B values from padded SLM
            float b0 = convert_float(B_slm[(kk + 0) * B_STRIDE + lx]);
            float b1 = convert_float(B_slm[(kk + 1) * B_STRIDE + lx]);
            float b2 = convert_float(B_slm[(kk + 2) * B_STRIDE + lx]);
            float b3 = convert_float(B_slm[(kk + 3) * B_STRIDE + lx]);
            float b4 = convert_float(B_slm[(kk + 4) * B_STRIDE + lx]);
            float b5 = convert_float(B_slm[(kk + 5) * B_STRIDE + lx]);
            float b6 = convert_float(B_slm[(kk + 6) * B_STRIDE + lx]);
            float b7 = convert_float(B_slm[(kk + 7) * B_STRIDE + lx]);

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
            if (row_base + 0 < M) C[(row_base + 0) * N + out_col] = convert_half(acc0);
            if (row_base + 1 < M) C[(row_base + 1) * N + out_col] = convert_half(acc1);
            if (row_base + 2 < M) C[(row_base + 2) * N + out_col] = convert_half(acc2);
            if (row_base + 3 < M) C[(row_base + 3) * N + out_col] = convert_half(acc3);
            if (row_base + 4 < M) C[(row_base + 4) * N + out_col] = convert_half(acc4);
            if (row_base + 5 < M) C[(row_base + 5) * N + out_col] = convert_half(acc5);
            if (row_base + 6 < M) C[(row_base + 6) * N + out_col] = convert_half(acc6);
            if (row_base + 7 < M) C[(row_base + 7) * N + out_col] = convert_half(acc7);
        }
    }
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 36.200):
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 36.200):
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

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.  
You did a solid job on correctness and basic tiling, but the current mapping leaves a lot of throughput on the table, especially for Xe/Intel-style subgroup hardware.

(II) Bottlenecks and suggestions for improvement:  
1. `float b0 = convert_float(B_slm[(kk + 0) * TILE_N + lx]);` (and same pattern for `b1..b3`, inside the inner loop): **reloading B from SLM every `kk` step is a hot-path bandwidth bottleneck**.  
   - Right now each thread repeatedly reads scalar `half` values from local memory and converts them one-by-one. Since `THREAD_N=1`, each thread computes only one output column, so B reuse per thread is low and SLM traffic dominates.  
   - **Improve it by increasing per-thread N work (register blocking on N)**, e.g. make each thread compute 2–4 columns (`THREAD_N=2/4`) and keep `b` vectors in registers. That way each loaded A value is reused across multiple output columns, reducing both SLM reads and conversion overhead.  
   - Concretely: change mapping so `lx` addresses a small column block; load `half2/half4` from `B_slm` and accumulate into `acc[THREAD_M][THREAD_N]`. This is usually the highest-impact change for this kernel shape.

2. `float a0 = convert_float(A_slm[aoff + 0]); ... float a3 = convert_float(A_slm[aoff + 3]);` inside `DO_ROW`, repeated for 8 rows: **very high conversion + scalar load pressure in the tightest loop**.  
   - For every `kk`, each thread does many scalar half loads/converts from SLM (8 rows × 4 A loads), which is expensive and inflates instruction count.  
   - **Improve with vectorized loads/converts from SLM**: load `half4`/`ushort4`-equivalent chunks and convert once (`convert_float4`) instead of 4 scalar converts; then FMA with vector lanes or manually unrolled float4 components.  
   - Also consider reducing `TILE_K` to 32 if register pressure is limiting occupancy (common when fully unrolling this pattern). A slightly smaller K tile can speed up runtime if it enables more resident warps/subgroups.

3. `A_slm[idx] = A[(wg_m + r) * K + k0 + c];` and `B_slm[idx] = B[(k0 + r) * N + wg_n + c];` in the load stage: **global-memory load pattern is not vectorized and can be poorly coalesced/aligned**.  
   - Each thread performs many scalar half loads (`32` for A, `4` for B). That creates extra memory transactions and address-generation overhead.  
   - **Improve by vectorizing global loads and SLM stores**: have each thread load `half2/half4/half8` contiguous elements (aligned) and write vector chunks to SLM. For A, remap loader indices so neighboring threads in `lx` fetch contiguous K elements to maximize coalescing; for B, use vector loads along N where contiguous in memory.  
   - If your target supports it, use subgroup block read/write builtins for SLM/global transfers; these typically outperform scalar pointer accesses in this exact tiling pattern.

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
1. Use Built-in Functions: Prefer built-in functions (mad, fma, dot, cross, length) over manual implementations. Use fast_* variants (fast_normalize, fast_length) when precision allows. Use native_* for maximum speed with reduced precision.
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
- **Online Algorithms**: Use single-pass algorithms with running statistics. For softmax: maintain running_max and running_sum, rescaling sum when max changes. For variance: use Welford's online algorithm.
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.