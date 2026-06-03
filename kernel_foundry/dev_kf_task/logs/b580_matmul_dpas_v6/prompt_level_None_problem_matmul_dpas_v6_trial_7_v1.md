

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 36.200):
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

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 42.700):
```OCL
// FP16 GEMM with double-buffered SLM tiling
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: TILE_M=128 x TILE_N=16, TILE_K=32
// THREAD_M=8, THREAD_N=1
// LWS = (16, 16, 1) = 256 work-items
// GWS = (ceil(N/16)*16, ceil(M/128)*16, 1)
// Double-buffered SLM: 2 x (A_slm[128*32] + B_slm[32*16])

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 128
#define TILE_N 16
#define TILE_K 32
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
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    const int out_col = wg_n + lx;
    const int row_base = wg_m + ly * THREAD_M;
    const int flat_id = ly * 16 + lx;

    // Scalar accumulators to minimize register pressure
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    // Double-buffered SLM
    // A: 128 rows x 32 cols = 4096 halfs per buffer
    // B: 32 rows x 16 cols = 512 halfs per buffer
    __local half A_slm[2][TILE_M * TILE_K];
    __local half B_slm[2][TILE_K * TILE_N];

    const int interior_m = (wg_m + TILE_M <= M);
    const int interior_n = (wg_n + TILE_N <= N);

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Macro for loading A tile into buffer `buf` at k-offset `k0`
    // A has 4096 elements, 256 threads => 16 each
    #define LOAD_A(buf, k0) do { \
        if (interior_m && (k0) + TILE_K <= K) { \
            _Pragma("unroll") \
            for (int _i = 0; _i < 16; _i++) { \
                int _idx = flat_id + _i * NUM_THREADS; \
                int _r = _idx >> 5; \
                int _c = _idx & 0x1F; \
                A_slm[buf][_idx] = A[(wg_m + _r) * K + (k0) + _c]; \
            } \
        } else { \
            _Pragma("unroll") \
            for (int _i = 0; _i < 16; _i++) { \
                int _idx = flat_id + _i * NUM_THREADS; \
                int _r = _idx >> 5; \
                int _c = _idx & 0x1F; \
                int _gm = wg_m + _r; \
                int _gk = (k0) + _c; \
                A_slm[buf][_idx] = (_gm < M && _gk < K) ? A[_gm * K + _gk] : (half)0.0h; \
            } \
        } \
    } while(0)

    // B has 512 elements, 256 threads => 2 each
    #define LOAD_B(buf, k0) do { \
        if (interior_n && (k0) + TILE_K <= K) { \
            _Pragma("unroll") \
            for (int _i = 0; _i < 2; _i++) { \
                int _idx = flat_id + _i * NUM_THREADS; \
                int _r = _idx >> 4; \
                int _c = _idx & 0xF; \
                B_slm[buf][_idx] = B[((k0) + _r) * N + wg_n + _c]; \
            } \
        } else { \
            _Pragma("unroll") \
            for (int _i = 0; _i < 2; _i++) { \
                int _idx = flat_id + _i * NUM_THREADS; \
                int _r = _idx >> 4; \
                int _c = _idx & 0xF; \
                int _gk = (k0) + _r; \
                int _gn = wg_n + _c; \
                B_slm[buf][_idx] = (_gk < K && _gn < N) ? B[_gk * N + _gn] : (half)0.0h; \
            } \
        } \
    } while(0)

    #define COMPUTE_TILE(buf) do { \
        const int _a_base = ly * THREAD_M * TILE_K; \
        _Pragma("unroll") \
        for (int _kk = 0; _kk < TILE_K; _kk++) { \
            float _bv = convert_float(B_slm[buf][_kk * TILE_N + lx]); \
            float _a0 = convert_float(A_slm[buf][_a_base + 0 * TILE_K + _kk]); \
            acc0 = fma(_a0, _bv, acc0); \
            float _a1 = convert_float(A_slm[buf][_a_base + 1 * TILE_K + _kk]); \
            acc1 = fma(_a1, _bv, acc1); \
            float _a2 = convert_float(A_slm[buf][_a_base + 2 * TILE_K + _kk]); \
            acc2 = fma(_a2, _bv, acc2); \
            float _a3 = convert_float(A_slm[buf][_a_base + 3 * TILE_K + _kk]); \
            acc3 = fma(_a3, _bv, acc3); \
            float _a4 = convert_float(A_slm[buf][_a_base + 4 * TILE_K + _kk]); \
            acc4 = fma(_a4, _bv, acc4); \
            float _a5 = convert_float(A_slm[buf][_a_base + 5 * TILE_K + _kk]); \
            acc5 = fma(_a5, _bv, acc5); \
            float _a6 = convert_float(A_slm[buf][_a_base + 6 * TILE_K + _kk]); \
            acc6 = fma(_a6, _bv, acc6); \
            float _a7 = convert_float(A_slm[buf][_a_base + 7 * TILE_K + _kk]); \
            acc7 = fma(_a7, _bv, acc7); \
        } \
    } while(0)

    // Load first tile into buffer 0
    LOAD_A(0, 0);
    LOAD_B(0, 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Double-buffered main loop
    for (int t = 0; t < num_k_tiles - 1; t++) {
        int cur = t & 1;
        int nxt = 1 - cur;
        int next_k0 = (t + 1) * TILE_K;

        // Prefetch next tile into other buffer
        LOAD_A(nxt, next_k0);
        LOAD_B(nxt, next_k0);

        // Compute on current buffer (no barrier needed - current buffer is ready)
        if (cur == 0) { COMPUTE_TILE(0); }
        else          { COMPUTE_TILE(1); }

        // Wait for prefetch to complete before next iteration uses it
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Compute last tile
    {
        int cur = (num_k_tiles - 1) & 1;
        if (cur == 0) { COMPUTE_TILE(0); }
        else          { COMPUTE_TILE(1); }
    }

    // Store results with select-based predication for N bound
    if (out_col < N) {
        if (row_base + 0 < M) C[(row_base + 0) * N + out_col] = convert_half(acc0);
        if (row_base + 1 < M) C[(row_base + 1) * N + out_col] = convert_half(acc1);
        if (row_base + 2 < M) C[(row_base + 2) * N + out_col] = convert_half(acc2);
        if (row_base + 3 < M) C[(row_base + 3) * N + out_col] = convert_half(acc3);
        if (row_base + 4 < M) C[(row_base + 4) * N + out_col] = convert_half(acc4);
        if (row_base + 5 < M) C[(row_base + 5) * N + out_col] = convert_half(acc5);
        if (row_base + 6 < M) C[(row_base + 6) * N + out_col] = convert_half(acc6);
        if (row_base + 7 < M) C[(row_base + 7) * N + out_col] = convert_half(acc7);
    }

    #undef LOAD_A
    #undef LOAD_B
    #undef COMPUTE_TILE
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 47.000):
```OCL
// FP16 GEMM with SLM tiling and register blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Each work-item computes THREAD_M x THREAD_N output elements
// THREAD_M=8, THREAD_N=1
// LWS = (16, 16, 1) = 256 work-items
// WG tile: 128 rows x 16 cols
// GWS = (ceil(N/16)*16, ceil(M/128)*16, 1)
// K-tile = 16

#define TILE_M 128
#define TILE_N 16
#define TILE_K 16
#define THREAD_M 8

__attribute__((reqd_work_group_size(16, 16, 1)))
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

    // SLM tiles
    __local half A_slm[TILE_M * TILE_K];  // 128 x 16
    __local half B_slm[TILE_K * TILE_N];  // 16 x 16

    const int flat_id = ly * 16 + lx;  // 0..255
    const int num_threads = 256;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperative load A_slm[128][16] = 2048 halfs, 256 threads => 8 each
        #pragma unroll
        for (int i = flat_id; i < TILE_M * TILE_K; i += num_threads) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int gm = wg_m + r;
            int gk = k0 + c;
            A_slm[i] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
        }

        // Cooperative load B_slm[16][16] = 256 halfs, 256 threads => 1 each
        {
            int i = flat_id;
            if (i < TILE_K * TILE_N) {
                int r = i / TILE_N;
                int c = i % TILE_N;
                int gk = k0 + r;
                int gn = wg_n + c;
                B_slm[i] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: each thread does THREAD_M rows x 1 col, iterating over K tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float b_val = convert_float(B_slm[kk * TILE_N + lx]);
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                float a_val = convert_float(A_slm[(ly * THREAD_M + r) * TILE_K + kk]);
                acc[r] += a_val * b_val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
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

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.

(II) Bottlenecks and suggestions for improvement:
1. `float b_val = convert_float(B_slm[kk * TILE_N + lx]);` + inner loop `for (int r = 0; r < THREAD_M; r++) { ... }`: you’re doing scalar math (`THREAD_N=1`), so each loaded `b_val` is reused only across 8 FMAs, and each thread computes just one output column. That underutilizes SIMD/ALU throughput and increases loop/control overhead per output element.  
   **Improve it by increasing per-thread N blocking** (e.g., `THREAD_N=2/4`) so each thread computes multiple columns, keeps multiple `b_val` in registers, and reuses each `A` value across more FMAs. Concretely: make local size x-dimension smaller accordingly, accumulate `acc[THREAD_M][THREAD_N]`, and load `B_slm[kk][lx*THREAD_N + tn]`. This usually gives a big jump in arithmetic intensity.

2. `A_slm[i] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;` and `B_slm[i] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;`: boundary checks are executed on every K tile load for all workgroups, causing extra predicate overhead and potential divergence on edge tiles.  
   **Improve by splitting kernels (or paths) into interior vs boundary tiles**:  
   - Fast path for full tiles (`wg_m + TILE_M <= M`, `wg_n + TILE_N <= N`, `k0 + TILE_K <= K`) with **no per-element bounds checks**.  
   - Slow path only for edge tiles with checks.  
   This removes a lot of control overhead from the dominant interior workload.

3. `barrier(CLK_LOCAL_MEM_FENCE); ... compute ... barrier(CLK_LOCAL_MEM_FENCE);` around each `k0` tile with single-buffer SLM (`__local half A_slm[...]`, `__local half B_slm[...]`): load and compute are fully serialized, so memory latency is exposed every iteration.  
   **Improve with double-buffered SLM (ping-pong)**: allocate `A_slm[2][...]`, `B_slm[2][...]`, preload tile 0, then for each iteration compute on buffer `p` while cooperatively loading next tile into buffer `1-p`, with one barrier between stage transitions. This overlaps global->local traffic with compute and typically reduces stall time significantly.

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
2. Async Memory Operations: Use async_work_group_copy() to overlap computation with memory transfers between global and local memory. Use async_work_group_strided_copy() for non-contiguous data. Wait with wait_group_events().

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

- **Shared Local Memory (SLM) Tiling**: Cache frequently accessed data in SLM using `group_local_memory_for_overwrite`. Synchronize with `group_barrier()` after writes and before reads. Use 16×16 or 32×32 tiles for float data.
- **Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing.