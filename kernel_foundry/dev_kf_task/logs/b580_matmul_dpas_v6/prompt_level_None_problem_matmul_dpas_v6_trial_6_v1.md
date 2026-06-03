

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 47.000):
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
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 48.200):
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

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.

(II) Bottlenecks and suggestions for improvement:
1. `float b_val = convert_float(B_slm[kk * TILE_N + lx]);` and `float a0 = convert_float(a_ptr[0 * A_STRIDE + kk]); ... float a7 = convert_float(a_ptr[7 * A_STRIDE + kk]);`:  
   You’re doing a lot of scalar half→float conversions inside the hottest inner loop (`kk` loop). That conversion pressure is expensive and limits throughput.  
   **Improve it by vectorizing loads/conversions**: load `half8` (or two `half4`) for A rows and convert once to `float8`; similarly for B use `half2/half4` where possible. Even if each thread still computes one output column, grouping A conversions reduces instruction count and register move overhead. If your target supports matrix instructions/subgroup block ops, this is the first place to map to them.

2. `barrier(CLK_LOCAL_MEM_FENCE);` (after every tile load, twice per K-tile):  
   The kernel is fully synchronous per tile with no overlap between global-memory fetch and compute, so latency from global loads is exposed every iteration.  
   **Improve it with double-buffered SLM ping-pong**: allocate two A/B local tiles (`A_slm0/1`, `B_slm0/1`), preload tile `k+1` while computing tile `k`, then swap buffers. This keeps ALUs busy while next tile is loading and typically gives a noticeable speedup for GEMM-style kernels.

3. `C[(row_base + r) * N + out_col] = convert_half(val);` with
   ```c
   switch(r) { case 0: val = acc0; ... case 7: val = acc7; }
   ```
   and edge path `float accs[8] = {acc0, ... , acc7};`  
   Store path adds avoidable scalar control/data-move overhead (switch, temp array), and each thread writes strided single elements.  
   **Improve by writing accumulators directly without switch/array materialization** (explicit 8 stores or packed stores), and if alignment permits, pack two results and use `vstore_half2` to reduce store instructions. Keep the fast path branchless and minimal—this is small compared to compute, but it’s still on every thread and every tile result.

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
1. Loop Unrolling: Use #pragma unroll N for small, fixed-iteration loops. Manually unroll critical loops when compiler doesn't optimize. Prefer compile-time loop bounds.
2. Use Kernel Attributes: Use __attribute__((reqd_work_group_size(X,Y,Z))) to specify fixed work-group size for compiler optimization. Use __attribute__((vec_type_hint(float4))) to hint vectorization opportunities.

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