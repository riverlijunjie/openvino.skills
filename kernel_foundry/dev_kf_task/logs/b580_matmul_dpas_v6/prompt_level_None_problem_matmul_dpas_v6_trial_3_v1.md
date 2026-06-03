

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 95.900):
```OCL
// FP16 GEMM with double-buffered SLM tiling and register blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: 64 rows x 64 cols, K-tile = 16
// Each work-item computes THREAD_M=8 rows x THREAD_N=4 cols
// LWS = (16, 8, 1) = 128 work-items
// GWS = (ceil(N/64)*16, ceil(M/64)*8, 1)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 8
#define THREAD_N 4
#define WG_X 16
#define WG_Y 8
#define NUM_THREADS 128

__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int lx = get_local_id(0);  // 0..15 -> maps to N dim
    const int ly = get_local_id(1);  // 0..7  -> maps to M dim

    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    // Each thread handles THREAD_N=4 consecutive columns and THREAD_M=8 consecutive rows
    const int col_base = lx * THREAD_N;  // 0,4,8,...,60
    const int row_base = ly * THREAD_M;  // 0,8,16,...,56

    // Accumulators: 8 rows x 4 cols
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            acc[i][j] = 0.0f;

    // Double-buffered SLM
    __local half A_slm[2][TILE_M * TILE_K];   // 64 x 16
    __local half B_slm[2][TILE_K * TILE_N];   // 16 x 64

    const int flat_id = ly * WG_X + lx;  // 0..127

    // Preload first tile (buffer 0)
    int k0 = 0;
    // Load A: 64*16 = 1024 halfs, 128 threads => 8 each
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = flat_id + i * NUM_THREADS;
        int r = idx / TILE_K;
        int c = idx % TILE_K;
        int gm = wg_m + r;
        int gk = c;
        A_slm[0][idx] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
    }
    // Load B: 16*64 = 1024 halfs, 128 threads => 8 each
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = flat_id + i * NUM_THREADS;
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        int gk = r;
        int gn = wg_n + c;
        B_slm[0][idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;

    for (k0 = 0; k0 < K; k0 += TILE_K) {
        int next_k = k0 + TILE_K;
        int next_buf = 1 - buf;

        // Prefetch next tile into next_buf (if there is a next tile)
        if (next_k < K) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx / TILE_K;
                int c = idx % TILE_K;
                int gm = wg_m + r;
                int gk = next_k + c;
                A_slm[next_buf][idx] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
            }
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id + i * NUM_THREADS;
                int r = idx / TILE_N;
                int c = idx % TILE_N;
                int gk = next_k + r;
                int gn = wg_n + c;
                B_slm[next_buf][idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }
        }

        // Compute on current buffer
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            // Load A values for this thread's rows (hoisted conversion)
            float a_val[THREAD_M];
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                a_val[r] = convert_float(A_slm[buf][(row_base + r) * TILE_K + kk]);
            }

            // Load B values for this thread's cols (hoisted conversion)
            float b_val[THREAD_N];
            #pragma unroll
            for (int c = 0; c < THREAD_N; c++) {
                b_val[c] = convert_float(B_slm[buf][kk * TILE_N + col_base + c]);
            }

            // Outer product accumulation
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                #pragma unroll
                for (int c = 0; c < THREAD_N; c++) {
                    acc[r][c] = fma(a_val[r], b_val[c], acc[r][c]);
                }
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

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 62.300):
```OCL
// Optimized FP16 GEMM with SLM tiling and register blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: TILE_M x TILE_N = 64 x 64
// K-tile: TILE_K = 16
// LWS = (16, 16, 1) = 256 work-items
// Each work-item computes THREAD_M x THREAD_N = 4 x 4 output elements
// GWS = (ceil(N/64)*16, ceil(M/64)*16, 1)
//
// SLM: A_slm[64][16] = 1024 halfs, B_slm[16][64] = 1024 halfs

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 4
#define THREAD_N 4
#define WG_X 16
#define WG_Y 16

__attribute__((reqd_work_group_size(16, 16, 1)))
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

    // Each work-item's output region
    // 16 threads along x cover 64 cols => 4 cols each, stride=16
    // 16 threads along y cover 64 rows => 4 rows each, stride=16
    // Thread (lx, ly) computes rows [ly, ly+16, ly+32, ly+48] x cols [lx, lx+16, lx+32, lx+48]
    // But contiguous is better. Let's do:
    // rows: ly*4 .. ly*4+3, cols: lx*4 .. lx*4+3
    // Actually for SLM access patterns, strided is better to avoid bank conflicts.
    // Use: rows at ly + r*16 (r=0..3), cols at lx + c*16 (c=0..3)

    // Accumulators: 4x4 = 16 floats
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            acc[i][j] = 0.0f;

    // SLM tiles
    __local half A_slm[TILE_M * TILE_K];  // 64 x 16 = 1024
    __local half B_slm[TILE_K * TILE_N];  // 16 x 64 = 1024

    const int flat_id = ly * WG_X + lx;  // 0..255
    const int num_threads = WG_X * WG_Y; // 256

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperative load A_slm[64][16] = 1024 halfs, 256 threads => 4 each
        #pragma unroll
        for (int i = flat_id; i < TILE_M * TILE_K; i += num_threads) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int gm = wg_m + r;
            int gk = k0 + c;
            A_slm[i] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
        }

        // Cooperative load B_slm[16][64] = 1024 halfs, 256 threads => 4 each
        #pragma unroll
        for (int i = flat_id; i < TILE_K * TILE_N; i += num_threads) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int gk = k0 + r;
            int gn = wg_n + c;
            B_slm[i] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: each thread does 4 rows x 4 cols
        // rows: ly + r*16, r=0..3
        // cols: lx + c*16, c=0..3
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            // Load A values for this thread's 4 rows
            float a_reg[THREAD_M];
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                a_reg[r] = convert_float(A_slm[(ly + r * WG_Y) * TILE_K + kk]);
            }

            // Load B values for this thread's 4 cols
            float b_reg[THREAD_N];
            #pragma unroll
            for (int c = 0; c < THREAD_N; c++) {
                b_reg[c] = convert_float(B_slm[kk * TILE_N + lx + c * WG_X]);
            }

            // Outer product accumulation
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                #pragma unroll
                for (int c = 0; c < THREAD_N; c++) {
                    acc[r][c] += a_reg[r] * b_reg[c];
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    #pragma unroll
    for (int r = 0; r < THREAD_M; r++) {
        int out_row = wg_m + ly + r * WG_Y;
        if (out_row < M) {
            #pragma unroll
            for (int c = 0; c < THREAD_N; c++) {
                int out_col = wg_n + lx + c * WG_X;
                if (out_col < N) {
                    C[out_row * N + out_col] = convert_half(acc[r][c]);
                }
            }
        }
    }
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
1. `float b_val = convert_float(B_slm[kk * TILE_N + lx]);` and `float a_val = convert_float(A_slm[(ly * THREAD_M + r) * TILE_K + kk]);`: you are converting `half -> float` inside the innermost FMAs, so conversion overhead is paid every iteration (`TILE_K * THREAD_M` times per thread per tile).  
   **Improve it:** keep math in half2/vec form as long as possible, and reduce conversion frequency. A practical step is to load `half2` from SLM and do two-k accumulation at once (unroll `kk` by 2), converting two values together. Even better on GPUs with native half FMA: accumulate in FP16/FP32 mixed instructions (vendor intrinsic path) so scalar converts are not on the critical path.

2. `__local half A_slm[TILE_M * TILE_K];` with access `A_slm[(ly * THREAD_M + r) * TILE_K + kk]`: this layout creates a high risk of local-memory bank conflicts when 16 threads in `ly` read different rows at the same `kk` (stride = `TILE_K=16`, often maps poorly to bank geometry).  
   **Improve it:** pad or swizzle A in SLM, e.g. allocate `A_slm[TILE_M * (TILE_K + 1)]` and index with `(TILE_K+1)` stride, or store transposed/swizzled so threads in a wavefront hit distinct banks. This usually gives a noticeable drop in local-memory stall cycles.

3. `THREAD_M=8, THREAD_N=1` and per-thread compute pattern “8 rows × 1 col”: arithmetic intensity per loaded `B_slm` value is low, and each thread reuses only one B element per `kk`. You’re underutilizing registers and issuing too many similar scalar memory ops/stores.  
   **Improve it:** increase register blocking in N (e.g., `THREAD_N=2` or `4`) so each thread computes multiple columns. Then each loaded `A` value is reused across multiple accumulators, improving FLOPs/byte and reducing pressure on SLM/L1. This typically outperforms `8x1` on GEMM unless occupancy collapses; tune `(THREAD_M, THREAD_N)` jointly with WG shape (for example `8x2` or `4x4`) to keep occupancy acceptable while increasing data reuse.

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
1. Memory Alignment: Align data structures to vector boundaries. Use __attribute__((aligned(N))) where N is 16, 32, or 64. Ensure global memory allocations are aligned for coalesced access.
2. Leverage Vectorization: Use vector types (float4, float8, float16) for explicit SIMD. Use vloadN() and vstoreN() for unaligned vector memory access. For aligned access, cast pointers: ((__global float4*)ptr)[idx]. Match native vector width of target device.

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