

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 57.100):
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

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 47.000):
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 71.000):
```OCL
// FP16 GEMM with SLM tiling and register blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// WG tile: 128 rows x 32 cols
// THREAD_M=8, THREAD_N=2
// LWS = (16, 16, 1) = 256 work-items
// GWS = (ceil(N/32)*16, ceil(M/128)*16, 1)
// TILE_K = 16
//
// SLM: A_slm[128*16] = 2048 halfs = 4KB
//       B_slm[16*32]  = 512  halfs = 1KB
// Total SLM = 5KB per WG

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 128
#define TILE_N 32
#define TILE_K 16
#define THREAD_M 8
#define THREAD_N 2

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

    // Thread (lx, ly) computes:
    //   rows: wg_m + ly*THREAD_M + r, r=0..7
    //   cols: wg_n + lx*2 + c, c=0..1
    const int row_base = wg_m + ly * THREAD_M;
    const int col_base = wg_n + lx * THREAD_N;

    // 8x2 = 16 accumulators
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++)
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++)
            acc[i][j] = 0.0f;

    __local half A_slm[TILE_M * TILE_K];  // 128 x 16
    __local half B_slm[TILE_K * TILE_N];  // 16 x 32

    const int flat_id = ly * 16 + lx;  // 0..255

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A_slm[128][16] = 2048 halfs, 256 threads => 8 each
        // Use vectorized load: each thread loads 8 contiguous halfs
        {
            int base = flat_id * 8;  // each thread loads 8 elements starting here
            int r = base / TILE_K;   // row in A_slm
            int c = base % TILE_K;   // col in A_slm
            // Since TILE_K=16, and we load 8 at a time, c is either 0 or 8
            int gm = wg_m + r;
            int gk = k0 + c;
            if (gm < M && gk + 7 < K) {
                half8 val = vload8(0, A + gm * K + gk);
                vstore8(val, 0, A_slm + base);
            } else {
                #pragma unroll
                for (int x = 0; x < 8; x++) {
                    int rr = (base + x) / TILE_K;
                    int cc = (base + x) % TILE_K;
                    int gm2 = wg_m + rr;
                    int gk2 = k0 + cc;
                    A_slm[base + x] = (gm2 < M && gk2 < K) ? A[gm2 * K + gk2] : (half)0.0h;
                }
            }
        }

        // Load B_slm[16][32] = 512 halfs, 256 threads => 2 each
        {
            int base = flat_id * 2;
            if (base < TILE_K * TILE_N) {
                int r = base / TILE_N;
                int c = base % TILE_N;
                int gk = k0 + r;
                int gn = wg_n + c;
                if (gk < K && gn + 1 < N) {
                    half2 val = vload2(0, B + gk * N + gn);
                    vstore2(val, 0, B_slm + base);
                } else {
                    B_slm[base] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
                    B_slm[base + 1] = (gk < K && (gn + 1) < N) ? B[gk * N + gn + 1] : (half)0.0h;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            // Load B values for this thread's 2 cols
            float b0 = convert_float(B_slm[kk * TILE_N + lx * 2]);
            float b1 = convert_float(B_slm[kk * TILE_N + lx * 2 + 1]);

            // Load A values and accumulate
            #pragma unroll
            for (int r = 0; r < THREAD_M; r++) {
                float a_val = convert_float(A_slm[(ly * THREAD_M + r) * TILE_K + kk]);
                acc[r][0] = mad(a_val, b0, acc[r][0]);
                acc[r][1] = mad(a_val, b1, acc[r][1]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    #pragma unroll
    for (int r = 0; r < THREAD_M; r++) {
        int out_row = row_base + r;
        if (out_row < M) {
            #pragma unroll
            for (int c = 0; c < THREAD_N; c++) {
                int out_col = col_base + c;
                if (out_col < N) {
                    C[out_row * N + out_col] = convert_half(acc[r][c]);
                }
            }
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.  
(II) Bottlenecks and suggestions for improvement:  
1. `float a_val = convert_float(A_slm[(ly * THREAD_M + r) * TILE_K + kk]);` + `float b0 = convert_float(B_slm[kk * TILE_N + lx * 2]);` / `float b1 = convert_float(B_slm[kk * TILE_N + lx * 2 + 1]);`: you are doing **FP16→FP32 conversion inside the innermost MAC loop** for every `kk` and `r`, which adds a lot of conversion and load pressure.  
   - Improve by computing in packed half2 where possible (2 columns naturally fit `THREAD_N=2`), e.g. load `half2 b = vload2(...)` once per `kk`, and use half-vector FMA paths (`fma` on vector floats after one conversion, or native half FMA if your target supports it).  
   - At minimum, hoist B conversion once per `kk` (already partly done), and for A load two rows at once via `half2`/`float2` to halve conversion instructions in the `r` loop.  
   - This directly reduces instruction count in your hottest loop.

2. `if (gm < M && gk + 7 < K) { ... } else { ... }` and `if (gk < K && gn + 1 < N) { ... } else { ... }` in the tile-load path: boundary checks are executed on **every K tile**, causing branch divergence and extra scalar ops even for interior workgroups.  
   - Split kernel behavior into fast interior path vs edge path: for tiles fully inside bounds, do unconditional vectorized loads/stores (no per-element guards); only edge workgroups run guarded loads.  
   - Concretely: compute `fullA = (wg_m + TILE_M <= M) && (k0 + TILE_K <= K)` and `fullB = (wg_n + TILE_N <= N) && (k0 + TILE_K <= K)` once per tile, then take one branch per WG instead of per-thread/per-element branches.  
   - This usually gives a noticeable speedup because most tiles are interior.

3. `__local half A_slm[TILE_M * TILE_K];` with access `A_slm[(ly * THREAD_M + r) * TILE_K + kk]`: this access pattern can create **local-memory bank conflicts** because many lanes in a wave access the same `kk` column with a fixed stride (`TILE_K=16`).  
   - Pad or swizzle A in SLM: e.g. allocate `A_slm[TILE_M * (TILE_K + 1)]` and index with `(row * (TILE_K + 1) + kk)` to break stride-aliasing across banks.  
   - Keep B as-is or similarly pad if profiling shows conflicts there too.  
   - This improves effective SLM bandwidth and helps the compute loop stay fed.

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
1. Leverage Vectorization: Use vector types (float4, float8, float16) for explicit SIMD. Use vloadN() and vstoreN() for unaligned vector memory access. For aligned access, cast pointers: ((__global float4*)ptr)[idx]. Match native vector width of target device.
2. Use Predication: Replace if(cond) x = a; else x = b; with x = select(b, a, cond) for scalar types or x = cond ? a : b. Use select() for vector types: result = select(false_val, true_val, condition).

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
- **Online Algorithms**: Use single-pass algorithms with running statistics. For softmax: maintain running_max and running_sum, rescaling sum when max changes. For variance: use Welford's online algorithm.
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.