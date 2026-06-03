

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Optimization goals:
- Use Intel OpenCL DPAS instruction when possible, e.g. intel_sub_group_f16_f16_matrix_mad_k16.
- Improve FLOPS and hide memory latency with tiling + subgroup-friendly mapping.
- Provide explicit launch metadata (GWS/LWS/subgroup hints) in kernel comments.
Current fallback/base implementation:
- Keep a correctness-first FP16 matmul kernel in EVOLVE region as baseline.

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 62.300):
```OCL
// Tiled FP16 matmul with SLM blocking
// C[M,N] = A[M,K] x B[K,N], all half, float accumulation
// Workgroup tile: TILE_M=64, TILE_N=64, TILE_K=16
// Each work-item computes a 4x4 sub-tile of C
// GWS = (ceil_div(N,64)*16, ceil_div(M,64)*16)
// LWS = (16, 16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define WI_ROWS 4
#define WI_COLS 4

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int lid_x = get_local_id(0);  // 0..15
    const int lid_y = get_local_id(1);  // 0..15
    const int gid_x = get_group_id(0);
    const int gid_y = get_group_id(1);

    const int tile_row = gid_y * TILE_M;
    const int tile_col = gid_x * TILE_N;

    __local half A_tile[TILE_M][TILE_K];
    __local half B_tile[TILE_K][TILE_N];

    float acc[WI_ROWS][WI_COLS];
    for (int i = 0; i < WI_ROWS; i++)
        for (int j = 0; j < WI_COLS; j++)
            acc[i][j] = 0.0f;

    const int flat_id = lid_y * 16 + lid_x; // 0..255

    for (int kt = 0; kt < K; kt += TILE_K) {
        // Load A_tile[64][16]: 64*16=1024 halfs, 256 threads, 4 each
        for (int i = 0; i < 4; i++) {
            int idx = flat_id * 4 + i;
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int gr = tile_row + r;
            int gc = kt + c;
            A_tile[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Load B_tile[16][64]: 16*64=1024 halfs, 256 threads, 4 each
        for (int i = 0; i < 4; i++) {
            int idx = flat_id * 4 + i;
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gr = kt + r;
            int gc = tile_col + c;
            B_tile[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: each WI handles rows [lid_y*4 .. lid_y*4+3] x cols [lid_x*4 .. lid_x*4+3]
        for (int kk = 0; kk < TILE_K; kk++) {
            for (int i = 0; i < WI_ROWS; i++) {
                float a_val = convert_float(A_tile[lid_y * WI_ROWS + i][kk]);
                for (int j = 0; j < WI_COLS; j++) {
                    float b_val = convert_float(B_tile[kk][lid_x * WI_COLS + j]);
                    acc[i][j] += a_val * b_val;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results
    for (int i = 0; i < WI_ROWS; i++) {
        for (int j = 0; j < WI_COLS; j++) {
            int gr = tile_row + lid_y * WI_ROWS + i;
            int gc = tile_col + lid_x * WI_COLS + j;
            if (gr < M && gc < N) {
                C[gr * N + gc] = convert_half(acc[i][j]);
            }
        }
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 27.700):
```OCL
// Tiled FP16 matmul with SLM blocking + vectorized loads + bank conflict padding
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
// LWS = (16, 16, 1)
// GWS = (ceil_div(N, 32)*16, ceil_div(M, 32)*16, 1)
// Subgroup size: 16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32
#define WG_X 16
#define WG_Y 16
// Pad to avoid bank conflicts (32 banks of 4 bytes = 64 halfs per bank cycle)
#define AS_STRIDE (TILE_K + 2)
#define BS_STRIDE (TILE_N + 2)

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
    const int tile_col = get_group_id(0) * TILE_N;
    const int tile_row = get_group_id(1) * TILE_M;

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lid = ly * WG_X + lx;

    const int out_row0 = tile_row + ly * 2;
    const int out_col0 = tile_col + lx * 2;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    __local half As[TILE_M * AS_STRIDE];
    __local half Bs[TILE_K * BS_STRIDE];

    for (int kk = 0; kk < K; kk += TILE_K) {
        // Load A tile [32 x 32] with padding stride
        for (int i = lid; i < TILE_M * TILE_K; i += 256) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int gr = tile_row + r;
            int gc = kk + c;
            As[r * AS_STRIDE + c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Load B tile [32 x 32] with padding stride
        for (int i = lid; i < TILE_K * TILE_N; i += 256) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int gr = kk + r;
            int gc = tile_col + c;
            Bs[r * BS_STRIDE + c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        int r0 = ly * 2;
        int r1 = ly * 2 + 1;
        int c0 = lx * 2;
        int c1 = lx * 2 + 1;

        // Unroll inner loop by 4
        int tk = 0;
        for (; tk + 3 < TILE_K; tk += 4) {
            float a0_0 = convert_float(As[r0 * AS_STRIDE + tk]);
            float a1_0 = convert_float(As[r1 * AS_STRIDE + tk]);
            float b0_0 = convert_float(Bs[tk * BS_STRIDE + c0]);
            float b1_0 = convert_float(Bs[tk * BS_STRIDE + c1]);

            float a0_1 = convert_float(As[r0 * AS_STRIDE + tk + 1]);
            float a1_1 = convert_float(As[r1 * AS_STRIDE + tk + 1]);
            float b0_1 = convert_float(Bs[(tk+1) * BS_STRIDE + c0]);
            float b1_1 = convert_float(Bs[(tk+1) * BS_STRIDE + c1]);

            float a0_2 = convert_float(As[r0 * AS_STRIDE + tk + 2]);
            float a1_2 = convert_float(As[r1 * AS_STRIDE + tk + 2]);
            float b0_2 = convert_float(Bs[(tk+2) * BS_STRIDE + c0]);
            float b1_2 = convert_float(Bs[(tk+2) * BS_STRIDE + c1]);

            float a0_3 = convert_float(As[r0 * AS_STRIDE + tk + 3]);
            float a1_3 = convert_float(As[r1 * AS_STRIDE + tk + 3]);
            float b0_3 = convert_float(Bs[(tk+3) * BS_STRIDE + c0]);
            float b1_3 = convert_float(Bs[(tk+3) * BS_STRIDE + c1]);

            acc00 = fma(a0_0, b0_0, acc00);
            acc01 = fma(a0_0, b1_0, acc01);
            acc10 = fma(a1_0, b0_0, acc10);
            acc11 = fma(a1_0, b1_0, acc11);

            acc00 = fma(a0_1, b0_1, acc00);
            acc01 = fma(a0_1, b1_1, acc01);
            acc10 = fma(a1_1, b0_1, acc10);
            acc11 = fma(a1_1, b1_1, acc11);

            acc00 = fma(a0_2, b0_2, acc00);
            acc01 = fma(a0_2, b1_2, acc01);
            acc10 = fma(a1_2, b0_2, acc10);
            acc11 = fma(a1_2, b1_2, acc11);

            acc00 = fma(a0_3, b0_3, acc00);
            acc01 = fma(a0_3, b1_3, acc01);
            acc10 = fma(a1_3, b0_3, acc10);
            acc11 = fma(a1_3, b1_3, acc11);
        }
        for (; tk < TILE_K; ++tk) {
            float a0 = convert_float(As[r0 * AS_STRIDE + tk]);
            float a1 = convert_float(As[r1 * AS_STRIDE + tk]);
            float b0 = convert_float(Bs[tk * BS_STRIDE + c0]);
            float b1 = convert_float(Bs[tk * BS_STRIDE + c1]);
            acc00 = fma(a0, b0, acc00);
            acc01 = fma(a0, b1, acc01);
            acc10 = fma(a1, b0, acc10);
            acc11 = fma(a1, b1, acc11);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (out_row0 < M && out_col0 < N)
        C[out_row0 * N + out_col0] = convert_half(acc00);
    if (out_row0 < M && (out_col0 + 1) < N)
        C[out_row0 * N + out_col0 + 1] = convert_half(acc01);
    if ((out_row0 + 1) < M && out_col0 < N)
        C[(out_row0 + 1) * N + out_col0] = convert_half(acc10);
    if ((out_row0 + 1) < M && (out_col0 + 1) < N)
        C[(out_row0 + 1) * N + out_col0 + 1] = convert_half(acc11);
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 26.000):
```OCL
// Tiled FP16 matmul with SLM blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
// Recommended launch:
//   LWS = (16, 16, 1)
//   GWS = (ceil_div(N, 32)*16, ceil_div(M, 32)*16, 1)
//   Subgroup size: 16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32
#define WG_X 16
#define WG_Y 16

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
    // Workgroup tile origin in C
    const int tile_col = get_group_id(0) * TILE_M;
    const int tile_row = get_group_id(1) * TILE_N;

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lid = ly * WG_X + lx; // linear local id [0..255]

    // Each thread computes 2x2 output elements
    const int out_row0 = tile_row + ly * 2;
    const int out_col0 = tile_col + lx * 2;

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    __local half As[TILE_M * TILE_K]; // 32x32
    __local half Bs[TILE_K * TILE_N]; // 32x32

    // Each WG has 256 threads, need to load 32*32=1024 halfs for A and B
    // Each thread loads 4 elements for A and 4 for B
    for (int kk = 0; kk < K; kk += TILE_K) {
        // Cooperative load A tile [TILE_M x TILE_K]
        // 1024 elements, 256 threads, 4 elements each
        for (int i = lid; i < TILE_M * TILE_K; i += WG_X * WG_Y) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int gr = tile_row + r;
            int gc = kk + c;
            As[i] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Cooperative load B tile [TILE_K x TILE_N]
        for (int i = lid; i < TILE_K * TILE_N; i += WG_X * WG_Y) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int gr = kk + r;
            int gc = tile_col + c;
            Bs[i] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: each thread accumulates its 2x2 block
        int r0 = ly * 2;
        int r1 = ly * 2 + 1;
        int c0 = lx * 2;
        int c1 = lx * 2 + 1;

        for (int tk = 0; tk < TILE_K; ++tk) {
            float a0 = convert_float(As[r0 * TILE_K + tk]);
            float a1 = convert_float(As[r1 * TILE_K + tk]);
            float b0 = convert_float(Bs[tk * TILE_N + c0]);
            float b1 = convert_float(Bs[tk * TILE_N + c1]);

            acc00 = fma(a0, b0, acc00);
            acc01 = fma(a0, b1, acc01);
            acc10 = fma(a1, b0, acc10);
            acc11 = fma(a1, b1, acc11);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store 2x2 results
    if (out_row0 < M && out_col0 < N)
        C[out_row0 * N + out_col0] = convert_half(acc00);
    if (out_row0 < M && (out_col0 + 1) < N)
        C[out_row0 * N + out_col0 + 1] = convert_half(acc01);
    if ((out_row0 + 1) < M && out_col0 < N)
        C[(out_row0 + 1) * N + out_col0] = convert_half(acc10);
    if ((out_row0 + 1) < M && (out_col0 + 1) < N)
        C[(out_row0 + 1) * N + out_col0 + 1] = convert_half(acc11);
}
```

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] PASSED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] PASSED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] PASSED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] PASSED         [100%]

======================= 4 passed, 1 deselected in 0.76s ========================
The kernel compiles and is correct, great job!

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
2. Minimize Global Memory Access: Cache frequently accessed values in private variables (registers). Use __constant address space for read-only data to enable hardware caching. Use __global const restrict for read-only buffers to hint compiler optimizations.

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