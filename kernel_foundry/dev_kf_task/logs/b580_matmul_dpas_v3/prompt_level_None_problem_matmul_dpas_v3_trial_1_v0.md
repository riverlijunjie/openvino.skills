

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Optimization goals:
- Use Intel OpenCL DPAS instruction when possible, e.g. intel_sub_group_f16_f16_matrix_mad_k16.
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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 230.000):
```OCL
// Tiled FP16 matmul with SLM blocking
// Each workgroup: 16x16 = 256 threads, computes 128x128 output tile
// Each thread computes 8x8 output elements
// K-tiles of size 32
//
// Recommended launch:
//   LWS = (16, 16, 1)
//   GWS = (ceil_div(N, 128)*16, ceil_div(M, 128)*16, 1)
//   Subgroup size: 16

#define TILE_M 128
#define TILE_N 128
#define TILE_K 32
#define THREAD_TILE_M 8
#define THREAD_TILE_N 8
#define WG_SIZE_X 16
#define WG_SIZE_Y 16

__attribute__((reqd_work_group_size(WG_SIZE_X, WG_SIZE_Y, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Workgroup tile origin
    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    // Local thread indices
    const int lx = get_local_id(0); // 0..15
    const int ly = get_local_id(1); // 0..15
    const int lid = ly * WG_SIZE_X + lx; // 0..255

    // Each thread's output tile position within the workgroup tile
    const int thread_row = ly * THREAD_TILE_M; // rows 0,8,16,...,120
    const int thread_col = lx * THREAD_TILE_N; // cols 0,8,16,...,120

    // Accumulators
    float acc[THREAD_TILE_M][THREAD_TILE_N];
    for (int i = 0; i < THREAD_TILE_M; i++)
        for (int j = 0; j < THREAD_TILE_N; j++)
            acc[i][j] = 0.0f;

    // SLM for A tile [TILE_M x TILE_K] and B tile [TILE_K x TILE_N]
    __local half A_slm[TILE_M * TILE_K];
    __local half B_slm[TILE_K * TILE_N];

    // Number of elements to load per thread
    // A_slm: 128*32 = 4096 halfs, 256 threads => 16 halfs/thread
    // B_slm: 32*128 = 4096 halfs, 256 threads => 16 halfs/thread

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperative load of A tile [TILE_M x TILE_K]
        // Each thread loads 16 halfs
        for (int i = 0; i < 16; i++) {
            int elem_idx = lid * 16 + i;
            int a_row = elem_idx / TILE_K;  // 0..127
            int a_col = elem_idx % TILE_K;  // 0..31
            int gm = wg_m + a_row;
            int gk = k0 + a_col;
            half val = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
            A_slm[a_row * TILE_K + a_col] = val;
        }

        // Cooperative load of B tile [TILE_K x TILE_N]
        for (int i = 0; i < 16; i++) {
            int elem_idx = lid * 16 + i;
            int b_row = elem_idx / TILE_N;  // 0..31
            int b_col = elem_idx % TILE_N;  // 0..127
            int gk = k0 + b_row;
            int gn = wg_n + b_col;
            half val = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            B_slm[b_row * TILE_N + b_col] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute partial results
        int actual_k = min(TILE_K, K - k0);
        for (int kk = 0; kk < TILE_K; kk++) {
            // Load A values for this thread's rows
            float a_val[THREAD_TILE_M];
            for (int i = 0; i < THREAD_TILE_M; i++) {
                a_val[i] = convert_float(A_slm[(thread_row + i) * TILE_K + kk]);
            }

            // Load B values for this thread's cols
            float b_val[THREAD_TILE_N];
            for (int j = 0; j < THREAD_TILE_N; j++) {
                b_val[j] = convert_float(B_slm[kk * TILE_N + thread_col + j]);
            }

            // Outer product accumulation
            for (int i = 0; i < THREAD_TILE_M; i++) {
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    acc[i][j] = fma(a_val[i], b_val[j], acc[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    for (int i = 0; i < THREAD_TILE_M; i++) {
        int gm = wg_m + thread_row + i;
        if (gm >= M) continue;
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int gn = wg_n + thread_col + j;
            if (gn >= N) continue;
            C[gm * N + gn] = convert_half(acc[i][j]);
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 65.800):
```OCL
// Tiled FP16 matmul using SLM blocking for Intel Xe2-HPG (Battlemage)
// C[M,N] = A[M,K] x B[K,N], accumulation in FP32
//
// Each workgroup computes a TILE_M x TILE_N block of C.
// TILE_M=64, TILE_N=64, TILE_K=16
// LWS = (16, 16) = 256 work-items
// Each work-item computes a 4x4 sub-tile of C.
// GWS = (ceil_div(N,64)*16, ceil_div(M,64)*16)

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define WG_X 16
#define WG_Y 16
#define ITEMS_PER_WI_M 4
#define ITEMS_PER_WI_N 4

__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Workgroup tile origin
    const int wg_id_x = get_group_id(0);
    const int wg_id_y = get_group_id(1);
    const int tile_col = wg_id_x * TILE_N;
    const int tile_row = wg_id_y * TILE_M;

    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lid = lid_y * WG_X + lid_x;

    // Each work-item accumulates a 4x4 sub-tile
    float acc[ITEMS_PER_WI_M][ITEMS_PER_WI_N];
    for (int i = 0; i < ITEMS_PER_WI_M; i++)
        for (int j = 0; j < ITEMS_PER_WI_N; j++)
            acc[i][j] = 0.0f;

    // SLM tiles for A and B
    __local half A_tile[TILE_M * TILE_K];  // 64 x 16
    __local half B_tile[TILE_K * TILE_N];  // 16 x 64

    const int total_wis = WG_X * WG_Y; // 256

    // Loop over K dimension in blocks of TILE_K
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperatively load A_tile[TILE_M][TILE_K] = 64*16 = 1024 elements
        // 256 work-items, each loads 4 elements
        for (int idx = lid; idx < TILE_M * TILE_K; idx += total_wis) {
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int global_r = tile_row + r;
            int global_c = k0 + c;
            half val = (global_r < M && global_c < K) ? A[global_r * K + global_c] : (half)0.0h;
            A_tile[r * TILE_K + c] = val;
        }

        // Cooperatively load B_tile[TILE_K][TILE_N] = 16*64 = 1024 elements
        for (int idx = lid; idx < TILE_K * TILE_N; idx += total_wis) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int global_r = k0 + r;
            int global_c = tile_col + c;
            half val = (global_r < K && global_c < N) ? B[global_r * N + global_c] : (half)0.0h;
            B_tile[r * TILE_N + c] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each WI computes its 4x4 sub-tile
        // WI (lid_x, lid_y) covers rows [lid_y*4 .. lid_y*4+3], cols [lid_x*4 .. lid_x*4+3]
        int local_row_base = lid_y * ITEMS_PER_WI_M;
        int local_col_base = lid_x * ITEMS_PER_WI_N;

        for (int kk = 0; kk < TILE_K; kk++) {
            // Load A values for this WI's rows
            float a_val[ITEMS_PER_WI_M];
            for (int i = 0; i < ITEMS_PER_WI_M; i++) {
                a_val[i] = convert_float(A_tile[(local_row_base + i) * TILE_K + kk]);
            }

            // Load B values for this WI's columns
            float b_val[ITEMS_PER_WI_N];
            for (int j = 0; j < ITEMS_PER_WI_N; j++) {
                b_val[j] = convert_float(B_tile[kk * TILE_N + local_col_base + j]);
            }

            // Outer product accumulation
            for (int i = 0; i < ITEMS_PER_WI_M; i++) {
                for (int j = 0; j < ITEMS_PER_WI_N; j++) {
                    acc[i][j] = fma(a_val[i], b_val[j], acc[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results back to C
    int local_row_base = lid_y * ITEMS_PER_WI_M;
    int local_col_base = lid_x * ITEMS_PER_WI_N;

    for (int i = 0; i < ITEMS_PER_WI_M; i++) {
        for (int j = 0; j < ITEMS_PER_WI_N; j++) {
            int global_r = tile_row + local_row_base + i;
            int global_c = tile_col + local_col_base + j;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = convert_half(acc[i][j]);
            }
        }
    }
}
```

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] PASSED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] PASSED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] PASSED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] PASSED         [100%]

======================= 4 passed, 1 deselected in 0.95s ========================
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
1. Minimize Global Memory Access: Cache frequently accessed values in private variables (registers). Use __constant address space for read-only data to enable hardware caching. Use __global const restrict for read-only buffers to hint compiler optimizations.
2. Coalesced Access: Ensure work-items in the same wavefront/warp access contiguous memory addresses. For row-major data, have adjacent work-items (consecutive get_global_id(0)) access adjacent memory locations. Use linear indexing that matches your data layout.

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