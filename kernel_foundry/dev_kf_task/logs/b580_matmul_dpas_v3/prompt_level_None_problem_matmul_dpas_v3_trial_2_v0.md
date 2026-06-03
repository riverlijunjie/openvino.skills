

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 172.000):
```OCL
// Tiled FP16 matmul with vectorized loads, large tiles
// C[M,N] = A[M,K] x B[K,N], accumulation in FP32
//
// TILE_M=128, TILE_N=128, TILE_K=16
// LWS = (16, 16) = 256 work-items
// Each WI computes 8x8 sub-tile of C
// GWS = (ceil_div(N,128)*16, ceil_div(M,128)*16)

#define TM 128
#define TN 128
#define TK 16
#define WG_X 16
#define WG_Y 16
#define WI_M 8
#define WI_N 8

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
    const int tile_col = get_group_id(0) * TN;
    const int tile_row = get_group_id(1) * TM;

    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lid = lid_y * WG_X + lid_x;

    // Accumulators: 8x8 per WI
    float acc[WI_M][WI_N];
    #pragma unroll
    for (int i = 0; i < WI_M; i++)
        #pragma unroll
        for (int j = 0; j < WI_N; j++)
            acc[i][j] = 0.0f;

    // SLM tiles
    __local half A_slm[TM * TK];   // 128*16 = 2048
    __local half B_slm[TK * TN];   // 16*128 = 2048

    const int total_wis = WG_X * WG_Y; // 256

    // Per-WI output position in tile
    const int row_base = lid_y * WI_M;
    const int col_base = lid_x * WI_N;

    for (int k0 = 0; k0 < K; k0 += TK) {
        // Load A_slm[TM][TK] = 2048 elements, 256 WIs => 8 each
        #pragma unroll
        for (int idx = lid; idx < TM * TK; idx += total_wis) {
            int r = idx >> 4;  // /16
            int c = idx & 15;  // %16
            int gr = tile_row + r;
            int gc = k0 + c;
            A_slm[idx] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Load B_slm[TK][TN] = 2048 elements
        #pragma unroll
        for (int idx = lid; idx < TK * TN; idx += total_wis) {
            int r = idx >> 7;  // /128
            int c = idx & 127; // %128
            int gr = k0 + r;
            int gc = tile_col + c;
            B_slm[idx] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 8x8 sub-tile
        #pragma unroll
        for (int kk = 0; kk < TK; kk++) {
            float a_val[WI_M];
            #pragma unroll
            for (int i = 0; i < WI_M; i++) {
                a_val[i] = convert_float(A_slm[(row_base + i) * TK + kk]);
            }

            float b_val[WI_N];
            #pragma unroll
            for (int j = 0; j < WI_N; j++) {
                b_val[j] = convert_float(B_slm[kk * TN + col_base + j]);
            }

            #pragma unroll
            for (int i = 0; i < WI_M; i++) {
                #pragma unroll
                for (int j = 0; j < WI_N; j++) {
                    acc[i][j] = fma(a_val[i], b_val[j], acc[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < WI_M; i++) {
        int gr = tile_row + row_base + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < WI_N; j++) {
            int gc = tile_col + col_base + j;
            if (gc < N) {
                C[gr * N + gc] = convert_half(acc[i][j]);
            }
        }
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 177.000):
```OCL
// Tiled FP16 matmul: C[M,N] = A[M,K] x B[K,N], accumulation in FP32
// TILE_M=128, TILE_N=128, TILE_K=16
// LWS = (16, 16) = 256 work-items
// Each WI computes 8x8 sub-tile of C
// GWS = (ceil_div(N,128)*16, ceil_div(M,128)*16)
// SLM: A_tile = 128*16 = 2048 halfs (4KB), B_tile = 16*128 = 2048 halfs (4KB), total 8KB

#define TILE_M 128
#define TILE_N 128
#define TILE_K 16
#define WG_X 16
#define WG_Y 16
#define TM 8
#define TN 8

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

    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lid = lid_y * WG_X + lid_x;

    // Accumulators in registers
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    __local half A_tile[TILE_M * TILE_K]; // 128*16 = 2048
    __local half B_tile[TILE_K * TILE_N]; // 16*128 = 2048

    const int total_wis = WG_X * WG_Y; // 256
    // A_tile: 2048 elems / 256 WIs = 8 per WI
    // B_tile: 2048 elems / 256 WIs = 8 per WI

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A_tile cooperatively - 8 elements per WI
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int idx = lid + t * total_wis;
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int gr = tile_row + r;
            int gc = k0 + c;
            A_tile[idx] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Load B_tile cooperatively - 8 elements per WI
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int idx = lid + t * total_wis;
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gr = k0 + r;
            int gc = tile_col + c;
            B_tile[idx] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute - each WI does 8x8 sub-tile
        int row_base = lid_y * TM;
        int col_base = lid_x * TN;

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            float a_val[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                a_val[i] = convert_float(A_tile[(row_base + i) * TILE_K + kk]);
            }

            float b_val[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                b_val[j] = convert_float(B_tile[kk * TILE_N + col_base + j]);
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] = fma(a_val[i], b_val[j], acc[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    int row_base = lid_y * TM;
    int col_base = lid_x * TN;

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gr = tile_row + row_base + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gc = tile_col + col_base + j;
            if (gc < N) {
                C[gr * N + gc] = convert_half(acc[i][j]);
            }
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
1. Avoid Bank Conflicts: Local memory is organized into banks (typically 32 banks). Pad shared arrays to avoid stride conflicts, e.g., __local float tile[TILE_SIZE][TILE_SIZE + 1] for transpose operations. Use sequential access patterns within wavefronts.
2. Use Appropriate Data Types: Use half (FP16) when precision allows - doubles throughput on many GPUs. Use native_* math functions (native_sqrt, native_divide) for faster, less precise operations. Use mad() for fused multiply-add. Size integer types appropriately (char, short, int).

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