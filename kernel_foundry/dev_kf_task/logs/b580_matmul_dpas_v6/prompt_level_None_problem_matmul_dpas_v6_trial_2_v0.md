

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

### Version 1 (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Optimized FP16 GEMM using Intel DPAS (intel_sub_group_f16_f16_matrix_mad_k16)
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Tile: 64 rows x 32 cols per work-group
// Subgroup size: 16
// WG layout: 2 subgroups in N (covers 32 cols), 2 rows of subgroups (covers 64 rows)
// Each subgroup computes 32x16 (THREAD_M_TILES=4 DPAS ops, each 8x16)
// K-tile: 16 (matches DPAS k=16)
//
// LWS = (32, 2, 1)  => 64 work-items = 4 subgroups
// GWS = (ceil(N/32)*32, ceil(M/64)*2, 1)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_TILE_M 64
#define WG_TILE_N 32
#define SG_TILE_M 32
#define SG_TILE_N 16
#define DPAS_M 8
#define DPAS_K 16
#define DPAS_N 16
#define THREAD_M_TILES 4
#define TILE_K 16

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Identify subgroup position within workgroup
    const int sg_id = get_sub_group_id();         // 0..3
    const int sg_lane = get_sub_group_local_id(); // 0..15

    // Subgroup grid within WG: 2 wide (N) x 2 tall (M)
    const int sg_col = sg_id % 2;  // 0 or 1
    const int sg_row = sg_id / 2;  // 0 or 1

    // Work-group tile origin
    const int wg_m = (get_group_id(1) * 2 + 0) * SG_TILE_M;  // get_group_id(1) indexes WG rows
    // Actually let me recompute based on GWS mapping
    // get_global_id(0) spans N, get_global_id(1) spans M/32
    const int wg_origin_n = (get_group_id(0) / 2) * WG_TILE_N;  // groups of 2 subgroups in x
    const int wg_origin_m = get_group_id(1) * WG_TILE_M;

    // This subgroup's output tile origin
    const int tile_m = wg_origin_m + sg_row * SG_TILE_M;
    const int tile_n = wg_origin_n + sg_col * SG_TILE_N;

    // Accumulator registers: THREAD_M_TILES dpas tiles, each float8
    float8 acc[THREAD_M_TILES];
    #pragma unroll
    for (int t = 0; t < THREAD_M_TILES; t++) {
        acc[t] = (float8)(0.0f);
    }

    // SLM for A and B tiles
    // A tile: WG_TILE_M x TILE_K = 64 x 16 halfs
    // B tile: TILE_K x WG_TILE_N = 16 x 32 halfs
    __local half A_slm[WG_TILE_M * TILE_K];
    __local half B_slm[TILE_K * WG_TILE_N];

    const int local_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int local_size = get_local_size(0) * get_local_size(1); // 64

    // Loop over K dimension
    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        // Cooperative load A[wg_origin_m .. +64, k0 .. +16] into SLM
        // 64*16 = 1024 halfs, 64 threads => 16 halfs each
        #pragma unroll
        for (int i = local_id; i < WG_TILE_M * TILE_K; i += local_size) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gm = wg_origin_m + row;
            int gk = k0 + col;
            A_slm[i] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
        }

        // Cooperative load B[k0 .. +16, wg_origin_n .. +32] into SLM
        // 16*32 = 512 halfs, 64 threads => 8 halfs each
        #pragma unroll
        for (int i = local_id; i < TILE_K * WG_TILE_N; i += local_size) {
            int row = i / WG_TILE_N;
            int col = i % WG_TILE_N;
            int gk = k0 + row;
            int gn = wg_origin_n + col;
            B_slm[i] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each subgroup reads its portion and does DPAS
        // B for this subgroup: 16 rows x 16 cols starting at col offset sg_col*16
        // Read B into registers: DPAS expects B distributed across subgroup
        // B layout for DPAS: int8 where each int packs 2 half values
        // B is K=16 rows x N=16 cols, each lane holds one column, 16 rows = 8 ints (2 halfs per int)
        int8 b_reg;
        __local const uint* B_slm_uint = (__local const uint*)(B_slm);
        // B_slm is [TILE_K][WG_TILE_N] in half
        // For sg_col subgroup, cols [sg_col*16 .. sg_col*16+15]
        // Lane sg_lane reads column (sg_col*16 + sg_lane)
        // Row r: B_slm[r * WG_TILE_N + sg_col*16 + sg_lane]
        // Pack pairs of rows into uint
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int r0 = 2 * i;
            int r1 = 2 * i + 1;
            ushort v0 = as_ushort(B_slm[r0 * WG_TILE_N + sg_col * DPAS_N + sg_lane]);
            ushort v1 = as_ushort(B_slm[r1 * WG_TILE_N + sg_col * DPAS_N + sg_lane]);
            ((uint*)&b_reg)[i] = ((uint)v1 << 16) | (uint)v0;
        }

        // For each DPAS M-tile
        #pragma unroll
        for (int t = 0; t < THREAD_M_TILES; t++) {
            // A for this dpas: 8 rows x 16 cols
            // Rows: sg_row * SG_TILE_M + t * DPAS_M .. + 8
            // A_slm layout: [WG_TILE_M][TILE_K]
            int a_row_base = sg_row * SG_TILE_M + t * DPAS_M;
            int8 a_reg;
            // DPAS A: each work-item holds data for its row(s)
            // For A: int8 where each int packs 2 halfs along K
            // Each of 8 rows distributed across the 8 components
            // Within each row, 16 K values = 8 uint (pairs of half)
            // But these are shared across subgroup via broadcast
            // Actually for DPAS: A is replicated, each lane has same A data
            // A[8 rows][16 K] packed as int8, each int = 2 halfs along K
            // Lane sg_lane reads K columns [sg_lane*1...] — no, A is broadcast

            // For intel_sub_group_f16_f16_matrix_mad_k16:
            // A: int8 — 8 rows, each row has K=16 halfs packed into 8 ints
            //    But spread across subgroup: each lane holds one pair of K values per row
            //    Lane l holds K indices [2l, 2l+1] for all 8 rows
            // So a_reg[row] = pack(A[row, 2*sg_lane], A[row, 2*sg_lane+1])

            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int slm_row = a_row_base + r;
                ushort v0 = as_ushort(A_slm[slm_row * TILE_K + 2 * sg_lane]);
                ushort v1 = as_ushort(A_slm[slm_row * TILE_K + 2 * sg_lane + 1]);
                ((uint*)&a_reg)[r] = ((uint)v1 << 16) | (uint)v0;
            }

            acc[t] = intel_sub_group_f16_f16_matrix_mad_k16(a_reg, b_reg, acc[t]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    // Each work-item in subgroup holds 8 values per DPAS tile for its column (tile_n + sg_lane)
    int out_col = tile_n + sg_lane;

    #pragma unroll
    for (int t = 0; t < THREAD_M_TILES; t++) {
        int row_base = tile_m + t * DPAS_M;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int out_row = row_base + r;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = convert_half(((float*)&acc[t])[r]);
            }
        }
    }
}
```

### Version 2 (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Optimized FP16 matmul using Intel DPAS (XMX) instructions
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Tiling: WG tile = 32x64 (MxN), DPAS tile = 8x16, K-tile = 16
// Each WG has 16 subgroups (4 along M, 4 along N), subgroup size = 16
// LWS = (64, 4) = 256 work-items
// GWS = (ceil(N/64)*64, ceil(M/32)*4)
// SLM usage: A_tile[32][16] + B_tile[16][64] = 1024 + 2048 = 3072 bytes (half)

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 4, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile origin in C
    const int wg_n = (get_group_id(0)) * 64;  // N offset
    const int wg_m = (get_group_id(1)) * 32;  // M offset

    // Local IDs
    const int lid_x = get_local_id(0); // 0..63
    const int lid_y = get_local_id(1); // 0..3
    const int lid = lid_y * 64 + lid_x; // flat local id 0..255

    // Subgroup identification
    const int sg_id = get_sub_group_id();        // 0..15
    const int sg_lid = get_sub_group_local_id();  // 0..15

    // Map 16 subgroups into 4x4 grid (4 along M, 4 along N)
    const int sg_m = sg_id / 4;  // 0..3, each covers 8 rows
    const int sg_n = sg_id % 4;  // 0..3, each covers 16 cols

    // Subgroup's tile origin within WG tile
    const int sg_row_offset = sg_m * 8;   // 0, 8, 16, 24
    const int sg_col_offset = sg_n * 16;  // 0, 16, 32, 48

    // SLM tiles
    __local half A_slm[32 * 16];  // 32 rows x 16 cols
    __local half B_slm[16 * 64];  // 16 rows x 64 cols

    // Accumulator: 8 floats per work-item (8 rows of the 8x16 DPAS output)
    float8 acc = (float8)(0.0f);

    // Loop over K dimension in tiles of 16
    for (int k_base = 0; k_base < K; k_base += 16) {
        // Cooperative load of A tile [32][16] into SLM
        // 256 work-items, 32*16 = 512 elements, 2 elements per work-item
        {
            int idx0 = lid * 2;
            int idx1 = lid * 2 + 1;

            int a_row0 = idx0 / 16;
            int a_col0 = idx0 % 16;
            int a_row1 = idx1 / 16;
            int a_col1 = idx1 % 16;

            int gm0 = wg_m + a_row0;
            int gk0 = k_base + a_col0;
            int gm1 = wg_m + a_row1;
            int gk1 = k_base + a_col1;

            A_slm[idx0] = (gm0 < M && gk0 < K) ? A[gm0 * K + gk0] : (half)0.0h;
            A_slm[idx1] = (gm1 < M && gk1 < K) ? A[gm1 * K + gk1] : (half)0.0h;
        }

        // Cooperative load of B tile [16][64] into SLM
        // 256 work-items, 16*64 = 1024 elements, 4 elements per work-item
        {
            for (int i = 0; i < 4; i++) {
                int idx = lid + i * 256;
                int b_row = idx / 64;
                int b_col = idx % 64;

                int gk = k_base + b_row;
                int gn = wg_n + b_col;

                B_slm[idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each subgroup performs DPAS: 8x16 = 8x16_k16 matmul
        // Load A sub-tile: 8 rows, 16 cols from A_slm
        // For DPAS a input: each work-item i holds A[sg_row_offset + row][col] 
        // packed as 8 halfs (8 rows, work-item's K-column = sg_lid-th pair)

        // a: each work-item holds 8 half values = rows 0..7, at k-columns
        // In intel_sub_group_f16_f16_matrix_mad_k16:
        //   a is int8 (8 x int, each int packs 2 half values) representing 8x16 matrix
        //   b is int8 representing 16x16 matrix
        //   Result is float8 (8 rows x 1 col per work-item, 16 work-items = 8x16)

        // Pack A: 8 rows x 16 cols, stored as int8 per work-item
        // Each work-item sg_lid reads column pair (sg_lid*2, sg_lid*2+1) -- but that's not right
        // Actually for DPAS, data layout:
        //   a[i] for work-item j: a is distributed across subgroup
        //   Each work-item holds 8 ints, each int = 2 halfs from K dimension
        //   work-item j holds k-indices [2j, 2j+1] for all 8 rows

        int8 a_packed;
        for (int r = 0; r < 8; r++) {
            int slm_row = sg_row_offset + r;
            int k0 = sg_lid * 2;
            int k1 = sg_lid * 2 + 1;
            half va0 = A_slm[slm_row * 16 + k0];
            half va1 = A_slm[slm_row * 16 + k1];
            // Pack two halfs into one int
            ushort u0 = as_ushort(va0);
            ushort u1 = as_ushort(va1);
            ((int*)&a_packed)[r] = (int)u0 | ((int)u1 << 16);
        }

        // Pack B: 16 rows x 16 cols (the subgroup's 16-col slice)
        // Each work-item j holds k-indices [2j, 2j+1] for all 16 output columns
        // But B is organized differently: 
        //   b[i] for work-item j: work-item j holds column j, rows packed in pairs
        //   Each int = 2 halfs from K dimension at column sg_col_offset + sg_lid

        int8 b_packed;
        for (int p = 0; p < 8; p++) {
            int k0 = p * 2;
            int k1 = p * 2 + 1;
            int col = sg_col_offset + sg_lid;
            half vb0 = B_slm[k0 * 64 + col];
            half vb1 = B_slm[k1 * 64 + col];
            ushort ub0 = as_ushort(vb0);
            ushort ub1 = as_ushort(vb1);
            ((int*)&b_packed)[p] = (int)ub0 | ((int)ub1 << 16);
        }

        // DPAS: 8x16 result = 8x16(A) * 16x16(B)
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: each work-item writes 8 values
    // Work-item sg_lid in subgroup writes column (sg_col_offset + sg_lid)
    // rows sg_row_offset + 0..7
    int out_col = wg_n + sg_col_offset + sg_lid;

    for (int r = 0; r < 8; r++) {
        int out_row = wg_m + sg_row_offset + r;
        if (out_row < M && out_col < N) {
            float val = ((float*)&acc)[r];
            C[out_row * N + out_col] = convert_half(val);
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
1. `float b_val = convert_float(B_slm[kk * TILE_N + lx]);` and `float a_val = convert_float(A_slm[(ly * THREAD_M + r) * TILE_K + kk]);`: you are doing FP16→FP32 scalar conversion inside the innermost FMA loop, which is repeated `TILE_K * THREAD_M` times per thread per tile. That conversion overhead is significant.  
   **Improve it:** vectorize and hoist conversions. For example, load `half8` from `A_slm` for the 8 rows handled by the thread, convert once to `float8`, and reuse for the current `kk`; similarly load `B` as packed types where possible. This reduces instruction count and improves ILP. If your target supports matrix ops (`cl_intel_subgroups`, WMMA-like paths), map this loop to subgroup/matrix instructions so conversion+FMA happens in hardware pipelines.

2. `A_slm[i] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;` and `B_slm[i] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;`: cooperative loads are scalar and involve per-element boundary branches, which hurts memory throughput and causes divergence on edge tiles.  
   **Improve it:** split kernel into **main tile path** (no bounds checks, vectorized global loads like `vload8/vload16`) and **tail path** (handles boundaries). In the main path, each thread should copy contiguous vectors from global to SLM; this both coalesces accesses and reduces branch/control overhead. Also pad/align SLM layouts to enable vector stores into local memory.

3. `barrier(CLK_LOCAL_MEM_FENCE);` (before and after compute in each `k0` iteration): the loop is strictly load → barrier → compute → barrier, so all threads stall twice per K-tile with no overlap of memory and compute.  
   **Improve it:** use double-buffered SLM (ping-pong): while computing on buffer 0, prefetch next K-tile into buffer 1, then swap. This removes one full synchronization phase from the critical path and hides global-memory latency. Structure it as preload tile 0, then for each `k0`: prefetch `k0+TILE_K` into other buffer, compute current buffer, single barrier for buffer swap. This is usually one of the biggest runtime wins for tiled GEMM kernels.

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
1. Exploit Constant Cache: Use __constant address space (limited to ~64KB) for frequently accessed read-only data. Declare as __constant float coeffs[N]. All work-items should access same location simultaneously for best performance.
2. Loop Unrolling: Use #pragma unroll N for small, fixed-iteration loops. Manually unroll critical loops when compiler doesn't optimize. Prefer compile-time loop bounds.

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
- **Kernel Fusion**: Combine sequential operations (e.g., exp → add → activation) into a single kernel. Eliminate intermediate buffers by computing in registers.
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.