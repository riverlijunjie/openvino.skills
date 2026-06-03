

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
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Optimized FP16 GEMM using Intel DPAS (XMX) on Battlemage Xe2-HPG
// C[M,N] = A[M,K] x B[K,N], all half, accumulation in float
//
// Each subgroup (16 work-items) computes an 8x16 tile of C using
// intel_sub_group_f16_f16_matrix_mad_k16.
//
// Launch configuration:
//   LWS = (16, SG_PER_WG)  e.g. (16, 4) → 64 WIs, 4 subgroups
//   GWS = (ceil(N/16)*16, ceil(M/(8*SG_PER_WG))*SG_PER_WG)
//   Subgroup size: 16
//
// Each work-group computes (8*SG_PER_WG) x 16 of C.
// With SG_PER_WG=4: 32 x 16 per work-group.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int sg_lid = get_sub_group_local_id();  // 0..15
    const int sg_id  = get_sub_group_id();        // subgroup within work-group

    // Tile position in output C
    const int tile_col = get_group_id(0) * 16;                          // N dimension
    const int tile_row = get_group_id(1) * (int)get_local_size(1) * 8   // M dimension
                       + sg_id * 8;

    // Early exit for out-of-bounds work-groups
    if (tile_row >= M || tile_col >= N) return;

    const int out_col = tile_col + sg_lid;

    // Accumulator: 8 rows, each work-item holds one column
    float8 acc = (float8)(0.0f);

    // K-loop in steps of 16 (DPAS k-dimension)
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        // ---- Load A tile [8 x 16] ----
        // DPAS A layout: int8 per work-item
        // work-item sg_lid holds, for each of 8 rows:
        //   { A[row][k + 2*sg_lid], A[row][k + 2*sg_lid+1] } packed as int
        int8 a_val;
        int a_k = k + 2 * sg_lid;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row = tile_row + r;
            // Use vload2 for aligned pair load
            ushort u0 = row < M ? as_ushort(A[row * K + a_k])     : (ushort)0;
            ushort u1 = row < M ? as_ushort(A[row * K + a_k + 1]) : (ushort)0;
            int packed = (int)u0 | ((int)u1 << 16);
            // Can't index int8 dynamically, unroll manually
            switch(r) {
                case 0: a_val.s0 = packed; break;
                case 1: a_val.s1 = packed; break;
                case 2: a_val.s2 = packed; break;
                case 3: a_val.s3 = packed; break;
                case 4: a_val.s4 = packed; break;
                case 5: a_val.s5 = packed; break;
                case 6: a_val.s6 = packed; break;
                case 7: a_val.s7 = packed; break;
            }
        }

        // ---- Load B tile [16 x 16] ----
        // DPAS B layout: int8 per work-item
        // work-item sg_lid holds column (tile_col + sg_lid) for pairs of k-rows
        // b[j] = { B[k+2*j][col], B[k+2*j+1][col] } packed as int
        int8 b_val;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int kr0 = k + 2 * j;
            int kr1 = kr0 + 1;
            ushort bv0 = (out_col < N) ? as_ushort(B[kr0 * N + out_col]) : (ushort)0;
            ushort bv1 = (out_col < N) ? as_ushort(B[kr1 * N + out_col]) : (ushort)0;
            int packed = (int)bv0 | ((int)bv1 << 16);
            switch(j) {
                case 0: b_val.s0 = packed; break;
                case 1: b_val.s1 = packed; break;
                case 2: b_val.s2 = packed; break;
                case 3: b_val.s3 = packed; break;
                case 4: b_val.s4 = packed; break;
                case 5: b_val.s5 = packed; break;
                case 6: b_val.s6 = packed; break;
                case 7: b_val.s7 = packed; break;
            }
        }

        // ---- DPAS: 8x16 matmul accumulate ----
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    }

    // Handle remaining K elements (K not multiple of 16) with scalar fallback
    if (k < K) {
        for (int kk = k; kk < K; kk++) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int row = tile_row + r;
                if (row < M && out_col < N) {
                    float a_elem = convert_float(A[row * K + kk]);
                    float b_elem = convert_float(B[kk * N + out_col]);
                    switch(r) {
                        case 0: acc.s0 += a_elem * b_elem; break;
                        case 1: acc.s1 += a_elem * b_elem; break;
                        case 2: acc.s2 += a_elem * b_elem; break;
                        case 3: acc.s3 += a_elem * b_elem; break;
                        case 4: acc.s4 += a_elem * b_elem; break;
                        case 5: acc.s5 += a_elem * b_elem; break;
                        case 6: acc.s6 += a_elem * b_elem; break;
                        case 7: acc.s7 += a_elem * b_elem; break;
                    }
                }
            }
        }
    }

    // ---- Store 8x1 column of results ----
    if (out_col < N) {
        if (tile_row + 0 < M) C[(tile_row + 0) * N + out_col] = convert_half(acc.s0);
        if (tile_row + 1 < M) C[(tile_row + 1) * N + out_col] = convert_half(acc.s1);
        if (tile_row + 2 < M) C[(tile_row + 2) * N + out_col] = convert_half(acc.s2);
        if (tile_row + 3 < M) C[(tile_row + 3) * N + out_col] = convert_half(acc.s3);
        if (tile_row + 4 < M) C[(tile_row + 4) * N + out_col] = convert_half(acc.s4);
        if (tile_row + 5 < M) C[(tile_row + 5) * N + out_col] = convert_half(acc.s5);
        if (tile_row + 6 < M) C[(tile_row + 6) * N + out_col] = convert_half(acc.s6);
        if (tile_row + 7 < M) C[(tile_row + 7) * N + out_col] = convert_half(acc.s7);
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel **is incorrect**.

(II) Summary of console output: All 4 correctness tests failed, and the produced output matrix is entirely zeros while PyTorch/reference outputs are non-zero. The key clue is:
`[WARN] matmul_opt.cl has no parsable reqd_work_group_size; fallback launch (N, M), local=None`.
That means your kernel was launched with a fallback NDRange that does not match your subgroup/tile mapping assumptions, so your indexing/work decomposition is effectively broken at runtime.

(III) Errors and suggestions for improvement:
1. `__attribute__((intel_reqd_sub_group_size(16)))` (kernel declaration): this is not enough for your mapping because your code assumes a **fixed local shape** (`LWS=(16, SG_PER_WG)`) and derives tile rows from `get_local_size(1)`. Add a required workgroup-size attribute (e.g. `reqd_work_group_size(16,4,1)`) and hard-code the corresponding tile math, or rewrite indexing so it is correct for arbitrary local size. Right now fallback `local=None` breaks your decomposition.

2. `const int tile_row = get_group_id(1) * (int)get_local_size(1) * 8 + sg_id * 8;`: this is tightly coupled to your intended local size and subgroup packing. Under fallback launch, `get_local_size(1)` won’t be your SG-per-WG design point, so row tiles are miscomputed. Use a compile-time constant `SG_PER_WG` in both launch contract and indexing, and guard kernel assumptions with required workgroup size attributes so scheduler/runtime cannot violate them.

3. `int a_k = k + 2 * sg_lid;` with loads `A[row*K + a_k]` and `A[row*K + a_k + 1]`: no bounds guard for `a_k+1 < K` inside DPAS loop body. You rely on `k+16<=K`, which is fine only if subgroup width and lane mapping are exactly as assumed; if subgroup semantics differ or compiler lowers differently, this can become unsafe. Make the DPAS path robust by explicitly guarding packed loads (zero-fill when out-of-range), same style as your tail path.

4. `acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);`: your operand packing/layout must exactly match intrinsic contract. Given all-zero output, verify the expected A/B fragment layout for Battlemage’s DPAS intrinsic variant; if layout is wrong, accumulation can be garbage/zero. In practice, switch to the exact documented fragment construction pattern (including lane/component mapping) and avoid ad-hoc `switch` packing unless it matches spec verbatim.

5. `switch`-based vector element assignment for `int8 a_val/b_val`: this is fragile and may lead to uninitialized lanes if compiler optimization/path pruning goes wrong. Replace with explicit vector constructors from precomputed scalars (`(int8)(p0,p1,...,p7)`) so every element is definitely initialized before DPAS.

6. Store path (`convert_half(acc.sX)`): you use default conversion mode. For numerical stability/consistency with reference tolerance, use explicit rounding mode (`convert_half_rte`) and consider saturation behavior only if needed. This won’t fix zero-output, but improves deterministic correctness once execution mapping is fixed.

7. Overall launch-contract dependency: because you cannot change host/test launcher, make kernel resilient to `local=None` by deriving work solely from global IDs (no subgroup-id dependent tiling assumptions), or add required work-group-size attributes that the harness can parse. Right now your algorithm is “correct only for a specific launch,” but tests run it with another launch shape, causing functional failure.

## Hardware specification:
Your code will run on the following hardware:
**Intel Battlemage** with specs: Xe-cores: 20, Render Slices: 5, Ray Tracing Units: 20, Intel® Xe Matrix Extensions (Intel® XMX) Engines: 160, Xe Vector Engines: 160, Graphics Clock: 2670, GPU Peak TOPS (Int8): 233, TBP: 190, PCI Express Configurations ‡: PCI Express 4.0 x8, Device ID: 0xE20B, Memory: 12 GB GDDR6, Memory Interface: 192 bit, Memory Bandwidth: 456, Memory Speed: 19, ISA_GPU: Xe2-HPG
Please consider the hardware specifications when improving the code. 

## Task:

**Your objectives**:
1. Analyze the previous kernel and its evaluation log.
2. Identify any errors or mismatches with the reference implementation.
3. Propose specific improvements or fixes, explaining your reasoning step by step.
4. Rewrite the kernel, providing the complete, corrected code in a code block.

**Critical Requirements:**

1. The kernel must exactly match the reference implementation's functionality.
2. The code must compile and run properly on the GPU.
3. Do not cache or reuse previous results; ensure the code executes fully on each run.
4. Keep all hyperparameters (e.g., batch size, dimensions) unchanged as specified in the reference implementation.
8. Beware of the critical error "Unexpected kernel lambda size. In such cases removing constexpr specifier aligns the captures between the host compiler and the device compiler"! Do not capture constexpr variables in lambda functions passed to kernel launches as this can lead to different lambda sizes between the host and device compiler.

Additional Guidance:

1. Clearly comment on any fixes and optimizations you implement.
2. If you change the kernel structure, explain why.
3. Anticipate and address possible runtime or compilation errors.

Please structure your response as follows:

1. Analysis:
    * Summarize the errors found in the previous kernel and log.
    * Explain your proposed changes.
2. Improved OCL code:
    * Provide the complete, corrected OCL code in a code block:
```OCL
Your code here
```


## Required Optimizations

Apply the following optimization techniques in your implementation:

- **Vectorized Memory Access**: Replace scalar loads with SYCL vectors (`float4` for general use, `float8`/`float16` for bulk transfers, `float2` for tight register pressure). Ensure adjacent work-items access adjacent memory addresses for coalesced access.
- **Kernel Fusion**: Combine sequential operations (e.g., exp → add → activation) into a single kernel. Eliminate intermediate buffers by computing in registers.
- **Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing.