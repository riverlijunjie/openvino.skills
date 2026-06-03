

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Launch metadata:
//   LWS = {16, 1, 1}  (1 subgroup of 16 lanes)
//   GWS = {ceil_div(N,16)*16, ceil_div(M,16), 1}
//   Each WG = 1 subgroup computing a 16×16 C tile
//   2 DPAS calls per K-chunk: rows[0:7] and rows[8:15]
//   Each lane = 1 column, float8 = 8 row results

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__attribute__((reqd_work_group_size(16, 1, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lane = get_local_id(0);
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);

    const int tile_row = gy * TILE_M;
    const int tile_col = gx * TILE_N;
    const int col = tile_col + lane;

    // Accumulators: rows 0-7 and 8-15
    float8 acc_lo = (float8)(0.0f);
    float8 acc_hi = (float8)(0.0f);

    // SLM for A tile [16][16] and B column vectors
    __local half Asub[TILE_M][TILE_K];
    __local half Bsub[TILE_K][TILE_N];

    const int k_full_end = (K / TILE_K) * TILE_K;

    for (int kb = 0; kb < k_full_end; kb += TILE_K) {
        // Load A: 16×16 = 256 elems, 16 threads => 16 each
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            int ar = t;
            int ak = lane;
            int gr = tile_row + ar;
            int gk = kb + ak;
            Asub[ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // Load B: 16×16 = 256 elems, 16 threads => 16 each
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            int bk = t;
            int bc = lane;
            int gk = kb + bk;
            int gc = tile_col + bc;
            Bsub[bk][bc] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // DPAS for rows 0-7: each lane reads A row for its "virtual row"
        // But DPAS broadcasts across subgroup - lane i's a_vec contributes to row i
        // For rows 0-7: lane i uses A[tile_row+i][kb:kb+15]
        // For rows 8-15: lane i uses A[tile_row+8+i][kb:kb+15]
        // b_vec: each lane loads B[:,col] = 16 k-values for its column

        half16 b_vec;
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            ((half*)&b_vec)[kk] = Bsub[kk][lane];
        }

        // Rows 0-7
        {
            int my_row = lane; // lane 0-15, but DPAS only uses lanes 0-7 for a
            half16 a_vec = vload16(0, &Asub[lane < 8 ? lane : 0][0]);
            acc_lo = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc_lo);
        }

        // Rows 8-15
        {
            half16 a_vec = vload16(0, &Asub[lane < 8 ? lane + 8 : 8][0]);
            acc_hi = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc_hi);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results + tail K
    if (col < N) {
        // Rows 0-7
        float acc_arr[8];
        acc_arr[0] = acc_lo.s0; acc_arr[1] = acc_lo.s1;
        acc_arr[2] = acc_lo.s2; acc_arr[3] = acc_lo.s3;
        acc_arr[4] = acc_lo.s4; acc_arr[5] = acc_lo.s5;
        acc_arr[6] = acc_lo.s6; acc_arr[7] = acc_lo.s7;

        for (int r = 0; r < 8; ++r) {
            int gr = tile_row + r;
            if (gr < M) {
                float v = acc_arr[r];
                for (int k = k_full_end; k < K; ++k)
                    v += convert_float(A[gr * K + k]) * convert_float(B[k * N + col]);
                C[gr * N + col] = convert_half(v);
            }
        }

        // Rows 8-15
        acc_arr[0] = acc_hi.s0; acc_arr[1] = acc_hi.s1;
        acc_arr[2] = acc_hi.s2; acc_arr[3] = acc_hi.s3;
        acc_arr[4] = acc_hi.s4; acc_arr[5] = acc_hi.s5;
        acc_arr[6] = acc_hi.s6; acc_arr[7] = acc_hi.s7;

        for (int r = 0; r < 8; ++r) {
            int gr = tile_row + 8 + r;
            if (gr < M) {
                float v = acc_arr[r];
                for (int k = k_full_end; k < K; ++k)
                    v += convert_float(A[gr * K + k]) * convert_float(B[k * N + col]);
                C[gr * N + col] = convert_half(v);
            }
        }
    }
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
// Suggested launch metadata (host-side):
//   - Local size (LWS):  {16, 16, 1}  (256 threads / work-group)
//   - Global size (GWS): {ceil_div(N,16)*16, ceil_div(M,16)*16, 1}
//   - Subgroup hint:     reqd_sub_group_size(16)
// Mapping:
//   - One work-group computes one 16x16 C tile at (group_y, group_x)
//   - K processed in chunks of 16
// Notes:
//   - Fast path uses Intel DPAS intrinsic on full tiles
//   - Boundary and tail-K handled by exact scalar accumulation

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__attribute__((reqd_work_group_size(16,16,1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int lx = get_local_id(0);   // 0..15
    const int ly = get_local_id(1);   // 0..15
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);

    const int row = gy * TILE_M + ly;
    const int col = gx * TILE_N + lx;

    // SLM tiles (padding to reduce bank conflicts)
    __local half Asub[TILE_M][TILE_K + 1];
    __local half Bsub[TILE_K][TILE_N + 1];

    float acc = 0.0f;

    // Number of full TILE_K chunks
    const int k_full_end = (K / TILE_K) * TILE_K;

    // Process full K tiles
    for (int kb = 0; kb < k_full_end; kb += TILE_K) {
        // Cooperative load A tile
        {
            const int a_r = row;
            const int a_c = kb + lx;
            Asub[ly][lx] = (a_r < M && a_c < K) ? A[a_r * K + a_c] : (half)0.0h;
        }

        // Cooperative load B tile
        {
            const int b_r = kb + ly;
            const int b_c = col;
            Bsub[ly][lx] = (b_r < K && b_c < N) ? B[b_r * N + b_c] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Fast full-tile DPAS path only when this C element is in-bounds.
        // We still keep exact behavior by using scalar path equivalence.
        if (row < M && col < N) {
            // Build packed operands for DPAS from SLM rows/cols.
            // Using vector loads to aid compiler generation of XMX instructions.
            half16 a_vec = vload16(0, &Asub[ly][0]); // A(row, kb:kb+15)
            half16 b_vec;
            // Gather B(k, col) as a vector.
            #pragma unroll
            for (int kk = 0; kk < 16; ++kk) {
                b_vec.s[kk] = Bsub[kk][lx];
            }

            // Accumulator vector type for DPAS intrinsic.
            // We use a 1-lane logical output and extract lane 0.
            float8 dpas_acc = (float8)(0.0f);
            // Intel DPAS: f16 x f16, k=16 accumulate into float.
            // Signature availability may vary by compiler version; this is the intended intrinsic.
            dpas_acc = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, dpas_acc);
            acc += dpas_acc.s0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Tail K (exact scalar cleanup)
    if (row < M && col < N) {
        #pragma unroll
        for (int k = k_full_end; k < K; ++k) {
            acc += convert_float(A[row * K + k]) * convert_float(B[k * N + col]);
        }
        C[row * N + col] = convert_half(acc);
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
// Intel Xe2 (Battlemage) tuned FP16 GEMM
// C[M,N] = A[M,K] x B[K,N]
// A/B/C: half, accumulation: float (reference-equivalent)
//
// Suggested launch metadata (host-side):
//   LWS = {16, 16, 1}   // 256 WI/WG, SG size 16
//   GWS = {ceil_div(N, 32) * 16, ceil_div(M, 32) * 16, 1}
// Subgroup hint:
//   __attribute__((intel_reqd_sub_group_size(16)))
//
// Mapping:
//   - One WG computes a 32x32 tile of C.
//   - One WI computes 2x2 outputs (register blocking).
//   - K is blocked by TILE_K=32 (two DPAS k16 ops per block).
//
// Key optimizations:
//   1) Register blocking (2x2 acc per WI)
//   2) SLM tiling with cooperative vectorized loads
//   3) Hoisted SLM->register vectors to avoid redundant reloads
//   4) DPAS intrinsic in hot loop + exact scalar tail handling

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

#define WG_X 16
#define WG_Y 16

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

#define THREAD_M 2
#define THREAD_N 2

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
    const int lx = get_local_id(0);   // 0..15
    const int ly = get_local_id(1);   // 0..15
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);

    const int c_row0 = gy * TILE_M + ly * THREAD_M; // row base this WI computes
    const int c_col0 = gx * TILE_N + lx * THREAD_N; // col base this WI computes

    // +1 padding to reduce bank conflicts
    __local half Aslm[TILE_M][TILE_K + 1];
    __local half BslmT[TILE_N][TILE_K + 1]; // transposed: [n][k]

    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const int k_full_end = (K / TILE_K) * TILE_K;

    for (int kb = 0; kb < k_full_end; kb += TILE_K) {
        // Cooperative load A tile (32x32 = 1024 halves = 256 * half4)
        {
            int linear = ly * WG_X + lx; // 0..255
            int ar = linear >> 3;         // /8 -> 0..31
            int a4 = linear & 7;          // 0..7
            int ak = a4 << 2;             // 0,4,...,28

            int g_r = gy * TILE_M + ar;
            int g_k = kb + ak;

            half4 v = (half4)(0.0h, 0.0h, 0.0h, 0.0h);
            if (g_r < M) {
                half t0 = (g_k + 0 < K) ? A[g_r * K + (g_k + 0)] : (half)0.0h;
                half t1 = (g_k + 1 < K) ? A[g_r * K + (g_k + 1)] : (half)0.0h;
                half t2 = (g_k + 2 < K) ? A[g_r * K + (g_k + 2)] : (half)0.0h;
                half t3 = (g_k + 3 < K) ? A[g_r * K + (g_k + 3)] : (half)0.0h;
                v = (half4)(t0, t1, t2, t3);
            }

            vstore4(v, 0, &Aslm[ar][ak]);
        }

        // Cooperative load B tile and transpose to BslmT (32x32)
        {
            int linear = ly * WG_X + lx; // 0..255
            int bk = linear >> 3;         // 0..31
            int b4 = linear & 7;          // 0..7
            int bn = b4 << 2;             // 0,4,...,28

            int g_k = kb + bk;
            int g_n = gx * TILE_N + bn;

            half4 v = (half4)(0.0h, 0.0h, 0.0h, 0.0h);
            if (g_k < K) {
                half t0 = (g_n + 0 < N) ? B[g_k * N + (g_n + 0)] : (half)0.0h;
                half t1 = (g_n + 1 < N) ? B[g_k * N + (g_n + 1)] : (half)0.0h;
                half t2 = (g_n + 2 < N) ? B[g_k * N + (g_n + 2)] : (half)0.0h;
                half t3 = (g_n + 3 < N) ? B[g_k * N + (g_n + 3)] : (half)0.0h;
                v = (half4)(t0, t1, t2, t3);
            }

            // transpose while storing: BslmT[n][k]
            BslmT[bn + 0][bk] = v.s0;
            BslmT[bn + 1][bk] = v.s1;
            BslmT[bn + 2][bk] = v.s2;
            BslmT[bn + 3][bk] = v.s3;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Two k16 DPAS steps within TILE_K=32.
        #pragma unroll
        for (int kk0 = 0; kk0 < TILE_K; kk0 += 16) {
            // Hoist A/B vectors to registers once per kk0
            half16 a_reg[THREAD_M];
            half16 b_reg[THREAD_N];

            #pragma unroll
            for (int mi = 0; mi < THREAD_M; ++mi) {
                const int ar = ly * THREAD_M + mi; // 0..31
                a_reg[mi] = vload16(0, &Aslm[ar][kk0]);
            }

            #pragma unroll
            for (int nj = 0; nj < THREAD_N; ++nj) {
                const int bc = lx * THREAD_N + nj; // 0..31
                b_reg[nj] = vload16(0, &BslmT[bc][kk0]);
            }

            // DPAS on 2x2 microtile
            #pragma unroll
            for (int mi = 0; mi < THREAD_M; ++mi) {
                #pragma unroll
                for (int nj = 0; nj < THREAD_N; ++nj) {
                    // Keep compiler-friendly DPAS form used by prior passing kernels.
                    float8 d = (float8)(0.0f);
                    d = intel_sub_group_f16_f16_matrix_mad_k16(a_reg[mi], b_reg[nj], d);
                    acc[mi][nj] += d.s0;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Exact tail-K cleanup + store
    #pragma unroll
    for (int mi = 0; mi < THREAD_M; ++mi) {
        const int r = c_row0 + mi;
        if (r >= M) continue;

        #pragma unroll
        for (int nj = 0; nj < THREAD_N; ++nj) {
            const int c = c_col0 + nj;
            if (c >= N) continue;

            float v = acc[mi][nj];
            for (int k = k_full_end; k < K; ++k) {
                v += convert_float(A[r * K + k]) * convert_float(B[k * N + c]);
            }
            C[r * N + c] = convert_half(v);
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.  

(II) Bottlenecks and suggestions for improvement:  
1. `float8 d = (float8)(0.0f); d = intel_sub_group_f16_f16_matrix_mad_k16(a_reg[mi], b_reg[nj], d); acc[mi][nj] += d.s0;`  
   You’re throwing away 7/8 of the DPAS result every time (`d.s1..d.s7`), so each DPAS does much more work than what you accumulate. That is a major compute-efficiency loss.  
   **Improve it by matching your micro-tile mapping to the DPAS output lanes**: either (a) have each lane own one DPAS output element and accumulate that lane’s scalar directly (no `float8` temp), or (b) enlarge per-thread output tile and map all 8 DPAS outputs into real accumulators. Right now the DPAS intrinsic is underutilized.

2. `for (int kb = 0; kb < k_full_end; kb += TILE_K) { ... barrier(...); ... barrier(...); }` with full-tile load into `Aslm`/`BslmT` then compute  
   You serialize load+compute phases with two barriers per K-block and no overlap. On Xe2 this often leaves memory latency exposed.  
   **Improve it with double-buffered SLM (ping-pong)**: allocate `Aslm[2][...]`, `BslmT[2][...]`, preload tile `kb+TILE_K` while computing tile `kb`, then swap buffers. This hides global/SLM load latency and reduces stall time at barriers.

3. `half t0 = ...; half t1 = ...; half t2 = ...; half t3 = ...;` (both A and B cooperative loads)  
   Your global loads are scalarized behind per-element bounds checks, which increases instruction count and hurts coalescing/vectorization.  
   **Improve boundary handling by splitting fast/edge paths**:  
   - Fast path for full interior tiles (`gy*TILE_M+31 < M`, `gx*TILE_N+31 < N`, and `kb+31 < K`) with unconditional vector loads (`vload8`/`vload16` where alignment allows).  
   - Separate slow edge kernel/branch for boundary tiles with masking.  
   This keeps the hot path branch-free and vector-friendly, which is where most runtime is spent.

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

- **Register Blocking**: Each work-item computes a THREAD_M×THREAD_N output block in private register arrays. Use `#pragma unroll` on inner loops. Combine with SLM tiling for multi-level memory hierarchy optimization.
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.