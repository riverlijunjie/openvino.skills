

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
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 34.000):
```OCL
// Intel Battlemage optimized FP16 GEMM with DPAS + double-buffering
//   C[M,N] = A[M,K] x B[K,N]
// dtype: A/B/C = half, accumulation = float
//
// Launch metadata (host-side):
//   LWS = {8, 8, 1}   // 64 WI = 4 subgroups (SG size 16)
//   GWS = {ceil_div(N, 64) * 8, ceil_div(M, 64) * 8, 1}
// Mapping:
//   - Each WG computes a 64×64 C tile
//   - Each WI computes 8×8 outputs in registers
//   - K processed in TILE_K=32 chunks with double-buffered SLM
//   - DPAS accumulates into live float8 registers across K-tiles
// Performance features:
//   - Register-cached SLM loads (no redundant reads)
//   - Full DPAS result utilization via float8 accumulators
//   - Pipelined memory/compute overlap
//   - 64 FP16 FMAs per thread per K-chunk (high arithmetic intensity)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define WG_X 8
#define WG_Y 8

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32

#define THREAD_M 8
#define THREAD_N 8

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
    const int lx = get_local_id(0);   // 0..7
    const int ly = get_local_id(1);   // 0..7
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);

    // WG tile origin in C
    const int c_row0 = gy * TILE_M + ly * THREAD_M;
    const int c_col0 = gx * TILE_N + lx * THREAD_N;

    // Double-buffered SLM tiles (padded to reduce bank conflicts)
    __local half Aslm[2][TILE_M][TILE_K + 1];
    __local half BslmT[2][TILE_N][TILE_K + 1];  // transposed B storage

    // Live DPAS accumulators (full float8 per output, extract lane 0 at end)
    float8 d_acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            d_acc[i][j] = (float8)(0.0f);
        }
    }

    const int k_tiles = (K + TILE_K - 1) / TILE_K;
    const int k_full_end = (K / TILE_K) * TILE_K;

    // Helper for cooperative loading (64 threads, each loads multiple elements)
    const int linear_id = ly * WG_X + lx;  // 0..63
    const int elems_per_tile = TILE_M * TILE_K;  // 64*32 = 2048
    const int loads_per_thread = (elems_per_tile + 63) / 64;  // 32

    // Prefetch first tile into buffer 0
    int buf = 0;
    int kb = 0;
    if (kb < k_full_end) {
        // Load A tile
        #pragma unroll 4
        for (int t = 0; t < loads_per_thread; ++t) {
            int idx = linear_id + t * 64;
            if (idx < elems_per_tile) {
                int ar = idx / TILE_K;
                int ak = idx % TILE_K;
                int g_r = gy * TILE_M + ar;
                int g_k = kb + ak;
                Aslm[buf][ar][ak] = (g_r < M && g_k < K) ? A[g_r * K + g_k] : (half)0.0h;
            }
        }

        // Load B tile (transposed into BslmT)
        #pragma unroll 4
        for (int t = 0; t < loads_per_thread; ++t) {
            int idx = linear_id + t * 64;
            if (idx < elems_per_tile) {
                int bk = idx / TILE_N;
                int bn = idx % TILE_N;
                int g_k = kb + bk;
                int g_n = gx * TILE_N + bn;
                BslmT[buf][bn][bk] = (g_k < K && g_n < N) ? B[g_k * N + g_n] : (half)0.0h;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K-tile loop with double-buffering
    for (int tile_idx = 0; tile_idx < k_tiles; ++tile_idx) {
        kb = tile_idx * TILE_K;
        if (kb >= K) break;

        buf = tile_idx % 2;
        int next_buf = 1 - buf;
        int next_kb = kb + TILE_K;

        // Async prefetch next tile while computing current (if not last)
        if (next_kb < k_full_end && tile_idx + 1 < k_tiles) {
            // Load A next tile
            #pragma unroll 4
            for (int t = 0; t < loads_per_thread; ++t) {
                int idx = linear_id + t * 64;
                if (idx < elems_per_tile) {
                    int ar = idx / TILE_K;
                    int ak = idx % TILE_K;
                    int g_r = gy * TILE_M + ar;
                    int g_k = next_kb + ak;
                    Aslm[next_buf][ar][ak] = (g_r < M && g_k < K) ? A[g_r * K + g_k] : (half)0.0h;
                }
            }

            // Load B next tile
            #pragma unroll 4
            for (int t = 0; t < loads_per_thread; ++t) {
                int idx = linear_id + t * 64;
                if (idx < elems_per_tile) {
                    int bk = idx / TILE_N;
                    int bn = idx % TILE_N;
                    int g_k = next_kb + bk;
                    int g_n = gx * TILE_N + bn;
                    BslmT[next_buf][bn][bk] = (g_k < K && g_n < N) ? B[g_k * N + g_n] : (half)0.0h;
                }
            }
        }

        // Compute current tile from current buffer
        // Cache SLM vectors in registers to avoid redundant loads
        half16 a_cache[THREAD_M];
        half16 b_cache[THREAD_N];

        // Process first half of K-tile (k=0..15)
        {
            // Load A rows for this thread's M-block
            #pragma unroll
            for (int mi = 0; mi < THREAD_M; ++mi) {
                int ar = ly * THREAD_M + mi;
                a_cache[mi] = vload16(0, &Aslm[buf][ar][0]);
            }

            // Load B cols for this thread's N-block
            #pragma unroll
            for (int nj = 0; nj < THREAD_N; ++nj) {
                int bc = lx * THREAD_N + nj;
                b_cache[nj] = vload16(0, &BslmT[buf][bc][0]);
            }

            // DPAS compute: 8×8 outputs per thread
            #pragma unroll
            for (int mi = 0; mi < THREAD_M; ++mi) {
                #pragma unroll
                for (int nj = 0; nj < THREAD_N; ++nj) {
                    d_acc[mi][nj] = intel_sub_group_f16_f16_matrix_mad_k16(
                        a_cache[mi], b_cache[nj], d_acc[mi][nj]);
                }
            }
        }

        // Process second half of K-tile (k=16..31) if full tile
        if (kb + TILE_K <= K) {
            #pragma unroll
            for (int mi = 0; mi < THREAD_M; ++mi) {
                int ar = ly * THREAD_M + mi;
                a_cache[mi] = vload16(0, &Aslm[buf][ar][16]);
            }

            #pragma unroll
            for (int nj = 0; nj < THREAD_N; ++nj) {
                int bc = lx * THREAD_N + nj;
                b_cache[nj] = vload16(0, &BslmT[buf][bc][16]);
            }

            #pragma unroll
            for (int mi = 0; mi < THREAD_M; ++mi) {
                #pragma unroll
                for (int nj = 0; nj < THREAD_N; ++nj) {
                    d_acc[mi][nj] = intel_sub_group_f16_f16_matrix_mad_k16(
                        a_cache[mi], b_cache[nj], d_acc[mi][nj]);
                }
            }
        }

        // Barrier for buffer swap (single barrier per K-tile)
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Extract DPAS results and write to global memory with tail-K cleanup
    #pragma unroll
    for (int mi = 0; mi < THREAD_M; ++mi) {
        int r = c_row0 + mi;
        if (r >= M) continue;

        #pragma unroll
        for (int nj = 0; nj < THREAD_N; ++nj) {
            int c = c_col0 + nj;
            if (c >= N) continue;

            // Extract accumulated result from DPAS (lane 0 of float8)
            float v = d_acc[mi][nj].s0;

            // Handle tail K (elements not covered by full TILE_K chunks)
            for (int k = k_full_end; k < K; ++k) {
                v = mad(convert_float(A[r * K + k]), 
                       convert_float(B[k * N + c]), v);
            }

            C[r * N + c] = convert_half(v);
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.  

(II) Bottlenecks and suggestions for improvement:  
1. `float8 d_acc[THREAD_M][THREAD_N];` and `d_acc[mi][nj] = intel_sub_group_f16_f16_matrix_mad_k16(..., d_acc[mi][nj]);`  
   You’re carrying **64 float8 accumulators per work-item** (`8x8`), which is extremely register-heavy and likely causing register spilling / reduced occupancy. Also, you only use `.s0` at the end (`float v = d_acc[mi][nj].s0;`), so most of each `float8` is wasted for your final mapping.  
   **Improve it by reducing per-thread output tile size** (e.g., `THREAD_M=4, THREAD_N=4`), or remap so each DPAS result vector lanes are all consumed (lane-mapped outputs instead of scalar `.s0` extraction). This will significantly cut register pressure and improve EU occupancy.

2. `__local half Aslm[2][TILE_M][TILE_K + 1];` and `__local half BslmT[2][TILE_N][TILE_K + 1];`  
   With double buffering and `+1` padding in both arrays, SLM footprint is large (~16.5 KB), and each tile step does full cooperative reload. This is not terrible, but it increases SLM traffic and can become a limiter versus DPAS throughput.  
   **Improve it by removing unnecessary padding where bank conflicts are already avoided by access pattern** (often only one operand needs skew), and by using a narrower `TILE_K` (e.g., 16) if it improves overlap/latency hiding on Battlemage. Benchmark `(TILE_K=16, no/less skew)` vs current; on many Intel GPUs it reduces SLM pressure and improves scheduling.

3. Tail handling in epilogue:  
   `for (int k = k_full_end; k < K; ++k) { v = mad(convert_float(A[r * K + k]), convert_float(B[k * N + c]), v); }`  
   This does **scalar global-memory FMAs per output element**, so every thread re-reads A/B from global for tail K. For small/medium K remainders this can dominate epilogue cost and disrupt vectorized flow.  
   **Improve it by handling tail K inside the tiled loop via zero-padded SLM loads** (you already zero-pad on load), and always run fixed k-steps (`0..15`, `16..31`) guarded by padded zeros instead of scalar fallback. Then remove this per-element tail loop entirely. That keeps computation in DPAS path and avoids scattered global reads in the writeback phase.

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
1. Use Kernel Attributes: Use __attribute__((reqd_work_group_size(X,Y,Z))) to specify fixed work-group size for compiler optimization. Use __attribute__((vec_type_hint(float4))) to hint vectorization opportunities.
2. Exploit Sub-groups (OpenCL 2.0+): Use sub-group functions for wavefront-level operations: sub_group_reduce_add(), sub_group_broadcast(), intel_sub_group_shuffle(), intel_sub_group_shuffle_down(). Query sub-group size with get_sub_group_size(). Use __attribute__((intel_reqd_sub_group_size(N))) for Intel GPUs.

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
- **Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade recomputation for memory savings (e.g., Flash-Attention style). Maintain running accumulators across blocks with proper rescaling.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.