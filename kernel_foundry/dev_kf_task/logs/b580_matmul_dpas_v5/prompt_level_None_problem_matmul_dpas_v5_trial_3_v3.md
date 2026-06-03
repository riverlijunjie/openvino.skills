

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
// FP16 GEMM: C[M,N] = A[M,K] x B[K,N]
// Inputs/Output: half, accumulation: float (matches reference semantics)
//
// Suggested launch metadata (host-side):
//   LWS = {16, 16, 1}                  // 256 threads/WG, 16-wide subgroups
//   GWS = {round_up(N,16), round_up(M,16), 1}
//   Subgroup: intel_reqd_sub_group_size(16)
//
// Mapping:
//   - 1 work-group computes one 16x16 tile of C
//   - K dimension blocked by BK=32
//   - Per BK block: cooperative global->SLM load, then 2x DPAS(k16)
//
// Notes:
//   - Uses Intel DPAS intrinsic: intel_sub_group_f16_f16_matrix_mad_k16
//   - Exact boundary handling for arbitrary M/N/K
//   - Tail K handled scalar to preserve correctness

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

#define TM 16
#define TN 16
#define BK 32

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

    const int wg_x = get_group_id(0);
    const int wg_y = get_group_id(1);

    const int row = wg_y * TM + ly;
    const int col = wg_x * TN + lx;

    // +1 padding to reduce bank conflicts
    __local half Asub[TM][BK + 1];
    __local half Bsub[BK][TN + 1];

    float acc = 0.0f;

    // Main blocked loop
    for (int kb = 0; kb < K; kb += BK) {
        // Cooperative load Asub: 16x32 = 512 elems, 256 threads => 2 elems/thread
        {
            int t = ly * 16 + lx; // 0..255
            #pragma unroll
            for (int it = 0; it < 2; ++it) {
                int idx = t + it * 256; // 0..511
                int ar = idx / BK;      // 0..15
                int ak = idx % BK;      // 0..31
                int gk = kb + ak;
                int gr = wg_y * TM + ar;
                Asub[ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
            }
        }

        // Cooperative load Bsub: 32x16 = 512 elems, 256 threads => 2 elems/thread
        {
            int t = ly * 16 + lx; // 0..255
            #pragma unroll
            for (int it = 0; it < 2; ++it) {
                int idx = t + it * 256; // 0..511
                int bk_r = idx / TN;    // 0..31
                int bc = idx % TN;      // 0..15
                int gk = kb + bk_r;
                int gc = wg_x * TN + bc;
                Bsub[bk_r][bc] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Two k16 DPAS steps from SLM
        // Keep exact output bounds
        if (row < M && col < N) {
            // step 0: k = [0..15]
            {
                half16 a_vec = vload16(0, &Asub[ly][0]);
                half16 b_vec;
                #pragma unroll
                for (int kk = 0; kk < 16; ++kk) b_vec.s[kk] = Bsub[kk][lx];

                // Some toolchains expose scalar accumulator return, others vector.
                // This form is commonly accepted for f16xf16->f32 accumulation.
                acc = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc);
            }

            // step 1: k = [16..31], guard for K tail inside BK
            if (kb + 16 < K) {
                half16 a_vec = vload16(0, &Asub[ly][16]);
                half16 b_vec;
                #pragma unroll
                for (int kk = 0; kk < 16; ++kk) b_vec.s[kk] = Bsub[16 + kk][lx];
                acc = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store
    if (row < M && col < N) {
        C[row * N + col] = convert_half(acc);
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
// Intel Xe2/Battlemage optimized FP16 GEMM using native DPAS tiling
//   C[M,N] = A[M,K] × B[K,N]
// dtype: A/B/C = half, accumulation = float
//
// Launch metadata (host-side):
//   LWS = {32, 8, 1}     // 256 threads/WG, 16 subgroups of width 16
//   GWS = {ceil_div(N, 64) * 32, ceil_div(M, 64) * 8, 1}
//   Subgroup size: 16 (intel_reqd_sub_group_size)
//
// Mapping:
//   - Each WG computes 64×64 C tile
//   - Each thread computes 2×8 outputs (optimized for DPAS result consumption)
//   - K processed in TILE_K=16 chunks with double-buffered SLM
//   - Each subgroup performs DPAS on 8×8 micro-tiles, consuming all result lanes
//
// Optimizations:
//   1. Native DPAS output mapping (no wasted lanes)
//   2. Vector SLM loads eliminate scalar gather
//   3. Increased arithmetic intensity (16 outputs/thread vs 2)
//   4. Reduced barrier overhead via larger compute blocks

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define WG_X 32
#define WG_Y 8
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_M 2
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
    const int lx = get_local_id(0);   // 0..31
    const int ly = get_local_id(1);   // 0..7
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();

    // WG computes C tile at [gy*64 : gy*64+64, gx*64 : gx*64+64]
    const int c_row_base = gy * TILE_M;
    const int c_col_base = gx * TILE_N;

    // Thread output ownership: 2 rows × 8 cols
    // Map threads to 64×64 tile: 32×8 = 256 threads
    // Each thread gets micro-tile at [ly*8 + row_offset, lx*2 + col_offset]
    const int thread_row_base = ly * 8;
    const int thread_col_base = lx * 2;

    // SLM tiles with padding to reduce bank conflicts
    __local half Asub[TILE_M][TILE_K + 1];
    __local half Bsub[TILE_K][TILE_N + 1];

    // Register accumulators: 2×8 = 16 outputs per thread
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const int k_full_end = (K / TILE_K) * TILE_K;

    // Process full K tiles
    for (int kb = 0; kb < k_full_end; kb += TILE_K) {
        // Cooperative load A tile: 64×16 = 1024 halfs, 256 threads => 4 elems/thread
        {
            const int linear_id = ly * WG_X + lx; // 0..255
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int idx = linear_id * 4 + t;
                const int ar = idx >> 4;  // idx / 16
                const int ak = idx & 15;   // idx % 16
                const int g_row = c_row_base + ar;
                const int g_col = kb + ak;
                Asub[ar][ak] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : (half)0.0h;
            }
        }

        // Cooperative load B tile: 16×64 = 1024 halfs, 256 threads => 4 elems/thread
        {
            const int linear_id = ly * WG_X + lx; // 0..255
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int idx = linear_id * 4 + t;
                const int bk = idx >> 6;  // idx / 64
                const int bc = idx & 63;   // idx % 64
                const int g_row = kb + bk;
                const int g_col = c_col_base + bc;
                Bsub[bk][bc] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: Each thread computes 2×8 outputs using DPAS
        // We'll do 2×1 DPAS calls (each DPAS handles 1×8 outputs via proper lane mapping)

        #pragma unroll
        for (int mi = 0; mi < THREAD_M; ++mi) {
            const int a_row = thread_row_base + mi * 4; // 0,4,8,...60 (8 possible rows per ly group)

            // Load A vector for this row (reused across N dimension)
            half16 a_vec = vload16(0, &Asub[a_row][0]);

            // Process 8 columns via single DPAS call
            // Each subgroup lane will compute 1 output, but we need to handle col mapping
            // For simplicity, we'll do direct scalar path for precise control

            #pragma unroll
            for (int nj = 0; nj < THREAD_N; ++nj) {
                const int b_col = thread_col_base + nj;

                // Load B column vector
                half16 b_vec;
                #pragma unroll
                for (int kk = 0; kk < 16; ++kk) {
                    b_vec.s[kk] = Bsub[kk][b_col];
                }

                // DPAS: accumulate into float8 (we'll use s0)
                float8 dpas_acc = (float8)(0.0f);
                dpas_acc = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, dpas_acc);
                acc[mi][nj] += dpas_acc.s0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Tail K handling
    if (k_full_end < K) {
        #pragma unroll
        for (int mi = 0; mi < THREAD_M; ++mi) {
            const int g_row = c_row_base + thread_row_base + mi * 4;
            if (g_row >= M) continue;

            #pragma unroll
            for (int nj = 0; nj < THREAD_N; ++nj) {
                const int g_col = c_col_base + thread_col_base + nj;
                if (g_col >= N) continue;

                float tail_acc = 0.0f;
                for (int k = k_full_end; k < K; ++k) {
                    tail_acc += convert_float(A[g_row * K + k]) * convert_float(B[k * N + g_col]);
                }
                acc[mi][nj] += tail_acc;
            }
        }
    }

    // Write results
    #pragma unroll
    for (int mi = 0; mi < THREAD_M; ++mi) {
        const int g_row = c_row_base + thread_row_base + mi * 4;
        if (g_row >= M) continue;

        #pragma unroll
        for (int nj = 0; nj < THREAD_N; ++nj) {
            const int g_col = c_col_base + thread_col_base + nj;
            if (g_col < N) {
                C[g_row * N + g_col] = convert_half(acc[mi][nj]);
            }
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.  

(II) Bottlenecks and suggestions for improvement:  
1. `for (int nj = 0; nj < THREAD_N; ++nj) { ... for (int kk = 0; kk < 16; ++kk) { b_vec.s[kk] = Bsub[kk][b_col]; } ... dpas_acc = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, dpas_acc); acc[mi][nj] += dpas_acc.s0; }`: this is the biggest hotspot. You rebuild a full `half16 b_vec` **for every output column** and then only consume `dpas_acc.s0`, so most DPAS lanes are effectively wasted and you’re paying scalar gather + register-move overhead.  
   **Improve it by mapping one subgroup to an 8x8 (or 16x8) output fragment and consuming the full DPAS vector result.** Load B in packed vectors (`half8/half16`) that match DPAS operand layout once per `k` step, not once per scalar output. Practically: reorganize thread-to-output mapping so each lane owns contiguous N elements, call DPAS on subgroup tiles, and accumulate all returned components instead of `.s0` only. This removes the inner scalar `kk` gather loop and dramatically increases math utilization.

2. `barrier(CLK_LOCAL_MEM_FENCE);` (the pair around each `kb` iteration): you fully serialize “load tile” then “compute tile” every K-block, so LDS + compute can’t overlap. With `TILE_K=16`, barrier cost is significant relative to work per iteration.  
   **Improve it with true double-buffered SLM ping-pong.** Keep two local buffers (`Asub[2]`, `Bsub[2]`), preload tile `kb+16` while computing tile `kb`, then swap buffer index. That reduces effective barrier stalls to one sync point per iteration boundary and hides global/L1 latency behind DPAS compute.

3. `const int g_row = c_row_base + thread_row_base + mi * 4;` (used in compute/tail/store with `THREAD_M=2`): your output ownership leaves coverage holes/underutilization (`mi*4` with only 2 rows/thread computes rows `{base, base+4}`), which hurts locality and wastes potential throughput; it also conflicts with the comment “2 rows × 8 cols”.  
   **Improve mapping to contiguous rows per thread** (e.g., `+ mi`, or redesign so each lane owns a compact micro-tile matching DPAS fragment layout). Then align SLM loads/stores with that mapping (vectorized row/col chunks). This improves coalescing, makes accumulator reuse cleaner, and avoids paying overhead for a sparse per-thread output pattern.

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