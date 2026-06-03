

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
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 34.000):
```OCL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Launch metadata:
//   LWS = {16, 4, 1}  (64 threads = 4 subgroups of 16 lanes)
//   GWS = {ceil_div(N,64)*16, ceil_div(M,32)*4, 1}
//   Each WG computes a 32x64 C tile
//   Each subgroup computes 32x16 (4 DPAS of 8x16 per K-chunk)
//   TILE_K = 16, double-buffered SLM

#define TILE_M 32
#define TILE_N 64
#define TILE_K 16
#define WG_X 16
#define WG_Y 4
#define NUM_THREADS (WG_X * WG_Y)  // 64

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
    const int lx = get_local_id(0);   // 0..15 lane
    const int ly = get_local_id(1);   // 0..3  subgroup index
    const int gx = get_group_id(0);   // N-tile index
    const int gy = get_group_id(1);   // M-tile index

    const int tile_row = gy * TILE_M;
    const int tile_col = gx * TILE_N;

    // Each subgroup ly handles columns [tile_col + ly*16 .. tile_col + ly*16 + 15]
    const int sg_col_base = tile_col + ly * 16;
    const int my_col = sg_col_base + lx;

    // Each subgroup computes ALL 32 rows for its 16 columns
    // 4 DPAS calls per K-chunk: rows 0-7, 8-15, 16-23, 24-31
    float8 acc0 = (float8)(0.0f);  // rows 0-7
    float8 acc1 = (float8)(0.0f);  // rows 8-15
    float8 acc2 = (float8)(0.0f);  // rows 16-23
    float8 acc3 = (float8)(0.0f);  // rows 24-31

    // Double-buffered SLM
    // A: [2][TILE_M][TILE_K] = 2*32*16 = 1024 halves = 2KB
    // B: [2][TILE_K][TILE_N] = 2*16*64 = 2048 halves = 4KB
    // Total: 6KB - fits easily in SLM
    __local half Asub[2][TILE_M][TILE_K];
    __local half Bsub[2][TILE_K][TILE_N];

    const int linear_id = ly * WG_X + lx;  // 0..63
    const int k_full_end = (K / TILE_K) * TILE_K;
    const int num_k_tiles = k_full_end / TILE_K;

    // Load first tile (buffer 0)
    int kb = 0;
    if (kb < k_full_end) {
        // Load A[32][16] = 512 elems, 64 threads => 8 each
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int idx = linear_id + t * NUM_THREADS;
            int ar = idx / TILE_K;
            int ak = idx % TILE_K;
            int gr = tile_row + ar;
            int gk = kb + ak;
            Asub[0][ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }
        // Load B[16][64] = 1024 elems, 64 threads => 16 each
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            int idx = linear_id + t * NUM_THREADS;
            int bk = idx / TILE_N;
            int bn = idx % TILE_N;
            int gk = kb + bk;
            int gc = tile_col + bn;
            Bsub[0][bk][bn] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int ki = 0; ki < num_k_tiles; ++ki) {
        int cur = ki & 1;
        int nxt = 1 - cur;
        int next_kb = (ki + 1) * TILE_K;

        // Prefetch next tile into nxt buffer
        if (next_kb < k_full_end) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                int idx = linear_id + t * NUM_THREADS;
                int ar = idx / TILE_K;
                int ak = idx % TILE_K;
                int gr = tile_row + ar;
                int gk = next_kb + ak;
                Asub[nxt][ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
            }
            #pragma unroll
            for (int t = 0; t < 16; ++t) {
                int idx = linear_id + t * NUM_THREADS;
                int bk = idx / TILE_N;
                int bn = idx % TILE_N;
                int gk = next_kb + bk;
                int gc = tile_col + bn;
                Bsub[nxt][bk][bn] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
            }
        }

        // Compute on current buffer
        // B vector for this subgroup's columns
        half16 b_vec;
        b_vec.s0 = Bsub[cur][0][sg_col_base - tile_col + lx];
        b_vec.s1 = Bsub[cur][1][sg_col_base - tile_col + lx];
        b_vec.s2 = Bsub[cur][2][sg_col_base - tile_col + lx];
        b_vec.s3 = Bsub[cur][3][sg_col_base - tile_col + lx];
        b_vec.s4 = Bsub[cur][4][sg_col_base - tile_col + lx];
        b_vec.s5 = Bsub[cur][5][sg_col_base - tile_col + lx];
        b_vec.s6 = Bsub[cur][6][sg_col_base - tile_col + lx];
        b_vec.s7 = Bsub[cur][7][sg_col_base - tile_col + lx];
        b_vec.s8 = Bsub[cur][8][sg_col_base - tile_col + lx];
        b_vec.s9 = Bsub[cur][9][sg_col_base - tile_col + lx];
        b_vec.sa = Bsub[cur][10][sg_col_base - tile_col + lx];
        b_vec.sb = Bsub[cur][11][sg_col_base - tile_col + lx];
        b_vec.sc = Bsub[cur][12][sg_col_base - tile_col + lx];
        b_vec.sd = Bsub[cur][13][sg_col_base - tile_col + lx];
        b_vec.se = Bsub[cur][14][sg_col_base - tile_col + lx];
        b_vec.sf = Bsub[cur][15][sg_col_base - tile_col + lx];

        // 4 DPAS calls for rows 0-7, 8-15, 16-23, 24-31
        // For DPAS: lane i (i<8) provides A row data, lanes 8-15 mirror lanes 0-7
        {
            half16 a_vec = vload16(0, &Asub[cur][lx % 8][0]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc0);
        }
        {
            half16 a_vec = vload16(0, &Asub[cur][8 + (lx % 8)][0]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc1);
        }
        {
            half16 a_vec = vload16(0, &Asub[cur][16 + (lx % 8)][0]);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc2);
        }
        {
            half16 a_vec = vload16(0, &Asub[cur][24 + (lx % 8)][0]);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results + tail K
    if (my_col < N) {
        float acc_arr[32];
        acc_arr[0] = acc0.s0; acc_arr[1] = acc0.s1; acc_arr[2] = acc0.s2; acc_arr[3] = acc0.s3;
        acc_arr[4] = acc0.s4; acc_arr[5] = acc0.s5; acc_arr[6] = acc0.s6; acc_arr[7] = acc0.s7;
        acc_arr[8] = acc1.s0; acc_arr[9] = acc1.s1; acc_arr[10] = acc1.s2; acc_arr[11] = acc1.s3;
        acc_arr[12] = acc1.s4; acc_arr[13] = acc1.s5; acc_arr[14] = acc1.s6; acc_arr[15] = acc1.s7;
        acc_arr[16] = acc2.s0; acc_arr[17] = acc2.s1; acc_arr[18] = acc2.s2; acc_arr[19] = acc2.s3;
        acc_arr[20] = acc2.s4; acc_arr[21] = acc2.s5; acc_arr[22] = acc2.s6; acc_arr[23] = acc2.s7;
        acc_arr[24] = acc3.s0; acc_arr[25] = acc3.s1; acc_arr[26] = acc3.s2; acc_arr[27] = acc3.s3;
        acc_arr[28] = acc3.s4; acc_arr[29] = acc3.s5; acc_arr[30] = acc3.s6; acc_arr[31] = acc3.s7;

        #pragma unroll
        for (int r = 0; r < 32; ++r) {
            int gr = tile_row + r;
            if (gr < M) {
                float v = acc_arr[r];
                for (int k = k_full_end; k < K; ++k) {
                    v += convert_float(A[gr * K + k]) * convert_float(B[k * N + my_col]);
                }
                C[gr * N + my_col] = convert_half(v);
            }
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.

(II) Bottlenecks and suggestions for improvement:
1. `barrier(CLK_LOCAL_MEM_FENCE);` (inside the `for (int ki = 0; ki < num_k_tiles; ++ki)` loop): this full workgroup barrier every K-tile is expensive, and your current “prefetch next + compute current” is not truly overlapped because both use the same threads and then synchronize globally.  
   **Improve it**: switch to a pipelined async copy pattern (`async_work_group_copy`/subgroup block loads if available on your target) so global→SLM movement of tile `ki+1` overlaps with DPAS compute on tile `ki`, then wait only on the async event before consuming the next tile. If async copy is unavailable, at least reduce synchronization scope by using subgroup-local staging for B (each subgroup owns disjoint 16 columns) and keep only one WG barrier for A visibility.

2. `half16 b_vec; b_vec.s0 = ...; ...; b_vec.sf = ...;` (16 scalar assignments): this creates a long scalar gather sequence from SLM every K-iteration, inflating instruction count and register pressure.  
   **Improve it**: replace scalar lane-by-lane assembly with a single vector load from contiguous memory, e.g. `half16 b_vec = vload16(0, &Bsub[cur][0][sg_col_base - tile_col + lx]);` only if layout is adjusted so the 16 K-elements are contiguous for each lane. Right now your `[TILE_K][TILE_N]` layout makes K strided by `TILE_N`; transpose B in SLM to `[TILE_N][TILE_K]` during load (or store subgroup-owned 16-column panels as `[16][16]`) so DPAS feed becomes one/two vector loads instead of 16 scalar reads.

3. `for (int r = 0; r < 32; ++r) { ... for (int k = k_full_end; k < K; ++k) { ... } C[...] = convert_half(v); }`: tail-K handling is done per output element at store time, causing repeated global loads of A/B and a large scalar epilogue in every lane.  
   **Improve it**: fold tail handling into the main tiled path by padding the last K tile in SLM with zeros and always running DPAS on it (or run one extra masked tile load when `K % TILE_K != 0`). That removes the nested scalar tail loop from the epilogue, keeps arithmetic in the matrix engine path, and greatly reduces divergent/global-memory-heavy cleanup work.

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
2. Use Built-in Functions: Prefer built-in functions (mad, fma, dot, cross, length) over manual implementations. Use fast_* variants (fast_normalize, fast_length) when precision allows. Use native_* for maximum speed with reduced precision.

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