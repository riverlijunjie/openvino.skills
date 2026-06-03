

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
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

// Launch metadata:
//   LWS = {16, 2, 1}  (32 threads = 2 subgroups of 16)
//   GWS = {ceil_div(N,32)*16, ceil_div(M,32)*2, 1}
//   Each WG computes a 32x32 C tile
//   Each subgroup computes 16 rows x 32 cols (4 DPAS calls per K-chunk)
//   B stored transposed in SLM for vectorized access

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define SG_SIZE 16

__attribute__((reqd_work_group_size(16, 2, 1)))
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_local_id(1);              // 0 or 1
    const int lane = get_sub_group_local_id();      // 0-15
    const int lid_linear = sg_id * 16 + lane;       // 0-31

    const int tile_row = get_group_id(1) * TILE_M;
    const int tile_col = get_group_id(0) * TILE_N;

    // SLM: A is 32x16, B transposed is 32x16 (stored as Bt[col][k])
    __local half A_slm[TILE_M][TILE_K];
    __local half Bt_slm[TILE_N][TILE_K];  // B transposed: Bt[n][k] = B[k][n]

    // Accumulators: each subgroup does 16 rows, split into 2x8 row blocks x 2 col blocks
    // Row block 0 (rows 0-7), col block 0 (cols 0-15)
    float8 acc_r0_c0 = (float8)(0.0f);
    // Row block 0, col block 1 (cols 16-31)
    float8 acc_r0_c1 = (float8)(0.0f);
    // Row block 1 (rows 8-15), col block 0
    float8 acc_r1_c0 = (float8)(0.0f);
    // Row block 1, col block 1
    float8 acc_r1_c1 = (float8)(0.0f);

    const int sg_row_base = tile_row + sg_id * 16;

    for (int kb = 0; kb < K; kb += TILE_K) {
        // === Cooperative load A[32][16]: 512 elems, 32 threads, 16 each ===
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int r = lid_linear;  // each thread loads one row
            int c = i;
            // But 32 threads x 16 cols = 512, so thread lid_linear loads row lid_linear, col i
            int gr = tile_row + r;
            int gk = kb + c;
            A_slm[r][c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // === Cooperative load B transposed: Bt[32][16] = 512 elems ===
        // Bt[n][k] = B[k][n], each thread loads one "column" (n = lid_linear)
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int n = lid_linear;  // column index in output
            int k = i;
            int gk = kb + k;
            int gn = tile_col + n;
            Bt_slm[n][k] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === DPAS compute ===
        // a_vec for rows 0-7 of this subgroup's 16 rows
        half16 a_lo = vload16(0, &A_slm[sg_id * 16 + (lane < 8 ? lane : 0)][0]);
        // a_vec for rows 8-15
        half16 a_hi = vload16(0, &A_slm[sg_id * 16 + (lane < 8 ? lane + 8 : 8)][0]);

        // b_vec for cols 0-15: each lane loads Bt[lane][0:15]
        half16 b_c0 = vload16(0, &Bt_slm[lane][0]);
        // b_vec for cols 16-31: each lane loads Bt[lane+16][0:15]
        half16 b_c1 = vload16(0, &Bt_slm[lane + 16][0]);

        // 4 DPAS calls
        acc_r0_c0 = intel_sub_group_f16_f16_matrix_mad_k16(a_lo, b_c0, acc_r0_c0);
        acc_r0_c1 = intel_sub_group_f16_f16_matrix_mad_k16(a_lo, b_c1, acc_r0_c1);
        acc_r1_c0 = intel_sub_group_f16_f16_matrix_mad_k16(a_hi, b_c0, acc_r1_c0);
        acc_r1_c1 = intel_sub_group_f16_f16_matrix_mad_k16(a_hi, b_c1, acc_r1_c1);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
    // acc_r0_c0: rows [sg_row_base+0..7], cols [tile_col+0..15] (lane = col within block)
    float acc_arr[8];

    #define STORE_BLOCK(acc_reg, row_off, col_off) \
    { \
        acc_arr[0]=(acc_reg).s0; acc_arr[1]=(acc_reg).s1; \
        acc_arr[2]=(acc_reg).s2; acc_arr[3]=(acc_reg).s3; \
        acc_arr[4]=(acc_reg).s4; acc_arr[5]=(acc_reg).s5; \
        acc_arr[6]=(acc_reg).s6; acc_arr[7]=(acc_reg).s7; \
        int gc = tile_col + (col_off) + lane; \
        if (gc < N) { \
            for (int r = 0; r < 8; ++r) { \
                int gr = sg_row_base + (row_off) + r; \
                if (gr < M) \
                    C[gr * N + gc] = convert_half(acc_arr[r]); \
            } \
        } \
    }

    STORE_BLOCK(acc_r0_c0, 0, 0)
    STORE_BLOCK(acc_r0_c1, 0, 16)
    STORE_BLOCK(acc_r1_c0, 8, 0)
    STORE_BLOCK(acc_r1_c1, 8, 16)

    #undef STORE_BLOCK
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.

(II) Bottlenecks and suggestions for improvement:
1. `half16 a_lo = vload16(0, &A_slm[sg_id * 16 + (lane < 8 ? lane : 0)][0]);` and `half16 a_hi = vload16(0, &A_slm[sg_id * 16 + (lane < 8 ? lane + 8 : 8)][0]);`:  
   You’re forcing **lane aliasing** (`lane>=8` reuses row 0/8), so half the lanes load duplicate A rows. That wastes SLM bandwidth and hurts DPAS utilization.  
   **Improve:** map rows to lanes without duplication and issue DPAS on packed operands per hardware-friendly layout. A practical fix is to have each lane own a unique row fragment (or use subgroup block reads to load the exact DPAS A payload format), e.g. split compute so lanes 0..15 each contribute distinct row data for one 16x16 output fragment, then iterate fragments. This removes redundant loads and increases useful math per SLM transaction.

2. `for (int i = 0; i < 16; ++i) { ... A_slm[r][c] = ... }` and `for (int i = 0; i < 16; ++i) { ... Bt_slm[n][k] = ... }` (plus the two `barrier(CLK_LOCAL_MEM_FENCE);` inside each `kb` loop):  
   The kernel is **SLM/barrier bound**: every K-block does full cooperative writes to SLM for both A and B, then synchronizes twice. The arithmetic intensity per barrier is low for TILE_K=16.  
   **Improve:** increase `TILE_K` (e.g., 32 if register/SLM budget allows) to halve barrier frequency, and pipeline loads/compute with double-buffered SLM (`A_slm[2]`, `Bt_slm[2]`) so you overlap global→SLM load of `kb+1` with DPAS on `kb`. Also prefer vectorized global loads/stores during staging (`vload4/vload8`) to reduce transaction count.

3. `C[gr * N + gc] = convert_half(acc_arr[r]);` inside `STORE_BLOCK` loop with scalar per-element writes and repeated bounds checks:  
   Your epilogue is **scalar and branchy** (8 scalar stores per lane per block, repeated for 4 blocks), which becomes a visible tail cost after optimizing compute.  
   **Improve:** write out contiguous vectors when in-bounds fast path is true (full tile case): precheck `tile_row+31<M && tile_col+31<N`, then do branch-free vector stores (`vstore8`/`vstore_half8`) per row chunk. Keep the current guarded scalar path only for edge tiles. This reduces control divergence and store instruction count significantly.

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
1. Minimize Host-Device Transfers: Use CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY flags appropriately. Chain kernels using clEnqueueNDRangeKernel with event dependencies instead of clFinish(). Use persistent device allocations across multiple kernel launches.
2. Exploit Constant Cache: Use __constant address space (limited to ~64KB) for frequently accessed read-only data. Declare as __constant float coeffs[N]. All work-items should access same location simultaneously for best performance.

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
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.