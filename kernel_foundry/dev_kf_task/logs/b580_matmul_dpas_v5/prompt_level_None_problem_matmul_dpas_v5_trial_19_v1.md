

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
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

// Launch metadata:
//   LWS = {16, 2, 1}  (32 threads = 2 subgroups of 16)
//   GWS = {ceil_div(N,32)*16, ceil_div(M,32)*2, 1}
//   Each WG computes a 32x32 C tile using double-buffered SLM
//   Each subgroup computes 16 rows x 32 cols (4 DPAS per K-chunk)
//   TILE_K=32 for 2x arithmetic intensity

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32
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

    // Double-buffered SLM: A[32][32], Bt[32][32] (transposed B)
    __local half A_slm[2][TILE_M][TILE_K];
    __local half Bt_slm[2][TILE_N][TILE_K];

    // Register tile: 16 rows x 32 cols split into 4 blocks
    float8 acc_r0_c0 = (float8)(0.0f);  // rows 0-7, cols 0-15
    float8 acc_r0_c1 = (float8)(0.0f);  // rows 0-7, cols 16-31
    float8 acc_r1_c0 = (float8)(0.0f);  // rows 8-15, cols 0-15
    float8 acc_r1_c1 = (float8)(0.0f);  // rows 8-15, cols 16-31

    const int sg_row_base = tile_row + sg_id * 16;
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prefetch first tile
    int buf_idx = 0;
    {
        // Load A[32][32]: 1024 half, 32 threads → 32 half each (2 vload8 per thread)
        int a_row = tile_row + lid_linear;
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            int a_col = kk + (lid_linear & 15);
            half val = (a_row < M && a_col < K) ? A[a_row * K + a_col] : (half)0.0h;
            A_slm[0][lid_linear][kk + (lid_linear & 15)] = val;
        }

        // Load Bt[32][32]: Bt[n][k] = B[k][n]
        int bt_col = tile_col + lid_linear;
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            int b_row = kk + (lid_linear & 15);
            half val = (b_row < K && bt_col < N) ? B[b_row * N + bt_col] : (half)0.0h;
            Bt_slm[0][lid_linear][kk + (lid_linear & 15)] = val;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K loop with double buffering
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int kb = kt * TILE_K;
        int curr_buf = buf_idx;
        int next_buf = 1 - buf_idx;

        // Prefetch next tile in background (if exists)
        if (kt + 1 < num_k_tiles) {
            int next_kb = (kt + 1) * TILE_K;

            // Async load A
            int a_row = tile_row + lid_linear;
            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk += 16) {
                int a_col = next_kb + kk + (lid_linear & 15);
                half val = (a_row < M && a_col < K) ? A[a_row * K + a_col] : (half)0.0h;
                A_slm[next_buf][lid_linear][kk + (lid_linear & 15)] = val;
            }

            // Async load Bt
            int bt_col = tile_col + lid_linear;
            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk += 16) {
                int b_row = next_kb + kk + (lid_linear & 15);
                half val = (b_row < K && bt_col < N) ? B[b_row * N + bt_col] : (half)0.0h;
                Bt_slm[next_buf][lid_linear][kk + (lid_linear & 15)] = val;
            }
        }

        // DPAS compute on current tile
        // Process K in two 16-element chunks for DPAS k=16 requirement
        #pragma unroll
        for (int k_chunk = 0; k_chunk < 2; ++k_chunk) {
            int k_off = k_chunk * 16;

            // Each lane loads its unique A row (no aliasing!)
            half16 a_lo = vload16(0, &A_slm[curr_buf][sg_row_base + lane][k_off]);
            half16 a_hi = vload16(0, &A_slm[curr_buf][sg_row_base + 8 + lane][k_off]);

            // Load B columns for both col blocks
            half16 b_c0 = vload16(0, &Bt_slm[curr_buf][lane][k_off]);
            half16 b_c1 = vload16(0, &Bt_slm[curr_buf][16 + lane][k_off]);

            // 4 DPAS calls
            acc_r0_c0 = intel_sub_group_f16_f16_matrix_mad_k16(a_lo, b_c0, acc_r0_c0);
            acc_r0_c1 = intel_sub_group_f16_f16_matrix_mad_k16(a_lo, b_c1, acc_r0_c1);
            acc_r1_c0 = intel_sub_group_f16_f16_matrix_mad_k16(a_hi, b_c0, acc_r1_c0);
            acc_r1_c1 = intel_sub_group_f16_f16_matrix_mad_k16(a_hi, b_c1, acc_r1_c1);
        }

        buf_idx = next_buf;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Vectorized epilogue
    const bool full_tile = (tile_row + 31 < M) && (tile_col + 31 < N);

    if (full_tile) {
        // Fast path: vectorized stores for in-bounds tile
        int c_col_base = tile_col + lane;

        // Block 0: rows 0-7, cols 0-15
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            float val = ((float*)&acc_r0_c0)[r];
            C[(sg_row_base + r) * N + c_col_base] = convert_half(val);
        }

        // Block 1: rows 0-7, cols 16-31
        c_col_base = tile_col + 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            float val = ((float*)&acc_r0_c1)[r];
            C[(sg_row_base + r) * N + c_col_base] = convert_half(val);
        }

        // Block 2: rows 8-15, cols 0-15
        c_col_base = tile_col + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            float val = ((float*)&acc_r1_c0)[r];
            C[(sg_row_base + 8 + r) * N + c_col_base] = convert_half(val);
        }

        // Block 3: rows 8-15, cols 16-31
        c_col_base = tile_col + 16 + lane;
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            float val = ((float*)&acc_r1_c1)[r];
            C[(sg_row_base + 8 + r) * N + c_col_base] = convert_half(val);
        }
    } else {
        // Slow path: guarded scalar stores for edge tiles
        int c_col = tile_col + lane;

        #define STORE_BLOCK_GUARDED(acc_reg, row_off, col_off) \
        { \
            int gc = tile_col + (col_off) + lane; \
            if (gc < N) { \
                for (int r = 0; r < 8; ++r) { \
                    int gr = sg_row_base + (row_off) + r; \
                    if (gr < M) { \
                        float val = ((float*)&(acc_reg))[r]; \
                        C[gr * N + gc] = convert_half(val); \
                    } \
                } \
            } \
        }

        STORE_BLOCK_GUARDED(acc_r0_c0, 0, 0)
        STORE_BLOCK_GUARDED(acc_r0_c1, 0, 16)
        STORE_BLOCK_GUARDED(acc_r1_c0, 8, 0)
        STORE_BLOCK_GUARDED(acc_r1_c1, 8, 16)

        #undef STORE_BLOCK_GUARDED
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.  
(II) Bottlenecks and suggestions for improvement:  
1. `A_slm[0][lid_linear][kk + (lid_linear & 15)] = val;` and `Bt_slm[0][lid_linear][kk + (lid_linear & 15)] = val;` (also same pattern in the “next tile” prefetch):  
   You are only filling **2 elements per thread per matrix per tile** (`kk = 0,16`), so per tile you load just 64 elements of A and 64 of Bt, while compute expects a full 32×32 tile. This underutilizes memory throughput and forces poor reuse behavior. Also, index mapping is irregular and likely causes bank conflicts.  
   **Improve it by remapping cooperative loads so all 1024 elements are populated contiguously per tile** (e.g., each thread loads a small vector block, such as `half8/half16`, with linearized index `t = lid_linear + 32*i`, then `row=t/32`, `col=t%32`). For Bt, store in a layout aligned to your DPAS read pattern (`Bt[col][k]` contiguous for each lane group). This both fixes effective bandwidth and reduces SLM bank conflicts.

2. `half16 a_lo = vload16(0, &A_slm[curr_buf][sg_row_base + lane][k_off]);` and `half16 a_hi = vload16(0, &A_slm[curr_buf][sg_row_base + 8 + lane][k_off]);`  
   `sg_row_base` is a **global row base** (`tile_row + sg_id*16`), but `A_slm` is a **local tile buffer indexed 0..31**. Using global coordinates to index SLM is wrong for locality and can cause out-of-bounds/undefined behavior depending on group id. Even when numerically “working” in tests, this is a major performance killer because accesses are not coherent to the local tile.  
   **Use local tile row indices**: subgroup 0 should read rows `[0..15]`, subgroup 1 rows `[16..31]`; i.e. `local_row_base = sg_id * 16`, then `A_slm[curr_buf][local_row_base + lane][k_off]` (and corresponding `+8` path if you truly want split rows). Keep global offsets only for global-memory loads/stores, never for SLM tile addressing.

3. `barrier(CLK_LOCAL_MEM_FENCE);` at tile boundaries with “background prefetch” done by normal stores  
   The code comments claim async overlap, but there is no real async copy primitive; prefetch and compute are serialized by full workgroup barriers each iteration. This leaves EU pipes idle and limits occupancy/latency hiding.  
   **Improve by using true async block copy (if available on your target, e.g., `async_work_group_copy`/vendor block-load intrinsics) into double buffers and overlapping with DPAS**, then synchronize with event/wait only when needed. Also consider reducing barrier scope/frequency by pipelining at subgroup granularity if your DPAS path permits. This typically gives a noticeable runtime drop in K-heavy GEMM.

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
2. Reduce Divergent Branches: Keep all work-items in a wavefront/warp on the same execution path. Move divergent code outside the kernel or restructure to process uniform batches. Group similar work together.

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
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.