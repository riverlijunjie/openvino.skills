

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
// Optimized FP16 MatMul using Intel DPAS (XMX) with hierarchical tiling
// C[M,N] = A[M,K] x B[K,N]
// Input/Output: half, Accumulation: float
//
// Launch configuration:
//   GWS: [((N+63)/64)*16, ((M+63)/64)*4, 1]
//   LWS: [16, 4, 1]  (64 work-items = 4 sub-groups of size 16)
//   Sub-group size: 16 (required)
//
// Tile mapping:
//   - Each work-group processes a 64x64 output tile
//   - K dimension processed in chunks of 32
//   - 4 sub-groups, each handles 16x64 rows
//   - Each work-item accumulates 2x4 output elements (8 FP32 accumulators)
//   - Uses DPAS for 8x16 matrix operations
//
// Memory layout:
//   - SLM A_tile: 64x32 with padding (4.25KB)
//   - SLM B_tile: 32x64 with padding (4.25KB)
//   - Total SLM per work-group: ~8.5KB

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define MICRO_M 2
#define MICRO_N 4

__attribute__((reqd_work_group_size(16, 4, 1)))
__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    // Work-group and sub-group identification
    const int sg_id = get_sub_group_id();           // 0-3
    const int sg_lid = get_sub_group_local_id();    // 0-15
    const int lx = get_local_id(0);                 // 0-15
    const int ly = get_local_id(1);                 // 0-3
    const int local_linear = ly * 16 + lx;          // 0-63

    // Global tile position
    const int tile_row = get_group_id(1) * TILE_M;
    const int tile_col = get_group_id(0) * TILE_N;

    // Shared Local Memory with padding to avoid bank conflicts
    __local half A_tile[TILE_M][TILE_K + 2];
    __local half B_tile[TILE_K][TILE_N + 2];

    // Each work-item handles a 2x4 micro-tile of C
    // Row mapping: sg_id * 16 + sg_lid gives base row (0-63)
    // We compute 2 consecutive rows
    const int micro_row_base = sg_id * 16 + sg_lid;

    // Column mapping: each work-item handles 4 columns spaced by 16
    // This gives good coalescing and DPAS-friendly access
    const int micro_col_base = sg_lid;  // 0-15

    // 8 accumulators for 2x4 micro-tile
    float acc[MICRO_M][MICRO_N];
    #pragma unroll
    for (int i = 0; i < MICRO_M; i++) {
        #pragma unroll
        for (int j = 0; j < MICRO_N; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Loop over K dimension in tiles of 32
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperative load A tile [64 x 32] = 2048 elements
        // 64 work-items, each loads 32 elements (2x16)
        #pragma unroll 2
        for (int i = 0; i < 2; i++) {
            int load_idx = local_linear + i * 64;
            if (load_idx < TILE_M * 2) {
                int tile_r = load_idx / 2;
                int half_k = load_idx % 2;
                int glob_r = tile_row + tile_r;
                int glob_c_base = k_tile + half_k * 16;

                if (glob_r < M && glob_c_base < K) {
                    // Vectorized load for better memory bandwidth
                    half16 a_vec = vload16(0, &A[glob_r * K + glob_c_base]);
                    vstore16(a_vec, 0, &A_tile[tile_r][half_k * 16]);
                } else {
                    // Boundary case
                    #pragma unroll
                    for (int kk = 0; kk < 16; kk++) {
                        int glob_c = glob_c_base + kk;
                        A_tile[tile_r][half_k * 16 + kk] = 
                            (glob_r < M && glob_c < K) ? A[glob_r * K + glob_c] : (half)0.0h;
                    }
                }
            }
        }

        // Cooperative load B tile [32 x 64] = 2048 elements
        // 64 work-items, each loads 32 elements (2x16)
        #pragma unroll 2
        for (int i = 0; i < 2; i++) {
            int load_idx = local_linear + i * 64;
            if (load_idx < TILE_K * 4) {
                int tile_r = load_idx / 4;
                int quarter_n = load_idx % 4;
                int glob_r = k_tile + tile_r;
                int glob_c_base = tile_col + quarter_n * 16;

                if (glob_r < K && glob_c_base < N) {
                    // Vectorized load
                    half16 b_vec = vload16(0, &B[glob_r * N + glob_c_base]);
                    vstore16(b_vec, 0, &B_tile[tile_r][quarter_n * 16]);
                } else {
                    // Boundary case
                    #pragma unroll
                    for (int nn = 0; nn < 16; nn++) {
                        int glob_c = glob_c_base + nn;
                        B_tile[tile_r][quarter_n * 16 + nn] = 
                            (glob_r < K && glob_c < N) ? B[glob_r * N + glob_c] : (half)0.0h;
                    }
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute using DPAS in two K=16 sub-steps
        #pragma unroll
        for (int k_sub = 0; k_sub < 2; k_sub++) {
            int k_offset = k_sub * 16;

            // Process 2x4 micro-tile
            #pragma unroll
            for (int mi = 0; mi < MICRO_M; mi++) {
                int row_idx = micro_row_base + mi;
                if (tile_row + row_idx >= M) continue;

                // Load A row vector (1x16)
                half16 a_vec = vload16(0, &A_tile[row_idx][k_offset]);

                #pragma unroll
                for (int mj = 0; mj < MICRO_N; mj++) {
                    int col_idx = micro_col_base + mj * 16;
                    if (tile_col + col_idx >= N) continue;

                    // Gather B column vector (16x1) from SLM
                    half16 b_vec;
                    #pragma unroll
                    for (int kk = 0; kk < 16; kk++) {
                        b_vec[kk] = B_tile[k_offset + kk][col_idx];
                    }

                    // DPAS: accumulate dot product into float
                    float8 dpas_acc = (float8)(acc[mi][mj]);
                    dpas_acc = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, dpas_acc);
                    acc[mi][mj] = dpas_acc.s0;
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results back to global memory
    #pragma unroll
    for (int mi = 0; mi < MICRO_M; mi++) {
        int global_row = tile_row + micro_row_base + mi;
        if (global_row >= M) continue;

        #pragma unroll
        for (int mj = 0; mj < MICRO_N; mj++) {
            int global_col = tile_col + micro_col_base + mj * 16;
            if (global_col < N) {
                C[global_row * N + global_col] = convert_half(acc[mi][mj]);
            }
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient. You’re leaving a lot of XMX throughput on the table because parts of the inner loop are still scalar/SLM-gather heavy instead of true subgroup-matrix style.

(II) Bottlenecks and suggestions for improvement:  
1. `for (int kk = 0; kk < 16; kk++) { b_vec[kk] = B_tile[k_offset + kk][col_idx]; }`: this per-FMA gather from SLM is a major bottleneck (16 scalar local loads for every output element).  
   **Improve it:** change the data mapping so each lane reads contiguous `half` from `B_tile` (or pre-transpose `B_tile` in SLM during the load stage) and feed DPAS with subgroup-friendly packed fragments. Right now you’re effectively emulating a column load with scalar accesses, which kills throughput. Store B in SLM as `[N_block][K_block]` layout (or an interleaved layout by subgroup lane) so `vload16`/block reads are possible instead of scalar gather.

2. `float8 dpas_acc = (float8)(acc[mi][mj]); dpas_acc = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, dpas_acc); acc[mi][mj] = dpas_acc.s0;`: you invoke DPAS in a scalarized way and only keep `.s0`, so most SIMD accumulator capacity is wasted and you pay conversion/packing overhead each iteration.  
   **Improve it:** keep accumulators in native DPAS vector form for the whole K loop (e.g., `float8`/`float16` register tiles per lane), and only extract/store at the end. Also increase per-lane output tile so one DPAS updates multiple C elements per call (instead of 1 dot-product-like usage). This raises arithmetic intensity and reduces instruction overhead.

3. `barrier(CLK_LOCAL_MEM_FENCE);` (both before and after compute in each `k_tile`): full-tile load → full-tile compute → full sync causes pipeline bubbles and stalls memory/compute overlap.  
   **Improve it:** implement double-buffered SLM tiles (`A_tile[2]`, `B_tile[2]`) and ping-pong between them. While computing on buffer `t`, start cooperative loads for `t+1`; keep only one barrier for handoff per stage. This hides global-memory latency and usually gives a noticeable runtime drop on larger K.

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
1. Kernel Fusion: Combine multiple small kernels into one to eliminate intermediate global memory writes and reduce launch overhead. Use barrier(CLK_LOCAL_MEM_FENCE) between logical kernel phases within a work-group.
2. Use Predication: Replace if(cond) x = a; else x = b; with x = select(b, a, cond) for scalar types or x = cond ? a : b. Use select() for vector types: result = select(false_val, true_val, condition).

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