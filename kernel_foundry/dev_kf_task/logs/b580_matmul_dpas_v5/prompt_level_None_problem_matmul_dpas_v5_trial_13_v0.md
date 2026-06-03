

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
//   LWS = {16, 16, 1}  (256 threads, 16 subgroups of size 16)
//   GWS = {ceil_div(N,128)*16, ceil_div(M,16)*16, 1}
//   Subgroup: intel_reqd_sub_group_size(16)
//
// Each WG computes a 16×128 tile of C
// Each thread computes 1 row × 8 columns via full float8 DPAS output
// lx selects which set of 8 columns (16 lanes × 8 cols = 128 cols)
// ly selects which row (16 rows)
// K blocked by 32 (two DPAS k16 calls per block)

#define TILE_M 16
#define TILE_N 128
#define TILE_K 32
#define SG_SIZE 16

__attribute__((reqd_work_group_size(16, 16, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lx = get_local_id(0);  // 0..15 (subgroup lane)
    const int ly = get_local_id(1);  // 0..15 (row within tile)
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);

    const int c_row = gy * TILE_M + ly;
    // Each lane handles 8 consecutive columns; 16 lanes cover 128 columns
    const int c_col_base = gx * TILE_N + lx * 8;

    // SLM: A[16][32+1], B[32][128+1]
    __local half Asub[TILE_M][TILE_K + 1];
    __local half Bsub[TILE_K][TILE_N + 1];

    // Accumulators: 8 columns per thread
    float8 acc0 = (float8)(0.0f);  // for first DPAS k16
    float8 acc1 = (float8)(0.0f);  // second half - wait, we accumulate into same

    // Actually accumulate across all K into one float8
    float8 acc = (float8)(0.0f);

    const int k_full_end = (K / TILE_K) * TILE_K;
    const int linear_id = ly * 16 + lx;  // 0..255

    for (int kb = 0; kb < k_full_end; kb += TILE_K) {
        // Load A: 16×32 = 512 elems, 256 threads => 2 each
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            int idx = linear_id + t * 256;
            int ar = idx / TILE_K;   // 0..15
            int ak = idx % TILE_K;   // 0..31
            int gr = gy * TILE_M + ar;
            int gk = kb + ak;
            Asub[ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // Load B: 32×128 = 4096 elems, 256 threads => 16 each
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            int idx = linear_id + t * 256;
            int bk = idx / TILE_N;   // 0..31
            int bc = idx % TILE_N;   // 0..127
            int gk = kb + bk;
            int gc = gx * TILE_N + bc;
            Bsub[bk][bc] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // DPAS step 0: k=[0..15]
        {
            half16 a_vec = vload16(0, &Asub[ly][0]);

            // B: need 8 columns for this lane, packed as half8 per k-row
            // But DPAS expects half8 b_vec where each lane's value forms a column
            // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
            //   a: half (scalar per lane, broadcast across subgroup for row selection)
            //   b: half8 (8 columns, shared across subgroup via subgroup register file)
            //   result: float8 (8 column results)
            // Wait - let me reconsider the DPAS signature more carefully.

            // The intrinsic does: result[i] += sum_k(a[k] * b[k][i]) for i in 0..7
            // where a is half16 (16 k-values) and b is half16 (?) 
            // Actually the signatures vary. Let me use what compiled before.

            // From Version 1 which compiled: 
            //   intel_sub_group_f16_f16_matrix_mad_k16(half16 a, half16 b, float/float8 acc)
            // where a=half16 is the row of A, b=half16 is gathered column of B
            // This returns float8 but only s0 was used before.

            // The float8 output means 8 rows of output are computed simultaneously
            // across subgroup lanes. So the DPAS computes an 8×16 matmul:
            //   8 rows (from 8 consecutive subgroup invocations' a vectors)
            //   × 16 k-values
            //   → 8 results per lane (each lane's b_vec selects a column)

            // So: each lane provides a_vec for its row, and b_vec for its column
            // Output float8: 8 row results for the lane's column
            // This means 1 DPAS call gives 8 rows × 16 lanes = 8×16 output tile

            // For 16 rows we need 2 DPAS calls (rows 0-7, rows 8-15)
            // For 128 columns we have 16 lanes × 8 results = only 16 columns from float8...
            // Wait no. float8 gives 8 rows. Each lane = 1 column. 16 lanes = 16 columns.
            // So one DPAS = 8 rows × 16 columns.

            // To cover 16×128: need 2 row groups × 8 column groups = 16 DPAS calls. Too many.
            // Let me scale back to 16×16 tile but use DPAS properly for 8 rows.

            // For a 16×16 tile: 2 DPAS calls (rows 0-7, rows 8-15), each producing 8×16.
            // But we have 256 threads and only need 2 DPAS calls... massive underutilization.

            // Better: use fewer threads per WG. E.g., LWS={16,1,1} = 16 threads = 1 subgroup.
            // Each subgroup computes 8×16 per DPAS. For 16 rows: 2 DPAS calls.
            // Each thread accumulates float8 (8 row results for its column).

            // Let me redesign completely.
            _ = a_vec; // placeholder
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // placeholder store
    if (c_row < M) {
        for (int j = 0; j < 8; ++j) {
            int gc = c_col_base + j;
            if (gc < N) {
                float v;
                switch(j) {
                    case 0: v = acc.s0; break;
                    case 1: v = acc.s1; break;
                    case 2: v = acc.s2; break;
                    case 3: v = acc.s3; break;
                    case 4: v = acc.s4; break;
                    case 5: v = acc.s5; break;
                    case 6: v = acc.s6; break;
                    case 7: v = acc.s7; break;
                }
                C[c_row * N + gc] = convert_half(v);
            }
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.  

(II) Bottlenecks and suggestions for improvement:  
1. `__attribute__((reqd_work_group_size(16, 16, 1)))` + `const int ly = get_local_id(1);` + your own comments in the DPAS section  
   You’re launching **256 threads/WG** and mapping one thread to one output row×8 cols, but the DPAS path you commented about is fundamentally **subgroup-oriented** (16 lanes). Right now, most of your structure is tuned for a scalar/FMA-style mapping, not a true DPAS mapping, so you pay high thread-management + barrier cost without getting the intended matrix engine efficiency.  
   **Improve it:** switch to a **single-subgroup WG** (`LWS={16,1,1}`) or a small integer number of subgroups where each subgroup computes a compact DPAS tile (e.g., 8x16 or 16x16 depending on intrinsic form), then replicate across tiles in `group_id` space. This removes the artificial `ly` dimension, cuts synchronization overhead, and aligns work decomposition with how `intel_sub_group_*matrix_mad*` actually executes.

2. `barrier(CLK_LOCAL_MEM_FENCE);` (twice per `kb` iteration) together with  
   `__local half Asub[TILE_M][TILE_K + 1];` and `__local half Bsub[TILE_K][TILE_N + 1];`  
   You force full-WG synchronization around large SLM staging every K-block. With your current 256-thread WG and big `Bsub` tile (32x128), these barriers are expensive and become a dominant stall source. Also, `+1` padding helps bank conflicts, but the bigger issue is that you are round-tripping through SLM for data that DPAS-friendly subgroup loads could often consume more directly/efficiently.  
   **Improve it:** use **double-buffered SLM** (ping-pong `Asub/Bsub`) and overlap next-tile loads with current compute; if possible, move B operand feeding to **subgroup block reads / packed loads** that match DPAS operand layout to reduce SLM traffic. At minimum, reduce barrier frequency to one synchronization point per stage transition instead of hard fencing both before and after compute.

3. Store path:  
   ```c
   for (int j = 0; j < 8; ++j) {
       ...
       switch(j) { case 0: v = acc.s0; ... case 7: v = acc.s7; }
       C[c_row * N + gc] = convert_half(v);
   }
   ```  
   This scalarized store with a per-element `switch` creates unnecessary control flow and inhibits vectorized memory writes. It also adds instruction overhead right at the bandwidth-sensitive epilogue.  
   **Improve it:** eliminate the switch entirely—use vector access (`acc.s01234567`) and perform a **packed half8 conversion + vector store** (e.g., convert to half8 once, then `vstore8` when in-bounds). For edge tiles, keep a fast full-tile path (no bounds checks) and a separate remainder path. This will noticeably reduce epilogue overhead and improve global store throughput.

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
1. Use Built-in Functions: Prefer built-in functions (mad, fma, dot, cross, length) over manual implementations. Use fast_* variants (fast_normalize, fast_length) when precision allows. Use native_* for maximum speed with reduced precision.
2. Minimize Host-Device Transfers: Use CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY flags appropriately. Chain kernels using clEnqueueNDRangeKernel with event dependencies instead of clFinish(). Use persistent device allocations across multiple kernel launches.

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
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.