

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
// Intel Xe2 / Battlemage optimized FP16 GEMM using DPAS
// C[M,N] = A[M,K] x B[K,N]
// A/B/C: half, accumulation: float
//
// Launch metadata (host-side recommendation):
//   LWS = {16, 1, 1}   // exactly one subgroup per work-group
//   GWS = {ceil_div(N,16)*16, ceil_div(M,16), 1}
//   Subgroup size = 16
//
// Mapping:
//   - One WG computes one 16x16 output tile at (tile_row, tile_col)
//   - lane x maps to one output column within tile
//   - Each lane accumulates 16 rows for its column as two float8 vectors
//   - K blocked by 16, full blocks use DPAS k16
//
// Why this is faster than prior:
//   - Uses DPAS outputs fully (float8) instead of only d.s0
//   - Subgroup-friendly mapping aligns with XMX execution model
//   - SLM tiling reduces global traffic and hides latency

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__attribute__((reqd_work_group_size(16,1,1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lane = (int)get_local_id(0);   // 0..15
    const int gx   = (int)get_group_id(0);
    const int gy   = (int)get_group_id(1);

    const int tile_row = gy * TILE_M;
    const int tile_col = gx * TILE_N;
    const int col      = tile_col + lane;

    // SLM tiles (+1 padding helps avoid bank conflicts)
    __local half Aslm[TILE_M][TILE_K + 1];
    __local half Bslm[TILE_K][TILE_N + 1];

    // Two 8-row accumulators => full 16 rows for this lane's column
    float8 acc0 = (float8)(0.0f); // rows 0..7
    float8 acc1 = (float8)(0.0f); // rows 8..15

    const int k_full_end = (K / TILE_K) * TILE_K;

    // Full k=16 blocks
    for (int kb = 0; kb < k_full_end; kb += TILE_K) {
        // Cooperative load A tile: each lane loads 16 elements (one k for all 16 rows)
        #pragma unroll
        for (int r = 0; r < TILE_M; ++r) {
            const int gr = tile_row + r;
            const int gk = kb + lane;
            Aslm[r][lane] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // Cooperative load B tile: each lane loads 16 elements (one col for all 16 k)
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            const int gk = kb + kk;
            const int gc = col;
            Bslm[kk][lane] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Build B vector for this lane's output column
        half16 bvec = vload16(0, &Bslm[0][lane]);

        // Rows 0..7
        {
            // Lanes 0..7 feed distinct rows, lanes 8..15 use safe duplicate row
            const int ar = (lane < 8) ? lane : 0;
            half16 avec = vload16(0, &Aslm[ar][0]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(avec, bvec, acc0);
        }

        // Rows 8..15
        {
            const int ar = (lane < 8) ? (lane + 8) : 8;
            half16 avec = vload16(0, &Aslm[ar][0]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(avec, bvec, acc1);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store + scalar tail K (exactness)
    if (col < N) {
        float tmp[8];

        // rows 0..7
        tmp[0] = acc0.s0; tmp[1] = acc0.s1; tmp[2] = acc0.s2; tmp[3] = acc0.s3;
        tmp[4] = acc0.s4; tmp[5] = acc0.s5; tmp[6] = acc0.s6; tmp[7] = acc0.s7;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gr = tile_row + r;
            if (gr < M) {
                float v = tmp[r];
                for (int k = k_full_end; k < K; ++k) {
                    v += convert_float(A[gr * K + k]) * convert_float(B[k * N + col]);
                }
                C[gr * N + col] = convert_half(v);
            }
        }

        // rows 8..15
        tmp[0] = acc1.s0; tmp[1] = acc1.s1; tmp[2] = acc1.s2; tmp[3] = acc1.s3;
        tmp[4] = acc1.s4; tmp[5] = acc1.s5; tmp[6] = acc1.s6; tmp[7] = acc1.s7;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gr = tile_row + 8 + r;
            if (gr < M) {
                float v = tmp[r];
                for (int k = k_full_end; k < K; ++k) {
                    v += convert_float(A[gr * K + k]) * convert_float(B[k * N + col]);
                }
                C[gr * N + col] = convert_half(v);
            }
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.

(II) Bottlenecks and suggestions for improvement:  
1. `const int ar = (lane < 8) ? lane : 0; ... intel_sub_group_f16_f16_matrix_mad_k16(avec, bvec, acc0);` and `const int ar = (lane < 8) ? (lane + 8) : 8; ... intel_sub_group_f16_f16_matrix_mad_k16(avec, bvec, acc1);`: you’re under-utilizing DPAS lanes. Half the subgroup feeds duplicated A rows (`0`/`8`), so many lanes do redundant work each K-block.  
   **Fix:** remap work so each subgroup computes a smaller tile that matches DPAS operand mapping (e.g., 8x16 or 16x8 micro-tile per subgroup) without duplicated row feeds. The key is: every lane should contribute unique A/B fragments to DPAS on every issue. If needed, use 2 subgroups per 16x16 tile (split rows) rather than forcing one subgroup with duplicated inputs.

2. `__local half Aslm[TILE_M][TILE_K + 1]; __local half Bslm[TILE_K][TILE_N + 1];` with per-iteration loads + `barrier(CLK_LOCAL_MEM_FENCE);` (twice inside K loop): SLM traffic + two barriers per `kb` is a major limiter for such a small tile. You’re paying synchronization overhead every 16 K even though access is highly regular.  
   **Fix:** remove SLM staging for at least B (and ideally both A/B) and load directly into registers using block/subgroup loads, then feed DPAS. If you keep SLM, use double-buffered SLM (ping-pong) with async copy so load of `kb+16` overlaps compute of `kb`, reducing barrier stalls to one logical sync point.

3. Tail handling in store path:  
   `for (int k = k_full_end; k < K; ++k) { v += ... }` inside both row loops. This recomputes the K-tail per output element (16 times per lane), causing heavy scalar overhead and poor instruction mix when `K % 16 != 0`.  
   **Fix:** split into two kernels: (a) fast main kernel assumes `K` multiple of 16 and contains only DPAS path, (b) tiny cleanup kernel handles tail-K (or pre-pad K on host). If you must keep one kernel, precompute tail contributions once per lane into a small row-vector accumulator before store (vectorized over rows), rather than re-running scalar tail loop separately for each row.

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
1. Exploit Sub-groups (OpenCL 2.0+): Use sub-group functions for wavefront-level operations: sub_group_reduce_add(), sub_group_broadcast(), intel_sub_group_shuffle(), intel_sub_group_shuffle_down(). Query sub-group size with get_sub_group_size(). Use __attribute__((intel_reqd_sub_group_size(N))) for Intel GPUs.
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
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.