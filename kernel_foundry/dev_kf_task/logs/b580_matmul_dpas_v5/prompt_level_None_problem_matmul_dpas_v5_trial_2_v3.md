

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
// Intel Xe2/Battlemage tuned FP16 GEMM:
//   C[M,N] = A[M,K] x B[K,N]
// dtype: A/B/C = half, accumulation = float
//
// Launch metadata (host-side recommendation):
//   LWS = {16, 16, 1}   // 256 WI = 16 subgroups (SG size 16)
//   GWS = {ceil_div(N, 32) * 16, ceil_div(M, 32) * 16, 1}
// Mapping:
//   - Each WG computes a 32x32 C tile.
//   - Each WI computes THREAD_M x THREAD_N = 2x2 outputs (register blocking).
//   - K processed in TILE_K=16 chunks; DPAS used on full chunks.
// Subgroup hint:
//   - intel_reqd_sub_group_size(16)

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define WG_X 16
#define WG_Y 16

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16

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

    // WG tile origin in C
    const int c_row0 = gy * TILE_M + ly * THREAD_M;
    const int c_col0 = gx * TILE_N + lx * THREAD_N;

    // SLM tiles:
    // A tile layout: [TILE_M][TILE_K+1] (pad to reduce bank conflicts)
    // B tile stored transposed: Bslm[col_in_tile][k] so each WI can read contiguous k-vectors per output col
    __local half Aslm[TILE_M][TILE_K + 1];
    __local half BslmT[TILE_N][TILE_K + 1];

    // Register-block accumulators (float)
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const int k_full_end = (K / TILE_K) * TILE_K;

    // Full K tiles (DPAS hot path)
    for (int kb = 0; kb < k_full_end; kb += TILE_K) {
        // Cooperative load A tile: 32x16 = 512 half, 256 WI => 2 elems/WI
        {
            int linear = ly * WG_X + lx; // 0..255
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                int idx = linear + t * 256;       // 0..511
                int ar = idx / TILE_K;            // 0..31
                int ak = idx - ar * TILE_K;       // 0..15
                int g_r = gy * TILE_M + ar;
                int g_k = kb + ak;
                Aslm[ar][ak] = (g_r < M && g_k < K) ? A[g_r * K + g_k] : (half)0.0h;
            }
        }

        // Cooperative load B tile and transpose into SLM:
        // source B[k][n], store BslmT[n][k]
        {
            int linear = ly * WG_X + lx; // 0..255
            #pragma unroll
            for (int t = 0; t < 2; ++t) {
                int idx = linear + t * 256;       // 0..511
                int bk = idx / TILE_N;            // 0..15
                int bn = idx - bk * TILE_N;       // 0..31
                int g_k = kb + bk;
                int g_n = gx * TILE_N + bn;
                BslmT[bn][bk] = (g_k < K && g_n < N) ? B[g_k * N + g_n] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 2x2 outputs per WI
        // For each output, build half16 vectors contiguous from SLM and call DPAS.
        #pragma unroll
        for (int mi = 0; mi < THREAD_M; ++mi) {
            const int ar = ly * THREAD_M + mi; // 0..31
            half16 a_vec = vload16(0, &Aslm[ar][0]);

            #pragma unroll
            for (int nj = 0; nj < THREAD_N; ++nj) {
                const int bc = lx * THREAD_N + nj; // 0..31
                half16 b_vec = vload16(0, &BslmT[bc][0]);

                // DPAS accumulate into float scalar via float8 carrier
                float8 d = (float8)(0.0f);
                d = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, d);
                acc[mi][nj] += d.s0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Tail K cleanup (exact reference behavior)
    #pragma unroll
    for (int mi = 0; mi < THREAD_M; ++mi) {
        int r = c_row0 + mi;
        if (r >= M) continue;

        #pragma unroll
        for (int nj = 0; nj < THREAD_N; ++nj) {
            int c = c_col0 + nj;
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

// Launch metadata (host-side recommendation):
//   LWS = {16, 16, 1}    // 256 threads / WG, 16 subgroups of width 16
//   GWS = { round_up(N, 32), round_up(M, 16), 1 }  // because each WI computes 2 cols
//   Subgroup: intel_reqd_sub_group_size(16)
//
// Mapping:
//   - One WG computes a C tile of 16 rows x 32 cols (register-blocked on N by 2)
//   - K processed in blocks of 16 (DPAS-friendly)
//   - SLM tiles per K-slice:
//       A_tile[16][16], B_tile[16][32]
//
// Correctness:
//   - Exact semantics vs reference: fp32 accumulation, fp16 output
//   - Full boundary checks on M/N and K-tail

#define TM 16
#define TN 32
#define TK 16

__attribute__((reqd_work_group_size(16,16,1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lx = get_local_id(0);   // 0..15
    const int ly = get_local_id(1);   // 0..15
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);

    // Each WI computes 2 output columns
    const int row = gy * TM + ly;
    const int col0 = gx * TN + (lx << 1);
    const int col1 = col0 + 1;

    __local half Asub[TM][TK + 1];
    __local half Bsub[TK][TN + 1];

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const int k_full_end = (K / TK) * TK;

    for (int kb = 0; kb < k_full_end; kb += TK) {
        // Cooperative load A: 16x16, one element per WI
        {
            const int a_r = gy * TM + ly;
            const int a_c = kb + lx;
            Asub[ly][lx] = (a_r < M && a_c < K) ? A[a_r * K + a_c] : (half)0.0h;
        }

        // Cooperative load B: 16x32, two phases because WG has 16x16 threads
        {
            const int b_r = kb + ly;
            const int b_c0 = gx * TN + lx;         // first 16 cols
            const int b_c1 = gx * TN + 16 + lx;    // second 16 cols

            Bsub[ly][lx]      = (b_r < K && b_c0 < N) ? B[b_r * N + b_c0] : (half)0.0h;
            Bsub[ly][16 + lx] = (b_r < K && b_c1 < N) ? B[b_r * N + b_c1] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Build DPAS operands from SLM for this WI's 2 cols.
        // We keep scalar FMA accumulation path as exact fallback behavior.
        half16 a_vec = vload16(0, &Asub[ly][0]);

        half16 b_vec0, b_vec1;
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            b_vec0.s[kk] = Bsub[kk][(lx << 1)];
            b_vec1.s[kk] = Bsub[kk][(lx << 1) + 1];
        }

        // Mandatory DPAS usage (Intel XMX path). We accumulate lane0 contribution.
        // Toolchains differ in exact lowering; this form is broadly accepted on Intel OpenCL.
        float8 d0 = (float8)(0.0f);
        float8 d1 = (float8)(0.0f);
        d0 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec0, d0);
        d1 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec1, d1);

        // Accumulate (plus exact scalar reinforcement to ensure robust correctness across toolchains)
        // If compiler fully lowers DPAS correctly, scalar loop may be optimized aggressively.
        acc0 += d0.s0;
        acc1 += d1.s0;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Exact K tail (and also covers all remainder behavior)
    if (row < M) {
        for (int k = k_full_end; k < K; ++k) {
            const float av = convert_float(A[row * K + k]);
            if (col0 < N) acc0 += av * convert_float(B[k * N + col0]);
            if (col1 < N) acc1 += av * convert_float(B[k * N + col1]);
        }

        if (col0 < N) C[row * N + col0] = convert_half(acc0);
        if (col1 < N) C[row * N + col1] = convert_half(acc1);
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.  

(II) Bottlenecks and suggestions for improvement:  
1. `for (int kk = 0; kk < 16; ++kk) { b_vec0.s[kk] = Bsub[kk][(lx << 1)]; b_vec1.s[kk] = Bsub[kk][(lx << 1) + 1]; }`: this is a high-overhead gather from SLM into private vectors every K-tile, and it is done per work-item. You’re paying 32 scalar local loads + register packing before each DPAS call, which can dominate runtime.  
   **Improve it:** eliminate this gather path and feed DPAS from data laid out natively for the subgroup operation. In practice, make each subgroup cooperatively load a DPAS-fragment-ready `B` tile (or use block reads into packed types like `half8/half16` with contiguous addressing), so you don’t reconstruct `b_vec*` element-by-element. If your compiler supports the joint-matrix style API, switch to that so tile load + MMA lowering is direct and avoids scalar lane assembly.

2. `acc0 += d0.s0;` and `acc1 += d1.s0;`: you are computing vector DPAS outputs (`float8`) and only using lane `s0`. That throws away most computed work and indicates the mapping between DPAS result shape and output ownership is inefficient.  
   **Improve it:** remap work so each work-item/subgroup consumes all DPAS result lanes it produces (e.g., each lane writes/accumulates multiple `N` outputs or a 2D micro-tile). If you keep the current 1-row/2-col ownership, then use scalar FMA directly for that mapping; otherwise redesign ownership to match DPAS native tile so `d*.s[0..7]` all contribute to C. This alone typically gives a major speedup.

3. `barrier(CLK_LOCAL_MEM_FENCE);` (both before and after compute in each `kb` iteration): full workgroup barriers twice per K-slice are expensive and limit overlap; plus your compute between barriers is relatively small (2 outputs/thread), so synchronization cost is significant.  
   **Improve it:** use double-buffered SLM tiles (`Asub0/Asub1`, `Bsub0/Bsub1`) and pipeline load/compute so each iteration has one synchronization point for buffer swap instead of two hard stops. Also increase per-thread compute (e.g., 4–8 cols per thread) so barrier cost is amortized over more FMAs/DPAS work. This raises arithmetic intensity and improves EU occupancy/utilization.

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
- **Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing.