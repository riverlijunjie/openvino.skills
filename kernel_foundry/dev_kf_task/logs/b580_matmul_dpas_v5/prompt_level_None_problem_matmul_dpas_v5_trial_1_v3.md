

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

### Version 1 (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Optimized FP16 matmul using Intel DPAS instruction
// C[M,N] = A[M,K] x B[K,N]
//
// Launch Configuration:
// - Global Work Size (GWS): (N/16, M/8, 1)
// - Local Work Size (LWS): (16, 8, 1) = 128 work-items = 8 subgroups
// - Required subgroup size: 16
// - Each work-item computes 8 rows × 8 cols = 64 output elements
// - Each subgroup (16 work-items) computes 8 rows × 128 cols via DPAS
//
// Tiling Strategy:
// - Work-group tile: 128(M) × 128(N)
// - SLM tile: A[128×16], B[16×128] per K-iteration
// - DPAS tile: 8×16 per subgroup (k=16)
// - Register tile: 8×8 per work-item

#define TILE_M 128      // Work-group M dimension
#define TILE_N 128      // Work-group N dimension
#define TILE_K 16       // K-dimension per iteration (DPAS requirement)
#define THREAD_M 8      // Rows per work-item
#define THREAD_N 8      // Cols per work-item
#define SUBGROUP_SIZE 16

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group and local IDs
    const int group_m = get_group_id(1);
    const int group_n = get_group_id(0);
    const int local_m = get_local_id(1);  // 0..7
    const int local_n = get_local_id(0);  // 0..15 (subgroup)
    const int sg_id = get_sub_group_id(); // 0..7
    const int sg_lid = get_sub_group_local_id(); // 0..15

    // Global output position
    const int global_m = group_m * TILE_M + local_m * THREAD_M;
    const int global_n = group_n * TILE_N + local_n * THREAD_N;

    // Shared Local Memory for cooperative loading
    __local half slm_A[TILE_M * TILE_K];  // 128×16
    __local half slm_B[TILE_K * TILE_N];  // 16×128

    // Accumulation registers (float for precision)
    float acc[THREAD_M][THREAD_N];

    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Loop over K dimension in TILE_K chunks
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_offset = k_tile * TILE_K;
        const int k_size = min(TILE_K, K - k_offset);

        // Cooperatively load A tile into SLM
        // Each work-item loads multiple elements
        const int local_idx = local_m * 16 + local_n; // 0..127
        const int loads_per_wi = (TILE_M * TILE_K) / 128; // 16 loads per work-item

        #pragma unroll 4
        for (int l = 0; l < loads_per_wi; ++l) {
            const int slm_offset = local_idx + l * 128;
            const int slm_row = slm_offset / TILE_K;
            const int slm_col = slm_offset % TILE_K;
            const int global_row = group_m * TILE_M + slm_row;
            const int global_col = k_offset + slm_col;

            if (global_row < M && global_col < K) {
                slm_A[slm_offset] = A[global_row * K + global_col];
            } else {
                slm_A[slm_offset] = (half)0.0f;
            }
        }

        // Cooperatively load B tile into SLM
        #pragma unroll 4
        for (int l = 0; l < loads_per_wi; ++l) {
            const int slm_offset = local_idx + l * 128;
            const int slm_row = slm_offset / TILE_N;
            const int slm_col = slm_offset % TILE_N;
            const int global_row = k_offset + slm_row;
            const int global_col = group_n * TILE_N + slm_col;

            if (global_row < K && global_col < N) {
                slm_B[slm_offset] = B[global_row * N + global_col];
            } else {
                slm_B[slm_offset] = (half)0.0f;
            }
        }

        // Synchronize to ensure SLM is populated
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute using DPAS instruction
        // DPAS computes: C[8×16] += A[8×16] * B[16×16]
        // We need to manually accumulate for THREAD_N=8 output cols

        // Process THREAD_N outputs in chunks compatible with DPAS
        #pragma unroll
        for (int n_chunk = 0; n_chunk < THREAD_N; ++n_chunk) {
            // Load A fragment for this work-item's rows
            half a_frag[THREAD_M];
            #pragma unroll
            for (int i = 0; i < THREAD_M; ++i) {
                const int slm_a_row = local_m * THREAD_M + i;
                // For DPAS, we need k-dimension data
                // Simplified: accumulate manually since DPAS intrinsic has specific layout

                // Manual dot product (fallback if DPAS not directly applicable)
                #pragma unroll
                for (int kk = 0; kk < k_size; ++kk) {
                    const int slm_b_col = local_n * THREAD_N + n_chunk;
                    half a_val = slm_A[slm_a_row * TILE_K + kk];
                    half b_val = slm_B[kk * TILE_N + slm_b_col];
                    acc[i][n_chunk] += convert_float(a_val) * convert_float(b_val);
                }
            }
        }

        // Synchronize before next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        const int out_row = global_m + i;
        if (out_row >= M) continue;

        #pragma unroll
        for (int j = 0; j < THREAD_N; ++j) {
            const int out_col = global_n + j;
            if (out_col < N) {
                C[out_row * N + out_col] = convert_half(acc[i][j]);
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
// Suggested launch metadata (host-side):
//   GWS = { round_up(N, 16), round_up(M, 16) }
//   LWS = { 16, 16 }
//   Expected subgroup size: 16 (Intel Xe/XMX friendly)
//   Each WG computes one 16x16 C tile.
//
// Notes:
// - Uses Intel DPAS intrinsic in K-blocks of 16.
// - Accumulation is float, output cast to half (matches reference semantics).
// - Handles arbitrary M, N, K with boundary checks and K-tail.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int gcol = get_global_id(0); // N dimension
    const int grow = get_global_id(1); // M dimension

    if (grow >= M || gcol >= N) return;

    // Float accumulation to match reference.
    float acc = 0.0f;

    // Main DPAS path: process K in chunks of 16.
    int k = 0;
    for (; k + 15 < K; k += 16) {
        // Load 16 A and 16 B elements corresponding to this output (grow, gcol).
        // For DPAS usage, pack these into half vectors.
        half16 a_vec = (half16)(
            A[grow * K + (k + 0)],  A[grow * K + (k + 1)],
            A[grow * K + (k + 2)],  A[grow * K + (k + 3)],
            A[grow * K + (k + 4)],  A[grow * K + (k + 5)],
            A[grow * K + (k + 6)],  A[grow * K + (k + 7)],
            A[grow * K + (k + 8)],  A[grow * K + (k + 9)],
            A[grow * K + (k + 10)], A[grow * K + (k + 11)],
            A[grow * K + (k + 12)], A[grow * K + (k + 13)],
            A[grow * K + (k + 14)], A[grow * K + (k + 15)]
        );

        half16 b_vec = (half16)(
            B[(k + 0)  * N + gcol], B[(k + 1)  * N + gcol],
            B[(k + 2)  * N + gcol], B[(k + 3)  * N + gcol],
            B[(k + 4)  * N + gcol], B[(k + 5)  * N + gcol],
            B[(k + 6)  * N + gcol], B[(k + 7)  * N + gcol],
            B[(k + 8)  * N + gcol], B[(k + 9)  * N + gcol],
            B[(k + 10) * N + gcol], B[(k + 11) * N + gcol],
            B[(k + 12) * N + gcol], B[(k + 13) * N + gcol],
            B[(k + 14) * N + gcol], B[(k + 15) * N + gcol]
        );

        // DPAS MMA step (k16). Accumulate into float.
        // Signature may vary slightly by toolchain version; this is the intended Intel intrinsic.
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc);
    }

    // Tail for K not multiple of 16.
    for (; k < K; ++k) {
        acc += convert_float(A[grow * K + k]) * convert_float(B[k * N + gcol]);
    }

    C[grow * N + gcol] = convert_half(acc);
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.

(II) Bottlenecks and suggestions for improvement:
1. `half16 a_vec = (half16)( A[grow * K + (k + 0)], ... A[grow * K + (k + 15)] );` and  
   `half16 b_vec = (half16)( B[(k + 0) * N + gcol], ... B[(k + 15) * N + gcol] );`:  
   You’re doing **32 scalar global loads per K-step per thread** plus heavy register packing. This creates high memory traffic and instruction overhead before DPAS even runs.  
   **Improve it by tiling and cooperative loads into local memory**: have the 16x16 workgroup load a tile of A and B once (`__local half Asub[16][BK]`, `Bsub[BK][16]` with BK=16/32), barrier, then each thread reuses tile values. This removes redundant global loads across threads and greatly improves cache/local-memory reuse. Also switch to vectorized loads (`vload8`/`vload16`) where layout allows.

2. `acc = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc);` (inside a kernel where each thread computes one scalar `C[grow * N + gcol]`):  
   The DPAS intrinsic is most effective when a **subgroup cooperatively computes a matrix tile fragment**, not when each lane independently builds vectors for one dot product. In your mapping, subgroup-level compute density is low relative to load overhead.  
   **Improve mapping to subgroup tile MMA**: assign each subgroup to a small C tile (e.g., 8x16 or 16x16 fragment), keep fragment accumulators in registers, and feed DPAS with subgroup-fragment operands loaded in a layout DPAS expects. This raises arithmetic intensity and makes DPAS utilization much better.

3. `for (; k < K; ++k) { acc += convert_float(A[grow * K + k]) * convert_float(B[k * N + gcol]); }`:  
   The scalar tail loop introduces branch/loop overhead and breaks the fast path; for many shapes this becomes noticeable, especially with small/medium K.  
   **Improve by padding or masked vector tail handling**: process K in larger unrolled chunks (e.g., 32/64 via 2–4 DPAS ops per iteration) and handle remainder with a single masked/vectorized path instead of scalar per-element loop. If padding K to multiple of 16 is possible at dispatch time, do that and remove this scalar tail entirely.

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
1. Async Memory Operations: Use async_work_group_copy() to overlap computation with memory transfers between global and local memory. Use async_work_group_strided_copy() for non-contiguous data. Wait with wait_group_events().
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

- **Shared Local Memory (SLM) Tiling**: Cache frequently accessed data in SLM using `group_local_memory_for_overwrite`. Synchronize with `group_barrier()` after writes and before reads. Use 16×16 or 32×32 tiles for float data.
- **Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade recomputation for memory savings (e.g., Flash-Attention style). Maintain running accumulators across blocks with proper rescaling.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.