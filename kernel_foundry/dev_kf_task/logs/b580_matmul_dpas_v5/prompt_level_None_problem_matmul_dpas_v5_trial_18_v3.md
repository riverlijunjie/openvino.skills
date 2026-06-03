

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

// -------------------------------------------------------------------------------------------------
// FP16 GEMM: C[M,N] = A[M,K] x B[K,N]
// Inputs/Output: half, accumulation: float (exact reference semantics)
//
// Launch metadata (recommended):
//   LWS = {16, 16, 1}                       // 256 threads / WG
//   GWS = {ceil_div(N,16)*16, ceil_div(M,32)*16, 1}
//   Subgroup: intel_reqd_sub_group_size(16)
//
// Mapping:
//   - WG computes a 32x16 C tile (2x16 rows, 16 cols)
//   - Each thread computes 2 outputs: (row0,col) and (row1,col)
//   - K blocked by BK=32, with 2 DPAS(k16) per block
//   - B is stored in SLM transposed as BsubT[col][k], enabling contiguous vload16 per lane
//
// Notes:
//   - Uses Intel DPAS intrinsic explicitly:
//       intel_sub_group_f16_f16_matrix_mad_k16
//   - Full boundary correctness for arbitrary M/N/K
// -------------------------------------------------------------------------------------------------

#define TM_PER_HALF 16
#define TM_TOTAL    32
#define TN          16
#define BK          32

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
    const int lx = get_local_id(0);   // lane in subgroup / output column within tile
    const int ly = get_local_id(1);   // selects output row within each 16-row half

    const int gx = get_group_id(0);
    const int gy = get_group_id(1);

    const int tile_col = gx * TN;
    const int tile_row = gy * TM_TOTAL;

    const int col  = tile_col + lx;
    const int row0 = tile_row + ly;        // top 16 rows
    const int row1 = tile_row + 16 + ly;   // bottom 16 rows

    // SLM tiles:
    // A: two 16xBK panels (top/bottom)
    // B: transposed layout [col][k] for contiguous lane-local vector loads
    __local half Asub0[TM_PER_HALF][BK + 1];
    __local half Asub1[TM_PER_HALF][BK + 1];
    __local half BsubT[TN][BK + 1];

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Full BK blocks
    const int k_full_end = (K / BK) * BK;

    for (int kb = 0; kb < k_full_end; kb += BK) {
        // Cooperative load A top/bottom panels: each thread loads 2 elems per panel (BK=32, lx=0..15)
        // top half
        {
            const int gk0 = kb + lx;
            const int gk1 = kb + 16 + lx;

            Asub0[ly][lx]      = (row0 < M && gk0 < K) ? A[row0 * K + gk0] : (half)0.0h;
            Asub0[ly][16 + lx] = (row0 < M && gk1 < K) ? A[row0 * K + gk1] : (half)0.0h;

            Asub1[ly][lx]      = (row1 < M && gk0 < K) ? A[row1 * K + gk0] : (half)0.0h;
            Asub1[ly][16 + lx] = (row1 < M && gk1 < K) ? A[row1 * K + gk1] : (half)0.0h;
        }

        // Cooperative load B tile in transposed SLM:
        // B[k, col] -> BsubT[col_local][k_local]
        {
            const int gc = tile_col + lx;
            const int gk0 = kb + ly;
            const int gk1 = kb + 16 + ly;

            BsubT[lx][ly]      = (gc < N && gk0 < K) ? B[gk0 * N + gc] : (half)0.0h;
            BsubT[lx][16 + ly] = (gc < N && gk1 < K) ? B[gk1 * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (col < N) {
            // lane-local contiguous B vectors (no scalar gather)
            half16 b0 = vload16(0, &BsubT[lx][0]);
            half16 b1 = vload16(0, &BsubT[lx][16]);

            // A vectors for row0 and row1
            half16 a00 = vload16(0, &Asub0[ly][0]);
            half16 a01 = vload16(0, &Asub0[ly][16]);
            half16 a10 = vload16(0, &Asub1[ly][0]);
            half16 a11 = vload16(0, &Asub1[ly][16]);

            // DPAS accumulation (required intrinsic)
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Tail K cleanup (exact)
    if (col < N) {
        for (int k = k_full_end; k < K; ++k) {
            const float b = convert_float(B[k * N + col]);
            if (row0 < M) acc0 += convert_float(A[row0 * K + k]) * b;
            if (row1 < M) acc1 += convert_float(A[row1 * K + k]) * b;
        }

        if (row0 < M) C[row0 * N + col] = convert_half(acc0);
        if (row1 < M) C[row1 * N + col] = convert_half(acc1);
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.

(II) Bottlenecks and suggestions for improvement:
1. `if (col < N) { ... DPAS ... }` inside the `kb` loop: this creates control-flow divergence at the right edge tiles and keeps the whole subgroup paying branch overhead every BK step.  
   **Improve it like this:** split the kernel into a **fast interior path** and an **edge path**. In the fast path (full tiles where `tile_col + 15 < N` and `tile_row + 31 < M`), remove all inner-loop bounds checks entirely and run pure DPAS. Keep your current guarded logic only for fringe work-groups. This usually gives a noticeable speedup because the hot loop becomes branch-free and compiler scheduling improves.

2. `barrier(CLK_LOCAL_MEM_FENCE);` (both barriers per BK iteration): you synchronize the full 16x16 work-group twice every K-block even though compute is subgroup-centric and B is reused identically by all lanes.  
   **Improve it like this:** reduce synchronization pressure with **double-buffered SLM ping-pong** (`buffer = kb & 1`) and prefetch next BK tile while computing current tile. That lets you keep one mandatory handoff barrier per stage instead of a strict load-barrier-compute-barrier pattern. Also consider restructuring so each subgroup owns its own A panel slice to avoid unnecessary cross-subgroup coupling; fewer full-WG barriers generally improves EU occupancy/latency hiding.

3. `half16 b0 = vload16(0, &BsubT[lx][0]);` and `half16 b1 = vload16(0, &BsubT[lx][16]);` (plus matching A vloads each iteration): reloading vectors from SLM every BK step is costly, and your B layout causes strided global loads during staging (`B[gk * N + gc]` per thread), which is not ideal for cache-line utilization.  
   **Improve it like this:** have each lane load B from global as wider contiguous chunks (e.g., `vload_half{4,8}` where possible) and store to SLM in a bank-friendly packed layout; then feed DPAS directly from registers when possible (or at least minimize repeated SLM vector reloads by unrolling BK and reusing loaded fragments across multiple MADs). In short: better global coalescing for B staging + more register reuse, less SLM traffic in the inner loop.

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
2. Minimize Synchronization: Reduce barrier() calls. Use barrier(CLK_LOCAL_MEM_FENCE) for local memory only, barrier(CLK_GLOBAL_MEM_FENCE) when needed for global. Sub-group operations are barrier-free. Use work-group-level reductions only when necessary.

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
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.