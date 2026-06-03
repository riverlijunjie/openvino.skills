

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
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Optimized FP16 MatMul using Intel DPAS (Dot Product Accumulate Systolic)
// Launch configuration:
//   GWS: [(N+31)/32 * 16, (M+31)/32 * 2, 1]
//   LWS: [16, 2, 1]  (32 work-items = 2 sub-groups of size 16)
//   Sub-group size: 16
//
// Performance characteristics:
//   - Uses intel_sub_group_f16_f16_matrix_mad_k16 for XMX acceleration
//   - Tile size: 32x32 output, K-tile: 16
//   - Each work-group processes 32x32 output elements
//   - SLM usage: 2KB for A tile + 2KB for B tile = 4KB per work-group
//
// C[M,N] = A[M,K] x B[K,N]

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define SG_SIZE 16

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group and sub-group identification
    const int sg_id = get_sub_group_id();           // 0 or 1 (2 sub-groups per work-group)
    const int sg_lid = get_sub_group_local_id();    // 0-15 within sub-group
    const int local_id = get_local_id(0) + get_local_id(1) * get_local_size(0);

    // Global tile position
    const int tile_row = get_group_id(1) * TILE_M;
    const int tile_col = get_group_id(0) * TILE_N;

    // Shared Local Memory for A and B tiles
    __local half A_tile[TILE_M * TILE_K];  // 32x16
    __local half B_tile[TILE_K * TILE_N];  // 16x32

    // Accumulators: each work-item accumulates 8x1 elements (using DPAS)
    // Sub-group layout: sg_id determines row block (0->rows 0-15, 1->rows 16-31)
    float acc[8];
    for (int i = 0; i < 8; i++) {
        acc[i] = 0.0f;
    }

    // Loop over K dimension in tiles of 16
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperatively load A tile [TILE_M x TILE_K] into SLM
        // 32 work-items load 32*16=512 half elements
        const int loads_per_item = (TILE_M * TILE_K) / 32;  // 16 loads per work-item
        for (int i = 0; i < loads_per_item; i++) {
            int idx = local_id * loads_per_item + i;
            int row = idx / TILE_K;
            int col = idx % TILE_K;
            int global_row = tile_row + row;
            int global_col = k_tile + col;

            if (global_row < M && global_col < K) {
                A_tile[row * TILE_K + col] = A[global_row * K + global_col];
            } else {
                A_tile[row * TILE_K + col] = 0.0h;
            }
        }

        // Cooperatively load B tile [TILE_K x TILE_N] into SLM
        for (int i = 0; i < loads_per_item; i++) {
            int idx = local_id * loads_per_item + i;
            int row = idx / TILE_N;
            int col = idx % TILE_N;
            int global_row = k_tile + row;
            int global_col = tile_col + col;

            if (global_row < K && global_col < N) {
                B_tile[row * TILE_N + col] = B[global_row * N + global_col];
            } else {
                B_tile[row * TILE_N + col] = 0.0h;
            }
        }

        // Synchronize to ensure all data is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute using DPAS: each sub-group processes 16 rows (determined by sg_id)
        // Each work-item in sub-group handles different columns via SIMD

        // DPAS operation: C += A x B
        // intel_sub_group_f16_f16_matrix_mad_k16(C, A, B)
        // A: 8x16 (8 rows per work-item, k=16)
        // B: 16x1 (k=16, broadcasted across sub-group for different columns)
        // C: 8x1 accumulator

        int row_base = sg_id * 16;  // Sub-group 0: rows 0-15, Sub-group 1: rows 16-31

        // Each work-item processes 8 consecutive rows and sg_lid determines column
        for (int i = 0; i < 8; i++) {
            int row = row_base + (i < 4 ? sg_lid / 2 : 8 + sg_lid / 2);

            // Load 16 elements from A_tile row
            half8 a_vec0, a_vec1;
            for (int k = 0; k < 8; k++) {
                if (k < 8) {
                    a_vec0[k] = A_tile[row * TILE_K + k];
                    a_vec1[k] = A_tile[row * TILE_K + 8 + k];
                }
            }

            // For each column (distributed via sg_lid)
            int col = sg_lid + (i % 2) * 16;
            if (col < TILE_N) {
                // Load B column
                half b_vec[16];
                for (int k = 0; k < 16; k++) {
                    b_vec[k] = B_tile[k * TILE_N + col];
                }

                // Manual dot product (DPAS emulation for clarity)
                for (int k = 0; k < 16; k++) {
                    half a_val = (k < 8) ? a_vec0[k] : a_vec1[k - 8];
                    acc[i] += convert_float(a_val) * convert_float(b_vec[k]);
                }
            }
        }

        // Synchronize before loading next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results back to global memory
    int row_base = sg_id * 16;
    for (int i = 0; i < 8; i++) {
        int row = row_base + (i < 4 ? sg_lid / 2 : 8 + sg_lid / 2);
        int col = sg_lid + (i % 2) * 16;

        int global_row = tile_row + row;
        int global_col = tile_col + col;

        if (global_row < M && global_col < N) {
            C[global_row * N + global_col] = convert_half(acc[i]);
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel **is incorrect**.

(II) Summary of console output: All 4 correctness tests failed (`wrt_pytorch` and `wrt_reference`, two seeds each). The output matrix has many zeros while a few positions match expected values, which means your kernel is computing only a sparse subset of `C`. There is also a launch warning: `matmul_opt.cl has no parsable reqd_work_group_size; fallback launch (N, M), local=None`, so your intended tiled work-group mapping (`16x2`) is not being used by the harness.

(III) Errors and suggestions for improvement:
1. `for (int i = 0; i < 8; i++) { ... int row = row_base + (i < 4 ? sg_lid / 2 : 8 + sg_lid / 2); ... int col = sg_lid + (i % 2) * 16; }`: this thread-to-output mapping is wrong and highly overlapping. `sg_lid/2` causes two lanes to write the same row, and `col` only toggles between two 16-wide column groups, so most of the 32x32 tile is never produced. Redesign mapping so each lane owns unique `(row,col)` outputs that fully cover 32x32 (or use a smaller tile that exactly matches your subgroup ownership).
2. `__attribute__((intel_reqd_sub_group_size(SG_SIZE)))` with comment-only launch config: your kernel is missing a required work-group-size attribute that the runner parses, so it falls back to global `(N,M)` execution and breaks your tile assumptions. Add `__attribute__((reqd_work_group_size(16,2,1)))` on the kernel so local size is fixed to 32 threads per WG as your indexing expects.
3. `// Uses intel_sub_group_f16_f16_matrix_mad_k16 ...` but compute path is manual dot-product loops with per-element loads from SLM: this is not DPAS and defeats both correctness intent and performance intent. Replace the emulation loop with actual subgroup matrix multiply intrinsics (or a consistent SIMD block-FMA strategy) and align/register-pack A/B fragments to the intrinsic contract.
4. `float acc[8]; ... C[...] = convert_half(acc[i]);`: accumulating to float is fine, but storing directly to FP16 at end is okay only if each output element is complete. Right now each `acc[i]` corresponds to ambiguous shared outputs due to bad mapping. Fix ownership first; then keep one accumulator per unique output element.
5. `half8 a_vec0, a_vec1; for (int k=0; k<8; k++) { ... }`: the `if (k < 8)` inside that loop is dead code and indicates fragmented logic. More importantly, this per-`i` reload of A and per-`i` reload of a full B column is bandwidth-heavy and non-coalesced for SLM. Load reusable fragments once per K-tile per lane/subgroup and reuse across multiple output accumulators.
6. `local_id = get_local_id(0) + get_local_id(1) * get_local_size(0);` plus two independent SLM load loops: load balance is numerically even, but access pattern is not tuned for contiguous vectorized transactions. For Battlemage, load `half8/half16` vectors with contiguous addresses where possible (especially B tile rows) to reduce SLM/global transaction overhead.
7. `barrier(CLK_LOCAL_MEM_FENCE);` placement is correct structurally, but because output mapping is wrong, barriers only synchronize incorrect work. After fixing mapping, keep exactly two barriers per K-tile (post-load, post-compute) and ensure no early exits before barriers.
8. `row_base = sg_id * 16` with only 2 subgroups/WG: this hardcodes exactly 32 tile rows, but your inner mapping doesn’t generate 32 distinct rows × 32 distinct cols. If you keep `TILE_M=TILE_N=32`, explicitly map subgroup 0/1 to disjoint row ranges and each lane to disjoint columns (or vice versa) with no aliasing.
9. Tail handling (`if (global_row < M && global_col < N)`) is fine, but because most outputs are never computed, bounds checks mask logic bugs by writing zeros elsewhere. After remapping, verify that every in-bounds element in a tile gets exactly one writer.
10. Comments claim “optimized FP16 MatMul using Intel DPAS”, but implementation diverges. Either implement true DPAS path end-to-end or rename/restructure to a coherent non-DPAS tiled kernel first, then optimize. Right now the mismatch makes debugging much harder and led to incorrect ownership logic.

## Hardware specification:
Your code will run on the following hardware:
**Intel Battlemage** with specs: Xe-cores: 20, Render Slices: 5, Ray Tracing Units: 20, Intel® Xe Matrix Extensions (Intel® XMX) Engines: 160, Xe Vector Engines: 160, Graphics Clock: 2670, GPU Peak TOPS (Int8): 233, TBP: 190, PCI Express Configurations ‡: PCI Express 4.0 x8, Device ID: 0xE20B, Memory: 12 GB GDDR6, Memory Interface: 192 bit, Memory Bandwidth: 456, Memory Speed: 19, ISA_GPU: Xe2-HPG
Please consider the hardware specifications when improving the code. 

## Task:

**Your objectives**:
1. Analyze the previous kernel and its evaluation log.
2. Identify any errors or mismatches with the reference implementation.
3. Propose specific improvements or fixes, explaining your reasoning step by step.
4. Rewrite the kernel, providing the complete, corrected code in a code block.

**Critical Requirements:**

1. The kernel must exactly match the reference implementation's functionality.
2. The code must compile and run properly on the GPU.
3. Do not cache or reuse previous results; ensure the code executes fully on each run.
4. Keep all hyperparameters (e.g., batch size, dimensions) unchanged as specified in the reference implementation.
8. Beware of the critical error "Unexpected kernel lambda size. In such cases removing constexpr specifier aligns the captures between the host compiler and the device compiler"! Do not capture constexpr variables in lambda functions passed to kernel launches as this can lead to different lambda sizes between the host and device compiler.

Additional Guidance:

1. Clearly comment on any fixes and optimizations you implement.
2. If you change the kernel structure, explain why.
3. Anticipate and address possible runtime or compilation errors.

Please structure your response as follows:

1. Analysis:
    * Summarize the errors found in the previous kernel and log.
    * Explain your proposed changes.
2. Improved OCL code:
    * Provide the complete, corrected OCL code in a code block:
```OCL
Your code here
```


## Required Optimizations

Apply the following optimization techniques in your implementation:

- **Register Blocking**: Each work-item computes a THREAD_M×THREAD_N output block in private register arrays. Use `#pragma unroll` on inner loops. Combine with SLM tiling for multi-level memory hierarchy optimization.
- **Online Algorithms**: Use single-pass algorithms with running statistics. For softmax: maintain running_max and running_sum, rescaling sum when max changes. For variance: use Welford's online algorithm.
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.