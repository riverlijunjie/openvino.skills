

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Optimization goals:
- Use Intel OpenCL DPAS instruction when possible, e.g. intel_sub_group_f16_f16_matrix_mad_k16.
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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 40.200):
```OCL
// Optimized FP16 matmul using Intel DPAS (XMX) for Battlemage
// Each subgroup computes 8x32 output (2x DPAS 8x16 per K step)
// Workgroup: 8 subgroups arranged as 4(row) x 2(col), WG tile: 32x64
// But we double the N-tile per subgroup: each SG does 8 rows x 32 cols
// WG tile: 32x64, SG grid: 4x2, each SG: 8x32
// GWS = (ceil_div(N,64)*16, ceil_div(M,32)*8)
// LWS = (16, 8)
// Subgroup size = 16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 16
#define SG_SIZE 16
#define SG_TILE_M 8
#define SG_TILE_N 32

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 8, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int wg_col = get_group_id(0) * TILE_N;
    const int wg_row = get_group_id(1) * TILE_M;

    const int lid_x = get_local_id(0); // 0..15 (subgroup lane)
    const int lid_y = get_local_id(1); // 0..7  (subgroup index)

    // Map 8 subgroups into 4x2 grid
    const int sg_row = lid_y / 2;  // 0..3
    const int sg_col = lid_y % 2;  // 0..1

    const int out_row = wg_row + sg_row * SG_TILE_M;
    const int out_col = wg_col + sg_col * SG_TILE_N;

    const int sg_lane = get_sub_group_local_id();

    // Accumulators: 8 rows x 32 cols = 2 DPAS tiles side by side
    float8 acc0 = (float8)(0.0f); // rows 0-7, cols 0-15
    float8 acc1 = (float8)(0.0f); // rows 0-7, cols 16-31

    // SLM for cooperative loading
    __local half A_slm[TILE_M * TILE_K];  // 32 x 16
    __local half B_slm[TILE_K * TILE_N];  // 16 x 64

    const int num_wi = 16 * 8; // 128 work-items

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperative load A_slm: 32*16 = 512 halfs, 128 WIs => 4 each
        {
            int linear_id = lid_y * 16 + lid_x;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = linear_id + i * num_wi;
                int r = idx / TILE_K;
                int c = idx % TILE_K;
                int gr = wg_row + r;
                int gc = k0 + c;
                A_slm[r * TILE_K + c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0;
            }
        }

        // Cooperative load B_slm: 16*64 = 1024 halfs, 128 WIs => 8 each
        {
            int linear_id = lid_y * 16 + lid_x;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = linear_id + i * num_wi;
                int r = idx / TILE_N;
                int c = idx % TILE_N;
                int gr = k0 + r;
                int gc = wg_col + c;
                B_slm[r * TILE_N + c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Load A for this subgroup: 8x16 from A_slm
        short8 a_packed;
        {
            int a_base = sg_row * SG_TILE_M;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                half v = A_slm[(a_base + r) * TILE_K + sg_lane];
                ((short*)&a_packed)[r] = as_short(v);
            }
        }

        // Load B for first 16 cols
        int8 b_int0;
        {
            int b_col_base = sg_col * SG_TILE_N;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int r0 = 2 * p;
                int r1 = 2 * p + 1;
                ushort s0 = as_ushort(B_slm[r0 * TILE_N + b_col_base + sg_lane]);
                ushort s1 = as_ushort(B_slm[r1 * TILE_N + b_col_base + sg_lane]);
                b_int0[p] = (int)s0 | ((int)s1 << 16);
            }
        }

        // Load B for second 16 cols
        int8 b_int1;
        {
            int b_col_base = sg_col * SG_TILE_N + 16;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int r0 = 2 * p;
                int r1 = 2 * p + 1;
                ushort s0 = as_ushort(B_slm[r0 * TILE_N + b_col_base + sg_lane]);
                ushort s1 = as_ushort(B_slm[r1 * TILE_N + b_col_base + sg_lane]);
                b_int1[p] = (int)s0 | ((int)s1 << 16);
            }
        }

        // Two DPAS calls, reusing A
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_int0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_int1, acc1);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: 8 rows x 32 cols
    // acc0: 8 rows, cols out_col+0..15 (lane selects column)
    // acc1: 8 rows, cols out_col+16..31
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int gr = out_row + r;
        int gc0 = out_col + sg_lane;
        if (gr < M && gc0 < N) {
            C[gr * N + gc0] = convert_half(((float*)&acc0)[r]);
        }
        int gc1 = out_col + 16 + sg_lane;
        if (gr < M && gc1 < N) {
            C[gr * N + gc1] = convert_half(((float*)&acc1)[r]);
        }
    }
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
// Optimized FP16 matmul using Intel DPAS (XMX) instructions for Battlemage
// Each subgroup computes 8x16 output tile via intel_sub_group_f16_f16_matrix_mad_k16
// Workgroup: 16 subgroups arranged as 4(row) x 4(col), each 8x16 => WG tile: 32x64
// GWS = (ceil_div(N,64)*16, ceil_div(M,32)*16)
// LWS = (16, 16)
// Subgroup size = 16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 16
#define SG_SIZE 16
#define SG_TILE_M 8
#define SG_TILE_N 16

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Workgroup tile origin
    const int wg_col = get_group_id(0) * TILE_N;  // in output columns
    const int wg_row = get_group_id(1) * TILE_M;  // in output rows

    // Subgroup ID within workgroup
    const int lid_x = get_local_id(0); // 0..15
    const int lid_y = get_local_id(1); // 0..15
    const int sg_linear = lid_y;       // subgroup index = local_id(1) since x is the subgroup lane

    // Map 16 subgroups into 4x4 grid
    const int sg_row = sg_linear / 4;  // 0..3
    const int sg_col = sg_linear % 4;  // 0..3

    // This subgroup's output tile origin
    const int out_row = wg_row + sg_row * SG_TILE_M;  // 8 rows
    const int out_col = wg_col + sg_col * SG_TILE_N;  // 16 cols

    // Accumulator: 8 floats per work-item (8 rows, column = lane id)
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // SLM for B tile: TILE_K x TILE_N = 16 x 64 halfs
    __local half B_slm[TILE_K * TILE_N];

    const int sg_lane = get_sub_group_local_id();

    // Loop over K dimension in blocks of TILE_K
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperatively load B tile [k0:k0+16, wg_col:wg_col+64] into SLM
        // Total elements: 16*64 = 1024 halfs, 256 work-items => 4 each
        {
            int linear_id = lid_y * 16 + lid_x; // 0..255
            for (int i = 0; i < 4; i++) {
                int idx = linear_id + i * 256;
                int bk = idx / TILE_N;  // row in B tile (0..15)
                int bn = idx % TILE_N;  // col in B tile (0..63)
                int global_k = k0 + bk;
                int global_n = wg_col + bn;
                half val = 0;
                if (global_k < K && global_n < N)
                    val = B[global_k * N + global_n];
                B_slm[bk * TILE_N + bn] = val;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Load A tile for this subgroup: 8 rows x 16 cols
        // For DPAS: A is 8x16 matrix, each WI holds 8 half values (one per row, broadcast across K)
        // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
        //   a: int8 (8 x short packed) per WI - 8 rows, k distributed across lanes
        //   b: int8 (8 x short packed) per WI - k distributed across lanes, 16 cols
        //   acc: float8 per WI

        // A matrix [8 x 16]: row-major, each WI reads one column (lane-th)
        // Packed as: each WI holds 8 halfs = rows 0..7 at column=lane
        // But DPAS expects specific layout. Let's use the sub_group block read.

        // For DPAS A input: 8 rows x 16 K, stored as 8 ushort per WI
        // WI[lane] holds A[row][lane] for row=0..7 => need A[out_row+r][k0+lane]
        short8 a_packed;
        {
            half a_vals[8];
            for (int r = 0; r < 8; r++) {
                int gr = out_row + r;
                int gc = k0 + sg_lane;
                if (gr < M && gc < K)
                    a_vals[r] = A[gr * K + gc];
                else
                    a_vals[r] = (half)0;
            }
            a_packed = (short8)(
                as_short(a_vals[0]), as_short(a_vals[1]),
                as_short(a_vals[2]), as_short(a_vals[3]),
                as_short(a_vals[4]), as_short(a_vals[5]),
                as_short(a_vals[6]), as_short(a_vals[7])
            );
        }

        // For DPAS B input: 16 K x 16 N, stored as 8 ushort per WI
        // B is 16x16 submatrix from B_slm at column offset sg_col*16
        // WI[lane] holds B[k][lane] for pairs of k => 8 x (2 halfs packed as int)
        // Actually layout: each WI holds 8 shorts = B[2*i][lane] and B[2*i+1][lane] interleaved
        short8 b_packed;
        {
            half b_vals[8];
            int b_col_base = sg_col * SG_TILE_N;
            // Each WI reads column sg_lane from B_slm, rows 0..15
            // Packed as pairs: (row0,row1), (row2,row3), ... BUT DPAS expects
            // 8 ushort per WI where each ushort is one row at the lane's column
            // For k16: 16 rows, so we need 16 halfs, packed as 8 ints (pairs)
            // Actually intel_sub_group_f16_f16_matrix_mad_k16 takes int8 for B
            int8 b_int;
            for (int p = 0; p < 8; p++) {
                int r0 = 2 * p;
                int r1 = 2 * p + 1;
                half v0 = B_slm[r0 * TILE_N + b_col_base + sg_lane];
                half v1 = B_slm[r1 * TILE_N + b_col_base + sg_lane];
                // Pack two halfs into one int
                ushort s0 = as_ushort(v0);
                ushort s1 = as_ushort(v1);
                b_int[p] = (int)s0 | ((int)s1 << 16);
            }

            // Call DPAS
            float8 acc_vec = (float8)(acc[0], acc[1], acc[2], acc[3],
                                       acc[4], acc[5], acc[6], acc[7]);
            acc_vec = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_int, acc_vec);
            acc[0] = acc_vec.s0; acc[1] = acc_vec.s1;
            acc[2] = acc_vec.s2; acc[3] = acc_vec.s3;
            acc[4] = acc_vec.s4; acc[5] = acc_vec.s5;
            acc[6] = acc_vec.s6; acc[7] = acc_vec.s7;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: each WI writes 8 values (8 rows at column out_col+sg_lane)
    for (int r = 0; r < 8; r++) {
        int gr = out_row + r;
        int gc = out_col + sg_lane;
        if (gr < M && gc < N) {
            C[gr * N + gc] = convert_half(acc[r]);
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 53.200):
```OCL
// Tiled FP16 matmul with SLM blocking
// Each work-item computes a 4x4 sub-tile of C
// Workgroup tile: 64x64, K-tile: 16
// GWS = (ceil_div(N,64)*16, ceil_div(M,64)*16)
// LWS = (16, 16)
// Subgroup size: 16 recommended
__attribute__((reqd_work_group_size(16, 16, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Workgroup tile: 64 rows x 64 cols
    // Each WI computes 4 rows x 4 cols
    const int lx = get_local_id(0);  // 0..15
    const int ly = get_local_id(1);  // 0..15

    const int wg_col = get_group_id(0) * 64;
    const int wg_row = get_group_id(1) * 64;

    // Each WI responsible for rows [wi_row, wi_row+3] and cols [wi_col, wi_col+3]
    const int wi_row = wg_row + ly * 4;
    const int wi_col = wg_col + lx * 4;

    // Accumulators: 4x4
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    // SLM tiles
    __local half A_tile[64][16]; // 64 rows x 16 K
    __local half B_tile[16][64]; // 16 K x 64 cols

    const int lid = ly * 16 + lx; // 0..255

    for (int k0 = 0; k0 < K; k0 += 16) {
        // Cooperative load A_tile: 64*16 = 1024 halfs, 256 WIs -> 4 each
        for (int i = 0; i < 4; i++) {
            int idx = lid * 4 + i;
            int r = idx / 16;
            int c = idx % 16;
            int gr = wg_row + r;
            int gc = k0 + c;
            A_tile[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Cooperative load B_tile: 16*64 = 1024 halfs, 256 WIs -> 4 each
        for (int i = 0; i < 4; i++) {
            int idx = lid * 4 + i;
            int r = idx / 64;
            int c = idx % 64;
            int gr = k0 + r;
            int gc = wg_col + c;
            B_tile[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 4x4 accumulation over K=16
        for (int kk = 0; kk < 16; kk++) {
            float a0 = convert_float(A_tile[ly * 4 + 0][kk]);
            float a1 = convert_float(A_tile[ly * 4 + 1][kk]);
            float a2 = convert_float(A_tile[ly * 4 + 2][kk]);
            float a3 = convert_float(A_tile[ly * 4 + 3][kk]);

            float b0 = convert_float(B_tile[kk][lx * 4 + 0]);
            float b1 = convert_float(B_tile[kk][lx * 4 + 1]);
            float b2 = convert_float(B_tile[kk][lx * 4 + 2]);
            float b3 = convert_float(B_tile[kk][lx * 4 + 3]);

            acc[0][0] += a0 * b0; acc[0][1] += a0 * b1; acc[0][2] += a0 * b2; acc[0][3] += a0 * b3;
            acc[1][0] += a1 * b0; acc[1][1] += a1 * b1; acc[1][2] += a1 * b2; acc[1][3] += a1 * b3;
            acc[2][0] += a2 * b0; acc[2][1] += a2 * b1; acc[2][2] += a2 * b2; acc[2][3] += a2 * b3;
            acc[3][0] += a3 * b0; acc[3][1] += a3 * b1; acc[3][2] += a3 * b2; acc[3][3] += a3 * b3;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int gr = wi_row + i;
            int gc = wi_col + j;
            if (gr < M && gc < N) {
                C[gr * N + gc] = convert_half(acc[i][j]);
            }
        }
    }
}
```

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] PASSED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] PASSED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] PASSED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] PASSED         [100%]

======================= 4 passed, 1 deselected in 0.89s ========================
The kernel compiles and is correct, great job!

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
1. Avoid Bank Conflicts: Local memory is organized into banks (typically 32 banks). Pad shared arrays to avoid stride conflicts, e.g., __local float tile[TILE_SIZE][TILE_SIZE + 1] for transpose operations. Use sequential access patterns within wavefronts.
2. Use Appropriate Data Types: Use half (FP16) when precision allows - doubles throughput on many GPUs. Use native_* math functions (native_sqrt, native_divide) for faster, less precise operations. Use mad() for fused multiply-add. Size integer types appropriately (char, short, int).

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
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.