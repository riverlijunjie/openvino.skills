

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

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
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

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
// Optimized FP16 MatMul using Intel DPAS (XMX)
// C[M,N] = A[M,K] x B[K,N]
// Input/Output: half, Accumulation: float
//
// Launch configuration:
//   GWS: [((N+31)/32)*16, ((M+31)/32)*2, 1]
//   LWS: [16, 2, 1]  (32 work-items = 2 sub-groups of size 16)
//   Sub-group size: 16 (required)
//
// Tile mapping:
//   - Each work-group processes a 32x32 output tile
//   - K dimension processed in chunks of 16
//   - Each sub-group handles 16 rows of the 32x32 tile
//   - Each work-item accumulates 1x2 output elements via DPAS
//
// Memory layout:
//   - SLM A_tile: 32x16 (1KB)
//   - SLM B_tile: 16x32 (1KB)
//   - Total SLM per work-group: 2KB

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define SG_SIZE 16

__attribute__((reqd_work_group_size(16, 2, 1)))
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
    const int sg_id = get_sub_group_id();           // 0 or 1
    const int sg_lid = get_sub_group_local_id();    // 0-15
    const int lx = get_local_id(0);                 // 0-15
    const int ly = get_local_id(1);                 // 0-1
    const int local_linear = ly * 16 + lx;          // 0-31

    // Global tile position
    const int tile_row = get_group_id(1) * TILE_M;
    const int tile_col = get_group_id(0) * TILE_N;

    // Shared Local Memory for A and B tiles
    __local half A_tile[TILE_M][TILE_K + 1];  // +1 padding to reduce bank conflicts
    __local half B_tile[TILE_K][TILE_N + 1];

    // Each sub-group handles 16 rows (sg_id * 16 + sg_lid gives row within tile)
    // Each work-item accumulates 2 output elements (columns 0-15 and 16-31)
    float acc0 = 0.0f;  // Column offset 0
    float acc1 = 0.0f;  // Column offset 16

    // Row index for this work-item within the 32x32 tile
    const int local_row = sg_id * 16 + sg_lid;
    const int global_row = tile_row + local_row;

    // Loop over K dimension in tiles of 16
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Cooperative load A tile [32 x 16]
        // 32 work-items load 512 half elements (16 loads per work-item)
        #pragma unroll 4
        for (int i = 0; i < 16; i++) {
            int idx = local_linear + i * 32;  // 0 to 511
            if (idx < TILE_M * TILE_K) {
                int tile_r = idx / TILE_K;
                int tile_c = idx % TILE_K;
                int glob_r = tile_row + tile_r;
                int glob_c = k_tile + tile_c;

                A_tile[tile_r][tile_c] = (glob_r < M && glob_c < K) 
                    ? A[glob_r * K + glob_c] 
                    : (half)0.0h;
            }
        }

        // Cooperative load B tile [16 x 32]
        // 32 work-items load 512 half elements (16 loads per work-item)
        #pragma unroll 4
        for (int i = 0; i < 16; i++) {
            int idx = local_linear + i * 32;  // 0 to 511
            if (idx < TILE_K * TILE_N) {
                int tile_r = idx / TILE_N;
                int tile_c = idx % TILE_N;
                int glob_r = k_tile + tile_r;
                int glob_c = tile_col + tile_c;

                B_tile[tile_r][tile_c] = (glob_r < K && glob_c < N) 
                    ? B[glob_r * N + glob_c] 
                    : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute using DPAS: intel_sub_group_f16_f16_matrix_mad_k16
        // Each work-item processes row=local_row, computing 2 output elements

        if (global_row < M) {
            // Load A row vector (1x16)
            half16 a_vec = vload16(0, &A_tile[local_row][0]);

            // First output column (sg_lid = 0-15)
            {
                int col0 = sg_lid;
                int global_col0 = tile_col + col0;

                if (global_col0 < N) {
                    // Gather B column vector (16x1) for this column
                    half16 b_vec;
                    #pragma unroll
                    for (int kk = 0; kk < 16; kk++) {
                        b_vec[kk] = B_tile[kk][col0];
                    }

                    // DPAS operation: accumulate a_vec · b_vec
                    acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc0);
                }
            }

            // Second output column (sg_lid + 16)
            {
                int col1 = sg_lid + 16;
                int global_col1 = tile_col + col1;

                if (global_col1 < N) {
                    // Gather B column vector (16x1) for this column
                    half16 b_vec;
                    #pragma unroll
                    for (int kk = 0; kk < 16; kk++) {
                        b_vec[kk] = B_tile[kk][col1];
                    }

                    // DPAS operation: accumulate a_vec · b_vec
                    acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc1);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results back to global memory
    if (global_row < M) {
        // First column
        {
            int global_col0 = tile_col + sg_lid;
            if (global_col0 < N) {
                C[global_row * N + global_col0] = convert_half(acc0);
            }
        }

        // Second column
        {
            int global_col1 = tile_col + sg_lid + 16;
            if (global_col1 < N) {
                C[global_row * N + global_col1] = convert_half(acc1);
            }
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 34.000):
```OCL
// Intel Xe2 (Battlemage) tuned FP16 GEMM
// C[M,N] = A[M,K] x B[K,N], A/B/C: half, accumulation: float
//
// Launch metadata (recommended host-side):
//   LWS = {16, 2, 2}   // 64 WI/WG = 4 subgroups (SG size 16)
//   GWS = {ceil_div(N,32)*16, ceil_div(M,32)*2, 2}
//   reqd_sub_group_size(16)
//
// Mapping:
//   - One WG computes a 32x32 C tile.
//   - WG has 2x2 subgroups; each subgroup computes one 16x16 subtile.
//   - Each lane computes 16 rows x 2 cols (two 8-row float8 accumulators per col).
//   - K processed in chunks of 16 via DPAS intrinsic.
//
// Notes:
//   - Uses Intel DPAS: intel_sub_group_f16_f16_matrix_mad_k16
//   - Tail-K handled exactly with scalar FP32 accumulation.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define SG_SIZE   16
#define TILE_M    32
#define TILE_N    32
#define TILE_K    16

__attribute__((reqd_work_group_size(16,2,2)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lane = get_local_id(0);      // 0..15 (subgroup lane)
    const int sgy  = get_local_id(1);      // 0..1  (subgroup row in WG)
    const int sgx  = get_local_id(2);      // 0..1  (subgroup col in WG)

    const int wg_x = get_group_id(0);
    const int wg_y = get_group_id(1);

    // WG tile origin
    const int wg_row0 = wg_y * TILE_M;
    const int wg_col0 = wg_x * TILE_N;

    // Subgroup tile origin (16x16)
    const int sg_row0 = wg_row0 + sgy * 16;
    const int sg_col0 = wg_col0 + sgx * 16;

    // Each lane handles one base column in subgroup tile and also +16 in-N direction via sgx split.
    // Here subgroup tile width is 16, so lane maps directly to one col in that subtile:
    const int col = sg_col0 + lane;

    // Local tiles for current K-block (cooperatively loaded by 64 threads)
    __local half Aslm[TILE_M][TILE_K + 1];
    __local half Bslm[TILE_K][TILE_N + 1];

    // Two float8 accumulators for 16 rows of this column:
    // rows [0..7], [8..15] relative to sg_row0
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);

    const int k_full_end = (K / TILE_K) * TILE_K;

    for (int kb = 0; kb < k_full_end; kb += TILE_K) {
        // Cooperative load A tile: 32x16 = 512 half
        {
            const int l0 = (sgx * 2 + sgy) * 16 + lane; // unique 0..63
            #pragma unroll
            for (int t = 0; t < 8; ++t) { // 64*8 = 512
                const int idx = l0 + t * 64;   // 0..511
                const int ar  = idx / TILE_K;  // 0..31
                const int ak  = idx - ar * TILE_K;
                const int gr  = wg_row0 + ar;
                const int gk  = kb + ak;
                Aslm[ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
            }
        }

        // Cooperative load B tile: 16x32 = 512 half
        {
            const int l0 = (sgx * 2 + sgy) * 16 + lane; // unique 0..63
            #pragma unroll
            for (int t = 0; t < 8; ++t) { // 64*8 = 512
                const int idx = l0 + t * 64;   // 0..511
                const int bk  = idx / TILE_N;  // 0..15
                const int bn  = idx - bk * TILE_N;
                const int gk  = kb + bk;
                const int gc  = wg_col0 + bn;
                Bslm[bk][bn] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Build b_vec = B[k=kb:kb+15, col] for this lane/column in subgroup subtile
        half16 b_vec;
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            b_vec.s[kk] = Bslm[kk][sgx * 16 + lane];
        }

        // Rows 0..7 in subgroup tile
        {
            // DPAS subgroup contract expects lane-participation; map rows via lane index with clamping.
            const int rr = (lane < 8) ? lane : 0;
            half16 a_vec = vload16(0, &Aslm[sgy * 16 + rr][0]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc0);
        }

        // Rows 8..15 in subgroup tile
        {
            const int rr = (lane < 8) ? (lane + 8) : 8;
            half16 a_vec = vload16(0, &Aslm[sgy * 16 + rr][0]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc1);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store full-tile accumulation + exact tail-k
    if (col < N) {
        float tmp[8];

        // acc0 -> rows sg_row0 + [0..7]
        tmp[0]=acc0.s0; tmp[1]=acc0.s1; tmp[2]=acc0.s2; tmp[3]=acc0.s3;
        tmp[4]=acc0.s4; tmp[5]=acc0.s5; tmp[6]=acc0.s6; tmp[7]=acc0.s7;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gr = sg_row0 + r;
            if (gr < M) {
                float v = tmp[r];
                for (int k = k_full_end; k < K; ++k) {
                    v += convert_float(A[gr * K + k]) * convert_float(B[k * N + col]);
                }
                C[gr * N + col] = convert_half(v);
            }
        }

        // acc1 -> rows sg_row0 + [8..15]
        tmp[0]=acc1.s0; tmp[1]=acc1.s1; tmp[2]=acc1.s2; tmp[3]=acc1.s3;
        tmp[4]=acc1.s4; tmp[5]=acc1.s5; tmp[6]=acc1.s6; tmp[7]=acc1.s7;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int gr = sg_row0 + 8 + r;
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
1. `const int rr = (lane < 8) ? lane : 0; ... acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc0);` and `const int rr = (lane < 8) ? (lane + 8) : 8; ... acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc1);`:  
   You’re feeding DPAS with duplicated A rows for lanes 8..15 (`rr` clamps to 0/8), so half the subgroup does redundant work each K-step. This leaves throughput on the table and can also underutilize the matrix engine’s intended lane mapping.  
   **Improve it by mapping lanes 0..15 to unique rows for each DPAS call** (or restructuring to one 16-row accumulator instead of two 8-row chunks). Concretely, make each lane load its own `Aslm[sgy*16 + lane][0..15]` for one DPAS, then use a second DPAS for another output column (or another tile fragment) rather than splitting rows with clamped lanes. This removes redundant lane participation and increases useful math per subgroup instruction.

2. `half16 b_vec; #pragma unroll for (int kk = 0; kk < 16; ++kk) { b_vec.s[kk] = Bslm[kk][sgx * 16 + lane]; }`:  
   This scalar gather from SLM is on the hot path of every K-block and creates 16 per-lane local-memory reads plus element-wise packing overhead.  
   **Improve it by changing B’s SLM layout so the load is vectorizable/contiguous per lane.** For example, store B tile transposed in SLM during cooperative load (so each lane can `vload16` directly), or use subgroup block-read style loads if available. You pay a little during tile load once, then save instructions every DPAS step. On Xe2 this usually gives a noticeable gain because it reduces SLM read instruction count and register shuffling around `b_vec`.

3. `for (int k = k_full_end; k < K; ++k) { v += convert_float(A[gr * K + k]) * convert_float(B[k * N + col]); }` inside both row-store loops:  
   Tail-K is computed per output element during store, which repeats global loads and scalar FMAs for every row/col result. Even if tail is small, this is in the epilogue and can dominate for many small/odd-K shapes.  
   **Improve it by handling tail-K inside the tiled loop with shared-memory buffering once per WG** (same cooperative load pattern, but masked for `K-kb < 16`) and accumulate into `acc0/acc1` before store. Then stores become pure writeback with no extra GEMM math. This removes duplicated tail work and keeps arithmetic in the same optimized path as full tiles.

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
2. Avoid Bank Conflicts: Local memory is organized into banks (typically 32 banks). Pad shared arrays to avoid stride conflicts, e.g., __local float tile[TILE_SIZE][TILE_SIZE + 1] for transpose operations. Use sequential access patterns within wavefronts.

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