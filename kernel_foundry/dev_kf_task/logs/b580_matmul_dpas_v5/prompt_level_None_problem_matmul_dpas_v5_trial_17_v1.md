

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
// Optimized FP16 GEMM: C[M,N] = A[M,K] × B[K,N]
// Input/Output: half, Accumulation: float
//
// Launch metadata (host-side):
//   LWS = {16, 16, 1}   // 256 threads per work-group
//   GWS = { round_up(N,16), round_up(M,16), 1 }
//   Subgroup size = 16 (required)
//
// Key optimizations:
//   1. Interior/boundary tile splitting (no divergence in hot path)
//   2. Transposed B tile in SLM for vectorized loads
//   3. Double-buffered SLM with prefetch (overlap memory/compute)
//   4. BK=64 for better memory-compute ratio (4× DPAS k16 per tile)
//
// Performance notes:
//   - Targets Intel Xe2-HPG with 160 XMX engines
//   - Designed to saturate XMX throughput and hide 456 GB/s memory latency

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

#define TM 16
#define TN 16
#define BK 64

// Double-buffered SLM with padding to reduce bank conflicts
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
    const int wg_x = get_group_id(0);
    const int wg_y = get_group_id(1);

    const int row = wg_y * TM + ly;
    const int col = wg_x * TN + lx;

    const int tid = ly * 16 + lx;  // flat thread ID: 0..255

    // Double-buffered SLM (ping-pong)
    // BsubT stored transposed: [TN][BK] for vectorized column access
    __local half Asub[2][TM][BK + 2];      // +2 padding
    __local half BsubT[2][TN][BK + 2];     // transposed, +2 padding

    float acc = 0.0f;
    const int k_tiles = (K + BK - 1) / BK;

    // Determine if this is an interior tile (no boundary checks needed)
    const bool is_interior = (wg_y * TM + TM <= M) && (wg_x * TN + TN <= N);

    // Prefetch first tile into buffer 0
    int buf = 0;
    int kb = 0;

    if (kb < K) {
        // Load A tile: 16×64 = 1024 elems, 256 threads => 4 elems/thread
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int idx = tid + it * 256;
            const int ar = idx / BK;
            const int ak = idx % BK;
            const int gr = wg_y * TM + ar;
            const int gk = kb + ak;

            if (is_interior || (gr < M && gk < K)) {
                Asub[buf][ar][ak] = A[gr * K + gk];
            } else {
                Asub[buf][ar][ak] = (half)0.0h;
            }
        }

        // Load B tile TRANSPOSED: store as [TN][BK]
        // Original B[kb+br][wg_x*TN+bc] → BsubT[bc][br]
        #pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int idx = tid + it * 256;
            const int br = idx / TN;  // 0..63
            const int bc = idx % TN;  // 0..15
            const int gk = kb + br;
            const int gc = wg_x * TN + bc;

            if (is_interior || (gk < K && gc < N)) {
                BsubT[buf][bc][br] = B[gk * N + gc];
            } else {
                BsubT[buf][bc][br] = (half)0.0h;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Main loop with double buffering
    for (int tile = 0; tile < k_tiles; ++tile) {
        kb = tile * BK;
        const int next_kb = kb + BK;
        const int next_buf = 1 - buf;

        // Prefetch next tile while computing current
        if (tile + 1 < k_tiles && next_kb < K) {
            // Load A tile for next iteration
            #pragma unroll
            for (int it = 0; it < 4; ++it) {
                const int idx = tid + it * 256;
                const int ar = idx / BK;
                const int ak = idx % BK;
                const int gr = wg_y * TM + ar;
                const int gk = next_kb + ak;

                if (is_interior || (gr < M && gk < K)) {
                    Asub[next_buf][ar][ak] = A[gr * K + gk];
                } else {
                    Asub[next_buf][ar][ak] = (half)0.0h;
                }
            }

            // Load B tile TRANSPOSED for next iteration
            #pragma unroll
            for (int it = 0; it < 4; ++it) {
                const int idx = tid + it * 256;
                const int br = idx / TN;
                const int bc = idx % TN;
                const int gk = next_kb + br;
                const int gc = wg_x * TN + bc;

                if (is_interior || (gk < K && gc < N)) {
                    BsubT[next_buf][bc][br] = B[gk * N + gc];
                } else {
                    BsubT[next_buf][bc][br] = (half)0.0h;
                }
            }
        }

        // Compute on current buffer - 4× DPAS k16 operations
        // Each covers 16 K-elements: [0:15], [16:31], [32:47], [48:63]

        if (is_interior || (row < M && col < N)) {
            #pragma unroll
            for (int kstep = 0; kstep < 4; ++kstep) {
                const int k_offset = kstep * 16;

                // Vectorized load from SLM for A (contiguous in K dimension)
                half16 a_vec = vload16(0, &Asub[buf][ly][k_offset]);

                // Vectorized load from TRANSPOSED B (now contiguous!)
                half16 b_vec = vload16(0, &BsubT[buf][lx][k_offset]);

                // Intel DPAS: 8×8 systolic with k=16
                // Returns float8 but we only need element 0 for our single output
                float8 dpas_result = intel_sub_group_f16_f16_matrix_mad_k16(
                    a_vec, b_vec, (float8)(acc)
                );
                acc = dpas_result.s0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Tail K cleanup (if K not multiple of BK)
    const int k_full = (K / BK) * BK;
    if ((is_interior || (row < M && col < N)) && k_full < K) {
        #pragma unroll 1  // Don't unroll tail (usually small)
        for (int k = k_full; k < K; ++k) {
            acc += convert_float(A[row * K + k]) * convert_float(B[k * N + col]);
        }
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = convert_half(acc);
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.  
(II) Bottlenecks and suggestions for improvement:  
1. `float acc = 0.0f;` + `acc = dpas_result.s0;` inside  
   `float8 dpas_result = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, (float8)(acc));`  
   You’re launching a subgroup MMA op that naturally computes a vector tile (`float8`), but you keep only one lane (`s0`) per iteration. That throws away most compute and forces scalar accumulation semantics on a matrix engine path. Keep a vector accumulator (`float8 acc8 = 0`) across the K loop and only extract the needed component at the end (or better, map each subgroup to produce multiple C values so all `acc8` components are used). This is the biggest throughput loss in your current hot loop.

2. `if (is_interior || (row < M && col < N)) { ... }` inside the main compute loop  
   plus per-load checks like  
   `if (is_interior || (gr < M && gk < K))` and `if (is_interior || (gk < K && gc < N))`  
   Even with `is_interior`, these conditionals still sit in the inner path and cost issue slots / predicate handling every tile. Split into two kernels (or two code paths selected once): a pure interior kernel with **no bounds checks anywhere in the K loop/prefetch**, and a boundary kernel for edges. You already computed `is_interior`; now fully specialize the interior path so the compiler can aggressively optimize and schedule DPAS + loads without control-flow baggage.

3. `__local half Asub[2][TM][BK + 2];` and `__local half BsubT[2][TN][BK + 2];` with `BK=64`  
   plus full-tile staging loops (`for (int it = 0; it < 4; ++it)`) for both A and B  
   You’re spending a lot of SLM bandwidth and barriers to move a full 16×64 tile for both operands each iteration, but each thread computes only one output (`acc`). That arithmetic intensity per thread is low relative to LDS traffic/synchronization. Increase per-thread (or per-subgroup) output work, e.g., compute a small register block (such as 2–4 columns per thread or subgroup-fragment tile) before writing back. That amortizes SLM loads/barriers and improves XMX utilization. If you keep one-output-per-thread, reduce staging pressure (smaller BK like 32 can win when synchronization/SLM is limiting) and retune unroll to match.

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
1. Coalesced Access: Ensure work-items in the same wavefront/warp access contiguous memory addresses. For row-major data, have adjacent work-items (consecutive get_global_id(0)) access adjacent memory locations. Use linear indexing that matches your data layout.
2. Leverage Vectorization: Use vector types (float4, float8, float16) for explicit SIMD. Use vloadN() and vstoreN() for unaligned vector memory access. For aligned access, cast pointers: ((__global float4*)ptr)[idx]. Match native vector width of target device.

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
- **Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing.