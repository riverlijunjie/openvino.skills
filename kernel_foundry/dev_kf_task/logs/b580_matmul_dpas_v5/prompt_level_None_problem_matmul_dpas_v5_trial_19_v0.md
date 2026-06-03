

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
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Launch metadata:
//   LWS = {16, 4, 1}  (64 WIs = 4 subgroups of 16)
//   GWS = {ceil_div(N,32)*16, ceil_div(M,32)*4, 1}
//   Each WG computes a 32x32 C tile
//   Each subgroup computes 8x32 (two DPAS calls per K-chunk: left 16 cols + right 16 cols)
//   All 8 float8 components are used per DPAS call

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define WG_X 16
#define WG_Y 4

__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lane = get_local_id(0);   // 0..15 (subgroup lane)
    const int sg_id = get_local_id(1);  // 0..3 (which subgroup)
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);

    const int tile_row = gy * TILE_M;
    const int tile_col = gx * TILE_N;

    // Each subgroup sg_id handles rows [sg_id*8 .. sg_id*8+7] of the tile
    // and all 32 columns (via two DPAS calls: cols 0-15 and cols 16-31)
    const int sg_row_base = sg_id * 8;  // 0, 8, 16, 24

    // Accumulators: float8 for left 16 cols and right 16 cols
    float8 acc_left = (float8)(0.0f);   // rows [sg_row_base..+7], cols [0..15]
    float8 acc_right = (float8)(0.0f);  // rows [sg_row_base..+7], cols [16..31]

    // Double-buffered SLM
    __local half Aslm[2][TILE_M][TILE_K];
    __local half Bslm[2][TILE_K][TILE_N];

    const int k_full_end = (K / TILE_K) * TILE_K;
    const int linear = sg_id * WG_X + lane; // 0..63
    int buf = 0;

    // Preload first tile
    if (0 < k_full_end) {
        // Load A: 32x16 = 512 elems, 64 WIs => 8 each
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int idx = linear + t * 64;
            int ar = idx / TILE_K;
            int ak = idx - ar * TILE_K;
            int gr = tile_row + ar;
            int gk = ak;
            Aslm[0][ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }
        // Load B: 16x32 = 512 elems, 64 WIs => 8 each
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int idx = linear + t * 64;
            int bk = idx / TILE_N;
            int bn = idx - bk * TILE_N;
            int gk = bk;
            int gn = tile_col + bn;
            Bslm[0][bk][bn] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kb = 0; kb < k_full_end; kb += TILE_K) {
        int next_kb = kb + TILE_K;
        int next_buf = 1 - buf;

        // Prefetch next tile into alternate buffer
        if (next_kb < k_full_end) {
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                int idx = linear + t * 64;
                int ar = idx / TILE_K;
                int ak = idx - ar * TILE_K;
                int gr = tile_row + ar;
                int gk = next_kb + ak;
                Aslm[next_buf][ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
            }
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                int idx = linear + t * 64;
                int bk = idx / TILE_N;
                int bn = idx - bk * TILE_N;
                int gk = next_kb + bk;
                int gn = tile_col + bn;
                Bslm[next_buf][bk][bn] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }
        }

        // Compute on current buffer
        // A fragment: rows [sg_row_base..sg_row_base+7], K=16
        // For DPAS: lane i provides A[sg_row_base + i%8][k] packed as half16
        // But we have 16 lanes and only 8 rows - lanes 0-7 map to rows, lanes 8-15 replicate
        // The hardware uses lanes 0-7 for the A operand
        half16 a_vec = vload16(0, &Aslm[buf][sg_row_base + (lane & 7)][0]);

        // B fragment for left cols (0-15): lane i provides B[k][tile_col + i]
        half16 b_left;
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            ((half*)&b_left)[kk] = Bslm[buf][kk][lane];
        }

        // B fragment for right cols (16-31): 
        half16 b_right;
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            ((half*)&b_right)[kk] = Bslm[buf][kk][16 + lane];
        }

        // DPAS: 8 rows x 16 cols, all 8 outputs used!
        acc_left = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_left, acc_left);
        acc_right = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_right, acc_right);

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Store results
    // acc_left.sN = C[tile_row + sg_row_base + N][tile_col + lane], N=0..7
    // acc_right.sN = C[tile_row + sg_row_base + N][tile_col + 16 + lane]

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int gr = tile_row + sg_row_base + r;
        int gc_left = tile_col + lane;
        int gc_right = tile_col + 16 + lane;

        float val_left, val_right;
        switch(r) {
            case 0: val_left = acc_left.s0; val_right = acc_right.s0; break;
            case 1: val_left = acc_left.s1; val_right = acc_right.s1; break;
            case 2: val_left = acc_left.s2; val_right = acc_right.s2; break;
            case 3: val_left = acc_left.s3; val_right = acc_right.s3; break;
            case 4: val_left = acc_left.s4; val_right = acc_right.s4; break;
            case 5: val_left = acc_left.s5; val_right = acc_right.s5; break;
            case 6: val_left = acc_left.s6; val_right = acc_right.s6; break;
            case 7: val_left = acc_left.s7; val_right = acc_right.s7; break;
        }

        if (gr < M) {
            // Tail K cleanup
            for (int k = k_full_end; k < K; ++k) {
                half a_val = (gr < M) ? A[gr * K + k] : (half)0.0h;
                if (gc_left < N)
                    val_left += convert_float(a_val) * convert_float(B[k * N + gc_left]);
                if (gc_right < N)
                    val_right += convert_float(a_val) * convert_float(B[k * N + gc_right]);
            }

            if (gc_left < N)
                C[gr * N + gc_left] = convert_half(val_left);
            if (gc_right < N)
                C[gr * N + gc_right] = convert_half(val_right);
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_matrix_multiply_accumulate : enable

// -----------------------------------------------------------------------------
// Intel Xe2 (Battlemage) DPAS-oriented FP16 GEMM
//   C[M,N] = A[M,K] x B[K,N]
//   input/output: half, accumulation: float
//
// Suggested launch metadata (host side):
//   LWS = {16, 1, 1}                         // exactly 1 subgroup / WG
//   GWS = { round_up(N,16), ceil_div(M,16), 1 }
//   subgroup size hint = 16
//
// Mapping:
//   - One WG (one subgroup) computes a 16x16 C tile:
//       rows: tile_row ... tile_row+15
//       cols: tile_col ... tile_col+15
//   - lane id selects column within the 16-column tile
//   - each lane accumulates 16 rows for its column (float8 lo + float8 hi)
//   - K processed in chunks of 16 using DPAS k16 intrinsic
// -----------------------------------------------------------------------------

#define TM 16
#define TN 16
#define TK 16

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
    const int lane = get_local_id(0);   // 0..15
    const int gx   = get_group_id(0);
    const int gy   = get_group_id(1);

    const int tile_row = gy * TM;
    const int tile_col = gx * TN;
    const int col      = tile_col + lane;

    // SLM tiles (+1 padding to reduce bank conflicts)
    __local half Asub[TM][TK + 1];
    __local half Bsub[TK][TN + 1];

    float8 acc_lo = (float8)(0.0f); // rows 0..7  for this lane's column
    float8 acc_hi = (float8)(0.0f); // rows 8..15 for this lane's column

    const int k_full_end = (K / TK) * TK;

    for (int kb = 0; kb < k_full_end; kb += TK) {
        // Cooperative load A tile: each lane loads one K-column for all 16 rows
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
            const int gr = tile_row + r;
            const int gk = kb + lane;
            Asub[r][lane] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // Cooperative load B tile: each lane loads one N-column for all 16 K rows
        #pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
            const int gk = kb + kk;
            const int gc = col;
            Bsub[kk][lane] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Build B vector for this lane's output column: B[kb:kb+15, col]
        half16 b_vec;
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            b_vec.s[kk] = Bsub[kk][lane];
        }

        // Rows 0..7
        {
            // Subgroup-consistent A fragment selection
            const int src_row = (lane < 8) ? lane : 0;
            half16 a_vec = vload16(0, &Asub[src_row][0]);
            acc_lo = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc_lo);
        }

        // Rows 8..15
        {
            const int src_row = (lane < 8) ? (lane + 8) : 8;
            half16 a_vec = vload16(0, &Asub[src_row][0]);
            acc_hi = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b_vec, acc_hi);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store with exact tail-K accumulation
    if (col < N) {
        float tmp[8];

        // lo rows 0..7
        tmp[0] = acc_lo.s0; tmp[1] = acc_lo.s1; tmp[2] = acc_lo.s2; tmp[3] = acc_lo.s3;
        tmp[4] = acc_lo.s4; tmp[5] = acc_lo.s5; tmp[6] = acc_lo.s6; tmp[7] = acc_lo.s7;

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

        // hi rows 8..15
        tmp[0] = acc_hi.s0; tmp[1] = acc_hi.s1; tmp[2] = acc_hi.s2; tmp[3] = acc_hi.s3;
        tmp[4] = acc_hi.s4; tmp[5] = acc_hi.s5; tmp[6] = acc_hi.s6; tmp[7] = acc_hi.s7;

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
1. `for (int r = 0; r < TM; ++r) { ... Asub[r][lane] = ... }` and `for (int kk = 0; kk < TK; ++kk) { ... Bsub[kk][lane] = ... }`: you’re doing scalarized local-memory staging for both A and B every K-block, then reading it back with barriers. This adds a lot of LDS traffic + synchronization overhead relative to compute.  
   **Improve it:** bypass `Asub`/`Bsub` for the DPAS path and build `a_vec`/`b_vec` directly from global (or at least use vectorized global loads like `vload8/vload16` into registers, then one coalesced local store if needed). At minimum, vectorize these loads/stores and unroll so each lane moves wider chunks; scalar per-element copies are a major runtime sink here.

2. `const int src_row = (lane < 8) ? lane : 0; ...` and `const int src_row = (lane < 8) ? (lane + 8) : 8; ...`: half the subgroup (`lane >= 8`) is forced to use a fixed row, so DPAS input mapping is imbalanced and you’re not feeding the matrix engine with a natural full-tile fragment layout. This can underutilize EU datapaths and creates redundant A-fragment fetch patterns.  
   **Improve it:** remap work so all 16 lanes contribute meaningful A fragments each DPAS call (e.g., lane-grouped row blocks with 2 DPAS ops per lane-group), or restructure to compute fewer rows per lane but with fully active subgroup fragments. The goal is to avoid “dummy/repeated row source” lanes and make fragment selection subgroup-uniform and dense.

3. `for (int k = k_full_end; k < K; ++k) { v += ... }` inside each row store loop: tail-K is recomputed separately for every output element (`16 rows × each lane/column`), causing repeated scalar global reads and poor cache reuse. This becomes expensive when `K % 16 != 0`.  
   **Improve it:** handle tail K as one extra padded K-block in the main loop (load out-of-range elements as zero, then run the same DPAS path), so you remove the per-element scalar tail loop entirely. This keeps execution uniform, reduces branchy epilogue work, and usually improves runtime significantly for non-multiple-of-16 K.

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
2. Minimize Global Memory Access: Cache frequently accessed values in private variables (registers). Use __constant address space for read-only data to enable hardware caching. Use __global const restrict for read-only buffers to hint compiler optimizations.

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