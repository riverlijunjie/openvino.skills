

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Performance optimization techniques to consider:
- Current kernel achieves 21% XMX utilization (1.07ms for 21.5 GFLOP on 96 TFLOPS peak).
    Target: <0.5ms (>45% utilization).
    Key bottleneck: LOW DPAS-to-barrier ratio. Current: 8 DPAS per 2 barriers.
    Optimization directions (in priority order):
        1. DOUBLE BUFFERING: Overlap SLM loads with DPAS compute (ping-pong buffers)
        2. LARGER TILE_M (64 or 128): More DPAS calls per barrier pair → amortize overhead
        3. INCREASE WG SIZE to 128-256 WIs: Better EU occupancy + faster cooperative loads  
        4. TILE_K=64: More DPAS per K-tile iteration (4 k16 steps, 16 DPAS per barrier pair)
        5. B in SLM: Avoid redundant L2 reads across subgroups
        6. Prefetch: async_work_group_copy or explicit load-ahead
    Hardware: B580 = 20 Xe2 cores, 96 TFLOPS FP16 XMX, ~456 GB/s, ~24 MB L2
    DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
- Must use Intel OpenCL DPAS instruction(XMX), e.g. intel_sub_group_f16_f16_matrix_mad_k16.
- Double-buffering: Overlap SLM loads with DPAS computation
- Prefetching: Use async_work_group_copy for next K-tile
- Register blocking: Maximize DPAS utilization per register load
- Loop unrolling: Fully unroll K-loop if TILE_K <= 64
- SLM bank conflict avoidance: Pad SLM arrays by +1 column
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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.310):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TILE_M=64, TILE_N=128, TILE_K=32
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 8 row-blocks of 8 rows x 16 cols = 64x16
// Per K-tile: 2 k16 steps x 8 row-blocks = 16 DPAS per subgroup
// True double buffering: load next A+B tile interleaved with DPAS compute on current tile
// SLM A: 2 x 64 x 34 halves, SLM B: 2 x 32 x 130 halves, total ~25KB
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)

#define TILE_M 64
#define TILE_N 128
#define TILE_K 32
#define SLM_A_STRIDE (TILE_K + 2)
#define SLM_B_STRIDE (TILE_N + 2)
#define NUM_WI 128

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid = get_local_id(0);

    const int n_base = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;

    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];
    __local half slm_B[2 * TILE_K * SLM_B_STRIDE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float8 acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const int col_base = n_base + sg_id * 16;
    const int col_idx = col_base + sg_lid;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const bool col_tile_valid = (n_base + TILE_N <= N);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Precompute cooperative load indices for A (16 elements per WI)
    // A tile: 64 x 32 = 2048 halves / 128 WIs = 16 each
    // B tile: 32 x 128 = 4096 halves / 128 WIs = 32 each

    // Load first tile (buffer 0)
    {
        __local half* a_dst = slm_A;
        __local half* b_dst = slm_B;

        if (row_tile_valid && TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * NUM_WI;
                int r = eid >> 5; // /32
                int c = eid & 31; // %32
                a_dst[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * NUM_WI;
                int r = eid >> 5;
                int c = eid & 31;
                int gr = row_base + r;
                a_dst[r * SLM_A_STRIDE + c] = (gr < M && c < K) ? A[gr * K + c] : (half)0.0h;
            }
        }

        if (col_tile_valid && TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int eid = lid + i * NUM_WI;
                int r = eid >> 7; // /128
                int c = eid & 127; // %128
                b_dst[r * SLM_B_STRIDE + c] = B[r * N + n_base + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int eid = lid + i * NUM_WI;
                int r = eid >> 7;
                int c = eid & 127;
                int gn = n_base + c;
                b_dst[r * SLM_B_STRIDE + c] = (r < K && gn < N) ? B[r * N + gn] : (half)0.0h;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int cur_buf = kt & 1;
        const int nxt_buf = 1 - cur_buf;
        const int k_cur = kt * TILE_K;
        const int k_next = k_cur + TILE_K;
        const bool has_next = (kt + 1 < num_k_tiles);

        __local const half* cur_A = slm_A + cur_buf * TILE_M * SLM_A_STRIDE;
        __local const half* cur_B = slm_B + cur_buf * TILE_K * SLM_B_STRIDE;

        // === k16 step 0: compute + start loading next tile ===
        {
            // Load B block for k16 step 0 from SLM
            int sg_col_off = sg_id * 16;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                short s0 = as_short(cur_B[(2 * p) * SLM_B_STRIDE + sg_col_off + sg_lid]);
                short s1 = as_short(cur_B[(2 * p + 1) * SLM_B_STRIDE + sg_col_off + sg_lid]);
                ((int*)&b_val)[p] = as_int((short2)(s0, s1));
            }

            // Load all 8 A blocks and DPAS
            short8 a_block;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[r * SLM_A_STRIDE + sg_lid]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc0);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(8 + r) * SLM_A_STRIDE + sg_lid]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc1);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(16 + r) * SLM_A_STRIDE + sg_lid]);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc2);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(24 + r) * SLM_A_STRIDE + sg_lid]);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc3);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(32 + r) * SLM_A_STRIDE + sg_lid]);
            acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc4);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(40 + r) * SLM_A_STRIDE + sg_lid]);
            acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc5);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(48 + r) * SLM_A_STRIDE + sg_lid]);
            acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc6);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(56 + r) * SLM_A_STRIDE + sg_lid]);
            acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc7);
        }

        // Interleave: start loading next A tile into nxt_buf while we still have k16 step 1 to compute
        // Since nxt_buf != cur_buf, no SLM read/write conflict
        if (has_next) {
            __local half* nxt_A = slm_A + nxt_buf * TILE_M * SLM_A_STRIDE;
            if (row_tile_valid && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int eid = lid + i * NUM_WI;
                    int r = eid >> 5;
                    int c = eid & 31;
                    nxt_A[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + k_next + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int eid = lid + i * NUM_WI;
                    int r = eid >> 5;
                    int c = eid & 31;
                    int gr = row_base + r;
                    int gk = k_next + c;
                    nxt_A[r * SLM_A_STRIDE + c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
                }
            }
        }

        // === k16 step 1: compute on current buffer ===
        {
            int sg_col_off = sg_id * 16;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                short s0 = as_short(cur_B[(16 + 2 * p) * SLM_B_STRIDE + sg_col_off + sg_lid]);
                short s1 = as_short(cur_B[(16 + 2 * p + 1) * SLM_B_STRIDE + sg_col_off + sg_lid]);
                ((int*)&b_val)[p] = as_int((short2)(s0, s1));
            }

            short8 a_block;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[r * SLM_A_STRIDE + 16 + sg_lid]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc0);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(8 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc1);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(16 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc2);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(24 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc3);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(32 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc4);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(40 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc5);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(48 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc6);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(56 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc7);
        }

        // Load next B tile (interleaved after k16 step 1 DPAS)
        if (has_next) {
            __local half* nxt_B = slm_B + nxt_buf * TILE_K * SLM_B_STRIDE;
            if (col_tile_valid && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int eid = lid + i * NUM_WI;
                    int r = eid >> 7;
                    int c = eid & 127;
                    nxt_B[r * SLM_B_STRIDE + c] = B[(k_next + r) * N + n_base + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int eid = lid + i * NUM_WI;
                    int r = eid >> 7;
                    int c = eid & 127;
                    int gk = k_next + r;
                    int gn = n_base + c;
                    nxt_B[r * SLM_B_STRIDE + c] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Store results
    const bool col_valid = col_idx < N;
    if (col_valid) {
        #define STORE_BLK(off, acc_v) \
        { \
            int ro = row_base + (off); \
            if (ro + 8 <= M) { \
                _Pragma("unroll") \
                for (int r = 0; r < 8; r++) \
                    C[(ro + r) * N + col_idx] = convert_half(((float*)&acc_v)[r]); \
            } else { \
                _Pragma("unroll") \
                for (int r = 0; r < 8; r++) \
                    if (ro + r < M) \
                        C[(ro + r) * N + col_idx] = convert_half(((float*)&acc_v)[r]); \
            } \
        }

        STORE_BLK(0, acc0)
        STORE_BLK(8, acc1)
        STORE_BLK(16, acc2)
        STORE_BLK(24, acc3)
        STORE_BLK(32, acc4)
        STORE_BLK(40, acc5)
        STORE_BLK(48, acc6)
        STORE_BLK(56, acc7)

        #undef STORE_BLK
    }
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.320):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TILE_M=64, TILE_N=128, TILE_K=32
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 8 row-blocks of 8 rows x 16 cols = 64x16 output tile
// Per K-tile: 2 k16 steps x 8 row-blocks = 16 DPAS per subgroup
// Double-buffered SLM: A: 2x64x34=8704B, B: 2x32x130=16640B, total ~25KB
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)

#define TILE_M 64
#define TILE_N 128
#define TILE_K 32
#define SLM_A_STRIDE 34   // TILE_K + 2 for bank conflict avoidance
#define SLM_B_STRIDE 130  // TILE_N + 2 for bank conflict avoidance
#define NUM_WI 128

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid = get_local_id(0);

    const int n_base = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;

    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];
    __local half slm_B[2 * TILE_K * SLM_B_STRIDE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float8 acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const int col_idx = n_base + sg_id * 16 + sg_lid;
    const int sg_col_off = sg_id * 16;
    const bool row_tile_full = (row_base + TILE_M <= M);
    const bool col_tile_full = (n_base + TILE_N <= N);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // === Load first tile into buffer 0 ===
    {
        __local half* a_dst = slm_A;
        __local half* b_dst = slm_B;

        // A: 64x32 = 2048 halves / 128 WIs = 16 per WI
        if (row_tile_full && TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * NUM_WI;
                int r = eid >> 5;
                int c = eid & 31;
                a_dst[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * NUM_WI;
                int r = eid >> 5;
                int c = eid & 31;
                int gr = row_base + r;
                a_dst[r * SLM_A_STRIDE + c] = (gr < M && c < K) ? A[gr * K + c] : (half)0.0h;
            }
        }

        // B: 32x128 = 4096 halves / 128 WIs = 32 per WI
        if (col_tile_full && TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int eid = lid + i * NUM_WI;
                int r = eid >> 7;
                int c = eid & 127;
                b_dst[r * SLM_B_STRIDE + c] = B[r * N + n_base + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int eid = lid + i * NUM_WI;
                int r = eid >> 7;
                int c = eid & 127;
                int gn = n_base + c;
                b_dst[r * SLM_B_STRIDE + c] = (r < K && gn < N) ? B[r * N + gn] : (half)0.0h;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int cur_buf = kt & 1;
        const int nxt_buf = 1 - cur_buf;
        const int k_cur = kt * TILE_K;
        const int k_next = k_cur + TILE_K;
        const bool has_next = (kt + 1 < num_k_tiles);

        __local const half* cur_A = slm_A + cur_buf * TILE_M * SLM_A_STRIDE;
        __local const half* cur_B = slm_B + cur_buf * TILE_K * SLM_B_STRIDE;

        // Pre-load both B blocks into registers before DPAS to hide SLM latency
        int8 b_val0, b_val1;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            short s0 = as_short(cur_B[(2 * p) * SLM_B_STRIDE + sg_col_off + sg_lid]);
            short s1 = as_short(cur_B[(2 * p + 1) * SLM_B_STRIDE + sg_col_off + sg_lid]);
            ((int*)&b_val0)[p] = as_int((short2)(s0, s1));
        }
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            short s0 = as_short(cur_B[(16 + 2 * p) * SLM_B_STRIDE + sg_col_off + sg_lid]);
            short s1 = as_short(cur_B[(16 + 2 * p + 1) * SLM_B_STRIDE + sg_col_off + sg_lid]);
            ((int*)&b_val1)[p] = as_int((short2)(s0, s1));
        }

        // === k16 step 0: 8 DPAS with b_val0 ===
        {
            short8 a_blk;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[r * SLM_A_STRIDE + sg_lid]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val0, acc0);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(8 + r) * SLM_A_STRIDE + sg_lid]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val0, acc1);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(16 + r) * SLM_A_STRIDE + sg_lid]);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val0, acc2);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(24 + r) * SLM_A_STRIDE + sg_lid]);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val0, acc3);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(32 + r) * SLM_A_STRIDE + sg_lid]);
            acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val0, acc4);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(40 + r) * SLM_A_STRIDE + sg_lid]);
            acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val0, acc5);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(48 + r) * SLM_A_STRIDE + sg_lid]);
            acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val0, acc6);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(56 + r) * SLM_A_STRIDE + sg_lid]);
            acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val0, acc7);
        }

        // === k16 step 1: 8 DPAS with b_val1 ===
        {
            short8 a_blk;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[r * SLM_A_STRIDE + 16 + sg_lid]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val1, acc0);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(8 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val1, acc1);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(16 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val1, acc2);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(24 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val1, acc3);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(32 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val1, acc4);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(40 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val1, acc5);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(48 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val1, acc6);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(56 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val1, acc7);
        }

        // === Load next tile into nxt_buf (after all DPAS on cur_buf) ===
        if (has_next) {
            __local half* nxt_A = slm_A + nxt_buf * TILE_M * SLM_A_STRIDE;
            __local half* nxt_B = slm_B + nxt_buf * TILE_K * SLM_B_STRIDE;

            if (row_tile_full && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int eid = lid + i * NUM_WI;
                    int r = eid >> 5;
                    int c = eid & 31;
                    nxt_A[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + k_next + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int eid = lid + i * NUM_WI;
                    int r = eid >> 5;
                    int c = eid & 31;
                    int gr = row_base + r;
                    int gk = k_next + c;
                    nxt_A[r * SLM_A_STRIDE + c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
                }
            }

            if (col_tile_full && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int eid = lid + i * NUM_WI;
                    int r = eid >> 7;
                    int c = eid & 127;
                    nxt_B[r * SLM_B_STRIDE + c] = B[(k_next + r) * N + n_base + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int eid = lid + i * NUM_WI;
                    int r = eid >> 7;
                    int c = eid & 127;
                    int gk = k_next + r;
                    int gn = n_base + c;
                    nxt_B[r * SLM_B_STRIDE + c] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Store results
    if (col_idx < N) {
        #define STORE_BLK(off, acc_v) \
        { \
            int ro = row_base + (off); \
            if (ro + 8 <= M) { \
                _Pragma("unroll") \
                for (int r = 0; r < 8; r++) \
                    C[(ro + r) * N + col_idx] = convert_half(((float*)&acc_v)[r]); \
            } else { \
                _Pragma("unroll") \
                for (int r = 0; r < 8; r++) \
                    if (ro + r < M) \
                        C[(ro + r) * N + col_idx] = convert_half(((float*)&acc_v)[r]); \
            } \
        }

        STORE_BLK(0, acc0)
        STORE_BLK(8, acc1)
        STORE_BLK(16, acc2)
        STORE_BLK(24, acc3)
        STORE_BLK(32, acc4)
        STORE_BLK(40, acc5)
        STORE_BLK(48, acc6)
        STORE_BLK(56, acc7)

        #undef STORE_BLK
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 2.740):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// TILE_M=64, TILE_N=128, TILE_K=64
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 8 row-blocks of 8x16 = 64x16 output
// Per K-tile: 4 k16 steps x 8 row-blocks = 32 DPAS per subgroup per barrier
// Only A in SLM (shared across 8 subgroups), B read from L2
// SLM A: 2 x 64 x 66 halves = ~17KB
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)

#define TILE_M 64
#define TILE_N 128
#define TILE_K 64
#define SLM_A_STRIDE (TILE_K + 2)
#define NUM_WI 128

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid = get_local_id(0);

    const int n_base = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;

    // Only A in SLM - double buffered
    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float8 acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const int col_base = n_base + sg_id * 16;
    const int col_idx = col_base + sg_lid;
    const bool row_tile_full = (row_base + TILE_M <= M);
    const bool col_valid = (col_idx < N);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // A tile: 64 x 64 = 4096 halves, 128 WIs => 32 each
    #define LOAD_A(dst, k_off, do_check)                                     \
    {                                                                         \
        _Pragma("unroll")                                                     \
        for (int _i = 0; _i < 32; _i++) {                                    \
            int _eid = lid + _i * NUM_WI;                                     \
            int _r = _eid >> 6;                                               \
            int _c = _eid & 63;                                               \
            if (do_check) {                                                   \
                int _gr = row_base + _r;                                      \
                int _gk = (k_off) + _c;                                       \
                (dst)[_r * SLM_A_STRIDE + _c] = (_gr < M && _gk < K) ? A[_gr * K + _gk] : (half)0.0h; \
            } else {                                                          \
                (dst)[_r * SLM_A_STRIDE + _c] = A[(row_base + _r) * K + (k_off) + _c]; \
            }                                                                 \
        }                                                                     \
    }

    // Load first A tile into buffer 0
    {
        bool safe = row_tile_full && (TILE_K <= K);
        if (safe) {
            LOAD_A(slm_A, 0, 0)
        } else {
            LOAD_A(slm_A, 0, 1)
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int cur_buf = kt & 1;
        const int nxt_buf = 1 - cur_buf;
        const int k_cur = kt * TILE_K;
        const int k_next = k_cur + TILE_K;
        const bool has_next = (kt + 1 < num_k_tiles);

        __local const half* cur_A = slm_A + cur_buf * TILE_M * SLM_A_STRIDE;

        // Load next A tile while computing (double buffer)
        if (has_next) {
            __local half* nxt_A = slm_A + nxt_buf * TILE_M * SLM_A_STRIDE;
            bool safe = row_tile_full && (k_next + TILE_K <= K);
            if (safe) {
                LOAD_A(nxt_A, k_next, 0)
            } else {
                LOAD_A(nxt_A, k_next, 1)
            }
        }

        // Compute: 4 k16 steps x 8 row-blocks = 32 DPAS
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            int gk = k_cur + kk;

            // Load B from global memory - each subgroup reads its own 16 columns
            // B layout: B[k][n], we need B[gk..gk+15][col_base..col_base+15]
            int8 b_val;
            if (col_valid && gk + 16 <= K) {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int kr0 = gk + 2 * p;
                    int kr1 = kr0 + 1;
                    short s0 = as_short(B[kr0 * N + col_base + sg_lid]);
                    short s1 = as_short(B[kr1 * N + col_base + sg_lid]);
                    ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                }
            } else {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int kr0 = gk + 2 * p;
                    int kr1 = kr0 + 1;
                    short s0 = (kr0 < K && col_idx < N) ? as_short(B[kr0 * N + col_base + sg_lid]) : (short)0;
                    short s1 = (kr1 < K && col_idx < N) ? as_short(B[kr1 * N + col_base + sg_lid]) : (short)0;
                    ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                }
            }

            // Load A blocks from SLM and DPAS
            short8 a_blk;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[r * SLM_A_STRIDE + kk + sg_lid]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val, acc0);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val, acc1);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val, acc2);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val, acc3);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(32 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val, acc4);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(40 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val, acc5);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(48 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val, acc6);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_blk)[r] = as_short(cur_A[(56 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a_blk, b_val, acc7);
        }

        if (has_next) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Store results
    if (col_valid) {
        #define STORE_BLK(off, acc_v)                                        \
        {                                                                     \
            int _ro = row_base + (off);                                       \
            if (_ro + 8 <= M) {                                               \
                _Pragma("unroll")                                             \
                for (int _r = 0; _r < 8; _r++)                               \
                    C[(_ro + _r) * N + col_idx] = convert_half(((float*)&(acc_v))[_r]); \
            } else {                                                          \
                _Pragma("unroll")                                             \
                for (int _r = 0; _r < 8; _r++)                               \
                    if (_ro + _r < M)                                         \
                        C[(_ro + _r) * N + col_idx] = convert_half(((float*)&(acc_v))[_r]); \
            }                                                                 \
        }

        STORE_BLK(0, acc0)
        STORE_BLK(8, acc1)
        STORE_BLK(16, acc2)
        STORE_BLK(24, acc3)
        STORE_BLK(32, acc4)
        STORE_BLK(40, acc5)
        STORE_BLK(48, acc6)
        STORE_BLK(56, acc7)

        #undef STORE_BLK
    }

    #undef LOAD_A
}
```

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] PASSED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] PASSED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] PASSED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] PASSED         [100%]

=============================== warnings summary ===============================
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0]
  /home/openvino-ci-74/miniforge3/envs/kernel_intel/lib/python3.12/site-packages/pyopencl/cache.py:517: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
    _create_built_program_from_source_cached(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 4 passed, 1 deselected, 1 warning in 0.77s ==================
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
1. Exploit Sub-groups (OpenCL 2.0+): Use sub-group functions for wavefront-level operations: sub_group_reduce_add(), sub_group_broadcast(), intel_sub_group_shuffle(), intel_sub_group_shuffle_down(). Query sub-group size with get_sub_group_size(). Use __attribute__((intel_reqd_sub_group_size(N))) for Intel GPUs.
2. Maximize Occupancy: Balance work-group size with register/local memory usage. Query CL_KERNEL_WORK_GROUP_SIZE and CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE. Typical work-group sizes: 64-256 for compute-bound, 32-128 for memory-bound kernels. Use multiples of wavefront/warp size (32 for NVIDIA, 64 for AMD).

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