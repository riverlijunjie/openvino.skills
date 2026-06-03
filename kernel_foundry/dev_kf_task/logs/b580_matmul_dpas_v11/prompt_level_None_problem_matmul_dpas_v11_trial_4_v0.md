

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 2.160):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TILE_M=64, TILE_N=128, TILE_K=32
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 8 row-blocks of 8 rows x 16 cols = 64x16
// 2 k16 steps x 8 row-blocks = 16 DPAS per k-step, 32 DPAS per K-tile
// A in double-buffered SLM, B read from global (L2 cached)
// SLM: 2 x 64 x 34 x 2 bytes = ~8.7 KB — good occupancy
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)

#define TILE_M 64
#define TILE_N 128
#define TILE_K 32
#define SLM_A_STRIDE (TILE_K + 2)
#define NUM_WI 128
#define NUM_SG 8

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
    const int sg_id = get_sub_group_id();        // 0..7
    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int lid = get_local_id(0);             // 0..127

    const int n_base = get_group_id(0) * TILE_N;
    const int col_base = n_base + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    // Double-buffered A in SLM: 2 x [64 x 34] halves = ~8.7 KB
    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];

    if (row_base >= M || n_base >= N)
        return;

    // 8 accumulators: 8 row-blocks of 8 rows each = 64 rows
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;
    float8 acc4 = 0.0f;
    float8 acc5 = 0.0f;
    float8 acc6 = 0.0f;
    float8 acc7 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // A tile: 64 x 32 = 2048 halves, 128 WIs => 16 each
    // Load first A tile into buffer 0
    {
        __local half* dst = slm_A;
        if (row_tile_valid && TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_K;
                int c = elem_id % TILE_K;
                dst[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_K;
                int c = elem_id % TILE_K;
                int gr = row_base + r;
                dst[r * SLM_A_STRIDE + c] = (gr < M && c < K) ? A[gr * K + c] : (half)0.0h;
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

        __local half* cur_A = slm_A + cur_buf * TILE_M * SLM_A_STRIDE;

        // ---- Compute: 2 k16 steps x 8 row-blocks = 16 DPAS per k16 step, 32 total ----
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load B from global: pack 16 k-rows into int8
            int8 b_val;
            if (col_valid) {
                int gk = k_cur + kk;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int k0 = gk + 2 * p;
                    int k1 = k0 + 1;
                    short s0 = (k0 < K) ? as_short(B[k0 * N + col_idx]) : (short)0;
                    short s1 = (k1 < K) ? as_short(B[k1 * N + col_idx]) : (short)0;
                    ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                }
            } else {
                b_val = (int8)(0);
            }

            // Load A blocks from SLM and compute
            short8 a_block;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[r * SLM_A_STRIDE + kk + sg_lid]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc0);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc1);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc2);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc3);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(32 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc4);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(40 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc5);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(48 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc6);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(cur_A[(56 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc7);
        }

        // ---- Load next A tile into nxt_buf ----
        if (has_next) {
            barrier(CLK_LOCAL_MEM_FENCE);
            __local half* nxt_A = slm_A + nxt_buf * TILE_M * SLM_A_STRIDE;
            if (row_tile_valid && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_K;
                    int c = elem_id % TILE_K;
                    nxt_A[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + k_next + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_K;
                    int c = elem_id % TILE_K;
                    int gr = row_base + r;
                    int gk = k_next + c;
                    nxt_A[r * SLM_A_STRIDE + c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // ---- Store results ----
    if (col_valid) {
        #define STORE_BLOCK(block_idx, acc_var) \
        { \
            int row_off = row_base + (block_idx) * 8; \
            if (row_off + 8 <= M) { \
                _Pragma("unroll") \
                for (int r = 0; r < 8; r++) \
                    C[(row_off + r) * N + col_idx] = convert_half(((float*)&acc_var)[r]); \
            } else { \
                _Pragma("unroll") \
                for (int r = 0; r < 8; r++) \
                    if (row_off + r < M) \
                        C[(row_off + r) * N + col_idx] = convert_half(((float*)&acc_var)[r]); \
            } \
        }

        STORE_BLOCK(0, acc0)
        STORE_BLOCK(1, acc1)
        STORE_BLOCK(2, acc2)
        STORE_BLOCK(3, acc3)
        STORE_BLOCK(4, acc4)
        STORE_BLOCK(5, acc5)
        STORE_BLOCK(6, acc6)
        STORE_BLOCK(7, acc7)

        #undef STORE_BLOCK
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 2.190):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 128 cols, K-step: 64
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// Double-buffered SLM for A and B
// GWS = (ceil(N/128)*128, ceil(M/32))  LWS = (128, 1)

#define TILE_M 32
#define TILE_N 128
#define TILE_K 64
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
    const int sg_id = get_sub_group_id();   // 0..7
    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int lid = get_local_id(0);        // 0..127

    const int n_base = get_group_id(0) * TILE_N;
    const int col_base = n_base + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    // Double-buffered SLM: 2 buffers for A (32 x 66) and B (64 x 130)
    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];
    __local half slm_B[2 * TILE_K * SLM_B_STRIDE];

    // Early exit for completely out-of-bounds workgroups
    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const bool col_tile_valid = (n_base + TILE_N <= N);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // A tile: 32 x 64 = 2048 halves, 128 WIs => 16 each
    // B tile: 64 x 128 = 8192 halves, 128 WIs => 64 each

    // --- Load first tile into buffer 0 ---
    int buf = 0;
    int k = 0;
    {
        // Load A[row_base..+32, k..+64] into slm_A[buf]
        __local half* slm_A_cur = slm_A + buf * TILE_M * SLM_A_STRIDE;
        if (row_tile_valid && k + TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_K;
                int c = elem_id % TILE_K;
                slm_A_cur[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + k + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_K;
                int c = elem_id % TILE_K;
                int gr = row_base + r;
                int gk = k + c;
                slm_A_cur[r * SLM_A_STRIDE + c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
            }
        }

        // Load B[k..+64, n_base..+128] into slm_B[buf]
        __local half* slm_B_cur = slm_B + buf * TILE_K * SLM_B_STRIDE;
        if (col_tile_valid && k + TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 64; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_N;
                int c = elem_id % TILE_N;
                slm_B_cur[r * SLM_B_STRIDE + c] = B[(k + r) * N + n_base + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 64; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_N;
                int c = elem_id % TILE_N;
                int gk = k + r;
                int gn = n_base + c;
                slm_B_cur[r * SLM_B_STRIDE + c] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Main loop with double buffering
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int cur_buf = kt & 1;
        int next_buf = 1 - cur_buf;
        int k_cur = kt * TILE_K;
        int k_next = (kt + 1) * TILE_K;

        __local half* slm_A_cur = slm_A + cur_buf * TILE_M * SLM_A_STRIDE;
        __local half* slm_B_cur = slm_B + cur_buf * TILE_K * SLM_B_STRIDE;

        // ---- Start loading next tile into next_buf (if exists) ----
        bool has_next = (kt + 1 < num_k_tiles);
        if (has_next) {
            __local half* slm_A_next = slm_A + next_buf * TILE_M * SLM_A_STRIDE;
            __local half* slm_B_next = slm_B + next_buf * TILE_K * SLM_B_STRIDE;

            if (row_tile_valid && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_K;
                    int c = elem_id % TILE_K;
                    slm_A_next[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + k_next + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_K;
                    int c = elem_id % TILE_K;
                    int gr = row_base + r;
                    int gk = k_next + c;
                    slm_A_next[r * SLM_A_STRIDE + c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
                }
            }

            if (col_tile_valid && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 64; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_N;
                    int c = elem_id % TILE_N;
                    slm_B_next[r * SLM_B_STRIDE + c] = B[(k_next + r) * N + n_base + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 64; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_N;
                    int c = elem_id % TILE_N;
                    int gk = k_next + r;
                    int gn = n_base + c;
                    slm_B_next[r * SLM_B_STRIDE + c] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
                }
            }
        }

        // ---- Compute on current buffer: 4 k16 steps x 4 row-blocks = 16 DPAS ----
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load A from SLM: 4 blocks of 8 rows, each short8
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a0)[r] = as_short(slm_A_cur[r * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a1)[r] = as_short(slm_A_cur[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a2)[r] = as_short(slm_A_cur[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a3)[r] = as_short(slm_A_cur[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);

            // Load B from SLM: int8 = 8 pairs of k-rows, packed as (short2)
            // B layout in SLM: [k_row][col], subgroup reads col_base..+16 via sg_lid
            int sg_col_offset = sg_id * 16;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_row0 = kk + 2 * p;
                int k_row1 = k_row0 + 1;
                short s0 = as_short(slm_B_cur[k_row0 * SLM_B_STRIDE + sg_col_offset + sg_lid]);
                short s1 = as_short(slm_B_cur[k_row1 * SLM_B_STRIDE + sg_col_offset + sg_lid]);
                ((int*)&b_val)[p] = as_int((short2)(s0, s1));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // Barrier before next iteration uses next_buf (which we just loaded)
        if (has_next) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Store results
    const bool col_valid = col_idx < N;
    if (col_valid) {
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + r) * N + col_idx] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 8 + r) * N + col_idx] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 16 + r) * N + col_idx] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 24 + r) * N + col_idx] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C[(row_base + r) * N + col_idx] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C[(row_base + 8 + r) * N + col_idx] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C[(row_base + 16 + r) * N + col_idx] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C[(row_base + 24 + r) * N + col_idx] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.330):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TILE_M=64, TILE_N=128, TILE_K=32
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 8 vertical 8x16 DPAS tiles = 64 rows x 16 cols
// Per K-tile: 2 k16 steps x 8 row-blocks = 16 DPAS per subgroup
// Double-buffered SLM for A and B
// SLM A: 2 x 64 x 34 = 8704 bytes, SLM B: 2 x 32 x 130 = 16640 bytes, total ~25KB
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
    const int sg_id = get_sub_group_id();        // 0..7
    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int lid = get_local_id(0);             // 0..127

    const int n_base = get_group_id(0) * TILE_N;
    const int col_base = n_base + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];
    __local half slm_B[2 * TILE_K * SLM_B_STRIDE];

    if (row_base >= M || n_base >= N)
        return;

    // 8 accumulators for 8 row-blocks of 8 rows each = 64 rows
    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float8 acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const bool col_tile_valid = (n_base + TILE_N <= N);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // A tile: 64 x 32 = 2048 halves, 128 WIs => 16 each
    // B tile: 32 x 128 = 4096 halves, 128 WIs => 32 each

    // --- Load first tile into buffer 0 ---
    {
        __local half* slm_A_cur = slm_A;
        __local half* slm_B_cur = slm_B;

        // Load A
        if (row_tile_valid && TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_K;
                int c = elem_id % TILE_K;
                slm_A_cur[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_K;
                int c = elem_id % TILE_K;
                int gr = row_base + r;
                slm_A_cur[r * SLM_A_STRIDE + c] = (gr < M && c < K) ? A[gr * K + c] : (half)0.0h;
            }
        }

        // Load B
        if (col_tile_valid && TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_N;
                int c = elem_id % TILE_N;
                slm_B_cur[r * SLM_B_STRIDE + c] = B[r * N + n_base + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_N;
                int c = elem_id % TILE_N;
                int gn = n_base + c;
                slm_B_cur[r * SLM_B_STRIDE + c] = (r < K && gn < N) ? B[r * N + gn] : (half)0.0h;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int cur_buf = kt & 1;
        int next_buf = 1 - cur_buf;
        int k_cur = kt * TILE_K;
        int k_next = k_cur + TILE_K;
        bool has_next = (kt + 1 < num_k_tiles);

        __local half* slm_A_cur = slm_A + cur_buf * TILE_M * SLM_A_STRIDE;
        __local half* slm_B_cur = slm_B + cur_buf * TILE_K * SLM_B_STRIDE;

        // Load next tile while computing (true double buffering - writes to different buffer)
        if (has_next) {
            __local half* slm_A_nxt = slm_A + next_buf * TILE_M * SLM_A_STRIDE;
            __local half* slm_B_nxt = slm_B + next_buf * TILE_K * SLM_B_STRIDE;

            if (row_tile_valid && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_K;
                    int c = elem_id % TILE_K;
                    slm_A_nxt[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + k_next + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_K;
                    int c = elem_id % TILE_K;
                    int gr = row_base + r;
                    int gk = k_next + c;
                    slm_A_nxt[r * SLM_A_STRIDE + c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
                }
            }

            if (col_tile_valid && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_N;
                    int c = elem_id % TILE_N;
                    slm_B_nxt[r * SLM_B_STRIDE + c] = B[(k_next + r) * N + n_base + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_N;
                    int c = elem_id % TILE_N;
                    int gk = k_next + r;
                    int gn = n_base + c;
                    slm_B_nxt[r * SLM_B_STRIDE + c] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
                }
            }
        }

        // Compute: 2 k16 steps x 8 row-blocks = 16 DPAS
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            short8 a0, a1, a2, a3, a4, a5, a6, a7;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a0)[r] = as_short(slm_A_cur[r * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a1)[r] = as_short(slm_A_cur[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a2)[r] = as_short(slm_A_cur[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a3)[r] = as_short(slm_A_cur[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a4)[r] = as_short(slm_A_cur[(32 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a5)[r] = as_short(slm_A_cur[(40 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a6)[r] = as_short(slm_A_cur[(48 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a7)[r] = as_short(slm_A_cur[(56 + r) * SLM_A_STRIDE + kk + sg_lid]);

            int sg_col_offset = sg_id * 16;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_row0 = kk + 2 * p;
                int k_row1 = k_row0 + 1;
                short s0 = as_short(slm_B_cur[k_row0 * SLM_B_STRIDE + sg_col_offset + sg_lid]);
                short s1 = as_short(slm_B_cur[k_row1 * SLM_B_STRIDE + sg_col_offset + sg_lid]);
                ((int*)&b_val)[p] = as_int((short2)(s0, s1));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
            acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a4, b_val, acc4);
            acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a5, b_val, acc5);
            acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a6, b_val, acc6);
            acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a7, b_val, acc7);
        }

        if (has_next) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Store results
    const bool col_valid = col_idx < N;
    if (col_valid) {
        #pragma unroll
        for (int r = 0; r < 8; r++)
            if (row_base + r < M) C[(row_base + r) * N + col_idx] = convert_half(((float*)&acc0)[r]);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            if (row_base + 8 + r < M) C[(row_base + 8 + r) * N + col_idx] = convert_half(((float*)&acc1)[r]);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            if (row_base + 16 + r < M) C[(row_base + 16 + r) * N + col_idx] = convert_half(((float*)&acc2)[r]);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            if (row_base + 24 + r < M) C[(row_base + 24 + r) * N + col_idx] = convert_half(((float*)&acc3)[r]);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            if (row_base + 32 + r < M) C[(row_base + 32 + r) * N + col_idx] = convert_half(((float*)&acc4)[r]);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            if (row_base + 40 + r < M) C[(row_base + 40 + r) * N + col_idx] = convert_half(((float*)&acc5)[r]);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            if (row_base + 48 + r < M) C[(row_base + 48 + r) * N + col_idx] = convert_half(((float*)&acc6)[r]);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            if (row_base + 56 + r < M) C[(row_base + 56 + r) * N + col_idx] = convert_half(((float*)&acc7)[r]);
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

=============================== warnings summary ===============================
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0]
  /home/openvino-ci-74/miniforge3/envs/kernel_intel/lib/python3.12/site-packages/pyopencl/cache.py:517: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
    _create_built_program_from_source_cached(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 4 passed, 1 deselected, 1 warning in 0.79s ==================
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
1. Kernel Fusion: Combine multiple small kernels into one to eliminate intermediate global memory writes and reduce launch overhead. Use barrier(CLK_LOCAL_MEM_FENCE) between logical kernel phases within a work-group.
2. Use Built-in Functions: Prefer built-in functions (mad, fma, dot, cross, length) over manual implementations. Use fast_* variants (fast_normalize, fast_length) when precision allows. Use native_* for maximum speed with reduced precision.

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