

## Analysis

The best implementation so far (2.190ms) uses TILE_M=32, TILE_N=128, TILE_K=64 with 128 WIs (8 subgroups), double-buffered A and B in SLM. The previous attempt (5.050ms) regressed by using smaller tiles (TILE_K=32) and only buffering A in SLM while reading B from global.

Key issues with the best version:
1. **Double buffering isn't truly overlapping** - the loads and computes are sequential within the loop body. The loads happen before the barrier, then compute happens, then barrier. True overlap requires loading next tile *while* computing current tile.
2. **Massive SLM usage for B** - 2 × 64 × 130 × 2 bytes = ~33KB for B alone, plus A. This limits occupancy.
3. **Too many unrolled loads** - 64 iterations for B loading per WI creates register pressure.

My approach:
- **TILE_M=64, TILE_N=128, TILE_K=32** - Larger M tile means 8 row-blocks × 2 k16 steps = 16 DPAS per k-iteration, with 8 subgroups covering 128 columns. Each subgroup computes 64×16.
- **Only A in SLM, B from global** - B reads from global with subgroup block reads are efficient since each subgroup reads the same 16 columns. This drastically reduces SLM usage and allows better occupancy.
- **256 WIs (16 subgroups)** - 8 subgroups across N (128 cols), 2 subgroups across M (each handling 32 rows of the 64-row tile). This increases EU occupancy.

Actually, let me reconsider. The best version at 2.19ms has both A and B in SLM. Let me try to improve it with better overlap and TILE_M=64.

Let me use: TILE_M=64, TILE_N=128, TILE_K=32, 256 WIs (16 subgroups). Each subgroup handles 8 rows × 16 cols. 16 subgroups = 2 rows × 8 cols. Per k-tile: 2 k16 steps × 1 row-block = 2 DPAS per subgroup, but 16 DPAS total across all subgroups sharing the same data.

Actually, DPAS count per subgroup matters. Let me keep each subgroup doing multiple row-blocks.

Better plan: **TILE_M=64, TILE_N=128, TILE_K=32, 128 WIs (8 subgroups)**. Each subgroup: 8 row-blocks of 8 rows = 64 rows × 16 cols. Per k16 step: 8 DPAS. 2 k16 steps = 16 DPAS per k-tile. Both A and B in SLM, double buffered. SLM: A = 2×64×34 = 8.5KB, B = 2×32×130 = 16.6KB, total ~25KB - fits well.

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

