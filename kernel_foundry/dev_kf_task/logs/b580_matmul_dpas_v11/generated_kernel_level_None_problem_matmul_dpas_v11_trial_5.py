

## Analysis

Looking at the previous versions:

1. **Version 1** (2.160ms): A in SLM, B from global. Double buffering with barrier between load and compute.
2. **Version 2** (2.170ms): Both A and B in SLM, TILE_K=64. But loads and computes aren't truly overlapped - loads happen before compute in the same iteration.
3. **Best version** (1.330ms): TILE_K=32, both A and B in SLM, true double buffering where next tile loads happen *before* compute (writes to different buffer, no conflict). Single barrier after compute+load.

Key issues with the 1.330ms version:
- The "double buffering" loads next tile data *before* computing current tile, but there's still a barrier serialization point
- Only 16 DPAS per barrier (2 k16 steps × 8 row-blocks)
- SLM B loads use scalar addressing which may cause bank conflicts

**Proposed improvements:**
1. **TILE_K=64**: 4 k16 steps × 8 row-blocks = 32 DPAS per barrier pair, doubling compute density
2. **True overlap**: Load next tile into separate buffer while computing current - no conflict since different SLM regions
3. **Use `vload4` for SLM loads** where possible to reduce instruction count
4. **Only A in SLM, B from global** with subgroup block reads - reduces SLM pressure and allows larger TILE_K
5. **Better B access pattern**: Each subgroup reads same B data from L2 cache, which should be hot after first subgroup reads it

Actually, let me reconsider. The best version puts both A and B in SLM at TILE_K=32. Let me try TILE_K=32 but with a cleaner double-buffering pattern where we truly overlap loads with compute by doing: compute current buffer while loading into next buffer, with only 1 barrier.

Let me try a different approach: **Only A in SLM, B read directly from global via subgroup block reads**. This halves SLM usage, allowing more occupancy. With TILE_K=32, B data is 32×16 halves per subgroup per k-tile step, which fits in L2 easily.

Actually, looking more carefully, the best version already does this pattern. The key difference is it interleaves loads before compute. Let me try TILE_K=64 with only A in SLM to keep SLM small, and read B from global with better vectorized loads.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TILE_M=64, TILE_N=128, TILE_K=32
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup handles 64 rows x 16 cols of output
// Only A in double-buffered SLM; B read from global (L2 cached)
// SLM: 2 x 64 x 34 x 2 bytes = ~8.7 KB -> good occupancy
// True double buffering: load next A while computing current A
// B loaded via subgroup reads from global (all 8 SGs read same K-rows, different columns)
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)

#define TILE_M 64
#define TILE_N 128
#define TILE_K 32
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
    const int col_base = n_base + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    // Double-buffered A in SLM
    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float8 acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

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

        __local const half* cur_A = slm_A + cur_buf * TILE_M * SLM_A_STRIDE;

        // Pre-load all 8 A blocks for first k16 step into registers
        // Then interleave B loads with DPAS calls

        // ---- k16 step 0 ----
        {
            const int kk = 0;
            // Load B tile for this k16 step
            int8 b_val;
            if (col_valid) {
                int gk = k_cur + kk;
                __global const half* B_base = B + gk * N + col_idx;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int k0 = 2 * p;
                    int k1 = k0 + 1;
                    short s0 = (gk + k0 < K) ? as_short(B_base[k0 * N]) : (short)0;
                    short s1 = (gk + k1 < K) ? as_short(B_base[k1 * N]) : (short)0;
                    ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                }
            } else {
                b_val = (int8)(0);
            }

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

        // ---- k16 step 1 ----
        {
            const int kk = 16;
            int8 b_val;
            if (col_valid) {
                int gk = k_cur + kk;
                __global const half* B_base = B + gk * N + col_idx;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int k0 = 2 * p;
                    int k1 = k0 + 1;
                    short s0 = (gk + k0 < K) ? as_short(B_base[k0 * N]) : (short)0;
                    short s1 = (gk + k1 < K) ? as_short(B_base[k1 * N]) : (short)0;
                    ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                }
            } else {
                b_val = (int8)(0);
            }

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

        // Load next A tile
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

    // Store results
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
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 32 + r) * N + col_idx] = convert_half(((float*)&acc4)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 40 + r) * N + col_idx] = convert_half(((float*)&acc5)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 48 + r) * N + col_idx] = convert_half(((float*)&acc6)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 56 + r) * N + col_idx] = convert_half(((float*)&acc7)[r]);
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
}
```

