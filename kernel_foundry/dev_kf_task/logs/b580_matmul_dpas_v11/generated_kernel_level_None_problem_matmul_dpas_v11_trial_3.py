

## Analysis

The best version so far (1.330ms) uses TILE_M=64, TILE_N=128, TILE_K=32 with both A and B in SLM, double-buffered. Key issues:

1. **Double buffering isn't truly overlapping** - loads and compute happen sequentially within each iteration (load next, then compute current). True overlap requires interleaving load instructions with DPAS instructions.
2. **SLM B is large** (2 × 32 × 130 × 2 = ~16.6KB) which combined with SLM A (~8.7KB) = ~25KB limits occupancy.
3. **Division/modulo in loops** for indexing is expensive - can be replaced with simpler arithmetic.
4. **Too many boundary checks** in the hot path.

**Key insight**: The best performing version already has the right tile sizes. Let me focus on:
- Truly interleaving SLM loads with DPAS compute within the same loop body
- Using TILE_K=64 to get more DPAS per barrier (4 k16 steps × 8 blocks = 32 DPAS per barrier)
- Only put A in SLM (shared across subgroups), read B directly from global/L2 (each subgroup reads different columns, no sharing benefit worth the SLM cost)
- Reduce SLM footprint to improve occupancy
- Precompute row/col indices to avoid division in loops

Removing B from SLM saves ~16KB, allowing better occupancy. B reads from L2 should be fine since the B580 has 24MB L2.

With TILE_K=64 and A-only in SLM: SLM = 2 × 64 × 66 × 2 = ~17KB - reasonable.

Each iteration: 4 k16 steps × 8 DPAS = 32 DPAS per barrier pair. Much better ratio.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TILE_M=64, TILE_N=128, TILE_K=64
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 8 row-blocks of 8 rows x 16 cols = 64x16
// Per K-tile: 4 k16 steps x 8 row-blocks = 32 DPAS per subgroup per barrier
// A in double-buffered SLM, B read from global (L2 cached)
// SLM: 2 x 64 x 66 x 2 bytes = ~16.9 KB
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)

#define TILE_M 64
#define TILE_N 128
#define TILE_K 64
#define SLM_A_STRIDE (TILE_K + 2)
#define NUM_WI 128
// A tile: 64 x 64 = 4096 halves, 128 WIs => 32 each

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

    // Double-buffered A in SLM
    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float8 acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_full = (row_base + TILE_M <= M);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Precompute load indices for A (avoid div/mod in loop)
    // 32 elements per WI, elem_id = lid + i*128, r = elem_id/64, c = elem_id%64
    int a_r[32], a_c[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int elem_id = lid + i * NUM_WI;
        a_r[i] = elem_id / TILE_K;  // row within tile (0..63)
        a_c[i] = elem_id % TILE_K;  // col within tile (0..63)
    }

    // --- Load first A tile into buffer 0 ---
    {
        __local half* dst = slm_A;
        if (row_tile_full && TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                dst[a_r[i] * SLM_A_STRIDE + a_c[i]] = A[(row_base + a_r[i]) * K + a_c[i]];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int gr = row_base + a_r[i];
                dst[a_r[i] * SLM_A_STRIDE + a_c[i]] = (gr < M && a_c[i] < K) ? A[gr * K + a_c[i]] : (half)0.0h;
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
        const bool k_tile_full = (k_cur + TILE_K <= K);

        __local half* cur_A = slm_A + cur_buf * TILE_M * SLM_A_STRIDE;

        // ---- Compute: 4 k16 steps x 8 row-blocks = 32 DPAS ----
        // Interleave B loads with DPAS for better latency hiding
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load B from global: pack 16 k-rows into int8
            int8 b_val;
            if (col_valid && k_tile_full) {
                int gk = k_cur + kk;
                __global const half* B_ptr = B + gk * N + col_idx;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    short s0 = as_short(B_ptr[0]);
                    short s1 = as_short(B_ptr[N]);
                    B_ptr += 2 * N;
                    ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                }
            } else if (col_valid) {
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

            // Load all 8 A blocks and fire DPAS immediately after each
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

        // ---- Load next A tile ----
        if (has_next) {
            barrier(CLK_LOCAL_MEM_FENCE);
            __local half* nxt_A = slm_A + nxt_buf * TILE_M * SLM_A_STRIDE;
            if (row_tile_full && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    nxt_A[a_r[i] * SLM_A_STRIDE + a_c[i]] = A[(row_base + a_r[i]) * K + k_next + a_c[i]];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int gr = row_base + a_r[i];
                    int gk = k_next + a_c[i];
                    nxt_A[a_r[i] * SLM_A_STRIDE + a_c[i]] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // ---- Store results ----
    if (col_valid) {
        if (row_tile_full) {
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

