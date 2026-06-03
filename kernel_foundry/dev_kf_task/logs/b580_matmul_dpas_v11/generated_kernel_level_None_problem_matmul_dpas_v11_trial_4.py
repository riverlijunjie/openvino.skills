

## Analysis

The best version so far (1.330ms) uses TILE_M=64, TILE_N=128, TILE_K=32 with both A and B in SLM and double buffering. Key issues:

1. **Double buffering is broken**: The code loads the next tile *before* the barrier, but writes to a different SLM buffer while reading from the current one. This is actually correct for double buffering - no WAR hazard since different buffers. But there's only ONE barrier at the end, which means the next iteration might read the next buffer before writes complete. The pattern needs: load next → compute current → barrier (ensures next writes visible).

2. **SLM B is expensive**: 32×128 = 4096 halves per tile, 32 loads per WI. This dominates load time. Since all 8 subgroups share the same B tile, SLM B makes sense, but the load cost is high.

3. **TILE_K=32 means only 2 k16 steps**: With 8 row-blocks × 2 k16 = 16 DPAS per barrier pair. Increasing to TILE_K=64 doubles this to 32 DPAS per barrier.

4. **Register pressure with 8 accumulators**: 8 × float8 = 64 floats = 256 bytes of register per subgroup lane. This is fine.

**Plan for improvement:**
- Use TILE_K=64 for more DPAS per barrier (32 DPAS per iteration)
- Keep TILE_M=64, TILE_N=128 for good compute density
- Both A and B in double-buffered SLM
- A: 64×64 = 4096 halves, 32 per WI; B: 64×128 = 8192 halves, 64 per WI
- SLM: 2×(64×66 + 64×130)×2 bytes ≈ 50KB — might be tight but should work
- Interleave loads with compute more carefully
- Actually, 50KB SLM might be too much. Let me reconsider.

On Xe2, SLM per Xe-core is 64KB. So 50KB should fit but limits occupancy to 1 WG per core.

Alternative: Keep B reads from global (L2 cached), only A in SLM. This worked in V1 at 2.16ms but V3 with both in SLM was 1.33ms. So SLM B helps.

Let me try: TILE_M=64, TILE_N=128, TILE_K=64, A+B in SLM, double-buffered.
- SLM A: 2 × 64 × 66 × 2 = 16,896 bytes
- SLM B: 2 × 64 × 130 × 2 = 33,280 bytes  
- Total: ~50KB — fits in 64KB SLM

This gives 4 k16 steps × 8 row-blocks = 32 DPAS per barrier pair.

Let me also try to reduce B SLM by using sub_group_block_read for B from global and only put A in SLM. But the V3 result shows SLM B is beneficial.

Let me go with the TILE_K=64 approach and carefully structure the double buffering.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TILE_M=64, TILE_N=128, TILE_K=64
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 8 vertical 8x16 DPAS tiles = 64 rows x 16 cols
// Per K-tile: 4 k16 steps x 8 row-blocks = 32 DPAS per subgroup per iteration
// Double-buffered SLM for A and B
// SLM A: 2 x 64 x 66 x 2 = ~17 KB
// SLM B: 2 x 64 x 130 x 2 = ~33 KB, total ~50 KB (fits 64KB SLM)
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)

#define TILE_M 64
#define TILE_N 128
#define TILE_K 64
#define SLM_A_STRIDE (TILE_K + 2)
#define SLM_B_STRIDE (TILE_N + 2)
#define NUM_WI 128

// A tile: 64 x 64 = 4096 halves, 128 WIs => 32 each
// B tile: 64 x 128 = 8192 halves, 128 WIs => 64 each

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

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float8 acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const bool col_tile_valid = (n_base + TILE_N <= N);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // --- Load first tile into buffer 0 ---
    {
        __local half* slm_A_cur = slm_A;
        __local half* slm_B_cur = slm_B;

        // Load A: 64x64 = 4096 halves, 32 per WI
        if (row_tile_valid && TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_K;
                int c = elem_id % TILE_K;
                slm_A_cur[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_K;
                int c = elem_id % TILE_K;
                int gr = row_base + r;
                slm_A_cur[r * SLM_A_STRIDE + c] = (gr < M && c < K) ? A[gr * K + c] : (half)0.0h;
            }
        }

        // Load B: 64x128 = 8192 halves, 64 per WI
        if (col_tile_valid && TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 64; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_N;
                int c = elem_id % TILE_N;
                slm_B_cur[r * SLM_B_STRIDE + c] = B[r * N + n_base + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 64; i++) {
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

        // --- Start loading next tile into next_buf while computing ---
        if (has_next) {
            __local half* slm_A_nxt = slm_A + next_buf * TILE_M * SLM_A_STRIDE;
            __local half* slm_B_nxt = slm_B + next_buf * TILE_K * SLM_B_STRIDE;

            if (row_tile_valid && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_K;
                    int c = elem_id % TILE_K;
                    slm_A_nxt[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + k_next + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
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
                for (int i = 0; i < 64; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_N;
                    int c = elem_id % TILE_N;
                    slm_B_nxt[r * SLM_B_STRIDE + c] = B[(k_next + r) * N + n_base + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 64; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_N;
                    int c = elem_id % TILE_N;
                    int gk = k_next + r;
                    int gn = n_base + c;
                    slm_B_nxt[r * SLM_B_STRIDE + c] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
                }
            }
        }

        // --- Compute: 4 k16 steps x 8 row-blocks = 32 DPAS per subgroup ---
        int sg_col_offset = sg_id * 16;

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load B from SLM: pack 16 k-rows into int8
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_row0 = kk + 2 * p;
                int k_row1 = k_row0 + 1;
                short s0 = as_short(slm_B_cur[k_row0 * SLM_B_STRIDE + sg_col_offset + sg_lid]);
                short s1 = as_short(slm_B_cur[k_row1 * SLM_B_STRIDE + sg_col_offset + sg_lid]);
                ((int*)&b_val)[p] = as_int((short2)(s0, s1));
            }

            // Load A blocks from SLM and DPAS
            short8 a_block;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(slm_A_cur[r * SLM_A_STRIDE + kk + sg_lid]);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc0);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(slm_A_cur[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc1);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(slm_A_cur[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc2);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(slm_A_cur[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc3);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(slm_A_cur[(32 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc4);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(slm_A_cur[(40 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc5);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(slm_A_cur[(48 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc6);

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a_block)[r] = as_short(slm_A_cur[(56 + r) * SLM_A_STRIDE + kk + sg_lid]);
            acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a_block, b_val, acc7);
        }

        if (has_next) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // --- Store results ---
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

