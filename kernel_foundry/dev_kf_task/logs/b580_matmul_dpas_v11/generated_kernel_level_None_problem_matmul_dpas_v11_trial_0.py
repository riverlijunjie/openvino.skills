

## Analysis

The initial implementation achieves only 21% XMX utilization with 8 DPAS per 2 barriers. The key optimizations I'll apply:

1. **Double buffering**: Use ping-pong SLM buffers to overlap loading the next K-tile with computing on the current one, eliminating one barrier per iteration.
2. **TILE_K=64**: 4 k16 steps × 4 row-blocks = 16 DPAS per barrier pair, dramatically improving the DPAS-to-barrier ratio.
3. **Larger WG (128 WIs, 8 subgroups)**: Better cooperative loading and EU occupancy. Tile becomes 32×128 (8 subgroups × 16 cols each).
4. **SLM padding**: +2 padding on A to avoid bank conflicts.
5. **B in SLM**: Cache B tile to avoid redundant L2 reads across subgroups.
6. **Loop unrolling**: Fully unroll the inner K-loop.

With double buffering and TILE_K=64:
- Load next tile into buffer[1] while computing on buffer[0]
- Only 1 barrier needed between iterations (to ensure previous load is done before compute starts on it)
- 16 DPAS per barrier → much better ratio

Tile: 32M × 128N × 64K, 8 subgroups × 16 WIs = 128 WIs per WG.

## OCL Code

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

**Key improvements over the initial implementation:**

1. **Double buffering**: Ping-pong SLM buffers allow loading the next K-tile while computing on the current one. This reduces barriers from 2 per iteration to 1, and overlaps memory latency with compute.

2. **TILE_K=64**: 4 k16 steps × 4 row-blocks = **16 DPAS per barrier** (up from 8 DPAS per 2 barriers). This is the single biggest improvement for DPAS-to-barrier ratio.

3. **8 subgroups (128 WIs)**: TILE_N=128 means each WG covers 32×128 output elements. More WIs for faster cooperative loads and better EU occupancy.

4. **B in SLM**: All 8 subgroups share the same B tile from SLM instead of each independently reading from global memory, reducing L2 pressure.

5. **SLM padding**: +2 on both A and B strides to avoid bank conflicts.

**Launch config**: GWS = (ceil(N/128)×128, ceil(M/32)), LWS = (128, 1).

