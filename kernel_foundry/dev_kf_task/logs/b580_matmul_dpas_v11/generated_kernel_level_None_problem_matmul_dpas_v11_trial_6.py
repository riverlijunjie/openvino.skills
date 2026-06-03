## Analysis

Comparing the versions:

1. **Best version (1.330ms)**: Uses both A and B in SLM with double buffering. Loads next tile *before* compute (not truly overlapped), then barriers after compute. The key issue is that loads and compute are sequential within each iteration - loads happen first, then all DPAS, then barrier.

2. **Version 1 (2.040ms)**: Only A in SLM, B from global. Double buffering but barriers placed poorly (barrier before load, barrier after load = compute must wait).

3. **Last tested (2.160ms)**: Similar to V1 but with macro-based store. Same barrier placement issue.

**Key bottleneck in best version**: The double buffering isn't truly overlapping - loads to next buffer happen before compute on current buffer. The barrier at the end of the loop body means we wait for both loads AND compute before proceeding.

**Improvements for this version**:

1. **True double buffering**: Load next tile AFTER starting compute, interleave them. Since we write to a different SLM buffer than we read from, we can overlap loads with DPAS compute without barriers between them. Only need barrier before next iteration to ensure loads completed.

2. **TILE_K=64**: 4 k16 steps × 8 row-blocks = 32 DPAS per barrier pair, dramatically improving DPAS-to-barrier ratio.

3. **Avoid division/modulo** in cooperative loads by using simpler indexing.

4. **Use intel_sub_group_block_read** for SLM reads where possible for better throughput.

5. **Interleave A loads and B loads with DPAS compute** - start DPAS on first k16 step while loading data for later steps.

Let me implement a version with TILE_K=64, true double buffering (load next while computing current), and both A+B in SLM:

- TILE_M=64, TILE_N=128, TILE_K=64
- SLM A: 2 × 64 × 66 × 2 = ~16.9 KB  
- SLM B: 2 × 64 × 130 × 2 = ~33.3 KB
- Total: ~50 KB - might be too much for good occupancy

Better: TILE_K=32 with better overlap, or keep A in SLM only with TILE_K=32 but fix the overlap.

Actually, let me reconsider. The best version already has both A+B in SLM at ~25KB. The main win is to truly overlap loads with compute. Let me do that properly:

- Iteration structure: barrier → compute(cur_buf) interleaved with load(next_buf) → barrier
- Since we read cur_buf and write next_buf simultaneously, no conflict

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

