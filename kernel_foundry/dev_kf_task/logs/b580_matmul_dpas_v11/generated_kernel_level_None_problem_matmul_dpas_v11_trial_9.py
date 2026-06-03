

## Analysis

Looking at the results:
- **Version 1 (TILE_K=32)**: 1.310ms - best performer
- **Version 2 (TILE_K=32)**: 1.330ms - similar, loads before compute (no true overlap)
- **TILE_K=64 versions**: 2.160ms - worse due to excessive SLM usage (~50KB), register pressure, and too many unrolled loads

The TILE_K=64 approach backfires because:
1. **SLM pressure**: 50KB SLM limits occupancy (only 1 WG per Xe-core)
2. **Register pressure**: 64 loads per WI for B tile causes spilling
3. **Division by 64/128**: The `eid / TILE_K` with TILE_K=64 can't use bit shifts efficiently for all cases

Key improvements for Version 1:
1. **Reduce SLM size** - stick with TILE_K=32 which uses ~25KB, allowing potentially 2 WGs per core
2. **Use power-of-2 strides** - TILE_K=32 means `eid >> 5` and `eid & 31` work perfectly
3. **Better interleaving** - load A between k16 steps, load B after all compute, minimize stalls
4. **Use `intel_sub_group_block_read`** for SLM reads where possible
5. **Reduce padding** - use +0 or minimal padding since TILE_K=32 and TILE_N=128 are already bank-friendly

Let me try a cleaner version based on Version 1 with these refinements:
- TILE_M=64, TILE_N=128, TILE_K=32 (proven best)
- Tighter double buffering: all loads happen *during* compute phase
- Minimize SLM padding (TILE_K=32 is already 32 banks wide - use stride 32 for A, 128 for B to avoid waste)
- Actually, stride 32 for A means all rows hit same banks. Use stride 34 for A, stride 130 for B (as before)
- Pre-load B into registers before DPAS to reduce SLM read latency impact

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

