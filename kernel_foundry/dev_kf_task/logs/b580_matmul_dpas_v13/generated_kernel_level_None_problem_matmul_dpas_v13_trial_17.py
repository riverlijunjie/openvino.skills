

## Analysis

Looking at the results:
- **Best version (1.140ms)**: Uses 32M×256N tile, K-step=16, only A in SLM, B from global. Simple double-buffering.
- **Last version (4.700ms)**: Puts both A and B in SLM (36KB double-buffered). The massive B tile loading (32 iterations per WI for B) creates a huge bottleneck - 32 scalar loads per WI per k-tile dominates execution time.
- **Version 1 (1.150ms)**: Similar to best, uses K-step=32 with A in SLM.

Key observations:
1. Loading B into SLM is counterproductive - it's too large (16KB per buffer) and each element is only read once per subgroup, so there's no reuse benefit.
2. The best versions keep only A in SLM (high reuse: 16 subgroups share A) and load B directly.
3. The bottleneck at ~1.14ms is likely: too many barriers, insufficient compute-to-memory overlap, or suboptimal tile shape.

**Proposed improvements:**
1. **Larger M tile (64M)** with fewer subgroups across N (8 SGs) → 64M×128N tile. Each SG handles 64 rows × 16 cols = 8 DPAS per k16. A reuse = 8x, but we compute 2x more per SG.
2. **Better: Keep 32M×256N but reduce barrier overhead** by doing K-step=32 with proper overlapping of A load and DPAS.
3. **Prefetch B into registers** while computing with current data.
4. Actually, let me try a different approach: **32M×128N with 8 subgroups**, allowing each SG to own 2 columns (32 cols) → more register work per SG, fewer SGs, less contention.

Let me try a cleaner version of the best approach with better pipelining: load next A tile while doing DPAS on current tile, and use K-step=32 to halve barriers.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Optimized GEMM: C[M,N] = A[M,K] * B[K,N], all half, acc in float
// Tile: 32M x 128N, K-step=32 (two k16 DPAS rounds per A load)
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16) per k16, 8 per k32
// A in SLM double-buffered: 2 * 32*32 * 2B = 4KB
// B loaded directly from global per SG
// A reuse: 8x (8 SGs share same A tile)
//
// GWS = (ceil(N/128)*128, ceil(M/32))  LWS = (128, 1)
// Subgroup size = 16

#define TILE_M 32
#define TILE_N 128
#define TILE_K 32
#define K16 16
#define SLM_A_SIZE (TILE_M * TILE_K)  // 1024 ushorts per buffer
#define WG_SIZE 128
#define NUM_SG 8

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id  = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid    = get_local_id(0);

    const int n_base   = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;
    const int col_base = n_base + sg_id * 16;

    // Double-buffered SLM for A
    __local ushort slm_a[2 * SLM_A_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const bool col_in_bounds = (col_base + 16 <= N);

    // Load first A tile (32x32) into buffer 0
    // 32*32 = 1024 elements, 1024/128 = 8 per WI
    {
        __local ushort* dst = slm_a;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;
            int ac = eid % TILE_K;
            int gr = row_base + ar;
            dst[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int cur_buf = kt & 1;
        __local const ushort* ca = slm_a + cur_buf * SLM_A_SIZE;

        // ======== K16 round 0: columns [0..15] ========
        short8 a0, a1, a2, a3;
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a0)[r] = intel_sub_group_block_read_us(ca + r * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a1)[r] = intel_sub_group_block_read_us(ca + (8 + r) * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a2)[r] = intel_sub_group_block_read_us(ca + (16 + r) * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a3)[r] = intel_sub_group_block_read_us(ca + (24 + r) * TILE_K);

        // Load B for k16 round 0
        int8 b_val0;
        if (col_in_bounds) {
            __global const ushort* bp = B_us + k_offset * N + col_base;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(bp + (2 * p) * N);
                ushort s1 = intel_sub_group_block_read_us(bp + (2 * p + 1) * N);
                ((int*)&b_val0)[p] = as_int((ushort2)(s0, s1));
            }
        } else {
            int col = col_base + sg_lid;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int kr0 = k_offset + 2 * p;
                int kr1 = kr0 + 1;
                ushort s0 = (kr0 < K && col < N) ? B_us[kr0 * N + col] : (ushort)0;
                ushort s1 = (kr1 < K && col < N) ? B_us[kr1 * N + col] : (ushort)0;
                ((int*)&b_val0)[p] = as_int((ushort2)(s0, s1));
            }
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val0, acc3);

        // ======== K16 round 1: columns [16..31] ========
        int k1 = k_offset + K16;

        short8 a0b, a1b, a2b, a3b;
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a0b)[r] = intel_sub_group_block_read_us(ca + r * TILE_K + K16);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a1b)[r] = intel_sub_group_block_read_us(ca + (8 + r) * TILE_K + K16);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a2b)[r] = intel_sub_group_block_read_us(ca + (16 + r) * TILE_K + K16);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a3b)[r] = intel_sub_group_block_read_us(ca + (24 + r) * TILE_K + K16);

        int8 b_val1;
        if (k1 < K) {
            if (col_in_bounds) {
                __global const ushort* bp = B_us + k1 * N + col_base;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = intel_sub_group_block_read_us(bp + (2 * p) * N);
                    ushort s1 = intel_sub_group_block_read_us(bp + (2 * p + 1) * N);
                    ((int*)&b_val1)[p] = as_int((ushort2)(s0, s1));
                }
            } else {
                int col = col_base + sg_lid;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int kr0 = k1 + 2 * p;
                    int kr1 = kr0 + 1;
                    ushort s0 = (kr0 < K && col < N) ? B_us[kr0 * N + col] : (ushort)0;
                    ushort s1 = (kr1 < K && col < N) ? B_us[kr1 * N + col] : (ushort)0;
                    ((int*)&b_val1)[p] = as_int((ushort2)(s0, s1));
                }
            }
        } else {
            b_val1 = 0;
        }

        // Overlap: start loading next A tile while doing DPAS round 1
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - cur_buf;
            const int next_k = (kt + 1) * TILE_K;
            __local ushort* dst = slm_a + next_buf * SLM_A_SIZE;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = row_base + ar;
                int gk = next_k + ac;
                dst[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }
        }

        // DPAS round 1 overlaps with SLM writes
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0b, b_val1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1b, b_val1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2b, b_val1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3b, b_val1, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ---- Store results using block writes where possible ----
    const int col_idx = col_base + sg_lid;
    if (col_idx < N) {
        __global ushort* C_us = (__global ushort*)C;

        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc0)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 8 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc1)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 16 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc2)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 24 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc3)[r]));
        }
    }
}
```

