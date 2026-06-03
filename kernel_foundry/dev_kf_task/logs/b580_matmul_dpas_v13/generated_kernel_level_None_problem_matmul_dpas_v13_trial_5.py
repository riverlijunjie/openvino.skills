

## Analysis

Looking at the previous versions:

1. **Version 1 (1.140ms)** - Best performer. Uses 32x256 tile with 16 subgroups, only A in SLM, B direct from global. Double-buffered A with K-pair unrolling.
2. **Version 2 (1.720ms)** - Same tile but single-buffered loop, slower due to less pipelining.
3. **Best so far (4.350ms)** - 32x128 tile with both A and B in SLM. B loading (64 elements/WI) dominates and creates a bottleneck.
4. **Last tested (11.800ms)** - 32x128 with K=64, massive SLM (40KB), 64 B-loads per WI kills performance.

**Key insights:**
- Putting B in SLM is counterproductive when TILE_N is large - the loading cost exceeds the benefit
- Version 1's approach (A in SLM, B from global) is best since A is shared across 16 SGs (16x reuse) while each SG needs different B columns
- The K-pair unrolling in V1 helps pipeline loads and compute
- We should try to increase compute density further

**Improvements for new version:**
1. Start from Version 1's structure (best so far)
2. Use K-step of 32 (2 k16 steps per SLM load) to amortize barrier cost
3. Keep only A in SLM with double buffering
4. Prefetch next A tile while computing current tile
5. Use `intel_sub_group_block_read` for B loads where possible
6. Use block writes for output

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Tile: 32M x 256N, K-step=32 (two k16 DPAS per SLM load)
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16) tiles
// Only A in SLM (shared across 16 subgroups = 16x reuse)
// B loaded directly from global (each SG loads its own 16x16 block)
// Double-buffered A in SLM: 2 * 32*32 * 2 bytes = 4 KB
// GWS = (ceil(N/256)*16, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16

#define TILE_M 32
#define TILE_N 256
#define TILE_K 32
#define K16 16
#define SLM_A_SIZE (TILE_M * TILE_K)  // 1024 ushorts = 2KB per buffer
#define WG_SIZE 256
#define NUM_SG 16

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

    // Load first A tile (k=0) into buffer 0
    // 32 * 32 = 1024 elements, 1024/256 = 4 per WI
    {
        __local ushort* dst = slm_a;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;
            int ac = eid % TILE_K;
            int gr = row_base + ar;
            dst[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;
    const bool col_in_bounds = (col_base + 16 <= N);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int next_kt = kt + 1;
        const bool has_next = (next_kt < num_k_tiles);

        __local const ushort* ca = slm_a + cur_buf * SLM_A_SIZE;

        // ======== K16 step 0 ========
        {
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

            int8 b_val;
            __global const ushort* bp = B_us + k_offset * N + col_base;
            if (col_in_bounds) {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = intel_sub_group_block_read_us(bp + (2*p) * N);
                    ushort s1 = intel_sub_group_block_read_us(bp + (2*p+1) * N);
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                }
            } else {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int col = col_base + sg_lid;
                    int kr0 = k_offset + 2*p;
                    int kr1 = k_offset + 2*p+1;
                    ushort s0 = (kr0 < K && col < N) ? B_us[kr0 * N + col] : (ushort)0;
                    ushort s1 = (kr1 < K && col < N) ? B_us[kr1 * N + col] : (ushort)0;
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                }
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ======== K16 step 1 ========
        {
            int k1 = k_offset + K16;
            // A is at offset +16 within the same SLM tile
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(ca + r * TILE_K + K16);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(ca + (8 + r) * TILE_K + K16);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(ca + (16 + r) * TILE_K + K16);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(ca + (24 + r) * TILE_K + K16);

            int8 b_val;
            if (k1 < K) {
                __global const ushort* bp = B_us + k1 * N + col_base;
                if (col_in_bounds) {
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        ushort s0 = intel_sub_group_block_read_us(bp + (2*p) * N);
                        ushort s1 = intel_sub_group_block_read_us(bp + (2*p+1) * N);
                        ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                    }
                } else {
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        int col = col_base + sg_lid;
                        int kr0 = k1 + 2*p;
                        int kr1 = k1 + 2*p+1;
                        ushort s0 = (kr0 < K && col < N) ? B_us[kr0 * N + col] : (ushort)0;
                        ushort s1 = (kr1 < K && col < N) ? B_us[kr1 * N + col] : (ushort)0;
                        ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                    }
                }
            } else {
                b_val = 0;
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ---- Load next A tile ----
        if (has_next) {
            int next_buf = 1 - cur_buf;
            int next_k = next_kt * TILE_K;

            barrier(CLK_LOCAL_MEM_FENCE);

            __local ushort* dst = slm_a + next_buf * SLM_A_SIZE;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = row_base + ar;
                int gk = next_k + ac;
                dst[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = next_buf;
        }
    }

    // ---- Store results ----
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

