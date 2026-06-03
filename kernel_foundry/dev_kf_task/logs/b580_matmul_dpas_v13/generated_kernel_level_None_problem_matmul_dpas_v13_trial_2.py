

## Analysis

Looking at the previous versions:

1. **Version 1 (5.13ms)** and **Version 3 (4.73ms)**: Both use 32×128 WG tile with 8 subgroups. The key difference is memory access patterns during cooperative loading. Version 3 uses `lid * 8 + i` indexing which is non-coalesced for A but the layout still works. Version 1 uses `lid + i * 128` which is coalesced.

2. **Version 2 (6.43ms)**: Uses 64×128 tile with 2×4 subgroup layout. Slower due to more SLM reads per subgroup (reading both A and B column blocks) and higher register pressure with 8 accumulators.

**Key bottlenecks in Version 3 (best so far at 4.73ms):**
- Non-coalesced global memory loads for A (`lid * 8 + i` pattern)
- Each of 8 subgroups reads the same A data from SLM independently (redundant SLM reads)
- Only 8 DPAS calls per k16 step, low compute density
- Store phase uses scalar writes

**Improvements for this version:**
1. **Coalesced global loads**: Use `lid + i * WG_SIZE` pattern for both A and B
2. **Larger TILE_K (64)** to reduce barrier/load overhead and increase compute per SLM load
3. **Interleave load and compute**: Start prefetching next tile while computing current
4. **Use block writes for output** via `intel_sub_group_block_write_us`
5. **Remove boundary checks** in the hot path when tiles are fully valid

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 128 cols, K-step: 16 (loaded from global per iteration)
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16) tiles
// A tile in SLM: 32 x 16 = 512 ushorts = 1 KB
// B tile in SLM: 16 x 128 = 2048 ushorts = 4 KB
// Single buffer SLM: 5 KB total (very small, fits easily)
// GWS = (ceil(N/128)*128, ceil(M/32))  LWS = (128, 1)
// Subgroup size = 16

#define TILE_M 32
#define TILE_N 128
#define TILE_K 16
#define SLM_A_SIZE (TILE_M * TILE_K)   // 512 ushorts
#define SLM_B_SIZE (TILE_K * TILE_N)   // 2048 ushorts
#define SLM_BUF_SIZE (SLM_A_SIZE + SLM_B_SIZE) // 2560 ushorts
#define WG_SIZE 128

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
    const int sg_id  = get_sub_group_id();        // 0..7
    const int sg_lid = get_sub_group_local_id();  // 0..15
    const int lid    = get_local_id(0);           // 0..127

    const int n_base   = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;
    const int col_base = n_base + sg_id * 16;

    // Double-buffered SLM
    __local ushort slm[2 * SLM_BUF_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // A load: 32 x 16 = 512 elements, 512/128 = 4 per WI
    // B load: 16 x 128 = 2048 elements, 2048/128 = 16 per WI

    // --- Load first tile into buffer 0 ---
    {
        __local ushort* slm_a = slm;
        __local ushort* slm_b = slm + SLM_A_SIZE;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;  // 0..31
            int ac = eid % TILE_K;  // 0..15
            int gr = row_base + ar;
            ushort val = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
            slm_a[eid] = val;
        }

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int eid = lid + i * WG_SIZE;
            int br = eid / TILE_N;  // 0..15
            int bc = eid % TILE_N;  // 0..127
            int gc = n_base + bc;
            ushort val = (br < K && gc < N) ? B_us[br * N + gc] : (ushort)0;
            slm_b[eid] = val;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int next_kt = kt + 1;
        const bool has_next = (next_kt < num_k_tiles);

        __local const ushort* cur_a = slm + cur_buf * SLM_BUF_SIZE;
        __local const ushort* cur_b = cur_a + SLM_A_SIZE;
        __local const ushort* my_b = cur_b + sg_id * 16;

        // ---- Single k16 DPAS step ----
        {
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(cur_a + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(cur_a + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(cur_a + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(cur_a + (24 + r) * TILE_K);

            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(my_b + (2 * p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(my_b + (2 * p + 1) * TILE_N);
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ---- Load next tile ----
        if (has_next) {
            int next_buf = 1 - cur_buf;
            __local ushort* next_a = slm + next_buf * SLM_BUF_SIZE;
            __local ushort* next_b = next_a + SLM_A_SIZE;
            int next_k = next_kt * TILE_K;

            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = row_base + ar;
                int gk = next_k + ac;
                next_a[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * WG_SIZE;
                int br = eid / TILE_N;
                int bc = eid % TILE_N;
                int gk = next_k + br;
                int gc = n_base + bc;
                next_b[eid] = (gk < K && gc < N) ? B_us[gk * N + gc] : (ushort)0;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = next_buf;
        }
    }

    // ---- Store results ----
    const int col_idx = col_base + sg_lid;
    if (col_idx < N) {
        __global half* C_ptr = C + col_idx;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc0)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 8 + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc1)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 16 + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc2)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 24 + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

