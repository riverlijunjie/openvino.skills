

## Analysis

Looking at the previous versions:

1. **Version 1 (4.730ms)**: TILE_K=32, double-buffered SLM, but sequential loading pattern (lid*8+i) causes poor coalescing.
2. **Best so far (4.350ms)**: TILE_K=16, double-buffered SLM, strided loading (lid+i*128) for better coalescing. Smaller K tile means less SLM but more iterations.
3. **Last tested (5.140ms)**: TILE_K=32, strided loading, but 32 B loads per WI is too much work per iteration.

Key bottlenecks:
- **B loading dominates**: Loading 4096 elements for B (32 per WI) is expensive with TILE_K=32.
- **Low compute-to-load ratio**: With TILE_K=16, we only do 4 DPAS per subgroup per iteration but pay barrier + load overhead each time.
- **Store is scalar**: Writing results element-by-element wastes bandwidth.

**Strategy for improvement:**
- Use **TILE_K=32** with better load strategy - use `intel_sub_group_block_read` for global loads where possible for higher bandwidth
- Increase compute density: try **TILE_M=64** so each subgroup does 8 DPAS tiles (64 rows x 16 cols), improving compute-to-memory ratio
- Actually, with 128 WIs and TILE_M=64, A becomes 64×32=2048 elements (16/WI) and B stays 32×128=4096 (32/WI). Too much.
- Better approach: Keep TILE_M=32, TILE_N=128, TILE_K=32 but use **prefetching** - overlap compute with loads by doing compute on current buffer while loading next.
- The real issue: the best version uses TILE_K=16 which has fewer loads but more loop iterations. Let me try TILE_K=32 with properly pipelined loads using async pattern.

Let me try a different approach: **larger TILE_N=256** with **16 subgroups** to increase the work per WG, or better yet, keep the tile sizes but improve the memory access patterns and reduce branch overhead.

Actually, looking more carefully - the best version at 4.35ms is still ~4.6x slower than 0.948ms target. The key insight: we need to **remove boundary checks** in the hot path and use **block reads from global memory** directly, avoiding SLM for B entirely since each subgroup reads its own B columns.

Let me try: **direct global memory reads with DPAS**, no SLM for B, only SLM for A (shared across subgroups).

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 256 cols, K-step: 16
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16) tiles
// A tile in SLM (shared): 32 x 16 = 512 ushorts = 1 KB, double-buffered = 2 KB
// B loaded directly from global memory by each subgroup (not shared)
// SLM total: 2 * 512 * 2 = 2 KB
// GWS = (ceil(N/256)*256, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16

#define TILE_M 32
#define TILE_N 256
#define TILE_K 16
#define SLM_A_SIZE (TILE_M * TILE_K)   // 512 ushorts
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
    const int sg_id  = get_sub_group_id();        // 0..15
    const int sg_lid = get_sub_group_local_id();  // 0..15
    const int lid    = get_local_id(0);           // 0..255

    const int n_base   = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;
    const int col_base = n_base + sg_id * 16;

    // Double-buffered SLM for A only
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

    // A load: 32 x 16 = 512 elements, 512/256 = 2 per WI
    // --- Load first A tile into buffer 0 ---
    {
        __local ushort* dst_a = slm_a;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;  // 0..31
            int ac = eid % TILE_K;  // 0..15
            int gr = row_base + ar;
            dst_a[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int next_kt = kt + 1;
        const bool has_next = (next_kt < num_k_tiles);

        __local const ushort* cur_a = slm_a + cur_buf * SLM_A_SIZE;

        // Load A from SLM
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

        // Load B directly from global memory
        // B[k_offset..k_offset+15, col_base..col_base+15]
        // Each subgroup reads its own 16x16 B block
        int8 b_val;
        __global const ushort* b_ptr = B_us + k_offset * N + col_base;
        
        if (col_base + 16 <= N) {
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int kr0 = k_offset + 2 * p;
                int kr1 = k_offset + 2 * p + 1;
                ushort s0 = (kr0 < K) ? intel_sub_group_block_read_us(b_ptr + (2 * p) * N) : (ushort)0;
                ushort s1 = (kr1 < K) ? intel_sub_group_block_read_us(b_ptr + (2 * p + 1) * N) : (ushort)0;
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }
        } else {
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int kr0 = k_offset + 2 * p;
                int kr1 = k_offset + 2 * p + 1;
                int col = col_base + sg_lid;
                ushort s0 = (kr0 < K && col < N) ? B_us[kr0 * N + col] : (ushort)0;
                ushort s1 = (kr1 < K && col < N) ? B_us[kr1 * N + col] : (ushort)0;
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);

        // ---- Load next A tile ----
        if (has_next) {
            barrier(CLK_LOCAL_MEM_FENCE);

            int next_buf = 1 - cur_buf;
            __local ushort* next_a = slm_a + next_buf * SLM_A_SIZE;
            int next_k = next_kt * TILE_K;

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = row_base + ar;
                int gk = next_k + ac;
                next_a[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
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

