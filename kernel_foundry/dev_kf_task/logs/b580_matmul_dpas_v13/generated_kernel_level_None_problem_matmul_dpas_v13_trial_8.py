## Analysis

The previous kernel (64x128 tile with 4x4 SG layout) produced incorrect results - zeros in the right portion of the output. The issue is the launch configuration. The test harness warning says: `"matmul_opt.cl has no parsable reqd_work_group_size; fallback launch (N, M), local=None"`. With `reqd_work_group_size(256, 1, 1)`, GWS should be `(ceil(N/128)*256, ceil(M/64))`, but the fallback launches with GWS=(N, M) and no local size, meaning workgroups don't cover the full N dimension properly with the 128-col tile.

The best performing correct versions (V1: 1.140ms, V2: 1.160ms) use 32x256 tiles with 16 SGs, only caching A in SLM. Let me go back to the proven V1/V2 approach but optimize it further:

**Key improvements over V1/V2:**
1. **Larger tile with both A and B in SLM** but with correct launch config that the harness can parse
2. **Use 32x128 tile** (like the "best" kernel at 4.35ms) but optimize by NOT putting B in SLM - B loading from global is fine with 16x reuse of A
3. Actually, V1 at 1.14ms is our best. Let me try to improve upon it with better K-loop pipelining and prefetching.

Let me stick with the V2 structure (cleanest, 1.16ms) and optimize:
- Use `__attribute__((reqd_work_group_size(256, 1, 1)))` so the harness can parse it
- Process K in chunks of 32 (two k16 DPAS per SLM load) to reduce barrier overhead
- Only A in SLM (16x reuse), B from global
- Double-buffer A in SLM: 2 * 32 * 32 * 2 = 4KB

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Tile: 32M x 256N, K-step=32 (two k16 DPAS rounds per SLM load)
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16) per k16, 8 per k32
// Only A in SLM (shared across 16 subgroups = 16x reuse)
// Double-buffered A: 2 * 32*32 * 2 bytes = 4 KB
// B loaded directly from global
// GWS = (ceil(N/256)*256, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16

#define TILE_M 32
#define TILE_N 256
#define TILE_K 32
#define K16 16
#define SLM_A_SIZE (TILE_M * TILE_K)  // 1024 ushorts = 2KB per buffer
#define WG_SIZE 256

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
    const bool col_in_bounds = (col_base + 16 <= N);

    // Load first A tile (32 rows x 32 cols) into buffer 0
    // 32*32 = 1024 elements, 1024/256 = 4 per WI
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

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int cur_buf = kt & 1;

        __local const ushort* ca = slm_a + cur_buf * SLM_A_SIZE;

        // Two k16 rounds per k32 tile
        #pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            int k16_off = ks * K16;
            int k_abs = k_offset + k16_off;

            // Read A from SLM: 32 rows, 16 cols starting at k16_off
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(ca + r * TILE_K + k16_off);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(ca + (8 + r) * TILE_K + k16_off);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(ca + (16 + r) * TILE_K + k16_off);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(ca + (24 + r) * TILE_K + k16_off);

            // Load B from global: 16 rows x 16 cols
            int8 b_val;
            if (col_in_bounds) {
                __global const ushort* bp = B_us + k_abs * N + col_base;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = intel_sub_group_block_read_us(bp + (2 * p) * N);
                    ushort s1 = intel_sub_group_block_read_us(bp + (2 * p + 1) * N);
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                }
            } else {
                int col = col_base + sg_lid;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int kr0 = k_abs + 2 * p;
                    int kr1 = kr0 + 1;
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

        // Load next A tile
        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - cur_buf;
            const int next_k = (kt + 1) * TILE_K;
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
        }
    }

    // Store results
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

