

## Analysis

Looking at the previous versions:

1. **Version 1 (1.140ms)** - 32x256 tile, K-step=16, double-buffered A in SLM, manual k-pair unrolling
2. **Best so far (1.160ms)** - Same tile but cleaner loop structure with single k-step
3. **Last tested (1.180ms)** - K-step=32 (two k16 per SLM load), but slower due to larger SLM footprint and more complex indexing

Key bottlenecks:
- **B matrix bandwidth**: Each subgroup independently loads B from global memory. With 16 subgroups loading different columns, there's no reuse of B.
- **Low compute density**: 4 DPAS per subgroup per k-step = 4 DPAS × 16 SGs = 64 DPAS per WG per k-step, but each DPAS is 8×16×16 = 2048 FP16 ops. Total = 131K ops per k-step vs loading 32×16 (A) + 16×16×16 (B) = 8.5KB.
- **SLM is underutilized**: Only 1-2KB of 128KB SLM used.

**Key insight**: We should also cache B in SLM to get reuse across the M-dimension rows. With both A and B in SLM, we maximize data reuse.

**New strategy**: 
- Tile: 32M × 256N, but cache both A (32×16) and B (16×256) in SLM
- SLM usage: 32×16×2 + 16×256×2 = 1KB + 8KB = 9KB per buffer, 18KB double-buffered — fits well in 128KB
- This gives 32× reuse for B (across M rows) and 16× reuse for A (across N columns)
- Each WG does 32×256×16×2 = 262K FP16 ops per k-step while loading only 9KB

Also improving: use `intel_sub_group_block_write_us` for stores and better prefetching.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32M x 256N, K-step=16
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16)
// Both A and B cached in SLM for maximum reuse:
//   A: 32x16 = 512 ushorts = 1KB (reused 16x across N)
//   B: 16x256 = 4096 ushorts = 8KB (reused 32x across M via DPAS)
//   Double-buffered: 2 * (512 + 4096) = 9216 ushorts = ~18KB << 128KB SLM
// GWS = (ceil(N/256)*16, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16

#define TILE_M 32
#define TILE_N 256
#define TILE_K 16
#define SLM_A_SIZE (TILE_M * TILE_K)   // 512 ushorts
#define SLM_B_SIZE (TILE_K * TILE_N)   // 4096 ushorts
#define SLM_BUF_SIZE (SLM_A_SIZE + SLM_B_SIZE)  // 4608 ushorts per buffer
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

    // Double-buffered SLM for A and B
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

    // Helper macro for SLM buffer offsets
    #define SLM_A(buf) (slm + (buf) * SLM_BUF_SIZE)
    #define SLM_B(buf) (slm + (buf) * SLM_BUF_SIZE + SLM_A_SIZE)

    // Load first tile (k=0) into buffer 0
    // A: 512 elements, B: 4096 elements, total 4608
    // 4608 / 256 = 18 elements per WI
    {
        __local ushort* dst_a = SLM_A(0);
        __local ushort* dst_b = SLM_B(0);

        // Load A: 512 elements = 2 per WI
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid >> 4;   // eid / 16
            int ac = eid & 15;   // eid % 16
            int gr = row_base + ar;
            dst_a[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }

        // Load B: 4096 elements = 16 per WI
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int eid = lid + i * WG_SIZE;
            int br = eid / TILE_N;  // row in [0,16)
            int bc = eid % TILE_N;  // col in [0,256)
            int gk = br;
            int gc = n_base + bc;
            dst_b[eid] = (gk < K && gc < N) ? B_us[gk * N + gc] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int cur_buf = kt & 1;

        __local const ushort* ca = SLM_A(cur_buf);
        __local const ushort* cb = SLM_B(cur_buf) + sg_id * 16;  // each SG reads its 16-col slice

        // Read A from SLM into registers
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

        // Read B from SLM: 16 rows x 16 cols -> int8 (pairs of ushort packed)
        int8 b_val;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            ushort s0 = intel_sub_group_block_read_us(cb + (2 * p) * TILE_N);
            ushort s1 = intel_sub_group_block_read_us(cb + (2 * p + 1) * TILE_N);
            ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
        }

        // Start loading next tile BEFORE compute to overlap with DPAS
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - cur_buf;
            const int next_k = (kt + 1) * TILE_K;
            __local ushort* dst_a = SLM_A(next_buf);
            __local ushort* dst_b = SLM_B(next_buf);

            // Load A: 512 elements = 2 per WI
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid >> 4;
                int ac = eid & 15;
                int gr = row_base + ar;
                int gk = next_k + ac;
                dst_a[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }

            // Load B: 4096 elements = 16 per WI
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * WG_SIZE;
                int br = eid / TILE_N;
                int bc = eid % TILE_N;
                int gk = next_k + br;
                int gc = n_base + bc;
                dst_b[eid] = (gk < K && gc < N) ? B_us[gk * N + gc] : (ushort)0;
            }
        }

        // DPAS compute - 4 DPAS instructions per subgroup
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
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

    #undef SLM_A
    #undef SLM_B
}
```

