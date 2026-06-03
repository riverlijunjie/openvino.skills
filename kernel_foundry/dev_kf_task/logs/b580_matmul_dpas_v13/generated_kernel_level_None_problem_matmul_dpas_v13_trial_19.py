

## Analysis

Looking at the previous versions (all ~1.14-1.16ms), the key bottlenecks are:

1. **Too many barriers per K-iteration** - Version 3 has 2+ barriers per K-pair, stalling the pipeline
2. **B matrix bandwidth** - Each subgroup independently loads B from global memory with no reuse
3. **Insufficient compute density** - 4 DPAS ops per K16 step with 16 subgroups = 64 DPAS per barrier, but we can do better
4. **SLM underutilized** - Only A is in SLM (1-2KB), while B580 has 128KB SLM per core

**Key insight**: We should also tile B into SLM to get reuse across the M-dimension. With a 32M x 64N x 32K tile, both A (32×32=2KB) and B (32×64=4KB) fit in SLM with double buffering (12KB total), and B gets 4x reuse (32 rows / 8 DPAS rows).

**New strategy**: 
- **Tile: 32M × 64N × 32K** per workgroup
- **4 subgroups × 16 WIs = 64 WIs** per WG
- Each subgroup handles 32 rows × 16 cols = 4 DPAS per k16
- Both A and B in SLM, double-buffered
- More workgroups → better occupancy across 20 Xe-cores
- K-step=32 means 2 DPAS rounds per SLM load → fewer barriers

Actually, let me reconsider. The problem with smaller N-tiles is less B reuse. Let me try a different approach:

**Better strategy: 64M × 128N tile, 8 subgroups**
- 8 SGs × 16 WIs = 128 WIs per WG
- Each SG: 8 rows × 16 cols, but we give each SG multiple 8×16 blocks
- Arrange: 4 SGs along N (4×16=64), 2 SGs along M... 

Let me stick with something proven but optimize the pipelining. The real issue is the barrier-to-compute ratio. Let me use **cooperative B loading into SLM** with a larger K-step.

**Final approach: 32M × 128N × 32K, 8 subgroups, both A+B in SLM**
- A: 32×32 = 2KB, B: 32×128 = 8KB, double-buffered = 20KB total
- 8 SGs along N dimension, each 32rows × 16cols = 4 DPAS per k16, 8 per k32
- Total: 64 DPAS per iteration with only 2 barriers
- B gets 4x reuse across M-blocks (32/8=4)
- 128 WIs cooperatively load both A and B

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Optimized GEMM: C[M,N] = A[M,K] * B[K,N], all half, acc in float
// Tile: 32M x 128N, K-step=32 (two k16 DPAS rounds per SLM load)
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16) per k16, 8 per k32
// Both A and B in SLM, double-buffered:
//   A: 2 * 32*32 * 2B = 4KB
//   B: 2 * 32*128 * 2B = 16KB
//   Total SLM: 20KB (well within 128KB limit)
// B gets 4x reuse across the 32 M-rows (each DPAS uses 8 rows, 4 blocks)
// GWS = (ceil(N/128)*128, ceil(M/32))  LWS = (128, 1)
// Subgroup size = 16

#define TILE_M 32
#define TILE_N 128
#define TILE_K 32
#define K16 16
#define SLM_A_SIZE (TILE_M * TILE_K)   // 1024 ushorts per buffer
#define SLM_B_SIZE (TILE_K * TILE_N)   // 4096 ushorts per buffer
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
    // Each of 8 subgroups handles a 16-col strip
    const int col_base = n_base + sg_id * 16;

    // Double-buffered SLM for A and B
    // Layout: slm_a[2][32][32], slm_b[2][32][128] stored as ushort
    // But B is stored transposed in SLM for better DPAS access:
    // Actually, for DPAS b argument (int8), we need pairs of rows packed.
    // Store B in SLM row-major: B_slm[k][n] where k in [0,32), n in [0,128)
    __local ushort slm_a[2 * SLM_A_SIZE];
    __local ushort slm_b[2 * SLM_B_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;  // rows [0..7]
    float8 acc1 = 0.0f;  // rows [8..15]
    float8 acc2 = 0.0f;  // rows [16..23]
    float8 acc3 = 0.0f;  // rows [24..31]

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Cooperative load: A is 32*32=1024 elements, B is 32*128=4096 elements
    // Total = 5120 elements, 5120/128 = 40 elements per WI

    // Load first tile into buffer 0
    {
        // Load A: 1024 elems, 1024/128 = 8 per WI
        __local ushort* dst_a = slm_a;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;  // row [0..31]
            int ac = eid % TILE_K;  // col [0..31]
            int gr = row_base + ar;
            dst_a[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }
        // Load B: 4096 elems, 4096/128 = 32 per WI
        __local ushort* dst_b = slm_b;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int eid = lid + i * WG_SIZE;
            int bk = eid / TILE_N;  // k-row [0..31]
            int bn = eid % TILE_N;  // n-col [0..127]
            int gn = n_base + bn;
            dst_b[eid] = (bk < K && gn < N) ? B_us[bk * N + gn] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int cur_buf = kt & 1;
        __local const ushort* ca = slm_a + cur_buf * SLM_A_SIZE;
        __local const ushort* cb = slm_b + cur_buf * SLM_B_SIZE;

        // ======== K16 round 0: k columns [0..15] ========
        // Read A from SLM: 4 blocks of 8 rows, each row reads 16 elements via subgroup
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

        // Read B from SLM for this subgroup's 16-col strip, k16 round 0
        // B_slm layout: [k][n], stride = TILE_N=128
        // For DPAS: need int8 b where each int packs 2 consecutive k-rows for same n
        int8 b_val0;
        {
            __local const ushort* bp = cb + sg_id * 16;  // offset to this SG's columns
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(bp + (2 * p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(bp + (2 * p + 1) * TILE_N);
                ((int*)&b_val0)[p] = as_int((ushort2)(s0, s1));
            }
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val0, acc3);

        // ======== K16 round 1: k columns [16..31] ========
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
        {
            __local const ushort* bp = cb + K16 * TILE_N + sg_id * 16;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(bp + (2 * p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(bp + (2 * p + 1) * TILE_N);
                ((int*)&b_val1)[p] = as_int((ushort2)(s0, s1));
            }
        }

        // Prefetch next tile into alternate buffer while computing DPAS round 1
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - cur_buf;
            const int next_k = (kt + 1) * TILE_K;

            __local ushort* dst_a = slm_a + next_buf * SLM_A_SIZE;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = row_base + ar;
                int gk = next_k + ac;
                dst_a[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }

            __local ushort* dst_b = slm_b + next_buf * SLM_B_SIZE;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int eid = lid + i * WG_SIZE;
                int bk = eid / TILE_N;
                int bn = eid % TILE_N;
                int gn = n_base + bn;
                int gk = next_k + bk;
                dst_b[eid] = (gk < K && gn < N) ? B_us[gk * N + gn] : (ushort)0;
            }
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0b, b_val1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1b, b_val1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2b, b_val1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3b, b_val1, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
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

