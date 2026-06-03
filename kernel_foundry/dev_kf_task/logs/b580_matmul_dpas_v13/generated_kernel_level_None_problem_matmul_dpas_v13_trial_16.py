

## Analysis

Looking at the results:
- **Version 1 (32M×256N)**: 1.160ms - good baseline
- **Best (32M×256N, K-pair unroll)**: 1.140ms - marginal improvement from K-unrolling
- **Previous (64M×128N)**: 3.370ms - **much worse** despite higher compute/memory ratio

The 64M×128N version failed because:
1. **Too many registers**: 8 acc registers (float8 each = 64 floats) + 8 A registers causes register spilling
2. **Too much SLM loading**: 16 iterations per WI to load A tile (64×32 = 2048 elements / 128 WIs)
3. **Only 8 subgroups**: less B reuse, each SG covers 16 cols, total 128N but with 64M rows requiring 8 DPAS calls per k-step

The 32M×256N config works better because:
- 4 acc registers = 32 floats, manageable register pressure
- 16 subgroups share A = high reuse
- Only 2 iterations to load A tile (32×16 = 512 / 256 WIs)

**Key improvements for this version:**
1. **Both A and B in SLM**: B is loaded 4 times redundantly (4 DPAS rows share same B). Put B in SLM too for reuse across the 4 row-groups within each subgroup.
2. **Better prefetching**: Overlap global loads with compute more effectively
3. **Use block reads for stores**: Use `intel_sub_group_block_write_us` for output
4. **Increase K-step to 32**: Load more K per SLM fill to amortize barrier cost

Let me try a refined 32M×256N with K=16, focusing on cleaner double-buffering and reducing redundant barriers.

Actually, the main bottleneck at 23% XMX utilization is likely memory-bound on B loads. Each subgroup loads its own 16×16 B block from global memory. With 16 subgroups, that's 16×16×16×2 = 8KB of B per k-step, but A is only 32×16×2 = 1KB shared. The compute is 32×256×16×2 = 262K FLOPs per k-step vs 9KB loaded = ~29 FLOPs/byte, which should be compute-bound at 96T/456G = 210 FLOPs/byte threshold. So we're compute-bound and the issue is DPAS pipeline stalls.

The real issue: we do 4 DPAS per subgroup per k-step, but we have barriers and SLM reads interleaved. Let me try **K=32 with 32M×256N** to double the DPAS work between barriers, keeping register count manageable by reading A in two k16 chunks from SLM.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Optimized GEMM: C[M,N] = A[M,K] * B[K,N], all half, acc in float
// Tile: 32M x 256N, K-step=32 (two k16 DPAS rounds per SLM load)
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16) per k16 round, 8 per k32
// A in SLM double-buffered: 2 * 32*32 * 2 bytes = 4 KB
// B loaded from global per SG (16x32 = 512 ushorts = 1KB per SG)
// GWS = (ceil(N/256)*256, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16
// Compute per k32: 16 SG * 8 DPAS * 8*16*16 = 2M FLOPs (doubled vs k16)
// Barriers halved vs k16 approach

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

    // Double-buffered SLM for A: 2 * 32 * 32 = 2048 ushorts = 4KB total
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
    // 32*32 = 1024 elements, 1024/256 = 4 per WI
    {
        __local ushort* dst = slm_a;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;   // row [0..31]
            int ac = eid % TILE_K;   // col [0..31]
            int gr = row_base + ar;
            dst[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int cur_buf = kt & 1;
        __local const ushort* ca = slm_a + cur_buf * SLM_A_SIZE;

        // ======== K16 round 0: columns [0..15] of the 32-wide A tile ========
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

        // ======== K16 round 1: columns [16..31] of the 32-wide A tile ========
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

        // Start prefetching next A tile before DPAS round 1
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - cur_buf;
            const int next_k = (kt + 1) * TILE_K;
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
        }

        // DPAS round 1 overlaps with SLM writes on Xe2
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

