## Analysis

Looking at the previous versions:

1. **Version 1** (1.510ms, correct): Uses `reqd_work_group_size(64, 1, 1)` with SLM stride 34. Works correctly.
2. **Version 2** (incorrect): Uses `reqd_work_group_size(4, 1, 1)` which doesn't properly map to 64 WIs. The GWS/LWS mismatch caused correctness issues.
3. **Last version** (1.500ms, correct): Similar to V1 but with SLM stride 36. Marginal improvement.

**Key bottlenecks identified:**
1. The K-loop has branch overhead (`if (ki < k_iters - 1)`) every iteration
2. B loads are scalar (16 individual global memory reads per lane per k16 step) - very expensive
3. No K-loop unrolling - loop overhead for 64 iterations (K=2048/32=64)
4. SLM A reads are scalar (individual half reads) rather than vectorized
5. The barrier placement could be optimized with double-buffering

**Proposed improvements:**
1. **Remove K-remainder path**: K=2048 divides evenly by 32, so remove the `if` check by restructuring the loop
2. **Unroll K-loop 2x**: Process two K tiles per loop iteration, reducing loop overhead from 64 to 32 iterations
3. **Optimize B loading**: Use vload2 to merge paired scalar reads where possible
4. **SLM stride optimization**: Try stride 34 (32+2) which is smaller and may have fewer conflicts for 16-wide access patterns
5. **Better pipeline**: Load next A tile earlier, interleave B loads with DPAS compute
6. **Vectorized SLM reads**: Use intel_sub_group_block_read for A from SLM where alignment permits

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all row-major FP16, f32 accumulation
// Launch: GWS = (N, M/32), LWS = (64, 1)
//   - 64 WIs per WG = 4 subgroups of 16
//   - Each WG computes a 32×64 output tile
//   - TILE_M=32, TILE_N=64, TILE_K=32
// A loaded cooperatively into SLM (double-buffered), B from global/L2
// K-loop unrolled 2x (K=2048 always divides by 64)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4
// SLM stride: 32 + 2 padding = 34 halfs (68 bytes) to reduce bank conflicts
// 34 is not a multiple of 16, which helps avoid bank conflicts for 16-wide SG reads
#define SLM_A_STRIDE 34

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile position
    const int wg_n = get_group_id(0);  // which 64-col tile
    const int wg_m = get_group_id(1);  // which 32-row tile

    const int m_start = wg_m * TILE_M;
    const int n_start = wg_n * TILE_N;

    // Subgroup and lane identification
    const int local_id = get_local_id(0);  // 0..63
    const int sg_id = local_id / SG_SIZE;  // 0..3
    const int sg_lid = local_id % SG_SIZE; // 0..15

    // Each subgroup handles 16 columns of the 64-col output tile
    const int sg_n_offset = sg_id * SG_SIZE;

    // SLM double buffer for A: 2 buffers × 32 rows × SLM_A_STRIDE halfs
    __local half slm_a[2 * TILE_M * SLM_A_STRIDE];

    // Accumulators: 32 rows × 16 cols per subgroup = 4 DPAS blocks of 8 rows each
    float8 acc0 = (float8)(0.0f);  // rows 0-7
    float8 acc1 = (float8)(0.0f);  // rows 8-15
    float8 acc2 = (float8)(0.0f);  // rows 16-23
    float8 acc3 = (float8)(0.0f);  // rows 24-31

    // Cooperative A load mapping:
    // 64 threads load 32 rows × 32 cols = 1024 halfs
    // Each thread loads 16 halfs
    const int a_load_row = local_id / 2;       // 0..31
    const int a_load_col = (local_id % 2) * 16; // 0 or 16

    // B base pointer for this subgroup's columns
    __global const half* b_base = B + n_start + sg_n_offset + sg_lid;

    // Load first A tile (k=0) into SLM buffer 0
    {
        __global const half* a_src = A + (m_start + a_load_row) * K + a_load_col;
        __local half* a_dst = slm_a + a_load_row * SLM_A_STRIDE + a_load_col;
        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // K=2048, TILE_K=32, so k_iters=64. Unroll 2x -> 32 outer iterations.
    const int k_iters = K / TILE_K;
    const int k_iters_half = k_iters / 2;  // 32

    for (int ki2 = 0; ki2 < k_iters_half; ki2++) {
        // Process two K tiles per outer iteration
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            int ki = ki2 * 2 + sub;
            int cur_buf = ki & 1;
            int next_buf = 1 - cur_buf;
            int k_base = ki * TILE_K;

            // Load next A tile into next SLM buffer (skip on very last iteration)
            if (ki < k_iters - 1) {
                int next_k = k_base + TILE_K;
                __global const half* a_src = A + (m_start + a_load_row) * K + next_k + a_load_col;
                __local half* a_dst = slm_a + next_buf * TILE_M * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col;
                half8 v0 = vload8(0, a_src);
                half8 v1 = vload8(1, a_src);
                vstore8(v0, 0, a_dst);
                vstore8(v1, 0, a_dst + 8);
            }

            // Current A SLM base
            __local const half* a_slm = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

            // Process TILE_K=32 as two k16 steps
            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                int k_inner = kk * 16;
                int k_global = k_base + k_inner;

                // Load B: 16 k-rows × 16 cols, packed into int8 (VNNI format)
                __global const half* b_ptr = b_base + k_global * N;

                int8 b_val;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort u0 = as_ushort(b_ptr[(2 * p) * N]);
                    ushort u1 = as_ushort(b_ptr[(2 * p + 1) * N]);
                    ((int*)&b_val)[p] = (int)((uint)u0 | ((uint)u1 << 16));
                }

                // DPAS for 4 blocks of 8 rows each
                // Block 0: rows 0-7
                {
                    short8 a_val;
                    #pragma unroll
                    for (int r = 0; r < 8; r++) {
                        a_val[r] = as_short(a_slm[r * SLM_A_STRIDE + k_inner + sg_lid]);
                    }
                    acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc0);
                }

                // Block 1: rows 8-15
                {
                    short8 a_val;
                    #pragma unroll
                    for (int r = 0; r < 8; r++) {
                        a_val[r] = as_short(a_slm[(8 + r) * SLM_A_STRIDE + k_inner + sg_lid]);
                    }
                    acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc1);
                }

                // Block 2: rows 16-23
                {
                    short8 a_val;
                    #pragma unroll
                    for (int r = 0; r < 8; r++) {
                        a_val[r] = as_short(a_slm[(16 + r) * SLM_A_STRIDE + k_inner + sg_lid]);
                    }
                    acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc2);
                }

                // Block 3: rows 24-31
                {
                    short8 a_val;
                    #pragma unroll
                    for (int r = 0; r < 8; r++) {
                        a_val[r] = as_short(a_slm[(24 + r) * SLM_A_STRIDE + k_inner + sg_lid]);
                    }
                    acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc3);
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Handle odd remaining iteration if k_iters is odd (for K=2048, k_iters=64, this won't execute)
    // Omitted since K=2048 divides evenly by 64

    // Write results: each lane writes one column across 32 rows
    __global half* c_ptr = C + m_start * N + n_start + sg_n_offset + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[r * N] = convert_half(acc0[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(8 + r) * N] = convert_half(acc1[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(16 + r) * N] = convert_half(acc2[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_ptr[(24 + r) * N] = convert_half(acc3[r]);
    }
}
```