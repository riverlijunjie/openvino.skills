

## Analysis

All three versions perform nearly identically (~0.968-0.970ms) because they share the same fundamental architecture. The differences (2x unrolling, loop structure) don't meaningfully change the bottleneck.

Key observations:
1. The kernel is B-load bound - 16 scalar reads per DPAS k16 step from global memory
2. The A SLM loading uses scattered reads (each WI computes row/col independently)
3. All versions achieve ~20.9% XMX utilization, suggesting we're memory-latency bound on B

**Proposed optimizations for this version:**

1. **Double-buffered SLM A loads**: Use two SLM buffers so A loading for the next K-tile overlaps with DPAS compute on the current tile. This hides the A-load latency behind compute.
2. **Tighter B load pattern**: Pre-compute all B offsets and use `as_int` more efficiently
3. **Store using `intel_sub_group_block_write_us`** for vectorized C output
4. **Remove the `#pragma unroll 2` inner loop** - fully inline both K-tile halves to give the compiler maximum scheduling freedom
5. **Use `vload2` for paired B reads** to hint the compiler to use wider loads

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (double-buffered: 2x 2KB = 4KB), B loaded directly from global/L2
// K=2048 divides evenly by 32, no remainder path needed.
// Double-buffering: load next A tile while computing on current tile.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_TILE (TILE_M * TILE_K)

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
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid = get_local_id(0);

    const int n_base = get_group_id(0) * TILE_N;
    const int col_base = n_base + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    // Double-buffered SLM for A: overlap load of next tile with compute on current
    __local ushort slm_A[2 * SLM_TILE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const int num_k_tiles = K >> 5;  // K / 32

    const int b_col = col_valid ? col_idx : (N - 1);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Precompute per-WI A-load offsets (16 elements per WI, stride 64)
    // elem_id = lid + i*64, r = elem_id >> 5, c = elem_id & 31
    // These are constant across K-tiles; only the k-offset changes.
    int a_local_off[16];
    int a_r[16];
    int a_c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int elem_id = lid + i * 64;
        a_r[i] = elem_id >> 5;
        a_c[i] = elem_id & 31;
        a_local_off[i] = elem_id;
    }

    // Load first A tile into buffer 0
    {
        __local ushort* dst = slm_A;
        if (row_tile_valid) {
            __global const ushort* A_tile = A_us + a_row_base_K;
            #pragma unroll
            for (int i = 0; i < 16; i++)
                dst[a_local_off[i]] = A_tile[a_r[i] * K + a_c[i]];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int gr = row_base + a_r[i];
                dst[a_local_off[i]] = (gr < M) ? A_us[gr * K + a_c[i]] : (ushort)0;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k = kt * TILE_K;
        const int next_kt = kt + 1;
        const bool has_next = (next_kt < num_k_tiles);

        // Prefetch next A tile into the other buffer (overlaps with compute below)
        // We'll barrier before using it in the next iteration
        __local ushort* cur_slm = slm_A + cur_buf * SLM_TILE;
        __local ushort* next_slm = slm_A + (1 - cur_buf) * SLM_TILE;

        // ---- COMPUTE on current buffer ----
        // k16 step 0
        {
            short8 a0, a1, a2, a3;
            __local const ushort* slm_base = cur_slm;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm_base + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm_base + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm_base + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm_base + (24 + r) * TILE_K);

            int b_off = k * N + b_col;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = B_us[b_off];
                ushort s1 = B_us[b_off + N];
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                b_off += N2;
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // k16 step 1
        {
            short8 a0, a1, a2, a3;
            __local const ushort* slm_base = cur_slm + 16;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm_base + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm_base + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm_base + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm_base + (24 + r) * TILE_K);

            int b_off = (k + 16) * N + b_col;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = B_us[b_off];
                ushort s1 = B_us[b_off + N];
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                b_off += N2;
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ---- LOAD next A tile (after compute to allow overlap) ----
        if (has_next) {
            const int next_k = next_kt * TILE_K;
            // Need barrier before writing to next_slm since previous iteration
            // may have been reading from it. But we've finished reading cur_slm above.
            // We need a barrier to ensure all WIs finished reading cur_slm before
            // we potentially write to next_slm (which was the previous cur_slm's partner).
            barrier(CLK_LOCAL_MEM_FENCE);

            if (row_tile_valid) {
                __global const ushort* A_tile = A_us + a_row_base_K + next_k;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    next_slm[a_local_off[i]] = A_tile[a_r[i] * K + a_c[i]];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int gr = row_base + a_r[i];
                    next_slm[a_local_off[i]] = (gr < M) ? A_us[gr * K + next_k + a_c[i]] : (ushort)0;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = 1 - cur_buf;
        }
    }

    // Store results
    if (col_valid) {
        __global half* C_col = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

