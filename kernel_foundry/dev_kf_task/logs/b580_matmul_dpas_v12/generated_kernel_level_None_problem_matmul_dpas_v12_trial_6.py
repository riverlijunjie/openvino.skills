

## Analysis

Looking at the results:
- **Best version (0.959ms)**: Double-buffered SLM, overlapping A load with compute
- **Version 1 (0.968ms)**: K-loop unrolled 2x with `#pragma unroll 2`
- **Last tested (1.010ms)**: Added B block reads with branching - the branch overhead and block read approach for B actually hurt performance

Key observations:
1. Double-buffering gave the best result by hiding A load latency behind compute
2. The B block read attempt was slower - the branching and `intel_sub_group_block_read_us` for global B didn't help
3. The main bottleneck is likely B memory access (16 scalar reads per k16 step, 32 per k32 tile)

**Proposed improvements over the best (0.959ms) version:**
1. **Tighter B loading**: Pack B reads more efficiently - precompute the base offset and use simpler pointer arithmetic
2. **Remove boundary checks where possible**: For the common case (M=2048 divisible by 32, N=2048 divisible by 64), skip all boundary checks entirely
3. **Reduce register pressure in A loads**: Use `intel_sub_group_block_read_us8` for 8-wide SLM reads instead of 8 individual reads
4. **Streamline double-buffer logic**: Eliminate the `has_next` branch inside the hot loop by peeling the last iteration

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
// Double-buffering with last-iteration peeling to eliminate branch in hot loop.
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
    const int num_k_tiles = K >> 5;

    const int b_col = col_valid ? col_idx : (N - 1);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Precompute per-WI A-load info
    int a_local_off[16];
    int a_row_off[16];  // r * K (precomputed)
    int a_c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int elem_id = lid + i * 64;
        int r = elem_id >> 5;
        int c = elem_id & 31;
        a_local_off[i] = elem_id;
        a_row_off[i] = r * K;
        a_c[i] = c;
    }

    // Precompute B base pointer for this subgroup's column
    __global const ushort* B_col = B_us + b_col;

    // Load first A tile into buffer 0
    {
        __local ushort* dst = slm_A;
        if (row_tile_valid) {
            __global const ushort* A_base = A_us + a_row_base_K;
            #pragma unroll
            for (int i = 0; i < 16; i++)
                dst[a_local_off[i]] = A_base[a_row_off[i] + a_c[i]];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int r = a_local_off[i] >> 5;
                int gr = row_base + r;
                dst[a_local_off[i]] = (gr < M) ? A_us[gr * K + a_c[i]] : (ushort)0;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    // Main loop: all iterations except the last, with double-buffered A prefetch
    // Peel the last iteration to avoid the has_next branch
    const int last_kt = num_k_tiles - 1;

    for (int kt = 0; kt < last_kt; kt++) {
        const int k = kt * TILE_K;
        __local const ushort* cur_slm = slm_A + cur_buf * SLM_TILE;
        __local ushort* next_slm = slm_A + (1 - cur_buf) * SLM_TILE;

        // ---- COMPUTE k16 step 0 ----
        {
            short8 a0, a1, a2, a3;
            __local const ushort* sb = cur_slm;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(sb + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(sb + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(sb + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(sb + (24 + r) * TILE_K);

            __global const ushort* bp = B_col + k * N;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = bp[0];
                ushort s1 = bp[N];
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                bp += N2;
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ---- COMPUTE k16 step 1 ----
        {
            short8 a0, a1, a2, a3;
            __local const ushort* sb = cur_slm + 16;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(sb + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(sb + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(sb + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(sb + (24 + r) * TILE_K);

            __global const ushort* bp = B_col + (k + 16) * N;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = bp[0];
                ushort s1 = bp[N];
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                bp += N2;
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ---- LOAD next A tile ----
        barrier(CLK_LOCAL_MEM_FENCE);

        const int next_k = (kt + 1) * TILE_K;
        if (row_tile_valid) {
            __global const ushort* A_base = A_us + a_row_base_K + next_k;
            #pragma unroll
            for (int i = 0; i < 16; i++)
                next_slm[a_local_off[i]] = A_base[a_row_off[i] + a_c[i]];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int r = a_local_off[i] >> 5;
                int gr = row_base + r;
                next_slm[a_local_off[i]] = (gr < M) ? A_us[gr * K + next_k + a_c[i]] : (ushort)0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        cur_buf = 1 - cur_buf;
    }

    // ---- LAST K-TILE: compute only, no next load ----
    {
        const int k = last_kt * TILE_K;
        __local const ushort* cur_slm = slm_A + cur_buf * SLM_TILE;

        // k16 step 0
        {
            short8 a0, a1, a2, a3;
            __local const ushort* sb = cur_slm;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(sb + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(sb + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(sb + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(sb + (24 + r) * TILE_K);

            __global const ushort* bp = B_col + k * N;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = bp[0];
                ushort s1 = bp[N];
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                bp += N2;
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // k16 step 1
        {
            short8 a0, a1, a2, a3;
            __local const ushort* sb = cur_slm + 16;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(sb + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(sb + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(sb + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(sb + (24 + r) * TILE_K);

            __global const ushort* bp = B_col + (k + 16) * N;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = bp[0];
                ushort s1 = bp[N];
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                bp += N2;
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
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

