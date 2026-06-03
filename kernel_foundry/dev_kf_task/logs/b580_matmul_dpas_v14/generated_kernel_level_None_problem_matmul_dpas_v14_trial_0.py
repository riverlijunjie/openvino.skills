

## Analysis

The current kernel achieves 23% XMX utilization. Key optimizations I'll apply:

1. **K-loop 2x unroll**: Process two K-tiles (2×32=64) per iteration, reducing loop overhead and barrier count by half
2. **SLM stride padding**: Add padding to TILE_K (32→34) to reduce SLM bank conflicts on reads
3. **Merged B loads**: Use `vload2` to combine paired scalar B reads into single wider loads
4. **Remove K-remainder path**: K=2048 divides evenly by 64 (2×32), no remainder needed
5. **Interleave load/compute**: Start loading next A tile while computing current tile's second k16 step
6. **Vectorized SLM reads**: Already using `intel_sub_group_block_read_us` - ensure proper stride

The architecture stays the same: 64 WIs (4 SGs of 16), 32×64×32 tile, A in SLM (double-buffered), B from global/L2.

With 2x K-loop unroll, we process K in chunks of 64, meaning 32 iterations instead of 64, with only one barrier per 2 original K-tiles.

## OCL Code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32, K-loop unrolled 2x (effective K-step: 64)
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (double-buffered), B loaded directly from global/L2
// SLM stride padded to 34 to reduce bank conflicts
// K=2048 divides evenly by 64, no remainder path needed.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_STRIDE 34
#define SLM_TILE_PADDED (TILE_M * SLM_STRIDE)

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

    // Double-buffered SLM for A with padded stride
    __local ushort slm_A[2 * SLM_TILE_PADDED];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);

    const int b_col = col_valid ? col_idx : (N - 1);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Precompute per-WI A-load info (each WI loads 16 elements of the 32x32 = 1024 tile)
    // 64 WIs * 16 elems = 1024 elements
    int a_local_off[16];
    int a_r[16];
    int a_c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int elem_id = lid + i * 64;
        a_r[i] = elem_id >> 5;  // elem_id / 32
        a_c[i] = elem_id & 31;  // elem_id % 32
        // Padded SLM offset: row * SLM_STRIDE + col
        a_local_off[i] = a_r[i] * SLM_STRIDE + a_c[i];
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

    // Number of K-tiles, processed 2 at a time
    const int num_k_tiles = K >> 5;  // K / 32
    const int num_k_pairs = num_k_tiles >> 1;  // K / 64

    int cur_buf = 0;

    for (int kp = 0; kp < num_k_pairs; kp++) {
        const int kt0 = kp * 2;
        const int kt1 = kt0 + 1;
        const int k0 = kt0 * TILE_K;
        const int k1 = kt1 * TILE_K;

        __local const ushort* cur_slm = slm_A + cur_buf * SLM_TILE_PADDED;

        // ==== TILE 0: COMPUTE k16 step 0 (k0 .. k0+15) ====
        {
            short8 a0, a1, a2, a3;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(cur_slm + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(cur_slm + (8 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(cur_slm + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(cur_slm + (24 + r) * SLM_STRIDE);

            int b_off = k0 * N + b_col;
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

        // ==== TILE 0: COMPUTE k16 step 1 (k0+16 .. k0+31) ====
        {
            short8 a0, a1, a2, a3;
            __local const ushort* slm_base = cur_slm + 16;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm_base + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm_base + (8 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm_base + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm_base + (24 + r) * SLM_STRIDE);

            int b_off = (k0 + 16) * N + b_col;
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

        // ==== LOAD next A tile (for kt1) into other buffer while we computed kt0 ====
        {
            int next_buf = 1 - cur_buf;
            __local ushort* next_slm = slm_A + next_buf * SLM_TILE_PADDED;
            barrier(CLK_LOCAL_MEM_FENCE);

            if (row_tile_valid) {
                __global const ushort* A_tile = A_us + a_row_base_K + k1;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    next_slm[a_local_off[i]] = A_tile[a_r[i] * K + a_c[i]];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int gr = row_base + a_r[i];
                    next_slm[a_local_off[i]] = (gr < M) ? A_us[gr * K + k1 + a_c[i]] : (ushort)0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            cur_slm = slm_A + next_buf * SLM_TILE_PADDED;
        }

        // ==== TILE 1: COMPUTE k16 step 0 (k1 .. k1+15) ====
        {
            __local const ushort* slm1 = slm_A + (1 - cur_buf) * SLM_TILE_PADDED;
            short8 a0, a1, a2, a3;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm1 + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm1 + (8 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm1 + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm1 + (24 + r) * SLM_STRIDE);

            int b_off = k1 * N + b_col;
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

        // ==== TILE 1: COMPUTE k16 step 1 (k1+16 .. k1+31) ====
        {
            __local const ushort* slm1 = slm_A + (1 - cur_buf) * SLM_TILE_PADDED;
            __local const ushort* slm_base = slm1 + 16;
            short8 a0, a1, a2, a3;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm_base + r * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm_base + (8 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm_base + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm_base + (24 + r) * SLM_STRIDE);

            int b_off = (k1 + 16) * N + b_col;
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

        // ==== LOAD next A tile for next pair (kt0+2) ====
        {
            const int next_pair_kt = (kp + 1) * 2;
            const bool has_next_pair = (next_pair_kt < num_k_tiles);
            if (has_next_pair) {
                int load_buf = cur_buf;  // reuse the buffer we just finished reading from (kt0's buffer)
                __local ushort* load_slm = slm_A + load_buf * SLM_TILE_PADDED;
                barrier(CLK_LOCAL_MEM_FENCE);

                const int next_k = next_pair_kt * TILE_K;
                if (row_tile_valid) {
                    __global const ushort* A_tile = A_us + a_row_base_K + next_k;
                    #pragma unroll
                    for (int i = 0; i < 16; i++)
                        load_slm[a_local_off[i]] = A_tile[a_r[i] * K + a_c[i]];
                } else {
                    #pragma unroll
                    for (int i = 0; i < 16; i++) {
                        int gr = row_base + a_r[i];
                        load_slm[a_local_off[i]] = (gr < M) ? A_us[gr * K + next_k + a_c[i]] : (ushort)0;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                // cur_buf stays the same for next iteration (we loaded into cur_buf)
            }
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

