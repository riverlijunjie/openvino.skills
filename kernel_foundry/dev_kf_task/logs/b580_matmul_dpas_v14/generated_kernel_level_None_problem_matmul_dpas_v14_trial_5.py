## Analysis

Looking at the three versions:

1. **Version 1 (1.140ms)** - Best performer. Clean double-buffered approach with 2 K-tiles per iteration. Interleaves A reads with B loads nicely. Only 2 barriers per K-pair (between tile 0 compute and tile 1 load, then between tile 1 compute and next tile 0 load).

2. **Version 2 (1.330ms)** - Worst. Pre-loads both buffers before the loop, then has 3 barriers per iteration (load buf0 + barrier, load buf1 + barrier). More barriers = more stalls.

3. **Version 3 (1.210ms)** - Middle. Has a bug in tile 1 compute (uses `slm1 = slm_A + (1 - cur_buf) * SLM_TILE_PADDED` which is wrong after reassigning `cur_slm`). Also has extra complexity with `cur_buf` tracking.

**Key insights for improvement:**
- Version 1's structure is best - minimize barriers
- Better interleaving of B loads with DPAS compute (start B load before all A reads complete)
- The A load pattern (16 scalar loads per WI) is fine but could try intel_sub_group_block_read for global A
- Simplify the loop: since K divides by 32, use single-buffered SLM with K-step=32 and only 2 barriers per tile (one before load, one after). This might be simpler and have less overhead than double-buffering with pairs.

Actually, let me reconsider. Version 1 processes 64 K-elements with 2 barriers (for the mid-pair load) + 2 more for next pair = 4 barriers per 64 K. A simple single-buffer approach would be 2 barriers per 32 K = 4 barriers per 64 K - same count but simpler code.

Let me focus on what made Version 1 fastest:
- Clean interleaving of A SLM reads with B global loads
- Minimal barrier overhead

**Improvements to try:**
1. Better interleaving: start B prefetch earlier, overlap with A SLM reads
2. Boustrophedon DPAS ordering (snake pattern) to improve register reuse
3. Remove the `row_tile_valid` branch from the hot loop (handle at boundaries only)
4. Use simpler single-buffer approach with tighter code

Let me go with Version 1's structure but with tighter interleaving and snake-pattern DPAS ordering.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (single buffer, barrier-protected), B from global/L2
// SLM stride padded to 34 to reduce bank conflicts
// K=2048 divides evenly by 32, no remainder path needed.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
// Optimizations vs V1:
//   - Single SLM buffer (saves SLM, same barrier count per K-step)
//   - Tighter interleaving: B loads overlapped with A SLM reads
//   - Snake-pattern (boustrophedon) DPAS ordering for register reuse
//   - Removed remainder path (K always divisible by 32)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_STRIDE 34
#define SLM_BUF_SIZE (TILE_M * SLM_STRIDE)

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
    const int row_base = get_group_id(1) * TILE_M;

    if (row_base >= M || n_base >= N)
        return;

    __local ushort slm_A[SLM_BUF_SIZE];

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int col_idx = n_base + sg_id * 16 + sg_lid;
    const bool col_valid = col_idx < N;
    const int b_col = col_valid ? col_idx : (N - 1);
    const bool row_tile_valid = (row_base + TILE_M <= M);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Precompute per-WI A-load mapping: 64 WIs load 1024 elements (32x32 tile)
    int a_slm_off[16], a_r[16], a_c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int eid = lid + i * 64;
        a_r[i] = eid >> 5;
        a_c[i] = eid & 31;
        a_slm_off[i] = a_r[i] * SLM_STRIDE + a_c[i];
    }

    const int num_k_tiles = K >> 5;  // K / 32

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_off = kt * TILE_K;

        // ==== LOAD A tile into SLM ====
        {
            __local ushort* dst = slm_A;
            if (row_tile_valid) {
                __global const ushort* src = A_us + a_row_base_K + k_off;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    dst[a_slm_off[i]] = src[a_r[i] * K + a_c[i]];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int gr = row_base + a_r[i];
                    dst[a_slm_off[i]] = (gr < M) ? A_us[gr * K + k_off + a_c[i]] : (ushort)0;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ==== k16 step 0: cols [0..15] ====
        // Interleave: read A rows 0-7 from SLM, start B load, read A rows 8-15, etc.
        {
            __local const ushort* s = slm_A;

            // Read A[0:8, 0:16] and start B[k_off:k_off+16] load simultaneously
            short8 a0, a1;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(s + r * SLM_STRIDE);

            // Start B load for k16 step 0
            int8 bv0;
            {
                int boff = k_off * N + b_col;
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    ushort s0 = B_us[boff];
                    ushort s1 = B_us[boff + N];
                    ((int*)&bv0)[p] = as_int((ushort2)(s0, s1));
                    boff += N2;
                }
            }

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(s + (8 + r) * SLM_STRIDE);

            // Finish B load
            {
                int boff = k_off * N + b_col + 4 * N2;
                #pragma unroll
                for (int p = 4; p < 8; p++) {
                    ushort s0 = B_us[boff];
                    ushort s1 = B_us[boff + N];
                    ((int*)&bv0)[p] = as_int((ushort2)(s0, s1));
                    boff += N2;
                }
            }

            short8 a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(s + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(s + (24 + r) * SLM_STRIDE);

            // DPAS forward order for step 0
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv0, acc3);

            // ==== k16 step 1: cols [16..31] ====
            __local const ushort* s16 = s + 16;

            short8 a0b, a1b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0b)[r] = intel_sub_group_block_read_us(s16 + r * SLM_STRIDE);

            int8 bv1;
            {
                int boff = (k_off + 16) * N + b_col;
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    ushort s0 = B_us[boff];
                    ushort s1 = B_us[boff + N];
                    ((int*)&bv1)[p] = as_int((ushort2)(s0, s1));
                    boff += N2;
                }
            }

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1b)[r] = intel_sub_group_block_read_us(s16 + (8 + r) * SLM_STRIDE);

            {
                int boff = (k_off + 16) * N + b_col + 4 * N2;
                #pragma unroll
                for (int p = 4; p < 8; p++) {
                    ushort s0 = B_us[boff];
                    ushort s1 = B_us[boff + N];
                    ((int*)&bv1)[p] = as_int((ushort2)(s0, s1));
                    boff += N2;
                }
            }

            short8 a2b, a3b;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2b)[r] = intel_sub_group_block_read_us(s16 + (16 + r) * SLM_STRIDE);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3b)[r] = intel_sub_group_block_read_us(s16 + (24 + r) * SLM_STRIDE);

            // DPAS reverse order (boustrophedon) for step 1
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3b, bv1, acc3);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2b, bv1, acc2);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1b, bv1, acc1);
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0b, bv1, acc0);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    if (col_valid) {
        __global half* C_out = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_out[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_out[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_out[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_out[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_out[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

