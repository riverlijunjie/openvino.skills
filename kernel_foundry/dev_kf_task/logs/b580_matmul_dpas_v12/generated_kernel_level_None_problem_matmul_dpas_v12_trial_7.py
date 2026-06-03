

## Analysis

Looking at the three versions:
- **V1** (0.968ms): K-loop unrolled 2x with `#pragma unroll 2` inner loop
- **V2** (0.969ms): K-loop manually unrolled 2x (explicit first/second tile)
- **V3** (0.959ms, best): Double-buffered SLM - overlaps A load with compute

V3 is best because double-buffering hides SLM load latency behind DPAS compute. However, it uses 4KB SLM (2x buffers) and has extra complexity with `has_next` branching and buffer swapping.

**Key bottleneck analysis:**
- At 0.959ms for 2048³ matmul: ~18 TFLOPS effective vs 96 TFLOPS peak (~19% utilization)
- B reads are the main bottleneck: 16 scalar reads per k16 step, 4 DPAS calls per step
- The double-buffering adds overhead (extra barrier, branch, buffer index swap)

**Proposed improvements for this version:**
1. Keep double-buffering but simplify: remove `has_next` branch by handling last iteration separately
2. Use `intel_sub_group_block_read` for B when possible (columns are contiguous per subgroup, stride=N though, so scalar is needed)
3. Precompute B offsets more aggressively to reduce address math
4. Try vectorized B loads: pack 2 ushort reads into a single uint read pattern
5. Reduce register pressure by combining the A load patterns
6. Use `__attribute__((opencl_unroll_hint))` for finer control

Given the constraints (64 WIs, no B in SLM, 32x64x32 tile), I'll focus on:
- Cleaner double-buffering with reduced branch overhead
- Tighter B load scheduling
- Minimizing barrier count (still need 2 per K-tile for double buffer)

Actually, let me reconsider: V1/V2 with single buffer use 2 barriers per K-tile (before/after compute). V3 double-buffer also uses ~2 barriers per K-tile. The win from V3 is marginal. Let me try a cleaner single-buffer approach with maximum compute density and minimal overhead.

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (double-buffered 2x1KB = 2KB per buf, 4KB total)
// B loaded directly from global/L2
// K=2048 divides evenly by 32, no remainder path.
// Double-buffered: preload next A while computing current.
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
    const int b_col = col_valid ? col_idx : (N - 1);
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const int num_k_tiles = K >> 5;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Precompute A-load constants per WI
    // Each WI loads 16 elements: elem_id = lid + i*64 for i in [0,16)
    // elem_id maps to row r = elem_id/32, col c = elem_id%32 in the 32x32 tile
    int a_slm_off[16];
    int a_row_off[16];  // r * K (stride in global A)
    int a_col[16];      // c (column within K-tile)
    int a_gr_valid[16]; // whether global row is valid (for boundary)

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int elem_id = lid + i * 64;
        int r = elem_id >> 5;
        int c = elem_id & 31;
        a_slm_off[i] = elem_id;
        a_row_off[i] = r * K;
        a_col[i] = c;
        a_gr_valid[i] = (row_base + r < M) ? 1 : 0;
    }

    // ---- Load first A tile into buffer 0 ----
    {
        __local ushort* dst = slm_A;
        __global const ushort* A_base = A_us + a_row_base_K;
        if (row_tile_valid) {
            #pragma unroll
            for (int i = 0; i < 16; i++)
                dst[a_slm_off[i]] = A_base[a_row_off[i] + a_col[i]];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++)
                dst[a_slm_off[i]] = a_gr_valid[i] ? A_base[a_row_off[i] + a_col[i]] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // ---- Main loop with double buffering ----
    // Process tile kt from buffer cur_buf, load tile kt+1 into 1-cur_buf
    int cur_buf = 0;

    for (int kt = 0; kt < num_k_tiles - 1; kt++) {
        const int k = kt << 5;
        const int next_k = k + TILE_K;

        __local const ushort* cur_slm = slm_A + cur_buf * SLM_TILE;

        // ---- COMPUTE: two k16 DPAS steps on current A tile ----
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

        // ---- LOAD next A tile into alternate buffer ----
        barrier(CLK_LOCAL_MEM_FENCE);

        {
            __local ushort* dst = slm_A + (1 - cur_buf) * SLM_TILE;
            __global const ushort* A_base = A_us + a_row_base_K + next_k;
            if (row_tile_valid) {
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    dst[a_slm_off[i]] = A_base[a_row_off[i] + a_col[i]];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    dst[a_slm_off[i]] = a_gr_valid[i] ? (A_base[a_row_off[i] + a_col[i]]) : (ushort)0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        cur_buf = 1 - cur_buf;
    }

    // ---- Last K-tile: compute only, no next load needed ----
    {
        const int k = (num_k_tiles - 1) << 5;
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
    }

    // ---- Store results ----
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

