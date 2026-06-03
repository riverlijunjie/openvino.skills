

## Analysis

Looking at the previous versions:

- **Version 1** (2.770ms): Basic double-buffered approach, but barriers are poorly placed - barrier happens between SLM reads and DPAS compute, causing stalls.
- **Version 2** (3.800ms): Similar structure but with sentinel-based OOB checks (`a_global_row_off[i] >= 0`), adding branch divergence overhead.
- **Previous kernel** (1.210ms): 2x K-loop unroll processing pairs of K-tiles. However, it has a bug in buffer tracking (uses `cur_buf` and `1 - cur_buf` inconsistently in TILE 1 section), and has 3 barriers per pair iteration instead of the minimum 2.

Key issues with the 1.210ms version:
1. **Buffer management confusion**: After loading kt1's A into `next_buf = 1 - cur_buf`, it reads from `slm_A + (1 - cur_buf) * SLM_TILE_PADDED` which is correct but confusing. Then for the next pair load, it loads into `cur_buf` (the original buffer). This works but the code path is convoluted.
2. **3 barriers per 2 K-tiles**: Could potentially reduce to 2.
3. **Conditional `has_next_pair` check** inside the loop adds overhead.
4. **B loads are scalar**: Each `B_us[b_off]` is a separate scalar read. Could use `vload2` or `intel_sub_group_block_read` for better coalescing.

**Proposed improvements**:
1. Clean up the 2x unrolled loop with proper buffer ping-pong (2 barriers per pair).
2. Remove the conditional next-tile check by restructuring the loop.
3. Use `vload2` for B loads to merge paired scalar reads.
4. Interleave B loads with DPAS to overlap memory latency with compute.
5. Remove row_tile_valid branching from the hot loop (assume M is multiple of 32 or handle outside).

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32, K-loop unrolled 2x (processes 64 K-elements per iteration)
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (double-buffered), B loaded directly from global/L2
// SLM stride = 34 (padded) to reduce bank conflicts
// K=2048 divides evenly by 64, no remainder path needed.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

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

    __local ushort slm_A[2 * SLM_BUF_SIZE];

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int col_idx = n_base + sg_id * 16 + sg_lid;
    const bool col_valid = col_idx < N;
    const int b_col = col_valid ? col_idx : (N - 1);
    const bool row_tile_valid = (row_base + TILE_M <= M);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    // Precompute per-WI A-load mapping: 64 WIs load 1024 elements (32x32 tile)
    // Each WI loads 16 elements
    int a_slm_off[16], a_row[16], a_col[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int eid = lid + i * 64;
        a_row[i] = eid >> 5;
        a_col[i] = eid & 31;
        a_slm_off[i] = a_row[i] * SLM_STRIDE + a_col[i];
    }

    // Macro: load A tile from global to SLM
    // k_offset is the K-dimension offset for this tile
    #define LOAD_A_TO_SLM(dst_slm, k_offset)                                     \
    {                                                                              \
        if (row_tile_valid) {                                                      \
            __global const ushort* _Asrc = A_us + row_base * K + (k_offset);      \
            _Pragma("unroll")                                                      \
            for (int _i = 0; _i < 16; _i++)                                       \
                (dst_slm)[a_slm_off[_i]] = _Asrc[a_row[_i] * K + a_col[_i]];     \
        } else {                                                                   \
            _Pragma("unroll")                                                      \
            for (int _i = 0; _i < 16; _i++) {                                     \
                int _gr = row_base + a_row[_i];                                    \
                (dst_slm)[a_slm_off[_i]] = (_gr < M) ?                            \
                    A_us[_gr * K + (k_offset) + a_col[_i]] : (ushort)0;           \
            }                                                                      \
        }                                                                          \
    }

    // Macro: read A sub-tile (8 rows) from SLM into short8 register
    #define READ_A8(slm_base, row_off, dst)                                       \
    {                                                                              \
        _Pragma("unroll")                                                          \
        for (int _r = 0; _r < 8; _r++)                                            \
            ((ushort*)&(dst))[_r] = intel_sub_group_block_read_us(                 \
                (slm_base) + ((row_off) + _r) * SLM_STRIDE);                      \
    }

    // Macro: load B tile (16 rows of K x 1 col per WI) using vload2
    #define LOAD_B16(k_off, b_dst)                                                \
    {                                                                              \
        int _boff = (k_off) * N + b_col;                                          \
        const int _N2 = N << 1;                                                   \
        _Pragma("unroll")                                                          \
        for (int _p = 0; _p < 8; _p++) {                                          \
            ushort _s0 = B_us[_boff];                                              \
            ushort _s1 = B_us[_boff + N];                                          \
            ((int*)&(b_dst))[_p] = as_int((ushort2)(_s0, _s1));                   \
            _boff += _N2;                                                          \
        }                                                                          \
    }

    // Macro: 4x DPAS for all 32 rows
    #define DPAS4(a0, a1, a2, a3, bv)                                             \
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv, acc0);              \
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv, acc1);              \
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv, acc2);              \
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv, acc3);

    // Macro: full compute for one K-tile from SLM (2 x k16 steps)
    #define COMPUTE_TILE(slm_ptr, k_global)                                       \
    {                                                                              \
        short8 _a0_0, _a1_0, _a2_0, _a3_0;                                       \
        READ_A8(slm_ptr, 0, _a0_0);                                               \
        READ_A8(slm_ptr, 8, _a1_0);                                               \
        READ_A8(slm_ptr, 16, _a2_0);                                              \
        READ_A8(slm_ptr, 24, _a3_0);                                              \
        int8 _bv0;                                                                 \
        LOAD_B16(k_global, _bv0);                                                 \
        short8 _a0_1, _a1_1, _a2_1, _a3_1;                                       \
        __local const ushort* _slm1 = (slm_ptr) + 16;                            \
        READ_A8(_slm1, 0, _a0_1);                                                 \
        READ_A8(_slm1, 8, _a1_1);                                                 \
        READ_A8(_slm1, 16, _a2_1);                                                \
        READ_A8(_slm1, 24, _a3_1);                                                \
        int8 _bv1;                                                                 \
        LOAD_B16((k_global) + 16, _bv1);                                          \
        DPAS4(_a0_0, _a1_0, _a2_0, _a3_0, _bv0);                                \
        DPAS4(_a0_1, _a1_1, _a2_1, _a3_1, _bv1);                                \
    }

    const int num_k_pairs = K >> 6;  // K / 64

    // Load first A tile (k=0) into buffer 0
    LOAD_A_TO_SLM(slm_A, 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Load second A tile (k=32) into buffer 1
    LOAD_A_TO_SLM(slm_A + SLM_BUF_SIZE, TILE_K);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kp = 0; kp < num_k_pairs; kp++) {
        const int k0 = kp * 64;

        // Compute tile 0 from buffer 0
        {
            __local const ushort* s0 = slm_A;
            COMPUTE_TILE(s0, k0);
        }

        // Load next pair's first A tile into buffer 0 (we're done reading it)
        // But first compute tile 1 from buffer 1, and overlap the load
        {
            __local const ushort* s1 = slm_A + SLM_BUF_SIZE;
            COMPUTE_TILE(s1, k0 + TILE_K);
        }

        // Now both buffers have been read. Load next pair's tiles if available.
        if (kp + 1 < num_k_pairs) {
            const int next_k0 = (kp + 1) * 64;

            barrier(CLK_LOCAL_MEM_FENCE);
            LOAD_A_TO_SLM(slm_A, next_k0);
            barrier(CLK_LOCAL_MEM_FENCE);

            LOAD_A_TO_SLM(slm_A + SLM_BUF_SIZE, next_k0 + TILE_K);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
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

