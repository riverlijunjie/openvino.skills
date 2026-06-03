

## Analysis

Looking at the three versions:
- **V1** (0.968ms): Unrolled 2x with `#pragma unroll 2` inner loop
- **V2** (0.969ms): Manually unrolled 2x (explicit first/second tile) - essentially same perf
- **V3/Last** (0.959ms): Double-buffered SLM - slight improvement by overlapping A loads with compute

The double-buffering approach in V3 gave a small win (~1%) by hiding A-load latency. However, it adds complexity (extra SLM, extra barriers, conditional `has_next` check).

Key observations:
1. The double-buffering helps slightly but adds overhead from the conditional branch and extra SLM pressure
2. All versions use the same B-loading pattern with scalar reads packed into int - this is a bottleneck
3. The `col_valid` check and `b_col` clamping add unnecessary overhead for the common case
4. Store operations are all scalar - could use subgroup block writes

**Proposed improvements for this iteration:**
1. **Simplify double-buffering**: Remove the conditional `has_next` check by peeling the last iteration
2. **Precompute B offsets more aggressively**: Reduce address arithmetic in the inner loop
3. **Use intel_sub_group_block_write for stores**: Vectorize C writes where possible
4. **Tighten the hot loop**: Minimize register pressure by computing B addresses inline
5. **Remove boundary checks for M/N**: For typical 2048x2048 matmul, tiles divide evenly - use select/predication instead of branches

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (double-buffered: 2x 2KB = 4KB), B from global/L2
// K=2048 divides evenly by 32, no remainder path.
// Double-buffered SLM: overlap next A load with current compute.
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

    // Precompute per-WI A-load constants
    int a_local_off[16];
    int a_row_off[16];  // r * K (stride in global A)
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

    // Macro for loading A tile into SLM buffer
    #define LOAD_A_TILE(dst_slm, k_offset) \
    { \
        if (row_tile_valid) { \
            __global const ushort* A_tile = A_us + a_row_base_K + (k_offset); \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) \
                (dst_slm)[a_local_off[i]] = A_tile[a_row_off[i] + a_c[i]]; \
        } else { \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) { \
                int gr = row_base + (a_local_off[i] >> 5); \
                (dst_slm)[a_local_off[i]] = (gr < M) ? A_us[gr * K + (k_offset) + a_c[i]] : (ushort)0; \
            } \
        } \
    }

    // Macro for DPAS compute step (one k16 chunk)
    #define DPAS_STEP(slm_base_ptr, b_k_offset) \
    { \
        short8 a0, a1, a2, a3; \
        _Pragma("unroll") \
        for (int r = 0; r < 8; r++) \
            ((ushort*)&a0)[r] = intel_sub_group_block_read_us((slm_base_ptr) + r * TILE_K); \
        _Pragma("unroll") \
        for (int r = 0; r < 8; r++) \
            ((ushort*)&a1)[r] = intel_sub_group_block_read_us((slm_base_ptr) + (8 + r) * TILE_K); \
        _Pragma("unroll") \
        for (int r = 0; r < 8; r++) \
            ((ushort*)&a2)[r] = intel_sub_group_block_read_us((slm_base_ptr) + (16 + r) * TILE_K); \
        _Pragma("unroll") \
        for (int r = 0; r < 8; r++) \
            ((ushort*)&a3)[r] = intel_sub_group_block_read_us((slm_base_ptr) + (24 + r) * TILE_K); \
        \
        int b_off_local = (b_k_offset) * N + b_col; \
        int8 b_val; \
        _Pragma("unroll") \
        for (int p = 0; p < 8; p++) { \
            ushort s0 = B_us[b_off_local]; \
            ushort s1 = B_us[b_off_local + N]; \
            ((int*)&b_val)[p] = as_int((ushort2)(s0, s1)); \
            b_off_local += N2; \
        } \
        \
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0); \
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1); \
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2); \
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3); \
    }

    // Load first A tile into buffer 0
    LOAD_A_TILE(slm_A, 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main loop: all tiles except the last, with double-buffered prefetch
    for (int kt = 0; kt < num_k_tiles - 1; kt++) {
        const int k = kt * TILE_K;
        __local ushort* cur_slm = slm_A + (kt & 1) * SLM_TILE;
        __local ushort* next_slm = slm_A + (1 - (kt & 1)) * SLM_TILE;

        // Compute two k16 steps on current buffer
        DPAS_STEP(cur_slm, k);
        DPAS_STEP(cur_slm + 16, k + 16);

        // Barrier: ensure all WIs done reading cur_slm before we overwrite next_slm
        // (next_slm was the previous iteration's cur_slm's partner)
        barrier(CLK_LOCAL_MEM_FENCE);

        // Load next A tile into the other buffer
        LOAD_A_TILE(next_slm, (kt + 1) * TILE_K);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Last tile: compute without loading next
    {
        const int last_kt = num_k_tiles - 1;
        const int k = last_kt * TILE_K;
        __local ushort* cur_slm = slm_A + (last_kt & 1) * SLM_TILE;

        DPAS_STEP(cur_slm, k);
        DPAS_STEP(cur_slm + 16, k + 16);
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

    #undef LOAD_A_TILE
    #undef DPAS_STEP
}
```

