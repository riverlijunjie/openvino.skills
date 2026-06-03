

## Analysis

Looking at the performance results:
- **Version 2 (1.080ms)** is the best - uses SLM for A with fast-path (no boundary checks) and TILE_K=32
- **Best so far (3.870ms)** - similar to V2 but always does boundary checks
- **Last attempt (6.660ms)** - TILE_K=64 with double-buffering, but the double-buffering was implemented incorrectly (load happens *after* compute with barriers serializing everything, defeating the purpose)
- **Version 1 (5.740ms)** - no SLM, direct global loads for A

Key insights:
1. TILE_K=32 outperforms TILE_K=64 - likely because smaller SLM footprint allows more occupancy
2. Fast-path without boundary checks matters significantly (V2 vs best-so-far)
3. The failed double-buffering attempt shows that naive barrier placement hurts more than helps
4. B is loaded from global per-subgroup anyway, so SLM for B isn't needed (each SG needs different columns)

**Improvements for this version:**
1. Start from Version 2's structure (best performer)
2. Also cache B in SLM to reduce global memory traffic - all 4 SGs load different B columns but from same K-rows, so B SLM sharing doesn't help directly. Keep B from global.
3. Use `intel_sub_group_block_read` for more efficient SLM reads where possible
4. Optimize the SLM cooperative load pattern - use shift operations instead of div/mod
5. Reduce register pressure by computing col_idx once
6. Try using sub-group block reads for B from global memory for coalesced access

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A tile cached in SLM, B loaded directly from global
// Fast path skips boundary checks when tile is fully in-bounds
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_A_STRIDE (TILE_K + 2)

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

    const int col_base = get_group_id(0) * TILE_N + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    __local half slm_A[TILE_M * SLM_A_STRIDE];

    // Early exit for completely out-of-bounds workgroups
    if (row_base >= M || col_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_full = (row_base + TILE_M <= M);

    const int num_full_k_tiles = K / TILE_K;
    const int k_remainder = K - num_full_k_tiles * TILE_K;

    // Precompute element mapping for SLM cooperative load
    // 1024 elements / 64 WIs = 16 elements per WI
    // elem_id = lid + i*64, r = elem_id >> 5, c = elem_id & 31

    // === Main loop over full K tiles ===
    for (int kt = 0; kt < num_full_k_tiles; kt++) {
        const int k = kt * TILE_K;

        // Cooperative load A into SLM
        if (row_tile_full) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;   // / 32
                int c = elem_id & 31;   // % 32
                slm_A[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + k + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                int gr = row_base + r;
                slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? A[gr * K + k + c] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Two k16 DPAS steps per k32 tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load A from SLM
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a0)[r] = as_short(slm_A[r * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a1)[r] = as_short(slm_A[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a2)[r] = as_short(slm_A[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a3)[r] = as_short(slm_A[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);

            // Load B from global
            int8 b_val;
            int gk_base = k + kk;
            if (col_valid) {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int k_row0 = gk_base + 2 * p;
                    int k_row1 = k_row0 + 1;
                    short s0 = as_short(B[k_row0 * N + col_idx]);
                    short s1 = as_short(B[k_row1 * N + col_idx]);
                    ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                }
            } else {
                b_val = (int8)(0);
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Remainder K tile ===
    if (k_remainder > 0) {
        const int k = num_full_k_tiles * TILE_K;

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int elem_id = lid + i * 64;
            int r = elem_id >> 5;
            int c = elem_id & 31;
            int gr = row_base + r;
            int gk = k + c;
            slm_A[r * SLM_A_STRIDE + c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a0)[r] = as_short(slm_A[r * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a1)[r] = as_short(slm_A[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a2)[r] = as_short(slm_A[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a3)[r] = as_short(slm_A[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);

            int8 b_val;
            int gk_base = k + kk;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_row0 = gk_base + 2 * p;
                int k_row1 = k_row0 + 1;
                short s0 = (k_row0 < K && col_valid) ? as_short(B[k_row0 * N + col_idx]) : (short)0;
                short s1 = (k_row1 < K && col_valid) ? as_short(B[k_row1 * N + col_idx]) : (short)0;
                ((int*)&b_val)[p] = as_int((short2)(s0, s1));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }
    }

    // Store results
    if (col_valid) {
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + r;
            if (row_idx < M)
                C[row_idx * N + col_idx] = convert_half(((float*)&acc0)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + 8 + r;
            if (row_idx < M)
                C[row_idx * N + col_idx] = convert_half(((float*)&acc1)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + 16 + r;
            if (row_idx < M)
                C[row_idx * N + col_idx] = convert_half(((float*)&acc2)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + 24 + r;
            if (row_idx < M)
                C[row_idx * N + col_idx] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

