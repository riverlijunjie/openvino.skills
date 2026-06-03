

## Analysis

Looking at the results:
- **Version 2** (52.1ms) is the best performing correct version - uses 32M×64N tiles, 4 subgroups, TILE_K=32
- **Best implementation** (75.7ms) - same tile size but TILE_K=64, more SLM loading overhead per iteration
- **Last tested** (207ms) - only 2 subgroups (32 WIs per WG), underutilizing the hardware

Key observations:
1. Version 2 with TILE_K=32 and 4 subgroups is fastest - good balance of compute vs memory loading
2. The TILE_K=64 version is slower despite more compute per barrier - the 32 scalar loads per thread for A and B are too expensive
3. The 2-subgroup version is slow due to insufficient parallelism and too many loads per thread
4. The SLM loading is a major bottleneck - element-by-element loads with bounds checking are slow

Improvements for this version:
1. Start from Version 2's structure (TILE_K=32, 4 subgroups, 32M×64N)
2. Use more efficient SLM loading with strided access patterns for better coalescing
3. Try using `intel_sub_group_block_read` for B matrix loads from SLM instead of scalar reads
4. Reduce bounds-checking overhead by separating fast-path (fully in-bounds tiles) from edge cases

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Tile: 32M x 64N per work-group, K-tile = 32 (two DPAS k=16 steps per SLM load)
// 4 subgroups per WG, each subgroup computes 8M x 64N (4 DPAS calls x 2 K-steps)
// LWS = (16, 4) => 64 WIs, 4 subgroups of 16
// GWS = (ceil(N/64)*16, ceil(M/32)*4)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 4, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    const int sg_id = get_local_id(1);           // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int lid0 = get_local_id(0);

    const int row_base = wg_m + sg_id * 8;

    // Accumulators: 8 rows x 64 cols = 4 DPAS blocks
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // SLM for A[32][32] and B in VNNI[2][8][64]
    __local ushort slm_a[TILE_M * TILE_K];       // 1024 ushorts = 2KB
    __local uint slm_b_vnni[2 * 8 * TILE_N];     // 1024 uints = 4KB

    const int flat_id = sg_id * 16 + lid0;       // 0..63

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    // Check if this WG is fully in-bounds for M and N dimensions
    const int wg_m_end = wg_m + TILE_M;
    const int wg_n_end = wg_n + TILE_N;
    const int m_safe = (wg_m_end <= M);
    const int n_safe = (wg_n_end <= N);

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int k_safe = (k0 + TILE_K <= K);

        // === Cooperative load A[32][32] into SLM ===
        // 1024 elements, 64 threads => 16 elements per thread (strided by 64)
        if (m_safe && k_safe) {
            // Fast path: no bounds checking needed
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 64;
                int r = idx >> 5;
                int c = idx & 31;
                int gr = wg_m + r;
                int gc = k0 + c;
                slm_a[r * TILE_K + c] = A_us[gr * K + gc];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 64;
                int r = idx >> 5;
                int c = idx & 31;
                int gr = wg_m + r;
                int gc = k0 + c;
                ushort val = 0;
                if (gr < M && gc < K)
                    val = A_us[gr * K + gc];
                slm_a[r * TILE_K + c] = val;
            }
        }

        // === Cooperative load B[32][64] into VNNI format ===
        // 1024 uints, 64 threads => 16 per thread (strided by 64)
        if (n_safe && k_safe) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 64;
                int block = idx >> 9;
                int rem = idx & 511;
                int pair = rem >> 6;
                int col = rem & 63;

                int k_base = k0 + block * 16;
                int k_row0 = k_base + pair * 2;
                int k_row1 = k_base + pair * 2 + 1;
                int gcol = wg_n + col;

                ushort v0 = B_us[k_row0 * N + gcol];
                ushort v1 = B_us[k_row1 * N + gcol];
                slm_b_vnni[block * 8 * TILE_N + pair * TILE_N + col] = ((uint)v1 << 16) | (uint)v0;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 64;
                int block = idx >> 9;
                int rem = idx & 511;
                int pair = rem >> 6;
                int col = rem & 63;

                int k_base = k0 + block * 16;
                int k_row0 = k_base + pair * 2;
                int k_row1 = k_base + pair * 2 + 1;
                int gcol = wg_n + col;

                ushort v0 = 0, v1 = 0;
                if (gcol < N) {
                    if (k_row0 < K) v0 = B_us[k_row0 * N + gcol];
                    if (k_row1 < K) v1 = B_us[k_row1 * N + gcol];
                }
                slm_b_vnni[block * 8 * TILE_N + pair * TILE_N + col] = ((uint)v1 << 16) | (uint)v0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === Two K-steps of 16 each ===
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            __local ushort* a_ptr = slm_a + sg_id * 8 * TILE_K + kk * 16;
            short8 a_val = as_short8((ushort8)(
                a_ptr[0 * TILE_K + sg_lid], a_ptr[1 * TILE_K + sg_lid],
                a_ptr[2 * TILE_K + sg_lid], a_ptr[3 * TILE_K + sg_lid],
                a_ptr[4 * TILE_K + sg_lid], a_ptr[5 * TILE_K + sg_lid],
                a_ptr[6 * TILE_K + sg_lid], a_ptr[7 * TILE_K + sg_lid]));

            __local uint* b_base = slm_b_vnni + kk * 8 * TILE_N;

            int8 b0 = (int8)(
                as_int(b_base[0*TILE_N + sg_lid]),      as_int(b_base[1*TILE_N + sg_lid]),
                as_int(b_base[2*TILE_N + sg_lid]),      as_int(b_base[3*TILE_N + sg_lid]),
                as_int(b_base[4*TILE_N + sg_lid]),      as_int(b_base[5*TILE_N + sg_lid]),
                as_int(b_base[6*TILE_N + sg_lid]),      as_int(b_base[7*TILE_N + sg_lid]));

            int8 b1 = (int8)(
                as_int(b_base[0*TILE_N + 16 + sg_lid]), as_int(b_base[1*TILE_N + 16 + sg_lid]),
                as_int(b_base[2*TILE_N + 16 + sg_lid]), as_int(b_base[3*TILE_N + 16 + sg_lid]),
                as_int(b_base[4*TILE_N + 16 + sg_lid]), as_int(b_base[5*TILE_N + 16 + sg_lid]),
                as_int(b_base[6*TILE_N + 16 + sg_lid]), as_int(b_base[7*TILE_N + 16 + sg_lid]));

            int8 b2 = (int8)(
                as_int(b_base[0*TILE_N + 32 + sg_lid]), as_int(b_base[1*TILE_N + 32 + sg_lid]),
                as_int(b_base[2*TILE_N + 32 + sg_lid]), as_int(b_base[3*TILE_N + 32 + sg_lid]),
                as_int(b_base[4*TILE_N + 32 + sg_lid]), as_int(b_base[5*TILE_N + 32 + sg_lid]),
                as_int(b_base[6*TILE_N + 32 + sg_lid]), as_int(b_base[7*TILE_N + 32 + sg_lid]));

            int8 b3 = (int8)(
                as_int(b_base[0*TILE_N + 48 + sg_lid]), as_int(b_base[1*TILE_N + 48 + sg_lid]),
                as_int(b_base[2*TILE_N + 48 + sg_lid]), as_int(b_base[3*TILE_N + 48 + sg_lid]),
                as_int(b_base[4*TILE_N + 48 + sg_lid]), as_int(b_base[5*TILE_N + 48 + sg_lid]),
                as_int(b_base[6*TILE_N + 48 + sg_lid]), as_int(b_base[7*TILE_N + 48 + sg_lid]));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b2, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b3, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = row_base + i;
        if (row < M) {
            float v0 = ((float*)&acc0)[i];
            float v1 = ((float*)&acc1)[i];
            float v2 = ((float*)&acc2)[i];
            float v3 = ((float*)&acc3)[i];

            int col0 = wg_n + sg_lid;
            int col1 = wg_n + 16 + sg_lid;
            int col2 = wg_n + 32 + sg_lid;
            int col3 = wg_n + 48 + sg_lid;

            __global half* out = C + row * N;
            if (col0 < N) out[col0] = convert_half(v0);
            if (col1 < N) out[col1] = convert_half(v1);
            if (col2 < N) out[col2] = convert_half(v2);
            if (col3 < N) out[col3] = convert_half(v3);
        }
    }
}
```

