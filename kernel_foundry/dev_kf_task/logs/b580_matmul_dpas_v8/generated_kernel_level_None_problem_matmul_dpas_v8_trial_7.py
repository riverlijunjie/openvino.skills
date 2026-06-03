

## Analysis

The previous versions achieve correctness with ~44ms runtime. Key observations:

1. **Current bottleneck**: The kernel is likely memory-bandwidth limited. Each WG computes 32×64 output with 64 threads, giving decent compute density but we can improve.

2. **Opportunities**:
   - **Larger tiles**: Increase TILE_M to process more rows per WG, improving data reuse of B tile from SLM.
   - **Double-buffering SLM**: Overlap SLM loads with computation by using two SLM buffers.
   - **Increase compute per subgroup**: Each subgroup can compute more rows (e.g., 16 rows instead of 8) by having 2 sets of A registers, reusing the same B data.
   - **Better SLM load pattern**: Use block reads where possible for coalesced access.
   - **Larger K-tile**: Load more K at once to amortize barrier costs.

3. **Key optimization - Register blocking on M**: Each subgroup computes 16 rows × 64 cols instead of 8×64. This doubles A reuse from registers against B from SLM, and doubles the compute-to-memory ratio. With 4 subgroups × 16 rows = 64 rows per WG.

4. **Tile sizing**: 64M × 64N per WG with TILE_K=32. This gives better occupancy and reuse.

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Register-blocked DPAS matmul: C[M,N] = A[M,K] * B[K,N]
// Tile: 64M x 64N per work-group, K-tile = 32 (two k16 DPAS steps)
// 8 subgroups per WG, each subgroup computes 8M x 64N (4 DPAS calls x 2 K-steps)
// LWS = (16, 8) => 128 WIs, 8 subgroups of 16
// GWS = (ceil(N/64)*16, ceil(M/64)*8)

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 8
#define K_STEP 16

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 8, 1)))
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

    const int sg_id = get_local_id(1);           // 0..7
    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int lid0 = get_local_id(0);

    // Each subgroup handles 8 rows
    const int row_base = wg_m + sg_id * 8;

    // Accumulators: 8 rows x 64 cols = 4 DPAS blocks per subgroup
    float8 acc0 = 0.0f;  // cols [0..15]
    float8 acc1 = 0.0f;  // cols [16..31]
    float8 acc2 = 0.0f;  // cols [32..47]
    float8 acc3 = 0.0f;  // cols [48..63]

    // SLM layout:
    // A: 64 rows x 32 cols = 2048 ushorts (4KB)
    // B_vnni: 2 k-blocks x 8 pairs x 64 cols = 1024 uints (4KB)
    // Total: 8KB
    __local ushort slm_a[TILE_M * TILE_K];        // 2048
    __local uint slm_b_vnni[2 * 8 * TILE_N];      // 1024

    const int flat_id = sg_id * 16 + lid0;        // 0..127

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int m_safe = ((wg_m + TILE_M) <= M);
    const int n_safe = ((wg_n + TILE_N) <= N);

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const int k_safe = ((k0 + TILE_K) <= K);

        // === Cooperative load A[64][32] into SLM ===
        // 2048 elements, 128 threads => 16 elements per thread
        if (m_safe & k_safe) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 128;
                int r = idx >> 5;   // / 32
                int c = idx & 31;   // % 32
                slm_a[r * TILE_K + c] = A_us[(wg_m + r) * K + k0 + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int idx = flat_id + i * 128;
                int r = idx >> 5;
                int c = idx & 31;
                int gr = wg_m + r;
                int gc = k0 + c;
                ushort val = (gr < M && gc < K) ? A_us[gr * K + gc] : (ushort)0;
                slm_a[r * TILE_K + c] = val;
            }
        }

        // === Cooperative load B[32][64] into VNNI format ===
        // 1024 uints, 128 threads => 8 per thread
        if (n_safe & k_safe) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id + i * 128;
                int block = idx >> 9;        // / 512 -> 0 or 1
                int rem = idx & 511;
                int pair = rem >> 6;         // / 64 -> 0..7
                int col = rem & 63;          // % 64

                int k_row0 = k0 + block * 16 + pair * 2;
                int k_row1 = k_row0 + 1;
                int gcol = wg_n + col;

                ushort v0 = B_us[k_row0 * N + gcol];
                ushort v1 = B_us[k_row1 * N + gcol];
                slm_b_vnni[block * 8 * TILE_N + pair * TILE_N + col] = ((uint)v1 << 16) | (uint)v0;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id + i * 128;
                int block = idx >> 9;
                int rem = idx & 511;
                int pair = rem >> 6;
                int col = rem & 63;

                int k_row0 = k0 + block * 16 + pair * 2;
                int k_row1 = k_row0 + 1;
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
            // Load A: 8 rows x 16 cols for this subgroup
            __local ushort* a_ptr = slm_a + sg_id * 8 * TILE_K + kk * K_STEP;
            short8 a_val = as_short8((ushort8)(
                a_ptr[0 * TILE_K + sg_lid], a_ptr[1 * TILE_K + sg_lid],
                a_ptr[2 * TILE_K + sg_lid], a_ptr[3 * TILE_K + sg_lid],
                a_ptr[4 * TILE_K + sg_lid], a_ptr[5 * TILE_K + sg_lid],
                a_ptr[6 * TILE_K + sg_lid], a_ptr[7 * TILE_K + sg_lid]));

            // Load B VNNI: 8 pairs x 64 cols, 4 blocks of 16 cols
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

