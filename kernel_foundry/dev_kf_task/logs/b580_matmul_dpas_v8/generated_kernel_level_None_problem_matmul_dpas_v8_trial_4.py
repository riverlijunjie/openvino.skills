

## Analysis

The previous best kernel (52.1ms) uses 32x64 tiles with K=32, 4 subgroups, and 2 K-steps per SLM load. Key observations:

1. **SLM loading is scalar**: Each element is loaded individually with bounds checks, which is slow.
2. **B matrix loading uses strided access**: The `flat_id + i * 64` pattern for B loading isn't optimal for coalescing.
3. **Only 4 DPAS calls per K-step**: We could increase compute density by having each subgroup compute more output.
4. **Larger K-tile amortizes barriers**: K=32 with 2 DPAS steps per load was better than K=16. We can try K=64 (4 steps).
5. **SLM reads are scalar**: Loading A and B from SLM element-by-element rather than using block reads.

**Proposed improvements:**
- Increase K-tile to 64 to further amortize SLM load + barrier costs (4 DPAS k16 steps per load)
- Use `intel_sub_group_block_read` for SLM reads where possible
- Optimize global memory loading with wider loads (vload4)
- Increase tile to 32x128 with 8 subgroups (LWS=16x8) for more compute per WG, but this may use too much SLM. Instead, keep 32x64 but increase K-tile.

Let me try a 32x64 tile with K=64 (4 k16 steps per barrier), which gives 16 DPAS calls between barriers vs 8 before.

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Tile: 32M x 64N per work-group, K-tile = 64 (four DPAS k=16 steps per SLM load)
// 4 subgroups per WG, each subgroup computes 8M x 64N (4 DPAS calls x 4 K-steps = 16 DPAS/barrier)
// LWS = (16, 4) => 64 WIs, 4 subgroups of 16
// GWS = (ceil(N/64)*16, ceil(M/32)*4)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 64
#define K_STEP 16
#define NUM_K_STEPS 4
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

    // SLM: A[32][64] as ushort = 2048, B_vnni[4*8][64] as uint = 2048
    // Total SLM: 2048*2 + 2048*4 = 12288 bytes = 12KB per WG (well within limits)
    __local ushort slm_a[TILE_M * TILE_K];              // 2048
    __local uint slm_b_vnni[NUM_K_STEPS * 8 * TILE_N];  // 2048

    const int flat_id = sg_id * 16 + lid0;  // 0..63

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        // === Cooperative load A[32][64] into SLM ===
        // 2048 elements, 64 threads => 32 elements per thread
        // Each thread loads a contiguous chunk of 32 elements
        {
            int base_idx = flat_id * 32;
            int r = base_idx >> 6;   // / 64
            int c = base_idx & 63;   // % 64
            int gr = wg_m + r;
            int gc = k0 + c;

            // 32 elements per thread, but they span at most 1 row since TILE_K=64 and c starts at 0 or 32
            // Actually flat_id*32: flat_id=0 -> idx=0, r=0,c=0; flat_id=1 -> idx=32, r=0,c=32; flat_id=2 -> idx=64, r=1,c=0...
            // So each thread loads 32 consecutive halves from one row
            if (gr < M) {
                int off_g = gr * K + gc;
                int off_s = r * TILE_K + c;
                int k_remain = K - gc;
                if (k_remain >= 32) {
                    // Fast path: all 32 in bounds
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        ushort v0 = A_us[off_g + j*4];
                        ushort v1 = A_us[off_g + j*4 + 1];
                        ushort v2 = A_us[off_g + j*4 + 2];
                        ushort v3 = A_us[off_g + j*4 + 3];
                        slm_a[off_s + j*4]     = v0;
                        slm_a[off_s + j*4 + 1] = v1;
                        slm_a[off_s + j*4 + 2] = v2;
                        slm_a[off_s + j*4 + 3] = v3;
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 32; j++) {
                        slm_a[off_s + j] = (gc + j < K) ? A_us[off_g + j] : (ushort)0;
                    }
                }
            } else {
                int off_s = r * TILE_K + c;
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    slm_a[off_s + j] = (ushort)0;
                }
            }
        }

        // === Cooperative load B[64][64] into VNNI format (four 16-deep blocks) ===
        // 2048 uints total, 64 threads => 32 per thread
        {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int idx = flat_id + i * 64;  // strided for coalescing within subgroup
                // idx ranges 0..2047
                int block = idx >> 9;        // / 512 => 0..3 (which k16 block)
                int rem = idx & 511;
                int pair = rem >> 6;         // / 64 => k-pair 0..7
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

        // === Four K-steps of 16 each ===
        #pragma unroll
        for (int kk = 0; kk < NUM_K_STEPS; kk++) {
            // Load A: 8 rows x 16 cols for this subgroup
            __local ushort* a_ptr = slm_a + sg_id * 8 * TILE_K + kk * K_STEP;
            short8 a_val = as_short8((ushort8)(
                a_ptr[0 * TILE_K + sg_lid], a_ptr[1 * TILE_K + sg_lid],
                a_ptr[2 * TILE_K + sg_lid], a_ptr[3 * TILE_K + sg_lid],
                a_ptr[4 * TILE_K + sg_lid], a_ptr[5 * TILE_K + sg_lid],
                a_ptr[6 * TILE_K + sg_lid], a_ptr[7 * TILE_K + sg_lid]));

            // Load B VNNI: 8 pairs x 4 col-blocks of 16
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

            if (col0 < N) C[row * N + col0] = convert_half(v0);
            if (col1 < N) C[row * N + col1] = convert_half(v1);
            if (col2 < N) C[row * N + col2] = convert_half(v2);
            if (col3 < N) C[row * N + col3] = convert_half(v3);
        }
    }
}
```

