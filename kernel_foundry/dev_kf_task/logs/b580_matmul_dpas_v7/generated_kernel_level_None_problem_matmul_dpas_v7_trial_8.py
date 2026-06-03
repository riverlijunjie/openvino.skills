

## Analysis

Looking at the previous versions, all achieve ~33.9ms. The key issues are:

1. **Scalar packing in the hot loop** - Every K-step does 16+ scalar shift-and-OR operations to pack halfs into DPAS format
2. **Small output tile per subgroup** - Only 8x8 per subgroup; we should have each subgroup compute more output (register blocking)
3. **Excessive barriers and bounds checks** in the main loop
4. **SLM loads not vectorized** - scalar half loads with per-element bounds checks

**Key improvements for this version:**

1. **Register blocking**: Each subgroup computes 8x32 (8 rows × 32 cols = 4 DPAS calls per K-step) instead of 8x8. This increases compute density per memory load.
2. **Pre-pack into SLM in DPAS-ready format**: Store A and B as packed ints during cooperative load, eliminating repacking in the hot loop.
3. **Vectorized SLM loads**: Use `vload4`/`vstore4` where possible.
4. **Larger work-group tile**: TM=32, TN=128 with 16 subgroups (4×4 grid, each doing 8×32 via 4 DPAS ops).
5. **Remove redundant trailing barrier** by using single barrier at top of loop.

Actually, let me reconsider. With subgroup size 8, each DPAS produces 8x8. To do register blocking, each subgroup can accumulate multiple 8x8 tiles. Let me use TM=32, TN=64, with 4x2 subgroup grid where each subgroup computes 8x32 (4 DPAS columns per K-step).

**WG config**: LWS=(16, 4, 1) = 64 WIs = 8 subgroups. Subgroup grid: 2 along N × 4 along M. Each subgroup: 8 rows × 32 cols (4 DPAS). TM=32, TN=64, TK=16.

```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS + register blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Launch metadata:
//   LWS = (16, 4, 1) => 64 work-items = 8 subgroups of 8
//   GWS = (ceil(N/64)*16, ceil(M/32)*4, 1)
//   Subgroup size = 8
//   Work-group tile: TM=32, TN=64
//   Subgroup grid: 2 along N x 4 along M
//   Each subgroup: 8 rows x 32 cols (4 DPAS calls per K-step) => register blocking
//   SLM: A stored as packed int [32][8], B stored as packed int [8][64]
//   SLM bytes: 32*8*4 + 8*64*4 = 1024 + 2048 = 3072 bytes per buffer

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Tile dimensions
    const int TM = 32;
    const int TN = 64;
    const int TK = 16;
    const int WG_SIZE = 64; // 16 * 4

    // Work-group origin
    const int wg_n = get_group_id(0) * TN;
    const int wg_m = get_group_id(1) * TM;

    const int lid0 = get_local_id(0);  // 0..15
    const int lid1 = get_local_id(1);  // 0..3
    const int flat_lid = lid1 * 16 + lid0;  // 0..63

    const int sg_local_id = get_sub_group_local_id(); // 0..7
    const int sg_id = get_sub_group_id(); // 0..7

    // Subgroup grid: 2 along N, 4 along M
    const int sg_n = sg_id % 2;  // 0..1
    const int sg_m = sg_id / 2;  // 0..3

    // Each subgroup computes 8 rows x 32 cols using 4 DPAS (each 8x8)
    // Output origin for this subgroup
    const int tile_m = wg_m + sg_m * 8;
    const int tile_n = wg_n + sg_n * 32;

    // SLM: A in DPAS-packed format (int), B in VNNI-packed format (int)
    // A: [32 rows][8 ints] = 32 rows x 16 halfs packed as 8 ints each
    // B: [8 k-pairs][64 cols] as int = VNNI packed
    __local int slm_A[32 * 8];      // 1024 bytes
    __local int slm_B[8 * 64];      // 2048 bytes

    // 4 accumulators for 4 column-groups of 8
    float8 acc0 = (float8)(0.0f);  // cols 0..7
    float8 acc1 = (float8)(0.0f);  // cols 8..15
    float8 acc2 = (float8)(0.0f);  // cols 16..23
    float8 acc3 = (float8)(0.0f);  // cols 24..31

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TK - 1) / TK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k0 = kt * TK;
        const int k_rem = K - k0;

        // === Cooperative load A into packed SLM ===
        // 32 rows x 8 ints = 256 ints, 64 WIs => 4 ints each
        for (int i = flat_lid; i < 32 * 8; i += WG_SIZE) {
            int r = i >> 3;    // row 0..31
            int p = i & 7;     // pair index 0..7
            int gr = wg_m + r;
            int gc0 = k0 + p * 2;
            int gc1 = gc0 + 1;
            ushort v0 = (gr < M && gc0 < K) ? A_us[gr * K + gc0] : (ushort)0;
            ushort v1 = (gr < M && gc1 < K) ? A_us[gr * K + gc1] : (ushort)0;
            slm_A[r * 8 + p] = (int)v0 | ((int)v1 << 16);
        }

        // === Cooperative load B into VNNI-packed SLM ===
        // 8 k-pairs x 64 cols = 512 ints, 64 WIs => 8 ints each
        for (int i = flat_lid; i < 8 * 64; i += WG_SIZE) {
            int kp = i >> 6;   // k-pair 0..7
            int c = i & 63;    // col 0..63
            int gk0 = k0 + kp * 2;
            int gk1 = gk0 + 1;
            int gc = wg_n + c;
            ushort v0 = (gk0 < K && gc < N) ? B_us[gk0 * N + gc] : (ushort)0;
            ushort v1 = (gk1 < K && gc < N) ? B_us[gk1 * N + gc] : (ushort)0;
            slm_B[kp * 64 + c] = (int)v0 | ((int)v1 << 16);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === Compute: load from SLM in DPAS-ready format ===
        // A operand: row = sg_m * 8 + sg_local_id, 8 ints
        int a_row = sg_m * 8 + sg_local_id;
        int8 a_packed = vload8(0, &slm_A[a_row * 8]);

        // B operand for 4 column groups
        // Each column group: sg_n * 32 + group * 8 + sg_local_id
        int b_base = sg_n * 32;

        int8 b0, b1, b2, b3;
        int bc0 = b_base + 0 + sg_local_id;
        int bc1 = b_base + 8 + sg_local_id;
        int bc2 = b_base + 16 + sg_local_id;
        int bc3 = b_base + 24 + sg_local_id;

        b0 = (int8)(slm_B[0*64+bc0], slm_B[1*64+bc0], slm_B[2*64+bc0], slm_B[3*64+bc0],
                     slm_B[4*64+bc0], slm_B[5*64+bc0], slm_B[6*64+bc0], slm_B[7*64+bc0]);
        b1 = (int8)(slm_B[0*64+bc1], slm_B[1*64+bc1], slm_B[2*64+bc1], slm_B[3*64+bc1],
                     slm_B[4*64+bc1], slm_B[5*64+bc1], slm_B[6*64+bc1], slm_B[7*64+bc1]);
        b2 = (int8)(slm_B[0*64+bc2], slm_B[1*64+bc2], slm_B[2*64+bc2], slm_B[3*64+bc2],
                     slm_B[4*64+bc2], slm_B[5*64+bc2], slm_B[6*64+bc2], slm_B[7*64+bc2]);
        b3 = (int8)(slm_B[0*64+bc3], slm_B[1*64+bc3], slm_B[2*64+bc3], slm_B[3*64+bc3],
                     slm_B[4*64+bc3], slm_B[5*64+bc3], slm_B[6*64+bc3], slm_B[7*64+bc3]);

        // 4 DPAS calls - register blocking along N
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b2, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b3, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store 8x32 output tile ===
    int out_row = tile_m + sg_local_id;
    if (out_row < M) {
        int base = out_row * N;

        // Store acc0: cols tile_n+0..7
        if (tile_n + 8 <= N) {
            vstore8(convert_half8(acc0), 0, C + base + tile_n);
        } else {
            for (int j = 0; j < 8 && (tile_n + j) < N; j++)
                C[base + tile_n + j] = convert_half(((float*)&acc0)[j]);
        }

        // Store acc1: cols tile_n+8..15
        if (tile_n + 16 <= N) {
            vstore8(convert_half8(acc1), 0, C + base + tile_n + 8);
        } else {
            for (int j = 0; j < 8 && (tile_n + 8 + j) < N; j++)
                C[base + tile_n + 8 + j] = convert_half(((float*)&acc1)[j]);
        }

        // Store acc2: cols tile_n+16..23
        if (tile_n + 24 <= N) {
            vstore8(convert_half8(acc2), 0, C + base + tile_n + 16);
        } else {
            for (int j = 0; j < 8 && (tile_n + 16 + j) < N; j++)
                C[base + tile_n + 16 + j] = convert_half(((float*)&acc2)[j]);
        }

        // Store acc3: cols tile_n+24..31
        if (tile_n + 32 <= N) {
            vstore8(convert_half8(acc3), 0, C + base + tile_n + 24);
        } else {
            for (int j = 0; j < 8 && (tile_n + 24 + j) < N; j++)
                C[base + tile_n + 24 + j] = convert_half(((float*)&acc3)[j]);
        }
    }
}
```

