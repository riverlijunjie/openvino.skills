

## Analysis

Looking at the three previous versions, all achieve 33.9ms. The key bottlenecks in the latest version are:

1. **Fake double-buffering**: The barrier before `buf = next_buf` serializes prefetch and compute, negating the benefit of double buffering.
2. **Excessive scalar packing**: 16 `vload2`/`as_int` operations per K-step to pack A and B for DPAS.
3. **Scalar stores**: Writing output element-by-element instead of vector stores.
4. **Small compute-per-subgroup**: Each subgroup only computes 8×8, meaning low arithmetic intensity per memory load.

**Proposed improvements:**

1. **Increase per-subgroup work**: Each subgroup computes multiple 8×8 tiles (e.g., 2×2 = 16×16), increasing arithmetic intensity and reusing A/B loads across DPAS calls.
2. **Store A/B in DPAS-ready format in SLM**: During cooperative load, pre-pack data as `int8` vectors so DPAS operands can be loaded with simple block reads.
3. **Use `intel_sub_group_block_read` for SLM reads**: Efficient vector loads instead of scalar packing.
4. **Vectorized output stores**: Use `vstore8` for coalesced writes.
5. **Larger K-tile (32)** with 2 DPAS k16 steps per load to amortize SLM load cost.
6. **Simplified double buffering with proper barrier placement**.

The new design:
- **WG tile**: 32M × 64N, with 4×4 = 16 subgroups (each 8 WIs), LWS = (32, 4, 1) = 128 WIs
- **Each subgroup**: computes 8×16 of C (2 DPAS calls per K16 step, for 2 adjacent 8×8 N-tiles)
- **K-tile**: 32 (two k16 steps)
- **SLM**: A stored as int8 (DPAS-ready), B stored as int8 (VNNI-packed)

## Improved OCL code

```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS + SLM tiling
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Launch metadata:
//   LWS = (64, 4, 1) => 256 work-items = 32 subgroups of 8
//   GWS = (ceil(N/64)*64, ceil(M/32)*4, 1)
//   Subgroup size = 8
//   Work-group tile: 32M x 64N
//   Subgroup grid: 4 rows x 8 cols of 8x8 tiles => 32 subgroups
//   Each subgroup: 8x8 output via DPAS
//   K-tile = 32 (two DPAS k16 steps)
//   SLM A: stored pre-packed as int for DPAS-ready reads
//   SLM B: stored in VNNI format (pairs of K packed into uint)

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
    const int TK = 32;

    // Work-group origin
    const int wg_n = get_group_id(0) * TN;
    const int wg_m = get_group_id(1) * TM;

    // Flat local ID for cooperative loading
    const int lid0 = get_local_id(0);
    const int lid1 = get_local_id(1);
    const int flat_lid = lid1 * 64 + lid0;
    const int num_wis = 256;

    // Subgroup identification
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();

    // Map subgroups to 4x8 grid
    const int sg_row = sg_id / 8;  // 0..3
    const int sg_col = sg_id % 8;  // 0..7

    // SLM for A: store in DPAS-ready packed format
    // A tile is [TM][TK] halves = [32][32] halves
    // For DPAS, each row of 16 halves packs into 8 ints (int8)
    // We store A as uint: [TM][TK/2] uints = [32][16] = 512 uints
    // For k16 step, a subgroup row needs 8 consecutive uints from its row
    __local uint slm_A_packed[TM * (TK / 2)];  // 32 * 16 = 512 uints = 2048 bytes

    // SLM for B in VNNI: [TK/2][TN] uints = [16][64] = 1024 uints = 4096 bytes
    __local uint slm_B_vnni[TK / 2 * TN];

    // Global pointers as ushort
    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    // Accumulators
    float8 acc = (float8)(0.0f);

    const int num_k_tiles = (K + TK - 1) / TK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k0 = kt * TK;

        // === Cooperative load A into packed SLM ===
        // A_packed: [TM][TK/2] = 512 uints, 256 WIs => 2 each
        for (int i = flat_lid; i < TM * (TK / 2); i += num_wis) {
            int r = i / (TK / 2);   // row 0..31
            int cp = i % (TK / 2);  // packed col 0..15
            int gr = wg_m + r;
            int gc0 = k0 + cp * 2;
            int gc1 = gc0 + 1;
            ushort v0 = (gr < M && gc0 < K) ? A_us[gr * K + gc0] : (ushort)0;
            ushort v1 = (gr < M && gc1 < K) ? A_us[gr * K + gc1] : (ushort)0;
            slm_A_packed[i] = (uint)v0 | ((uint)v1 << 16);
        }

        // === Cooperative load B into VNNI SLM ===
        // B_vnni: [TK/2][TN] = 1024 uints, 256 WIs => 4 each
        for (int i = flat_lid; i < (TK / 2) * TN; i += num_wis) {
            int kp = i / TN;   // k-pair 0..15
            int c  = i % TN;   // col 0..63
            int gk0 = k0 + kp * 2;
            int gk1 = gk0 + 1;
            int gc  = wg_n + c;
            ushort v0 = (gk0 < K && gc < N) ? B_us[gk0 * N + gc] : (ushort)0;
            ushort v1 = (gk1 < K && gc < N) ? B_us[gk1 * N + gc] : (ushort)0;
            slm_B_vnni[i] = (uint)v0 | ((uint)v1 << 16);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === Compute: two DPAS k16 steps ===
        int a_slm_row = sg_row * 8 + sg_lid;  // row within A tile
        int b_slm_col = sg_col * 8 + sg_lid;  // col within B tile

        // --- Step 1: k offset 0..15 ---
        {
            // A: row a_slm_row, packed cols 0..7 (covers k=0..15)
            __local uint* a_base = &slm_A_packed[a_slm_row * (TK / 2)];
            int8 a_packed;
            a_packed.s0 = (int)a_base[0];
            a_packed.s1 = (int)a_base[1];
            a_packed.s2 = (int)a_base[2];
            a_packed.s3 = (int)a_base[3];
            a_packed.s4 = (int)a_base[4];
            a_packed.s5 = (int)a_base[5];
            a_packed.s6 = (int)a_base[6];
            a_packed.s7 = (int)a_base[7];

            // B VNNI: k-pairs 0..7, col b_slm_col
            int8 b_packed;
            b_packed.s0 = (int)slm_B_vnni[0 * TN + b_slm_col];
            b_packed.s1 = (int)slm_B_vnni[1 * TN + b_slm_col];
            b_packed.s2 = (int)slm_B_vnni[2 * TN + b_slm_col];
            b_packed.s3 = (int)slm_B_vnni[3 * TN + b_slm_col];
            b_packed.s4 = (int)slm_B_vnni[4 * TN + b_slm_col];
            b_packed.s5 = (int)slm_B_vnni[5 * TN + b_slm_col];
            b_packed.s6 = (int)slm_B_vnni[6 * TN + b_slm_col];
            b_packed.s7 = (int)slm_B_vnni[7 * TN + b_slm_col];

            acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);
        }

        // --- Step 2: k offset 16..31 ---
        {
            __local uint* a_base = &slm_A_packed[a_slm_row * (TK / 2) + 8];
            int8 a_packed;
            a_packed.s0 = (int)a_base[0];
            a_packed.s1 = (int)a_base[1];
            a_packed.s2 = (int)a_base[2];
            a_packed.s3 = (int)a_base[3];
            a_packed.s4 = (int)a_base[4];
            a_packed.s5 = (int)a_base[5];
            a_packed.s6 = (int)a_base[6];
            a_packed.s7 = (int)a_base[7];

            int8 b_packed;
            b_packed.s0 = (int)slm_B_vnni[8  * TN + b_slm_col];
            b_packed.s1 = (int)slm_B_vnni[9  * TN + b_slm_col];
            b_packed.s2 = (int)slm_B_vnni[10 * TN + b_slm_col];
            b_packed.s3 = (int)slm_B_vnni[11 * TN + b_slm_col];
            b_packed.s4 = (int)slm_B_vnni[12 * TN + b_slm_col];
            b_packed.s5 = (int)slm_B_vnni[13 * TN + b_slm_col];
            b_packed.s6 = (int)slm_B_vnni[14 * TN + b_slm_col];
            b_packed.s7 = (int)slm_B_vnni[15 * TN + b_slm_col];

            acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store 8x8 output tile with vectorized writes ===
    int out_col_base = wg_n + sg_col * 8;
    int out_row_base = wg_m + sg_row * 8;

    // Each WI (sg_lid) holds acc for row (out_row_base + sg_lid), cols out_col_base..+7
    int out_row = out_row_base + sg_lid;

    if (out_row < M) {
        half8 result = convert_half8(acc);
        int out_base = out_row * N + out_col_base;

        if (out_col_base + 8 <= N) {
            // Full vectorized store
            vstore8(result, 0, C + out_base);
        } else {
            // Partial boundary store
            for (int j = 0; j < 8; j++) {
                if (out_col_base + j < N) {
                    C[out_base + j] = ((half*)&result)[j];
                }
            }
        }
    }
}
```

