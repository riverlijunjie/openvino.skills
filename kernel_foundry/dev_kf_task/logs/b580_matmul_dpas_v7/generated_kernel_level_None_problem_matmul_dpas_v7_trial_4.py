

## Analysis

Looking at the previous versions (all at 33.9ms), the key bottlenecks identified in the feedback are:

1. **Scalar repacking of A from SLM** - 16 scalar loads + pack ops per K-tile per lane
2. **Low arithmetic intensity** - only 1 DPAS per barrier, so barrier/SLM latency dominates
3. **Scalar stores** - using pointer casting instead of vector stores

My improvements:
- **Store A in packed uint form in SLM** (like B's VNNI format), eliminating scalar repack
- **Increase K blocking to 64** per SLM load, issuing 4 DPAS ops per barrier (4x better compute/sync ratio)
- **Increase tile size to 32x64** for better data reuse (each WG: 4x8 subgroups)
- **Use vector stores** for output with `vstore8`
- **Double-buffered SLM** to overlap loads with compute

## Improved OCL code

```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Launch metadata:
//   LWS = (64, 4, 1) => 256 work-items = 32 subgroups of 8
//   GWS = (ceil(N/64)*64, ceil(M/32)*4, 1)
//   Subgroup size = 8
//   Work-group tile: 32 rows x 64 cols of C
//   Subgroup grid: 4 rows x 8 cols of 8x8 tiles
//   K-blocking: 64 per iteration (4 DPAS k16 ops per barrier)
//   SLM per buffer: A_packed[32][32] uint + B_vnni[32][64] uint
//     = 32*32*4 + 32*64*4 = 4096 + 8192 = 12288 bytes per buffer
//   Double-buffered: ~24576 bytes total SLM

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
    // WG tile: 32 rows x 64 cols
    // K-block: 64 (4 DPAS k16 steps per SLM load)
    // Subgroup grid: 4x8 = 32 subgroups, each handles 8x8

    const int wg_col = get_group_id(0) * 64;
    const int wg_row = get_group_id(1) * 32;

    const int lid0 = get_local_id(0);  // 0..63
    const int lid1 = get_local_id(1);  // 0..3
    const int flat_lid = lid1 * 64 + lid0;  // 0..255

    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();

    // Subgroup position in 4x8 grid
    const int sg_row = sg_id / 8;  // 0..3
    const int sg_col = sg_id % 8;  // 0..7

    // SLM: A stored as packed uint (2 halfs per uint), B in VNNI uint format
    // A_packed: [32][32] uint = 32 rows x 64 K-cols, packed as 32 uints per row
    // B_vnni:   [32][64] uint = 32 k-pairs x 64 cols
    // Double buffered
    __local uint slm_A_packed[2][32 * 32];  // [buf][row * 32 + k_pair]
    __local uint slm_B_vnni[2][32 * 64];    // [buf][k_pair * 64 + col]

    float8 acc = (float8)(0.0f);

    int num_k_tiles = (K + 63) / 64;
    int buf = 0;

    // Helper macro-like inline: load a K-block of 64 into SLM buffer
    // A: 32 rows x 64 cols = 32*64 = 2048 halfs = 1024 uints
    // 256 threads => 4 uints each
    // B: 64 rows x 64 cols => but we store as 32 k-pairs x 64 cols = 2048 uints
    // 256 threads => 8 uints each

    // Preload first tile
    {
        int k0 = 0;
        // Load A packed: 32*32 = 1024 uints, 256 threads => 4 each
        for (int i = flat_lid; i < 32 * 32; i += 256) {
            int r = i >> 5;       // i / 32
            int kp = i & 31;     // i % 32, k-pair index
            int gr = wg_row + r;
            int gk = k0 + kp * 2;
            ushort v0 = (gr < M && gk < K)     ? as_ushort(A[gr * K + gk])     : (ushort)0;
            ushort v1 = (gr < M && gk + 1 < K) ? as_ushort(A[gr * K + gk + 1]) : (ushort)0;
            slm_A_packed[0][i] = (uint)v0 | ((uint)v1 << 16);
        }
        // Load B VNNI: 32*64 = 2048 uints, 256 threads => 8 each
        for (int i = flat_lid; i < 32 * 64; i += 256) {
            int kp = i >> 6;     // i / 64, k-pair index 0..31
            int c = i & 63;      // i % 64, column index
            int gk = k0 + kp * 2;
            int gc = wg_col + c;
            ushort v0 = (gk < K && gc < N)     ? as_ushort(B[gk * N + gc])       : (ushort)0;
            ushort v1 = (gk + 1 < K && gc < N) ? as_ushort(B[(gk + 1) * N + gc]) : (ushort)0;
            slm_B_vnni[0][i] = (uint)v0 | ((uint)v1 << 16);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int next_kt = kt + 1;
        int next_buf = buf ^ 1;

        // Prefetch next K-tile
        if (next_kt < num_k_tiles) {
            int k0_next = next_kt * 64;
            for (int i = flat_lid; i < 32 * 32; i += 256) {
                int r = i >> 5;
                int kp = i & 31;
                int gr = wg_row + r;
                int gk = k0_next + kp * 2;
                ushort v0 = (gr < M && gk < K)     ? as_ushort(A[gr * K + gk])     : (ushort)0;
                ushort v1 = (gr < M && gk + 1 < K) ? as_ushort(A[gr * K + gk + 1]) : (ushort)0;
                slm_A_packed[next_buf][i] = (uint)v0 | ((uint)v1 << 16);
            }
            for (int i = flat_lid; i < 32 * 64; i += 256) {
                int kp = i >> 6;
                int c = i & 63;
                int gk = k0_next + kp * 2;
                int gc = wg_col + c;
                ushort v0 = (gk < K && gc < N)     ? as_ushort(B[gk * N + gc])       : (ushort)0;
                ushort v1 = (gk + 1 < K && gc < N) ? as_ushort(B[(gk + 1) * N + gc]) : (ushort)0;
                slm_B_vnni[next_buf][i] = (uint)v0 | ((uint)v1 << 16);
            }
        }

        // Compute: 4 DPAS k16 ops covering 64 K-elements
        // A layout in SLM: slm_A_packed[buf][row * 32 + k_pair]
        //   row 0..31, k_pair 0..31 (each uint = 2 halfs along K)
        //   For DPAS step s (0..3): k_pairs [s*4 .. s*4+3] give 8 uints = int8
        //     Wait - k16 needs 16 halfs = 8 uints per row
        //   For step s: K range [s*16 .. s*16+15], k_pairs [s*8 .. s*8+7]

        int a_slm_row = sg_row * 8 + sg_lid;
        int b_slm_col = sg_col * 8 + sg_lid;

        // Unrolled 4 DPAS instructions
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            int a_base = a_slm_row * 32 + s * 8;
            int8 a_packed = (int8)(
                (int)slm_A_packed[buf][a_base + 0],
                (int)slm_A_packed[buf][a_base + 1],
                (int)slm_A_packed[buf][a_base + 2],
                (int)slm_A_packed[buf][a_base + 3],
                (int)slm_A_packed[buf][a_base + 4],
                (int)slm_A_packed[buf][a_base + 5],
                (int)slm_A_packed[buf][a_base + 6],
                (int)slm_A_packed[buf][a_base + 7]
            );

            int b_base_kp = s * 8;  // k-pair offset
            int8 b_packed = (int8)(
                (int)slm_B_vnni[buf][(b_base_kp + 0) * 64 + b_slm_col],
                (int)slm_B_vnni[buf][(b_base_kp + 1) * 64 + b_slm_col],
                (int)slm_B_vnni[buf][(b_base_kp + 2) * 64 + b_slm_col],
                (int)slm_B_vnni[buf][(b_base_kp + 3) * 64 + b_slm_col],
                (int)slm_B_vnni[buf][(b_base_kp + 4) * 64 + b_slm_col],
                (int)slm_B_vnni[buf][(b_base_kp + 5) * 64 + b_slm_col],
                (int)slm_B_vnni[buf][(b_base_kp + 6) * 64 + b_slm_col],
                (int)slm_B_vnni[buf][(b_base_kp + 7) * 64 + b_slm_col]
            );

            acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Store 8x8 output tile using vector store
    int out_row_base = wg_row + sg_row * 8;
    int out_col_base = wg_col + sg_col * 8;

    // Convert accumulator to half8
    half8 result = convert_half8(acc);

    // Each work-item writes one element per row (sg_lid selects column)
    // But acc layout: WI sg_lid holds row[sg_lid] of the 8x8 tile,
    // acc.s0..s7 are columns 0..7
    // So WI sg_lid has row (out_row_base + sg_lid), cols out_col_base..out_col_base+7

    int out_row = out_row_base + sg_lid;
    if (out_row < M) {
        int out_base = out_row * N + out_col_base;
        if (out_col_base + 8 <= N) {
            // Full vector store
            vstore8(result, 0, C + out_base);
        } else {
            // Partial store for edge
            for (int j = 0; j < 8 && (out_col_base + j) < N; j++) {
                C[out_base + j] = ((half*)&result)[j];
            }
        }
    }
}
```

