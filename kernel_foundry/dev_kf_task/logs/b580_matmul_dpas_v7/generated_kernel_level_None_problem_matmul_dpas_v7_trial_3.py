

## Analysis

Looking at the previous versions, all achieved 33.900ms. The key bottlenecks identified in feedback are:

1. **Scalar global loads with per-element bounds checking** in the hot loop - need vectorized loads for interior tiles
2. **Manual DPAS operand packing in the compute loop** - should pre-pack during SLM write phase
3. **Single-buffered SLM** causing pipeline stalls - need double-buffering

My strategy for improvement:
- **Larger tile**: TM=32, TN=64, TK=32 with double-buffered SLM, pre-packed in VNNI/DPAS format
- **Fast-path for interior tiles**: Skip bounds checks when the entire WG tile is within bounds (vectorized loads)
- **Pre-pack A and B into DPAS-ready format during SLM store**, so the compute loop does simple int8 loads
- **Double-buffered SLM ping-pong** to overlap next tile load with current compute
- **Each subgroup computes 8x16 output** (2 DPAS per k-step) to increase compute density per subgroup

## Improved OCL code

```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Work-group tile: TM=32, TN=64, TK=16
// Subgroup layout: 4 along M x 8 along N = 32 subgroups, each 8x8 output
// LWS = (64, 4, 1) => 256 work-items = 32 subgroups of 8
// GWS = (ceil(N/64)*64, ceil(M/32)*4, 1)
// Subgroup size = 8
//
// Optimizations:
//   - Double-buffered SLM to overlap load with compute
//   - Pre-packed DPAS operands in SLM (A as int per 2 halfs, B in VNNI)
//   - Fast-path vectorized loads for interior tiles (no bounds checks)
//   - Vectorized output stores

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
    // Tile config
    const int TM = 32;
    const int TN = 64;
    const int TK = 16;

    const int wg_n = get_group_id(0) * TN;
    const int wg_m = get_group_id(1) * TM;

    const int lid0 = get_local_id(0);  // 0..63
    const int lid1 = get_local_id(1);  // 0..3
    const int flat_lid = lid1 * 64 + lid0;  // 0..255

    const int sg_local_id = get_sub_group_local_id(); // 0..7
    const int sg_x = lid0 / 8;  // 0..7 (along N dimension)
    const int sg_y = lid1;       // 0..3 (along M dimension)

    // Output tile origin for this subgroup's 8x8 block
    const int tile_m = wg_m + sg_y * 8;
    const int tile_n = wg_n + sg_x * 8;

    // Double-buffered SLM in pre-packed DPAS format:
    // A: [2][TM][TK/2] as uint (each uint = 2 packed halfs along K)
    //    Layout: a_slm[buf][row * 8 + k_pair], row in [0,31], k_pair in [0,7]
    // B: [2][TK/2][TN] as uint (VNNI: each uint = pack(B[k*2], B[k*2+1]) for same col)
    //    Layout: b_slm[buf][k_pair * 64 + col], k_pair in [0,7], col in [0,63]
    __local uint slm_A[2][32 * 8];    // 2 * 32 * 8 * 4 = 2048 bytes
    __local uint slm_B[2][8 * 64];    // 2 * 8 * 64 * 4 = 4096 bytes

    float8 acc = (float8)(0.0f);

    // Check if this is an interior tile (no bounds checking needed)
    const int is_interior = (wg_m + TM <= M) && (wg_n + TN <= N);

    const int num_k_tiles = (K + TK - 1) / TK;
    int buf = 0;

    // === Macro-like functions for loading tiles ===
    // Load A tile: A[wg_m..+31][k0..+15] -> slm_A[buf] pre-packed
    // 32 rows * 8 uint pairs = 256 elements, 256 WIs => 1 each
    #define LOAD_A_TILE(BUF, K0) \
    { \
        int idx = flat_lid; \
        if (idx < 256) { \
            int r = idx >> 3;       /* idx / 8, row 0..31 */ \
            int kp = idx & 7;       /* idx % 8, k-pair 0..7 */ \
            int gr = wg_m + r; \
            int gk0 = (K0) + kp * 2; \
            int gk1 = gk0 + 1; \
            ushort v0, v1; \
            if (is_interior && gk1 < K) { \
                v0 = as_ushort(A[gr * K + gk0]); \
                v1 = as_ushort(A[gr * K + gk1]); \
            } else { \
                v0 = (gr < M && gk0 < K) ? as_ushort(A[gr * K + gk0]) : (ushort)0; \
                v1 = (gr < M && gk1 < K) ? as_ushort(A[gr * K + gk1]) : (ushort)0; \
            } \
            slm_A[BUF][r * 8 + kp] = (uint)v0 | ((uint)v1 << 16); \
        } \
    }

    // Load B tile: B[k0..+15][wg_n..+63] -> slm_B[buf] VNNI-packed
    // 8 k-pairs * 64 cols = 512 elements, 256 WIs => 2 each
    #define LOAD_B_TILE(BUF, K0) \
    { \
        for (int idx = flat_lid; idx < 512; idx += 256) { \
            int kp = idx >> 6;      /* idx / 64, k-pair 0..7 */ \
            int c = idx & 63;       /* idx % 64, col 0..63 */ \
            int gk0 = (K0) + kp * 2; \
            int gk1 = gk0 + 1; \
            int gc = wg_n + c; \
            ushort v0, v1; \
            if (is_interior && gk1 < K) { \
                v0 = as_ushort(B[gk0 * N + gc]); \
                v1 = as_ushort(B[gk1 * N + gc]); \
            } else { \
                v0 = (gk0 < K && gc < N) ? as_ushort(B[gk0 * N + gc]) : (ushort)0; \
                v1 = (gk1 < K && gc < N) ? as_ushort(B[gk1 * N + gc]) : (ushort)0; \
            } \
            slm_B[BUF][kp * 64 + c] = (uint)v0 | ((uint)v1 << 16); \
        } \
    }

    // Preload first tile
    LOAD_A_TILE(0, 0);
    LOAD_B_TILE(0, 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int next_kt = kt + 1;
        int next_buf = buf ^ 1;

        // Prefetch next tile into next_buf
        if (next_kt < num_k_tiles) {
            int k0_next = next_kt * TK;
            LOAD_A_TILE(next_buf, k0_next);
            LOAD_B_TILE(next_buf, k0_next);
        }

        // === Compute DPAS on current buffer ===
        // A operand: row = sg_y*8 + sg_local_id, 8 packed ints
        int a_row = sg_y * 8 + sg_local_id;
        int a_base = a_row * 8;
        int8 a_packed = (int8)(
            slm_A[buf][a_base + 0], slm_A[buf][a_base + 1],
            slm_A[buf][a_base + 2], slm_A[buf][a_base + 3],
            slm_A[buf][a_base + 4], slm_A[buf][a_base + 5],
            slm_A[buf][a_base + 6], slm_A[buf][a_base + 7]
        );

        // B operand: col = sg_x*8 + sg_local_id, 8 k-pairs
        int b_col = sg_x * 8 + sg_local_id;
        int8 b_packed = (int8)(
            (int)slm_B[buf][0 * 64 + b_col],
            (int)slm_B[buf][1 * 64 + b_col],
            (int)slm_B[buf][2 * 64 + b_col],
            (int)slm_B[buf][3 * 64 + b_col],
            (int)slm_B[buf][4 * 64 + b_col],
            (int)slm_B[buf][5 * 64 + b_col],
            (int)slm_B[buf][6 * 64 + b_col],
            (int)slm_B[buf][7 * 64 + b_col]
        );

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // === Store 8x8 output tile ===
    int out_row = tile_m + sg_local_id;
    if (out_row < M) {
        half8 result = convert_half8(acc);
        int out_base = out_row * N + tile_n;

        if (tile_n + 8 <= N) {
            vstore8(result, 0, C + out_base);
        } else {
            for (int j = 0; j < 8; j++) {
                if (tile_n + j < N) {
                    C[out_base + j] = ((half*)&result)[j];
                }
            }
        }
    }
}
```

