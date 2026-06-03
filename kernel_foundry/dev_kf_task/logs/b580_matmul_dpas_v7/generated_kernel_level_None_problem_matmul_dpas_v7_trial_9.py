

## Analysis

Looking at the previous versions, all achieve ~33.9ms. The key bottlenecks identified in the feedback are:

1. **Scalar SLM loads for B**: Building `int8` vectors from individual SLM reads is expensive. We need a layout where `vload8` works directly.
2. **No double-buffering**: Load and compute are serialized by barriers each K-step.
3. **Branchy store epilogue**: Multiple conditional stores add overhead.

My plan:
- Use **TM=32, TN=64** tile with **LWS=(16,4,1)** = 64 WIs = 8 subgroups of 8
- Each subgroup computes **8 rows × 32 cols** (4 DPAS calls per K-step) - register blocking
- **Double-buffered SLM** to overlap load and compute
- **Transpose B in SLM** to column-major `[col][kpair]` layout so `vload8` works for B operand
- A stored as `[row][8]` ints (already contiguous for `vload8`)
- Minimize barriers with ping-pong buffering

## Improved OCL code

```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS + double-buffered SLM
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Launch metadata:
//   LWS = (16, 4, 1) => 64 work-items = 8 subgroups of 8
//   GWS = (ceil(N/64)*16, ceil(M/32)*4, 1)
//   Subgroup size = 8
//   Work-group tile: TM=32, TN=64
//   Subgroup grid: 2 along N x 4 along M
//   Each subgroup: 8 rows x 32 cols (4 DPAS per K-step)
//   Double-buffered SLM:
//     A: [2][32][8] ints = 2*1024 = 2048 bytes
//     B: [2][64][8] ints (col-major for vload8) = 2*2048 = 4096 bytes
//   Total SLM: 6144 bytes

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
    const int TM = 32;
    const int TN = 64;
    const int TK = 16;
    const int WG_SIZE = 64;

    const int wg_n = get_group_id(0) * TN;
    const int wg_m = get_group_id(1) * TM;

    const int lid0 = get_local_id(0);
    const int lid1 = get_local_id(1);
    const int flat_lid = lid1 * 16 + lid0;

    const int sg_local_id = get_sub_group_local_id();
    const int sg_id = get_sub_group_id();

    // Subgroup grid: 2 along N, 4 along M
    const int sg_n = sg_id & 1;   // 0..1
    const int sg_m = sg_id >> 1;  // 0..3

    const int tile_m = wg_m + sg_m * 8;
    const int tile_n = wg_n + sg_n * 32;

    // Double-buffered SLM
    // A: [buf][row][8 ints] - row-major, each row = 16 halfs packed as 8 ints
    // B: [buf][col][8 ints] - col-major! Each col = 8 k-pairs packed as 8 ints
    //    This allows vload8 for B operand
    __local int slm_A[2][32 * 8];     // 2 * 256 ints = 2048 bytes
    __local int slm_B[2][64 * 8];     // 2 * 512 ints = 4096 bytes

    // 4 accumulators for 4 column-groups of 8
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TK - 1) / TK;

    // === Macro for loading a tile into SLM buffer ===
    // Load A: 32*8 = 256 ints, 64 WIs => 4 each
    // Load B: 64*8 = 512 ints, 64 WIs => 8 each
    // B stored col-major: slm_B[buf][col * 8 + kpair]

    #define LOAD_TILE(buf_idx, k0_val) \
    { \
        /* Load A */ \
        for (int i = flat_lid; i < 32 * 8; i += WG_SIZE) { \
            int r = i >> 3; \
            int p = i & 7; \
            int gr = wg_m + r; \
            int gc0 = (k0_val) + p * 2; \
            int gc1 = gc0 + 1; \
            ushort v0 = (gr < M && gc0 < K) ? A_us[gr * K + gc0] : (ushort)0; \
            ushort v1 = (gr < M && gc1 < K) ? A_us[gr * K + gc1] : (ushort)0; \
            slm_A[buf_idx][r * 8 + p] = (int)v0 | ((int)v1 << 16); \
        } \
        /* Load B in col-major layout: slm_B[buf][col * 8 + kpair] */ \
        for (int i = flat_lid; i < 64 * 8; i += WG_SIZE) { \
            int c = i >> 3;  /* col 0..63 */ \
            int kp = i & 7;  /* k-pair 0..7 */ \
            int gk0 = (k0_val) + kp * 2; \
            int gk1 = gk0 + 1; \
            int gc = wg_n + c; \
            ushort v0 = (gk0 < K && gc < N) ? B_us[gk0 * N + gc] : (ushort)0; \
            ushort v1 = (gk1 < K && gc < N) ? B_us[gk1 * N + gc] : (ushort)0; \
            slm_B[buf_idx][c * 8 + kp] = (int)v0 | ((int)v1 << 16); \
        } \
    }

    // Preload first tile into buffer 0
    int buf = 0;
    if (num_k_tiles > 0) {
        LOAD_TILE(0, 0);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int next_buf = buf ^ 1;

        // Prefetch next K-tile into next_buf (overlaps with compute below after barrier)
        if (kt + 1 < num_k_tiles) {
            int k0_next = (kt + 1) * TK;
            LOAD_TILE(next_buf, k0_next);
        }

        // === Compute on current buffer ===
        // Load A: each WI loads its row's 16 halfs as int8 via vload8
        int a_row = sg_m * 8 + sg_local_id;
        int8 a_packed = vload8(0, &slm_A[buf][a_row * 8]);

        // Load B: col-major layout allows vload8 per column
        // Each of 4 column-groups needs int8 from slm_B[buf][(sg_n*32 + group*8 + sg_local_id) * 8]
        int b_base = sg_n * 32;

        int8 b0 = vload8(0, &slm_B[buf][(b_base + 0  + sg_local_id) * 8]);
        int8 b1 = vload8(0, &slm_B[buf][(b_base + 8  + sg_local_id) * 8]);
        int8 b2 = vload8(0, &slm_B[buf][(b_base + 16 + sg_local_id) * 8]);
        int8 b3 = vload8(0, &slm_B[buf][(b_base + 24 + sg_local_id) * 8]);

        // 4 DPAS calls - register blocking along N dimension
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b2, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b3, acc3);

        // Single barrier: ensures next tile load is complete before we swap
        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    #undef LOAD_TILE

    // === Store 8x32 output tile ===
    int out_row = tile_m + sg_local_id;
    if (out_row >= M) return;

    int base = out_row * N;
    bool full_n = (tile_n + 32 <= N);

    if (full_n) {
        // Fast path: all 32 columns are valid
        vstore8(convert_half8(acc0), 0, C + base + tile_n);
        vstore8(convert_half8(acc1), 0, C + base + tile_n + 8);
        vstore8(convert_half8(acc2), 0, C + base + tile_n + 16);
        vstore8(convert_half8(acc3), 0, C + base + tile_n + 24);
    } else {
        // Slow path: boundary handling
        half8 r0 = convert_half8(acc0);
        half8 r1 = convert_half8(acc1);
        half8 r2 = convert_half8(acc2);
        half8 r3 = convert_half8(acc3);

        for (int j = 0; j < 8; j++) {
            int c = tile_n + j;
            if (c < N) C[base + c] = ((half*)&r0)[j];
        }
        for (int j = 0; j < 8; j++) {
            int c = tile_n + 8 + j;
            if (c < N) C[base + c] = ((half*)&r1)[j];
        }
        for (int j = 0; j < 8; j++) {
            int c = tile_n + 16 + j;
            if (c < N) C[base + c] = ((half*)&r2)[j];
        }
        for (int j = 0; j < 8; j++) {
            int c = tile_n + 24 + j;
            if (c < N) C[base + c] = ((half*)&r3)[j];
        }
    }
}
```

