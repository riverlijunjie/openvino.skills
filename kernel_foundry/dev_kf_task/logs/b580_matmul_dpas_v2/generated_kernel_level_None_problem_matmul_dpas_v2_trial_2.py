

## Analysis

The best kernel so far (33.9ms) uses DPAS with a 32x64 WG tile, 16 subgroups (4x4 grid), each computing 8x16 output. The 40.2ms version uses 8 subgroups (4x2), each computing 8x32 (two DPAS calls per K step). The slowest (53.2ms) doesn't use DPAS at all.

Key observations:
1. The best version loads A directly from global memory per subgroup (redundant reads across sg_col), while B goes through SLM. This is suboptimal - A should also be in SLM for reuse.
2. The best version has 16 subgroups but only 4x4=16 mapping, which is good occupancy.
3. Version 1 (40.2ms) does 2 DPAS per K step per subgroup, reusing A - good idea but SLM loading is the bottleneck with more elements per WI.

**Improvements for new version:**
- Use larger WG tile (32x128) with 32 subgroups (4x8), each doing 8x16 via DPAS
- Keep A in SLM for reuse across column subgroups
- Pad SLM to avoid bank conflicts
- Use `restrict` and `#pragma unroll` throughout
- Double-buffer K loop iterations to overlap compute and memory

Let me try a refined approach: 32x64 WG tile like the best, but with A also in SLM, proper padding, and unrolled cooperative loads using contiguous access patterns.

```OCL
// Optimized FP16 matmul using Intel DPAS (XMX) for Battlemage
// WG tile: 32x64, SG grid: 4x4, each SG: 8x16
// A and B both loaded into SLM with padding to avoid bank conflicts
// GWS = (ceil_div(N,64)*16, ceil_div(M,32)*16)
// LWS = (16, 16)
// Subgroup size = 16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define SG_TILE_M 8
#define SG_TILE_N 16
#define A_PAD 1
#define B_PAD 1

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int wg_col = get_group_id(0) * TILE_N;
    const int wg_row = get_group_id(1) * TILE_M;

    const int lid_x = get_local_id(0); // 0..15
    const int lid_y = get_local_id(1); // 0..15

    // 16 subgroups in 4x4 grid
    const int sg_row = lid_y / 4;  // 0..3
    const int sg_col = lid_y % 4;  // 0..3

    const int out_row = wg_row + sg_row * SG_TILE_M;
    const int out_col = wg_col + sg_col * SG_TILE_N;

    const int sg_lane = get_sub_group_local_id();

    // Accumulators
    float8 acc = (float8)(0.0f);

    // SLM with padding to reduce bank conflicts
    __local half A_slm[TILE_M * (TILE_K + A_PAD)];  // 32 x 33
    __local half B_slm[TILE_K * (TILE_N + B_PAD)];  // 32 x 65

    const int A_stride = TILE_K + A_PAD;
    const int B_stride = TILE_N + B_PAD;

    const int linear_id = lid_y * 16 + lid_x; // 0..255
    const int num_wi = 256;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A: 32 x 32 = 1024 halfs, 256 WIs => 4 each
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = linear_id + i * num_wi;
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int gr = wg_row + r;
            int gc = k0 + c;
            A_slm[r * A_stride + c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0;
        }

        // Load B: 32 x 64 = 2048 halfs, 256 WIs => 8 each
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = linear_id + i * num_wi;
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gr = k0 + r;
            int gc = wg_col + c;
            B_slm[r * B_stride + c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Process two K=16 blocks within this K=32 tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load A: 8 rows x 16 K
            short8 a_packed;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int a_row = sg_row * SG_TILE_M + r;
                a_packed[r] = as_short(A_slm[a_row * A_stride + kk + sg_lane]);
            }

            // Load B: 16 K x 16 N, packed as int8
            int8 b_int;
            int b_col_base = sg_col * SG_TILE_N;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int r0 = kk + 2 * p;
                int r1 = kk + 2 * p + 1;
                ushort s0 = as_ushort(B_slm[r0 * B_stride + b_col_base + sg_lane]);
                ushort s1 = as_ushort(B_slm[r1 * B_stride + b_col_base + sg_lane]);
                b_int[p] = (int)s0 | ((int)s1 << 16);
            }

            acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_int, acc);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: 8 rows x 16 cols
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int gr = out_row + r;
        int gc = out_col + sg_lane;
        if (gr < M && gc < N) {
            C[gr * N + gc] = convert_half(acc[r]);
        }
    }
}
```

