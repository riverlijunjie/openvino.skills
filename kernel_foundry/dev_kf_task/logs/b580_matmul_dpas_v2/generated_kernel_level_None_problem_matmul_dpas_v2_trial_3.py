

## Analysis

The best kernel so far (27.3ms) uses:
- WG tile: 32x64, SG grid: 4x4, each SG: 8x16
- TILE_K=32 (two DPAS calls per SLM load)
- SLM with padding for A and B

The slower kernel (40.2ms) uses:
- WG tile: 32x64, SG grid: 4x2, each SG: 8x32 (two DPAS per SG)
- TILE_K=16 (only one K step per SLM load)
- Fewer work-items (128 vs 256), so more SLM loads per WI and less cooperative loading bandwidth

Key observations:
1. Larger TILE_K amortizes SLM load cost over more compute
2. The 4x4 SG grid with 256 WIs loads data faster cooperatively
3. We can try increasing TILE_K further (e.g., 64) to get more compute per barrier
4. We can try larger WG tiles (e.g., 64x64) to improve data reuse
5. Using `intel_sub_group_block_read` for SLM reads could be faster than scalar reads

Let me try TILE_K=64 with the best config to get 4x DPAS calls per SLM load, improving compute-to-memory ratio.

## Improved OCL code

```OCL
// Optimized FP16 matmul using Intel DPAS (XMX) for Battlemage
// WG tile: 32x64, SG grid: 4x4, each SG: 8x16
// TILE_K=64: 4 DPAS calls per SLM load to maximize compute/memory ratio
// GWS = (ceil_div(N,64)*16, ceil_div(M,32)*16)
// LWS = (16, 16)
// Subgroup size = 16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 64
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

    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);

    const int sg_row = lid_y / 4;
    const int sg_col = lid_y % 4;

    const int out_row = wg_row + sg_row * SG_TILE_M;
    const int out_col = wg_col + sg_col * SG_TILE_N;

    const int sg_lane = get_sub_group_local_id();

    float8 acc = (float8)(0.0f);

    // SLM with padding
    const int A_stride = TILE_K + A_PAD;  // 65
    const int B_stride = TILE_N + B_PAD;  // 65
    __local half A_slm[TILE_M * (TILE_K + A_PAD)];  // 32 x 65
    __local half B_slm[TILE_K * (TILE_N + B_PAD)];  // 64 x 65

    const int linear_id = lid_y * 16 + lid_x; // 0..255
    const int num_wi = 256;

    // A: 32*64 = 2048 halfs => 8 per WI
    // B: 64*64 = 4096 halfs => 16 per WI

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A: 32 x 64 = 2048, 8 per WI
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = linear_id + i * num_wi;
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int gr = wg_row + r;
            int gc = k0 + c;
            A_slm[r * A_stride + c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0;
        }

        // Load B: 64 x 64 = 4096, 16 per WI
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int idx = linear_id + i * num_wi;
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gr = k0 + r;
            int gc = wg_col + c;
            B_slm[r * B_stride + c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 4 DPAS calls for K=64 (4 x K=16 blocks)
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load A: 8 rows x 16 K from SLM
            short8 a_packed;
            int a_base = sg_row * SG_TILE_M;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                a_packed[r] = as_short(A_slm[(a_base + r) * A_stride + kk + sg_lane]);
            }

            // Load B: 16 K x 16 N from SLM, packed as int8
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

    // Store results
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

