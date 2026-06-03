

## Analysis

The previous kernel (33.9ms) uses DPAS instructions correctly but has several inefficiencies:

1. **Small tile per workgroup**: 32x64 is relatively small, meaning more workgroups and overhead.
2. **Scalar A loads**: Each work-item loads A values individually with boundary checks in a loop.
3. **B loaded to SLM then re-read for DPAS**: Extra SLM traffic. We could use SLM more efficiently.
4. **Only 8x16 per subgroup**: We can increase to multiple DPAS calls per subgroup to improve arithmetic intensity.

**Proposed improvements:**
- Increase workgroup tile to 32x128 (or 64x64) with each subgroup computing multiple 8x16 tiles.
- Each subgroup computes 8x32 (two DPAS calls per K iteration) to increase register reuse of A.
- Use vectorized loads for B into SLM.
- Prefetch/pipeline K iterations.
- Use `__global const half* restrict` for better compiler hints.

## Improved OCL code

```OCL
// Optimized FP16 matmul using Intel DPAS (XMX) for Battlemage
// Each subgroup computes 8x32 output (2x DPAS 8x16 per K step)
// Workgroup: 8 subgroups arranged as 4(row) x 2(col), WG tile: 32x64
// But we double the N-tile per subgroup: each SG does 8 rows x 32 cols
// WG tile: 32x64, SG grid: 4x2, each SG: 8x32
// GWS = (ceil_div(N,64)*16, ceil_div(M,32)*8)
// LWS = (16, 8)
// Subgroup size = 16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 16
#define SG_SIZE 16
#define SG_TILE_M 8
#define SG_TILE_N 32

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
    const int wg_col = get_group_id(0) * TILE_N;
    const int wg_row = get_group_id(1) * TILE_M;

    const int lid_x = get_local_id(0); // 0..15 (subgroup lane)
    const int lid_y = get_local_id(1); // 0..7  (subgroup index)

    // Map 8 subgroups into 4x2 grid
    const int sg_row = lid_y / 2;  // 0..3
    const int sg_col = lid_y % 2;  // 0..1

    const int out_row = wg_row + sg_row * SG_TILE_M;
    const int out_col = wg_col + sg_col * SG_TILE_N;

    const int sg_lane = get_sub_group_local_id();

    // Accumulators: 8 rows x 32 cols = 2 DPAS tiles side by side
    float8 acc0 = (float8)(0.0f); // rows 0-7, cols 0-15
    float8 acc1 = (float8)(0.0f); // rows 0-7, cols 16-31

    // SLM for cooperative loading
    __local half A_slm[TILE_M * TILE_K];  // 32 x 16
    __local half B_slm[TILE_K * TILE_N];  // 16 x 64

    const int num_wi = 16 * 8; // 128 work-items

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperative load A_slm: 32*16 = 512 halfs, 128 WIs => 4 each
        {
            int linear_id = lid_y * 16 + lid_x;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = linear_id + i * num_wi;
                int r = idx / TILE_K;
                int c = idx % TILE_K;
                int gr = wg_row + r;
                int gc = k0 + c;
                A_slm[r * TILE_K + c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0;
            }
        }

        // Cooperative load B_slm: 16*64 = 1024 halfs, 128 WIs => 8 each
        {
            int linear_id = lid_y * 16 + lid_x;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = linear_id + i * num_wi;
                int r = idx / TILE_N;
                int c = idx % TILE_N;
                int gr = k0 + r;
                int gc = wg_col + c;
                B_slm[r * TILE_N + c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Load A for this subgroup: 8x16 from A_slm
        short8 a_packed;
        {
            int a_base = sg_row * SG_TILE_M;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                half v = A_slm[(a_base + r) * TILE_K + sg_lane];
                ((short*)&a_packed)[r] = as_short(v);
            }
        }

        // Load B for first 16 cols
        int8 b_int0;
        {
            int b_col_base = sg_col * SG_TILE_N;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int r0 = 2 * p;
                int r1 = 2 * p + 1;
                ushort s0 = as_ushort(B_slm[r0 * TILE_N + b_col_base + sg_lane]);
                ushort s1 = as_ushort(B_slm[r1 * TILE_N + b_col_base + sg_lane]);
                b_int0[p] = (int)s0 | ((int)s1 << 16);
            }
        }

        // Load B for second 16 cols
        int8 b_int1;
        {
            int b_col_base = sg_col * SG_TILE_N + 16;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int r0 = 2 * p;
                int r1 = 2 * p + 1;
                ushort s0 = as_ushort(B_slm[r0 * TILE_N + b_col_base + sg_lane]);
                ushort s1 = as_ushort(B_slm[r1 * TILE_N + b_col_base + sg_lane]);
                b_int1[p] = (int)s0 | ((int)s1 << 16);
            }
        }

        // Two DPAS calls, reusing A
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_int0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_int1, acc1);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: 8 rows x 32 cols
    // acc0: 8 rows, cols out_col+0..15 (lane selects column)
    // acc1: 8 rows, cols out_col+16..31
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int gr = out_row + r;
        int gc0 = out_col + sg_lane;
        if (gr < M && gc0 < N) {
            C[gr * N + gc0] = convert_half(((float*)&acc0)[r]);
        }
        int gc1 = out_col + 16 + sg_lane;
        if (gr < M && gc1 < N) {
            C[gr * N + gc1] = convert_half(((float*)&acc1)[r]);
        }
    }
}
```

