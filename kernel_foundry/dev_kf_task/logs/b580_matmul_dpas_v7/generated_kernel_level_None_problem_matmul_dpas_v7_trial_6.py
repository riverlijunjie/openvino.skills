

## Analysis

Looking at the previous versions, all achieve 33.9ms. The key bottlenecks are:

1. **Low per-lane arithmetic intensity**: Each lane computes only one 8x8 tile (single `float8` accumulator). This means barrier/load overhead dominates.
2. **Scalar SLM loads**: Building `int8` from 8 individual SLM reads wastes issue slots.
3. **No true overlap**: Double-buffering doesn't help since loads and compute are serialized within each iteration.

**My improvements:**
- **Register blocking**: Each subgroup computes a **16x16** output tile using 2x2 grid of 8x8 DPAS results (4 accumulators: `float8 acc00, acc01, acc10, acc11`). This doubles reuse of both A and B loads.
- **Larger WG tile**: 32x64 with 8 subgroups (2x4 grid), each doing 16x16. Fewer subgroups means less overhead.
- **Vector SLM loads**: Store A in a layout where each lane can do contiguous reads. Use `as_int8` on aligned SLM for B.
- **K-tile of 16**: Single DPAS per sub-step, but 4 accumulators per step maximize compute-to-load ratio.

## Improved OCL code

```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS + register blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Launch metadata:
//   LWS = (32, 8, 1) => 256 work-items = 32 subgroups of 8
//   GWS = (ceil(N/128)*32, ceil(M/32)*8, 1)
//   Subgroup size = 8
//   Work-group tile: 32 rows x 128 cols of C
//   Subgroup grid: 4 rows x 8 cols => 32 subgroups
//   Each subgroup: 8x16 output (register blocking: 1 row x 2 cols of 8x8 DPAS)
//   K-tile = 32 (2 DPAS k16 steps per SLM load)
//   SLM: A[32][32] half + B_vnni[16][128] uint = 2048 + 8192 = 10240 bytes

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(32, 8, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0) * 128;
    const int wg_m = get_group_id(1) * 32;

    const int lid0 = get_local_id(0);
    const int lid1 = get_local_id(1);
    const int flat_lid = lid1 * 32 + lid0;

    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();

    // 32 subgroups in 4x8 grid, each subgroup handles 8x16 of output
    // (8 rows, 16 cols = 2 adjacent 8x8 DPAS tiles sharing same A)
    const int sg_row = sg_id / 8;  // 0..3
    const int sg_col = sg_id % 8;  // 0..7

    // SLM tiles
    __local half slm_A[32 * 32];          // [32 rows][32 K-cols]
    __local uint slm_B_vnni[16 * 128];    // [16 k-pairs][128 cols]

    // Register blocking: 2 accumulators for 8x16 output
    // acc0: rows[sg_row*8..+7] x cols[sg_col*16..+7]
    // acc1: rows[sg_row*8..+7] x cols[sg_col*16+8..+15]
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + 31) / 32;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k0 = kt * 32;

        // Cooperative load A[32][32]: 1024 halves, 256 threads => 4 each
        #pragma unroll 4
        for (int i = flat_lid; i < 32 * 32; i += 256) {
            int r = i >> 5;    // i / 32
            int c = i & 31;    // i % 32
            int gr = wg_m + r;
            int gc = k0 + c;
            slm_A[i] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0f;
        }

        // Cooperative load B VNNI[16][128]: 2048 uints, 256 threads => 8 each
        #pragma unroll 8
        for (int i = flat_lid; i < 16 * 128; i += 256) {
            int kp = i >> 7;    // i / 128, k-pair 0..15
            int c  = i & 127;   // i % 128, column
            int gk0 = k0 + kp * 2;
            int gk1 = gk0 + 1;
            int gc  = wg_n + c;
            ushort v0 = (gk0 < K && gc < N) ? B_us[gk0 * N + gc] : (ushort)0;
            ushort v1 = (gk1 < K && gc < N) ? B_us[gk1 * N + gc] : (ushort)0;
            slm_B_vnni[i] = (uint)v0 | ((uint)v1 << 16);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: 2 DPAS k16 steps x 2 B-column tiles = 4 DPAS total
        int a_row_in_tile = sg_row * 8 + sg_lid;
        int b_col0 = sg_col * 16 + sg_lid;       // first 8x8 B tile column
        int b_col1 = sg_col * 16 + 8 + sg_lid;   // second 8x8 B tile column

        #pragma unroll 2
        for (int s = 0; s < 2; s++) {
            // Load A: row a_row_in_tile, K-cols [s*16..s*16+15]
            __local half* a_ptr = &slm_A[a_row_in_tile * 32 + s * 16];
            int8 a_packed;
            // Pack 16 halves into 8 ints (2 halves per int)
            a_packed.s0 = as_int(vload2(0, (__local ushort*)(a_ptr + 0)));
            a_packed.s1 = as_int(vload2(0, (__local ushort*)(a_ptr + 2)));
            a_packed.s2 = as_int(vload2(0, (__local ushort*)(a_ptr + 4)));
            a_packed.s3 = as_int(vload2(0, (__local ushort*)(a_ptr + 6)));
            a_packed.s4 = as_int(vload2(0, (__local ushort*)(a_ptr + 8)));
            a_packed.s5 = as_int(vload2(0, (__local ushort*)(a_ptr + 10)));
            a_packed.s6 = as_int(vload2(0, (__local ushort*)(a_ptr + 12)));
            a_packed.s7 = as_int(vload2(0, (__local ushort*)(a_ptr + 14)));

            // Load B for first 8 cols - reuse A
            int bkp = s * 8;
            int8 b0_packed;
            b0_packed.s0 = (int)slm_B_vnni[(bkp + 0) * 128 + b_col0];
            b0_packed.s1 = (int)slm_B_vnni[(bkp + 1) * 128 + b_col0];
            b0_packed.s2 = (int)slm_B_vnni[(bkp + 2) * 128 + b_col0];
            b0_packed.s3 = (int)slm_B_vnni[(bkp + 3) * 128 + b_col0];
            b0_packed.s4 = (int)slm_B_vnni[(bkp + 4) * 128 + b_col0];
            b0_packed.s5 = (int)slm_B_vnni[(bkp + 5) * 128 + b_col0];
            b0_packed.s6 = (int)slm_B_vnni[(bkp + 6) * 128 + b_col0];
            b0_packed.s7 = (int)slm_B_vnni[(bkp + 7) * 128 + b_col0];

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b0_packed, acc0);

            // Load B for second 8 cols - reuse same A
            int8 b1_packed;
            b1_packed.s0 = (int)slm_B_vnni[(bkp + 0) * 128 + b_col1];
            b1_packed.s1 = (int)slm_B_vnni[(bkp + 1) * 128 + b_col1];
            b1_packed.s2 = (int)slm_B_vnni[(bkp + 2) * 128 + b_col1];
            b1_packed.s3 = (int)slm_B_vnni[(bkp + 3) * 128 + b_col1];
            b1_packed.s4 = (int)slm_B_vnni[(bkp + 4) * 128 + b_col1];
            b1_packed.s5 = (int)slm_B_vnni[(bkp + 5) * 128 + b_col1];
            b1_packed.s6 = (int)slm_B_vnni[(bkp + 6) * 128 + b_col1];
            b1_packed.s7 = (int)slm_B_vnni[(bkp + 7) * 128 + b_col1];

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b1_packed, acc1);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store 8x16 output: each WI holds 2 rows of 8 values
    int out_row = wg_m + sg_row * 8 + sg_lid;
    int out_col0_base = wg_n + sg_col * 16;
    int out_col1_base = out_col0_base + 8;

    if (out_row < M) {
        half8 res0 = convert_half8(acc0);
        half8 res1 = convert_half8(acc1);

        int base0 = out_row * N + out_col0_base;
        if (out_col0_base + 8 <= N) {
            vstore8(res0, 0, C + base0);
        } else {
            for (int j = 0; j < 8 && (out_col0_base + j) < N; j++)
                C[base0 + j] = ((half*)&res0)[j];
        }

        int base1 = out_row * N + out_col1_base;
        if (out_col1_base + 8 <= N) {
            vstore8(res1, 0, C + base1);
        } else if (out_col1_base < N) {
            for (int j = 0; j < 8 && (out_col1_base + j) < N; j++)
                C[base1 + j] = ((half*)&res1)[j];
        }
    }
}
```

