

## Analysis

The previous kernel failed because:

1. **Missing `reqd_work_group_size` attribute** - The harness couldn't parse the required work group size, so it used a fallback launch that broke all the tiling assumptions.
2. **No actual DPAS usage** - Despite comments about DPAS, it used scalar FMA loops.
3. **Subgroup size 16 vs 8** - The best working versions use subgroup size 8 with `intel_sub_group_f16_f16_matrix_mad_k16`.

The best kernel so far (33.9ms) uses a 32x32 tile with SLM and DPAS. I'll improve it by:
- Using a larger tile (32x64) to increase compute density per work-group
- Using `reqd_work_group_size` attribute so the harness can parse it
- Keeping double-buffered SLM for latency hiding
- Each subgroup computes 8x8 via DPAS k16
- Larger tile means more compute per global memory load

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Work-group tile: 32x64 of C
// Subgroup layout: 4 in M x 8 in N = 32 subgroups, sg_size=8 => 256 WIs
// Each subgroup: 8x8 via DPAS k16
// LWS = (64, 4, 1) = 256 work-items
// GWS = (ceil(N/64)*64, ceil(M/32)*4, 1)
// SLM: A[32][16] + B_vnni[8][64] per buffer, double-buffered

__attribute__((intel_reqd_sub_group_size(8)))
__attribute__((reqd_work_group_size(64, 4, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int TILE_M = 32;
    const int TILE_N = 64;
    const int TILE_K = 16;

    const int wg_col = get_group_id(0) * TILE_N;
    const int wg_row = get_group_id(1) * TILE_M;

    const int lid0 = get_local_id(0);  // 0..63
    const int lid1 = get_local_id(1);  // 0..3
    const int flat_lid = lid1 * 64 + lid0;  // 0..255

    const int sg_id = get_sub_group_id();        // 0..31
    const int sg_lid = get_sub_group_local_id(); // 0..7

    // 4 rows x 8 cols of subgroups
    const int sg_row = sg_id / 8;  // 0..3
    const int sg_col = sg_id % 8;  // 0..7

    // Double-buffered SLM
    __local half slm_A[2][32 * 16];          // A: 32 rows x 16 cols
    __local uint slm_B_vnni[2][8 * 64];      // B VNNI: 8 k-pairs x 64 cols

    float8 acc = (float8)(0.0f);

    int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Preload first tile into buffer 0
    {
        int k0 = 0;
        // Load A: 32*16=512 elements, 256 threads => 2 each
        for (int i = flat_lid; i < 32 * 16; i += 256) {
            int r = i >> 4;
            int c = i & 15;
            int gr = wg_row + r;
            int gc = k0 + c;
            slm_A[0][i] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0f;
        }
        // Load B VNNI: 8*64=512 elements, 256 threads => 2 each
        for (int i = flat_lid; i < 8 * 64; i += 256) {
            int kp = i >> 6;     // i / 64
            int c = i & 63;      // i % 64
            int gk0 = k0 + kp * 2;
            int gk1 = gk0 + 1;
            int gc = wg_col + c;
            ushort v0 = (gk0 < K && gc < N) ? as_ushort(B[gk0 * N + gc]) : (ushort)0;
            ushort v1 = (gk1 < K && gc < N) ? as_ushort(B[gk1 * N + gc]) : (ushort)0;
            slm_B_vnni[0][i] = (uint)v0 | ((uint)v1 << 16);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int next_kt = kt + 1;
        int next_buf = buf ^ 1;

        // Prefetch next tile
        if (next_kt < num_k_tiles) {
            int k0_next = next_kt * TILE_K;
            for (int i = flat_lid; i < 32 * 16; i += 256) {
                int r = i >> 4;
                int c = i & 15;
                int gr = wg_row + r;
                int gc = k0_next + c;
                slm_A[next_buf][i] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0f;
            }
            for (int i = flat_lid; i < 8 * 64; i += 256) {
                int kp = i >> 6;
                int c = i & 63;
                int gk0 = k0_next + kp * 2;
                int gk1 = gk0 + 1;
                int gc = wg_col + c;
                ushort v0 = (gk0 < K && gc < N) ? as_ushort(B[gk0 * N + gc]) : (ushort)0;
                ushort v1 = (gk1 < K && gc < N) ? as_ushort(B[gk1 * N + gc]) : (ushort)0;
                slm_B_vnni[next_buf][i] = (uint)v0 | ((uint)v1 << 16);
            }
        }

        // Compute: pack A and B from SLM, call DPAS
        int a_slm_row = sg_row * 8 + sg_lid;
        __local half* a_ptr = &slm_A[buf][a_slm_row * 16];

        int8 a_packed;
        {
            ushort s0=as_ushort(a_ptr[0]),  s1=as_ushort(a_ptr[1]),  s2=as_ushort(a_ptr[2]),  s3=as_ushort(a_ptr[3]);
            ushort s4=as_ushort(a_ptr[4]),  s5=as_ushort(a_ptr[5]),  s6=as_ushort(a_ptr[6]),  s7=as_ushort(a_ptr[7]);
            ushort s8=as_ushort(a_ptr[8]),  s9=as_ushort(a_ptr[9]),  s10=as_ushort(a_ptr[10]),s11=as_ushort(a_ptr[11]);
            ushort s12=as_ushort(a_ptr[12]),s13=as_ushort(a_ptr[13]),s14=as_ushort(a_ptr[14]),s15=as_ushort(a_ptr[15]);
            a_packed = (int8)(
                as_int((ushort2)(s0,s1)),   as_int((ushort2)(s2,s3)),
                as_int((ushort2)(s4,s5)),   as_int((ushort2)(s6,s7)),
                as_int((ushort2)(s8,s9)),   as_int((ushort2)(s10,s11)),
                as_int((ushort2)(s12,s13)), as_int((ushort2)(s14,s15))
            );
        }

        int b_slm_col = sg_col * 8 + sg_lid;
        int8 b_packed = (int8)(
            (int)slm_B_vnni[buf][0 * 64 + b_slm_col],
            (int)slm_B_vnni[buf][1 * 64 + b_slm_col],
            (int)slm_B_vnni[buf][2 * 64 + b_slm_col],
            (int)slm_B_vnni[buf][3 * 64 + b_slm_col],
            (int)slm_B_vnni[buf][4 * 64 + b_slm_col],
            (int)slm_B_vnni[buf][5 * 64 + b_slm_col],
            (int)slm_B_vnni[buf][6 * 64 + b_slm_col],
            (int)slm_B_vnni[buf][7 * 64 + b_slm_col]
        );

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Store 8x8 output
    int out_col = wg_col + sg_col * 8 + sg_lid;
    if (out_col < N) {
        for (int i = 0; i < 8; i++) {
            int out_row = wg_row + sg_row * 8 + i;
            if (out_row < M) {
                C[out_row * N + out_col] = convert_half(((float*)&acc)[i]);
            }
        }
    }
}
```

