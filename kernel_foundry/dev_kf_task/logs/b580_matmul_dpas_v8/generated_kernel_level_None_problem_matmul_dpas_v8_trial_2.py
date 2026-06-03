## Analysis

The previous kernel failed correctness tests. The issue is with the launch configuration. The console output shows: `[WARN] matmul_opt.cl has no parsable reqd_work_group_size; fallback launch (N, M), local=None`. This means the framework is launching with GWS=(N, M) and no specified local work size, which doesn't match our kernel's expected launch configuration of LWS=(16, 4) with GWS=(ceil(N/32)*16, ceil(M/32)*4).

The best implementation (102ms) also has this warning but works because its structure with `get_group_id` and `get_local_id` still functions correctly with the fallback launch. Let me look more carefully...

Actually, the fallback launch with GWS=(N, M) and local=None means the runtime picks arbitrary local sizes. With the previous kernel using `get_local_id(1)` as subgroup ID (0..3), if the local size in dim 1 isn't 4, the kernel breaks.

The best implementation at 102ms has the same issue but somehow works. Let me re-examine - it uses `get_local_id(1)` as `sg_id` (0..3) and `get_local_id(0)` within that. If fallback launches with local=None, the OpenCL runtime may choose something compatible, but it's unreliable.

To fix this, I should add `__attribute__((reqd_work_group_size(16, 4, 1)))` so the framework can parse it and use proper GWS/LWS. Let me base my solution on the best implementation (102ms) but add the required work group size attribute and try to improve performance with a larger tile or prefetching.

Let me also try to increase the tile size to get better compute-to-memory ratio. I'll use 32M x 64N with 4 subgroups, same as the best version, but add `reqd_work_group_size` and optimize the loading with better patterns.

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Tile: 32M x 64N per work-group, K-tile = 16
// 4 subgroups per WG, each subgroup computes 8M x 64N (4 DPAS calls per K-step)
// LWS = (16, 4) => 64 WIs, 4 subgroups of 16
// GWS = (ceil(N/64)*16, ceil(M/32)*4)
// reqd_work_group_size ensures framework parses correct launch config

#define TILE_M 32
#define TILE_N 64
#define TILE_K 16
#define SG_SIZE 16
#define NUM_SG 4
#define A_STRIDE (TILE_K)  // 16

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 4, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0) * TILE_N;  // base column of this WG tile
    const int wg_m = get_group_id(1) * TILE_M;  // base row of this WG tile

    const int sg_id = get_local_id(1);           // 0..3, which subgroup (row block)
    const int sg_lid = get_sub_group_local_id(); // 0..15

    const int row_base = wg_m + sg_id * 8;      // each subgroup handles 8 rows

    // Accumulators: 4 DPAS blocks of 8x16 each = 8 rows x 64 cols
    float8 acc0 = 0.0f; // cols 0-15
    float8 acc1 = 0.0f; // cols 16-31
    float8 acc2 = 0.0f; // cols 32-47
    float8 acc3 = 0.0f; // cols 48-63

    // SLM for A tile [32][16] as ushort and B tile in VNNI format [8][64] as uint
    // A: 32*16 = 512 ushorts = 1024 bytes
    // B_vnni: 8*64 = 512 uints = 2048 bytes
    __local ushort slm_a[TILE_M * TILE_K];    // 512 ushorts
    __local uint slm_b_vnni[8 * TILE_N];      // 512 uints

    // Flat thread ID within WG for cooperative loading
    const int flat_id = sg_id * 16 + get_local_id(0); // 0..63

    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        // === Cooperative load A[32][16] into SLM ===
        // 512 elements, 64 threads => 8 elements per thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = flat_id * 8 + i;
            int r = idx >> 4;   // idx / 16
            int c = idx & 15;   // idx % 16
            int gr = wg_m + r;
            int gc = k0 + c;
            ushort val = 0;
            if (gr < M && gc < K)
                val = as_ushort(A[gr * K + gc]);
            slm_a[r * TILE_K + c] = val;
        }

        // === Cooperative load B[16][64] and pack into VNNI format ===
        // VNNI packs pairs of k-rows: slm_b_vnni[pair][col] = pack(B[2*pair][col], B[2*pair+1][col])
        // 8 pairs x 64 cols = 512 uints, 64 threads => 8 per thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = flat_id * 8 + i;
            int pair = idx >> 6;  // idx / 64 -> which k-pair (0..7)
            int col = idx & 63;   // idx % 64 -> which column (0..63)
            int k_row0 = k0 + pair * 2;
            int k_row1 = k0 + pair * 2 + 1;
            int gcol = wg_n + col;
            ushort v0 = 0, v1 = 0;
            if (k_row0 < K && gcol < N)
                v0 = as_ushort(B[k_row0 * N + gcol]);
            if (k_row1 < K && gcol < N)
                v1 = as_ushort(B[k_row1 * N + gcol]);
            slm_b_vnni[pair * TILE_N + col] = ((uint)v1 << 16) | (uint)v0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === Load A for this subgroup: 8 rows x 16 cols ===
        __local ushort* a_ptr = slm_a + sg_id * 8 * TILE_K;
        short8 a_val = as_short8((ushort8)(
            a_ptr[0 * TILE_K + sg_lid], a_ptr[1 * TILE_K + sg_lid],
            a_ptr[2 * TILE_K + sg_lid], a_ptr[3 * TILE_K + sg_lid],
            a_ptr[4 * TILE_K + sg_lid], a_ptr[5 * TILE_K + sg_lid],
            a_ptr[6 * TILE_K + sg_lid], a_ptr[7 * TILE_K + sg_lid]));

        // === Load B VNNI blocks ===
        __local uint* b_base = slm_b_vnni;

        int8 b0 = (int8)(
            as_int(b_base[0 * TILE_N + sg_lid]),      as_int(b_base[1 * TILE_N + sg_lid]),
            as_int(b_base[2 * TILE_N + sg_lid]),      as_int(b_base[3 * TILE_N + sg_lid]),
            as_int(b_base[4 * TILE_N + sg_lid]),      as_int(b_base[5 * TILE_N + sg_lid]),
            as_int(b_base[6 * TILE_N + sg_lid]),      as_int(b_base[7 * TILE_N + sg_lid]));

        int8 b1 = (int8)(
            as_int(b_base[0 * TILE_N + 16 + sg_lid]), as_int(b_base[1 * TILE_N + 16 + sg_lid]),
            as_int(b_base[2 * TILE_N + 16 + sg_lid]), as_int(b_base[3 * TILE_N + 16 + sg_lid]),
            as_int(b_base[4 * TILE_N + 16 + sg_lid]), as_int(b_base[5 * TILE_N + 16 + sg_lid]),
            as_int(b_base[6 * TILE_N + 16 + sg_lid]), as_int(b_base[7 * TILE_N + 16 + sg_lid]));

        int8 b2 = (int8)(
            as_int(b_base[0 * TILE_N + 32 + sg_lid]), as_int(b_base[1 * TILE_N + 32 + sg_lid]),
            as_int(b_base[2 * TILE_N + 32 + sg_lid]), as_int(b_base[3 * TILE_N + 32 + sg_lid]),
            as_int(b_base[4 * TILE_N + 32 + sg_lid]), as_int(b_base[5 * TILE_N + 32 + sg_lid]),
            as_int(b_base[6 * TILE_N + 32 + sg_lid]), as_int(b_base[7 * TILE_N + 32 + sg_lid]));

        int8 b3 = (int8)(
            as_int(b_base[0 * TILE_N + 48 + sg_lid]), as_int(b_base[1 * TILE_N + 48 + sg_lid]),
            as_int(b_base[2 * TILE_N + 48 + sg_lid]), as_int(b_base[3 * TILE_N + 48 + sg_lid]),
            as_int(b_base[4 * TILE_N + 48 + sg_lid]), as_int(b_base[5 * TILE_N + 48 + sg_lid]),
            as_int(b_base[6 * TILE_N + 48 + sg_lid]), as_int(b_base[7 * TILE_N + 48 + sg_lid]));

        // 4 DPAS calls per subgroup per K-step
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b2, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b3, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
    // Each accumulator float8 holds 8 rows, each WI holds one column (sg_lid offset)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = row_base + i;
        if (row < M) {
            float v0 = ((float*)&acc0)[i];
            float v1 = ((float*)&acc1)[i];
            float v2 = ((float*)&acc2)[i];
            float v3 = ((float*)&acc3)[i];

            int col0 = wg_n + sg_lid;
            int col1 = wg_n + 16 + sg_lid;
            int col2 = wg_n + 32 + sg_lid;
            int col3 = wg_n + 48 + sg_lid;

            if (col0 < N) C[row * N + col0] = convert_half(v0);
            if (col1 < N) C[row * N + col1] = convert_half(v1);
            if (col2 < N) C[row * N + col2] = convert_half(v2);
            if (col3 < N) C[row * N + col3] = convert_half(v3);
        }
    }
}
```

