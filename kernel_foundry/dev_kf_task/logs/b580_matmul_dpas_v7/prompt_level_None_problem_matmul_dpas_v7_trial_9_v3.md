

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Optimization goals:
- Must use Intel OpenCL DPAS instruction, e.g. intel_sub_group_f16_f16_matrix_mad_k16.
- Improve FLOPS and hide memory latency with tiling + subgroup-friendly mapping.
- Provide explicit launch metadata (GWS/LWS/subgroup hints) in kernel comments.

## Reference code / Task:

This is the reference OCL implementation:
```
// Simple row-major FP16 matmul:
//   C[M,N] = A[M,K] x B[K,N]
// Input/Output dtype: half
// Accumulation dtype: float
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += convert_float(A[row * K + k]) * convert_float(B[k * N + col]);
    }

    C[row * N + col] = convert_half(acc);
}

```

## Previous OCL implementations with scores:

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 34.000):
```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS + register blocking
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Launch metadata:
//   LWS = (16, 4, 1) => 64 work-items = 8 subgroups of 8
//   GWS = (ceil(N/64)*16, ceil(M/32)*4, 1)
//   Subgroup size = 8
//   Work-group tile: TM=32, TN=64
//   Subgroup grid: 2 along N x 4 along M
//   Each subgroup: 8 rows x 32 cols (4 DPAS calls per K-step) => register blocking
//   SLM: A stored as packed int [32][8], B stored as packed int [8][64]
//   SLM bytes: 32*8*4 + 8*64*4 = 1024 + 2048 = 3072 bytes per buffer

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
    const int TK = 16;
    const int WG_SIZE = 64; // 16 * 4

    // Work-group origin
    const int wg_n = get_group_id(0) * TN;
    const int wg_m = get_group_id(1) * TM;

    const int lid0 = get_local_id(0);  // 0..15
    const int lid1 = get_local_id(1);  // 0..3
    const int flat_lid = lid1 * 16 + lid0;  // 0..63

    const int sg_local_id = get_sub_group_local_id(); // 0..7
    const int sg_id = get_sub_group_id(); // 0..7

    // Subgroup grid: 2 along N, 4 along M
    const int sg_n = sg_id % 2;  // 0..1
    const int sg_m = sg_id / 2;  // 0..3

    // Each subgroup computes 8 rows x 32 cols using 4 DPAS (each 8x8)
    // Output origin for this subgroup
    const int tile_m = wg_m + sg_m * 8;
    const int tile_n = wg_n + sg_n * 32;

    // SLM: A in DPAS-packed format (int), B in VNNI-packed format (int)
    // A: [32 rows][8 ints] = 32 rows x 16 halfs packed as 8 ints each
    // B: [8 k-pairs][64 cols] as int = VNNI packed
    __local int slm_A[32 * 8];      // 1024 bytes
    __local int slm_B[8 * 64];      // 2048 bytes

    // 4 accumulators for 4 column-groups of 8
    float8 acc0 = (float8)(0.0f);  // cols 0..7
    float8 acc1 = (float8)(0.0f);  // cols 8..15
    float8 acc2 = (float8)(0.0f);  // cols 16..23
    float8 acc3 = (float8)(0.0f);  // cols 24..31

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TK - 1) / TK;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k0 = kt * TK;
        const int k_rem = K - k0;

        // === Cooperative load A into packed SLM ===
        // 32 rows x 8 ints = 256 ints, 64 WIs => 4 ints each
        for (int i = flat_lid; i < 32 * 8; i += WG_SIZE) {
            int r = i >> 3;    // row 0..31
            int p = i & 7;     // pair index 0..7
            int gr = wg_m + r;
            int gc0 = k0 + p * 2;
            int gc1 = gc0 + 1;
            ushort v0 = (gr < M && gc0 < K) ? A_us[gr * K + gc0] : (ushort)0;
            ushort v1 = (gr < M && gc1 < K) ? A_us[gr * K + gc1] : (ushort)0;
            slm_A[r * 8 + p] = (int)v0 | ((int)v1 << 16);
        }

        // === Cooperative load B into VNNI-packed SLM ===
        // 8 k-pairs x 64 cols = 512 ints, 64 WIs => 8 ints each
        for (int i = flat_lid; i < 8 * 64; i += WG_SIZE) {
            int kp = i >> 6;   // k-pair 0..7
            int c = i & 63;    // col 0..63
            int gk0 = k0 + kp * 2;
            int gk1 = gk0 + 1;
            int gc = wg_n + c;
            ushort v0 = (gk0 < K && gc < N) ? B_us[gk0 * N + gc] : (ushort)0;
            ushort v1 = (gk1 < K && gc < N) ? B_us[gk1 * N + gc] : (ushort)0;
            slm_B[kp * 64 + c] = (int)v0 | ((int)v1 << 16);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === Compute: load from SLM in DPAS-ready format ===
        // A operand: row = sg_m * 8 + sg_local_id, 8 ints
        int a_row = sg_m * 8 + sg_local_id;
        int8 a_packed = vload8(0, &slm_A[a_row * 8]);

        // B operand for 4 column groups
        // Each column group: sg_n * 32 + group * 8 + sg_local_id
        int b_base = sg_n * 32;

        int8 b0, b1, b2, b3;
        int bc0 = b_base + 0 + sg_local_id;
        int bc1 = b_base + 8 + sg_local_id;
        int bc2 = b_base + 16 + sg_local_id;
        int bc3 = b_base + 24 + sg_local_id;

        b0 = (int8)(slm_B[0*64+bc0], slm_B[1*64+bc0], slm_B[2*64+bc0], slm_B[3*64+bc0],
                     slm_B[4*64+bc0], slm_B[5*64+bc0], slm_B[6*64+bc0], slm_B[7*64+bc0]);
        b1 = (int8)(slm_B[0*64+bc1], slm_B[1*64+bc1], slm_B[2*64+bc1], slm_B[3*64+bc1],
                     slm_B[4*64+bc1], slm_B[5*64+bc1], slm_B[6*64+bc1], slm_B[7*64+bc1]);
        b2 = (int8)(slm_B[0*64+bc2], slm_B[1*64+bc2], slm_B[2*64+bc2], slm_B[3*64+bc2],
                     slm_B[4*64+bc2], slm_B[5*64+bc2], slm_B[6*64+bc2], slm_B[7*64+bc2]);
        b3 = (int8)(slm_B[0*64+bc3], slm_B[1*64+bc3], slm_B[2*64+bc3], slm_B[3*64+bc3],
                     slm_B[4*64+bc3], slm_B[5*64+bc3], slm_B[6*64+bc3], slm_B[7*64+bc3]);

        // 4 DPAS calls - register blocking along N
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b2, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b3, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store 8x32 output tile ===
    int out_row = tile_m + sg_local_id;
    if (out_row < M) {
        int base = out_row * N;

        // Store acc0: cols tile_n+0..7
        if (tile_n + 8 <= N) {
            vstore8(convert_half8(acc0), 0, C + base + tile_n);
        } else {
            for (int j = 0; j < 8 && (tile_n + j) < N; j++)
                C[base + tile_n + j] = convert_half(((float*)&acc0)[j]);
        }

        // Store acc1: cols tile_n+8..15
        if (tile_n + 16 <= N) {
            vstore8(convert_half8(acc1), 0, C + base + tile_n + 8);
        } else {
            for (int j = 0; j < 8 && (tile_n + 8 + j) < N; j++)
                C[base + tile_n + 8 + j] = convert_half(((float*)&acc1)[j]);
        }

        // Store acc2: cols tile_n+16..23
        if (tile_n + 24 <= N) {
            vstore8(convert_half8(acc2), 0, C + base + tile_n + 16);
        } else {
            for (int j = 0; j < 8 && (tile_n + 16 + j) < N; j++)
                C[base + tile_n + 16 + j] = convert_half(((float*)&acc2)[j]);
        }

        // Store acc3: cols tile_n+24..31
        if (tile_n + 32 <= N) {
            vstore8(convert_half8(acc3), 0, C + base + tile_n + 24);
        } else {
            for (int j = 0; j < 8 && (tile_n + 24 + j) < N; j++)
                C[base + tile_n + 24 + j] = convert_half(((float*)&acc3)[j]);
        }
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Launch metadata:
//   LWS = (32, 4, 1)  => 128 work-items = 16 subgroups of 8
//   GWS = (ceil(N/32)*32, ceil(M/32)*4, 1)
//   Subgroup size = 8
//   Work-group tile: 32x32 of C
//   Each subgroup: 8x8 of C via DPAS with k=16 steps
//   SLM: 32*16*2 + 16*32*2 = 1024 + 1024 = 2048 bytes per k-step

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile origin
    const int tN = 32;  // tile cols
    const int tM = 32;  // tile rows
    const int tK = 16;  // tile K depth (matches DPAS k16)

    const int wg_col = get_group_id(0) * tN;  // column offset in C
    const int wg_row = get_group_id(1) * tM;  // row offset in C

    // Local IDs
    const int lid0 = get_local_id(0);  // 0..31
    const int lid1 = get_local_id(1);  // 0..3
    const int flat_lid = lid1 * 32 + lid0;  // 0..127

    // Subgroup info
    const int sg_id = get_sub_group_id();        // 0..15
    const int sg_lid = get_sub_group_local_id(); // 0..7

    // Each subgroup handles an 8x8 tile within the 32x32 work-group tile
    // sg_id maps to (sg_row, sg_col) in a 4x4 grid
    const int sg_row = sg_id / 4;  // 0..3
    const int sg_col = sg_id % 4;  // 0..3

    // SLM for A tile [32][16] and B tile [16][32], stored as half
    __local half slm_A[32 * 16];
    __local half slm_B[16 * 32];

    // Accumulator: 8 rows x 8 cols per subgroup, each work-item holds 8 floats
    // (one column of the 8x8 tile, across 8 rows)
    float8 acc = (float8)(0.0f);

    // Loop over K in steps of 16
    for (int k0 = 0; k0 < K; k0 += tK) {
        // Cooperative load of A[wg_row..wg_row+31][k0..k0+15] into slm_A
        // 128 work-items load 32*16 = 512 halves => 4 halves each
        for (int i = flat_lid; i < 32 * 16; i += 128) {
            int r = i / 16;
            int c = i % 16;
            int gr = wg_row + r;
            int gc = k0 + c;
            slm_A[r * 16 + c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0f;
        }

        // Cooperative load of B[k0..k0+15][wg_col..wg_col+31] into slm_B
        for (int i = flat_lid; i < 16 * 32; i += 128) {
            int r = i / 32;
            int c = i % 32;
            int gr = k0 + r;
            int gc = wg_col + c;
            slm_B[r * 32 + c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each subgroup computes its 8x8 tile using DPAS
        // A sub-tile: rows [sg_row*8 .. sg_row*8+7], cols [0..15] from slm_A
        // B sub-tile: rows [0..15], cols [sg_col*8 .. sg_col*8+7] from slm_B

        // Pack A for DPAS: each work-item i in subgroup holds row (sg_row*8 + i)
        // Need 16 half values packed into 8 floats (2 halves per float)
        int a_row = sg_row * 8 + sg_lid;
        float8 a_packed;
        __local half* a_ptr = &slm_A[a_row * 16];
        // Pack pairs of halves into float using as_float(as_uint((ushort2)(v0,v1)))
        for (int j = 0; j < 8; j++) {
            ushort lo = as_ushort(a_ptr[j * 2]);
            ushort hi = as_ushort(a_ptr[j * 2 + 1]);
            ((float*)&a_packed)[j] = as_float((uint)lo | ((uint)hi << 16));
        }

        // Pack B for DPAS: B is [16][8] sub-tile
        // For intel_sub_group_f16_f16_matrix_mad_k16, B needs VNNI-like layout
        // B is packed as: for each group of 2 K-rows, interleave 8 columns
        // B layout: [K/2][8][2] packed into floats
        // Each work-item holds 8 floats corresponding to 8 pairs of K-rows
        int b_col_base = sg_col * 8;
        float8 b_packed;
        for (int j = 0; j < 8; j++) {
            int k_pair = j;  // pairs 0..7 cover k=0..15
            ushort lo = as_ushort(slm_B[(k_pair * 2) * 32 + b_col_base + sg_lid]);
            ushort hi = as_ushort(slm_B[(k_pair * 2 + 1) * 32 + b_col_base + sg_lid]);
            ((float*)&b_packed)[j] = as_float((uint)lo | ((uint)hi << 16));
        }

        // DPAS: C[8x8] += A[8x16] * B[16x8]
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: each work-item in subgroup writes one column of the 8x8 tile
    int out_col = wg_col + sg_col * 8 + sg_lid;
    for (int i = 0; i < 8; i++) {
        int out_row = wg_row + sg_row * 8 + i;
        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = convert_half(((float*)&acc)[i]);
        }
    }
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS + double-buffered SLM
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Launch metadata:
//   LWS = (32, 4, 1) => 128 work-items = 16 subgroups of 8
//   GWS = (ceil(N/32)*32, ceil(M/32)*4, 1)
//   Subgroup size = 8
//   Work-group tile: 32x32 of C
//   Each subgroup: 8x8 of C via DPAS with k=16 steps
//   Double-buffered SLM: 2 * (32*16 + 16*32) * 2 bytes = 4096 bytes
//   A stored row-major in SLM, B stored in VNNI format (k-pairs interleaved)

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
    const int tM = 32;
    const int tN = 32;
    const int tK = 16;

    const int wg_col = get_group_id(0) * tN;
    const int wg_row = get_group_id(1) * tM;

    const int lid0 = get_local_id(0);
    const int lid1 = get_local_id(1);
    const int flat_lid = lid1 * 32 + lid0;

    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();

    const int sg_row = sg_id / 4;
    const int sg_col = sg_id % 4;

    // Double-buffered SLM
    // A: [2][32][16] as uint (pairs of halfs) -> [2][32][8] uints
    // B: [2][8][32] as uint (VNNI: k-pair interleaved) -> [2][8][32] uints
    // Store A as half for simplicity, but load as uint for DPAS
    __local half slm_A[2][32 * 16];
    // B stored in VNNI format: slm_B[buf][k_pair][col] as uint = pack(B[k0+2*kp, col], B[k0+2*kp+1, col])
    __local uint slm_B_vnni[2][8 * 32];

    float8 acc = (float8)(0.0f);

    int num_k_tiles = (K + tK - 1) / tK;
    int buf = 0;

    // Preload first tile into buffer 0
    {
        int k0 = 0;
        // Load A[wg_row..+31][k0..+15] into slm_A[0]
        // 128 threads, 512 elements => 4 each
        for (int i = flat_lid; i < 32 * 16; i += 128) {
            int r = i >> 4;      // i / 16
            int c = i & 15;      // i % 16
            int gr = wg_row + r;
            int gc = k0 + c;
            slm_A[0][i] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0f;
        }
        // Load B into VNNI format: 8 k-pairs x 32 cols = 256 uints
        // 128 threads => 2 each
        for (int i = flat_lid; i < 8 * 32; i += 128) {
            int kp = i >> 5;     // i / 32, k-pair index 0..7
            int c = i & 31;      // i % 32, column index
            int gk0 = k0 + kp * 2;
            int gk1 = gk0 + 1;
            int gc = wg_col + c;
            ushort v0 = (gk0 < K && gc < N) ? as_ushort(B[gk0 * N + gc]) : (ushort)0;
            ushort v1 = (gk1 < K && gc < N) ? as_ushort(B[gk1 * N + gc]) : (ushort)0;
            slm_B_vnni[0][i] = (uint)v0 | ((uint)v1 << 16);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int next_kt = kt + 1;
        int next_buf = buf ^ 1;

        // Prefetch next K-tile into next_buf (if exists)
        if (next_kt < num_k_tiles) {
            int k0_next = next_kt * tK;
            for (int i = flat_lid; i < 32 * 16; i += 128) {
                int r = i >> 4;
                int c = i & 15;
                int gr = wg_row + r;
                int gc = k0_next + c;
                slm_A[next_buf][i] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0f;
            }
            for (int i = flat_lid; i < 8 * 32; i += 128) {
                int kp = i >> 5;
                int c = i & 31;
                int gk0 = k0_next + kp * 2;
                int gk1 = gk0 + 1;
                int gc = wg_col + c;
                ushort v0 = (gk0 < K && gc < N) ? as_ushort(B[gk0 * N + gc]) : (ushort)0;
                ushort v1 = (gk1 < K && gc < N) ? as_ushort(B[gk1 * N + gc]) : (ushort)0;
                slm_B_vnni[next_buf][i] = (uint)v0 | ((uint)v1 << 16);
            }
        }

        // Compute on current buffer
        // Load A for this subgroup: rows [sg_row*8 + sg_lid], 16 cols
        int a_slm_row = sg_row * 8 + sg_lid;
        __local half* a_ptr = &slm_A[buf][a_slm_row * 16];

        // Pack A into int8 (16 halfs = 8 ints) - use vector load
        int8 a_packed;
        {
            ushort s0  = as_ushort(a_ptr[0]);  ushort s1  = as_ushort(a_ptr[1]);
            ushort s2  = as_ushort(a_ptr[2]);  ushort s3  = as_ushort(a_ptr[3]);
            ushort s4  = as_ushort(a_ptr[4]);  ushort s5  = as_ushort(a_ptr[5]);
            ushort s6  = as_ushort(a_ptr[6]);  ushort s7  = as_ushort(a_ptr[7]);
            ushort s8  = as_ushort(a_ptr[8]);  ushort s9  = as_ushort(a_ptr[9]);
            ushort s10 = as_ushort(a_ptr[10]); ushort s11 = as_ushort(a_ptr[11]);
            ushort s12 = as_ushort(a_ptr[12]); ushort s13 = as_ushort(a_ptr[13]);
            ushort s14 = as_ushort(a_ptr[14]); ushort s15 = as_ushort(a_ptr[15]);
            a_packed = (int8)(
                as_int((ushort2)(s0, s1)),   as_int((ushort2)(s2, s3)),
                as_int((ushort2)(s4, s5)),   as_int((ushort2)(s6, s7)),
                as_int((ushort2)(s8, s9)),   as_int((ushort2)(s10, s11)),
                as_int((ushort2)(s12, s13)), as_int((ushort2)(s14, s15))
            );
        }

        // Load B VNNI for this subgroup: 8 k-pairs, cols [sg_col*8 + sg_lid]
        int b_slm_col = sg_col * 8 + sg_lid;
        int8 b_packed;
        {
            b_packed = (int8)(
                (int)slm_B_vnni[buf][0 * 32 + b_slm_col],
                (int)slm_B_vnni[buf][1 * 32 + b_slm_col],
                (int)slm_B_vnni[buf][2 * 32 + b_slm_col],
                (int)slm_B_vnni[buf][3 * 32 + b_slm_col],
                (int)slm_B_vnni[buf][4 * 32 + b_slm_col],
                (int)slm_B_vnni[buf][5 * 32 + b_slm_col],
                (int)slm_B_vnni[buf][6 * 32 + b_slm_col],
                (int)slm_B_vnni[buf][7 * 32 + b_slm_col]
            );
        }

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);

        // Single barrier for double-buffering: protect next_buf writes
        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Store 8x8 output tile
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Optimized FP16 GEMM using Intel DPAS (XMX) instructions for Battlemage Xe2-HPG
// C[M,N] = A[M,K] x B[K,N], all half precision, float accumulation
// Launch config:
//   LWS = (16, 1)  -- one subgroup per work-group
//   GWS = (ceil(N/16)*16, ceil(M/8))
//   Subgroup size = 16
//   Each subgroup computes an 8x16 output tile
// TILE_M=8, TILE_N=16, TILE_K=16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_sub_group_local_id(); // 0..15, corresponds to output column within tile
    const int tile_col = get_group_id(0) * 16;  // starting output column for this tile
    const int tile_row = get_group_id(1) * 8;   // starting output row for this tile

    float8 acc = (float8)(0.0f);

    // Loop over K in chunks of 16
    for (int k = 0; k < K; k += 16) {
        // Load A tile: 8 rows x 16 cols (k-chunk)
        // Each lane loads one column of A; for DPAS we need int8 'a' where
        // a[row] = pack of 2 fp16 values. We use sub_group block reads.
        int8 a_val;
        for (int r = 0; r < 8; r++) {
            int row = tile_row + r;
            // Each subgroup lane sg_id reads A[row, k + sg_id*1] but we need pairs
            // For DPAS: a[r] contains 2 packed halfs: A[row, k+2*sg_id] and A[row, k+2*sg_id+1]
            short lo = 0, hi = 0;
            if (row < M && (k + 2 * sg_id) < K)
                lo = as_short(A[row * K + k + 2 * sg_id]);
            if (row < M && (k + 2 * sg_id + 1) < K)
                hi = as_short(A[row * K + k + 2 * sg_id + 1]);
            ((int*)&a_val)[r] = as_int((short2)(lo, hi));
        }

        // Load B tile: 16 rows (k-chunk) x 16 cols
        // For DPAS b format: b[i] packs 2 rows of B, each row is 16 halfs across the subgroup
        // b[i] at lane sg_id = pack(B[k+2*i, tile_col+sg_id], B[k+2*i+1, tile_col+sg_id])
        int8 b_val;
        for (int i = 0; i < 8; i++) {
            int brow0 = k + 2 * i;
            int brow1 = k + 2 * i + 1;
            int col = tile_col + sg_id;
            short v0 = 0, v1 = 0;
            if (brow0 < K && col < N)
                v0 = as_short(B[brow0 * N + col]);
            if (brow1 < K && col < N)
                v1 = as_short(B[brow1 * N + col]);
            ((int*)&b_val)[i] = as_int((short2)(v0, v1));
        }

        // DPAS: acc += A_tile * B_tile
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    }

    // Store results: each lane sg_id writes column (tile_col + sg_id) for 8 rows
    int col = tile_col + sg_id;
    if (col < N) {
        for (int r = 0; r < 8; r++) {
            int row = tile_row + r;
            if (row < M) {
                C[row * N + col] = convert_half(((float*)&acc)[r]);
            }
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel **is incorrect**.

(II) Summary of console output:  
All 4 correctness tests failed, and the output matrix is entirely zeros while PyTorch/reference produce nonzero values. The key clue is:  
`[WARN] matmul_opt.cl has no parsable reqd_work_group_size; fallback launch (N, M), local=None`  
So your kernel was launched with a different NDRange/work-group shape than the one your tiling logic assumes (`group_id(0)=tile N/16`, `group_id(1)=tile M/8`, subgroup=16). With that mismatch, your addressing and subgroup-DPAS mapping don’t correspond to valid tiles, so computation effectively collapses to wrong/zero output.

(III) Errors and suggestions for improvement:  
1. `__attribute__((intel_reqd_sub_group_size(16)))` (kernel declaration): This enforces subgroup width, but you did **not** provide a parsable required work-group size for the harness, so it falls back to `local=None`. Add a required work-group-size attribute that matches your design (single subgroup WG):  
   `__attribute__((reqd_work_group_size(16,1,1)))`  
   so the launcher won’t use incompatible geometry.

2. `const int tile_col = get_group_id(0) * 16;` and `const int tile_row = get_group_id(1) * 8;`: These are only correct for `GWS=(ceil(N/16)*16, ceil(M/8))` with `LWS=(16,1)`. Under fallback `(N,M)` this mapping is invalid. Make indexing robust to actual NDRange by deriving tile coords from global IDs (or enforce required WG size as above and keep strict tiling). Right now you rely on launch assumptions the runtime is not honoring.

3. `((int*)&a_val)[r] = ...`, `((int*)&b_val)[i] = ...`, and `((float*)&acc)[r]`: These pointer casts on OpenCL vector types are non-portable/fragile and can break codegen for DPAS paths. Replace with explicit vector construction/insertion (`int8` component assignments like `.s0 ... .s7`, or helper pack functions) and explicit `acc.s0...s7` stores. This removes aliasing ambiguity and makes the DPAS operand layout unambiguous for the compiler.

## Hardware specification:
Your code will run on the following hardware:
**Intel Battlemage** with specs: Xe-cores: 20, Render Slices: 5, Ray Tracing Units: 20, Intel® Xe Matrix Extensions (Intel® XMX) Engines: 160, Xe Vector Engines: 160, Graphics Clock: 2670, GPU Peak TOPS (Int8): 233, TBP: 190, PCI Express Configurations ‡: PCI Express 4.0 x8, Device ID: 0xE20B, Memory: 12 GB GDDR6, Memory Interface: 192 bit, Memory Bandwidth: 456, Memory Speed: 19, ISA_GPU: Xe2-HPG
Please consider the hardware specifications when improving the code. 

## Task:

**Your objectives**:
1. Analyze the previous kernel and its evaluation log.
2. Identify any errors or mismatches with the reference implementation.
3. Propose specific improvements or fixes, explaining your reasoning step by step.
4. Rewrite the kernel, providing the complete, corrected code in a code block.

**Critical Requirements:**

1. The kernel must exactly match the reference implementation's functionality.
2. The code must compile and run properly on the GPU.
3. Do not cache or reuse previous results; ensure the code executes fully on each run.
4. Keep all hyperparameters (e.g., batch size, dimensions) unchanged as specified in the reference implementation.
8. Beware of the critical error "Unexpected kernel lambda size. In such cases removing constexpr specifier aligns the captures between the host compiler and the device compiler"! Do not capture constexpr variables in lambda functions passed to kernel launches as this can lead to different lambda sizes between the host and device compiler.

Additional Guidance:

1. Clearly comment on any fixes and optimizations you implement.
2. If you change the kernel structure, explain why.
3. Anticipate and address possible runtime or compilation errors.

Please structure your response as follows:

1. Analysis:
    * Summarize the errors found in the previous kernel and log.
    * Explain your proposed changes.
2. Improved OCL code:
    * Provide the complete, corrected OCL code in a code block:
```OCL
Your code here
```


## Required Optimizations

Apply the following optimization techniques in your implementation:

- **Shared Local Memory (SLM) Tiling**: Cache frequently accessed data in SLM using `group_local_memory_for_overwrite`. Synchronize with `group_barrier()` after writes and before reads. Use 16×16 or 32×32 tiles for float data.
- **Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade recomputation for memory savings (e.g., Flash-Attention style). Maintain running accumulators across blocks with proper rescaling.
- **Work-Group Reductions**: Replace atomic operations with O(log N) tree-based reductions in local memory. Synchronize with `group_barrier()` between iterations.