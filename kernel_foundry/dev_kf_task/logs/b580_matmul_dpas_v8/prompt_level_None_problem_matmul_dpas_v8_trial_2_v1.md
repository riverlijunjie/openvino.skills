

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Optimization goals:
- Must use Intel OpenCL DPAS instruction(XMX), e.g. intel_sub_group_f16_f16_matrix_mad_k16.
    CRITICAL: For intel_sub_group_f16_f16_matrix_mad_k16, you MUST use:
        - First operand: short8 (NOT float8)
        - Second operand: int8 (NOT float8)
        - Accumulator: float8 (this one is float8)
    Example:
        short8 a_val = as_short8(intel_sub_group_block_read_us8(...));
        int8 b_val = as_int8(intel_sub_group_block_read_ui8(...));
        float8 acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    Using float8 for the first two operands will compile but will NOT use
    XMX hardware acceleration.
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

### Version 1 (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Tiled DPAS matmul: C[M,N] = A[M,K] * B[K,N], half precision, float accumulation
// Tile: 32M x 32N per work-group, K stepped by 32 (two DPAS k16 steps)
// 4 subgroups per WG, each subgroup computes 8 rows x 32 cols (two 8x16 DPAS outputs)
// LWS = (16, 4), subgroup_size = 16
// GWS = (ceil(N/32)*16, ceil(M/32)*4)

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define SG_COUNT 4

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);
    const int base_m = wg_m * TILE_M;
    const int base_n = wg_n * TILE_N;

    const int sg_id = get_local_id(1);  // 0..3, which subgroup (row block)
    const int sg_lid = get_sub_group_local_id(); // 0..15

    // Each subgroup handles 8 rows x 32 cols = two 8x16 DPAS blocks
    const int my_row_base = base_m + sg_id * 8;

    // Accumulators: 2 DPAS outputs (left 16 cols, right 16 cols)
    float8 acc0 = 0.0f;  // cols [0..15]
    float8 acc1 = 0.0f;  // cols [16..31]

    // SLM: A tile [32][16+2] padded to avoid bank conflicts, B in VNNI format [8][32] as uint
    // A: 32 rows x 16 cols of half, padded stride = 18
    // B VNNI: 8 pairs x 32 cols as uint = 8*32 uint
    __local ushort slm_a[TILE_M * (TILE_K + 2)];  // 32 * 18 = 576 ushorts
    __local uint slm_b_vnni[8 * TILE_N];           // 8 * 32 = 256 uints

    const int a_stride = TILE_K + 2;  // 18, padded to reduce bank conflicts

    // flat thread id within WG: sg_id * 16 + sg_lid, total 64 threads
    const int flat_id = sg_id * 16 + sg_lid;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        // === Load A[32][16] into slm_a[32][18] ===
        // 64 threads, 512 elements -> 8 elements per thread
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = flat_id + i * 64;
            int r = idx >> 4;       // idx / 16
            int c = idx & 15;       // idx % 16
            int gr = base_m + r;
            int gc = k0 + c;
            ushort val = 0;
            if (gr < M && gc < K)
                val = as_ushort(A[gr * K + gc]);
            slm_a[r * a_stride + c] = val;
        }

        // === Load B[16][32] and pack into VNNI format in SLM ===
        // VNNI: slm_b_vnni[pair][col] = pack(B[2*pair][col], B[2*pair+1][col])
        // 256 uints to write, 64 threads -> 4 per thread
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = flat_id + i * 64;
            int pair = idx >> 5;    // idx / 32 -> which pair (0..7)
            int col = idx & 31;     // idx % 32 -> which column

            int k_row0 = k0 + pair * 2;
            int k_row1 = k0 + pair * 2 + 1;
            int n_col = base_n + col;

            ushort v0 = 0, v1 = 0;
            if (k_row0 < K && n_col < N)
                v0 = as_ushort(B[k_row0 * N + n_col]);
            if (k_row1 < K && n_col < N)
                v1 = as_ushort(B[k_row1 * N + n_col]);

            slm_b_vnni[pair * TILE_N + col] = ((uint)v1 << 16) | (uint)v0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === DPAS computation ===
        // Load A for this subgroup: 8 rows x 16 cols from slm_a
        // Each WI reads its column (sg_lid) across 8 rows
        __local ushort* a_base = &slm_a[sg_id * 8 * a_stride];

        short8 a_val = as_short8((ushort8)(
            a_base[0 * a_stride + sg_lid],
            a_base[1 * a_stride + sg_lid],
            a_base[2 * a_stride + sg_lid],
            a_base[3 * a_stride + sg_lid],
            a_base[4 * a_stride + sg_lid],
            a_base[5 * a_stride + sg_lid],
            a_base[6 * a_stride + sg_lid],
            a_base[7 * a_stride + sg_lid]));

        // Load B VNNI for cols 0-15: each WI reads its column (sg_lid) from 8 pairs
        __local uint* b_base0 = &slm_b_vnni[0];
        int8 b0 = (int8)(
            as_int(b_base0[0 * TILE_N + sg_lid]),
            as_int(b_base0[1 * TILE_N + sg_lid]),
            as_int(b_base0[2 * TILE_N + sg_lid]),
            as_int(b_base0[3 * TILE_N + sg_lid]),
            as_int(b_base0[4 * TILE_N + sg_lid]),
            as_int(b_base0[5 * TILE_N + sg_lid]),
            as_int(b_base0[6 * TILE_N + sg_lid]),
            as_int(b_base0[7 * TILE_N + sg_lid]));

        // Load B VNNI for cols 16-31
        int8 b1 = (int8)(
            as_int(b_base0[0 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[1 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[2 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[3 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[4 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[5 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[6 * TILE_N + 16 + sg_lid]),
            as_int(b_base0[7 * TILE_N + 16 + sg_lid]));

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b1, acc1);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
    int col0 = base_n + sg_lid;
    int col1 = base_n + 16 + sg_lid;

    float acc0_arr[8], acc1_arr[8];
    vstore8(acc0, 0, acc0_arr);
    vstore8(acc1, 0, acc1_arr);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = my_row_base + i;
        if (row < M) {
            if (col0 < N) C[row * N + col0] = convert_half(acc0_arr[i]);
            if (col1 < N) C[row * N + col1] = convert_half(acc1_arr[i]);
        }
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 332.000):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

// Tiling: each work-group (1 subgroup of 16 WIs) computes 32x32 output
// DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc) -> 8x16 output per call
// Register blocking: 4 vertical (4*8=32 rows) x 2 horizontal (2*16=32 cols) DPAS calls
// K tiled in steps of 16
// GWS = (ceil(N/32)*16, ceil(M/32)*1)  -- x has 16 WIs per subgroup
// LWS = (16, 1)

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile position
    const int wg_n = get_group_id(0); // which 32-col tile
    const int wg_m = get_group_id(1); // which 32-row tile

    const int base_m = wg_m * 32;
    const int base_n = wg_n * 32;

    const int sg_lid = get_sub_group_local_id(); // 0..15

    // Accumulators: 4 vertical x 2 horizontal = 8 float8 registers
    float8 acc00 = 0.0f, acc01 = 0.0f;
    float8 acc10 = 0.0f, acc11 = 0.0f;
    float8 acc20 = 0.0f, acc21 = 0.0f;
    float8 acc30 = 0.0f, acc31 = 0.0f;

    // SLM for A tile [32][16] and B tile [16][32] in half
    // A: 32*16 = 512 halfs = 1024 bytes
    // B: 16*32 = 512 halfs = 1024 bytes
    __local half slm_a[32 * 16];
    __local half slm_b[16 * 32];

    for (int k0 = 0; k0 < K; k0 += 16) {
        // Cooperative load A[32][16] - 512 halfs, 16 WIs, each loads 32 halfs
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int row = base_m + i;
            int col = k0 + sg_lid;
            half val = 0;
            if (row < M && col < K)
                val = A[row * K + col];
            slm_a[i * 16 + sg_lid] = val;
        }

        // Cooperative load B[16][32] - 512 halfs, 16 WIs, each loads 32 halfs
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int row = k0 + i;
            // Load two columns per WI to cover 32 columns
            int col0 = sg_lid;
            int col1 = sg_lid + 16;
            half val0 = 0, val1 = 0;
            if (row < K && (base_n + col0) < N)
                val0 = B[row * N + base_n + col0];
            if (row < K && (base_n + col1) < N)
                val1 = B[row * N + base_n + col1];
            slm_b[i * 32 + col0] = val0;
            slm_b[i * 32 + col1] = val1;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // DPAS calls: read from SLM using subgroup block reads
        // A: for each 8-row chunk, read short8 (8 halfs = 8 rows, k=16 deep packed)
        // For DPAS, A is short8 where each short holds 1 fp16, 8 shorts = 8 rows × 1 k-element?
        // Actually: short8 a means 8×16-bit = 8 fp16 values. For k16, we need 8 rows each with 16 fp16 values.
        // The DPAS k16 variant: a is short8 per WI, across subgroup of 16 WIs gives 8×16 = 128 fp16 = 8 rows × 16 k-elements
        // So each WI's short8 has: row[i]'s k-element at position sg_lid, for i=0..7
        // A layout in SLM: [row][k] with k stride 16, row stride 16 halfs
        // Block read: intel_sub_group_block_read_us8 reads 8 consecutive ushorts per WI from 16-aligned address
        // We need A in column-major within the 8x16 block for the DPAS format

        // A SLM layout is [32][16] row-major. For block read of 8 rows:
        // Address = &slm_a[chunk*8*16], stride = 16 halfs = 32 bytes
        // intel_sub_group_block_read_us8 with stride reads 8 consecutive rows

        // For A: need to use block_read with pitch. Let's manually construct short8.
        short8 a0, a1, a2, a3;

        __local ushort* slm_a_us = (__local ushort*)slm_a;

        // Chunk 0: rows 0-7
        a0 = as_short8((ushort8)(
            slm_a_us[0*16 + sg_lid], slm_a_us[1*16 + sg_lid],
            slm_a_us[2*16 + sg_lid], slm_a_us[3*16 + sg_lid],
            slm_a_us[4*16 + sg_lid], slm_a_us[5*16 + sg_lid],
            slm_a_us[6*16 + sg_lid], slm_a_us[7*16 + sg_lid]));

        a1 = as_short8((ushort8)(
            slm_a_us[8*16 + sg_lid], slm_a_us[9*16 + sg_lid],
            slm_a_us[10*16 + sg_lid], slm_a_us[11*16 + sg_lid],
            slm_a_us[12*16 + sg_lid], slm_a_us[13*16 + sg_lid],
            slm_a_us[14*16 + sg_lid], slm_a_us[15*16 + sg_lid]));

        a2 = as_short8((ushort8)(
            slm_a_us[16*16 + sg_lid], slm_a_us[17*16 + sg_lid],
            slm_a_us[18*16 + sg_lid], slm_a_us[19*16 + sg_lid],
            slm_a_us[20*16 + sg_lid], slm_a_us[21*16 + sg_lid],
            slm_a_us[22*16 + sg_lid], slm_a_us[23*16 + sg_lid]));

        a3 = as_short8((ushort8)(
            slm_a_us[24*16 + sg_lid], slm_a_us[25*16 + sg_lid],
            slm_a_us[26*16 + sg_lid], slm_a_us[27*16 + sg_lid],
            slm_a_us[28*16 + sg_lid], slm_a_us[29*16 + sg_lid],
            slm_a_us[30*16 + sg_lid], slm_a_us[31*16 + sg_lid]));

        // For B: [16][32] row-major. DPAS B operand int8 per WI.
        // B matrix for DPAS: 16 k-elements × 16 n-elements (one DPAS n-block)
        // int8 = 8 ints = 8×32bit = 8×2×fp16 = 16 fp16 per WI, across 16 WIs = 256 fp16 = 16×16
        // B layout for DPAS: VNNI format - pairs of k are packed into 32-bit
        // B[k][n] -> for VNNI: B_packed[k/2][n][2] as uint
        // SLM B is [16][32]. For n-block 0 (cols 0-15):
        //   We need B[0..15][0..15] in VNNI format

        __local uint* slm_b_ui = (__local uint*)slm_b;
        // slm_b as ushort: slm_b[k*32 + n]
        // As uint (pairs of halfs): slm_b_ui[k*16 + n/2]... no, that pairs adjacent columns
        // We need VNNI: pack k=2i and k=2i+1 for same n column
        // slm_b[2i*32+n] and slm_b[(2i+1)*32+n] -> packed as uint

        // B VNNI for n-block 0 (n=0..15): b_vnni[i][n] = pack(B[2i][n], B[2i+1][n])
        // i=0..7, n=0..15 -> int8 per WI (WI sg_lid reads n=sg_lid)

        __local ushort* slm_b_us = (__local ushort*)slm_b;

        int8 b0, b1;

        // B block 0: cols 0-15
        b0 = (int8)(
            as_int((ushort2)(slm_b_us[0*32 + sg_lid], slm_b_us[1*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[2*32 + sg_lid], slm_b_us[3*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[4*32 + sg_lid], slm_b_us[5*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[6*32 + sg_lid], slm_b_us[7*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[8*32 + sg_lid], slm_b_us[9*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[10*32 + sg_lid], slm_b_us[11*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[12*32 + sg_lid], slm_b_us[13*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[14*32 + sg_lid], slm_b_us[15*32 + sg_lid])));

        // B block 1: cols 16-31
        b1 = (int8)(
            as_int((ushort2)(slm_b_us[0*32 + 16 + sg_lid], slm_b_us[1*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[2*32 + 16 + sg_lid], slm_b_us[3*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[4*32 + 16 + sg_lid], slm_b_us[5*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[6*32 + 16 + sg_lid], slm_b_us[7*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[8*32 + 16 + sg_lid], slm_b_us[9*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[10*32 + 16 + sg_lid], slm_b_us[11*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[12*32 + 16 + sg_lid], slm_b_us[13*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[14*32 + 16 + sg_lid], slm_b_us[15*32 + 16 + sg_lid])));

        // 4×2 = 8 DPAS calls
        acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
        acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
        acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
        acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
        acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc20);
        acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc21);
        acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc30);
        acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc31);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: each accumulator is float8 = 8 rows × 1 col per WI, 16 WIs = 8 rows × 16 cols
    // acc[row_chunk][col_chunk].s[i] = C[base_m + chunk*8 + i][base_n + col_chunk*16 + sg_lid]

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = base_m + 0*8 + i;
        int col0 = base_n + sg_lid;
        int col1 = base_n + 16 + sg_lid;
        if (row < M) {
            float v0, v1;
            switch(i) {
                case 0: v0 = acc00.s0; v1 = acc01.s0; break;
                case 1: v0 = acc00.s1; v1 = acc01.s1; break;
                case 2: v0 = acc00.s2; v1 = acc01.s2; break;
                case 3: v0 = acc00.s3; v1 = acc01.s3; break;
                case 4: v0 = acc00.s4; v1 = acc01.s4; break;
                case 5: v0 = acc00.s5; v1 = acc01.s5; break;
                case 6: v0 = acc00.s6; v1 = acc01.s6; break;
                case 7: v0 = acc00.s7; v1 = acc01.s7; break;
            }
            if (col0 < N) C[row * N + col0] = convert_half(v0);
            if (col1 < N) C[row * N + col1] = convert_half(v1);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = base_m + 8 + i;
        int col0 = base_n + sg_lid;
        int col1 = base_n + 16 + sg_lid;
        if (row < M) {
            float v0, v1;
            switch(i) {
                case 0: v0 = acc10.s0; v1 = acc11.s0; break;
                case 1: v0 = acc10.s1; v1 = acc11.s1; break;
                case 2: v0 = acc10.s2; v1 = acc11.s2; break;
                case 3: v0 = acc10.s3; v1 = acc11.s3; break;
                case 4: v0 = acc10.s4; v1 = acc11.s4; break;
                case 5: v0 = acc10.s5; v1 = acc11.s5; break;
                case 6: v0 = acc10.s6; v1 = acc11.s6; break;
                case 7: v0 = acc10.s7; v1 = acc11.s7; break;
            }
            if (col0 < N) C[row * N + col0] = convert_half(v0);
            if (col1 < N) C[row * N + col1] = convert_half(v1);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = base_m + 16 + i;
        int col0 = base_n + sg_lid;
        int col1 = base_n + 16 + sg_lid;
        if (row < M) {
            float v0, v1;
            switch(i) {
                case 0: v0 = acc20.s0; v1 = acc21.s0; break;
                case 1: v0 = acc20.s1; v1 = acc21.s1; break;
                case 2: v0 = acc20.s2; v1 = acc21.s2; break;
                case 3: v0 = acc20.s3; v1 = acc21.s3; break;
                case 4: v0 = acc20.s4; v1 = acc21.s4; break;
                case 5: v0 = acc20.s5; v1 = acc21.s5; break;
                case 6: v0 = acc20.s6; v1 = acc21.s6; break;
                case 7: v0 = acc20.s7; v1 = acc21.s7; break;
            }
            if (col0 < N) C[row * N + col0] = convert_half(v0);
            if (col1 < N) C[row * N + col1] = convert_half(v1);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = base_m + 24 + i;
        int col0 = base_n + sg_lid;
        int col1 = base_n + 16 + sg_lid;
        if (row < M) {
            float v0, v1;
            switch(i) {
                case 0: v0 = acc30.s0; v1 = acc31.s0; break;
                case 1: v0 = acc30.s1; v1 = acc31.s1; break;
                case 2: v0 = acc30.s2; v1 = acc31.s2; break;
                case 3: v0 = acc30.s3; v1 = acc31.s3; break;
                case 4: v0 = acc30.s4; v1 = acc31.s4; break;
                case 5: v0 = acc30.s5; v1 = acc31.s5; break;
                case 6: v0 = acc30.s6; v1 = acc31.s6; break;
                case 7: v0 = acc30.s7; v1 = acc31.s7; break;
            }
            if (col0 < N) C[row * N + col0] = convert_half(v0);
            if (col1 < N) C[row * N + col1] = convert_half(v1);
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 102.000):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Tile: 32M x 64N per work-group, K-tile = 16
// 4 subgroups per WG, each subgroup computes 8M x 64N (using 4 DPAS calls per K-step)
// Actually: 4 subgroups, each handles 8 rows x 64 cols = 4 x (8x16) DPAS blocks
// LWS = (16, 4) => 64 WIs, 4 subgroups of 16
// GWS = (ceil(N/64)*16, ceil(M/32)*4)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 16
#define SG_SIZE 16
#define NUM_SG 4

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;

    const int sg_id = get_local_id(1);  // 0..3, which subgroup (row block)
    const int sg_lid = get_sub_group_local_id(); // 0..15

    const int row_base = wg_m + sg_id * 8; // each subgroup handles 8 rows

    // Each subgroup computes 8 rows x 64 cols = 4 DPAS blocks (8x16 each)
    float8 acc0 = 0.0f; // cols 0-15
    float8 acc1 = 0.0f; // cols 16-31
    float8 acc2 = 0.0f; // cols 32-47
    float8 acc3 = 0.0f; // cols 48-63

    // SLM: A[32][16] + B in VNNI format [8][64] as uint (= 16 k-rows x 64 cols packed)
    // A: 32*16 halfs = 512 halfs = 1024 bytes
    // B_vnni: 8*64 uints = 512 uints = 2048 bytes (16 k-rows x 64 cols, k-pairs packed)
    __local ushort slm_a[TILE_M * TILE_K]; // 32*16 = 512
    __local uint slm_b_vnni[8 * TILE_N];   // 8*64 = 512

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // === Cooperative load A[32][16] ===
        // 64 WIs total, 512 elements => 8 elements per WI
        {
            int flat_id = get_local_id(1) * 16 + get_local_id(0); // 0..63
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id * 8 + i;
                int r = idx >> 4;  // idx / 16
                int c = idx & 15;  // idx % 16
                int gr = wg_m + r;
                int gc = k0 + c;
                ushort val = 0;
                if (gr < M && gc < K)
                    val = as_ushort(A[gr * K + gc]);
                slm_a[r * TILE_K + c] = val;
            }
        }

        // === Cooperative load B[16][64] into VNNI format ===
        // VNNI: pack k-pairs => 8 pairs x 64 cols = 512 uints
        // 64 WIs, 512 elements => 8 per WI
        {
            int flat_id = get_local_id(1) * 16 + get_local_id(0);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id * 8 + i;
                int pair = idx >> 6;  // idx / 64, which k-pair (0..7)
                int col = idx & 63;   // idx % 64, which column (0..63)
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
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === Load A for this subgroup: 8 rows x 16 cols ===
        __local ushort* a_ptr = slm_a + sg_id * 8 * TILE_K;
        short8 a_val = as_short8((ushort8)(
            a_ptr[0*TILE_K + sg_lid], a_ptr[1*TILE_K + sg_lid],
            a_ptr[2*TILE_K + sg_lid], a_ptr[3*TILE_K + sg_lid],
            a_ptr[4*TILE_K + sg_lid], a_ptr[5*TILE_K + sg_lid],
            a_ptr[6*TILE_K + sg_lid], a_ptr[7*TILE_K + sg_lid]));

        // === Load B VNNI blocks (already packed) ===
        // Block j: cols [j*16 .. j*16+15], 8 pairs
        // Each WI reads its column (sg_lid) from each pair
        __local uint* b_base = slm_b_vnni;

        int8 b0 = (int8)(
            as_int(b_base[0*TILE_N + 0*16 + sg_lid]), as_int(b_base[1*TILE_N + 0*16 + sg_lid]),
            as_int(b_base[2*TILE_N + 0*16 + sg_lid]), as_int(b_base[3*TILE_N + 0*16 + sg_lid]),
            as_int(b_base[4*TILE_N + 0*16 + sg_lid]), as_int(b_base[5*TILE_N + 0*16 + sg_lid]),
            as_int(b_base[6*TILE_N + 0*16 + sg_lid]), as_int(b_base[7*TILE_N + 0*16 + sg_lid]));

        int8 b1 = (int8)(
            as_int(b_base[0*TILE_N + 16 + sg_lid]), as_int(b_base[1*TILE_N + 16 + sg_lid]),
            as_int(b_base[2*TILE_N + 16 + sg_lid]), as_int(b_base[3*TILE_N + 16 + sg_lid]),
            as_int(b_base[4*TILE_N + 16 + sg_lid]), as_int(b_base[5*TILE_N + 16 + sg_lid]),
            as_int(b_base[6*TILE_N + 16 + sg_lid]), as_int(b_base[7*TILE_N + 16 + sg_lid]));

        int8 b2 = (int8)(
            as_int(b_base[0*TILE_N + 32 + sg_lid]), as_int(b_base[1*TILE_N + 32 + sg_lid]),
            as_int(b_base[2*TILE_N + 32 + sg_lid]), as_int(b_base[3*TILE_N + 32 + sg_lid]),
            as_int(b_base[4*TILE_N + 32 + sg_lid]), as_int(b_base[5*TILE_N + 32 + sg_lid]),
            as_int(b_base[6*TILE_N + 32 + sg_lid]), as_int(b_base[7*TILE_N + 32 + sg_lid]));

        int8 b3 = (int8)(
            as_int(b_base[0*TILE_N + 48 + sg_lid]), as_int(b_base[1*TILE_N + 48 + sg_lid]),
            as_int(b_base[2*TILE_N + 48 + sg_lid]), as_int(b_base[3*TILE_N + 48 + sg_lid]),
            as_int(b_base[4*TILE_N + 48 + sg_lid]), as_int(b_base[5*TILE_N + 48 + sg_lid]),
            as_int(b_base[6*TILE_N + 48 + sg_lid]), as_int(b_base[7*TILE_N + 48 + sg_lid]));

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b2, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b3, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
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

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] PASSED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] PASSED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] PASSED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] PASSED         [100%]

=============================== warnings summary ===============================
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0]
  /home/openvino-ci-74/miniforge3/envs/kernel_intel/lib/python3.12/site-packages/pyopencl/cache.py:517: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
    _create_built_program_from_source_cached(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 4 passed, 1 deselected, 1 warning in 1.09s ==================
The kernel compiles and is correct, great job!

## Hardware specification:
Your code will run on the following hardware:
**Intel Battlemage** with specs: Xe-cores: 20, Render Slices: 5, Ray Tracing Units: 20, Intel® Xe Matrix Extensions (Intel® XMX) Engines: 160, Xe Vector Engines: 160, Graphics Clock: 2670, GPU Peak TOPS (Int8): 233, TBP: 190, PCI Express Configurations ‡: PCI Express 4.0 x8, Device ID: 0xE20B, Memory: 12 GB GDDR6, Memory Interface: 192 bit, Memory Bandwidth: 456, Memory Speed: 19, ISA_GPU: Xe2-HPG
Please consider the hardware specifications when improving the code. 

## Task:

**Your objectives**:
1. Analyze the previous versions and their results (why does one achieve better results than the other?).
2. Identify any inefficiencies and bottlenecks.
3. Propose specific improvements or options to take the best of all prior versions, explaining your reasoning step by step.

4. Provide a new kernel that achieves better performance **on the target hardware**. Provide the complete, improved code in a code block.

**Optimization strategies**:

Here are some potential strategies to improve the kernel runtime:
1. Minimize Synchronization: Reduce barrier() calls. Use barrier(CLK_LOCAL_MEM_FENCE) for local memory only, barrier(CLK_GLOBAL_MEM_FENCE) when needed for global. Sub-group operations are barrier-free. Use work-group-level reductions only when necessary.
2. Maximize Occupancy: Balance work-group size with register/local memory usage. Query CL_KERNEL_WORK_GROUP_SIZE and CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE. Typical work-group sizes: 64-256 for compute-bound, 32-128 for memory-bound kernels. Use multiples of wavefront/warp size (32 for NVIDIA, 64 for AMD).

**Critical Requirements:**

1. The kernel must exactly match the reference implementation's functionality.
2. The code must compile and run properly on the GPU.
3. Do not cache or reuse previous results; ensure the code executes fully on each run.
4. Keep all hyperparameters (e.g., batch size, dimensions) unchanged as specified in the reference implementation.
8. Beware of the critical error "Unexpected kernel lambda size. In such cases removing constexpr specifier aligns the captures between the host compiler and the device compiler"! Do not capture constexpr variables in lambda functions passed to kernel launches as this can lead to different lambda sizes between the host and device compiler.

Additional Guidance:

1. Clearly comment on any performance optimizations you implement.
2. If you change the kernel structure, explain why.
3. Anticipate and address possible runtime or compilation errors.

Please structure your response as follows:

1. Analysis:
    * Summarize the issues found in the previous kernel and log.
    * Explain your proposed changes and optimizations.
2. Improved OCL code:
    * Provide the complete, improved OCL code in a code block:
```OCL
Your code here
```


## Required Optimizations

Apply the following optimization techniques in your implementation:

- **Register Blocking**: Each work-item computes a THREAD_M×THREAD_N output block in private register arrays. Use `#pragma unroll` on inner loops. Combine with SLM tiling for multi-level memory hierarchy optimization.
- **Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade recomputation for memory savings (e.g., Flash-Attention style). Maintain running accumulators across blocks with proper rescaling.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.