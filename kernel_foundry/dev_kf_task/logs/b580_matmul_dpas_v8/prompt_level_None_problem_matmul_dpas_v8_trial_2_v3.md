

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 332.000):
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

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 102.000):
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// GWS = (N/16 * 16, M/32) = (N, M/32)  -- dim0 is subgroup-aware
// LWS = (64, 1)  -- 4 subgroups of 16
// Subgroup size = 16
// Each workgroup computes a 32x64 tile of C
// Each subgroup computes a 32x16 tile (4 DPAS calls of 8x16 each)

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
    const int sg_id = get_sub_group_id();           // 0..3
    const int sg_lid = get_sub_group_local_id();    // 0..15

    // Workgroup tile position
    const int tile_row = get_group_id(1) * 32;      // 32 rows per WG
    const int tile_col = get_group_id(0) * 64 + sg_id * 16; // 64 cols per WG, 16 per SG

    // Accumulators: 4 blocks of 8 rows each = 32 rows
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // K-loop in steps of 16
    for (int k = 0; k < K; k += 16) {
        // Load A: 32 rows x 16 cols (half), each DPAS needs 8 rows x 16 cols = short8 per WI
        // A is [M, K], row-major. For rows [tile_row + r*8 .. tile_row + r*8+7], cols [k..k+15]
        // Each work-item in subgroup reads the same A data (broadcast across subgroup)
        // For DPAS, A (short8) = 8 rows, each row has 16 halfs, packed as short8 per work-item
        // A layout for DPAS: A[row][k] where each WI holds rows 0-7, with k distributed across WIs

        // A block read: 8 rows x 16 halfs. Use block_read_us8 from A pointer
        // A is row-major [M,K]. We need contiguous 16 halfs per row.
        // block_read_us8 reads 8 sequential ushorts per WI, with subgroup reading 8*16 = 128 ushorts
        // For DPAS A matrix: need VNNI-like format or row-major with stride

        // For intel_sub_group_f16_f16_matrix_mad_k16:
        // A (short8): each WI holds 8 pairs of fp16 values. WI_i holds A[row][2*i] and A[row][2*i+1] packed.
        // Actually, A is packed so that each short8 element corresponds to one row, 
        // with the 16 k-values distributed across the 16 subgroup lanes (1 half per lane per row).
        // But short holds 2 halfs packed, so it's actually k=16 with pairs.

        // A: short8 means 8 elements, each short packs 2 fp16. 
        // Layout: a[i] = {A[row_base+i][k + 2*sg_lid], A[row_base+i][k + 2*sg_lid+1]}

        short8 a0, a1, a2, a3;

        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = tile_row + r;
            if (row_idx < M) {
                __global const ushort* a_ptr = (__global const ushort*)(A + row_idx * K + k);
                ((short*)&a0)[r] = as_short(intel_sub_group_block_read_us(a_ptr));
            } else {
                ((short*)&a0)[r] = 0;
            }
        }

        // Actually, let me reconsider the A/B formats for DPAS.
        // 
        // intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
        // - Computes C[8][16] += A[8][16] * B[16][16]
        // - a (short8): A matrix, 8 rows x 16 cols. Each short packs 2 fp16.
        //   WI_lane holds: a[j] = pack(A[j][2*lane], A[j][2*lane+1]) for j=0..7
        // - b (int8): B matrix, 16 rows x 16 cols. Each int packs 2 fp16.
        //   WI_lane holds: b[j] = pack(B[2*j][lane], B[2*j+1][lane]) for j=0..7
        // - acc (float8): C[j][lane] for j=0..7

        // Load A tiles (4 blocks of 8 rows)
        // A[tile_row + block*8 + j][k + 2*lane : k + 2*lane + 1]

        #pragma unroll
        for (int block = 0; block < 4; block++) {
            short8 a_val;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int r = tile_row + block * 8 + j;
                if (r < M && (k + 2 * sg_lid + 1) < K) {
                    __global const half* a_base = A + r * K + k;
                    ushort2 tmp;
                    tmp.x = *((__global const ushort*)(a_base + 2 * sg_lid));
                    tmp.y = *((__global const ushort*)(a_base + 2 * sg_lid + 1));
                    ((short*)&a_val)[j] = as_short(tmp.x) | (as_short(tmp.y) << 16);
                } else {
                    ((short*)&a_val)[j] = 0;
                }
            }
            if (block == 0) a0 = a_val;
            else if (block == 1) a1 = a_val;
            else if (block == 2) a2 = a_val;
            else a3 = a_val;
        }

        // Load B tile: B[k..k+15][tile_col..tile_col+15]
        // b[j] = pack(B[k+2*j][tile_col+lane], B[k+2*j+1][tile_col+lane])
        int8 b_val;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int kr0 = k + 2 * j;
            int kr1 = k + 2 * j + 1;
            ushort v0 = 0, v1 = 0;
            if (kr0 < K && (tile_col + sg_lid) < N)
                v0 = *((__global const ushort*)(B + kr0 * N + tile_col + sg_lid));
            if (kr1 < K && (tile_col + sg_lid) < N)
                v1 = *((__global const ushort*)(B + kr1 * N + tile_col + sg_lid));
            ((int*)&b_val)[j] = (int)v0 | ((int)v1 << 16);
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
    }

    // Store results
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int r = tile_row + j;
        int c = tile_col + sg_lid;
        if (r < M && c < N)
            C[r * N + c] = convert_half(((float*)&acc0)[j]);
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int r = tile_row + 8 + j;
        int c = tile_col + sg_lid;
        if (r < M && c < N)
            C[r * N + c] = convert_half(((float*)&acc1)[j]);
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int r = tile_row + 16 + j;
        int c = tile_col + sg_lid;
        if (r < M && c < N)
            C[r * N + c] = convert_half(((float*)&acc2)[j]);
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int r = tile_row + 24 + j;
        int c = tile_col + sg_lid;
        if (r < M && c < N)
            C[r * N + c] = convert_half(((float*)&acc3)[j]);
    }
}
```

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] FAILED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] FAILED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] FAILED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] FAILED         [100%]

=================================== FAILURES ===================================
________________ TestMatmulOCL.test_correctness_wrt_pytorch[0] _________________

self = <task.TestMatmulOCL object at 0x72ba4655cc50>
kernel = <function initialize_matmul_kernel.<locals>.<lambda> at 0x72ba4659f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72ba465a9940>, _run = 0

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        args, expected = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=_run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        assert got.shape == expected.shape
>       assert np.allclose(got, expected, rtol=2e-2, atol=2e-2), "matmul result mismatch vs pytorch/numpy"
E       AssertionError: matmul result mismatch vs pytorch/numpy
E       assert False
E        +  where False = <function allclose at 0x72ba9ef5c430>(array([[  83.375    ,  -40.9375   ,    2.5117188, ...,    0.       ,\n           0.       ,    0.       ],\n       [ -37.25     ,  -60.65625  ,  110.875    , ...,    0.       ,\n           0.       ,    0.       ],\n       [ -46.5      ,  -36.5      ,   38.34375  , ...,    0.       ,\n           0.       ,    0.       ],\n       ...,\n       [ -36.4375   ,  -29.359375 ,  -41.71875  , ...,    0.       ,\n           0.       ,    0.       ],\n       [ -21.75     ,  -44.90625  ,   25.140625 , ...,    0.       ,\n           0.       ,    0.       ],\n       [  65.3125   ,  -14.53125  , -102.25     , ...,    0.       ,\n           0.       ,    0.       ]], shape=(2048, 2048), dtype=float32), array([[-12.434087 , -85.22102  ,  45.86866  , ..., -67.074715 ,\n        -64.52674  ,  37.798523 ],\n       [-82.95244  ,  28.332115 ,   4.3084497, ...,  37.17192  ,\n         48.87541  ,  55.1519   ],\n       [ 31.096529 , -51.77693  ,  -9.3054905, ...,   8.124319 ,\n         61.21928  ,   4.7092314],\n       ...,\n       [-65.29967  , -27.73106  ,  74.195465 , ..., 122.09403  ,\n        -41.569603 ,  10.711429 ],\n       [ 44.6838   ,   2.3142765,  22.61605  , ..., -35.807106 ,\n         42.793472 ,  52.60636  ],\n       [ 50.399834 ,  -3.015791 ,  21.545517 , ..., -21.399685 ,\n        -36.035267 ,  49.01544  ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72ba9ef5c430> = np.allclose

task.py:99: AssertionError
---------------------------- Captured stdout setup -----------------------------
[WARN] matmul_opt.cl has no parsable reqd_work_group_size; fallback launch (N, M), local=None
________________ TestMatmulOCL.test_correctness_wrt_pytorch[1] _________________

self = <task.TestMatmulOCL object at 0x72ba4655e000>
kernel = <function initialize_matmul_kernel.<locals>.<lambda> at 0x72ba4659f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72ba465a9940>, _run = 1

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_pytorch(self, kernel, ocl_queue, _run):
        args, expected = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=_run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        assert got.shape == expected.shape
>       assert np.allclose(got, expected, rtol=2e-2, atol=2e-2), "matmul result mismatch vs pytorch/numpy"
E       AssertionError: matmul result mismatch vs pytorch/numpy
E       assert False
E        +  where False = <function allclose at 0x72ba9ef5c430>(array([[-37.84375 , -68.9375  ,  45.3125  , ...,   0.      ,   0.      ,\n          0.      ],\n       [-72.75    , -31.796875, -32.09375 , ...,   0.      ,   0.      ,\n          0.      ],\n       [ 94.9375  , -20.171875,  65.5     , ...,   0.      ,   0.      ,\n          0.      ],\n       ...,\n       [-26.78125 ,  39.5     , -28.5625  , ...,   0.      ,   0.      ,\n          0.      ],\n       [-43.5     ,  36.59375 ,  80.0625  , ...,   0.      ,   0.      ,\n          0.      ],\n       [ 43.59375 ,  58.03125 , -12.65625 , ...,   0.      ,   0.      ,\n          0.      ]], shape=(2048, 2048), dtype=float32), array([[  63.220627 ,   13.663691 ,   62.708282 , ...,   26.950535 ,\n        -100.14888  ,  -76.10468  ],\n       [  30.338015 ,   -9.576593 ,  -15.848044 , ...,  -86.66203  ,\n           6.3691177,    9.569207 ],\n       [  -9.825886 ,   26.83852  ,  -39.88768  , ...,   94.32298  ,\n         -40.437588 ,   13.349518 ],\n       ...,\n       [ -50.946926 ,  -10.7210655,  -18.652342 , ...,   -4.0612535,\n         -29.112085 ,   -2.7683525],\n       [ -41.46417  ,   -5.034666 ,   35.500336 , ...,    3.5289268,\n          14.26104  ,   55.58531  ],\n       [ -33.896618 ,   51.45737  ,   13.108513 , ...,   11.92079  ,\n         -64.022385 ,   63.048595 ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72ba9ef5c430> = np.allclose

task.py:99: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[0] ________________

self = <task.TestMatmulOCL object at 0x72ba46596c90>
kernel = <function initialize_matmul_kernel.<locals>.<lambda> at 0x72ba4659f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72ba465a9940>, _run = 0

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        args, _ = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=100 + _run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        ref_kernel = initialize_matmul_kernel("matmul_reference.cl", ocl_queue)
        ref_kernel(*args)
        ref_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, ref_flat, out_buf)
        ref = ref_flat.reshape((int(m), int(n))).astype(np.float32)

>       assert np.allclose(got, ref, rtol=2e-2, atol=2e-2), "matmul result mismatch vs reference"
E       AssertionError: matmul result mismatch vs reference
E       assert False
E        +  where False = <function allclose at 0x72ba9ef5c430>(array([[ -67.3125   ,   73.125    ,  -19.296875 , ...,    0.       ,\n           0.       ,    0.       ],\n       [  19.984375 ,   30.609375 ,  -11.1171875, ...,    0.       ,\n           0.       ,    0.       ],\n       [ -57.84375  ,  -64.75     ,   44.65625  , ...,    0.       ,\n           0.       ,    0.       ],\n       ...,\n       [ -27.859375 ,  -78.6875   , -117.       , ...,    0.       ,\n           0.       ,    0.       ],\n       [ -31.234375 ,  -19.296875 ,    7.84375  , ...,    0.       ,\n           0.       ,    0.       ],\n       [  -1.90625  ,  -24.765625 ,  -25.5      , ...,    0.       ,\n           0.       ,    0.       ]], shape=(2048, 2048), dtype=float32), array([[-5.7125000e+01, -2.8015625e+01,  6.5562500e+01, ...,\n        -5.0218750e+01,  2.7000000e+01,  2.6109375e+01],\n       [ 9.2687500e+01,  1.8343750e+01, -1.4343750e+01, ...,\n        -4.0843750e+01, -2.5828125e+01, -3.1914062e+00],\n       [-2.5585938e+00, -3.0015625e+01,  4.9937500e+01, ...,\n        -5.7156250e+01, -9.2250000e+01,  1.2921875e+01],\n       ...,\n       [-4.6031250e+01, -1.4262500e+02, -6.1500000e+01, ...,\n        -1.2194824e-01,  6.6445312e+00, -5.3156250e+01],\n       [ 3.1859375e+01, -2.3484375e+01,  4.2750000e+01, ...,\n         1.8390625e+01, -2.0507812e+00, -1.6000000e+02],\n       [-3.8968750e+01,  9.1750000e+01,  2.3953125e+01, ...,\n         4.5437500e+01,  5.4437500e+01,  1.0787500e+02]],\n      shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72ba9ef5c430> = np.allclose

task.py:117: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[1] ________________

self = <task.TestMatmulOCL object at 0x72ba46596f90>
kernel = <function initialize_matmul_kernel.<locals>.<lambda> at 0x72ba4659f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x72ba465a9940>, _run = 1

    @pytest.mark.parametrize("_run", range(2))
    def test_correctness_wrt_reference(self, kernel, ocl_queue, _run):
        args, _ = get_data_on_device(DEFAULT_SHAPE, ocl_queue, seed=100 + _run)
        kernel(*args)

        _, _, out_buf, m, _, n = args
        got_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, got_flat, out_buf)
        got = got_flat.reshape((int(m), int(n))).astype(np.float32)

        ref_kernel = initialize_matmul_kernel("matmul_reference.cl", ocl_queue)
        ref_kernel(*args)
        ref_flat = np.empty(int(m) * int(n), dtype=np.float16)
        cl.enqueue_copy(ocl_queue, ref_flat, out_buf)
        ref = ref_flat.reshape((int(m), int(n))).astype(np.float32)

>       assert np.allclose(got, ref, rtol=2e-2, atol=2e-2), "matmul result mismatch vs reference"
E       AssertionError: matmul result mismatch vs reference
E       assert False
E        +  where False = <function allclose at 0x72ba9ef5c430>(array([[-22.890625  ,  65.9375    , -63.78125   , ...,   0.        ,\n          0.        ,   0.        ],\n       [ -2.6757812 ,  17.71875   , -56.3125    , ...,   0.        ,\n          0.        ,   0.        ],\n       [ 61.125     ,  18.5625    ,  78.75      , ...,   0.        ,\n          0.        ,   0.        ],\n       ...,\n       [ -5.8789062 , -81.5       ,  40.1875    , ...,   0.        ,\n          0.        ,   0.        ],\n       [ -0.59277344, -21.4375    ,  32.25      , ...,   0.        ,\n          0.        ,   0.        ],\n       [-59.4375    ,  37.75      ,   7.40625   , ...,   0.        ,\n          0.        ,   0.        ]], shape=(2048, 2048), dtype=float32), array([[  46.4375   ,  -21.75     ,  -36.71875  , ...,  -39.96875  ,\n           3.125    , -147.25     ],\n       [ -13.859375 ,  -17.453125 ,   17.375    , ...,   57.90625  ,\n          25.859375 ,  -68.0625   ],\n       [  34.1875   ,   -9.421875 ,  -49.       , ...,  -46.71875  ,\n         -64.5625   ,   37.6875   ],\n       ...,\n       [   3.3261719,  -60.03125  ,  -64.       , ...,   18.328125 ,\n         -17.234375 ,  -30.90625  ],\n       [  -0.9145508,   97.25     ,  -18.109375 , ...,   35.75     ,\n         -45.0625   ,   21.78125  ],\n       [ -36.5      ,  -63.8125   ,   68.125    , ...,  -48.09375  ,\n         -29.515625 ,   11.1640625]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x72ba9ef5c430> = np.allclose

task.py:117: AssertionError
=============================== warnings summary ===============================
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0]
  /home/openvino-ci-74/miniforge3/envs/kernel_intel/lib/python3.12/site-packages/pyopencl/cache.py:517: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
    _create_built_program_from_source_cached(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] - AssertionErr...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] - AssertionErr...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[0] - AssertionE...
FAILED task.py::TestMatmulOCL::test_correctness_wrt_reference[1] - AssertionE...
================== 4 failed, 1 deselected, 1 warning in 1.09s ==================

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

- **Kernel Fusion**: Combine sequential operations (e.g., exp → add → activation) into a single kernel. Eliminate intermediate buffers by computing in registers.
- **Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing.