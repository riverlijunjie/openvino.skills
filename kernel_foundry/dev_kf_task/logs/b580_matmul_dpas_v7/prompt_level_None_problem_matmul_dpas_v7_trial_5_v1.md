

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
// Optimized FP16 matmul for Intel Battlemage (Xe2-HPG) using DPAS
// C[M,N] = A[M,K] x B[K,N], all half, accumulation in float
//
// Each subgroup (8 work-items) computes an 8x8 output tile using
// intel_sub_group_f16_f16_matrix_mad_k16 DPAS instruction.
//
// Work-group tile: TM x TN = 32 x 64
//   Arranged as 4 x 8 subgroups (4 along M, 8 along N) = 32 subgroups
//   Each subgroup: 8 work-items => WG size = 256
//
// LWS = (64, 4, 1)  [64 = 8 subgroups_x * 8 WIs, 4 = subgroups_y]
// GWS = (ceil(N/64)*64, ceil(M/32)*4, 1)
// Subgroup size = 8
//
// K-loop steps by 16 (DPAS k=16)

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
    // Identify subgroup position
    const int sg_local_id = get_sub_group_local_id(); // 0..7, lane within subgroup

    // Work-group position
    const int wg_x = get_group_id(0); // along N
    const int wg_y = get_group_id(1); // along M

    // Local IDs
    const int lid_x = get_local_id(0); // 0..63
    const int lid_y = get_local_id(1); // 0..3

    // Subgroup index within the work-group
    const int sg_idx_x = lid_x / 8; // 0..7 (which 8-col sub-tile along N)
    const int sg_idx_y = lid_y;      // 0..3 (which 8-row sub-tile along M)

    // Global tile origin for this subgroup's 8x8 output
    const int tile_m = wg_y * 32 + sg_idx_y * 8; // starting row
    const int tile_n = wg_x * 64 + sg_idx_x * 8; // starting col

    // Each work-item holds one row of the 8x8 output (accumulator)
    float8 acc = (float8)(0.0f);

    // Iterate over K dimension in steps of 16 (DPAS k-depth)
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        // Load A sub-tile: 8 rows x 16 cols
        // Each work-item (lane i) loads row (tile_m + i), cols [k, k+16)
        // We need half16 per work-item -> pack into appropriate DPAS format

        // For DPAS a-operand: each WI holds 8 halfs (half8) representing
        // one row of the 8x16 A-tile. But DPAS k=16 means we need 16 halfs.
        // Actually intel_sub_group_f16_f16_matrix_mad_k16 takes:
        //   a: int4 (8 halfs packed as 4 ints) per WI - but this is for 8x16
        //   Actually the signature uses short8 for a-operand for fp16

        // Load A: each WI loads 16 half values from its row
        int a_row = tile_m + sg_local_id;
        short8 a_val0, a_val1;

        if (a_row < M) {
            __global const short* A_short = (__global const short*)A;
            int a_base = a_row * K + k;
            // Load 16 halfs = 16 shorts
            a_val0 = (short8)(
                A_short[a_base + 0], A_short[a_base + 1],
                A_short[a_base + 2], A_short[a_base + 3],
                A_short[a_base + 4], A_short[a_base + 5],
                A_short[a_base + 6], A_short[a_base + 7]);
            a_val1 = (short8)(
                A_short[a_base + 8], A_short[a_base + 9],
                A_short[a_base + 10], A_short[a_base + 11],
                A_short[a_base + 12], A_short[a_base + 13],
                A_short[a_base + 14], A_short[a_base + 15]);
        } else {
            a_val0 = (short8)(0);
            a_val1 = (short8)(0);
        }

        // Load B sub-tile: 16 rows x 8 cols
        // For DPAS b-operand with subgroup size 8:
        // Each WI loads a column-pair from B, packed appropriately.
        // B is row-major: B[k_idx, tile_n + lane]
        // For k16: we need 16 rows of 8 cols.
        // int8 b: each int packs 2 halfs (consecutive k values for same n)

        int8 b_val;
        __global const short* B_short = (__global const short*)B;
        int b_col = tile_n + sg_local_id;

        if (b_col < N) {
            // Pack pairs of B values: B[k+2i, col] and B[k+2i+1, col] into one int
            int b0  = (int)(ushort)B_short[(k + 0) * N + b_col]  | ((int)(ushort)B_short[(k + 1) * N + b_col] << 16);
            int b1  = (int)(ushort)B_short[(k + 2) * N + b_col]  | ((int)(ushort)B_short[(k + 3) * N + b_col] << 16);
            int b2  = (int)(ushort)B_short[(k + 4) * N + b_col]  | ((int)(ushort)B_short[(k + 5) * N + b_col] << 16);
            int b3  = (int)(ushort)B_short[(k + 6) * N + b_col]  | ((int)(ushort)B_short[(k + 7) * N + b_col] << 16);
            int b4  = (int)(ushort)B_short[(k + 8) * N + b_col]  | ((int)(ushort)B_short[(k + 9) * N + b_col] << 16);
            int b5  = (int)(ushort)B_short[(k + 10) * N + b_col] | ((int)(ushort)B_short[(k + 11) * N + b_col] << 16);
            int b6  = (int)(ushort)B_short[(k + 12) * N + b_col] | ((int)(ushort)B_short[(k + 13) * N + b_col] << 16);
            int b7  = (int)(ushort)B_short[(k + 14) * N + b_col] | ((int)(ushort)B_short[(k + 15) * N + b_col] << 16);
            b_val = (int8)(b0, b1, b2, b3, b4, b5, b6, b7);
        } else {
            b_val = (int8)(0);
        }

        // DPAS: 8x16 * 16x8 -> 8x8, accumulated in float
        // a-operand for k16: need int4 per WI (16 halfs = 8 ints? No, 16 halfs = 16 shorts = 8 ints)
        // The intrinsic signature: float8 intel_sub_group_f16_f16_matrix_mad_k16(int4 a, int8 b, float8 acc)
        // Pack a into int4: each int holds 2 halfs
        int4 a_packed;
        a_packed.s0 = as_int((short2)(a_val0.s0, a_val0.s1));
        a_packed.s1 = as_int((short2)(a_val0.s2, a_val0.s3));
        a_packed.s2 = as_int((short2)(a_val0.s4, a_val0.s5));
        a_packed.s3 = as_int((short2)(a_val0.s6, a_val0.s7));

        int4 a_packed2;
        a_packed2.s0 = as_int((short2)(a_val1.s0, a_val1.s1));
        a_packed2.s1 = as_int((short2)(a_val1.s2, a_val1.s3));
        a_packed2.s2 = as_int((short2)(a_val1.s4, a_val1.s5));
        a_packed2.s3 = as_int((short2)(a_val1.s6, a_val1.s7));

        // Full int8 for a-operand (16 halfs = 8 ints)
        int8 a_full = (int8)(a_packed.s0, a_packed.s1, a_packed.s2, a_packed.s3,
                              a_packed2.s0, a_packed2.s1, a_packed2.s2, a_packed2.s3);

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_full, b_val, acc);
    }

    // Handle remainder K (k < K) with scalar fallback
    for (; k < K; k++) {
        int a_row = tile_m + sg_local_id;
        float a_val = 0.0f;
        if (a_row < M) {
            a_val = convert_float(A[a_row * K + k]);
        }

        // Each lane needs B[k, tile_n + 0..7]
        int b_col = tile_n + sg_local_id;
        float b_lane = 0.0f;
        if (b_col < N) {
            b_lane = convert_float(B[k * N + b_col]);
        }

        // Broadcast a_val and multiply with all 8 B columns via shuffle
        for (int j = 0; j < 8; j++) {
            float bj = intel_sub_group_shuffle(b_lane, j);
            // We need: acc[j] += a_val * bj for each WI's row
            // acc.sN corresponds to column N of the output
            switch(j) {
                case 0: acc.s0 += a_val * intel_sub_group_shuffle(b_lane, 0); break;
                case 1: acc.s1 += a_val * intel_sub_group_shuffle(b_lane, 1); break;
                case 2: acc.s2 += a_val * intel_sub_group_shuffle(b_lane, 2); break;
                case 3: acc.s3 += a_val * intel_sub_group_shuffle(b_lane, 3); break;
                case 4: acc.s4 += a_val * intel_sub_group_shuffle(b_lane, 4); break;
                case 5: acc.s5 += a_val * intel_sub_group_shuffle(b_lane, 5); break;
                case 6: acc.s6 += a_val * intel_sub_group_shuffle(b_lane, 6); break;
                case 7: acc.s7 += a_val * intel_sub_group_shuffle(b_lane, 7); break;
            }
        }
    }

    // Store 8x8 output tile
    // Each WI (lane sg_local_id) holds row (tile_m + sg_local_id), cols tile_n..tile_n+7
    int out_row = tile_m + sg_local_id;
    if (out_row < M) {
        half8 result = convert_half8(acc);
        int out_base = out_row * N + tile_n;

        if (tile_n + 8 <= N) {
            // Full 8-wide store
            vstore8(result, 0, C + out_base);
        } else {
            // Partial store for boundary
            for (int j = 0; j < 8 && (tile_n + j) < N; j++) {
                switch(j) {
                    case 0: C[out_base + 0] = result.s0; break;
                    case 1: C[out_base + 1] = result.s1; break;
                    case 2: C[out_base + 2] = result.s2; break;
                    case 3: C[out_base + 3] = result.s3; break;
                    case 4: C[out_base + 4] = result.s4; break;
                    case 5: C[out_base + 5] = result.s5; break;
                    case 6: C[out_base + 6] = result.s6; break;
                    case 7: C[out_base + 7] = result.s7; break;
                }
            }
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS + SLM tiling
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Launch metadata:
//   LWS = (64, 4, 1) => 256 work-items = 32 subgroups of 8
//   GWS = (ceil(N/64)*64, ceil(M/32)*4, 1)
//   Subgroup size = 8
//   Work-group tile: 32 rows x 64 cols of C
//   Subgroup grid: 4 rows x 8 cols => 32 subgroups, each 8x8 output
//   K-tile = 32 (two DPAS k16 steps per SLM load)
//   SLM: A[32][32] + B_vnni[16][64] as uint = 32*32*2 + 16*64*4 = 2048+4096 = 6144 bytes per buffer
//   Double-buffered: ~12288 bytes total

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
    const int TM = 32;   // rows per work-group
    const int TN = 64;   // cols per work-group
    const int TK = 32;   // K depth per SLM load (2x DPAS k16)

    // Work-group origin in output C
    const int wg_n = get_group_id(0) * TN;
    const int wg_m = get_group_id(1) * TM;

    // Flat local ID for cooperative loading
    const int lid0 = get_local_id(0);  // 0..63
    const int lid1 = get_local_id(1);  // 0..3
    const int flat_lid = lid1 * 64 + lid0;  // 0..255

    // Subgroup identification
    const int sg_id = get_sub_group_id();         // 0..31
    const int sg_lid = get_sub_group_local_id();  // 0..7

    // Map subgroups to 4x8 grid (4 along M, 8 along N)
    const int sg_row = sg_id / 8;  // 0..3
    const int sg_col = sg_id % 8;  // 0..7

    // Double-buffered SLM
    // A: [2][TM][TK] as half = [2][32][32] halves
    // B in VNNI: [2][TK/2][TN] as uint = [2][16][64] uints
    __local half slm_A[2][TM * TK];        // 2 * 32 * 32 * 2 = 4096 bytes
    __local uint slm_B_vnni[2][TK/2 * TN]; // 2 * 16 * 64 * 4 = 8192 bytes

    // Accumulator: 8x8 per subgroup
    float8 acc = (float8)(0.0f);

    const int num_k_tiles = (K + TK - 1) / TK;
    int buf = 0;

    // Global pointers cast to ushort for packing
    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    // === Preload first tile into buffer 0 ===
    {
        int k0 = 0;

        // Load A[wg_m..+31][k0..+31]: 32*32=1024 halves, 256 threads => 4 each
        for (int i = flat_lid; i < TM * TK; i += 256) {
            int r = i / TK;  // row within tile
            int c = i % TK;  // col within tile (K dim)
            int gr = wg_m + r;
            int gc = k0 + c;
            slm_A[0][i] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0f;
        }

        // Load B into VNNI: (TK/2) * TN = 16*64 = 1024 uints, 256 threads => 4 each
        for (int i = flat_lid; i < (TK/2) * TN; i += 256) {
            int kp = i / TN;   // k-pair index 0..15
            int c  = i % TN;   // column index 0..63
            int gk0 = k0 + kp * 2;
            int gk1 = gk0 + 1;
            int gc  = wg_n + c;
            ushort v0 = (gk0 < K && gc < N) ? B_us[gk0 * N + gc] : (ushort)0;
            ushort v1 = (gk1 < K && gc < N) ? B_us[gk1 * N + gc] : (ushort)0;
            slm_B_vnni[0][i] = (uint)v0 | ((uint)v1 << 16);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // === Main K-loop ===
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int next_kt = kt + 1;
        int next_buf = buf ^ 1;

        // --- Prefetch next K-tile into next_buf ---
        if (next_kt < num_k_tiles) {
            int k0_next = next_kt * TK;

            for (int i = flat_lid; i < TM * TK; i += 256) {
                int r = i / TK;
                int c = i % TK;
                int gr = wg_m + r;
                int gc = k0_next + c;
                slm_A[next_buf][i] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0f;
            }

            for (int i = flat_lid; i < (TK/2) * TN; i += 256) {
                int kp = i / TN;
                int c  = i % TN;
                int gk0 = k0_next + kp * 2;
                int gk1 = gk0 + 1;
                int gc  = wg_n + c;
                ushort v0 = (gk0 < K && gc < N) ? B_us[gk0 * N + gc] : (ushort)0;
                ushort v1 = (gk1 < K && gc < N) ? B_us[gk1 * N + gc] : (ushort)0;
                slm_B_vnni[next_buf][i] = (uint)v0 | ((uint)v1 << 16);
            }
        }

        // --- Compute on current buffer: two DPAS k16 steps ---
        int a_row_in_tile = sg_row * 8 + sg_lid;  // which row of A tile this WI uses
        int b_col_in_tile = sg_col * 8 + sg_lid;  // which col of B tile this WI uses

        // Step 1: k offset 0..15
        {
            // Pack A: row a_row_in_tile, cols 0..15 from slm_A
            __local half* a_ptr = &slm_A[buf][a_row_in_tile * TK];  // points to start of row
            int8 a_packed;
            // Each int packs 2 consecutive halves
            a_packed.s0 = as_int(vload2(0, (__local ushort*)(a_ptr + 0)));
            a_packed.s1 = as_int(vload2(0, (__local ushort*)(a_ptr + 2)));
            a_packed.s2 = as_int(vload2(0, (__local ushort*)(a_ptr + 4)));
            a_packed.s3 = as_int(vload2(0, (__local ushort*)(a_ptr + 6)));
            a_packed.s4 = as_int(vload2(0, (__local ushort*)(a_ptr + 8)));
            a_packed.s5 = as_int(vload2(0, (__local ushort*)(a_ptr + 10)));
            a_packed.s6 = as_int(vload2(0, (__local ushort*)(a_ptr + 12)));
            a_packed.s7 = as_int(vload2(0, (__local ushort*)(a_ptr + 14)));

            // Pack B VNNI: k-pairs 0..7, column b_col_in_tile
            int8 b_packed;
            b_packed.s0 = (int)slm_B_vnni[buf][0 * TN + b_col_in_tile];
            b_packed.s1 = (int)slm_B_vnni[buf][1 * TN + b_col_in_tile];
            b_packed.s2 = (int)slm_B_vnni[buf][2 * TN + b_col_in_tile];
            b_packed.s3 = (int)slm_B_vnni[buf][3 * TN + b_col_in_tile];
            b_packed.s4 = (int)slm_B_vnni[buf][4 * TN + b_col_in_tile];
            b_packed.s5 = (int)slm_B_vnni[buf][5 * TN + b_col_in_tile];
            b_packed.s6 = (int)slm_B_vnni[buf][6 * TN + b_col_in_tile];
            b_packed.s7 = (int)slm_B_vnni[buf][7 * TN + b_col_in_tile];

            acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);
        }

        // Step 2: k offset 16..31
        {
            __local half* a_ptr = &slm_A[buf][a_row_in_tile * TK + 16];
            int8 a_packed;
            a_packed.s0 = as_int(vload2(0, (__local ushort*)(a_ptr + 0)));
            a_packed.s1 = as_int(vload2(0, (__local ushort*)(a_ptr + 2)));
            a_packed.s2 = as_int(vload2(0, (__local ushort*)(a_ptr + 4)));
            a_packed.s3 = as_int(vload2(0, (__local ushort*)(a_ptr + 6)));
            a_packed.s4 = as_int(vload2(0, (__local ushort*)(a_ptr + 8)));
            a_packed.s5 = as_int(vload2(0, (__local ushort*)(a_ptr + 10)));
            a_packed.s6 = as_int(vload2(0, (__local ushort*)(a_ptr + 12)));
            a_packed.s7 = as_int(vload2(0, (__local ushort*)(a_ptr + 14)));

            // B VNNI k-pairs 8..15
            int8 b_packed;
            b_packed.s0 = (int)slm_B_vnni[buf][8  * TN + b_col_in_tile];
            b_packed.s1 = (int)slm_B_vnni[buf][9  * TN + b_col_in_tile];
            b_packed.s2 = (int)slm_B_vnni[buf][10 * TN + b_col_in_tile];
            b_packed.s3 = (int)slm_B_vnni[buf][11 * TN + b_col_in_tile];
            b_packed.s4 = (int)slm_B_vnni[buf][12 * TN + b_col_in_tile];
            b_packed.s5 = (int)slm_B_vnni[buf][13 * TN + b_col_in_tile];
            b_packed.s6 = (int)slm_B_vnni[buf][14 * TN + b_col_in_tile];
            b_packed.s7 = (int)slm_B_vnni[buf][15 * TN + b_col_in_tile];

            acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // === Store 8x8 output tile ===
    int out_col = wg_n + sg_col * 8 + sg_lid;
    if (out_col < N) {
        for (int i = 0; i < 8; i++) {
            int out_row = wg_m + sg_row * 8 + i;
            if (out_row < M) {
                C[out_row * N + out_col] = convert_half(((float*)&acc)[i]);
            }
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.  

(II) Bottlenecks and suggestions for improvement:  
1. `barrier(CLK_LOCAL_MEM_FENCE);` (right before `buf = next_buf;` in every K-iteration): this barrier kills overlap between prefetch and compute, so your “double buffering” is mostly serialized. Right now every work-item finishes compute, waits, and only then advances, which leaves latency-hiding on the table.  
   **Improve it:** switch to true pipelining with `async_work_group_copy` (or Xe equivalent async block copy) for A/B tile loads into `next_buf`, keep compute running on `buf`, and wait only when the data is actually needed (`wait_group_events` right before using `next_buf`). Keep one barrier for SLM visibility at tile handoff, not a full stop around the whole loop body.

2. 
```c
a_packed.s0 = as_int(vload2(...));
...
a_packed.s7 = as_int(vload2(...));
```
(and same pattern for `b_packed.s0...s7` in both DPAS steps): this is a lot of scalar packing instructions per iteration, repeated twice per tile, and it inflates register pressure + instruction count. You’re spending cycles repacking data instead of feeding DPAS.  
   **Improve it:** store SLM in a DPAS-ready layout so each subgroup/lane can load packed vectors directly (e.g., `intel_sub_group_block_read`/vector block loads of prepacked `int8` chunks). Prepack A and B during the cooperative load stage once, then reuse. This removes most per-iteration `vload2/as_int` scalarization and reduces backend pressure.

3. 
```c
C[out_row * N + out_col] = convert_half(((float*)&acc)[i]);
```
(looped 8 times): scalar stores + pointer-cast lane extraction are inefficient and can generate poor codegen for the accumulator writeback path.  
   **Improve it:** use subgroup/block writeout in vector form (e.g., convert `acc` to `half8` once, then `vstore8`/subgroup block write) with contiguous per-lane mapping. If needed, remap output ownership so each lane writes contiguous elements to enable coalesced stores. This cuts store instructions and improves global-memory throughput on the epilogue.

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
1. Kernel Fusion: Combine multiple small kernels into one to eliminate intermediate global memory writes and reduce launch overhead. Use barrier(CLK_LOCAL_MEM_FENCE) between logical kernel phases within a work-group.
2. Use Appropriate Data Types: Use half (FP16) when precision allows - doubles throughput on many GPUs. Use native_* math functions (native_sqrt, native_divide) for faster, less precise operations. Use mad() for fused multiply-add. Size integer types appropriately (char, short, int).

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

- **Shared Local Memory (SLM) Tiling**: Cache frequently accessed data in SLM using `group_local_memory_for_overwrite`. Synchronize with `group_barrier()` after writes and before reads. Use 16×16 or 32×32 tiles for float data.
- **Kernel Fusion**: Combine sequential operations (e.g., exp → add → activation) into a single kernel. Eliminate intermediate buffers by computing in registers.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.