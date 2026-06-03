

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
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 33.900):
```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Work-group tile: TM=32, TN=64, TK=16
// Subgroup grid: 4x4 = 16 subgroups, each 8 WIs => LWS = 128
// Each subgroup computes 8x16 output (2 DPAS k16 along N)
//
// SLM (double-buffered):
//   A: [2][32][8] uint (32 rows x 16 halfs packed as 8 uints per row) - contiguous for vector load
//   B: [2][8][64] uint (8 k-pairs x 64 cols in VNNI) 
// Total: 2*(32*8*4 + 8*64*4) = 2*(1024+2048) = 6144 bytes
//
// LWS = (32, 4, 1)  [32 WIs x 4 = 128 WIs total]
// GWS = (ceil(N/64)*32, ceil(M/32)*4, 1)
// Subgroup size = 8

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
    const int TM = 32;
    const int TN = 64;
    const int TK = 16;

    const int wg_n = get_group_id(0) * TN;
    const int wg_m = get_group_id(1) * TM;

    const int lid0 = get_local_id(0);  // 0..31
    const int lid1 = get_local_id(1);  // 0..3
    const int flat_lid = lid1 * 32 + lid0;  // 0..127

    const int sg_lid = get_sub_group_local_id(); // 0..7
    const int sg_id = get_sub_group_id();        // 0..15

    // 4x4 subgroup grid
    const int sg_y = sg_id / 4;  // 0..3 (along M)
    const int sg_x = sg_id % 4;  // 0..3 (along N, each covers 16 cols)

    // A stored as uint (pairs of halfs): [32][8] per buffer - contiguous rows
    __local uint slm_A[2][32 * 8];      // 2 * 1024 bytes
    // B stored in VNNI: [8][64] per buffer
    __local uint slm_B[2][8 * 64];      // 2 * 2048 bytes

    // Two accumulators: each subgroup computes 8x8 twice (8x16 total)
    float8 acc0 = (float8)(0.0f);  // columns [sg_x*16 + 0..7]
    float8 acc1 = (float8)(0.0f);  // columns [sg_x*16 + 8..15]

    int num_k_tiles = (K + TK - 1) / TK;

    // Preload first tile into buffer 0
    {
        // Load A: 32*8 = 256 uints, 128 WIs => 2 each
        for (int i = flat_lid; i < 32 * 8; i += 128) {
            int r = i >> 3;       // row 0..31
            int p = i & 7;        // pair index 0..7
            int gr = wg_m + r;
            int gc0 = p * 2;
            int gc1 = gc0 + 1;
            ushort v0 = (gr < M && gc0 < K) ? as_ushort(A[gr * K + gc0]) : (ushort)0;
            ushort v1 = (gr < M && gc1 < K) ? as_ushort(A[gr * K + gc1]) : (ushort)0;
            slm_A[0][r * 8 + p] = (uint)v0 | ((uint)v1 << 16);
        }
        // Load B: 8*64 = 512 uints, 128 WIs => 4 each
        for (int i = flat_lid; i < 8 * 64; i += 128) {
            int kp = i >> 6;      // k-pair 0..7
            int c = i & 63;       // col 0..63
            int gk0 = kp * 2;
            int gk1 = gk0 + 1;
            int gc = wg_n + c;
            ushort b0 = (gk0 < K && gc < N) ? as_ushort(B[gk0 * N + gc]) : (ushort)0;
            ushort b1 = (gk1 < K && gc < N) ? as_ushort(B[gk1 * N + gc]) : (ushort)0;
            slm_B[0][kp * 64 + c] = (uint)b0 | ((uint)b1 << 16);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int next_buf = buf ^ 1;

        // Prefetch next tile
        if (kt + 1 < num_k_tiles) {
            int k_next = (kt + 1) * TK;
            for (int i = flat_lid; i < 32 * 8; i += 128) {
                int r = i >> 3;
                int p = i & 7;
                int gr = wg_m + r;
                int gc0 = k_next + p * 2;
                int gc1 = gc0 + 1;
                ushort v0 = (gr < M && gc0 < K) ? as_ushort(A[gr * K + gc0]) : (ushort)0;
                ushort v1 = (gr < M && gc1 < K) ? as_ushort(A[gr * K + gc1]) : (ushort)0;
                slm_A[next_buf][r * 8 + p] = (uint)v0 | ((uint)v1 << 16);
            }
            for (int i = flat_lid; i < 8 * 64; i += 128) {
                int kp = i >> 6;
                int c = i & 63;
                int gk0 = k_next + kp * 2;
                int gk1 = gk0 + 1;
                int gc = wg_n + c;
                ushort b0 = (gk0 < K && gc < N) ? as_ushort(B[gk0 * N + gc]) : (ushort)0;
                ushort b1 = (gk1 < K && gc < N) ? as_ushort(B[gk1 * N + gc]) : (ushort)0;
                slm_B[next_buf][kp * 64 + c] = (uint)b0 | ((uint)b1 << 16);
            }
        }

        // Compute: load A once, use for two DPAS (two sets of B columns)
        int a_row_local = sg_y * 8 + sg_lid;
        __local uint* a_base_ptr = &slm_A[buf][a_row_local * 8];

        // Vector load A: 8 contiguous uints
        int8 a_packed = *(__local int8*)a_base_ptr;

        // B for first 8 cols of this subgroup's 16-col tile
        int b_col0 = sg_x * 16 + sg_lid;        // cols 0..7
        int b_col1 = sg_x * 16 + 8 + sg_lid;    // cols 8..15

        int8 b_packed0 = (int8)(
            (int)slm_B[buf][0 * 64 + b_col0],
            (int)slm_B[buf][1 * 64 + b_col0],
            (int)slm_B[buf][2 * 64 + b_col0],
            (int)slm_B[buf][3 * 64 + b_col0],
            (int)slm_B[buf][4 * 64 + b_col0],
            (int)slm_B[buf][5 * 64 + b_col0],
            (int)slm_B[buf][6 * 64 + b_col0],
            (int)slm_B[buf][7 * 64 + b_col0]
        );

        int8 b_packed1 = (int8)(
            (int)slm_B[buf][0 * 64 + b_col1],
            (int)slm_B[buf][1 * 64 + b_col1],
            (int)slm_B[buf][2 * 64 + b_col1],
            (int)slm_B[buf][3 * 64 + b_col1],
            (int)slm_B[buf][4 * 64 + b_col1],
            (int)slm_B[buf][5 * 64 + b_col1],
            (int)slm_B[buf][6 * 64 + b_col1],
            (int)slm_B[buf][7 * 64 + b_col1]
        );

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed1, acc1);

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Store 8x16 output: two 8x8 blocks
    int out_row_base = wg_m + sg_y * 8;
    int out_col0 = wg_n + sg_x * 16 + sg_lid;
    int out_col1 = out_col0 + 8;

    for (int i = 0; i < 8; i++) {
        int out_row = out_row_base + i;
        if (out_row < M) {
            if (out_col0 < N)
                C[out_row * N + out_col0] = convert_half(((float*)&acc0)[i]);
            if (out_col1 < N)
                C[out_row * N + out_col1] = convert_half(((float*)&acc1)[i]);
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel is correct, but could be made more efficient.  
(II) Bottlenecks and suggestions for improvement:  
1. `for (int i = flat_lid; i < 32 * 8; i += 128) { ... A[gr * K + gc0] ... A[gr * K + gc1] ... }` and `for (int i = flat_lid; i < 8 * 64; i += 128) { ... B[gk0 * N + gc] ... B[gk1 * N + gc] ... }`: you’re doing scalar global loads with per-element bounds checks for every preload/prefetch, which is expensive and limits memory throughput.  
   **Improve it** by splitting into fast path + tail path:
   - Fast path when the tile is fully in-bounds (`wg_m+31 < M`, `wg_n+63 < N`, `k_next+15 < K`): use vectorized loads/stores (`vload_half2/4/8` then pack) and no conditionals.  
   - Tail path only for edge tiles with current guarded scalar loads.  
   Also, map each WI to contiguous segments (e.g., each WI loads 16B/32B contiguous) to improve coalescing and reduce address arithmetic.

2. `barrier(CLK_LOCAL_MEM_FENCE);` (inside every `kt` iteration, plus preload barrier): the double-buffering exists, but prefetch and compute are still serialized at workgroup scope, so you’re not actually overlapping memory and DPAS much.  
   **Improve it** by converting to true async pipeline:
   - Use `async_work_group_copy` (or Intel block read intrinsics if available) into `slm_A[next_buf]` / `slm_B[next_buf]`.
   - Compute on `buf` while async copies for `next_buf` are in flight.
   - Replace unconditional barrier with `wait_group_events` + minimal fence only when switching buffers.  
   This reduces stall time from full workgroup synchronization every K-tile.

3. `int8 b_packed0 = (int8)( slm_B[buf][0 * 64 + b_col0], ... slm_B[buf][7 * 64 + b_col0] );` and similarly for `b_packed1`: this is 16 separate SLM scalar reads per subgroup per K-tile just to assemble DPAS operands, creating high SLM traffic and instruction overhead.  
   **Improve it** by changing B’s SLM layout to make DPAS operand fetch contiguous for each lane:
   - Store B in SLM as lane-major blocks for each `sg_x` chunk (e.g., `[sg_x][lane][kpair]`), so each lane can do one vector load (`int8`) directly.
   - Then `b_packed0/1` become simple vector loads instead of manual gather construction.
   This typically gives a meaningful speedup because DPAS becomes compute-bound instead of SLM-gather-bound.

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
1. Minimize Host-Device Transfers: Use CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY flags appropriately. Chain kernels using clEnqueueNDRangeKernel with event dependencies instead of clFinish(). Use persistent device allocations across multiple kernel launches.
2. Leverage Vectorization: Use vector types (float4, float8, float16) for explicit SIMD. Use vloadN() and vstoreN() for unaligned vector memory access. For aligned access, cast pointers: ((__global float4*)ptr)[idx]. Match native vector width of target device.

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