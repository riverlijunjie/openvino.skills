

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

### Version 1 (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Optimized FP16 GEMM using Intel DPAS (intel_sub_group_f16_f16_matrix_mad_k16)
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Tile: 64 rows x 32 cols per work-group
// Subgroup size: 16
// WG layout: 2 subgroups in N (covers 32 cols), 2 rows of subgroups (covers 64 rows)
// Each subgroup computes 32x16 (THREAD_M_TILES=4 DPAS ops, each 8x16)
// K-tile: 16 (matches DPAS k=16)
//
// LWS = (32, 2, 1)  => 64 work-items = 4 subgroups
// GWS = (ceil(N/32)*32, ceil(M/64)*2, 1)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define WG_TILE_M 64
#define WG_TILE_N 32
#define SG_TILE_M 32
#define SG_TILE_N 16
#define DPAS_M 8
#define DPAS_K 16
#define DPAS_N 16
#define THREAD_M_TILES 4
#define TILE_K 16

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Identify subgroup position within workgroup
    const int sg_id = get_sub_group_id();         // 0..3
    const int sg_lane = get_sub_group_local_id(); // 0..15

    // Subgroup grid within WG: 2 wide (N) x 2 tall (M)
    const int sg_col = sg_id % 2;  // 0 or 1
    const int sg_row = sg_id / 2;  // 0 or 1

    // Work-group tile origin
    const int wg_m = (get_group_id(1) * 2 + 0) * SG_TILE_M;  // get_group_id(1) indexes WG rows
    // Actually let me recompute based on GWS mapping
    // get_global_id(0) spans N, get_global_id(1) spans M/32
    const int wg_origin_n = (get_group_id(0) / 2) * WG_TILE_N;  // groups of 2 subgroups in x
    const int wg_origin_m = get_group_id(1) * WG_TILE_M;

    // This subgroup's output tile origin
    const int tile_m = wg_origin_m + sg_row * SG_TILE_M;
    const int tile_n = wg_origin_n + sg_col * SG_TILE_N;

    // Accumulator registers: THREAD_M_TILES dpas tiles, each float8
    float8 acc[THREAD_M_TILES];
    #pragma unroll
    for (int t = 0; t < THREAD_M_TILES; t++) {
        acc[t] = (float8)(0.0f);
    }

    // SLM for A and B tiles
    // A tile: WG_TILE_M x TILE_K = 64 x 16 halfs
    // B tile: TILE_K x WG_TILE_N = 16 x 32 halfs
    __local half A_slm[WG_TILE_M * TILE_K];
    __local half B_slm[TILE_K * WG_TILE_N];

    const int local_id = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int local_size = get_local_size(0) * get_local_size(1); // 64

    // Loop over K dimension
    for (int k0 = 0; k0 < K; k0 += TILE_K) {

        // Cooperative load A[wg_origin_m .. +64, k0 .. +16] into SLM
        // 64*16 = 1024 halfs, 64 threads => 16 halfs each
        #pragma unroll
        for (int i = local_id; i < WG_TILE_M * TILE_K; i += local_size) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gm = wg_origin_m + row;
            int gk = k0 + col;
            A_slm[i] = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0h;
        }

        // Cooperative load B[k0 .. +16, wg_origin_n .. +32] into SLM
        // 16*32 = 512 halfs, 64 threads => 8 halfs each
        #pragma unroll
        for (int i = local_id; i < TILE_K * WG_TILE_N; i += local_size) {
            int row = i / WG_TILE_N;
            int col = i % WG_TILE_N;
            int gk = k0 + row;
            int gn = wg_origin_n + col;
            B_slm[i] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each subgroup reads its portion and does DPAS
        // B for this subgroup: 16 rows x 16 cols starting at col offset sg_col*16
        // Read B into registers: DPAS expects B distributed across subgroup
        // B layout for DPAS: int8 where each int packs 2 half values
        // B is K=16 rows x N=16 cols, each lane holds one column, 16 rows = 8 ints (2 halfs per int)
        int8 b_reg;
        __local const uint* B_slm_uint = (__local const uint*)(B_slm);
        // B_slm is [TILE_K][WG_TILE_N] in half
        // For sg_col subgroup, cols [sg_col*16 .. sg_col*16+15]
        // Lane sg_lane reads column (sg_col*16 + sg_lane)
        // Row r: B_slm[r * WG_TILE_N + sg_col*16 + sg_lane]
        // Pack pairs of rows into uint
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int r0 = 2 * i;
            int r1 = 2 * i + 1;
            ushort v0 = as_ushort(B_slm[r0 * WG_TILE_N + sg_col * DPAS_N + sg_lane]);
            ushort v1 = as_ushort(B_slm[r1 * WG_TILE_N + sg_col * DPAS_N + sg_lane]);
            ((uint*)&b_reg)[i] = ((uint)v1 << 16) | (uint)v0;
        }

        // For each DPAS M-tile
        #pragma unroll
        for (int t = 0; t < THREAD_M_TILES; t++) {
            // A for this dpas: 8 rows x 16 cols
            // Rows: sg_row * SG_TILE_M + t * DPAS_M .. + 8
            // A_slm layout: [WG_TILE_M][TILE_K]
            int a_row_base = sg_row * SG_TILE_M + t * DPAS_M;
            int8 a_reg;
            // DPAS A: each work-item holds data for its row(s)
            // For A: int8 where each int packs 2 halfs along K
            // Each of 8 rows distributed across the 8 components
            // Within each row, 16 K values = 8 uint (pairs of half)
            // But these are shared across subgroup via broadcast
            // Actually for DPAS: A is replicated, each lane has same A data
            // A[8 rows][16 K] packed as int8, each int = 2 halfs along K
            // Lane sg_lane reads K columns [sg_lane*1...] — no, A is broadcast

            // For intel_sub_group_f16_f16_matrix_mad_k16:
            // A: int8 — 8 rows, each row has K=16 halfs packed into 8 ints
            //    But spread across subgroup: each lane holds one pair of K values per row
            //    Lane l holds K indices [2l, 2l+1] for all 8 rows
            // So a_reg[row] = pack(A[row, 2*sg_lane], A[row, 2*sg_lane+1])

            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int slm_row = a_row_base + r;
                ushort v0 = as_ushort(A_slm[slm_row * TILE_K + 2 * sg_lane]);
                ushort v1 = as_ushort(A_slm[slm_row * TILE_K + 2 * sg_lane + 1]);
                ((uint*)&a_reg)[r] = ((uint)v1 << 16) | (uint)v0;
            }

            acc[t] = intel_sub_group_f16_f16_matrix_mad_k16(a_reg, b_reg, acc[t]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    // Each work-item in subgroup holds 8 values per DPAS tile for its column (tile_n + sg_lane)
    int out_col = tile_n + sg_lane;

    #pragma unroll
    for (int t = 0; t < THREAD_M_TILES; t++) {
        int row_base = tile_m + t * DPAS_M;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int out_row = row_base + r;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = convert_half(((float*)&acc[t])[r]);
            }
        }
    }
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Optimized FP16 GEMM using Intel DPAS (XMX) on Battlemage Xe2-HPG
// C[M,N] = A[M,K] x B[K,N], all half, accumulation in float
//
// Each subgroup (16 work-items) computes an 8x16 tile of C using
// intel_sub_group_f16_f16_matrix_mad_k16.
//
// Launch configuration:
//   LWS = (16, SG_PER_WG)  e.g. (16, 4) → 64 WIs, 4 subgroups
//   GWS = (ceil(N/16)*16, ceil(M/(8*SG_PER_WG))*SG_PER_WG)
//   Subgroup size: 16
//
// Each work-group computes (8*SG_PER_WG) x 16 of C.
// With SG_PER_WG=4: 32 x 16 per work-group.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int sg_lid = get_sub_group_local_id();  // 0..15
    const int sg_id  = get_sub_group_id();        // subgroup within work-group

    // Tile position in output C
    const int tile_col = get_group_id(0) * 16;                          // N dimension
    const int tile_row = get_group_id(1) * (int)get_local_size(1) * 8   // M dimension
                       + sg_id * 8;

    // Early exit for out-of-bounds work-groups
    if (tile_row >= M || tile_col >= N) return;

    const int out_col = tile_col + sg_lid;

    // Accumulator: 8 rows, each work-item holds one column
    float8 acc = (float8)(0.0f);

    // K-loop in steps of 16 (DPAS k-dimension)
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        // ---- Load A tile [8 x 16] ----
        // DPAS A layout: int8 per work-item
        // work-item sg_lid holds, for each of 8 rows:
        //   { A[row][k + 2*sg_lid], A[row][k + 2*sg_lid+1] } packed as int
        int8 a_val;
        int a_k = k + 2 * sg_lid;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row = tile_row + r;
            // Use vload2 for aligned pair load
            ushort u0 = row < M ? as_ushort(A[row * K + a_k])     : (ushort)0;
            ushort u1 = row < M ? as_ushort(A[row * K + a_k + 1]) : (ushort)0;
            int packed = (int)u0 | ((int)u1 << 16);
            // Can't index int8 dynamically, unroll manually
            switch(r) {
                case 0: a_val.s0 = packed; break;
                case 1: a_val.s1 = packed; break;
                case 2: a_val.s2 = packed; break;
                case 3: a_val.s3 = packed; break;
                case 4: a_val.s4 = packed; break;
                case 5: a_val.s5 = packed; break;
                case 6: a_val.s6 = packed; break;
                case 7: a_val.s7 = packed; break;
            }
        }

        // ---- Load B tile [16 x 16] ----
        // DPAS B layout: int8 per work-item
        // work-item sg_lid holds column (tile_col + sg_lid) for pairs of k-rows
        // b[j] = { B[k+2*j][col], B[k+2*j+1][col] } packed as int
        int8 b_val;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int kr0 = k + 2 * j;
            int kr1 = kr0 + 1;
            ushort bv0 = (out_col < N) ? as_ushort(B[kr0 * N + out_col]) : (ushort)0;
            ushort bv1 = (out_col < N) ? as_ushort(B[kr1 * N + out_col]) : (ushort)0;
            int packed = (int)bv0 | ((int)bv1 << 16);
            switch(j) {
                case 0: b_val.s0 = packed; break;
                case 1: b_val.s1 = packed; break;
                case 2: b_val.s2 = packed; break;
                case 3: b_val.s3 = packed; break;
                case 4: b_val.s4 = packed; break;
                case 5: b_val.s5 = packed; break;
                case 6: b_val.s6 = packed; break;
                case 7: b_val.s7 = packed; break;
            }
        }

        // ---- DPAS: 8x16 matmul accumulate ----
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    }

    // Handle remaining K elements (K not multiple of 16) with scalar fallback
    if (k < K) {
        for (int kk = k; kk < K; kk++) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int row = tile_row + r;
                if (row < M && out_col < N) {
                    float a_elem = convert_float(A[row * K + kk]);
                    float b_elem = convert_float(B[kk * N + out_col]);
                    switch(r) {
                        case 0: acc.s0 += a_elem * b_elem; break;
                        case 1: acc.s1 += a_elem * b_elem; break;
                        case 2: acc.s2 += a_elem * b_elem; break;
                        case 3: acc.s3 += a_elem * b_elem; break;
                        case 4: acc.s4 += a_elem * b_elem; break;
                        case 5: acc.s5 += a_elem * b_elem; break;
                        case 6: acc.s6 += a_elem * b_elem; break;
                        case 7: acc.s7 += a_elem * b_elem; break;
                    }
                }
            }
        }
    }

    // ---- Store 8x1 column of results ----
    if (out_col < N) {
        if (tile_row + 0 < M) C[(tile_row + 0) * N + out_col] = convert_half(acc.s0);
        if (tile_row + 1 < M) C[(tile_row + 1) * N + out_col] = convert_half(acc.s1);
        if (tile_row + 2 < M) C[(tile_row + 2) * N + out_col] = convert_half(acc.s2);
        if (tile_row + 3 < M) C[(tile_row + 3) * N + out_col] = convert_half(acc.s3);
        if (tile_row + 4 < M) C[(tile_row + 4) * N + out_col] = convert_half(acc.s4);
        if (tile_row + 5 < M) C[(tile_row + 5) * N + out_col] = convert_half(acc.s5);
        if (tile_row + 6 < M) C[(tile_row + 6) * N + out_col] = convert_half(acc.s6);
        if (tile_row + 7 < M) C[(tile_row + 7) * N + out_col] = convert_half(acc.s7);
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Optimized FP16 matmul using Intel DPAS (XMX) instructions
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Tiling: WG tile = 32x64 (MxN), DPAS tile = 8x16, K-tile = 16
// Each WG has 16 subgroups (4 along M, 4 along N), subgroup size = 16
// LWS = (64, 4) = 256 work-items
// GWS = (ceil(N/64)*64, ceil(M/32)*4)
// SLM usage: A_tile[32][16] + B_tile[16][64] = 1024 + 2048 = 3072 bytes (half)

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 4, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile origin in C
    const int wg_n = (get_group_id(0)) * 64;  // N offset
    const int wg_m = (get_group_id(1)) * 32;  // M offset

    // Local IDs
    const int lid_x = get_local_id(0); // 0..63
    const int lid_y = get_local_id(1); // 0..3
    const int lid = lid_y * 64 + lid_x; // flat local id 0..255

    // Subgroup identification
    const int sg_id = get_sub_group_id();        // 0..15
    const int sg_lid = get_sub_group_local_id();  // 0..15

    // Map 16 subgroups into 4x4 grid (4 along M, 4 along N)
    const int sg_m = sg_id / 4;  // 0..3, each covers 8 rows
    const int sg_n = sg_id % 4;  // 0..3, each covers 16 cols

    // Subgroup's tile origin within WG tile
    const int sg_row_offset = sg_m * 8;   // 0, 8, 16, 24
    const int sg_col_offset = sg_n * 16;  // 0, 16, 32, 48

    // SLM tiles
    __local half A_slm[32 * 16];  // 32 rows x 16 cols
    __local half B_slm[16 * 64];  // 16 rows x 64 cols

    // Accumulator: 8 floats per work-item (8 rows of the 8x16 DPAS output)
    float8 acc = (float8)(0.0f);

    // Loop over K dimension in tiles of 16
    for (int k_base = 0; k_base < K; k_base += 16) {
        // Cooperative load of A tile [32][16] into SLM
        // 256 work-items, 32*16 = 512 elements, 2 elements per work-item
        {
            int idx0 = lid * 2;
            int idx1 = lid * 2 + 1;

            int a_row0 = idx0 / 16;
            int a_col0 = idx0 % 16;
            int a_row1 = idx1 / 16;
            int a_col1 = idx1 % 16;

            int gm0 = wg_m + a_row0;
            int gk0 = k_base + a_col0;
            int gm1 = wg_m + a_row1;
            int gk1 = k_base + a_col1;

            A_slm[idx0] = (gm0 < M && gk0 < K) ? A[gm0 * K + gk0] : (half)0.0h;
            A_slm[idx1] = (gm1 < M && gk1 < K) ? A[gm1 * K + gk1] : (half)0.0h;
        }

        // Cooperative load of B tile [16][64] into SLM
        // 256 work-items, 16*64 = 1024 elements, 4 elements per work-item
        {
            for (int i = 0; i < 4; i++) {
                int idx = lid + i * 256;
                int b_row = idx / 64;
                int b_col = idx % 64;

                int gk = k_base + b_row;
                int gn = wg_n + b_col;

                B_slm[idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each subgroup performs DPAS: 8x16 = 8x16_k16 matmul
        // Load A sub-tile: 8 rows, 16 cols from A_slm
        // For DPAS a input: each work-item i holds A[sg_row_offset + row][col] 
        // packed as 8 halfs (8 rows, work-item's K-column = sg_lid-th pair)

        // a: each work-item holds 8 half values = rows 0..7, at k-columns
        // In intel_sub_group_f16_f16_matrix_mad_k16:
        //   a is int8 (8 x int, each int packs 2 half values) representing 8x16 matrix
        //   b is int8 representing 16x16 matrix
        //   Result is float8 (8 rows x 1 col per work-item, 16 work-items = 8x16)

        // Pack A: 8 rows x 16 cols, stored as int8 per work-item
        // Each work-item sg_lid reads column pair (sg_lid*2, sg_lid*2+1) -- but that's not right
        // Actually for DPAS, data layout:
        //   a[i] for work-item j: a is distributed across subgroup
        //   Each work-item holds 8 ints, each int = 2 halfs from K dimension
        //   work-item j holds k-indices [2j, 2j+1] for all 8 rows

        int8 a_packed;
        for (int r = 0; r < 8; r++) {
            int slm_row = sg_row_offset + r;
            int k0 = sg_lid * 2;
            int k1 = sg_lid * 2 + 1;
            half va0 = A_slm[slm_row * 16 + k0];
            half va1 = A_slm[slm_row * 16 + k1];
            // Pack two halfs into one int
            ushort u0 = as_ushort(va0);
            ushort u1 = as_ushort(va1);
            ((int*)&a_packed)[r] = (int)u0 | ((int)u1 << 16);
        }

        // Pack B: 16 rows x 16 cols (the subgroup's 16-col slice)
        // Each work-item j holds k-indices [2j, 2j+1] for all 16 output columns
        // But B is organized differently: 
        //   b[i] for work-item j: work-item j holds column j, rows packed in pairs
        //   Each int = 2 halfs from K dimension at column sg_col_offset + sg_lid

        int8 b_packed;
        for (int p = 0; p < 8; p++) {
            int k0 = p * 2;
            int k1 = p * 2 + 1;
            int col = sg_col_offset + sg_lid;
            half vb0 = B_slm[k0 * 64 + col];
            half vb1 = B_slm[k1 * 64 + col];
            ushort ub0 = as_ushort(vb0);
            ushort ub1 = as_ushort(vb1);
            ((int*)&b_packed)[p] = (int)ub0 | ((int)ub1 << 16);
        }

        // DPAS: 8x16 result = 8x16(A) * 16x16(B)
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: each work-item writes 8 values
    // Work-item sg_lid in subgroup writes column (sg_col_offset + sg_lid)
    // rows sg_row_offset + 0..7
    int out_col = wg_n + sg_col_offset + sg_lid;

    for (int r = 0; r < 8; r++) {
        int out_row = wg_m + sg_row_offset + r;
        if (out_row < M && out_col < N) {
            float val = ((float*)&acc)[r];
            C[out_row * N + out_col] = convert_half(val);
        }
    }
}
```

Feedback on this kernel:
(I) General feedback: Your kernel **is incorrect**.

(II) Summary of console output: All 4 correctness tests failed (`wrt_pytorch` and `wrt_reference`, two seeds each). The output matrix is essentially all zeros for a 2048x2048 case, while expected/reference contain large nonzero values. This strongly indicates your compute path is not producing valid accumulations (DPAS operand layout/type mismatch), rather than a small numerical tolerance issue.

(III) Errors and suggestions for improvement:
1. `acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);`: You’re calling the DPAS intrinsic with operands packed manually into `int8` via pointer-casts, but the lane/register layout expected by this intrinsic is very strict and your mapping is almost certainly wrong. Replace the manual pack with the exact vendor-recommended data types/layout for Battlemage DPAS (typically `sub_group` block load + required vector/int type form), or switch to a known-correct subgroup MMA builtin variant for your compiler version. Right now, mathematically you intend 8x16x16, but physically you’re feeding an incompatible register arrangement, which explains zero outputs.

2. `((int*)&a_packed)[r] = ...` and `((int*)&b_packed)[p] = ...`: This is undefined/fragile in OpenCL C (strict aliasing + vector element access by casted pointer). On Intel GPUs this can compile into unexpected code and silently break. Write vector elements directly (e.g., `a_packed.s0 ... s7`) or use `as_int2/as_int4`-style safe packing primitives per spec. Also ensure half-pair packing order (low/high 16 bits) matches the intrinsic contract; a swapped order can fully corrupt results.

3. `const int sg_id = get_sub_group_id();` with `__attribute__((reqd_work_group_size(64, 4, 1)))`: Your subgroup-to-tile mapping assumes exactly 16 contiguous subgroups arranged as 4x4 over (M,N), but subgroup enumeration/order across 2D local IDs is not guaranteed to match your implicit mapping. Derive `sg_m/sg_n` from `lid_x/lid_y` deterministically (or flatten local ID then divide by subgroup size) so each subgroup writes a unique 8x16 tile you actually loaded. A wrong subgroup mapping can make all lanes read wrong B columns and write incoherent/zeroed regions.

4. `float8 acc` + final store `float val = ((float*)&acc)[r];`: Same pointer-cast issue for vector element extraction. Use `acc.s0 ... acc.s7` (or a `switch`) to avoid undefined behavior and register spilling/incorrect loads. This is a correctness issue first, performance second.

5. `int k0 = sg_lid * 2; int k1 = sg_lid * 2 + 1;` for A packing: This hardcodes lane `j` to K pair `(2j,2j+1)` for all rows, which may not match DPAS operand A lane distribution required by `*_k16` form. Verify and remap lanes exactly per intrinsic docs (often different lane-to-K/chunk mapping). If lane mapping is off, dot-products collapse to garbage/zeros even though SLM tiles are correct.

6. `int col = sg_col_offset + sg_lid;` for B packing: You bind each lane to one output column and pack K pairs across `p`, but DPAS B operand layout is typically VNNI-like packed in K-major chunks expected by hardware. Reorder B in SLM into the required packed layout before compute (or load B already transformed), then feed the intrinsic directly. Correcting B layout is usually the single biggest fix for both correctness and XMX throughput.

7. `A_slm`/`B_slm` scalar cooperative loads: Functional but not aligned with DPAS-friendly layout. Improve by storing B tile in packed/interleaved form in SLM during load (e.g., pairwise K packing) so compute loop avoids per-iteration scalar repacking. This removes expensive per-lane pack loops and aligns memory layout with DPAS expectations, improving both correctness robustness and performance.

8. `C[out_row * N + out_col] = convert_half(val);`: You always downcast FP32 accum to FP16, which matches test buffers, but make sure conversion mode is explicit (`convert_half_rte`) to avoid compiler-dependent rounding differences near tolerance boundaries. This won’t fix zeros, but it prevents future borderline mismatches once DPAS path is corrected.

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
- **Sub-Group Collectives**: Use `reduce_over_group(sg, val, op)` for hardware-accelerated SIMD reductions. Use `group_broadcast` and `shift_group_*` for efficient data sharing.