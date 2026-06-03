

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
- Current kernel achieves 0.948ms = 23% XMX utilization on B580 (96 TFLOPS peak).
    Hardware: B580 = 20 Xe2 cores, 96 TFLOPS FP16 XMX, 456 GB/s, 32MB L2, 128 KB SLM per core.
    DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
- Focus on improving memory access efficiency and compute utilization.
- Adopts better thread walker and blocking strategy to maximize DPAS usage and hide memory latency.
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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 5.130):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 128 cols, K-step: 32
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup computes 32 rows x 16 cols = 4 vertical DPAS(8x16) tiles
// Both A (32x32) and B (32x128) cached in SLM, double-buffered
// SLM usage: 2 * (32*32 + 32*128) * 2 bytes = 2 * (1024 + 4096) * 2 = 20480 bytes = 20KB
// GWS = (ceil(N/128)*128, ceil(M/32))  LWS = (128, 1)
// Subgroup size = 16

#define TILE_M 32
#define TILE_N 128
#define TILE_K 32
#define SLM_A_TILE (TILE_M * TILE_K)   // 1024 elements
#define SLM_B_TILE (TILE_K * TILE_N)   // 4096 elements
#define SLM_BUF_SIZE (SLM_A_TILE + SLM_B_TILE) // 5120 elements per buffer

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_sub_group_id();    // 0..7
    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int lid = get_local_id(0);         // 0..127

    const int n_base = get_group_id(0) * TILE_N;   // column base for this WG
    const int row_base = get_group_id(1) * TILE_M;  // row base for this WG
    const int col_base = n_base + sg_id * 16;       // column base for this subgroup

    // Double-buffered SLM for A and B
    __local ushort slm[2 * SLM_BUF_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    // Accumulators: 4 tiles of 8x16 each = 32 rows x 16 cols
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const bool row_tile_full = (row_base + TILE_M <= M);

    // Cooperative loading helpers
    // 128 WIs load A: 32x32 = 1024 elements, 1024/128 = 8 elements per WI
    // 128 WIs load B: 32x128 = 4096 elements, 4096/128 = 32 elements per WI

    // --- Load first tile into buffer 0 ---
    {
        __local ushort* slm_a = slm;
        __local ushort* slm_b = slm + SLM_A_TILE;
        int k_off = 0;

        // Load A[row_base..row_base+31, k_off..k_off+31] into slm_a[row][col] row-major
        // Each WI loads 8 elements
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int elem = lid + i * 128;
            int ar = elem / TILE_K;  // row within tile (0..31)
            int ac = elem % TILE_K;  // col within tile (0..31)
            int gr = row_base + ar;
            ushort val = (gr < M && (k_off + ac) < K) ? A_us[gr * K + k_off + ac] : (ushort)0;
            slm_a[elem] = val;
        }

        // Load B[k_off..k_off+31, n_base..n_base+127] into slm_b[row][col] row-major
        // Each WI loads 32 elements
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int elem = lid + i * 128;
            int br = elem / TILE_N;  // row within tile (0..31)
            int bc = elem % TILE_N;  // col within tile (0..127)
            int gk = k_off + br;
            int gn = n_base + bc;
            ushort val = (gk < K && gn < N) ? B_us[gk * N + gn] : (ushort)0;
            slm_b[elem] = val;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int next_kt = kt + 1;
        const bool has_next = (next_kt < num_k_tiles);

        __local const ushort* cur_a = slm + cur_buf * SLM_BUF_SIZE;
        __local const ushort* cur_b = slm + cur_buf * SLM_BUF_SIZE + SLM_A_TILE;

        // B offset for this subgroup: sg_id * 16 columns
        __local const ushort* b_sg = cur_b + sg_id * 16;

        // ---- COMPUTE k16 step 0 (columns 0..15 of K tile) ----
        {
            // Load A sub-tiles from SLM: 4 x short8 (rows 0-7, 8-15, 16-23, 24-31)
            // A is stored as slm_a[row * TILE_K + col], we need k=0..15
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(cur_a + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(cur_a + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(cur_a + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(cur_a + (24 + r) * TILE_K);

            // Load B: need 16 rows x 16 cols from SLM
            // B stored as slm_b[k_row * TILE_N + col], subgroup offset = sg_id*16
            // DPAS b format: int8 where each int packs 2 consecutive k-row fp16 values
            // b_val[p] = pack(B[2p, sg_lid], B[2p+1, sg_lid])
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(b_sg + (2 * p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(b_sg + (2 * p + 1) * TILE_N);
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ---- COMPUTE k16 step 1 (columns 16..31 of K tile) ----
        {
            short8 a0, a1, a2, a3;
            __local const ushort* a_base = cur_a + 16;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(a_base + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(a_base + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(a_base + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(a_base + (24 + r) * TILE_K);

            __local const ushort* b_base = b_sg + 16 * TILE_N;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(b_base + (2 * p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(b_base + (2 * p + 1) * TILE_N);
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ---- PREFETCH next tile into other buffer ----
        if (has_next) {
            int next_buf = 1 - cur_buf;
            __local ushort* next_a = slm + next_buf * SLM_BUF_SIZE;
            __local ushort* next_b = slm + next_buf * SLM_BUF_SIZE + SLM_A_TILE;
            int next_k = next_kt * TILE_K;

            barrier(CLK_LOCAL_MEM_FENCE);

            // Load A
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int elem = lid + i * 128;
                int ar = elem / TILE_K;
                int ac = elem % TILE_K;
                int gr = row_base + ar;
                ushort val = (gr < M && (next_k + ac) < K) ? A_us[gr * K + next_k + ac] : (ushort)0;
                next_a[elem] = val;
            }

            // Load B
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int elem = lid + i * 128;
                int br = elem / TILE_N;
                int bc = elem % TILE_N;
                int gk = next_k + br;
                int gn = n_base + bc;
                ushort val = (gk < K && gn < N) ? B_us[gk * N + gn] : (ushort)0;
                next_b[elem] = val;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = next_buf;
        }
    }

    // ---- Store results ----
    const int col_idx = col_base + sg_lid;
    if (col_idx < N) {
        __global half* C_ptr = C + col_idx;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc0)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 8 + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc1)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 16 + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc2)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 24 + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 6.430):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// =============================================================================
// Optimized FP16 GEMM: C[M,N] = A[M,K] * B[K,N]
// Row-major, half I/O, float accumulation
//
// WG tile: 64 rows x 128 cols, K-step: 32
// 8 subgroups x 16 WIs = 128 WIs per WG
// Subgroup layout: 2 rows x 4 cols
//   sg_row = sg_id / 4 (0..1), sg_col = sg_id % 4 (0..3)
//   Each subgroup: 32 rows x 32 cols = 4 height x 2 width DPAS(8x16) blocks
//   = 8 accumulators (float8) per subgroup
//
// SLM double-buffered:
//   A tile: 64 x 32 = 2048 ushorts = 4 KB
//   B tile: 32 x 128 = 4096 ushorts = 8 KB
//   Per buffer: 12 KB, total: 24 KB (fits in 128 KB SLM)
//
// Cooperative loads (128 WIs):
//   A: 2048 elements / 128 = 16 per WI
//   B: 4096 elements / 128 = 32 per WI
//
// Per K-tile iteration: 2 k16 steps x 8 DPAS calls = 16 DPAS per subgroup
// Total DPAS per WG per K-tile: 16 * 8 = 128
//
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)
// Subgroup size = 16
// =============================================================================

#define TILE_M 64
#define TILE_N 128
#define TILE_K 32
#define SG_TILE_M 32
#define SG_TILE_N 32
#define WG_SIZE 128

#define SLM_A_SIZE (TILE_M * TILE_K)   // 2048 ushorts
#define SLM_B_SIZE (TILE_K * TILE_N)   // 4096 ushorts
#define SLM_BUF_SIZE (SLM_A_SIZE + SLM_B_SIZE) // 6144 ushorts

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id  = get_sub_group_id();        // 0..7
    const int sg_lid = get_sub_group_local_id();  // 0..15
    const int lid    = get_local_id(0);           // 0..127

    // WG tile origin
    const int n_base   = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;

    // Subgroup position: 2 rows x 4 cols
    const int sg_row = sg_id >> 2;    // 0 or 1
    const int sg_col = sg_id & 3;     // 0..3
    const int sg_m_off = sg_row * SG_TILE_M;  // 0 or 32
    const int sg_n_off = sg_col * SG_TILE_N;  // 0, 32, 64, 96

    // Double-buffered SLM
    __local ushort slm[2 * SLM_BUF_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    // 8 accumulators: 4 row-blocks x 2 col-blocks of DPAS 8x16
    // Rows 0-7 x cols 0-15, rows 0-7 x cols 16-31, ...
    float8 acc00 = 0.0f;  // rows 0-7,   cols 0-15
    float8 acc01 = 0.0f;  // rows 0-7,   cols 16-31
    float8 acc10 = 0.0f;  // rows 8-15,  cols 0-15
    float8 acc11 = 0.0f;  // rows 8-15,  cols 16-31
    float8 acc20 = 0.0f;  // rows 16-23, cols 0-15
    float8 acc21 = 0.0f;  // rows 16-23, cols 16-31
    float8 acc30 = 0.0f;  // rows 24-31, cols 0-15
    float8 acc31 = 0.0f;  // rows 24-31, cols 16-31

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // ---- Load first tile into buffer 0 ----
    {
        __local ushort* slm_a = slm;
        __local ushort* slm_b = slm + SLM_A_SIZE;

        // Load A: 64x32 = 2048 elements, 16 per WI, coalesced
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid >> 5;    // eid / 32
            int ac = eid & 31;    // eid % 32
            int gr = row_base + ar;
            ushort val = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
            slm_a[eid] = val;
        }

        // Load B: 32x128 = 4096 elements, 32 per WI, coalesced
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int eid = lid + i * WG_SIZE;
            int br = eid >> 7;    // eid / 128
            int bc = eid & 127;   // eid % 128
            int gc = n_base + bc;
            ushort val = (br < K && gc < N) ? B_us[br * N + gc] : (ushort)0;
            slm_b[eid] = val;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int next_kt = kt + 1;
        const bool has_next = (next_kt < num_k_tiles);

        __local const ushort* cur_a = slm + cur_buf * SLM_BUF_SIZE;
        __local const ushort* cur_b = cur_a + SLM_A_SIZE;

        // SLM pointers for this subgroup
        __local const ushort* sg_a = cur_a + sg_m_off * TILE_K;  // A rows for this SG
        __local const ushort* sg_b0 = cur_b + sg_n_off;          // B cols 0-15 for this SG
        __local const ushort* sg_b1 = cur_b + sg_n_off + 16;     // B cols 16-31 for this SG

        // ---- k16 step 0 (k columns 0..15) ----
        {
            // Load A: 32 rows x k16, from SLM (stride = TILE_K = 32)
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(sg_a + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(sg_a + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(sg_a + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(sg_a + (24 + r) * TILE_K);

            // Load B col-block 0: k16 rows x 16 cols (stride = TILE_N = 128)
            int8 bv0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(sg_b0 + (2 * p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(sg_b0 + (2 * p + 1) * TILE_N);
                ((int*)&bv0)[p] = as_int((ushort2)(s0, s1));
            }

            // Load B col-block 1: k16 rows x 16 cols
            int8 bv1;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(sg_b1 + (2 * p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(sg_b1 + (2 * p + 1) * TILE_N);
                ((int*)&bv1)[p] = as_int((ushort2)(s0, s1));
            }

            // 8 DPAS calls
            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv0, acc00);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv1, acc01);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv0, acc10);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv1, acc11);
            acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv0, acc20);
            acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv1, acc21);
            acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv0, acc30);
            acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv1, acc31);
        }

        // ---- k16 step 1 (k columns 16..31) ----
        {
            __local const ushort* sg_a2 = sg_a + 16;
            __local const ushort* sg_b0_2 = sg_b0 + 16 * TILE_N;
            __local const ushort* sg_b1_2 = sg_b1 + 16 * TILE_N;

            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(sg_a2 + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(sg_a2 + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(sg_a2 + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(sg_a2 + (24 + r) * TILE_K);

            int8 bv0;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(sg_b0_2 + (2 * p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(sg_b0_2 + (2 * p + 1) * TILE_N);
                ((int*)&bv0)[p] = as_int((ushort2)(s0, s1));
            }

            int8 bv1;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(sg_b1_2 + (2 * p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(sg_b1_2 + (2 * p + 1) * TILE_N);
                ((int*)&bv1)[p] = as_int((ushort2)(s0, s1));
            }

            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv0, acc00);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, bv1, acc01);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv0, acc10);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, bv1, acc11);
            acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv0, acc20);
            acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, bv1, acc21);
            acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv0, acc30);
            acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, bv1, acc31);
        }

        // ---- Load next K-tile ----
        if (has_next) {
            int next_buf = 1 - cur_buf;
            __local ushort* next_a = slm + next_buf * SLM_BUF_SIZE;
            __local ushort* next_b = next_a + SLM_A_SIZE;
            int next_k = next_kt * TILE_K;

            barrier(CLK_LOCAL_MEM_FENCE);

            // Load A: coalesced
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid >> 5;
                int ac = eid & 31;
                int gr = row_base + ar;
                int gk = next_k + ac;
                next_a[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }

            // Load B: coalesced
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int eid = lid + i * WG_SIZE;
                int br = eid >> 7;
                int bc = eid & 127;
                int gk = next_k + br;
                int gc = n_base + bc;
                next_b[eid] = (gk < K && gc < N) ? B_us[gk * N + gc] : (ushort)0;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = next_buf;
        }
    }

    // ---- Store results ----
    // Each subgroup writes 32 rows x 32 cols as two 32x16 blocks
    {
        // Block 0: cols 0-15
        const int out_col0 = n_base + sg_n_off + sg_lid;
        if (out_col0 < N) {
            __global half* C_ptr = C + out_col0;
            int r_base = row_base + sg_m_off;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = r_base + r;
                if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc00)[r]);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = r_base + 8 + r;
                if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc10)[r]);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = r_base + 16 + r;
                if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc20)[r]);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = r_base + 24 + r;
                if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc30)[r]);
            }
        }

        // Block 1: cols 16-31
        const int out_col1 = n_base + sg_n_off + 16 + sg_lid;
        if (out_col1 < N) {
            __global half* C_ptr = C + out_col1;
            int r_base = row_base + sg_m_off;
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = r_base + r;
                if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc01)[r]);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = r_base + 8 + r;
                if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc11)[r]);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = r_base + 16 + r;
                if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc21)[r]);
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = r_base + 24 + r;
                if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc31)[r]);
            }
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 4.730):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 128 cols, K-step: 32
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// Both A and B cached in SLM, double-buffered
// SLM: 2 * (32*32 + 32*128) * 2 bytes = 20 KB per WG
// GWS = (ceil(N/128)*128, ceil(M/32))  LWS = (128, 1)

#define TILE_M 32
#define TILE_N 128
#define TILE_K 32
#define SLM_A_TILE (TILE_M * TILE_K)       // 1024 ushorts
#define SLM_B_TILE (TILE_K * TILE_N)       // 4096 ushorts
#define SLM_TILE_TOTAL (SLM_A_TILE + SLM_B_TILE)  // 5120 ushorts

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_sub_group_id();       // 0..7
    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int lid = get_local_id(0);            // 0..127

    const int n_base = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;

    // Each subgroup handles a 16-col slice of the 128-col tile
    const int col_base = n_base + sg_id * 16;

    // Double-buffered SLM
    __local ushort slm[2 * SLM_TILE_TOTAL];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;  // rows 0-7
    float8 acc1 = 0.0f;  // rows 8-15
    float8 acc2 = 0.0f;  // rows 16-23
    float8 acc3 = 0.0f;  // rows 24-31

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const bool row_tile_valid = (row_base + TILE_M <= M);
    const bool col_tile_valid = (n_base + TILE_N <= N);
    const int num_k_tiles = K / TILE_K;

    // Cooperative load helpers
    // 128 WIs load 1024 A elements: each WI loads 8 elements
    // 128 WIs load 4096 B elements: each WI loads 32 elements

    // --- Load first tiles into buffer 0 ---
    {
        __local ushort* slm_a = slm;
        __local ushort* slm_b = slm + SLM_A_TILE;

        // Load A[row_base..row_base+31, 0..31] into SLM row-major with stride TILE_K
        // 1024 elements / 128 WIs = 8 per WI
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int elem = lid * 8 + i;
            int r = elem / TILE_K;  // row within tile
            int c = elem % TILE_K;  // col within tile (k dim)
            int gr = row_base + r;
            ushort val = 0;
            if (gr < M && c < K)
                val = A_us[gr * K + c];
            slm_a[r * TILE_K + c] = val;
        }

        // Load B[0..31, n_base..n_base+127] into SLM row-major with stride TILE_N
        // 4096 elements / 128 WIs = 32 per WI
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int elem = lid * 32 + i;
            int r = elem / TILE_N;  // row within tile (k dim)
            int c = elem % TILE_N;  // col within tile
            int gc = n_base + c;
            ushort val = 0;
            if (r < K && gc < N)
                val = B_us[r * N + gc];
            slm_b[r * TILE_N + c] = val;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k = kt * TILE_K;
        const int next_kt = kt + 1;
        const bool has_next = (next_kt < num_k_tiles);

        __local const ushort* cur_a = slm + cur_buf * SLM_TILE_TOTAL;
        __local const ushort* cur_b = cur_a + SLM_A_TILE;

        // Offset B SLM pointer to this subgroup's 16-col slice
        __local const ushort* my_b = cur_b + sg_id * 16;

        // ---- k16 step 0 (k offset 0..15) ----
        {
            short8 a0, a1, a2, a3;
            // Load A from SLM: each DPAS needs 8 rows × k16
            // A in SLM: row-major, stride TILE_K=32
            // For k16 step 0, columns 0..15
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(cur_a + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(cur_a + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(cur_a + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(cur_a + (24 + r) * TILE_K);

            // Load B from SLM: k16 × 16 cols
            // B in SLM: row-major, stride TILE_N=128
            // For k16 step 0, rows 0..15 of B tile
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(my_b + (2 * p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(my_b + (2 * p + 1) * TILE_N);
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ---- k16 step 1 (k offset 16..31) ----
        {
            short8 a0, a1, a2, a3;
            __local const ushort* a_base = cur_a + 16;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(a_base + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(a_base + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(a_base + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(a_base + (24 + r) * TILE_K);

            __local const ushort* b_base = my_b + 16 * TILE_N;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(b_base + (2 * p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(b_base + (2 * p + 1) * TILE_N);
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ---- LOAD next tiles ----
        if (has_next) {
            barrier(CLK_LOCAL_MEM_FENCE);

            int next_buf = 1 - cur_buf;
            __local ushort* next_a = slm + next_buf * SLM_TILE_TOTAL;
            __local ushort* next_b = next_a + SLM_A_TILE;
            int next_k = next_kt * TILE_K;

            // Load A
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int elem = lid * 8 + i;
                int r = elem / TILE_K;
                int c = elem % TILE_K;
                int gr = row_base + r;
                ushort val = 0;
                if (gr < M && (next_k + c) < K)
                    val = A_us[gr * K + next_k + c];
                next_a[r * TILE_K + c] = val;
            }

            // Load B
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int elem = lid * 32 + i;
                int r = elem / TILE_N;
                int c = elem % TILE_N;
                int gk = next_k + r;
                int gc = n_base + c;
                ushort val = 0;
                if (gk < K && gc < N)
                    val = B_us[gk * N + gc];
                next_b[r * TILE_N + c] = val;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = next_buf;
        }
    }

    // ---- Store results ----
    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;

    if (col_valid) {
        __global half* C_ptr = C + col_idx;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc0)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 8 + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc1)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 16 + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc2)[r]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 24 + r;
            if (gr < M) C_ptr[gr * N] = convert_half(((float*)&acc3)[r]);
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
================== 4 passed, 1 deselected, 1 warning in 0.74s ==================
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
1. Loop Unrolling: Use #pragma unroll N for small, fixed-iteration loops. Manually unroll critical loops when compiler doesn't optimize. Prefer compile-time loop bounds.
2. Kernel Fusion: Combine multiple small kernels into one to eliminate intermediate global memory writes and reduce launch overhead. Use barrier(CLK_LOCAL_MEM_FENCE) between logical kernel phases within a work-group.

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