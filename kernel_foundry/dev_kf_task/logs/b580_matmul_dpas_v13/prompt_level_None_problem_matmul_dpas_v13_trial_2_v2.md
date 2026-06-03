

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

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 4.730):
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Optimized GEMM: C[M,N] = A[M,K] * B[K,N], all half, float accumulation
//
// WG tile: 64 rows x 64 cols, K-step: 32
// 8 subgroups x 16 WIs = 128 WIs per WG
// Subgroup layout: 2 cols (sg_col) x 4 rows (sg_row)
//   sg_col = sg_id & 1,  sg_row = sg_id >> 1
//   Each subgroup computes 16 rows x 32 cols (2 DPAS-width x 2 k16 steps)
//   Actually: each subgroup computes 16 rows x 32 cols via 2 horizontal DPAS tiles
//   = 2 float8 accumulators per row-group, 2 row-groups of 8 = 16 rows
//   Wait, DPAS does 8 rows x 16 cols. So 16 rows x 32 cols = 2x2 = 4 DPAS calls per k16.
//
// Both A(64x32) and B(32x64) cached in SLM, double-buffered.
//   A SLM: 64 * 32 * 2 bytes = 4096 bytes per buffer
//   B SLM: 32 * 64 * 2 bytes = 4096 bytes per buffer  (stored as 32 rows x 64 cols, row-major)
//   Total: 2 * (4096 + 4096) = 16384 bytes = 16 KB << 128 KB limit
//
// GWS = (ceil(N/64)*128, ceil(M/64))  LWS = (128, 1)
// Subgroup size = 16

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define SLM_A_SIZE (TILE_M * TILE_K)   // 2048 ushorts = 4096 bytes
#define SLM_B_SIZE (TILE_K * TILE_N)   // 2048 ushorts = 4096 bytes
#define SLM_BUF_SIZE (SLM_A_SIZE + SLM_B_SIZE)  // 4096 ushorts per buffer

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

    // Subgroup grid: 2 columns x 4 rows
    const int sg_col = sg_id & 1;      // 0 or 1
    const int sg_row = sg_id >> 1;     // 0..3

    const int n_base = get_group_id(0) * TILE_N;
    const int m_base = get_group_id(1) * TILE_M;

    // Each subgroup's output tile origin
    const int out_row = m_base + sg_row * 16;  // 16 rows per subgroup
    const int out_col = n_base + sg_col * 32;  // 32 cols per subgroup

    // Double-buffered SLM
    // Layout: [buf0_A | buf0_B | buf1_A | buf1_B]
    __local ushort slm[2 * SLM_BUF_SIZE];

    if (m_base >= M && n_base >= N)
        return;

    // Accumulators: 2 row-blocks (8 rows each) x 2 col-blocks (16 cols each) = 4 float8
    float8 acc00 = 0.0f, acc01 = 0.0f;  // row-block 0, col-block 0 and 1
    float8 acc10 = 0.0f, acc11 = 0.0f;  // row-block 1, col-block 0 and 1

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Cooperative load helpers - 128 WIs load 2048 elements for A and 2048 for B
    // A tile: 64 rows x 32 cols = 2048 elements, 128 WIs => 16 elements per WI
    // B tile: 32 rows x 64 cols = 2048 elements, 128 WIs => 16 elements per WI

    // ---- Load first tiles into buffer 0 ----
    {
        __local ushort* slm_a = slm;
        __local ushort* slm_b = slm + SLM_A_SIZE;

        // Load A: each WI loads 16 elements
        // A is stored in SLM as [row][k] with stride TILE_K=32
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int elem = lid + i * 128;
            int r = elem / TILE_K;   // elem / 32
            int c = elem % TILE_K;   // elem & 31
            int gr = m_base + r;
            slm_a[elem] = (gr < M && c < K) ? A_us[gr * K + c] : (ushort)0;
        }

        // Load B: each WI loads 16 elements
        // B is stored in SLM as [k][col] with stride TILE_N=64
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int elem = lid + i * 128;
            int r = elem / TILE_N;   // elem / 64
            int c = elem % TILE_N;   // elem & 63
            int gc = n_base + c;
            slm_b[elem] = (r < K && gc < N) ? B_us[r * N + gc] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k = kt * TILE_K;
        const int next_kt = kt + 1;
        const bool has_next = (next_kt < num_k_tiles);

        __local const ushort* cur_a = slm + cur_buf * SLM_BUF_SIZE;
        __local const ushort* cur_b = slm + cur_buf * SLM_BUF_SIZE + SLM_A_SIZE;

        // ---- K16 step 0 (k offset 0..15 within tile) ----
        {
            // Load A sub-tiles from SLM for this subgroup's rows
            // sg_row*16 gives the row offset within the 64-row tile
            // Each DPAS needs short8 = 8 rows x 16 cols (k)
            short8 a0, a1;
            __local const ushort* a_base = cur_a + sg_row * 16 * TILE_K; // row offset * stride

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(a_base + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(a_base + (8 + r) * TILE_K);

            // Load B sub-tiles from SLM for this subgroup's columns
            // B in SLM: [k_row][col], stride TILE_N=64
            // Need int8 for DPAS B operand: 16 k-values x 16 cols packed as int8
            // Each int packs 2 consecutive k-values for one column
            int8 b0, b1;
            __local const ushort* b_base0 = cur_b + sg_col * 32;      // col offset for first 16-col block
            __local const ushort* b_base1 = cur_b + sg_col * 32 + 16; // col offset for second 16-col block

            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(b_base0 + (2*p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(b_base0 + (2*p+1) * TILE_N);
                ((int*)&b0)[p] = as_int((ushort2)(s0, s1));
            }
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(b_base1 + (2*p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(b_base1 + (2*p+1) * TILE_N);
                ((int*)&b1)[p] = as_int((ushort2)(s0, s1));
            }

            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
        }

        // ---- K16 step 1 (k offset 16..31 within tile) ----
        {
            short8 a0, a1;
            __local const ushort* a_base = cur_a + sg_row * 16 * TILE_K + 16;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(a_base + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(a_base + (8 + r) * TILE_K);

            int8 b0, b1;
            __local const ushort* b_base0 = cur_b + 16 * TILE_N + sg_col * 32;
            __local const ushort* b_base1 = cur_b + 16 * TILE_N + sg_col * 32 + 16;

            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(b_base0 + (2*p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(b_base0 + (2*p+1) * TILE_N);
                ((int*)&b0)[p] = as_int((ushort2)(s0, s1));
            }
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(b_base1 + (2*p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(b_base1 + (2*p+1) * TILE_N);
                ((int*)&b1)[p] = as_int((ushort2)(s0, s1));
            }

            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
        }

        // ---- Prefetch next tile ----
        if (has_next) {
            int next_buf = 1 - cur_buf;
            __local ushort* next_a = slm + next_buf * SLM_BUF_SIZE;
            __local ushort* next_b = slm + next_buf * SLM_BUF_SIZE + SLM_A_SIZE;
            int next_k = next_kt * TILE_K;

            barrier(CLK_LOCAL_MEM_FENCE);

            // Load next A tile
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem = lid + i * 128;
                int r = elem >> 5;     // / 32
                int c = elem & 31;     // % 32
                int gr = m_base + r;
                int gk = next_k + c;
                next_a[elem] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }

            // Load next B tile
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem = lid + i * 128;
                int r = elem >> 6;     // / 64
                int c = elem & 63;     // % 64
                int gk = next_k + r;
                int gc = n_base + c;
                next_b[elem] = (gk < K && gc < N) ? B_us[gk * N + gc] : (ushort)0;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = next_buf;
        }
    }

    // ---- Store results ----
    // Each subgroup writes 16 rows x 32 cols
    // acc00: rows [0..7], cols [0..15]   (sg_lid selects column within 16)
    // acc01: rows [0..7], cols [16..31]
    // acc10: rows [8..15], cols [0..15]
    // acc11: rows [8..15], cols [16..31]

    {
        int store_row_base = out_row;
        int store_col0 = out_col + sg_lid;        // first 16-col block
        int store_col1 = out_col + 16 + sg_lid;   // second 16-col block

        bool col0_valid = (store_col0 < N);
        bool col1_valid = (store_col1 < N);

        __global ushort* C_us = (__global ushort*)C;

        // Write acc00 (rows 0..7, cols 0..15)
        if (col0_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = store_row_base + r;
                if (gr < M)
                    C_us[gr * N + store_col0] = as_ushort(convert_half(((float*)&acc00)[r]));
            }
        }

        // Write acc01 (rows 0..7, cols 16..31)
        if (col1_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = store_row_base + r;
                if (gr < M)
                    C_us[gr * N + store_col1] = as_ushort(convert_half(((float*)&acc01)[r]));
            }
        }

        // Write acc10 (rows 8..15, cols 0..15)
        if (col0_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = store_row_base + 8 + r;
                if (gr < M)
                    C_us[gr * N + store_col0] = as_ushort(convert_half(((float*)&acc10)[r]));
            }
        }

        // Write acc11 (rows 8..15, cols 16..31)
        if (col1_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = store_row_base + 8 + r;
                if (gr < M)
                    C_us[gr * N + store_col1] = as_ushort(convert_half(((float*)&acc11)[r]));
            }
        }
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

self = <task.TestMatmulOCL object at 0x7d689e6fde80>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7d689e73f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7d689e7498f0>, _run = 0

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
E        +  where False = <function allclose at 0x7d68f6f642f0>(array([[-12.4375   , -85.25     ,  45.875    , ...,   0.       ,\n          0.       ,   0.       ],\n       [-82.9375   ,  28.328125 ,   4.3085938, ...,   0.       ,\n          0.       ,   0.       ],\n       [ 31.09375  , -51.78125  ,  -9.3046875, ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [-65.3125   , -27.734375 ,  74.1875   , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 44.6875   ,   2.3144531,  22.609375 , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 50.40625  ,  -3.015625 ,  21.546875 , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[-12.434087 , -85.22102  ,  45.86866  , ..., -67.074715 ,\n        -64.52674  ,  37.798523 ],\n       [-82.95244  ,  28.332115 ,   4.3084497, ...,  37.17192  ,\n         48.87541  ,  55.1519   ],\n       [ 31.096529 , -51.77693  ,  -9.3054905, ...,   8.124319 ,\n         61.21928  ,   4.7092314],\n       ...,\n       [-65.29967  , -27.73106  ,  74.195465 , ..., 122.09403  ,\n        -41.569603 ,  10.711429 ],\n       [ 44.6838   ,   2.3142765,  22.61605  , ..., -35.807106 ,\n         42.793472 ,  52.60636  ],\n       [ 50.399834 ,  -3.015791 ,  21.545517 , ..., -21.399685 ,\n        -36.035267 ,  49.01544  ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7d68f6f642f0> = np.allclose

task.py:99: AssertionError
________________ TestMatmulOCL.test_correctness_wrt_pytorch[1] _________________

self = <task.TestMatmulOCL object at 0x7d689e692900>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7d689e73f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7d689e7498f0>, _run = 1

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
E        +  where False = <function allclose at 0x7d68f6f642f0>(array([[ 63.21875  ,  13.6640625,  62.71875  , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 30.34375  ,  -9.578125 , -15.8515625, ...,   0.       ,\n          0.       ,   0.       ],\n       [ -9.828125 ,  26.84375  , -39.875    , ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [-50.9375   , -10.71875  , -18.65625  , ...,   0.       ,\n          0.       ,   0.       ],\n       [-41.46875  ,  -5.0351562,  35.5      , ...,   0.       ,\n          0.       ,   0.       ],\n       [-33.90625  ,  51.46875  ,  13.109375 , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[  63.220627 ,   13.663691 ,   62.708282 , ...,   26.950535 ,\n        -100.14888  ,  -76.10468  ],\n       [  30.338015 ,   -9.576593 ,  -15.848044 , ...,  -86.66203  ,\n           6.3691177,    9.569207 ],\n       [  -9.825886 ,   26.83852  ,  -39.88768  , ...,   94.32298  ,\n         -40.437588 ,   13.349518 ],\n       ...,\n       [ -50.946926 ,  -10.7210655,  -18.652342 , ...,   -4.0612535,\n         -29.112085 ,   -2.7683525],\n       [ -41.46417  ,   -5.034666 ,   35.500336 , ...,    3.5289268,\n          14.26104  ,   55.58531  ],\n       [ -33.896618 ,   51.45737  ,   13.108513 , ...,   11.92079  ,\n         -64.022385 ,   63.048595 ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7d68f6f642f0> = np.allclose

task.py:99: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[0] ________________

self = <task.TestMatmulOCL object at 0x7d689e736db0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7d689e73f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7d689e7498f0>, _run = 0

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
E        +  where False = <function allclose at 0x7d68f6f642f0>(array([[ -57.125    ,  -28.015625 ,   65.5625   , ...,    0.       ,\n           0.       ,    0.       ],\n       [  92.6875   ,   18.34375  ,  -14.34375  , ...,    0.       ,\n           0.       ,    0.       ],\n       [  -2.5585938,  -30.015625 ,   49.9375   , ...,    0.       ,\n           0.       ,    0.       ],\n       ...,\n       [ -46.03125  , -142.625    ,  -61.5      , ...,    0.       ,\n           0.       ,    0.       ],\n       [  31.859375 ,  -23.484375 ,   42.75     , ...,    0.       ,\n           0.       ,    0.       ],\n       [ -38.96875  ,   91.75     ,   23.953125 , ...,    0.       ,\n           0.       ,    0.       ]], shape=(2048, 2048), dtype=float32), array([[-5.7125000e+01, -2.8015625e+01,  6.5562500e+01, ...,\n        -5.0218750e+01,  2.7000000e+01,  2.6109375e+01],\n       [ 9.2687500e+01,  1.8343750e+01, -1.4343750e+01, ...,\n        -4.0843750e+01, -2.5828125e+01, -3.1914062e+00],\n       [-2.5585938e+00, -3.0015625e+01,  4.9937500e+01, ...,\n        -5.7156250e+01, -9.2250000e+01,  1.2921875e+01],\n       ...,\n       [-4.6031250e+01, -1.4262500e+02, -6.1500000e+01, ...,\n        -1.2194824e-01,  6.6445312e+00, -5.3156250e+01],\n       [ 3.1859375e+01, -2.3484375e+01,  4.2750000e+01, ...,\n         1.8390625e+01, -2.0507812e+00, -1.6000000e+02],\n       [-3.8968750e+01,  9.1750000e+01,  2.3953125e+01, ...,\n         4.5437500e+01,  5.4437500e+01,  1.0787500e+02]],\n      shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7d68f6f642f0> = np.allclose

task.py:117: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[1] ________________

self = <task.TestMatmulOCL object at 0x7d689e736ed0>
kernel = <function initialize_matmul_kernel.<locals>.run at 0x7d689e73f060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x7d689e7498f0>, _run = 1

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
E        +  where False = <function allclose at 0x7d68f6f642f0>(array([[ 46.4375   , -21.75     , -36.71875  , ...,   0.       ,\n          0.       ,   0.       ],\n       [-13.859375 , -17.453125 ,  17.375    , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 34.1875   ,  -9.421875 , -49.       , ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [  3.3261719, -60.03125  , -64.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [ -0.9145508,  97.25     , -18.109375 , ...,   0.       ,\n          0.       ,   0.       ],\n       [-36.5      , -63.8125   ,  68.125    , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[  46.4375   ,  -21.75     ,  -36.71875  , ...,  -39.96875  ,\n           3.125    , -147.25     ],\n       [ -13.859375 ,  -17.453125 ,   17.375    , ...,   57.90625  ,\n          25.859375 ,  -68.0625   ],\n       [  34.1875   ,   -9.421875 ,  -49.       , ...,  -46.71875  ,\n         -64.5625   ,   37.6875   ],\n       ...,\n       [   3.3261719,  -60.03125  ,  -64.       , ...,   18.328125 ,\n         -17.234375 ,  -30.90625  ],\n       [  -0.9145508,   97.25     ,  -18.109375 , ...,   35.75     ,\n         -45.0625   ,   21.78125  ],\n       [ -36.5      ,  -63.8125   ,   68.125    , ...,  -48.09375  ,\n         -29.515625 ,   11.1640625]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x7d68f6f642f0> = np.allclose

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
================== 4 failed, 1 deselected, 1 warning in 0.98s ==================

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