

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

### Version 1 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.140):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Tile: 32M x 256N, K-step=16
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16)
// Only A in SLM (shared across 16 subgroups = 16x reuse)
// B loaded directly from global (each SG loads its own 16x16 block)
// Double-buffered A in SLM: 2 * 32*16 * 2 bytes = 2 KB
// GWS = (ceil(N/256)*256, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16
// K unrolled by 2: load 2 k-tiles of A, do 2 rounds of DPAS per iteration

#define TILE_M 32
#define TILE_N 256
#define TILE_K 16
#define SLM_A_SIZE (TILE_M * TILE_K)
#define WG_SIZE 256
#define NUM_SG 16

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
    const int sg_id  = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid    = get_local_id(0);

    const int n_base   = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;
    const int col_base = n_base + sg_id * 16;

    __local ushort slm_a[2 * SLM_A_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_pairs = K / (2 * TILE_K);
    const int k_remainder = K - num_k_pairs * 2 * TILE_K;

    // Load first A tile (k=0) into buffer 0
    {
        __local ushort* dst = slm_a;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;
            int ac = eid % TILE_K;
            int gr = row_base + ar;
            dst[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Process K in pairs of 2 tiles for better pipelining
    for (int kp = 0; kp < num_k_pairs; kp++) {
        int k0 = kp * 2 * TILE_K;
        int k1 = k0 + TILE_K;

        // === K-step 0: compute with buffer 0, load A k1 into buffer 1 ===
        {
            __local const ushort* ca = slm_a;

            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(ca + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(ca + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(ca + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(ca + (24 + r) * TILE_K);

            int8 b_val;
            __global const ushort* bp = B_us + k0 * N + col_base;
            if (col_base + 16 <= N) {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = intel_sub_group_block_read_us(bp + (2*p) * N);
                    ushort s1 = intel_sub_group_block_read_us(bp + (2*p+1) * N);
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                }
            } else {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int col = col_base + sg_lid;
                    ushort s0 = (col < N) ? B_us[(k0 + 2*p) * N + col] : (ushort)0;
                    ushort s1 = (col < N) ? B_us[(k0 + 2*p+1) * N + col] : (ushort)0;
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                }
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // Load A for k1 into buffer 1
        barrier(CLK_LOCAL_MEM_FENCE);
        {
            __local ushort* dst = slm_a + SLM_A_SIZE;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = row_base + ar;
                int gk = k1 + ac;
                dst[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // === K-step 1: compute with buffer 1, load A for next pair into buffer 0 ===
        {
            __local const ushort* ca = slm_a + SLM_A_SIZE;

            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(ca + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(ca + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(ca + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(ca + (24 + r) * TILE_K);

            int8 b_val;
            __global const ushort* bp = B_us + k1 * N + col_base;
            if (col_base + 16 <= N) {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = intel_sub_group_block_read_us(bp + (2*p) * N);
                    ushort s1 = intel_sub_group_block_read_us(bp + (2*p+1) * N);
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                }
            } else {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int col = col_base + sg_lid;
                    ushort s0 = (col < N) ? B_us[(k1 + 2*p) * N + col] : (ushort)0;
                    ushort s1 = (col < N) ? B_us[(k1 + 2*p+1) * N + col] : (ushort)0;
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                }
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // Load next A into buffer 0
        int next_k = (kp + 1) * 2 * TILE_K;
        if (next_k < K) {
            barrier(CLK_LOCAL_MEM_FENCE);
            __local ushort* dst = slm_a;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = row_base + ar;
                int gk = next_k + ac;
                dst[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Handle remainder (0 or 1 k-tiles)
    if (k_remainder > 0) {
        int k0 = num_k_pairs * 2 * TILE_K;
        __local const ushort* ca = slm_a;

        short8 a0, a1, a2, a3;
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a0)[r] = intel_sub_group_block_read_us(ca + r * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a1)[r] = intel_sub_group_block_read_us(ca + (8 + r) * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a2)[r] = intel_sub_group_block_read_us(ca + (16 + r) * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a3)[r] = intel_sub_group_block_read_us(ca + (24 + r) * TILE_K);

        int8 b_val;
        __global const ushort* bp = B_us + k0 * N + col_base;
        if (col_base + 16 <= N) {
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int kr0 = k0 + 2*p;
                int kr1 = k0 + 2*p + 1;
                ushort s0 = (kr0 < K) ? intel_sub_group_block_read_us(bp + (2*p) * N) : (ushort)0;
                ushort s1 = (kr1 < K) ? intel_sub_group_block_read_us(bp + (2*p+1) * N) : (ushort)0;
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }
        } else {
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int col = col_base + sg_lid;
                int kr0 = k0 + 2*p;
                int kr1 = k0 + 2*p + 1;
                ushort s0 = (kr0 < K && col < N) ? B_us[kr0 * N + col] : (ushort)0;
                ushort s1 = (kr1 < K && col < N) ? B_us[kr1 * N + col] : (ushort)0;
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
    }

    // ---- Store results ----
    const int col_idx = col_base + sg_lid;
    if (col_idx < N) {
        __global ushort* C_us = (__global ushort*)C;

        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc0)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 8 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc1)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 16 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc2)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 24 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc3)[r]));
        }
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 1.160):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32M x 256N, K-step=16
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16)
// A in SLM (shared across 16 subgroups = 16x reuse), double-buffered
// B loaded directly from global (each SG loads its own 16x16 block)
// GWS = (ceil(N/256)*256, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16
// Key optimization: overlap next-tile SLM writes with DPAS compute

#define TILE_M 32
#define TILE_N 256
#define TILE_K 16
#define SLM_A_SIZE (TILE_M * TILE_K)  // 512 ushorts = 1KB per buffer
#define WG_SIZE 256

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
    const int sg_id  = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid    = get_local_id(0);

    const int n_base   = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;
    const int col_base = n_base + sg_id * 16;

    __local ushort slm_a[2 * SLM_A_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const bool col_in_bounds = (col_base + 16 <= N);

    // Load first A tile into buffer 0
    {
        __local ushort* dst = slm_a;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid >> 4;
            int ac = eid & 15;
            int gr = row_base + ar;
            dst[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int cur_buf = kt & 1;

        __local const ushort* ca = slm_a + cur_buf * SLM_A_SIZE;

        // Read A from SLM into registers
        short8 a0, a1, a2, a3;
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a0)[r] = intel_sub_group_block_read_us(ca + r * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a1)[r] = intel_sub_group_block_read_us(ca + (8 + r) * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a2)[r] = intel_sub_group_block_read_us(ca + (16 + r) * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a3)[r] = intel_sub_group_block_read_us(ca + (24 + r) * TILE_K);

        // Load B from global memory: 16 rows x 16 cols packed as int8
        int8 b_val;
        if (col_in_bounds) {
            __global const ushort* bp = B_us + k_offset * N + col_base;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(bp + (2 * p) * N);
                ushort s1 = intel_sub_group_block_read_us(bp + (2 * p + 1) * N);
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }
        } else {
            int col = col_base + sg_lid;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int kr0 = k_offset + 2 * p;
                int kr1 = kr0 + 1;
                ushort s0 = (kr0 < K && col < N) ? B_us[kr0 * N + col] : (ushort)0;
                ushort s1 = (kr1 < K && col < N) ? B_us[kr1 * N + col] : (ushort)0;
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }
        }

        // Start loading next A tile BEFORE compute (overlap with DPAS)
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - cur_buf;
            const int next_k = (kt + 1) * TILE_K;
            __local ushort* dst = slm_a + next_buf * SLM_A_SIZE;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid >> 4;
                int ac = eid & 15;
                int gr = row_base + ar;
                int gk = next_k + ac;
                dst[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }
        }

        // DPAS compute - overlaps with SLM writes above on Xe2
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    const int col_idx = col_base + sg_lid;
    if (col_idx < N) {
        __global ushort* C_us = (__global ushort*)C;

        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc0)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 8 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc1)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 16 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc2)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 24 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc3)[r]));
        }
    }
}
```

## Best implementation so far:

This is the code of the best kernel we have tested so far (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 4.350):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 128 cols, K-step: 16 (loaded from global per iteration)
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16) tiles
// A tile in SLM: 32 x 16 = 512 ushorts = 1 KB
// B tile in SLM: 16 x 128 = 2048 ushorts = 4 KB
// Single buffer SLM: 5 KB total (very small, fits easily)
// GWS = (ceil(N/128)*128, ceil(M/32))  LWS = (128, 1)
// Subgroup size = 16

#define TILE_M 32
#define TILE_N 128
#define TILE_K 16
#define SLM_A_SIZE (TILE_M * TILE_K)   // 512 ushorts
#define SLM_B_SIZE (TILE_K * TILE_N)   // 2048 ushorts
#define SLM_BUF_SIZE (SLM_A_SIZE + SLM_B_SIZE) // 2560 ushorts
#define WG_SIZE 128

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

    const int n_base   = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;
    const int col_base = n_base + sg_id * 16;

    // Double-buffered SLM
    __local ushort slm[2 * SLM_BUF_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // A load: 32 x 16 = 512 elements, 512/128 = 4 per WI
    // B load: 16 x 128 = 2048 elements, 2048/128 = 16 per WI

    // --- Load first tile into buffer 0 ---
    {
        __local ushort* slm_a = slm;
        __local ushort* slm_b = slm + SLM_A_SIZE;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;  // 0..31
            int ac = eid % TILE_K;  // 0..15
            int gr = row_base + ar;
            ushort val = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
            slm_a[eid] = val;
        }

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int eid = lid + i * WG_SIZE;
            int br = eid / TILE_N;  // 0..15
            int bc = eid % TILE_N;  // 0..127
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
        __local const ushort* my_b = cur_b + sg_id * 16;

        // ---- Single k16 DPAS step ----
        {
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

        // ---- Load next tile ----
        if (has_next) {
            int next_buf = 1 - cur_buf;
            __local ushort* next_a = slm + next_buf * SLM_BUF_SIZE;
            __local ushort* next_b = next_a + SLM_A_SIZE;
            int next_k = next_kt * TILE_K;

            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = row_base + ar;
                int gk = next_k + ac;
                next_a[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * WG_SIZE;
                int br = eid / TILE_N;
                int bc = eid % TILE_N;
                int gk = next_k + br;
                int gc = n_base + bc;
                next_b[eid] = (gk < K && gc < N) ? B_us[gk * N + gc] : (ushort)0;
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

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 11.800):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// =============================================================================
// Optimized FP16 GEMM: C[M,N] = A[M,K] * B[K,N]
// Row-major, half I/O, float accumulation
//
// WG tile: 32 rows x 128 cols,  K-step: 64
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 32 rows x 16 cols (4 stacked 8x16 DPAS blocks)
//
// SLM layout (double-buffered):
//   A tile: 32 x 64 = 2048 halfs, padded stride = 64 (no pad needed, 64 is even)
//   B tile: 64 x 128 = 8192 halfs
//   Per buffer: (2048 + 8192) = 10240 ushorts = 20 KB
//   Total SLM: 2 * 20 KB = 40 KB (within 128 KB limit)
//
// Cooperative loading (coalesced):
//   A: 2048 elems / 128 WIs = 16 per WI
//   B: 8192 elems / 128 WIs = 64 per WI
//
// Compute per K-step: 32 * 128 * 64 * 2 = 524,288 FLOPs
// Load per K-step: (2048 + 8192) * 2 = 20,480 bytes
// Compute intensity: ~25.6 FLOPs/byte
//
// GWS = (ceil(N/128)*128, ceil(M/32))  LWS = (128, 1)
// intel_reqd_sub_group_size(16)
// =============================================================================

#define TILE_M 32
#define TILE_N 128
#define TILE_K 64
#define A_STRIDE 64          // A SLM row stride (= TILE_K, no padding needed)
#define B_STRIDE 128         // B SLM row stride (= TILE_N)
#define SLM_A_TILE (TILE_M * A_STRIDE)   // 2048 ushorts
#define SLM_B_TILE (TILE_K * B_STRIDE)   // 8192 ushorts
#define SLM_BUF_SIZE (SLM_A_TILE + SLM_B_TILE)  // 10240 ushorts

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
    const int sg_id  = get_sub_group_id();        // 0..7
    const int sg_lid = get_sub_group_local_id();  // 0..15
    const int lid    = get_local_id(0);           // 0..127

    const int n_base   = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;

    // Double-buffered SLM
    __local ushort slm[2 * SLM_BUF_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    // Accumulators: 4 DPAS blocks of 8x16 = 32 rows x 16 cols per subgroup
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // ---- Macro for cooperative tile loading (coalesced pattern) ----
    // A: 2048 elements, 128 WIs, 16 per WI. lid + i*128 gives coalesced access.
    // B: 8192 elements, 128 WIs, 64 per WI. lid + i*128 gives coalesced access.

    #define LOAD_TILE(slm_a_ptr, slm_b_ptr, k_off) \
    { \
        _Pragma("unroll") \
        for (int i = 0; i < 16; i++) { \
            int elem = lid + i * 128; \
            int ar = elem / A_STRIDE; \
            int ac = elem % A_STRIDE; \
            int gr = row_base + ar; \
            int gk = (k_off) + ac; \
            (slm_a_ptr)[elem] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0; \
        } \
        _Pragma("unroll") \
        for (int i = 0; i < 64; i++) { \
            int elem = lid + i * 128; \
            int br = elem / B_STRIDE; \
            int bc = elem % B_STRIDE; \
            int gk = (k_off) + br; \
            int gn = n_base + bc; \
            (slm_b_ptr)[elem] = (gk < K && gn < N) ? B_us[gk * N + gn] : (ushort)0; \
        } \
    }

    // ---- Macro for DPAS k16 compute step ----
    #define COMPUTE_K16(a_ptr, b_ptr) \
    { \
        short8 a0, a1, a2, a3; \
        _Pragma("unroll") \
        for (int r = 0; r < 8; r++) \
            ((ushort*)&a0)[r] = intel_sub_group_block_read_us((a_ptr) + r * A_STRIDE); \
        _Pragma("unroll") \
        for (int r = 0; r < 8; r++) \
            ((ushort*)&a1)[r] = intel_sub_group_block_read_us((a_ptr) + (8 + r) * A_STRIDE); \
        _Pragma("unroll") \
        for (int r = 0; r < 8; r++) \
            ((ushort*)&a2)[r] = intel_sub_group_block_read_us((a_ptr) + (16 + r) * A_STRIDE); \
        _Pragma("unroll") \
        for (int r = 0; r < 8; r++) \
            ((ushort*)&a3)[r] = intel_sub_group_block_read_us((a_ptr) + (24 + r) * A_STRIDE); \
        int8 b_val; \
        _Pragma("unroll") \
        for (int p = 0; p < 8; p++) { \
            ushort s0 = intel_sub_group_block_read_us((b_ptr) + (2*p) * B_STRIDE); \
            ushort s1 = intel_sub_group_block_read_us((b_ptr) + (2*p+1) * B_STRIDE); \
            ((int*)&b_val)[p] = as_int((ushort2)(s0, s1)); \
        } \
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0); \
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1); \
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2); \
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3); \
    }

    // ---- Load first tile into buffer 0 ----
    LOAD_TILE(slm, slm + SLM_A_TILE, 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        __local const ushort* cur_a = slm + cur_buf * SLM_BUF_SIZE;
        __local const ushort* cur_b = cur_a + SLM_A_TILE + sg_id * 16;

        // 4 k16 steps within the k64 tile
        COMPUTE_K16(cur_a,      cur_b);
        COMPUTE_K16(cur_a + 16, cur_b + 16 * B_STRIDE);
        COMPUTE_K16(cur_a + 32, cur_b + 32 * B_STRIDE);
        COMPUTE_K16(cur_a + 48, cur_b + 48 * B_STRIDE);

        // ---- Prefetch next tile into alternate buffer ----
        const int next_kt = kt + 1;
        if (next_kt < num_k_tiles) {
            barrier(CLK_LOCAL_MEM_FENCE);

            int next_buf = 1 - cur_buf;
            __local ushort* next_a = slm + next_buf * SLM_BUF_SIZE;
            __local ushort* next_b = next_a + SLM_A_TILE;
            int next_k = next_kt * TILE_K;

            LOAD_TILE(next_a, next_b, next_k);

            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = next_buf;
        }
    }

    // ---- Store results ----
    const int col_idx = n_base + sg_id * 16 + sg_lid;
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
================== 4 passed, 1 deselected, 1 warning in 0.78s ==================
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
1. Exploit Sub-groups (OpenCL 2.0+): Use sub-group functions for wavefront-level operations: sub_group_reduce_add(), sub_group_broadcast(), intel_sub_group_shuffle(), intel_sub_group_shuffle_down(). Query sub-group size with get_sub_group_size(). Use __attribute__((intel_reqd_sub_group_size(N))) for Intel GPUs.
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