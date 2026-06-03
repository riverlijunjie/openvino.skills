

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
Here is the last OCL kernel we tested (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// =============================================================================
// Cooperative GEMM: Both A and B tiles cached in SLM
// Tile: 64M x 128N, K-step=32 (two k16 DPAS rounds per SLM load)
// WG: 16 subgroups x 16 WIs = 256 WIs
// SG layout: 4 along M x 4 along N
//   Each SG: 16M x 32N = 2 DPAS(8x16) x 2 N-blocks = 4 DPAS per k16, 8 per k32
// SLM usage: A = 64*32*2 = 4KB, B = 32*128*2 = 8KB => 12KB per buffer, x2 = 24KB
// A reused 4x across N-dim SGs, B reused 4x across M-dim SGs
// GWS = (ceil(N/128)*256, ceil(M/64))  LWS = (256, 1)
// Subgroup size = 16
// =============================================================================

#define TILE_M 64
#define TILE_N 128
#define TILE_K 32
#define K16 16
#define SLM_A_SIZE (TILE_M * TILE_K)   // 2048 ushorts = 4KB
#define SLM_B_SIZE (TILE_K * TILE_N)   // 4096 ushorts = 8KB
#define SLM_TILE_SIZE (SLM_A_SIZE + SLM_B_SIZE) // 6144 ushorts = 12KB
#define WG_SIZE 256
#define NUM_SG 16
#define SG_M 4   // subgroups along M
#define SG_N 4   // subgroups along N

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

    // SG position in the 4x4 grid
    const int sg_row = sg_id / SG_N;  // 0..3
    const int sg_col = sg_id % SG_N;  // 0..3

    const int n_base   = get_group_id(0) * TILE_N;
    const int m_base   = get_group_id(1) * TILE_M;

    // Each SG handles 16 rows x 32 cols of C
    const int sg_m_base = m_base + sg_row * 16;
    const int sg_n_base = n_base + sg_col * 32;

    // Double-buffered SLM for A and B
    __local ushort slm[2 * SLM_TILE_SIZE];

    if (m_base >= M || n_base >= N)
        return;

    // Accumulators: 2 DPAS blocks along M (rows 0-7, 8-15) x 2 along N (cols 0-15, 16-31)
    float8 acc00 = 0.0f;  // rows 0-7, cols 0-15
    float8 acc10 = 0.0f;  // rows 8-15, cols 0-15
    float8 acc01 = 0.0f;  // rows 0-7, cols 16-31
    float8 acc11 = 0.0f;  // rows 8-15, cols 16-31

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Total elements to load per tile:
    // A: 64*32 = 2048 elements, 2048/256 = 8 per WI
    // B: 32*128 = 4096 elements, 4096/256 = 16 per WI
    // Total: 24 per WI

    // Load first tile into buffer 0
    {
        __local ushort* dst_a = slm;
        __local ushort* dst_b = slm + SLM_A_SIZE;

        // Load A: 64x32, each WI loads 8 elements
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;    // row in tile (0..63)
            int ac = eid % TILE_K;    // col in tile (0..31)
            int gr = m_base + ar;
            dst_a[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }

        // Load B: 32x128, each WI loads 16 elements
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int eid = lid + i * WG_SIZE;
            int br = eid / TILE_N;    // row in tile (0..31)
            int bc = eid % TILE_N;    // col in tile (0..127)
            int gk = br;
            int gn = n_base + bc;
            dst_b[eid] = (gk < K && gn < N) ? B_us[gk * N + gn] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int cur_buf = kt & 1;

        __local const ushort* ca = slm + cur_buf * SLM_TILE_SIZE;
        __local const ushort* cb = slm + cur_buf * SLM_TILE_SIZE + SLM_A_SIZE;

        // Process two k16 steps
        #pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            int k16_off = ks * K16;

            // Load A sub-tile from SLM: 16 rows for this SG, k16 columns
            // A layout in SLM: [TILE_M][TILE_K] = [64][32]
            // We need rows [sg_row*16 .. sg_row*16+15], cols [k16_off .. k16_off+15]
            short8 a0, a1;
            int a_row_base = sg_row * 16;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(
                    ca + (a_row_base + r) * TILE_K + k16_off);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(
                    ca + (a_row_base + 8 + r) * TILE_K + k16_off);

            // Load B sub-tile from SLM: k16 rows, 32 cols for this SG
            // B layout in SLM: [TILE_K][TILE_N] = [32][128]
            // We need rows [k16_off .. k16_off+15], cols [sg_col*32 .. sg_col*32+31]
            // Two 16-wide B blocks (cols 0-15 and cols 16-31 of this SG's region)
            int8 b0, b1;
            int b_col_base0 = sg_col * 32;
            int b_col_base1 = sg_col * 32 + 16;

            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int brow = k16_off + 2 * p;
                ushort s0 = intel_sub_group_block_read_us(cb + brow * TILE_N + b_col_base0);
                ushort s1 = intel_sub_group_block_read_us(cb + (brow + 1) * TILE_N + b_col_base0);
                ((int*)&b0)[p] = as_int((ushort2)(s0, s1));
            }
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int brow = k16_off + 2 * p;
                ushort s0 = intel_sub_group_block_read_us(cb + brow * TILE_N + b_col_base1);
                ushort s1 = intel_sub_group_block_read_us(cb + (brow + 1) * TILE_N + b_col_base1);
                ((int*)&b1)[p] = as_int((ushort2)(s0, s1));
            }

            // 4 DPAS operations per k16 step
            acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
            acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
            acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
            acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
        }

        // Load next tile into other buffer
        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - cur_buf;
            const int next_k = (kt + 1) * TILE_K;

            barrier(CLK_LOCAL_MEM_FENCE);

            __local ushort* dst_a = slm + next_buf * SLM_TILE_SIZE;
            __local ushort* dst_b = slm + next_buf * SLM_TILE_SIZE + SLM_A_SIZE;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = m_base + ar;
                int gk = next_k + ac;
                dst_a[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * WG_SIZE;
                int br = eid / TILE_N;
                int bc = eid % TILE_N;
                int gk = next_k + br;
                int gn = n_base + bc;
                dst_b[eid] = (gk < K && gn < N) ? B_us[gk * N + gn] : (ushort)0;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Store results
    __global ushort* C_us = (__global ushort*)C;

    // acc00: rows [sg_m_base..+7], cols [sg_n_base..+15]
    // acc10: rows [sg_m_base+8..+15], cols [sg_n_base..+15]
    // acc01: rows [sg_m_base..+7], cols [sg_n_base+16..+31]
    // acc11: rows [sg_m_base+8..+15], cols [sg_n_base+16..+31]

    int col0 = sg_n_base + sg_lid;
    int col1 = sg_n_base + 16 + sg_lid;

    if (col0 < N) {
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = sg_m_base + r;
            if (gr < M) C_us[gr * N + col0] = as_ushort(convert_half(((float*)&acc00)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = sg_m_base + 8 + r;
            if (gr < M) C_us[gr * N + col0] = as_ushort(convert_half(((float*)&acc10)[r]));
        }
    }
    if (col1 < N) {
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = sg_m_base + r;
            if (gr < M) C_us[gr * N + col1] = as_ushort(convert_half(((float*)&acc01)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = sg_m_base + 8 + r;
            if (gr < M) C_us[gr * N + col1] = as_ushort(convert_half(((float*)&acc11)[r]));
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

self = <task.TestMatmulOCL object at 0x741f27e72510>
kernel = <function initialize_matmul_kernel.<locals>.<lambda> at 0x741ecbea3060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x741ecbea98f0>, _run = 0

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
E        +  where False = <function allclose at 0x741f2455c170>(array([[-12.4375   , -85.25     ,  45.875    , ...,   0.       ,\n          0.       ,   0.       ],\n       [-82.9375   ,  28.328125 ,   4.3085938, ...,   0.       ,\n          0.       ,   0.       ],\n       [ 31.09375  , -51.78125  ,  -9.3046875, ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [-65.3125   , -27.734375 ,  74.1875   , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 44.6875   ,   2.3144531,  22.609375 , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 50.40625  ,  -3.015625 ,  21.546875 , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[-12.434087 , -85.22102  ,  45.86866  , ..., -67.074715 ,\n        -64.52674  ,  37.798523 ],\n       [-82.95244  ,  28.332115 ,   4.3084497, ...,  37.17192  ,\n         48.87541  ,  55.1519   ],\n       [ 31.096529 , -51.77693  ,  -9.3054905, ...,   8.124319 ,\n         61.21928  ,   4.7092314],\n       ...,\n       [-65.29967  , -27.73106  ,  74.195465 , ..., 122.09403  ,\n        -41.569603 ,  10.711429 ],\n       [ 44.6838   ,   2.3142765,  22.61605  , ..., -35.807106 ,\n         42.793472 ,  52.60636  ],\n       [ 50.399834 ,  -3.015791 ,  21.545517 , ..., -21.399685 ,\n        -36.035267 ,  49.01544  ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x741f2455c170> = np.allclose

task.py:99: AssertionError
---------------------------- Captured stdout setup -----------------------------
[WARN] matmul_opt.cl has no parsable reqd_work_group_size; fallback launch (N, M), local=None
________________ TestMatmulOCL.test_correctness_wrt_pytorch[1] _________________

self = <task.TestMatmulOCL object at 0x741ecbe4ea80>
kernel = <function initialize_matmul_kernel.<locals>.<lambda> at 0x741ecbea3060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x741ecbea98f0>, _run = 1

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
E        +  where False = <function allclose at 0x741f2455c170>(array([[ 63.21875  ,  13.6640625,  62.71875  , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 30.34375  ,  -9.578125 , -15.8515625, ...,   0.       ,\n          0.       ,   0.       ],\n       [ -9.828125 ,  26.84375  , -39.875    , ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [-50.9375   , -10.71875  , -18.65625  , ...,   0.       ,\n          0.       ,   0.       ],\n       [-41.46875  ,  -5.0351562,  35.5      , ...,   0.       ,\n          0.       ,   0.       ],\n       [-33.90625  ,  51.46875  ,  13.109375 , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[  63.220627 ,   13.663691 ,   62.708282 , ...,   26.950535 ,\n        -100.14888  ,  -76.10468  ],\n       [  30.338015 ,   -9.576593 ,  -15.848044 , ...,  -86.66203  ,\n           6.3691177,    9.569207 ],\n       [  -9.825886 ,   26.83852  ,  -39.88768  , ...,   94.32298  ,\n         -40.437588 ,   13.349518 ],\n       ...,\n       [ -50.946926 ,  -10.7210655,  -18.652342 , ...,   -4.0612535,\n         -29.112085 ,   -2.7683525],\n       [ -41.46417  ,   -5.034666 ,   35.500336 , ...,    3.5289268,\n          14.26104  ,   55.58531  ],\n       [ -33.896618 ,   51.45737  ,   13.108513 , ...,   11.92079  ,\n         -64.022385 ,   63.048595 ]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x741f2455c170> = np.allclose

task.py:99: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[0] ________________

self = <task.TestMatmulOCL object at 0x741ecbe96fc0>
kernel = <function initialize_matmul_kernel.<locals>.<lambda> at 0x741ecbea3060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x741ecbea98f0>, _run = 0

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
E        +  where False = <function allclose at 0x741f2455c170>(array([[ -57.125    ,  -28.015625 ,   65.5625   , ...,    0.       ,\n           0.       ,    0.       ],\n       [  92.6875   ,   18.34375  ,  -14.34375  , ...,    0.       ,\n           0.       ,    0.       ],\n       [  -2.5585938,  -30.015625 ,   49.9375   , ...,    0.       ,\n           0.       ,    0.       ],\n       ...,\n       [ -46.03125  , -142.625    ,  -61.5      , ...,    0.       ,\n           0.       ,    0.       ],\n       [  31.859375 ,  -23.484375 ,   42.75     , ...,    0.       ,\n           0.       ,    0.       ],\n       [ -38.96875  ,   91.75     ,   23.953125 , ...,    0.       ,\n           0.       ,    0.       ]], shape=(2048, 2048), dtype=float32), array([[-5.7125000e+01, -2.8015625e+01,  6.5562500e+01, ...,\n        -5.0218750e+01,  2.7000000e+01,  2.6109375e+01],\n       [ 9.2687500e+01,  1.8343750e+01, -1.4343750e+01, ...,\n        -4.0843750e+01, -2.5828125e+01, -3.1914062e+00],\n       [-2.5585938e+00, -3.0015625e+01,  4.9937500e+01, ...,\n        -5.7156250e+01, -9.2250000e+01,  1.2921875e+01],\n       ...,\n       [-4.6031250e+01, -1.4262500e+02, -6.1500000e+01, ...,\n        -1.2194824e-01,  6.6445312e+00, -5.3156250e+01],\n       [ 3.1859375e+01, -2.3484375e+01,  4.2750000e+01, ...,\n         1.8390625e+01, -2.0507812e+00, -1.6000000e+02],\n       [-3.8968750e+01,  9.1750000e+01,  2.3953125e+01, ...,\n         4.5437500e+01,  5.4437500e+01,  1.0787500e+02]],\n      shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x741f2455c170> = np.allclose

task.py:117: AssertionError
_______________ TestMatmulOCL.test_correctness_wrt_reference[1] ________________

self = <task.TestMatmulOCL object at 0x741ecbe970e0>
kernel = <function initialize_matmul_kernel.<locals>.<lambda> at 0x741ecbea3060>
ocl_queue = <pyopencl._cl.CommandQueue object at 0x741ecbea98f0>, _run = 1

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
E        +  where False = <function allclose at 0x741f2455c170>(array([[ 46.4375   , -21.75     , -36.71875  , ...,   0.       ,\n          0.       ,   0.       ],\n       [-13.859375 , -17.453125 ,  17.375    , ...,   0.       ,\n          0.       ,   0.       ],\n       [ 34.1875   ,  -9.421875 , -49.       , ...,   0.       ,\n          0.       ,   0.       ],\n       ...,\n       [  3.3261719, -60.03125  , -64.       , ...,   0.       ,\n          0.       ,   0.       ],\n       [ -0.9145508,  97.25     , -18.109375 , ...,   0.       ,\n          0.       ,   0.       ],\n       [-36.5      , -63.8125   ,  68.125    , ...,   0.       ,\n          0.       ,   0.       ]], shape=(2048, 2048), dtype=float32), array([[  46.4375   ,  -21.75     ,  -36.71875  , ...,  -39.96875  ,\n           3.125    , -147.25     ],\n       [ -13.859375 ,  -17.453125 ,   17.375    , ...,   57.90625  ,\n          25.859375 ,  -68.0625   ],\n       [  34.1875   ,   -9.421875 ,  -49.       , ...,  -46.71875  ,\n         -64.5625   ,   37.6875   ],\n       ...,\n       [   3.3261719,  -60.03125  ,  -64.       , ...,   18.328125 ,\n         -17.234375 ,  -30.90625  ],\n       [  -0.9145508,   97.25     ,  -18.109375 , ...,   35.75     ,\n         -45.0625   ,   21.78125  ],\n       [ -36.5      ,  -63.8125   ,   68.125    , ...,  -48.09375  ,\n         -29.515625 ,   11.1640625]], shape=(2048, 2048), dtype=float32), rtol=0.02, atol=0.02)
E        +    where <function allclose at 0x741f2455c170> = np.allclose

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
================== 4 failed, 1 deselected, 1 warning in 0.76s ==================

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

- **Register Blocking**: Each work-item computes a THREAD_M×THREAD_N output block in private register arrays. Use `#pragma unroll` on inner loops. Combine with SLM tiling for multi-level memory hierarchy optimization.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.