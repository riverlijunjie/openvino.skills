

## Analysis

Looking at the previous versions, all achieve ~1.14-1.16ms with 23% XMM utilization. Key issues:

1. **Too many barriers**: Each k-tile requires 2 barriers for A loading, serializing compute and memory
2. **SLM reads are inefficient**: Reading A from SLM row-by-row with `intel_sub_group_block_read_us` one element at a time (8 reads per 8x16 A block = 32 reads total)
3. **B loaded redundantly**: Each of 16 subgroups loads B independently - no reuse
4. **Low compute density**: 4 DPAS per k16 step with heavy memory overhead between them
5. **Tile shape**: 32M×256N means each WG covers a wide but short tile. With B580's 20 Xe-cores, we need enough WGs but also enough compute per WG

**Strategy for improvement:**
- **Both A and B in SLM** for maximum reuse - A reused across N-dimension, B reused across M-dimension
- **Larger per-subgroup tile**: Each subgroup computes 32M×32N (4×2=8 DPAS per k16) instead of 32M×16N (4 DPAS)
- **Fewer subgroups, more work each**: 8 subgroups × 16 WIs = 128 WIs per WG
- **WG tile: 32M×256N** with 8 subgroups each handling 32M×32N
- **Prefetch B into SLM** to amortize global memory reads
- **K-step=32** with double-buffering to hide latency

Let me try a different approach: **cooperative loading of both A and B into SLM**, with each subgroup computing a larger tile.

Tile: 64M × 128N, 8 subgroups arranged 4×2 (M×N), each SG does 16M×64N = 2×4=8 DPAS per k16.

Actually, let me keep it simpler but focus on the key bottleneck: **B memory bandwidth**. With 16 subgroups each loading their own B tile, we're loading B 16× more than needed per WG. Loading B into SLM gives 16× reuse.

New design:
- WG tile: 32M × 256N, K-step=16
- Both A (32×16) and B (16×256) in SLM  
- SLM: 32×16×2 + 16×256×2 = 1KB + 8KB = 9KB per buffer, double-buffered = 18KB (fits 128KB)
- 16 subgroups, each does 32M×16N = 4 DPAS per k16

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32M x 256N, K-step=16, double-buffered
// Both A and B loaded cooperatively into SLM for maximum reuse
// A: 32x16 = 512 ushorts (1KB), B: 16x256 = 4096 ushorts (8KB), per buffer
// Double-buffered: 2*(1KB+8KB) = 18KB SLM total
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32M x 16N = 4 DPAS(8x16) per k-step
// GWS = (ceil(N/256)*256, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16

#define TILE_M 32
#define TILE_N 256
#define TILE_K 16
#define SLM_A_ELEMS (TILE_M * TILE_K)    // 512
#define SLM_B_ELEMS (TILE_K * TILE_N)    // 4096
#define SLM_BUF_SIZE (SLM_A_ELEMS + SLM_B_ELEMS)  // 4608
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

    // Double-buffered SLM for A and B
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

    // Cooperative load of A (512 elems) and B (4096 elems) = 4608 total
    // 4608 / 256 = 18 elements per WI
    // Split: first 512 = A, next 4096 = B
    // Use simpler scheme: each WI loads ~18 elements total

    #define LOAD_TILE(buf_idx, k_off) \
    { \
        __local ushort* slm_a = slm + (buf_idx) * SLM_BUF_SIZE; \
        __local ushort* slm_b = slm_a + SLM_A_ELEMS; \
        /* Load A: 512 elements, 2 per WI */ \
        if (lid < 256) { \
            int eid0 = lid * 2; \
            int eid1 = eid0 + 1; \
            int ar0 = eid0 / TILE_K; int ac0 = eid0 % TILE_K; \
            int ar1 = eid1 / TILE_K; int ac1 = eid1 % TILE_K; \
            int gr0 = row_base + ar0; int gk0 = (k_off) + ac0; \
            int gr1 = row_base + ar1; int gk1 = (k_off) + ac1; \
            slm_a[eid0] = (gr0 < M && gk0 < K) ? A_us[gr0 * K + gk0] : (ushort)0; \
            slm_a[eid1] = (gr1 < M && gk1 < K) ? A_us[gr1 * K + gk1] : (ushort)0; \
        } \
        /* Load B: 4096 elements, 16 per WI */ \
        for (int i = 0; i < 16; i++) { \
            int eid = lid + i * WG_SIZE; \
            int br = eid / TILE_N; int bc = eid % TILE_N; \
            int gk = (k_off) + br; int gc = n_base + bc; \
            slm_b[eid] = (gk < K && gc < N) ? B_us[gk * N + gc] : (ushort)0; \
        } \
    }

    // Load first tile
    LOAD_TILE(0, 0);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int cur_buf = kt & 1;
        __local const ushort* ca = slm + cur_buf * SLM_BUF_SIZE;
        __local const ushort* cb = ca + SLM_A_ELEMS;

        // Read A from SLM
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

        // Read B from SLM: 16 rows x 16 cols for this subgroup
        // B in SLM is 16 x 256, subgroup sg_id reads columns [sg_id*16 .. sg_id*16+15]
        int8 b_val;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            ushort s0 = intel_sub_group_block_read_us(cb + (2 * p) * TILE_N + sg_id * 16);
            ushort s1 = intel_sub_group_block_read_us(cb + (2 * p + 1) * TILE_N + sg_id * 16);
            ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
        }

        // Prefetch next tile while computing
        if (kt + 1 < num_k_tiles) {
            barrier(CLK_LOCAL_MEM_FENCE);
            LOAD_TILE(1 - cur_buf, (kt + 1) * TILE_K);
        }

        // DPAS compute
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

