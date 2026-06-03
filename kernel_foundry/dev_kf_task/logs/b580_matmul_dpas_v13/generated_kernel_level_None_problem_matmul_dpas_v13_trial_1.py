

## Analysis

Looking at the previous versions:

1. **Version 1** (5.130ms) and **Version 3** (4.730ms) are correct but slow - ~23% XMM utilization
2. **Version 2** (incorrect) tried 64x64 tiles but had bugs in the first tile load (used `ac < K` instead of `0 + ac < K` for first tile)

Key bottlenecks in the working versions:
- **Poor memory coalescing during SLM loads**: Using `lid * 8 + i` or `lid * 32 + i` indexing means adjacent work-items access non-contiguous memory. Should use `lid + i * 128` for coalesced global reads.
- **B matrix load dominates**: 32×128 = 4096 elements, 32 loads per WI is heavy. The large B tile causes register pressure and long load times.
- **Insufficient compute density per SLM load**: Only 2 DPAS k16 steps per K-tile load. Need larger K-tile or more compute per load.
- **Store pattern**: Scalar stores are inefficient.

**Proposed improvements:**
1. **Use 64×64 tile with 2×4 subgroup layout** - reduces B load (32×64 vs 32×128), better balance
2. **Increase TILE_K to 64** for more compute per SLM load (4 k16 steps), better arithmetic intensity
3. **Coalesced loads**: Use strided `lid + i*WG_SIZE` pattern for global memory coalescing
4. **Fix Version 2's bug**: The first tile load needs `k_off=0` properly handled
5. **SLM: 2*(64*64 + 64*64)*2 = 2*16KB = 32KB** - fits well within 128KB
6. **Each subgroup: 32 rows × 16 cols = 4 DPAS blocks × 4 k16 steps = 16 DPAS per K-tile**

Actually, let me reconsider. The working version 1 gets 5.13ms, version 3 gets 4.73ms. The main difference is load pattern (`lid + i*128` vs `lid*8 + i`). Version 1 uses coalesced pattern but is slower - possibly due to SLM bank conflicts from the interleaved storage.

Let me try a bigger approach: **128×128 tile, 16 subgroups (256 WIs), TILE_K=32**, with 4×4 subgroup grid. Each SG does 32×32 = 4×2 = 8 DPAS blocks. But 256 WIs might be too many.

Actually, let me stick with a proven approach but optimize better: **64×128 tile, 16 subgroups, TILE_K=32**. 4×8 SG layout, each SG: 16 rows × 16 cols = 2 DPAS blocks. That's too few DPAS per SG.

Better: **64×128 tile, 8 subgroups (128 WIs), 2×4 SG layout**. Each SG: 32×32 = 4 DPAS height × 2 DPAS width = 8 accumulators. With K=32 (2 k16 steps): 16 DPAS per iteration. This gives much better compute density.

SLM: 64×32 + 32×128 = 2048 + 4096 = 6144 ushorts = 12KB per buffer, 24KB total.

Let me implement this:

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

