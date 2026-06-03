

## Analysis

Looking at the previous versions, all achieve ~1.14-1.16ms. The key bottlenecks:

1. **Too many barriers per K-iteration** - Version 3 has 2 barriers per K-step pair, stalling the pipeline
2. **SLM A reads are scattered** - Reading A with `intel_sub_group_block_read_us` row by row (32 reads per SG per k-tile) is expensive
3. **B is loaded from global per-subgroup without reuse** - Each of 16 SGs loads its own B tile, no sharing
4. **Low compute density** - 4 DPAS per k16 step with lots of load overhead

**Strategy for improvement:**
- **Increase tile size in M** to 64 rows → 8 DPAS calls per k-step per SG, doubling compute density
- **Use both A and B in SLM** - B gets 4x reuse across M-dimension subgroups (4 SGs share same B column tile)
- **Arrange SGs as 4×4 grid**: 4 along M (each handles 16 rows = 2 DPAS), 4 along N (each handles 16 cols)
- **TILE_N = 64, TILE_M = 64, TILE_K = 32** with double-buffered SLM
- **Prefetch next tile overlapped with compute**

Wait - with only 16 SGs and wanting high reuse, let me reconsider. The key insight: with 32M×256N tiles, A gets 16x reuse (good) but B gets only 1x reuse (bad). Let me try **64M×128N** with 8 SGs along N and 2 along M (or similar), giving A 8x reuse and B 2x reuse, with both in SLM.

Actually, let me try a different approach: **increase per-subgroup work** by having each SG compute more rows (register blocking). Use 32M×128N with 8 SGs, each SG computes 32 rows × 16 cols = 4 DPAS. Put both A (32×16=1KB) and B (16×128=4KB) in SLM. B gets 2x reuse minimum. Total SLM = 2*(1KB+4KB) = 10KB double-buffered.

Let me try **64M × 128N** with 8 SGs in N-direction, each SG doing 64 rows (8 DPAS per k-step). A in SLM = 64×16×2 = 2KB, B in SLM = 16×128×2 = 4KB. Double-buffered = 12KB. 8 SGs × 16 WIs = 128 WIs per WG. Each SG does 8 DPAS per k16.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Tile: 64M x 128N, K-step=16
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 64 rows x 16 cols = 8 DPAS(8x16) per k-step
// Both A and B in SLM for reuse:
//   A: 64x16 = 1024 ushorts = 2KB (shared across 8 SGs = 8x reuse)
//   B: 16x128 = 2048 ushorts = 4KB (shared across 1 SG in N... no reuse)
// Actually: rearrange as 2 SGs in M x 4 SGs in N
//   Each SG: 32 rows x 32 cols (4 DPAS x 2 N-blocks = 8 DPAS per k-step)
//   A: 64x16 = 2KB (4x reuse along N)
//   B: 16x128 = 4KB (2x reuse along M)
// Double-buffered: 2*(2KB+4KB) = 12KB << 128KB SLM
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)
// Subgroup size = 16

#define TILE_M 64
#define TILE_N 128
#define TILE_K 16
#define SG_M 32
#define SG_N 32
#define SG_ROWS_M 2
#define SG_COLS_N 4
#define SLM_A_SIZE (TILE_M * TILE_K)    // 1024 ushorts
#define SLM_B_SIZE (TILE_K * TILE_N)    // 2048 ushorts
#define SLM_TILE_SIZE (SLM_A_SIZE + SLM_B_SIZE)  // 3072 ushorts
#define WG_SIZE 128
#define NUM_SG 8

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

    // SG grid: 2 rows x 4 cols
    const int sg_row = sg_id / SG_COLS_N;  // 0 or 1
    const int sg_col = sg_id % SG_COLS_N;  // 0..3

    const int n_base   = get_group_id(0) * TILE_N;
    const int m_base   = get_group_id(1) * TILE_M;

    // Each SG handles 32 rows starting at sg_row*32, and 32 cols at sg_col*32
    // But DPAS produces 8 rows x 16 cols, so 32 cols = 2 DPAS horizontally
    // 32 rows = 4 DPAS vertically, total 8 DPAS per SG per k-step
    const int sg_m_base = m_base + sg_row * SG_M;
    const int sg_n_base = n_base + sg_col * SG_N;

    __local ushort slm[2 * SLM_TILE_SIZE];

    if (m_base >= M || n_base >= N)
        return;

    // Accumulators: 4 row-blocks x 2 col-blocks = 8 float8
    float8 acc00 = 0.0f, acc10 = 0.0f, acc20 = 0.0f, acc30 = 0.0f;
    float8 acc01 = 0.0f, acc11 = 0.0f, acc21 = 0.0f, acc31 = 0.0f;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Helper: load A and B into SLM buffer
    // A: 64 x 16 = 1024 elements, B: 16 x 128 = 2048 elements
    // Total: 3072 elements / 128 WIs = 24 elements per WI

    #define LOAD_AB_TO_SLM(buf_offset, k_off) \
    { \
        __local ushort* dst_a = slm + (buf_offset) + 0; \
        __local ushort* dst_b = slm + (buf_offset) + SLM_A_SIZE; \
        /* Load A: 1024 / 128 = 8 per WI */ \
        _Pragma("unroll") \
        for (int i = 0; i < 8; i++) { \
            int eid = lid + i * WG_SIZE; \
            int ar = eid / TILE_K; \
            int ac = eid % TILE_K; \
            int gr = m_base + ar; \
            int gk = (k_off) + ac; \
            dst_a[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0; \
        } \
        /* Load B: 2048 / 128 = 16 per WI */ \
        _Pragma("unroll") \
        for (int i = 0; i < 16; i++) { \
            int eid = lid + i * WG_SIZE; \
            int br = eid / TILE_N; \
            int bc = eid % TILE_N; \
            int gk = (k_off) + br; \
            int gn = n_base + bc; \
            dst_b[eid] = (gk < K && gn < N) ? B_us[gk * N + gn] : (ushort)0; \
        } \
    }

    // Load first tile
    LOAD_AB_TO_SLM(0, 0)
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int cur_buf = (kt & 1) * SLM_TILE_SIZE;

        __local const ushort* ca = slm + cur_buf;
        __local const ushort* cb = slm + cur_buf + SLM_A_SIZE;

        // Read A from SLM: 32 rows at sg_row*32, 16 cols
        // A layout in SLM: row-major, 64 rows x 16 cols, stride = TILE_K = 16
        short8 a0, a1, a2, a3;
        {
            __local const ushort* ap = ca + sg_row * SG_M * TILE_K;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(ap + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(ap + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(ap + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(ap + (24 + r) * TILE_K);
        }

        // Read B from SLM: 2 blocks of 16x16
        // B layout: row-major, 16 rows x 128 cols, stride = TILE_N = 128
        // Block 0: cols [sg_col*32 .. sg_col*32+15]
        // Block 1: cols [sg_col*32+16 .. sg_col*32+31]
        int8 b0, b1;
        {
            __local const ushort* bp0 = cb + sg_col * SG_N;
            __local const ushort* bp1 = cb + sg_col * SG_N + 16;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(bp0 + (2*p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(bp0 + (2*p+1) * TILE_N);
                ((int*)&b0)[p] = as_int((ushort2)(s0, s1));
            }
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(bp1 + (2*p) * TILE_N);
                ushort s1 = intel_sub_group_block_read_us(bp1 + (2*p+1) * TILE_N);
                ((int*)&b1)[p] = as_int((ushort2)(s0, s1));
            }
        }

        // Prefetch next A/B tile into other buffer
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kt + 1 < num_k_tiles) {
            int next_k = (kt + 1) * TILE_K;
            int next_buf = ((kt + 1) & 1) * SLM_TILE_SIZE;
            LOAD_AB_TO_SLM(next_buf, next_k)
        }

        // 8 DPAS operations: 4 row-blocks x 2 col-blocks
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

    // Store results
    __global ushort* C_us = (__global ushort*)C;

    #define STORE_ACC(acc_var, row_off, col_off) \
    { \
        int col_idx = sg_n_base + (col_off) + sg_lid; \
        if (col_idx < N) { \
            _Pragma("unroll") \
            for (int r = 0; r < 8; r++) { \
                int gr = sg_m_base + (row_off) + r; \
                if (gr < M) \
                    C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc_var)[r])); \
            } \
        } \
    }

    STORE_ACC(acc00,  0,  0)
    STORE_ACC(acc01,  0, 16)
    STORE_ACC(acc10,  8,  0)
    STORE_ACC(acc11,  8, 16)
    STORE_ACC(acc20, 16,  0)
    STORE_ACC(acc21, 16, 16)
    STORE_ACC(acc30, 24,  0)
    STORE_ACC(acc31, 24, 16)

    #undef STORE_ACC
    #undef LOAD_AB_TO_SLM
}
```

