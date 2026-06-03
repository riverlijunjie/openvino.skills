

## Analysis

Looking at the previous versions, all achieve ~1.14-1.16ms with 23% XMM utilization. The key bottlenecks:

1. **B matrix loading from global memory**: Each subgroup independently loads B, causing redundant global memory traffic. With 16 subgroups loading the same B tile, there's no reuse.
2. **SLM underutilization**: Only A is cached in SLM (~2KB), but SLM is 128KB per core. We should cache both A and B.
3. **Compute-to-memory ratio**: 4 DPAS per k-step with 32 SLM reads + 16 global reads is memory-bound.
4. **Tile shape**: 32M×256N means each WG does 32×256×K FLOPs but loads 32×K (A) + K×256 (B) elements. The N-dimension is too wide, reducing A reuse.

**Proposed improvements:**
- **Cache both A and B in SLM**: Use a 64M×64N tile with 4 subgroups. Each subgroup computes 32M×16N. A (64×16=2KB) and B (16×64=2KB) per k-step fit easily in SLM.
- **Increase M-tile to 64**: More A reuse per B load. Each SG does 4 DPAS calls for 32 rows; with 2 SG rows we get 64M.
- **Better blocking**: 64M×64N tile, 4 subgroups (each 16 WIs), 64 WIs per WG. Each SG computes 32M×16N = 4 DPAS per k-step.
- **Actually, let's try a different approach**: 32M×128N with 8 subgroups, both A and B in SLM. This gives better reuse and allows prefetching.

Let me try: **Tile 32M×128N, 8 SGs, both A+B in SLM, K-step=16**
- SLM per k-step: A=32×16=1KB, B=16×128=4KB, total=5KB, double-buffered=10KB. Fits easily.
- Each SG: 32 rows × 16 cols = 4 DPAS
- A reuse: 8× (across 8 SGs), B reuse: 1× (each SG has unique cols)... not great.

Better: **Tile 64M×128N, 8 SGs arranged 2×4, both in SLM**
- 2 SG rows × 4 SG cols
- Each SG: 32M×32N = 4 DPAS × 2 (for 32N with two B blocks) = 8 DPAS per k-step
- A reuse: 4×, B reuse: 2×
- SLM: A=64×16=2KB, B=16×128=4KB, double-buffered=12KB

Actually, let me reconsider. The key issue is that **B loads dominate**. With 32M×256N tiles, we load 256×16 = 4096 half values of B per k-step from global, while A is only 32×16 = 512. The compute is 32×256×16 = 131072 FLOPs vs 4608 elements loaded = 28.4 FLOPs/element. That's actually decent.

The real problem might be **too many workgroups competing for bandwidth**. Let me try increasing per-SG work with register blocking - each SG computes multiple N-tiles.

**New approach: 32M×256N, 8 SGs, each SG computes 32M×32N using 2 B-column blocks per k-step (register blocking in N)**

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Tile: 64M x 128N, K-step=16
// 8 subgroups arranged as 2 (M-dim) x 4 (N-dim), 16 WIs each = 128 WIs per WG
// Each subgroup: 32 rows x 32 cols = 4 DPAS x 2 B-columns = 8 DPAS per k-step
// Both A and B in SLM, double-buffered
// SLM per buffer: A = 64*16*2 = 2KB, B = 16*128*2 = 4KB, total = 6KB, x2 = 12KB
// A reuse: 4x (across N-dim SGs), B reuse: 2x (across M-dim SGs)
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)
// Subgroup size = 16

#define TILE_M 64
#define TILE_N 128
#define TILE_K 16
#define SG_ROWS 2
#define SG_COLS 4
#define NUM_SG 8
#define WG_SIZE 128
#define SLM_A_SIZE (TILE_M * TILE_K)       // 1024 ushorts = 2KB
#define SLM_B_SIZE (TILE_K * TILE_N)       // 2048 ushorts = 4KB
#define SLM_BUF_SIZE (SLM_A_SIZE + SLM_B_SIZE)  // 3072 ushorts = 6KB

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

    // SG arrangement: sg_row = sg_id / SG_COLS, sg_col = sg_id % SG_COLS
    const int sg_row = sg_id / SG_COLS;  // 0 or 1
    const int sg_col = sg_id % SG_COLS;  // 0..3

    // Each SG handles 32 M-rows and 32 N-cols (2 x 16-col DPAS blocks)
    const int sg_m_base = row_base + sg_row * 32;
    const int sg_n_base = n_base + sg_col * 32;

    // Double-buffered SLM for both A and B
    __local ushort slm[2 * SLM_BUF_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    // 8 accumulators: 4 row-blocks x 2 col-blocks
    float8 acc00 = 0.0f, acc10 = 0.0f, acc20 = 0.0f, acc30 = 0.0f;
    float8 acc01 = 0.0f, acc11 = 0.0f, acc21 = 0.0f, acc31 = 0.0f;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Helper: load A tile (64 x 16) and B tile (16 x 128) into SLM buffer
    // Total elements: A = 1024, B = 2048, sum = 3072
    // 128 WIs, each loads 3072/128 = 24 elements
    // Split: A = 1024 elems = 8 per WI, B = 2048 elems = 16 per WI

    // Load first tile into buffer 0
    {
        __local ushort* slm_a = slm;
        __local ushort* slm_b = slm + SLM_A_SIZE;

        // Load A: 64*16 = 1024, 1024/128 = 8 per WI
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;   // row in [0,63]
            int ac = eid % TILE_K;   // col in [0,15]
            int gr = row_base + ar;
            slm_a[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }

        // Load B: 16*128 = 2048, 2048/128 = 16 per WI
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int eid = lid + i * WG_SIZE;
            int br = eid / TILE_N;   // row in [0,15]
            int bc = eid % TILE_N;   // col in [0,127]
            int gk = br;
            int gn = n_base + bc;
            slm_b[eid] = (gk < K && gn < N) ? B_us[gk * N + gn] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int cur_buf = kt & 1;

        __local const ushort* ca = slm + cur_buf * SLM_BUF_SIZE;
        __local const ushort* cb = slm + cur_buf * SLM_BUF_SIZE + SLM_A_SIZE;

        // Read A from SLM: 32 rows for this SG's M-portion
        // A is stored as [64][16], our rows start at sg_row*32
        int a_offset = sg_row * 32 * TILE_K;
        short8 a0, a1, a2, a3;
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a0)[r] = intel_sub_group_block_read_us(ca + a_offset + r * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a1)[r] = intel_sub_group_block_read_us(ca + a_offset + (8 + r) * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a2)[r] = intel_sub_group_block_read_us(ca + a_offset + (16 + r) * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a3)[r] = intel_sub_group_block_read_us(ca + a_offset + (24 + r) * TILE_K);

        // Read B from SLM: two 16x16 blocks for sg_col's two 16-col portions
        // B is stored as [16][128], our cols start at sg_col*32
        int b_offset0 = sg_col * 32;       // first 16 cols
        int b_offset1 = sg_col * 32 + 16;  // second 16 cols

        int8 b0, b1;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            ushort s0 = intel_sub_group_block_read_us(cb + (2*p) * TILE_N + b_offset0);
            ushort s1 = intel_sub_group_block_read_us(cb + (2*p+1) * TILE_N + b_offset0);
            ((int*)&b0)[p] = as_int((ushort2)(s0, s1));
        }
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            ushort s0 = intel_sub_group_block_read_us(cb + (2*p) * TILE_N + b_offset1);
            ushort s1 = intel_sub_group_block_read_us(cb + (2*p+1) * TILE_N + b_offset1);
            ((int*)&b1)[p] = as_int((ushort2)(s0, s1));
        }

        // Prefetch next tile into other buffer
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - cur_buf;
            const int next_k = (kt + 1) * TILE_K;
            __local ushort* na = slm + next_buf * SLM_BUF_SIZE;
            __local ushort* nb = slm + next_buf * SLM_BUF_SIZE + SLM_A_SIZE;

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = row_base + ar;
                int gk = next_k + ac;
                na[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * WG_SIZE;
                int br = eid / TILE_N;
                int bc = eid % TILE_N;
                int gk = next_k + br;
                int gn = n_base + bc;
                nb[eid] = (gk < K && gn < N) ? B_us[gk * N + gn] : (ushort)0;
            }
        }

        // DPAS compute: 4 row-blocks x 2 col-blocks = 8 DPAS
        acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
        acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
        acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc20);
        acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc30);
        acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
        acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
        acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc21);
        acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc31);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    __global ushort* C_us = (__global ushort*)C;

    // First 16-col block
    {
        int col_idx = sg_n_base + sg_lid;
        if (col_idx < N) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = sg_m_base + r;
                if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc00)[r]));
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = sg_m_base + 8 + r;
                if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc10)[r]));
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = sg_m_base + 16 + r;
                if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc20)[r]));
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = sg_m_base + 24 + r;
                if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc30)[r]));
            }
        }
    }

    // Second 16-col block
    {
        int col_idx = sg_n_base + 16 + sg_lid;
        if (col_idx < N) {
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = sg_m_base + r;
                if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc01)[r]));
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = sg_m_base + 8 + r;
                if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc11)[r]));
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = sg_m_base + 16 + r;
                if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc21)[r]));
            }
            #pragma unroll
            for (int r = 0; r < 8; r++) {
                int gr = sg_m_base + 24 + r;
                if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc31)[r]));
            }
        }
    }
}
```

