

## Analysis

Looking at the previous versions, all achieve ~1.14-1.16ms. The key bottleneck is that with a 32M×256N tile, each workgroup has 16 subgroups each computing 32×16 output, but the B matrix load is not shared through SLM - each of the 16 subgroups loads its own 16×16 B tile independently from global memory. Since A is only 32×16 per k-step (small), the real bandwidth pressure is on B.

**Key issues:**
1. **B not in SLM**: B is loaded 16 times (once per SG) from global memory. Putting B in SLM would enable reuse across the M-dimension subgroups if we reorganize the tiling.
2. **Tile shape**: 32M×256N means only 16x reuse for A but 1x for B. For a more balanced approach, we should increase M-tile to get B reuse too, or put both A and B in SLM.
3. **Barrier overhead**: Too many barriers per k-iteration.

**New approach**: Use a larger tile with both A and B in SLM. Tile: 64M × 128N, K-step=32. This gives 128KB SLM budget check: 64×32×2 + 128×32×2 = 4KB + 8KB = 12KB per buffer, 24KB double-buffered - fits in 128KB SLM.

Each workgroup: 8 subgroups (128 WIs). Each subgroup computes 8M × 128N = 16 DPAS per k16 step (8 row-blocks × 2 = but that's too many). Let me reconsider.

**Better approach**: 32M × 128N tile, 8 subgroups. Each SG handles 32M × 16N = 4 DPAS per k16. A in SLM: 32×16=1KB, B in SLM: 16×128=4KB. Both double-buffered = 10KB. 8 SGs share both A (8× reuse) and B (2× reuse for 32 rows). Not great.

Let me try: **64M × 256N, 32 subgroups (512 WIs)**. Each SG: 8M×16N. A: 64×16=2KB, B: 16×256=8KB. Double-buffered: 20KB. This gives 32× A reuse and 4× B reuse. 32 DPAS per k16 step across 32 SGs = 1 DPAS/SG.

Actually, let me try a different blocking: each subgroup computes more output by handling multiple N-columns. **32M × 256N, 8 subgroups, each SG computes 32M × 32N = 8 DPAS per k16** (4 row-blocks × 2 col-blocks). Both A and B in SLM. A: 32×16=1KB, B: 16×256=8KB, double-buffered = 18KB. B reuse = 2× (32 rows share B). Each SG does 8 DPAS = good utilization.

Let me go with putting **both A and B in SLM** with the existing 32×256 tile but having each subgroup compute 32M×32N (2 B columns), reducing to 8 subgroups (128 WIs). This doubles compute per SG and shares B across row blocks.

Actually, the simplest high-impact change: **put B in SLM too**. Keep 32M×256N tile, 16 SGs, but load B cooperatively into SLM once per k-step instead of each SG loading independently.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Tile: 32M x 256N, K-step=16
// Both A and B in SLM (cooperative load, maximum reuse)
// A SLM: 32x16 = 512 ushorts = 1KB per buffer
// B SLM: 16x256 = 4096 ushorts = 8KB per buffer
// Double-buffered: 2*(1+8) = 18KB << 128KB SLM limit
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16) per k16
// A reuse: 16x (shared across 16 SGs), B reuse: 2x (shared across 32/16=2 row-groups... actually all SGs share same B)
// Actually B reuse = 4x because 32 rows / 8 rows per DPAS = 4 row-groups all use same B column
// GWS = (ceil(N/256)*256, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16
// Total elements to load per k-step: 512 + 4096 = 4608, per WI = 18 => 9 uint loads

#define TILE_M 32
#define TILE_N 256
#define TILE_K 16
#define SLM_A_SIZE (TILE_M * TILE_K)       // 512 ushorts
#define SLM_B_SIZE (TILE_K * TILE_N)       // 4096 ushorts
#define SLM_BUF_SIZE (SLM_A_SIZE + SLM_B_SIZE)  // 4608 ushorts per buffer
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
    __local ushort slm_a[2 * SLM_A_SIZE];
    __local ushort slm_b[2 * SLM_B_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Load first A tile into buffer 0: 512 elements, 2 per WI
    // Load first B tile into buffer 0: 4096 elements, 16 per WI
    {
        __local ushort* da = slm_a;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;
            int ac = eid % TILE_K;
            int gr = row_base + ar;
            da[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }

        __local ushort* db = slm_b;
        // 4096 / 256 = 16 elements per WI
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int eid = lid + i * WG_SIZE;
            int br = eid / TILE_N;   // row in B tile [0..15]
            int bc = eid % TILE_N;   // col in B tile [0..255]
            int gk = br;
            int gc = n_base + bc;
            db[eid] = (gk < K && gc < N) ? B_us[gk * N + gc] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int cur_buf = kt & 1;

        __local const ushort* ca = slm_a + cur_buf * SLM_A_SIZE;
        __local const ushort* cb = slm_b + cur_buf * SLM_B_SIZE;

        // Read A from SLM: 4 blocks of 8 rows
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
        // B in SLM is stored as [TILE_K][TILE_N], we need cols [sg_id*16 .. sg_id*16+15]
        int8 b_val;
        __local const ushort* bp = cb + sg_id * 16;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            ushort s0 = intel_sub_group_block_read_us(bp + (2 * p) * TILE_N);
            ushort s1 = intel_sub_group_block_read_us(bp + (2 * p + 1) * TILE_N);
            ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
        }

        // Start loading next tile into other buffer BEFORE compute
        // This overlaps memory operations with DPAS on Xe2
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - cur_buf;
            const int next_k = (kt + 1) * TILE_K;

            __local ushort* da = slm_a + next_buf * SLM_A_SIZE;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = row_base + ar;
                int gk = next_k + ac;
                da[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }

            __local ushort* db = slm_b + next_buf * SLM_B_SIZE;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * WG_SIZE;
                int br = eid / TILE_N;
                int bc = eid % TILE_N;
                int gk = next_k + br;
                int gc = n_base + bc;
                db[eid] = (gk < K && gc < N) ? B_us[gk * N + gc] : (ushort)0;
            }
        }

        // DPAS compute - overlaps with SLM writes on Xe2
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

