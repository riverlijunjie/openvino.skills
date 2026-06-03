

## Analysis

Looking at the previous versions, all achieve ~1.14-1.16ms with 23% XMM utilization. Key issues:

1. **Too many barriers**: The double-buffered A loading has barriers that serialize compute and memory access instead of overlapping them.
2. **SLM reads are inefficient**: Reading A from SLM uses `intel_sub_group_block_read_us` row by row (32 reads per subgroup per k-tile), which is slow.
3. **B is not shared via SLM**: Each of 16 subgroups independently loads B from global memory - no reuse across the M dimension.
4. **Tile shape**: 32M x 256N means 16 subgroups each handle 16 columns of B. But B has no reuse across subgroups. A better approach: put BOTH A and B in SLM, so both get reused.
5. **Compute-to-memory ratio**: With 32x256x16 = 131K FLOPs per k-tile, we load 32×16=512 A values (shared) + 16×16×16=4096 B values (not shared, 16 SGs each load 16×16). Total global loads = 512+4096 = 4608 half values = 9216 bytes for 131K FLOPs = 14.2 FLOPs/byte. Need higher.

**Strategy for improvement:**
- Use a **64M × 128N** tile with **8 subgroups** (each SG handles 8M×16N = 1 DPAS, but we do 8 rows of DPAS per SG).
- Actually, let me try: **32M × 64N** tile, 4 subgroups, both A and B in SLM. This gives 32×16 + 64×16 = 1536 values per k-tile for 32×64×32 = 65K FLOPs.
- Better: **64M × 64N** with 4 SGs, each SG does 64M×16N. Load A(64×16) and B(16×64) into SLM. 64×16+64×16=2048 values = 4KB for 64×64×32 = 131K FLOPs. Reuse = 32 FLOPs/byte.
- Even better: Increase tile to **128M × 128N** with 8 SGs (each 128M×16N), k=16. A=128×16, B=16×128 in SLM. 4096+4096=8192 values=16KB. FLOPs=128×128×32=524K. 32 FLOPs/byte. But 128 rows of DPAS per SG = 16 DPAS calls = lots of registers.

Let me try a different approach: **Each subgroup computes multiple N-columns** to increase register reuse of A. Specifically, each SG computes 32M × 32N (2 columns of 16), with 4 DPAS per column × 2 columns = 8 DPAS per k16 step. 8 SGs → 32M × 256N tile. A is shared (32×16 in SLM), B is shared too (16×256 in SLM). A=1KB, B=8KB, total=9KB per buffer × 2 = 18KB. Fits in 128KB SLM.

Actually let me try a cleaner approach: **Both A and B in SLM, larger K-step, and proper prefetching with async copy pattern.**

```
Tile: 32M × 256N, K=32
A in SLM: 32×32 = 2KB per buffer
B in SLM: 32×256 = 16KB per buffer  
Total SLM: 2 × (2+16) = 36KB - fits in 128KB

8 subgroups, each SG computes 32M × 32N = 4 DPAS × 2 k16 rounds × 2 N-blocks = could work
Wait, 256/16 = 16 SG columns needed. Let's keep 16 SGs.

16 SGs, each 32M × 16N. Both A(32×32) and B(32×256) in SLM.
A reuse: 16×, B reuse: 2× (from k=32 vs k=16)
SLM per buffer: 32*32*2 + 32*256*2 = 2KB + 16KB = 18KB
Double buffered: 36KB. Fits.
```

Let me try a fundamentally different approach - put B in SLM too and use cooperative loading.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Optimized GEMM: C[M,N] = A[M,K] * B[K,N], all FP16, FP32 accumulation
//
// Tile: 32M x 256N, K-step=32 (two k16 DPAS rounds per SLM load)
// Both A and B stored in SLM for maximum reuse
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32M x 16N = 4 DPAS(8x16) per k16 round
// SLM usage: A: 32*32*2=2KB, B: 32*256*2=16KB, per buffer = 18KB
// Double-buffered: 36KB total SLM (fits in 128KB)
// A reuse: 16x across subgroups (N dimension)
// B reuse: loaded once cooperatively, used by all SGs needing same K strip
//
// GWS = (ceil(N/256)*256, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16

#define TILE_M 32
#define TILE_N 256
#define TILE_K 32
#define K16 16
#define SLM_A_STRIDE TILE_K
#define SLM_A_SIZE (TILE_M * TILE_K)        // 1024 ushorts = 2KB
#define SLM_B_STRIDE TILE_N
#define SLM_B_SIZE (TILE_K * TILE_N)        // 8192 ushorts = 16KB
#define SLM_BUF_SIZE (SLM_A_SIZE + SLM_B_SIZE)  // 9216 ushorts = 18KB
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

    // Helper macro for SLM buffer offsets
    // Buffer layout: [A: TILE_M * TILE_K][B: TILE_K * TILE_N]

    // Cooperative load of A (32x32 = 1024 elements) and B (32x256 = 8192 elements)
    // Total = 9216 elements, 256 WIs -> 36 elements per WI
    // A: 1024/256 = 4 per WI, B: 8192/256 = 32 per WI
    // Load A: first 4 iterations, Load B: next 32 iterations
    // Better: interleave. WI lid loads:
    //   A: 4 elements, B: 32 elements = 36 total

    // Load first tile into buffer 0
    {
        __local ushort* dst_a = slm;
        __local ushort* dst_b = slm + SLM_A_SIZE;

        // Load A: 32*32 = 1024 elements, 4 per WI
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int eid = lid + i * WG_SIZE;
            int ar = eid / TILE_K;
            int ac = eid % TILE_K;
            int gr = row_base + ar;
            dst_a[eid] = (gr < M && ac < K) ? A_us[gr * K + ac] : (ushort)0;
        }

        // Load B: 32*256 = 8192 elements, 32 per WI
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int eid = lid + i * WG_SIZE;
            int br = eid / TILE_N;
            int bc = eid % TILE_N;
            int gk = br;
            int gn = n_base + bc;
            dst_b[eid] = (gk < K && gn < N) ? B_us[gk * N + gn] : (ushort)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int cur_buf = kt & 1;

        __local const ushort* ca = slm + cur_buf * SLM_BUF_SIZE;
        __local const ushort* cb = slm + cur_buf * SLM_BUF_SIZE + SLM_A_SIZE;

        // Two k16 rounds per k32 tile
        #pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            int k16_off = ks * K16;

            // Read A from SLM: 32 rows x 16 cols
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(ca + r * SLM_A_STRIDE + k16_off);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(ca + (8 + r) * SLM_A_STRIDE + k16_off);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(ca + (16 + r) * SLM_A_STRIDE + k16_off);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(ca + (24 + r) * SLM_A_STRIDE + k16_off);

            // Read B from SLM: 16 rows x 16 cols (for this subgroup's column)
            // B in SLM is stored as [TILE_K][TILE_N], we need [k16_off..k16_off+15][col_base-n_base..+15]
            int b_col_off = sg_id * 16;  // column offset within tile
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(cb + (k16_off + 2 * p) * SLM_B_STRIDE + b_col_off);
                ushort s1 = intel_sub_group_block_read_us(cb + (k16_off + 2 * p + 1) * SLM_B_STRIDE + b_col_off);
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }

            // DPAS compute
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // Prefetch next tile into alternate buffer
        if (kt + 1 < num_k_tiles) {
            barrier(CLK_LOCAL_MEM_FENCE);
            const int next_buf = 1 - cur_buf;
            const int next_k = (kt + 1) * TILE_K;
            __local ushort* dst_a = slm + next_buf * SLM_BUF_SIZE;
            __local ushort* dst_b = slm + next_buf * SLM_BUF_SIZE + SLM_A_SIZE;

            // Load A: 4 per WI
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int eid = lid + i * WG_SIZE;
                int ar = eid / TILE_K;
                int ac = eid % TILE_K;
                int gr = row_base + ar;
                int gk = next_k + ac;
                dst_a[eid] = (gr < M && gk < K) ? A_us[gr * K + gk] : (ushort)0;
            }

            // Load B: 32 per WI
            #pragma unroll
            for (int i = 0; i < 32; i++) {
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

