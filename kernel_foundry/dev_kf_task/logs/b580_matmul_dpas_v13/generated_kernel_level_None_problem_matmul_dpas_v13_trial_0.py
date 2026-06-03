

## Analysis

Looking at the current kernel, I can identify several bottlenecks:

1. **B matrix loads are scalar** - each work-item loads individual elements from global memory, causing poor memory coalescing
2. **A matrix SLM layout** - the current layout stores A in row-major with stride TILE_K=32, but DPAS needs k16 chunks, requiring careful addressing
3. **Insufficient compute-to-memory ratio** - with 32x64 tile and K-step=32, we do 32×64×32×2 = 131K FLOPs per tile step, but load 32×32×2 + 32×64×2 = 6KB, giving ~21 FLOP/byte
4. **Double buffering placement** - barrier is inside the compute path, stalling DPAS

**Key improvements:**
1. **Cache B in SLM too** - use cooperative loading with block reads for coalesced access
2. **Larger K-step (64)** to increase compute density and amortize SLM loads
3. **Use `intel_sub_group_block_read` for B loads** - subgroup-coalesced reads from global memory
4. **Better tile: 64x64 with 8 subgroups** - more compute per WG, better L2 reuse
5. **Prefetch B into SLM** for uniform access patterns

After analysis, I'll use a **32x128 WG tile** with **8 subgroups**, each handling 32×16 output. This gives 4× more N-work per WG, improving A reuse from SLM. Both A and B go through SLM with double buffering.

Actually, let me be more pragmatic and focus on the key wins while keeping the structure manageable:

- **Tile: 32M × 128N**, 8 subgroups (each 32×16), K-step=32
- **Both A and B in SLM** with cooperative loading using block reads
- **Double buffer both A and B**
- SLM usage: 2 × (32×32 + 32×128) × 2 bytes = 2 × (1024 + 4096) × 2 = 20,480 bytes — fits in 128KB

Let me reconsider. 128 WIs = 8 subgroups × 16 WIs. SLM for A: 32×32×2=2KB, B: 32×128×2=8KB, double buffered = 20KB. Fine.

Each subgroup computes 32rows × 16cols = 4 DPAS tiles vertically × 2 k16 steps = 8 DPAS calls per k-step.

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

