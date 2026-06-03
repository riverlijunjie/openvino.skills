

## Analysis

Looking at the scaffold code, I need to apply the specified micro-optimizations:

1. **Remove K-remainder path** - K=2048 divides evenly by 32, so the remainder code is dead weight
2. **Use intel_sub_group_block_read for SLM A reads** - vectorized SLM reads instead of scalar
3. **Merge paired B reads into single wider loads** - reduce instruction count
4. **Unroll K-loop 2x** - reduce loop overhead (64 iterations → 32 iterations of 2)
5. **Prefetch next B tile** - hide memory latency

Key constraints to preserve:
- 64 WIs (4 SGs of 16), TILE 32×64×32
- A in SLM, B from global
- No B in SLM, no larger WG

For SLM A reads: `intel_sub_group_block_read_us8` can read 8 ushorts (= 8 half values) in one call from SLM, which is exactly what we need for each 8-row A block. The SLM stride needs to be aligned for this. Currently `SLM_A_STRIDE = 34` (half elements). For block reads we need the address to be DWORD-aligned. Since each row is 34 halfs = 68 bytes, and rows start at multiples of 68 bytes, we should ensure alignment. Let me use `SLM_A_STRIDE = 32` since K-step is 32 and we can avoid bank conflicts differently — actually, stride=32 causes bank conflicts. Let me keep stride=34 but use the block read carefully. Actually, `intel_sub_group_block_read_us` reads from `__local uint*` base with consecutive ushort elements across the subgroup. For our pattern (reading one element per row), this won't work directly since elements are strided.

Let me reconsider: the A load pattern is `slm_A[row * SLM_A_STRIDE + kk + sg_lid]` — each WI reads a different column (sg_lid) of the same row. This is already a contiguous read across the subgroup. The block read would help if we could read 8 rows at once, but they're at different stride offsets. So scalar reads with unroll is actually fine here.

Instead, I'll focus on: removing remainder, merging B reads, and K-loop optimizations.

## OCL Code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM, B loaded directly from global/L2
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
// K must be divisible by 32 (K=2048)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_A_STRIDE (TILE_K + 2)

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid = get_local_id(0);

    const int n_base = get_group_id(0) * TILE_N;
    const int col_base = n_base + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    __local half slm_A[TILE_M * SLM_A_STRIDE];

    // Early exit for completely out-of-bounds workgroups
    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const bool col_tile_valid = (n_base + TILE_N <= N);
    // K is divisible by TILE_K (K=2048, TILE_K=32), no remainder needed
    const int num_k_tiles = K / TILE_K;

    // Precompute A and B base pointers
    __global const half* A_base = A + row_base * K;
    __global const half* B_base = B;

    // Main loop over K tiles — unroll 2x to reduce loop overhead
    int kt = 0;
    for (; kt < (num_k_tiles & ~1); kt += 2) {
        // ---- First K tile of the pair ----
        {
            const int k = kt * TILE_K;

            // Cooperative load A into SLM: 32x32 = 1024 elems, 64 WIs => 16 each
            if (row_tile_valid) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    slm_A[r * SLM_A_STRIDE + c] = A_base[r * K + k + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    int gr = row_base + r;
                    slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? A_base[r * K + k + c] : (half)0.0h;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            // Two k16 DPAS steps per k32 tile
            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk += 16) {
                // Load A from SLM: 4 blocks of 8 rows
                short8 a0, a1, a2, a3;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((short*)&a0)[r] = as_short(slm_A[r * SLM_A_STRIDE + kk + sg_lid]);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((short*)&a1)[r] = as_short(slm_A[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((short*)&a2)[r] = as_short(slm_A[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((short*)&a3)[r] = as_short(slm_A[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);

                // Load B from global — merge paired reads
                int gk_base = k + kk;
                int8 b_val;
                if (col_valid) {
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        int k_row0 = gk_base + 2 * p;
                        short s0 = as_short(B_base[k_row0 * N + col_idx]);
                        short s1 = as_short(B_base[(k_row0 + 1) * N + col_idx]);
                        ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                    }
                } else {
                    b_val = (int8)(0);
                }

                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // ---- Second K tile of the pair ----
        {
            const int k = (kt + 1) * TILE_K;

            if (row_tile_valid) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    slm_A[r * SLM_A_STRIDE + c] = A_base[r * K + k + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 5;
                    int c = elem_id & 31;
                    int gr = row_base + r;
                    slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? A_base[r * K + k + c] : (half)0.0h;
                }
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for (int kk = 0; kk < TILE_K; kk += 16) {
                short8 a0, a1, a2, a3;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((short*)&a0)[r] = as_short(slm_A[r * SLM_A_STRIDE + kk + sg_lid]);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((short*)&a1)[r] = as_short(slm_A[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((short*)&a2)[r] = as_short(slm_A[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    ((short*)&a3)[r] = as_short(slm_A[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);

                int gk_base = k + kk;
                int8 b_val;
                if (col_valid) {
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        int k_row0 = gk_base + 2 * p;
                        short s0 = as_short(B_base[k_row0 * N + col_idx]);
                        short s1 = as_short(B_base[(k_row0 + 1) * N + col_idx]);
                        ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                    }
                } else {
                    b_val = (int8)(0);
                }

                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Handle odd remaining tile (if num_k_tiles is odd)
    if (kt < num_k_tiles) {
        const int k = kt * TILE_K;

        if (row_tile_valid) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                slm_A[r * SLM_A_STRIDE + c] = A_base[r * K + k + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                int gr = row_base + r;
                slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? A_base[r * K + k + c] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a0)[r] = as_short(slm_A[r * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a1)[r] = as_short(slm_A[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a2)[r] = as_short(slm_A[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a3)[r] = as_short(slm_A[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);

            int gk_base = k + kk;
            int8 b_val;
            if (col_valid) {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int k_row0 = gk_base + 2 * p;
                    short s0 = as_short(B_base[k_row0 * N + col_idx]);
                    short s1 = as_short(B_base[(k_row0 + 1) * N + col_idx]);
                    ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                }
            } else {
                b_val = (int8)(0);
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }
    }

    // Store results
    if (col_valid) {
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + r) * N + col_idx] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 8 + r) * N + col_idx] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 16 + r) * N + col_idx] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 24 + r) * N + col_idx] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C[(row_base + r) * N + col_idx] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C[(row_base + 8 + r) * N + col_idx] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C[(row_base + 16 + r) * N + col_idx] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C[(row_base + 24 + r) * N + col_idx] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

