

## Analysis

The best kernel so far (1.010ms) uses:
- SLM stride of 32 (no padding), `ushort` type for SLM
- `intel_sub_group_block_read_us` for SLM A reads
- Single K-loop iteration (no 2x unrolling)
- `col_valid` branch inside the inner loop

The slower version (1.230ms) used:
- SLM stride of 34 (padding), `half` type with `as_short` conversions
- Scalar SLM reads
- 2x K-loop unrolling with remainder path
- More code bloat from unrolling + remainder

Key observations:
1. The best version's simplicity (single K-loop, no unrolling) won — likely better instruction cache usage
2. `intel_sub_group_block_read_us` helps with SLM reads
3. The `col_valid` check inside the inner loop adds a branch per iteration — we can hoist it by clamping `col_idx`
4. B reads are the bottleneck — 16 scalar reads per k16 step, each from different rows

**Proposed improvements over best (1.010ms):**
1. Remove `col_valid` branch from inner loop — clamp b_col like Version 1 did, zero-mask only at store
2. Precompute B row offsets to reduce address arithmetic
3. Use `ushort` consistently to avoid type conversion overhead
4. Try converting stores to use `intel_sub_group_block_write_us` for vectorized writes

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (2KB), B loaded directly from global/L2
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)
// K must be divisible by 32 (K=2048, no remainder)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32

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

    // SLM for A tile: 32 rows x 32 cols = 2KB (ushort)
    __local ushort slm_A[TILE_M * TILE_K];

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
    const int num_k_tiles = K >> 5;

    // Clamp col for B reads to avoid OOB — invalid lanes read garbage but results discarded at store
    const int b_col = col_valid ? col_idx : (N - 1);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    // Precompute A tile base for row_tile_valid path
    const int a_row_base_K = row_base * K;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k = kt * TILE_K;

        // Cooperative load A into SLM: 1024 elems, 64 WIs => 16 each
        if (row_tile_valid) {
            __global const ushort* A_tile = A_us + a_row_base_K + k;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                slm_A[elem_id] = A_tile[r * K + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                int gr = row_base + r;
                slm_A[elem_id] = (gr < M) ? A_us[gr * K + k + c] : (ushort)0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Two k16 DPAS steps per k32 tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load A from SLM using vectorized subgroup block reads
            short8 a0, a1, a2, a3;
            __local const ushort* slm_base = slm_A + kk;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a0)[r] = intel_sub_group_block_read_us(slm_base + r * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a1)[r] = intel_sub_group_block_read_us(slm_base + (8 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a2)[r] = intel_sub_group_block_read_us(slm_base + (16 + r) * TILE_K);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((ushort*)&a3)[r] = intel_sub_group_block_read_us(slm_base + (24 + r) * TILE_K);

            // Load B from global — no branch, clamped col, paired reads packed into int
            const int gk_base = k + kk;
            // Precompute base offset for B
            int b_base_offset = gk_base * N + b_col;
            int8 b_val;

            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = B_us[b_base_offset];
                ushort s1 = B_us[b_base_offset + N];
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                b_base_offset += (N << 1);  // advance by 2 rows
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results — only valid columns write
    if (col_valid) {
        __global half* C_col = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

