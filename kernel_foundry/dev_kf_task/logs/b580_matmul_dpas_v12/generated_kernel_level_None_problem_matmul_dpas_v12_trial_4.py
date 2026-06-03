

## Analysis

Looking at the two versions:
- **Version 1** (0.970ms): 2x K-loop unrolling with single SLM buffer, reducing barriers from 64 to 32
- **Version 2** (0.985ms): Double-buffered SLM, but 2x SLM usage and still one barrier per tile

Version 1 wins because the 2x unroll genuinely halves barrier count while keeping SLM small. Double buffering in V2 doesn't help because the extra SLM pressure and the load-before-barrier pattern doesn't effectively overlap compute with memory.

**Key bottleneck analysis for V1 (0.970ms):**
1. B loads are scalar (16 individual ushort reads per k16 step) — these could be more efficient
2. The K-remainder path is dead code for K=2048 but adds code size / instruction cache pressure
3. The store path does scalar writes — could use block writes

**Plan for improvement:**
1. Remove K-remainder path (K=2048 divides by 64, so 2x unrolled loop handles everything)
2. Keep the 2x K-loop unrolling from V1 (proven best at 0.970ms)
3. Use `vload2` for paired B reads to potentially get better memory coalescing
4. Minimize redundant computations in the inner loop
5. Use `intel_sub_group_block_write_us` for output stores where possible

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (2KB), B loaded directly from global/L2
// K-loop unrolled 2x: 32 barriers instead of 64 (K=2048 divides evenly by 64)
// No K-remainder path needed.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

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

    __local ushort slm_A[TILE_M * TILE_K];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);

    // K=2048 divides by 64, so num_k_pairs = 32, no remainder
    const int num_k_pairs = K >> 6;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int b_col = col_valid ? col_idx : (N - 1);
    const int a_row_base_K = row_base * K;
    const int N2 = N << 1;

    // Main K-loop: unrolled 2x
    for (int kp = 0; kp < num_k_pairs; kp++) {
        // Process two K-tiles per iteration
        #pragma unroll 2
        for (int sub = 0; sub < 2; sub++) {
            const int k = (kp * 2 + sub) * TILE_K;

            // Cooperative load A into SLM
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

                // Load B from global — paired reads packed into int
                int b_off = (k + kk) * N + b_col;
                int8 b_val;

                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort s0 = B_us[b_off];
                    ushort s1 = B_us[b_off + N];
                    ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
                    b_off += N2;
                }

                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Store results
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

