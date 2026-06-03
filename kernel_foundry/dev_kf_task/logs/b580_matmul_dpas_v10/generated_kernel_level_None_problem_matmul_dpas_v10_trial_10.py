

## Analysis

Looking at the results:
- **Version 2** (1.080ms) is the fastest, using SLM for A with a simple loop structure
- **Best so far** (1.260ms) and **Previous** (1.280ms) use double-buffering but are slightly slower, likely due to increased SLM usage (2x buffers) reducing occupancy

Key observations:
1. Double-buffering doubled SLM usage but didn't help because the next A-tile load overlaps with B loads from global memory anyway
2. Version 2's simpler approach with single SLM buffer performs better - less register pressure, better occupancy
3. B is loaded from global memory with strided access (row-major B, each row accessed at different K iterations) - this is bandwidth-intensive
4. We should also cache B in SLM to reduce global memory traffic
5. On Battlemage with 20 Xe-cores, we want high occupancy

**Proposed improvements over Version 2:**
1. Cache both A and B in SLM to reduce global memory bandwidth pressure
2. Use single SLM buffer (better occupancy than double-buffer)
3. Use `intel_sub_group_block_read` for coalesced SLM reads where possible
4. Keep the fast-path / boundary-check split from Version 2
5. Increase TILE_K to 32 to amortize barrier cost

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 64 cols, K-step: 32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// Both A and B cached in SLM to reduce global memory traffic
// A SLM: 32 x 34 half (padded stride to avoid bank conflicts)
// B SLM: 32 x 66 half (padded stride to avoid bank conflicts)
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_A_STRIDE (TILE_K + 2)   // 34, pad to avoid bank conflicts
#define SLM_B_STRIDE (TILE_N + 2)   // 66, pad to avoid bank conflicts
#define SLM_A_SIZE (TILE_M * SLM_A_STRIDE)
#define SLM_B_SIZE (TILE_K * SLM_B_STRIDE)

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

    const int col_base = get_group_id(0) * TILE_N + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    __local half slm_A[SLM_A_SIZE];  // 32 x 34
    __local half slm_B[SLM_B_SIZE];  // 32 x 66

    if (row_base >= M || col_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const int num_full_k_tiles = K / TILE_K;
    const int k_remainder = K - num_full_k_tiles * TILE_K;

    // Each WI loads multiple elements for A and B cooperatively
    // A: 32x32 = 1024 elements, 64 WIs => 16 elements each
    // B: 32x64 = 2048 elements, 64 WIs => 32 elements each

    for (int kt = 0; kt < num_full_k_tiles; kt++) {
        const int k = kt * TILE_K;

        // ---- Cooperative load A[row_base:row_base+32, k:k+32] into SLM ----
        if (row_tile_valid) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;   // elem_id / 32
                int c = elem_id & 31;   // elem_id % 32
                slm_A[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + k + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                int gr = row_base + r;
                slm_A[r * SLM_A_STRIDE + c] = (gr < M) ? A[gr * K + k + c] : (half)0.0h;
            }
        }

        // ---- Cooperative load B[k:k+32, col_group_base:col_group_base+64] into SLM ----
        // B tile: 32 rows x 64 cols = 2048 elements, 64 WIs => 32 elements each
        {
            const int n_base = get_group_id(0) * TILE_N;
            const bool n_tile_valid = (n_base + TILE_N <= N);

            if (n_tile_valid) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 6;   // elem_id / 64
                    int c = elem_id & 63;   // elem_id % 64
                    slm_B[r * SLM_B_STRIDE + c] = B[(k + r) * N + n_base + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int elem_id = lid + i * 64;
                    int r = elem_id >> 6;
                    int c = elem_id & 63;
                    int gc = n_base + c;
                    slm_B[r * SLM_B_STRIDE + c] = (gc < N) ? B[(k + r) * N + gc] : (half)0.0h;
                }
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

            // Load B from SLM: 16 K-rows x 16 cols for this subgroup
            // sg_id determines which 16-col slice
            int b_col_off = sg_id * 16;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_row0 = kk + 2 * p;
                int k_row1 = k_row0 + 1;
                short s0 = as_short(slm_B[k_row0 * SLM_B_STRIDE + b_col_off + sg_lid]);
                short s1 = as_short(slm_B[k_row1 * SLM_B_STRIDE + b_col_off + sg_lid]);
                ((int*)&b_val)[p] = as_int((short2)(s0, s1));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Handle remainder K tile
    if (k_remainder > 0) {
        const int k = num_full_k_tiles * TILE_K;
        const int n_base = get_group_id(0) * TILE_N;

        // Load A remainder
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int elem_id = lid + i * 64;
            int r = elem_id >> 5;
            int c = elem_id & 31;
            int gr = row_base + r;
            int gk = k + c;
            slm_A[r * SLM_A_STRIDE + c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // Load B remainder
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int elem_id = lid + i * 64;
            int r = elem_id >> 6;
            int c = elem_id & 63;
            int gk = k + r;
            int gc = n_base + c;
            slm_B[r * SLM_B_STRIDE + c] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0.0h;
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

            int b_col_off = sg_id * 16;
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_row0 = kk + 2 * p;
                int k_row1 = k_row0 + 1;
                short s0 = as_short(slm_B[k_row0 * SLM_B_STRIDE + b_col_off + sg_lid]);
                short s1 = as_short(slm_B[k_row1 * SLM_B_STRIDE + b_col_off + sg_lid]);
                ((int*)&b_val)[p] = as_int((short2)(s0, s1));
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

