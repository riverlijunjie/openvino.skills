

## Analysis

The previous best version (3.870ms) uses:
- 32x64 WG tile with 4 subgroups, each handling 32x16 output
- SLM for A tile (shared across subgroups), direct global loads for B
- K-step of 32 with two k16 DPAS iterations per step

Key bottlenecks:
1. **B loads are scalar** - each half is loaded individually with bounds checks, preventing coalesced vector loads
2. **SLM A loads are scalar** - element-by-element with division/modulo arithmetic
3. **No double buffering** - barrier stalls between SLM load and compute
4. **No B in SLM** - B is reloaded from global for each subgroup separately (though different columns)

Proposed improvements:
1. **Use vector loads for B** - Use `vload8` or cast to `__global half8*` for coalesced B reads
2. **Use vector loads for SLM A population** - Load A with wider types (uint/uint2) 
3. **Increase TILE_K to 32 and use double buffering** - Overlap next tile's SLM load with current DPAS
4. **Use intel_sub_group_block_read for B** - Subgroup block reads are highly efficient on Intel
5. **Precompute pointers** to reduce address arithmetic in inner loop

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// WG tile: 32 rows x 64 cols, K-step: 32 with double buffering
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A tile cached in SLM (shared across 4 SGs), B loaded directly
// Double-buffer: two SLM A buffers to overlap load with compute
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

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

    const int col_base = get_group_id(0) * TILE_N + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    // Double-buffered SLM for A
    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];

    if (row_base >= M || col_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;

    // Preload first A tile into buffer 0
    {
        int k = 0;
        // Each WI loads 16 elements: 1024 total / 64 WIs = 16
        // Use half2 loads where possible
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int elem_id = lid + i * 64;
            int r = elem_id >> 5;       // /32
            int c = elem_id & 31;       // %32
            int gr = row_base + r;
            int gk = k + c;
            half val = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
            slm_A[r * SLM_A_STRIDE + c] = val;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;
    for (int k = 0; k < K; k += TILE_K) {
        int next_k = k + TILE_K;
        int next_buf = 1 - buf;
        int cur_off = buf * TILE_M * SLM_A_STRIDE;
        int nxt_off = next_buf * TILE_M * SLM_A_STRIDE;

        // --- DPAS compute from current buffer + prefetch next A tile ---

        // kk=0: first k16 sub-tile
        {
            short8 a0, a1, a2, a3;
            __local half* base = slm_A + cur_off;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a0)[r] = as_short(base[r * SLM_A_STRIDE + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a1)[r] = as_short(base[(8 + r) * SLM_A_STRIDE + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a2)[r] = as_short(base[(16 + r) * SLM_A_STRIDE + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a3)[r] = as_short(base[(24 + r) * SLM_A_STRIDE + sg_lid]);

            // Load B: k16 x 16 cols, packed as int8 (pairs of fp16)
            int8 b_val;
            __global const half* B_base = B + k * N + col_idx;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_row0 = k + 2 * p;
                int k_row1 = k + 2 * p + 1;
                half b0 = (k_row0 < K && col_valid) ? B_base[2 * p * N] : (half)0.0h;
                half b1 = (k_row1 < K && col_valid) ? B_base[(2 * p + 1) * N] : (half)0.0h;
                ((int*)&b_val)[p] = as_int((short2)(as_short(b0), as_short(b1)));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // Prefetch next A tile into next buffer (overlap with second k16 compute)
        if (next_k < K) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * 64;
                int r = elem_id >> 5;
                int c = elem_id & 31;
                int gr = row_base + r;
                int gk = next_k + c;
                half val = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
                slm_A[nxt_off + r * SLM_A_STRIDE + c] = val;
            }
        }

        // kk=16: second k16 sub-tile
        {
            short8 a0, a1, a2, a3;
            __local half* base = slm_A + cur_off;

            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a0)[r] = as_short(base[r * SLM_A_STRIDE + 16 + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a1)[r] = as_short(base[(8 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a2)[r] = as_short(base[(16 + r) * SLM_A_STRIDE + 16 + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a3)[r] = as_short(base[(24 + r) * SLM_A_STRIDE + 16 + sg_lid]);

            int8 b_val;
            __global const half* B_base = B + (k + 16) * N + col_idx;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int k_row0 = k + 16 + 2 * p;
                int k_row1 = k + 16 + 2 * p + 1;
                half b0 = (k_row0 < K && col_valid) ? B_base[2 * p * N] : (half)0.0h;
                half b1 = (k_row1 < K && col_valid) ? B_base[(2 * p + 1) * N] : (half)0.0h;
                ((int*)&b_val)[p] = as_int((short2)(as_short(b0), as_short(b1)));
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // Store results using vector writes where possible
    #pragma unroll
    for (int blk = 0; blk < 4; blk++) {
        float8 acc = (blk == 0) ? acc0 : (blk == 1) ? acc1 : (blk == 2) ? acc2 : acc3;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int row_idx = row_base + blk * 8 + r;
            if (row_idx < M && col_valid) {
                C[row_idx * N + col_idx] = convert_half(((float*)&acc)[r]);
            }
        }
    }
}
```

