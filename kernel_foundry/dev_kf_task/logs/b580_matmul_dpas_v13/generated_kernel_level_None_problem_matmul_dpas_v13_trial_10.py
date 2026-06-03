

## Analysis

Looking at the results:
- **Version 1** (32M×256N, K16, double-buffered pairs): 1.140ms
- **Version 2** (32M×256N, K16, single-tile loop): 1.160ms  
- **Best** (32M×256N, K32): 1.160ms
- **Last** (64M×128N, K32, 8 SGs): 3.370ms — much worse due to register pressure (8 acc registers) and only 128 WIs loading 2048 elements (16 loads per WI)

Key observations:
1. The 32M×256N tile with 16 subgroups (256 WIs) works best — good balance of A reuse (16x) and parallelism
2. The 64M version has too much register pressure and SLM loading overhead
3. Version 1's paired K processing was slightly better than Version 2's generic loop
4. Both A and B are loaded from global — B has no reuse across subgroups since each SG covers different columns

**Key improvements to try:**
1. **Cache both A and B in SLM** — B is loaded redundantly when we could share it. Actually with 16 SGs each needing different B columns, B can't be shared. But we can use SLM for A more efficiently.
2. **Reduce barrier overhead** — use prefetch-style loading where DPAS overlaps with next tile load
3. **Use block reads for stores** — `intel_sub_group_block_write_us` for output
4. **Better A loading** — use `intel_sub_group_block_read` for coalesced loads from global to SLM
5. **Try 32M×256N with K16 step, tighter loop** — minimize overhead per iteration
6. **Prefetch B** — load B for next iteration while computing current

Let me stick with the winning 32M×256N config but optimize the inner loop to minimize barriers and maximize overlap.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Tile: 32M x 256N, K-step=16
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16) per k16 step
// A in SLM (16x reuse across subgroups), double-buffered
// B loaded from global per subgroup
// SLM: 2 * 32*16 * 2 = 2KB (tiny, fits easily)
// GWS = (ceil(N/256)*256, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16
//
// Key optimizations vs previous:
// 1. Tighter inner loop with prefetch of next A overlapping DPAS
// 2. Use block writes for output
// 3. Minimize branching in hot loop

#define TILE_M 32
#define TILE_N 256
#define TILE_K 16
#define SLM_A_SIZE (TILE_M * TILE_K)  // 512 ushorts = 1KB per buffer
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

    __local ushort slm_a[2 * SLM_A_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const bool col_in_bounds = (col_base + 16 <= N);

    // Precompute A loading parameters for this work-item
    // Each WI loads 2 elements: lid and lid+256
    const int eid0 = lid;
    const int eid1 = lid + WG_SIZE;
    const int ar0 = eid0 >> 4;  // eid0 / 16
    const int ac0 = eid0 & 15;  // eid0 % 16
    const int ar1 = eid1 >> 4;
    const int ac1 = eid1 & 15;
    const int gr0 = row_base + ar0;
    const int gr1 = row_base + ar1;
    const bool r0_valid = (gr0 < M);
    const bool r1_valid = (gr1 < M);

    // Load first A tile into buffer 0
    {
        __local ushort* dst = slm_a;
        dst[eid0] = (r0_valid && ac0 < K) ? A_us[gr0 * K + ac0] : (ushort)0;
        dst[eid1] = (r1_valid && ac1 < K) ? A_us[gr1 * K + ac1] : (ushort)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int cur_buf = kt & 1;
        __local const ushort* ca = slm_a + cur_buf * SLM_A_SIZE;

        // Read A tile from SLM into registers
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

        // Load B from global: 16 rows x 16 cols packed as int8
        int8 b_val;
        if (col_in_bounds) {
            __global const ushort* bp = B_us + k_offset * N + col_base;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(bp + (2 * p) * N);
                ushort s1 = intel_sub_group_block_read_us(bp + (2 * p + 1) * N);
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }
        } else {
            int col = col_base + sg_lid;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int kr0 = k_offset + 2 * p;
                int kr1 = kr0 + 1;
                ushort s0 = (kr0 < K && col < N) ? B_us[kr0 * N + col] : (ushort)0;
                ushort s1 = (kr1 < K && col < N) ? B_us[kr1 * N + col] : (ushort)0;
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }
        }

        // Start prefetching next A tile while DPAS executes
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - cur_buf;
            const int next_k = (kt + 1) * TILE_K;
            __local ushort* dst = slm_a + next_buf * SLM_A_SIZE;
            int gk0 = next_k + ac0;
            int gk1 = next_k + ac1;
            dst[eid0] = (r0_valid && gk0 < K) ? A_us[gr0 * K + gk0] : (ushort)0;
            dst[eid1] = (r1_valid && gk1 < K) ? A_us[gr1 * K + gk1] : (ushort)0;
        }

        // DPAS compute - should overlap with SLM writes on Xe2
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results using block writes where possible
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

