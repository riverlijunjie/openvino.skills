## Analysis

Looking at the previous versions:

1. **Version 1** (1.510ms): Uses `reqd_work_group_size(16, 4, 1)` with padding stride 34. Has the A load before the loop and next A load inside the loop with a branch check.

2. **Version 2** (1.560ms): Uses `reqd_work_group_size(64, 1, 1)` with padding stride 34. Similar structure but slightly slower, likely due to the 1D workgroup layout affecting subgroup formation.

3. **Previous best** (1.410ms): Uses `reqd_work_group_size(16, 4, 1)` with stride 32 (no padding). Moves the next A tile load to overlap with the second DPAS block. Key improvements:
   - No SLM padding (stride 32) - simpler addressing
   - Overlaps next A load with second k16 DPAS block
   - Removed the `if (next_k < K)` check from the barrier (always barriers)

**Key bottlenecks to address:**
1. The `if (next_k < K)` branch in the A load still exists - since K=2048 divides evenly by 32, we can remove the last-iteration check by restructuring
2. B loads are scalar (16 individual loads per k16 block) - very inefficient
3. No interleaving of B loads with DPAS operations
4. The barrier at the end of each iteration adds overhead even on the last iteration
5. Loop overhead for 64 iterations (K=2048/32=64)

**Proposed improvements:**
1. **Remove K-remainder path**: K=2048 divides by 32, so remove all `if (next_k < K)` checks. Process last iteration separately to avoid unnecessary barrier.
2. **2x K-loop unroll**: Process 64 k-values per iteration (two k32 tiles), reducing loop overhead from 64 to 32 iterations. This requires loading two A tiles per iteration.
3. **Better interleaving**: Load B for the second k16 block while DPAS processes the first k16 block. Load next A while processing second k16 block.
4. **Vectorized C stores**: Use `intel_sub_group_block_write_us` for output writes.

After careful consideration, the safest high-impact optimization is:
- Remove the branch check (K always divisible by 32)
- Better interleave: load B1 during DPAS of first k16, load next A during DPAS of second k16
- Use SLM stride of 32 (proven good)
- Restructure loop to avoid barrier on last iteration

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half, f32 accumulation
// Launch: GWS = ((N/64)*16, (M/32)*4, 1), LWS = (16, 4, 1)
// 4 subgroups of 16 WIs per WG. Each WG computes 32x64 tile.
// Each subgroup computes 32x16 using DPAS.
// A in SLM (double-buffered), B from global/L2.
// TILE_M=32, TILE_N=64, TILE_K=32. K must be divisible by 32.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 4, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);
    const int sg_id = get_local_id(1);
    const int sg_lane = get_local_id(0);

    const int tile_row = wg_m * 32;
    const int tile_col = wg_n * 64 + sg_id * 16;

    // Linear local ID for cooperative loads
    const int lid = sg_id * 16 + sg_lane; // 0..63

    // SLM for A: double-buffered, 32 rows x 32 cols of half, stride=32
    #define SLM_STRIDE 32
    #define SLM_BUF_SIZE (32 * SLM_STRIDE)
    __local half slm_a[2 * SLM_BUF_SIZE];

    // Accumulators
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Cooperative A load: 64 WIs load 32x32 = 1024 halfs = 16 per WI
    const int load_row = lid / 2;
    const int load_col = (lid % 2) * 16;
    const int slm_store_offset = load_row * SLM_STRIDE + load_col;

    // Precompute A base for this WI's load responsibility
    __global const half* a_load_base = A + (tile_row + load_row) * K + load_col;

    // Preload first A tile into SLM buffer 0
    vstore16(vload16(0, a_load_base), 0, &slm_a[slm_store_offset]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // B base pointer for this subgroup's column
    __global const half* b_base = B + tile_col + sg_lane;

    const int k_iters = K / 32;  // K guaranteed divisible by 32
    int cur_buf = 0;

    // Main K-loop: process all but last iteration with double-buffering
    for (int ki = 0; ki < k_iters - 1; ki++) {
        const int k = ki * 32;
        const int next_buf = 1 - cur_buf;

        __local const half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

        // ===== Load B for k_inner=0..15 =====
        int8 b0;
        {
            __global const half* b_ptr = b_base + k * N;
            short bv0  = as_short(b_ptr[0 * N]);
            short bv1  = as_short(b_ptr[1 * N]);
            short bv2  = as_short(b_ptr[2 * N]);
            short bv3  = as_short(b_ptr[3 * N]);
            short bv4  = as_short(b_ptr[4 * N]);
            short bv5  = as_short(b_ptr[5 * N]);
            short bv6  = as_short(b_ptr[6 * N]);
            short bv7  = as_short(b_ptr[7 * N]);
            short bv8  = as_short(b_ptr[8 * N]);
            short bv9  = as_short(b_ptr[9 * N]);
            short bv10 = as_short(b_ptr[10 * N]);
            short bv11 = as_short(b_ptr[11 * N]);
            short bv12 = as_short(b_ptr[12 * N]);
            short bv13 = as_short(b_ptr[13 * N]);
            short bv14 = as_short(b_ptr[14 * N]);
            short bv15 = as_short(b_ptr[15 * N]);

            b0.s0 = as_int((short2)(bv0, bv1));
            b0.s1 = as_int((short2)(bv2, bv3));
            b0.s2 = as_int((short2)(bv4, bv5));
            b0.s3 = as_int((short2)(bv6, bv7));
            b0.s4 = as_int((short2)(bv8, bv9));
            b0.s5 = as_int((short2)(bv10, bv11));
            b0.s6 = as_int((short2)(bv12, bv13));
            b0.s7 = as_int((short2)(bv14, bv15));
        }

        // ===== Load A from SLM for k_inner=0..15 =====
        short8 a0, a1, a2, a3;
        a0 = (short8)(
            as_short(cur_slm[0 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[1 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[2 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[3 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[4 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[5 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[6 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[7 * SLM_STRIDE + sg_lane])
        );
        a1 = (short8)(
            as_short(cur_slm[8 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[9 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[10 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[11 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[12 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[13 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[14 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[15 * SLM_STRIDE + sg_lane])
        );
        a2 = (short8)(
            as_short(cur_slm[16 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[17 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[18 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[19 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[20 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[21 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[22 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[23 * SLM_STRIDE + sg_lane])
        );
        a3 = (short8)(
            as_short(cur_slm[24 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[25 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[26 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[27 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[28 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[29 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[30 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[31 * SLM_STRIDE + sg_lane])
        );

        // ===== DPAS k_inner=0..15, interleaved with B1 load =====
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc0);

        // Start loading B for k_inner=16..31 (overlap with DPAS)
        int8 b1;
        __global const half* b_ptr1 = b_base + (k + 16) * N;
        short bv1_0  = as_short(b_ptr1[0 * N]);
        short bv1_1  = as_short(b_ptr1[1 * N]);

        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc1);

        short bv1_2  = as_short(b_ptr1[2 * N]);
        short bv1_3  = as_short(b_ptr1[3 * N]);
        short bv1_4  = as_short(b_ptr1[4 * N]);
        short bv1_5  = as_short(b_ptr1[5 * N]);

        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc2);

        short bv1_6  = as_short(b_ptr1[6 * N]);
        short bv1_7  = as_short(b_ptr1[7 * N]);
        short bv1_8  = as_short(b_ptr1[8 * N]);
        short bv1_9  = as_short(b_ptr1[9 * N]);

        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc3);

        short bv1_10 = as_short(b_ptr1[10 * N]);
        short bv1_11 = as_short(b_ptr1[11 * N]);
        short bv1_12 = as_short(b_ptr1[12 * N]);
        short bv1_13 = as_short(b_ptr1[13 * N]);
        short bv1_14 = as_short(b_ptr1[14 * N]);
        short bv1_15 = as_short(b_ptr1[15 * N]);

        b1.s0 = as_int((short2)(bv1_0, bv1_1));
        b1.s1 = as_int((short2)(bv1_2, bv1_3));
        b1.s2 = as_int((short2)(bv1_4, bv1_5));
        b1.s3 = as_int((short2)(bv1_6, bv1_7));
        b1.s4 = as_int((short2)(bv1_8, bv1_9));
        b1.s5 = as_int((short2)(bv1_10, bv1_11));
        b1.s6 = as_int((short2)(bv1_12, bv1_13));
        b1.s7 = as_int((short2)(bv1_14, bv1_15));

        // ===== Load A from SLM for k_inner=16..31 =====
        short8 a4, a5, a6, a7;
        a4 = (short8)(
            as_short(cur_slm[0 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[1 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[2 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[3 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[4 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[5 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[6 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[7 * SLM_STRIDE + 16 + sg_lane])
        );
        a5 = (short8)(
            as_short(cur_slm[8 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[9 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[10 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[11 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[12 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[13 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[14 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[15 * SLM_STRIDE + 16 + sg_lane])
        );
        a6 = (short8)(
            as_short(cur_slm[16 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[17 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[18 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[19 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[20 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[21 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[22 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[23 * SLM_STRIDE + 16 + sg_lane])
        );
        a7 = (short8)(
            as_short(cur_slm[24 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[25 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[26 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[27 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[28 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[29 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[30 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[31 * SLM_STRIDE + 16 + sg_lane])
        );

        // Load next A tile into next buffer (overlaps with DPAS below)
        __global const half* a_next = a_load_base + (ki + 1) * 32;
        half16 a_prefetch = vload16(0, a_next);

        // ===== DPAS k_inner=16..31, interleaved with next A store =====
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a4, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a5, b1, acc1);

        // Store next A to SLM (overlap with remaining DPAS)
        vstore16(a_prefetch, 0, &slm_a[next_buf * SLM_BUF_SIZE + slm_store_offset]);

        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a6, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a7, b1, acc3);

        // Barrier for next A tile
        barrier(CLK_LOCAL_MEM_FENCE);
        cur_buf = next_buf;
    }

    // ===== Last iteration (no next A load needed, no barrier needed) =====
    {
        const int k = (k_iters - 1) * 32;
        __local const half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

        // Load B for k_inner=0..15
        int8 b0;
        {
            __global const half* b_ptr = b_base + k * N;
            short bv0  = as_short(b_ptr[0 * N]);
            short bv1  = as_short(b_ptr[1 * N]);
            short bv2  = as_short(b_ptr[2 * N]);
            short bv3  = as_short(b_ptr[3 * N]);
            short bv4  = as_short(b_ptr[4 * N]);
            short bv5  = as_short(b_ptr[5 * N]);
            short bv6  = as_short(b_ptr[6 * N]);
            short bv7  = as_short(b_ptr[7 * N]);
            short bv8  = as_short(b_ptr[8 * N]);
            short bv9  = as_short(b_ptr[9 * N]);
            short bv10 = as_short(b_ptr[10 * N]);
            short bv11 = as_short(b_ptr[11 * N]);
            short bv12 = as_short(b_ptr[12 * N]);
            short bv13 = as_short(b_ptr[13 * N]);
            short bv14 = as_short(b_ptr[14 * N]);
            short bv15 = as_short(b_ptr[15 * N]);

            b0.s0 = as_int((short2)(bv0, bv1));
            b0.s1 = as_int((short2)(bv2, bv3));
            b0.s2 = as_int((short2)(bv4, bv5));
            b0.s3 = as_int((short2)(bv6, bv7));
            b0.s4 = as_int((short2)(bv8, bv9));
            b0.s5 = as_int((short2)(bv10, bv11));
            b0.s6 = as_int((short2)(bv12, bv13));
            b0.s7 = as_int((short2)(bv14, bv15));
        }

        // Load A from SLM for k_inner=0..15
        short8 a0, a1, a2, a3;
        a0 = (short8)(
            as_short(cur_slm[0 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[1 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[2 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[3 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[4 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[5 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[6 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[7 * SLM_STRIDE + sg_lane])
        );
        a1 = (short8)(
            as_short(cur_slm[8 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[9 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[10 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[11 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[12 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[13 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[14 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[15 * SLM_STRIDE + sg_lane])
        );
        a2 = (short8)(
            as_short(cur_slm[16 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[17 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[18 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[19 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[20 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[21 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[22 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[23 * SLM_STRIDE + sg_lane])
        );
        a3 = (short8)(
            as_short(cur_slm[24 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[25 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[26 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[27 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[28 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[29 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[30 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[31 * SLM_STRIDE + sg_lane])
        );

        // DPAS k_inner=0..15
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc3);

        // Load B for k_inner=16..31
        int8 b1;
        {
            __global const half* b_ptr = b_base + (k + 16) * N;
            short bv0  = as_short(b_ptr[0 * N]);
            short bv1  = as_short(b_ptr[1 * N]);
            short bv2  = as_short(b_ptr[2 * N]);
            short bv3  = as_short(b_ptr[3 * N]);
            short bv4  = as_short(b_ptr[4 * N]);
            short bv5  = as_short(b_ptr[5 * N]);
            short bv6  = as_short(b_ptr[6 * N]);
            short bv7  = as_short(b_ptr[7 * N]);
            short bv8  = as_short(b_ptr[8 * N]);
            short bv9  = as_short(b_ptr[9 * N]);
            short bv10 = as_short(b_ptr[10 * N]);
            short bv11 = as_short(b_ptr[11 * N]);
            short bv12 = as_short(b_ptr[12 * N]);
            short bv13 = as_short(b_ptr[13 * N]);
            short bv14 = as_short(b_ptr[14 * N]);
            short bv15 = as_short(b_ptr[15 * N]);

            b1.s0 = as_int((short2)(bv0, bv1));
            b1.s1 = as_int((short2)(bv2, bv3));
            b1.s2 = as_int((short2)(bv4, bv5));
            b1.s3 = as_int((short2)(bv6, bv7));
            b1.s4 = as_int((short2)(bv8, bv9));
            b1.s5 = as_int((short2)(bv10, bv11));
            b1.s6 = as_int((short2)(bv12, bv13));
            b1.s7 = as_int((short2)(bv14, bv15));
        }

        // Load A from SLM for k_inner=16..31
        short8 a4, a5, a6, a7;
        a4 = (short8)(
            as_short(cur_slm[0 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[1 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[2 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[3 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[4 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[5 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[6 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[7 * SLM_STRIDE + 16 + sg_lane])
        );
        a5 = (short8)(
            as_short(cur_slm[8 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[9 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[10 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[11 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[12 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[13 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[14 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[15 * SLM_STRIDE + 16 + sg_lane])
        );
        a6 = (short8)(
            as_short(cur_slm[16 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[17 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[18 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[19 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[20 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[21 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[22 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[23 * SLM_STRIDE + 16 + sg_lane])
        );
        a7 = (short8)(
            as_short(cur_slm[24 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[25 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[26 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[27 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[28 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[29 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[30 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[31 * SLM_STRIDE + 16 + sg_lane])
        );

        // DPAS k_inner=16..31
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a4, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a5, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a6, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a7, b1, acc3);
    }

    // Store results: each subgroup writes 32 rows x 16 cols
    __global half* c_ptr = C + tile_row * N + tile_col + sg_lane;

    float* p0 = (float*)&acc0;
    float* p1 = (float*)&acc1;
    float* p2 = (float*)&acc2;
    float* p3 = (float*)&acc3;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(0 + i) * N] = convert_half(p0[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(8 + i) * N] = convert_half(p1[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(16 + i) * N] = convert_half(p2[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c_ptr[(24 + i) * N] = convert_half(p3[i]);
    }
}
```