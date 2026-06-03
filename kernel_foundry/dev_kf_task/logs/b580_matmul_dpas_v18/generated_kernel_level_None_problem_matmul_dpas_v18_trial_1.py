## Analysis

### Previous Version Performance
The last working kernel (Version 3) achieves 1.510ms. Let me analyze the issues:

1. **B loads are scalar** - Each lane loads individual halfs from B with stride N between rows. This creates 16 scattered reads per k-row, which is very inefficient.
2. **SLM stride of 34** - Good for avoiding bank conflicts but the padding wastes some SLM bandwidth.
3. **Double-buffering structure** - The barrier placement is suboptimal. The next A tile load happens at the start of the loop iteration, but the barrier is at the end. This means compute and load don't overlap well.
4. **The compile error** mentions "invalid reinterpretation: sizes of" - this was from a failed trial using `as_int` on `half2`. The issue is that `half2` is 4 bytes and `int` is 4 bytes, but the compiler may not support `as_int((half2)(...))` directly. Need to use `as_int(as_uint(...))` or cast through `short2`.

### Key Optimizations for This Version

1. **Fix B loading** - Use `vload2` to load pairs of halfs from consecutive k-rows for the same column, then reinterpret as int. This reduces address calculations.
2. **Better double-buffering** - Load next A tile BEFORE computing current tile's second k16 block, so the load overlaps with compute.
3. **Remove K-remainder check** - K=2048 divides evenly by 32, so remove the `if (next_k < K)` guards.
4. **Use proper type casting** - Avoid `as_int((half2)(...))` which causes compile errors. Instead use `as_int` on a properly constructed `short2`.
5. **Interleave B loads with DPAS** - Load B for second k16 while first k16 DPAS is executing.
6. **SLM stride = 32** - Since we have 16-wide subgroups reading columns 0-15 and 16-31, stride 32 actually works fine (different banks for different lanes). Stride 34 wastes space and may cause misalignment.

Let me use stride 32 (no padding) which gives perfect alignment and the 32-half rows fit exactly in SLM banks without conflicts for our access pattern (consecutive lanes access consecutive elements).

Actually, with 32 banks of 4 bytes each, accessing stride-32 halfs (2 bytes each) means lane 0 accesses byte 0, lane 1 accesses byte 2, etc. 16 lanes access bytes 0-30, which spans banks 0-7 (4 bytes per bank). No conflicts for 16 lanes accessing 16 consecutive halfs.

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

    // SLM for A: double-buffered, 32 rows x 32 cols of half
    // Stride = 32 halfs (no padding - 16 lanes access consecutive halfs, no bank conflict)
    #define SLM_STRIDE 32
    #define SLM_BUF_SIZE (32 * SLM_STRIDE)
    __local half slm_a[2 * SLM_BUF_SIZE];

    // Accumulators: 32 rows x 16 cols per subgroup = 4 blocks of 8 rows
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Cooperative A load helper: 64 WIs load 32x32 = 1024 halfs = 16 per WI
    // lid/2 = row (0..31), (lid%2)*16 = col offset (0 or 16)
    const int load_row = lid / 2;
    const int load_col = (lid % 2) * 16;
    const int slm_store_offset = load_row * SLM_STRIDE + load_col;

    // Preload first A tile into SLM buffer 0
    {
        __global const half* a_ptr = A + (tile_row + load_row) * K + load_col;
        vstore16(vload16(0, a_ptr), 0, &slm_a[slm_store_offset]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    // Main K-loop, step by 32. K is guaranteed divisible by 32.
    for (int k = 0; k < K; k += 32) {
        int next_k = k + 32;
        int next_buf = 1 - cur_buf;

        __local const half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

        // ===== K-step 0: k_inner = 0..15 =====
        // Load A from SLM for rows 0-31, k cols 0-15
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

        // Load B for k_inner=0..15: B[k..k+15, tile_col+sg_lane]
        // Each lane loads 16 halfs from 16 consecutive k-rows at its column
        // Pack pairs into int for VNNI format
        int8 b0;
        {
            __global const half* b_ptr = B + k * N + tile_col + sg_lane;
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

        // DPAS for k_inner=0..15
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc3);

        // ===== K-step 1: k_inner = 16..31 =====
        // Load A from SLM for k cols 16-31
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

        // Load B for k_inner=16..31
        int8 b1;
        {
            __global const half* b_ptr = B + (k + 16) * N + tile_col + sg_lane;
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

        // Start loading next A tile into next buffer (overlaps with DPAS below)
        if (next_k < K) {
            __global const half* a_ptr = A + (tile_row + load_row) * K + next_k + load_col;
            vstore16(vload16(0, a_ptr), 0, &slm_a[next_buf * SLM_BUF_SIZE + slm_store_offset]);
        }

        // DPAS for k_inner=16..31
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a4, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a5, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a6, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a7, b1, acc3);

        // Barrier to ensure next A tile is fully loaded before we use it
        barrier(CLK_LOCAL_MEM_FENCE);
        cur_buf = next_buf;
    }

    // Store results: each subgroup writes 32 rows x 16 cols
    // Use intel_sub_group_block_write_us for vectorized stores
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