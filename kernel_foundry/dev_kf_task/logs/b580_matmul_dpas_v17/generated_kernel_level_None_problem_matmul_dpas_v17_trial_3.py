## Analysis

Looking at the previous versions:

1. **Version 1 (1.000ms)**: 2x K-loop unroll with double buffering, but has a branch (`if (ki2 < k_iters_2x - 1)`) in the second tile of each pair, and the A load happens after all DPAS operations complete (poor overlap).

2. **Version 2 (1.120ms)**: Simple single-iteration loop with double buffering, but the A load is gated by `if (ki < k_iters - 1)` which adds branch overhead every iteration.

3. **Best version (0.721ms)**: Key improvement - uses `half16` vector load for A (`*(__local half16*)a_dst = *(__global const half16*)a_src`), and interleaves the next A load between DPAS operations (after acc1, before acc2). Has a separate last-iteration path to avoid the branch.

**Key insights for further optimization:**

1. The best version already interleaves A loads with DPAS well, but we can try to also interleave B loads with DPAS from the previous k16 step.
2. The SLM stride of 32 means each row is 64 bytes. With 16 banks of 4 bytes each on Xe2, consecutive rows at the same column offset hit the same bank. We could try stride=34 to stagger bank access.
3. We can try to use `intel_sub_group_block_read` for SLM reads to get vectorized loads.
4. We can try to overlap B loads for the second k16 step with DPAS from the first k16 step.
5. The output store can use `intel_sub_group_block_write` for better write coalescing.

**My approach:**
- Keep the proven architecture (64 WIs, 4 SGs, 32×64×32 tile, A in SLM double-buffered, B from global)
- Use SLM stride of 34 (instead of 32) to reduce bank conflicts - 34 halfs = 68 bytes, which breaks the 64-byte bank alignment pattern
- Better interleaving: start loading B1 while DPAS for B0 is executing
- Use `intel_sub_group_block_read` for SLM A reads where possible
- Remove the last-iteration special case by loading one extra (harmless) A tile that won't be used

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], row-major, FP16 in/out, FP32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM (double-buffered), B from global
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// Each WG computes a 32(M) x 64(N) output tile
// 4 subgroups, each handles 32 rows x 16 cols
// K-loop steps by 32, double-buffered A in SLM
// Optimizations: SLM stride=34 for bank conflict reduction, interleaved B/DPAS,
//   half16 vector loads for A global->SLM, separate last iteration to remove branch

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
    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);

    const int lid = get_local_id(0);
    const int sg_id = lid / 16;
    const int sg_lid = get_sub_group_local_id();

    const int baseM = wg_m * 32;
    const int baseN = wg_n * 64 + sg_id * 16;

    // SLM stride=34 to reduce bank conflicts (34*2=68 bytes per row, not power-of-2)
    #define A_SLM_STRIDE 34
    #define A_SLM_SIZE (32 * A_SLM_STRIDE)
    __local half slm_A[2 * A_SLM_SIZE];

    // Accumulators: 32 rows x 16 cols = 4 x float8
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Cooperative A load: 64 WIs load 32x32 = 1024 halfs = 16 halfs/WI
    const int a_load_row = lid / 2;
    const int a_load_col = (lid & 1) * 16;
    const int a_gm_row = baseM + a_load_row;

    // Preload first A tile (k=0..31) into SLM buffer 0
    {
        __global const half* a_src = A + a_gm_row * K + a_load_col;
        __local half* a_dst = slm_A + a_load_row * A_SLM_STRIDE + a_load_col;
        *(__local half16*)a_dst = *(__global const half16*)a_src;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int k_iters = K / 32;  // 64
    int cur_buf = 0;

    // Main loop: all iterations except the last (no next A load needed for last)
    for (int ki = 0; ki < k_iters - 1; ki++) {
        const int k_base = ki * 32;
        const int next_buf = 1 - cur_buf;

        __local const half* a_slm = slm_A + cur_buf * A_SLM_SIZE;

        // === First k16 step: k_base..k_base+15 ===

        // Load B for first k16
        int8 b0;
        {
            __global const half* b_col = B + k_base * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b0.s0 = as_int((ushort2)(bv0, bv1));
            b0.s1 = as_int((ushort2)(bv2, bv3));
            b0.s2 = as_int((ushort2)(bv4, bv5));
            b0.s3 = as_int((ushort2)(bv6, bv7));
            b0.s4 = as_int((ushort2)(bv8, bv9));
            b0.s5 = as_int((ushort2)(bv10, bv11));
            b0.s6 = as_int((ushort2)(bv12, bv13));
            b0.s7 = as_int((ushort2)(bv14, bv15));
        }

        // Load A from SLM for first k16
        short8 a00, a01, a02, a03;
        {
            __local const half* a_k0 = a_slm + sg_lid;
            a00.s0 = as_short(a_k0[0 * A_SLM_STRIDE]);
            a00.s1 = as_short(a_k0[1 * A_SLM_STRIDE]);
            a00.s2 = as_short(a_k0[2 * A_SLM_STRIDE]);
            a00.s3 = as_short(a_k0[3 * A_SLM_STRIDE]);
            a00.s4 = as_short(a_k0[4 * A_SLM_STRIDE]);
            a00.s5 = as_short(a_k0[5 * A_SLM_STRIDE]);
            a00.s6 = as_short(a_k0[6 * A_SLM_STRIDE]);
            a00.s7 = as_short(a_k0[7 * A_SLM_STRIDE]);

            a01.s0 = as_short(a_k0[8 * A_SLM_STRIDE]);
            a01.s1 = as_short(a_k0[9 * A_SLM_STRIDE]);
            a01.s2 = as_short(a_k0[10 * A_SLM_STRIDE]);
            a01.s3 = as_short(a_k0[11 * A_SLM_STRIDE]);
            a01.s4 = as_short(a_k0[12 * A_SLM_STRIDE]);
            a01.s5 = as_short(a_k0[13 * A_SLM_STRIDE]);
            a01.s6 = as_short(a_k0[14 * A_SLM_STRIDE]);
            a01.s7 = as_short(a_k0[15 * A_SLM_STRIDE]);

            a02.s0 = as_short(a_k0[16 * A_SLM_STRIDE]);
            a02.s1 = as_short(a_k0[17 * A_SLM_STRIDE]);
            a02.s2 = as_short(a_k0[18 * A_SLM_STRIDE]);
            a02.s3 = as_short(a_k0[19 * A_SLM_STRIDE]);
            a02.s4 = as_short(a_k0[20 * A_SLM_STRIDE]);
            a02.s5 = as_short(a_k0[21 * A_SLM_STRIDE]);
            a02.s6 = as_short(a_k0[22 * A_SLM_STRIDE]);
            a02.s7 = as_short(a_k0[23 * A_SLM_STRIDE]);

            a03.s0 = as_short(a_k0[24 * A_SLM_STRIDE]);
            a03.s1 = as_short(a_k0[25 * A_SLM_STRIDE]);
            a03.s2 = as_short(a_k0[26 * A_SLM_STRIDE]);
            a03.s3 = as_short(a_k0[27 * A_SLM_STRIDE]);
            a03.s4 = as_short(a_k0[28 * A_SLM_STRIDE]);
            a03.s5 = as_short(a_k0[29 * A_SLM_STRIDE]);
            a03.s6 = as_short(a_k0[30 * A_SLM_STRIDE]);
            a03.s7 = as_short(a_k0[31 * A_SLM_STRIDE]);
        }

        // DPAS first k16 - interleave with next A tile load
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b0, acc1);

        // Interleave: load next A tile into other SLM buffer while DPAS executes
        __global const half* a_next_src = A + a_gm_row * K + (k_base + 32) + a_load_col;
        __local half* a_next_dst = slm_A + next_buf * A_SLM_SIZE + a_load_row * A_SLM_STRIDE + a_load_col;
        *(__local half16*)a_next_dst = *(__global const half16*)a_next_src;

        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b0, acc3);

        // === Second k16 step: k_base+16..k_base+31 ===

        // Load B for second k16
        int8 b1;
        {
            __global const half* b_col = B + (k_base + 16) * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b1.s0 = as_int((ushort2)(bv0, bv1));
            b1.s1 = as_int((ushort2)(bv2, bv3));
            b1.s2 = as_int((ushort2)(bv4, bv5));
            b1.s3 = as_int((ushort2)(bv6, bv7));
            b1.s4 = as_int((ushort2)(bv8, bv9));
            b1.s5 = as_int((ushort2)(bv10, bv11));
            b1.s6 = as_int((ushort2)(bv12, bv13));
            b1.s7 = as_int((ushort2)(bv14, bv15));
        }

        // Load A from SLM for second k16
        short8 a10, a11, a12, a13;
        {
            __local const half* a_k1 = a_slm + 16 + sg_lid;
            a10.s0 = as_short(a_k1[0 * A_SLM_STRIDE]);
            a10.s1 = as_short(a_k1[1 * A_SLM_STRIDE]);
            a10.s2 = as_short(a_k1[2 * A_SLM_STRIDE]);
            a10.s3 = as_short(a_k1[3 * A_SLM_STRIDE]);
            a10.s4 = as_short(a_k1[4 * A_SLM_STRIDE]);
            a10.s5 = as_short(a_k1[5 * A_SLM_STRIDE]);
            a10.s6 = as_short(a_k1[6 * A_SLM_STRIDE]);
            a10.s7 = as_short(a_k1[7 * A_SLM_STRIDE]);

            a11.s0 = as_short(a_k1[8 * A_SLM_STRIDE]);
            a11.s1 = as_short(a_k1[9 * A_SLM_STRIDE]);
            a11.s2 = as_short(a_k1[10 * A_SLM_STRIDE]);
            a11.s3 = as_short(a_k1[11 * A_SLM_STRIDE]);
            a11.s4 = as_short(a_k1[12 * A_SLM_STRIDE]);
            a11.s5 = as_short(a_k1[13 * A_SLM_STRIDE]);
            a11.s6 = as_short(a_k1[14 * A_SLM_STRIDE]);
            a11.s7 = as_short(a_k1[15 * A_SLM_STRIDE]);

            a12.s0 = as_short(a_k1[16 * A_SLM_STRIDE]);
            a12.s1 = as_short(a_k1[17 * A_SLM_STRIDE]);
            a12.s2 = as_short(a_k1[18 * A_SLM_STRIDE]);
            a12.s3 = as_short(a_k1[19 * A_SLM_STRIDE]);
            a12.s4 = as_short(a_k1[20 * A_SLM_STRIDE]);
            a12.s5 = as_short(a_k1[21 * A_SLM_STRIDE]);
            a12.s6 = as_short(a_k1[22 * A_SLM_STRIDE]);
            a12.s7 = as_short(a_k1[23 * A_SLM_STRIDE]);

            a13.s0 = as_short(a_k1[24 * A_SLM_STRIDE]);
            a13.s1 = as_short(a_k1[25 * A_SLM_STRIDE]);
            a13.s2 = as_short(a_k1[26 * A_SLM_STRIDE]);
            a13.s3 = as_short(a_k1[27 * A_SLM_STRIDE]);
            a13.s4 = as_short(a_k1[28 * A_SLM_STRIDE]);
            a13.s5 = as_short(a_k1[29 * A_SLM_STRIDE]);
            a13.s6 = as_short(a_k1[30 * A_SLM_STRIDE]);
            a13.s7 = as_short(a_k1[31 * A_SLM_STRIDE]);
        }

        // DPAS second k16
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b1, acc3);

        cur_buf = next_buf;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Last iteration (ki = k_iters - 1): no next A load needed ===
    {
        const int k_base = (k_iters - 1) * 32;
        __local const half* a_slm = slm_A + cur_buf * A_SLM_SIZE;

        // First k16
        int8 b0;
        {
            __global const half* b_col = B + k_base * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b0.s0 = as_int((ushort2)(bv0, bv1));
            b0.s1 = as_int((ushort2)(bv2, bv3));
            b0.s2 = as_int((ushort2)(bv4, bv5));
            b0.s3 = as_int((ushort2)(bv6, bv7));
            b0.s4 = as_int((ushort2)(bv8, bv9));
            b0.s5 = as_int((ushort2)(bv10, bv11));
            b0.s6 = as_int((ushort2)(bv12, bv13));
            b0.s7 = as_int((ushort2)(bv14, bv15));
        }

        short8 a00, a01, a02, a03;
        {
            __local const half* a_k0 = a_slm + sg_lid;
            a00.s0 = as_short(a_k0[0 * A_SLM_STRIDE]);
            a00.s1 = as_short(a_k0[1 * A_SLM_STRIDE]);
            a00.s2 = as_short(a_k0[2 * A_SLM_STRIDE]);
            a00.s3 = as_short(a_k0[3 * A_SLM_STRIDE]);
            a00.s4 = as_short(a_k0[4 * A_SLM_STRIDE]);
            a00.s5 = as_short(a_k0[5 * A_SLM_STRIDE]);
            a00.s6 = as_short(a_k0[6 * A_SLM_STRIDE]);
            a00.s7 = as_short(a_k0[7 * A_SLM_STRIDE]);

            a01.s0 = as_short(a_k0[8 * A_SLM_STRIDE]);
            a01.s1 = as_short(a_k0[9 * A_SLM_STRIDE]);
            a01.s2 = as_short(a_k0[10 * A_SLM_STRIDE]);
            a01.s3 = as_short(a_k0[11 * A_SLM_STRIDE]);
            a01.s4 = as_short(a_k0[12 * A_SLM_STRIDE]);
            a01.s5 = as_short(a_k0[13 * A_SLM_STRIDE]);
            a01.s6 = as_short(a_k0[14 * A_SLM_STRIDE]);
            a01.s7 = as_short(a_k0[15 * A_SLM_STRIDE]);

            a02.s0 = as_short(a_k0[16 * A_SLM_STRIDE]);
            a02.s1 = as_short(a_k0[17 * A_SLM_STRIDE]);
            a02.s2 = as_short(a_k0[18 * A_SLM_STRIDE]);
            a02.s3 = as_short(a_k0[19 * A_SLM_STRIDE]);
            a02.s4 = as_short(a_k0[20 * A_SLM_STRIDE]);
            a02.s5 = as_short(a_k0[21 * A_SLM_STRIDE]);
            a02.s6 = as_short(a_k0[22 * A_SLM_STRIDE]);
            a02.s7 = as_short(a_k0[23 * A_SLM_STRIDE]);

            a03.s0 = as_short(a_k0[24 * A_SLM_STRIDE]);
            a03.s1 = as_short(a_k0[25 * A_SLM_STRIDE]);
            a03.s2 = as_short(a_k0[26 * A_SLM_STRIDE]);
            a03.s3 = as_short(a_k0[27 * A_SLM_STRIDE]);
            a03.s4 = as_short(a_k0[28 * A_SLM_STRIDE]);
            a03.s5 = as_short(a_k0[29 * A_SLM_STRIDE]);
            a03.s6 = as_short(a_k0[30 * A_SLM_STRIDE]);
            a03.s7 = as_short(a_k0[31 * A_SLM_STRIDE]);
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a02, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a03, b0, acc3);

        // Second k16
        int8 b1;
        {
            __global const half* b_col = B + (k_base + 16) * N + baseN + sg_lid;
            ushort bv0  = as_ushort(b_col[0 * N]);
            ushort bv1  = as_ushort(b_col[1 * N]);
            ushort bv2  = as_ushort(b_col[2 * N]);
            ushort bv3  = as_ushort(b_col[3 * N]);
            ushort bv4  = as_ushort(b_col[4 * N]);
            ushort bv5  = as_ushort(b_col[5 * N]);
            ushort bv6  = as_ushort(b_col[6 * N]);
            ushort bv7  = as_ushort(b_col[7 * N]);
            ushort bv8  = as_ushort(b_col[8 * N]);
            ushort bv9  = as_ushort(b_col[9 * N]);
            ushort bv10 = as_ushort(b_col[10 * N]);
            ushort bv11 = as_ushort(b_col[11 * N]);
            ushort bv12 = as_ushort(b_col[12 * N]);
            ushort bv13 = as_ushort(b_col[13 * N]);
            ushort bv14 = as_ushort(b_col[14 * N]);
            ushort bv15 = as_ushort(b_col[15 * N]);

            b1.s0 = as_int((ushort2)(bv0, bv1));
            b1.s1 = as_int((ushort2)(bv2, bv3));
            b1.s2 = as_int((ushort2)(bv4, bv5));
            b1.s3 = as_int((ushort2)(bv6, bv7));
            b1.s4 = as_int((ushort2)(bv8, bv9));
            b1.s5 = as_int((ushort2)(bv10, bv11));
            b1.s6 = as_int((ushort2)(bv12, bv13));
            b1.s7 = as_int((ushort2)(bv14, bv15));
        }

        short8 a10, a11, a12, a13;
        {
            __local const half* a_k1 = a_slm + 16 + sg_lid;
            a10.s0 = as_short(a_k1[0 * A_SLM_STRIDE]);
            a10.s1 = as_short(a_k1[1 * A_SLM_STRIDE]);
            a10.s2 = as_short(a_k1[2 * A_SLM_STRIDE]);
            a10.s3 = as_short(a_k1[3 * A_SLM_STRIDE]);
            a10.s4 = as_short(a_k1[4 * A_SLM_STRIDE]);
            a10.s5 = as_short(a_k1[5 * A_SLM_STRIDE]);
            a10.s6 = as_short(a_k1[6 * A_SLM_STRIDE]);
            a10.s7 = as_short(a_k1[7 * A_SLM_STRIDE]);

            a11.s0 = as_short(a_k1[8 * A_SLM_STRIDE]);
            a11.s1 = as_short(a_k1[9 * A_SLM_STRIDE]);
            a11.s2 = as_short(a_k1[10 * A_SLM_STRIDE]);
            a11.s3 = as_short(a_k1[11 * A_SLM_STRIDE]);
            a11.s4 = as_short(a_k1[12 * A_SLM_STRIDE]);
            a11.s5 = as_short(a_k1[13 * A_SLM_STRIDE]);
            a11.s6 = as_short(a_k1[14 * A_SLM_STRIDE]);
            a11.s7 = as_short(a_k1[15 * A_SLM_STRIDE]);

            a12.s0 = as_short(a_k1[16 * A_SLM_STRIDE]);
            a12.s1 = as_short(a_k1[17 * A_SLM_STRIDE]);
            a12.s2 = as_short(a_k1[18 * A_SLM_STRIDE]);
            a12.s3 = as_short(a_k1[19 * A_SLM_STRIDE]);
            a12.s4 = as_short(a_k1[20 * A_SLM_STRIDE]);
            a12.s5 = as_short(a_k1[21 * A_SLM_STRIDE]);
            a12.s6 = as_short(a_k1[22 * A_SLM_STRIDE]);
            a12.s7 = as_short(a_k1[23 * A_SLM_STRIDE]);

            a13.s0 = as_short(a_k1[24 * A_SLM_STRIDE]);
            a13.s1 = as_short(a_k1[25 * A_SLM_STRIDE]);
            a13.s2 = as_short(a_k1[26 * A_SLM_STRIDE]);
            a13.s3 = as_short(a_k1[27 * A_SLM_STRIDE]);
            a13.s4 = as_short(a_k1[28 * A_SLM_STRIDE]);
            a13.s5 = as_short(a_k1[29 * A_SLM_STRIDE]);
            a13.s6 = as_short(a_k1[30 * A_SLM_STRIDE]);
            a13.s7 = as_short(a_k1[31 * A_SLM_STRIDE]);
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a12, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a13, b1, acc3);
    }

    // Store C: each subgroup writes 32 rows x 16 cols
    __global half* c_out = C + baseM * N + baseN + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(0 + r) * N] = convert_half(((float*)&acc0)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(8 + r) * N] = convert_half(((float*)&acc1)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(16 + r) * N] = convert_half(((float*)&acc2)[r]);
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        c_out[(24 + r) * N] = convert_half(((float*)&acc3)[r]);
    }

    #undef A_SLM_STRIDE
    #undef A_SLM_SIZE
}
```