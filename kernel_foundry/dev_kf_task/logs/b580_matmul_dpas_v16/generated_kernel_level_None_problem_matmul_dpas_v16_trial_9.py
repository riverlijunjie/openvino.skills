## Analysis

The previous kernel failed correctness tests. The issue is a **synchronization bug**: the kernel loads both A buffers (k+64 into buffer 0, k+96 into buffer 1) at the end of Phase 1, but the barrier at the top of the loop (`barrier(CLK_LOCAL_MEM_FENCE)`) is placed incorrectly. Specifically:

1. The barrier at the **beginning** of the loop means the first iteration reads SLM data that was loaded before the loop (correct for iteration 0), but subsequent iterations need the barrier to happen **after** the SLM writes, not before the reads of the **same** iteration.

2. The kernel preloads buffer 1 (k+96) during Phase 1 computation, but then immediately reads from buffer 1 in the next iteration's Phase 1 without a proper barrier between the write and read of buffer 1.

3. The `sg_col` computation includes `+ sg_lid` which is wrong for the B address calculation - in the best implementation, `sg_col = base_col + sg_id * 16` (without sg_lid), and then B is accessed at `sg_col + sg_lid`. The failed kernel uses `sg_col = base_col + sg_id * 16 + sg_lid` and then accesses B at `sg_col` directly. This is actually equivalent for B reads, but the C write uses `sg_col` which already includes `sg_lid` - this is fine.

Wait, looking more carefully: the real issue is the **double-buffer synchronization**. The best implementation loads A(k+32) into buffer 1 during Phase 0, then barriers, then reads buffer 1 in Phase 1, then loads A(k+64) into buffer 0, then barriers. This is correct because each barrier separates writes from reads of the same buffer.

The failed kernel loads BOTH buffers at the end of Phase 1 (before the barrier at loop top), meaning buffer 0 is written and then read in Phase 0 of the SAME iteration (after the barrier at loop top of the NEXT iteration). But buffer 1 is written and read within the same iteration without an intervening barrier - that's the bug.

**Fix**: Return to the proven best implementation structure with proper double-buffering. I'll apply micro-optimizations that maintain correctness:
- Remove the `if (k + 64 < K)` branch (K=2048 divides by 64, but last iteration k=1984 would try k+64=2048 which equals K, so we need to handle this)
- Actually K=2048, loop steps by 64: k=0,64,128,...,1984. At k=1984, k+64=2048=K, so the condition `k+64 < K` is false for the last iteration. We can restructure to avoid the branch.

I'll stick with the best implementation's structure and apply safe optimizations.

## Improved OCL code

```OCL
// GEMM: C[M,N] = A[M,K] * B[K,N], all half, f32 accumulation
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// 4 subgroups of 16 WIs. Tile: 32x64x32. A in SLM (double-buffered), B from global.
// K=2048, M=2048, N=2048 - all divisible by 64.
// K-loop unrolled 2x (step 64), proper double-buffer synchronization.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

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
    const int lid = get_local_id(0);
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();

    const int wg_m = get_group_id(1);
    const int wg_n = get_group_id(0);

    const int base_row = wg_m * 32;
    const int base_col = wg_n * 64;
    const int sg_col = base_col + sg_id * 16;

    // SLM double buffer for A: 2 * 32 rows * 32 cols = 2048 halfs = 4096 bytes
    // Using stride 32 (no padding needed since 32 halfs = 64 bytes, 16 banks of 4 bytes each = no conflicts for stride-32 access)
    #define SLM_S 32
    __local half slm_A[2 * 32 * SLM_S];

    // Accumulators: 4 groups of 8 rows = 32 rows total
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Cooperative A load mapping: 64 WIs load 32x32 = 1024 halfs, 16 per WI
    const int a_row = lid >> 1;            // 0..31
    const int a_col_base = (lid & 1) << 4; // 0 or 16
    const int a_row_global = base_row + a_row;

    // Preload first A tile (k=0..31) into buffer 0
    {
        __global const half* a_src = A + a_row_global * K + a_col_base;
        __local half* a_dst = slm_A + a_row * SLM_S + a_col_base;
        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K-loop: step by 64 (2x unrolled)
    // K=2048 divides evenly by 64, so 32 iterations total.
    // Structure per iteration:
    //   Phase 0: Read A from buffer 0, compute k..k+31
    //            Load A(k+32) into buffer 1
    //   Barrier
    //   Phase 1: Read A from buffer 1, compute k+32..k+63
    //            Load A(k+64) into buffer 0 (if not last iter)
    //   Barrier

    for (int k = 0; k < K; k += 64) {

        // ===== Phase 0: Compute from buffer 0, load next A into buffer 1 =====
        {
            __local const half* slm_base = slm_A;
            __global const half* b_ptr = B + k * N + sg_col + sg_lid;

            // Load A from SLM: k-step 0 (cols 0..15)
            short8 a00, a10, a20, a30;
            {
                __local const half* ap = slm_base + sg_lid;
                a00.s0 = as_short(ap[0*SLM_S]); a00.s1 = as_short(ap[1*SLM_S]);
                a00.s2 = as_short(ap[2*SLM_S]); a00.s3 = as_short(ap[3*SLM_S]);
                a00.s4 = as_short(ap[4*SLM_S]); a00.s5 = as_short(ap[5*SLM_S]);
                a00.s6 = as_short(ap[6*SLM_S]); a00.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a10.s0 = as_short(ap[0*SLM_S]); a10.s1 = as_short(ap[1*SLM_S]);
                a10.s2 = as_short(ap[2*SLM_S]); a10.s3 = as_short(ap[3*SLM_S]);
                a10.s4 = as_short(ap[4*SLM_S]); a10.s5 = as_short(ap[5*SLM_S]);
                a10.s6 = as_short(ap[6*SLM_S]); a10.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a20.s0 = as_short(ap[0*SLM_S]); a20.s1 = as_short(ap[1*SLM_S]);
                a20.s2 = as_short(ap[2*SLM_S]); a20.s3 = as_short(ap[3*SLM_S]);
                a20.s4 = as_short(ap[4*SLM_S]); a20.s5 = as_short(ap[5*SLM_S]);
                a20.s6 = as_short(ap[6*SLM_S]); a20.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a30.s0 = as_short(ap[0*SLM_S]); a30.s1 = as_short(ap[1*SLM_S]);
                a30.s2 = as_short(ap[2*SLM_S]); a30.s3 = as_short(ap[3*SLM_S]);
                a30.s4 = as_short(ap[4*SLM_S]); a30.s5 = as_short(ap[5*SLM_S]);
                a30.s6 = as_short(ap[6*SLM_S]); a30.s7 = as_short(ap[7*SLM_S]);
            }

            // Load B k-step 0 (k..k+15)
            int8 b0;
            {
                __global const half* bp = b_ptr;
                b0.s0 = as_int((half2)(bp[0], bp[N]));
                b0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // DPAS k-step 0
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            // Load A from SLM: k-step 1 (cols 16..31)
            short8 a01, a11, a21, a31;
            {
                __local const half* ap = slm_base + 16 + sg_lid;
                a01.s0 = as_short(ap[0*SLM_S]); a01.s1 = as_short(ap[1*SLM_S]);
                a01.s2 = as_short(ap[2*SLM_S]); a01.s3 = as_short(ap[3*SLM_S]);
                a01.s4 = as_short(ap[4*SLM_S]); a01.s5 = as_short(ap[5*SLM_S]);
                a01.s6 = as_short(ap[6*SLM_S]); a01.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a11.s0 = as_short(ap[0*SLM_S]); a11.s1 = as_short(ap[1*SLM_S]);
                a11.s2 = as_short(ap[2*SLM_S]); a11.s3 = as_short(ap[3*SLM_S]);
                a11.s4 = as_short(ap[4*SLM_S]); a11.s5 = as_short(ap[5*SLM_S]);
                a11.s6 = as_short(ap[6*SLM_S]); a11.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a21.s0 = as_short(ap[0*SLM_S]); a21.s1 = as_short(ap[1*SLM_S]);
                a21.s2 = as_short(ap[2*SLM_S]); a21.s3 = as_short(ap[3*SLM_S]);
                a21.s4 = as_short(ap[4*SLM_S]); a21.s5 = as_short(ap[5*SLM_S]);
                a21.s6 = as_short(ap[6*SLM_S]); a21.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a31.s0 = as_short(ap[0*SLM_S]); a31.s1 = as_short(ap[1*SLM_S]);
                a31.s2 = as_short(ap[2*SLM_S]); a31.s3 = as_short(ap[3*SLM_S]);
                a31.s4 = as_short(ap[4*SLM_S]); a31.s5 = as_short(ap[5*SLM_S]);
                a31.s6 = as_short(ap[6*SLM_S]); a31.s7 = as_short(ap[7*SLM_S]);
            }

            // Load B k-step 1 (k+16..k+31)
            int8 b1;
            {
                __global const half* bp = b_ptr + 16*N;
                b1.s0 = as_int((half2)(bp[0], bp[N]));
                b1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // DPAS k-step 1
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);

            // Load A(k+32) into buffer 1 (always valid since k+32 <= K-32 = 2016 for k < K)
            {
                __global const half* a_src = A + a_row_global * K + (k + 32) + a_col_base;
                __local half* a_dst = slm_A + 32*SLM_S + a_row * SLM_S + a_col_base;
                half8 v0 = vload8(0, a_src);
                half8 v1 = vload8(1, a_src);
                vstore8(v0, 0, a_dst);
                vstore8(v1, 0, a_dst + 8);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ===== Phase 1: Compute from buffer 1, load next A into buffer 0 =====
        {
            __local const half* slm_base = slm_A + 32*SLM_S;
            __global const half* b_ptr = B + (k+32) * N + sg_col + sg_lid;

            // Load A from SLM: k-step 0 (cols 0..15)
            short8 a00, a10, a20, a30;
            {
                __local const half* ap = slm_base + sg_lid;
                a00.s0 = as_short(ap[0*SLM_S]); a00.s1 = as_short(ap[1*SLM_S]);
                a00.s2 = as_short(ap[2*SLM_S]); a00.s3 = as_short(ap[3*SLM_S]);
                a00.s4 = as_short(ap[4*SLM_S]); a00.s5 = as_short(ap[5*SLM_S]);
                a00.s6 = as_short(ap[6*SLM_S]); a00.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a10.s0 = as_short(ap[0*SLM_S]); a10.s1 = as_short(ap[1*SLM_S]);
                a10.s2 = as_short(ap[2*SLM_S]); a10.s3 = as_short(ap[3*SLM_S]);
                a10.s4 = as_short(ap[4*SLM_S]); a10.s5 = as_short(ap[5*SLM_S]);
                a10.s6 = as_short(ap[6*SLM_S]); a10.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a20.s0 = as_short(ap[0*SLM_S]); a20.s1 = as_short(ap[1*SLM_S]);
                a20.s2 = as_short(ap[2*SLM_S]); a20.s3 = as_short(ap[3*SLM_S]);
                a20.s4 = as_short(ap[4*SLM_S]); a20.s5 = as_short(ap[5*SLM_S]);
                a20.s6 = as_short(ap[6*SLM_S]); a20.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a30.s0 = as_short(ap[0*SLM_S]); a30.s1 = as_short(ap[1*SLM_S]);
                a30.s2 = as_short(ap[2*SLM_S]); a30.s3 = as_short(ap[3*SLM_S]);
                a30.s4 = as_short(ap[4*SLM_S]); a30.s5 = as_short(ap[5*SLM_S]);
                a30.s6 = as_short(ap[6*SLM_S]); a30.s7 = as_short(ap[7*SLM_S]);
            }

            // Load B k-step 0 (k+32..k+47)
            int8 b0;
            {
                __global const half* bp = b_ptr;
                b0.s0 = as_int((half2)(bp[0], bp[N]));
                b0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // DPAS k-step 0
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            // Load A from SLM: k-step 1 (cols 16..31)
            short8 a01, a11, a21, a31;
            {
                __local const half* ap = slm_base + 16 + sg_lid;
                a01.s0 = as_short(ap[0*SLM_S]); a01.s1 = as_short(ap[1*SLM_S]);
                a01.s2 = as_short(ap[2*SLM_S]); a01.s3 = as_short(ap[3*SLM_S]);
                a01.s4 = as_short(ap[4*SLM_S]); a01.s5 = as_short(ap[5*SLM_S]);
                a01.s6 = as_short(ap[6*SLM_S]); a01.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a11.s0 = as_short(ap[0*SLM_S]); a11.s1 = as_short(ap[1*SLM_S]);
                a11.s2 = as_short(ap[2*SLM_S]); a11.s3 = as_short(ap[3*SLM_S]);
                a11.s4 = as_short(ap[4*SLM_S]); a11.s5 = as_short(ap[5*SLM_S]);
                a11.s6 = as_short(ap[6*SLM_S]); a11.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a21.s0 = as_short(ap[0*SLM_S]); a21.s1 = as_short(ap[1*SLM_S]);
                a21.s2 = as_short(ap[2*SLM_S]); a21.s3 = as_short(ap[3*SLM_S]);
                a21.s4 = as_short(ap[4*SLM_S]); a21.s5 = as_short(ap[5*SLM_S]);
                a21.s6 = as_short(ap[6*SLM_S]); a21.s7 = as_short(ap[7*SLM_S]);
                ap += 8*SLM_S;
                a31.s0 = as_short(ap[0*SLM_S]); a31.s1 = as_short(ap[1*SLM_S]);
                a31.s2 = as_short(ap[2*SLM_S]); a31.s3 = as_short(ap[3*SLM_S]);
                a31.s4 = as_short(ap[4*SLM_S]); a31.s5 = as_short(ap[5*SLM_S]);
                a31.s6 = as_short(ap[6*SLM_S]); a31.s7 = as_short(ap[7*SLM_S]);
            }

            // Load B k-step 1 (k+48..k+63)
            int8 b1;
            {
                __global const half* bp = b_ptr + 16*N;
                b1.s0 = as_int((half2)(bp[0], bp[N]));
                b1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // DPAS k-step 1
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);

            // Load A(k+64) into buffer 0 for next iteration
            // K=2048, last k=1984, k+64=2048=K so skip on last iter
            if (k + 64 < K) {
                __global const half* a_src = A + a_row_global * K + (k + 64) + a_col_base;
                __local half* a_dst = slm_A + a_row * SLM_S + a_col_base;
                half8 v0 = vload8(0, a_src);
                half8 v1 = vload8(1, a_src);
                vstore8(v0, 0, a_dst);
                vstore8(v1, 0, a_dst + 8);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write C: each WI writes 32 rows to its column
    __global half* c_ptr = C + base_row * N + sg_col + sg_lid;
    c_ptr[0*N] = convert_half(acc0.s0); c_ptr[1*N] = convert_half(acc0.s1);
    c_ptr[2*N] = convert_half(acc0.s2); c_ptr[3*N] = convert_half(acc0.s3);
    c_ptr[4*N] = convert_half(acc0.s4); c_ptr[5*N] = convert_half(acc0.s5);
    c_ptr[6*N] = convert_half(acc0.s6); c_ptr[7*N] = convert_half(acc0.s7);
    c_ptr[8*N] = convert_half(acc1.s0); c_ptr[9*N] = convert_half(acc1.s1);
    c_ptr[10*N] = convert_half(acc1.s2); c_ptr[11*N] = convert_half(acc1.s3);
    c_ptr[12*N] = convert_half(acc1.s4); c_ptr[13*N] = convert_half(acc1.s5);
    c_ptr[14*N] = convert_half(acc1.s6); c_ptr[15*N] = convert_half(acc1.s7);
    c_ptr[16*N] = convert_half(acc2.s0); c_ptr[17*N] = convert_half(acc2.s1);
    c_ptr[18*N] = convert_half(acc2.s2); c_ptr[19*N] = convert_half(acc2.s3);
    c_ptr[20*N] = convert_half(acc2.s4); c_ptr[21*N] = convert_half(acc2.s5);
    c_ptr[22*N] = convert_half(acc2.s6); c_ptr[23*N] = convert_half(acc2.s7);
    c_ptr[24*N] = convert_half(acc3.s0); c_ptr[25*N] = convert_half(acc3.s1);
    c_ptr[26*N] = convert_half(acc3.s2); c_ptr[27*N] = convert_half(acc3.s3);
    c_ptr[28*N] = convert_half(acc3.s4); c_ptr[29*N] = convert_half(acc3.s5);
    c_ptr[30*N] = convert_half(acc3.s6); c_ptr[31*N] = convert_half(acc3.s7);
}
```