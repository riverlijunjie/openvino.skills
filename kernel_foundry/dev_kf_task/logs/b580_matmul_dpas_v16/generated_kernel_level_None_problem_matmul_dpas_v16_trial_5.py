# Analysis

## Previous Kernel Issues

The last tested kernel (34ms) had a critical issue: it used `READ_A_BLOCK` macro with `sg_lid * 2` addressing, which reads pairs of halfs (crosspack=2 format). However, this conflicts with how the best kernel (1.3ms) reads A - where each WI reads a single half per row at column `sg_lid`. The 34ms kernel's macro approach with compound literals for B reads (`*((__global const int*)&(half2){...})`) likely caused the compiler to generate very poor code (possibly scalar operations instead of vectorized).

The best kernel (1.3ms) uses the correct approach:
- A from SLM: each WI reads column `sg_lid` (one half per row), giving short8 per 8-row block
- B from global: manual construction of int8 by reading pairs of rows and packing as half2→int

## Key Optimizations to Apply

1. **Remove the `if (k + 64 < K)` branch** - K=2048 divides evenly by 64, so the last iteration's prefetch is unnecessary but the branch adds overhead
2. **Interleave A loads with DPAS** - Load A for k-step 1 while DPAS is executing for k-step 0
3. **Interleave B loads with DPAS** - Start loading B1 while DPAS for B0 is executing
4. **Use SLM stride of 34** to avoid bank conflicts (32 = power of 2 causes conflicts)
5. **Remove redundant barrier** - restructure to minimize barriers
6. **Vectorized C store** - use intel_sub_group_block_write for C output

Let me build on the best kernel (1.3ms) with these micro-optimizations:

- SLM stride 34 (bank conflict avoidance)
- Remove conditional in A prefetch (K%64==0 guaranteed)
- Better interleaving of loads and compute
- Unrolled loop body with loads overlapping DPAS latency

```OCL
// GEMM: C[M,N] = A[M,K] * B[K,N], all half, f32 accumulation
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// 4 subgroups of 16 WIs. Tile: 32x64x32. A in SLM (double-buffered), B from global.
// K=2048, M=2048, N=2048 - all divisible by 64.
// SLM stride = 34 to avoid bank conflicts.

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

    // SLM double buffer for A: 2 * 32 rows * 34 cols (stride=34 for bank conflict avoidance)
    // Total: 2 * 32 * 34 = 2176 halfs = 4352 bytes
    #define SLM_S 34
    __local half slm_A[2 * 32 * SLM_S];

    // Accumulators: 4 blocks of 8 rows each
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Cooperative A load: 64 WIs, 32 rows x 32 cols = 1024 halfs, 16 per WI
    const int a_row = lid >> 1;            // 0..31
    const int a_col_base = (lid & 1) << 4; // 0 or 16

    // Preload first A tile (k=0) into buffer 0
    {
        __global const half* a_src = A + (base_row + a_row) * K + a_col_base;
        __local half* a_dst = slm_A + a_row * SLM_S + a_col_base;
        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K-loop: step by 64 (2x unrolled), K=2048 divides by 64 exactly (32 iterations)
    // Last iteration: k=1984, we load A at k+32=2016 (valid), and k+64=2048 (out of bounds)
    // So we handle the last iteration specially by not loading A at k+64
    const int K_end = K - 64; // last iteration where we still need to prefetch

    for (int k = 0; k < K; k += 64) {
        // ===== Phase 0: Compute from buffer 0, load A(k+32) into buffer 1 =====
        {
            __local const half* slm_base = slm_A;
            __global const half* b_ptr = B + k * N + sg_col + sg_lid;

            // Load B k-step 0 (k..k+15) - interleave with A SLM reads
            int8 b0;
            {
                __global const half* bp = b_ptr;
                b0.s0 = as_int((half2)(bp[0*N], bp[1*N]));
                b0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // Read A from SLM for k-step 0: rows 0-31, cols 0-15
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

            // DPAS k-step 0
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);

            // Start loading B k-step 1 while DPAS is in flight
            int8 b1;
            {
                __global const half* bp = b_ptr + 16*N;
                b1.s0 = as_int((half2)(bp[0*N], bp[1*N]));
                b1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            // Read A from SLM for k-step 1: rows 0-31, cols 16-31
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

            // DPAS k-step 1
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);

            // Load A(k+32) into buffer 1
            {
                __global const half* a_src = A + (base_row + a_row) * K + (k + 32) + a_col_base;
                __local half* a_dst = slm_A + 32*SLM_S + a_row * SLM_S + a_col_base;
                half8 v0 = vload8(0, a_src);
                half8 v1 = vload8(1, a_src);
                vstore8(v0, 0, a_dst);
                vstore8(v1, 0, a_dst + 8);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ===== Phase 1: Compute from buffer 1, load A(k+64) into buffer 0 =====
        {
            __local const half* slm_base = slm_A + 32*SLM_S;
            __global const half* b_ptr = B + (k+32) * N + sg_col + sg_lid;

            // Load B k-step 0 (k+32..k+47)
            int8 b0;
            {
                __global const half* bp = b_ptr;
                b0.s0 = as_int((half2)(bp[0*N], bp[1*N]));
                b0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // Read A from SLM for k-step 0
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

            // DPAS k-step 0
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);

            // Start loading B k-step 1 while DPAS in flight
            int8 b1;
            {
                __global const half* bp = b_ptr + 16*N;
                b1.s0 = as_int((half2)(bp[0*N], bp[1*N]));
                b1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            // Read A from SLM for k-step 1
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

            // DPAS k-step 1
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);

            // Load A(k+64) into buffer 0 - skip on last iteration
            if (k < K_end) {
                __global const half* a_src = A + (base_row + a_row) * K + (k + 64) + a_col_base;
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