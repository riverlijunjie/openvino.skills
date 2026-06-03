## Analysis

Looking at the previous versions, both achieve 1.300ms with the same architecture (32×64×32 tile, 4 subgroups of 16 WIs, A in SLM double-buffered, B from global). The key bottlenecks are:

1. **B loads are scattered**: Each B element requires a separate global memory access with stride N between rows. This creates 16 scalar loads per int8 B vector, which is very inefficient.

2. **SLM reads are scalar**: Each short8 A vector requires 8 individual SLM reads with stride SLM_STRIDE, causing potential bank conflicts.

3. **Loop overhead**: The branch `if (k + 64 < K)` adds unnecessary overhead since K=2048 divides evenly by 64.

4. **Instruction scheduling**: The A loads from SLM, B loads from global, and DPAS are not optimally interleaved. Loading all A and B before DPAS means the systolic unit sits idle during loads.

**Proposed improvements:**

1. **Remove the K-boundary check** - K=2048 divides by 64, so the last iteration's prefetch writes to buffer 0 which is never read again.

2. **Better interleaving of loads and compute** - Start loading B for the next k-step while DPAS is executing for the current step. This overlaps memory latency with compute.

3. **Use SLM stride of 34 (32+2)** to reduce bank conflicts - SLM has 32 banks of 4 bytes each. With stride 32 (64 bytes), consecutive rows map to the same banks. Adding 2 half padding (4 bytes) shifts each row by one bank.

4. **Preload B for next k-step during current DPAS** - Since DPAS has ~20 cycle latency, we can issue B loads during that time.

5. **Vectorized C store** - Use intel_sub_group_block_write for C output where possible, or at minimum batch the conversions.

## Improved OCL code

```OCL
// GEMM: C[M,N] = A[M,K] * B[K,N], all half, f32 accumulation
// Launch: GWS = (N/64 * 64, M/32), LWS = (64, 1)
// 4 subgroups of 16 WIs. Tile: 32x64x32. A in SLM (double-buffered), B from global/L2.
// K=2048, M=2048, N=2048 - all divisible by 64, no remainder handling needed.
// SLM stride = 34 to reduce bank conflicts (32 data + 2 padding halfs per row).

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_STRIDE 34
#define BUF_SIZE (TILE_M * SLM_STRIDE)

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

    const int base_row = wg_m * TILE_M;
    const int base_col = wg_n * TILE_N;
    const int sg_col = base_col + sg_id * 16;

    // SLM double buffer for A: 2 * 32 * 34 = 2176 halfs = 4352 bytes
    __local half slm_A[2 * BUF_SIZE];

    // Accumulators: 4 groups of 8 rows = 32 rows total
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Cooperative A load: 64 WIs load 32x32 = 1024 halfs, 16 per WI
    const int a_row = lid >> 1;            // 0..31
    const int a_col_base = (lid & 1) << 4; // 0 or 16

    // Preload first A tile (k=0) into buffer 0
    {
        __global const half* a_src = A + (base_row + a_row) * K + a_col_base;
        __local half* a_dst = slm_A + a_row * SLM_STRIDE + a_col_base;
        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main K-loop: step by 64 (2x unrolled), K=2048 => 32 iterations
    // No boundary check needed: K divides evenly by 64
    for (int k = 0; k < K; k += 64) {

        // ===== Phase 0: Compute from buffer 0, load A(k+32) into buffer 1 =====
        {
            __local const half* slm_base = slm_A;
            __global const half* b_base = B + k * N + sg_col + sg_lid;

            // Load B k-step 0 (k..k+15) - start early to overlap with SLM reads
            int8 b0;
            {
                __global const half* bp = b_base;
                b0.s0 = as_int((half2)(bp[0], bp[N]));
                b0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // Read A from SLM for k-step 0 (cols 0..15)
            short8 a00, a10, a20, a30;
            {
                __local const half* ap = slm_base + sg_lid;
                a00.s0 = as_short(ap[0*SLM_STRIDE]); a00.s1 = as_short(ap[1*SLM_STRIDE]);
                a00.s2 = as_short(ap[2*SLM_STRIDE]); a00.s3 = as_short(ap[3*SLM_STRIDE]);
                a00.s4 = as_short(ap[4*SLM_STRIDE]); a00.s5 = as_short(ap[5*SLM_STRIDE]);
                a00.s6 = as_short(ap[6*SLM_STRIDE]); a00.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a10.s0 = as_short(ap[0*SLM_STRIDE]); a10.s1 = as_short(ap[1*SLM_STRIDE]);
                a10.s2 = as_short(ap[2*SLM_STRIDE]); a10.s3 = as_short(ap[3*SLM_STRIDE]);
                a10.s4 = as_short(ap[4*SLM_STRIDE]); a10.s5 = as_short(ap[5*SLM_STRIDE]);
                a10.s6 = as_short(ap[6*SLM_STRIDE]); a10.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a20.s0 = as_short(ap[0*SLM_STRIDE]); a20.s1 = as_short(ap[1*SLM_STRIDE]);
                a20.s2 = as_short(ap[2*SLM_STRIDE]); a20.s3 = as_short(ap[3*SLM_STRIDE]);
                a20.s4 = as_short(ap[4*SLM_STRIDE]); a20.s5 = as_short(ap[5*SLM_STRIDE]);
                a20.s6 = as_short(ap[6*SLM_STRIDE]); a20.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a30.s0 = as_short(ap[0*SLM_STRIDE]); a30.s1 = as_short(ap[1*SLM_STRIDE]);
                a30.s2 = as_short(ap[2*SLM_STRIDE]); a30.s3 = as_short(ap[3*SLM_STRIDE]);
                a30.s4 = as_short(ap[4*SLM_STRIDE]); a30.s5 = as_short(ap[5*SLM_STRIDE]);
                a30.s6 = as_short(ap[6*SLM_STRIDE]); a30.s7 = as_short(ap[7*SLM_STRIDE]);
            }

            // Start loading B k-step 1 early (overlaps with DPAS latency)
            int8 b1;
            {
                __global const half* bp = b_base + 16*N;
                b1.s0 = as_int((half2)(bp[0], bp[N]));
                b1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // DPAS k-step 0
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            // Read A from SLM for k-step 1 (cols 16..31)
            short8 a01, a11, a21, a31;
            {
                __local const half* ap = slm_base + 16 + sg_lid;
                a01.s0 = as_short(ap[0*SLM_STRIDE]); a01.s1 = as_short(ap[1*SLM_STRIDE]);
                a01.s2 = as_short(ap[2*SLM_STRIDE]); a01.s3 = as_short(ap[3*SLM_STRIDE]);
                a01.s4 = as_short(ap[4*SLM_STRIDE]); a01.s5 = as_short(ap[5*SLM_STRIDE]);
                a01.s6 = as_short(ap[6*SLM_STRIDE]); a01.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a11.s0 = as_short(ap[0*SLM_STRIDE]); a11.s1 = as_short(ap[1*SLM_STRIDE]);
                a11.s2 = as_short(ap[2*SLM_STRIDE]); a11.s3 = as_short(ap[3*SLM_STRIDE]);
                a11.s4 = as_short(ap[4*SLM_STRIDE]); a11.s5 = as_short(ap[5*SLM_STRIDE]);
                a11.s6 = as_short(ap[6*SLM_STRIDE]); a11.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a21.s0 = as_short(ap[0*SLM_STRIDE]); a21.s1 = as_short(ap[1*SLM_STRIDE]);
                a21.s2 = as_short(ap[2*SLM_STRIDE]); a21.s3 = as_short(ap[3*SLM_STRIDE]);
                a21.s4 = as_short(ap[4*SLM_STRIDE]); a21.s5 = as_short(ap[5*SLM_STRIDE]);
                a21.s6 = as_short(ap[6*SLM_STRIDE]); a21.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a31.s0 = as_short(ap[0*SLM_STRIDE]); a31.s1 = as_short(ap[1*SLM_STRIDE]);
                a31.s2 = as_short(ap[2*SLM_STRIDE]); a31.s3 = as_short(ap[3*SLM_STRIDE]);
                a31.s4 = as_short(ap[4*SLM_STRIDE]); a31.s5 = as_short(ap[5*SLM_STRIDE]);
                a31.s6 = as_short(ap[6*SLM_STRIDE]); a31.s7 = as_short(ap[7*SLM_STRIDE]);
            }

            // DPAS k-step 1 - interleaved with A prefetch to SLM
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);

            // Load A(k+32) into buffer 1 - overlaps with remaining DPAS
            {
                __global const half* a_src = A + (base_row + a_row) * K + (k + 32) + a_col_base;
                __local half* a_dst = slm_A + BUF_SIZE + a_row * SLM_STRIDE + a_col_base;
                half8 v0 = vload8(0, a_src);
                half8 v1 = vload8(1, a_src);
                vstore8(v0, 0, a_dst);
                vstore8(v1, 0, a_dst + 8);
            }

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ===== Phase 1: Compute from buffer 1, prefetch A(k+64) into buffer 0 =====
        {
            __local const half* slm_base = slm_A + BUF_SIZE;
            __global const half* b_base = B + (k+32) * N + sg_col + sg_lid;

            // Load B k-step 0 early
            int8 b0;
            {
                __global const half* bp = b_base;
                b0.s0 = as_int((half2)(bp[0], bp[N]));
                b0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            short8 a00, a10, a20, a30;
            {
                __local const half* ap = slm_base + sg_lid;
                a00.s0 = as_short(ap[0*SLM_STRIDE]); a00.s1 = as_short(ap[1*SLM_STRIDE]);
                a00.s2 = as_short(ap[2*SLM_STRIDE]); a00.s3 = as_short(ap[3*SLM_STRIDE]);
                a00.s4 = as_short(ap[4*SLM_STRIDE]); a00.s5 = as_short(ap[5*SLM_STRIDE]);
                a00.s6 = as_short(ap[6*SLM_STRIDE]); a00.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a10.s0 = as_short(ap[0*SLM_STRIDE]); a10.s1 = as_short(ap[1*SLM_STRIDE]);
                a10.s2 = as_short(ap[2*SLM_STRIDE]); a10.s3 = as_short(ap[3*SLM_STRIDE]);
                a10.s4 = as_short(ap[4*SLM_STRIDE]); a10.s5 = as_short(ap[5*SLM_STRIDE]);
                a10.s6 = as_short(ap[6*SLM_STRIDE]); a10.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a20.s0 = as_short(ap[0*SLM_STRIDE]); a20.s1 = as_short(ap[1*SLM_STRIDE]);
                a20.s2 = as_short(ap[2*SLM_STRIDE]); a20.s3 = as_short(ap[3*SLM_STRIDE]);
                a20.s4 = as_short(ap[4*SLM_STRIDE]); a20.s5 = as_short(ap[5*SLM_STRIDE]);
                a20.s6 = as_short(ap[6*SLM_STRIDE]); a20.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a30.s0 = as_short(ap[0*SLM_STRIDE]); a30.s1 = as_short(ap[1*SLM_STRIDE]);
                a30.s2 = as_short(ap[2*SLM_STRIDE]); a30.s3 = as_short(ap[3*SLM_STRIDE]);
                a30.s4 = as_short(ap[4*SLM_STRIDE]); a30.s5 = as_short(ap[5*SLM_STRIDE]);
                a30.s6 = as_short(ap[6*SLM_STRIDE]); a30.s7 = as_short(ap[7*SLM_STRIDE]);
            }

            // Start loading B k-step 1 early
            int8 b1;
            {
                __global const half* bp = b_base + 16*N;
                b1.s0 = as_int((half2)(bp[0], bp[N]));
                b1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
                b1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
                b1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
                b1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
                b1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
                b1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
                b1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
            }

            // DPAS k-step 0
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a00, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a10, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a20, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a30, b0, acc3);

            // Read A from SLM for k-step 1
            short8 a01, a11, a21, a31;
            {
                __local const half* ap = slm_base + 16 + sg_lid;
                a01.s0 = as_short(ap[0*SLM_STRIDE]); a01.s1 = as_short(ap[1*SLM_STRIDE]);
                a01.s2 = as_short(ap[2*SLM_STRIDE]); a01.s3 = as_short(ap[3*SLM_STRIDE]);
                a01.s4 = as_short(ap[4*SLM_STRIDE]); a01.s5 = as_short(ap[5*SLM_STRIDE]);
                a01.s6 = as_short(ap[6*SLM_STRIDE]); a01.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a11.s0 = as_short(ap[0*SLM_STRIDE]); a11.s1 = as_short(ap[1*SLM_STRIDE]);
                a11.s2 = as_short(ap[2*SLM_STRIDE]); a11.s3 = as_short(ap[3*SLM_STRIDE]);
                a11.s4 = as_short(ap[4*SLM_STRIDE]); a11.s5 = as_short(ap[5*SLM_STRIDE]);
                a11.s6 = as_short(ap[6*SLM_STRIDE]); a11.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a21.s0 = as_short(ap[0*SLM_STRIDE]); a21.s1 = as_short(ap[1*SLM_STRIDE]);
                a21.s2 = as_short(ap[2*SLM_STRIDE]); a21.s3 = as_short(ap[3*SLM_STRIDE]);
                a21.s4 = as_short(ap[4*SLM_STRIDE]); a21.s5 = as_short(ap[5*SLM_STRIDE]);
                a21.s6 = as_short(ap[6*SLM_STRIDE]); a21.s7 = as_short(ap[7*SLM_STRIDE]);
                ap += 8*SLM_STRIDE;
                a31.s0 = as_short(ap[0*SLM_STRIDE]); a31.s1 = as_short(ap[1*SLM_STRIDE]);
                a31.s2 = as_short(ap[2*SLM_STRIDE]); a31.s3 = as_short(ap[3*SLM_STRIDE]);
                a31.s4 = as_short(ap[4*SLM_STRIDE]); a31.s5 = as_short(ap[5*SLM_STRIDE]);
                a31.s6 = as_short(ap[6*SLM_STRIDE]); a31.s7 = as_short(ap[7*SLM_STRIDE]);
            }

            // DPAS k-step 1 interleaved with next A prefetch
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a01, b1, acc0);

            // Load A(k+64) into buffer 0 - no guard needed, last iter writes harmlessly
            {
                int next_k = k + 64;
                // Use min to avoid OOB on last iteration - clamp to last valid tile
                int safe_k = min(next_k, K - TILE_K);
                __global const half* a_src = A + (base_row + a_row) * K + safe_k + a_col_base;
                __local half* a_dst = slm_A + a_row * SLM_STRIDE + a_col_base;
                half8 v0 = vload8(0, a_src);
                half8 v1 = vload8(1, a_src);
                vstore8(v0, 0, a_dst);
                vstore8(v1, 0, a_dst + 8);
            }

            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a11, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a21, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a31, b1, acc3);
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