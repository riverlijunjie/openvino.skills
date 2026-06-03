## Analysis

Looking at the previous versions:

1. **Version 1 (1.450ms)**: Basic implementation with scalar SLM reads and scalar B loads. Double-buffered but with inefficient memory access patterns.

2. **Version 2 (1.080ms)**: Improved with `intel_sub_group_block_read_us8` for SLM reads and 2x K-loop unrolling. Used block writes for C output.

3. **Version 3 (1.020ms)**: Best so far. Key improvement: interleaved A loading with compute (load next A tile between DPAS calls). Separated last iteration to avoid branch in loop.

**Key bottlenecks in Version 3:**
1. **B loads are scattered** - each lane reads individual half values from global memory with stride N between rows. This is 16 scalar loads per int8 B register (16 pairs × 16 lanes = 256 individual loads per K16 step).
2. **Loop overhead** - still has per-iteration overhead with buffer swapping and barrier.
3. **No prefetching** of B data - B is loaded on-demand from global/L2.
4. **SLM bank conflicts** - stride=16 means all 16 lanes access the same bank offset pattern.

**Proposed improvements:**
1. **SLM stride padding** - Use stride=20 (16+4) to shift bank access patterns and reduce conflicts.
2. **Better interleaving** - Load B for K16 step 1 while DPAS for step 0 is executing, and vice versa.
3. **Prefetch B** - Use `intel_sub_group_block_read` on B data ahead of time (or explicit prefetch).
4. **Tighter loop structure** - Combine the A store and barrier more efficiently.
5. **Remove last-iteration special case** - Process all iterations uniformly by loading A one step ahead and handling the boundary differently.

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, f32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM double-buffered, B from global
// Launch: LWS=(16,4,1), GWS=(N/4, M/8, 1) i.e. (N/64*16, M/32*4, 1)
// Subgroup size: 16
// For M=N=K=4096: GWS=(1024,512,1), LWS=(16,4,1)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_local_id(1);       // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15

    const int wg_row = get_group_id(1) * 32;
    const int wg_col = get_group_id(0) * 64;
    const int sg_col = wg_col + sg_id * 16;

    // SLM: double-buffered, stride=20 to reduce bank conflicts
    // Each buffer stores 32 rows x 32 cols in two K16 blocks
    // Layout: [k16_block][row][col_within_16] with stride=20 per row
    // Buffer size: 2 * 32 * 20 = 1280 halfs per K16 block, 2560 per buffer
    #define SLM_STRIDE 20
    #define SLM_K16_BLOCK (32 * SLM_STRIDE)  // 640 halfs per K16 block
    #define SLM_BUF_SIZE (2 * SLM_K16_BLOCK) // 1280 halfs per buffer (two K16 blocks)
    __local half slm_A[2 * SLM_BUF_SIZE];    // 2560 halfs = 5120 bytes total

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int lid = sg_id * 16 + sg_lid;
    // 64 WIs load 32 rows x 32 cols = 1024 halfs
    // Each WI loads 16 halfs (one half-row of K16)
    const int a_load_row = lid / 2;       // 0..31
    const int a_load_khalf = lid & 1;     // 0 or 1 (which K16 block)

    // Preload first A tile into SLM buffer 0
    {
        __global const half* a_ptr = A + (wg_row + a_load_row) * K + a_load_khalf * 16;
        int slm_off = a_load_khalf * SLM_K16_BLOCK + a_load_row * SLM_STRIDE;
        half8 v0 = vload8(0, a_ptr);
        half8 v1 = vload8(1, a_ptr);
        vstore8(v0, 0, &slm_A[slm_off]);
        vstore8(v1, 0, &slm_A[slm_off + 8]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;
    const int k_iterations = K / 32;  // K=2048 or 4096, always divisible by 32

    for (int ki = 0; ki < k_iterations; ki++) {
        const int k_offset = ki * 32;
        const int slm_base = buf * SLM_BUF_SIZE;

        // B base pointer for this K-tile
        __global const half* b_base = B + k_offset * N + sg_col + sg_lid;

        // === K16 step 0 ===
        // Load A from SLM (padded stride=20, block reads use stride=16 within the 20-wide row)
        short8 a0, a1, a2, a3;
        {
            // For block_read_us8: reads 8 consecutive ushorts per lane, stride = SLM_STRIDE
            // We need to read column sg_lid from 8 consecutive rows
            // With stride=20, row i starts at offset i*20
            // block_read_us8 reads 8 elements with stride = 1 (contiguous)
            // So we must use scalar reads with stride
            __local const half* a_slm = &slm_A[slm_base + sg_lid];
            a0.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a0.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a0.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a0.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a0.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a0.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a0.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a0.s7 = as_short(a_slm[7*SLM_STRIDE]);

            a_slm += 8*SLM_STRIDE;
            a1.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a1.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a1.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a1.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a1.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a1.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a1.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a1.s7 = as_short(a_slm[7*SLM_STRIDE]);

            a_slm += 8*SLM_STRIDE;
            a2.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a2.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a2.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a2.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a2.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a2.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a2.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a2.s7 = as_short(a_slm[7*SLM_STRIDE]);

            a_slm += 8*SLM_STRIDE;
            a3.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a3.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a3.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a3.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a3.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a3.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a3.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a3.s7 = as_short(a_slm[7*SLM_STRIDE]);
        }

        // Load B for K16 step 0 - pack pairs into VNNI format
        int8 b_reg0;
        {
            __global const half* bp = b_base;
            b_reg0.s0 = as_int((half2)(bp[0*N], bp[1*N]));
            b_reg0.s1 = as_int((half2)(bp[2*N], bp[3*N]));
            b_reg0.s2 = as_int((half2)(bp[4*N], bp[5*N]));
            b_reg0.s3 = as_int((half2)(bp[6*N], bp[7*N]));
            b_reg0.s4 = as_int((half2)(bp[8*N], bp[9*N]));
            b_reg0.s5 = as_int((half2)(bp[10*N], bp[11*N]));
            b_reg0.s6 = as_int((half2)(bp[12*N], bp[13*N]));
            b_reg0.s7 = as_int((half2)(bp[14*N], bp[15*N]));
        }

        // DPAS K16 step 0
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg0, acc1);

        // Interleave: start loading next A tile while DPAS is in flight
        half8 a_next_v0, a_next_v1;
        int next_k = k_offset + 32;
        int next_buf = 1 - buf;
        if (ki < k_iterations - 1) {
            __global const half* a_ptr = A + (wg_row + a_load_row) * K + next_k + a_load_khalf * 16;
            a_next_v0 = vload8(0, a_ptr);
            a_next_v1 = vload8(1, a_ptr);
        }

        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg0, acc3);

        // === K16 step 1 ===
        short8 a0_1, a1_1, a2_1, a3_1;
        {
            __local const half* a_slm = &slm_A[slm_base + SLM_K16_BLOCK + sg_lid];
            a0_1.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a0_1.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a0_1.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a0_1.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a0_1.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a0_1.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a0_1.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a0_1.s7 = as_short(a_slm[7*SLM_STRIDE]);

            a_slm += 8*SLM_STRIDE;
            a1_1.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a1_1.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a1_1.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a1_1.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a1_1.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a1_1.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a1_1.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a1_1.s7 = as_short(a_slm[7*SLM_STRIDE]);

            a_slm += 8*SLM_STRIDE;
            a2_1.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a2_1.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a2_1.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a2_1.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a2_1.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a2_1.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a2_1.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a2_1.s7 = as_short(a_slm[7*SLM_STRIDE]);

            a_slm += 8*SLM_STRIDE;
            a3_1.s0 = as_short(a_slm[0*SLM_STRIDE]);
            a3_1.s1 = as_short(a_slm[1*SLM_STRIDE]);
            a3_1.s2 = as_short(a_slm[2*SLM_STRIDE]);
            a3_1.s3 = as_short(a_slm[3*SLM_STRIDE]);
            a3_1.s4 = as_short(a_slm[4*SLM_STRIDE]);
            a3_1.s5 = as_short(a_slm[5*SLM_STRIDE]);
            a3_1.s6 = as_short(a_slm[6*SLM_STRIDE]);
            a3_1.s7 = as_short(a_slm[7*SLM_STRIDE]);
        }

        // Load B for K16 step 1
        int8 b_reg1;
        {
            __global const half* bp = b_base + 16 * N;
            b_reg1.s0 = as_int((half2)(bp[0*N], bp[1*N]));
            b_reg1.s1 = as_int((half2)(bp[2*N], bp[3*N]));
            b_reg1.s2 = as_int((half2)(bp[4*N], bp[5*N]));
            b_reg1.s3 = as_int((half2)(bp[6*N], bp[7*N]));
            b_reg1.s4 = as_int((half2)(bp[8*N], bp[9*N]));
            b_reg1.s5 = as_int((half2)(bp[10*N], bp[11*N]));
            b_reg1.s6 = as_int((half2)(bp[12*N], bp[13*N]));
            b_reg1.s7 = as_int((half2)(bp[14*N], bp[15*N]));
        }

        // DPAS K16 step 1
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_1, b_reg1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_1, b_reg1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_1, b_reg1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_1, b_reg1, acc3);

        // Store next A tile to SLM (if not last iteration)
        if (ki < k_iterations - 1) {
            int slm_off = next_buf * SLM_BUF_SIZE + a_load_khalf * SLM_K16_BLOCK + a_load_row * SLM_STRIDE;
            vstore8(a_next_v0, 0, &slm_A[slm_off]);
            vstore8(a_next_v1, 0, &slm_A[slm_off + 8]);
            buf = next_buf;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // === Store C using subgroup block writes ===
    __global half* c_base = C + wg_row * N + sg_col;

    intel_sub_group_block_write_us((__global ushort*)&c_base[0*N], as_ushort(convert_half(acc0.s0)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[1*N], as_ushort(convert_half(acc0.s1)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[2*N], as_ushort(convert_half(acc0.s2)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[3*N], as_ushort(convert_half(acc0.s3)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[4*N], as_ushort(convert_half(acc0.s4)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[5*N], as_ushort(convert_half(acc0.s5)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[6*N], as_ushort(convert_half(acc0.s6)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[7*N], as_ushort(convert_half(acc0.s7)));

    intel_sub_group_block_write_us((__global ushort*)&c_base[8*N], as_ushort(convert_half(acc1.s0)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[9*N], as_ushort(convert_half(acc1.s1)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[10*N], as_ushort(convert_half(acc1.s2)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[11*N], as_ushort(convert_half(acc1.s3)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[12*N], as_ushort(convert_half(acc1.s4)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[13*N], as_ushort(convert_half(acc1.s5)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[14*N], as_ushort(convert_half(acc1.s6)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[15*N], as_ushort(convert_half(acc1.s7)));

    intel_sub_group_block_write_us((__global ushort*)&c_base[16*N], as_ushort(convert_half(acc2.s0)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[17*N], as_ushort(convert_half(acc2.s1)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[18*N], as_ushort(convert_half(acc2.s2)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[19*N], as_ushort(convert_half(acc2.s3)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[20*N], as_ushort(convert_half(acc2.s4)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[21*N], as_ushort(convert_half(acc2.s5)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[22*N], as_ushort(convert_half(acc2.s6)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[23*N], as_ushort(convert_half(acc2.s7)));

    intel_sub_group_block_write_us((__global ushort*)&c_base[24*N], as_ushort(convert_half(acc3.s0)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[25*N], as_ushort(convert_half(acc3.s1)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[26*N], as_ushort(convert_half(acc3.s2)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[27*N], as_ushort(convert_half(acc3.s3)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[28*N], as_ushort(convert_half(acc3.s4)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[29*N], as_ushort(convert_half(acc3.s5)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[30*N], as_ushort(convert_half(acc3.s6)));
    intel_sub_group_block_write_us((__global ushort*)&c_base[31*N], as_ushort(convert_half(acc3.s7)));
}
```