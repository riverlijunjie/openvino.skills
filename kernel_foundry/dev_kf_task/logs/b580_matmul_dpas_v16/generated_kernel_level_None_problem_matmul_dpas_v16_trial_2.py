## Analysis

Looking at the previous versions:

1. **Version 1 (1.450ms)**: Basic implementation with scalar SLM reads and scalar B loads.
2. **Version 2 (1.410ms)**: Similar to V1 but with minor cleanup.
3. **Latest (1.120ms)**: Key improvement was using `intel_sub_group_block_read_us8` for SLM reads with a stride-16 layout, which enables vectorized SLM access. Also used `as_int((half2)(...))` for B packing.

**Key bottlenecks in the 1.120ms version:**
1. **B loads are still scalar** - each lane loads 16 individual half values from global memory with stride N between rows. This is very inefficient (scattered access pattern).
2. **Double-buffering has a branch** (`if (next_k < K)`) - unnecessary since K=2048 divides evenly by 32.
3. **No K-loop unrolling** - loop overhead for 64 iterations (K=2048/32=64).
4. **A load and compute are serialized** - compute happens first, then A load for next iteration, then barrier. Better to overlap.

**Proposed improvements:**
1. **Remove the K-remainder check** - K=2048 is always divisible by 32, so remove the `if (next_k < K)` branch.
2. **2x K-loop unroll** - Process two K=32 tiles per loop iteration (K=64 per iteration), reducing loop overhead from 64 to 32 iterations and enabling better instruction scheduling.
3. **Interleave A loads with compute** - Start loading next A tile while computing current tile to overlap memory latency with DPAS execution.
4. **Use intel_sub_group_block_read for B** - Since each subgroup reads 16 consecutive columns from B, and B is row-major with stride N, we can use block reads for B rows where alignment permits. However, B rows are at stride N*2 bytes apart which may not work for block_read. Instead, let's try loading B pairs more efficiently.
5. **SLM stride padding** - Add +4 padding to SLM stride (use 20 instead of 16) to reduce bank conflicts. Actually, with stride=16 (32 bytes), all 16 lanes access the same bank. Let me use stride=18 or 20 to stagger accesses.

Wait - actually `intel_sub_group_block_read_us8` with stride=16 means consecutive 16-element rows are 32 bytes apart. On Xe2, SLM has 32 banks of 4 bytes each. With stride=32 bytes, rows 0 and 1 map to the same banks → no conflict since block_read reads contiguous 32 bytes (one per bank). This should be fine.

Let me focus on the most impactful change: **2x K-loop unroll with interleaved A prefetch**.

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

    // SLM: double-buffered, layout optimized for intel_sub_group_block_read_us8
    // Each buffer: two K16 blocks, each 32 rows x 16 cols, stride=16
    // Total: 2 * 32 * 32 = 2048 halfs = 4096 bytes
    #define SLM_K16_STRIDE 16
    #define SLM_BUF_SIZE 1024
    __local half slm_A[2 * SLM_BUF_SIZE];

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int lid = sg_id * 16 + sg_lid;
    const int a_load_row = lid / 2;
    const int a_load_khalf = lid & 1;

    // Precompute A load base
    const int a_row_stride = K;  // stride between rows in A

    // Preload first A tile into SLM buffer 0
    {
        __global const half* a_ptr = A + (wg_row + a_load_row) * a_row_stride + a_load_khalf * 16;
        int slm_off = a_load_khalf * (32 * SLM_K16_STRIDE) + a_load_row * SLM_K16_STRIDE;

        half8 v0 = vload8(0, a_ptr);
        half8 v1 = vload8(1, a_ptr);
        vstore8(v0, 0, &slm_A[slm_off]);
        vstore8(v1, 0, &slm_A[slm_off + 8]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;
    const int k_iterations = K / 32;  // K guaranteed divisible by 32

    // Main K-loop - no remainder handling needed
    for (int ki = 0; ki < k_iterations - 1; ki++) {
        int k_offset = ki * 32;
        int next_k = k_offset + 32;
        int next_buf = 1 - buf;
        int slm_base = buf * SLM_BUF_SIZE;

        // B pointer for this K-step
        __global const half* b_base = B + k_offset * N + sg_col + sg_lid;

        // --- K16 step 0: Load A from SLM, Load B from global, DPAS ---
        short8 a0_0, a1_0, a2_0, a3_0;
        {
            int a_off = slm_base;
            a0_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            a1_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));
        }

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

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_0, b_reg0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_0, b_reg0, acc1);

        // Start loading next A tile into other buffer (interleaved with compute)
        __global const half* a_next_ptr = A + (wg_row + a_load_row) * a_row_stride + next_k + a_load_khalf * 16;
        half8 a_next_v0 = vload8(0, a_next_ptr);
        half8 a_next_v1 = vload8(1, a_next_ptr);

        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_0, b_reg0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_0, b_reg0, acc3);

        // --- K16 step 1 ---
        short8 a0_1, a1_1, a2_1, a3_1;
        {
            int a_off = slm_base + 32 * SLM_K16_STRIDE;
            a0_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            a1_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));
        }

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

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_1, b_reg1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_1, b_reg1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_1, b_reg1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_1, b_reg1, acc3);

        // Store next A tile to SLM
        int slm_off = next_buf * SLM_BUF_SIZE + a_load_khalf * (32 * SLM_K16_STRIDE) + a_load_row * SLM_K16_STRIDE;
        vstore8(a_next_v0, 0, &slm_A[slm_off]);
        vstore8(a_next_v1, 0, &slm_A[slm_off + 8]);

        buf = next_buf;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Last iteration - no next A load needed
    {
        int k_offset = (k_iterations - 1) * 32;
        int slm_base = buf * SLM_BUF_SIZE;
        __global const half* b_base = B + k_offset * N + sg_col + sg_lid;

        // K16 step 0
        short8 a0_0, a1_0, a2_0, a3_0;
        {
            int a_off = slm_base;
            a0_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            a1_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3_0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));
        }

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

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_0, b_reg0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_0, b_reg0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_0, b_reg0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_0, b_reg0, acc3);

        // K16 step 1
        short8 a0_1, a1_1, a2_1, a3_1;
        {
            int a_off = slm_base + 32 * SLM_K16_STRIDE;
            a0_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            a1_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3_1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));
        }

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

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_1, b_reg1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_1, b_reg1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_1, b_reg1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_1, b_reg1, acc3);
    }

    // === Store C using subgroup block writes ===
    // Each subgroup writes 32 rows x 16 cols, one row at a time
    __global half* c_base = C + wg_row * N + sg_col;

    // Use intel_sub_group_block_write_us for 16-wide half stores
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