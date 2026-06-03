## Analysis

Looking at the previous versions:

1. **Version 2 (1.220ms)** - Best performer. Uses `intel_sub_group_block_read_us` for B loads and `intel_sub_group_block_write_us` for C stores. SLM stride = 32 (no padding). Simple double-buffered loop with separate last iteration.

2. **Version 1 (1.340ms)** - Similar to V2 but uses scalar B loads (`as_ushort(b_ptr[...])`) and scalar C stores. The block reads/writes in V2 clearly help.

3. **Last tested (1.610ms)** - Attempted 2x K-loop unroll with SLM stride=34 padding. The padding actually hurt (odd stride = more complex addressing), and the 2x unroll with doubled code size increased instruction cache pressure without benefit.

4. **Best so far (1.500ms)** - Uses SLM stride=36 padding. Again, padding hurts more than helps.

**Key insights:**
- SLM stride = 32 (no padding) is best - keeps addressing simple
- `intel_sub_group_block_read_us` for B is beneficial
- `intel_sub_group_block_write_us` for C output is beneficial  
- Simple double-buffer loop (V2 style) beats complex 2x unroll
- The main bottleneck is likely B memory bandwidth (16 scattered reads per DPAS)

**Proposed improvements over V2:**
1. Keep SLM_A_STRIDE = 32 (proven best)
2. Keep `intel_sub_group_block_read_us` for B and C
3. Use `intel_sub_group_block_read_us` for SLM A reads too (vectorized SLM access)
4. Interleave B loads with DPAS compute (load B for next DPAS block while current one executes)
5. Remove the separate last-iteration code path - just use a conditional in the loop (reduces code size / i-cache pressure)
6. Use `restrict` qualifiers

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all row-major FP16, f32 accumulation
// Launch: GWS = (N, M/32), LWS = (64, 1)
//   - 64 WIs per WG = 4 subgroups of 16
//   - Each WG computes a 32×64 output tile
//   - TILE_M=32, TILE_N=64, TILE_K=32
// A loaded cooperatively into SLM (double-buffered), B from global/L2
// K=2048 always divides by 32, no remainder needed.
// Hardware: Intel B580, Xe2, exec_size=16, 96 TFLOPS FP16 XMX

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4
#define SLM_A_STRIDE 32

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
    const int m_start = wg_m * TILE_M;
    const int n_start = wg_n * TILE_N;

    const int local_id = get_local_id(0);
    const int sg_id = local_id / SG_SIZE;
    const int sg_lid = local_id % SG_SIZE;
    const int sg_n_offset = sg_id * SG_SIZE;

    // SLM double buffer for A: 2 × 32 × 32 halfs = 4096 bytes
    __local half slm_a[2 * TILE_M * SLM_A_STRIDE];

    // Accumulators
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Cooperative A load: 64 threads, each loads 16 halfs
    const int a_load_row = local_id / 2;
    const int a_load_col = (local_id & 1) * 16;
    __global const half* a_row_ptr = A + (m_start + a_load_row) * K + a_load_col;

    // B base for this subgroup
    __global const ushort* b_base = (__global const ushort*)(B + n_start + sg_n_offset);

    // Load first A tile into buffer 0
    {
        __local half* a_dst = slm_a + a_load_row * SLM_A_STRIDE + a_load_col;
        half8 v0 = vload8(0, a_row_ptr);
        half8 v1 = vload8(1, a_row_ptr);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int k_iters = K / TILE_K;

    for (int ki = 0; ki < k_iters - 1; ki++) {
        const int cur_buf = ki & 1;
        const int next_buf = 1 - cur_buf;
        const int k_base = ki * TILE_K;

        // Load next A tile into next buffer (overlapped with compute)
        {
            __global const half* a_src = a_row_ptr + (k_base + TILE_K);
            __local half* a_dst = slm_a + next_buf * TILE_M * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col;
            half8 v0 = vload8(0, a_src);
            half8 v1 = vload8(1, a_src);
            vstore8(v0, 0, a_dst);
            vstore8(v1, 0, a_dst + 8);
        }

        // Compute from current SLM buffer
        __local const half* a_slm = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int k_inner = kk * 16;
            __global const ushort* b_row = b_base + (k_base + k_inner) * N;

            // Load B: 16 rows × 16 cols packed into VNNI int8
            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort lo = intel_sub_group_block_read_us(b_row + (2 * p) * N);
                ushort hi = intel_sub_group_block_read_us(b_row + (2 * p + 1) * N);
                ((int*)&b_val)[p] = (int)((uint)lo | ((uint)hi << 16));
            }

            // 4 DPAS blocks of 8 rows each
            __local const half* a_ptr0 = a_slm + k_inner + sg_lid;
            {
                short8 a_val;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr0[r * SLM_A_STRIDE]);
                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc0);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_ptr0 + 8 * SLM_A_STRIDE;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc1);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_ptr0 + 16 * SLM_A_STRIDE;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc2);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_ptr0 + 24 * SLM_A_STRIDE;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc3);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Last iteration: compute only
    {
        const int cur_buf = (k_iters - 1) & 1;
        const int k_base = (k_iters - 1) * TILE_K;
        __local const half* a_slm = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int k_inner = kk * 16;
            __global const ushort* b_row = b_base + (k_base + k_inner) * N;

            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort lo = intel_sub_group_block_read_us(b_row + (2 * p) * N);
                ushort hi = intel_sub_group_block_read_us(b_row + (2 * p + 1) * N);
                ((int*)&b_val)[p] = (int)((uint)lo | ((uint)hi << 16));
            }

            __local const half* a_ptr0 = a_slm + k_inner + sg_lid;
            {
                short8 a_val;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr0[r * SLM_A_STRIDE]);
                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc0);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_ptr0 + 8 * SLM_A_STRIDE;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc1);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_ptr0 + 16 * SLM_A_STRIDE;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc2);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_ptr0 + 24 * SLM_A_STRIDE;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc3);
            }
        }
    }

    // Write results using block writes
    __global ushort* c_ptr = (__global ushort*)(C + m_start * N + n_start + sg_n_offset);

    #pragma unroll
    for (int r = 0; r < 8; r++)
        intel_sub_group_block_write_us(c_ptr + r * N, as_ushort(convert_half(acc0[r])));
    #pragma unroll
    for (int r = 0; r < 8; r++)
        intel_sub_group_block_write_us(c_ptr + (8 + r) * N, as_ushort(convert_half(acc1[r])));
    #pragma unroll
    for (int r = 0; r < 8; r++)
        intel_sub_group_block_write_us(c_ptr + (16 + r) * N, as_ushort(convert_half(acc2[r])));
    #pragma unroll
    for (int r = 0; r < 8; r++)
        intel_sub_group_block_write_us(c_ptr + (24 + r) * N, as_ushort(convert_half(acc3[r])));
}
```