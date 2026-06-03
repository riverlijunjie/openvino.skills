## Analysis

Looking at the previous versions:

1. **Best version (1.340ms)**: Uses SLM_STRIDE=32 (no padding), simple double-buffering with separate main loop and last-iteration handling, scalar B loads per lane.

2. **Previous version (1.490ms)**: Uses SLM_STRIDE=36 (with padding), 2x unrolled K-loop. The padding actually hurt performance - likely because it increases SLM footprint and the stride=32 already has no bank conflicts (16 consecutive lanes reading consecutive addresses).

3. **Version 1 (1.550ms)**: Similar to best but uses `intel_sub_group_block_read_us` for B loads and stores. The block reads for B require aligned pointers which may not always be satisfied.

**Key observations:**
- SLM_STRIDE=32 is better than 36 (no padding needed since access pattern is conflict-free)
- The best version's structure (separate last iteration, no 2x unroll overhead) works well
- Scalar B loads work fine since each lane reads its own column

**Proposed improvements over the best version:**
1. **Use `intel_sub_group_block_read_us` for B loads** - B is accessed at `b_base + k_global * N` where N=2048 and addresses are 16-element aligned (n_start is multiple of 64, sg_n_offset is multiple of 16). This should give coalesced reads.
2. **Use `intel_sub_group_block_write_us` for C stores** - Similarly aligned.
3. **Interleave A loads with compute** - Place the A SLM load before the compute but after reading the first k16 block of A from SLM, to overlap memory latency.
4. **Remove branch in main loop** - Since we handle last iteration separately, the `if (ki < k_iters - 1)` check is unnecessary in the main loop.
5. **Use vload2 for paired B reads** - Instead of two scalar reads + pack, use a single 32-bit read when possible.

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all row-major FP16, f32 accumulation
// Launch: GWS = (N, M/32), LWS = (64, 1)
//   - 64 WIs per WG = 4 subgroups of 16
//   - Each WG computes a 32×64 output tile
//   - TILE_M=32, TILE_N=64, TILE_K=32
// A loaded cooperatively into SLM (double-buffered), B from global/L2
// K=2048 always divides by 32, no remainder handling needed.
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
    // Work-group tile position
    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);

    const int m_start = wg_m * TILE_M;
    const int n_start = wg_n * TILE_N;

    // Subgroup and lane identification
    const int local_id = get_local_id(0);
    const int sg_id = local_id / SG_SIZE;
    const int sg_lid = local_id % SG_SIZE;

    // Each subgroup handles 16 columns of the 64-col output tile
    const int sg_n_offset = sg_id * SG_SIZE;

    // SLM double buffer for A: 2 × 32 rows × 32 halfs = 4096 bytes
    __local half slm_a[2 * TILE_M * SLM_A_STRIDE];

    // Accumulators: 32 rows × 16 cols per subgroup
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Cooperative A load: 64 threads load 32×32 = 1024 halfs
    const int a_load_row = local_id / 2;
    const int a_load_col = (local_id & 1) * 16;

    // Precompute A row pointer
    __global const half* a_row_ptr = A + (m_start + a_load_row) * K + a_load_col;

    // B base pointer for this subgroup's columns (aligned to 16 halfs = 32 bytes)
    __global const ushort* b_base = (__global const ushort*)(B + n_start + sg_n_offset);

    // Load first A tile (k=0) into SLM buffer 0
    {
        __local half* a_dst = slm_a + a_load_row * SLM_A_STRIDE + a_load_col;
        half8 v0 = vload8(0, a_row_ptr);
        half8 v1 = vload8(1, a_row_ptr);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int k_iters = K / TILE_K;  // 64

    // Main loop: iterations 0..62 (load next A + compute current)
    for (int ki = 0; ki < k_iters - 1; ki++) {
        const int cur_buf = ki & 1;
        const int next_buf = 1 - cur_buf;
        const int k_base = ki * TILE_K;

        // Load next A tile into next buffer
        {
            const int next_k = k_base + TILE_K;
            __global const half* a_src = a_row_ptr + next_k;
            __local half* a_dst = slm_a + next_buf * TILE_M * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col;
            half8 v0 = vload8(0, a_src);
            half8 v1 = vload8(1, a_src);
            vstore8(v0, 0, a_dst);
            vstore8(v1, 0, a_dst + 8);
        }

        // Compute current tile from SLM
        __local const half* a_slm = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int k_inner = kk * 16;
            const int k_global = k_base + k_inner;

            // Load B using intel_sub_group_block_read_us for coalesced access
            // B layout: B[k][n], each row has N elements, we read 16 consecutive cols
            __global const ushort* b_row = b_base + k_global * N;

            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort lo = intel_sub_group_block_read_us(b_row + (2 * p) * N);
                ushort hi = intel_sub_group_block_read_us(b_row + (2 * p + 1) * N);
                ((int*)&b_val)[p] = (int)((uint)lo | ((uint)hi << 16));
            }

            // DPAS blocks - 4 blocks of 8 rows each
            {
                short8 a_val;
                __local const half* a_ptr = a_slm + k_inner + sg_lid;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc0);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_slm + 8 * SLM_A_STRIDE + k_inner + sg_lid;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc1);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_slm + 16 * SLM_A_STRIDE + k_inner + sg_lid;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc2);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_slm + 24 * SLM_A_STRIDE + k_inner + sg_lid;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc3);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Last iteration (ki = k_iters - 1): no next A load needed
    {
        const int cur_buf = (k_iters - 1) & 1;
        const int k_base = (k_iters - 1) * TILE_K;
        __local const half* a_slm = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int k_inner = kk * 16;
            const int k_global = k_base + k_inner;
            __global const ushort* b_row = b_base + k_global * N;

            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort lo = intel_sub_group_block_read_us(b_row + (2 * p) * N);
                ushort hi = intel_sub_group_block_read_us(b_row + (2 * p + 1) * N);
                ((int*)&b_val)[p] = (int)((uint)lo | ((uint)hi << 16));
            }

            {
                short8 a_val;
                __local const half* a_ptr = a_slm + k_inner + sg_lid;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc0);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_slm + 8 * SLM_A_STRIDE + k_inner + sg_lid;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc1);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_slm + 16 * SLM_A_STRIDE + k_inner + sg_lid;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc2);
            }
            {
                short8 a_val;
                __local const half* a_ptr = a_slm + 24 * SLM_A_STRIDE + k_inner + sg_lid;
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc3);
            }
        }
    }

    // Write results using block writes for coalesced output
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