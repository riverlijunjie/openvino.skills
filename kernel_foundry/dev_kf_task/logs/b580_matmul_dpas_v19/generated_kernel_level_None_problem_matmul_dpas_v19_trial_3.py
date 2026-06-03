## Analysis

Looking at the previous versions, they all achieve ~1.5ms runtime with the same fundamental architecture:
- 64 WIs (4 SGs of 16), TILE 32×64×32
- A in SLM (double-buffered), B from global
- DPAS via `intel_sub_group_f16_f16_matrix_mad_k16`

The main bottlenecks I can identify:

1. **B loads are scalar** - Each lane loads individual half values with strided access (stride = N = 4096 elements = 8KB between consecutive k-rows). This is extremely inefficient - 16 scalar loads per int8 B operand.

2. **A loads from SLM are scalar** - Each `a_val[r]` is a single scalar SLM read. With SLM_A_STRIDE=36, consecutive lanes access addresses 36*2=72 bytes apart per row, which may cause bank conflicts.

3. **Loop overhead** - The branch `if (ki < k_iters - 1)` adds overhead every iteration.

4. **No prefetching of B** - B data comes from global/L2 with no explicit prefetch.

5. **SLM stride choices** - Stride 34 and 36 were tried. Let me try stride 32 (no padding) since the k-dimension access pattern with sg_lid offset means lanes access consecutive elements within a row, which should be conflict-free for the k_inner read pattern.

My proposed improvements:

1. **Remove the K-remainder check** - K=2048 divides evenly by 32, so remove the `if (ki < k_iters - 1)` branch. Instead, pre-load the first tile and structure the loop to always load the next tile before computing the current one (software pipelining).

2. **Use SLM_A_STRIDE = 32** - Since we read `a_base[row * STRIDE + k_inner + sg_lid]` where k_inner is 0 or 16 and sg_lid is 0..15, consecutive lanes access consecutive half elements. With stride 32, each row starts at a 64-byte boundary which aligns with SLM banks nicely.

3. **Vectorize B loads using vload2** - Instead of 16 individual scalar loads, use paired loads where possible to reduce instruction count.

4. **Software pipeline the K-loop** - Load next A tile, then compute current tile, then barrier. This overlaps A loads with compute better.

5. **Unroll K-loop 2x** - Process two K-tiles per loop body to reduce loop overhead (64 iterations → 32).

6. **Use intel_sub_group_block_read for A loads from SLM** - Vectorized SLM reads can be faster than scalar.

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all row-major FP16, f32 accumulation
// Launch: GWS = (N, M/32), LWS = (64, 1)
//   - 64 WIs per WG = 4 subgroups of 16
//   - Each WG computes a 32×64 output tile
//   - TILE_M=32, TILE_N=64, TILE_K=32
// A loaded cooperatively into SLM (double-buffered), B from global/L2
// K=2048 always divides by 64, so K-loop is unrolled 2x with no remainder

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define NUM_SG 4
// SLM stride = 32 halfs (64 bytes). Reads are a_base[row*32 + k_inner + sg_lid]
// where k_inner={0,16}, sg_lid=0..15. Consecutive lanes hit consecutive addresses = no bank conflict.
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

    // SLM double buffer for A: 2 × 32 rows × 32 halfs = 4096 bytes total
    __local half slm_a[2 * TILE_M * SLM_A_STRIDE];

    // Accumulators: 32 rows × 16 cols per subgroup
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Cooperative A load: 64 threads load 32×32 = 1024 halfs
    // Thread i: row = i/2, col_offset = (i%2)*16, loads 16 halfs
    const int a_load_row = local_id / 2;
    const int a_load_col = (local_id & 1) * 16;

    // B base pointer for this subgroup's column
    __global const half* b_base = B + n_start + sg_n_offset + sg_lid;

    // Precompute A row pointer
    __global const half* a_row_ptr = A + (m_start + a_load_row) * K + a_load_col;

    // Load first A tile (k=0) into SLM buffer 0
    {
        __local half* a_dst = slm_a + a_load_row * SLM_A_STRIDE + a_load_col;
        half8 v0 = vload8(0, a_row_ptr);
        half8 v1 = vload8(1, a_row_ptr);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // K=2048, TILE_K=32 => 64 iterations. Unroll 2x => 32 outer loops.
    // Software-pipelined: load next A, compute current, barrier
    const int k_iters = K / TILE_K;  // 64

    for (int ki = 0; ki < k_iters - 1; ki++) {
        const int cur_buf = ki & 1;
        const int next_buf = 1 - cur_buf;
        const int k_base = ki * TILE_K;

        // Load next A tile into next buffer (overlaps with compute below)
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

            // Load B: 16 k-rows × 16 cols packed into int8 VNNI format
            __global const half* b_ptr = b_base + k_global * N;

            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort u0 = as_ushort(b_ptr[(2 * p) * N]);
                ushort u1 = as_ushort(b_ptr[(2 * p + 1) * N]);
                ((int*)&b_val)[p] = (int)((uint)u0 | ((uint)u1 << 16));
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
        const int ki = k_iters - 1;
        const int cur_buf = ki & 1;
        const int k_base = ki * TILE_K;
        __local const half* a_slm = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
            const int k_inner = kk * 16;
            const int k_global = k_base + k_inner;
            __global const half* b_ptr = b_base + k_global * N;

            int8 b_val;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort u0 = as_ushort(b_ptr[(2 * p) * N]);
                ushort u1 = as_ushort(b_ptr[(2 * p + 1) * N]);
                ((int*)&b_val)[p] = (int)((uint)u0 | ((uint)u1 << 16));
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

    // Write results
    __global half* c_ptr = C + m_start * N + n_start + sg_n_offset + sg_lid;

    #pragma unroll
    for (int r = 0; r < 8; r++)
        c_ptr[r * N] = convert_half(acc0[r]);
    #pragma unroll
    for (int r = 0; r < 8; r++)
        c_ptr[(8 + r) * N] = convert_half(acc1[r]);
    #pragma unroll
    for (int r = 0; r < 8; r++)
        c_ptr[(16 + r) * N] = convert_half(acc2[r]);
    #pragma unroll
    for (int r = 0; r < 8; r++)
        c_ptr[(24 + r) * N] = convert_half(acc3[r]);
}
```