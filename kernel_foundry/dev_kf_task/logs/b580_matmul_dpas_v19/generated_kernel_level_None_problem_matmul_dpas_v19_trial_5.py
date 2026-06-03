# Analysis

Looking at the previous versions:

1. **Version 1 (1.220ms)** - Best performer. Uses `intel_sub_group_block_read_us` for B loads (coalesced), `restrict` keyword, and block writes for C output.
2. **Version 2 (1.340ms)** - Uses scalar B loads (`b_ptr[(2*p)*N]`) which are less efficient than block reads.
3. **Previous kernel (1.500ms)** - Uses SLM_A_STRIDE=36 (padding), which actually hurts because it wastes SLM bandwidth and the original stride=32 already has no bank conflicts (consecutive lanes hit consecutive addresses). Also has a branch `if (ki < k_iters - 1)` inside the loop.

**Key observations:**
- Version 1 is best because: (a) `intel_sub_group_block_read_us` for B gives coalesced reads, (b) `restrict` helps compiler optimize, (c) block writes for C output, (d) no branch in main loop (separate last iteration).
- SLM_A_STRIDE=32 is optimal (no padding needed since k_inner+sg_lid spans 0..31 continuously).

**Improvements to try on top of Version 1:**
1. **K-loop 2x unroll** - Process two TILE_K=32 steps per iteration, reducing loop overhead by 50% (64 iterations → 32 iterations with 2 barriers instead of 63).
2. **Interleave B loads with DPAS** - Load B for the second k16 step while computing the first k16 step.
3. **Use `intel_sub_group_block_read_us` for SLM A reads** - Vectorized SLM reads instead of scalar.
4. **Remove unnecessary address recomputation** - Use pointer arithmetic incrementally.

The 2x K-loop unroll is the most promising since it halves barrier count (from 63 to 31 barriers) and reduces loop control overhead.

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all row-major FP16, f32 accumulation
// Launch: GWS = (N, M/32), LWS = (64, 1)
//   - 64 WIs per WG = 4 subgroups of 16
//   - Each WG computes a 32×64 output tile
//   - TILE_M=32, TILE_N=64, TILE_K=32
// A loaded cooperatively into SLM (double-buffered), B from global/L2
// K=2048 always divides by 64 (2*TILE_K), K-loop unrolled 2x
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

    // Precompute base pointers
    __global const half* a_row_base = A + (m_start + a_load_row) * K + a_load_col;
    __global const ushort* b_base = (__global const ushort*)(B + n_start + sg_n_offset);

    // Load first A tile (k=0) into SLM buffer 0
    {
        __local half* a_dst = slm_a + a_load_row * SLM_A_STRIDE + a_load_col;
        half8 v0 = vload8(0, a_row_base);
        half8 v1 = vload8(1, a_row_base);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // K=2048, TILE_K=32 => 64 k-tiles. Unroll 2x => 32 outer iterations.
    // Each outer iteration processes 2 consecutive k-tiles (64 K elements).
    // Double-buffering: compute tile from buf[cur], load next into buf[next].
    // With 2x unroll: iter processes ki and ki+1, loads ki+2 into next buf.
    // Actually simpler: just keep the proven double-buffer but unroll the loop body 2x.
    
    const int k_iters = K / TILE_K;  // 64
    const int k_pairs = k_iters / 2; // 32

    for (int kp = 0; kp < k_pairs - 1; kp++) {
        // Process two k-tiles per outer iteration
        // First k-tile: ki = 2*kp (already in SLM buf = (2*kp)&1 = 0 for even kp)
        // Second k-tile: ki = 2*kp+1
        
        int ki0 = 2 * kp;
        int ki1 = ki0 + 1;
        
        // === First k-tile (ki0): compute from cur_buf, load next into next_buf ===
        {
            int cur_buf = ki0 & 1;
            int next_buf = 1 - cur_buf;
            int k_base = ki0 * TILE_K;

            // Load next A tile
            {
                int next_k = k_base + TILE_K;
                __global const half* a_src = a_row_base + next_k;
                __local half* a_dst = slm_a + next_buf * TILE_M * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col;
                half8 v0 = vload8(0, a_src);
                half8 v1 = vload8(1, a_src);
                vstore8(v0, 0, a_dst);
                vstore8(v1, 0, a_dst + 8);
            }

            // Compute current tile
            __local const half* a_slm = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                int k_inner = kk * 16;
                __global const ushort* b_row = b_base + (k_base + k_inner) * N;

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

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // === Second k-tile (ki1): compute from cur_buf, load next into next_buf ===
        {
            int cur_buf = ki1 & 1;
            int next_buf = 1 - cur_buf;
            int k_base = ki1 * TILE_K;

            // Load next A tile
            {
                int next_k = k_base + TILE_K;
                __global const half* a_src = a_row_base + next_k;
                __local half* a_dst = slm_a + next_buf * TILE_M * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col;
                half8 v0 = vload8(0, a_src);
                half8 v1 = vload8(1, a_src);
                vstore8(v0, 0, a_dst);
                vstore8(v1, 0, a_dst + 8);
            }

            // Compute current tile
            __local const half* a_slm = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                int k_inner = kk * 16;
                __global const ushort* b_row = b_base + (k_base + k_inner) * N;

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

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Last pair: kp = k_pairs - 1, process ki = k_iters-2 and ki = k_iters-1
    {
        int ki0 = k_iters - 2;
        int ki1 = k_iters - 1;

        // First of last pair: load next A, compute
        {
            int cur_buf = ki0 & 1;
            int next_buf = 1 - cur_buf;
            int k_base = ki0 * TILE_K;

            {
                int next_k = k_base + TILE_K;
                __global const half* a_src = a_row_base + next_k;
                __local half* a_dst = slm_a + next_buf * TILE_M * SLM_A_STRIDE + a_load_row * SLM_A_STRIDE + a_load_col;
                half8 v0 = vload8(0, a_src);
                half8 v1 = vload8(1, a_src);
                vstore8(v0, 0, a_dst);
                vstore8(v1, 0, a_dst + 8);
            }

            __local const half* a_slm = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                int k_inner = kk * 16;
                __global const ushort* b_row = b_base + (k_base + k_inner) * N;

                int8 b_val;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort lo = intel_sub_group_block_read_us(b_row + (2 * p) * N);
                    ushort hi = intel_sub_group_block_read_us(b_row + (2 * p + 1) * N);
                    ((int*)&b_val)[p] = (int)((uint)lo | ((uint)hi << 16));
                }

                { short8 a_val; __local const half* a_ptr = a_slm + k_inner + sg_lid;
                  #pragma unroll
                  for (int r = 0; r < 8; r++) a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                  acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc0); }
                { short8 a_val; __local const half* a_ptr = a_slm + 8*SLM_A_STRIDE + k_inner + sg_lid;
                  #pragma unroll
                  for (int r = 0; r < 8; r++) a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                  acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc1); }
                { short8 a_val; __local const half* a_ptr = a_slm + 16*SLM_A_STRIDE + k_inner + sg_lid;
                  #pragma unroll
                  for (int r = 0; r < 8; r++) a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                  acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc2); }
                { short8 a_val; __local const half* a_ptr = a_slm + 24*SLM_A_STRIDE + k_inner + sg_lid;
                  #pragma unroll
                  for (int r = 0; r < 8; r++) a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                  acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc3); }
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Last k-tile: no next load needed
        {
            int cur_buf = ki1 & 1;
            int k_base = ki1 * TILE_K;
            __local const half* a_slm = slm_a + cur_buf * TILE_M * SLM_A_STRIDE;

            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                int k_inner = kk * 16;
                __global const ushort* b_row = b_base + (k_base + k_inner) * N;

                int8 b_val;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    ushort lo = intel_sub_group_block_read_us(b_row + (2 * p) * N);
                    ushort hi = intel_sub_group_block_read_us(b_row + (2 * p + 1) * N);
                    ((int*)&b_val)[p] = (int)((uint)lo | ((uint)hi << 16));
                }

                { short8 a_val; __local const half* a_ptr = a_slm + k_inner + sg_lid;
                  #pragma unroll
                  for (int r = 0; r < 8; r++) a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                  acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc0); }
                { short8 a_val; __local const half* a_ptr = a_slm + 8*SLM_A_STRIDE + k_inner + sg_lid;
                  #pragma unroll
                  for (int r = 0; r < 8; r++) a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                  acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc1); }
                { short8 a_val; __local const half* a_ptr = a_slm + 16*SLM_A_STRIDE + k_inner + sg_lid;
                  #pragma unroll
                  for (int r = 0; r < 8; r++) a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                  acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc2); }
                { short8 a_val; __local const half* a_ptr = a_slm + 24*SLM_A_STRIDE + k_inner + sg_lid;
                  #pragma unroll
                  for (int r = 0; r < 8; r++) a_val[r] = as_short(a_ptr[r * SLM_A_STRIDE]);
                  acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc3); }
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