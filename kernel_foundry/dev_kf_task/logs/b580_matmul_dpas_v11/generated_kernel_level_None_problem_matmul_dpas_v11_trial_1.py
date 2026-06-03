

## Analysis

Looking at the previous versions:

1. **Version 1 (3.88ms)**: TILE_M=32, TILE_N=64, TILE_K=64, 64 WIs, only A in SLM. B loaded from global each time - redundant across subgroups.

2. **Version 2 (9.08ms)**: TILE_M=64, TILE_N=64, TILE_K=32, 64 WIs, both A and B in SLM. Too much register pressure (8 accumulators) and too much SLM usage slowed it down.

3. **Version 3 (2.19ms)**: TILE_M=32, TILE_N=128, TILE_K=64, 128 WIs, both A and B in SLM. The B tile is huge (64×130 = 8320 halves per buffer × 2 = ~33KB just for B). SLM pressure is too high, and loading 64 elements per WI for B is excessive. Also, the double buffering has a structural problem: loads and compute aren't truly overlapped because loads happen before compute within the same iteration.

**Key issues with Version 3:**
- Massive SLM usage (~50KB total) likely exceeds efficient SLM capacity
- B loading is 64 iterations per WI - too much overhead
- Double buffering structure is wrong: loads next tile, then computes, then barriers - this serializes rather than overlaps

**Strategy for improvement:**
- Go back to Version 1's approach (best performer) but fix the double buffering properly
- Use TILE_M=32, TILE_N=64, TILE_K=32 - smaller K tile means less B data to load
- Only cache A in SLM (small: 32×34 = ~2KB per buffer), load B from global (it's accessed contiguously and benefits from L2 cache)
- True double buffering: load next A tile while computing on current
- 64 WIs (4 subgroups) - proven to work well
- Each subgroup: 4 blocks × 8 rows = 32 rows × 16 cols, 2 k16 steps = 8 DPAS per k-tile
- With proper double buffering, we reduce barrier overhead

Actually, let me reconsider. Version 1 at 3.88ms had TILE_K=64 and loaded B from global. Let me try a cleaner approach: TILE_M=32, TILE_N=64, TILE_K=32, A in SLM only, proper double buffering where we interleave load and compute without extra barriers.

Better yet: let me try to minimize barriers. With proper double buffering using two A buffers, the pattern should be:
1. Load buf[0], barrier
2. Loop: compute buf[cur] while loading buf[next], barrier
3. Compute last buf

This gives 1 barrier per K-tile in steady state (not 2).

Let me also try block reads via `intel_sub_group_block_read` for B to improve memory throughput.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Optimized matmul: TILE_M=32, TILE_N=64, TILE_K=32
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 row-blocks x 8 rows = 32 rows x 16 cols
// 2 k16 steps per K-tile => 8 DPAS per barrier
// A in double-buffered SLM, B read directly from global (L2 cached)
// True double buffering: overlap SLM load of next tile with DPAS compute
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_A_STRIDE (TILE_K + 2)  // pad to avoid bank conflicts
#define NUM_WI 64

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
    const int sg_id = get_sub_group_id();    // 0..3
    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int lid = get_local_id(0);         // 0..63

    const int n_base = get_group_id(0) * TILE_N;
    const int col_base = n_base + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    // Double-buffered A in SLM: 2 x [32 x 34] = ~4.25 KB total
    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // A tile: 32 x 32 = 1024 halves, 64 WIs => 16 elements each
    // Load first A tile into buffer 0
    {
        __local half* dst = slm_A;
        if (row_tile_valid && TILE_K <= K) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_K;
                int c = elem_id % TILE_K;
                dst[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int elem_id = lid + i * NUM_WI;
                int r = elem_id / TILE_K;
                int c = elem_id % TILE_K;
                int gr = row_base + r;
                dst[r * SLM_A_STRIDE + c] = (gr < M && c < K) ? A[gr * K + c] : (half)0.0h;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Main loop
    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int cur_buf = kt & 1;
        const int nxt_buf = 1 - cur_buf;
        const int k_cur = kt * TILE_K;
        const bool has_next = (kt + 1 < num_k_tiles);
        const int k_next = k_cur + TILE_K;

        __local half* cur_A = slm_A + cur_buf * TILE_M * SLM_A_STRIDE;

        // ---- Compute: 2 k16 steps x 4 row-blocks = 8 DPAS ----
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += 16) {
            // Load A from SLM
            short8 a0, a1, a2, a3;
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a0)[r] = as_short(cur_A[r * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a1)[r] = as_short(cur_A[(8 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a2)[r] = as_short(cur_A[(16 + r) * SLM_A_STRIDE + kk + sg_lid]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                ((short*)&a3)[r] = as_short(cur_A[(24 + r) * SLM_A_STRIDE + kk + sg_lid]);

            // Load B from global: pack 16 k-rows into int8
            int8 b_val;
            if (col_valid) {
                int gk = k_cur + kk;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    int k0 = gk + 2 * p;
                    int k1 = k0 + 1;
                    short s0 = (k0 < K) ? as_short(B[k0 * N + col_idx]) : (short)0;
                    short s1 = (k1 < K) ? as_short(B[k1 * N + col_idx]) : (short)0;
                    ((int*)&b_val)[p] = as_int((short2)(s0, s1));
                }
            } else {
                b_val = (int8)(0);
            }

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);
        }

        // ---- Load next A tile into nxt_buf (overlapped after compute) ----
        if (has_next) {
            barrier(CLK_LOCAL_MEM_FENCE);
            __local half* nxt_A = slm_A + nxt_buf * TILE_M * SLM_A_STRIDE;
            if (row_tile_valid && k_next + TILE_K <= K) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_K;
                    int c = elem_id % TILE_K;
                    nxt_A[r * SLM_A_STRIDE + c] = A[(row_base + r) * K + k_next + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int elem_id = lid + i * NUM_WI;
                    int r = elem_id / TILE_K;
                    int c = elem_id % TILE_K;
                    int gr = row_base + r;
                    int gk = k_next + c;
                    nxt_A[r * SLM_A_STRIDE + c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // ---- Store results ----
    if (col_valid) {
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + r) * N + col_idx] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 8 + r) * N + col_idx] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 16 + r) * N + col_idx] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C[(row_base + 24 + r) * N + col_idx] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C[(row_base + r) * N + col_idx] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C[(row_base + 8 + r) * N + col_idx] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C[(row_base + 16 + r) * N + col_idx] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C[(row_base + 24 + r) * N + col_idx] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

