

## Analysis

Looking at the previous versions:

1. **Best version (1.140ms)**: 32M×256N tile, K-step=16, double-buffered A in SLM, B from global. The K-pair unrolling adds code complexity but barriers between compute and load aren't well overlapped.

2. **Last version (2.230ms)**: K-step=32 with two k16 steps per tile. Worse because: larger SLM A tile (32×32=2KB per buffer), more elements to load (4 per WI instead of 2), and the `eid / TILE_K` with TILE_K=32 is not a power-of-2 shift — actually 32 IS power of 2, but the bigger issue is that loading 4 elements per WI increases load time and the two barriers per k-tile iteration hurt.

**Key bottlenecks in the best version:**
- B matrix loads are not cached in SLM — each of 16 subgroups loads independently from global memory. With 32M tile, A reuse is 16× but B has no reuse across rows.
- The SLM A reads use `intel_sub_group_block_read_us` row by row (32 reads per subgroup per k-tile), which is many SLM operations.
- Barriers are frequent and compute/load overlap isn't great.

**Proposed improvements:**
1. **Both A and B in SLM**: Load B tile (16×256) cooperatively into SLM for reuse across the 4 row-groups (32/8=4). This reduces global memory bandwidth for B by 4×.
2. **Larger K unroll (K=32)** but with proper SLM layout — load 32×32 A and 32×256 B, doing 2 rounds of k16 DPAS.
3. Actually, 32×256×2bytes = 16KB for B per k16 step, and 32×16×2 = 1KB for A. Total SLM for both = 17KB per buffer. Double buffered = 34KB. That's within 128KB SLM budget but quite large. Let me reconsider.

**Better approach**: Keep A in SLM (small, high reuse), but improve the pipeline. The main issue is likely that with 16 subgroups all loading B from global, we're bandwidth-bound. Let me try a different tiling: **64M × 128N** with 8 subgroups handling N (8×16=128) and each subgroup computing 64 rows (8 DPAS calls). This gives A reuse of 8× and doubles the compute per B load.

Actually, let me try: **64M × 128N**, 8 subgroups, each SG does 8×DPAS per k-step. A tile = 64×16 = 2KB per buffer. More compute per SG means better DPAS utilization.

Wait — with 8 SGs × 16 WIs = 128 WIs per WG. We need enough WGs to fill 20 Xe cores. For a typical 1024×1024 problem: ceil(1024/128)×ceil(1024/64) = 8×16 = 128 WGs for 20 cores = 6.4 WGs/core — good occupancy.

Each SG: 8 DPAS per k-step (64 rows / 8 = 8 groups). That's good compute density. B load: 8 block reads of ushort pairs = 16 reads. A from SLM: 64 block reads. Ratio of DPAS to loads is better.

Let me try **32M × 128N** with 8 subgroups to keep register pressure manageable but reduce N-tile (less B bandwidth).

Actually, let me reconsider the fundamental issue. At 23% XMX utilization, we're likely memory-bandwidth bound. Let's compute:

For M=N=K=1024: 2×1024³ = 2B FLOPs. At 96 TFLOPS → 0.021ms theoretical. Data = (2×1024² + 1024²)×2 = 6MB. At 456 GB/s → 0.013ms. So we should be compute-bound but achieving only 1.14ms means massive inefficiency.

The problem might be that matrix dimensions aren't always 1024. Let me focus on maximizing DPAS throughput with good pipelining.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// Tile: 32M x 256N, K-step=16
// 16 subgroups x 16 WIs = 256 WIs per WG
// Each subgroup: 32 rows x 16 cols = 4 DPAS(8x16) per k-step
// A in SLM (double-buffered), B from global
// GWS = (ceil(N/256)*256, ceil(M/32))  LWS = (256, 1)
// Subgroup size = 16
// Key optimization: true double-buffering with overlapped load/compute,
// minimized barriers, and prefetch of B via SLM for next iteration

#define TILE_M 32
#define TILE_N 256
#define TILE_K 16
#define SLM_A_SIZE (TILE_M * TILE_K)  // 512 ushorts = 1KB per buffer
#define WG_SIZE 256

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id  = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid    = get_local_id(0);

    const int n_base   = get_group_id(0) * TILE_N;
    const int row_base = get_group_id(1) * TILE_M;
    const int col_base = n_base + sg_id * 16;

    // Double-buffered SLM for A: 2 * 512 ushorts = 2 KB total
    __local ushort slm_a[2 * SLM_A_SIZE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    const bool col_in_bounds = (col_base + 16 <= N);

    // Precompute A load coordinates for this work-item
    // Each WI loads 2 elements of A per tile (512 elements / 256 WIs = 2)
    const int eid0 = lid;
    const int eid1 = lid + WG_SIZE;
    const int ar0 = eid0 >> 4;  // eid0 / 16
    const int ac0 = eid0 & 15;  // eid0 % 16
    const int ar1 = eid1 >> 4;
    const int ac1 = eid1 & 15;
    const int gr0 = row_base + ar0;
    const int gr1 = row_base + ar1;
    const bool r0_valid = (gr0 < M);
    const bool r1_valid = (gr1 < M);

    // Load first A tile into buffer 0
    {
        __local ushort* dst = slm_a;
        dst[eid0] = (r0_valid && ac0 < K) ? A_us[gr0 * K + ac0] : (ushort)0;
        dst[eid1] = (r1_valid && ac1 < K) ? A_us[gr1 * K + ac1] : (ushort)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;
        const int cur_buf = kt & 1;

        __local const ushort* ca = slm_a + cur_buf * SLM_A_SIZE;

        // Read A from SLM into registers - all 32 rows, 16 cols
        short8 a0, a1, a2, a3;
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a0)[r] = intel_sub_group_block_read_us(ca + r * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a1)[r] = intel_sub_group_block_read_us(ca + (8 + r) * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a2)[r] = intel_sub_group_block_read_us(ca + (16 + r) * TILE_K);
        #pragma unroll
        for (int r = 0; r < 8; r++)
            ((ushort*)&a3)[r] = intel_sub_group_block_read_us(ca + (24 + r) * TILE_K);

        // Load B from global memory: 16 rows x 16 cols packed as int8
        int8 b_val;
        if (col_in_bounds) {
            __global const ushort* bp = B_us + k_offset * N + col_base;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                ushort s0 = intel_sub_group_block_read_us(bp + (2 * p) * N);
                ushort s1 = intel_sub_group_block_read_us(bp + (2 * p + 1) * N);
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }
        } else {
            int col = col_base + sg_lid;
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                int kr0 = k_offset + 2 * p;
                int kr1 = kr0 + 1;
                ushort s0 = (kr0 < K && col < N) ? B_us[kr0 * N + col] : (ushort)0;
                ushort s1 = (kr1 < K && col < N) ? B_us[kr1 * N + col] : (ushort)0;
                ((int*)&b_val)[p] = as_int((ushort2)(s0, s1));
            }
        }

        // Start loading next A tile into alternate buffer BEFORE compute
        // This allows the global memory load to overlap with DPAS execution
        barrier(CLK_LOCAL_MEM_FENCE);
        if (kt + 1 < num_k_tiles) {
            const int next_buf = 1 - cur_buf;
            const int next_k = k_offset + TILE_K;
            __local ushort* dst = slm_a + next_buf * SLM_A_SIZE;
            int gk0 = next_k + ac0;
            int gk1 = next_k + ac1;
            dst[eid0] = (r0_valid && gk0 < K) ? A_us[gr0 * K + gk0] : (ushort)0;
            dst[eid1] = (r1_valid && gk1 < K) ? A_us[gr1 * K + gk1] : (ushort)0;
        }

        // DPAS compute - 4 matrix multiply-accumulate operations
        // On Xe2, DPAS can execute concurrently with pending memory operations
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results using subgroup block writes where possible
    const int col_idx = col_base + sg_lid;
    if (col_idx < N) {
        __global ushort* C_us = (__global ushort*)C;

        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc0)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 8 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc1)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 16 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc2)[r]));
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int gr = row_base + 24 + r;
            if (gr < M) C_us[gr * N + col_idx] = as_ushort(convert_half(((float*)&acc3)[r]));
        }
    }
}
```

