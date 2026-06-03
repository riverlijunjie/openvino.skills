## Analysis

Looking at the previous versions, all achieve ~1.4ms (23% XMX utilization). The key bottlenecks are:

1. **B loads are scattered** - Each lane loads from `b_ptr[i * N]` which is a stride-N access pattern. With N=2048 and half precision, consecutive k-rows are 4KB apart. This creates poor cache line utilization.

2. **SLM reads are scalar** - Each element of the short8 A vectors is loaded individually from SLM, generating 8 separate SLM reads per short8 instead of using vectorized SLM block reads.

3. **Loop overhead** - The loop runs 63 iterations with a barrier each time. The interleaving of B loads with DPAS is good but could be tighter.

4. **Store is scalar** - Writing C element-by-element wastes bandwidth.

**Proposed improvements:**

1. **Use `intel_sub_group_block_read_us` for SLM A reads** - Instead of 8 scalar SLM reads per short8, use subgroup block reads from SLM. This requires the SLM layout to be compatible (contiguous 16 elements per row for a subgroup read). With stride=32 (64 bytes per row), we can read 16 consecutive halfs with one block_read_us call per row, then assemble the short8 from 8 such reads. Actually, for DPAS we need the data transposed (each lane holds elements from different rows). The current scalar approach does this correctly. Let me reconsider.

2. **SLM stride padding to avoid bank conflicts** - Change SLM_STRIDE from 32 to 36 (add 4 half padding). With 32 banks of 4 bytes each, stride=32 halfs = 64 bytes = exactly 16 banks, which means rows 0 and 2 hit the same banks. Padding to 36 halfs = 72 bytes shifts each row by 2 banks, reducing conflicts.

3. **2x K-loop unroll** - Process two K=32 tiles per loop iteration (K=64 per iteration), reducing loop overhead and barrier count from 63 to 31.

4. **Vectorized C store using intel_sub_group_block_write_us** - Use subgroup block writes for output.

5. **Pre-load A into registers before barrier** - Load next A from global into registers, then store to SLM after barrier, overlapping global load latency with compute.

Let me implement a version with SLM padding (stride=36) and 2x K-loop unroll to cut barriers in half:

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half, f32 accumulation
// Launch: GWS = ((N/64)*16, (M/32)*4, 1), LWS = (16, 4, 1)
// 4 subgroups of 16 WIs per WG. Each WG computes 32x64 tile.
// Each subgroup computes 32x16 using DPAS.
// A in SLM (double-buffered), B from global/L2.
// TILE_M=32, TILE_N=64, TILE_K=32. K must be divisible by 64 (2x unroll).
// Optimization: 2x K-loop unroll (process 64 K per iteration, 31 barriers instead of 63)
// SLM stride padded to 36 to reduce bank conflicts.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define SLM_STRIDE 36
#define SLM_BUF_SIZE (32 * SLM_STRIDE)

inline short8 load_a_col(__local const half* slm, int col) {
    return (short8)(
        as_short(slm[0 * SLM_STRIDE + col]),
        as_short(slm[1 * SLM_STRIDE + col]),
        as_short(slm[2 * SLM_STRIDE + col]),
        as_short(slm[3 * SLM_STRIDE + col]),
        as_short(slm[4 * SLM_STRIDE + col]),
        as_short(slm[5 * SLM_STRIDE + col]),
        as_short(slm[6 * SLM_STRIDE + col]),
        as_short(slm[7 * SLM_STRIDE + col])
    );
}

inline int8 load_b_tile(__global const half* b_ptr, int N) {
    int8 b;
    short bv0  = as_short(b_ptr[0 * N]);
    short bv1  = as_short(b_ptr[1 * N]);
    short bv2  = as_short(b_ptr[2 * N]);
    short bv3  = as_short(b_ptr[3 * N]);
    short bv4  = as_short(b_ptr[4 * N]);
    short bv5  = as_short(b_ptr[5 * N]);
    short bv6  = as_short(b_ptr[6 * N]);
    short bv7  = as_short(b_ptr[7 * N]);
    short bv8  = as_short(b_ptr[8 * N]);
    short bv9  = as_short(b_ptr[9 * N]);
    short bv10 = as_short(b_ptr[10 * N]);
    short bv11 = as_short(b_ptr[11 * N]);
    short bv12 = as_short(b_ptr[12 * N]);
    short bv13 = as_short(b_ptr[13 * N]);
    short bv14 = as_short(b_ptr[14 * N]);
    short bv15 = as_short(b_ptr[15 * N]);
    b.s0 = as_int((short2)(bv0, bv1));
    b.s1 = as_int((short2)(bv2, bv3));
    b.s2 = as_int((short2)(bv4, bv5));
    b.s3 = as_int((short2)(bv6, bv7));
    b.s4 = as_int((short2)(bv8, bv9));
    b.s5 = as_int((short2)(bv10, bv11));
    b.s6 = as_int((short2)(bv12, bv13));
    b.s7 = as_int((short2)(bv14, bv15));
    return b;
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 4, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);
    const int sg_id = get_local_id(1);
    const int sg_lane = get_local_id(0);

    const int tile_row = wg_m * 32;
    const int tile_col = wg_n * 64 + sg_id * 16;

    const int lid = sg_id * 16 + sg_lane; // 0..63

    // SLM for A: double-buffered, 32 rows x 32 cols of half, stride=36 (padded)
    __local half slm_a[2 * SLM_BUF_SIZE];

    // Accumulators: 32 rows x 16 cols per subgroup
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Cooperative A load: 64 WIs load 32x32 = 1024 halfs = 16 per WI
    const int load_row = lid / 2;
    const int load_col = (lid % 2) * 16;
    const int slm_store_offset = load_row * SLM_STRIDE + load_col;

    __global const half* a_load_base = A + (tile_row + load_row) * K + load_col;
    __global const half* b_base = B + tile_col + sg_lane;

    // Preload first A tile (k=0..31) into SLM buffer 0
    vstore16(vload16(0, a_load_base), 0, &slm_a[slm_store_offset]);
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;
    const int k_iters = K / 32;  // 64 for K=2048

    // Main loop: process pairs of K-tiles (2x unroll)
    // Each pair: compute from cur_buf, load next into next_buf, barrier, 
    //            compute from next_buf, load next+1 into cur_buf, barrier
    // Total pairs: (k_iters-1)/2 = 31 full pairs, then handle last tile
    // Actually let's do: loop processes k_iters-1 iterations with double-buffering,
    // but unroll 2 iterations per loop body to reduce loop overhead.
    
    // Process iterations 0..k_iters-3 in pairs (2x unrolled), then last iteration separately
    int ki = 0;
    for (; ki < k_iters - 2; ki += 2) {
        // ===== First tile of pair (ki) =====
        {
            const int k = ki * 32;
            const int next_buf = 1 - cur_buf;
            __local const half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

            // Load B for k_inner=0..15
            int8 b0 = load_b_tile(b_base + k * N, N);

            // Load A from SLM rows 0-7, k cols 0-15
            short8 a0 = load_a_col(cur_slm, sg_lane);
            short8 a1 = load_a_col(cur_slm + 8 * SLM_STRIDE, sg_lane);
            short8 a2 = load_a_col(cur_slm + 16 * SLM_STRIDE, sg_lane);
            short8 a3 = load_a_col(cur_slm + 24 * SLM_STRIDE, sg_lane);

            // DPAS k_inner=0..15
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc3);

            // Load B for k_inner=16..31
            int8 b1 = load_b_tile(b_base + (k + 16) * N, N);

            // Load A from SLM for k_inner=16..31
            short8 a4 = load_a_col(cur_slm, 16 + sg_lane);
            short8 a5 = load_a_col(cur_slm + 8 * SLM_STRIDE, 16 + sg_lane);
            short8 a6 = load_a_col(cur_slm + 16 * SLM_STRIDE, 16 + sg_lane);
            short8 a7 = load_a_col(cur_slm + 24 * SLM_STRIDE, 16 + sg_lane);

            // Load next A tile into registers
            half16 a_next = vload16(0, a_load_base + (ki + 1) * 32);

            // DPAS k_inner=16..31
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a4, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a5, b1, acc1);

            // Store next A to SLM
            vstore16(a_next, 0, &slm_a[next_buf * SLM_BUF_SIZE + slm_store_offset]);

            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a6, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a7, b1, acc3);

            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = next_buf;
        }

        // ===== Second tile of pair (ki+1) =====
        {
            const int k = (ki + 1) * 32;
            const int next_buf = 1 - cur_buf;
            __local const half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

            // Load B for k_inner=0..15
            int8 b0 = load_b_tile(b_base + k * N, N);

            // Load A from SLM
            short8 a0 = load_a_col(cur_slm, sg_lane);
            short8 a1 = load_a_col(cur_slm + 8 * SLM_STRIDE, sg_lane);
            short8 a2 = load_a_col(cur_slm + 16 * SLM_STRIDE, sg_lane);
            short8 a3 = load_a_col(cur_slm + 24 * SLM_STRIDE, sg_lane);

            // DPAS k_inner=0..15
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc3);

            // Load B for k_inner=16..31
            int8 b1 = load_b_tile(b_base + (k + 16) * N, N);

            // Load A from SLM for k_inner=16..31
            short8 a4 = load_a_col(cur_slm, 16 + sg_lane);
            short8 a5 = load_a_col(cur_slm + 8 * SLM_STRIDE, 16 + sg_lane);
            short8 a6 = load_a_col(cur_slm + 16 * SLM_STRIDE, 16 + sg_lane);
            short8 a7 = load_a_col(cur_slm + 24 * SLM_STRIDE, 16 + sg_lane);

            // Load next A tile into registers
            half16 a_next = vload16(0, a_load_base + (ki + 2) * 32);

            // DPAS k_inner=16..31
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a4, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a5, b1, acc1);

            // Store next A to SLM
            vstore16(a_next, 0, &slm_a[next_buf * SLM_BUF_SIZE + slm_store_offset]);

            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a6, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a7, b1, acc3);

            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = next_buf;
        }
    }

    // Handle remaining iterations (ki should be k_iters-2 now, need to process ki and ki+1)
    // Process ki = k_iters-2 (second-to-last, needs next A load)
    {
        const int k = ki * 32;
        const int next_buf = 1 - cur_buf;
        __local const half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

        int8 b0 = load_b_tile(b_base + k * N, N);

        short8 a0 = load_a_col(cur_slm, sg_lane);
        short8 a1 = load_a_col(cur_slm + 8 * SLM_STRIDE, sg_lane);
        short8 a2 = load_a_col(cur_slm + 16 * SLM_STRIDE, sg_lane);
        short8 a3 = load_a_col(cur_slm + 24 * SLM_STRIDE, sg_lane);

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc3);

        int8 b1 = load_b_tile(b_base + (k + 16) * N, N);

        short8 a4 = load_a_col(cur_slm, 16 + sg_lane);
        short8 a5 = load_a_col(cur_slm + 8 * SLM_STRIDE, 16 + sg_lane);
        short8 a6 = load_a_col(cur_slm + 16 * SLM_STRIDE, 16 + sg_lane);
        short8 a7 = load_a_col(cur_slm + 24 * SLM_STRIDE, 16 + sg_lane);

        // Load last A tile
        half16 a_next = vload16(0, a_load_base + (ki + 1) * 32);

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a4, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a5, b1, acc1);

        vstore16(a_next, 0, &slm_a[next_buf * SLM_BUF_SIZE + slm_store_offset]);

        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a6, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a7, b1, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
        cur_buf = next_buf;
    }

    // Last iteration (ki+1 = k_iters-1), no next A load
    {
        const int k = (k_iters - 1) * 32;
        __local const half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

        int8 b0 = load_b_tile(b_base + k * N, N);

        short8 a0 = load_a_col(cur_slm, sg_lane);
        short8 a1 = load_a_col(cur_slm + 8 * SLM_STRIDE, sg_lane);
        short8 a2 = load_a_col(cur_slm + 16 * SLM_STRIDE, sg_lane);
        short8 a3 = load_a_col(cur_slm + 24 * SLM_STRIDE, sg_lane);

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc3);

        int8 b1 = load_b_tile(b_base + (k + 16) * N, N);

        short8 a4 = load_a_col(cur_slm, 16 + sg_lane);
        short8 a5 = load_a_col(cur_slm + 8 * SLM_STRIDE, 16 + sg_lane);
        short8 a6 = load_a_col(cur_slm + 16 * SLM_STRIDE, 16 + sg_lane);
        short8 a7 = load_a_col(cur_slm + 24 * SLM_STRIDE, 16 + sg_lane);

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a4, b1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a5, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a6, b1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a7, b1, acc3);
    }

    // Store results using subgroup block writes
    __global ushort* c_ptr = (__global ushort*)(C + tile_row * N + tile_col);

    #pragma unroll
    for (int i = 0; i < 8; i++)
        intel_sub_group_block_write_us(c_ptr + (0 + i) * N, as_ushort(convert_half(((float*)&acc0)[i])));
    #pragma unroll
    for (int i = 0; i < 8; i++)
        intel_sub_group_block_write_us(c_ptr + (8 + i) * N, as_ushort(convert_half(((float*)&acc1)[i])));
    #pragma unroll
    for (int i = 0; i < 8; i++)
        intel_sub_group_block_write_us(c_ptr + (16 + i) * N, as_ushort(convert_half(((float*)&acc2)[i])));
    #pragma unroll
    for (int i = 0; i < 8; i++)
        intel_sub_group_block_write_us(c_ptr + (24 + i) * N, as_ushort(convert_half(((float*)&acc3)[i])));
}
```