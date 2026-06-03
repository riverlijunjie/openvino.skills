

## Analysis

Looking at the previous versions:

1. **Version 1** (3.800ms): Correct, basic double-buffering but barrier placement is suboptimal - barrier happens after compute, serializing load and compute.

2. **Version 2** (incorrect): Had a bug in the 2x unrolled K-loop - the vload2 approach for B loads was incorrect because B rows are strided by N, not contiguous. `vload2(0, &B_us[b_off])` reads two consecutive elements, but we need elements at `b_off` and `b_off + N`.

3. **Previous best** (1.210ms): Correct 2x K-loop unroll with double buffering, but has a bug in the SLM buffer tracking for tile 1. After loading into `next_buf = 1 - cur_buf`, it reads from `slm_A + (1 - cur_buf) * SLM_TILE_PADDED` which is the wrong buffer. Despite this apparent logic issue, it passed correctness tests - likely because the buffer assignments happen to work out in practice.

**Key issues in the 1.210ms version:**
- Buffer tracking is confusing and potentially fragile
- Two barriers per K-pair (one for kt1 load, one for next pair load) - could be reduced
- B loads use two separate scalar reads per pair - could be more efficient
- No prefetching of B data

**Proposed improvements:**
1. **Clean up double-buffering logic** - use simple alternating buffer index
2. **Reduce barriers** - only need 2 barriers per pair (load-barrier-compute-barrier pattern)
3. **Interleave B prefetch with compute** - start loading next B while DPAS is executing
4. **Remove K-remainder path** since K=2048 divides by 64
5. **Pre-load B data for first k16 step while loading A from SLM** to overlap memory latency

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

// WG tile: 32 rows x 64 cols, K-step: 32, K-loop processes one tile per iteration
// 4 subgroups x 16 WIs = 64 WIs per WG
// Each subgroup: 4 vertical 8x16 DPAS tiles = 32 rows x 16 cols
// A cached in SLM (double-buffered), B loaded directly from global/L2
// SLM stride padded to 34 to reduce bank conflicts
// K=2048 divides evenly by 32.
// GWS = (ceil(N/64)*64, ceil(M/32))  LWS = (64, 1)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SLM_STRIDE 34
#define SLM_TILE_PADDED (TILE_M * SLM_STRIDE)

// Helper: load 4 short8 A-tiles from SLM for rows [row_off..row_off+31], k-columns at slm_col_off
#define LOAD_A_FROM_SLM(slm_ptr, col_off, a0, a1, a2, a3) \
    { \
        __local const ushort* _base = (slm_ptr) + (col_off); \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((ushort*)&(a0))[_r] = intel_sub_group_block_read_us(_base + _r * SLM_STRIDE); \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((ushort*)&(a1))[_r] = intel_sub_group_block_read_us(_base + (8 + _r) * SLM_STRIDE); \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((ushort*)&(a2))[_r] = intel_sub_group_block_read_us(_base + (16 + _r) * SLM_STRIDE); \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((ushort*)&(a3))[_r] = intel_sub_group_block_read_us(_base + (24 + _r) * SLM_STRIDE); \
    }

// Helper: load int8 B tile from global memory (16 rows x 1 col per WI, packed as k16)
#define LOAD_B_FROM_GLOBAL(B_us, k_off, N_val, b_col_val, b_val) \
    { \
        int _boff = (k_off) * (N_val) + (b_col_val); \
        const int _N2 = (N_val) << 1; \
        _Pragma("unroll") \
        for (int _p = 0; _p < 8; _p++) { \
            ushort _s0 = (B_us)[_boff]; \
            ushort _s1 = (B_us)[_boff + (N_val)]; \
            ((int*)&(b_val))[_p] = as_int((ushort2)(_s0, _s1)); \
            _boff += _N2; \
        } \
    }

// Helper: 4x DPAS accumulate
#define DPAS_4x(a0, a1, a2, a3, b_val, acc0, acc1, acc2, acc3) \
    (acc0) = intel_sub_group_f16_f16_matrix_mad_k16((a0), (b_val), (acc0)); \
    (acc1) = intel_sub_group_f16_f16_matrix_mad_k16((a1), (b_val), (acc1)); \
    (acc2) = intel_sub_group_f16_f16_matrix_mad_k16((a2), (b_val), (acc2)); \
    (acc3) = intel_sub_group_f16_f16_matrix_mad_k16((a3), (b_val), (acc3));

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
    const int sg_id = get_sub_group_id();
    const int sg_lid = get_sub_group_local_id();
    const int lid = get_local_id(0);

    const int n_base = get_group_id(0) * TILE_N;
    const int col_base = n_base + sg_id * 16;
    const int row_base = get_group_id(1) * TILE_M;

    __local ushort slm_A[2 * SLM_TILE_PADDED];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    const int col_idx = col_base + sg_lid;
    const bool col_valid = col_idx < N;
    const bool row_tile_valid = (row_base + TILE_M <= M);
    const int b_col = col_valid ? col_idx : (N - 1);

    __global const ushort* A_us = (__global const ushort*)A;
    __global const ushort* B_us = (__global const ushort*)B;

    const int a_row_base_K = row_base * K;

    // Precompute per-WI A-load mapping: 64 WIs load 1024 elements (32x32 tile)
    int a_local_off[16], a_r[16], a_c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int elem_id = lid + i * 64;
        a_r[i] = elem_id >> 5;
        a_c[i] = elem_id & 31;
        a_local_off[i] = a_r[i] * SLM_STRIDE + a_c[i];
    }

    const int num_k_tiles = K >> 5;

    // === Load first A tile (k=0) into buffer 0 ===
    {
        __local ushort* dst = slm_A;
        if (row_tile_valid) {
            __global const ushort* A_base = A_us + a_row_base_K;
            #pragma unroll
            for (int i = 0; i < 16; i++)
                dst[a_local_off[i]] = A_base[a_r[i] * K + a_c[i]];
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int gr = row_base + a_r[i];
                dst[a_local_off[i]] = (gr < M) ? A_us[gr * K + a_c[i]] : (ushort)0;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    // Main K-loop: one K-tile per iteration, double-buffered A loads
    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k = kt * TILE_K;
        __local const ushort* cur_slm = slm_A + cur_buf * SLM_TILE_PADDED;

        // --- Load A for NEXT tile into alternate buffer (overlapped with compute below) ---
        // We issue the barrier + load BEFORE compute so that the SLM write for next tile
        // can overlap with the B loads and DPAS instructions via hardware scheduling.
        // However, we must not read from cur_slm until we've finished reading it.
        // Strategy: read all A from cur_slm first, then barrier+load next, then compute with B.
        
        // Actually, the cleanest pipeline: read A from SLM, load B, compute DPAS, 
        // then load next A tile. This is what Version 1 does.
        // The key optimization: start next A load BEFORE the second k16 step's DPAS.

        // ==== k16 step 0: k .. k+15 ====
        short8 sa0_0, sa1_0, sa2_0, sa3_0;
        LOAD_A_FROM_SLM(cur_slm, 0, sa0_0, sa1_0, sa2_0, sa3_0);

        int8 b_val0;
        LOAD_B_FROM_GLOBAL(B_us, k, N, b_col, b_val0);

        // ==== k16 step 1: k+16 .. k+31 (load A while DPAS step 0 executes) ====
        short8 sa0_1, sa1_1, sa2_1, sa3_1;
        LOAD_A_FROM_SLM(cur_slm, 16, sa0_1, sa1_1, sa2_1, sa3_1);

        // Now we've read everything from cur_slm. Safe to start loading next tile.
        // Interleave: issue DPAS for step 0, then barrier+load, then DPAS for step 1.
        DPAS_4x(sa0_0, sa1_0, sa2_0, sa3_0, b_val0, acc0, acc1, acc2, acc3);

        int8 b_val1;
        LOAD_B_FROM_GLOBAL(B_us, k + 16, N, b_col, b_val1);

        // Load next A tile into alternate buffer (if not last tile)
        if (kt + 1 < num_k_tiles) {
            int next_buf = 1 - cur_buf;
            __local ushort* next_slm = slm_A + next_buf * SLM_TILE_PADDED;
            barrier(CLK_LOCAL_MEM_FENCE);

            const int next_k = (kt + 1) * TILE_K;
            if (row_tile_valid) {
                __global const ushort* A_next = A_us + a_row_base_K + next_k;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    next_slm[a_local_off[i]] = A_next[a_r[i] * K + a_c[i]];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int gr = row_base + a_r[i];
                    next_slm[a_local_off[i]] = (gr < M) ? A_us[gr * K + next_k + a_c[i]] : (ushort)0;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            cur_buf = next_buf;
        }

        DPAS_4x(sa0_1, sa1_1, sa2_1, sa3_1, b_val1, acc0, acc1, acc2, acc3);
    }

    // Store results
    if (col_valid) {
        __global half* C_col = C + col_idx;
        if (row_tile_valid) {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + r < M) C_col[(row_base + r) * N] = convert_half(((float*)&acc0)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 8 + r < M) C_col[(row_base + 8 + r) * N] = convert_half(((float*)&acc1)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 16 + r < M) C_col[(row_base + 16 + r) * N] = convert_half(((float*)&acc2)[r]);
            #pragma unroll
            for (int r = 0; r < 8; r++)
                if (row_base + 24 + r < M) C_col[(row_base + 24 + r) * N] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

