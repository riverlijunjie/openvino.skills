

## Analysis

The previous versions all achieve ~1.31-1.33ms. The key bottleneck is the low DPAS-to-barrier ratio. With TILE_K=32, we get 2 k16 steps × 8 row-blocks = 16 DPAS per barrier pair, but the load overhead is still significant.

Key issues:
1. **TILE_K=32 limits DPAS density** - Only 2 k16 steps per iteration means loads dominate
2. **SLM load pattern** - Element-wise half loads are inefficient; we should use wider loads (vload4/vload8)
3. **Double buffering not truly overlapping** - The compiler may serialize loads and computes since they're in the same thread

**Proposed improvements:**
1. **TILE_K=64** - 4 k16 steps × 8 row-blocks = 32 DPAS per barrier pair (2× improvement)
2. **Use vload4 for SLM loads** - Wider memory transactions for cooperative loading
3. **Reduce SLM padding** - Use +0 or minimal padding since TILE_K=64 already has good alignment
4. **Tighter interleaving** - Load A during first 2 k16 steps, load B during last 2 k16 steps

SLM budget with TILE_K=64:
- A: 2 × 64 × 66 × 2 = 16896 bytes  
- B: 2 × 64 × 130 × 2 = 33280 bytes
- Total: ~50KB - might be too much. Let's use TILE_N=64 with 4 subgroups instead.

Actually, let me reconsider: TILE_M=64, TILE_N=64, TILE_K=64, 4 subgroups × 16 = 64 WIs:
- A: 2 × 64 × 66 × 2 = 16896 bytes
- B: 2 × 64 × 66 × 2 = 16896 bytes  
- Total: ~34KB - fine
- DPAS: 4 k16 × 8 row-blocks = 32 per subgroup per iteration
- But fewer WGs cover N dimension → may reduce parallelism

Better: Keep TILE_N=128, TILE_K=64, 8 subgroups:
- A: 2 × 64 × 66 × 2 = 16.5KB
- B: 2 × 64 × 130 × 2 = 32.5KB
- Total ~49KB - tight but should work on Xe2

A: 64×64 = 4096 halves / 128 WIs = 32 each
B: 64×128 = 8192 halves / 128 WIs = 64 each

That's a lot of loads per WI. Let me use vload4 to make them efficient.

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TILE_M=64, TILE_N=128, TILE_K=64
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 8 row-blocks of 8x16 = 64x16 output
// Per K-tile: 4 k16 steps x 8 row-blocks = 32 DPAS per subgroup per barrier
// Double-buffered SLM for A and B
// SLM A: 2 x 64 x 66 halves = 16896 bytes
// SLM B: 2 x 64 x 130 halves = 33280 bytes, total ~49KB
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)

#define TILE_M 64
#define TILE_N 128
#define TILE_K 64
#define SLM_A_STRIDE (TILE_K + 2)   // 66
#define SLM_B_STRIDE (TILE_N + 2)   // 130
#define NUM_WI 128

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128, 1, 1)))
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
    const int row_base = get_group_id(1) * TILE_M;

    __local half slm_A[2 * TILE_M * SLM_A_STRIDE];
    __local half slm_B[2 * TILE_K * SLM_B_STRIDE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float8 acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const int col_idx = n_base + sg_id * 16 + sg_lid;
    const int sg_col_off = sg_id * 16;
    const bool row_tile_full = (row_base + TILE_M <= M);
    const bool col_tile_full = (n_base + TILE_N <= N);
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // A tile: 64 x 64 = 4096 halves, 128 WIs => 32 each
    // B tile: 64 x 128 = 8192 halves, 128 WIs => 64 each

    // --- Helper: Load A tile cooperatively ---
    // Each WI loads 32 halves (use 8 iterations of vload4)
    #define LOAD_A(dst, k_off, need_bc) \
    { \
        _Pragma("unroll") \
        for (int _i = 0; _i < 32; _i++) { \
            int _eid = lid + _i * NUM_WI; \
            int _r = _eid >> 6; \
            int _c = _eid & 63; \
            if (need_bc) { \
                int _gr = row_base + _r; \
                int _gk = (k_off) + _c; \
                (dst)[_r * SLM_A_STRIDE + _c] = (_gr < M && _gk < K) ? A[_gr * K + _gk] : (half)0.0h; \
            } else { \
                (dst)[_r * SLM_A_STRIDE + _c] = A[(row_base + _r) * K + (k_off) + _c]; \
            } \
        } \
    }

    // --- Helper: Load B tile cooperatively ---
    // Each WI loads 64 halves
    #define LOAD_B(dst, k_off, need_bc) \
    { \
        _Pragma("unroll") \
        for (int _i = 0; _i < 64; _i++) { \
            int _eid = lid + _i * NUM_WI; \
            int _r = _eid >> 7; \
            int _c = _eid & 127; \
            if (need_bc) { \
                int _gk = (k_off) + _r; \
                int _gn = n_base + _c; \
                (dst)[_r * SLM_B_STRIDE + _c] = (_gk < K && _gn < N) ? B[_gk * N + _gn] : (half)0.0h; \
            } else { \
                (dst)[_r * SLM_B_STRIDE + _c] = B[((k_off) + _r) * N + n_base + _c]; \
            } \
        } \
    }

    // --- Helper: DPAS compute for one k16 step ---
    #define DPAS_K16(cur_A, cur_B, kk_off) \
    { \
        int8 _b; \
        _Pragma("unroll") \
        for (int _p = 0; _p < 8; _p++) { \
            short _s0 = as_short((cur_B)[((kk_off) + 2*_p) * SLM_B_STRIDE + sg_col_off + sg_lid]); \
            short _s1 = as_short((cur_B)[((kk_off) + 2*_p + 1) * SLM_B_STRIDE + sg_col_off + sg_lid]); \
            ((int*)&_b)[_p] = as_int((short2)(_s0, _s1)); \
        } \
        short8 _a; \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((short*)&_a)[_r] = as_short((cur_A)[_r * SLM_A_STRIDE + (kk_off) + sg_lid]); \
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(_a, _b, acc0); \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((short*)&_a)[_r] = as_short((cur_A)[(8+_r) * SLM_A_STRIDE + (kk_off) + sg_lid]); \
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(_a, _b, acc1); \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((short*)&_a)[_r] = as_short((cur_A)[(16+_r) * SLM_A_STRIDE + (kk_off) + sg_lid]); \
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(_a, _b, acc2); \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((short*)&_a)[_r] = as_short((cur_A)[(24+_r) * SLM_A_STRIDE + (kk_off) + sg_lid]); \
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(_a, _b, acc3); \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((short*)&_a)[_r] = as_short((cur_A)[(32+_r) * SLM_A_STRIDE + (kk_off) + sg_lid]); \
        acc4 = intel_sub_group_f16_f16_matrix_mad_k16(_a, _b, acc4); \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((short*)&_a)[_r] = as_short((cur_A)[(40+_r) * SLM_A_STRIDE + (kk_off) + sg_lid]); \
        acc5 = intel_sub_group_f16_f16_matrix_mad_k16(_a, _b, acc5); \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((short*)&_a)[_r] = as_short((cur_A)[(48+_r) * SLM_A_STRIDE + (kk_off) + sg_lid]); \
        acc6 = intel_sub_group_f16_f16_matrix_mad_k16(_a, _b, acc6); \
        _Pragma("unroll") \
        for (int _r = 0; _r < 8; _r++) \
            ((short*)&_a)[_r] = as_short((cur_A)[(56+_r) * SLM_A_STRIDE + (kk_off) + sg_lid]); \
        acc7 = intel_sub_group_f16_f16_matrix_mad_k16(_a, _b, acc7); \
    }

    const bool k_full = (K % TILE_K == 0);
    const bool all_full = row_tile_full && col_tile_full && k_full;

    // Load first tile into buffer 0
    if (all_full) {
        LOAD_A(slm_A, 0, 0)
        LOAD_B(slm_B, 0, 0)
    } else {
        LOAD_A(slm_A, 0, 1)
        LOAD_B(slm_B, 0, 1)
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int cur_buf = kt & 1;
        int nxt_buf = 1 - cur_buf;
        int k_next = (kt + 1) * TILE_K;
        bool has_next = (kt + 1 < num_k_tiles);

        __local const half* cur_A = slm_A + cur_buf * TILE_M * SLM_A_STRIDE;
        __local const half* cur_B = slm_B + cur_buf * TILE_K * SLM_B_STRIDE;

        // Compute k16 steps 0 and 1
        DPAS_K16(cur_A, cur_B, 0)
        DPAS_K16(cur_A, cur_B, 16)

        // Interleave: load next A while computing k16 steps 2-3
        if (has_next) {
            __local half* nxt_A = slm_A + nxt_buf * TILE_M * SLM_A_STRIDE;
            if (all_full || (row_tile_full && k_next + TILE_K <= K)) {
                LOAD_A(nxt_A, k_next, 0)
            } else {
                LOAD_A(nxt_A, k_next, 1)
            }
        }

        DPAS_K16(cur_A, cur_B, 32)
        DPAS_K16(cur_A, cur_B, 48)

        // Load next B
        if (has_next) {
            __local half* nxt_B = slm_B + nxt_buf * TILE_K * SLM_B_STRIDE;
            if (all_full || (col_tile_full && k_next + TILE_K <= K)) {
                LOAD_B(nxt_B, k_next, 0)
            } else {
                LOAD_B(nxt_B, k_next, 1)
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Store results
    if (col_idx < N) {
        #define STORE_BLK(off, acc_v) \
        { \
            int _ro = row_base + (off); \
            if (_ro + 8 <= M) { \
                _Pragma("unroll") \
                for (int _r = 0; _r < 8; _r++) \
                    C[(_ro + _r) * N + col_idx] = convert_half(((float*)&(acc_v))[_r]); \
            } else { \
                _Pragma("unroll") \
                for (int _r = 0; _r < 8; _r++) \
                    if (_ro + _r < M) \
                        C[(_ro + _r) * N + col_idx] = convert_half(((float*)&(acc_v))[_r]); \
            } \
        }

        STORE_BLK(0, acc0)
        STORE_BLK(8, acc1)
        STORE_BLK(16, acc2)
        STORE_BLK(24, acc3)
        STORE_BLK(32, acc4)
        STORE_BLK(40, acc5)
        STORE_BLK(48, acc6)
        STORE_BLK(56, acc7)

        #undef STORE_BLK
    }

    #undef LOAD_A
    #undef LOAD_B
    #undef DPAS_K16
}
```

