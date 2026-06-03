

## Analysis

Looking at the previous versions, all achieve ~1.3ms with TILE_M=64, TILE_N=128, TILE_K=32. The key bottleneck is the low DPAS-to-barrier ratio. With 16 DPAS per K-tile and barriers for double buffering, we're not getting enough compute density.

Key observations:
1. **Version 1 & 2** both use double-buffered SLM for A and B with similar structure - ~1.3ms
2. **The last tested kernel** reads B from global instead of SLM - 2.16ms (worse due to redundant global reads across subgroups)
3. The `elem_id / TILE_K` and `elem_id % TILE_K` use expensive integer division - should use power-of-2 shifts/masks

**Proposed improvements:**
1. **TILE_K=64**: 4 k16 steps × 8 row-blocks = 32 DPAS per barrier pair (vs 16 currently)
2. **Use bitwise ops** for division by TILE_K (64 is power of 2: shift by 6, mask with 63)
3. **Pre-load B into registers** before DPAS loop to hide SLM latency
4. **Reduce SLM usage**: With TILE_K=64, A is 64×64=8192 halves, B is 64×128=16384 halves per buffer. Double-buffered: ~49KB. This might be too much. Let's keep TILE_K=32 but try to minimize barriers.
5. **Better approach**: Keep TILE_K=32, but reorganize to do all loads + barrier + all compute + barrier pattern more efficiently. Or use TILE_K=64 with single-buffered + prefetch into registers.

Actually, let me try a different approach: **TILE_M=64, TILE_N=128, TILE_K=32** but with the loads truly overlapped with compute using double buffering, and ensure we use bitwise operations (TILE_K=32 → shift by 5, mask 31).

The real issue might be that versions 1&2 have the loads and compute serialized despite "double buffering" - the barrier still serializes them. The key insight is: with double buffering, we load the NEXT tile while computing the CURRENT tile, then one barrier suffices. Let me ensure this is done correctly and minimize overhead.

Let me try **TILE_K=64** to double the DPAS count per barrier:
- A tile: 64×64 = 4096 halves, 32 per WI
- B tile: 64×128 = 8192 halves, 64 per WI  
- SLM: single buffer (no double buffering needed if we prefetch into registers)

Actually with 64 per WI for B, that's too many loads. Let me try a different approach: **register prefetch** - load B from global directly into registers (no SLM for B), only SLM for A.

Wait, that was the worst performer (2.16ms). The issue is each subgroup reads different B columns but same B rows, so SLM sharing helps.

Let me stick with the best structure (Version 1/2) but fix the integer division issue and ensure proper double buffering:

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// TILE_M=64, TILE_N=128, TILE_K=32
// 8 subgroups x 16 WIs = 128 WIs per WG
// Each subgroup: 8 row-blocks of 8 rows x 16 cols = 64x16
// Per K-tile: 2 k16 steps x 8 row-blocks = 16 DPAS per subgroup
// True double buffering: load next tile concurrent with DPAS on current tile
// SLM A: 2 x 64 x 34 halves, SLM B: 2 x 32 x 130 halves
// GWS = (ceil(N/128)*128, ceil(M/64))  LWS = (128, 1)

#define TM 64
#define TN 128
#define TK 32
#define A_STRIDE 34
#define B_STRIDE 130
#define NWI 128

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

    const int n_base = get_group_id(0) * TN;
    const int row_base = get_group_id(1) * TM;

    __local half slm_A[2 * TM * A_STRIDE];
    __local half slm_B[2 * TK * B_STRIDE];

    if (row_base >= M || n_base >= N)
        return;

    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float8 acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    const int col_idx = n_base + sg_id * 16 + sg_lid;
    const int sg_col_off = sg_id * 16;
    const bool row_full = (row_base + TM <= M);
    const bool col_full = (n_base + TN <= N);
    const int nkt = (K + TK - 1) / TK;

    // Load first tile into buffer 0
    // A: 64x32 = 2048 halves / 128 WIs = 16 per WI
    // Use bit ops: TK=32, so /32 = >>5, %32 = &31
    {
        __local half* a0 = slm_A;
        __local half* b0 = slm_B;
        if (row_full && TK <= K) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * NWI;
                int r = eid >> 5;
                int c = eid & 31;
                a0[r * A_STRIDE + c] = A[(row_base + r) * K + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int eid = lid + i * NWI;
                int r = eid >> 5;
                int c = eid & 31;
                int gr = row_base + r;
                a0[r * A_STRIDE + c] = (gr < M && c < K) ? A[gr * K + c] : (half)0.0h;
            }
        }
        // B: 32x128 = 4096 halves / 128 WIs = 32 per WI
        // /128 = >>7, %128 = &127
        if (col_full && TK <= K) {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int eid = lid + i * NWI;
                int r = eid >> 7;
                int c = eid & 127;
                b0[r * B_STRIDE + c] = B[r * N + n_base + c];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int eid = lid + i * NWI;
                int r = eid >> 7;
                int c = eid & 127;
                int gn = n_base + c;
                b0[r * B_STRIDE + c] = (r < K && gn < N) ? B[r * N + gn] : (half)0.0h;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kt = 0; kt < nkt; kt++) {
        int cb = kt & 1;
        int nb = 1 - cb;
        int k_next = (kt + 1) * TK;
        bool has_next = (kt + 1 < nkt);

        __local const half* cA = slm_A + cb * TM * A_STRIDE;
        __local const half* cB = slm_B + cb * TK * B_STRIDE;

        // Pre-load all B blocks into registers for both k16 steps
        int8 bv0, bv1;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            short s0 = as_short(cB[(2*p) * B_STRIDE + sg_col_off + sg_lid]);
            short s1 = as_short(cB[(2*p+1) * B_STRIDE + sg_col_off + sg_lid]);
            ((int*)&bv0)[p] = as_int((short2)(s0, s1));
        }
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            short s0 = as_short(cB[(16+2*p) * B_STRIDE + sg_col_off + sg_lid]);
            short s1 = as_short(cB[(16+2*p+1) * B_STRIDE + sg_col_off + sg_lid]);
            ((int*)&bv1)[p] = as_int((short2)(s0, s1));
        }

        // Pre-load all A blocks into registers for k16 step 0
        short8 a0r, a1r, a2r, a3r, a4r, a5r, a6r, a7r;
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a0r)[r] = as_short(cA[r * A_STRIDE + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a1r)[r] = as_short(cA[(8+r) * A_STRIDE + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a2r)[r] = as_short(cA[(16+r) * A_STRIDE + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a3r)[r] = as_short(cA[(24+r) * A_STRIDE + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a4r)[r] = as_short(cA[(32+r) * A_STRIDE + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a5r)[r] = as_short(cA[(40+r) * A_STRIDE + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a6r)[r] = as_short(cA[(48+r) * A_STRIDE + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a7r)[r] = as_short(cA[(56+r) * A_STRIDE + sg_lid]);

        // k16 step 0: 8 DPAS
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0r, bv0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1r, bv0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2r, bv0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3r, bv0, acc3);
        acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a4r, bv0, acc4);
        acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a5r, bv0, acc5);
        acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a6r, bv0, acc6);
        acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a7r, bv0, acc7);

        // Load A blocks for k16 step 1
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a0r)[r] = as_short(cA[r * A_STRIDE + 16 + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a1r)[r] = as_short(cA[(8+r) * A_STRIDE + 16 + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a2r)[r] = as_short(cA[(16+r) * A_STRIDE + 16 + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a3r)[r] = as_short(cA[(24+r) * A_STRIDE + 16 + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a4r)[r] = as_short(cA[(32+r) * A_STRIDE + 16 + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a5r)[r] = as_short(cA[(40+r) * A_STRIDE + 16 + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a6r)[r] = as_short(cA[(48+r) * A_STRIDE + 16 + sg_lid]);
        #pragma unroll
        for (int r = 0; r < 8; r++) ((short*)&a7r)[r] = as_short(cA[(56+r) * A_STRIDE + 16 + sg_lid]);

        // k16 step 1: 8 DPAS
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0r, bv1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1r, bv1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2r, bv1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3r, bv1, acc3);
        acc4 = intel_sub_group_f16_f16_matrix_mad_k16(a4r, bv1, acc4);
        acc5 = intel_sub_group_f16_f16_matrix_mad_k16(a5r, bv1, acc5);
        acc6 = intel_sub_group_f16_f16_matrix_mad_k16(a6r, bv1, acc6);
        acc7 = intel_sub_group_f16_f16_matrix_mad_k16(a7r, bv1, acc7);

        // Load next tile
        if (has_next) {
            __local half* nA = slm_A + nb * TM * A_STRIDE;
            __local half* nB = slm_B + nb * TK * B_STRIDE;

            if (row_full && k_next + TK <= K) {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int eid = lid + i * NWI;
                    int r = eid >> 5;
                    int c = eid & 31;
                    nA[r * A_STRIDE + c] = A[(row_base + r) * K + k_next + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    int eid = lid + i * NWI;
                    int r = eid >> 5;
                    int c = eid & 31;
                    int gr = row_base + r;
                    int gk = k_next + c;
                    nA[r * A_STRIDE + c] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
                }
            }

            if (col_full && k_next + TK <= K) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int eid = lid + i * NWI;
                    int r = eid >> 7;
                    int c = eid & 127;
                    nB[r * B_STRIDE + c] = B[(k_next + r) * N + n_base + c];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    int eid = lid + i * NWI;
                    int r = eid >> 7;
                    int c = eid & 127;
                    int gk = k_next + r;
                    int gn = n_base + c;
                    nB[r * B_STRIDE + c] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    // Store
    if (col_idx < N) {
        #define ST(off, av) \
        { \
            int ro = row_base + (off); \
            if (ro + 8 <= M) { \
                _Pragma("unroll") \
                for (int r = 0; r < 8; r++) \
                    C[(ro+r)*N+col_idx] = convert_half(((float*)&av)[r]); \
            } else { \
                _Pragma("unroll") \
                for (int r = 0; r < 8; r++) \
                    if (ro+r < M) C[(ro+r)*N+col_idx] = convert_half(((float*)&av)[r]); \
            } \
        }
        ST(0,acc0) ST(8,acc1) ST(16,acc2) ST(24,acc3)
        ST(32,acc4) ST(40,acc5) ST(48,acc6) ST(56,acc7)
        #undef ST
    }
}
```

