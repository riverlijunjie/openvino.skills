

## Analysis

Looking at the previous versions, all achieve 33.9ms. The key issues are:

1. **Small tile sizes**: 16×16 or 32×32 tiles don't amortize memory load costs well enough
2. **SLM overhead**: Scalar loads into SLM with barriers dominate runtime
3. **Underutilized subgroups**: With only 1 subgroup per WG (16×1), there's not enough parallelism to hide latency
4. **Tail-K handled separately**: Scalar cleanup loop is expensive

**Proposed improvements:**
- **Larger tiles (64×64)** with register blocking: each subgroup computes multiple 8×16 output blocks
- **4 subgroups per WG** (64 work-items) for cooperative loading
- **Register blocking**: each subgroup accumulates 4 DPAS results (8×16 each → 32×16 per subgroup, 4 subgroups → 32×64 per WG... adjusted to 64×32)
- **Pad K to multiple of 16** by loading zeros, eliminating tail-K loop
- **Vectorized global loads** where possible

I'll use a 32×32 tile with 4 subgroups, where each subgroup computes 8×32 (two DPAS calls), but with direct register loading from global memory instead of going through SLM for B, and precomputing A tiles more efficiently.

Actually, let me try a cleaner approach: **64×32 tile**, 4 subgroups, each subgroup handles 16 rows × 32 cols (2 DPAS for lo/hi rows × 2 DPAS for left/right cols = 4 DPAS per K-step). This maximizes compute density.

```OCL
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// Launch metadata:
//   LWS = {16, 4, 1}  (64 WIs = 4 subgroups of 16 lanes)
//   GWS = {ceil_div(N,64)*16, ceil_div(M,32)*4, 1}
//   Each WG computes a 32×64 C tile
//   Each subgroup computes 8 rows × 64 cols = 4 DPAS calls per K-chunk
//   Register blocking: 4 float8 accumulators per subgroup

#define TILE_M 32
#define TILE_N 64
#define TILE_K 16
#define WG_X 16
#define WG_Y 4
#define NUM_WI 64

__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    const int lane = get_local_id(0);
    const int sg_id = get_local_id(1);
    const int gx = get_group_id(0);
    const int gy = get_group_id(1);

    const int tile_row = gy * TILE_M;
    const int tile_col = gx * TILE_N;

    // Each subgroup handles 8 rows
    const int sg_row_base = sg_id * 8;

    // Register blocking: 4 accumulators for 8 rows × 4×16 cols
    float8 acc0 = (float8)(0.0f);  // cols [0..15]
    float8 acc1 = (float8)(0.0f);  // cols [16..31]
    float8 acc2 = (float8)(0.0f);  // cols [32..47]
    float8 acc3 = (float8)(0.0f);  // cols [48..63]

    __local half Aslm[TILE_M][TILE_K];
    __local half Bslm[TILE_K][TILE_N];

    const int linear = sg_id * WG_X + lane;
    // Round K up to multiple of TILE_K to eliminate tail
    const int K_padded = ((K + TILE_K - 1) / TILE_K) * TILE_K;

    for (int kb = 0; kb < K_padded; kb += TILE_K) {
        // Cooperative load A: 32×16 = 512 elems, 64 WIs => 8 each
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int idx = linear + t * NUM_WI;
            int ar = idx >> 4;  // idx / 16
            int ak = idx & 15;  // idx % 16
            int gr = tile_row + ar;
            int gk = kb + ak;
            Aslm[ar][ak] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0.0h;
        }

        // Cooperative load B: 16×64 = 1024 elems, 64 WIs => 16 each
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            int idx = linear + t * NUM_WI;
            int bk = idx >> 6;  // idx / 64
            int bn = idx & 63;  // idx % 64
            int gk = kb + bk;
            int gn = tile_col + bn;
            Bslm[bk][bn] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Build A fragment: lanes 0-7 provide rows sg_row_base+0..7
        half16 a_vec = vload16(0, &Aslm[sg_row_base + (lane & 7)][0]);

        // Build B fragments for 4 column groups
        half16 b0, b1, b2, b3;
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            ((half*)&b0)[kk] = Bslm[kk][lane];
            ((half*)&b1)[kk] = Bslm[kk][16 + lane];
            ((half*)&b2)[kk] = Bslm[kk][32 + lane];
            ((half*)&b3)[kk] = Bslm[kk][48 + lane];
        }

        // 4 DPAS calls - register blocking maximizes compute per SLM load
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b2, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_vec, b3, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results - no tail K needed since we padded
    float acc_vals[4];
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        int gr = tile_row + sg_row_base + r;
        if (gr >= M) continue;

        switch(r) {
            case 0: acc_vals[0]=acc0.s0; acc_vals[1]=acc1.s0; acc_vals[2]=acc2.s0; acc_vals[3]=acc3.s0; break;
            case 1: acc_vals[0]=acc0.s1; acc_vals[1]=acc1.s1; acc_vals[2]=acc2.s1; acc_vals[3]=acc3.s1; break;
            case 2: acc_vals[0]=acc0.s2; acc_vals[1]=acc1.s2; acc_vals[2]=acc2.s2; acc_vals[3]=acc3.s2; break;
            case 3: acc_vals[0]=acc0.s3; acc_vals[1]=acc1.s3; acc_vals[2]=acc2.s3; acc_vals[3]=acc3.s3; break;
            case 4: acc_vals[0]=acc0.s4; acc_vals[1]=acc1.s4; acc_vals[2]=acc2.s4; acc_vals[3]=acc3.s4; break;
            case 5: acc_vals[0]=acc0.s5; acc_vals[1]=acc1.s5; acc_vals[2]=acc2.s5; acc_vals[3]=acc3.s5; break;
            case 6: acc_vals[0]=acc0.s6; acc_vals[1]=acc1.s6; acc_vals[2]=acc2.s6; acc_vals[3]=acc3.s6; break;
            case 7: acc_vals[0]=acc0.s7; acc_vals[1]=acc1.s7; acc_vals[2]=acc2.s7; acc_vals[3]=acc3.s7; break;
        }

        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            int gc = tile_col + c * 16 + lane;
            if (gc < N)
                C[gr * N + gc] = convert_half(acc_vals[c]);
        }
    }
}
```

