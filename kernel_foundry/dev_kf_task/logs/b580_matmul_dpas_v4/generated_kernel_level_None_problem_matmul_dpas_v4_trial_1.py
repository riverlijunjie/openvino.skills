

## Analysis

The previous best kernel (11.6ms) uses a single subgroup per workgroup computing an 8x16 output tile. Key inefficiencies:

1. **No SLM tiling** - Each subgroup loads A and B from global memory every K iteration, causing redundant global memory traffic.
2. **Small workgroup** - Only 1 subgroup per WG means low occupancy and poor latency hiding.
3. **Scalar loads with bounds checks** - Per-element loads with conditionals are slow; we should use block loads where possible.
4. **No data reuse** - A tiles loaded by one subgroup could be shared with others computing different N-tiles.

**Proposed improvements:**
- Use **SLM tiling**: Load A and B tiles into local memory, share A across subgroups computing different N-tiles.
- Use **multiple subgroups per workgroup** (e.g., 4 subgroups = 64 WIs): Each WG computes a 32x32 tile (4 subgroups in a 4x1 layout sharing A, each doing 8x32... or 2x2 layout doing 16x32).
- **Larger WG tile** = better A reuse through SLM.
- Keep DPAS for the inner compute.

**Design:** 
- WG tile: 32 rows × 32 cols
- 4 subgroups per WG arranged as 4 along M (each subgroup: 8 rows × 32 cols)
- Each subgroup does 2 DPAS calls per K-step (for 2 × 16 cols)
- A tile (32×16) loaded into SLM cooperatively, B tile (16×32) loaded into SLM cooperatively
- K iterated in steps of 16

```OCL
// Tiled FP16 matmul with SLM + DPAS
// C[M,N] = A[M,K] x B[K,N]
// WG tile: 32 rows x 32 cols, K-tile: 16
// 4 subgroups per WG (sg_size=16), LWS=(16,4,1)=64 WIs
// GWS = (ceil(N/32)*16, ceil(M/32)*4, 1)
// Each subgroup computes 8 rows x 32 cols (two 8x16 DPAS ops per K-step)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define SG_ROWS 8

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
    // SLM for A tile [32][16] and B tile [16][32]
    __local half A_slm[TILE_M * TILE_K];   // 32*16 = 512 halfs = 1KB
    __local half B_slm[TILE_K * TILE_N];   // 16*32 = 512 halfs = 1KB

    const int sg_lid = get_sub_group_local_id(); // 0..15
    const int sg_id  = get_sub_group_id();       // 0..3
    const int lid    = get_local_id(0) + get_local_id(1) * 16; // 0..63

    const int wg_m = get_group_id(1) * TILE_M;  // row start
    const int wg_n = get_group_id(0) * TILE_N;  // col start (note: gid(0) maps N/32 tiles, but GWS x = ceil(N/32)*16)

    // Actually let's re-derive group_id mapping:
    // GWS = (ceil(N/32)*16, ceil(M/32)*4)
    // get_group_id(0) = n_tile_idx (since LWS x=16, groups along x = ceil(N/32))
    // get_group_id(1) = m_tile_idx (since LWS y=4, groups along y = ceil(M/32))

    const int row_base = wg_m + sg_id * SG_ROWS; // each subgroup owns 8 rows
    // Each subgroup computes 8 rows x 32 cols (two 8x16 blocks: cols [0..15] and [16..31])

    // Accumulators: two 8x16 tiles
    float8 acc0 = 0.0f; // rows [row_base..+7], cols [wg_n..+15]
    float8 acc1 = 0.0f; // rows [row_base..+7], cols [wg_n+16..+31]

    for (int k = 0; k < K; k += TILE_K) {
        // Cooperative load of A_slm[32][16]: 512 elements, 64 WIs -> 8 elements each
        for (int i = lid; i < TILE_M * TILE_K; i += 64) {
            int r = i / TILE_K;  // row within tile (0..31)
            int c = i % TILE_K;  // col within tile (0..15)
            int grow = wg_m + r;
            int gcol = k + c;
            A_slm[i] = (grow < M && gcol < K) ? A[grow * K + gcol] : (half)0.0h;
        }

        // Cooperative load of B_slm[16][32]: 512 elements, 64 WIs -> 8 elements each
        for (int i = lid; i < TILE_K * TILE_N; i += 64) {
            int r = i / TILE_N;  // k-row (0..15)
            int c = i % TILE_N;  // n-col (0..31)
            int grow = k + r;
            int gcol = wg_n + c;
            B_slm[i] = (grow < K && gcol < N) ? B[grow * N + gcol] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each subgroup: load A from SLM for its 8 rows
        short8 a_val;
        for (int r = 0; r < 8; r++) {
            int slm_idx = (sg_id * SG_ROWS + r) * TILE_K + sg_lid;
            ((short*)&a_val)[r] = as_short(A_slm[slm_idx]);
        }

        // Load B for first 16 cols (col offset 0)
        int8 b_val0;
        for (int p = 0; p < 8; p++) {
            int k0 = 2 * p;
            int k1 = 2 * p + 1;
            half bv0 = B_slm[k0 * TILE_N + sg_lid];       // cols 0..15
            half bv1 = B_slm[k1 * TILE_N + sg_lid];
            short2 packed = (short2)(as_short(bv0), as_short(bv1));
            ((int*)&b_val0)[p] = as_int(packed);
        }

        // Load B for second 16 cols (col offset 16)
        int8 b_val1;
        for (int p = 0; p < 8; p++) {
            int k0 = 2 * p;
            int k1 = 2 * p + 1;
            half bv0 = B_slm[k0 * TILE_N + 16 + sg_lid];  // cols 16..31
            half bv1 = B_slm[k1 * TILE_N + 16 + sg_lid];
            short2 packed = (short2)(as_short(bv0), as_short(bv1));
            ((int*)&b_val1)[p] = as_int(packed);
        }

        // Two DPAS calls: 8x16 * 16x16 each
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val1, acc1);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    // acc0: 8 rows x 16 cols (lane sg_lid -> col sg_lid)
    // acc1: 8 rows x 16 cols (lane sg_lid -> col 16+sg_lid)
    for (int r = 0; r < 8; r++) {
        int row_idx = row_base + r;
        if (row_idx < M) {
            int col0 = wg_n + sg_lid;
            if (col0 < N)
                C[row_idx * N + col0] = convert_half(((float*)&acc0)[r]);

            int col1 = wg_n + 16 + sg_lid;
            if (col1 < N)
                C[row_idx * N + col1] = convert_half(((float*)&acc1)[r]);
        }
    }
}
```

