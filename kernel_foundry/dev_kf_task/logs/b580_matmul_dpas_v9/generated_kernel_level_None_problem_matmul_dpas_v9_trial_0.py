

## Analysis

This is a FP16 matrix multiplication C[M,N] = A[M,K] × B[K,N] that needs to leverage Intel XMX (DPAS) instructions on Battlemage GPU.

**Key design decisions:**

1. **DPAS instruction**: `intel_sub_group_f16_f16_matrix_mad_k16` computes an 8×16 × 16×16 → 8×16 tile per subgroup. Each work-item in a subgroup of 16 holds one column, computing 8 rows. The `short8` input A represents 8 rows × 16 K-elements (packed as fp16), `int8` input B represents 16K × 16N (packed), and `float8` is the accumulator for 8 output rows.

2. **Tiling strategy**: 
   - TILE_M = 32, TILE_N = 64, TILE_K = 16
   - Each workgroup: 2 subgroups along M (2×8=16... let me reconsider)
   - Actually, let me use: TILE_M=32, TILE_N=32, with each subgroup handling 8×16 output tile
   - Workgroup has multiple subgroups to cover the tile

3. **Register blocking**: Each subgroup computes 8 rows × 16 cols via DPAS. To get larger tiles, we can have each work-item do multiple DPAS calls (e.g., 2 along M for 16 rows, or 2 along N for 32 cols).

4. **SLM**: Load tiles of A and B into SLM, then feed DPAS from SLM. Pad SLM to avoid bank conflicts.

Let me design a concrete approach:
- **TILE_M = 32, TILE_N = 64, TILE_K = 16**
- Subgroup size = 16
- Each workgroup: 4×4 = 16 subgroups? No, too many.
- Better: Each subgroup computes 8×16 output. For 32×64 tile, need 4×4=16 subgroups → 256 work-items. That's reasonable.
- Actually let me simplify: **TILE_M=32, TILE_N=64, TILE_K=32**
- 4 subgroups along M (4×8=32), 4 along N (4×16=64) → 16 subgroups, 256 WIs
- Each subgroup does 2 DPAS calls per K-tile iteration (K=32, each DPAS does k=16)

Let me reconsider for simplicity and correctness. I'll use:
- **TILE_M=32, TILE_N=16, TILE_K=16** with 4 subgroups (each 8×16), giving 32×16 output per WG
- Or better: each subgroup computes multiple 8×16 blocks via register blocking.

**Final design:**
- WG tile: 32M × 32N
- Subgroup count per WG: 4 (along M) × 2 (along N) = 8 subgroups = 128 WIs
- Each subgroup: 8M × 16N via one DPAS per k=16 step
- K-loop iterates over K in steps of 16
- Load A[32×16] and B[16×32] into SLM per iteration
- SLM padding: A is 32×(16+1), B is 16×(32+1)

Actually, for DPAS the B matrix needs to be in a specific packed format. Let me use direct global memory reads with `intel_sub_group_block_read` instead of SLM for simplicity and correctness, then add SLM.

Let me go with a proven approach using SLM tiles.

```OCL
// Optimized FP16 GEMM using Intel XMX DPAS instructions
// C[M,N] = A[M,K] x B[K,N], all half precision, float accumulation
//
// Tiling: TILE_M=32, TILE_N=32, TILE_K=16
// Workgroup: 8 subgroups of 16 WIs = 128 WIs
//   - 4 subgroups along M (4*8=32 rows)
//   - 2 subgroups along N (2*16=32 cols)
// Each subgroup computes 8x16 output via DPAS
// K-loop steps by 16, using SLM for A and B tiles
//
// Launch config:
//   GWS = (N/32 * 16, M/32 * 8)  -- adjusted for subgroup mapping
//   LWS = (16, 8)  -- 128 WIs = 8 subgroups of 16
//   Subgroup size = 16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define SG_SIZE 16
#define SLM_A_STRIDE (TILE_K + 2)   // pad to avoid bank conflicts
#define SLM_B_STRIDE (TILE_N + 2)

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Workgroup tile position
    const int wg_n = get_group_id(0) * TILE_N;  // column start
    const int wg_m = get_group_id(1) * TILE_M;  // row start

    // Local IDs
    const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int sg_id = lid / SG_SIZE;        // subgroup index [0..7]
    const int sg_lid = lid % SG_SIZE;       // lane within subgroup [0..15]

    // Subgroup position within workgroup tile
    // 4 subgroups along M, 2 along N
    const int sg_row = sg_id / 2;   // [0..3] -> which 8-row block
    const int sg_col = sg_id % 2;   // [0..1] -> which 16-col block

    const int out_row_start = wg_m + sg_row * 8;
    const int out_col_start = wg_n + sg_col * 16;

    // Accumulator: 8 rows × 16 cols, each WI holds 8 floats (one per row, its column is sg_lid)
    float8 acc = (float8)(0.0f);

    // SLM for tiles
    __local half slm_a[TILE_M * SLM_A_STRIDE];  // 32 x 18
    __local half slm_b[TILE_K * SLM_B_STRIDE];  // 16 x 34

    const int total_wis = 128; // 8 subgroups * 16

    // K-loop
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperative load A[wg_m..wg_m+31, k0..k0+15] into SLM
        // Total elements: 32 * 16 = 512, with 128 WIs -> 4 elements each
        for (int i = lid; i < TILE_M * TILE_K; i += total_wis) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gm = wg_m + row;
            int gk = k0 + col;
            half val = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0f;
            slm_a[row * SLM_A_STRIDE + col] = val;
        }

        // Cooperative load B[k0..k0+15, wg_n..wg_n+31] into SLM
        // Total elements: 16 * 32 = 512, with 128 WIs -> 4 elements each
        for (int i = lid; i < TILE_K * TILE_N; i += total_wis) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int gk = k0 + row;
            int gn = wg_n + col;
            half val = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0f;
            slm_b[row * SLM_B_STRIDE + col] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each subgroup reads its A block (8 rows x 16 K) and B block (16 K x 16 cols)
        // A: rows [sg_row*8 .. sg_row*8+7], cols [0..15] in SLM
        // B: rows [0..15], cols [sg_col*16 .. sg_col*16+15] in SLM

        // Read A: 8 rows x 16 half values = short8 per WI
        // For DPAS, A needs to be in VNNI-like format: short8 where each short is a pair of halfs
        // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
        //   a: short8 - each WI holds 8 shorts, representing 8 rows, each with k-data distributed across subgroup
        //   b: int8 - packed B matrix

        // Load A tile for this subgroup from SLM
        // A is 8 rows x 16 cols of half. We need short8 per work-item.
        // Each work-item sg_lid provides the k-index. For row r, the value is A[r, sg_lid].
        // short8 means 8 rows, each row has one half value at column sg_lid, packed as short.
        short8 a_val;
        __local half* a_base = slm_a + sg_row * 8 * SLM_A_STRIDE;
        a_val.s0 = as_short(a_base[0 * SLM_A_STRIDE + sg_lid]);
        a_val.s1 = as_short(a_base[1 * SLM_A_STRIDE + sg_lid]);
        a_val.s2 = as_short(a_base[2 * SLM_A_STRIDE + sg_lid]);
        a_val.s3 = as_short(a_base[3 * SLM_A_STRIDE + sg_lid]);
        a_val.s4 = as_short(a_base[4 * SLM_A_STRIDE + sg_lid]);
        a_val.s5 = as_short(a_base[5 * SLM_A_STRIDE + sg_lid]);
        a_val.s6 = as_short(a_base[6 * SLM_A_STRIDE + sg_lid]);
        a_val.s7 = as_short(a_base[7 * SLM_A_STRIDE + sg_lid]);

        // Load B tile for this subgroup from SLM
        // B is 16 rows x 16 cols of half. For DPAS, int8 packing:
        // int8 = 8 ints, each int packs 2 consecutive half values (2 rows) for one column
        // Layout: b_val.s0 = pack(B[0, sg_lid], B[1, sg_lid]), etc.
        __local half* b_base = slm_b + sg_col * 16;
        int8 b_val;
        b_val.s0 = as_int((half2)(b_base[0 * SLM_B_STRIDE + sg_lid], b_base[1 * SLM_B_STRIDE + sg_lid]));
        b_val.s1 = as_int((half2)(b_base[2 * SLM_B_STRIDE + sg_lid], b_base[3 * SLM_B_STRIDE + sg_lid]));
        b_val.s2 = as_int((half2)(b_base[4 * SLM_B_STRIDE + sg_lid], b_base[5 * SLM_B_STRIDE + sg_lid]));
        b_val.s3 = as_int((half2)(b_base[6 * SLM_B_STRIDE + sg_lid], b_base[7 * SLM_B_STRIDE + sg_lid]));
        b_val.s4 = as_int((half2)(b_base[8 * SLM_B_STRIDE + sg_lid], b_base[9 * SLM_B_STRIDE + sg_lid]));
        b_val.s5 = as_int((half2)(b_base[10 * SLM_B_STRIDE + sg_lid], b_base[11 * SLM_B_STRIDE + sg_lid]));
        b_val.s6 = as_int((half2)(b_base[12 * SLM_B_STRIDE + sg_lid], b_base[13 * SLM_B_STRIDE + sg_lid]));
        b_val.s7 = as_int((half2)(b_base[14 * SLM_B_STRIDE + sg_lid], b_base[15 * SLM_B_STRIDE + sg_lid]));

        // DPAS: 8x16 = 8x16 * 16x16
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    // Each WI holds 8 float values for rows [out_row_start..+7], col [out_col_start + sg_lid]
    int out_col = out_col_start + sg_lid;
    if (out_col < N) {
        if (out_row_start + 0 < M) C[(out_row_start + 0) * N + out_col] = convert_half(acc.s0);
        if (out_row_start + 1 < M) C[(out_row_start + 1) * N + out_col] = convert_half(acc.s1);
        if (out_row_start + 2 < M) C[(out_row_start + 2) * N + out_col] = convert_half(acc.s2);
        if (out_row_start + 3 < M) C[(out_row_start + 3) * N + out_col] = convert_half(acc.s3);
        if (out_row_start + 4 < M) C[(out_row_start + 4) * N + out_col] = convert_half(acc.s4);
        if (out_row_start + 5 < M) C[(out_row_start + 5) * N + out_col] = convert_half(acc.s5);
        if (out_row_start + 6 < M) C[(out_row_start + 6) * N + out_col] = convert_half(acc.s6);
        if (out_row_start + 7 < M) C[(out_row_start + 7) * N + out_col] = convert_half(acc.s7);
    }
}
```

