

## Analysis

For Intel Battlemage (Xe2-HPG) with XMX engines, I need to leverage the DPAS (Dot Product Accumulate Systolic) instructions via `intel_sub_group_f16_f16_matrix_mad_k16` for optimal FP16 matmul performance.

**Key design decisions:**

1. **Subgroup size = 16** (native for Xe2-HPG)
2. **DPAS instruction**: `intel_sub_group_f16_f16_matrix_mad_k16` computes an 8x16 output tile per subgroup, where each work-item in the subgroup holds 8 accumulator values (8 rows), and the subgroup covers 16 columns. The K dimension is processed in chunks of 16.
3. **Tiling**: Each subgroup computes an 8×16 tile of C. A workgroup of (16, 16) = 256 work-items = 16 subgroups. With subgroups mapped along the N dimension, each workgroup handles a 128×16 block (16 subgroups × 8 rows each × 16 columns) — but actually better to tile as rows and columns.
4. **Workgroup tiling**: With LWS=(16,16), I'll have 16 subgroups per workgroup. Each subgroup (16 WIs along x) handles 16 columns and 8 rows. The 16 rows of WIs in y-dimension mean each WI-row maps to a different set of 8 rows → workgroup covers 8 rows × 16 columns per subgroup.

**Simplified approach**: Map each subgroup to compute an 8×16 output tile using DPAS. Each work-item holds 8 float accumulators. Loop over K in steps of 16, loading A and B tiles into registers, then calling DPAS.

- **A tile**: 8 rows × 16 K elements, loaded as 8 `int` values per WI (packed half pairs)
- **B tile**: 16 K × 16 N, loaded as 8 `int` values per WI via subgroup block reads

**Launch config**: 
- Each subgroup: 8 rows × 16 cols
- Workgroup (16,16): 16 subgroups, each handling 8×16
- GWS_X = ceil(N/16)*16, GWS_Y = ceil(M/8)*(16) → but we need to map subgroups to tiles

Let me use a simpler tiled approach with SLM and subgroup operations that's more portable while still being efficient:

## OCL Code

```OCL
// Tiled FP16 matmul using SLM blocking for Intel Xe2-HPG (Battlemage)
// C[M,N] = A[M,K] x B[K,N], accumulation in FP32
//
// Each workgroup computes a TILE_M x TILE_N block of C.
// TILE_M=64, TILE_N=64, TILE_K=16
// LWS = (16, 16) = 256 work-items
// Each work-item computes a 4x4 sub-tile of C.
// GWS = (ceil_div(N,64)*16, ceil_div(M,64)*16)

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define WG_X 16
#define WG_Y 16
#define ITEMS_PER_WI_M 4
#define ITEMS_PER_WI_N 4

__attribute__((reqd_work_group_size(WG_X, WG_Y, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Workgroup tile origin
    const int wg_id_x = get_group_id(0);
    const int wg_id_y = get_group_id(1);
    const int tile_col = wg_id_x * TILE_N;
    const int tile_row = wg_id_y * TILE_M;

    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lid = lid_y * WG_X + lid_x;

    // Each work-item accumulates a 4x4 sub-tile
    float acc[ITEMS_PER_WI_M][ITEMS_PER_WI_N];
    for (int i = 0; i < ITEMS_PER_WI_M; i++)
        for (int j = 0; j < ITEMS_PER_WI_N; j++)
            acc[i][j] = 0.0f;

    // SLM tiles for A and B
    __local half A_tile[TILE_M * TILE_K];  // 64 x 16
    __local half B_tile[TILE_K * TILE_N];  // 16 x 64

    const int total_wis = WG_X * WG_Y; // 256

    // Loop over K dimension in blocks of TILE_K
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperatively load A_tile[TILE_M][TILE_K] = 64*16 = 1024 elements
        // 256 work-items, each loads 4 elements
        for (int idx = lid; idx < TILE_M * TILE_K; idx += total_wis) {
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int global_r = tile_row + r;
            int global_c = k0 + c;
            half val = (global_r < M && global_c < K) ? A[global_r * K + global_c] : (half)0.0h;
            A_tile[r * TILE_K + c] = val;
        }

        // Cooperatively load B_tile[TILE_K][TILE_N] = 16*64 = 1024 elements
        for (int idx = lid; idx < TILE_K * TILE_N; idx += total_wis) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int global_r = k0 + r;
            int global_c = tile_col + c;
            half val = (global_r < K && global_c < N) ? B[global_r * N + global_c] : (half)0.0h;
            B_tile[r * TILE_N + c] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each WI computes its 4x4 sub-tile
        // WI (lid_x, lid_y) covers rows [lid_y*4 .. lid_y*4+3], cols [lid_x*4 .. lid_x*4+3]
        int local_row_base = lid_y * ITEMS_PER_WI_M;
        int local_col_base = lid_x * ITEMS_PER_WI_N;

        for (int kk = 0; kk < TILE_K; kk++) {
            // Load A values for this WI's rows
            float a_val[ITEMS_PER_WI_M];
            for (int i = 0; i < ITEMS_PER_WI_M; i++) {
                a_val[i] = convert_float(A_tile[(local_row_base + i) * TILE_K + kk]);
            }

            // Load B values for this WI's columns
            float b_val[ITEMS_PER_WI_N];
            for (int j = 0; j < ITEMS_PER_WI_N; j++) {
                b_val[j] = convert_float(B_tile[kk * TILE_N + local_col_base + j]);
            }

            // Outer product accumulation
            for (int i = 0; i < ITEMS_PER_WI_M; i++) {
                for (int j = 0; j < ITEMS_PER_WI_N; j++) {
                    acc[i][j] = fma(a_val[i], b_val[j], acc[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write results back to C
    int local_row_base = lid_y * ITEMS_PER_WI_M;
    int local_col_base = lid_x * ITEMS_PER_WI_N;

    for (int i = 0; i < ITEMS_PER_WI_M; i++) {
        for (int j = 0; j < ITEMS_PER_WI_N; j++) {
            int global_r = tile_row + local_row_base + i;
            int global_c = tile_col + local_col_base + j;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = convert_half(acc[i][j]);
            }
        }
    }
}
```

