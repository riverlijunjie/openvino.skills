

## Analysis

The best version so far (53.6ms) uses TILE_M=64, TILE_N=64, TILE_K=32 with SLM padding to avoid bank conflicts. Key observations:

1. **Version 1 (65.8ms)**: TILE_K=16, no padding, no subgroup hint - decent but leaves performance on the table
2. **Version 2 (172ms)**: TILE_M/N=128 too large - 8x8 per WI means too many SLM reads and register pressure
3. **Best version (53.6ms)**: TILE_K=32 with A padding works well, but still uses scalar half loads from SLM

**Improvements to try:**
- Use vectorized loads (vload4/vload8) for both global→SLM and SLM→registers to increase memory throughput
- Increase TILE_N to 128 while keeping TILE_M=64, giving each WI a 4x8 block (32 accumulators, manageable register pressure)
- Use `__global const half8*` style vectorized global loads for better bandwidth utilization
- Double-buffer or prefetch pattern isn't easy in OCL, but we can try larger K tiles
- Keep A padding for bank conflict avoidance

Let me try TILE_M=64, TILE_N=128, TILE_K=32 with LWS=(16,16), each WI computes 4x8. This gives more work per WI while keeping register pressure reasonable (32 floats).

Actually, let me reconsider - the best version already works well with 4x4. Let me try to optimize the memory access patterns more aggressively with vectorized loads and a slightly different tiling.

## Improved OCL code

```OCL
// Tiled FP16 matmul: C[M,N] = A[M,K] x B[K,N], accumulation in FP32
// TILE_M=64, TILE_N=128, TILE_K=32
// LWS = (16, 16) = 256 work-items
// Each WI computes a 4x8 sub-tile of C (32 accumulators)
// GWS = (ceil_div(N,128)*16, ceil_div(M,64)*16)
// SLM: A_tile = 64*32 = 2048 halfs (4KB padded), B_tile = 32*128 = 4096 halfs (8KB)

#define TILE_M 64
#define TILE_N 128
#define TILE_K 32
#define WG_X 16
#define WG_Y 16
#define ITEMS_M 4
#define ITEMS_N 8
#define TOTAL_WIS 256

// Pad A stride to avoid SLM bank conflicts
#define A_STRIDE (TILE_K + 2)
// Pad B stride similarly
#define B_STRIDE (TILE_N + 2)

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
    const int tile_col = get_group_id(0) * TILE_N;
    const int tile_row = get_group_id(1) * TILE_M;

    const int lid_x = get_local_id(0);
    const int lid_y = get_local_id(1);
    const int lid = lid_y * WG_X + lid_x;

    // 4x8 accumulators per WI
    float acc[ITEMS_M][ITEMS_N];
    #pragma unroll
    for (int i = 0; i < ITEMS_M; i++)
        #pragma unroll
        for (int j = 0; j < ITEMS_N; j++)
            acc[i][j] = 0.0f;

    // SLM tiles
    __local half A_tile[TILE_M * A_STRIDE];   // 64 * 34
    __local half B_tile[TILE_K * B_STRIDE];   // 32 * 130

    // Per-WI sub-tile position
    const int local_row_base = lid_y * ITEMS_M;  // 0,4,8,...,60
    const int local_col_base = lid_x * ITEMS_N;  // 0,8,16,...,120

    // A_tile: 64*32 = 2048 elements, 256 WIs => 8 each
    // B_tile: 32*128 = 4096 elements, 256 WIs => 16 each

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A_tile[TILE_M][TILE_K] into padded SLM
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int idx = lid + t * TOTAL_WIS;
            int r = idx >> 5;   // /32
            int c = idx & 31;   // %32
            int gr = tile_row + r;
            int gc = k0 + c;
            A_tile[r * A_STRIDE + c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Load B_tile[TILE_K][TILE_N] into padded SLM - 4096 elements, 16 per WI
        #pragma unroll
        for (int t = 0; t < 16; t++) {
            int idx = lid + t * TOTAL_WIS;
            int r = idx >> 7;    // /128
            int c = idx & 127;   // %128
            int gr = k0 + r;
            int gc = tile_col + c;
            B_tile[r * B_STRIDE + c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute 4x8 sub-tile
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            // Load 4 A values for this WI's rows
            float a_val[ITEMS_M];
            #pragma unroll
            for (int i = 0; i < ITEMS_M; i++) {
                a_val[i] = convert_float(A_tile[(local_row_base + i) * A_STRIDE + kk]);
            }

            // Load 8 B values for this WI's columns
            float b_val[ITEMS_N];
            #pragma unroll
            for (int j = 0; j < ITEMS_N; j++) {
                b_val[j] = convert_float(B_tile[kk * B_STRIDE + local_col_base + j]);
            }

            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < ITEMS_M; i++) {
                #pragma unroll
                for (int j = 0; j < ITEMS_N; j++) {
                    acc[i][j] = fma(a_val[i], b_val[j], acc[i][j]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < ITEMS_M; i++) {
        int gr = tile_row + local_row_base + i;
        if (gr >= M) continue;
        #pragma unroll
        for (int j = 0; j < ITEMS_N; j++) {
            int gc = tile_col + local_col_base + j;
            if (gc < N) {
                C[gr * N + gc] = convert_half(acc[i][j]);
            }
        }
    }
}
```

