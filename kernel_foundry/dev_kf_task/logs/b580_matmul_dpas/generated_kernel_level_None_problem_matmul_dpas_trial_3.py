

## Analysis

The previous versions show progressive improvement:
- **V1 (62.3ms)**: 64x64 tile, 4x4 per WI, but uses 2D SLM arrays with potential bank conflicts
- **V2 (27.7ms)**: 32x32 tile with padding to avoid bank conflicts, unrolled inner loop
- **V3 (26.0ms)**: Similar to V2 but without padding - slightly faster due to simpler indexing

Key bottlenecks in V3:
1. Only 2x2 output per work-item → low arithmetic intensity
2. Scalar loads from SLM → not utilizing vector capabilities
3. TILE_K=32 inner loop not unrolled
4. Small tile size means more global memory traffic relative to compute

**Proposed improvements:**
- Increase to **4x4 output per work-item** (TILE_M=64, TILE_N=64) for higher arithmetic intensity
- Use **vectorized SLM loads** (vload4) to reduce load instruction count
- **Fully unroll** inner K loop with `#pragma unroll`
- Use **register blocking** with arrays and FMA
- Pre-load A values outside the j-loop to reduce redundant SLM reads

## Improved OCL code

```OCL
// Tiled FP16 matmul: C[M,N] = A[M,K] x B[K,N]
// Each WI computes 4x4 output block, accumulate in float
// LWS = (16, 16, 1), GWS = (ceil_div(N,64)*16, ceil_div(M,64)*16, 1)
// Subgroup size: 16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define WG_X 16
#define WG_Y 16
#define TM 4
#define TN 4

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

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lid = ly * WG_X + lx;

    // Each WI owns a 4x4 block
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0.0f;

    // SLM tiles
    __local half As[TILE_M * TILE_K];  // 64*16 = 1024
    __local half Bs[TILE_K * TILE_N];  // 16*64 = 1024

    for (int kk = 0; kk < K; kk += TILE_K) {
        // Load A tile [64 x 16] = 1024 elements, 256 threads -> 4 each
        #pragma unroll
        for (int i = lid; i < TILE_M * TILE_K; i += 256) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int gr = tile_row + r;
            int gc = kk + c;
            As[i] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }

        // Load B tile [16 x 64] = 1024 elements, 256 threads -> 4 each
        #pragma unroll
        for (int i = lid; i < TILE_K * TILE_N; i += 256) {
            int r = i / TILE_N;
            int c = i % TILE_N;
            int gr = kk + r;
            int gc = tile_col + c;
            Bs[i] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: WI (lx, ly) handles rows [ly*4..ly*4+3], cols [lx*4..lx*4+3]
        int base_r = ly * TM;
        int base_c = lx * TN;

        #pragma unroll
        for (int tk = 0; tk < TILE_K; ++tk) {
            // Load A values into registers
            float a_reg[TM];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a_reg[i] = convert_float(As[(base_r + i) * TILE_K + tk]);

            // Load B values into registers
            float b_reg[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b_reg[j] = convert_float(Bs[tk * TILE_N + base_c + j]);

            // Outer product accumulation
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store 4x4 results
    int out_row0 = tile_row + ly * TM;
    int out_col0 = tile_col + lx * TN;

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gr = out_row0 + i;
        if (gr < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int gc = out_col0 + j;
                if (gc < N) {
                    C[gr * N + gc] = convert_half(acc[i][j]);
                }
            }
        }
    }
}
```

