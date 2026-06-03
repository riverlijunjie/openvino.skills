

## Analysis

For this Intel Battlemage (Xe2-HPG) optimized matmul kernel, I need to leverage:

1. **Intel XMX DPAS instructions** (`intel_sub_group_f16_f16_matrix_mad_k16`) for FP16 matrix multiply
2. **SLM tiling** to reduce global memory traffic
3. **Subgroup-level cooperative loading** for efficient memory access

### DPAS Instruction Semantics
`intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc)` performs:
- Each subgroup (16 work-items) computes an 8×16 result tile
- `a`: 8 half values per work-item (8×16 tile, K=16)
- `b`: 8 half values per work-item (16×16 tile, K=16)  
- `acc`: 8 float values per work-item (8×16 accumulator)
- Result: 8×16 output tile in float

### Tiling Strategy
- **Work-group tile**: 32×64 (M×N) — each work-group handles this block of C
- **Subgroup tile**: 8×16 — natural DPAS output size
- **Work-group composition**: 4 subgroups along M (4×8=32), 4 along N (4×16=64) → 16 subgroups → 256 work-items per work-group (16 per subgroup × 16 subgroups)
- **K-loop tile**: 16 (matches DPAS k16)
- **SLM**: Store A tile (32×16 = 1KB) and B tile (16×64 = 2KB) in half precision

### Launch Configuration
- LWS: (64, 4) — 256 work-items, 16 subgroups of size 16
- GWS: (ceil(N/64)*64, ceil(M/32)*4)

```OCL
// Optimized FP16 matmul using Intel DPAS (XMX) instructions
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Tiling: WG tile = 32x64 (MxN), DPAS tile = 8x16, K-tile = 16
// Each WG has 16 subgroups (4 along M, 4 along N), subgroup size = 16
// LWS = (64, 4) = 256 work-items
// GWS = (ceil(N/64)*64, ceil(M/32)*4)
// SLM usage: A_tile[32][16] + B_tile[16][64] = 1024 + 2048 = 3072 bytes (half)

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(64, 4, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile origin in C
    const int wg_n = (get_group_id(0)) * 64;  // N offset
    const int wg_m = (get_group_id(1)) * 32;  // M offset

    // Local IDs
    const int lid_x = get_local_id(0); // 0..63
    const int lid_y = get_local_id(1); // 0..3
    const int lid = lid_y * 64 + lid_x; // flat local id 0..255

    // Subgroup identification
    const int sg_id = get_sub_group_id();        // 0..15
    const int sg_lid = get_sub_group_local_id();  // 0..15

    // Map 16 subgroups into 4x4 grid (4 along M, 4 along N)
    const int sg_m = sg_id / 4;  // 0..3, each covers 8 rows
    const int sg_n = sg_id % 4;  // 0..3, each covers 16 cols

    // Subgroup's tile origin within WG tile
    const int sg_row_offset = sg_m * 8;   // 0, 8, 16, 24
    const int sg_col_offset = sg_n * 16;  // 0, 16, 32, 48

    // SLM tiles
    __local half A_slm[32 * 16];  // 32 rows x 16 cols
    __local half B_slm[16 * 64];  // 16 rows x 64 cols

    // Accumulator: 8 floats per work-item (8 rows of the 8x16 DPAS output)
    float8 acc = (float8)(0.0f);

    // Loop over K dimension in tiles of 16
    for (int k_base = 0; k_base < K; k_base += 16) {
        // Cooperative load of A tile [32][16] into SLM
        // 256 work-items, 32*16 = 512 elements, 2 elements per work-item
        {
            int idx0 = lid * 2;
            int idx1 = lid * 2 + 1;
            
            int a_row0 = idx0 / 16;
            int a_col0 = idx0 % 16;
            int a_row1 = idx1 / 16;
            int a_col1 = idx1 % 16;
            
            int gm0 = wg_m + a_row0;
            int gk0 = k_base + a_col0;
            int gm1 = wg_m + a_row1;
            int gk1 = k_base + a_col1;
            
            A_slm[idx0] = (gm0 < M && gk0 < K) ? A[gm0 * K + gk0] : (half)0.0h;
            A_slm[idx1] = (gm1 < M && gk1 < K) ? A[gm1 * K + gk1] : (half)0.0h;
        }

        // Cooperative load of B tile [16][64] into SLM
        // 256 work-items, 16*64 = 1024 elements, 4 elements per work-item
        {
            for (int i = 0; i < 4; i++) {
                int idx = lid + i * 256;
                int b_row = idx / 64;
                int b_col = idx % 64;
                
                int gk = k_base + b_row;
                int gn = wg_n + b_col;
                
                B_slm[idx] = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each subgroup performs DPAS: 8x16 = 8x16_k16 matmul
        // Load A sub-tile: 8 rows, 16 cols from A_slm
        // For DPAS a input: each work-item i holds A[sg_row_offset + row][col] 
        // packed as 8 halfs (8 rows, work-item's K-column = sg_lid-th pair)
        
        // a: each work-item holds 8 half values = rows 0..7, at k-columns
        // In intel_sub_group_f16_f16_matrix_mad_k16:
        //   a is int8 (8 x int, each int packs 2 half values) representing 8x16 matrix
        //   b is int8 representing 16x16 matrix
        //   Result is float8 (8 rows x 1 col per work-item, 16 work-items = 8x16)

        // Pack A: 8 rows x 16 cols, stored as int8 per work-item
        // Each work-item sg_lid reads column pair (sg_lid*2, sg_lid*2+1) -- but that's not right
        // Actually for DPAS, data layout:
        //   a[i] for work-item j: a is distributed across subgroup
        //   Each work-item holds 8 ints, each int = 2 halfs from K dimension
        //   work-item j holds k-indices [2j, 2j+1] for all 8 rows
        
        int8 a_packed;
        for (int r = 0; r < 8; r++) {
            int slm_row = sg_row_offset + r;
            int k0 = sg_lid * 2;
            int k1 = sg_lid * 2 + 1;
            half va0 = A_slm[slm_row * 16 + k0];
            half va1 = A_slm[slm_row * 16 + k1];
            // Pack two halfs into one int
            ushort u0 = as_ushort(va0);
            ushort u1 = as_ushort(va1);
            ((int*)&a_packed)[r] = (int)u0 | ((int)u1 << 16);
        }

        // Pack B: 16 rows x 16 cols (the subgroup's 16-col slice)
        // Each work-item j holds k-indices [2j, 2j+1] for all 16 output columns
        // But B is organized differently: 
        //   b[i] for work-item j: work-item j holds column j, rows packed in pairs
        //   Each int = 2 halfs from K dimension at column sg_col_offset + sg_lid
        
        int8 b_packed;
        for (int p = 0; p < 8; p++) {
            int k0 = p * 2;
            int k1 = p * 2 + 1;
            int col = sg_col_offset + sg_lid;
            half vb0 = B_slm[k0 * 64 + col];
            half vb1 = B_slm[k1 * 64 + col];
            ushort ub0 = as_ushort(vb0);
            ushort ub1 = as_ushort(vb1);
            ((int*)&b_packed)[p] = (int)ub0 | ((int)ub1 << 16);
        }

        // DPAS: 8x16 result = 8x16(A) * 16x16(B)
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: each work-item writes 8 values
    // Work-item sg_lid in subgroup writes column (sg_col_offset + sg_lid)
    // rows sg_row_offset + 0..7
    int out_col = wg_n + sg_col_offset + sg_lid;
    
    for (int r = 0; r < 8; r++) {
        int out_row = wg_m + sg_row_offset + r;
        if (out_row < M && out_col < N) {
            float val = ((float*)&acc)[r];
            C[out_row * N + out_col] = convert_half(val);
        }
    }
}
```

