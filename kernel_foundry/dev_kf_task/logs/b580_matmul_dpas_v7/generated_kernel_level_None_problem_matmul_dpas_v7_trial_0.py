

# Analysis

## Task Requirements
We need to implement an FP16 matrix multiplication C[M,N] = A[M,K] × B[K,N] optimized for Intel Battlemage (Xe2-HPG) using DPAS instructions.

## Key Design Decisions

**DPAS Instruction**: `intel_sub_group_f16_f16_matrix_mad_k16` performs a matrix multiply-accumulate on FP16 data with FP32 accumulation. Each DPAS call in a subgroup of 8 work-items computes an 8×8 result tile from 8×16 × 16×8 inputs (with K=16 depth).

**Tiling Strategy**:
- **Work-group tile**: 32×32 (each work-group computes a 32×32 block of C)
- **Subgroup tile**: Each subgroup handles an 8×8 output tile via DPAS
- **Work-group has 4×4 = 16 subgroups** → but simpler: use 4 subgroups, each handling 8×32 or similar. 

Let me reconsider for simplicity and correctness:

- Work-group: 32 rows × 32 cols of C
- Each work-group has 32×4 = 128 work-items → 16 subgroups of 8
- Each subgroup computes an 8×8 tile of C
- 4×4 = 16 subgroups tile the 32×32 block
- K is tiled in chunks of 16 (matching DPAS k=16)
- SLM stores 32×16 tile of A and 16×32 tile of B

**DPAS details for `intel_sub_group_f16_f16_matrix_mad_k16`**:
- Subgroup size = 8
- Computes C[8×8] += A[8×16] × B[16×8]
- A is passed as 8 floats per work-item (each float packs 2 half values, 8 floats = 16 halves per row, 8 rows across subgroup)
- B is passed as 8 floats per work-item (packed similarly)
- Result is 8 floats per work-item

**SLM Layout**:
- Load A tile [32×16] and B tile [16×32] into SLM per K-step
- Each subgroup reads its portion for DPAS

```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Launch metadata:
//   LWS = (32, 4, 1)  => 128 work-items = 16 subgroups of 8
//   GWS = (ceil(N/32)*32, ceil(M/32)*4, 1)
//   Subgroup size = 8
//   Work-group tile: 32x32 of C
//   Each subgroup: 8x8 of C via DPAS with k=16 steps
//   SLM: 32*16*2 + 16*32*2 = 1024 + 1024 = 2048 bytes per k-step

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile origin
    const int tN = 32;  // tile cols
    const int tM = 32;  // tile rows
    const int tK = 16;  // tile K depth (matches DPAS k16)

    const int wg_col = get_group_id(0) * tN;  // column offset in C
    const int wg_row = get_group_id(1) * tM;  // row offset in C

    // Local IDs
    const int lid0 = get_local_id(0);  // 0..31
    const int lid1 = get_local_id(1);  // 0..3
    const int flat_lid = lid1 * 32 + lid0;  // 0..127

    // Subgroup info
    const int sg_id = get_sub_group_id();        // 0..15
    const int sg_lid = get_sub_group_local_id(); // 0..7

    // Each subgroup handles an 8x8 tile within the 32x32 work-group tile
    // sg_id maps to (sg_row, sg_col) in a 4x4 grid
    const int sg_row = sg_id / 4;  // 0..3
    const int sg_col = sg_id % 4;  // 0..3

    // SLM for A tile [32][16] and B tile [16][32], stored as half
    __local half slm_A[32 * 16];
    __local half slm_B[16 * 32];

    // Accumulator: 8 rows x 8 cols per subgroup, each work-item holds 8 floats
    // (one column of the 8x8 tile, across 8 rows)
    float8 acc = (float8)(0.0f);

    // Loop over K in steps of 16
    for (int k0 = 0; k0 < K; k0 += tK) {
        // Cooperative load of A[wg_row..wg_row+31][k0..k0+15] into slm_A
        // 128 work-items load 32*16 = 512 halves => 4 halves each
        for (int i = flat_lid; i < 32 * 16; i += 128) {
            int r = i / 16;
            int c = i % 16;
            int gr = wg_row + r;
            int gc = k0 + c;
            slm_A[r * 16 + c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0f;
        }

        // Cooperative load of B[k0..k0+15][wg_col..wg_col+31] into slm_B
        for (int i = flat_lid; i < 16 * 32; i += 128) {
            int r = i / 32;
            int c = i % 32;
            int gr = k0 + r;
            int gc = wg_col + c;
            slm_B[r * 32 + c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each subgroup computes its 8x8 tile using DPAS
        // A sub-tile: rows [sg_row*8 .. sg_row*8+7], cols [0..15] from slm_A
        // B sub-tile: rows [0..15], cols [sg_col*8 .. sg_col*8+7] from slm_B

        // Pack A for DPAS: each work-item i in subgroup holds row (sg_row*8 + i)
        // Need 16 half values packed into 8 floats (2 halves per float)
        int a_row = sg_row * 8 + sg_lid;
        float8 a_packed;
        __local half* a_ptr = &slm_A[a_row * 16];
        // Pack pairs of halves into float using as_float(as_uint((ushort2)(v0,v1)))
        for (int j = 0; j < 8; j++) {
            ushort lo = as_ushort(a_ptr[j * 2]);
            ushort hi = as_ushort(a_ptr[j * 2 + 1]);
            ((float*)&a_packed)[j] = as_float((uint)lo | ((uint)hi << 16));
        }

        // Pack B for DPAS: B is [16][8] sub-tile
        // For intel_sub_group_f16_f16_matrix_mad_k16, B needs VNNI-like layout
        // B is packed as: for each group of 2 K-rows, interleave 8 columns
        // B layout: [K/2][8][2] packed into floats
        // Each work-item holds 8 floats corresponding to 8 pairs of K-rows
        int b_col_base = sg_col * 8;
        float8 b_packed;
        for (int j = 0; j < 8; j++) {
            int k_pair = j;  // pairs 0..7 cover k=0..15
            ushort lo = as_ushort(slm_B[(k_pair * 2) * 32 + b_col_base + sg_lid]);
            ushort hi = as_ushort(slm_B[(k_pair * 2 + 1) * 32 + b_col_base + sg_lid]);
            ((float*)&b_packed)[j] = as_float((uint)lo | ((uint)hi << 16));
        }

        // DPAS: C[8x8] += A[8x16] * B[16x8]
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: each work-item in subgroup writes one column of the 8x8 tile
    int out_col = wg_col + sg_col * 8 + sg_lid;
    for (int i = 0; i < 8; i++) {
        int out_row = wg_row + sg_row * 8 + i;
        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = convert_half(((float*)&acc)[i]);
        }
    }
}
```

This kernel:
1. **Uses DPAS** (`intel_sub_group_f16_f16_matrix_mad_k16`) for peak throughput on XMX engines
2. **SLM tiling**: Loads 32×16 (A) and 16×32 (B) tiles cooperatively into shared local memory
3. **Hierarchical parallelism**: Work-groups own 32×32 tiles, 16 subgroups of 8 each own 8×8 tiles, DPAS operates at register level
4. **Subgroup size 8**: Matches Battlemage Xe2-HPG DPAS requirements
5. **K=16 inner tile**: Matches the DPAS k16 variant for FP16
6. **Packs half pairs into float**: Required VNNI-style packing for DPAS operands

