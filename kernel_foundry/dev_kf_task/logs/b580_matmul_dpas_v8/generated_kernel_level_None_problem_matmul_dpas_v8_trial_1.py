

## Analysis

The last working kernel (332ms) has several inefficiencies:

1. **Single subgroup per work-group**: Only 16 work-items per WG means low occupancy and poor latency hiding.
2. **Scalar SLM loads**: Loading A and B element-by-element from SLM is inefficient. Each element requires a separate memory transaction.
3. **Manual VNNI packing**: B matrix VNNI packing via individual ushort reads is slow.
4. **Store with switch statements**: Extracting float8 components via switch is suboptimal.
5. **Only 32×32 tile per WG with 1 subgroup**: Compute density is low relative to SLM traffic.

**Proposed improvements:**
- Use **multiple subgroups per work-group** (e.g., 4 subgroups) to increase occupancy and share SLM data.
- Use **larger K tile** (32 instead of 16) to amortize SLM load cost with two DPAS calls per K-tile load.
- **Prefetch B into VNNI format in SLM** to avoid per-subgroup repacking.
- Use **block reads from SLM** where possible for A data.
- Streamline the store path.

## Improved OCL code

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Tile: 32M x 64N per work-group, K-tile = 16
// 4 subgroups per WG, each subgroup computes 8M x 64N (using 4 DPAS calls per K-step)
// Actually: 4 subgroups, each handles 8 rows x 64 cols = 4 x (8x16) DPAS blocks
// LWS = (16, 4) => 64 WIs, 4 subgroups of 16
// GWS = (ceil(N/64)*16, ceil(M/32)*4)

#define TILE_M 32
#define TILE_N 64
#define TILE_K 16
#define SG_SIZE 16
#define NUM_SG 4

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int wg_n = get_group_id(0) * TILE_N;
    const int wg_m = get_group_id(1) * TILE_M;
    
    const int sg_id = get_local_id(1);  // 0..3, which subgroup (row block)
    const int sg_lid = get_sub_group_local_id(); // 0..15

    const int row_base = wg_m + sg_id * 8; // each subgroup handles 8 rows

    // Each subgroup computes 8 rows x 64 cols = 4 DPAS blocks (8x16 each)
    float8 acc0 = 0.0f; // cols 0-15
    float8 acc1 = 0.0f; // cols 16-31
    float8 acc2 = 0.0f; // cols 32-47
    float8 acc3 = 0.0f; // cols 48-63

    // SLM: A[32][16] + B in VNNI format [8][64] as uint (= 16 k-rows x 64 cols packed)
    // A: 32*16 halfs = 512 halfs = 1024 bytes
    // B_vnni: 8*64 uints = 512 uints = 2048 bytes (16 k-rows x 64 cols, k-pairs packed)
    __local ushort slm_a[TILE_M * TILE_K]; // 32*16 = 512
    __local uint slm_b_vnni[8 * TILE_N];   // 8*64 = 512

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // === Cooperative load A[32][16] ===
        // 64 WIs total, 512 elements => 8 elements per WI
        {
            int flat_id = get_local_id(1) * 16 + get_local_id(0); // 0..63
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id * 8 + i;
                int r = idx >> 4;  // idx / 16
                int c = idx & 15;  // idx % 16
                int gr = wg_m + r;
                int gc = k0 + c;
                ushort val = 0;
                if (gr < M && gc < K)
                    val = as_ushort(A[gr * K + gc]);
                slm_a[r * TILE_K + c] = val;
            }
        }

        // === Cooperative load B[16][64] into VNNI format ===
        // VNNI: pack k-pairs => 8 pairs x 64 cols = 512 uints
        // 64 WIs, 512 elements => 8 per WI
        {
            int flat_id = get_local_id(1) * 16 + get_local_id(0);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int idx = flat_id * 8 + i;
                int pair = idx >> 6;  // idx / 64, which k-pair (0..7)
                int col = idx & 63;   // idx % 64, which column (0..63)
                int k_row0 = k0 + pair * 2;
                int k_row1 = k0 + pair * 2 + 1;
                int gcol = wg_n + col;
                ushort v0 = 0, v1 = 0;
                if (k_row0 < K && gcol < N)
                    v0 = as_ushort(B[k_row0 * N + gcol]);
                if (k_row1 < K && gcol < N)
                    v1 = as_ushort(B[k_row1 * N + gcol]);
                slm_b_vnni[pair * TILE_N + col] = ((uint)v1 << 16) | (uint)v0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === Load A for this subgroup: 8 rows x 16 cols ===
        __local ushort* a_ptr = slm_a + sg_id * 8 * TILE_K;
        short8 a_val = as_short8((ushort8)(
            a_ptr[0*TILE_K + sg_lid], a_ptr[1*TILE_K + sg_lid],
            a_ptr[2*TILE_K + sg_lid], a_ptr[3*TILE_K + sg_lid],
            a_ptr[4*TILE_K + sg_lid], a_ptr[5*TILE_K + sg_lid],
            a_ptr[6*TILE_K + sg_lid], a_ptr[7*TILE_K + sg_lid]));

        // === Load B VNNI blocks (already packed) ===
        // Block j: cols [j*16 .. j*16+15], 8 pairs
        // Each WI reads its column (sg_lid) from each pair
        __local uint* b_base = slm_b_vnni;

        int8 b0 = (int8)(
            as_int(b_base[0*TILE_N + 0*16 + sg_lid]), as_int(b_base[1*TILE_N + 0*16 + sg_lid]),
            as_int(b_base[2*TILE_N + 0*16 + sg_lid]), as_int(b_base[3*TILE_N + 0*16 + sg_lid]),
            as_int(b_base[4*TILE_N + 0*16 + sg_lid]), as_int(b_base[5*TILE_N + 0*16 + sg_lid]),
            as_int(b_base[6*TILE_N + 0*16 + sg_lid]), as_int(b_base[7*TILE_N + 0*16 + sg_lid]));

        int8 b1 = (int8)(
            as_int(b_base[0*TILE_N + 16 + sg_lid]), as_int(b_base[1*TILE_N + 16 + sg_lid]),
            as_int(b_base[2*TILE_N + 16 + sg_lid]), as_int(b_base[3*TILE_N + 16 + sg_lid]),
            as_int(b_base[4*TILE_N + 16 + sg_lid]), as_int(b_base[5*TILE_N + 16 + sg_lid]),
            as_int(b_base[6*TILE_N + 16 + sg_lid]), as_int(b_base[7*TILE_N + 16 + sg_lid]));

        int8 b2 = (int8)(
            as_int(b_base[0*TILE_N + 32 + sg_lid]), as_int(b_base[1*TILE_N + 32 + sg_lid]),
            as_int(b_base[2*TILE_N + 32 + sg_lid]), as_int(b_base[3*TILE_N + 32 + sg_lid]),
            as_int(b_base[4*TILE_N + 32 + sg_lid]), as_int(b_base[5*TILE_N + 32 + sg_lid]),
            as_int(b_base[6*TILE_N + 32 + sg_lid]), as_int(b_base[7*TILE_N + 32 + sg_lid]));

        int8 b3 = (int8)(
            as_int(b_base[0*TILE_N + 48 + sg_lid]), as_int(b_base[1*TILE_N + 48 + sg_lid]),
            as_int(b_base[2*TILE_N + 48 + sg_lid]), as_int(b_base[3*TILE_N + 48 + sg_lid]),
            as_int(b_base[4*TILE_N + 48 + sg_lid]), as_int(b_base[5*TILE_N + 48 + sg_lid]),
            as_int(b_base[6*TILE_N + 48 + sg_lid]), as_int(b_base[7*TILE_N + 48 + sg_lid]));

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b2, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b3, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = row_base + i;
        if (row < M) {
            float v0 = ((float*)&acc0)[i];
            float v1 = ((float*)&acc1)[i];
            float v2 = ((float*)&acc2)[i];
            float v3 = ((float*)&acc3)[i];

            int col0 = wg_n + sg_lid;
            int col1 = wg_n + 16 + sg_lid;
            int col2 = wg_n + 32 + sg_lid;
            int col3 = wg_n + 48 + sg_lid;

            if (col0 < N) C[row * N + col0] = convert_half(v0);
            if (col1 < N) C[row * N + col1] = convert_half(v1);
            if (col2 < N) C[row * N + col2] = convert_half(v2);
            if (col3 < N) C[row * N + col3] = convert_half(v3);
        }
    }
}
```

