

## Analysis

For Intel Battlemage (Xe2-HPG) with XMX engines, the key optimization is using `intel_sub_group_f16_f16_matrix_mad_k16` DPAS instructions. This instruction computes a small matrix multiply where each subgroup (16 work-items) cooperatively computes an 8x16 tile of results, with K=16 reduction.

**Strategy:**
- Each subgroup computes a tile of C: 8 rows × 16 columns (DPAS native size: M=8, K=16, N=16)
- Workgroup: 16×16 = 256 work-items = 16 subgroups
- Each workgroup computes a 32×64 tile of C (arranged as 4×4 grid of 8×16 DPAS tiles... let me simplify)

Actually, let me use a cleaner approach:
- Subgroup size = 16
- Each subgroup handles one 8×16 output tile using DPAS
- Workgroup = (16, 16, 1) → 16 subgroups
- Arrange subgroups as 4 rows × 4 cols within workgroup → workgroup tile = 32×64

Wait, let me reconsider for simplicity and correctness. Each DPAS call: `intel_sub_group_f16_f16_matrix_mad_k16` takes:
- A: 8 half values per work-item (8×16 matrix distributed across 16 WIs)  
- B: 8 half values per work-item (16×16 matrix distributed across 16 WIs)
- Acc: 8 float values per work-item (8×16 result)

For the workgroup of 256 WIs = 16 subgroups, I'll arrange them as 4×4 grid of subgroups. Each subgroup computes 8×16 output. Total per WG: 32×64.

Loop over K in steps of 16, loading A tiles into registers and B tiles into SLM for reuse.

Let me use a tiled approach with SLM for B and register blocking for A.

## OCL Code

```OCL
// Optimized FP16 matmul using Intel DPAS (XMX) instructions for Battlemage
// Each subgroup computes 8x16 output tile via intel_sub_group_f16_f16_matrix_mad_k16
// Workgroup: 16 subgroups arranged as 4(row) x 4(col), each 8x16 => WG tile: 32x64
// GWS = (ceil_div(N,64)*16, ceil_div(M,32)*16)
// LWS = (16, 16)
// Subgroup size = 16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 16
#define SG_SIZE 16
#define SG_TILE_M 8
#define SG_TILE_N 16

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Workgroup tile origin
    const int wg_col = get_group_id(0) * TILE_N;  // in output columns
    const int wg_row = get_group_id(1) * TILE_M;  // in output rows

    // Subgroup ID within workgroup
    const int lid_x = get_local_id(0); // 0..15
    const int lid_y = get_local_id(1); // 0..15
    const int sg_linear = lid_y;       // subgroup index = local_id(1) since x is the subgroup lane

    // Map 16 subgroups into 4x4 grid
    const int sg_row = sg_linear / 4;  // 0..3
    const int sg_col = sg_linear % 4;  // 0..3

    // This subgroup's output tile origin
    const int out_row = wg_row + sg_row * SG_TILE_M;  // 8 rows
    const int out_col = wg_col + sg_col * SG_TILE_N;  // 16 cols

    // Accumulator: 8 floats per work-item (8 rows, column = lane id)
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // SLM for B tile: TILE_K x TILE_N = 16 x 64 halfs
    __local half B_slm[TILE_K * TILE_N];

    const int sg_lane = get_sub_group_local_id();

    // Loop over K dimension in blocks of TILE_K
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperatively load B tile [k0:k0+16, wg_col:wg_col+64] into SLM
        // Total elements: 16*64 = 1024 halfs, 256 work-items => 4 each
        {
            int linear_id = lid_y * 16 + lid_x; // 0..255
            for (int i = 0; i < 4; i++) {
                int idx = linear_id + i * 256;
                int bk = idx / TILE_N;  // row in B tile (0..15)
                int bn = idx % TILE_N;  // col in B tile (0..63)
                int global_k = k0 + bk;
                int global_n = wg_col + bn;
                half val = 0;
                if (global_k < K && global_n < N)
                    val = B[global_k * N + global_n];
                B_slm[bk * TILE_N + bn] = val;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Load A tile for this subgroup: 8 rows x 16 cols
        // For DPAS: A is 8x16 matrix, each WI holds 8 half values (one per row, broadcast across K)
        // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
        //   a: int8 (8 x short packed) per WI - 8 rows, k distributed across lanes
        //   b: int8 (8 x short packed) per WI - k distributed across lanes, 16 cols
        //   acc: float8 per WI

        // A matrix [8 x 16]: row-major, each WI reads one column (lane-th)
        // Packed as: each WI holds 8 halfs = rows 0..7 at column=lane
        // But DPAS expects specific layout. Let's use the sub_group block read.

        // For DPAS A input: 8 rows x 16 K, stored as 8 ushort per WI
        // WI[lane] holds A[row][lane] for row=0..7 => need A[out_row+r][k0+lane]
        short8 a_packed;
        {
            half a_vals[8];
            for (int r = 0; r < 8; r++) {
                int gr = out_row + r;
                int gc = k0 + sg_lane;
                if (gr < M && gc < K)
                    a_vals[r] = A[gr * K + gc];
                else
                    a_vals[r] = (half)0;
            }
            a_packed = (short8)(
                as_short(a_vals[0]), as_short(a_vals[1]),
                as_short(a_vals[2]), as_short(a_vals[3]),
                as_short(a_vals[4]), as_short(a_vals[5]),
                as_short(a_vals[6]), as_short(a_vals[7])
            );
        }

        // For DPAS B input: 16 K x 16 N, stored as 8 ushort per WI
        // B is 16x16 submatrix from B_slm at column offset sg_col*16
        // WI[lane] holds B[k][lane] for pairs of k => 8 x (2 halfs packed as int)
        // Actually layout: each WI holds 8 shorts = B[2*i][lane] and B[2*i+1][lane] interleaved
        short8 b_packed;
        {
            half b_vals[8];
            int b_col_base = sg_col * SG_TILE_N;
            // Each WI reads column sg_lane from B_slm, rows 0..15
            // Packed as pairs: (row0,row1), (row2,row3), ... BUT DPAS expects
            // 8 ushort per WI where each ushort is one row at the lane's column
            // For k16: 16 rows, so we need 16 halfs, packed as 8 ints (pairs)
            // Actually intel_sub_group_f16_f16_matrix_mad_k16 takes int8 for B
            int8 b_int;
            for (int p = 0; p < 8; p++) {
                int r0 = 2 * p;
                int r1 = 2 * p + 1;
                half v0 = B_slm[r0 * TILE_N + b_col_base + sg_lane];
                half v1 = B_slm[r1 * TILE_N + b_col_base + sg_lane];
                // Pack two halfs into one int
                ushort s0 = as_ushort(v0);
                ushort s1 = as_ushort(v1);
                b_int[p] = (int)s0 | ((int)s1 << 16);
            }

            // Call DPAS
            float8 acc_vec = (float8)(acc[0], acc[1], acc[2], acc[3],
                                       acc[4], acc[5], acc[6], acc[7]);
            acc_vec = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_int, acc_vec);
            acc[0] = acc_vec.s0; acc[1] = acc_vec.s1;
            acc[2] = acc_vec.s2; acc[3] = acc_vec.s3;
            acc[4] = acc_vec.s4; acc[5] = acc_vec.s5;
            acc[6] = acc_vec.s6; acc[7] = acc_vec.s7;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: each WI writes 8 values (8 rows at column out_col+sg_lane)
    for (int r = 0; r < 8; r++) {
        int gr = out_row + r;
        int gc = out_col + sg_lane;
        if (gr < M && gc < N) {
            C[gr * N + gc] = convert_half(acc[r]);
        }
    }
}
```

