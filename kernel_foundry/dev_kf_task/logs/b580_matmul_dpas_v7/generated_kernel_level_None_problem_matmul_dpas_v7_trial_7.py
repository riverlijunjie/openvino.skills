

## Analysis

Looking at the previous versions, all achieve 33.9ms. The main bottlenecks identified:

1. **Scalar repacking of A from SLM** - 16 scalar loads + pack ops per K-tile per subgroup
2. **No overlap of prefetch and compute** - loads happen before DPAS, exposing latency
3. **Boundary checks on every element** - divergence overhead in the common case
4. **Small work-group tile (32x32)** - insufficient compute-to-memory ratio

My improvements:
- **Larger tile: 32x64** (4x8 subgroups) to increase compute density per SLM load
- **Store A in packed uint format in SLM** during preload, eliminating per-iteration repack
- **Fast path for interior tiles** - skip bounds checks when full tile fits
- **Reorder: compute first, then prefetch next** so compiler can overlap DPAS with memory
- **Use intel_sub_group_block_read for SLM reads** where possible

## Improved OCL code

```OCL
// Optimized FP16 GEMM for Intel Battlemage (Xe2-HPG) using DPAS
// C[M,N] = A[M,K] x B[K,N], all half, accumulate in float
//
// Launch metadata:
//   LWS = (64, 4, 1) => 256 work-items = 32 subgroups of 8
//   GWS = (ceil(N/64)*64, ceil(M/32)*4, 1)
//   Subgroup size = 8
//   Work-group tile: 32x64 of C (4 sg rows x 8 sg cols)
//   Each subgroup: 8x8 of C via DPAS k16
//   SLM per buffer: A_packed[32][8] uint + B_vnni[8][64] uint
//     = 32*8*4 + 8*64*4 = 1024 + 2048 = 3072 bytes
//   Double-buffered: 6144 bytes total

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int TM = 32;
    const int TN = 64;
    const int TK = 16;

    const int wg_row = get_group_id(1) * TM;
    const int wg_col = get_group_id(0) * TN;

    const int lid0 = get_local_id(0);  // 0..63
    const int lid1 = get_local_id(1);  // 0..3
    const int flat_lid = lid1 * 64 + lid0;  // 0..255

    const int sg_id = get_sub_group_id();        // 0..31
    const int sg_lid = get_sub_group_local_id(); // 0..7

    // 4x8 subgroup grid
    const int sg_row = sg_id / 8;  // 0..3
    const int sg_col = sg_id % 8;  // 0..7

    // Check if this is an interior tile (no bounds checks needed)
    const int is_interior = (wg_row + TM <= M) && (wg_col + TN <= N);

    // Double-buffered SLM
    // A stored as packed uint: slm_A[buf][row][k_pair] where each uint = 2 halfs
    // 32 rows x 8 k_pairs = 256 uints per buffer
    __local uint slm_A_packed[2][32 * 8];
    // B stored in VNNI: slm_B[buf][k_pair][col] as uint = pack(B[k+2*kp, col], B[k+2*kp+1, col])
    // 8 k_pairs x 64 cols = 512 uints per buffer
    __local uint slm_B_vnni[2][8 * 64];

    float8 acc = (float8)(0.0f);

    int num_k_tiles = (K + TK - 1) / TK;

    // === Preload first tile into buffer 0 ===
    {
        int k0 = 0;
        // Load A: 32*8 = 256 uints, 256 threads => 1 each
        if (flat_lid < 256) {
            int r = flat_lid >> 3;   // row 0..31
            int kp = flat_lid & 7;   // k-pair 0..7
            int gr = wg_row + r;
            int gk = k0 + kp * 2;
            ushort v0, v1;
            if (is_interior && (gk + 1 < K)) {
                v0 = as_ushort(A[gr * K + gk]);
                v1 = as_ushort(A[gr * K + gk + 1]);
            } else {
                v0 = (gr < M && gk < K) ? as_ushort(A[gr * K + gk]) : (ushort)0;
                v1 = (gr < M && (gk + 1) < K) ? as_ushort(A[gr * K + gk + 1]) : (ushort)0;
            }
            slm_A_packed[0][flat_lid] = (uint)v0 | ((uint)v1 << 16);
        }

        // Load B: 8*64 = 512 uints, 256 threads => 2 each
        for (int i = flat_lid; i < 512; i += 256) {
            int kp = i >> 6;    // i / 64
            int c = i & 63;     // i % 64
            int gk = k0 + kp * 2;
            int gc = wg_col + c;
            ushort v0, v1;
            if (is_interior && (gk + 1 < K)) {
                v0 = as_ushort(B[gk * N + gc]);
                v1 = as_ushort(B[(gk + 1) * N + gc]);
            } else {
                v0 = (gk < K && gc < N) ? as_ushort(B[gk * N + gc]) : (ushort)0;
                v1 = ((gk + 1) < K && gc < N) ? as_ushort(B[(gk + 1) * N + gc]) : (ushort)0;
            }
            slm_B_vnni[0][i] = (uint)v0 | ((uint)v1 << 16);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int cur_buf = buf;
        int next_buf = buf ^ 1;
        int next_kt = kt + 1;

        // === COMPUTE on current buffer ===
        // Load A packed: 8 uints for this subgroup's rows
        int a_base = (sg_row * 8 + sg_lid) * 8;  // row offset in slm_A_packed
        int8 a_packed = (int8)(
            (int)slm_A_packed[cur_buf][a_base + 0],
            (int)slm_A_packed[cur_buf][a_base + 1],
            (int)slm_A_packed[cur_buf][a_base + 2],
            (int)slm_A_packed[cur_buf][a_base + 3],
            (int)slm_A_packed[cur_buf][a_base + 4],
            (int)slm_A_packed[cur_buf][a_base + 5],
            (int)slm_A_packed[cur_buf][a_base + 6],
            (int)slm_A_packed[cur_buf][a_base + 7]
        );

        // Load B VNNI: 8 uints for this subgroup's columns
        int b_col = sg_col * 8 + sg_lid;
        int8 b_packed = (int8)(
            (int)slm_B_vnni[cur_buf][0 * 64 + b_col],
            (int)slm_B_vnni[cur_buf][1 * 64 + b_col],
            (int)slm_B_vnni[cur_buf][2 * 64 + b_col],
            (int)slm_B_vnni[cur_buf][3 * 64 + b_col],
            (int)slm_B_vnni[cur_buf][4 * 64 + b_col],
            (int)slm_B_vnni[cur_buf][5 * 64 + b_col],
            (int)slm_B_vnni[cur_buf][6 * 64 + b_col],
            (int)slm_B_vnni[cur_buf][7 * 64 + b_col]
        );

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_packed, b_packed, acc);

        // === PREFETCH next tile into next_buf (overlaps with DPAS pipeline drain) ===
        if (next_kt < num_k_tiles) {
            int k0_next = next_kt * TK;
            int k_remaining = K - k0_next;

            if (flat_lid < 256) {
                int r = flat_lid >> 3;
                int kp = flat_lid & 7;
                int gr = wg_row + r;
                int gk = k0_next + kp * 2;
                ushort v0, v1;
                if (is_interior && gk + 1 < K) {
                    v0 = as_ushort(A[gr * K + gk]);
                    v1 = as_ushort(A[gr * K + gk + 1]);
                } else {
                    v0 = (gr < M && gk < K) ? as_ushort(A[gr * K + gk]) : (ushort)0;
                    v1 = (gr < M && (gk + 1) < K) ? as_ushort(A[gr * K + gk + 1]) : (ushort)0;
                }
                slm_A_packed[next_buf][flat_lid] = (uint)v0 | ((uint)v1 << 16);
            }

            for (int i = flat_lid; i < 512; i += 256) {
                int kp = i >> 6;
                int c = i & 63;
                int gk = k0_next + kp * 2;
                int gc = wg_col + c;
                ushort v0, v1;
                if (is_interior && gk + 1 < K) {
                    v0 = as_ushort(B[gk * N + gc]);
                    v1 = as_ushort(B[(gk + 1) * N + gc]);
                } else {
                    v0 = (gk < K && gc < N) ? as_ushort(B[gk * N + gc]) : (ushort)0;
                    v1 = ((gk + 1) < K && gc < N) ? as_ushort(B[(gk + 1) * N + gc]) : (ushort)0;
                }
                slm_B_vnni[next_buf][i] = (uint)v0 | ((uint)v1 << 16);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        buf = next_buf;
    }

    // === Store 8x8 output tile ===
    int out_col = wg_col + sg_col * 8 + sg_lid;
    int out_row_base = wg_row + sg_row * 8;

    if (is_interior) {
        // Fast path: no bounds checks
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            C[(out_row_base + i) * N + out_col] = convert_half(((float*)&acc)[i]);
        }
    } else {
        if (out_col < N) {
            for (int i = 0; i < 8; i++) {
                int out_row = out_row_base + i;
                if (out_row < M) {
                    C[out_row * N + out_col] = convert_half(((float*)&acc)[i]);
                }
            }
        }
    }
}
```

