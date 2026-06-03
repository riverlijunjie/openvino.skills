# Analysis

## Requirements

We need an efficient FP16 GEMM kernel for Intel Battlemage (Xe2-HPG) GPU. The key constraints from the instructions are:

1. **Architecture**: 64 work-items (4 subgroups of 16), A in SLM, B from global/L2
2. **Tile size**: 32×64×32 (TILE_M=32, TILE_N=64, TILE_K=32)
3. **Use DPAS**: `intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)`
4. **K=2048 divides evenly by 32**, so no remainder path needed

## Proposed Kernel Structure

**Work-group layout**: 64 WIs = 4 subgroups × 16 WIs each
- Each WG computes a 32×64 tile of C
- 4 subgroups split the N dimension: each SG handles 32×16 of C
- DPAS with repcount=8 produces 8 rows per call, so each SG does 4 DPAS calls per K-step (covering 32 rows)

**Memory strategy**:
- A tile (32×32 f16) loaded cooperatively into SLM with double-buffering
- B tile (32×64 f16) loaded directly from global memory into registers
- SLM padding to avoid bank conflicts (stride = 34 instead of 32)

**K-loop optimizations**:
- 2× unroll of K-loop (process K=64 per iteration, 32 iterations for K=2048)
- Double-buffering: load next A into SLM buffer while computing with current
- Prefetch B tiles ahead

**DPAS mapping on Xe2** (exec_size=16):
- `intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)` 
- a: 8 rows × 16 K elements (as short8 per WI in SG of 16)
- b: 16 K elements × 16 N columns (as int8 per WI)
- Result: 8 rows × 16 columns accumulated into float8

Each subgroup computes 32 rows × 16 cols = 4 DPAS calls (for M=32, stepping 8 rows each), repeated for 2 K-steps of 16 within each K=32 block.

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, f32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM, B from global
// Launch: GWS = (N/16, M/32, 1) * (16, 4, 1), LWS = (16, 4, 1)
// Subgroup size = 16
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
    // Work-group and subgroup IDs
    const int wg_n = get_group_id(0);  // which 64-col tile
    const int wg_m = get_group_id(1);  // which 32-row tile
    const int sg_id = get_sub_group_id();  // 0..3, each handles 16 columns
    const int sg_lid = get_sub_group_local_id();  // 0..15

    // Global tile offsets
    const int tile_row = wg_m * 32;
    const int tile_col = wg_n * 64 + sg_id * 16;

    // SLM for A: double-buffered, 32 rows x 32 cols with padding
    // Stride = 34 to avoid bank conflicts (34 halfs = 68 bytes per row)
    #define SLM_STRIDE 34
    __local half slm_A[2 * 32 * SLM_STRIDE];  // 2 buffers

    // Accumulators: 32 rows x 16 cols = 4 DPAS results of float8
    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);
    float8 acc2 = (float8)(0.0f);
    float8 acc3 = (float8)(0.0f);

    // Each WI in the WG (64 total) cooperatively loads A tile (32x32 = 1024 halfs)
    // 64 WIs, each loads 1024/64 = 16 halfs per K-step
    const int local_id = sg_id * 16 + sg_lid;  // 0..63

    // Preload first A tile into buffer 0
    {
        // Each WI loads 16 halfs. 1024 halfs / 64 WIs = 16 halfs per WI
        // Map: local_id covers 64 positions, each loads 16 consecutive halfs
        // Total elements = 32*32 = 1024, arranged as 32 rows x 32 cols
        // Each WI handles: row = (local_id * 16) / 32, starting col = (local_id * 16) % 32
        // But simpler: treat as linear, each WI loads elements [local_id*16 .. local_id*16+15]
        // Then store into SLM with proper stride
        int base_elem = local_id * 16;  // 0, 16, 32, ... 1008
        int row = base_elem / 32;       // 0..31
        int col = base_elem % 32;       // 0 or 16

        __global const half* a_ptr = A + (tile_row + row) * K + col;

        // Load 16 halfs from global (contiguous in K dimension for row-major A)
        half16 a_val = vload16(0, a_ptr);

        // Store to SLM buffer 0
        __local half* slm_dst = slm_A + row * SLM_STRIDE + col;
        vstore16(a_val, 0, slm_dst);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;

    // Main K-loop: K/32 iterations
    const int k_iters = K / 32;

    for (int ki = 0; ki < k_iters; ki++) {
        int k_offset = ki * 32;
        int next_k = k_offset + 32;

        // Start loading next A tile into other buffer (if not last iteration)
        // We'll do compute first, then load next (with barrier management)

        // Read A from current SLM buffer and B from global, do DPAS
        __local half* cur_slm = slm_A + buf * 32 * SLM_STRIDE;

        // For DPAS k16: we do 2 DPAS per accumulator (k=0..15, k=16..31)
        // Each DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
        // a (short8): 8 rows, each row has 16 f16 values packed across the subgroup
        //   - Each WI holds short8 = 8 pairs of f16 (but actually 8 x 1 short = 8 f16 values?)
        //   No: short8 per WI, 16 WIs in SG. Total = 8*16 shorts = 128 shorts = 128 f16 values
        //   That's 8 rows x 16 K elements.
        //   So each WI's short8 contains: for each of 8 rows, one pair of f16 at k-positions [lid*1..?]
        //   Actually: a is 8 rows x 16K. Distributed: each WI holds 8 shorts.
        //   short = 2 f16 packed? No, short = 16 bits = 1 f16.
        //   Wait - intel_sub_group_f16_f16_matrix_mad_k16 with short8 a:
        //   8 rows, k=16. Total a elements = 8*16 = 128 f16 = 128 shorts.
        //   Across 16 WIs: 128/16 = 8 shorts per WI = short8. 
        //   Layout: WI[i] holds a[row][i] for each row packed sequentially.
        //   So a.s0 = row0_k[lid], a.s1 = row1_k[lid], ... a.s7 = row7_k[lid]
        //   Wait no - for k16: each row has 16 k-elements distributed 1 per WI.
        //   So WI[lid] holds: s0=A[row0,k_base+lid], s1=A[row1,k_base+lid], ...

        // Actually for Xe2 DPAS with k16 and f16:
        // a: short8 per WI. 8 = repcount (rows). Each element = 1 f16 at k-position = sg_lid
        // b: int8 per WI. int = 2 f16 packed. 8 ints = 16 f16 values.
        //   b represents 16 K-elements for column = sg_lid (N-direction).
        //   int8 = 16 f16 values for k=0..15 at n=sg_lid

        // Load B: for k16 step, each WI needs int8 = 16 f16 values from B
        // B[k, n]: row-major, so B[k_base + k_idx, tile_col + sg_lid]
        // For 16 consecutive k values at fixed n: stride = N between elements

        // Let's load B for first k16 block
        __global const half* b_base0 = B + k_offset * N + tile_col + sg_lid;
        __global const half* b_base1 = B + (k_offset + 16) * N + tile_col + sg_lid;

        // Load B k=0..15 for this subgroup's 16 columns
        // Each WI loads 16 halfs with stride N
        short b_raw0[16], b_raw1[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            b_raw0[i] = as_short(b_base0[i * N]);
        }
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            b_raw1[i] = as_short(b_base1[i * N]);
        }

        // Pack into int8 for DPAS (pairs of f16 -> int)
        int8 b_val0, b_val1;
        b_val0.s0 = (int)((uint)(ushort)b_raw0[0] | ((uint)(ushort)b_raw0[1] << 16));
        b_val0.s1 = (int)((uint)(ushort)b_raw0[2] | ((uint)(ushort)b_raw0[3] << 16));
        b_val0.s2 = (int)((uint)(ushort)b_raw0[4] | ((uint)(ushort)b_raw0[5] << 16));
        b_val0.s3 = (int)((uint)(ushort)b_raw0[6] | ((uint)(ushort)b_raw0[7] << 16));
        b_val0.s4 = (int)((uint)(ushort)b_raw0[8] | ((uint)(ushort)b_raw0[9] << 16));
        b_val0.s5 = (int)((uint)(ushort)b_raw0[10] | ((uint)(ushort)b_raw0[11] << 16));
        b_val0.s6 = (int)((uint)(ushort)b_raw0[12] | ((uint)(ushort)b_raw0[13] << 16));
        b_val0.s7 = (int)((uint)(ushort)b_raw0[14] | ((uint)(ushort)b_raw0[15] << 16));

        b_val1.s0 = (int)((uint)(ushort)b_raw1[0] | ((uint)(ushort)b_raw1[1] << 16));
        b_val1.s1 = (int)((uint)(ushort)b_raw1[2] | ((uint)(ushort)b_raw1[3] << 16));
        b_val1.s2 = (int)((uint)(ushort)b_raw1[4] | ((uint)(ushort)b_raw1[5] << 16));
        b_val1.s3 = (int)((uint)(ushort)b_raw1[6] | ((uint)(ushort)b_raw1[7] << 16));
        b_val1.s4 = (int)((uint)(ushort)b_raw1[8] | ((uint)(ushort)b_raw1[9] << 16));
        b_val1.s5 = (int)((uint)(ushort)b_raw1[10] | ((uint)(ushort)b_raw1[11] << 16));
        b_val1.s6 = (int)((uint)(ushort)b_raw1[12] | ((uint)(ushort)b_raw1[13] << 16));
        b_val1.s7 = (int)((uint)(ushort)b_raw1[14] | ((uint)(ushort)b_raw1[15] << 16));

        // Load A from SLM for 4 groups of 8 rows, k=0..15
        // A in SLM: row r, col c -> slm[r * SLM_STRIDE + c]
        // For DPAS a (short8): WI[lid] needs A[row_base+r, k_base+lid] for r=0..7
        // k_base=0 for first k16, k_base=16 for second k16

        short8 a_val00, a_val01, a_val02, a_val03;
        short8 a_val10, a_val11, a_val12, a_val13;

        // First k16 block (k=0..15 within the 32-wide tile)
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val00)[r] = as_short(cur_slm[(0 + r) * SLM_STRIDE + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val01)[r] = as_short(cur_slm[(8 + r) * SLM_STRIDE + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val02)[r] = as_short(cur_slm[(16 + r) * SLM_STRIDE + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val03)[r] = as_short(cur_slm[(24 + r) * SLM_STRIDE + sg_lid]);
        }

        // Second k16 block (k=16..31)
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val10)[r] = as_short(cur_slm[(0 + r) * SLM_STRIDE + 16 + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val11)[r] = as_short(cur_slm[(8 + r) * SLM_STRIDE + 16 + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val12)[r] = as_short(cur_slm[(16 + r) * SLM_STRIDE + 16 + sg_lid]);
        }
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            ((short*)&a_val13)[r] = as_short(cur_slm[(24 + r) * SLM_STRIDE + 16 + sg_lid]);
        }

        // DPAS calls: 4 row-groups x 2 k-steps = 8 DPAS
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val00, b_val0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val01, b_val0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val02, b_val0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val03, b_val0, acc3);

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a_val10, b_val1, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a_val11, b_val1, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a_val12, b_val1, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a_val13, b_val1, acc3);

        // Load next A tile (if not last iteration)
        if (ki < k_iters - 1) {
            int next_buf = 1 - buf;
            int base_elem = local_id * 16;
            int row = base_elem / 32;
            int col = base_elem % 32;

            __global const half* a_ptr = A + (tile_row + row) * K + next_k + col;
            half16 a_load = vload16(0, a_ptr);

            __local half* slm_dst = slm_A + next_buf * 32 * SLM_STRIDE + row * SLM_STRIDE + col;
            vstore16(a_load, 0, slm_dst);

            buf = next_buf;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write C: 32 rows x 16 cols per subgroup
    // acc0 = rows 0..7, acc1 = rows 8..15, acc2 = rows 16..23, acc3 = rows 24..31
    // Each float8 element corresponds to one row, sg_lid selects column

    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = tile_row + r;
        int col_idx = tile_col + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc0)[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = tile_row + 8 + r;
        int col_idx = tile_col + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc1)[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = tile_row + 16 + r;
        int col_idx = tile_col + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc2)[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_idx = tile_row + 24 + r;
        int col_idx = tile_col + sg_lid;
        if (row_idx < M && col_idx < N) {
            C[row_idx * N + col_idx] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```