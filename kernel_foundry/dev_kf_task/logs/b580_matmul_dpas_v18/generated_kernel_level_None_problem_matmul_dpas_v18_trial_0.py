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
- SLM stride padded to avoid bank conflicts

**K-loop optimizations**:
- 2x unroll of K-loop (process K_STEP=32 twice per iteration = 64 per loop body)
- Double-buffering: load next A into SLM buffer while computing with current
- Prefetch B for next iteration

**DPAS mapping**:
- `intel_sub_group_f16_f16_matrix_mad_k16`: takes short8 A (8 rows × 16 cols packed as short8), int8 B (16 rows × 16 cols packed), produces float8 (8 results)
- For 32 M-rows × 16 N-cols per SG with K_STEP=32: 4 M-blocks × 2 K-blocks = 8 DPAS calls per SG per K-step

**Launch config**:
- LWS = (16, 4, 1) → 64 WIs, 4 SGs
- GWS = (N/4, M/32, 1) × LWS = (N/4, M/32 * 4, 1) ... actually GWS_x = (N/64)*16, GWS_y = (M/32)*4

Let me work through the details carefully:

- GWS = ((N/64)*16, (M/32)*4, 1)
- LWS = (16, 4, 1)

Each WG: get_group_id(0) selects N-tile (64 cols), get_group_id(1) selects M-tile (32 rows).
Within WG: get_local_id(1) identifies subgroup (0-3), each handles 16 consecutive columns.

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half, f32 accumulation
// Launch: GWS = ((N/64)*16, (M/32)*4, 1), LWS = (16, 4, 1)
// 4 subgroups of 16 WIs per WG. Each WG computes 32x64 tile.
// Each subgroup computes 32x16 using DPAS.
// A in SLM (double-buffered), B from global/L2.
// TILE_M=32, TILE_N=64, TILE_K=32. K must be divisible by 32.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

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
    // Work-group and subgroup identification
    const int wg_n = get_group_id(0);  // which 64-col tile
    const int wg_m = get_group_id(1);  // which 32-row tile
    const int sg_id = get_local_id(1); // subgroup index 0-3
    const int sg_lane = get_local_id(0); // lane within subgroup 0-15

    const int tile_row = wg_m * 32;
    const int tile_col = wg_n * 64 + sg_id * 16;

    // Linear local ID for cooperative loads
    const int lid = sg_id * 16 + sg_lane; // 0..63

    // SLM for A: double-buffered, 32 rows x 32 cols of half
    // Pad stride to 34 to avoid bank conflicts (34 halfs = 68 bytes per row)
    #define SLM_STRIDE 34
    #define SLM_BUF_SIZE (32 * SLM_STRIDE)  // per buffer in halfs
    __local half slm_a[2 * SLM_BUF_SIZE];

    // Accumulators: 32 rows x 16 cols per subgroup = 4 blocks of 8 rows
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // Preload first A tile into SLM buffer 0
    // A tile: 32 rows x 32 cols. 64 WIs load 32*32/64 = 16 halfs each
    // Each WI loads one half16 vector (16 consecutive elements)
    // Map: lid covers 64 positions, we need 32*32=1024 halfs = 64*16
    {
        // Each WI loads 16 halfs. lid determines which row/col chunk.
        // 32 cols / 16 = 2 chunks per row, so 64 WIs cover 32 rows * 2 chunks
        int chunk_row = lid / 2;
        int chunk_col = (lid % 2) * 16;
        int a_row = tile_row + chunk_row;
        int a_col = chunk_col; // k=0

        __global const half* a_ptr = A + a_row * K + a_col;
        half16 a_val = vload16(0, a_ptr);

        int slm_offset = chunk_row * SLM_STRIDE + chunk_col;
        vstore16(a_val, 0, &slm_a[slm_offset]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    // Main K-loop, step by 32
    for (int k = 0; k < K; k += 32) {
        int next_k = k + 32;
        int next_buf = 1 - cur_buf;

        // Load next A tile into next SLM buffer (if not last iteration)
        if (next_k < K) {
            int chunk_row = lid / 2;
            int chunk_col = (lid % 2) * 16;
            int a_row = tile_row + chunk_row;
            int a_col = next_k + chunk_col;

            __global const half* a_ptr = A + a_row * K + a_col;
            half16 a_val = vload16(0, a_ptr);

            int slm_offset = next_buf * SLM_BUF_SIZE + chunk_row * SLM_STRIDE + chunk_col;
            vstore16(a_val, 0, &slm_a[slm_offset]);
        }

        // Compute with current buffer
        // Load B from global: 32 rows x 16 cols for this subgroup
        // DPAS needs B in packed format: int8 = 16 halfs packed as 8 ints (16 rows x 16 cols, k16)
        // B layout: row-major [K, N], so B[k_offset, tile_col] 
        // For DPAS k16: we process k in two steps of 16

        __local half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

        // Process k_inner = 0..15 (first k16 block)
        {
            // Load A from SLM for DPAS: need short8 per 8-row block
            // short8 a means 8 rows x 16 cols packed: each short = 1 half
            // Actually for DPAS k16: short8 a = 8 rows, each with k16 packed as short
            // intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)
            // a: short8 - each lane holds 8 consecutive k-values for its row? No.
            // a: short8 across subgroup: 16 lanes × short8 = 16×8 = 128 shorts = 8 rows × 16 k-values
            //    Lane i holds rows' k-values at column i? 
            // Actually: a is 8×16 matrix (8 rows, 16 k-elements), distributed as:
            //    Each WI holds short8 = 8 values = one column of the 8×16 tile
            //    WI lane_i holds a[row][lane_i] for row=0..7? No, it's the k-dimension.
            // For DPAS k16: a[m,k] where m=8 (repcount), k=16
            //    Distributed: lane_i holds k=i for all 8 m-rows → short8 per lane
            // So lane_i of short8 a = {a[0,i], a[1,i], ..., a[7,i]} but packed as halfs in short

            // B: int8 = 16×16 matrix (k=16, n=16 per subgroup)
            //    Distributed: lane_i holds column i, int8 = 8 ints = 16 halfs = k=0..15
            // So lane_i of int8 b = {b[0,i]b[1,i], b[2,i]b[3,i], ...} pairs packed as int

            // Load A block 0 (rows 0-7, k 0-15) from SLM
            short8 a0, a1, a2, a3;
            // For a0: rows 0-7, k=0..15. Lane sg_lane reads column sg_lane.
            // a0.s0 = slm[row0 * STRIDE + sg_lane] as short (half)
            #define LOAD_A_ROW(block, row_offset) \
                block.s0 = as_short(cur_slm[(row_offset + 0) * SLM_STRIDE + sg_lane]); \
                block.s1 = as_short(cur_slm[(row_offset + 1) * SLM_STRIDE + sg_lane]); \
                block.s2 = as_short(cur_slm[(row_offset + 2) * SLM_STRIDE + sg_lane]); \
                block.s3 = as_short(cur_slm[(row_offset + 3) * SLM_STRIDE + sg_lane]); \
                block.s4 = as_short(cur_slm[(row_offset + 4) * SLM_STRIDE + sg_lane]); \
                block.s5 = as_short(cur_slm[(row_offset + 5) * SLM_STRIDE + sg_lane]); \
                block.s6 = as_short(cur_slm[(row_offset + 6) * SLM_STRIDE + sg_lane]); \
                block.s7 = as_short(cur_slm[(row_offset + 7) * SLM_STRIDE + sg_lane]);

            LOAD_A_ROW(a0, 0)
            LOAD_A_ROW(a1, 8)
            LOAD_A_ROW(a2, 16)
            LOAD_A_ROW(a3, 24)

            // Load B block (k 0-15, 16 cols for this SG)
            // B[k + k_inner, tile_col + lane] - lane handles column
            // int8 b: lane_i holds 16 halfs (k=0..15) packed as 8 ints
            // Each int = 2 consecutive halfs: b[2j, lane], b[2j+1, lane]
            int8 b0;
            __global const half* b_ptr = B + (k + 0) * N + tile_col + sg_lane;
            // Load 16 rows of B, each row contributes one half per lane
            half b_vals[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                b_vals[i] = b_ptr[i * N];
            }
            // Pack pairs into int
            b0.s0 = as_int((short2)(as_short(b_vals[0]), as_short(b_vals[1])));
            b0.s1 = as_int((short2)(as_short(b_vals[2]), as_short(b_vals[3])));
            b0.s2 = as_int((short2)(as_short(b_vals[4]), as_short(b_vals[5])));
            b0.s3 = as_int((short2)(as_short(b_vals[6]), as_short(b_vals[7])));
            b0.s4 = as_int((short2)(as_short(b_vals[8]), as_short(b_vals[9])));
            b0.s5 = as_int((short2)(as_short(b_vals[10]), as_short(b_vals[11])));
            b0.s6 = as_int((short2)(as_short(b_vals[12]), as_short(b_vals[13])));
            b0.s7 = as_int((short2)(as_short(b_vals[14]), as_short(b_vals[15])));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc3);
        }

        // Process k_inner = 16..31 (second k16 block)
        {
            short8 a0, a1, a2, a3;
            // Load from SLM columns 16..31
            #define LOAD_A_ROW2(block, row_offset) \
                block.s0 = as_short(cur_slm[(row_offset + 0) * SLM_STRIDE + 16 + sg_lane]); \
                block.s1 = as_short(cur_slm[(row_offset + 1) * SLM_STRIDE + 16 + sg_lane]); \
                block.s2 = as_short(cur_slm[(row_offset + 2) * SLM_STRIDE + 16 + sg_lane]); \
                block.s3 = as_short(cur_slm[(row_offset + 3) * SLM_STRIDE + 16 + sg_lane]); \
                block.s4 = as_short(cur_slm[(row_offset + 4) * SLM_STRIDE + 16 + sg_lane]); \
                block.s5 = as_short(cur_slm[(row_offset + 5) * SLM_STRIDE + 16 + sg_lane]); \
                block.s6 = as_short(cur_slm[(row_offset + 6) * SLM_STRIDE + 16 + sg_lane]); \
                block.s7 = as_short(cur_slm[(row_offset + 7) * SLM_STRIDE + 16 + sg_lane]);

            LOAD_A_ROW2(a0, 0)
            LOAD_A_ROW2(a1, 8)
            LOAD_A_ROW2(a2, 16)
            LOAD_A_ROW2(a3, 24)

            int8 b1;
            __global const half* b_ptr = B + (k + 16) * N + tile_col + sg_lane;
            half b_vals[16];
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                b_vals[i] = b_ptr[i * N];
            }
            b1.s0 = as_int((short2)(as_short(b_vals[0]), as_short(b_vals[1])));
            b1.s1 = as_int((short2)(as_short(b_vals[2]), as_short(b_vals[3])));
            b1.s2 = as_int((short2)(as_short(b_vals[4]), as_short(b_vals[5])));
            b1.s3 = as_int((short2)(as_short(b_vals[6]), as_short(b_vals[7])));
            b1.s4 = as_int((short2)(as_short(b_vals[8]), as_short(b_vals[9])));
            b1.s5 = as_int((short2)(as_short(b_vals[10]), as_short(b_vals[11])));
            b1.s6 = as_int((short2)(as_short(b_vals[12]), as_short(b_vals[13])));
            b1.s7 = as_int((short2)(as_short(b_vals[14]), as_short(b_vals[15])));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc3);
        }

        // Wait for next A tile load to complete
        if (next_k < K) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        cur_buf = next_buf;
    }

    // Store results: each subgroup writes 32 rows x 16 cols
    // acc0 = rows 0-7, acc1 = rows 8-15, acc2 = rows 16-23, acc3 = rows 24-31
    // Each float8 has 8 values for 8 rows, lane sg_lane handles column tile_col+sg_lane

    #define STORE_ROW(acc_block, row_base, idx) \
    { \
        int r = tile_row + row_base + idx; \
        int c = tile_col + sg_lane; \
        if (r < M && c < N) \
            C[r * N + c] = convert_half(acc_block[idx]); \
    }

    #define STORE_BLOCK(acc_block, row_base) \
        STORE_ROW(acc_block, row_base, 0) \
        STORE_ROW(acc_block, row_base, 1) \
        STORE_ROW(acc_block, row_base, 2) \
        STORE_ROW(acc_block, row_base, 3) \
        STORE_ROW(acc_block, row_base, 4) \
        STORE_ROW(acc_block, row_base, 5) \
        STORE_ROW(acc_block, row_base, 6) \
        STORE_ROW(acc_block, row_base, 7)

    // Use array access for float8
    float* p0 = (float*)&acc0;
    float* p1 = (float*)&acc1;
    float* p2 = (float*)&acc2;
    float* p3 = (float*)&acc3;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = tile_row + i;
        C[r * N + tile_col + sg_lane] = convert_half(p0[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = tile_row + 8 + i;
        C[r * N + tile_col + sg_lane] = convert_half(p1[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = tile_row + 16 + i;
        C[r * N + tile_col + sg_lane] = convert_half(p2[i]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int r = tile_row + 24 + i;
        C[r * N + tile_col + sg_lane] = convert_half(p3[i]);
    }
}
```

Wait, let me reconsider the approach. The float8 indexing with pointer cast might not work well in OpenCL. Let me also reconsider the DPAS operand layout more carefully.

For `intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)`:
- **a** (short8): Represents an 8×16 matrix (M=8 rows, K=16). Across 16 lanes, each lane holds 8 shorts. Lane `i` holds `a[row][i]` for row=0..7. So `a.s0` in lane `i` = element at row 0, k-col `i`.
- **b** (int8): Represents a 16×16 matrix (K=16, N=16). Each lane `j` holds column `j`. int8 = 8 ints = 16 halfs packed as pairs. `b.s0` in lane `j` = pack of `b[0,j]` and `b[1,j]`.
- **result** (float8): 8×16 output. Lane `j` holds 8 results for column `j`, rows 0..7.

Wait, that doesn't match. Let me reconsider. The result is float8 per lane, and there are 16 lanes, so we get 8×16 = 128 values. That's the C tile: 8 rows × 16 columns. Lane `j` holds the 8 row values for column `j`.

So for A loading from SLM: A is stored row-major in SLM (32 rows × 32 cols). For the first k16 block (cols 0-15), lane `sg_lane` needs `A[row][sg_lane]` for rows in the block. That matches my LOAD_A_ROW macro.

For B: B is row-major [K, N]. For k16 block starting at row `k`, column `tile_col`, lane `sg_lane` needs B[k_row, tile_col + sg_lane] for k_row = 0..15, packed in pairs as ints.

This looks correct. Let me clean up and finalize:

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half, f32 accumulation
// Architecture: 64 WIs (4 SGs of 16), A in SLM (double-buffered), B from global/L2
// Tile: 32x64x32 (TILE_M=32, TILE_N=64, TILE_K=32)
// Launch: LWS = (16, 4, 1), GWS = ((N/64)*16, (M/32)*4, 1)
// Each WG computes 32x64 output tile. Each SG computes 32x16.
// K must be divisible by 32.

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

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
    const int wg_n = get_group_id(0);
    const int wg_m = get_group_id(1);
    const int sg_id = get_local_id(1);
    const int sg_lane = get_local_id(0);
    const int lid = sg_id * 16 + sg_lane;

    const int tile_row = wg_m * 32;
    const int tile_col = wg_n * 64 + sg_id * 16;

    // SLM double buffer for A: 32 rows x 32 cols, stride=34 to avoid bank conflicts
    #define SLM_STRIDE 34
    #define SLM_BUF_SIZE (32 * SLM_STRIDE)
    __local half slm_a[2 * SLM_BUF_SIZE];

    // Accumulators
    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Cooperative A load helper: 64 WIs load 32x32 = 1024 halfs = 16 per WI
    int chunk_row = lid / 2;
    int chunk_col = (lid & 1) * 16;

    // Preload first A tile into buffer 0
    {
        __global const half* a_ptr = A + (tile_row + chunk_row) * K + chunk_col;
        half16 a_val = vload16(0, a_ptr);
        vstore16(a_val, 0, &slm_a[chunk_row * SLM_STRIDE + chunk_col]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    for (int k = 0; k < K; k += 32) {
        int next_k = k + 32;
        int next_buf = 1 - cur_buf;
        __local half* cur_slm = &slm_a[cur_buf * SLM_BUF_SIZE];

        // Start loading next A tile (overlapped with compute)
        half16 next_a_val;
        int do_next = (next_k < K);
        if (do_next) {
            __global const half* a_ptr = A + (tile_row + chunk_row) * K + next_k + chunk_col;
            next_a_val = vload16(0, a_ptr);
        }

        // === K-inner block 0: k..k+15 ===
        short8 a0_0, a1_0, a2_0, a3_0;
        
        // Load A from SLM for rows 0-7, k-cols 0-15
        a0_0 = (short8)(
            as_short(cur_slm[0 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[1 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[2 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[3 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[4 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[5 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[6 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[7 * SLM_STRIDE + sg_lane]));
        
        a1_0 = (short8)(
            as_short(cur_slm[8 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[9 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[10 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[11 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[12 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[13 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[14 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[15 * SLM_STRIDE + sg_lane]));

        a2_0 = (short8)(
            as_short(cur_slm[16 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[17 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[18 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[19 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[20 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[21 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[22 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[23 * SLM_STRIDE + sg_lane]));

        a3_0 = (short8)(
            as_short(cur_slm[24 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[25 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[26 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[27 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[28 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[29 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[30 * SLM_STRIDE + sg_lane]),
            as_short(cur_slm[31 * SLM_STRIDE + sg_lane]));

        // Load B for k..k+15
        int8 b0;
        {
            __global const half* b_ptr = B + k * N + tile_col + sg_lane;
            short bv0  = as_short(b_ptr[0 * N]);
            short bv1  = as_short(b_ptr[1 * N]);
            short bv2  = as_short(b_ptr[2 * N]);
            short bv3  = as_short(b_ptr[3 * N]);
            short bv4  = as_short(b_ptr[4 * N]);
            short bv5  = as_short(b_ptr[5 * N]);
            short bv6  = as_short(b_ptr[6 * N]);
            short bv7  = as_short(b_ptr[7 * N]);
            short bv8  = as_short(b_ptr[8 * N]);
            short bv9  = as_short(b_ptr[9 * N]);
            short bv10 = as_short(b_ptr[10 * N]);
            short bv11 = as_short(b_ptr[11 * N]);
            short bv12 = as_short(b_ptr[12 * N]);
            short bv13 = as_short(b_ptr[13 * N]);
            short bv14 = as_short(b_ptr[14 * N]);
            short bv15 = as_short(b_ptr[15 * N]);
            b0.s0 = as_int((short2)(bv0, bv1));
            b0.s1 = as_int((short2)(bv2, bv3));
            b0.s2 = as_int((short2)(bv4, bv5));
            b0.s3 = as_int((short2)(bv6, bv7));
            b0.s4 = as_int((short2)(bv8, bv9));
            b0.s5 = as_int((short2)(bv10, bv11));
            b0.s6 = as_int((short2)(bv12, bv13));
            b0.s7 = as_int((short2)(bv14, bv15));
        }

        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0_0, b0, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1_0, b0, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2_0, b0, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3_0, b0, acc3);

        // === K-inner block 1: k+16..k+31 ===
        short8 a0_1, a1_1, a2_1, a3_1;
        
        a0_1 = (short8)(
            as_short(cur_slm[0 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[1 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[2 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[3 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[4 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[5 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[6 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[7 * SLM_STRIDE + 16 + sg_lane]));

        a1_1 = (short8)(
            as_short(cur_slm[8 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[9 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[10 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[11 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[12 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[13 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[14 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[15 * SLM_STRIDE + 16 + sg_lane]));

        a2_1 = (short8)(
            as_short(cur_slm[16 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[17 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[18 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[19 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[20 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[21 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[22 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[23 * SLM_STRIDE + 16 + sg_lane]));

        a3_1 = (short8)(
            as_short(cur_slm[24 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[25 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[26 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[27 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[28 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[29 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[30 * SLM_STRIDE + 16 + sg_lane]),
            as_short(cur_slm[31 * SLM_STRIDE + 16 + sg_lane]));

        // Load B for k+16..k+31
        int8 b1;
        {
            __global const half* b_ptr = B + (k + 16) * N + tile_col + sg_lane;
            short bv0  = as_short(b_ptr[0 * N]);
            short bv1  = as_short(b_ptr[1 * N]);
            short bv2  = as_short(b_ptr[2 * N]);
            short bv3  = as_short(b_ptr[3 * N]);
            short bv4  = as_short(b_ptr[4 * N]);
            short bv5  = as_short(b_ptr[5 * N]);
            short bv6  = as_short(b_ptr[6 