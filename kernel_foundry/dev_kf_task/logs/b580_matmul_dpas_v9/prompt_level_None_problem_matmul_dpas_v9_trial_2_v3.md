

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Performance optimization techniques to consider:
- Must use Intel OpenCL DPAS instruction(XMX), e.g. intel_sub_group_f16_f16_matrix_mad_k16.
    CRITICAL: For intel_sub_group_f16_f16_matrix_mad_k16, you MUST use:
        - First operand: short8 (NOT float8)
        - Second operand: int8 (NOT float8)
        - Accumulator: float8 (this one is float8)
    Example:
        short8 a_val = as_short8(intel_sub_group_block_read_us8(...));
        int8 b_val = as_int8(intel_sub_group_block_read_ui8(...));
        float8 acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    Using float8 for the first two operands will compile but will NOT use
    XMX hardware acceleration.
- Improve FLOPS and hide memory latency with tiling + subgroup-friendly mapping.
- Double-buffering: Overlap SLM loads with DPAS computation
- Prefetching: Use async_work_group_copy for next K-tile
- Register blocking: Maximize DPAS utilization per register load
- Loop unrolling: Fully unroll K-loop if TILE_K <= 64
- SLM bank conflict avoidance: Pad SLM arrays by +1 column
- Provide explicit launch metadata (GWS/LWS/subgroup hints) in kernel comments.

## Reference code / Task:

This is the reference OCL implementation:
```
// Simple row-major FP16 matmul:
//   C[M,N] = A[M,K] x B[K,N]
// Input/Output dtype: half
// Accumulation dtype: float
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += convert_float(A[row * K + k]) * convert_float(B[k * N + col]);
    }

    C[row * N + col] = convert_half(acc);
}

```

## Previous OCL implementations with scores:

### Version 1 (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Workgroup tile: TILE_M=32, TILE_N=32, TILE_K=16
// LWS: (16, 2) -> 2 subgroups of 16
// Each subgroup: 32 rows x 16 cols = 4 DPAS (8x16) outputs
// GWS: (ceil(N/32)*16, ceil(M/32)*2)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define SG_SIZE 16

__attribute__((intel_reqd_sub_group_size(SG_SIZE)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Workgroup indices
    const int wg_n = get_group_id(0); // which 32-col tile
    const int wg_m = get_group_id(1); // which 32-row tile

    const int sg_id = get_sub_group_id();        // 0 or 1
    const int sg_lid = get_sub_group_local_id(); // 0..15

    const int base_m = wg_m * TILE_M;
    const int base_n = wg_n * TILE_N + sg_id * 16; // each subgroup handles 16 cols

    // SLM for tiles - padded to avoid bank conflicts
    // A tile: 32 x 16 halfs, padded to 32 x 17
    // B tile: 16 x 32 halfs, padded to 16 x 33
    __local half A_slm[TILE_M * (TILE_K + 1)];
    __local half B_slm[TILE_K * (TILE_N + 1)];

    // 4 accumulators for 4 blocks of 8 rows each
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    const int lid = get_local_id(1) * 16 + get_local_id(0); // flat local id, 0..31

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // === Load A tile [TILE_M x TILE_K] into SLM ===
        // 32 work items load 32x16 = 512 halfs -> 16 halfs per work item
        for (int i = 0; i < 16; i++) {
            int row = lid;       // which row (0..31)
            int col = i;         // which col (0..15)
            int src_row = base_m + row;
            int src_col = k0 + col;
            half val = (src_row < M && src_col < K) ? A[src_row * K + src_col] : (half)0.0h;
            A_slm[row * (TILE_K + 1) + col] = val;
        }

        // === Load B tile [TILE_K x TILE_N] into SLM ===
        // 32 work items load 16x32 = 512 halfs -> 16 halfs per work item
        for (int i = 0; i < 16; i++) {
            int idx = lid * 16 + i; // flat index 0..511
            int row = idx / TILE_N; // 0..15
            int col = idx % TILE_N; // 0..31
            int src_row = k0 + row;
            int src_col = base_n - sg_id * 16 + col; // base_n without sg offset for full tile
            // Wait, B tile should cover full 32 cols for the workgroup
            src_col = wg_n * TILE_N + col;
            half val = (src_row < K && src_col < N) ? B[src_row * N + src_col] : (half)0.0h;
            B_slm[row * (TILE_N + 1) + col] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === DPAS computation ===
        // Each subgroup reads A from SLM and B from SLM
        // For each of 4 row-blocks (8 rows each):
        //   Read A[8x16] as short8 per work item (each work item reads one row's 16 halfs = short8)
        //   Read B[16x16] as int8 per work item
        //   DPAS: acc = mad(a, b, acc)

        // Load B sub-tile for this subgroup: B_slm rows 0..15, cols sg_id*16..sg_id*16+15
        // B for DPAS needs to be in specific format: 16x16 packed as int8 per work item
        // intel_sub_group_block_read reads 16 consecutive values across subgroup
        // For B[16x16]: each of 16 rows, 16 cols. We need column-major packing for DPAS.
        // Actually for DPAS B matrix: it's read as int8 which is 8x(2 halfs) = 8 pairs of halfs = 16 halfs per lane
        // The B matrix is K=16 rows by N=16 cols, stored in a specific VNNI format

        // Let me use direct SLM block reads instead
        // For B: need to read 16x16 sub-tile. DPAS expects B in VNNI format (pairs of K packed with N)
        // B_vnni[k/2][n][2] - pairs of k values for each n

        // Actually, let me just read from SLM using subgroup block reads properly
        // For the A matrix (short8): each work-item in the subgroup holds 8 consecutive rows' worth of data
        //   for a single K=16 step. short8 = 8 halfs = 8 values, but we need 16 K values per row...
        //   Actually short8 = 16 bytes = 8 shorts = 8 halfs... but DPAS k16 needs 16 halfs per row
        //   Wait: short8 for A means 8 shorts = 8 halfs. With K=16, that's only half.
        //   But the DPAS k16 instruction: the 'a' operand short8 encodes 8 rows × 2 halfs,
        //   and the subgroup of 16 items provides 16 × 2 = 32... no.

        // Let me reconsider the DPAS semantics:
        // intel_sub_group_f16_f16_matrix_mad_k16:
        //   a: short8 per work-item -> across 16 work items = 8 * 16 shorts = 128 halfs = 8 rows × 16 K
        //   b: int8 per work-item -> across 16 work items = 8 * 16 ints = 8*16*2 halfs = 16 K × 16 N
        //   result: float8 per work-item -> 8 rows × 16 cols (each WI holds 8 rows, 1 col)

        // So for A: each WI holds short8 = 8 shorts. WI i holds A[row][k], where:
        //   row = 0..7 (mapped to short index 0..7)
        //   The 16 halfs of K are spread across the 16 work items... no, short8 = 8 values per WI,
        //   16 WIs × 8 = 128 values = 8 rows × 16 K. So WI j holds K values [j*1...], hmm.

        // The standard layout for A (row-major, 8×16 halfs):
        //   Use intel_sub_group_block_read_us8 from a pointer to row-major 8×16 half matrix
        //   with stride = 16 halfs between rows. Each WI gets short8 where element i = A[i][sg_lid].
        //   Wait no, for k16: each WI needs A[row][k_pair], and the subgroup collectively holds all K=16.
        //   Actually with short8: 8 values of 16 bits each. For k16, we need 16 half values per row.
        //   short8 = 8 shorts = 128 bits. For 8 rows × 16K: total = 8 × 16 × 16 bits = 2048 bits.
        //   Across 16 WIs: 16 × 128 = 2048 bits. ✓
        //   So each WI holds 8 values, each corresponding to a row, and within each row, 
        //   the K dimension is split across WIs. But 16 K values / 16 WIs = 1 K per WI... 
        //   but we have short (16-bit) values, and there are 8 of them per WI for 8 rows.
        //   So WI j holds: A[0][j], A[1][j], ..., A[7][j] - NO, that's only K stride of 1.
        //   But we need K=16. 
        //   
        //   Actually for k16: short8 means each short holds 1 half, so 8 halfs per WI.
        //   8 halfs × 16 WIs = 128 halfs. 8 rows × 16 K = 128. 
        //   Layout: WI j holds A[r][j] for r=0..7. But that gives K=16 (j=0..15). 
        //   So it IS a simple row-major block read where each WI reads column j of all 8 rows.

        // For B (VNNI format for k16): int8 per WI. int = 32 bits = 2 halfs.
        //   int8 = 8 ints = 8 × 2 halfs = 16 halfs per WI. 16 WIs × 16 = 256 halfs = 16K × 16N. ✓
        //   VNNI layout: B_vnni[k/2][n][2] where n is the column.
        //   WI j holds: for i=0..7: B[2i][j] and B[2i+1][j] packed as int.
        //   So B needs to be in VNNI format: pairs of K rows interleaved with N.

        // For SLM, I need to either:
        // 1. Store B in VNNI format in SLM, or
        // 2. Store B row-major and convert when reading

        // Let me store both A and B in formats amenable to block reads.

        // This is getting complex. Let me restructure.

        // APPROACH: Store A row-major in SLM with stride (TILE_K+padding).
        //           Store B in VNNI format in SLM.
        //           Use intel_sub_group_block_read for both.

        // For now, let me just manually construct the DPAS operands from SLM.

        // --- Construct A operand for each row block ---
        // Row block rb (0..3): rows rb*8 .. rb*8+7
        // A[row][k] from A_slm[row * (TILE_K+1) + k]
        // For DPAS: WI sg_lid reads A[rb*8 + r][sg_lid] for r=0..7

        short8 a0, a1, a2, a3;

        // Block 0: rows 0-7
        a0 = (short8)(
            as_short(A_slm[0 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[1 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[2 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[3 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[4 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[5 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[6 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[7 * (TILE_K+1) + sg_lid])
        );

        // Block 1: rows 8-15
        a1 = (short8)(
            as_short(A_slm[8 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[9 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[10 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[11 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[12 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[13 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[14 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[15 * (TILE_K+1) + sg_lid])
        );

        // Block 2: rows 16-23
        a2 = (short8)(
            as_short(A_slm[16 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[17 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[18 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[19 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[20 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[21 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[22 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[23 * (TILE_K+1) + sg_lid])
        );

        // Block 3: rows 24-31
        a3 = (short8)(
            as_short(A_slm[24 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[25 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[26 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[27 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[28 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[29 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[30 * (TILE_K+1) + sg_lid]),
            as_short(A_slm[31 * (TILE_K+1) + sg_lid])
        );

        // --- Construct B operand ---
        // B[k][n] from B_slm[k * (TILE_N+1) + sg_id*16 + sg_lid]
        // For DPAS VNNI: int8 where each int packs 2 halfs from consecutive K rows
        // WI sg_lid reads column (sg_id*16 + sg_lid), pairs of K
        int n_col = sg_id * 16 + sg_lid;
        int8 b_val;
        b_val.s0 = as_int((half2)(B_slm[0 * (TILE_N+1) + n_col], B_slm[1 * (TILE_N+1) + n_col]));
        b_val.s1 = as_int((half2)(B_slm[2 * (TILE_N+1) + n_col], B_slm[3 * (TILE_N+1) + n_col]));
        b_val.s2 = as_int((half2)(B_slm[4 * (TILE_N+1) + n_col], B_slm[5 * (TILE_N+1) + n_col]));
        b_val.s3 = as_int((half2)(B_slm[6 * (TILE_N+1) + n_col], B_slm[7 * (TILE_N+1) + n_col]));
        b_val.s4 = as_int((half2)(B_slm[8 * (TILE_N+1) + n_col], B_slm[9 * (TILE_N+1) + n_col]));
        b_val.s5 = as_int((half2)(B_slm[10 * (TILE_N+1) + n_col], B_slm[11 * (TILE_N+1) + n_col]));
        b_val.s6 = as_int((half2)(B_slm[12 * (TILE_N+1) + n_col], B_slm[13 * (TILE_N+1) + n_col]));
        b_val.s7 = as_int((half2)(B_slm[14 * (TILE_N+1) + n_col], B_slm[15 * (TILE_N+1) + n_col]));

        // DPAS
        acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_val, acc0);
        acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_val, acc1);
        acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_val, acc2);
        acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_val, acc3);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
    // Each WI holds 4×8 = 32 rows, 1 column (base_n + sg_id*16 + sg_lid)
    int out_col = base_n + sg_lid;
    // Wait, base_n already includes sg_id*16
    // base_n = wg_n * TILE_N + sg_id * 16
    // out_col = base_n + sg_lid = wg_n*32 + sg_id*16 + sg_lid

    if (out_col < N) {
        for (int r = 0; r < 8; r++) {
            int out_row = base_m + r;
            if (out_row < M)
                C[out_row * N + out_col] = convert_half(((float*)&acc0)[r]);
        }
        for (int r = 0; r < 8; r++) {
            int out_row = base_m + 8 + r;
            if (out_row < M)
                C[out_row * N + out_col] = convert_half(((float*)&acc1)[r]);
        }
        for (int r = 0; r < 8; r++) {
            int out_row = base_m + 16 + r;
            if (out_row < M)
                C[out_row * N + out_col] = convert_half(((float*)&acc2)[r]);
        }
        for (int r = 0; r < 8; r++) {
            int out_row = base_m + 24 + r;
            if (out_row < M)
                C[out_row * N + out_col] = convert_half(((float*)&acc3)[r]);
        }
    }
}
```

### Version 2 (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 53.400):
```OCL
// Optimized FP16 GEMM using Intel XMX DPAS instructions
// C[M,N] = A[M,K] x B[K,N], all half precision, float accumulation
//
// Tiling: TILE_M=32, TILE_N=32, TILE_K=16
// Workgroup: 8 subgroups of 16 WIs = 128 WIs
//   - 4 subgroups along M (4*8=32 rows)
//   - 2 subgroups along N (2*16=32 cols)
// Each subgroup computes 8x16 output via DPAS
// K-loop steps by 16, using SLM for A and B tiles
//
// Launch config:
//   GWS = (N/32 * 16, M/32 * 8)  -- adjusted for subgroup mapping
//   LWS = (16, 8)  -- 128 WIs = 8 subgroups of 16
//   Subgroup size = 16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define TILE_M 32
#define TILE_N 32
#define TILE_K 16
#define SG_SIZE 16
#define SLM_A_STRIDE (TILE_K + 2)   // pad to avoid bank conflicts
#define SLM_B_STRIDE (TILE_N + 2)

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Workgroup tile position
    const int wg_n = get_group_id(0) * TILE_N;  // column start
    const int wg_m = get_group_id(1) * TILE_M;  // row start

    // Local IDs
    const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int sg_id = lid / SG_SIZE;        // subgroup index [0..7]
    const int sg_lid = lid % SG_SIZE;       // lane within subgroup [0..15]

    // Subgroup position within workgroup tile
    // 4 subgroups along M, 2 along N
    const int sg_row = sg_id / 2;   // [0..3] -> which 8-row block
    const int sg_col = sg_id % 2;   // [0..1] -> which 16-col block

    const int out_row_start = wg_m + sg_row * 8;
    const int out_col_start = wg_n + sg_col * 16;

    // Accumulator: 8 rows × 16 cols, each WI holds 8 floats (one per row, its column is sg_lid)
    float8 acc = (float8)(0.0f);

    // SLM for tiles
    __local half slm_a[TILE_M * SLM_A_STRIDE];  // 32 x 18
    __local half slm_b[TILE_K * SLM_B_STRIDE];  // 16 x 34

    const int total_wis = 128; // 8 subgroups * 16

    // K-loop
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperative load A[wg_m..wg_m+31, k0..k0+15] into SLM
        // Total elements: 32 * 16 = 512, with 128 WIs -> 4 elements each
        for (int i = lid; i < TILE_M * TILE_K; i += total_wis) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gm = wg_m + row;
            int gk = k0 + col;
            half val = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0f;
            slm_a[row * SLM_A_STRIDE + col] = val;
        }

        // Cooperative load B[k0..k0+15, wg_n..wg_n+31] into SLM
        // Total elements: 16 * 32 = 512, with 128 WIs -> 4 elements each
        for (int i = lid; i < TILE_K * TILE_N; i += total_wis) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int gk = k0 + row;
            int gn = wg_n + col;
            half val = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0f;
            slm_b[row * SLM_B_STRIDE + col] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Each subgroup reads its A block (8 rows x 16 K) and B block (16 K x 16 cols)
        // A: rows [sg_row*8 .. sg_row*8+7], cols [0..15] in SLM
        // B: rows [0..15], cols [sg_col*16 .. sg_col*16+15] in SLM

        // Read A: 8 rows x 16 half values = short8 per WI
        // For DPAS, A needs to be in VNNI-like format: short8 where each short is a pair of halfs
        // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
        //   a: short8 - each WI holds 8 shorts, representing 8 rows, each with k-data distributed across subgroup
        //   b: int8 - packed B matrix

        // Load A tile for this subgroup from SLM
        // A is 8 rows x 16 cols of half. We need short8 per work-item.
        // Each work-item sg_lid provides the k-index. For row r, the value is A[r, sg_lid].
        // short8 means 8 rows, each row has one half value at column sg_lid, packed as short.
        short8 a_val;
        __local half* a_base = slm_a + sg_row * 8 * SLM_A_STRIDE;
        a_val.s0 = as_short(a_base[0 * SLM_A_STRIDE + sg_lid]);
        a_val.s1 = as_short(a_base[1 * SLM_A_STRIDE + sg_lid]);
        a_val.s2 = as_short(a_base[2 * SLM_A_STRIDE + sg_lid]);
        a_val.s3 = as_short(a_base[3 * SLM_A_STRIDE + sg_lid]);
        a_val.s4 = as_short(a_base[4 * SLM_A_STRIDE + sg_lid]);
        a_val.s5 = as_short(a_base[5 * SLM_A_STRIDE + sg_lid]);
        a_val.s6 = as_short(a_base[6 * SLM_A_STRIDE + sg_lid]);
        a_val.s7 = as_short(a_base[7 * SLM_A_STRIDE + sg_lid]);

        // Load B tile for this subgroup from SLM
        // B is 16 rows x 16 cols of half. For DPAS, int8 packing:
        // int8 = 8 ints, each int packs 2 consecutive half values (2 rows) for one column
        // Layout: b_val.s0 = pack(B[0, sg_lid], B[1, sg_lid]), etc.
        __local half* b_base = slm_b + sg_col * 16;
        int8 b_val;
        b_val.s0 = as_int((half2)(b_base[0 * SLM_B_STRIDE + sg_lid], b_base[1 * SLM_B_STRIDE + sg_lid]));
        b_val.s1 = as_int((half2)(b_base[2 * SLM_B_STRIDE + sg_lid], b_base[3 * SLM_B_STRIDE + sg_lid]));
        b_val.s2 = as_int((half2)(b_base[4 * SLM_B_STRIDE + sg_lid], b_base[5 * SLM_B_STRIDE + sg_lid]));
        b_val.s3 = as_int((half2)(b_base[6 * SLM_B_STRIDE + sg_lid], b_base[7 * SLM_B_STRIDE + sg_lid]));
        b_val.s4 = as_int((half2)(b_base[8 * SLM_B_STRIDE + sg_lid], b_base[9 * SLM_B_STRIDE + sg_lid]));
        b_val.s5 = as_int((half2)(b_base[10 * SLM_B_STRIDE + sg_lid], b_base[11 * SLM_B_STRIDE + sg_lid]));
        b_val.s6 = as_int((half2)(b_base[12 * SLM_B_STRIDE + sg_lid], b_base[13 * SLM_B_STRIDE + sg_lid]));
        b_val.s7 = as_int((half2)(b_base[14 * SLM_B_STRIDE + sg_lid], b_base[15 * SLM_B_STRIDE + sg_lid]));

        // DPAS: 8x16 = 8x16 * 16x16
        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    // Each WI holds 8 float values for rows [out_row_start..+7], col [out_col_start + sg_lid]
    int out_col = out_col_start + sg_lid;
    if (out_col < N) {
        if (out_row_start + 0 < M) C[(out_row_start + 0) * N + out_col] = convert_half(acc.s0);
        if (out_row_start + 1 < M) C[(out_row_start + 1) * N + out_col] = convert_half(acc.s1);
        if (out_row_start + 2 < M) C[(out_row_start + 2) * N + out_col] = convert_half(acc.s2);
        if (out_row_start + 3 < M) C[(out_row_start + 3) * N + out_col] = convert_half(acc.s3);
        if (out_row_start + 4 < M) C[(out_row_start + 4) * N + out_col] = convert_half(acc.s4);
        if (out_row_start + 5 < M) C[(out_row_start + 5) * N + out_col] = convert_half(acc.s5);
        if (out_row_start + 6 < M) C[(out_row_start + 6) * N + out_col] = convert_half(acc.s6);
        if (out_row_start + 7 < M) C[(out_row_start + 7) * N + out_col] = convert_half(acc.s7);
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 32.300):
```OCL
// Optimized FP16 GEMM using Intel XMX DPAS instructions
// C[M,N] = A[M,K] x B[K,N], all half precision, float accumulation
//
// Tiling: TILE_M=32, TILE_N=64, TILE_K=32
// Workgroup: 16 subgroups of 16 WIs = 256 WIs
//   - 4 subgroups along M (4*8=32 rows)
//   - 4 subgroups along N (4*16=64 cols)
// Each subgroup computes 8x16 output via DPAS
// K-loop steps by 32, two DPAS calls per iteration
//
// Launch config:
//   LWS = (16, 16) = 256 WIs
//   GWS = (ceil(N/64)*16, ceil(M/32)*16)
//   Subgroup size = 16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define TILE_M 32
#define TILE_N 64
#define TILE_K 32
#define SG_SIZE 16
#define SLM_A_STRIDE (TILE_K + 2)
#define SLM_B_STRIDE (TILE_N + 2)
#define NUM_WIS 256

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

    const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
    const int sg_id = lid / SG_SIZE;
    const int sg_lid = lid % SG_SIZE;

    // 4 along M, 4 along N
    const int sg_row = sg_id / 4;   // [0..3]
    const int sg_col = sg_id % 4;   // [0..3]

    const int out_row_start = wg_m + sg_row * 8;
    const int out_col_start = wg_n + sg_col * 16;

    float8 acc = (float8)(0.0f);

    // SLM: A[32 x 34], B[32 x 66]  (padded)
    // For TILE_K=32, we do 2 DPAS calls (k=0..15, k=16..31)
    __local half slm_a[TILE_M * SLM_A_STRIDE];
    __local half slm_b[TILE_K * SLM_B_STRIDE];

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A[wg_m..+32, k0..+32] into SLM: 32*32=1024 elements, 256 WIs -> 4 each
        for (int i = lid; i < TILE_M * TILE_K; i += NUM_WIS) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gm = wg_m + row;
            int gk = k0 + col;
            half val = (gm < M && gk < K) ? A[gm * K + gk] : (half)0.0f;
            slm_a[row * SLM_A_STRIDE + col] = val;
        }

        // Load B[k0..+32, wg_n..+64] into SLM: 32*64=2048 elements, 256 WIs -> 8 each
        for (int i = lid; i < TILE_K * TILE_N; i += NUM_WIS) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int gk = k0 + row;
            int gn = wg_n + col;
            half val = (gk < K && gn < N) ? B[gk * N + gn] : (half)0.0f;
            slm_b[row * SLM_B_STRIDE + col] = val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Two DPAS iterations: k_inner=0 and k_inner=16
        #pragma unroll
        for (int ki = 0; ki < TILE_K; ki += 16) {
            // Load A: 8 rows x 16 K cols
            __local half* a_base = slm_a + sg_row * 8 * SLM_A_STRIDE + ki;
            short8 a_val;
            a_val.s0 = as_short(a_base[0 * SLM_A_STRIDE + sg_lid]);
            a_val.s1 = as_short(a_base[1 * SLM_A_STRIDE + sg_lid]);
            a_val.s2 = as_short(a_base[2 * SLM_A_STRIDE + sg_lid]);
            a_val.s3 = as_short(a_base[3 * SLM_A_STRIDE + sg_lid]);
            a_val.s4 = as_short(a_base[4 * SLM_A_STRIDE + sg_lid]);
            a_val.s5 = as_short(a_base[5 * SLM_A_STRIDE + sg_lid]);
            a_val.s6 = as_short(a_base[6 * SLM_A_STRIDE + sg_lid]);
            a_val.s7 = as_short(a_base[7 * SLM_A_STRIDE + sg_lid]);

            // Load B: 16 K rows x 16 cols, packed as int8
            __local half* b_base = slm_b + ki * SLM_B_STRIDE + sg_col * 16;
            int8 b_val;
            b_val.s0 = as_int((half2)(b_base[0 * SLM_B_STRIDE + sg_lid], b_base[1 * SLM_B_STRIDE + sg_lid]));
            b_val.s1 = as_int((half2)(b_base[2 * SLM_B_STRIDE + sg_lid], b_base[3 * SLM_B_STRIDE + sg_lid]));
            b_val.s2 = as_int((half2)(b_base[4 * SLM_B_STRIDE + sg_lid], b_base[5 * SLM_B_STRIDE + sg_lid]));
            b_val.s3 = as_int((half2)(b_base[6 * SLM_B_STRIDE + sg_lid], b_base[7 * SLM_B_STRIDE + sg_lid]));
            b_val.s4 = as_int((half2)(b_base[8 * SLM_B_STRIDE + sg_lid], b_base[9 * SLM_B_STRIDE + sg_lid]));
            b_val.s5 = as_int((half2)(b_base[10 * SLM_B_STRIDE + sg_lid], b_base[11 * SLM_B_STRIDE + sg_lid]));
            b_val.s6 = as_int((half2)(b_base[12 * SLM_B_STRIDE + sg_lid], b_base[13 * SLM_B_STRIDE + sg_lid]));
            b_val.s7 = as_int((half2)(b_base[14 * SLM_B_STRIDE + sg_lid], b_base[15 * SLM_B_STRIDE + sg_lid]));

            acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    int out_col = out_col_start + sg_lid;
    if (out_col < N) {
        if (out_row_start + 0 < M) C[(out_row_start + 0) * N + out_col] = convert_half(acc.s0);
        if (out_row_start + 1 < M) C[(out_row_start + 1) * N + out_col] = convert_half(acc.s1);
        if (out_row_start + 2 < M) C[(out_row_start + 2) * N + out_col] = convert_half(acc.s2);
        if (out_row_start + 3 < M) C[(out_row_start + 3) * N + out_col] = convert_half(acc.s3);
        if (out_row_start + 4 < M) C[(out_row_start + 4) * N + out_col] = convert_half(acc.s4);
        if (out_row_start + 5 < M) C[(out_row_start + 5) * N + out_col] = convert_half(acc.s5);
        if (out_row_start + 6 < M) C[(out_row_start + 6) * N + out_col] = convert_half(acc.s6);
        if (out_row_start + 7 < M) C[(out_row_start + 7) * N + out_col] = convert_half(acc.s7);
    }
}
```

Console output from running this kernel:

Test result on platform Intel Corporation Battlemage G21 [Intel Graphics]:
==== test session starts

task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0] PASSED           [ 25%]
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[1] PASSED           [ 50%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[0] PASSED         [ 75%]
task.py::TestMatmulOCL::test_correctness_wrt_reference[1] PASSED         [100%]

=============================== warnings summary ===============================
task.py::TestMatmulOCL::test_correctness_wrt_pytorch[0]
  /home/openvino-ci-74/miniforge3/envs/kernel_intel/lib/python3.12/site-packages/pyopencl/cache.py:517: CompilerWarning: Non-empty compiler output encountered. Set the environment variable PYOPENCL_COMPILER_OUTPUT=1 to see more.
    _create_built_program_from_source_cached(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================== 4 passed, 1 deselected, 1 warning in 0.84s ==================
The kernel compiles and is correct, great job!

## Hardware specification:
Your code will run on the following hardware:
**Intel Battlemage** with specs: Xe-cores: 20, Render Slices: 5, Ray Tracing Units: 20, Intel® Xe Matrix Extensions (Intel® XMX) Engines: 160, Xe Vector Engines: 160, Graphics Clock: 2670, GPU Peak TOPS (Int8): 233, TBP: 190, PCI Express Configurations ‡: PCI Express 4.0 x8, Device ID: 0xE20B, Memory: 12 GB GDDR6, Memory Interface: 192 bit, Memory Bandwidth: 456, Memory Speed: 19, ISA_GPU: Xe2-HPG
Please consider the hardware specifications when improving the code. 

## Task:

**Your objectives**:
1. Analyze the previous versions and their results (why does one achieve better results than the other?).
2. Identify any inefficiencies and bottlenecks.
3. Propose specific improvements or options to take the best of all prior versions, explaining your reasoning step by step.

4. Provide a new kernel that achieves better performance **on the target hardware**. Provide the complete, improved code in a code block.

**Optimization strategies**:

Here are some potential strategies to improve the kernel runtime:
1. Minimize Synchronization: Reduce barrier() calls. Use barrier(CLK_LOCAL_MEM_FENCE) for local memory only, barrier(CLK_GLOBAL_MEM_FENCE) when needed for global. Sub-group operations are barrier-free. Use work-group-level reductions only when necessary.
2. Maximize Occupancy: Balance work-group size with register/local memory usage. Query CL_KERNEL_WORK_GROUP_SIZE and CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE. Typical work-group sizes: 64-256 for compute-bound, 32-128 for memory-bound kernels. Use multiples of wavefront/warp size (32 for NVIDIA, 64 for AMD).

**Critical Requirements:**

1. The kernel must exactly match the reference implementation's functionality.
2. The code must compile and run properly on the GPU.
3. Do not cache or reuse previous results; ensure the code executes fully on each run.
4. Keep all hyperparameters (e.g., batch size, dimensions) unchanged as specified in the reference implementation.
8. Beware of the critical error "Unexpected kernel lambda size. In such cases removing constexpr specifier aligns the captures between the host compiler and the device compiler"! Do not capture constexpr variables in lambda functions passed to kernel launches as this can lead to different lambda sizes between the host and device compiler.

Additional Guidance:

1. Clearly comment on any performance optimizations you implement.
2. If you change the kernel structure, explain why.
3. Anticipate and address possible runtime or compilation errors.

Please structure your response as follows:

1. Analysis:
    * Summarize the issues found in the previous kernel and log.
    * Explain your proposed changes and optimizations.
2. Improved OCL code:
    * Provide the complete, improved OCL code in a code block:
```OCL
Your code here
```


## Required Optimizations

Apply the following optimization techniques in your implementation:

- **Register Blocking**: Each work-item computes a THREAD_M×THREAD_N output block in private register arrays. Use `#pragma unroll` on inner loops. Combine with SLM tiling for multi-level memory hierarchy optimization.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.