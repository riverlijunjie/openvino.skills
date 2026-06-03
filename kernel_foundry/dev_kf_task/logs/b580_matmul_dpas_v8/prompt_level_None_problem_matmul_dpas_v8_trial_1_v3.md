

# You are a OCL programming expert specializing in GPU kernel optimization. 
Given a reference OCL implementation, your objective is to create a performant OCL kernel with identical functionality as the reference.

The code you generate will be pasted into an existing project. Make sure to follow the existing code structure and function signatures.

## The user provided the following additional instructions for you:
Optimization goals:
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
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// DPAS-based matmul: C[M,N] = A[M,K] * B[K,N], all half, acc in float
// Each sub-group computes 8 rows x 16 cols using intel_sub_group_f16_f16_matrix_mad_k16
// TILE_M=32 (4 sub-groups vertically), TILE_N=16 (1 sub-group wide)
// LWS = (16, 4), sub-group_size=16 => 4 sub-groups per WG
// GWS = (ceil(N/16)*16, ceil(M/32)*4)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_local_id(1);       // which sub-group row [0..3]
    const int sg_lane = get_sub_group_local_id(); // lane within sub-group [0..15]

    const int tile_col = get_group_id(0) * 16;   // N tile start
    const int tile_row = get_group_id(1) * 32;    // M tile start

    const int row_base = tile_row + sg_id * 8;    // this sub-group's starting row

    float8 acc = (float8)(0.0f);

    for (int k = 0; k < K; k += 16) {
        // Load A: 8 rows x 16 cols (k-dim)
        // Each work-item loads one column of 8 half values -> short8
        // A[row_base + r][k + sg_lane] for r=0..7
        short8 a_val;
        if (row_base + 7 < M && k + 15 < K) {
            half8 a_tmp;
            a_tmp.s0 = A[(row_base + 0) * K + k + sg_lane];
            a_tmp.s1 = A[(row_base + 1) * K + k + sg_lane];
            a_tmp.s2 = A[(row_base + 2) * K + k + sg_lane];
            a_tmp.s3 = A[(row_base + 3) * K + k + sg_lane];
            a_tmp.s4 = A[(row_base + 4) * K + k + sg_lane];
            a_tmp.s5 = A[(row_base + 5) * K + k + sg_lane];
            a_tmp.s6 = A[(row_base + 6) * K + k + sg_lane];
            a_tmp.s7 = A[(row_base + 7) * K + k + sg_lane];
            a_val = as_short8(a_tmp);
        } else {
            half a_arr[8];
            for (int r = 0; r < 8; r++) {
                int row_idx = row_base + r;
                int col_idx = k + sg_lane;
                a_arr[r] = (row_idx < M && col_idx < K) ? A[row_idx * K + col_idx] : (half)0.0h;
            }
            a_val = as_short8(vload8(0, a_arr));
        }

        // Load B: 16 rows x 16 cols
        // B is [K, N], we need B[k..k+15][tile_col..tile_col+15]
        // For DPAS, B needs to be in VNNI-like format: int8 per lane
        // int8 b_val: each int packs 2 halfs (row-pair), 8 ints = 16 rows, distributed across 16 lanes (cols)
        int8 b_val;
        if (k + 15 < K && tile_col + 15 < N) {
            // Each lane sg_lane handles column (tile_col + sg_lane)
            // Pack pairs of rows into ints
            for (int p = 0; p < 8; p++) {
                int r0 = k + p * 2;
                int r1 = k + p * 2 + 1;
                half bv0 = B[r0 * N + tile_col + sg_lane];
                half bv1 = B[r1 * N + tile_col + sg_lane];
                ((int*)&b_val)[p] = as_int((half2)(bv0, bv1));
            }
        } else {
            for (int p = 0; p < 8; p++) {
                int r0 = k + p * 2;
                int r1 = k + p * 2 + 1;
                half bv0 = (r0 < K && tile_col + sg_lane < N) ? B[r0 * N + tile_col + sg_lane] : (half)0.0h;
                half bv1 = (r1 < K && tile_col + sg_lane < N) ? B[r1 * N + tile_col + sg_lane] : (half)0.0h;
                ((int*)&b_val)[p] = as_int((half2)(bv0, bv1));
            }
        }

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);
    }

    // Store C: 8 rows x 16 cols
    int out_col = tile_col + sg_lane;
    for (int r = 0; r < 8; r++) {
        int out_row = row_base + r;
        if (out_row < M && out_col < N) {
            C[out_row * N + out_col] = convert_half(((float*)&acc)[r]);
        }
    }
}
```

### Version 2 (Result: Correctness score: 3 / 5 (compiled but incorrect), runtime in ms: -1.000):
```OCL
// Launch config:
//   GWS = (ceil(N/32)*16, ceil(M/32)*8)
//   LWS = (16, 8)
//   Subgroup size = 16

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

#define TM 32
#define TN 32
#define TK 16

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile position
    const int wg_n = (get_group_id(0)) * TN;  // column start
    const int wg_m = (get_group_id(1)) * TM;  // row start

    // Local IDs
    const int lid_x = get_local_id(0); // 0..15
    const int lid_y = get_local_id(1); // 0..7

    // Subgroup ID within work-group (0..7)
    const int sg_id = lid_y;
    // Map subgroups: sg_row = sg_id / 2 (0..3), sg_col = sg_id % 2 (0..1)
    const int sg_row = sg_id >> 1;  // which 8-row block (0..3)
    const int sg_col = sg_id & 1;   // which 16-col block (0..1)

    const int sub_m = wg_m + sg_row * 8;  // starting row for this subgroup's tile
    const int sub_n = wg_n + sg_col * 16; // starting col for this subgroup's tile

    // SLM for A tile [TM][TK] and B tile [TK][TN], stored as half
    __local half slm_a[TM * TK];   // 32 * 16 = 512 halfs
    __local half slm_b[TK * TN];   // 16 * 32 = 512 halfs

    // Accumulator: 8 rows × 16 cols, each work-item holds float8 (8 rows, 1 col per item)
    float8 acc = (float8)(0.0f);

    const int lane = lid_x; // sub_group_local_id equivalent

    // Loop over K in steps of TK=16
    for (int k0 = 0; k0 < K; k0 += TK) {

        // === Cooperative load of A[wg_m..wg_m+31][k0..k0+15] into slm_a ===
        // 128 work-items, 512 elements to load -> 4 elements per work-item
        {
            int flat_id = lid_y * 16 + lid_x; // 0..127
            for (int i = 0; i < 4; i++) {
                int idx = flat_id + i * 128;
                int r = idx / TK;  // row within tile
                int c = idx % TK;  // col within tile
                int global_r = wg_m + r;
                int global_c = k0 + c;
                half val = (global_r < M && global_c < K) ? A[global_r * K + global_c] : (half)0.0h;
                slm_a[r * TK + c] = val;
            }
        }

        // === Cooperative load of B[k0..k0+15][wg_n..wg_n+31] into slm_b ===
        // 512 elements, 128 items -> 4 per item
        {
            int flat_id = lid_y * 16 + lid_x;
            for (int i = 0; i < 4; i++) {
                int idx = flat_id + i * 128;
                int r = idx / TN;
                int c = idx % TN;
                int global_r = k0 + r;
                int global_c = wg_n + c;
                half val = (global_r < K && global_c < N) ? B[global_r * N + global_c] : (half)0.0h;
                slm_b[r * TN + c] = val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // === DPAS computation ===
        // This subgroup computes C[sub_m..sub_m+7][sub_n..sub_n+15]
        // A sub-tile: slm_a[sg_row*8 .. sg_row*8+7][0..15] -> 8 rows × 16 cols
        // B sub-tile: slm_b[0..15][sg_col*16 .. sg_col*16+15] -> 16 rows × 16 cols

        // Load A: short8 per work-item
        // For DPAS, A is packed row-major: each work-item k reads rows 0..7 at column k
        // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
        // A (short8): VNNI-like layout - subgroup block read from row-major 8×16 half matrix
        // B (int8): subgroup block read from 16×16 half matrix (VNNI packed)

        // A: 8 rows × 16 cols of half = 8×16 = 128 halfs = 8 ushort per lane (16 lanes)
        // Use intel_sub_group_block_read_us8 on the SLM A tile
        // A is stored row-major in SLM: slm_a[row][col], stride = TK = 16
        // For block read: base = &slm_a[sg_row * 8 * TK], stride between rows = TK
        // Block read reads 8 consecutive ushort values per work-item from contiguous memory
        // For matrix A we need row-major 8x16, which is 128 halfs contiguous (since TK=16, rows are contiguous)

        __local ushort* a_ptr = (__local ushort*)&slm_a[sg_row * 8 * TK];
        short8 a_val = as_short8(intel_sub_group_block_read_us8(a_ptr));

        // B: 16 rows × 16 cols of half, need VNNI format for DPAS
        // For the B operand, data should be in VNNI format (pairs of halfs packed as uint)
        // slm_b is row-major: slm_b[row * TN + col]
        // We need columns [sg_col*16 .. sg_col*16+15]
        // VNNI packing: pairs of k-rows are interleaved
        // For block_read_ui8: reads 8 uint per lane from 16 lanes = 8×16 uint = 16×16 half (VNNI packed)

        // Repack B into VNNI format in registers via block read
        // B tile for this subgroup: 16 rows × 16 cols starting at slm_b[0 * TN + sg_col * 16]
        // With TN=32, stride=32, we can't do a simple contiguous block read.
        // Instead, let's read B manually and pack.

        // Actually, for B in VNNI format with block reads, data needs to be contiguous.
        // Since our B tile cols aren't contiguous in SLM (stride=TN=32), we need to handle this.

        // Manual VNNI packing for B:
        // VNNI format: for each pair of k-rows (k, k+1), interleave across the 16 columns
        // int8 b_val: 8 ints per work-item, where each int packs 2 halfs from consecutive k-rows

        __local uint* b_base = (__local uint*)&slm_b[sg_col * 16];
        // b_base points to row 0, col sg_col*16, but stride is TN=32 halfs = 16 uints
        // Each row has TN/2 = 16 uints when viewed as uint (pairs of halfs)
        // But we want VNNI: pairs of rows packed together

        // For VNNI, we need to rearrange. Let's just manually construct:
        int8 b_val;
        // Each work-item (lane) handles one column out of 16
        // For k-pair i (i=0..7), we read rows 2i and 2i+1 at our column
        // Pack as: low half from row 2i, high half from row 2i+1

        // Actually, let me reconsider. The VNNI format for B expected by DPAS:
        // B is 16×16 half in VNNI = 8×16 uint where each uint = (half[k+1] << 16) | half[k]
        // With intel_sub_group_block_read_ui8, each lane reads 8 consecutive uints,
        // and across 16 lanes we get 8×16 = 128 uints = 16×16 halfs.

        // We need B data laid out as VNNI in memory for block read to work.
        // Since it's not, let's pack manually:

        ushort b_raw[16];
        for (int kk = 0; kk < 16; kk++) {
            b_raw[kk] = as_ushort(slm_b[kk * TN + sg_col * 16 + lane]);
        }
        // Pack into VNNI: pairs of consecutive k values
        uint b_packed[8];
        for (int p = 0; p < 8; p++) {
            b_packed[p] = ((uint)b_raw[2*p+1] << 16) | (uint)b_raw[2*p];
        }
        b_val = (int8)(as_int(b_packed[0]), as_int(b_packed[1]), as_int(b_packed[2]), as_int(b_packed[3]),
                        as_int(b_packed[4]), as_int(b_packed[5]), as_int(b_packed[6]), as_int(b_packed[7]));

        acc = intel_sub_group_f16_f16_matrix_mad_k16(a_val, b_val, acc);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store results ===
    // Each work-item in subgroup holds float8 = 8 rows, lane determines column
    int col = sub_n + lane;
    for (int i = 0; i < 8; i++) {
        int row = sub_m + i;
        if (row < M && col < N) {
            float val;
            switch(i) {
                case 0: val = acc.s0; break;
                case 1: val = acc.s1; break;
                case 2: val = acc.s2; break;
                case 3: val = acc.s3; break;
                case 4: val = acc.s4; break;
                case 5: val = acc.s5; break;
                case 6: val = acc.s6; break;
                case 7: val = acc.s7; break;
            }
            C[row * N + col] = convert_half(val);
        }
    }
}
```

### Previous kernel and evaluation:
Here is the last OCL kernel we tested (Result: Correctness score: 5 / 5 (compiles and correct), runtime in ms: 332.000):
```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

// Tiling: each work-group (1 subgroup of 16 WIs) computes 32x32 output
// DPAS: intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc) -> 8x16 output per call
// Register blocking: 4 vertical (4*8=32 rows) x 2 horizontal (2*16=32 cols) DPAS calls
// K tiled in steps of 16
// GWS = (ceil(N/32)*16, ceil(M/32)*1)  -- x has 16 WIs per subgroup
// LWS = (16, 1)

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile position
    const int wg_n = get_group_id(0); // which 32-col tile
    const int wg_m = get_group_id(1); // which 32-row tile

    const int base_m = wg_m * 32;
    const int base_n = wg_n * 32;

    const int sg_lid = get_sub_group_local_id(); // 0..15

    // Accumulators: 4 vertical x 2 horizontal = 8 float8 registers
    float8 acc00 = 0.0f, acc01 = 0.0f;
    float8 acc10 = 0.0f, acc11 = 0.0f;
    float8 acc20 = 0.0f, acc21 = 0.0f;
    float8 acc30 = 0.0f, acc31 = 0.0f;

    // SLM for A tile [32][16] and B tile [16][32] in half
    // A: 32*16 = 512 halfs = 1024 bytes
    // B: 16*32 = 512 halfs = 1024 bytes
    __local half slm_a[32 * 16];
    __local half slm_b[16 * 32];

    for (int k0 = 0; k0 < K; k0 += 16) {
        // Cooperative load A[32][16] - 512 halfs, 16 WIs, each loads 32 halfs
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int row = base_m + i;
            int col = k0 + sg_lid;
            half val = 0;
            if (row < M && col < K)
                val = A[row * K + col];
            slm_a[i * 16 + sg_lid] = val;
        }

        // Cooperative load B[16][32] - 512 halfs, 16 WIs, each loads 32 halfs
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int row = k0 + i;
            // Load two columns per WI to cover 32 columns
            int col0 = sg_lid;
            int col1 = sg_lid + 16;
            half val0 = 0, val1 = 0;
            if (row < K && (base_n + col0) < N)
                val0 = B[row * N + base_n + col0];
            if (row < K && (base_n + col1) < N)
                val1 = B[row * N + base_n + col1];
            slm_b[i * 32 + col0] = val0;
            slm_b[i * 32 + col1] = val1;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // DPAS calls: read from SLM using subgroup block reads
        // A: for each 8-row chunk, read short8 (8 halfs = 8 rows, k=16 deep packed)
        // For DPAS, A is short8 where each short holds 1 fp16, 8 shorts = 8 rows × 1 k-element?
        // Actually: short8 a means 8×16-bit = 8 fp16 values. For k16, we need 8 rows each with 16 fp16 values.
        // The DPAS k16 variant: a is short8 per WI, across subgroup of 16 WIs gives 8×16 = 128 fp16 = 8 rows × 16 k-elements
        // So each WI's short8 has: row[i]'s k-element at position sg_lid, for i=0..7
        // A layout in SLM: [row][k] with k stride 16, row stride 16 halfs
        // Block read: intel_sub_group_block_read_us8 reads 8 consecutive ushorts per WI from 16-aligned address
        // We need A in column-major within the 8x16 block for the DPAS format

        // A SLM layout is [32][16] row-major. For block read of 8 rows:
        // Address = &slm_a[chunk*8*16], stride = 16 halfs = 32 bytes
        // intel_sub_group_block_read_us8 with stride reads 8 consecutive rows

        // For A: need to use block_read with pitch. Let's manually construct short8.
        short8 a0, a1, a2, a3;

        __local ushort* slm_a_us = (__local ushort*)slm_a;

        // Chunk 0: rows 0-7
        a0 = as_short8((ushort8)(
            slm_a_us[0*16 + sg_lid], slm_a_us[1*16 + sg_lid],
            slm_a_us[2*16 + sg_lid], slm_a_us[3*16 + sg_lid],
            slm_a_us[4*16 + sg_lid], slm_a_us[5*16 + sg_lid],
            slm_a_us[6*16 + sg_lid], slm_a_us[7*16 + sg_lid]));

        a1 = as_short8((ushort8)(
            slm_a_us[8*16 + sg_lid], slm_a_us[9*16 + sg_lid],
            slm_a_us[10*16 + sg_lid], slm_a_us[11*16 + sg_lid],
            slm_a_us[12*16 + sg_lid], slm_a_us[13*16 + sg_lid],
            slm_a_us[14*16 + sg_lid], slm_a_us[15*16 + sg_lid]));

        a2 = as_short8((ushort8)(
            slm_a_us[16*16 + sg_lid], slm_a_us[17*16 + sg_lid],
            slm_a_us[18*16 + sg_lid], slm_a_us[19*16 + sg_lid],
            slm_a_us[20*16 + sg_lid], slm_a_us[21*16 + sg_lid],
            slm_a_us[22*16 + sg_lid], slm_a_us[23*16 + sg_lid]));

        a3 = as_short8((ushort8)(
            slm_a_us[24*16 + sg_lid], slm_a_us[25*16 + sg_lid],
            slm_a_us[26*16 + sg_lid], slm_a_us[27*16 + sg_lid],
            slm_a_us[28*16 + sg_lid], slm_a_us[29*16 + sg_lid],
            slm_a_us[30*16 + sg_lid], slm_a_us[31*16 + sg_lid]));

        // For B: [16][32] row-major. DPAS B operand int8 per WI.
        // B matrix for DPAS: 16 k-elements × 16 n-elements (one DPAS n-block)
        // int8 = 8 ints = 8×32bit = 8×2×fp16 = 16 fp16 per WI, across 16 WIs = 256 fp16 = 16×16
        // B layout for DPAS: VNNI format - pairs of k are packed into 32-bit
        // B[k][n] -> for VNNI: B_packed[k/2][n][2] as uint
        // SLM B is [16][32]. For n-block 0 (cols 0-15):
        //   We need B[0..15][0..15] in VNNI format

        __local uint* slm_b_ui = (__local uint*)slm_b;
        // slm_b as ushort: slm_b[k*32 + n]
        // As uint (pairs of halfs): slm_b_ui[k*16 + n/2]... no, that pairs adjacent columns
        // We need VNNI: pack k=2i and k=2i+1 for same n column
        // slm_b[2i*32+n] and slm_b[(2i+1)*32+n] -> packed as uint

        // B VNNI for n-block 0 (n=0..15): b_vnni[i][n] = pack(B[2i][n], B[2i+1][n])
        // i=0..7, n=0..15 -> int8 per WI (WI sg_lid reads n=sg_lid)

        __local ushort* slm_b_us = (__local ushort*)slm_b;

        int8 b0, b1;

        // B block 0: cols 0-15
        b0 = (int8)(
            as_int((ushort2)(slm_b_us[0*32 + sg_lid], slm_b_us[1*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[2*32 + sg_lid], slm_b_us[3*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[4*32 + sg_lid], slm_b_us[5*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[6*32 + sg_lid], slm_b_us[7*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[8*32 + sg_lid], slm_b_us[9*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[10*32 + sg_lid], slm_b_us[11*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[12*32 + sg_lid], slm_b_us[13*32 + sg_lid])),
            as_int((ushort2)(slm_b_us[14*32 + sg_lid], slm_b_us[15*32 + sg_lid])));

        // B block 1: cols 16-31
        b1 = (int8)(
            as_int((ushort2)(slm_b_us[0*32 + 16 + sg_lid], slm_b_us[1*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[2*32 + 16 + sg_lid], slm_b_us[3*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[4*32 + 16 + sg_lid], slm_b_us[5*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[6*32 + 16 + sg_lid], slm_b_us[7*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[8*32 + 16 + sg_lid], slm_b_us[9*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[10*32 + 16 + sg_lid], slm_b_us[11*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[12*32 + 16 + sg_lid], slm_b_us[13*32 + 16 + sg_lid])),
            as_int((ushort2)(slm_b_us[14*32 + 16 + sg_lid], slm_b_us[15*32 + 16 + sg_lid])));

        // 4×2 = 8 DPAS calls
        acc00 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b0, acc00);
        acc01 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b1, acc01);
        acc10 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b0, acc10);
        acc11 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b1, acc11);
        acc20 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b0, acc20);
        acc21 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b1, acc21);
        acc30 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b0, acc30);
        acc31 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b1, acc31);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results: each accumulator is float8 = 8 rows × 1 col per WI, 16 WIs = 8 rows × 16 cols
    // acc[row_chunk][col_chunk].s[i] = C[base_m + chunk*8 + i][base_n + col_chunk*16 + sg_lid]

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = base_m + 0*8 + i;
        int col0 = base_n + sg_lid;
        int col1 = base_n + 16 + sg_lid;
        if (row < M) {
            float v0, v1;
            switch(i) {
                case 0: v0 = acc00.s0; v1 = acc01.s0; break;
                case 1: v0 = acc00.s1; v1 = acc01.s1; break;
                case 2: v0 = acc00.s2; v1 = acc01.s2; break;
                case 3: v0 = acc00.s3; v1 = acc01.s3; break;
                case 4: v0 = acc00.s4; v1 = acc01.s4; break;
                case 5: v0 = acc00.s5; v1 = acc01.s5; break;
                case 6: v0 = acc00.s6; v1 = acc01.s6; break;
                case 7: v0 = acc00.s7; v1 = acc01.s7; break;
            }
            if (col0 < N) C[row * N + col0] = convert_half(v0);
            if (col1 < N) C[row * N + col1] = convert_half(v1);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = base_m + 8 + i;
        int col0 = base_n + sg_lid;
        int col1 = base_n + 16 + sg_lid;
        if (row < M) {
            float v0, v1;
            switch(i) {
                case 0: v0 = acc10.s0; v1 = acc11.s0; break;
                case 1: v0 = acc10.s1; v1 = acc11.s1; break;
                case 2: v0 = acc10.s2; v1 = acc11.s2; break;
                case 3: v0 = acc10.s3; v1 = acc11.s3; break;
                case 4: v0 = acc10.s4; v1 = acc11.s4; break;
                case 5: v0 = acc10.s5; v1 = acc11.s5; break;
                case 6: v0 = acc10.s6; v1 = acc11.s6; break;
                case 7: v0 = acc10.s7; v1 = acc11.s7; break;
            }
            if (col0 < N) C[row * N + col0] = convert_half(v0);
            if (col1 < N) C[row * N + col1] = convert_half(v1);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = base_m + 16 + i;
        int col0 = base_n + sg_lid;
        int col1 = base_n + 16 + sg_lid;
        if (row < M) {
            float v0, v1;
            switch(i) {
                case 0: v0 = acc20.s0; v1 = acc21.s0; break;
                case 1: v0 = acc20.s1; v1 = acc21.s1; break;
                case 2: v0 = acc20.s2; v1 = acc21.s2; break;
                case 3: v0 = acc20.s3; v1 = acc21.s3; break;
                case 4: v0 = acc20.s4; v1 = acc21.s4; break;
                case 5: v0 = acc20.s5; v1 = acc21.s5; break;
                case 6: v0 = acc20.s6; v1 = acc21.s6; break;
                case 7: v0 = acc20.s7; v1 = acc21.s7; break;
            }
            if (col0 < N) C[row * N + col0] = convert_half(v0);
            if (col1 < N) C[row * N + col1] = convert_half(v1);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = base_m + 24 + i;
        int col0 = base_n + sg_lid;
        int col1 = base_n + 16 + sg_lid;
        if (row < M) {
            float v0, v1;
            switch(i) {
                case 0: v0 = acc30.s0; v1 = acc31.s0; break;
                case 1: v0 = acc30.s1; v1 = acc31.s1; break;
                case 2: v0 = acc30.s2; v1 = acc31.s2; break;
                case 3: v0 = acc30.s3; v1 = acc31.s3; break;
                case 4: v0 = acc30.s4; v1 = acc31.s4; break;
                case 5: v0 = acc30.s5; v1 = acc31.s5; break;
                case 6: v0 = acc30.s6; v1 = acc31.s6; break;
                case 7: v0 = acc30.s7; v1 = acc31.s7; break;
            }
            if (col0 < N) C[row * N + col0] = convert_half(v0);
            if (col1 < N) C[row * N + col1] = convert_half(v1);
        }
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
================== 4 passed, 1 deselected, 1 warning in 2.01s ==================
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
1. Use Local Memory Strategically: Use __local memory for data reused across work-items in a work-group. Implement tile-based algorithms where each work-group loads a tile into local memory, processes it, then moves to the next tile. Declare with __local float tile[TILE_SIZE].
2. Use Appropriate Data Types: Use half (FP16) when precision allows - doubles throughput on many GPUs. Use native_* math functions (native_sqrt, native_divide) for faster, less precise operations. Use mad() for fused multiply-add. Size integer types appropriately (char, short, int).

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
- **Blocked/Tiled Algorithms**: Process input in blocks to bound peak memory. Trade recomputation for memory savings (e.g., Flash-Attention style). Maintain running accumulators across blocks with proper rescaling.
- **Hierarchical Parallelism**: Structure work at three levels - work-groups (tile assignment), sub-groups (sub-tile processing), work-items (register tile). Use sub-group shuffles to share data within sub-group without SLM.