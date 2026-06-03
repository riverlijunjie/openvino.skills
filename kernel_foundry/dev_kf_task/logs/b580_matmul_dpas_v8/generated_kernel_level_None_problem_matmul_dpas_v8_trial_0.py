

## Analysis

This is a matrix multiplication C[M,N] = A[M,K] × B[K,N] with FP16 inputs and FP32 accumulation.

**DPAS instruction details:**
- `intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)` computes an 8×16 × 16×16 → 8×16 tile per subgroup (16 work-items).
- Each work-item holds 8 rows of A (as short8 = 8×fp16 packed) and the B matrix is distributed across the subgroup (int8 = 8×2×fp16 = 16 fp16 values per work-item).

**Tiling strategy:**
- Subgroup size = 16 (Xe2-HPG)
- Each DPAS call: computes 8 rows × 16 cols with K=16 reduction
- To increase register blocking, each work-item computes THREAD_M=4 DPAS calls vertically (32 rows) and THREAD_N=2 horizontally (32 cols) → 32×32 tile per subgroup
- Work-group: 1 subgroup for simplicity, or multiple subgroups. Let's use 4 subgroups per work-group (64 work-items).
- Each work-group handles a 32×64 tile (2 subgroups wide) or we can do 32×32 per subgroup.

Let me use a cleaner approach:
- **Tile per subgroup**: 32 rows × 16 cols (THREAD_M=4 DPAS calls, THREAD_N=1)
- **Work-group**: 2×2 subgroups → 64×32 tile per work-group? 

Actually, let me keep it simpler and effective:
- Each subgroup computes a TM×TN = 32×32 output tile (4 DPAS calls vertically × 2 DPAS calls horizontally)
- K-loop tiles in steps of 16
- Use SLM to stage A and B tiles
- Work-group = 1 subgroup (16 WIs) to start, focusing on register blocking

Let me use SLM with a work-group of 4 subgroups (64 WIs), where each subgroup computes 32×32.

Actually for clarity and correctness, let me use a single subgroup per workgroup computing 32×32 with SLM.

**Final design:**
- WG_M=32, WG_N=32, K_TILE=16
- 1 subgroup (16 WIs) per work-group
- 4 vertical × 2 horizontal DPAS calls per k-step
- Load A[32×16] and B[16×32] into SLM, then DPAS

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

Wait, the switch statements with compile-time constant `i` under `#pragma unroll` should optimize away, but let me use a cleaner approach with array indexing or a helper macro. Actually, let me refactor the store to use a private float array to make it cleaner and more maintainable.

Let me reconsider and write a cleaner version:

```OCL
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable

// Launch config:
// LWS = (16, 1, 1)  -- one subgroup of 16 work-items
// GWS = (ceil(N/32)*16, ceil(M/32), 1)
// Each work-group computes a 32x32 output tile using 8 DPAS calls per k-step
// Register blocking: 4 row-chunks × 2 col-chunks of 8×16 DPAS tiles

__attribute__((intel_reqd_sub_group_size(16)))
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
    const int base_m = wg_m * 32;
    const int base_n = wg_n * 32;
    const int sg_lid = get_sub_group_local_id();

    float8 acc[4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        acc[i][0] = 0.0f;
        acc[i][1] = 0.0f;
    }

    __local ushort slm_a[32 * 16];
    __local ushort slm_b[16 * 32];

    for (int k0 = 0; k0 < K; k0 += 16) {
        // Load A[32][16] into SLM
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            int row = base_m + i;
            int col = k0 + sg_lid;
            ushort val = 0;
            if (row < M && col < K)
                val = as_ushort(A[row * K + col]);
            slm_a[i * 16 + sg_lid] = val;
        }

        // Load B[16][32] into SLM
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            int row = k0 + i;
            ushort val0 = 0, val1 = 0;
            if (row < K) {
                int col0 = base_n + sg_lid;
                int col1 = base_n + 16 + sg_lid;
                if (col0 < N) val0 = as_ushort(B[row * N + col0]);
                if (col1 < N) val1 = as_ushort(B[row * N + col1]);
            }
            slm_b[i * 32 + sg_lid] = val0;
            slm_b[i * 32 + 16 + sg_lid] = val1;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Build A operands (short8 per row-chunk)
        short8 a[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            a[r] = as_short8((ushort8)(
                slm_a[(r*8+0)*16 + sg_lid], slm_a[(r*8+1)*16 + sg_lid],
                slm_a[(r*8+2)*16 + sg_lid], slm_a[(r*8+3)*16 + sg_lid],
                slm_a[(r*8+4)*16 + sg_lid], slm_a[(r*8+5)*16 + sg_lid],
                slm_a[(r*8+6)*16 + sg_lid], slm_a[(r*8+7)*16 + sg_lid]));
        }

        // Build B operands in VNNI format (int8 per col-chunk)
        int8 b[2];
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            int off = c * 16;
            b[c] = (int8)(
                as_int((ushort2)(slm_b[0*32+off+sg_lid], slm_b[1*32+off+sg_lid])),
                as_int((ushort2)(slm_b[2*32+off+sg_lid], slm_b[3*32+off+sg_lid])),
                as_int((ushort2)(slm_b[4*32+off+sg_lid], slm_b[5*32+off+sg_lid])),
                as_int((ushort2)(slm_b[6*32+off+sg_lid], slm_b[7*32+off+sg_lid])),
                as_int((ushort2)(slm_b[8*32+off+sg_lid], slm_b[9*32+off+sg_lid])),
                as_int((ushort2)(slm_b[10*32+off+sg_lid], slm_b[11*32+off+sg_lid])),
                as_int((ushort2)(slm_b[12*32+off+sg_lid], slm_b[13*32+off+sg_lid])),
                as_int((ushort2)(slm_b[14*32+off+sg_lid], slm_b[15*32+off+sg_lid])));
        }

        // 4x2 DPAS calls
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                acc[r][c] = intel_sub_group_f16_f16_matrix_mad_k16(a[r], b[c], acc[r][c]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            float8 v = acc[r][c];
            int col = base_n + c * 16 + sg_lid;
            if (col < N) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int row = base_m + r * 8 + i;
                    if (row < M) {
                        float val;
                        switch(i) {
                            case 0: val = v.s0; break;
                            case 1: val = v.s1; break;
                            case 2: val = v.s2; break;
                            case 3: val = v.s3; break;
                            case 4: val = v.s4; break;
                            case 5: val = v.s5; break;
                            case 6: val = v.s6; break;
                            case 7: val = v.s7; break;
                        }
                        C[row * N + col] = convert_half(val);
                    }
                }
            }
        }
    }
}
```

