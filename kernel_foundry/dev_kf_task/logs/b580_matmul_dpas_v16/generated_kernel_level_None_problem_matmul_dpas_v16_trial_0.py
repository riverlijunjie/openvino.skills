# Analysis

## Requirements

We need an efficient FP16 GEMM kernel for Intel Battlemage (Xe2-HPG) GPU. The key constraints from the instructions are:

1. **Architecture**: 64 work-items (4 subgroups of 16), A in SLM, B from global/L2
2. **Tile size**: 32×64×32 (TILE_M=32, TILE_N=64, TILE_K=32)
3. **Use DPAS**: `intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc)`
4. **Do NOT**: add B to SLM, increase WG beyond 64, use 32×256 tile, use K-step < 32

## Proposed Kernel Structure

- **Work-group**: 64 WIs = 4 subgroups × 16 WIs each
- **Each WG computes**: 32×64 output tile
- **K-loop**: steps of 32, with K=2048 dividing evenly (no remainder needed)
- **SLM**: Double-buffered A tile (32×32 = 2KB per buffer, total ~4KB)
- **B access**: Direct from global memory with prefetch
- **DPAS**: Each subgroup handles 8 rows × 16 columns via DPAS

### Detailed breakdown:
- 4 subgroups handle 4 columns of 16 each → 64 columns total (TILE_N=64)
- Each subgroup computes 32 rows × 16 cols = 4 DPAS calls per K16 step (8 rows each)
- Per K=32 step: 2 K16 iterations × 4 DPAS = 8 DPAS per subgroup
- A is loaded cooperatively into SLM (32×32 half = 2048 bytes)
- B is loaded per-subgroup from global (32×16 = 512 bytes per SG)

### K-loop 2x unroll:
- Process K in steps of 64 (two K=32 tiles per iteration)
- Double-buffer SLM: load next A while computing current A
- Reduces loop overhead for 32 iterations (K=2048/64=32 loops)

### GWS/LWS:
- LWS = (16, 4, 1) → 64 WIs, subgroup size = 16
- GWS = (N/64 * 16, M/32 * 4, 1) → for 4096×4096: (1024, 512, 1)

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, f32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM double-buffered, B from global
// Launch: LWS=(16,4,1), GWS=(N/4, M/8, 1) i.e. (N/64*16, M/32*4, 1)
// Subgroup size: 16
// For M=N=K=4096: GWS=(1024,512,1), LWS=(16,4,1)

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile: 32 rows x 64 cols
    // 4 subgroups, each handles 32 rows x 16 cols
    const int sg_id = get_local_id(1);       // 0..3 (which subgroup)
    const int sg_lid = get_sub_group_local_id(); // 0..15 (lane within SG)
    
    // WG position in output
    const int wg_row = (get_group_id(1)) * 32;  // M tile start
    const int wg_col = (get_group_id(0)) * 64;  // N tile start
    
    // Each subgroup handles 16 consecutive columns
    const int sg_col = wg_col + sg_id * 16;
    
    // SLM for A tile: double-buffered, 32 rows x 32 cols (half)
    // Use stride of 34 to avoid bank conflicts (34 halfs = 68 bytes per row)
    #define SLM_STRIDE 34
    __local half slm_A[2 * 32 * SLM_STRIDE];  // double buffer
    
    // Accumulators: 32 rows x 16 cols per subgroup = 4 DPAS results of float8
    float8 acc0 = 0.0f;  // rows 0-7
    float8 acc1 = 0.0f;  // rows 8-15
    float8 acc2 = 0.0f;  // rows 16-23
    float8 acc3 = 0.0f;  // rows 24-31
    
    // Linear local ID for cooperative loading
    const int lid = sg_id * 16 + sg_lid;  // 0..63
    
    // Preload first A tile into SLM buffer 0
    // A tile: 32 rows x 32 cols = 1024 halfs, 64 WIs load 16 halfs each
    // Each WI loads one row-segment: lid maps to (row, col_chunk)
    // 64 WIs, 32 rows x 32 cols = 1024 elements, 16 per WI
    {
        // Each WI loads 16 consecutive halfs
        // Map: lid/2 = row (0..31), (lid%2)*16 = col offset (0 or 16)
        int a_row = lid / 2;
        int a_col_off = (lid % 2) * 16;
        int a_global_offset = (wg_row + a_row) * K + a_col_off;
        
        __global const half* a_ptr = A + a_global_offset;
        
        // Load 16 halfs (use vload8 x2 for efficiency)
        half8 v0 = vload8(0, a_ptr);
        half8 v1 = vload8(1, a_ptr);
        
        // Store to SLM buffer 0
        int slm_offset = a_row * SLM_STRIDE + a_col_off;
        vstore8(v0, 0, &slm_A[slm_offset]);
        vstore8(v1, 0, &slm_A[slm_offset + 8]);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int buf = 0;  // current buffer being consumed
    
    // Main K-loop, step by 32
    for (int k = 0; k < K; k += 32) {
        int next_k = k + 32;
        
        // Start loading next A tile into other buffer (if not last iteration)
        // We'll do compute first, then load next (with barrier management)
        
        // === COMPUTE from current SLM buffer ===
        // A is 32x32 in SLM, we process in two K16 steps
        
        // B pointer for this K-step: B[k:k+32, sg_col:sg_col+16]
        __global const half* b_ptr = B + k * N + sg_col;
        
        // --- K16 step 0 (k_offset = 0..15) ---
        {
            // Load B[k:k+16, sg_col:sg_col+16] into registers
            // Each lane reads one column, 16 rows = int8 (16 halfs packed as 8 ints)
            int8 b_reg;
            // B is row-major: B[row, col] = B[row*N + col]
            // For DPAS, B needs to be in VNNI format: pairs of k packed into 32-bit
            // int8 b means 8 ints = 16 halfs = 16 k-values for one column
            // Each subgroup lane handles one column (sg_lid = column within 16)
            // We need B in format: b[i] = (B[2i, lane], B[2i+1, lane]) packed as int
            
            __global const int* b_int_ptr = (__global const int*)(b_ptr + sg_lid);
            b_reg.s0 = *(b_int_ptr); b_int_ptr = (__global const int*)(((__global const half*)b_int_ptr) + N);
            b_reg.s1 = *(__global const int*)(b_ptr + 2*N + sg_lid);  
            
            // Actually, for VNNI format we need consecutive k pairs in 32-bit lanes
            // B[k, col] with k=0..15, col=sg_lid
            // Packed as int: (B[2i, col], B[2i+1, col]) for i=0..7
            
            half b_vals[16];
            for (int i = 0; i < 16; i++) {
                b_vals[i] = b_ptr[i * N + sg_lid];
            }
            // Pack into int8 (VNNI format)
            int* b_int = (int*)&b_reg;
            for (int i = 0; i < 8; i++) {
                short h0 = as_short(b_vals[2*i]);
                short h1 = as_short(b_vals[2*i+1]);
                b_int[i] = (int)((uint)(ushort)h0 | ((uint)(ushort)h1 << 16));
            }
            
            // Load A from SLM for rows 0-7, 8-15, 16-23, 24-31 (k=0..15)
            // A in SLM: slm_A[buf*32*SLM_STRIDE + row*SLM_STRIDE + k_col]
            // For DPAS A operand (short8): 8 rows x 16 k-values packed
            // short8 a means 8 shorts = 8 k-values for one row? No...
            // Actually for intel_sub_group_f16_f16_matrix_mad_k16:
            //   a is short8: across the subgroup, represents 8 rows x 16 k-values
            //   Each lane holds 8 shorts = 8 k-values, lane i holds k-values for specific pattern
            // The A matrix for DPAS k16: 8 rows x 16 cols
            // Distributed: each of 16 lanes holds one column (k-index), 8 rows = short8
            
            int slm_base = buf * 32 * SLM_STRIDE;
            
            // For rows 0-7, each lane (sg_lid) reads column sg_lid of k=0..15
            // A[row, k] = slm_A[slm_base + row*SLM_STRIDE + k]
            // Lane sg_lid reads k=sg_lid for all 8 rows
            short8 a0, a1, a2, a3;
            
            // rows 0-7
            a0.s0 = as_short(slm_A[slm_base + 0*SLM_STRIDE + sg_lid]);
            a0.s1 = as_short(slm_A[slm_base + 1*SLM_STRIDE + sg_lid]);
            a0.s2 = as_short(slm_A[slm_base + 2*SLM_STRIDE + sg_lid]);
            a0.s3 = as_short(slm_A[slm_base + 3*SLM_STRIDE + sg_lid]);
            a0.s4 = as_short(slm_A[slm_base + 4*SLM_STRIDE + sg_lid]);
            a0.s5 = as_short(slm_A[slm_base + 5*SLM_STRIDE + sg_lid]);
            a0.s6 = as_short(slm_A[slm_base + 6*SLM_STRIDE + sg_lid]);
            a0.s7 = as_short(slm_A[slm_base + 7*SLM_STRIDE + sg_lid]);
            
            // rows 8-15
            a1.s0 = as_short(slm_A[slm_base + 8*SLM_STRIDE + sg_lid]);
            a1.s1 = as_short(slm_A[slm_base + 9*SLM_STRIDE + sg_lid]);
            a1.s2 = as_short(slm_A[slm_base + 10*SLM_STRIDE + sg_lid]);
            a1.s3 = as_short(slm_A[slm_base + 11*SLM_STRIDE + sg_lid]);
            a1.s4 = as_short(slm_A[slm_base + 12*SLM_STRIDE + sg_lid]);
            a1.s5 = as_short(slm_A[slm_base + 13*SLM_STRIDE + sg_lid]);
            a1.s6 = as_short(slm_A[slm_base + 14*SLM_STRIDE + sg_lid]);
            a1.s7 = as_short(slm_A[slm_base + 15*SLM_STRIDE + sg_lid]);
            
            // rows 16-23
            a2.s0 = as_short(slm_A[slm_base + 16*SLM_STRIDE + sg_lid]);
            a2.s1 = as_short(slm_A[slm_base + 17*SLM_STRIDE + sg_lid]);
            a2.s2 = as_short(slm_A[slm_base + 18*SLM_STRIDE + sg_lid]);
            a2.s3 = as_short(slm_A[slm_base + 19*SLM_STRIDE + sg_lid]);
            a2.s4 = as_short(slm_A[slm_base + 20*SLM_STRIDE + sg_lid]);
            a2.s5 = as_short(slm_A[slm_base + 21*SLM_STRIDE + sg_lid]);
            a2.s6 = as_short(slm_A[slm_base + 22*SLM_STRIDE + sg_lid]);
            a2.s7 = as_short(slm_A[slm_base + 23*SLM_STRIDE + sg_lid]);
            
            // rows 24-31
            a3.s0 = as_short(slm_A[slm_base + 24*SLM_STRIDE + sg_lid]);
            a3.s1 = as_short(slm_A[slm_base + 25*SLM_STRIDE + sg_lid]);
            a3.s2 = as_short(slm_A[slm_base + 26*SLM_STRIDE + sg_lid]);
            a3.s3 = as_short(slm_A[slm_base + 27*SLM_STRIDE + sg_lid]);
            a3.s4 = as_short(slm_A[slm_base + 28*SLM_STRIDE + sg_lid]);
            a3.s5 = as_short(slm_A[slm_base + 29*SLM_STRIDE + sg_lid]);
            a3.s6 = as_short(slm_A[slm_base + 30*SLM_STRIDE + sg_lid]);
            a3.s7 = as_short(slm_A[slm_base + 31*SLM_STRIDE + sg_lid]);
            
            // DPAS calls
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }
        
        // --- K16 step 1 (k_offset = 16..31) ---
        {
            __global const half* b_ptr2 = b_ptr + 16 * N;
            
            half b_vals[16];
            for (int i = 0; i < 16; i++) {
                b_vals[i] = b_ptr2[i * N + sg_lid];
            }
            int8 b_reg;
            int* b_int = (int*)&b_reg;
            for (int i = 0; i < 8; i++) {
                short h0 = as_short(b_vals[2*i]);
                short h1 = as_short(b_vals[2*i+1]);
                b_int[i] = (int)((uint)(ushort)h0 | ((uint)(ushort)h1 << 16));
            }
            
            int slm_base = buf * 32 * SLM_STRIDE + 16; // offset by 16 in k
            
            short8 a0, a1, a2, a3;
            
            a0.s0 = as_short(slm_A[slm_base + 0*SLM_STRIDE + sg_lid]);
            a0.s1 = as_short(slm_A[slm_base + 1*SLM_STRIDE + sg_lid]);
            a0.s2 = as_short(slm_A[slm_base + 2*SLM_STRIDE + sg_lid]);
            a0.s3 = as_short(slm_A[slm_base + 3*SLM_STRIDE + sg_lid]);
            a0.s4 = as_short(slm_A[slm_base + 4*SLM_STRIDE + sg_lid]);
            a0.s5 = as_short(slm_A[slm_base + 5*SLM_STRIDE + sg_lid]);
            a0.s6 = as_short(slm_A[slm_base + 6*SLM_STRIDE + sg_lid]);
            a0.s7 = as_short(slm_A[slm_base + 7*SLM_STRIDE + sg_lid]);
            
            a1.s0 = as_short(slm_A[slm_base + 8*SLM_STRIDE + sg_lid]);
            a1.s1 = as_short(slm_A[slm_base + 9*SLM_STRIDE + sg_lid]);
            a1.s2 = as_short(slm_A[slm_base + 10*SLM_STRIDE + sg_lid]);
            a1.s3 = as_short(slm_A[slm_base + 11*SLM_STRIDE + sg_lid]);
            a1.s4 = as_short(slm_A[slm_base + 12*SLM_STRIDE + sg_lid]);
            a1.s5 = as_short(slm_A[slm_base + 13*SLM_STRIDE + sg_lid]);
            a1.s6 = as_short(slm_A[slm_base + 14*SLM_STRIDE + sg_lid]);
            a1.s7 = as_short(slm_A[slm_base + 15*SLM_STRIDE + sg_lid]);
            
            a2.s0 = as_short(slm_A[slm_base + 16*SLM_STRIDE + sg_lid]);
            a2.s1 = as_short(slm_A[slm_base + 17*SLM_STRIDE + sg_lid]);
            a2.s2 = as_short(slm_A[slm_base + 18*SLM_STRIDE + sg_lid]);
            a2.s3 = as_short(slm_A[slm_base + 19*SLM_STRIDE + sg_lid]);
            a2.s4 = as_short(slm_A[slm_base + 20*SLM_STRIDE + sg_lid]);
            a2.s5 = as_short(slm_A[slm_base + 21*SLM_STRIDE + sg_lid]);
            a2.s6 = as_short(slm_A[slm_base + 22*SLM_STRIDE + sg_lid]);
            a2.s7 = as_short(slm_A[slm_base + 23*SLM_STRIDE + sg_lid]);
            
            a3.s0 = as_short(slm_A[slm_base + 24*SLM_STRIDE + sg_lid]);
            a3.s1 = as_short(slm_A[slm_base + 25*SLM_STRIDE + sg_lid]);
            a3.s2 = as_short(slm_A[slm_base + 26*SLM_STRIDE + sg_lid]);
            a3.s3 = as_short(slm_A[slm_base + 27*SLM_STRIDE + sg_lid]);
            a3.s4 = as_short(slm_A[slm_base + 28*SLM_STRIDE + sg_lid]);
            a3.s5 = as_short(slm_A[slm_base + 29*SLM_STRIDE + sg_lid]);
            a3.s6 = as_short(slm_A[slm_base + 30*SLM_STRIDE + sg_lid]);
            a3.s7 = as_short(slm_A[slm_base + 31*SLM_STRIDE + sg_lid]);
            
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }
        
        // === Load next A tile (if not last) ===
        if (next_k < K) {
            int next_buf = 1 - buf;
            int a_row = lid / 2;
            int a_col_off = (lid % 2) * 16;
            int a_global_offset = (wg_row + a_row) * K + next_k + a_col_off;
            
            __global const half* a_ptr = A + a_global_offset;
            half8 v0 = vload8(0, a_ptr);
            half8 v1 = vload8(1, a_ptr);
            
            int slm_offset = next_buf * 32 * SLM_STRIDE + a_row * SLM_STRIDE + a_col_off;
            vstore8(v0, 0, &slm_A[slm_offset]);
            vstore8(v1, 0, &slm_A[slm_offset + 8]);
            
            buf = next_buf;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // === Store C ===
    // Each subgroup writes 32 rows x 16 cols
    // acc0 = rows 0-7, acc1 = rows 8-15, acc2 = rows 16-23, acc3 = rows 24-31
    // Each float8 has 8 values for 8 rows, one column per lane (sg_lid)
    
    __global half* c_ptr = C + wg_row * N + sg_col + sg_lid;
    
    // Store rows 0-7
    c_ptr[0*N] = convert_half(acc0.s0);
    c_ptr[1*N] = convert_half(acc0.s1);
    c_ptr[2*N] = convert_half(acc0.s2);
    c_ptr[3*N] = convert_half(acc0.s3);
    c_ptr[4*N] = convert_half(acc0.s4);
    c_ptr[5*N] = convert_half(acc0.s5);
    c_ptr[6*N] = convert_half(acc0.s6);
    c_ptr[7*N] = convert_half(acc0.s7);
    
    // Store rows 8-15
    c_ptr[8*N] = convert_half(acc1.s0);
    c_ptr[9*N] = convert_half(acc1.s1);
    c_ptr[10*N] = convert_half(acc1.s2);
    c_ptr[11*N] = convert_half(acc1.s3);
    c_ptr[12*N] = convert_half(acc1.s4);
    c_ptr[13*N] = convert_half(acc1.s5);
    c_ptr[14*N] = convert_half(acc1.s6);
    c_ptr[15*N] = convert_half(acc1.s7);
    
    // Store rows 16-23
    c_ptr[16*N] = convert_half(acc2.s0);
    c_ptr[17*N] = convert_half(acc2.s1);
    c_ptr[18*N] = convert_half(acc2.s2);
    c_ptr[19*N] = convert_half(acc2.s3);
    c_ptr[20*N] = convert_half(acc2.s4);
    c_ptr[21*N] = convert_half(acc2.s5);
    c_ptr[22*N] = convert_half(acc2.s6);
    c_ptr[23*N] = convert_half(acc2.s7);
    
    // Store rows 24-31
    c_ptr[24*N] = convert_half(acc3.s0);
    c_ptr[25*N] = convert_half(acc3.s1);
    c_ptr[26*N] = convert_half(acc3.s2);
    c_ptr[27*N] = convert_half(acc3.s3);
    c_ptr[28*N] = convert_half(acc3.s4);
    c_ptr[29*N] = convert_half(acc3.s5);
    c_ptr[30*N] = convert_half(acc3.s6);
    c_ptr[31*N] = convert_half(acc3.s7);
}
```

Wait, let me reconsider the approach more carefully. The current kernel achieves 0.948ms. Let me think about what optimizations will actually help.

Key issues with the above draft:
1. The B loading with scalar reads is very inefficient
2. The SLM A loading and compute are serialized (load next AFTER compute)
3. Need proper double-buffering where load overlaps with compute

Let me redesign with proper double-buffering and better B loading:

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], row-major, half precision I/O, f32 accumulation
// Architecture: 64 WIs = 4 subgroups x 16 WIs, TILE 32x64x32
// A in SLM (double-buffered), B from global/L2
// Launch: LWS=(16,4,1), GWS=(N/4, M/8, 1) => for 4096x4096: GWS=(1024,512,1)
// Subgroup size: 16

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* A,
    __global const half* B,
    __global half* C,
    const int M,
    const int K,
    const int N)
{
    const int sg_id = get_local_id(1);          // 0..3 subgroup index
    const int sg_lid = get_sub_group_local_id(); // 0..15 lane
    const int lid = sg_id * 16 + sg_lid;         // 0..63 linear local id

    // WG tile position
    const int wg_row = get_group_id(1) * 32;
    const int wg_col = get_group_id(0) * 64;

    // Each subgroup handles 16 columns
    const int sg_col = wg_col + sg_id * 16;

    // SLM: double-buffered A, 32 rows x 32 cols with stride 32 (no padding needed if we use careful access)
    // Using stride 32 for simplicity; 32 halfs = 64 bytes = 4 banks of 16 bytes
    // Actually use stride 32 since K-step is 32 and we want contiguous k access
    #define SLM_A_STRIDE 32
    __local half slm_A[2 * 32 * SLM_A_STRIDE]; // 2 buffers x 32 rows x 32 cols = 4096 halfs = 8KB

    // Accumulators: 32 rows x 16 cols = 4 x float8
    float8 acc0 = 0.0f;
    float8 acc1 = 0.0f;
    float8 acc2 = 0.0f;
    float8 acc3 = 0.0f;

    // === Cooperative load of first A tile into buffer 0 ===
    // 32x32 = 1024 halfs, 64 WIs, each loads 16 halfs
    {
        int a_row = lid / 2;          // 0..31
        int a_col_off = (lid & 1) * 16; // 0 or 16
        __global const half* a_src = A + (wg_row + a_row) * K + a_col_off;
        __local half* a_dst = slm_A + a_row * SLM_A_STRIDE + a_col_off;
        
        half8 v0 = vload8(0, a_src);
        half8 v1 = vload8(1, a_src);
        vstore8(v0, 0, a_dst);
        vstore8(v1, 0, a_dst + 8);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int cur_buf = 0;

    // Main K-loop: K is guaranteed divisible by 32
    for (int k = 0; k < K; k += 32) {
        int next_k = k + 32;
        int next_buf = 1 - cur_buf;

        // === Start loading next A tile into next_buf (overlapped with compute) ===
        // We load first, then barrier before next iteration's compute
        // Actually for proper overlap: load next A, then compute current, then barrier
        // But SLM writes and reads to different buffers don't conflict
        
        // Load next A tile (if exists)
        half8 next_a0, next_a1;
        int has_next = (next_k < K);
        int a_row_load = lid / 2;
        int a_col_load = (lid & 1) * 16;
        
        if (has_next) {
            __global const half* a_src = A + (wg_row + a_row_load) * K + next_k + a_col_load;
            next_a0 = vload8(0, a_src);
            next_a1 = vload8(1, a_src);
        }

        // === COMPUTE: process current A tile from cur_buf ===
        __local const half* slm_base = slm_A + cur_buf * 32 * SLM_A_STRIDE;
        __global const half* b_base = B + k * N + sg_col;

        // K16 step 0: k_local = 0..15
        {
            // Load B: 16 rows x 16 cols, pack into VNNI int8
            // Each lane reads its column (sg_lid), 16 rows with stride N
            int8 b_reg;
            __global const half* b_col = b_base + sg_lid;
            
            // Pack pairs of k-rows into int (VNNI format)
            short h0, h1;
            h0 = as_short(b_col[0*N]); h1 = as_short(b_col[1*N]);
            ((int*)&b_reg)[0] = ((uint)(ushort)h0) | ((uint)(ushort)h1 << 16);
            h0 = as_short(b_col[2*N]); h1 = as_short(b_col[3*N]);
            ((int*)&b_reg)[1] = ((uint)(ushort)h0) | ((uint)(ushort)h1 << 16);
            h0 = as_short(b_col[4*N]); h1 = as_short(b_col[5*N]);
            ((int*)&b_reg)[2] = ((uint)(ushort)h0) | ((uint)(ushort)h1 << 16);
            h0 = as_short(b_col[6*N]); h1 = as_short(b_col[7*N]);
            ((int*)&b_reg)[3] = ((uint)(ushort)h0) | ((uint)(ushort)h1 << 16);
            h0 = as_short(b_col[8*N]); h1 = as_short(b_col[9*N]);
            ((int*)&b_reg)[4] = ((uint)(ushort)h0) | ((uint)(ushort)h1 << 16);
            h0 = as_short(b_col[10*N]); h1 = as_short(b_col[11*N]);
            ((int*)&b_reg)[5] = ((uint)(ushort)h0) | ((uint)(ushort)h1 << 16);
            h0 = as_short(b_col[12*N]); h1 = as_short(b_col[13*N]);
            ((int*)&b_reg)[6] = ((uint)(ushort)h0) | ((uint)(ushort)h1 << 16);
            h0 = as_short(b_col[14*N]); h1 = as_short(b_col[15*N]);
            ((int*)&b_reg)[7] = ((uint)(ushort)h0) | ((uint)(ushort)h1 << 16);

            // Load A from SLM: each lane reads k=sg_lid for 8 rows
            short8 a0, a1, a2, a3;
            
            a0 = (short8)(as_short(slm_base[0*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[1*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[2*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[3*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[4*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[5*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[6*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[7*SLM_A_STRIDE + sg_lid]));
            
            a1 = (short8)(as_short(slm_base[8*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[9*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[10*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[11*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[12*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[13*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[14*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[15*SLM_A_STRIDE + sg_lid]));
            
            a2 = (short8)(as_short(slm_base[16*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[17*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[18*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[19*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[20*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[21*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[22*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[23*SLM_A_STRIDE + sg_lid]));
            
            a3 = (short8)(as_short(slm_base[24*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[25*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[26*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[27*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[28*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[29*SLM_A_STRIDE + sg_lid]),
                          as_short(slm_base[30*SLM_A_STRIDE + sg_lid]),
                          as