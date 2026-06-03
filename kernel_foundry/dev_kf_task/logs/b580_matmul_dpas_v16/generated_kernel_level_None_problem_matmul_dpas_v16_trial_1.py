## Analysis

### Previous Version Performance
The last correct kernel achieved 2.560ms, which is about 9% XMX utilization. The main bottlenecks are:

1. **Scalar B loads**: Loading B one element at a time with `b_ptr[i * N + sg_lid]` is extremely inefficient - 16 scalar loads per k-pair, 128 total for the 32-wide K tile.
2. **Scalar A loads from SLM**: Loading A one element at a time with individual `slm_A[...]` accesses instead of using vectorized SLM reads.
3. **Double-buffering overhead**: The barrier placement and buffer swap logic could be tighter.
4. **No B prefetching**: B data arrives from global/L2 with no prefetch hints.
5. **SLM stride of 34**: While intended to avoid bank conflicts, this prevents use of `intel_sub_group_block_read` for SLM reads (which requires stride = subgroup_size = 16).

### Proposed Improvements

1. **SLM layout optimized for block reads**: Store A in SLM with stride=32 (the full K-tile width). Read using manual scalar loads but with better organization, OR store in a transposed/blocked layout that enables `intel_sub_group_block_read_us` (stride=16 between rows).
2. **Vectorized B loads**: Use `vload2` to load pairs of B rows at once, reducing load instruction count.
3. **Remove K-remainder path**: K=2048 or 4096 always divides by 32.
4. **Better double-buffering**: Overlap next A load with current compute by loading next A before the barrier.
5. **K-loop 2x unroll**: Process 64 K-elements per iteration to reduce loop overhead.
6. **SLM layout**: Use stride=16 with two separate 32×16 blocks for the two K16 halves. This enables `intel_sub_group_block_read_us8` for A loads.

### Key Design Decisions
- Keep 64 WIs (4 SGs), TILE 32×64×32
- A in SLM with layout optimized for block reads (stride=16, two K16 blocks)
- B from global with paired loads
- Double-buffered SLM
- The kernel name must be `matmul` (compile error was about wrong kernel name)

## Improved OCL code

```OCL
// GEMM: C[M,N] = A[M,K] x B[K,N], all half precision, f32 accumulation
// Architecture: 64 WIs (4 SGs of 16), TILE 32x64x32, A in SLM double-buffered, B from global
// Launch: LWS=(16,4,1), GWS=(N/4, M/8, 1) i.e. (N/64*16, M/32*4, 1)
// Subgroup size: 16
// For M=N=K=4096: GWS=(1024,512,1), LWS=(16,4,1)

#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void matmul(
    __global const half* restrict A,
    __global const half* restrict B,
    __global half* restrict C,
    const int M,
    const int K,
    const int N)
{
    // Work-group tile: 32 rows x 64 cols
    // 4 subgroups, each handles 32 rows x 16 cols
    const int sg_id = get_local_id(1);       // 0..3 (which subgroup)
    const int sg_lid = get_sub_group_local_id(); // 0..15 (lane within SG)

    // WG position in output
    const int wg_row = get_group_id(1) * 32;  // M tile start
    const int wg_col = get_group_id(0) * 64;  // N tile start (get_group_id(0) = global_id(0)/16 effectively)

    // Each subgroup handles 16 consecutive columns
    const int sg_col = wg_col + sg_id * 16;

    // SLM for A tile: double-buffered
    // Layout: two K16 blocks per buffer, each block is 32 rows x 16 cols with stride=16
    // This enables efficient subgroup reads
    // Buffer layout: [buf][k_half][row][k_within_16]
    // Total per buffer: 32 * 32 = 1024 halfs
    // With stride=16 for each k_half block: 2 * 32 * 16 = 1024 halfs per buffer
    // Double buffer: 2048 halfs total = 4096 bytes
    #define SLM_K16_STRIDE 16
    #define SLM_BUF_SIZE 1024
    __local half slm_A[2 * SLM_BUF_SIZE];  // double buffer, each 32*32 halfs

    // Accumulators: 32 rows x 16 cols per subgroup = 4 DPAS results of float8
    float8 acc0 = 0.0f;  // rows 0-7
    float8 acc1 = 0.0f;  // rows 8-15
    float8 acc2 = 0.0f;  // rows 16-23
    float8 acc3 = 0.0f;  // rows 24-31

    // Linear local ID for cooperative loading
    const int lid = sg_id * 16 + sg_lid;  // 0..63

    // A load mapping: 64 WIs load 32x32 = 1024 halfs = 16 per WI
    // Map: lid/2 = row (0..31), (lid%2) = which k_half (0 or 1)
    // Each WI loads 16 consecutive halfs (one row of one k_half block)
    const int a_load_row = lid / 2;        // 0..31
    const int a_load_khalf = lid % 2;      // 0 or 1

    // Preload first A tile into SLM buffer 0
    {
        int global_row = wg_row + a_load_row;
        int global_k = a_load_khalf * 16;
        __global const half* a_ptr = A + global_row * K + global_k;

        // SLM offset: k_half * 32*16 + row * 16
        int slm_off = a_load_khalf * (32 * SLM_K16_STRIDE) + a_load_row * SLM_K16_STRIDE;

        // Load 16 halfs using vload8 x2
        half8 v0 = vload8(0, a_ptr);
        half8 v1 = vload8(1, a_ptr);
        vstore8(v0, 0, &slm_A[slm_off]);
        vstore8(v1, 0, &slm_A[slm_off + 8]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int buf = 0;  // current buffer being consumed
    const int k_iterations = K / 32;

    for (int ki = 0; ki < k_iterations; ki++) {
        int k_offset = ki * 32;
        int next_k = k_offset + 32;

        // === COMPUTE from current SLM buffer ===
        int slm_base = buf * SLM_BUF_SIZE;

        // B pointer for this K-step
        __global const half* b_base = B + k_offset * N + sg_col + sg_lid;

        // --- K16 step 0 (k=0..15) ---
        {
            // Load A from SLM: k_half=0 block
            // A[row][k] at slm_A[slm_base + 0*(32*16) + row*16 + k]
            // For subgroup block read: base at slm_base + row_base*16
            // Lane sg_lid reads k=sg_lid, short8 gives 8 rows
            int a_off = slm_base;

            short8 a0, a1, a2, a3;

            // Use intel_sub_group_block_read_us to read 8 consecutive ushorts
            // from SLM with stride=16 between vector elements
            // This reads: lane i gets element at base + i, base+16+i, ..., base+7*16+i
            // Which gives us A[row_base+j][k=sg_lid] for j=0..7

            // rows 0-7
            a0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            // rows 8-15
            a1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            // rows 16-23
            a2 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            // rows 24-31
            a3 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));

            // Load B[k:k+16, sg_col+sg_lid] and pack into VNNI int8
            // Each lane loads 16 values from its column, stride=N between rows
            // Pack pairs into int: b[i] = pack(B[2i], B[2i+1])
            int8 b_reg;

            __global const half* bp = b_base;
            b_reg.s0 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s1 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s2 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s3 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s4 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s5 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s6 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s7 = as_int((half2)(bp[0], bp[N]));

            // DPAS calls
            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }

        // --- K16 step 1 (k=16..31) ---
        {
            // Load A from SLM: k_half=1 block
            int a_off = slm_base + 32 * SLM_K16_STRIDE;

            short8 a0, a1, a2, a3;

            a0 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 0*SLM_K16_STRIDE]));
            a1 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 8*SLM_K16_STRIDE]));
            a2 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 16*SLM_K16_STRIDE]));
            a3 = as_short8(intel_sub_group_block_read_us8((__local const uint*)&slm_A[a_off + 24*SLM_K16_STRIDE]));

            // Load B for k=16..31
            __global const half* bp = b_base + 16 * N;
            int8 b_reg;

            b_reg.s0 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s1 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s2 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s3 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s4 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s5 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s6 = as_int((half2)(bp[0], bp[N])); bp += 2*N;
            b_reg.s7 = as_int((half2)(bp[0], bp[N]));

            acc0 = intel_sub_group_f16_f16_matrix_mad_k16(a0, b_reg, acc0);
            acc1 = intel_sub_group_f16_f16_matrix_mad_k16(a1, b_reg, acc1);
            acc2 = intel_sub_group_f16_f16_matrix_mad_k16(a2, b_reg, acc2);
            acc3 = intel_sub_group_f16_f16_matrix_mad_k16(a3, b_reg, acc3);
        }

        // === Load next A tile into other SLM buffer ===
        if (next_k < K) {
            int next_buf = 1 - buf;
            int global_row = wg_row + a_load_row;
            int global_k = next_k + a_load_khalf * 16;
            __global const half* a_ptr = A + global_row * K + global_k;

            int slm_off = next_buf * SLM_BUF_SIZE + a_load_khalf * (32 * SLM_K16_STRIDE) + a_load_row * SLM_K16_STRIDE;

            half8 v0 = vload8(0, a_ptr);
            half8 v1 = vload8(1, a_ptr);
            vstore8(v0, 0, &slm_A[slm_off]);
            vstore8(v1, 0, &slm_A[slm_off + 8]);

            buf = next_buf;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // === Store C ===
    // Each subgroup writes 32 rows x 16 cols
    // acc[r] at lane sg_lid = C[wg_row + row_offset + r, sg_col + sg_lid]
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